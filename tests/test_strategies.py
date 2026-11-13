"""Strategies run side by side, each on its own paper book: trend following, the config
sections, the daily job and the combined recap."""

import json

import numpy as np
import pandas as pd
import pytest

from pairs import cli, notify
from pairs.backtest import BacktestConfig, signal_frame
from pairs.config import Config, load_config, strategy_params
from pairs.paper import PaperBook
from pairs.paths import STATE_DIR, strategy_path
from pairs.portfolio import WeightBook
from pairs.signals import leg_weights
from pairs.strategies import (
    STRATEGIES,
    low_vol,
    momentum,
    picks,
    reversal,
    risk_parity,
    sector_arb,
    trend,
)


def _etfs(n=400, end="2026-10-26"):
    """UPP climbs, DWN falls, CALM climbs with a tenth of their volatility."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range(end=end, periods=n)

    def walk(drift, vol, start):
        return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))

    return pd.DataFrame({"UPP": walk(0.001, 0.01, 50), "DWN": walk(-0.001, 0.01, 80),
                         "CALM": walk(0.0005, 0.001, 100)}, idx)


def _params(**over):
    return strategy_params("trend", {"universe": ["UPP", "DWN", "CALM"], **over})


def _vol(s):
    return np.log(s).diff().iloc[-60:].std() * np.sqrt(252)


# ---------------------------------------------------------------- trend following
def test_trend_goes_long_what_rose_and_short_what_fell():
    px = _etfs()
    weights, reasons = trend.targets(px, _params(universe=["UPP", "DWN"], max_gross=10))
    assert weights["UPP"] > 0 > weights["DWN"]
    up = px["UPP"].iloc[-1] / px["UPP"].iloc[-253] - 1
    assert reasons["UPP"].startswith(f"up {up:.1%} over 12 months: long")
    assert reasons["DWN"].startswith("down ") and "short" in reasons["DWN"]
    assert f"{_vol(px['UPP']):.0%} volatility" in reasons["UPP"]


def test_trend_sizes_each_etf_for_an_equal_share_of_the_volatility_target():
    px = _etfs()
    weights, _ = trend.targets(px, _params(vol_target=0.10, max_gross=100))
    for t in px:
        assert abs(weights[t]) == pytest.approx(0.10 / 3 / _vol(px[t]))
    assert abs(weights["CALM"]) > 5 * abs(weights["UPP"])  # calmer, so held in more size


def test_trend_caps_gross_exposure():
    px = _etfs()
    free, _ = trend.targets(px, _params(max_gross=100))
    assert sum(abs(w) for w in free.values()) > 1.5
    capped, _ = trend.targets(px, _params(max_gross=1.5))
    assert sum(abs(w) for w in capped.values()) == pytest.approx(1.5)
    assert capped["UPP"] / capped["DWN"] == pytest.approx(free["UPP"] / free["DWN"])


def test_trend_needs_a_year_of_history():
    px = _etfs()
    px.loc[px.index[:-252], "DWN"] = np.nan  # 252 closes: one short of a 12-month return
    weights, reasons = trend.targets(px, _params())
    assert weights["DWN"] == 0 and reasons["DWN"] == "not enough history"
    assert weights["UPP"] != 0
    weights, reasons = trend.targets(px[["UPP", "CALM"]], _params())  # no data at all
    assert weights["DWN"] == 0 and reasons["DWN"] == "not enough history"


def test_trend_does_not_look_ahead():
    px = _etfs()
    day = px.index[-40]
    then = trend.targets(px.loc[:day], _params())
    assert trend.targets(px, _params(), asof=day) == then  # the later rows change nothing
    assert trend.targets(px, _params()) != then


@pytest.mark.parametrize("last, date, due", [
    (None, "2026-10-19", True),  # never rebalanced
    ("2026-10-01", "2026-10-30", False),  # same month
    ("2026-10-01", "2026-11-02", True),  # first session of November
    ("2026-11-02", "2026-11-02", False),  # catch-up run on the same close
    ("2026-12-01", "2027-01-04", True),
    ("2026-11-02", "2026-10-30", False),  # prices behind the book: wait
])
def test_trend_rebalances_once_a_month(last, date, due):
    assert trend.due(last, date) is due


# ---------------------------------------------------------------- config
def test_config_loads_a_strategy_section(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('[strategy]\ncost_bps = 3\n[strategies.trend]\nstarting_cash = 50000\n'
                 'universe = ["spy", "TLT"]\nvol_target = 0.15\n')
    c = load_config(f)
    t = c.strategies["trend"]
    assert t["enabled"] is True and t["starting_cash"] == 50_000
    assert t["universe"] == ["SPY", "TLT"] and t["vol_target"] == 0.15
    assert t["max_gross"] == 1.5  # default kept
    assert t["cost_bps"] == 3  # the pairs strategy's, unless set here
    f.write_text("[strategies.trend]\ncost_bps = 1\nenabled = false\n")
    t = load_config(f).strategies["trend"]
    assert t["cost_bps"] == 1 and t["enabled"] is False and "SPY" in t["universe"]
    f.write_text("")
    assert load_config(f).strategies == {}


@pytest.mark.parametrize("text, match", [
    ("[strategies.trend]\nvol_traget = 0.1\n", "vol_traget"),
    ("[strategies.trned]\nenabled = true\n", "trned"),
    ("[strategies.trend]\nuniverse = []\n", "universe"),
    ("[strategies.trend]\nuniverse = \"SPY\"\n", "universe"),
    ("[strategies.trend]\nstarting_cash = 0\n", "starting_cash"),
    ("[strategies.trend]\nvol_target = -0.1\n", "vol_target"),
    ("[strategies.trend]\nenabled = \"yes\"\n", "enabled"),
    ("[strategies.momentum]\nn_each = 0\n", "n_each"),
    ("[strategies.momentum]\nn_each = 2.5\n", "n_each"),
])
def test_config_rejects_bad_strategy_sections(tmp_path, text, match):
    f = tmp_path / "config.toml"
    f.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_config(f)


def test_repo_config_runs_trend_on_its_own_book():
    t = load_config().strategies["trend"]
    assert t["enabled"] and "SPY" in t["universe"] and t["starting_cash"] > 0
    assert strategy_path("trend") == STATE_DIR / "trend.json"
    assert "trend" in STRATEGIES


# ---------------------------------------------------------------- risk parity
def _rp(**over):
    return strategy_params("risk_parity", {"universe": ["UPP", "DWN", "CALM"], **over})


def test_risk_parity_holds_everything_long_sized_to_equal_risk():
    px = _etfs()
    weights, reasons = risk_parity.targets(px, _rp(exposure=1.0))
    assert all(w > 0 for w in weights.values())  # long only, even what fell
    assert sum(weights.values()) == pytest.approx(1.0)
    risk = [weights[t] * _vol(px[t]) for t in px]  # each brings the same volatility
    assert max(risk) == pytest.approx(min(risk))
    assert reasons["CALM"] == (f"{_vol(px['CALM']):.0%} volatility: "
                               f"{weights['CALM']:.0%} of the book")


def test_risk_parity_exposure_sets_the_total():
    weights, _ = risk_parity.targets(_etfs(), _rp(exposure=1.5))
    assert sum(weights.values()) == pytest.approx(1.5)


def test_risk_parity_shares_out_what_lacks_history():
    px = _etfs()
    px.loc[px.index[:-30], "DWN"] = np.nan  # only 30 closes
    weights, reasons = risk_parity.targets(px, _rp())
    assert weights["DWN"] == 0 and reasons["DWN"] == "not enough history"
    assert sum(weights.values()) == pytest.approx(1.0)


def test_risk_parity_does_not_look_ahead():
    px = _etfs()
    early = px.index[-100]
    assert risk_parity.targets(px.loc[:early], _rp()) == risk_parity.targets(px, _rp(), asof=early)


def test_risk_parity_rebalances_monthly_like_trend():
    assert risk_parity.due(None, "2026-11-02")
    assert not risk_parity.due("2026-11-02", "2026-11-30")
    assert risk_parity.due("2026-11-30", "2026-12-01")


def test_repo_config_runs_risk_parity_on_its_own_book():
    rp = load_config().strategies["risk_parity"]
    assert rp["enabled"] and {"SPY", "TLT", "GLD"} <= set(rp["universe"])
    assert strategy_path("risk_parity") == STATE_DIR / "risk_parity.json"


# ---------------------------------------------------------------- momentum long/short
def _stocks(n=300, k=30):
    """k stocks, S00 drifting down hardest through S29 drifting up hardest."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range(end="2026-10-26", periods=n)
    return pd.DataFrame({f"S{i:02d}": 50 * np.exp(np.cumsum(rng.normal((i - k / 2) * 1e-4,
                                                                     0.01, n)))
                         for i in range(k)}, idx)


def _mom(**over):
    return strategy_params("momentum", {"n_each": 5, **over})


def _score(s):
    return s.iloc[-22] / s.iloc[-253] - 1


def test_momentum_buys_the_winners_and_shorts_the_losers():
    px = _stocks()
    weights, reasons = momentum.targets(px, _mom(exposure=1.0))
    ranked = sorted(px, key=lambda t: _score(px[t]))
    assert {t for t, w in weights.items() if w > 0} == set(ranked[-5:])
    assert {t for t, w in weights.items() if w < 0} == set(ranked[:5])
    assert all(abs(w) == pytest.approx(0.1) for w in weights.values())  # 1.0 / 2 / 5
    best = ranked[-1]
    assert reasons[best] == (f"up {_score(px[best]):.1%} over the 12 months to a month ago: "
                             "among the 5 best of 30, long")


def test_momentum_skips_the_latest_month():
    px = _stocks()
    loser = min(px, key=lambda t: _score(px[t]))
    px.loc[px.index[-21:], loser] *= 3  # a late rally doesn't count
    weights, _ = momentum.targets(px, _mom())
    assert weights[loser] < 0


def test_momentum_leaves_out_stocks_without_a_years_history():
    px = _stocks()
    best = max(px, key=lambda t: _score(px[t]))
    px.loc[px.index[:-200], best] = np.nan
    weights, _ = momentum.targets(px, _mom())
    assert best not in weights


def test_momentum_does_not_look_ahead():
    px = _stocks()
    early = px.index[-30]
    assert momentum.targets(px.loc[:early], _mom()) == momentum.targets(px, _mom(), asof=early)


def test_momentum_trades_the_sp500_from_universe_csv():
    tickers = momentum.tickers(_mom())
    assert 450 < len(tickers) < 520 and "AAPL" in tickers and len(set(tickers)) == len(tickers)


def test_repo_config_runs_momentum_on_its_own_book():
    m = load_config().strategies["momentum"]
    assert m["enabled"] and m["index"] == "S&P 500" and m["n_each"] >= 5


def test_a_short_that_cannot_be_borrowed_is_dropped(tmp_path, market, monkeypatch):
    prices, _ = market
    from pairs.borrow import BorrowInfo
    monkeypatch.setattr(cli, "borrow_table", lambda tickers: {
        "DWN": BorrowInfo(shortable=False, fee_rate=0.0, available=0, source="ibkr")})
    _daily(tmp_path, prices)
    book = WeightBook.load(tmp_path / "trend.json")
    assert "DWN" not in book.holdings and book.holdings["UPP"] > 0
    assert any(e["pair"] == "TREND DWN" and "can't borrow" in e["reason"] for e in book.log)


def test_between_rebalances_only_the_holdings_are_priced(tmp_path, market, monkeypatch):
    prices, _ = market
    config = Config(strategies={"trend": _params(universe=["UPP", "DWN", "CALM", "NEW"])})
    asked = []

    def load(tickers, **k):
        asked.append(sorted(tickers))
        return prices[[t for t in tickers if t in prices]]
    monkeypatch.setattr(cli, "load_prices", lambda tickers, **k: load(tickers)
                        .loc[:prices.index[-2]])
    book = PaperBook.load(tmp_path / "book.json")
    cli.daily(book, config, now=_after_close(prices.iloc[:-1]), sleep=lambda s: None)
    held = sorted(WeightBook.load(tmp_path / "trend.json").holdings)
    asked.clear()
    monkeypatch.setattr(cli, "load_prices", lambda tickers, **k: load(tickers))
    cli.daily(book, config, now=_after_close(prices), sleep=lambda s: None)
    assert asked == [held]  # same month: no need to price the whole universe


# ---------------------------------------------------------------- low volatility
def _calm_to_jumpy(n=300, k=20):
    """k stocks, V00 the calmest through V19 the jumpiest."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range(end="2026-10-26", periods=n)
    return pd.DataFrame({f"V{i:02d}": 40 * np.exp(np.cumsum(rng.normal(0, 0.005 + i * 0.002, n)))
                         for i in range(k)}, idx)


def _lv(**over):
    return strategy_params("low_vol", {"n_each": 4, **over})


def _yvol(s):
    return np.log(s).diff().iloc[-252:].std() * np.sqrt(252)


def test_low_vol_buys_the_calmest_and_shorts_the_jumpiest():
    px = _calm_to_jumpy()
    weights, reasons = low_vol.targets(px, _lv())
    assert {t for t, w in weights.items() if w > 0} == {"V00", "V01", "V02", "V03"}
    assert {t for t, w in weights.items() if w < 0} == {"V16", "V17", "V18", "V19"}
    assert reasons["V00"] == f"{_yvol(px['V00']):.0%} volatility: among the 4 calmest of 20, long"


def test_low_vol_balances_the_risk_of_the_two_sides():
    px = _calm_to_jumpy()
    weights, _ = low_vol.targets(px, _lv(exposure=1.0))
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)
    risk = [w * _yvol(px[t]) for t, w in weights.items()]
    assert sum(r for r in risk if r > 0) == pytest.approx(-sum(r for r in risk if r < 0))
    assert weights["V00"] > -weights["V19"]  # the calm side is held in more size


def test_low_vol_does_not_look_ahead():
    px = _calm_to_jumpy()
    early = px.index[-30]
    assert low_vol.targets(px.loc[:early], _lv()) == low_vol.targets(px, _lv(), asof=early)


def test_repo_config_runs_low_vol_on_its_own_book():
    lv = load_config().strategies["low_vol"]
    assert lv["enabled"] and lv["index"] == "S&P 500"
    assert len(low_vol.tickers(lv)) > 450


# ---------------------------------------------------------------- stock vs sector ETF
def _sector(n=300, kick=None):
    """XLK and four tech stocks that track it closely; `kick` = {ticker: last-day move}."""
    rng = np.random.default_rng(13)
    idx = pd.bdate_range(end="2026-10-26", periods=n)
    etf = np.cumsum(rng.normal(0, 0.01, n))
    out = {"XLK": 100 * np.exp(etf)}
    for t in ("AAPL", "MSFT", "NVDA", "ORCL"):
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.003)
        out[t] = 50 * np.exp(etf + noise)
    df = pd.DataFrame(out, idx)
    for t, move in (kick or {}).items():
        df.loc[df.index[-1], t] *= 1 + move
    return df


def _sa(**over):
    return strategy_params("sector_arb", {"max_positions": 10, **over})


def test_sector_arb_maps_each_stock_to_its_sector_etf():
    assert sector_arb.sector_etf()["AAPL"] == "XLK" and sector_arb.sector_etf()["JPM"] == "XLF"
    tickers = sector_arb.tickers(_sa())
    assert {"AAPL", "XLK", "XLF", "XLU"} <= set(tickers) and len(tickers) > 450


def test_sector_arb_shorts_a_stock_that_ran_ahead_of_its_sector():
    px = _sector(kick={"AAPL": 0.015})
    weights, reasons = sector_arb.targets(px, _sa(position_size=0.1))
    frame = signal_frame(px["AAPL"], px["XLK"], BacktestConfig())
    z, beta = frame["z"].iloc[-1], frame["beta"].iloc[-1]
    w_y, w_x = leg_weights(beta)
    assert 2 <= z < 4 and weights["AAPL"] == pytest.approx(-0.1 * w_y)
    assert weights["XLK"] == pytest.approx(0.1 * w_x)  # the hedge
    assert reasons["AAPL"].startswith(f"z {z:+.2f} vs XLK: AAPL ran ahead of its sector, "
                                      "so short AAPL / buy XLK")
    assert set(weights) == {"AAPL", "XLK"}  # the others are near normal


def test_sector_arb_keeps_a_trade_until_it_converges():
    px = _sector(kick={"AAPL": 0.015})
    weights, reasons = sector_arb.targets(px, _sa(entry=10.0), held={"AAPL": -40})
    assert weights["AAPL"] < 0 and "holding" in reasons["AAPL"]  # no new entries at 10, kept


def test_sector_arb_closes_a_trade_once_it_converges():
    px = _sector()
    zs = {t: signal_frame(px[t], px["XLK"], BacktestConfig())["z"].iloc[-1]
          for t in ("AAPL", "MSFT", "NVDA", "ORCL")}
    t = min(zs, key=lambda k: abs(zs[k]))
    assert abs(zs[t]) <= 1.0
    weights, reasons = sector_arb.targets(px, _sa(exit=1.0), held={t: -40})
    assert weights[t] == 0 and f"z {zs[t]:+.2f} vs XLK: back within ±1.0, converged" == reasons[t]


def test_sector_arb_takes_the_widest_gaps_first():
    px = _sector(kick={"AAPL": 0.02, "MSFT": 0.011, "NVDA": -0.03})
    weights, _ = sector_arb.targets(px, _sa(max_positions=2))
    assert {t for t in weights if t != "XLK"} == {"AAPL", "NVDA"}


def test_sector_arb_nets_the_hedges_on_one_etf():
    px = _sector(kick={"AAPL": 0.015, "NVDA": -0.025})
    weights, _ = sector_arb.targets(px, _sa(position_size=0.1))
    legs = {t: leg_weights(signal_frame(px[t], px["XLK"], BacktestConfig())["beta"].iloc[-1])[1]
            for t in ("AAPL", "NVDA")}
    assert weights["XLK"] == pytest.approx(0.1 * legs["AAPL"] - 0.1 * legs["NVDA"])


def test_repo_config_runs_sector_arb_on_its_own_book():
    sa = load_config().strategies["sector_arb"]
    assert sa["enabled"] and sa["index"] == "S&P 500" and sa["max_positions"] >= 1


# ---------------------------------------------------------------- short-term reversal
def _rev(**over):
    return strategy_params("reversal", {"n_each": 5, **over})


def _week(s):
    return s.iloc[-1] / s.iloc[-6] - 1


def test_reversal_buys_last_weeks_losers_and_shorts_its_winners():
    px = _stocks(n=60)
    weights, reasons = reversal.targets(px, _rev(exposure=1.0))
    ranked = sorted(px, key=lambda t: _week(px[t]))
    assert {t for t, w in weights.items() if w > 0} == set(ranked[:5])
    assert {t for t, w in weights.items() if w < 0} == set(ranked[-5:])
    assert all(abs(w) == pytest.approx(0.1) for w in weights.values())
    worst = ranked[0]
    assert reasons[worst] == (f"{'up' if _week(px[worst]) >= 0 else 'down'} "
                              f"{abs(_week(px[worst])):.1%} over the past week: among the 5 "
                              "worst of 30, long for a bounce")


def test_reversal_does_not_look_ahead():
    px = _stocks(n=60)
    early = px.index[-10]
    assert reversal.targets(px.loc[:early], _rev()) == reversal.targets(px, _rev(), asof=early)


@pytest.mark.parametrize("last, date, due", [
    (None, "2026-11-12", True),
    ("2026-11-09", "2026-11-13", False),  # Monday, then Friday of the same week
    ("2026-11-13", "2026-11-16", True),  # Friday, then the next Monday
    ("2026-12-31", "2027-01-04", True),  # across the new year
])
def test_reversal_trades_at_the_first_close_of_each_week(last, date, due):
    assert reversal.due(last, date) is due


def test_repo_config_runs_reversal_on_its_own_book():
    r = load_config().strategies["reversal"]
    assert r["enabled"] and r["index"] == "S&P 500"


# ---------------------------------------------------------------- your own picks
def _picks(**over):
    return strategy_params("picks", {"long": ["UPP", "CALM"], "short": ["DWN"], **over})


def test_picks_hold_your_longs_and_shorts_at_equal_weights():
    weights, reasons = picks.targets(_etfs(), _picks(exposure=1.5))
    assert weights == pytest.approx({"UPP": 0.5, "CALM": 0.5, "DWN": -0.5})
    assert reasons["UPP"] == "your pick: long"
    assert reasons["DWN"] == "your pick: short"
    assert picks.tickers(_picks()) == ["CALM", "DWN", "UPP"]


def test_picks_without_a_price_are_left_out():
    weights, _ = picks.targets(_etfs(), _picks(long=["UPP", "NOPE"]))
    assert "NOPE" not in weights and weights["UPP"] == pytest.approx(0.5)


def test_picks_are_checked_every_close():
    assert picks.due("2026-11-12", "2026-11-13")


@pytest.mark.parametrize("section, match", [
    ({"long": [], "short": []}, "long or short"),
    ({"long": ["AAPL"], "short": ["aapl"]}, "both"),
    ({"long": "AAPL"}, "long"),
])
def test_picks_config_is_checked(section, match):
    with pytest.raises(ValueError, match=match):
        strategy_params("picks", section)


def test_repo_config_has_a_picks_section_switched_off_until_you_add_some():
    assert load_config().strategies.get("picks", {"enabled": False})["enabled"] is False


# ---------------------------------------------------------------- recap by strategy
def _eq(d, equity, open_=1):
    return {"date": d, "cash": 100_000.0, "unrealised": equity - 100_000.0, "equity": equity,
            "open": open_}


def _book(equity=(), log=(), orders=(), warnings=(), start=100_000.0):
    return {"starting_cash": start, "cash": start, "equity": list(equity), "log": list(log),
            "orders": list(orders), "last_run_warnings": list(warnings)}


def _state(tmp_path, name, **books):
    d = tmp_path / name
    d.mkdir()
    for stem, book in books.items():
        (d / f"{stem}.json").write_text(json.dumps(book))
    return d


TREND_ORDER = {"pair": "TREND SPY", "event": "rebalance", "signal_date": "2026-10-28",
               "execute_on": "2026-10-29", "z": 0.0,
               "legs": [{"ticker": "SPY", "side": "BUY", "qty": 12, "ref_price": 600.0}]}
TREND_LOG = {"date": "2026-10-28", "pair": "TREND SPY", "action": "rebalance", "z": 0.0,
             "reason": "up 14.2% over 12 months: long, sized for 18% volatility"}
DAY1 = "2026-10-27"
DAY2 = "2026-10-28"


def _two_days():
    old = {"book": _book([_eq(DAY1, 100_100.0)]),
           "trend": _book([_eq(DAY1, 50_000.0, 0)], start=50_000.0)}
    new = {"book": _book([_eq(DAY1, 100_100.0), _eq(DAY2, 100_410.0)]),
           "trend": _book([_eq(DAY1, 50_000.0, 0), _eq(DAY2, 49_900.0, 9)], [TREND_LOG],
                          [TREND_ORDER], ["⚠ TREND: no price for UUP"], start=50_000.0)}
    return old, new


PNL = ("**P&L after the Wed 2026-10-28 close**\n"
       "Total: $150,310 · +$310 (+0.21%) overall · +$210 today\n"
       "Pairs: $100,410 · +$410 (+0.41%) overall · +$310 today\n"
       "Trend: $49,900 · -$100 (-0.20%) overall · -$100 today")


def test_recap_opens_with_overall_and_per_strategy_pnl():
    old, new = _two_days()
    text = notify.full_recap(old, new)
    assert text.startswith(PNL + "\n\n" + notify.recap(old["book"], new["book"]))
    assert "**Trend after the Wed 2026-10-28 close**" in text
    section = text[text.index("**Trend after"):]
    assert "$49,900" in section and "-$100" in section and "-0.20%" in section
    assert "$50,000 start" in section and "since Tue 2026-10-27" in section
    assert "9 holdings" in section
    assert "**TREND SPY** REBALANCE: up 14.2% over 12 months" in section
    assert "BUY 12 SPY" in section and "⚠ TREND: no price for UUP" in section


def test_recap_is_quiet_when_no_book_has_a_new_close():
    _, new = _two_days()
    assert notify.full_recap(new, new) == ""


def test_recap_skips_the_section_of_a_strategy_with_no_new_close():
    old, new = _two_days()
    new["trend"] = old["trend"]  # e.g. its prices were late: only pairs moved on
    text = notify.full_recap(old, new)
    assert "Paper book after" in text and "Trend after" not in text
    assert "Trend: $50,000 · +$0 (+0.00%) overall · +$0 today" in text  # still in the totals


def test_a_new_book_counts_its_first_day_from_its_start():
    old, new = _two_days()
    del old["trend"]
    new["trend"]["equity"] = new["trend"]["equity"][-1:]
    assert "Trend: $49,900 · -$100 (-0.20%) overall · -$100 today" in notify.full_recap(old, new)


def test_recap_posts_a_strategy_warning_even_without_a_new_close():
    old, new = _two_days()
    new = {"book": old["book"], "trend": {**old["trend"], "last_run_warnings": ["⚠ Trend: boom"]}}
    assert notify.full_recap(old, new) == "⚠ Trend: boom"
    assert notify.full_recap(new, new) == ""  # once, not on every run


def test_recap_with_only_the_pairs_book_still_shows_pnl():
    old, new = _two_days()
    text = notify.full_recap({"book": old["book"]}, {"book": new["book"]})
    assert text == ("**P&L after the Wed 2026-10-28 close**\n"
                    "Total: $100,410 · +$410 (+0.41%) overall · +$310 today\n"
                    "Pairs: $100,410 · +$410 (+0.41%) overall · +$310 today\n\n"
                    + notify.recap(old["book"], new["book"]))


def test_notifier_takes_two_state_directories(tmp_path, monkeypatch):
    old, new = _two_days()
    old_dir, new_dir = _state(tmp_path, "old", **old), _state(tmp_path, "state", **new)
    posted = []
    monkeypatch.setattr(notify, "post_discord", lambda text, url: posted.append(text) or True)
    notify.main([str(old_dir), str(new_dir)])
    assert posted == [notify.full_recap(old, new)]
    posted.clear()
    notify.main([str(new_dir), str(new_dir)])  # the catch-up run: nothing new
    assert posted == []
    notify.main([str(tmp_path / "missing"), str(new_dir)])  # first run: no old state at all
    assert len(posted) == 1 and "Trend after" in posted[0] and "Total:" in posted[0]


def test_notifier_still_takes_two_book_files(tmp_path, monkeypatch):
    old, new = _two_days()
    old_dir, new_dir = _state(tmp_path, "old", **old), _state(tmp_path, "state", **new)
    posted = []
    monkeypatch.setattr(notify, "post_discord", lambda text, url: posted.append(text) or True)
    notify.main([str(old_dir / "book.json"), str(new_dir / "book.json")])
    assert posted == [notify.recap(old["book"], new["book"])]


# ---------------------------------------------------------------- daily job
def _after_close(prices):
    return pd.Timestamp(prices.index[-1]).tz_localize("America/New_York") + pd.Timedelta(hours=20)


@pytest.fixture
def market(monkeypatch, coint_pair):
    """A cointegrated pair plus three ETFs, ending on the same close. Returns (prices, posted)."""
    y, x = coint_pair
    etfs = _etfs()
    pair = pd.concat([y.rename("YYY"), x.rename("XXX")], axis=1)
    pair.index = pd.bdate_range(end=etfs.index[-1], periods=len(pair))
    prices = pair.join(etfs)
    monkeypatch.setattr(cli, "load_prices",
                        lambda tickers, **k: prices[[t for t in tickers if t in prices]])
    monkeypatch.setattr(cli, "borrow_table", lambda t: {})
    posted = []
    monkeypatch.setattr(cli, "post_discord", lambda text, url: posted.append(text) or True)
    return prices, posted


def _daily(tmp_path, prices, config=None, **kw):
    book = PaperBook.load(tmp_path / "book.json")
    book.last_bar.setdefault("YYY/XXX", str(prices.index[-200].date()))
    config = config or Config(watchlist=[("YYY", "XXX")], strategies={"trend": _params()})
    new = cli.daily(book, config, webhook_url="https://hook", now=_after_close(prices),
                    sleep=lambda s: None, **kw)
    return book, new


def test_daily_runs_each_strategy_on_its_own_book(tmp_path, market):
    prices, posted = market
    book, new = _daily(tmp_path, prices)
    close = str(prices.index[-1].date())
    trend_book = WeightBook.load(tmp_path / "trend.json")
    assert book.orders and trend_book.orders
    assert book.equity[-1]["date"] == trend_book.equity[-1]["date"] == close
    assert trend_book.last_rebalance == close and trend_book.starting_cash == 100_000
    assert set(trend_book.holdings) == {"UPP", "DWN", "CALM"}
    assert trend_book.holdings["UPP"] > 0 > trend_book.holdings["DWN"]
    assert {o["pair"] for o in new} >= {"YYY/XXX", "TREND UPP", "TREND DWN", "TREND CALM"}
    (text,) = posted
    assert "Paper book after" in text and "Trend after" in text
    assert "**TREND UPP** REBALANCE" in text and "over 12 months" in text
    assert text.startswith("**P&L after") and "Total: $" in text
    assert "Pairs: $" in text and "Trend: $" in text and "overall" in text

    posted.clear()
    _, new = _daily(tmp_path, prices)  # catch-up run on the same close
    assert new == [] and posted == []
    assert WeightBook.load(tmp_path / "trend.json").to_dict() == trend_book.to_dict()


def test_daily_marks_every_day_but_rebalances_monthly(tmp_path, market, monkeypatch):
    prices, _ = market
    monkeypatch.setattr(cli, "load_prices", lambda tickers, **k: prices.iloc[:-1][
        [t for t in tickers if t in prices]])
    book = PaperBook.load(tmp_path / "book.json")
    config = Config(strategies={"trend": _params()})
    cli.daily(book, config, now=_after_close(prices.iloc[:-1]), sleep=lambda s: None)
    first = WeightBook.load(tmp_path / "trend.json")
    monkeypatch.setattr(cli, "load_prices",
                        lambda tickers, **k: prices[[t for t in tickers if t in prices]])
    cli.daily(book, config, now=_after_close(prices), sleep=lambda s: None)
    second = WeightBook.load(tmp_path / "trend.json")
    assert len(second.equity) == 2 and second.equity[-1]["equity"] != first.equity[-1]["equity"]
    assert second.orders == first.orders and second.holdings == first.holdings  # same month


def test_a_strategy_that_crashes_does_not_stop_the_pairs_book(tmp_path, market, monkeypatch):
    prices, posted = market

    def boom(prices, params):
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(trend, "targets", boom)
    book, _ = _daily(tmp_path, prices)
    saved = PaperBook.load(tmp_path / "book.json")
    assert saved.orders == book.orders and saved.orders and saved.equity
    trend_book = WeightBook.load(tmp_path / "trend.json")
    (w,) = trend_book.last_run_warnings
    assert "Trend" in w and "boom" in w
    assert trend_book.holdings == {} and trend_book.equity == []
    (text,) = posted
    assert "Paper book after" in text and "YYY/XXX" in text and w in text


def test_a_strategy_with_no_prices_does_not_stop_the_pairs_book(tmp_path, market):
    prices, posted = market
    config = Config(watchlist=[("YYY", "XXX")],
                    strategies={"trend": _params(universe=["NOPE", "NADA"])})
    _daily(tmp_path, prices, config)
    assert PaperBook.load(tmp_path / "book.json").equity
    (w,) = WeightBook.load(tmp_path / "trend.json").last_run_warnings
    assert "no price data" in w and w in posted[0]


def test_a_disabled_strategy_is_not_run(tmp_path, market):
    prices, posted = market
    config = Config(watchlist=[("YYY", "XXX")], strategies={"trend": _params(enabled=False)})
    _daily(tmp_path, prices, config)
    assert not (tmp_path / "trend.json").exists() and "Trend: $" not in posted[0]


def test_pairs_with_no_prices_still_fails_the_run(tmp_path, market):
    prices, _ = market
    config = Config(watchlist=[("ZZZ", "QQQQ")], strategies={"trend": _params()})
    with pytest.raises(RuntimeError, match="No price data"):
        cli.daily(PaperBook.load(tmp_path / "book.json"), config, now=_after_close(prices),
                  sleep=lambda s: None, attempts=1)


def test_a_strategy_does_not_wait_again_for_a_close_pairs_already_waited_for(tmp_path, market):
    """The hosted job has 15 minutes: one wait for a late Yahoo close, not one per book."""
    prices, posted = market
    slept = []
    book = PaperBook.load(tmp_path / "book.json")
    config = Config(watchlist=[("YYY", "XXX")], strategies={"trend": _params()})
    late = _after_close(prices) + pd.Timedelta(days=1)  # Tuesday evening, prices end Monday
    cli.daily(book, config, webhook_url="https://hook", now=late, sleep=slept.append, attempts=3)
    assert len(slept) == 2
    assert posted[0].count("hasn't published") == 1  # said once, not once per book
    trend_book = WeightBook.load(tmp_path / "trend.json")
    assert trend_book.equity[-1]["date"] == str(prices.index[-1].date())  # used what it had
    assert any("hasn't published" in w for w in trend_book.last_run_warnings)


def test_a_strategy_book_that_cannot_be_read_is_left_alone(tmp_path, market):
    prices, posted = market
    (tmp_path / "trend.json").write_text("{not json")
    book, _ = _daily(tmp_path, prices)
    assert (tmp_path / "trend.json").read_text() == "{not json"  # for a human to look at
    assert any("trend.json" in w for w in PaperBook.load(tmp_path / "book.json").last_run_warnings)
    assert book.equity and "couldn't read trend.json" in posted[0]


def test_daily_credits_dividends_from_the_restated_closes(tmp_path, market, monkeypatch):
    """The day after: CALM (held long) paid 2% and DWN (held short) 1%, so Yahoo's series
    now shows every earlier close of theirs scaled down by that much."""
    prices, _ = market
    monkeypatch.setattr(cli, "load_prices", lambda tickers, **k: prices.iloc[:-1][
        [t for t in tickers if t in prices]])
    book = PaperBook.load(tmp_path / "book.json")
    config = Config(strategies={"trend": _params()})
    cli.daily(book, config, now=_after_close(prices.iloc[:-1]), sleep=lambda s: None)
    first = WeightBook.load(tmp_path / "trend.json")
    assert first.holdings["CALM"] > 0 > first.holdings["DWN"]

    paid = prices.copy()
    paid.loc[paid.index[:-1], "CALM"] *= 0.98
    paid.loc[paid.index[:-1], "DWN"] *= 0.99
    monkeypatch.setattr(cli, "load_prices",
                        lambda tickers, **k: paid[[t for t in tickers if t in paid]])
    cli.daily(book, config, now=_after_close(paid), sleep=lambda s: None)
    second = WeightBook.load(tmp_path / "trend.json")
    was = prices.iloc[-2]
    dividends = (first.holdings["CALM"] * was["CALM"] * 0.02
                 + first.holdings["DWN"] * was["DWN"] * 0.01)
    assert second.cash - first.cash == pytest.approx(dividends)
    assert second.holdings == first.holdings and not second.last_run_warnings
    moved = sum(n * (prices[t].iloc[-1] - was[t]) for t, n in first.holdings.items())
    assert second.equity[-1]["equity"] - first.equity[-1]["equity"] == pytest.approx(
        moved + dividends)  # total return, not just the price change

    cli.daily(book, config, now=_after_close(paid), sleep=lambda s: None)  # same close again
    assert WeightBook.load(tmp_path / "trend.json").cash == second.cash
