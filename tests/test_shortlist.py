"""Shortlisting tradeable pairs, and capping how many trades are open at once."""

import numpy as np
import pandas as pd
import pytest

from pairs.backtest import BacktestConfig, signal_frame
from pairs.cli import replay
from pairs.config import load_config
from pairs.data import load_universe
from pairs.paper import PaperBook
from pairs.scanner import shortlist


def _pair(seed, names, n=800, phi=0.93, beta=1.2):
    rng = np.random.default_rng(seed)
    log_x = np.log(50) + np.cumsum(rng.normal(0, 0.012, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] * phi + rng.normal(0, 0.012)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({names[0]: np.exp(0.3 + beta * log_x + ou + rng.normal(0, 0.004, n)),
                         names[1]: np.exp(log_x)}, idx)


def _walks(seed, names, n=800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({t: np.exp(np.log(40) + np.cumsum(rng.normal(0, 0.015, n)))
                         for t in names}, idx)


def test_shortlist_keeps_mean_reverting_pairs_and_drops_noise():
    prices = pd.concat([_pair(1, ("AAA", "BBB")), _walks(2, ("CCC", "DDD"))], axis=1)
    out = shortlist(prices, {"g": list(prices.columns)}, BacktestConfig())
    pairs = {(r.y, r.x) for r in out.itertuples()} | {(r.x, r.y) for r in out.itertuples()}
    assert ("AAA", "BBB") in pairs
    assert not any("CCC" in p or "DDD" in p for p in pairs)
    for col in ("pvalue", "pvalue_1y", "half_life", "corr", "sharpe", "return_1y", "trades"):
        assert col in out.columns


def test_shortlist_caps_pairs_per_group():
    prices = pd.concat([_pair(1, ("AAA", "BBB")), _pair(3, ("EEE", "FFF"))], axis=1)
    both = shortlist(prices, {"g": list(prices.columns)}, BacktestConfig(), per_group=5)
    one = shortlist(prices, {"g": list(prices.columns)}, BacktestConfig(), per_group=1)
    assert len(both) >= 2 and len(one) == 1
    assert one.iloc[0]["sharpe"] == both["sharpe"].max()  # keeps the best


def test_shortlist_skips_near_identical_funds():
    base = _pair(1, ("AAA", "BBB"))
    twin = base[["BBB"]].rename(columns={"BBB": "BBB2"}) * 1.0001  # same fund, new ticker
    prices = pd.concat([base, twin], axis=1)
    out = shortlist(prices, {"g": list(prices.columns)}, BacktestConfig())
    assert not any({r.y, r.x} == {"BBB", "BBB2"} for r in out.itertuples())


def test_sp500_universe_groups_by_sub_industry(tmp_path):
    f = tmp_path / "u.csv"
    f.write_text("ticker,name,sector,sub_industry\n"
                 "KO,Coca-Cola,Staples,Soft Drinks & Non-alcoholic Beverages\n"
                 "PEP,PepsiCo,Staples,Soft Drinks & Non-alcoholic Beverages\n"
                 "BRK-B,Berkshire,Financials,Multi-Sector Holdings\n"
                 "UNP,Union Pacific,Industrials,Rail Transportation\n"
                 "CSX,CSX,Industrials,Rail Transportation\n")
    groups = load_universe(f)
    assert groups == {"Soft Drinks & Non-alcoholic Beverages": ["KO", "PEP"],
                      "Rail Transportation": ["CSX", "UNP"]}  # one-stock groups can't pair


def test_bundled_sp500_universe_is_usable():
    groups = load_universe()
    tickers = [t for g in groups.values() for t in g]
    assert len(tickers) > 400 and len(tickers) == len(set(tickers))
    assert all(len(g) >= 2 for g in groups.values())
    assert "BF-B" in tickers and not any("." in t for t in tickers)  # Yahoo spelling


# ---------------------------------------------------------------- position cap
def _kw(**kw):
    return {"entry": 2, "exit": 0.5, "stop": 4, "notional": 10_000, "beta": 1.0, **kw}


def test_cap_blocks_new_trades_but_not_exits(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    assert book.step(("A", "B"), date="2026-10-19", y_px=10, x_px=10, z=2.5,
                     **_kw(max_open=1)) == "open short_spread"
    assert book.step(("C", "D"), date="2026-10-19", y_px=10, x_px=10, z=2.5,
                     **_kw(max_open=1)) == "skip: max open positions"
    assert list(book.positions) == ["A/B"]
    assert book.step(("A", "B"), date="2026-10-20", y_px=10, x_px=10, z=0.1,
                     **_kw(max_open=1)) == "close"


def test_replay_reports_capped_signals(tmp_path, coint_pair):
    y, x = coint_pair
    book = PaperBook.load(tmp_path / "b.json")
    book.positions["Q/R"] = {"y": "Q", "x": "R", "side": "short_spread", "y_shares": -1,
                             "x_shares": 1, "entry_y": 1.0, "entry_x": 1.0, "beta": 1.0,
                             "entry_date": "2020-01-01", "entry_z": 2.5}
    fr = signal_frame(y, x, BacktestConfig())
    first = fr.index.get_loc(fr.index[(fr["z"].abs() >= 2) & (fr["beta"] > 0)][0])
    book.last_bar["YYY/XXX"] = str(fr.index[first - 1].date())
    actions = replay(book, ("YYY", "XXX"), y, x, fr.iloc[: first + 1], BacktestConfig(),
                     10_000, max_open=1)
    assert any("skip: max open positions" in a for a in actions)
    assert "YYY/XXX" not in book.positions


def test_config_max_open_positions(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[paper]\nmax_open_positions = 5\n")
    c = load_config(f)
    assert c.max_open_positions == 5 and isinstance(c.max_open_positions, int)
    assert load_config(tmp_path / "c.toml").notional == 10_000
    f.write_text("[paper]\nnotional = 5000\n")
    assert load_config(f).max_open_positions is None  # no cap unless set


def test_config_rejects_bad_max_open_positions(tmp_path):
    f = tmp_path / "c.toml"
    for bad in ("2.5", "0", "-1"):
        f.write_text(f"[paper]\nmax_open_positions = {bad}\n")
        with pytest.raises(ValueError, match="max_open_positions"):
            load_config(f)
