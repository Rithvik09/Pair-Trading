"""Order sheet, Discord notifications, config file and price-cache behaviour."""

import json

import pandas as pd
import pytest

from pairs import data, notify
from pairs.config import load_config
from pairs.orders import next_session
from pairs.paper import PaperBook


def _open(book, z=2.5, y_px=60.0, x_px=170.0, beta=1.0, notional=10_000, date="2026-10-19"):
    return book.step(("KO", "PEP"), date=date, y_px=y_px, x_px=x_px, beta=beta, z=z,
                     entry=2, exit=0.5, stop=4, notional=notional)


# ---------------------------------------------------------------- whole shares
def test_positions_use_whole_shares(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)
    p = book.positions["KO/PEP"]
    assert p["y_shares"] == -83  # $5,000 / $60 = 83.3 -> 83 short
    assert p["x_shares"] == 29  # $5,000 / $170 = 29.4 -> 29 long
    assert isinstance(p["y_shares"], int) and isinstance(p["x_shares"], int)


def test_too_small_to_trade(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    assert _open(book, notional=200, x_px=170.0) == "too small PEP"  # $100 < 1 share of PEP
    assert book.positions == {} and book.orders == []


def test_rebase_keeps_real_share_count(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)
    idx = pd.to_datetime(["2026-10-19"])
    book.rebase_entry("KO/PEP", pd.Series([59.0], idx), pd.Series([168.0], idx))
    p = book.positions["KO/PEP"]
    assert (p["y_shares"], p["x_shares"]) == (-83, 29)
    assert (p["entry_y"], p["entry_x"]) == (59.0, 168.0)


# ---------------------------------------------------------------- order sheet
def test_open_writes_orders_for_both_legs(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)  # Monday
    (o,) = book.orders
    assert o["pair"] == "KO/PEP" and o["event"] == "open short_spread"
    assert o["signal_date"] == "2026-10-19" and o["execute_on"] == "2026-10-20"
    assert o["legs"] == [
        {"ticker": "KO", "side": "SELL SHORT", "qty": 83, "ref_price": 60.0},
        {"ticker": "PEP", "side": "BUY", "qty": 29, "ref_price": 170.0},
    ]


def test_close_reverses_each_leg(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)
    book.step(("KO", "PEP"), date="2026-10-23", y_px=58.0, x_px=173.0, beta=1.0, z=0.1,
              entry=2, exit=0.5, stop=4, notional=10_000)  # Friday
    o = book.orders[-1]
    assert o["event"] == "close" and o["execute_on"] == "2026-10-26"  # next Monday
    assert o["legs"] == [
        {"ticker": "KO", "side": "BUY TO COVER", "qty": 83, "ref_price": 58.0},
        {"ticker": "PEP", "side": "SELL", "qty": 29, "ref_price": 173.0},
    ]


def test_stop_is_labelled(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)
    book.step(("KO", "PEP"), date="2026-10-20", y_px=66.0, x_px=165.0, beta=1.0, z=4.4,
              entry=2, exit=0.5, stop=4, notional=10_000)
    assert book.orders[-1]["event"] == "stop"


def test_orders_persist(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)
    book.save()
    assert PaperBook.load(tmp_path / "b.json").orders == book.orders


def test_next_session_skips_weekends_and_market_holidays():
    assert next_session("2026-10-16") == "2026-10-19"  # Fri -> Mon
    assert next_session("2026-11-25") == "2026-11-27"  # day before Thanksgiving
    assert next_session("2026-12-24") == "2026-12-28"  # Christmas Friday, then weekend
    assert next_session("2027-03-25") == "2027-03-29"  # Good Friday 2027-03-26


# ---------------------------------------------------------------- discord
def test_format_order_sheet(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _open(book)
    text = notify.format_orders(book.orders)
    assert "KO/PEP" in text and "SELL SHORT 83 KO" in text and "BUY 29 PEP" in text
    assert "Tue 2026-10-20" in text and "at the open" in text


def test_discord_posts_in_chunks(monkeypatch):
    sent = []

    class Resp:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent.append(json.loads(req.data)["content"])
        return Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    lines = "\n".join(f"line {i} " + "x" * 50 for i in range(100))
    notify.post_discord(lines, "https://discord.com/api/webhooks/1/abc")
    assert len(sent) > 1 and all(len(c) <= 2000 for c in sent)
    assert "\n".join(sent) == lines


def test_discord_skipped_without_url(monkeypatch):
    monkeypatch.setattr(notify.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("should not post"))
    assert notify.post_discord("hi", None) is False


# ---------------------------------------------------------------- config
def test_load_config(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('watchlist = [["KO", "PEP"], ["gs", "ms"]]\n'
                 "[strategy]\nentry = 2.5\nz_window = 20\n"
                 "[paper]\nnotional = 5000\nstarting_cash = 50000\n")
    c = load_config(f)
    assert c.watchlist == [("KO", "PEP"), ("GS", "MS")]
    assert c.strategy.entry == 2.5 and c.strategy.z_window == 20
    assert c.strategy.exit == 0.5  # default kept
    assert (c.notional, c.starting_cash) == (5000, 50000)


def test_config_rejects_unknown_keys(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[strategy]\nentyr = 2.5\n")
    with pytest.raises(ValueError, match="entyr"):
        load_config(f)


def test_repo_config_is_valid():
    load_config()  # the committed config.toml must always parse


# ---------------------------------------------------------------- price cache
def test_intraday_download_is_not_reused_after_the_close(tmp_path, monkeypatch):
    calls = []
    idx = pd.to_datetime(["2026-10-19", "2026-10-20"])

    def fake_download(tickers, **kw):
        calls.append(tickers)
        price = 10.0 if len(calls) == 1 else 11.0  # intraday vs closing price
        cols = pd.MultiIndex.from_product([["Close"], tickers])
        return pd.DataFrame([[9.0] * len(tickers), [price] * len(tickers)], idx, cols)

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.yf, "download", fake_download)
    noon = pd.Timestamp("2026-10-20 12:00", tz="America/New_York")
    evening = pd.Timestamp("2026-10-20 17:00", tz="America/New_York")

    df = data.load_prices(["KO"], now=noon)
    assert list(df.index) == [idx[0]]  # unfinished bar hidden
    df = data.load_prices(["KO"], now=evening)
    assert len(calls) == 2  # re-fetched after the close
    assert df["KO"].iloc[-1] == 11.0


# ---------------------------------------------------------------- daily job
def _no_sleep(seconds):
    pytest.fail("the daily job waited for fresh data in a test that has fresh data")


def _after_close(prices):
    """The evening of the last bar, so that bar counts as the latest completed close."""
    return pd.Timestamp(prices.index[-1]).tz_localize("America/New_York") + pd.Timedelta(hours=20)


def test_daily_posts_only_new_orders(tmp_path, monkeypatch, coint_pair):
    from pairs import cli
    from pairs.config import Config

    y, x = coint_pair
    prices = pd.concat([y.rename("YYY"), x.rename("XXX")], axis=1)
    monkeypatch.setattr(cli, "load_prices", lambda *a, **k: prices)
    monkeypatch.setattr(cli, "borrow_table", lambda t: {})
    posted = []
    monkeypatch.setattr(cli, "post_discord", lambda text, url: posted.append((text, url)) or True)

    book = PaperBook.load(tmp_path / "book.json")
    book.orders.append({"pair": "OLD/ONE", "event": "close", "signal_date": "2020-01-02",
                        "execute_on": "2020-01-03", "z": 0.0, "legs": []})
    book.last_bar["YYY/XXX"] = str(prices.index[-200].date())  # replay ~200 bars
    cfg = Config(watchlist=[("YYY", "XXX")])

    new = cli.daily(book, cfg, webhook_url="https://hook", now=_after_close(prices), sleep=_no_sleep)
    assert new and all(o["pair"] == "YYY/XXX" for o in new)
    (text, url), = posted
    assert url == "https://hook" and "OLD/ONE" not in text and "YYY/XXX" in text
    assert PaperBook.load(tmp_path / "book.json").orders[-len(new):] == new  # saved

    posted.clear()
    assert cli.daily(book, cfg, webhook_url="https://hook", now=_after_close(prices),
                     sleep=_no_sleep) == []  # nothing new
    assert posted == []


def test_daily_fails_loudly_when_no_prices_arrive(tmp_path, monkeypatch):
    from pairs import cli
    from pairs.config import Config

    monkeypatch.setattr(cli, "load_prices", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cli, "borrow_table", lambda t: {})
    monkeypatch.setattr(cli, "post_discord", lambda *a: True)
    with pytest.raises(RuntimeError, match="No price data"):
        cli.daily(PaperBook.load(tmp_path / "b.json"), Config(watchlist=[("KO", "PEP")]),
                  sleep=_no_sleep, attempts=1)


def test_daily_warns_about_a_missing_ticker(tmp_path, monkeypatch, coint_pair):
    from pairs import cli
    from pairs.config import Config

    y, x = coint_pair
    prices = pd.concat([y.rename("YYY"), x.rename("XXX")], axis=1)
    monkeypatch.setattr(cli, "load_prices", lambda *a, **k: prices)
    monkeypatch.setattr(cli, "borrow_table", lambda t: {})
    posted = []
    monkeypatch.setattr(cli, "post_discord", lambda text, url: posted.append(text) or True)
    book = PaperBook.load(tmp_path / "b.json")
    cli.daily(book, Config(watchlist=[("YYY", "XXX"), ("ZZZ", "XXX")]), webhook_url="https://hook",
              now=_after_close(prices), sleep=_no_sleep)
    assert any("ZZZ/XXX" in t and "no price data" in t for t in posted)
    # stored for the post-commit notifier too, and cleared by a clean run
    assert PaperBook.load(tmp_path / "b.json").last_run_warnings == book.last_run_warnings
    assert "ZZZ/XXX" in book.last_run_warnings[0]
    cli.daily(book, Config(watchlist=[("YYY", "XXX")]), now=_after_close(prices), sleep=_no_sleep)
    assert book.last_run_warnings == []


# ---------------------------------------------------------------- post-commit notifier
def _book_json(tmp_path, name, orders, warnings=()):
    f = tmp_path / name
    f.write_text(json.dumps({"cash": 1.0, "orders": orders, "last_run_warnings": list(warnings)}))
    return f


def _order(pair, signal, execute):
    return {"pair": pair, "event": "close", "signal_date": signal, "execute_on": execute,
            "z": 0.1, "legs": [{"ticker": pair[:2], "side": "SELL", "qty": 1, "ref_price": 1.0}]}


def test_notifier_posts_only_orders_new_since_last_commit(tmp_path, monkeypatch):
    a, b = _order("AA/BB", "2026-10-19", "2026-10-20"), _order("CC/DD", "2026-10-20", "2026-10-21")
    old = _book_json(tmp_path, "old.json", [a])
    new = _book_json(tmp_path, "new.json", [a, b], ["⚠ no price data for ZZ/XX"])
    posted = []
    monkeypatch.setattr(notify, "post_discord", lambda text, url: posted.append(text) or True)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    notify.main([str(old), str(new)])
    (text,) = posted
    assert "CC/DD" in text and "AA/BB" not in text and "ZZ/XX" in text


def test_notifier_first_run_and_quiet_day(tmp_path, monkeypatch):
    a = _order("AA/BB", "2026-10-19", "2026-10-20")
    new = _book_json(tmp_path, "new.json", [a])
    posted = []
    monkeypatch.setattr(notify, "post_discord", lambda text, url: posted.append(text) or True)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    notify.main([str(tmp_path / "missing.json"), str(new)])  # no previous book
    assert len(posted) == 1 and "AA/BB" in posted[0]
    posted.clear()
    notify.main([str(new), str(new)])  # nothing new, no warnings
    assert posted == []


def test_order_sheet_sorted_by_session(tmp_path):
    orders = [_order("AA/BB", "2026-10-19", "2026-10-20"), _order("CC/DD", "2026-10-15", "2026-10-16"),
              _order("EE/FF", "2026-10-19", "2026-10-20")]
    text = notify.format_orders(orders, today=pd.Timestamp("2026-10-01").date())
    assert text.count("Orders for") == 2
    assert text.index("2026-10-16") < text.index("Orders for Tue 2026-10-20")


def test_fractional_share_books_are_rejected(tmp_path):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"cash": 1.0, "positions": {"A/B": {"y_shares": -83.7, "x_shares": 2}}}))
    with pytest.raises(ValueError, match="whole shares"):
        PaperBook.load(f)


# ---------------------------------------------------------------- Oct 27 fixes
def test_decide_respects_can_enter():
    from pairs.signals import decide
    assert decide(0, False, 2.5, 2, 0.5, 4, can_enter=False) == "flat"
    assert decide(-1, False, 0.1, 2, 0.5, 4, can_enter=False) == "exit"  # exits still work


def test_backtest_never_enters_with_negative_hedge_ratio():
    import numpy as np

    from pairs.backtest import BacktestConfig, signal_frame
    rng = np.random.default_rng(5)
    n = 600
    log_x = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] * 0.9 + rng.normal(0, 0.01)
    idx = pd.bdate_range("2023-01-02", periods=n)
    y = pd.Series(np.exp(5 - 0.8 * log_x + ou), idx)  # moves opposite to X
    fr = signal_frame(y, pd.Series(np.exp(log_x), idx), BacktestConfig())
    assert (fr["beta"].dropna() < 0).all()
    assert fr["z"].abs().max() > 2  # it does diverge...
    assert (fr["position"] == 0).all()  # ...but is never traded


@pytest.mark.parametrize("now, expected", [
    ("2026-10-26 20:50", "2026-10-26"),  # Monday evening: Monday's close
    ("2026-10-26 10:00", "2026-10-23"),  # Monday morning: Friday's close
    ("2026-10-24 12:00", "2026-10-23"),  # Saturday
    ("2026-11-26 20:00", "2026-11-25"),  # Thanksgiving evening
])
def test_last_completed_session(now, expected):
    from pairs.orders import last_completed_session
    assert last_completed_session(pd.Timestamp(now, tz="America/New_York")) == expected


def _daily_with(prices_end, tmp_path, monkeypatch, coint_pair):
    from pairs import cli
    from pairs.config import Config

    y, x = coint_pair
    full = pd.concat([y.rename("YYY"), x.rename("XXX")], axis=1)
    full.index = pd.bdate_range(end="2026-10-26", periods=len(full))
    calls = []
    monkeypatch.setattr(cli, "load_prices",
                        lambda *a, **k: calls.append(1) or full.loc[:prices_end])
    monkeypatch.setattr(cli, "borrow_table", lambda t: {})
    monkeypatch.setattr(cli, "post_discord", lambda *a: True)
    slept = []
    book = PaperBook.load(tmp_path / "b.json")
    cli.daily(book, Config(watchlist=[("YYY", "XXX")]),
              now=pd.Timestamp("2026-10-26 20:50", tz="America/New_York"),
              sleep=slept.append, attempts=3)
    return book, calls, slept


def test_daily_retries_and_warns_when_yahoo_is_behind(tmp_path, monkeypatch, coint_pair):
    book, calls, slept = _daily_with("2026-10-23", tmp_path, monkeypatch, coint_pair)
    assert len(calls) == 3 and len(slept) == 2  # retried, waiting between attempts
    assert any("2026-10-26" in w and "2026-10-23" in w for w in book.last_run_warnings)
    assert book.last_bar["YYY/XXX"] == "2026-10-23"  # used what it had; next run catches up


def test_daily_fresh_data_no_retry(tmp_path, monkeypatch, coint_pair):
    book, calls, slept = _daily_with("2026-10-26", tmp_path, monkeypatch, coint_pair)
    assert len(calls) == 1 and slept == [] and book.last_run_warnings == []


def test_notifier_posts_a_warning_once_not_every_run(tmp_path, monkeypatch):
    w = ["⚠ Yahoo hasn't published the 2026-10-26 close yet"]
    old = _book_json(tmp_path, "old.json", [], w)
    new = _book_json(tmp_path, "new.json", [], w)
    posted = []
    monkeypatch.setattr(notify, "post_discord", lambda text, url: posted.append(text) or True)
    notify.main([str(old), str(new)])  # catch-up run, same warning as before
    assert posted == []
    notify.main([str(tmp_path / "none.json"), str(new)])  # warning is new
    assert len(posted) == 1
