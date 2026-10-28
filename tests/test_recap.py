"""Why each decision was made, the book's value after every close, and the daily recap."""

import json

import pandas as pd
import pytest

from pairs import cli, notify, paper
from pairs.backtest import BacktestConfig
from pairs.config import Config, load_config
from pairs.paper import PaperBook


def _after_close(prices):
    return pd.Timestamp(prices.index[-1]).tz_localize("America/New_York") + pd.Timedelta(hours=20)


@pytest.fixture
def run(tmp_path, monkeypatch, coint_pair):
    """daily() on a cointegrated pair, replaying ~200 bars; returns (book, prices, posted)."""
    y, x = coint_pair
    prices = pd.concat([y.rename("YYY"), x.rename("XXX")], axis=1)
    monkeypatch.setattr(cli, "load_prices", lambda *a, **k: prices)
    monkeypatch.setattr(cli, "borrow_table", lambda t: {})
    posted = []
    monkeypatch.setattr(cli, "post_discord", lambda text, url: posted.append(text) or True)
    book = PaperBook.load(tmp_path / "book.json")
    book.last_bar["YYY/XXX"] = str(prices.index[-200].date())
    cli.daily(book, Config(watchlist=[("YYY", "XXX")]), webhook_url="https://hook",
              now=_after_close(prices), sleep=lambda s: None)
    return book, prices, posted


# ---------------------------------------------------------------- reasons
def test_every_decision_is_logged_with_its_reason(run):
    book, _, _ = run
    assert book.orders and len(book.log) >= len(book.orders)
    for e in book.log:
        assert set(e) >= {"date", "pair", "action", "z", "reason"} and e["reason"]
    opens = [e for e in book.log if e["action"].startswith("open")]
    assert opens and all("2.0" in e["reason"] for e in opens)  # names the entry threshold
    assert PaperBook.load(book.path).log == book.log  # saved


@pytest.mark.parametrize("action, z, needle", [
    ("open short_spread", 2.31, "short YYY"),
    ("open long_spread", -2.31, "buy YYY"),
    ("close", 0.2, "converged"),
    ("stop", 4.3, "stop"),
    ("skip: max open positions", 2.4, "10 positions already open"),
    ("skip: hedge ratio not positive", 2.4, "hedge ratio -0.30"),
    ("unshortable XXX", 2.4, "can't borrow XXX"),
    ("too small XXX", 2.4, "less than one share of XXX"),
])
def test_explain(action, z, needle):
    text = cli.explain(("YYY", "XXX"), action, z, -0.3, BacktestConfig(), max_open=10)
    assert needle in text and f"{z:+.2f}" in text


def test_log_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "LOG_KEEP", 3)
    book = PaperBook.load(tmp_path / "b.json")
    book.log = [{"date": str(i)} for i in range(5)]
    book.save()
    assert [e["date"] for e in PaperBook.load(tmp_path / "b.json").log] == ["2", "3", "4"]


# ---------------------------------------------------------------- equity
def test_equity_is_recorded_once_per_close(run):
    book, prices, _ = run
    last = book.equity[-1]
    assert last["date"] == str(prices.index[-1].date())
    assert last["equity"] == pytest.approx(last["cash"] + last["unrealised"])
    assert last["open"] == len(book.positions)
    n = len(book.equity)
    cli.daily(book, Config(watchlist=[("YYY", "XXX")]), now=_after_close(prices),
              sleep=lambda s: None)  # catch-up run, same close
    assert len(book.equity) == n


def test_starting_cash_is_kept_in_the_book(tmp_path):
    book = PaperBook.load(tmp_path / "b.json", starting_cash=50_000)
    book.save()
    assert json.loads((tmp_path / "b.json").read_text())["starting_cash"] == 50_000
    assert PaperBook.load(tmp_path / "b.json", starting_cash=1).starting_cash == 50_000


# ---------------------------------------------------------------- recap
def _book(equity=(), log=(), orders=(), warnings=()):
    return {"starting_cash": 100_000.0, "cash": 100_000.0, "equity": list(equity),
            "log": list(log), "orders": list(orders), "last_run_warnings": list(warnings)}


def _eq(d, equity, open_=1):
    return {"date": d, "cash": 100_000.0, "unrealised": equity - 100_000.0, "equity": equity,
            "open": open_}


def test_recap_summarises_the_day():
    day1, day2 = _eq("2026-10-27", 100_100.0), _eq("2026-10-28", 100_350.0, open_=2)
    opened = {"date": "2026-10-28", "pair": "SW/PKG", "action": "open long_spread", "z": -2.21,
              "reason": "z -2.21 ≤ -2.0: SW fell behind PKG"}
    skipped = {"date": "2026-10-28", "pair": "KNX/JBHT", "action": "skip: max open positions",
               "z": 2.4, "reason": "z +2.40 crossed the entry, but 10 positions already open"}
    text = notify.recap(_book([day1]), _book([day1, day2], [skipped, opened]))
    assert "Wed 2026-10-28" in text
    assert "$100,350" in text and "+$250" in text and "+0.35%" in text and "2 open" in text
    assert "SW/PKG" in text and "SW fell behind PKG" in text
    assert "KNX/JBHT" in text and "10 positions already open" in text
    assert text.index("SW/PKG") < text.index("KNX/JBHT")  # trades before skips


def test_recap_on_a_quiet_day_still_reports_the_book():
    old = _book([_eq("2026-10-27", 100_000.0)])
    new = _book([_eq("2026-10-27", 100_000.0), _eq("2026-10-28", 99_900.0)])
    text = notify.recap(old, new)
    assert "$99,900" in text and "-$100" in text and "No trades" in text


def test_recap_is_empty_when_nothing_happened():
    b = _book([_eq("2026-10-27", 100_000.0)])
    assert notify.recap(b, b) == ""


def test_notifier_posts_the_recap(tmp_path, monkeypatch):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(_book([_eq("2026-10-27", 100_000.0)])))
    new.write_text(json.dumps(_book([_eq("2026-10-27", 100_000.0), _eq("2026-10-28", 100_020.0)])))
    posted = []
    monkeypatch.setattr(notify, "post_discord", lambda text, url: posted.append(text) or True)
    notify.main([str(old), str(new)])
    (text,) = posted
    assert "$100,020" in text


def test_daily_posts_one_recap_with_reasons_and_orders(run):
    book, _, posted = run
    (text,) = posted
    assert "Paper book" in text and "Orders for" in text
    assert any(e["reason"] in text for e in book.log)


# ---------------------------------------------------------------- review fixes
def test_rebase_ignores_a_missing_price(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    book.step(("A", "B"), date="2026-10-19", y_px=10, x_px=10, beta=1.0, z=2.5, entry=2,
              exit=0.5, stop=4, notional=10_000)
    idx = pd.to_datetime(["2026-10-19"])
    book.rebase_entry("A/B", pd.Series([float("nan")], idx), pd.Series([11.0], idx))
    p = book.positions["A/B"]
    assert (p["entry_y"], p["entry_x"]) == (10.0, 10.0)  # left alone, not half-rebased


def test_equity_counts_the_cost_of_closing(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    book.step(("A", "B"), date="2026-10-19", y_px=10, x_px=10, beta=1.0, z=2.5, entry=2,
              exit=0.5, stop=4, notional=10_000, cost_bps=5)
    marked = book.record_equity("2026-10-19", {"A": 10.0, "B": 10.0}, cost_bps=5)
    book.step(("A", "B"), date="2026-10-19", y_px=10, x_px=10, beta=1.0, z=0.1, entry=2,
              exit=0.5, stop=4, notional=10_000, cost_bps=5)  # close at the same prices
    assert marked["equity"] == pytest.approx(book.cash)


def test_recap_dates_decisions_from_a_catch_up_run():
    a = {"date": "2026-10-27", "pair": "A/B", "action": "close", "z": 0.1, "reason": "r1"}
    b = {"date": "2026-10-28", "pair": "C/D", "action": "close", "z": 0.1, "reason": "r2"}
    text = notify.recap(_book(), _book([_eq("2026-10-28", 1e5)], [a, b]))
    assert "Tue 2026-10-27" in text and "Wed 2026-10-28" in text
    one_day = notify.recap(_book(), _book([_eq("2026-10-28", 1e5)], [b]))
    assert one_day.count("2026-10-28") == 1  # just the header


# ---------------------------------------------------------------- cash in use
def test_equity_tracks_stock_held_margin_and_free_cash(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    book.step(("A", "B"), date="2026-10-19", y_px=10, x_px=10, beta=1.0, z=2.5, entry=2,
              exit=0.5, stop=4, notional=10_000)  # short 500 A, buy 500 B
    snap = book.record_equity("2026-10-20", {"A": 12.0, "B": 10.0}, margin_rate=0.5)
    assert snap["gross"] == pytest.approx(500 * 12 + 500 * 10)  # both legs, at today's prices
    assert snap["margin"] == pytest.approx(0.5 * snap["gross"])
    assert snap["free"] == pytest.approx(snap["equity"] - snap["margin"])


def test_config_margin_rate(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[paper]\nmargin_rate = 0.3\n")
    assert load_config(f).margin_rate == 0.3
    f.write_text("")
    assert load_config(f).margin_rate == 0.5  # Reg T initial margin
    f.write_text("[paper]\nmargin_rate = 0\n")
    with pytest.raises(ValueError, match="margin_rate"):
        load_config(f)


def test_recap_shows_cash_in_use_and_warns_when_short_of_it():
    def eq(free):
        return {**_eq("2026-10-28", 100_000.0), "gross": 80_000.0, "margin": 40_000.0,
                "free": free}
    text = notify.recap(_book(), _book([eq(60_000.0)]))
    assert "$80,000 of stock" in text and "$40,000 margin" in text and "$60,000 free" in text
    assert "⚠" not in text
    assert "⚠" in notify.recap(_book(), _book([eq(-5_000.0)]))
