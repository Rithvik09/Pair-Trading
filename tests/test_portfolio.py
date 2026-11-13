"""The target-weight paper book each strategy other than pairs runs on."""

import json

import pytest

from pairs import notify
from pairs.paper import PaperBook
from pairs.portfolio import WeightBook

PX = {"AAA": 33.0, "BBB": 70.0}


def _book(tmp_path, cash=100_000.0):
    return WeightBook.load(tmp_path / "trend.json", starting_cash=cash)


def _rebalance(book, targets, prices=PX, date="2026-10-19", cost_bps=0.0, **kw):
    reasons = {t: f"why {t}" for t in targets}
    return book.rebalance(date, prices, targets, reasons, cost_bps, **kw)


# ---------------------------------------------------------------- cash and equity
def test_cash_and_equity_after_a_long_and_a_short(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2}, cost_bps=5)
    assert book.holdings == {"AAA": 1515, "BBB": -285}  # $50,000 / $33, -$20,000 / $70
    traded = 1515 * 33 + 285 * 70
    cost = traded * 5 / 1e4
    assert book.cash == pytest.approx(100_000 - 1515 * 33 + 285 * 70 - cost)
    snap = book.record_equity("2026-10-19", PX)
    assert snap["equity"] == pytest.approx(100_000 - cost)  # nothing moved: down the costs

    snap = book.record_equity("2026-10-20", {"AAA": 35.0, "BBB": 72.0})
    moved = 1515 * 2 - 285 * 2  # the long gained, the short lost
    assert snap["equity"] - book.starting_cash == pytest.approx(moved - cost)


def test_shares_are_whole_and_round_toward_zero(tmp_path):
    book = _book(tmp_path, cash=1_000)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.5})
    assert book.holdings == {"AAA": 15, "BBB": -7}  # 15.15 -> 15, -7.14 -> -7
    assert all(isinstance(n, int) for n in book.holdings.values())


def test_borrow_accrues_per_calendar_day_on_the_short(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2})
    book.record_equity("2026-10-19", PX)
    cash = book.cash
    owed = book.accrue_borrow("2026-10-22", {"BBB": 0.036})  # AAA is long, and has no fee
    assert owed == pytest.approx(285 * 70 * 0.036 * 3 / 360)
    assert book.cash == pytest.approx(cash - owed)
    book.record_equity("2026-10-22", PX)
    assert book.accrue_borrow("2026-10-22", {"BBB": 0.036}) == 0  # a rerun charges nothing more
    assert book.equity[-1]["equity"] - book.starting_cash == pytest.approx(-owed)


def test_equity_snapshot_has_the_same_keys_as_the_pairs_book(tmp_path):
    pairs = PaperBook.load(tmp_path / "book.json").record_equity("2026-10-19", {})
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2})
    snap = book.record_equity("2026-10-19", PX, margin_rate=0.5)
    assert set(snap) == set(pairs)
    assert snap["equity"] == pytest.approx(snap["cash"] + snap["unrealised"])
    assert snap["open"] == 2
    assert snap["gross"] == pytest.approx(1515 * 33 + 285 * 70)
    assert snap["margin"] == pytest.approx(0.5 * snap["gross"])
    assert snap["free"] == pytest.approx(snap["equity"] - snap["margin"])
    book.record_equity("2026-10-19", PX)  # a rerun replaces the day's entry
    assert [e["date"] for e in book.equity] == ["2026-10-19"]


def test_book_round_trips_through_its_file(tmp_path):
    book = _book(tmp_path, cash=50_000)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2})
    book.record_equity("2026-10-19", PX)
    book.save()
    again = WeightBook.load(tmp_path / "trend.json", starting_cash=1)
    assert again.to_dict() == book.to_dict()
    assert again.starting_cash == 50_000 and again.last_rebalance == "2026-10-19"
    assert set(json.loads((tmp_path / "trend.json").read_text())) >= {
        "cash", "starting_cash", "holdings", "orders", "log", "equity", "last_run_warnings",
        "last_rebalance"}


# ---------------------------------------------------------------- when it trades
def test_small_drifts_are_not_traded(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5})
    orders = _rebalance(book, {"AAA": 0.5}, {"AAA": 34.0}, date="2026-11-02")
    assert orders == [] and book.holdings == {"AAA": 1515}  # 23 shares off: inside the band
    orders = _rebalance(book, {"AAA": 0.9}, {"AAA": 34.0}, date="2026-12-01")
    assert len(orders) == 1 and book.holdings == {"AAA": 2687}  # $91,363 / $34


def test_a_zero_target_always_closes_in_full(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2})
    (o,) = _rebalance(book, {"AAA": 0.5, "BBB": 0.0}, date="2026-11-02")
    assert o["legs"] == [{"ticker": "BBB", "side": "BUY TO COVER", "qty": 285, "ref_price": 70.0}]
    (o,) = _rebalance(book, {}, date="2026-12-01")  # dropped from the targets: sold too
    assert o["legs"] == [{"ticker": "AAA", "side": "SELL", "qty": 1515, "ref_price": 33.0}]
    assert book.holdings == {}


def test_a_flip_is_two_orders(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.02})  # 60 shares: a move to -61 would be inside the band
    (o,) = _rebalance(book, {"AAA": -0.02}, date="2026-11-02")
    assert o["legs"] == [{"ticker": "AAA", "side": "SELL", "qty": 60, "ref_price": 33.0},
                         {"ticker": "AAA", "side": "SELL SHORT", "qty": 60, "ref_price": 33.0}]
    (o,) = _rebalance(book, {"AAA": 0.02}, date="2026-12-01")
    assert [(leg["side"], leg["qty"]) for leg in o["legs"]] == [("BUY TO COVER", 60), ("BUY", 60)]
    assert book.holdings == {"AAA": 60}


def test_targets_are_scaled_to_the_margin_the_book_supports(tmp_path):
    book = _book(tmp_path)
    px = {"AAA": 10.0, "BBB": 10.0}
    _rebalance(book, {"AAA": 1.5, "BBB": -1.5}, px, margin_rate=0.5)  # asks for 3x, 2x fits
    assert book.holdings == {"AAA": 10_000, "BBB": -10_000}
    snap = book.record_equity("2026-10-19", px, margin_rate=0.5)
    assert snap["free"] == pytest.approx(0)


def test_a_missing_price_leaves_the_holding_and_warns(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2})
    book.record_equity("2026-10-19", PX)
    orders = _rebalance(book, {"AAA": 0.0, "BBB": 0.0}, {"BBB": 70.0}, date="2026-11-02")
    assert [o["pair"] for o in orders] == ["TREND BBB"]
    assert book.holdings == {"AAA": 1515}
    (w,) = book.last_run_warnings
    assert "AAA" in w and "no price" in w
    snap = book.record_equity("2026-11-02", {"BBB": 70.0})  # valued at its last known price
    assert snap["unrealised"] == pytest.approx(1515 * 33)


# ---------------------------------------------------------------- orders and log
def test_orders_fit_the_order_sheet(tmp_path):
    book = _book(tmp_path)
    orders = _rebalance(book, {"AAA": 0.5, "BBB": -0.2})  # Monday
    assert orders == book.orders
    assert [o["pair"] for o in orders] == ["TREND AAA", "TREND BBB"]
    o = orders[1]
    assert o["event"] == "rebalance"
    assert o["signal_date"] == "2026-10-19" and o["execute_on"] == "2026-10-20"
    assert o["legs"] == [{"ticker": "BBB", "side": "SELL SHORT", "qty": 285, "ref_price": 70.0}]
    text = notify.format_orders(orders)
    assert "**TREND BBB** REBALANCE" in text and "SELL SHORT 285 BBB" in text
    assert "BUY 1515 AAA" in text and "Tue 2026-10-20" in text
    assert "z " not in text  # a rebalance has no z-score to show


def test_every_trade_is_logged_with_its_reason(tmp_path):
    book = _book(tmp_path)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.2})
    assert [(e["date"], e["pair"], e["action"], e["reason"]) for e in book.log] == [
        ("2026-10-19", "TREND AAA", "rebalance", "why AAA"),
        ("2026-10-19", "TREND BBB", "rebalance", "why BBB")]
    assert book.last_rebalance == "2026-10-19"


# ---------------------------------------------------------------- dividends and splits
def _held(tmp_path, weight, px=100.0):
    """A book holding AAA at $100 (100 shares per 0.1 of weight), marked on Monday."""
    book = _book(tmp_path)
    _rebalance(book, {"AAA": weight}, {"AAA": px})
    book.record_equity("2026-10-19", {"AAA": px})
    return book


def test_a_dividend_is_paid_to_a_long(tmp_path):
    """Yahoo's closes are dividend-adjusted: after a $2 dividend, Monday's $100 close reads
    $98. The price drop alone would lose the dividend; the book gets it as cash."""
    book = _held(tmp_path, 0.1)
    assert book.holdings == {"AAA": 100} and book.price_dates == {"AAA": "2026-10-19"}
    before = book.equity[-1]["equity"]
    assert book.credit_dividends({"AAA": 98.0}) == pytest.approx(200)
    after = book.record_equity("2026-10-20", {"AAA": 98.5})["equity"]
    assert after - before == pytest.approx(100 * (98.5 + 2 - 100))  # the share's total return


def test_a_dividend_is_charged_to_a_short(tmp_path):
    book = _held(tmp_path, -0.1)
    before = book.equity[-1]["equity"]
    assert book.credit_dividends({"AAA": 98.0}) == pytest.approx(-200)
    after = book.record_equity("2026-10-20", {"AAA": 100.0})["equity"]
    assert after - before == pytest.approx(-200)  # price unchanged: down the dividend it owes


def test_no_dividend_no_cash(tmp_path):
    book = _held(tmp_path, 0.1)
    cash = book.cash
    assert book.credit_dividends({"AAA": 100.0}) == 0 and book.cash == cash
    assert book.credit_dividends({}) == 0 and book.cash == cash  # no restated close to compare
    assert book.credit_dividends({"AAA": 100.5}) == 0 and book.cash == cash  # a data correction


def test_a_dividend_is_credited_once(tmp_path):
    book = _held(tmp_path, 0.1)
    book.credit_dividends({"AAA": 98.0})
    book.record_equity("2026-10-20", {"AAA": 98.5})
    cash = book.cash
    assert book.price_dates == {"AAA": "2026-10-20"}
    assert book.credit_dividends({"AAA": 98.5}) == 0 and book.cash == cash  # rerun, same close


def test_a_split_changes_the_share_count_not_the_cash(tmp_path):
    book = _held(tmp_path, 0.1)
    cash, before = book.cash, book.equity[-1]["equity"]
    assert book.credit_dividends({"AAA": 50.0}) == 0  # 2-for-1: Monday's $100 now reads $50
    assert book.holdings == {"AAA": 200} and book.cash == cash
    assert book.record_equity("2026-10-20", {"AAA": 50.0})["equity"] == pytest.approx(before)
    (w,) = book.last_run_warnings
    assert "AAA" in w and "split" in w and "100" in w and "200" in w


def test_a_reverse_split_pays_cash_for_the_part_share(tmp_path):
    book = _held(tmp_path, -0.015)  # short 15
    before = book.equity[-1]["equity"]
    book.credit_dividends({"AAA": 1000.0})  # 1-for-10: 1.5 shares, so 1 and $500 of cash
    assert book.holdings == {"AAA": -1}
    assert book.record_equity("2026-10-20", {"AAA": 1000.0})["equity"] == pytest.approx(before)


def test_pending_orders_are_those_whose_session_has_not_passed():
    from pairs.orders import pending_orders
    orders = [{"pair": "A/B", "execute_on": "2026-10-19"}, {"pair": "TREND SPY",
              "execute_on": "2026-10-20"}, {"pair": "C/D", "execute_on": None},
              {"pair": "E/F", "execute_on": "2026-10-21"}]
    assert [o["pair"] for o in pending_orders(orders, "2026-10-20")] == ["TREND SPY", "E/F"]


def test_a_target_too_small_for_one_share_is_logged(tmp_path):
    book = _book(tmp_path, cash=1_000)
    _rebalance(book, {"AAA": 0.5, "BBB": -0.05}, prices={"AAA": 33.0, "BBB": 700.0})
    assert "BBB" not in book.holdings and not [o for o in book.orders if "BBB" in o["pair"]]
    (entry,) = [e for e in book.log if e["pair"] == "TREND BBB"]
    assert entry["action"] == "too small BBB"
    assert entry["reason"] == "why BBB, but -$50 doesn't buy one share at $700.00"
