"""The sizing calculator: what a trade size and account size add up to."""

import pytest

from pairs import sizing
from pairs.paper import PaperBook


@pytest.mark.parametrize("beta", [0.4, 1.0, 1.24, 3.0])
def test_smallest_trade_matches_the_paper_books_share_rule(tmp_path, beta):
    y_px, x_px = 253.74, 42.24
    need = sizing.smallest_trade(y_px, x_px, beta)
    kw = {"date": "2026-10-26", "y_px": y_px, "x_px": x_px, "beta": beta, "z": 2.5,
          "entry": 2, "exit": 0.5, "stop": 4}
    book = PaperBook.load(tmp_path / "a.json")
    assert book.step(("A", "B"), notional=need * 1.0001, **kw) == "open short_spread"
    book = PaperBook.load(tmp_path / "b.json")
    assert book.step(("A", "B"), notional=need * 0.999, **kw).startswith("too small")


def test_plan_adds_up_the_cash():
    p = sizing.plan(account=60_000, notional=5_000, max_open=10, margin_rate=0.5,
                    smallest={"MTD/A": 3_400.0, "SW/PKG": 900.0, "X/Y": 5_000.0},
                    loss_rates=[-0.046, 0.03, -0.19])
    assert p["gross"] == 50_000 and p["margin"] == 25_000 and p["free"] == 35_000
    assert p["too_small"] == []  # 5,000 buys a share of everything here
    assert p["worst_loss"] == pytest.approx(-0.19 * 5_000)
    assert p["worst_rate"] == -0.19


def test_plan_lists_pairs_the_size_is_too_small_for():
    p = sizing.plan(account=20_000, notional=3_000, max_open=4, margin_rate=0.5,
                    smallest={"MTD/A": 3_400.0, "SW/PKG": 900.0, "B/C": 7_000.0},
                    loss_rates=[])
    assert p["too_small"] == ["B/C", "MTD/A"]  # biggest requirement first
    assert p["worst_loss"] is None and p["worst_rate"] is None  # nothing to go on yet


def test_plan_flags_margin_beyond_the_account():
    p = sizing.plan(account=10_000, notional=10_000, max_open=10, margin_rate=0.5,
                    smallest={}, loss_rates=[])
    assert p["free"] == -40_000


def test_loss_rates_come_from_closed_and_open_trades():
    closed = [{"pnl": -456.0, "notional": 10_000.0}, {"pnl": 120.0, "notional": 10_000.0},
              {"pnl": 5.0}]  # no notional: skipped
    open_ = [(-1_870.0, 10_000.0)]
    assert sizing.loss_rates(closed, open_) == pytest.approx([-0.0456, 0.012, -0.187])
