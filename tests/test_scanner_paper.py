import numpy as np
import pandas as pd
import pytest

from pairs.paper import PaperBook
from pairs.scanner import scan


def test_scan_finds_planted_pair(coint_pair, random_walks):
    y, x = coint_pair
    a, b = random_walks
    prices = pd.concat([y, x, a, b], axis=1)
    res = scan(prices, max_pvalue=0.05)
    assert len(res) >= 1
    top = res.iloc[0]
    assert {top["y"], top["x"]} == {"YYY", "XXX"}


def test_scan_only_compares_within_groups(coint_pair, random_walks):
    y, x = coint_pair
    a, b = random_walks
    prices = pd.concat([y, x, a, b], axis=1)
    res = scan(prices, groups={"g1": ["YYY", "AAA"], "g2": ["XXX", "BBB"]}, max_pvalue=1.0)
    pairs = {frozenset((r.y, r.x)) for r in res.itertuples()}
    assert pairs == {frozenset(("YYY", "AAA")), frozenset(("XXX", "BBB"))}


def test_paper_book_open_close_and_persist(tmp_path):
    path = tmp_path / "book.json"
    book = PaperBook.load(path, starting_cash=100_000)
    key = ("KO", "PEP")

    # z = +2.5 -> KO rich vs PEP: short KO, buy PEP
    act = book.step(key, date="2026-01-02", y_px=60.0, x_px=170.0, beta=1.0, z=2.5,
                    entry=2.0, exit=0.5, stop=4.0, notional=10_000)
    assert act == "open short_spread"
    pos = book.positions["KO/PEP"]
    assert pos["y_shares"] < 0 < pos["x_shares"]

    # holds while still diverged
    assert book.step(key, date="2026-01-03", y_px=61.0, x_px=170.0, beta=1.0, z=1.5,
                     entry=2.0, exit=0.5, stop=4.0, notional=10_000) == "hold"

    # converges: KO falls back, PEP rises -> close for a profit
    act = book.step(key, date="2026-01-09", y_px=58.0, x_px=173.0, beta=1.0, z=0.2,
                    entry=2.0, exit=0.5, stop=4.0, notional=10_000)
    assert act == "close"
    assert "KO/PEP" not in book.positions
    pnl = book.closed[-1]["pnl"]
    exp = -pos["y_shares"] * (60 - 58) + pos["x_shares"] * (173 - 170)
    assert pnl == pytest.approx(exp)
    assert book.cash == pytest.approx(100_000 + pnl)

    book.save()
    again = PaperBook.load(path)
    assert again.cash == pytest.approx(book.cash)
    assert len(again.closed) == 1


def test_paper_book_ignores_nan_z(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    assert book.step(("A", "B"), date="d", y_px=1, x_px=1, beta=1, z=np.nan,
                     entry=2, exit=0.5, stop=4, notional=1000) == "none"
