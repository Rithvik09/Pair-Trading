import pytest

from pairs.backtest import BacktestConfig, backtest
from pairs.borrow import DEFAULT_FEE, BorrowInfo, borrow_table, parse_ibkr
from pairs.paper import PaperBook

IBKR_SAMPLE = """#BOF|2026.10.16|09:15:02
#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|
KO|USD|COCA-COLA CO/THE|8894|US1912161007|3.9814|0.2500|>10000000|
GME|USD|GAMESTOP CORP-CLASS A|36285627|US36467W1099|-12.3|16.2000|150000|
XYZ|USD|NO BORROW INC|1|US0000000001|0|NA|0|
#EOF|3
"""


def test_parse_ibkr_feed():
    t = parse_ibkr(IBKR_SAMPLE)
    assert t["KO"] == BorrowInfo(shortable=True, fee_rate=0.0025, available=10_000_000,
                                 source="ibkr")
    assert t["GME"].fee_rate == pytest.approx(0.162)
    assert t["GME"].available == 150_000
    assert t["XYZ"].shortable is False


def test_borrow_table_precedence(tmp_path):
    overrides = tmp_path / "borrow.csv"
    overrides.write_text("ticker,shortable,fee_rate\nGME,false,\nPEP,,0.01\n")
    t = borrow_table(["KO", "GME", "PEP", "ZZZ"], feed_text=IBKR_SAMPLE,
                     overrides_path=overrides)
    assert t["KO"].source == "ibkr"
    assert t["GME"].shortable is False and t["GME"].source == "override"
    assert t["GME"].fee_rate == pytest.approx(0.162)  # blank override keeps feed value
    # PEP is absent from the feed (=> IBKR has none to lend) but the fee is overridden
    assert t["PEP"].fee_rate == pytest.approx(0.01) and t["PEP"].shortable is False
    assert t["ZZZ"].shortable is False


def test_borrow_table_without_feed_is_unknown(tmp_path):
    t = borrow_table(["KO"], feed_text=None, overrides_path=tmp_path / "none.csv")
    assert t["KO"] == BorrowInfo(shortable=None, fee_rate=DEFAULT_FEE, available=None,
                                 source="default")


def test_backtest_charges_borrow_only_on_short_leg(coint_pair):
    y, x = coint_pair
    cfg = BacktestConfig(cost_bps=0)
    free = backtest(y, x, cfg)
    fee_y = backtest(y, x, cfg, borrow_fees=(0.10, 0.0))
    held = free.position.shift(1).fillna(0)
    # Y is only short in a short-spread trade (beta > 0 here)
    assert (fee_y.borrow_cost[held != -1] == 0).all()
    assert (fee_y.borrow_cost[held == -1] > 0).all()
    assert fee_y.stats["total_return"] < free.stats["total_return"]
    assert fee_y.stats["borrow_paid"] > 0 and free.stats["borrow_paid"] == 0


def test_backtest_borrow_accrues_per_calendar_day(coint_pair):
    """Same convention as the paper book: actual/360, so a Monday bar carries the weekend."""
    y, x = coint_pair
    res = backtest(y, x, BacktestConfig(cost_bps=0), borrow_fees=(0.0, 0.36))
    held = res.position.shift(1).fillna(0)
    pos = res.frame["position"]
    opened = pos.ne(pos.shift(1, fill_value=0)) & pos.ne(0)
    beta = res.frame["beta"].where(opened).ffill().shift(1)
    bars = res.borrow_cost[held == 1].index
    monday = next(t for t in bars if t.dayofweek == 0)
    tuesday = next(t for t in bars if t.dayofweek == 1)
    # long spread shorts X at weight w_x; 0.36/360 = 0.001 per calendar day
    assert res.borrow_cost[tuesday] == pytest.approx(beta[tuesday] / (1 + beta[tuesday]) * 0.001)
    assert res.borrow_cost[monday] == pytest.approx(beta[monday] / (1 + beta[monday]) * 0.003)


def _step(book, z, date, y_px=100.0, x_px=100.0, **kw):
    args = {"beta": 1.0, "entry": 2, "exit": 0.5, "stop": 4, "notional": 10_000}
    return book.step(("A", "B"), date=date, y_px=y_px, x_px=x_px, z=z, **(args | kw))


def test_paper_refuses_trade_when_short_leg_unborrowable(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    # short spread needs to short A
    assert _step(book, 2.5, "2026-01-02", shortable=(False, True)) == "unshortable A"
    assert book.positions == {}
    # long spread needs to short B, which is fine
    assert _step(book, -2.5, "2026-01-02", shortable=(False, True)) == "open long_spread"
    # unknown (None) does not block
    book2 = PaperBook.load(tmp_path / "c.json")
    assert _step(book2, 2.5, "2026-01-02", shortable=(None, None)) == "open short_spread"


def test_paper_accrues_borrow_per_calendar_day(tmp_path):
    book = PaperBook.load(tmp_path / "b.json", starting_cash=100_000)
    _step(book, 2.5, "2026-01-02", borrow_fee=(0.036, 0.5))  # short A: $5,000
    assert book.borrow_accrued("A/B", "2026-01-12") == pytest.approx(5.0)  # 5000*.036*10/360
    _step(book, 0.0, "2026-01-12", borrow_fee=(0.036, 0.5))
    closed = book.closed[-1]
    assert closed["borrow"] == pytest.approx(5.0)
    assert closed["pnl"] == pytest.approx(-5.0)  # prices flat, no commissions
    assert book.cash == pytest.approx(100_000 - 5.0)


def test_paper_skips_negative_hedge_ratio(tmp_path):
    """A negative hedge ratio would short (or buy) both legs: a directional bet, not a pair
    trade. The strategy only trades pairs that move together, so it skips these."""
    book = PaperBook.load(tmp_path / "b.json")
    for beta in (-1.0, 0.0):
        assert _step(book, -2.5, "2026-01-02", beta=beta) == "skip: hedge ratio not positive"
        assert _step(book, 2.5, "2026-01-02", beta=beta) == "skip: hedge ratio not positive"
    assert book.positions == {} and book.orders == []


def test_unparseable_dates_accrue_nothing(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    _step(book, 2.5, "d1", borrow_fee=(1.0, 1.0))
    assert book.borrow_accrued("A/B", "d2") == 0.0


def test_old_books_without_borrow_fields_still_load(tmp_path):
    p = tmp_path / "old.json"
    p.write_text('{"cash": 1000, "positions": {"A/B": {"y": "A", "x": "B", '
                 '"side": "short_spread", "entry_date": "2026-01-02", "entry_y": 1, '
                 '"entry_x": 1, "y_shares": -1, "x_shares": 1, "beta": 1, "entry_z": 2.5}}}')
    book = PaperBook.load(p)
    assert book.borrow_accrued("A/B", "2026-02-02") == 0.0


def test_override_file_comments_are_ignored(tmp_path):
    f = tmp_path / "borrow.csv"
    f.write_text("ticker,shortable,fee_rate\n# GME,true,0.15\nKO,false,\n")
    t = borrow_table(["KO", "GME"], feed_text=None, overrides_path=f)
    assert t["KO"].shortable is False
    assert t["GME"].source == "default"


def test_class_share_symbols_match_across_yahoo_and_ibkr(tmp_path):
    feed = IBKR_SAMPLE.replace("XYZ|", "BRK B|").replace("#EOF|3", "BF.B|USD|x|1|x|0|0.30|500|\n#EOF|4")
    t = borrow_table(["BRK-B", "BF-B"], feed_text=feed, overrides_path=tmp_path / "n.csv")
    assert t["BRK-B"].source == "ibkr" and t["BRK-B"].shortable is False  # 0 available
    assert t["BF-B"].shortable is True and t["BF-B"].fee_rate == pytest.approx(0.003)


def test_garbage_feed_is_treated_as_unavailable(tmp_path):
    t = borrow_table(["KO"], feed_text="<html>503 Service Unavailable</html>",
                     overrides_path=tmp_path / "n.csv")
    assert t["KO"].source == "default" and t["KO"].shortable is None


def test_parse_ibkr_edge_values():
    t = parse_ibkr("ko|USD|x|1|x|0|nan||\nAB|USD|x|1|x|0|0.5|n/a|\n")
    assert t["KO"].fee_rate == DEFAULT_FEE  # NaN fee rejected, symbol upper-cased
    assert t["KO"].shortable is None and t["AB"].shortable is None  # unknown availability


@pytest.mark.parametrize("row, msg", [("KO,,5", "decimal"), ("KO,ture,", "shortable"),
                                      ("KO,,abc", "fee_rate")])
def test_bad_override_values_raise_clearly(tmp_path, row, msg):
    f = tmp_path / "borrow.csv"
    f.write_text(f"ticker,shortable,fee_rate\n{row}\n")
    with pytest.raises(ValueError, match=msg):
        borrow_table(["KO"], feed_text=None, overrides_path=f)


def test_override_accepts_common_spellings(tmp_path):
    f = tmp_path / "borrow.csv"
    f.write_text("ticker,shortable,fee_rate\nKO, TRUE ,\nPEP,No,\n")
    t = borrow_table(["KO", "PEP"], feed_text=None, overrides_path=f)
    assert t["KO"].shortable is True and t["PEP"].shortable is False
