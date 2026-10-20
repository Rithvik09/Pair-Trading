"""Command line: scan for pairs, or run the daily step on the config.toml watchlist.

    uv run python -m pairs.cli scan
    uv run python -m pairs.cli step            # update state/book.json, print new orders
    uv run python -m pairs.cli step --notify   # ...and post them to Discord

The hosted job doesn't use --notify: it commits the book first and then posts with
`python -m pairs.notify`, so a failed commit can't leave orders posted but unsaved.
"""

import argparse
import os

import pandas as pd

from pairs.backtest import BacktestConfig, signal_frame
from pairs.borrow import BorrowInfo, borrow_table
from pairs.config import Config, load_config
from pairs.data import UNIVERSE, load_prices
from pairs.notify import format_orders, post_discord
from pairs.paper import PaperBook, pair_key
from pairs.paths import BOOK_PATH
from pairs.scanner import scan


def replay(book: PaperBook, pair, y_px: pd.Series, x_px: pd.Series, frame: pd.DataFrame,
           cfg: BacktestConfig, notional: float,
           borrow: dict[str, BorrowInfo] | None = None) -> list[str]:
    """Step one pair through every bar since it was last stepped.

    If the daily run is skipped, a stop or exit that happened in between is still caught
    on the day it happened. A pair stepped for the first time only sees the latest bar.
    """
    k = pair_key(pair)
    legs = [(borrow or {}).get(t) for t in pair]
    shortable = tuple(b.shortable if b else None for b in legs)
    fees = tuple(b.fee_rate if b else 0.0 for b in legs)
    if k in book.positions:
        book.rebase_entry(k, y_px, x_px)
    since = book.last_bar.get(k)
    bars = frame.index[-1:] if since is None else frame.index[frame.index > pd.Timestamp(since)]
    actions = []
    for t in bars:
        # an open trade keeps the hedge ratio it was entered with
        beta = book.positions[k]["beta"] if k in book.positions else frame.at[t, "beta"]
        a = book.step(pair, date=t.date(), y_px=y_px[t], x_px=x_px[t], beta=beta,
                      z=frame.at[t, "z"], entry=cfg.entry, exit=cfg.exit, stop=cfg.stop,
                      notional=notional, cost_bps=cfg.cost_bps, shortable=shortable,
                      borrow_fee=fees)
        if a not in ("none", "hold", "blocked"):
            actions.append(f"{t.date()}: {a}")
    if len(frame):
        book.last_bar[k] = str(frame.index[-1].date())
    return actions


def run_step(book: PaperBook, config: Config, period: str = "2y"):
    """Fetch latest prices and catch every watchlist pair up to the latest close."""
    cfg, notional = config.strategy, config.notional
    tickers = sorted({t for pair in config.watchlist for t in pair})
    if not tickers:
        return []
    prices = load_prices(tickers, period=period, refresh=True)
    borrow = borrow_table(tickers)
    results = []
    for y, x in config.watchlist:
        if y not in prices or x not in prices:
            results.append((y, x, "no data", float("nan")))
            continue
        df = prices[[y, x]].dropna()
        frame = signal_frame(df[y], df[x], cfg)
        actions = replay(book, (y, x), df[y], df[x], frame, cfg, notional, borrow)
        holding = pair_key((y, x)) in book.positions
        state = "; ".join(actions) or ("holding" if holding else "no trade")
        results.append((y, x, state, frame["z"].iloc[-1]))
    book.save()
    return results


def daily(book: PaperBook, config: Config, webhook_url: str | None = None) -> list[dict]:
    """The scheduled job: step every pair, save the book, post any new orders.
    Returns the orders created by this run."""
    first_new = len(book.orders)
    results = run_step(book, config)
    for y, x, action, z in results:
        print(f"{pair_key((y, x))}: z={z:+.2f}  {action}")
    missing = [pair_key((y, x)) for y, x, action, _ in results if action == "no data"]
    if results and len(missing) == len(results):
        # fail the run (GitHub emails on failure) rather than silently skip a day
        raise RuntimeError(f"No price data for any watchlist pair: {', '.join(missing)}")
    book.last_run_warnings = ([(f"⚠ no price data for {', '.join(missing)}: check the tickers "
                                "in config.toml")] if missing else [])
    book.save()
    if book.last_run_warnings:
        post_discord("\n".join(book.last_run_warnings), webhook_url)
    new = book.orders[first_new:]
    print(f"cash ${book.cash:,.2f}, open positions: {len(book.positions)}, "
          f"new orders: {len(new)}")
    if new:
        sheet = format_orders(new)
        print(sheet)
        post_discord(sheet, webhook_url)
    return new


def main() -> None:
    p = argparse.ArgumentParser(prog="pairs")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="find cointegrated pairs in the built-in universe")
    s.add_argument("--period", default="3y")
    s.add_argument("--max-pvalue", type=float, default=0.05)
    t = sub.add_parser("step", help="run the daily step for the config.toml watchlist")
    t.add_argument("--notify", action="store_true",
                   help="post new orders to the Discord webhook in $DISCORD_WEBHOOK_URL")
    args = p.parse_args()

    if args.cmd == "scan":
        prices = load_prices([t for g in UNIVERSE.values() for t in g], period=args.period)
        print(scan(prices, UNIVERSE, max_pvalue=args.max_pvalue).to_string())
    else:
        config = load_config()
        url = os.environ.get("DISCORD_WEBHOOK_URL") if args.notify else None
        if args.notify and not url:
            raise SystemExit("--notify needs the DISCORD_WEBHOOK_URL environment variable")
        daily(PaperBook.load(BOOK_PATH, starting_cash=config.starting_cash), config, url)


if __name__ == "__main__":
    main()
