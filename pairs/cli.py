"""Command line: scan for pairs, or run the daily step on the config.toml watchlist.

    uv run python -m pairs.cli scan            # cointegrated pairs in the S&P 500
    uv run python -m pairs.cli shortlist       # ...that also pass the backtest filters
    uv run python -m pairs.cli step            # update state/book.json, print new orders
    uv run python -m pairs.cli step --notify   # ...and post them to Discord

The hosted job doesn't use --notify: it commits the book first and then posts with
`python -m pairs.notify`, so a failed commit can't leave orders posted but unsaved.
"""

import argparse
import os
import time

import pandas as pd

from pairs.backtest import BacktestConfig, signal_frame
from pairs.borrow import BorrowInfo, borrow_table
from pairs.config import Config, load_config
from pairs.data import UNIVERSE, load_prices, load_universe, update_universe
from pairs.notify import format_orders, post_discord
from pairs.orders import last_completed_session
from pairs.paper import PaperBook, pair_key
from pairs.paths import BOOK_PATH
from pairs.scanner import scan, shortlist


def replay(book: PaperBook, pair, y_px: pd.Series, x_px: pd.Series, frame: pd.DataFrame,
           cfg: BacktestConfig, notional: float,
           borrow: dict[str, BorrowInfo] | None = None,
           max_open: int | None = None) -> list[str]:
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
                      borrow_fee=fees, max_open=max_open)
        if a not in ("none", "hold", "blocked"):
            actions.append(f"{t.date()}: {a}")
    if len(frame):
        book.last_bar[k] = str(frame.index[-1].date())
    return actions


def load_fresh_prices(tickers, period, now, attempts=3, wait=300, sleep=time.sleep):
    """Prices through the latest completed session. Yahoo sometimes publishes a day's close
    late, so retry a few times; if it's still behind, return what there is plus a warning
    (the next run catches up)."""
    expected = last_completed_session(now)
    for i in range(attempts):
        prices = load_prices(tickers, period=period, refresh=True, now=now)
        latest = str(prices.index.max().date()) if len(prices) else None
        if latest is not None and latest >= expected:
            return prices, None
        if i < attempts - 1:
            print(f"Prices end {latest}, expected {expected}: retrying in {wait}s")
            sleep(wait)
    return prices, (f"⚠ Yahoo hasn't published the {expected} close yet (latest: {latest}). "
                    "Orders are based on the latest close available; the next run catches up.")


def run_step(book: PaperBook, config: Config, period: str = "2y", now=None,
             sleep=time.sleep, attempts: int = 3):
    """Fetch latest prices and catch every watchlist pair up to the latest close.
    Returns (per-pair results, warnings)."""
    cfg, notional = config.strategy, config.notional
    tickers = sorted({t for pair in config.watchlist for t in pair})
    if not tickers:
        return [], []
    now = now or pd.Timestamp.now(tz="America/New_York")
    prices, stale = load_fresh_prices(tickers, period, now, attempts=attempts, sleep=sleep)
    borrow = borrow_table(tickers)
    results = []
    for y, x in config.watchlist:
        if y not in prices or x not in prices:
            results.append((y, x, "no data", float("nan")))
            continue
        df = prices[[y, x]].dropna()
        frame = signal_frame(df[y], df[x], cfg)
        actions = replay(book, (y, x), df[y], df[x], frame, cfg, notional, borrow,
                         config.max_open_positions)
        holding = pair_key((y, x)) in book.positions
        state = "; ".join(actions) or ("holding" if holding else "no trade")
        results.append((y, x, state, frame["z"].iloc[-1]))
    book.save()
    return results, [stale] if stale else []


def daily(book: PaperBook, config: Config, webhook_url: str | None = None, now=None,
          sleep=time.sleep, attempts: int = 3) -> list[dict]:
    """The scheduled job: step every pair, save the book, post any new orders.
    Returns the orders created by this run."""
    first_new = len(book.orders)
    results, notes = run_step(book, config, now=now, sleep=sleep, attempts=attempts)
    for y, x, action, z in results:
        print(f"{pair_key((y, x))}: z={z:+.2f}  {action}")
    missing = [pair_key((y, x)) for y, x, action, _ in results if action == "no data"]
    capped = [pair_key((y, x)) for y, x, action, _ in results
              if "skip: max open positions" in action]
    if capped:
        notes = notes + [(f"⚠ signal skipped, {config.max_open_positions} positions already "
                         f"open: {', '.join(capped)}")]
    if results and len(missing) == len(results):
        # fail the run (GitHub emails on failure) rather than silently skip a day
        raise RuntimeError(f"No price data for any watchlist pair: {', '.join(missing)}")
    book.last_run_warnings = notes + ([(f"⚠ no price data for {', '.join(missing)}: check the "
                                        "tickers in config.toml")] if missing else [])
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
    s = sub.add_parser("scan", help="find cointegrated pairs")
    s.add_argument("--max-pvalue", type=float, default=0.05)
    sl = sub.add_parser("shortlist", help="pairs worth trading (filters + backtest)")
    for cmd in (s, sl):
        cmd.add_argument("--period", default="3y")
        cmd.add_argument("--universe", choices=["sp500", "core"], default="sp500",
                         help="sp500: S&P 500 by sub-industry (sp500.csv); "
                              "core: the hand-picked groups in pairs/data.py")
    sub.add_parser("update-universe", help="refresh sp500.csv from Wikipedia "
                                           "(run with `uv run --with lxml`)")
    t = sub.add_parser("step", help="run the daily step for the config.toml watchlist")
    t.add_argument("--notify", action="store_true",
                   help="post new orders to the Discord webhook in $DISCORD_WEBHOOK_URL")
    args = p.parse_args()

    if args.cmd == "update-universe":
        out = update_universe()
        print(f"{len(out)} tickers, {out['sub_industry'].nunique()} sub-industries. "
              "Review the diff before committing.")
    elif args.cmd in ("scan", "shortlist"):
        groups = load_universe() if args.universe == "sp500" else UNIVERSE
        tickers = sorted({t for g in groups.values() for t in g})
        prices = load_prices(tickers, period=args.period)
        if args.cmd == "scan":
            print(scan(prices, groups, max_pvalue=args.max_pvalue).to_string())
            return
        fees = {t: b.fee_rate for t, b in borrow_table(tickers).items()}
        out = shortlist(prices, groups, load_config().strategy, borrow_fees=fees)
        print(out.round(3).to_string())
        print("\nconfig.toml watchlist lines:")
        print("\n".join(f'  ["{r.y}", "{r.x}"],' for r in out.itertuples()))
    else:
        config = load_config()
        url = os.environ.get("DISCORD_WEBHOOK_URL") if args.notify else None
        if args.notify and not url:
            raise SystemExit("--notify needs the DISCORD_WEBHOOK_URL environment variable")
        daily(PaperBook.load(BOOK_PATH, starting_cash=config.starting_cash), config, url)


if __name__ == "__main__":
    main()
