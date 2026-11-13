"""The daily recap (book value, decisions and why, order sheet), posted to Discord via webhook.

Stdlib only on purpose: the hosted job runs this module with the webhook secret and without
installing any third-party package.

    python -m pairs.notify OLD_STATE_DIR NEW_STATE_DIR   # post the recap of what's new in NEW
    python -m pairs.notify OLD_BOOK.json NEW_BOOK.json   # ...for the pairs book alone
"""

import json
import os
import sys
import urllib.request
from datetime import date
from itertools import groupby
from pathlib import Path

DISCORD_LIMIT = 2000  # characters per message

EVENT_LABELS = {
    "open short_spread": "OPEN",
    "open long_spread": "OPEN",
    "close": "CLOSE (converged)",
    "stop": "STOP-LOSS EXIT",
    "replace": "CLOSE (swapped for a better trade)",
    "rebalance": "REBALANCE",  # a strategy book traded back to its target weights
}
PAIRS_BOOK = "book"  # the pairs book's file name in the state directory, without .json


def _day(iso: str | None) -> str:
    if not iso:
        return "next session"
    return date.fromisoformat(iso).strftime("%a ") + iso


def format_orders(orders: list[dict], today: date | None = None) -> str:
    """Order sheet grouped by the session to execute in, e.g.

    **Orders for Tue 2026-10-20, at the open**
    **KO/PEP** OPEN (z +2.50, signal from Mon 2026-10-19 close)
    • SELL SHORT 83 KO  (ref $60.00)
    """
    today = today or date.today()
    out = []
    orders = sorted(orders, key=lambda o: (o["execute_on"] or "", o["pair"]))
    for when, group in groupby(orders, key=lambda o: o["execute_on"]):
        late = when is not None and date.fromisoformat(when) < today
        out.append(f"**Orders for {_day(when)}, at the open**"
                   + (" ⚠ LATE: that session has passed, check prices first" if late else ""))
        for o in group:
            z = "" if o["event"] == "rebalance" else f"z {o['z']:+.2f}, "  # no z-score behind it
            out.append(f"**{o['pair']}** {EVENT_LABELS.get(o['event'], o['event'])} "
                       f"({z}signal from {_day(o['signal_date'])} close)")
            out.extend(f"• {leg['side']} {leg['qty']} {leg['ticker']}  (ref ${leg['ref_price']:,.2f})"
                       for leg in o["legs"])
        out.append("")
    return "\n".join(out).rstrip()


def _chunks(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Split on line boundaries so no message exceeds Discord's limit."""
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # pathological single line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks


def post_discord(text: str, webhook_url: str | None, timeout: float = 15.0) -> bool:
    """Post text to a Discord webhook. Returns False (and sends nothing) without a URL."""
    if not webhook_url:
        return False
    for chunk in _chunks(text):
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps({"content": chunk}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "pairs-trader"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    return True


def _order_id(o: dict) -> tuple:
    return o["pair"], o["event"], o["signal_date"]


def new_orders(old_book: dict, new_book: dict) -> list[dict]:
    """Orders in new_book that weren't in old_book (the previous commit)."""
    seen = {_order_id(o) for o in old_book.get("orders", [])}
    return [o for o in new_book.get("orders", []) if _order_id(o) not in seen]


def _money(v: float) -> str:
    return f"{'-' if v < 0 else '+'}${abs(v):,.0f}"


def _log_id(e: dict) -> tuple:
    return e.get("date"), e.get("pair"), e.get("action")


def _decision_lines(decisions: list[dict]) -> list[str]:
    """Trades first, then the signals that weren't traded. Dated when a catch-up run covers
    more than one close."""
    dated = len({e["date"] for e in decisions}) > 1

    def when(e):
        return f"{_day(e['date'])} " if dated else ""

    traded = [e for e in decisions if e["action"] in EVENT_LABELS]
    skipped = [e for e in decisions if e["action"] not in EVENT_LABELS]
    lines = [f"• {when(e)}**{e['pair']}** {EVENT_LABELS[e['action']]}: {e['reason']}"
             for e in traded]
    if skipped:
        lines.append("Not traded:")
        lines += [f"• {when(e)}{e['pair']}: {e['reason']}" for e in skipped]
    return lines


def recap(old: dict, new: dict, today: date | None = None) -> str:
    """What changed between two saves of the book, as one message: after a new close, the
    book's value and every decision with its reason; any new orders; any new warnings.
    Empty when nothing changed (e.g. a catch-up run on a day that was already done)."""
    parts = []
    seen = {_log_id(e) for e in old.get("log", [])}
    decisions = [e for e in new.get("log", []) if _log_id(e) not in seen]
    equity = new.get("equity", [])
    if equity and equity[-1]["date"] not in {e["date"] for e in old.get("equity", [])}:
        now = equity[-1]
        start = new.get("starting_cash") or now["equity"]
        # since the previous snapshot, which is the previous close unless a run was missed
        day = (f"{_money(now['equity'] - equity[-2]['equity'])} since "
               f"{_day(equity[-2]['date'])}, " if len(equity) > 1 else "")
        lines = [f"**Paper book after the {_day(now['date'])} close**",
                 (f"Value ${now['equity']:,.0f} ({day}{now['equity'] / start - 1:+.2%} since "
                  f"the ${start:,.0f} start) · {now['open']} open")]
        if "margin" in now:  # snapshots from before cash in use was tracked don't have it
            lines.append(f"Cash in use: ${now['gross']:,.0f} of stock held (long + short) "
                         f"needs ${now['margin']:,.0f} margin · ${now['free']:,.0f} free")
            if now["free"] < 0:
                lines.append("⚠ The open trades need more margin than the book is worth: a "
                             "real account would get a margin call. Lower notional in config.toml.")
        lines += _decision_lines(decisions) or ["No trades today."]
        parts.append("\n".join(lines))
    elif decisions:
        parts.append("\n".join(_decision_lines(decisions)))
    if orders := new_orders(old, new):
        parts.append(format_orders(orders, today))
    # a warning is posted when it first appears, not again on every run while it lasts
    parts += [w for w in new.get("last_run_warnings", [])
              if w not in old.get("last_run_warnings", [])]
    return "\n\n".join(parts)


def title(name: str) -> str:
    """A strategy's name as shown to the user, from its book's file name: 'trend' -> 'Trend'."""
    return "Pairs" if name == PAIRS_BOOK else name.replace("_", " ").capitalize()


def _new_close(old: dict, new: dict) -> bool:
    """Whether `new` has a close recorded that `old` doesn't."""
    equity = new.get("equity", [])
    return bool(equity) and equity[-1]["date"] not in {e["date"] for e in old.get("equity", [])}


def strategy_recap(name: str, old: dict, new: dict, today: date | None = None,
                   said: tuple[str, ...] = ()) -> str:
    """The same as recap, for a strategy's book: its value and profit or loss after a new
    close, the trades of a rebalance with their reasons, new orders, new warnings.
    `said` are warnings another book already carries (e.g. Yahoo is late), not to repeat."""
    parts = []
    seen = {_log_id(e) for e in old.get("log", [])}
    decisions = [e for e in new.get("log", []) if _log_id(e) not in seen]
    if _new_close(old, new):
        equity = new["equity"]
        now = equity[-1]
        start = new.get("starting_cash") or now["equity"]
        day = (f" · {_money(now['equity'] - equity[-2]['equity'])} since "
               f"{_day(equity[-2]['date'])}" if len(equity) > 1 else "")
        lines = [f"**{title(name)} after the {_day(now['date'])} close**",
                 (f"Value ${now['equity']:,.0f} · {_money(now['equity'] - start)} "
                  f"({now['equity'] / start - 1:+.2%}) since the ${start:,.0f} start{day} · "
                  f"{now['open']} holdings")]
        parts.append("\n".join(lines + _decision_lines(decisions)))
    elif decisions:
        parts.append("\n".join(_decision_lines(decisions)))
    if orders := new_orders(old, new):
        parts.append(format_orders(orders, today))
    parts += [w for w in new.get("last_run_warnings", [])
              if w not in old.get("last_run_warnings", []) and w not in said]
    return "\n\n".join(parts)


def by_strategy(books: dict[str, dict]) -> str:
    """One line per book (what it's worth, and its profit or loss since it started), then
    all of them together. A book with no close recorded yet counts at its cash."""
    lines, worth, started = ["**By strategy**"], 0.0, 0.0
    for name in sorted(books, key=lambda n: (n != PAIRS_BOOK, n)):
        b = books[name]
        value = b["equity"][-1]["equity"] if b.get("equity") else b.get("cash")
        start = b.get("starting_cash") or value
        if not start:
            continue
        lines.append(f"{title(name)}: ${value:,.0f} ({_money(value - start)}, "
                     f"{value / start - 1:+.2%})")
        worth, started = worth + value, started + start
    if not started:
        return ""
    lines.append(f"Total: ${worth:,.0f} ({_money(worth - started)}, {worth / started - 1:+.2%}) "
                 f"on ${started:,.0f} started")
    return "\n".join(lines)


def full_recap(old: dict[str, dict], new: dict[str, dict], today: date | None = None) -> str:
    """One message for every book, keyed by file name without .json ('book' is pairs): the
    pairs recap, a section per strategy, then profit and loss by strategy and in total.
    The totals go out only with a new close, so a catch-up run on a finished day stays quiet.
    With only the pairs book this is exactly recap()."""
    strategies = sorted(n for n in new if n != PAIRS_BOOK)
    parts = [recap(old.get(PAIRS_BOOK, {}), new[PAIRS_BOOK], today)] if PAIRS_BOOK in new else []
    said = tuple(new.get(PAIRS_BOOK, {}).get("last_run_warnings", []))
    parts += [strategy_recap(n, old.get(n, {}), new[n], today, said) for n in strategies]
    if strategies and any(_new_close(old.get(n, {}), b) for n, b in new.items()):
        parts.append(by_strategy(new))
    return "\n\n".join(p for p in parts if p)


def load_state(path) -> dict[str, dict]:
    """Every book in a state directory, keyed by file name without .json. Empty if the
    directory isn't there (the first run)."""
    return {f.stem: json.loads(f.read_text()) for f in sorted(Path(path).glob("*.json"))}


def main(argv: list[str]) -> None:
    old_path, new_path = argv
    if os.path.isdir(new_path):
        text = full_recap(load_state(old_path), load_state(new_path))
    else:
        old = json.loads(Path(old_path).read_text()) if os.path.exists(old_path) else {}
        text = recap(old, json.loads(Path(new_path).read_text()))
    if text:
        print(text)
        post_discord(text, os.environ.get("DISCORD_WEBHOOK_URL"))
    else:
        print("Nothing new since the last run: nothing to post.")


if __name__ == "__main__":
    main(sys.argv[1:])
