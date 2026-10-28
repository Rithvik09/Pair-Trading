"""The daily recap (book value, decisions and why, order sheet), posted to Discord via webhook.

Stdlib only on purpose: the hosted job runs this module with the webhook secret and without
installing any third-party package.

    python -m pairs.notify OLD_BOOK.json NEW_BOOK.json   # post the recap of what's new in NEW
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
}


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
            out.append(f"**{o['pair']}** {EVENT_LABELS.get(o['event'], o['event'])} "
                       f"(z {o['z']:+.2f}, signal from {_day(o['signal_date'])} close)")
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
                             "real account would get a margin call. Lower notional or "
                             "max_open_positions in config.toml.")
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


def main(argv: list[str]) -> None:
    old_path, new_path = argv
    old = json.loads(Path(old_path).read_text()) if os.path.exists(old_path) else {}
    new = json.loads(Path(new_path).read_text())
    text = recap(old, new)
    if text:
        print(text)
        post_discord(text, os.environ.get("DISCORD_WEBHOOK_URL"))
    else:
        print("Nothing new since the last run: nothing to post.")


if __name__ == "__main__":
    main(sys.argv[1:])
