"""Order sheet text, and posting it to a Discord channel via webhook.

Stdlib only on purpose: the hosted job runs this module with the webhook secret and without
installing any third-party package.

    python -m pairs.notify OLD_BOOK.json NEW_BOOK.json   # post orders new in NEW, + warnings
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


def main(argv: list[str]) -> None:
    old_path, new_path = argv
    old = json.loads(Path(old_path).read_text()) if os.path.exists(old_path) else {}
    new = json.loads(Path(new_path).read_text())
    parts = [format_orders(o)] if (o := new_orders(old, new)) else []
    parts += new.get("last_run_warnings", [])
    if parts:
        print("\n\n".join(parts))
        post_discord("\n\n".join(parts), os.environ.get("DISCORD_WEBHOOK_URL"))
    else:
        print("No new orders and no warnings: nothing to post.")


if __name__ == "__main__":
    main(sys.argv[1:])
