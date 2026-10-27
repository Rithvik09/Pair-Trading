"""Simulated (paper) positions, persisted to a JSON file. No broker, no real money."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pairs.orders import order_record
from pairs.signals import decide, leg_weights


def pair_key(pair) -> str:
    """'Y/X': how a pair is keyed in the book and labelled in the UI."""
    y, x = pair
    return f"{y}/{x}"


def parse_pair_key(k: str) -> list[str]:
    """Inverse of pair_key."""
    return k.split("/")


def short_legs(pair, side: int, beta: float) -> list[str]:
    """Tickers that are sold short for a spread position (-1 short spread, +1 long spread).

    Y is short in a short spread. X is short when it's sold against Y: in a long spread
    with a positive hedge ratio, or a short spread with a negative one.
    """
    y, x = pair
    legs = [y] if side == -1 else []
    if side * math.copysign(1, beta) > 0:
        legs.append(x)
    return legs


@dataclass
class PaperBook:
    path: Path
    cash: float = 100_000.0
    blocked: list[str] = field(default_factory=list)  # pairs stopped out, waiting for reset
    positions: dict[str, dict] = field(default_factory=dict)
    closed: list[dict] = field(default_factory=list)
    last_bar: dict[str, str] = field(default_factory=dict)  # last date stepped, per pair
    orders: list[dict] = field(default_factory=list)  # every order the strategy asked for
    last_run_warnings: list[str] = field(default_factory=list)  # e.g. tickers with no data

    @classmethod
    def load(cls, path, starting_cash: float = 100_000.0) -> "PaperBook":
        path = Path(path)
        if not path.exists():
            return cls(path=path, cash=starting_cash)
        d = json.loads(path.read_text())
        for k, p in d.get("positions", {}).items():
            if any(float(p.get(f"{leg}_shares", 0)) % 1 for leg in ("y", "x")):
                raise ValueError(f"{path}: position {k} isn't in whole shares; it was saved by "
                                 "an older version. Close it by hand and remove it from the book.")
        return cls(path=path, cash=d["cash"], blocked=d.get("blocked", []),
                   positions=d.get("positions", {}), closed=d.get("closed", []),
                   last_bar=d.get("last_bar", {}), orders=d.get("orders", []),
                   last_run_warnings=d.get("last_run_warnings", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        d = {"cash": self.cash, "blocked": self.blocked, "positions": self.positions,
             "closed": self.closed, "last_bar": self.last_bar, "orders": self.orders,
             "last_run_warnings": self.last_run_warnings}
        self.path.write_text(json.dumps(d, indent=2, default=str))

    def step(self, pair, *, date, y_px, x_px, beta, z, entry, exit, stop, notional,
             cost_bps: float = 0.0, shortable=(None, None), borrow_fee=(0.0, 0.0)) -> str:
        """Apply one bar of the strategy to one pair.

        Entry/exit/stop decisions come from signals.decide, the same rule the backtest uses.

        cost_bps is charged on gross notional when opening and again when closing.
        shortable / borrow_fee are per leg (Y, X): a trade whose short leg is known to be
        unborrowable (False) is skipped; None means unknown and doesn't block. borrow_fee is
        an annual rate accrued per calendar day on the short leg's entry value.

        Positions are whole shares (rounded down), as you would actually trade them. Every
        open, close and stop appends an order record to self.orders.

        Returns one of: "none", "hold", "open short_spread", "open long_spread",
        "close", "stop", "blocked", "unshortable <tickers>", "too small <tickers>",
        "skip: hedge ratio not positive".
        """
        k = pair_key(pair)
        pos = self.positions.get(k)
        if z is None or math.isnan(z):
            return "hold" if pos else "none"

        current = 0 if not pos else -1 if pos["side"] == "short_spread" else 1
        action = decide(current, k in self.blocked, z, entry, exit, stop)
        if action == "stop":
            self._close(k, date, y_px, x_px, "stop", cost_bps, z)
            self.blocked.append(k)
            return "stop"
        if action == "exit":
            self._close(k, date, y_px, x_px, "converged", cost_bps, z)
            return "close"
        if action == "unblock":
            self.blocked.remove(k)
            return "none"
        if action in ("hold", "blocked"):
            return action
        if action == "flat":
            return "none"

        if beta <= 0:  # same rule as the backtest's can_enter: never short or buy both legs
            return "skip: hedge ratio not positive"
        side = action  # short_spread: Y rose too far, short Y / buy X. long_spread: the reverse.
        sign = -1 if side == "short_spread" else 1
        w_y, w_x = leg_weights(beta)
        y, x = pair
        y_shares = int(sign * math.floor(notional * w_y / y_px))
        x_shares = -int(math.copysign(math.floor(notional * abs(w_x) / x_px), w_x)) * sign
        can_short = dict(zip(pair, shortable, strict=True))
        blocked_legs = [t for t in short_legs(pair, sign, beta) if can_short[t] is False]
        if blocked_legs:
            return "unshortable " + ",".join(blocked_legs)
        too_small = [t for t, n in ((y, y_shares), (x, x_shares)) if n == 0]
        if too_small:
            return "too small " + ",".join(too_small)
        self.positions[k] = {
            "y": y, "x": x, "side": side, "entry_date": str(date),
            "entry_y": float(y_px), "entry_x": float(x_px),
            "y_shares": y_shares, "x_shares": x_shares,
            "beta": float(beta), "entry_z": float(z), "notional": float(notional),
            "costs": notional * cost_bps / 1e4,
            "borrow_fee_y": float(borrow_fee[0]) if y_shares < 0 else 0.0,
            "borrow_fee_x": float(borrow_fee[1]) if x_shares < 0 else 0.0,
        }
        self.orders.append(order_record(k, self.positions[k], f"open {side}", date, z,
                                        (y_px, x_px)))
        return f"open {side}"

    def borrow_accrued(self, k: str, date) -> float:
        """Borrow fees owed on the short leg(s) from entry up to `date` (actual/360)."""
        p = self.positions[k]
        try:
            days = (pd.Timestamp(str(date)) - pd.Timestamp(p["entry_date"])).days
        except ValueError:
            return 0.0
        owed = sum(abs(p[f"{leg}_shares"]) * p[f"entry_{leg}"] * p.get(f"borrow_fee_{leg}", 0.0)
                   for leg in ("y", "x") if p[f"{leg}_shares"] < 0)
        return owed * max(days, 0) / 360

    def rebase_entry(self, k: str, y_adj, x_adj) -> None:
        """Re-read entry prices from today's adjusted series.

        Yahoo restates adjusted history after every dividend/split, so an entry price saved
        weeks ago is on a different basis from today's price. Share counts are real
        holdings and stay as they are.
        """
        p = self.positions[k]
        d = pd.Timestamp(p["entry_date"])
        if d in y_adj.index and d in x_adj.index:
            p["entry_y"], p["entry_x"] = float(y_adj[d]), float(x_adj[d])

    def mark(self, k: str, y_px: float, x_px: float) -> float:
        """Unrealised P&L of an open position at the given prices."""
        p = self.positions[k]
        return p["y_shares"] * (y_px - p["entry_y"]) + p["x_shares"] * (x_px - p["entry_x"])

    def _close(self, k, date, y_px, x_px, reason, cost_bps, z) -> None:
        p = self.positions[k]
        costs = p.get("costs", 0.0) + p.get("notional", 0.0) * cost_bps / 1e4
        borrow = self.borrow_accrued(k, date)
        pnl = self.mark(k, y_px, x_px) - costs - borrow
        event = "stop" if reason == "stop" else "close"
        self.orders.append(order_record(k, p, event, date, z, (y_px, x_px)))
        del self.positions[k]
        self.cash += pnl
        self.closed.append({**p, "exit_date": str(date), "exit_y": float(y_px),
                            "exit_x": float(x_px), "costs": costs, "borrow": borrow,
                            "pnl": pnl, "reason": reason})
