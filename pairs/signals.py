"""Turn a spread z-score into a position.

Position is in the *spread* (Y minus beta * X):
  -1 = short spread: Y has risen too far vs X -> short Y, buy X
  +1 = long spread:  Y has fallen too far vs X -> buy Y, short X
   0 = flat

The entry/exit/stop rule lives in `decide`, shared by the backtest (`positions_from_z`)
and the paper book (`PaperBook.step`), so the two cannot drift apart.
"""

import math

import pandas as pd


def decide(pos: int, blocked: bool, z: float, entry: float, exit: float, stop: float) -> str:
    """One bar of the strategy, given the current position and a non-NaN z.

    Enter at |z| >= entry, exit once z is back inside +/-exit, bail at |z| >= stop.
    After a stop-out we stay flat ("blocked") until z returns inside the exit band, so we
    don't immediately re-enter a pair whose relationship may have broken.

    In a position: "stop", "exit" or "hold".
    Flat and blocked: "unblock" or "blocked".
    Flat: "short_spread", "long_spread" or "flat".
    """
    if pos != 0:
        if abs(z) >= stop:
            return "stop"
        if (pos == -1 and z <= exit) or (pos == 1 and z >= -exit):
            return "exit"
        return "hold"
    if blocked:
        return "unblock" if abs(z) <= exit else "blocked"
    if entry <= z < stop:
        return "short_spread"
    if -stop < z <= -entry:
        return "long_spread"
    return "flat"


def positions_from_z(z: pd.Series, entry: float, exit: float, stop: float) -> pd.Series:
    """Position held at each bar under `decide`. A NaN z keeps the previous position."""
    out = []
    pos = 0
    blocked = False
    for v in z:
        if v is None or math.isnan(v):
            out.append(pos)
            continue
        action = decide(pos, blocked, v, entry, exit, stop)
        if action == "stop":
            pos, blocked = 0, True
        elif action == "exit":
            pos = 0
        elif action == "unblock":
            blocked = False
        elif action == "short_spread":
            pos = -1
        elif action == "long_spread":
            pos = 1
        out.append(pos)
    return pd.Series(out, index=z.index, dtype=int)


def trade_opens(pos: pd.Series) -> pd.Series:
    """True on each bar where a new position is taken (from flat or a flipped side)."""
    return pos.ne(pos.shift(1, fill_value=0)) & pos.ne(0)


def leg_weights(beta):
    """Dollar weights (w_y, w_x) per unit of gross notional, |w_y| + |w_x| = 1.

    A long spread holds +w_y of Y and -w_x of X; a short spread the opposite. w_x carries
    beta's sign, so with a negative hedge ratio both legs are on the same side. Works on a
    float or elementwise on a Series.
    """
    return 1 / (1 + abs(beta)), beta / (1 + abs(beta))
