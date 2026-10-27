"""Turn paper-book events into the orders you would place, and when to place them.

Signals come from a day's close, so orders are for the open of the next NYSE session.
"""

from datetime import date, timedelta
from functools import cache


def _observed(d: date) -> date:
    """Saturday holidays move to Friday, Sunday ones to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l_) // 433
    month = (h + l_ - 7 * m + 90) // 25
    return date(year, month, (h + l_ - 7 * m + 33 * month + 19) % 32)


@cache
def nyse_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures. Early-close days (e.g. Black Friday) are normal sessions."""
    days = {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # NYSE doesn't close the prior Friday for a Saturday New Year
        days.add(_observed(new_year))
    return frozenset(days)


def is_session(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def last_completed_session(now) -> str:
    """ISO date of the latest NYSE session whose close is final (4:15pm ET) as of `now`,
    a timezone-aware datetime in US/Eastern."""
    d = now.date()
    if not (is_session(d) and (now.hour, now.minute) >= (16, 15)):
        d -= timedelta(days=1)
        while not is_session(d):
            d -= timedelta(days=1)
    return d.isoformat()


def next_session(signal_date) -> str | None:
    """ISO date of the first NYSE session after `signal_date` (date or ISO string).
    None if the date can't be parsed."""
    try:
        d = date.fromisoformat(str(signal_date)) + timedelta(days=1)
    except ValueError:
        return None
    while not is_session(d):
        d += timedelta(days=1)
    return d.isoformat()


def leg_orders(position: dict, prices: tuple[float, float], closing: bool) -> list[dict]:
    """One order per leg. Opening: short legs SELL SHORT, long legs BUY.
    Closing: the reverse (BUY TO COVER / SELL), same share counts."""
    out = []
    for leg, px in zip(("y", "x"), prices, strict=True):
        qty = position[f"{leg}_shares"]
        if closing:
            side = "BUY TO COVER" if qty < 0 else "SELL"
        else:
            side = "SELL SHORT" if qty < 0 else "BUY"
        out.append({"ticker": position[leg], "side": side, "qty": abs(int(qty)),
                    "ref_price": float(px)})
    return out


def order_record(pair: str, position: dict, event: str, signal_date, z: float,
                 prices: tuple[float, float]) -> dict:
    return {
        "pair": pair,
        "event": event,
        "signal_date": str(signal_date),
        "execute_on": next_session(signal_date),
        "z": float(z),
        "legs": leg_orders(position, prices, closing=not event.startswith("open")),
    }
