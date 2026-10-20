"""Can a stock be shorted, and what does borrowing it cost?

Sources, highest priority first:
  1. borrow.csv in the project root (columns: ticker,shortable,fee_rate). Your broker's
     numbers beat anything public; blank cells fall through to the next source.
     fee_rate is an annual DECIMAL (0.0025 = 0.25%/yr), unlike IBKR's file which is in
     percent. shortable is true/false (yes/no, 1/0 also accepted).
  2. Interactive Brokers' public short-stock file (anonymous FTP, refreshed through the day).
     A ticker missing from it means IBKR has no shares to lend.
  3. Fallback: a general-collateral fee and shortability "unknown".
"""

import csv
import math
import re
import urllib.request
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from pairs.paths import DATA_DIR, OVERRIDES_PATH

IBKR_URL = "ftp://shortstock:@ftp3.interactivebrokers.com/usa.txt"
DEFAULT_FEE = 0.0025  # 0.25%/yr, typical for easy-to-borrow large caps


def normalize_symbol(sym: str) -> str:
    """Yahoo writes class shares as BRK-B, IBKR as 'BRK B' or BRK.B: map all to BRK.B."""
    return re.sub(r"[-. ]+", ".", sym.strip().upper())


@dataclass(frozen=True)
class BorrowInfo:
    shortable: bool | None  # None = unknown
    fee_rate: float  # annualised, as a decimal (0.0025 = 0.25% per year)
    available: int | None  # shares available to borrow, if known
    source: str  # "override", "ibkr" or "default"


def parse_ibkr(text: str) -> dict[str, BorrowInfo]:
    """Parse IBKR's pipe-delimited file: SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 8:
            continue
        sym, fee, avail = normalize_symbol(f[0]), f[6].strip(), f[7].strip()
        try:
            n = int(avail.lstrip(">"))
        except ValueError:
            n = None
        try:
            rate = float(fee) / 100
        except ValueError:
            rate = DEFAULT_FEE
        if not math.isfinite(rate) or rate < 0:
            rate = DEFAULT_FEE
        out[sym] = BorrowInfo(shortable=None if n is None else n > 0, fee_rate=rate,
                              available=n, source="ibkr")
    return out


def fetch_ibkr(timeout: float = 10.0) -> str | None:
    """Today's IBKR file, cached once per day. None if it can't be reached."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"ibkr_shortstock_{date.today().isoformat()}.txt"
    if cache.exists():
        return cache.read_text(encoding="latin-1")
    try:
        with urllib.request.urlopen(IBKR_URL, timeout=timeout) as r:
            text = r.read(20_000_000).decode("latin-1")  # the real file is ~2MB
    except OSError:
        return None
    if parse_ibkr(text):  # never cache an error page or a truncated download
        cache.write_text(text, encoding="latin-1")
    return text


_TRUE, _FALSE = {"true", "yes", "y", "1"}, {"false", "no", "n", "0"}


def _overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            t = (row.get("ticker") or "").strip()
            if not t or t.startswith("#"):
                continue
            where = f"{path.name} line {line_no} ({t})"
            o = {}
            s = (row.get("shortable") or "").strip().lower()
            if s in _TRUE | _FALSE:
                o["shortable"] = s in _TRUE
            elif s:
                raise ValueError(f"{where}: shortable must be true/false, got {s!r}")
            r = (row.get("fee_rate") or "").strip()
            if r:
                try:
                    rate = float(r)
                except ValueError:
                    raise ValueError(f"{where}: fee_rate must be a number, got {r!r}") from None
                if not 0 <= rate <= 1:
                    raise ValueError(f"{where}: fee_rate is an annual decimal "
                                     f"(0.0025 = 0.25%/yr), got {rate}")
                o["fee_rate"] = rate
            out[normalize_symbol(t)] = o
    return out


_NOT_PASSED = object()


def borrow_table(tickers, *, feed_text=_NOT_PASSED,
                 overrides_path: Path = OVERRIDES_PATH) -> dict[str, BorrowInfo]:
    """BorrowInfo for each ticker. Pass feed_text=None to skip the network entirely."""
    if feed_text is _NOT_PASSED:
        feed_text = fetch_ibkr()
    feed = (parse_ibkr(feed_text) or None) if feed_text else None  # unparseable = unavailable
    over = _overrides(overrides_path)
    out = {}
    for t in tickers:
        key = normalize_symbol(t)
        if feed is None:
            info = BorrowInfo(shortable=None, fee_rate=DEFAULT_FEE, available=None,
                              source="default")
        else:
            info = feed.get(key) or BorrowInfo(shortable=False, fee_rate=DEFAULT_FEE,
                                             available=0, source="ibkr")
        if over.get(key):
            info = replace(info, **over[key], source="override")
        out[t] = info
    return out


def fee_label(rate: float) -> str:
    """An annual decimal rate as shown to the user: 0.0025 -> '0.25%/yr'."""
    return f"{rate:.2%}/yr"


def borrow_label(info: BorrowInfo) -> str:
    """Short human label, e.g. '0.25%/yr' or 'NOT SHORTABLE'."""
    if info.shortable is False:
        return "NOT SHORTABLE"
    label = fee_label(info.fee_rate)
    return label if info.shortable else f"{label} (unverified)"
