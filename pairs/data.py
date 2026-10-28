"""Price data from Yahoo Finance (via yfinance), cached on disk.

The cache is split at the 4:15pm ET close: a download made during the session never stands
in for the closing prices, which are fetched fresh once the market has closed.
"""

import io
import time as clock
import urllib.request
from datetime import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from pairs.paths import DATA_DIR, UNIVERSE_PATH

CLOSE_CUTOFF = time(16, 15)  # ET; closing prices are settled by then

# Pairs only make economic sense between similar businesses, so the scanner only
# compares tickers within the same group. Edit freely.
UNIVERSE: dict[str, list[str]] = {
    "Beverages & staples": ["KO", "PEP", "KDP", "MNST", "PG", "CL", "KMB", "CHD", "CLX"],
    "Big banks": ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC"],
    "Oil majors": ["XOM", "CVX", "COP", "EOG", "OXY", "BP", "SHEL", "TTE"],
    "Payments": ["V", "MA", "AXP", "PYPL", "COF"],
    "Home improvement & retail": ["HD", "LOW", "WMT", "TGT", "COST", "DG", "DLTR"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "ADI"],
    "Rails & parcels": ["UNP", "CSX", "NSC", "UPS", "FDX"],
    "Utilities": ["DUK", "SO", "D", "AEP", "XEL", "NEE", "ED"],
    "Telecom": ["T", "VZ", "TMUS"],
    "Gold & miners ETFs": ["GLD", "IAU", "GDX", "GDXJ", "NEM"],
    "Index ETFs": ["SPY", "VOO", "IVV", "QQQ", "DIA", "IWM"],
}


INDEX_PAGES = {  # Wikipedia lists with each member's GICS sub-industry
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def load_universe(path: Path = UNIVERSE_PATH,
                  indexes: set[str] | None = None) -> dict[str, list[str]]:
    """S&P 1500 tickers (large, mid and small caps) grouped by GICS sub-industry: a much wider
    search than UNIVERSE, still only pairing like with like. `indexes` limits it, e.g.
    {"S&P 500"}. Sub-industries with a single member are dropped."""
    df = pd.read_csv(path)
    if indexes is not None:
        df = df[df["index"].isin(indexes)]
    groups = {sub: sorted(g["ticker"]) for sub, g in df.groupby("sub_industry", sort=False)}
    return {sub: tickers for sub, tickers in groups.items() if len(tickers) >= 2}


def update_universe(path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    """Rewrite `path` from Wikipedia's index lists (needs lxml: `uv run --with lxml`)."""
    parts = []
    for index, url in INDEX_PAGES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "pairs-trader/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read(10_000_000).decode()
        t = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
        parts.append(pd.DataFrame({
            "ticker": t["Symbol"].str.replace(".", "-", regex=False),  # Yahoo spelling: BRK-B
            "name": t["Security"],
            "sector": t["GICS Sector"],
            "sub_industry": t["GICS Sub-Industry"],
            "index": index,
        }))
    out = (pd.concat(parts).drop_duplicates("ticker")  # mid-rebalance a stock can be in two
           .sort_values(["sub_industry", "ticker"]))
    out.to_csv(path, index=False)
    return out


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York")


def drop_unfinished_bar(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's row while US markets are still open; it is not a close yet."""
    now = now or _now_et()
    if len(df) and df.index[-1].date() == now.date() and now.time() < CLOSE_CUTOFF:
        return df.iloc[:-1]
    return df


def _download(tickers: list[str], period: str, attempts: int = 3) -> pd.DataFrame:
    """Adjusted closes from Yahoo, retrying: it sometimes rate-limits cloud servers and
    answers with an empty frame rather than an error."""
    for i in range(attempts):
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                          group_by="column", threads=True)
        closes = raw["Close"] if "Close" in raw else pd.DataFrame()
        if isinstance(closes, pd.Series):  # single ticker
            closes = closes.to_frame(tickers[0])
        closes = closes.dropna(axis=1, how="all")
        if not closes.empty or i == attempts - 1:
            return closes
        clock.sleep(5 * 2**i)
    return closes


def load_prices(tickers: list[str], period: str = "3y", refresh: bool = False,
                now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker. Tickers with no data are dropped."""
    tickers = sorted(set(tickers))
    now = now or _now_et()
    phase = "closed" if now.time() >= CLOSE_CUTOFF else "open"
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"prices_{period}_{now.date().isoformat()}_{phase}.csv"

    cached = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else pd.DataFrame()
    missing = tickers if refresh else [t for t in tickers if t not in cached.columns]

    if missing:
        closes = _download(missing, period)
        cached = cached.drop(columns=closes.columns, errors="ignore").join(closes, how="outer") \
            if not cached.empty else closes
        cached.to_csv(cache)

    out = cached[[t for t in tickers if t in cached.columns]].dropna(how="all")
    return drop_unfinished_bar(out, now)
