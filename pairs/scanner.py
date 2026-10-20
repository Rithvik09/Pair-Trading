"""Search a universe of tickers for cointegrated (mean-reverting) pairs."""

from itertools import combinations

import pandas as pd

from pairs.stats import pair_stats

COLUMNS = ["group", "y", "x", "pvalue", "beta", "alpha", "half_life", "corr"]


def scan(
    prices: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    max_pvalue: float = 0.05,
    min_obs: int = 200,
) -> pd.DataFrame:
    """Test every pair within each group; return those with p-value <= max_pvalue.

    Engle-Granger depends on which leg is the dependent variable, so both orderings
    are tested and the stronger one is kept.
    """
    if groups is None:
        groups = {"all": list(prices.columns)}

    rows = []
    for group, tickers in groups.items():
        present = [t for t in tickers if t in prices.columns]
        for a, b in combinations(present, 2):
            df = prices[[a, b]].dropna()
            if len(df) < min_obs:
                continue
            best = None
            for y, x in ((a, b), (b, a)):
                st = pair_stats(df[y], df[x])
                if best is None or st.pvalue < best[2].pvalue:
                    best = (y, x, st)
            y, x, st = best
            if st.pvalue <= max_pvalue:
                rows.append([group, y, x, st.pvalue, st.beta, st.alpha, st.half_life, st.corr])

    return pd.DataFrame(rows, columns=COLUMNS).sort_values("pvalue", ignore_index=True)
