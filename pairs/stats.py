"""Core statistics for a pair: hedge ratio, spread, z-score, half-life, cointegration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class PairStats:
    pvalue: float  # Engle-Granger cointegration p-value (lower = more reliably mean-reverting)
    beta: float  # hedge ratio: log(Y) ~ alpha + beta * log(X)
    alpha: float
    half_life: float  # days for a divergence to close halfway
    corr: float  # correlation of daily log returns


def hedge_ratio(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float]:
    """OLS fit of log_y = alpha + beta * log_x. Returns (beta, alpha)."""
    beta, alpha = np.polyfit(np.asarray(log_x, float), np.asarray(log_y, float), 1)
    return float(beta), float(alpha)


def spread(y: pd.Series, x: pd.Series, beta: float, alpha: float = 0.0) -> pd.Series:
    """Log-price spread. Positive = Y is rich relative to X."""
    return np.log(y) - beta * np.log(x) - alpha


def zscore(s: pd.Series, window: int) -> pd.Series:
    """How many rolling standard deviations the spread is from its rolling mean."""
    roll = s.rolling(window)
    return (s - roll.mean()) / roll.std()


def half_life(s: pd.Series) -> float:
    """Mean-reversion half-life in bars, from an AR(1) fit of the spread's changes."""
    s = pd.Series(s).dropna()
    lagged = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    b, _ = np.polyfit(lagged.to_numpy(), delta.to_numpy(), 1)
    if b >= 0 or b <= -1:
        return float("inf")
    return float(-np.log(2) / np.log(1 + b))


def pair_stats(y: pd.Series, x: pd.Series) -> PairStats:
    df = pd.concat([y, x], axis=1).dropna()
    ly, lx = np.log(df.iloc[:, 0]), np.log(df.iloc[:, 1])
    beta, alpha = hedge_ratio(ly, lx)
    _, pvalue, _ = coint(ly, lx)
    corr = float(ly.diff().corr(lx.diff()))
    return PairStats(
        pvalue=float(pvalue),
        beta=beta,
        alpha=alpha,
        half_life=half_life(ly - beta * lx - alpha),
        corr=corr,
    )
