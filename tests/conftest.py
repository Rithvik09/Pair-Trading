import numpy as np
import pandas as pd
import pytest


def _dates(n):
    return pd.bdate_range("2022-01-03", periods=n)


@pytest.fixture
def coint_pair():
    """Two log-price series where log(Y) = 0.2 + 1.5*log(X) + OU noise (mean-reverting)."""
    rng = np.random.default_rng(7)
    n = 750
    log_x = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] * 0.9 + rng.normal(0, 0.01)
    log_y = 0.2 + 1.5 * log_x + ou
    idx = _dates(n)
    return pd.Series(np.exp(log_y), idx, name="YYY"), pd.Series(np.exp(log_x), idx, name="XXX")


@pytest.fixture
def random_walks():
    rng = np.random.default_rng(11)
    n = 750
    idx = _dates(n)
    a = np.exp(np.log(40) + np.cumsum(rng.normal(0, 0.015, n)))
    b = np.exp(np.log(80) + np.cumsum(rng.normal(0, 0.015, n)))
    return pd.Series(a, idx, name="AAA"), pd.Series(b, idx, name="BBB")
