import numpy as np
import pandas as pd

from pairs.stats import half_life, hedge_ratio, pair_stats, spread, zscore


def test_hedge_ratio_recovers_beta(coint_pair):
    y, x = coint_pair
    beta, alpha = hedge_ratio(np.log(y), np.log(x))
    assert abs(beta - 1.5) < 0.05
    assert abs(alpha - 0.2) < 0.25


def test_spread_is_log_y_minus_beta_log_x(coint_pair):
    y, x = coint_pair
    s = spread(y, x, beta=1.5, alpha=0.2)
    expected = np.log(y) - 1.5 * np.log(x) - 0.2
    pd.testing.assert_series_equal(s, expected, check_names=False)


def test_zscore_uses_rolling_window():
    s = pd.Series([0.0, 0, 0, 0, 10])
    z = zscore(s, window=5)
    assert z.iloc[:4].isna().all()
    assert z.iloc[-1] > 1.5


def test_half_life_of_ou_process():
    rng = np.random.default_rng(3)
    phi = 0.9  # half-life = ln(0.5)/ln(0.9) ~= 6.6 days
    x = np.zeros(5000)
    for t in range(1, len(x)):
        x[t] = phi * x[t - 1] + rng.normal()
    hl = half_life(pd.Series(x))
    assert 5.5 < hl < 8.0


def test_pair_stats_flags_cointegrated_pair(coint_pair):
    y, x = coint_pair
    st = pair_stats(y, x)
    assert st.pvalue < 0.01
    assert abs(st.beta - 1.5) < 0.05
    assert 2 < st.half_life < 20


def test_pair_stats_rejects_independent_walks(random_walks):
    a, b = random_walks
    assert pair_stats(a, b).pvalue > 0.05
