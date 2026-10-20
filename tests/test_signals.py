import numpy as np
import pandas as pd

from pairs.signals import positions_from_z


def test_enters_short_spread_when_z_high_and_holds_until_exit():
    # z > entry -> short spread (-1): short the rich leg (Y), buy the cheap leg (X)
    z = pd.Series([0.0, 1.0, 2.5, 1.8, 1.0, 0.4, 0.0])
    pos = positions_from_z(z, entry=2.0, exit=0.5, stop=4.0)
    assert pos.tolist() == [0, 0, -1, -1, -1, 0, 0]


def test_enters_long_spread_when_z_low():
    z = pd.Series([0.0, -2.2, -1.0, -0.3])
    pos = positions_from_z(z, entry=2.0, exit=0.5, stop=4.0)
    assert pos.tolist() == [0, 1, 1, 0]


def test_stop_loss_flattens_and_waits_for_reset_before_reentry():
    # blows through stop at 4.5 -> flat; still above entry at 3.0 but must not re-enter
    # until z has come back inside the exit band.
    z = pd.Series([2.5, 4.5, 3.0, 2.5, 0.2, 2.1])
    pos = positions_from_z(z, entry=2.0, exit=0.5, stop=4.0)
    assert pos.tolist() == [-1, 0, 0, 0, 0, -1]


def test_nan_z_means_flat():
    z = pd.Series([np.nan, np.nan, 2.5, np.nan])
    pos = positions_from_z(z, entry=2.0, exit=0.5, stop=4.0)
    assert pos.tolist() == [0, 0, -1, -1]  # NaN mid-trade holds; NaN before data is flat
