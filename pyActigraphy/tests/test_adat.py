# from generate_dataset import generate_sequences
# from generate_dataset import generate_series

# import numpy as np
import pandas as pd
import pyActigraphy
# import pytest

frequency = pd.Timedelta('60s')
start_time = '2020-01-01 00:00:00'
period = pd.Timedelta('7D')

# Uniform data over 7 days
raw_uniform = pyActigraphy.io.BaseRaw(
    name='raw_uniform',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=pd.Series(
        data=10,
        index=pd.date_range(
            start=start_time,
            periods=int(period/frequency),
            freq=frequency)
    ),
    light=None
)

# Uniform data over 7 days, corrupted (i.e set activity counts to zeros)
# during the first and last days
raw_uniform_corrupted = pyActigraphy.io.BaseRaw(
    name='raw_uniform',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=pd.Series(
        data=10,
        index=pd.date_range(
            start=start_time,
            periods=int(period/frequency),
            freq=frequency)
    ),
    light=None
)
raw_uniform_corrupted.raw_data.loc[
    '2020-01-01 08:00:00':'2020-01-01 10:00:00'] = 0
raw_uniform_corrupted.raw_data.loc[
    '2020-01-07 08:00:00':'2020-01-07 10:00:00'] = 0

# Mask corrupted periods
raw_uniform_corrupted.inactivity_length = "1h"
raw_uniform_corrupted.mask_inactivity = True


# Test  ADAT
def test_adat():

    assert raw_uniform.ADAT(
        binarize=False, rescale=False, exclude_ends=False
    ) == 10*int(pd.Timedelta('24h')/frequency)


def test_adat_binarize():

    assert raw_uniform.ADAT(
        binarize=True, rescale=False, exclude_ends=False
    ) == int(pd.Timedelta('24h')/frequency)


def test_adat_rescale():

    assert raw_uniform_corrupted.ADAT(
        binarize=False, rescale=True, exclude_ends=False
    ) == 10*int(pd.Timedelta('24h')/frequency)


def test_adat_exclude():

    assert raw_uniform_corrupted.ADAT(
        binarize=False, rescale=False, exclude_ends=True
    ) == 10*int(pd.Timedelta('24h')/frequency)
