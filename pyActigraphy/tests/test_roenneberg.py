import numpy as np
import pandas as pd
import pyActigraphy

from pytest import approx
from generate_dataset import generate_series
from generate_dataset import generate_sinewave


sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '2018/01/01 08:00:00'
N = 20160
period = pd.Timedelta(N, unit='T')

sine_wave = generate_series(
    generate_sinewave(N=N, offset=True),
    start=start_time,
    sampling_period=sampling_period
)

# Sleep characteristics:
# - start_time: 8h
# - sleep onset: 22h01
# - periode awake: 14h
# - sleep offset: 6h
# - periode asleep:7h59min
n_epochs_nan_trend = int(pd.Timedelta('12h')/frequency)
n_epochs_awake_prior_sleep = int(pd.Timedelta('2h01min')/frequency)
n_epochs_awake_after_sleep = int(pd.Timedelta('14h')/frequency)
n_epochs_asleep = int(pd.Timedelta('7h59min')/frequency)
n_days = int(period/pd.Timedelta('1D'))-1

# Create associated sleep signal (awake:0, sleep:1)
sleep_wave = generate_series(
    n_epochs_nan_trend*[np.nan] +
    (n_epochs_awake_prior_sleep*[0] +
     n_epochs_asleep*[1] +
     n_epochs_awake_after_sleep*[0])*n_days +
    [0] + (n_epochs_nan_trend-1)*[np.nan],  # At midnight on the last day,
    # there are still enough data left to calculate the trend.
    start=start_time,
    sampling_period=sampling_period
)
raw_sinewave = pyActigraphy.io.BaseRaw(
    name='raw_sinewave',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=sine_wave,
    light=None
)
raw_sleepwave = pyActigraphy.io.BaseRaw(
    name='raw_sleep',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=sleep_wave,
    light=None
)

# @pytest.mark.parametrize("a,b,expected", testdata)
# def test_timedistance_v0(a, b, expected):
#     diff = a - b
#     assert diff == expected


def test_roenneberg_sinewave():

    test = raw_sinewave.Roenneberg(threshold=0.5, min_trend_period='24h')
    index_equality = np.alltrue(raw_sleepwave.data.index == test.index)
    value_equality = np.isclose(raw_sleepwave.data, test, equal_nan=True)

    assert (
        index_equality and
        np.sum(value_equality)/len(value_equality) == approx(1, rel=0.01)
    )
