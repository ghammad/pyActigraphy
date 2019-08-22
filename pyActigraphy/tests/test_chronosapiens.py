from generate_dataset import generate_series
from generate_dataset import generate_sinewave
from generate_dataset import generate_squarewave

import numpy as np
import pandas as pd
import pyActigraphy
# from pytest import approx

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '2018/01/01 08:00:00'
N = 20160
period = pd.Timedelta(N, unit='T')

square_wave = generate_series(
    generate_squarewave(N=N, offset=True),
    start=start_time,
    sampling_period=sampling_period
)
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
    # generate_squarewave(N=N-1, A=1, offset=False),
    n_epochs_nan_trend*[np.nan] +
    (n_epochs_awake_prior_sleep*[0] +
     n_epochs_asleep*[1] +
     n_epochs_awake_after_sleep*[0])*n_days +
    n_epochs_nan_trend*[np.nan],
    start=start_time,
    sampling_period=sampling_period
)
raw_squarewave = pyActigraphy.io.BaseRaw(
    name='raw_square',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=square_wave,
    light=None
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


def test_chronosapiens_sinewave():

    assert pd.testing.assert_series_equal(
        raw_sinewave.Chronosapiens(threshold=0.5, min_trend_period='24h'),
        raw_sleepwave.data
    )


# def test_rar_ra_squarewave():
#
#     assert raw_squarewave.RA(binarize=True) == approx(1.0)
