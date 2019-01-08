from generate_dataset import generate_gaussian_noise
from generate_dataset import generate_series
from generate_dataset import generate_sinewave

import pandas as pd
import pyActigraphy
import pytest

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2018 08:00:00'
N = 20160*sampling_period
period = pd.Timedelta(N, unit='s')

gaussian_noise = generate_series(
    generate_gaussian_noise(N=N-1),
    start=start_time,
    sampling_period=sampling_period
)

sine_wave = generate_series(
    generate_sinewave(N=N-1),
    start=start_time,
    sampling_period=sampling_period
)

raw_gaussian = pyActigraphy.io.BaseRaw(
    name='raw_gaussian',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=gaussian_noise,
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


def test_rar_is_gaussian():

    assert raw_gaussian.IS(
        freq='1min', binarize=False
        ) == pytest.approx(1/period.days, rel=0.1)


def test_rar_is_sinewave():

    assert raw_sinewave.IS(
        freq='1min', binarize=False
        ) == pytest.approx(1.0, 0.01)


def test_rar_iv_gaussian():

    assert raw_gaussian.IV(
        freq='1min', binarize=False
        ) == pytest.approx(2.0, 0.01)


def test_rar_iv_sinewave():

    assert raw_sinewave.IV(
        freq='1min', binarize=False
        ) == pytest.approx(0.0, abs=0.001)
