# from generate_dataset import generate_gaussian_noise
from generate_dataset import generate_series
# from generate_dataset import generate_squarewave
from generate_dataset import generate_sinewave

import pandas as pd
import pyActigraphy
from pytest import approx

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2018 08:00:00'
N = 20160
period = pd.Timedelta(N*sampling_period, unit='s')

sine_wave_mask = generate_series(
    generate_sinewave(N=N, offset=True),
    start=start_time,
    sampling_period=sampling_period
)
sine_wave_mask.loc['2018-01-03 06:00:00':'2018-01-03 14:00:00'] = 0

raw_sinewave_mask = pyActigraphy.io.BaseRaw(
    name='raw_sinewave',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=sine_wave_mask,
    light=None
)


def test_mask_is_automatic():

    raw_sinewave_mask.create_inactivity_mask(duration='90min')
    raw_sinewave_mask.mask_inactivity = True

    assert raw_sinewave_mask.IS(
        freq='1min', binarize=False
        ) == approx(1.0, 0.01)


def test_mask_is_manual():

    # Reset and deactivate mask
    raw_sinewave_mask.inactivity_length = None
    # Add mask period manually
    raw_sinewave_mask.add_mask_period(
        start='2018-01-03 06:00:00', stop='2018-01-03 14:00:00'
    )
    # Activate mask
    raw_sinewave_mask.mask_inactivity = True

    assert raw_sinewave_mask.IS(
        freq='1min', binarize=False
        ) == approx(1.0, 0.01)
