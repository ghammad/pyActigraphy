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
N = 14*1440  # 14 DAYS
period = pd.Timedelta(N*sampling_period, unit='s')

sine_wave_mask = generate_series(
    generate_sinewave(N=N, offset=True),
    start=start_time,
    sampling_period=sampling_period
)
mask_start = pd.to_datetime('2018-01-03 06:00:00')
mask_end = pd.to_datetime('2018-01-03 14:00:00')
mask_nepochs = (mask_end-mask_start)/frequency

sine_wave_mask.loc[mask_start:mask_end] = 0

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


def test_mask_fraction():

    # Test if mask fraction is correctly calculated as the length of the masked
    # period divided by the total length of the recording/mask
    assert raw_sinewave_mask.mask_fraction() == approx(mask_nepochs/N, 0.01)


def test_mask_fraction_restricted():

    # Test if mask fraction is correctly updated when the recording is
    # restricted to a certain period.
    # Ex: restrict to '01/01/2018 08:00:00':'03/01/2018 10:00:00'
    # so that only half of original mask overlaps with the new recording period
    raw_sinewave_mask.period = pd.Timedelta('2days2h')
    assert raw_sinewave_mask.mask_fraction() == approx(
        (pd.Timedelta('4h')/pd.Timedelta('2days2h')),
        0.01
    )


def test_mask_fraction_period():

    # Test if mask fraction per period is correctly calculated.
    # Restrict recording to 3 days. Starting at 8h00.
    # The first day, there is no mask. Mask_frac == 0.
    # The second day, the last 2h are masked. Mask_frac == 2/24.
    # The third (and last) day, there is 6h of masking. Mask_frac == 6/24.
    raw_sinewave_mask.period = pd.Timedelta('3days')
    assert raw_sinewave_mask.mask_fraction_period(period='1D') == approx(
        [0, 2/24, 6/24], 0.01
    )
