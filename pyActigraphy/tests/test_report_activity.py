# from generate_dataset import generate_gaussian_noise
from generate_dataset import generate_series
# from generate_dataset import generate_squarewave
from generate_dataset import generate_sinewave

import numpy as np
import pandas as pd
import pyActigraphy
import pytest

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2000 08:00:00'
N = 1440
period = pd.Timedelta(N*sampling_period, unit='s')
A = 100  # oscillation amplitude

# Set seed for reproducibility
np.random.seed(0)

sine_wave = generate_series(
    generate_sinewave(
        N=N,  # number of samples
        T=24*60*sampling_period,  # period in sec: 24*60*60
        Ts=sampling_period,  # sampling rate (sec.)
        A=A,  # oscillation amplitude
        add_noise=False,  # add gaussian noise
        noise_power=0,
        offset=True  # offset oscillations between 0 and +2A
    ),
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

grd_truth = pd.DataFrame.from_dict(
    {'activity level': {0: 'Sedentary', 1: 'Moderate', 2: 'Vigorous'},
     'sum': {0: 26163.44911160938, 1: 117836.55088839062, 2: 0.0},
     'mean': {0: 36.33812376612414, 1: 163.66187623387586, 2: np.nan},
     'median': {0: 29.289321881345245, 1: 170.71067811865476, 2: np.nan},
     'std': {0: 30.797448785194323, 1: 30.797448785194312, 2: np.nan},
     'count': {0: 720, 1: 720, 2: 0},
     'ID': {0: 'raw_sinewave', 1: 'raw_sinewave', 2: 'raw_sinewave'}}
)
grd_truth.loc[:, 'activity level'] = pd.Categorical(
    ['Sedentary', 'Moderate', 'Vigorous'],
    categories=['Sedentary', 'Moderate', 'Vigorous'],
    ordered=True
)


def test_report_activity_default():

    raw_sinewave.create_activity_report(
        cut_points=[100, 200],
        labels=['Sedentary', 'Moderate', 'Vigorous']
    )
    # assert (raw_sinewave.activity_report.equals(grd_truth))
    pd.testing.assert_frame_equal(
        raw_sinewave.activity_report,
        grd_truth,
        check_dtype=True,
        check_exact=False,
        rtol=1.0e-3
    )


def test_report_activity_perc():

    raw_sinewave.create_activity_report(
        cut_points=[1/2, 1],
        labels=['Sedentary', 'Moderate', 'Vigorous']
    )
    report_sum = raw_sinewave.activity_report[
        ['sum', 'mean', 'median', 'count']
    ].sum()

    assert (
        (report_sum['sum'] == N*A)
        & (report_sum['mean'] == pytest.approx(2*A, rel=0.01))
        & (report_sum['median'] == pytest.approx(2*A, rel=0.01))
        & (report_sum['count'] == pytest.approx(N, rel=0.05))
    )
