# from generate_dataset import generate_gaussian_noise
from generate_dataset import generate_series
# from generate_dataset import generate_squarewave
from generate_dataset import generate_sinewave

import numpy as np
from lmfit import Parameters
import pandas as pd
import pyActigraphy
from pyActigraphy.analysis import Cosinor
from pytest import approx

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2000 08:00:00'
N = 10080
period = pd.Timedelta(N*sampling_period, unit='s')

# Set seed for reproducibility
np.random.seed(0)

sine_wave = generate_series(
    generate_sinewave(
        N=N,  # number of samples
        T=1440*60,  # period in sec: 24*60*60
        Ts=60,  # sampling rate (sec.)
        A=100,  # oscillation amplitude
        add_noise=True,  # add gaussian noise
        noise_power=100,
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

fit_params = Parameters()
fit_params.add('Acrophase', value=0, min=-np.pi, max=np.pi)
fit_params.add('Amplitude', value=50, min=0)
fit_params.add('Period', value=1440, min=0)  # Dummy value
fit_params.add('Mesor', value=100, min=0)

cosinor = Cosinor()


def test_cosinor_default_params():

    assert (
        (cosinor.fit_initial_params['Acrophase'] == np.pi)
        & (cosinor.fit_initial_params['Amplitude'] == 50)
        & (cosinor.fit_initial_params['Period'] == 1440)
        & (cosinor.fit_initial_params['Mesor'] == 50)
    )


def test_cosinor_set_params():

    cosinor.fit_initial_params = fit_params
    assert (
        (cosinor.fit_initial_params['Acrophase'] == 0)
        & (cosinor.fit_initial_params['Amplitude'] == 50)
        & (cosinor.fit_initial_params['Period'] == 1440)
        & (cosinor.fit_initial_params['Mesor'] == 100)
    )


def test_cosinor_fit():

    # From sinus (data) to cosinus (fit function): Add - Pi/2 to initial phase
    results = cosinor.fit(raw_sinewave.data)
    assert (
        (results.params['Acrophase'].value == approx(-np.pi/2, abs=0.05))
        & (results.params['Amplitude'].value == approx(100.0, rel=0.05))
        & (results.params['Period'].value == approx(1440.0, rel=0.05))
        & (results.params['Mesor'].value == approx(100.0, rel=0.05))
    )
