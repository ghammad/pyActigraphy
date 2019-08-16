# coding: utf-8

# Synthetic datasets

# pyActigraphy uses synthetic datasets in order to validate the implementation
# of its functionalities.

import pandas as pd
import numpy as np


# Helper functions
def generate_datetime_index(start='01/01/2018', N=1000, sampling_period=60):
    """Generates a datetime index

    Parameters
    ----------

    start: str or datetime-like, optional
        Left bound for generating dates.
        Default is '01/01/2018'
    N: int, optional
        Number of periods to generate
        Default is 1000.
    sampling_period: int, optional
        Sampling period, in seconds.
        Default is 60.

    Returns
    -------

    dti: pandas.DatetimeIndex
    """

    dti = pd.date_range(
        start=start,
        periods=N,
        freq=pd.offsets.Second(sampling_period)
    )
    return dti


def generate_series(data, start='01/01/2018', sampling_period=60):
    """Generates a time series with index

    Parameters
    ----------

    data: array-like
        Data array.
    start: str or datetime-like, optional
        Left bound for generating dates.
        Default is '01/01/2018'
    sampling_period: int, optional
        Sampling period, in seconds.
        Default is 60.

    Returns
    -------

    idata: pandas.Series
    """

    index = generate_datetime_index(
        start=start,
        N=len(data),
        sampling_period=sampling_period
    )

    idata = pd.Series(
        data=data,
        index=index
    )

    return idata


# Periodic datasets

def generate_sinewave(
    N=10080,
    T=86400,
    Ts=60,
    A=100,
    add_noise=False,
    noise_power=100,
    offset=False
):
    """Generates a synthetic sine wave, corrupted by a white noise.

    Parameters
    ----------

    N: int, optional
        Number of points to generate.
        Default is 10080 (7 days at a sampling period of 60s).
    T: float, optional
        Period (in seconds) of the sinusoidal signal.
        Default is 24*60*60 (1 day).
    Ts: float, optional
        Sampling period (in seconds).
        Default is 60.
    A: float, optional
        Amplitude of the sinusoidal signal.
        Default is 100.
    add_noise: bool, optional
        If set to true, add white Gaussian noise to the main signal.
        Default is False.
    noise_power: float, optional
        Amplitude of the white Gaussian noise.
        Default is 100.
    offset: bool, optional
        If set to True, an offset is applied to the signal so that
        it is comprised between (0, 2A), instead of (-A, +A).
    """

    time = np.arange(N) * Ts
    signal = A*np.sin((2*np.pi/T)*time)

    if offset:
        signal += A

    if add_noise:
        signal += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

    return signal


def generate_squarewave(
    N=10080,
    T=86400,
    Ts=60,
    A=100,
    add_noise=False,
    noise_power=100,
    offset=False
):
    """Generates a synthetic square wave, corrupted by a white noise.

    Parameters
    ----------

    N: int, optional
        Number of points to generate.
        Default is 10080 (7 days at a sampling period of 60s).
    T: float, optional
        Period (in seconds) of the sinusoidal signal.
        Default is 24*60*60 (1 day).
    Ts: float, optional
        Sampling period (in seconds).
        Default is 60.
    A: float, optional
        Amplitude of the sinusoidal signal.
        Default is 100.
    add_noise: bool, optional
        If set to true, add white Gaussian noise to the main signal.
        Default is False.
    noise_power: float, optional
        Amplitude of the white Gaussian noise.
        Default is 100.
    offset: bool, optional
        If set to True, an offset is applied to the signal so that
        it is comprised between (0, 2A), instead of (-A, +A).
    """

    signal = A*np.sign(generate_sinewave(N, T, Ts, 1, False, 0))

    if offset:
        signal += A

    if add_noise:
        signal += np.random.normal(
            scale=np.sqrt(noise_power),
            size=signal.shape
        )

    return signal

# Non-Periodic datasets


def generate_gaussian_noise(N=1000, mu=100, sigma=10):
    """Generates a Gaussian noise signal

    Parameters
    ----------

    N: int, optional
        Number of points to generate.
        Default is 1000.
    mu: float, optional
        Mean of the Gaussian distribution.
        Default is 100.
    sigma: float, optional
        Standard deviation of the Gaussian distribution.
        Default is 10.

    """

    signal = np.random.normal(mu, sigma, N)

    return signal


def generate_sequences(N=1000, p=.3, max_rand_int=10, seed=0):
    """Generates consecutive sequences of identical random numbers
    whose lengths follow a Geometric distribution.

    Parameters
    ----------

    N: int, optional
        Number of sequences to generate.
        Default is 1000.
    p: float, optional
        Probability of success of each individual trial.
        Default is 10.
    max_rand_int: int, optional
        Maximum value for the random integer generator.
        Default is 10.
    seed: int, optional
        Seed for the random number generator.
        Default is 0.
    """

    # Set the random generator seed
    np.random.seed(seed)

    # Define the lengths of the consecutive sequences
    s = np.random.geometric(p=p, size=N)

    # Define the activity counts in each sequence
    rand_ints = np.random.randint(0, max_rand_int+1, N)

    # Create the signal as the repetition of the activity count
    # in each sequence.
    signal = np.repeat(rand_ints, s, axis=0)

    return signal


def generate_inactivity(
    N=1000, inactivity_index=500, inactivity_length=100, seed=0
):
    """Generates white gaussian noise signal,
    mixed with a flat signal at a user-specified index.

    Parameters
    ----------

    N: int, optional
        Number of white gaussian noise signal points to generate.
        Default is 1000.
    inactivity_index: int, optional
        Index at which to insert the flat signal.
        Default is 500.
    inactivity_length: int, optional
        Number of flat signal (i.e 0) points to generate.
        Default is 100.
    seed: int, optional
        Seed for the random number generator.
        Default is 0.
    """

    # Set the random generator seed
    np.random.seed(seed)

    # Generate a white gaussian noise signal
    abs_gaussian_noise = np.abs(np.random.normal(size=N))

    # Create the signal by concatenating the white gaussian noise
    # and the flat signals.
    signal = np.concatenate(
        (abs_gaussian_noise[:inactivity_index],
         np.zeros(inactivity_length),
         abs_gaussian_noise[inactivity_index:]),
        axis=0)

    return signal
