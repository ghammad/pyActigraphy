from generate_dataset import generate_sequences
from generate_dataset import generate_series

import numpy as np
import pandas as pd
import pyActigraphy
import pytest

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2018 00:00:00'
N = 30240*sampling_period
period = pd.Timedelta(N, unit='s')
seq_prob = 0.7311  # ~ e/e+1
seq_max_length = 50

sequences = generate_series(
    generate_sequences(
        N=N-1,
        p=seq_prob,
        max_rand_int=seq_max_length
    ),
    start=start_time,
    sampling_period=sampling_period
)

seq = pyActigraphy.io.BaseRaw(
    name='raw_sequence',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=sequences,
    light=None
)


def test_kra_sequences():

    assert seq.kRA(0) == pytest.approx(seq_prob, rel=0.01)


def test_kra_logit():

    assert seq.kRA(0, logit=True) == pytest.approx(1.0, rel=0.04)


def test_kar_sequences():

    assert seq.kAR(0) == pytest.approx(0, abs=0.02)


def test_kar_logit():

    assert seq.kAR(0, logit=True) == pytest.approx(np.log(0.01), rel=0.1)
