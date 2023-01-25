from generate_dataset import generate_series
from stochastic.processes.noise import BrownianNoise, FractionalGaussianNoise
from stochastic.processes.continuous import FractionalBrownianMotion
from stochastic import random

import numpy as np
import pandas as pd
from pyActigraphy.analysis import Fractal
import pytest

##################
# Simulated data #
##################

sampling_period = 30
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2018 00:00:00'
N = 7*1440*2  # *sampling_period
n_array = np.geomspace(5, 1000, num=40, endpoint=True, dtype=int)
q_array = [-5, -3, -1, 0, 1, 3, 5]

# Stochastic generators
bn = BrownianNoise(t=1)
fbm = FractionalBrownianMotion(hurst=0.9, t=1)
fgn = FractionalGaussianNoise(hurst=0.6, t=1)

random.seed(0)

# Brownian noise: h(q) = 1+H with H=0.5
bn_sample = generate_series(
    bn.sample(N-1),
    start=start_time,
    sampling_period=sampling_period
)

# Fractional Brownian motion: h(q) = 1+H with H = 0.9 (in this example)
fbm_sample = generate_series(
    fbm.sample(N-1),
    start=start_time,
    sampling_period=sampling_period
)

# Fractional Gaussian noise: h(q) = H with H = 0.6 (in this example)
fgn_sample = generate_series(
    fgn.sample(N),
    start=start_time,
    sampling_period=sampling_period
)

# Associated fluctuations
test_bn = Fractal.dfa(bn_sample, n_array, deg=2, log=False)
test_bn_overlap = Fractal.dfa(
    bn_sample, n_array, deg=2, overlap=True, log=False
)
test_bn_parallel = Fractal.dfa_parallel(
    bn_sample, n_array, deg=2, log=False, n_jobs=4
)
test_bn_q = Fractal.mfdfa(bn_sample, n_array, q_array, deg=2, log=False)
test_bn_q_overlap = Fractal.mfdfa(
    bn_sample, n_array, q_array, overlap=True, deg=2, log=False
)
test_bn_q_parrallel = Fractal.mfdfa_parallel(
    bn_sample, n_array, q_array, deg=2, log=False, n_jobs=4
)

test_fbm = Fractal.dfa(fbm_sample, n_array, deg=2, log=False)
test_fbm_overlap = Fractal.dfa(
    fbm_sample, n_array, deg=2, overlap=True, log=False
)

test_fgn = Fractal.dfa(fgn_sample, n_array, deg=2, log=False)

# Generalized Hurst exponents
bn_h, bn_h_err = Fractal.generalized_hurst_exponent(
    F_n=test_bn, n_array=n_array, log=False, x_center=False
)
bn_h_overlap, bn_h_overlap_err = Fractal.generalized_hurst_exponent(
    F_n=test_bn_overlap, n_array=n_array, log=False, x_center=False
)
bn_q_h = np.fromiter((Fractal.generalized_hurst_exponent(
        F_n=test_bn_q[:, q_idx], n_array=n_array, log=False, x_center=False
    )[0] for q_idx in range(len(q_array))),
    dtype='float',
    count=len(q_array)
)

bn_q_h_overlap = np.fromiter((Fractal.generalized_hurst_exponent(
        F_n=test_bn_q_overlap[:, q_idx],
        n_array=n_array,
        log=False,
        x_center=False
    )[0] for q_idx in range(len(q_array))),
    dtype='float',
    count=len(q_array)
)

fbm_h, fbm_h_err = Fractal.generalized_hurst_exponent(
    F_n=test_fbm, n_array=n_array, log=False, x_center=False
)
fbm_h_overlap, fbm_h_overlap_err = Fractal.generalized_hurst_exponent(
    F_n=test_fbm_overlap, n_array=n_array, log=False, x_center=False
)

fgn_h, fgn_h_err = Fractal.generalized_hurst_exponent(
    F_n=test_fgn, n_array=n_array, log=False, x_center=False
)

# Crossover with a straight line as input
h_ratios, h_ratios_err, n_x = Fractal.crossover_search(
    F_n=n_array, n_array=n_array, n_min=3, log=True
)
n_sigma = 3


def test_dfa_bn():

    assert bn_h-1 == pytest.approx(0.5, rel=0.05)


def test_dfa_bn_overlap():

    assert bn_h_overlap-1 == pytest.approx(0.5, rel=0.01)


def test_dfa_fbm():

    assert fbm_h-1 == pytest.approx(0.9, rel=0.05)


def test_dfa_fbm_overlap():

    assert fbm_h_overlap-1 == pytest.approx(0.9, rel=0.025)


def test_dfa_fgn():

    assert fgn_h == pytest.approx(0.6, rel=0.05)


def test_dfa_parallel():

    assert np.all(test_bn == test_bn_parallel)


def test_mfdfa_bn():

    assert np.mean(bn_q_h-1) == pytest.approx(0.5, rel=0.05)


def test_mfdfa_bn_overlap():

    assert np.mean(bn_q_h_overlap-1) == pytest.approx(0.5, rel=0.01)


def test_mfdfa_parallel():

    assert np.all(test_bn_q == test_bn_q_parrallel)


def test_crossover_search():
    # Ratio should be constant and equal to 1 (+/-n_sigma)
    assert np.all(h_ratios == pytest.approx(1.0, rel=0.001))
