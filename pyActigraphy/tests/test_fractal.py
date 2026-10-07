from generate_dataset import generate_series
import inspect
import numpy as np
import os.path as op
import pandas as pd
from pyActigraphy.analysis import Fractal
import pytest


############
# Settings #
############

sampling_period = 30
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2018 00:00:00'
N = 7*1440*2  # *sampling_period
n_array = np.geomspace(5, 500, num=20, endpoint=True, dtype=int)
q_array = [-5, -3, -1, 0, 1, 3, 5]

rel_err = 0.025 # relative error of 2.5% on Hurst exponent estimations

####################
# Fractal datasets #
####################

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')

# Brownian noise: h(q) = 1+H with H=0.5
with np.load(op.join(data_dir,'fractals_brownian_noise_h_0p5.npz')) as bn:
    bn_sample = generate_series(
        bn['bn_sample'],
        start=start_time,
        sampling_period=sampling_period
    )

# Fractional Brownian motion: h(q) = 1+H with H = 0.9 (in this example)
with np.load(op.join(data_dir,'fractals_fbrownian_motion_h_0p9.npz')) as fbm:
    fbm_sample = generate_series(
        fbm['fbm_sample'],
        start=start_time,
        sampling_period=sampling_period
    )

# Fractional Gaussian noise: h(q) = H with H = 0.6 (in this example)
with np.load(op.join(data_dir,'fractals_fgaussian_noise_h_0p6.npz')) as fgn:
    fgn_sample = generate_series(
        fgn['fgn_sample'],
        start=start_time,
        sampling_period=sampling_period
    )


###########################
# Associated fluctuations #
###########################

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


###############################
# Generalized Hurst exponents #
###############################

# DFA on Brownian noise
bn_h, bn_h_err = Fractal.generalized_hurst_exponent(
    F_n=test_bn, n_array=n_array, log=False, x_center=False
)

# DFA on Brownian noise with overlapping windows
bn_h_overlap, bn_h_overlap_err = Fractal.generalized_hurst_exponent(
    F_n=test_bn_overlap, n_array=n_array, log=False, x_center=False
)

# MFDFA on Brownian noise
bn_q_h = np.fromiter((Fractal.generalized_hurst_exponent(
        F_n=test_bn_q[:, q_idx], n_array=n_array, log=False, x_center=False
    )[0] for q_idx in range(len(q_array))),
    dtype='float',
    count=len(q_array)
)

# MFDFA on Brownian noise with overlapping windows
bn_q_h_overlap = np.fromiter((Fractal.generalized_hurst_exponent(
        F_n=test_bn_q_overlap[:, q_idx],
        n_array=n_array,
        log=False,
        x_center=False
    )[0] for q_idx in range(len(q_array))),
    dtype='float',
    count=len(q_array)
)

# DFA on Fractional Brownian motion
fbm_h, fbm_h_err = Fractal.generalized_hurst_exponent(
    F_n=test_fbm, n_array=n_array, log=False, x_center=False
)

# DFA on Fractional Brownian motion with overlapping windows
fbm_h_overlap, fbm_h_overlap_err = Fractal.generalized_hurst_exponent(
    F_n=test_fbm_overlap, n_array=n_array, log=False, x_center=False
)

# DFA on Fractional Gaussian noise
fgn_h, fgn_h_err = Fractal.generalized_hurst_exponent(
    F_n=test_fgn, n_array=n_array, log=False, x_center=False
)

# Crossover with a straight line as input
h_ratios, h_ratios_err, n_x = Fractal.crossover_search(
    F_n=n_array, n_array=n_array, n_min=3, log=True
)


def test_dfa_bn():

    assert bn_h-1 == pytest.approx(0.5, rel=rel_err)


def test_dfa_bn_overlap():

    assert bn_h_overlap-1 == pytest.approx(0.5, rel=rel_err)


def test_dfa_fbm():

    assert fbm_h-1 == pytest.approx(0.9, rel=rel_err)


def test_dfa_fbm_overlap():

    assert fbm_h_overlap-1 == pytest.approx(0.9, rel=rel_err)


def test_dfa_fgn():

    assert fgn_h == pytest.approx(0.6, rel=rel_err)


def test_dfa_parallel():

    assert np.all(test_bn == test_bn_parallel)


def test_mfdfa_bn():

    assert np.mean(bn_q_h-1) == pytest.approx(0.5, rel=rel_err)


def test_mfdfa_bn_overlap():

    assert np.mean(bn_q_h_overlap-1) == pytest.approx(0.5, rel=rel_err)


def test_mfdfa_parallel():

    assert np.all(test_bn_q == test_bn_q_parrallel)


def test_crossover_search():
    # Ratio should be constant and equal to 1 (+/-n_sigma)
    assert np.all(h_ratios == pytest.approx(1.0, rel=0.001))
