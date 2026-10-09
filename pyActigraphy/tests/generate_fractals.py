# coding: utf-8

# Fractal datasets

# pyActigraphy uses fractal datasets in order to validate the implementation
# of the DFA method.

import numpy as np

from stochastic.processes.noise import BrownianNoise, FractionalGaussianNoise
from stochastic.processes.continuous import FractionalBrownianMotion
from stochastic import random


# Number of samples
N = 7*1440*2  # *sampling_period

# Random number generator
rng = np.random.default_rng(0)

# Stochastic generators
bn = BrownianNoise(t=1,rng=rng)
fbm = FractionalBrownianMotion(hurst=0.9, t=1,rng=rng)
fgn = FractionalGaussianNoise(hurst=0.6, t=1,rng=rng)

# Brownian noise: h(q) = 1+H with H=0.5
bn_sample = bn.sample(N-1)
np.savez('data/fractals_brownian_noise_h_0p5', bn_sample)

# Fractional Brownian motion: h(q) = 1+H with H = 0.9 (in this example)
fbm_sample = fbm.sample(N-1)
np.savez('data/fractals_fbrownian_motion_h_0p9', fbm_sample)

# Fractional Gaussian noise: h(q) = H with H = 0.6 (in this example)
fgn_sample = fgn.sample(N)
np.savez('data/fractals_fgaussian_noise_h_0p6', fgn_sample)