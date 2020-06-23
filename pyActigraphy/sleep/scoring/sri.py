import numpy as np
import pandas as pd


def prob_stability(ts, threshold):
    r''' Compute the probability that any two consecutive time
    points are in the same state (wake or sleep)'''

    # Construct binarized data if requested
    data = np.where(ts > threshold, 1, 0) if threshold is not None else ts

    # Compute stability as $\delta(s_i,s_{i+1}) = 1$ if $s_i = s_{i+}$
    # Two consecutive values are equal if the 1st order diff is equal to zero.
    # The 1st order diff is either +1 or -1 otherwise.
    prob = np.mean(1-np.abs(np.diff(data)))

    return prob


def sri_profile(data, threshold):
    r''' Compute daily profile of sleep regularity indices '''
    # Group data by hour/minute/second across all the days contained in the
    # recording
    data_grp = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ])
    # Apply prob_stability to each data group (i.e series of consecutive points
    # that are 24h apart for a given time of day)
    sri_prof = data_grp.apply(prob_stability, threshold=threshold)
    sri_prof.index = pd.timedelta_range(
        start='0 day',
        end='1 day',
        freq=data.index.freq,
        closed='left'
    )
    return sri_prof


def sri(data, threshold=None):
    r''' Compute sleep regularity index (SRI)'''

    # Compute daily profile of sleep regularity indices
    sri_prof = sri_profile(data, threshold)

    # Calculate SRI coefficient
    sri_coef = 200*np.mean(sri_prof.values)-100

    return sri_coef
