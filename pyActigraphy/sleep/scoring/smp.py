import numpy as np
import pandas as pd


def minutes_since_midnight(index, scale=2*np.pi, norm=1440.):
    r''' Translate time index into number of minutes since midnight '''

    # Input tuple represents (hour,minute,second)

    # Minutes since midnight
    msm = 60.0*index[0]+index[1]+index[2]/60.0

    return msm*scale/norm


def sum_of_sine(msm, scores):
    r''' Compute sum of sines '''

    return np.sum(scores*np.sin(msm))


def sum_of_cosine(msm, scores):
    r''' Compute sum of cosines '''

    return np.sum(scores*np.cos(msm))


def sum_over_time_of_day(data):
    r''' Compute the sum of the sum of cosine/sine of the data across days '''

    # Group data by hour/minute/second across all the days contained in the
    # recording
    grouped_data = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ])

    # Sum of sine
    sos = np.zeros(len(grouped_data.indices))
    # Sum of cosine
    soc = np.zeros_like(sos)

    # For each time of day
    for idx, (k, v) in enumerate(grouped_data.indices.items()):

        # Minutes since midnight
        msm = minutes_since_midnight(k)

        sos[idx] = sum_of_sine(msm, data.iloc[v])
        soc[idx] = sum_of_cosine(msm, data.iloc[v])

    return np.sum(sos), np.sum(soc)


def sleep_midpoint(data, threshold=None):
    r''' Calculate the sleep midpoint (in minute since midnight) '''

    # Construct binarized data if requested
    data = pd.Series(
        np.where(data > threshold, 1, 0),
        index=data.index
    ) if threshold is not None else data

    # Sum of sum of sine/cosine
    ssos, ssoc = sum_over_time_of_day(data)

    smp = 1440/(2*np.pi)*np.arctan2(ssos, ssoc)

    return smp
