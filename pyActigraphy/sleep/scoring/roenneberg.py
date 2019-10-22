import numpy as np
import pandas as pd
from .utils import consecutive_values, correlation_series
from .utils import find_highest_peak_idx


def _extract_trend(data, period='24h', min_period='12h', closed='right'):
    r''' Calculate the trend of the time series using a centered 24 hours window

    Parameters
    ----------
    data : pandas.Series
        Data from which to extract the trend.

    period: str, optional
        Time period for the rolling window.
        Default is '24h'.

    min_period: str, optional
        Minimum time period required for the rolling window to produce a
        value. Values default to NaN otherwise.
        Default is '12h'.

    closed: str, optional
        Make the interval closed on the 'right', 'left', 'both' or 'neither'
        endpoints.
        Default is 'right'.
    Returns
    -------
    trend : pandas.Series
        Trend data

        '''
    win_size = int(pd.Timedelta(period)/data.index.freq)
    min_win_size = int(pd.Timedelta(min_period)/data.index.freq)

    return data.rolling(
        win_size,
        center=True,
        min_periods=min_win_size,
        closed=closed).mean()


def _sleep_wake_categorization(data, trend, threshold=0.15):
    r'''Categorize data as wake (0) or sleep (1)

    If the data are above `thr` % of the trend, they are assigned wake.
    Otherwise, they are assigned sleep.

    Parameters
    ----------
    data : pandas.Series
        Data to categorize

    trend : pandas.Series
        Data with trend

    threshold : int, optional
        Fraction of the trend to use as a threshold for categorization.
        Default is '0.15'.

    Return
    ----------
    sw : pandas.Series
        Categorized data

    '''

    # Binarize data according to specified trend and threshold.
    # True evaluates to 1
    sw = (data <= threshold*trend).astype(int)

    # Set to np.nan values for which the trend is nan. Needed because
    # comparison with nan yields False
    return sw.where(trend.notnull())


def _find_sleep_bout_seeds(data, min_period='30Min'):
    r'''Find indices of the start of all series of consecutive values of 1.

    Parameters
    ----------
    data : pandas.Series
        Binarized data

    min_period : str, optional
        Minimum time period required for the series of consecutive values.
        Default is '30Min'.

    Return
    ----------
    seeds : (N,) array_like
        Array with the start indices.

    '''

    win_size = int(pd.Timedelta(min_period)/data.index.freq)

    seed_indices = consecutive_values(data.values, 1, win_size)

    # Shift seed index by the window size as the window closes on the right.
    return data.index[seed_indices[:, 0]]


def _test_sleep_bout(uncleaned_binary_data, period='12h'):
    r'''Calculate the correlation of the raw input binary sleep bout with the
    test series.

    Parameters
    ----------
    uncleaned_binary_data : pd.Series
        Raw binary sleep series.

    period : str, optional
        Maximal period of the test series.
        Default is '12h'


    Return
    ----------
    corr : (N,) array-like
        List of Pearson's correlations.

    '''
    win_size = int(pd.Timedelta(period)/uncleaned_binary_data.index.freq)

    # Creation of the data to test
    test_data = uncleaned_binary_data.iloc[:win_size].values

    # Creation of the test series (i.e triangular upper matrix with only ones)
    test_series = np.tril(np.ones(min(win_size, len(test_data))))

    # Calculate the list of Pearson's correlations with the test series
    corr = correlation_series(test_data, test_series)

    return corr


def _clean_sleep_bout(uncleaned_binary_data, period='12h', n_succ=3):
    r'''Find the time index of the sleep offset.

    The sleep offset is defined as the index at which the current time series
    maximizes the Pearson's correlation with a consolidated sleep bout of the
    same size.

    Parameters
    ----------
    uncleaned_binary_data : pd.Series
        Raw binary sleep series.

    period : str, optional
        Maximal period of the test series.
        Default is '12h'

    n_succ : int, optional
        Number of successive elements to consider when searching for the
        maximum correlation peak.
        Default is 3.

    Return
    ----------
    sleep_offset : pd.date_time
        Time index of the sleep offset. None if none is found.

    '''
    # Calculate the list of Pearson's correlations with the test series
    corr = _test_sleep_bout(uncleaned_binary_data, period=period)

    # Find the date_time index corresponding to the highest correlation peak
    sleep_offset_idx = find_highest_peak_idx(corr, n_succ=n_succ)

    if sleep_offset_idx is not None:
        return uncleaned_binary_data.index[sleep_offset_idx]
    else:
        return sleep_offset_idx


def roenneberg(
    data,
    trend_period='24h',
    min_trend_period='12h',
    threshold=0.15,
    min_seed_period='30Min',
    max_test_period='12h',
    n_succ=3
):
    r'''Automatic sleep detection.

    Identification of consolidated sleep episodes using the
    algorithm developped by Roenneberg et al. [1]_.

    Parameters
    ----------
    trend_period: str, optional
        Time period for the rolling window.
        Default is '24h'.

    min_trend_period: str, optional
        Minimum time period required for the rolling window to produce a
        value. Values default to NaN otherwise.
        Default is '12h'.

    threshold : int, optional
        Fraction of the trend to use as a threshold for categorization.
        Default is '0.15'.

    min_seed_period: str, optional
        Minimum time period required to identify a potential sleep onset.
        Default is '30Min'.

    max_test_period : str, optional
        Maximal period of the test series.
        Default is '12h'

    n_succ : int, optional
        Number of successive elements to consider when searching for the
        maximum correlation peak.
        Default is 3.

    Return
    ----------
    sw : pandas.core.Series
        Time series containing the estimated periods of rest (1) and
        activity (0).

    '''

    # Extract trend
    trend = _extract_trend(
        data=data,
        period=trend_period,
        min_period=min_trend_period
    )

    # Categorize as sleep or wake
    sw = _sleep_wake_categorization(data, trend, threshold=threshold)

    # Find start time of binary sleep series of length 'min_seed_period'
    seeds = _find_sleep_bout_seeds(sw, min_period=min_seed_period)

    # Score all potential sleep epochs (1) before the first seed as wake (0)
    sw.iloc[:sw.index.get_loc(seeds[0])].replace(1, 0, inplace=True)

    # Loop over the seeds
    sot = []  # list of sleep onset and offset times
    for seed in seeds:
        # if the seed is anterior the last sleep offset, discard it.
        if(len(sot) > 0 and seed < sot[-1][1]):
            continue

        # Score all potential sleep epochs (1) before current seed as wake (0)
        if(len(sot) > 0):
            sw.iloc[
                sw.index.get_loc(sot[-1][1])+1:sw.index.get_loc(seed)
            ].replace(1, 0, inplace=True)

        # Find sleep offset
        sleep_onset = seed
        sleep_offset = _clean_sleep_bout(
            sw.loc[sleep_onset:sleep_onset+pd.Timedelta(max_test_period)],
            period=max_test_period,
            n_succ=n_succ
        )
        if sleep_offset is not None:
            sw.loc[sleep_onset:sleep_offset] = 1
            sot.append((sleep_onset, sleep_offset))

    # Score all potential sleep epochs (1) after last sleep offset as wake (0)
    sw.iloc[sw.index.get_loc(sot[-1][1])+1:].replace(1, 0, inplace=True)
    # return sot
    return sw
