import numpy as np
import pandas as pd
from .utils import consecutive_values, correlation_series
from .utils import find_highest_peak_idx


def _extract_trend(data, period='24h', min_period='12h'):
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

    Returns
    -------
    trend : pandas.Series
        Trend data

        '''
    win_size = int(pd.Timedelta(period)/data.index.freq)
    min_win_size = int(pd.Timedelta(min_period)/data.index.freq)

    return data.rolling(win_size, center=True, min_periods=min_win_size).mean()


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
    return pd.Series(
        np.where(data > threshold*trend, 0, 1),
        index=data.index
    )


def _find_sleep_bout_seeds(data, min_period='30Min'):

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

    # creation of the data to test
    test_data = uncleaned_binary_data.iloc[:win_size].values

    # creation of the test series (i.e triangular upper matrix with only ones)
    test_series = np.tril(np.ones(min(win_size, len(test_data))))

    # calculate the list of Pearson's correlations with the test series
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
    # calculate the list of Pearson's correlations with the test series
    corr = _test_sleep_bout(uncleaned_binary_data, period=period)

    # find the date_time index corresponding to the highest correlation peak
    sleep_offset_idx = find_highest_peak_idx(corr, n_succ=n_succ)

    if sleep_offset_idx is not None:
        return uncleaned_binary_data.index[sleep_offset_idx]
    else:
        return sleep_offset_idx


def chronosapiens(
    data,
    trend_period='24h',
    min_trend_period='12h',
    threshold=0.15,
    min_seed_period='30Min',
    min_corr_period='12h',
    n_succ=3
):

    # Extract trend
    trend = _extract_trend(
        data=data,
        period=trend_period,
        min_period=min_trend_period
    )

    # Categorize as sleep or wake
    sw = _sleep_wake_categorization(data, trend, threshold=threshold)

    # Find binary sleep series of length 'min_seed_period' (a.k.a. seeds):
    seed_ts = _find_sleep_bout_seeds(sw, min_period=min_seed_period)

    sot = []
    # Loop over the seeds
    for seed in seed_ts:
        # if the seed is anterior the last sleep offset, discard it.
        if(len(sot) > 0 and seed < sot[-1][1]):
            continue
        sleep_onset = seed
        sleep_offset = _clean_sleep_bout(
            sw.loc[sleep_onset:sleep_onset+pd.Timedelta(min_corr_period)],
            period=min_corr_period,
            n_succ=n_succ
        )
        if sleep_offset is not None:
            sot.append((sleep_onset, sleep_offset))

    return sot
