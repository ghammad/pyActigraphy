import numpy as np
from numba import jit, prange


@jit(nopython=True)
def pearsonr(x, y):
    r'''Local implementation of the calculation of the Pearson's r correlation

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------
    r : float
        Pearson's correlation coefficient

    '''
    xmean = x.mean()
    ymean = y.mean()

    xm = x - xmean
    ym = y - ymean

    normxm = np.linalg.norm(xm)
    normym = np.linalg.norm(ym)

    r = np.dot(xm/normxm, ym/normym)

    # If r>1 or r<-1, due to precision effect, return 1 or -1.
    return max(min(r, 1.0), -1.0)


@jit(nopython=True, parallel=True)
def correlation_series(x, Y):
    r'''Correlation between the input array x and each column of the matrix Y

    Parameters
    ----------
    x : (N,) array_like
        Input
    Y : (N,M) array_like
        Input

    Returns
    -------
    corr : (N,) array_like
        List of Pearson's correlation coefficients

    '''
    n, m = Y.shape
    corr = np.empty(n, dtype=np.float64)
    for i in prange(n):
        corr[i] = pearsonr(x, Y[i, :])
    return corr


def rolling_window(a, win_size):
    r'''Split input array in an array of window-sized arrays, shifted by one
    element. Emulate rolling function of pandas.Series.

    Parameters
    ----------
    a : (N,) array_like
        Input
    win_size : int
        Size of the rolling window

    Returns
    -------
    roll : (N,win_size) array_like
        Array containing the successive windows.

    '''
    shape = a.shape[:-1] + (a.shape[-1] - win_size + 1, win_size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def is_a_peak(x):
    r'''Define a peak as an element whose value is greater than those of all
    the following elements.

    Parameters
    ----------
    x : (N,) array_like
        Input

    Returns
    -------
    peak : bool
        True if the first value is greater than any succesive one.

    '''

    return np.all(x[0] > x[1:])


def find_highest_peak_idx(x, n_succ=3):
    r'''Find the index of the highest peak.

    A peak is defined as an element whose value is higher than those of the
    successive Nth elements.

    Parameters
    ----------
    x : np.ndarray
        Array containing the peak candidates.

    n_succ : int, optional
        Number of successive elements to consider when searching for a peak.
        Default is 3.

    Return
    ----------
    idx : int
        Index of the highest peak.


    Notes
    -----
    When several peaks with the same heigh are found, the index of the first
    peak is returned.

    '''

    peak_candidate_idx, = np.apply_along_axis(
        is_a_peak,
        axis=1,
        arr=rolling_window(x, n_succ+1)
    ).nonzero()

    if(len(peak_candidate_idx) > 0):
        highest_peak_idx, = np.where(x == np.max(x[peak_candidate_idx]))
        return highest_peak_idx[0]
    else:
        return None


def consecutive_values(x, target=1, min_length=10):
    r'''Returns the start and end indices of series of consecutive values


    Parameters
    ----------
    x : (N,) array_like
        Input

    target : int, optional
        Value of the successive elements to consider.
        Default is 1.

    min_length : int, optional
        Minimal number of successive elements.

    Return
    ----------
    ranges : (N,2) array_like
        Array with the start and end of the series.

    '''
    # Create an array that is 1 where x is equal to target,
    # and pad each end with an extra 0.
    targets = np.concatenate(([0], np.equal(x, target).view(np.int8), [0]))
    absdiff = np.abs(np.diff(targets))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    return ranges[np.where(ranges[:, 1] - ranges[:, 0] >= min_length)]


def rescore_if_preceded(scoring, n_periods, n_previous, sleep_score=1):
    r'''Returns a binary series with 0 at indices of epochs scored as sleep
    that are preceded by epochs scored as wake.

    Parameters
    ----------
    scoring : np.ndarray
        Binary series with sleep-wake scoring

    n_periods : int
        Number of successive epochs scored as sleep to search for.

    n_previous : int
        Number of successive epochs scored as wake, preceding sleep epochs
        to search for.

    sleep_score: int, optional
        Sleep score.
        Default is 1.

    Return
    ----------
    mask : array_like
        Binary array.

    '''
    # Check input type
    if not isinstance(scoring, (np.ndarray)):
        raise TypeError(
            "Wrong input type for 'scoring': {}.\nExpect np.array.".format(
                type(scoring)
            ))

    # Search for the indices of the stretches of epochs scored as sleep
    indices = consecutive_values(
            scoring, target=sleep_score, min_length=n_periods
        )

    # Create mask
    mask = np.ones_like(scoring)

    # For each stretch of epochs scored as sleep:
    for index in indices:
        if (index[0] < n_previous) or (index[1] > len(scoring)):
            continue
        # If all preceding epochs are scored as wake:
        if np.all(scoring[index[0]-n_previous:index[0]] == 0):
            mask[index[0]:index[0]+n_periods] = 0

    return mask


def rescore_if_surrounded(scoring, n_periods, n_surround, sleep_score=1):
    r'''Returns a binary series with 0 at indices of epochs scored as sleep
    that are surrounded by epochs scored as wake.

    Parameters
    ----------
    scoring : np.ndarray
        Binary series with sleep-wake scoring

    n_periods : int
        Number of successive epochs scored as sleep to search for.

    n_previous : int
        Number of successive epochs scored as wake, surrounding sleep epochs
        to search for.

    sleep_score: int, optional
        Sleep score.
        Default is 1.

    Return
    ----------
    mask : array_like
        Binary array.

    '''
    # Check input type
    if not isinstance(scoring, (np.ndarray)):
        raise TypeError(
            "Wrong input type for 'scoring': {}.\nExpect np.array.".format(
                type(scoring)
            ))

    # Search for the indices of the stretches of epochs scored as wake
    indices = consecutive_values(
        scoring, target=np.abs(sleep_score-1), min_length=n_surround
    )

    # Create mask
    mask = np.ones_like(scoring)

    # For each stretch of epochs scored as wake:
    for idx in range(indices.shape[0]-1):
        # Diff between first element of the next wake period and the last
        # element of the current wake period
        sleep_duration = indices[idx+1][0] - indices[idx][1]

        # If the number of epochs scored as sleep is below threshold:
        if sleep_duration <= n_periods:
            mask[indices[idx][1]:indices[idx+1][0]] = 0

    return mask


def rescore(scoring, sleep_score=1):
    r'''Returns a binary series with 0 at indices of epochs scored as sleep
    that should rescored as wake according to Webster's rules.

    Parameters
    ----------
    scoring : np.ndarray
        Binary series with sleep-wake scoring

    sleep_score: int, optional
        Sleep score.
        Default is 1.

    Return
    ----------
    mask : array_like
        Binary array.

    '''
    # create an initial series of 1
    rescoring_masks = np.empty((5, len(scoring)))

    # Rule 1: Search for series of at least 4 minutes scored as wake...
    rescoring_masks[0] = rescore_if_preceded(
        scoring, n_periods=1, n_previous=4, sleep_score=1)

    # Rule 2: Search for series of at least 3 minutes scored as sleep...
    rescoring_masks[1] = rescore_if_preceded(
        scoring, n_periods=3, n_previous=10, sleep_score=1)

    # Rule 3: Search for series of at least 3 minutes scored as sleep...
    rescoring_masks[2] = rescore_if_preceded(
        scoring, n_periods=4, n_previous=15, sleep_score=1)

    # Rule 4: Search for series of at least 10 minutes scored as wake...
    rescoring_masks[3] = rescore_if_surrounded(
        scoring, n_periods=6, n_surround=10, sleep_score=1)

    # Rule 5: Search for series of at least 20 minutes scored as wake...
    rescoring_masks[4] = rescore_if_surrounded(
        scoring, n_periods=10, n_surround=20, sleep_score=1)

    # Multiply all the rescoring mask elements across the different rules.
    # If there is any zero (i.e. an epoch rescored as wake by any of the 5
    # rules), the result is zero.
    mask = np.prod(rescoring_masks, axis=0)

    return mask
