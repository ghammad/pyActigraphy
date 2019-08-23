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

    return ranges[np.where(ranges[:, 1] - ranges[:, 0] > min_length)]
