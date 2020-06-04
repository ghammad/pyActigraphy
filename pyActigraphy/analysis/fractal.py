# # (Multi-fractal)Detrended fluctuation analysis
from joblib import Parallel, delayed
from numba import jit  # , prange
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial


@jit(nopython=True)
def _integrate_detrended(X):

    trend = np.mean(X)

    detrended_series = X - trend

    integrated_series = np.cumsum(detrended_series)

    return integrated_series


class Fractal():
    """ Class for Fractality Analysis"""

    def __init__(self, n_array=None, q_array=None):

        self.__n_array = n_array
        self.__q_array = q_array

    @classmethod
    def integrate_detrended(cls, ts):
        int_s = _integrate_detrended(ts.values)
        return pd.Series(int_s, index=ts.index, name=ts.name)

    @classmethod
    def date_range(cls, ts, n_seg, unit='m', backward=False):
        r"""Split a date index range

        Split the datetime index of a time series into non-overlapping chunks

        """

        # Calculate duration of the time series
        duration = ts.index[-1]-ts.index[0]

        # Calculate the length of a non-overlapping segment
        segment_length = duration/n_seg

        # Start and end of the non-overlapping chunks
        if backward:
            dt_range = [(ts.index[-1]-(n+1)*segment_length,
                         ts.index[-1]-n*segment_length)
                        for n in range(n_seg)]
        else:
            dt_range = [(ts.index[0]+n*segment_length,
                         ts.index[0]+(n+1)*segment_length)
                        for n in range(n_seg)]

        # Convert the segment length to the required unit
        segment_length /= np.timedelta64('1', 'm')

        # Return
        return segment_length, dt_range

    @classmethod
    def date_indices(cls, ts, n_seg, unit='m', backward=False):
        r"""Split a date index range

        Split the datetime index of a time series into non-overlapping chunks

        """

        # Calculate duration of the time series
        duration = ts.index[-1]-ts.index[0]

        # Calculate the length of a non-overlapping segment
        segment_length = duration/n_seg

        # Start and end of the non-overlapping chunks
        if backward:
            dt_range = [(ts.index[-1]-(n+1)*segment_length,
                         ts.index[-1]-n*segment_length)
                        for n in range(n_seg)]
        else:
            dt_range = [(ts.index[0]+n*segment_length,
                         ts.index[0]+(n+1)*segment_length)
                        for n in range(n_seg)]

        # Convert index into positional index
        dt_indices = [(ts.index.get_loc(idx[0], 'nearest'),
                       ts.index.get_loc(idx[1], 'nearest'))
                      for idx in dt_range]

        # Convert the segment length to the required unit
        segment_length /= np.timedelta64('1', 'm')

        # Return
        return segment_length, dt_indices

    @classmethod
    def local_msq_residuals(cls, ts, freq, deg):
        r'''Mean squared residuals

        Mean squared residuals of the least squares polynomial fit.

        Parameters
        ----------
        ts: pandas.Series
            Activity time series.
        freq: pandas.Timedelta
            Sampling frequency of the time series.
            Needs to be set explicitly for unevenly spaced tim series.
        deg: int
            Degree(s) of the fitting polynomials.

        Returns
        -------
        residuals_msq: numpy.float
            Mean squared residuals.
        '''

        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = ((ts.index - ts.index[0])/freq).values

        # Fit the data
        _, fit_result = polynomial.polyfit(
            y=ts.values,
            x=x,
            deg=deg,
            full=True
        )

        # Return mean squared residuals
        return fit_result[0][0]/len(x) if fit_result[0].size != 0 else np.nan

    @classmethod
    def fluctuations(cls, ts, freq, n, deg, unit='m'):
        r'''Fluctuation function

        The fluctuations are defined as the series of mean squared residuals
        of the least squares fit in each non-overlapping segment.

        Parameters
        ----------
        ts: pandas.Series
            Activity time series.
        freq: pandas.Timedelta
            Sampling frequency of the time series.
            Needs to be set explicitly for unevenly spaced tim series.
        n: int
            Number of non-overlapping segment of equal length
        deg: int
            Degree(s) of the fitting polynomials.
        unit: str, optional
            Units of the segment length.
            Default is 'm' (minute).

        Returns
        -------
        segment_length,residuals_msq: numpy.float, numpy.array
            Length of the segment (in minutes) and array of mean
            squared residuals in each segment.
        '''

        # Detrend and integrate time series
        int_ts = cls.integrate_detrended(ts)

        # Compute the length, the start and end times of each of the n
        # non-overlapping segment
        segment_length, dt_range = cls.date_range(int_ts, n)

        # Compute the sum of the squared residuals for each segment
        iterable = (
            cls.local_msq_residuals(
                int_ts.loc[times[0]:times[1]],
                freq,
                deg
            ) for times in dt_range)

        residuals_msq = np.fromiter(
            iterable,
            dtype=np.float,
            count=len(dt_range)
        )

        return segment_length, residuals_msq

    @classmethod
    def q_th_order_mean_square(cls, data, q):

        return np.power(np.nanmean(np.power(data, q/2.)), 1/q)

    @classmethod
    def dfa(cls, raw, n_array, deg=1, log=False):
        r'''Detrended Fluctuation Analysis function

        Compute the q-th order mean squared fluctuations for different segment
        lengths.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be used.
        freq: pandas.Timedelta
            Sampling frequency of the time series.
            Needs to be set explicitly for unevenly spaced tim series.
        n_array: array of int
            Array of the numbers of non-overlapping segments of equal length
        deg: int
            Degree(s) of the fitting polynomials.
        log: bool, optional
            If set to True, returned values are log-transformed.
            Default is False.

        Returns
        -------
        lengths,q_th_order_msq_fluc: numpy.array, numpy.array
            Array of segment lengths (in minutes) and array of q-th order mean
            squared fluctuations.
        '''

        lengths = np.empty_like(n_array, dtype=np.float)
        q_th_order_msq_fluc = np.empty_like(n_array, dtype=np.float)
        for idx, n in enumerate(n_array):
            seg_length, fluct = cls.fluctuations(
                raw.data,
                freq=raw.frequency,
                n=n,
                deg=deg
            )
            lengths[idx] = seg_length
            q_th_order_msq_fluc[idx] = cls.q_th_order_mean_square(fluct, q=2)

        if log:
            return np.log(lengths), np.log(q_th_order_msq_fluc)
        else:
            return lengths, q_th_order_msq_fluc

    @classmethod
    def dfa_parallel(
        cls,
        raw,
        n_array,
        deg=1,
        log=False,
        n_jobs=2,
        prefer=None,
        verbose=0
    ):
        r'''Detrended Fluctuation Analysis function

        Compute, in parallel,  the q-th order mean squared fluctuations for
        different segment lengths.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be used.
        freq: pandas.Timedelta
            Sampling frequency of the time series.
            Needs to be set explicitly for unevenly spaced tim series.
        n_array: array of int
            Array of the numbers of non-overlapping segments of equal length
        deg: int
            Degree(s) of the fitting polynomials.
        log: bool, optional
            If set to True, returned values are log-transformed.
            Default is False.
        n_jobs: int, optional
            Number of CPU to use for parallel fitting.
            Default is 2.
        prefer: str, optional
            Soft hint to choose the default backendself.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose: int, optional
            Display a progress meter if set to a value > 0.
            Default is 0.

        Returns
        -------
        lengths,q_th_order_msq_fluc: numpy.array, numpy.array
            Array of segment lengths (in minutes) and array of q-th order mean
            squared fluctuations.
        '''

        lengths, flucts = zip(
            *Parallel(
                n_jobs=n_jobs,
                prefer=prefer,
                verbose=verbose
            )(delayed(cls.fluctuations)(
                raw.data,
                freq=raw.frequency,
                n=n,
                deg=deg
            ) for n in n_array)
        )

        q_th_order_msq_fluctuations = [
            cls.q_th_order_mean_square(fluct, q=2) for fluct in flucts
        ]

        if log:
            return np.log(lengths), np.log(q_th_order_msq_fluctuations)
        else:
            return np.array(lengths), q_th_order_msq_fluctuations

    @classmethod
    def hurst_exponent(cls, lengths, fluctuations, log=True):
        if log:
            c, stats = polynomial.polyfit(
                y=fluctuations,
                x=lengths,
                deg=1,
                full=True
            )
        else:
            c, stats = polynomial.polyfit(
                y=np.log(fluctuations),
                x=np.log(lengths),
                deg=1,
                full=True
            )

        return c[1]

    @classmethod
    def break_points(
        cls,
        n,
        f_n,
        start_idx_offset=3,
        stop_idx_offset=3,
        log=True
    ):

        times = []
        exponents = []
        # If the number of points for a single linear fit is less than 3
        if(start_idx_offset < 3 or stop_idx_offset < 3):
            print(
                ("Cannot perform a linear fit on series of less than"
                 " 3 points. Exiting now.")
            )
        # If the number of points to fit is less than 2*3
        elif((len(n)-stop_idx_offset+1-start_idx_offset) < 1):
            print(
                ("Total number of points to fit is less than 2*3."
                 "Exiting now.")
            )
        else:
            for t in np.arange(start_idx_offset, len(n)-stop_idx_offset+1):
                # Fit the series of points (F(n) vs n) up to point n_t
                alpha_1 = cls.hurst_exponent(n[:t], f_n[:t], log)
                # Fit the series of points (F(n) vs n) from point n_t to n_max
                alpha_2 = cls.hurst_exponent(n[t:], f_n[t:], log)
                # Append n,a_1/a_2
                times.append(n[t])
                exponents.append(alpha_1/alpha_2)

            if log:
                times = np.exp(times)

        return times, exponents

    @classmethod
    def mfdfa(cls, raw, n_array, q_array, deg=1, log=False):

        lengths = np.empty_like(n_array, dtype=np.float)
        q_th_order_msq_fluctuations = np.empty(
            (len(n_array), len(q_array)),
            dtype=np.float
        )
        for idx, n in enumerate(n_array):
            seg_length, fluct = cls.fluctuations(
                raw.data,
                freq=raw.frequency,
                n=n,
                deg=deg
            )
            lengths[idx] = seg_length
            q_th_order_msq_fluctuations[idx] = [
                cls.q_th_order_mean_square(fluct, q=q) for q in q_array
            ]

        if log:
            return np.log(lengths), np.log(q_th_order_msq_fluctuations)
        else:
            return lengths, q_th_order_msq_fluctuations
