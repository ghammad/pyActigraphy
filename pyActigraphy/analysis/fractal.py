# # (Multi-fractal)Detrended fluctuation analysis
from joblib import Parallel, delayed
from numba import njit  # , prange
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial
from scipy.stats import linregress


@njit
def _profile(X):

    trend = np.mean(X)

    prof = np.cumsum(X - trend)

    return prof


@njit  # (float64[:,:](float64[:],int64,boolean))
def _segmentation(x, n, backward=False):

    # Number of elements
    N = len(x)

    # Number of segments of length n (and remainder r)
    nseg, r = divmod(N, n)

    # Non-overlapping segments
    if backward:
        segments = x[r:].reshape(nseg, n)
    else:
        segments = x[:N-r].reshape(nseg, n)

    return segments


class Fractal():
    r''' Class for Fractality Analysis

    This class implements methods used to perform a (multifractal) detrended
    fluctuation analysis, (MF)DFA.

    The implementation follows the original description made in [1]_ and [2]_.

    The (MF)DFA consists in:

    1. removing the global mean and integrating the time series of a signal:

       .. math::

           X_{t} = \sum_i^N(x_i - \bar{x})

       where :math:`\bar{x}` denotes the mean value of the time series
       :math:`\{x_i\}_{i\in[1:N]}`;

    2. dividing the integrated signal into N non-overlapping windows of the
       same chosen size n;

    3. detrending the integrated signal in each window using polynomial
       functions to obtain residuals, that is:

       .. math::

           \widehat{X_t} = X_{t} - Y_{t}

       where :math:`Y_t` denotes the trend obtained by polynomial fit and
       :math:`\widehat{X_t}` the integrated time series after detrending;

    4. calculating the root mean square of residuals in all windows as
       detrended fluctuation amplitude :math:`F_q(n)`, that is:

       .. math::

          F_q(n) = \left[\frac{1}{N} \sum_{t=1}^N \widehat{X_t}^q\right]^{1/q}

    For :math:`q=2`, the DFA is retrieved.

    In the context of actigraphy, further informations can be found in:

    * Hu, K., Ivanov, P. C., Chen, Z., Hilton, M. F., Stanley, H. E., & Shea,
      S. A. (2004). Non-random fluctuations and multi-scale dynamics regulation
      of human activity. Physica A: Statistical Mechanics and Its Applications,
      337(1–2), 307–318. https://doi.org/10.1016/j.physa.2004.01.042


    References
    ----------

    .. [1] Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley,
           H. E., & Goldberger, A. L. (1994). Mosaic organization of DNA
           nucleotides. Physical Review E, 49(2), 1685–1689.
           https://doi.org/10.1103/PhysRevE.49.1685
    .. [2] Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin,
           S., Bunde, A., & Stanley, H. E. (2002). Multifractal detrended
           fluctuation analysis of nonstationary time series. Physica A:
           Statistical Mechanics and Its Applications, 316(1–4), 87–114.
           https://doi.org/10.1016/S0378-4371(02)01383-3

    '''

    def __init__(self, n_array=None, q_array=None):

        self.__n_array = n_array
        self.__q_array = q_array

    @classmethod
    def profile(cls, X):
        r'''Profile function

        Detrend and integrate the signal.

        Parameters
        ----------
        x: numpy.array
            Input array.
        n: int
            Window size.

        Returns
        -------
        segments: numpy.array
            Non-overlappping windows of size n.
        '''

        return _profile(X)

    @classmethod
    def segmentation(cls, x, n, backward=False):
        r'''Segmentation function

        Segment the signal into non-overlapping windows of equal size.

        Parameters
        ----------
        x: numpy.array
            Input array.
        n: int
            Window size.
        backward: bool
            If set to True, start segmentation for the end of the signal.
            Default is False.

        Returns
        -------
        segments: numpy.array
            Non-overlappping windows of size n.
        '''
        return _segmentation(x, n, backward)

    @classmethod
    def local_msq_residuals(cls, segment, deg):
        r'''Mean squared residuals

        Mean squared residuals of the least squares polynomial fit.

        Parameters
        ----------
        segment: numpy.array
            Input array.
        deg: int
            Degree(s) of the fitting polynomials.

        Returns
        -------
        residuals_msq: numpy.float
            Mean squared residuals.
        '''

        # Segment length
        n = len(segment)

        # X-axis
        x = np.linspace(1, n, n)

        # Fit the data
        _, fit_result = polynomial.polyfit(y=segment, x=x, deg=deg, full=True)

        # Return mean squared residuals
        return fit_result[0][0]/n if fit_result[0].size != 0 else np.nan

    @classmethod
    def fluctuations(cls, X, n, deg):
        r'''Fluctuation function

        The fluctuations are defined as the mean squared residuals
        of the least squares fit in each non-overlapping window.

        Parameters
        ----------
        X: numpy.array
            Array of activity counts.
        n: int
            Window size.
        deg: int
            Degree(s) of the fitting polynomials.

        Returns
        -------
        F: numpy.array
            Array of mean squared residuals in each segment.
        '''

        # Detrend and integrate time series
        Y = cls.profile(X)

        # Define non-overlapping segments
        segments_fwd = cls.segmentation(Y, n, backward=False)
        segments_bwd = cls.segmentation(Y, n, backward=True)

        # Assert equal numbers of segments
        assert(segments_fwd.shape == segments_bwd.shape)

        F = np.empty(len(segments_fwd)+len(segments_bwd))
        # Compute the sum of the squared residuals for each segment
        for i in range(len(segments_fwd)):
            F[i*2] = cls.local_msq_residuals(segments_fwd[i], deg)
            F[i*2+1] = cls.local_msq_residuals(segments_bwd[i], deg)

        return F

    @classmethod
    def q_th_order_mean_square(cls, F, q):
        r'''Qth-order mean square function

        Compute the q-th order mean squares.

        Parameters
        ----------
        F: numpy.array
            Array of fluctuations.
        q: scalar
            Order.

        Returns
        -------
        qth_msq: numpy.float
            Q-th order mean square.
        '''

        if q == 0:
            qth_msq = np.exp(0.5*np.nanmean(np.log(F)))
        else:
            qth_msq = np.power(np.nanmean(np.power(F, q/2)), 1/q)

        return qth_msq

    @classmethod
    def dfa(cls, ts, n_array, deg=1, log=False):
        r'''Detrended Fluctuation Analysis function

        Compute the q-th order mean squared fluctuations for different segment
        lengths.

        Parameters
        ----------
        ts: pandas.Series
            Input signal.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        deg: int, optional
            Degree(s) of the fitting polynomials.
            Default is 1.
        log: bool, optional
            If set to True, returned values are log-transformed.
            Default is False.

        Returns
        -------
        q_th_order_msq_fluc: numpy.array
            Array of q-th order mean squared fluctuations.
        '''

        # Check sampling frequency
        freq = ts.index.freq
        if freq is None:
            raise ValueError(
                "Cannot check the sampling frequency of the input data.\n"
                "Might be the case for unevenly spaced data or masked data.\n"
                "Please, consider using pandas.Series.resample."
            )
        else:
            factor = pd.Timedelta('1min')/freq

        iterable = (
            cls.q_th_order_mean_square(
                cls.fluctuations(ts.values, n=int(n), deg=deg),
                q=2
            ) for n in factor*n_array
        )

        q_th_order_msq_fluc = np.fromiter(
            iterable,
            dtype=np.float,
            count=len(n_array)
        )

        if log:
            q_th_order_msq_fluc = np.log(q_th_order_msq_fluc)

        return q_th_order_msq_fluc

    @classmethod
    def dfa_parallel(
        cls, ts, n_array, deg=1, log=False, n_jobs=2, prefer=None, verbose=0
    ):
        r'''Detrended Fluctuation Analysis function

        Compute, in parallel,  the q-th order mean squared fluctuations for
        different segment lengths.

        Parameters
        ----------
        ts : pandas.Series
            Input signal.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        deg: int, optional
            Degree(s) of the fitting polynomials.
            Default is 1.
        log: bool, optional
            If set to True, returned values are log-transformed.
            Default is False.
        n_jobs: int, optional
            Number of CPU to use for parallel fitting.
            Default is 2.
        prefer: str, optional
            Soft hint to choose the default backend.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose: int, optional
            Display a progress meter if set to a value > 0.
            Default is 0.

        Returns
        -------
        q_th_order_msq_fluc: numpy.array
            Array of q-th order mean squared fluctuations.
        '''

        # Check sampling frequency
        freq = ts.index.freq
        if freq is None:
            raise ValueError(
                "Cannot check the sampling frequency of the input data.\n"
                "Might be the case for unevenly spaced data or masked data.\n"
                "Please, consider using pandas.Series.resample."
            )
        else:
            factor = pd.Timedelta('1min')/freq

        flucts = Parallel(
            n_jobs=n_jobs,
            prefer=prefer,
            verbose=verbose
        )(delayed(cls.fluctuations)(
            ts.values,
            n=int(n),
            deg=deg
        ) for n in factor*n_array)

        q_th_order_msq_fluc = np.fromiter(
            (cls.q_th_order_mean_square(fluct, q=2) for fluct in flucts),
            dtype=np.float,
            count=len(flucts)
        )

        if log:
            q_th_order_msq_fluc = np.log(q_th_order_msq_fluc)

        return q_th_order_msq_fluc

    @classmethod
    def generalized_hurst_exponent(
        cls, F_n, n_array, x_center=False, log=False
    ):
        r'''Generalized Hurst exponent

        Compute the generalized Hurst exponent, :math:`h(q)`, by fitting .

        Parameters
        ----------
        F_n : array
            Array of fluctuations.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        x_center: bool, optional
            If set to True, time scales are mean-centered.
            Default is false.
        log: bool, optional
            If set to True, assume that the input values have already been
            log-transformed.
            Default is False.

        Returns
        -------
        h, _h_err: float,float
            Estimate of the generalized Hurst exponent and its standard error.
        '''

        y = np.log(F_n) if not log else F_n
        x = np.log(n_array) if not log else n_array

        if x_center:
            x = x - np.mean(x)

        r = linregress(y=y, x=x)

        return r.slope, r.stderr

    @classmethod
    def crossover_search(cls, F_n, n_array, n_min=3, log=False):
        r'''Search for crossovers

        A crossover is defined as a change in scaling properties of the
        fluctuations with respect time scales. A search is performed by
        calculating the series of ratios between the generalized Hurst exponent
        :math:`h(q)` obtained at time scales :math:`n<n_x` and time scales
        :math:`n>n_x`, for various values of :math:`n_x`.

        Parameters
        ----------
        F_n : array
            Array of fluctuations.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        n_min: int, optional
            Minimal number of time scales required to estimate the generalized
            Hurst exponent.
            Default is 3.
        log: bool, optional
            If set to True, assume that the input values have already been
            log-transformed.
            Default is False.

        Returns
        -------
        h_ratios, h_ratios_err, n_x: arrays of floats
            Ratio of h(q), and associated uncertainties, obtained for various
            time scales n_x.

        Notes
        -----

        .. warning::

            The calculation of the uncertainty on the ratio of scaling
            exponents assumes uncorrelated variables:
            :math:`\sigma_{A/B}^2=(A/B)^2(\sigma_{A}^2/A^2+\sigma_{B}^2/B^2)`.
            Most likely, the scaling exponents calculated for time scales
            :math:`n<n_x` is not uncorrelated to the scaling exponents
            calculated for time scales :math:`n>n_x`. Therefore, the resulting
            uncertainty is either overestimated in case of positively
            correlated variables or underestimated otherwise. However, the
            magnitude of the calculated uncertainty provides a rough estimate.
        '''

        n_x = np.empty(len(n_array)-2*n_min+1)
        h_ratios = np.empty(len(n_array)-2*n_min+1)
        h_ratios_err = np.empty(len(n_array)-2*n_min+1)
        # If the number of points for a single linear fit is less than 3
        if(n_min < 3):
            print(
                ("Cannot perform a linear fit on series of less than"
                 " 3 points. Exiting now.")
            )
        # If the number of points to fit is less than 2*3
        elif((len(n_array)-2*n_min+1) < 1):
            print(
                ("Total number of points to fit is less than 2*3."
                 "Exiting now.")
            )
        else:
            for t in np.arange(n_min, len(n_array)-n_min+1):
                # Fit the series of points (F(n) vs n) up to point n_x
                alpha_1, alpha_1_err = cls.generalized_hurst_exponent(
                    F_n[:t], n_array[:t], log
                )
                # Fit the series of points (F(n) vs n) from point n_x to n_max
                alpha_2, alpha_2_err = cls.generalized_hurst_exponent(
                    F_n[t:], n_array[t:], log
                )
                # Alpha ratio and relative uncertainties
                ratio = alpha_1/alpha_2
                alpha_1_rel_err = alpha_1_err/alpha_1
                alpha_2_rel_err = alpha_2_err/alpha_2

                n_x[t-n_min] = n_array[t]
                h_ratios[t-n_min] = ratio
                h_ratios_err[t-n_min] = ratio*np.sqrt(
                    alpha_1_rel_err*alpha_1_rel_err +
                    alpha_2_rel_err*alpha_2_rel_err
                )

            if log:
                n_x = np.exp(n_x)

        return h_ratios, h_ratios_err, n_x

    @classmethod
    def mfdfa(cls, ts, n_array, q_array, deg=1, log=False):
        r'''Multifractal Detrended Fluctuation Analysis function

        Compute the q-th order mean squared fluctuations for different segment
        lengths and different index q values.

        Parameters
        ----------
        ts: pandas.Series
            Input signal.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        q_array: array of float
            Orders of the mean squares.
        deg: int, optional
            Degree(s) of the fitting polynomials.
            Default is 1.
        log: bool, optional
            If set to True, returned values are log-transformed.
            Default is False.

        Returns
        -------
        q_th_order_msq_fluctuations: numpy.array
            (n,q) array of q-th order mean squared fluctuations.
        '''

        # Check sampling frequency
        freq = ts.index.freq
        if freq is None:
            raise ValueError(
                "Cannot check the sampling frequency of the input data.\n"
                "Might be the case for unevenly spaced data or masked data.\n"
                "Please, consider using pandas.Series.resample."
            )
        else:
            factor = pd.Timedelta('1min')/freq

        q_th_order_msq_fluctuations = np.empty(
            (len(n_array), len(q_array)),
            dtype=np.float
        )
        for idx, n in enumerate(factor*n_array):

            fluct = cls.fluctuations(ts.values, n=int(n), deg=deg)
            q_th_order_msq_fluctuations[idx] = [
                cls.q_th_order_mean_square(fluct, q=q) for q in q_array
            ]

        if log:
            q_th_order_msq_fluctuations = np.log(q_th_order_msq_fluctuations)

        return q_th_order_msq_fluctuations

    @classmethod
    def mfdfa_parallel(
        cls,
        ts,
        n_array,
        q_array,
        deg=1,
        log=False,
        n_jobs=2,
        prefer=None,
        verbose=0
    ):
        r'''Multifractal Detrended Fluctuation Analysis function

        Compute, in parallel, the q-th order mean squared fluctuations for
        different segment lengths and different index q values.

        Parameters
        ----------
        ts : pandas.Series
            Input signal.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        q_array: array of float
            Orders of the mean squares.
        deg: int, optional
            Degree(s) of the fitting polynomials.
            Default is 1.
        log: bool, optional
            If set to True, returned values are log-transformed.
            Default is False.
        n_jobs: int, optional
            Number of CPU to use for parallel fitting.
            Default is 2.
        prefer: str, optional
            Soft hint to choose the default backend.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose: int, optional
            Display a progress meter if set to a value > 0.
            Default is 0.

        Returns
        -------
        q_th_order_msq_fluctuations: numpy.array
            (n,q) array of q-th order mean squared fluctuations.
        '''

        # Check sampling frequency
        freq = ts.index.freq
        if freq is None:
            raise ValueError(
                "Cannot check the sampling frequency of the input data.\n"
                "Might be the case for unevenly spaced data or masked data.\n"
                "Please, consider using pandas.Series.resample."
            )
        else:
            factor = pd.Timedelta('1min')/freq

        flucts = Parallel(
            n_jobs=n_jobs,
            prefer=prefer,
            verbose=verbose
        )(delayed(cls.fluctuations)(
            ts.values,
            n=int(n),
            deg=deg
        ) for n in factor*n_array)

        q_th_order_msq_fluctuations = np.array([
            cls.q_th_order_mean_square(fluct, q=q)
            for fluct in flucts for q in q_array
        ]).reshape(len(n_array), len(q_array))

        if log:
            q_th_order_msq_fluctuations = np.log(q_th_order_msq_fluctuations)

        return q_th_order_msq_fluctuations
