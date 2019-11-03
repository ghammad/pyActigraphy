import numpy as np
import pandas as pd
import warnings

from lmfit import fit_report, minimize, Parameters
from scipy.signal import find_peaks
from scipy.stats import pearsonr, poisson


def _zero_crossing_points(x):
    r'''Zero crossing points'''
    x_sign = np.sign(x)
    zero_crossing = ((np.roll(x_sign, 1) - x_sign) != 0).astype(int)
    # the first point is set to 1 if the last and the first points have
    # different signs as the rolling window is cyclic
    zero_crossing[0] = 0
    return zero_crossing


def _extrema_points(df_dx, d2f_dx2):
    r'''Extrema (either minimum or maximum) points'''
    # Extrema are located where the first derivative, df_dx = 0
    extrema = _zero_crossing_points(df_dx)
    # Second derivative is used to differentiate maxima (d2f_dx2<0)
    # from minima (d2f_dx2>0)
    extrema = extrema*np.sign(d2f_dx2)
    return extrema


def _inflexion_points(df_dx, d2f_dx2):
    r'''Inflexion points'''
    # Inflexion points are located where the second derivative, d2f_dx2 = 0
    # The first derivative is then used to distinguish between an
    # 'increasing' or 'decreasing' inflexion.
    return _extrema_points(d2f_dx2, df_dx)


def _lids_func(x):
    r'''LIDS transformation function'''

    return 100/(x+1)


def _lids_inverse_func(x):
    r'''LIDS inverse transformation function'''

    return 100/x - 1


def _lids_pmf(x_lids, mu_lids):
    r'''Probability mass function of the LIDS'''

    # Expected number of counts
    mu = _lids_inverse_func(mu_lids)

    k1 = _lids_inverse_func(x_lids)
    k2 = _lids_inverse_func(x_lids+1)
    return poisson.cdf(k=k1, mu=mu) - poisson.cdf(k=k2, mu=mu)


def _cosine(x, params):
    r'''1-harmonic cosine function'''

    A = params['amp']
    phi = params['phase']
    T = params['period']
    offset = params['offset']

    return A*np.cos(2*np.pi/T*x+phi) + offset


def _lfm(x, params):
    r'''Linear frequency modulated cosine function'''

    A = params['amp']
    k = params['k']
    phi = params['phase']
    T = params['period']
    offset = params['offset']
    slope = params['slope']

    return A*np.cos(2*np.pi*(x/T+k*x*x)+phi) + offset + slope*x


def _lfam(x, params):
    r'''Linear frequency and amplitude modulated cosine function'''

    A = params['amp']
    b = params['mod']
    k = params['k']
    phi = params['phase']
    T = params['period']
    offset = params['offset']
    slope = params['slope']

    return (A + b*x)*np.cos(2*np.pi*(x/T+k*x*x)+phi) + offset + slope*x


def _lfamd(x, params):
    r'''Linear frequency and amplitude modulated cosine function, associated
    with an exponential decay'''

    A = params['amp']
    b = params['mod']
    k = params['k']
    phi = params['phase']
    T = params['period']
    offset = params['offset']
    amp_exp = params['amp_exp']
    tau = params['tau']

    return (A + b*x)*np.cos(
            2*np.pi*(x/T+k*x*x)+phi
        ) + offset + amp_exp*np.exp(-x/tau)


def _residual(params, x, data, fit_func):
    r'''Residual function to minimize'''

    model = fit_func(x, params)
    return (data-model)


def _residual_rel(params, x, data, sigma, fit_func):
    r'''Residual function to minimize'''

    model = fit_func(x, params)
    return (data-model)/sigma


def _lids_likelihood(params, x, data, fit_func):
    r'''LIDS likelihood function

    Defined as the product of the probability mass functions, evaluated at each
    data point, using the current fit value as the expected value.

    NB: when the difference between the expected value and the observed one is
    large, the probability drops to zero, due to finite floating precision.
    A temporary solution consists in replacing all values below eps with eps.
    '''

    # Expected LIDS counts (i.e fitted values, 'mu_i')
    expected_val = fit_func(x, params)

    # Create empty array
    lids_ll = np.empty_like(expected_val)

    # Iterate over all the values of the currently fitted function
    it = np.nditer(expected_val, flags=['c_index'])
    while not it.finished:
        # lids_ll[it.index] = np.sqrt(
        #     -2*np.log(_lids_pmf(data[it.index], it[0]))
        # )
        lids_ll[it.index] = _lids_pmf(data[it.index], it[0])
        it.iternext()

    # Replace zeros with eps
    eps = np.finfo(data.dtype).eps
    np.place(lids_ll, lids_ll < eps, [eps])

    return lids_ll


def _nlog(x):
    return -2*np.log(np.prod(x))


class LIDS():
    """
    Class for Locomotor inactivity during sleep (LIDS) Analysis

    Winnebeck, E. C., Fischer, D., Leise, T., & Roenneberg, T. (2018).
    Dynamics and Ultradian Structure of Human Sleep in Real Life.
    Current Biology, 28(1), 49–59.e5. http://doi.org/10.1016/j.cub.2017.11.063

    """

    lids_func_list = ['lids']
    fit_func_list = ['cosine', 'chirp', 'modchirp', 'modchirp_exp']

    def __init__(
        self,
        lids_func='lids',
        fit_func='cosine',
        fit_obj_func='residuals',
        fit_params=None
    ):

        # LIDS functions
        lids_funcs = {'lids': _lids_func}
        if lids_func not in lids_funcs.keys():
            raise ValueError(
                '`LIDS function` must be "%s". You passed: "%s"' %
                ('" or "'.join(list(lids_funcs.keys())), lids_func)
            )

        # Fit functions
        fit_funcs = {
            'cosine': _cosine,
            'chirp': _lfm,
            'modchirp': _lfam,
            'modchirp_exp': _lfamd
        }
        if fit_func not in fit_funcs.keys():
            raise ValueError(
                '`Fit function` must be "%s". You passed: "%s"' %
                ('" or "'.join(list(fit_funcs.keys())), fit_func)
            )

        # Fit objective functions (i.e. functions to be minimized)
        fit_obj_funcs = {
            'residuals': _residual,
            'nll': _lids_likelihood
        }
        # and associated functions to convert the residuals to a scalar value
        fit_reduc_funcs = {
            'residuals': None,
            'nll': _nlog
        }

        if fit_obj_func not in fit_obj_funcs.keys():
            raise ValueError(
                '`Fit objective function` must be "%s". You passed: "%s"' %
                ('" or "'.join(list(fit_obj_funcs.keys())), fit_obj_func)
            )

        self.__lids_func = lids_funcs[lids_func]  # LIDS transformation fct
        self.__fit_func = fit_funcs[fit_func]  # Fit function to LIDS
        self.__fit_obj_func = fit_obj_funcs[fit_obj_func]  # Fit obj function
        self.__fit_reduc_func = fit_reduc_funcs[fit_obj_func]

        if fit_params is None:
            fit_params = Parameters()
            # Default parameters for the cosine fit function
            fit_params.add('amp', value=50, min=0, max=100)
            fit_params.add('phase', value=np.pi/2, min=0, max=2*np.pi)
            fit_params.add('period', value=9, min=0)  # Dummy value
            # Introduce inequality amp+offset < 100
            fit_params.add('delta', value=60, min=0, max=100, vary=True)
            fit_params.add('offset', expr='delta-amp')
            # Additional parameters for the chirp fit function
            if fit_func == 'chirp':
                fit_params.add('k', value=-.0001, min=-1, max=1)
                fit_params.add('slope', value=-0.5)
            # Additional parameters for the modchirp fit function
            if fit_func == 'modchirp':
                fit_params.add('k', value=-.0001, min=-1, max=1)
                fit_params.add('slope', value=-0.5)
                fit_params.add('mod', value=0.0001, min=-10, max=10)
            # Additional parameters for the modchirp_exp fit function
            if fit_func == 'modchirp_exp':
                fit_params.add('k', value=-.0001, min=-1, max=1)
                fit_params.add('mod', value=0.0001, min=-10, max=10)
                fit_params.add('tau', value=0.5)
                fit_params.add('amp_exp', value=10, min=0, max=100)

        self.__fit_initial_params = fit_params
        # self.__fit_params = None
        self.__fit_results = None
        self.__fit_mri_profile = None
        # self.__fit_period = None

    @property
    def lids_func(self):
        r'''LIDS transformation function'''
        return self.__lids_func

    @property
    def lids_fit_func(self):
        r'''Fit function to LIDS oscillations'''
        return self.__fit_func

    @lids_fit_func.setter
    def lids_fit_func(self, func):
        self.__fit_func = func

    @property
    def lids_fit_initial_params(self):
        r'''Initial parameters of the fit function to LIDS oscillations'''
        return self.__fit_initial_params

    @property
    def lids_fit_results(self):
        r'''Results of the LIDS fit'''
        return self.__fit_results

    @classmethod
    def concat(cls, ts_list, time_delta='15min'):
        r'''Concatenate time series

        Consecutive time series that are apart by less than a user-defined
        thresold are concatenanted. The gaps between the time series are filled
        with NaN so that the resulting time series remain evenly spaced.

        Parameters
        ----------
        ts_list: list of pandas.Series
            Series to concatenate. Must have the same sampling frequency.
        time_delta: str, optional
            Maximal time delta between LIDS series to concatenate.
            Default is '15min'.

        Returns
        -------
        concat_indices: list of lists of the indices of the series to be merged
        concat_ts: list of pandas.Series

        '''
        if len(ts_list) < 2:
            warnings.warn(
                'The number of time series to concatenate must be greater' +
                ' than 2.\n' +
                'Actual number of time series: {}.\n'.format(len(ts_list)) +
                'Returning empty lists.',
                UserWarning
            )
            return [], []

        # Store the sampling frequency of all the input time series
        freqs = [ts.index.freq for ts in ts_list]
        # Check if one is None
        if None in freqs:
            raise ValueError(
                "One of the input time series has no index frequency. "
                "This could indicate unevenly sampled data.\n"
                "The current implementation of the LIDS analysis does not "
                "support such data. A possible workaround would consist in "
                "resampling the data with the assumed acquisition frequency."
            )
        # Check if all the time series to concatenate have the same sampling
        # frequency
        if len(set(freqs)) != 1:
            raise ValueError(
                "One of the input time series has no index frequency. "
                "This could indicate unevenly sampled data.\n"
                "The current implementation of the LIDS analysis does not "
                "support such data. A possible workaround would consist in "
                "resampling the data with the assumed acquisition frequency."
            )

        # Store current sampling frequency
        freq = set(freqs).pop()

        td = pd.Timedelta(time_delta)

        # Check if two consecutive series are separated by 'time_delta'
        intervals = [
            (ts_list[idx+1].index[0]-ts_list[idx].index[-1]) < td
            for idx in range(len(ts_list)-1)
        ]

        concat_indices = []
        to_concat = []

        for idx, interval in enumerate(intervals):
            to_concat.append(idx)
            if not interval:
                concat_indices.append(to_concat)
                to_concat = []
        if len(to_concat) > 0:
            concat_indices.append(to_concat)
        if intervals[-1]:
            concat_indices[-1].append(len(intervals))
        else:
            concat_indices.append([len(intervals)])

        concat_ts = []
        for idx in concat_indices:
            if len(idx) == 1:
                concat_ts.append(ts_list[idx[0]].asfreq(freq))
            else:
                concat_ts.append(
                    pd.concat([ts_list[i] for i in idx]).asfreq(freq)
                )

        return concat_indices, concat_ts

    @classmethod
    def filter(cls, ts_list, duration_min='3H', duration_max='12H'):
        r'''Filter data according to their time duration.

        Parameters
        ----------
        ts_list: list of pandas.Series
            Data to filter.
        duration_min: str, optional
            Minimal time duration for a time series to be kept.
            If set to None, this criterion is not used for filtering.
            Default is '30min'.
        duration_max: str, optional
            Maximal time duration for a time series to be kept.
            If set to None, this criterion is not used for filtering.
            Default is '12h'.

        Returns
        -------
        filtered: list of pandas.Series
            List of filtered time series.
        '''

        def duration(s):
            return s.index[-1]-s.index[0] if len(s) > 0 else pd.Timedelta(0)

        td_min = pd.Timedelta(duration_min)
        td_max = pd.Timedelta(duration_max)

        from itertools import filterfalse
        filtered = []
        filtered[:] = filterfalse(
            lambda x: duration(x) < td_min or duration(x) > td_max,
            ts_list
        )
        return filtered

    @classmethod
    def smooth(cls, ts, method, resolution):
        r'''Smooth data using a rolling window

        Parameters
        ----------
        ts: pandas.Series
            Time series to smooth.
        method: str, optional
            Method to smooth the data.
            Available options are:
                'mva': moving average
                'gaussian': gaussian kernel
                'none': no smoothing
        resolution: float
            If method='mva': Size of the rolling window.
            If method='gaussian': Standard deviation of the gaussian kernel.
            The window size is then set to :math:`[-3*\sigma,3*\sigma]`.

        Returns
        -------
        smooth_lids: pandas.Series
        '''

        # Smooth functions
        smooth_funcs = ['mva', 'gaussian', 'none']
        if method not in smooth_funcs:
            raise ValueError(
                '`Smooth function` must be "%s". You passed: "%s"' %
                ('" or "'.join(list(smooth_funcs)), method)
            )

        if method == 'mva':
            win_size = int(resolution)
            return ts.rolling(win_size, center=True, min_periods=1).mean()
        elif method == 'gaussian':
            win_size = int(3*2*resolution)
            return ts.rolling(
                win_size,
                win_type=method,
                center=True,
                min_periods=1).mean(std=resolution)
            # smooth_ts = spm_smooth(ts.values, fwhm=win_size)
            # return pd.Series(data=smooth_ts, index=ts.index)
        elif method == 'none':
            return ts

    def lids_transform(
        self, ts, resampling_freq=None, method='mva', resolution='30min'
    ):
        r'''Apply LIDS transformation to activity data

        This transformation comprises:
        * resampling via summation (optional)
        * non-linear LIDS transformation
        * smoothing with a centered moving average

        Parameters
        ----------
        ts: pandas.Series
            Data identified as locomotor activity during sleep.
        resampling_freq: str, optional
            Frequency of the resampling applied prior to LIDS transformation.
            Default is None.
        method: str, optional
            Method to smooth the data.
            Available options are:
                'mva': moving average
                'gaussian': gaussian kernel
                'none': no smoothing
            Default is 'mva'.
        resolution: str, optional
            If method='mva': Size of the rolling window.
            If method='gaussian': Standard deviation of the gaussian kernel.
            Default is '30min'.

        Returns
        -------
        smooth_lids: pandas.Series
        '''

        # Resample data to the required frequency
        if resampling_freq is not None:
            rs = ts.resample(resampling_freq).sum()
        else:
            rs = ts

        # Check if index frequency is not None
        if rs.index.freq is None:
            raise ValueError(
                "The input data have no index frequency. "
                "This could indicate unevenly sampled data.\n"
                "The current implementation of the LIDS analysis does not "
                "support such data. A possible workaround would consist in "
                "resampling the data with the assumed acquisition frequency."
            )
        # Apply LIDS transformation x: 100/(x+1)
        lids = rs.apply(self.lids_func)

        # Series with a DateTimeIndex don't accept 'time-aware' centered window
        # Convert win_size (TimeDelta) into a number of time bins
        resolution_norm = pd.Timedelta(resolution)/lids.index.freq

        # Smooth LIDS-transformed data
        smooth_lids = self.smooth(
            lids,
            method=method,
            resolution=resolution_norm
        )

        return smooth_lids

    def lids_fit(
        self,
        lids,
        method='leastsq',
        scan_period=True,
        bounds=('30min', '180min'),
        step='5min',
        mri_profile=False,
        nan_policy='raise',
        verbose=False
    ):
        r'''Fit oscillations of the LIDS data

        The fit is performed with a fixed period ranging from 30 min to 180 min
        with a step of 5 min by default. The best-fit criterion is the maximal
        Munich Rhythmicity Index (MRI).

        Parameters
        ----------
        lids: pandas.Series
            Output data from LIDS transformation.
        method: str, optional
            Name of the fitting method to use [1]_.
            Default is 'leastsq'.
        scan_period: bool, optional
            If set to True, the period of the LIDS fit function is fixed and
            varied between the specified bounds.
            The selected period corresponds to the highest MRI value.
            Otherwise, the period is a free parameter of the fit.
            Default is True.
        bounds: 2-tuple of str, optional
            Lower and upper bounds for the periods to be tested.
            If scan_period is set to False, the bounds are ignored.
            Default is ('30min','180min').
        step: str, optional
            Time delta between the periods to be tested.
            Default is '5min'
        mri_profile: bool, optional
            If set to True, the function returns a list with the MRI calculated
            for each test period.
            Default is False.
        nan_policy: str, optional
            Specifies action if the objective function returns NaN values.
            One of:
                'raise': a ValueError is raised
                'propagate': the values returned from userfcn are un-altered
                'omit': non-finite values are filtered
            Default is 'raise'.
        verbose: bool, optional
            If set to True, display fit informations

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html
        '''

        # Store actual sampling frequency
        freq = pd.Timedelta(lids.index.freq)
        # Check if index frequency is not None
        if freq is None:
            raise ValueError(
                "The input data have no index frequency. "
                "This could indicate unevenly sampled data.\n"
                "The current implementation of the LIDS analysis does not "
                "support such data. A possible workaround would consist in "
                "resampling the data with the assumed acquisition frequency."
            )

        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = ((lids.index - lids.index[0])/freq).values
        mri = []

        if scan_period:

            # Define bounds for the period
            period_start = pd.Timedelta(bounds[0])/freq
            period_end = pd.Timedelta(bounds[1])/freq
            period_range = period_end-period_start
            period_step = pd.Timedelta(step)/freq

            test_periods = np.linspace(
                period_start,
                period_end,
                num=int(period_range/period_step)+1
            )

            # Fit data for each test period
            fit_results = []
            initial_period = self.__fit_initial_params['period'].value
            for test_period in test_periods:
                # Fix test period
                self.__fit_initial_params['period'].value = test_period
                self.__fit_initial_params['period'].vary = False

                # Minimize residuals
                fit_results.append(
                    minimize(
                        self.__fit_obj_func,
                        self.__fit_initial_params,
                        method=method,
                        args=(x, lids.values, self.lids_fit_func),
                        nan_policy=nan_policy,
                        reduce_fcn=self.__fit_reduc_func
                    )
                )
                # Print fit parameters if verbose
                if verbose:
                    print(fit_report(fit_results[-1]))
                # Calculate the MR index
                mri.append(self.lids_mri(lids, fit_results[-1].params))
                if verbose:
                    pearson_r = self.lids_pearson_r(
                        lids, fit_results[-1].params)[0]
                    print('Pearson r: {}'.format(pearson_r))
                    print('MRI: {}'.format(mri[-1]))

            # Fit the highest MRI peak
            peaks, peak_properties = find_peaks(mri, height=0)
            if len(peak_properties['peak_heights']) > 0:
                # Store index of the highest MRI peak
                peak_index = peaks[np.argmax(peak_properties['peak_heights'])]
                # Store fit parameters corresponding the highest MRI peak
                self.__fit_results = fit_results[peak_index]
                if verbose:
                    print('Highest MRI: {}'.format(mri[peak_index]))
                # Add lids index frequency to fit result parameters
                self.__fit_results.params.add(
                    'freq', value=freq.total_seconds()
                )
                # Add last value of the ordinal index of the input lids
                # Needed for calculation of the phase at sleep offset
                self.__fit_results.params.add(
                    'x_max', value=x[-1]
                )

            else:
                # Store fit parameters
                self.__fit_results = None
                if verbose:
                    print(
                        'No highest MRI could be found. '
                        'No peak was found in the MRI profile'
                    )

            # Set back original value
            self.__fit_initial_params['period'].value = initial_period
            self.__fit_initial_params['period'].vary = True

        else:

            # Minimize residuals
            self.__fit_results = minimize(
                self.__fit_obj_func,
                self.__fit_initial_params,
                method=method,
                args=(x,  lids.values, self.lids_fit_func),
                nan_policy=nan_policy,
                reduce_fcn=self.__fit_reduc_func
            )
            # Add lids index frequency to fit result parameters
            self.__fit_results.params.add(
                'freq', value=freq.total_seconds()
            )
            # Add last value of the ordinal index of the input lids
            # Needed for calculation of the phase at sleep offset
            self.__fit_results.params.add(
                'x_max', value=x[-1]
            )
            # Print fit parameters if verbose
            if verbose:
                print(fit_report(self.lids_fit_results))
            # Calculate the MR index
            mri.append(
                self.lids_mri(lids, self.lids_fit_results.params)
            )
            if verbose:
                pearson_r = self.lids_pearson_r(
                    lids, self.lids_fit_results.params)[0]
                print('Pearson r: {}'.format(pearson_r))
                print('MRI: {}'.format(mri[-1]))

        if mri_profile:
            if scan_period:
                return pd.Series(index=test_periods*freq, data=mri)
            else:
                return pd.Series(
                    index=[
                        self.lids_fit_results.params[
                            'period'
                        ].value*freq
                    ],
                    data=mri
                )

    def lids_pearson_r(self, lids, params=None):
        r'''Pearson correlation factor

        Pearson correlation factor between LIDS data and its fit function

        Parameters
        ----------
        lids: pandas.Series
            Output data from LIDS transformation.
        params: lmfit.Parameters, optional
            Parameters for the fit function.
            If None, self.lids_fit_params is used instead.
            Default is None.

        Returns
        -------
        r: numpy.float64
            Pearson’s correlation coefficient
        p: numpy.float64
            2-tailed p-value
        '''

        # Store actual sampling frequency
        freq = pd.Timedelta(lids.index.freq)
        # Check if index frequency is not None
        if freq is None:
            raise ValueError(
                "The input data have no index frequency. "
                "This could indicate unevenly sampled data.\n"
                "The current implementation of the LIDS analysis does not "
                "support such data. A possible workaround would consist in "
                "resampling the data with the assumed acquisition frequency."
            )
        # Drop potential NaN
        lids_cleaned = lids.dropna()
        # Create associated ordinal index
        x = ((lids_cleaned.index - lids_cleaned.index[0])/freq).values
        if params is None:
            params = self.lids_fit_results.params
        return pearsonr(lids_cleaned, self.lids_fit_func(x, params))

    def lids_mri(self, lids, params=None):
        r'''Munich Rhythmicity Index

        The Munich Rhythmicity Index (MRI) is defined as
        :math:`MRI = A \times r` with :math:`A`, the cosine fit amplitude and
        :math:`r`, the bivariate correlation coefficient (a.k.a. Pearson'r).

        Parameters
        ----------
        lids: pandas.Series
            Output data from LIDS transformation.
        params: lmfit.Parameters, optional
            Parameters for the fit function.
            If None, self.lids_fit_params is used instead.
            Default is None.

        Returns
        -------
        mri: numpy.float64
            Munich Rhythmicity Index
        '''
        if params is None:
            params = self.lids_fit_results.params

        # Pearson's r
        pearson_r = self.lids_pearson_r(lids, params)[0]

        # Oscillation range = [-A,+A] => 2*A
        oscillation_range = 2*params['amp'].value

        # MRI
        mri = pearson_r*oscillation_range

        return mri

    def lids_period(self, params=None, freq='s'):
        r'''LIDS period

        Convert the period of the LIDS oscillations as estimated by the fit
        function to a TimeDelta.

        Parameters
        ----------
        params: lmfit.Parameters, optional
            Parameters for the fit function.
            If None, self.lids_fit_params is used instead.
            Default is None.
        s: str, optional
            Frequency to cast the output timedelta to.
            Default is 's'.

        Returns
        -------
        lids_period: numpy.timedelta64[freq]


        Note
        ----
        As there is no way to automatically derive the LIDS period from the fit
        parameters, the fitted period needs to be set via its own setter
        function.
        '''
        # Access fit parameters
        if params is None:
            if self.lids_fit_results is None:
                warnings.warn(
                    'The LIDS fit results are not available.\n'
                    'Run lids_fit() before accessing this method.\n'
                    'Returning None.',
                    UserWarning
                )
                # TODO: evaluate if raise ValueError('') more appropriate
                return None
            else:
                params = self.lids_fit_results.params

        lids_period = params['period']*pd.Timedelta(
                params['freq'],
                unit='s'
            )
        return lids_period.astype('timedelta64[{}]'.format(freq))

    def lids_phases(self, params=None, radians=False):
        r'''LIDS onset and offset phases

        These phases are defined as the minimal distance to the first/last peak
        from sleep onset/offset, respectively.

        Parameters
        ----------
        params: lmfit.Parameters, optional
            Parameters for the fit function.
            If None, self.lids_fit_params is used instead.
            Default is None.
        radians: bool, optional
            If set to True, the phases are calculated in radians instead of
            degrees.
            Default is False.

        Returns
        -------
        onset_phase, offset_phase: numpy.float64
        '''

        # Access fit parameters
        if params is None:
            if self.lids_fit_results is None:
                warnings.warn(
                    'The LIDS fit results are not available.\n'
                    'Run lids_fit() before accessing this method.\n'
                    'Returning None.',
                    UserWarning
                )
                # TODO: evaluate if raise ValueError('') more appropriate
                return None
            else:
                params = self.lids_fit_results.params

        # Phase at sleep onset
        onset_phase = 2*np.pi - params['phase'].value

        # Phase at sleep offset
        # Defined as the value of an inverse cosine fit function at sleep
        # offset t_1,
        # modulo 2*Pi: Phi@Offset = 2*pi*t_1/T + phi [2*pi]
        t_1 = params['x_max'].value
        T = params['period'].value
        phi = params['phase'].value
        offset_phase = (2*np.pi*t_1/T + phi) % (2*np.pi)

        if not radians:
            onset_phase *= 180/np.pi
            offset_phase *= 180/np.pi

        return onset_phase, offset_phase

    def lids_convert_to_internal_time(self, lids, t_norm='90min'):
        r'''Convert LIDS data index to internal time.

        XXX

        Parameters
        ----------
        lids: pandas.Series
            Output data from LIDS transformation.
        t_norm: str, optional
            Default period used to normalize the fitted LIDS period.
            Default is '90min'.

        Returns
        -------
        ts: pandas.Series
            LIDS data with internal time since sleep onset as index.
        '''
        # Store actual sampling frequency
        freq = pd.Timedelta(lids.index.freq)

        # External timeline of the current LIDS data since sleep onset
        t_ext = pd.timedelta_range(
            start='0 day',
            periods=lids.index.size,
            freq=freq
        )

        # Scaling factor, relative to the LIDS period, normalized to t_norm
        scaling_factor = pd.Timedelta(t_norm)/self.lids_period()

        # Internal timeline (aka: external timeline, rescaled to LIDS period)
        # t_int = scaling_factor*t_ext
        t_int = pd.TimedeltaIndex(scaling_factor*t_ext.values, freq='infer')

        # Construct a new Series with internal timeline as index
        lids_rescaled = pd.Series(lids.values, index=t_int)

        # Resample LIDS data to restore the original bin width of its Index
        # Infer missing data via interpolation
        lids_resampled = lids_rescaled.resample(
            freq
            # label='right',
            # closed='right'
        ).mean()

        return lids_resampled.interpolate(method='linear')

    def lids_preprocessing(
        self,
        sleep_bouts,
        duration_min='3H',
        duration_max='12H',
        resampling_freq='10min',
        smooth_method='mva',
        smooth_resolution='30Min',
        concat_time_delta=None
    ):
        r'''Data preprocessing for LIDS analysis

        Filter data according to their time duration, smooth them and
        eventually concatenate them.

        Parameters
        ----------
        sleep_bouts: list of pandas.Series
            Data to filter and smooth.
        duration_min: str, optional
            Minimal time duration for a time series to be kept.
            Default is '30min'.
        duration_max: str, optional
            Maximal time duration for a time series to be kept.
            Default is '12h'.
        resampling_freq: str, optional
            Frequency of the resampling applied prior to LIDS transformation.
            Default is '10min'.
        smooth_method: str, optional
            Method used to smooth the data.
            Available options are:
                'mva': moving average
                'gaussian': gaussian kernel
                'none': no smoothing
            Default is 'mva'.
        smooth_resolution: str, optional
            If method='mva': Size of the rolling window.
            If method='gaussian': Standard deviation of the gaussian kernel.
            Default is '30min'.
        concat_time_delta: str, optional
            If not set to None, consecutive sleep bouts separated by less than
            'concat_time_delta' are concatenated.
            Default is None.

        Returns
        -------
        concat_lids: pandas.Series
        '''

        # Filtering
        # Sleep bouts shorted than duration_min and
        # longer than duration_max are discarded
        filtered_sleep_bouts = self.filter(
            sleep_bouts,
            duration_min=duration_min,
            duration_max=duration_max
        )

        # LIDS conversion
        # Resample activity counts and apply LIDS transformation
        smooth_lids = [
            self.lids_transform(
                ts,
                resampling_freq=resampling_freq,
                method=smooth_method,
                resolution=smooth_resolution
            ) for ts in filtered_sleep_bouts
        ]

        # LIDS concatenation
        # Concatenate consecutive sleep bouts if delta_time < concat_time_delta
        if concat_time_delta is not None:
            concat_indices, concat_lids = self.concat(
                smooth_lids,
                time_delta=concat_time_delta
            )
        else:
            concat_lids = smooth_lids

        return concat_lids

    def lids_summary(
        self,
        subject_id,
        lids_bouts,
        method='leastsq',
        scan_period=True,
        bounds=('30min', '180min'),
        step='5min',
        nan_policy='raise',
        verbose_fit=False,
        verbose=False
    ):
        r'''Estimated parameters and goodness-of-fit statistics for LIDS

        Fit the input LIDS-transformed bout and create a DataFrame containing
        the estimated values of the LIDS fit parameters as well as
        goodness-of-fit statistics such as AIC, BIC, etc.

        Parameters
        ----------
        subject_id: scalar or str
            Subject ID to insert into the returned pandas.DataFrame.
        lids_bouts: list of pandas.Series
            Output data from LIDS transformation.
        method: str, optional
            Name of the fitting method to use [1]_.
            Default is 'leastsq'.
        scan_period: bool, optional
            If set to True, the period of the LIDS fit function is fixed and
            varied between the specified bounds.
            The selected period corresponds to the highest MRI value.
            Otherwise, the period is a free parameter of the fit.
            Default is True.
        bounds: 2-tuple of str, optional
            Lower and upper bounds for the periods to be tested.
            If scan_period is set to False, the bounds are ignored.
            Default is ('30min','180min').
        step: str, optional
            Time delta between the periods to be tested.
            Default is '5min'
        nan_policy: str, optional
            Specifies action if the objective function returns NaN values.
            One of:
                'raise': a ValueError is raised
                'propagate': the values returned from userfcn are un-altered
                'omit': non-finite values are filtered
            Default is 'raise'.
        verbose_fit: bool, optional
            If set to True, display fit informations
        verbose: bool, optional
            If set to True, print summary statistics.
            Default is False.

        Returns
        -------
        df_params: pandas.DataFrame
            DataFrame with the estimated parameters and goodness-of-fit
            statistics.

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html
        '''

        ldf = []
        for idx, lids in enumerate(lids_bouts):
            # Fit LIDS data
            self.lids_fit(
                lids,
                method=method,
                scan_period=scan_period,
                bounds=bounds,
                step=step,
                mri_profile=False,
                nan_policy=nan_policy,
                verbose=verbose_fit
            )

            if self.lids_fit_results is None:
                continue

            # Extract fit parameters
            fit_params = self.lids_fit_results.params.valuesdict()

            # Add subject ID
            fit_params['subject_id'] = subject_id

            # Add pearson correlation factor to fit parameters
            fit_params['pearson_r'] = self.lids_pearson_r(lids)[0]
            # Add MRI to fit parameters
            fit_params['mri'] = self.lids_mri(lids)

            # Add fit results to fit parameters
            fit_params['status'] = int(self.lids_fit_results.success)
            fit_params['aic'] = self.lids_fit_results.aic
            fit_params['bic'] = self.lids_fit_results.bic
            fit_params['redchisq'] = self.lids_fit_results.redchi

            # Calculate phase at sleep onset and offset
            lids_onset_phase, lids_offset_phase = self.lids_phases()

            # Add phases to fit parameters
            fit_params['phase_onset'] = lids_onset_phase
            fit_params['phase_offset'] = lids_offset_phase

            # Add sleep bout duration
            fit_params['duration'] = lids.index[-1]-lids.index[0]

            # Create a DF with the fit parameters
            df_params = pd.DataFrame(fit_params, index=[idx])

            if verbose:
                print(df_params)

            ldf.append(df_params)

        return pd.concat(ldf) if len(ldf) > 0 else None
