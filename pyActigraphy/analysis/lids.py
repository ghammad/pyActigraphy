import numpy as np
import pandas as pd
import warnings

from functools import reduce
from lmfit import fit_report, minimize, Parameters
from scipy.stats import pearsonr, poisson
from spm1d.util import smooth as spm_smooth


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
    fit_func_list = ['cosine', 'chirp', 'modchirp']

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
        fit_funcs = {'cosine': _cosine, 'chirp': _lfm, 'modchirp': _lfam}
        if fit_func not in fit_funcs.keys():
            raise ValueError(
                '`Fit function` must be "%s". You passed: "%s"' %
                ('" or "'.join(list(fit_funcs.keys())), fit_func)
            )

        # Fit objective functions (i.e. funcitons to be minimized)
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

        self.__freq = None  # pd.Timedelta
        self.__lids_func = lids_funcs[lids_func]  # LIDS transformation fct
        self.__fit_func = fit_funcs[fit_func]  # Fit function to LIDS
        self.__fit_obj_func = fit_obj_funcs[fit_obj_func]  # Fit obj function
        self.__fit_reduc_func = fit_reduc_funcs[fit_obj_func]

        if fit_params is None:
            fit_params = Parameters()
            # Default parameters for the cosine fit function
            fit_params.add('amp', value=50, min=0, max=100)
            fit_params.add('phase', value=0.0, min=-2*np.pi, max=2*np.pi)
            fit_params.add('period', value=9, min=0)  # Dummy value
            # Introduce inequality amp+offset < 100
            fit_params.add('delta', value=60, max=100, vary=True)
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

        self.__fit_initial_params = fit_params
        # self.__fit_params = None
        self.__fit_results = None
        # self.__fit_period = None

    @property
    def freq(self):
        r'''Sampling frequency of the LIDS transformed data'''
        if self.__freq is None:
            warnings.warn(
                'The sampling frequency of the LIDS data is not set. '
                'Run lids_transform() before accessing this attribute.',
                UserWarning
            )
        return self.__freq

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
        if self.__fit_results is None:
            warnings.warn(
                'The fit results is None. '
                'Run lids_fit() before accessing this attribute.',
                UserWarning
            )
        return self.__fit_results

    def filter(self, ts, duration_min='3H', duration_max='12H'):
        r'''Filter data according to their duration

        Before performing a LIDS analysis, it is necessary to drop sleep bouts
        that too short or too long.
        '''

        def duration(s):
            return s.index[-1]-s.index[0]

        td_min = pd.Timedelta(duration_min)
        td_max = pd.Timedelta(duration_max)

        from itertools import filterfalse
        filtered = []
        filtered[:] = filterfalse(
            lambda x: duration(x) < td_min or duration(x) > td_max,
            ts
        )
        return filtered

    def __smooth(self, lids, method, win_size):
        r'''Smooth LIDS data

        By default, smooth with a centered moving average using a `win_size`
        window'''

        # Smooth functions
        lids_smooth_funcs = ['mva', 'kernel', 'none']
        if method not in lids_smooth_funcs:
            raise ValueError(
                '`LIDS smooth function` must be "%s". You passed: "%s"' %
                ('" or "'.join(list(lids_smooth_funcs)), method)
            )

        if method == 'mva':
            return lids.rolling(win_size, center=True, min_periods=1).mean()
        elif method == 'kernel':
            smooth_lids = spm_smooth(lids.values, fwhm=win_size)
            return pd.Series(data=smooth_lids, index=lids.index)
        elif method == 'none':
            return lids

    def lids_transform(
        self, ts, method='mva', win_td='30min', resampling_freq=None
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
        method: str, optional
            Method to smooth the data.
            Available options are:

            * 'mva': moving average
            * 'kernel': gaussian kernel
            * 'none': no smoothing

            Default is 'mva'.
        win_td: str, optional
            Size of the moving average window.
            Default is '30min'.
        resampling_freq: str, optional
            Frequency of the resampling applied prior to LIDS transformation.
            Default is None.

        Returns
        -------
        smooth_lids: pandas.Series
        '''

        # Resample data to the required frequency
        if resampling_freq is not None:
            rs = ts.resample(resampling_freq).sum()
        else:
            rs = ts
        # Apply LIDS transformation x: 100/(x+1)
        lids = rs.apply(self.lids_func)

        # Store actual sampling frequency
        self.__freq = pd.Timedelta(lids.index.freq)

        # Series with a DateTimeIndex don't accept 'time-aware' centered window
        # Convert win_size (TimeDelta) into a number of time bins
        win_size = int(pd.Timedelta(win_td)/self.__freq)

        # Smooth LIDS-transformed data
        smooth_lids = self.__smooth(lids, method=method, win_size=win_size)

        return smooth_lids

    def lids_fit(
        self,
        lids,
        method='leastsq',
        scan_period=True,
        bounds=('30min', '180min'),
        step='5min',
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
        nan_policy: str, optional
            Specifies action if the objective function returns NaN values.
            One of:

            * 'raise': a ValueError is raised
            * 'propagate': the values returned from userfcn are un-altered
            * 'omit': non-finite values are filtered

            Default is 'raise'.
        verbose: bool, optional
            If set to True, display fit informations

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html
        '''

        # # Define residual function to minimize
        # def residual(params, x, data):
        #     model = self.lids_fit_func(x, params)
        #     return (data-model)

        # Define the x range
        x = np.arange(lids.index.size)

        if scan_period:

            # Define bounds for the period
            period_start = pd.Timedelta(bounds[0])/self.__freq
            period_end = pd.Timedelta(bounds[1])/self.__freq
            period_range = period_end-period_start
            period_step = pd.Timedelta(step)/self.__freq

            test_periods = np.linspace(
                period_start,
                period_end,
                num=int(period_range/period_step)+1
            )

            # Fit data for each test period
            mri = -np.inf
            fit_result_tmp = None
            initial_period = self.__fit_initial_params['period'].value
            for test_period in test_periods:
                # Fix test period
                self.__fit_initial_params['period'].value = test_period
                self.__fit_initial_params['period'].vary = False

                # Minimize residuals
                fit_result_tmp = minimize(
                    self.__fit_obj_func,
                    self.__fit_initial_params,
                    args=(x,  lids.values, self.lids_fit_func),
                    nan_policy=nan_policy,
                    reduce_fcn=self.__fit_reduc_func
                )
                # Print fit parameters if verbose
                if verbose:
                    print(fit_report(fit_result_tmp))
                # Calculate the MR index
                mri_tmp = self.lids_mri(lids, fit_result_tmp.params)
                if verbose:
                    pearson_r = self.lids_pearson_r(
                        lids, fit_result_tmp.params)[0]
                    print('Pearson r: {}'.format(pearson_r))
                    print('MRI: {}'.format(mri_tmp))

                # If the newly calculated MRI is higher than the current MRI
                if mri_tmp > mri and (test_period != period_end):
                    # Store MRI
                    mri = mri_tmp
                    # Store fit parameters
                    fit_result = fit_result_tmp

            if verbose:
                print('Highest MRI: {}'.format(mri))
            # Set back original value
            self.__fit_initial_params['period'].value = initial_period
            self.__fit_initial_params['period'].vary = True
        else:

            # Minimize residuals
            fit_result = minimize(
                self.__fit_obj_func,
                self.__fit_initial_params,
                args=(x,  lids.values, self.lids_fit_func),
                nan_policy=nan_policy,
                reduce_fcn=self.__fit_reduc_func
            )
            # Print fit parameters if verbose
            if verbose:
                print(fit_report(fit_result))
            if verbose:
                # Calculate the MR index
                pearson_r = self.lids_pearson_r(lids, fit_result.params)[0]
                mri = self.lids_mri(lids, fit_result.params)
                print('Pearson r: {}'.format(pearson_r))
                print('MRI: {}'.format(mri))

        self.__fit_results = fit_result
        # self.lids_fit_params = fit_result.params
        # self.lids_fit_period = fit_result.params['period'].value

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

        x = np.arange(lids.index.size)
        if params is None:
            params = self.lids_fit_results.params
        return pearsonr(lids, self.lids_fit_func(x, params))

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

    def lids_period(self, freq='s'):
        r'''LIDS period

        Convert the period of the LIDS oscillations as estimated by the fit
        function to a TimeDelta.

        Parameters
        ----------
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
        if self.freq is None:
            # TODO: evaluate if raise ValueError('') more appropriate
            return None
        elif self.lids_fit_results is None:
            # TODO: evaluate if raise ValueError('') more appropriate
            return None
        else:
            lids_period = self.lids_fit_results.params['period']*self.freq
            return lids_period.astype('timedelta64[{}]'.format(freq))

    def lids_phases(self, lids, step=.1):
        r'''LIDS onset and offset phases in degrees

        Parameters
        ----------
        lids: pandas.Series
            Output data from LIDS transformation.
        step: float, optional
            Step between points at which the LIDS fit is evaluated.
            Default is '0.1'.

        Returns
        -------
        onset_phase, offset_phase: numpy.float64
        '''

        if self.lids_fit_results is None:
            # TODO: evaluate if raise ValueError('') more appropriate
            return None

        if self.lids_fit_results is None:
            # TODO: evaluate if raise ValueError('') more appropriate
            return None

        # Access fit parameters
        params = self.lids_fit_results.params

        # Fit support range
        x = np.arange(lids.index.size, step=step)

        # LIDS fit derivatives (1st and 2nd)
        df_dx = np.gradient(self.lids_fit_func(x, params), step)
        d2f_dx2 = np.gradient(df_dx, step)

        # Index of the 1st maxima (i.e 1st maximum of the LIDS oscillations)
        first_max_idx = np.argmax(_extrema_points(df_dx, d2f_dx2))
        # Convert the index into a phase using the fitted period
        onset_phase = (first_max_idx*step)/params['period']*360

        # Index of the last 'increasing' inflexion points in LIDS oscillations
        # before sleep offset
        last_inflex_idx = -1 * (
            # reverse order to find last
            np.argmax(_inflexion_points(df_dx, d2f_dx2)[::-1]) +
            1  # to account for index shifting during reverse (-1: 0th elem)
        )
        # Convert the index into a phase using the fitted period
        offset_phase = np.abs(last_inflex_idx*step/params['period']*360)

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

        # External timeline of the current LIDS data since sleep onset
        t_ext = pd.timedelta_range(
            start='0 day',
            periods=lids.index.size,
            freq=self.freq
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
            self.freq
            # label='right',
            # closed='right'
        ).mean()

        return lids_resampled.interpolate(method='linear')

    def lids_summary(self, lids, verbose=False):
        r'''Calculate summary statistics for LIDS

        Fit all LIDS-transformed bouts and calculate the mean period, the mean
        mri, the mean number of LIDS cycles and the dampening factor of the
        mean LIDS profile.

        Parameters
        ----------
        lids: list of pandas.Series
            Output data from LIDS transformation.
        verbose: bool, optional
            If set to True, print summary statistics.
            Default is False.

        Returns
        -------
        summary: dict
            Dictionary with the summary statistics.
        '''

        ilids = []  # LIDS profiles
        periods = []  # List of LIDS periods
        mris = []  # MRI indices
        ncycles = []  # Number of LIDS cycles/sleep bout

        for idx, s in enumerate(lids):
            # Fit LIDS data
            self.lids_fit(s, verbose=False)

            # Verify LIDS period
            period = self.lids_period(freq='s')

            # Calculate MRI
            mri = self.lids_mri(s)

            # Calculate the number of LIDS cycle (as sleep bout length/period):
            ncycle = s.index.values.ptp()/np.timedelta64(1, 's')
            ncycle /= period.astype(float)

            if verbose:
                print('-'*20)
                print('Sleep bout nr {}'.format(idx))
                print('- Period: {!s}'.format(period))
                print('- MRI: {}'.format(mri))
                print('- Number of cycles: {}'.format(ncycle))

            # Rescale LIDS timeline to LIDS period
            rescaled_lids = self.lids_convert_to_internal_time(s)

            periods.append(period)
            mris.append(mri)
            ncycles.append(ncycle)
            ilids.append(rescaled_lids)

        # Create the mean LIDS profile
        lids_profile = reduce(
            (lambda x, y: x.add(y, fill_value=0)),
            ilids
        )/len(ilids)

        # Fit mean LIDS profile with a pol0
        fit_params = np.polyfit(
            x=range(len(lids_profile.index)),
            y=lids_profile.values,
            deg=1
        )

        # LIDS summary
        summary = {}
        summary['Mean number of LIDS cycles'] = np.mean(ncycles)
        summary['Mean LIDS period (s)'] = np.mean(periods).astype(float)
        summary['Mean MRI'] = np.mean(mris)
        summary[
            'LIDS dampening factor (counts/{})'.format(self.freq)
        ] = fit_params[0]

        return summary
