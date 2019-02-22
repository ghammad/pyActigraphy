import numpy as np
import pandas as pd
import warnings

from lmfit import fit_report, minimize, Parameters
from scipy.stats import pearsonr


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


def _cosine(x, params):
    r'''1-harmonic cosine function'''

    A = params['amp']
    phi = params['phase']
    T = params['period']
    offset = params['offset']

    return A*np.cos(2*np.pi/T*x+phi) + offset


class LIDS():
    """
    Class for Locomotor inactivity during sleep (LIDS) Analysis

    Winnebeck, E. C., Fischer, D., Leise, T., & Roenneberg, T. (2018).
    Dynamics and Ultradian Structure of Human Sleep in Real Life.
    Current Biology, 28(1), 49–59.e5. http://doi.org/10.1016/j.cub.2017.11.063

    """

    def __init__(self):

        self.__freq = None  # pd.Timedelta
        self.__lids_func = lambda x: 100/(x+1)  # LIDS transformation function
        self.__fit_func = _cosine  # Default fit function to LIDS oscillations
        self.__fit_args = None
        self.__fit_period = None

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
    def lids_fit_params(self):
        r'''Arguments of the fit function to LIDS oscillations'''
        return self.__fit_params

    @property
    def lids_fit_period(self):
        r'''Period of the fit function to LIDS oscillations'''
        if self.__fit_period is None:
            warnings.warn(
                'The period of the fit to the LIDS oscillations is not set.',
                UserWarning
            )
            # TODO: evaluate if raise ValueError('') more appropriate
        return self.__fit_period

    @lids_fit_period.setter
    def lids_fit_period(self, value):
        r'''Period of the fit function to LIDS oscillations'''
        self.__fit_period = value

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

    def __smooth(self, lids, win_size):
        r'''Smooth LIDS data with a centered moving average using a `win_size`
        window'''

        return lids.rolling(win_size, center=True, min_periods=1).mean()

    def lids_transform(self, ts, win_td='30min', resampling_freq=None):
        r'''Apply LIDS transformation to activity data

        This transformation comprises:
        * resampling via summation (optional)
        * non-linear LIDS trasformation
        * smoothing with a centered moving average

        Parameters
        ----------
        ts: pandas.Series
            Data identified as locomotor activity during sleep.
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
        smooth_lids = self.__smooth(lids, win_size=win_size)

        return smooth_lids

    def lids_fit(
        self,
        lids,
        p0=None,
        bounds=('30min', '180min'),
        step='5min',
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
        p0: array, optional
            Initial values of the fit parameters.
        bounds: 2-tuple of str, optional
            Lower and upper bounds on periods to be tested.
            Default is ('30min','180min').
        step: str, optional
            Time delta between the periods to be tested.
        verbose: bool, optional
            If set to True, display fit informations
        '''

        x = np.arange(lids.index.size)

        # Define periods
        period_start = pd.Timedelta(bounds[0])/self.__freq
        period_end = pd.Timedelta(bounds[1])/self.__freq
        period_range = period_end-period_start
        period_step = pd.Timedelta(step)/self.__freq
        test_periods = np.linspace(
            period_start,
            period_end,
            num=int(period_range/period_step)
        )

        # Fit parameters
        params = Parameters()
        params.add('amp', value=50, min=0, max=100)
        params.add('offset', value=50, min=0, max=100)
        params.add('phase', value=0.0, min=-np.inf, max=np.inf)
        params.add('period', value=test_periods[0], vary=False)  # Fixed period

        # TODO: include reading values from p0

        # Define residuals
        def residual(params, x, data):
            model = self.lids_fit_func(x, params)
            return (data-model)

        # Fit data with 1-harmonic cosine function for each test period
        fit_results = []
        mri = -np.inf
        for test_period in test_periods:
            # Fix test period
            params['period'].value = test_period
            # Minimize residuals
            out = minimize(residual, params, args=(x,  lids.values))
            # Print fit parameters if verbose
            if verbose:
                print(fit_report(out))
                # out.params.pretty_print()
            # Store fit results
            fit_results.append(out)
            # Calculate the MR index
            pearson_r = self.lids_pearson_r(lids, out.params)[0]
            # Oscillation range = [-A,+A] => 2*A
            # oscillation_range = 2*out.params['amp'].value
            # mri_tmp = pearson_r*oscillation_range
            mri_tmp = self.lids_mri(lids, out.params)
            if verbose:
                print('Pearson r: {}'.format(pearson_r))
                print('MRI: {}'.format(mri_tmp))

            # If the newly calculated MRI is higher than the current MRI
            if mri_tmp > mri:
                # Store MRI
                mri = mri_tmp
                # Store fit parameters
                self.__fit_params = out.params
                self.lids_fit_period = out.params['period'].value

        if verbose:
            print('Highest MRI: {}'.format(mri))

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
            params = self.lids_fit_params
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
            params = self.lids_fit_params

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
        elif self.lids_fit_period is None:
            # TODO: evaluate if raise ValueError('') more appropriate
            return None
        else:
            lids_period = self.lids_fit_period*self.freq
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

        if self.lids_fit_period is None:
            # TODO: evaluate if raise ValueError('') more appropriate
            return None

        if self.lids_fit_params is None:
            warnings.warn(
                'LIDS fit parameters are not set. '
                'Run lids_fit() before calling this function.\n'
                'Returning None.',
                UserWarning
            )
            return None

        # from scipy.misc import derivative

        # Fit support range
        x = np.arange(lids.index.size, step=step)

        # LIDS fit derivatives (1st and 2nd)
        df_dx = np.gradient(self.lids_fit_func(x, self.lids_fit_params), step)
        # derivative(
        #     func=self.lids_fit_func,
        #     x0=x, dx=step, n=1,
        #     args=self.lids_fit_params
        # )
        d2f_dx2 = np.gradient(df_dx, step)
        # derivative(
        #     func=self.lids_fit_func,
        #     x0=x, dx=step, n=2,
        #     args=self.lids_fit_params
        # )

        # Index of the 1st maxima (i.e 1st maximum of the LIDS oscillations)
        first_max_idx = np.argmax(_extrema_points(df_dx, d2f_dx2))
        # Convert the index into a phase using the fitted period
        onset_phase = (first_max_idx*step)/self.lids_fit_period*360

        # Index of the last 'increasing' inflexion points in LIDS oscillations
        # before sleep offset
        last_inflex_idx = -1 * (
            # reverse order to find last
            np.argmax(_inflexion_points(df_dx, d2f_dx2)[::-1])
            + 1  # to account for index shifting during reverse (-1: 0th elem)
        )
        # Convert the index into a phase using the fitted period
        offset_phase = np.abs(last_inflex_idx*step/self.lids_fit_period*360)

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
        t_int = scaling_factor*t_ext

        # Construct a new Series with internal timeline as index
        lids_rescaled = pd.Series(lids.values, index=t_int)

        # Resample LIDS data to restore the original bin width of its Index
        # Infer missing data via interpolation
        lids_resampled = lids_rescaled.resample(self.freq).mean()

        return lids_resampled.interpolate(method='linear')
