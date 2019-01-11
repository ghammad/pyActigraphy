import numpy as np
import pandas as pd
import re
from scipy.ndimage import binary_closing, binary_opening
from ..filters import _create_inactivity_mask
from ..metrics import _average_daily_activity


def _td_format(td):
    return '{:02}:{:02}:{:02}'.format(
        td.components.hours,
        td.components.minutes,
        td.components.seconds
    )


def _activity_onset_time(data, whs=4):

    avgdaily = _average_daily_activity(data=data, cyclic=True)

    r = avgdaily.rolling(whs*2, center=True)

    aot = r.apply(
        lambda x: np.mean(x[whs:])/np.mean(x[0:whs])-1,
        raw=False
    ).idxmax()

    return aot


def _activity_offset_time(data, whs=4):

    avgdaily = _average_daily_activity(data=data, cyclic=True)

    r = avgdaily.rolling(whs*2, center=True)

    aot = r.apply(
        lambda x: np.mean(x[0:whs])/np.mean(x[whs:])-1,
        raw=False
    ).idxmax()

    return aot


def _padded_data(data, value, periods, frequency):

    date_offset = pd.DateOffset(seconds=frequency.total_seconds())
    pad_beginning = pd.Series(
        data=value,
        index=pd.date_range(
            end=data.index[0],
            periods=periods,
            freq=date_offset,
            closed='left'
        ),
        dtype=int
    )
    pad_end = pd.Series(
        data=value,
        index=pd.date_range(
            start=data.index[-1],
            periods=periods,
            freq=date_offset,
            closed='right'
        ),
        dtype=int
    )
    return pd.concat([pad_beginning, data, pad_end])


def _ratio_sequences_of_zeroes(
    data, seq_length, n_boostrap, seed=0, with_replacement=True
):
    # Calculate a moving sum with a window of size 'seq_length'
    rolling_sum = data.rolling(seq_length).sum()
    # Set seed for reproducibility
    np.random.seed(seed)
    random_sample = np.random.choice(
        rolling_sum.values,
        size=n_boostrap*len(rolling_sum),
        replace=with_replacement
    )
    # Calculate the ratio of zero elements
    ratio = 1 - np.count_nonzero(random_sample)/len(random_sample)
    return ratio


def _estimate_zeta(data, seq_length_max, n_boostrap=100, level=0.05):
    ratios = np.fromiter((
        _ratio_sequences_of_zeroes(data, n, n_boostrap) for n in np.arange(
            1, seq_length_max+1
        )),
        np.float,
        seq_length_max
    )
    zeta_est = np.argmax(ratios < level)
    return zeta_est


def _window_convolution(x, scale, window, offset=0.0):

    return scale * np.dot(x, window) + offset


def _cole_kripke(data, scale, window, threshold):

    """Automatic scoring methods from Cole and Kripke"""

    ck = data.rolling(
        window.size, center=True
    ).apply(_window_convolution, args=(scale, window), raw=False)

    return (ck < threshold).astype(int)


def _sadeh(data, offset, weights, threshold):

    """ Activity-Based Sleep-Wake Identification"""

    r = data.rolling(11, center=True)

    mean_W5 = r.mean()

    NAT = r.apply(lambda x: np.size(np.where((x > 50) & (x < 100))), raw=False)

    sd_Last6 = data.rolling(6).std()

    logAct = data.shift(-1).apply(lambda x: np.log(1+x))

    sadeh = pd.concat(
        [mean_W5, NAT, sd_Last6, logAct],
        axis=1,
        keys=['mean_W5', 'NAT', 'sd_Last6', 'logAct']
    )

    sadeh['PS'] = sadeh.apply(
        _window_convolution, axis=1, args=(1.0, weights, offset)
    )

    return (sadeh['PS'] > threshold).astype(int)


def _scripps(data, scale, window, threshold):

    """Scripps Clinic algorithm for sleep-wake scoring."""

    scripps = data.rolling(
        window.size, center=True
    ).apply(_window_convolution, args=(scale, window), raw=False)

    return (scripps < threshold).astype(int)


class ScoringMixin(object):
    """ Mixin Class for scoring:
    - sleep/wake periods
    - ...
    """

    def AonT(self, freq='5min', whs=12, binarize=True, threshold=4):
        r"""Activity onset time.

        Activity onset time derived from the daily activity profile.

        Parameters
        ----------
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
            Default is '5min'.
        whs: int, optional
            Window half size.
            Default is 12.
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 4.

        Returns
        -------
        aot: Timedelta
            Activity onset time.


        Notes
        -----

        This automatic detection of the activity onset time is based on the
        daily activity profile. It returns the time point where difference
        between the mean activity over :math:`whs` epochs before and after this
        time point is maximum:

        .. math::
            AonT = \max_{t}(
            \frac{\sum_{i=-whs}^{0} x_{t+i}}{\sum_{i=1}^{whs} x_{t+i}} - 1
            )

        with:

        * :math:`x_{i}` is the activity count at time :math:`i`.

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.AonT()
            Timedelta('0 days 07:15:00')
            >>> raw.AonT(binarize=False)
            Timedelta('0 days 07:05:00')

        """
        data = self.resampled_data(freq, binarize, threshold)

        return _activity_onset_time(data, whs=whs)

    def AoffT(self, freq='5min', whs=12, binarize=True, threshold=4):
        r"""Activity offset time.

        Activity offset time derived from the daily activity profile.

        Parameters
        ----------
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
            Default is '5min'.
        whs: int, optional
            Window half size.
            Default is 12.
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 4.

        Returns
        -------
        aot: Timedelta
            Activity offset time.


        Notes
        -----

        This automatic detection of the activity offset time is based on the
        daily activity profile. It returns the time point where relative
        difference between the mean activity over :math:`whs` epochs after and
        before this time point is maximum:

        .. math::
            AoffT = \max_{t}(
            \frac{\sum_{i=1}^{whs} x_{t+i}}{\sum_{i=-whs}^{0} x_{t+i}} - 1
            )

        with:

        * :math:`x_{i}` is the activity count at time :math:`i`.

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.AoffT()
            Timedelta('0 days 23:20:00')
            >>> raw.AoffT(binarize=False)
            Timedelta('0 days 23:05:00')

        """

        data = self.resampled_data(freq, binarize, threshold)

        return _activity_offset_time(data, whs=whs)

    def CK(
        self,
        scale=0.00001,
        window=np.array([400, 600, 300, 400, 1400, 500, 350, 0, 0], np.int32),
        threshold=1.0
    ):

        r"""Cole&Kripke algorithm for sleep-wake identification.

        Algorithm for automatic sleep scoring based on wrist activity,
        developped by Cole, Kripke et al [1]_.


        Parameters
        ----------
        scale: float, optional
            Scale parameter P.
            Default is 0.00001.
        window: np.array
            Array of weighting factors, :math:`W_{i}`.
            Default is [400, 600, 300, 400, 1400, 500, 350, 0, 0]
        threshold: float, optional
            Threshold value for scoring sleep/wake.
            Default is 1.0.

        Returns
        -------
        ck: pandas.core.Series
            Time series containing the `D` scores (0: sleep, 1: wake) for each
            epoch.

        Notes
        -----

        The output variable D of the CK algorithm is defined in [1]_ as:

        .. math::

            D = P*(
                [W_{-4},\dots,W_{0},\dots,W_{+2}]
                \cdot
                [A_{-4},\dots,A_{0},\dots,A_{+2}])

        with:

        * D < 1 == sleep, D >= 1 == wake;
        * P, scale factor;
        * :math:`W_{0},W_{-1},W_{+1},\dots`, weighting factors for the present
          minute, the previous minute, the following minute, etc.;
        * :math:`A_{0},A_{-1},A_{+1},\dots`, activity scores for the present
          minute, the previous minute, the following minute, etc.


        References
        ----------

        .. [1] Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J.,
               & Gillin, J. C. (1992). Automatic Sleep/Wake Identification
               From Wrist Activity. Sleep, 15(5), 461–469.
               http://doi.org/10.1093/sleep/15.5.461

        Examples
        --------


        """

        return _cole_kripke(self.data, scale, window, threshold)

    def Sadeh(
        self,
        offset=7.601,
        weights=np.array([-0.065, -1.08, -0.056, -0.703], np.float),
        threshold=0.0
    ):

        r"""Sadeh algorithm for sleep identification

        Algorithm for automatic sleep scoring based on wrist activity,
        developped by Sadeh et al [1]_.

        Parameters
        ----------
        offset: float, optional
            Offset parameter.
            Default is 7.601.
        weights: np.array
            Array of weighting factors for mean_W5, NAT, sd_Last6 and logAct.
            Default is [-0.065, -1.08, -0.056, -0.703].
        threshold: float, optional
            Threshold value for scoring sleep/wake.
            Default is 0.0.

        Returns
        -------

        Notes
        -----

        The output variable PS of the Sadeh algorithm is defined in [2]_ as:

        .. math::

            PS = 7.601 - 0.065·mean_W5 - 1.08·NAT - 0.056·sd_Last6 - 0.703·logAct

        with:

        * PS >= 0 == sleep, PS < 0 == wake;
        * mean_W5, the average number of activity counts during the scored
          epoch and the window of five epochs preceding and following it;
        * sd_Last6, the standard deviation of the activity counts during
          the scored epoch and the five epochs preceding it;
        * NAT, the number of epochs with activity level equal to or higher
          than 50 but lower than 100 activity counts in a window of 11 minutes
          that includes the scored epoch and the five epochs preceding and
          following it;
        * logAct, the natural logarithm of the number of activity counts during
          the scored epoch + 1.

        References
        ----------

        .. [1] Sadeh, A., Alster, J., Urbach, D., & Lavie, P. (1989).
               Actigraphically based automatic bedtime sleep-wake scoring:
               validity and clinical applications.
               Journal of ambulatory monitoring, 2(3), 209-216.
        .. [2] Sadeh, A., Sharkey, M., & Carskadon, M. A. (1994).
               Activity-Based Sleep-Wake Identification: An Empirical Test of
               Methodological Issues. Sleep, 17(3), 201–207.
               http://doi.org/10.1093/sleep/17.3.201

        Examples
        --------


        """

        return _sadeh(self.data, offset, weights, threshold)

    def Scripps(
        self,
        scale=0.204,
        window=np.array([
            0.0064,  # b_{-10}
            0.0074,  # b_{-9}
            0.0112,  # b_{-8}
            0.0112,  # b_{-7}
            0.0118,  # b_{-6}
            0.0118,  # b_{-5}
            0.0128,  # b_{-4}
            0.0188,  # b_{-3}
            0.0280,  # b_{-2}
            0.0664,  # b_{-1}
            0.0300,  # b_{+0}
            0.0112,  # b_{+1}
            0.0100,  # b_{+2}
            0.0000,  # b_{+3}
            0.0000,  # b_{+4}
            0.0000,  # b_{+5}
            0.0000,  # b_{+6}
            0.0000,  # b_{+7}
            0.0000,  # b_{+8}
            0.0000,  # b_{+9}
            0.0000   # b_{+10}
            ], np.float),
        threshold=1.0
    ):

        r"""Scripps Clinic algorithm for sleep-wake identification.

        Algorithm for automatic sleep scoring based on wrist activity,
        developed by Kripke et al [1]_.


        Parameters
        ----------
        scale: float, optional
            Scale parameter P
            Default is 0.204.
        window: np.array, optional
            Array of weighting factors :math:`W_{i}`
            Default values are identical to those found in the original
            publication [1]_.
        threshold: float, optional
            Threshold value for scoring sleep/wake.
            Default is 1.0.

        Returns
        -------

        scripps: pandas.core.Series
            Time series containing the `D` scores (0: sleep, 1: wake) for each
            epoch.


        Notes
        -----

        The output variable D of the Scripps algorithm is defined in [1]_ as:

        .. math::

            D = P*(
                [W_{-10},\dots,W_{0},\dots,W_{+10}]
                \cdot
                [A_{-10},\dots,A_{0},\dots,A_{+10}])

        with:

        * D < 1 == sleep, D >= 1 == wake;
        * P, scale factor;
        * :math:`W_{0},W_{-1},W_{+1},\dots`, weighting factors for the present
          epoch, the previous epoch, the following epoch, etc.;
        * :math:`A_{0},A_{-1},A_{+1},\dots`, activity scores for the present
          epoch, the previous epoch, the following epoch, etc.


        References
        ----------

        .. [1] Kripke, D. F., Hahn, E. K., Grizas, A. P., Wadiak, K. H.,
               Loving, R. T., Poceta, J. S., … Kline, L. E. (2010).
               Wrist actigraphic scoring for sleep laboratory patients:
               algorithm development. Journal of Sleep Research, 19(4),
               612–619. http://doi.org/10.1111/j.1365-2869.2010.00835.x

        """

        return _scripps(self.data, scale, window, threshold)

    def SoD(
        self,
        freq='5min',
        whs=4,
        start='12:00:00',
        period='5h',
        algo='unanimous'
    ):

        r"""Sleep over Daytime

        Quantify the volume of epochs identified as sleep over daytime (SoD),
        using sleep-wake scoring algorithms.

        Parameters
        ----------
        freq: str, optional
            Resampling frequency.
            Default is '5min'
        whs: int, optional
            Window half size.
            Default is 4
        start: str, optional
            Start time of the period of interest.
            Default: '12:00:00'
            Supported times: 'AonT', 'AoffT', any 'HH:MM:SS'
        period: str, optional
            Period length.
            Default is '10h'
        algo: str, optional
            Sleep scoring algorithm to use.
            Default is 'unanimous'.
            Supported algorithms: 'ck', 'sadeh', 'scripps', 'majority',
            'unanimous'

        Returns
        -------
        sod: pandas.core.Series
            Time series containing the epochs of rest (1) and
            activity (0) over the specified period.

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> SoD = rawAWD.SoD()
            >>> SoD
            2018-03-26 04:16:00    1
            2018-03-26 04:17:00    1
            2018-03-26 04:18:00    1
            (...)
            2018-04-05 16:59:00    0
            2018-04-05 16:59:00    0
            2018-04-05 17:00:00    0
            Length: 3175, dtype: int64

        """

        # Regex pattern for HH:MM:SS time string
        pattern = re.compile(r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$")
        if start == 'AonT':
            td = _activity_onset_time(self.data, freq=freq, whs=whs)
        elif start == 'AoffT':
            td = _activity_offset_time(self.data, freq=freq, whs=whs)
        elif pattern.match(start):
            td = pd.Timedelta(start)
        else:
            print('Input string for start ({}) not supported.'.format(start))

        start_time = _td_format(td)
        end_time = _td_format(td+pd.Timedelta(period))

        ts_days = self.data.between_time(start_time, end_time)

        if algo == 'ck':
            sod = _cole_kripke(
                ts_days,
                scale=0.00001,
                window=np.array(
                    [400, 600, 300, 400, 1400, 500, 350, 0, 0],
                    np.int32
                ),
                threshold=1.0
            )
        elif algo == 'sadeh':
            sod = _sadeh(
                ts_days,
                offset=7.601,
                weights=np.array([-0.065, -1.08, -0.056, -0.703], np.float),
                threshold=0.0
            )
        elif algo == 'scripps':
            sod = _scripps(
                ts_days,
                scale=0.204,
                window=np.array(
                    [
                        0.0064,  # b_{-10}
                        0.0074,  # b_{-9}
                        0.0112,  # b_{-8}
                        0.0112,  # b_{-7}
                        0.0118,  # b_{-6}
                        0.0118,  # b_{-5}
                        0.0128,  # b_{-4}
                        0.0188,  # b_{-3}
                        0.0280,  # b_{-2}
                        0.0664,  # b_{-1}
                        0.0300,  # b_{+0}
                        0.0112,  # b_{+1}
                        0.0100,  # b_{+2}
                        0.0000,  # b_{+3}
                        0.0000,  # b_{+4}
                        0.0000,  # b_{+5}
                        0.0000,  # b_{+6}
                        0.0000,  # b_{+7}
                        0.0000,  # b_{+8}
                        0.0000,  # b_{+9}
                        0.0000   # b_{+10}
                    ], np.float
                ),
                threshold=1.0
            )
        elif algo == 'majority':
            ck = _cole_kripke(
                ts_days,
                scale=0.00001,
                window=np.array(
                    [400, 600, 300, 400, 1400, 500, 350, 0, 0],
                    np.int32
                ),
                threshold=1.0
            )
            sadeh = _sadeh(
                ts_days,
                offset=7.601,
                weights=np.array([-0.065, -1.08, -0.056, -0.703], np.float),
                threshold=0.0
            )
            scripps = _scripps(
                ts_days,
                scale=0.204,
                window=np.array(
                    [
                        0.0064,  # b_{-10}
                        0.0074,  # b_{-9}
                        0.0112,  # b_{-8}
                        0.0112,  # b_{-7}
                        0.0118,  # b_{-6}
                        0.0118,  # b_{-5}
                        0.0128,  # b_{-4}
                        0.0188,  # b_{-3}
                        0.0280,  # b_{-2}
                        0.0664,  # b_{-1}
                        0.0300,  # b_{+0}
                        0.0112,  # b_{+1}
                        0.0100,  # b_{+2}
                        0.0000,  # b_{+3}
                        0.0000,  # b_{+4}
                        0.0000,  # b_{+5}
                        0.0000,  # b_{+6}
                        0.0000,  # b_{+7}
                        0.0000,  # b_{+8}
                        0.0000,  # b_{+9}
                        0.0000   # b_{+10}
                    ],
                    np.float
                ),
                threshold=1.0
            )
            sod = ((ck + sadeh + scripps) > 1).astype(int)
        elif algo == 'unanimous':
            ck = _cole_kripke(
                ts_days,
                scale=0.00001,
                window=np.array(
                    [400, 600, 300, 400, 1400, 500, 350, 0, 0],
                    np.int32
                ),
                threshold=1.0
            )
            sadeh = _sadeh(
                ts_days,
                offset=7.601,
                weights=np.array([-0.065, -1.08, -0.056, -0.703], np.float),
                threshold=0.0
            )
            scripps = _scripps(
                ts_days,
                scale=0.204,
                window=np.array(
                    [
                        0.0064,  # b_{-10}
                        0.0074,  # b_{-9}
                        0.0112,  # b_{-8}
                        0.0112,  # b_{-7}
                        0.0118,  # b_{-6}
                        0.0118,  # b_{-5}
                        0.0128,  # b_{-4}
                        0.0188,  # b_{-3}
                        0.0280,  # b_{-2}
                        0.0664,  # b_{-1}
                        0.0300,  # b_{+0}
                        0.0112,  # b_{+1}
                        0.0100,  # b_{+2}
                        0.0000,  # b_{+3}
                        0.0000,  # b_{+4}
                        0.0000,  # b_{+5}
                        0.0000,  # b_{+6}
                        0.0000,  # b_{+7}
                        0.0000,  # b_{+8}
                        0.0000,  # b_{+9}
                        0.0000   # b_{+10}
                    ],
                    np.float
                ),
                threshold=1.0
            )
            sod = ((ck * sadeh * scripps) > 0).astype(int)

        return sod

    def fSoD(
        self,
        freq='5min',
        whs=12,
        start='12:00:00',
        period='5h',
        algo='unanimous'
    ):

        r"""Fraction of Sleep over Daytime

        Fractional volume of epochs identified as sleep over daytime (SoD),
        using sleep-wake scoring algorithms.

        Parameters
        ----------
        freq: str, optional
            Resampling frequency.
            Default is '5min'
        whs: int, optional
            Window half size.
            Default is 4
        start: str, optional
            Start time of the period of interest.
            Default: '12:00:00'
            Supported times: 'AonT', 'AoffT', any 'HH:MM:SS'
        period: str, optional
            Period length.
            Default is '10h'
        algo: str, optional
            Sleep scoring algorithm to use.
            Default is 'unanimous'.
            Supported algorithms: 'ck', 'sadeh', 'scripps', 'majority',
            'unanimous'

        Returns
        -------
        fsod: float
            Fraction of epochs scored as sleep, relatively to the length of
            the specified period.

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.fSoD()
            0.17763779527559054
            >>> raw.fSoD(algo='ck')
            0.23811023622047245

        """

        SoD = self.SoD(
            freq=freq, whs=whs, start=start, period=period, algo=algo
        )

        return SoD.sum()/len(SoD)

    def SleepFragmentation(self):
        """Sleep Fragmentation is an index of restlessness during the sleep
        period expressed as a percentage. The higher the index, the more sleep
        is disrupted. ActiLife calculates three values for sleep fragmentation:
        movement Index, a fragmentation index, and total sleep fragmentation
        Index.
        - The Movement Index (MI) is  the percentage of epochs with y-axis
        counts greater than zero in the sleep period.
        - The Fragmentation Index (FI) is the percentage of one minute periods
        of sleep vs. all periods of sleep during the sleep period.
        - The Total Sleep Fragmentation Index (SFI) is the sum of the MI and
        the FI"""
        pass

    def Crespo(
        self,
        zeta=15, zeta_r=30, zeta_a=2,
        t=.33, alpha='8h', beta='1h',
        estimate_zeta=False, seq_length_max=100,
        verbose=False
    ):

        r"""Crespo algorithm for activity/rest identification

        Algorithm for automatic identification of activity-rest periods based
        on actigraphy, developped by Crespo et al. [1]_.

        Parameters
        ----------
        zeta: int, optional
            Maximum number of consecutive zeroes considered valid.
            Default is 15.
        zeta_r: int, optional
            Maximum number of consecutive zeroes considered valid (rest).
            Default is 30.
        zeta_a: int, optional
            Maximum number of consecutive zeroes considered valid (active).
            Default is 2.
        t: float, optional
            Percentile for invalid zeroes.
            Default is 0.33.
        alpha: str, optional
            Average hours of sleep per night.
            Default is '8h'.
        beta: str, optional
            Length of the padding sequence used during the processing.
            Default is '1h'.
        estimate_zeta: bool, optional
            If set to True, zeta values are estimated from the distribution of
            ratios of the number of series of consecutive zeroes to
            the number of series randomly chosen from the actigraphy data.
            Default is False.
        seq_length_max: int, optional
            Maximal length of the aforementioned random series.
            Default is 100.
        verbose: bool, optional
            If set to True, print the estimated values of zeta.
            Default is False.

        Returns
        -------
        crespo : pandas.core.Series
            Time series containing the estimated periods of rest (0) and
            activity (1).

        References
        ----------

        .. [1] Crespo, C., Aboy, M., Fernández, J. R., & Mojón, A. (2012).
               Automatic identification of activity–rest periods based on
               actigraphy. Medical & Biological Engineering & Computing, 50(4),
               329–340. http://doi.org/10.1007/s11517-012-0875-y

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> crespo = rawAWD.Crespo()
            >>> crespo
            2018-03-26 14:16:00    1
            2018-03-26 14:17:00    0
            2018-03-26 14:18:00    0
            (...)
            2018-04-06 08:22:00    0
            2018-04-06 08:23:00    0
            2018-04-06 08:24:00    1
            Length: 15489, dtype: int64
        """

        # 1. Pre-processing
        # This stage produces an initial estimate of the rest-activity periods

        # 1.1. Signal conditioning based on empirical probability model
        # This step replaces sequences of more than $\zeta$ "zeroes"
        # with the t-percentile value of the actigraphy data
        # zeta = 15
        if estimate_zeta:
            zeta = _estimate_zeta(self.data, seq_length_max)
            if verbose:
                print("CRESPO: estimated zeta = {}".format(zeta))
        # Determine the sequences of consecutive zeroes
        mask_zeta = _create_inactivity_mask(self.data, zeta, 1)

        # Determine the user-specified t-percentile value
        s_t = self.data.quantile(t)

        # Replace zeroes with the t-percentile value
        x = self.data.copy()
        x[mask_zeta > 0] = s_t

        # Median filter window length L_w
        L_w = int(pd.Timedelta(alpha)/self.frequency)+1
        L_w_over_2 = int((L_w-1)/2)

        # Pad the signal at the beginning and at the end with a sequence of
        # $\alpha/2$ h of elements of value $m = max(s(t))$.
        #
        # alpha_epochs = int(pd.Timedelta(alpha)/self.frequency)
        # alpha_epochs_half = int(alpha_epochs/2)
        # beta_epochs = int(pd.Timedelta(beta)/self.frequency)

        s_t_max = self.data.max()

        x_p = _padded_data(
            self.data, s_t_max, L_w_over_2, self.frequency
        )

        # 1.2 Rank-order processing and decision logic
        # Apply a median filter to the $x_p$ series
        x_f = x_p.rolling(L_w, center=True).median()

        # Rank-order thresholding
        # Create a series $y_1(n)$ where $y_1(n) = 1$ for $x_f(n)>p$, $0$ otw.
        # The threshold $p$ is the percentile of $x_f(n)$ corresponding to
        # $(h_s/24)\times 100\%$

        p_threshold = x_f.quantile((pd.Timedelta(alpha)/pd.Timedelta('24h')))
        y_1 = pd.Series(np.where(x_f > p_threshold, 1, 0), index=x_f.index)

        # 1.3 Morphological filtering

        # Morph. filter window length, L_p
        L_p = int(pd.Timedelta(beta)/self.frequency)+1

        # Morph. filter, M_f
        M_f = np.ones(L_p)

        # Apply a morphological closing operation

        y_1_close = binary_closing(y_1, M_f).astype(int)

        # Apply a morphological opening operation

        y_1_close_and_open = binary_opening(y_1_close, M_f).astype(int)

        y_e = pd.Series(y_1_close_and_open, index=y_1.index)

        # 2. Processing and decision logic
        # This stage uses the estimates of the rest-activity periods
        # from the previous stage.

        # 2.1 Model-based data validation

        # Create a mask for sequences of more than $\zeta_{rest}$ zeros
        # during the rest periods

        # zeta_r = 30
        # zeta_a = 2
        if estimate_zeta:
            zeta_r = _estimate_zeta(self.data[y_e < 1], seq_length_max)
            zeta_a = _estimate_zeta(self.data[y_e > 0], seq_length_max)
            if verbose:
                print("CRESPO: estimated zeta@rest= {}".format(zeta_r))
                print("CRESPO: estimated zeta@actv= {}".format(zeta_a))

        # Find sequences of zeroes during the rest and the active periods
        # and mark as invalid sequences of more $\zeta_x$ zeroes.

        # Use y_e series as a filter for the rest periods
        mask_rest = _create_inactivity_mask(
            self.data[y_e < 1], zeta_r, 1
        )

        # Use y_e series as a filter for the active periods
        mask_actv = _create_inactivity_mask(
            self.data[y_e > 0], zeta_a, 1
        )

        mask = pd.concat([mask_actv, mask_rest], verify_integrity=True)

        # 2.2 Adaptative rank-order processing and decision logic

        # Replace masked values by NaN so that they are not taken into account
        # by the median filter.
        # Done before padding to avoid unaligned time series.

        x_nan = self.data.copy()
        x_nan[mask < 1] = np.NaN

        # Pad the signal at the beginning and at the end with a sequence of 1h
        # of elements of value m = max(s(t)).
        x_sp = _padded_data(
            x_nan, s_t_max,
            L_p-1,
            self.frequency
        )

        # Apply an adaptative median filter to the $x_{sp}$ series

        # no need to use a time-aware window as there is no time gap
        # in this time series by definition.
        x_fa = x_sp.rolling(L_w, center=True, min_periods=L_p-1).median()

        # The 'alpha' hour window is biased at the edges as it is not
        # symmetrical anymore. In the regions (start, start+alpha/2,
        # the median needs to be calculate by hand.
        # The range is start, start+alpha as the window is centered.
        median_start = x_sp.iloc[0:L_w].expanding(
                center=True
            ).median()
        median_end = x_sp.iloc[-L_w-1:-1].sort_index(
                ascending=False
            ).expanding(center=True).median()[::-1]

        # replace values in the original x_fa series with the new values
        # within the range (start, start+alpha/2) only.
        x_fa.iloc[0:L_w_over_2] = median_start.iloc[0:L_w_over_2]
        x_fa.iloc[-L_w_over_2-1:-1] = median_end.iloc[0:L_w_over_2]

        # restore original time range
        x_fa = x_fa[self.data.index[0]:self.data.index[-1]]

        p_threshold = x_fa.quantile((pd.Timedelta(alpha)/pd.Timedelta('24h')))

        y_2 = pd.Series(np.where(x_fa > p_threshold, 1, 0), index=x_fa.index)

        # ### 2.3 Morphological filtering
        y_2_close = binary_closing(
            y_2,
            structure=np.ones(2*(L_p-1)+1)
        ).astype(int)

        y_2_open = binary_opening(
            y_2_close,
            structure=np.ones(2*(L_p-1)+1)
        ).astype(int)

        crespo = pd.Series(
            y_2_open,
            index=y_2.index)

        # Manual post-processing
        crespo.iloc[0] = 1
        crespo.iloc[-1] = 1

        return crespo

    def Crespo_AoT(
        self,
        zeta=15, zeta_r=30, zeta_a=2,
        t=.33, alpha='8h', beta='1h',
        estimate_zeta=False, seq_length_max=100,
        verbose=False
    ):

        """Automatic identification of activity onset/offset times, based on
        the Crespo algorithm.

        Identification of the activity onset and offset time using the
        algorithm for automatic identification of activity-rest periods based
        on actigraphy, developped by Crespo et al. [1]_.

        Parameters
        ----------
        zeta: int
            Maximum number of consecutive zeroes considered valid.
            Default is 15.
        zeta_r: int
            Maximum number of consecutive zeroes considered valid (rest).
            Default is 30.
        zeta_a: int
            Maximum number of consecutive zeroes considered valid (active).
            Default is 2.
        t: float
            Percentile for invalid zeroes.
            Default is 0.33.
        alpha: offset
            Average hours of sleep per night.
            Default is '8h'.
        beta: offset
            Length of the padding sequence used during the processing.
            Default is '1h'.
        estimate_zeta: Boolean
            If set to True, zeta values are estimated from the distribution of
            ratios of the number of series of consecutive zeroes to
            the number of series randomly chosen from the actigraphy data.
            Default is False.
        seq_length_max: int
            Maximal length of the aforementioned random series.
            Default is 100.
        verbose:
            If set to True, print the estimated values of zeta.
            Default is False.

        Returns
        -------
        aot : (ndarray, ndarray)
            Arrays containing the estimated activity onset and offset times,
            respectively.

        References
        ----------

        .. [1] Crespo, C., Aboy, M., Fernández, J. R., & Mojón, A. (2012).
               Automatic identification of activity–rest periods based on
               actigraphy. Medical & Biological Engineering & Computing, 50(4),
               329–340. http://doi.org/10.1007/s11517-012-0875-y

        Examples
        --------

        """

        crespo = self.Crespo(
            zeta=zeta, zeta_r=zeta_r, zeta_a=zeta_a,
            t=t, alpha=alpha, beta=beta,
            estimate_zeta=estimate_zeta, seq_length_max=seq_length_max,
            verbose=verbose
        )

        AonT = crespo[crespo.diff(1) == 1].index
        AoffT = crespo[crespo.diff(1) == -1].index

        return (AonT, AoffT)
