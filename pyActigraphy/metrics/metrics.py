import pandas as pd
import numpy as np
# from functools import lru_cache
# from ..sleep import _td_format
from statistics import mean
import statsmodels.api as sm


def _average_daily_activity(data, cyclic=False):
    """Calculate the average daily activity distribution"""

    avgdaily = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ]).mean()

    if cyclic:
        avgdaily = pd.concat([avgdaily, avgdaily])
        index = pd.timedelta_range(
            start='0 day',
            end='2 days',
            freq=data.index.freq,
            closed='left'
        )
    else:
        index = pd.timedelta_range(
            start='0 day',
            end='1 day',
            freq=data.index.freq,
            closed='left'
        )

    avgdaily.index = index

    return avgdaily


def _average_daily_total_activity(data):

    return data.resample('1D').sum().mean()


def _interdaily_stability(data):
    """Calculate the interdaily stability as defined in
    [Eus J. W. Van Someren, Dick F. Swaab, Christopher C. Colenda, Wayne Cohen,
    W. Vaughn McCall & Peter B. Rosenquist (1999) Bright Light Therapy:
    Improved Sensitivity to Its Effects on Rest-Activity Rhythms in Alzheimer
    Patients by Application of Nonparametric Methods,
    Chronobiology International, 16:4, 505-518, DOI: 10.3109/07420529908998724]

    ## Definition of the Interdaily stability (IS):

    \begin{equation*}
    IS = \frac{d^{24h}}{d^{1h}}
    \end{equation*}

    with:

    \begin{equation*}
    d^{1h}=\sum_{i}^{n}\frac{\left( x_{i}-\bar{x}\right)^{2}}{n}
    \end{equation*}

    where $x_{i}$ is the number of active (counts higher than a predefined
    threshold) minutes during the $i^{th}$ period, $\bar{x}$ is the mean of all
    data and $n$ is the number of periods covered by the actigraphy data,

    and with:

    \begin{equation*}
    d^{24h}=\sum_{i}^{p}\frac{\left( \bar{x}_{h,i}-\bar{x}\right)^{2}}{p}
    \end{equation*}

    where $\bar{x}^{h,i}$ is the average number of active minutes over
    the $i^{th}$ period and p is the number of periods per day.
    The average runs over all the days.


    """
    # resampled_data = data.resample(freq).sum()

    d_24h = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second]
    ).mean().var()

    d_1h = data.var()

    return (d_24h / d_1h)


def _intradaily_variability(data):
    """Calculate the intradaily variability as defined in
    [Eus J. W. Van Someren, Dick F. Swaab, Christopher C. Colenda, Wayne Cohen,
    W. Vaughn McCall & Peter B. Rosenquist (1999) Bright Light Therapy:
    Improved Sensitivity to Its Effects on Rest-Activity Rhythms in Alzheimer
    Patients by Application of Nonparametric Methods,
    Chronobiology International, 16:4, 505-518, DOI: 10.3109/07420529908998724]

    ## Definition of the Intradaily variability (IV):

    \begin{equation*}
    IV = \frac{c^{1h}}{d^{1h}}
    \end{equation*}

    with:

    \begin{equation*}
    d^{1h}=\sum_{i}^{n}\frac{\left( x_{i}-\bar{x}\right)^{2}}{n}
    \end{equation*}

    where $x_{i}$ is the number of active (counts higher than a predefined
    threshold) minutes during the $i^{th}$ period, $\bar{x}$ is the mean of all
    data and $n$ is the number of periods covered by the actigraphy data,

    and with:

    \begin{equation*}
    c^{1h}=\sum_{i}^{n-1}\frac{\left( x_{i+1}-x_{i}\right)^{2}}{n-1}
    \end{equation*}.

    """
    c_1h = data.diff(1).pow(2).mean()

    d_1h = data.var()

    return (c_1h / d_1h)


def _lmx(data, epochs, lowest=True):
    """Calculate the start time and mean activity of the period of
    lowest/highest activity"""

    avgdaily = _average_daily_activity(data=data, cyclic=True)

    mean_activity = avgdaily.rolling(epochs).sum().shift(-epochs+1)

    if lowest:
        t_start = mean_activity.idxmin()
    else:
        t_start = mean_activity.idxmax()

    lmx = mean_activity[t_start]/epochs
    return t_start, lmx


def _interval_maker(index, period, verbose):
    """ """
    # TODO: test if period is a valid string

    (num_periods, td) = divmod(
        (index[-1] - index[0]), pd.Timedelta(period)
    )
    if verbose:
        print("Number of periods: {0}\n Time unaccounted for: {1}".format(
            num_periods,
            '{} days, {}h, {}m, {}s'.format(
                td.days,
                td.seconds//3600,
                (td.seconds//60) % 60,
                td.seconds % 60
            )
        ))

    intervals = [(
        index[0] + (i)*pd.Timedelta(period),
        index[0] + (i+1)*pd.Timedelta(period))
        for i in range(0, num_periods)
    ]

    return intervals


def _count_consecutive_values(data):
    """ Create a count list for identical consecutive numbers
    together with a state for each series:
     - 1 if the sum of the consecutive series numbers is positive
     - 0 otherwise
    """

    consecutive_values = data.groupby(
        # create identical 'labels' for identical consecutive numbers
        [data.diff().ne(0).cumsum()]
    ).agg(['count', lambda x: (np.sum(x) > 0).astype(int)])
    # rename columns
    consecutive_values.columns = ['counts', 'state']

    return consecutive_values


def _count_consecutive_zeros(data):
    ccz = _count_consecutive_values(data)
    ccz['end'] = ccz['counts'].cumsum()
    ccz['start'] = ccz['end'].shift(1).fillna(0).astype(int)
    return ccz[ccz['state'] < 1]


def _transition_prob(data, from_zero_to_one):

    # Create a list of consecutive sequence of active/rest epochs
    ccv = _count_consecutive_values(data)
    # filter out sequences of active epochs
    if from_zero_to_one is True:
        bouts = ccv[ccv['state'] < 1]['counts']
    else:
        bouts = ccv[ccv['state'] > 0]['counts']
    # Count the number of sequences of length N for N=1...Nmax
    Nt = bouts.groupby(bouts).count()
    # Create its reverse cumulative sum so that Nt at index t is equal to
    # the number of sequences of lengths t or longer.
    Nt = np.cumsum(Nt[::-1])[::-1]
    # Rest->Activity (or Activity->Rest) transition probability at time t,
    # defined as the number of sequences for which R->A at time t+1 / Nt
    prob = Nt.diff(-1)/Nt
    # Correct pRA for discontinuities due to sparse data
    prob = prob.dropna() / np.diff(prob.index.tolist())
    # Define the weights as the square root of the number of runs
    # contributing to each probability estimate
    prob_weights = np.sqrt(Nt+Nt.shift(-1)).dropna()

    return prob, prob_weights


def _transition_prob_sustain_region(prob, prob_weights, frac=.3, it=0):

    # Fit the 'prob' distribution with a LOWESS
    lowess = sm.nonparametric.lowess(
        prob.values, prob.index, return_sorted=False, frac=frac, it=it
    )

    # Calculate the pRA std
    std = prob.std()

    # Check which residuals are below 1 sigma
    prob_residuals_below_one_std = _count_consecutive_values(
        ((prob-lowess).abs() < std).astype(int)
    )

    # Find the index of the longest series of consecutive values below 1 SD
    index = prob_residuals_below_one_std[
        prob_residuals_below_one_std['state'] > 0
    ]['counts'].idxmax()-1

    # Calculate the cumulative sum of the indices of series of consecutive
    # values of residuals below 1 SD in order to find the number of points
    # before the "index".
    prob_cumsum = prob_residuals_below_one_std['counts'].cumsum()

    # Calculate the start and end indices
    if index < prob_cumsum.index.min():
        start_index = 0
    else:
        start_index = prob_cumsum[index]
    # start_index = prob_cumsum[index]+1
    end_index = prob_cumsum[index+1]

    kProb = np.average(
        prob[start_index:end_index],
        weights=prob_weights[start_index:end_index]
    )
    return kProb


def _td_format(td):
    return '{:02}:{:02}:{:02}'.format(
        td.components.hours,
        td.components.minutes,
        td.components.seconds
    )


class MetricsMixin(object):
    """ Mixin Class """

    def average_daily_activity(
        self, freq=None, cyclic=False, binarize=True, threshold=4
    ):

        data = self.resampled_data(freq, binarize, threshold)

        avgdaily = _average_daily_activity(data, cyclic=cyclic)

        return avgdaily

    def average_daily_light(self, freq=None, cyclic=False):
        """Average daily light (in lux)"""

        light = self.resampled_light(freq)

        avgdaily_light = _average_daily_activity(light, cyclic=cyclic)

        return avgdaily_light

    def ADAT(self, binarize=True, threshold=4):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        adat = _average_daily_total_activity(data)

        return adat

    def ADATp(self, period='7D', binarize=True, threshold=4, verbose=False):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        intervals = _interval_maker(data.index, period, verbose)

        results = [
            _average_daily_total_activity(
                data[time[0]:time[1]]
            ) for time in intervals
        ]

        return results

    def L5(self, binarize=True, threshold=4):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        n_epochs = int(pd.Timedelta('5H')/self.frequency)

        _, l5 = _lmx(data, n_epochs, lowest=True)

        return l5

    def M10(self, binarize=True, threshold=4):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        n_epochs = int(pd.Timedelta('10H')/self.frequency)

        _, m10 = _lmx(data, n_epochs, lowest=False)

        return m10

    def RA(self, binarize=True, threshold=4):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        n_epochs = int(pd.Timedelta('5H')/self.frequency)

        _, l5 = _lmx(data, n_epochs, lowest=True)
        _, m10 = _lmx(data, n_epochs*2, lowest=False)

        return (m10-l5)/(m10+l5)

    def L5p(self, period='7D', binarize=True, threshold=4, verbose=False):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        n_epochs = int(pd.Timedelta('5H')/self.frequency)

        intervals = _interval_maker(data.index, period, verbose)

        results = [
            _lmx(
                data[time[0]:time[1]],
                n_epochs,
                lowest=True
            ) for time in intervals
        ]
        return [res[1] for res in results]

    def M10p(self, period='7D', binarize=True, threshold=4, verbose=False):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        n_epochs = int(pd.Timedelta('10H')/self.frequency)

        intervals = _interval_maker(data.index, period, verbose)

        results = [
            _lmx(
                data[time[0]:time[1]],
                n_epochs,
                lowest=False
            ) for time in intervals
        ]
        return [res[1] for res in results]

    def RAp(self, period='7D', binarize=True, threshold=4, verbose=False):

        if binarize is True:
            data = self.binarized_data(threshold)
        else:
            data = self.data

        n_epochs = int(pd.Timedelta('5H')/self.frequency)

        intervals = _interval_maker(data.index, period, verbose)

        results = []

        for time in intervals:
            data_subset = data[time[0]:time[1]]
            _, l5 = _lmx(data_subset, n_epochs, lowest=True)
            _, m10 = _lmx(data_subset, n_epochs*2, lowest=False)
            results.append((m10-l5)/(m10+l5))

        return results

    # @lru_cache(maxsize=6)
    def IS(self, freq='1H', binarize=True, threshold=4):

        data = self.resampled_data(
            freq=freq,
            binarize=binarize,
            threshold=threshold
        )
        return _interdaily_stability(data)

    def ISm(
        self,
        freqs=[
            '1T', '2T', '3T', '4T', '5T', '6T', '8T', '9T', '10T',
            '12T', '15T', '16T', '18T', '20T', '24T', '30T',
            '32T', '36T', '40T', '45T', '48T', '60T'
        ],
        binarize=True,
        threshold=4
    ):

        data = [
            self.resampled_data(freq, binarize, threshold) for freq in freqs
        ]

        return mean([_interdaily_stability(datum) for datum in data])

    def ISp(self, period='7D', freq='1H',
            binarize=True, threshold=4, verbose=False):

        data = self.resampled_data(freq, binarize, threshold)

        intervals = _interval_maker(data.index, period, verbose)

        results = [
            _interdaily_stability(data[time[0]:time[1]]) for time in intervals
        ]
        return results

    # @lru_cache(maxsize=6)
    def IV(self, freq='1H', binarize=True, threshold=4):

        data = self.resampled_data(freq, binarize, threshold)

        return _intradaily_variability(data)

    def IVm(
        self,
        freqs=[
            '1T', '2T', '3T', '4T', '5T', '6T', '8T', '9T', '10T',
            '12T', '15T', '16T', '18T', '20T', '24T', '30T',
            '32T', '36T', '40T', '45T', '48T', '60T'
        ],
        binarize=True,
        threshold=4
    ):

        data = [
            self.resampled_data(freq, binarize, threshold) for freq in freqs
        ]

        return mean([_intradaily_variability(datum) for datum in data])

    def IVp(self, period='7D', freq='1H',
            binarize=True, threshold=4, verbose=False):

        data = self.resampled_data(freq, binarize, threshold)

        intervals = _interval_maker(data.index, period, verbose)

        results = [
            _intradaily_variability(data[time[0]:time[1]])
            for time in intervals
        ]
        return results

    def pRA(self, threshold, start=None, period=None):

        # Restrict data range to period 'Start, Start+Period'
        if start is not None:
            end = _td_format(
                pd.Timedelta(start)+pd.Timedelta(period)
            )

            data = self.binarized_data(
                threshold
            ).between_time(start, end)
        else:
            data = self.binarized_data(threshold)
        # Rest->Activity transition probability:
        pRA, pRA_weights = _transition_prob(
            data, True
        )

        return pRA, pRA_weights

    def pAR(self, threshold, start=None, period=None):

        # Restrict data range to period 'Start, Start+Period'
        if start is not None:
            end = _td_format(
                pd.Timedelta(start)+pd.Timedelta(period)
            )

            data = self.binarized_data(
                threshold
            ).between_time(start, end)
        else:
            data = self.binarized_data(threshold)
        # Activity->Rest transition probability:
        pAR, pAR_weights = _transition_prob(
            data, False
        )

        return pAR, pAR_weights

    def kRA(self, threshold, start=None, period=None, frac=.3, it=0):

        # Calculate the pRA probabilities and their weights.
        pRA, pRA_weights = self.pRA(threshold, start=start, period=period)
        # Fit the pRA distribution with a LOWESS and return mean value for
        # the constant region (i.e. the region where |pRA-lowess|<1SD)
        kRA = _transition_prob_sustain_region(
            pRA,
            pRA_weights,
            frac=frac,
            it=it
            )
        return kRA

    def kAR(self, threshold, start=None, period=None, frac=.3, it=0):

        # Calculate the pAR probabilities and their weights.
        pAR, pAR_weights = self.pAR(threshold, start=start, period=period)
        # Fit the pAR distribution with a LOWESS and return mean value for
        # the constant region (i.e. the region where |pAR-lowess|<1SD)
        kAR = _transition_prob_sustain_region(
            pAR,
            pAR_weights,
            frac=frac,
            it=it
            )
        return kAR


class ForwardMetricsMixin(object):
    """ Mixin Class """

    def mask_fraction(self):

        return {
            iread.display_name: iread.mask_fraction() for iread in self.readers
        }

    def start_time(self):

        return {
            iread.display_name: str(iread.start_time) for iread in self.readers
        }

    def duration(self):

        return {
            iread.display_name: str(iread.duration()) for iread in self.readers
        }

    def ADAT(self, binarize=True, threshold=4):

        return {
            iread.display_name: iread.ADAT(
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def ADATp(self, period='7D', binarize=True, threshold=4, verbose=False):

        return {
            iread.display_name: iread.ADATp(
                period=period,
                binarize=binarize,
                threshold=threshold,
                verbose=verbose
            ) for iread in self.readers
        }

    def L5(self, binarize=True, threshold=4):

        return {
            iread.display_name: iread.L5(
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def M10(self, binarize=True, threshold=4):

        return {
            iread.display_name: iread.M10(
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def RA(self, binarize=True, threshold=4):

        return {
            iread.display_name: iread.RA(
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def L5p(self, period='7D', binarize=True, threshold=4, verbose=False):

        return {
            iread.display_name: iread.L5p(
                period=period,
                binarize=binarize,
                threshold=threshold,
                verbose=verbose
            ) for iread in self.readers
        }

    def M10p(self, period='7D', binarize=True, threshold=4, verbose=False):

        return {
            iread.display_name: iread.M10p(
                period=period,
                binarize=binarize,
                threshold=threshold,
                verbose=verbose
            ) for iread in self.readers
        }

    def RAp(self, period='7D', binarize=True, threshold=4, verbose=False):

        return {
            iread.display_name: iread.RAp(
                period=period,
                binarize=binarize,
                threshold=threshold,
                verbose=verbose
            ) for iread in self.readers
        }

    def IS(self, freq='1H', binarize=True, threshold=4):

        return {
            iread.display_name: iread.IS(
                freq=freq,
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def ISm(
        self,
        freqs=[
            '1T', '2T', '3T', '4T', '5T', '6T', '8T', '9T', '10T',
            '12T', '15T', '16T', '18T', '20T', '24T', '30T',
            '32T', '36T', '40T', '45T', '48T', '60T'
        ],
        binarize=True,
        threshold=4
    ):

        return {
            iread.display_name: iread.ISm(
                freqs=freqs,
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def ISp(self, period='7D', freq='1H',
            binarize=True, threshold=4, verbose=False):

        return {
            iread.display_name: iread.ISp(
                period=period,
                freq=freq,
                binarize=binarize,
                threshold=threshold,
                verbose=verbose
            ) for iread in self.readers
        }

    def IV(self, freq='1H', binarize=True, threshold=4):

        return {
            iread.display_name: iread.IV(
                freq=freq,
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def IVm(
        self,
        freqs=[
            '1T', '2T', '3T', '4T', '5T', '6T', '8T', '9T', '10T',
            '12T', '15T', '16T', '18T', '20T', '24T', '30T',
            '32T', '36T', '40T', '45T', '48T', '60T'
        ],
        binarize=True,
        threshold=4
    ):

        return {
            iread.display_name: iread.IVm(
                freqs=freqs,
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def IVp(self, period='7D', freq='1H',
            binarize=True, threshold=4, verbose=False):

        return {
            iread.display_name: iread.IVp(
                period=period,
                freq=freq,
                binarize=binarize,
                threshold=threshold,
                verbose=verbose
            ) for iread in self.readers
        }

    def kRA(self, threshold=4, start=None, period=None, frac=.3, it=0):

        return {
            iread.display_name: iread.kRA(
                threshold=threshold,
                start=start,
                period=period,
                frac=frac,
                it=it
            ) for iread in self.readers
        }

    def kAR(self, threshold=4, start=None, period=None, frac=.3, it=0):

        return {
            iread.display_name: iread.kAR(
                threshold=threshold,
                start=start,
                period=period,
                frac=frac,
                it=it
            ) for iread in self.readers
        }

    def AonT(self, freq='5min', whs=12, binarize=True, threshold=4):

        return {
            iread.display_name: iread.AonT(
                freq=freq,
                whs=whs,
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def AoffT(self, freq='5min', whs=12, binarize=True, threshold=4):

        return {
            iread.display_name: iread.AoffT(
                freq=freq,
                whs=whs,
                binarize=binarize,
                threshold=threshold
            ) for iread in self.readers
        }

    def fSoD(
        self,
        freq='5min',
        whs=12,
        start='12:00:00',
        period='5h',
        algo='unanimous'
    ):

        return {
            iread.display_name: iread.fSoD(
                freq=freq,
                whs=whs,
                period=period,
                algo=algo
            ) for iread in self.readers
        }

    def daily_light_average(self):

        return {
            iread.display_name:
            iread.daily_light_average() for iread in self.readers
        }

    def Summary(self, mask_inactivity=True):

        # set inactivity mask
        for iread in self.readers:
            iread.mask_inactivity = mask_inactivity

        # dict of dictionnaries
        ldic = {}
        ldic['Start_time'] = self.start_time()
        ldic['Mask_fraction'] = self.mask_fraction()
        ldic['Duration'] = self.duration()
        ldic['ADAT'] = self.ADAT()
        ldic['ADATp'] = self.ADATp()
        ldic['L5'] = self.L5()
        ldic['M10'] = self.M10()
        ldic['RA'] = self.RA()
        ldic['L5p'] = self.L5p()
        ldic['M10p'] = self.M10p()
        ldic['RAp'] = self.RAp()
        ldic['IS'] = self.IS()
        ldic['IV'] = self.IV()
        ldic['ISm'] = self.ISm()
        ldic['IVm'] = self.IVm()
        ldic['ISp'] = self.ISp()
        ldic['IVp'] = self.IVp()
        ldic['kRA(night)'] = self.kRA(start='00:00:00', period='5h')
        ldic['kAR(Mid-day)'] = self.kAR(start='12:00:00', period='5h')
        ldic['AonT'] = self.AonT()
        ldic['AoffT'] = self.AoffT()
        ldic['fSoD(Mid-day)'] = self.fSoD()
        if self.reader_type == 'RPX':
            ldic['daily_light_average'] = self.daily_light_average()

        # list keys of dictionnaries whose number of columns is variable:
        var_dic = ['ADATp', 'L5p', 'M10p', 'RAp', 'ISp', 'IVp']

        # list of corresponding dataframes
        dfs = []
        for key, value in ldic.items():
            columns = []
            if key in var_dic:
                # Get max length of value arrays
                max_length = np.max([len(x) for x in list(value.values())])
                for i in range(max_length):
                    columns.append(
                        key+'(duration={0},period={1})'.format('7D', i+1)
                    )
            else:
                columns.append(key)

            df = pd.DataFrame(
                list(value.values()),
                index=value.keys(),
                columns=columns
            )
            dfs.append(df)

        # join the dataframes recursively
        from functools import reduce
        df = reduce((lambda x, y: x.join(y)), dfs)
        return df
