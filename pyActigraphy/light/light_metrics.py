#############################################################################
# Copyright (c) 2022, DLA
# Author: Grégory Hammad
# Owner: Daylight Academy (https://daylight.academy)
# Maintainer: Grégory Hammad
# Email: gregory.hammad@uliege.be
# Status: developpement
#############################################################################
# The development of the light module has been initially funded by the
# Daylight Academy under the supervision of Prof. Mirjam Münch and
# Prof. Manuel Spitschan.
# This module is part of the pyActigraphy software.
# pyActigraphy is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# pyActigraphy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
############################################################################
import pandas as pd
import re
from ..utils.utils import _average_daily_activity
from ..utils.utils import _shift_time_axis

__all__ = [
    'LightMetricsMixin',
]


class LightMetricsMixin(object):
    """ Mixin Class """

    def average_daily_profile(
        self,
        channel,
        rsfreq='5min',
        cyclic=False,
        binarize=False,
        threshold=None,
        time_origin=None
    ):
        r"""Average daily light profile

        Calculate the daily profile of light exposure. Data are averaged over
        all the days.

        Parameters
        ----------
        channel: str,
            Channel to be used (i.e column of the input data).
        rsfreq: str, optional
            Data resampling frequency.
            Cf. #timeseries-offset-aliases in
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        cyclic: bool, optional
            If set to True, two daily profiles are concatenated to ensure
            continuity between the last point of the day and the first one.
            Default is False.
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is False.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is None.
        time_origin: str or pd.Timedelta, optional
            If not None, origin of the time axis for the daily profile.
            Original time bins are translated as time delta with respect to
            this new origin.
            Default is None
            Supported time string: 'HH:MM:SS'

        Returns
        -------
        raw : pandas.Series
            A Series containing the daily light profile with a 24h/48h index.
        """
        # Check if requested channel is available
        if channel not in self.data.columns:
            raise ValueError(
                'The light channel you tried to access ({}) '.format(channel)
                + 'is not available.\nAvailable channels:\n-{}'.format(
                    '\n- '.join(self.data.columns)
                )
            )

        # Binarize (+resample) data, if required.
        if binarize:
            data = self.binarized_data(
                threshold,
                rsfreq=rsfreq,
                agg='sum'
            )
        elif rsfreq is not None:
            data = self.resampled_data(rsfreq=rsfreq, agg='sum')
        else:
            data = self.data

        # Select requested channel
        data = data.loc[:, channel]

        if time_origin is None:

            return _average_daily_activity(data, cyclic=cyclic)

        else:
            if cyclic is True:
                raise NotImplementedError(
                    'Setting a time origin while cyclic option is True is not '
                    'implemented yet.'
                )

            avgdaily = _average_daily_activity(data, cyclic=False)

            if isinstance(time_origin, str):
                # Regex pattern for HH:MM:SS time string
                pattern = re.compile(
                    r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$"
                )

                if pattern.match(time_origin):
                    time_origin = pd.Timedelta(time_origin)
                else:
                    raise ValueError(
                        'Time origin format ({}) not supported.\n'.format(
                            time_origin
                        ) + 'Supported format: HH:MM:SS.'
                    )

            elif not isinstance(time_origin, pd.Timedelta):
                raise ValueError(
                    'Time origin is neither a time string with a supported '
                    'format, nor a pd.Timedelta.'
                )

            # Round time origin to the required frequency
            time_origin = time_origin.round(data.index.freq)

            shift = int((pd.Timedelta('12h')-time_origin)/data.index.freq)

            return _shift_time_axis(avgdaily, shift)

    def average_daily_profile_auc(
        self,
        channel=None,
        start_time=None,
        stop_time=None,
        binarize=False,
        threshold=None,
        time_origin=None
    ):
        r"""AUC of the average daily light profile

        Calculate the area under the curve of the daily profile of light
        exposure. Data are averaged over all the days.

        Parameters
        ----------
        channel: str,
            Channel to be used (i.e column of the input data).
        start_time: str, optional
            If not set to None, compute AUC from start time.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, compute AUC until stop time.
            Supported time string: 'HH:MM:SS'
            Default is None.
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is False.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is None.
        time_origin: str or pd.Timedelta, optional
            If not None, origin of the time axis for the daily profile.
            Original time bins are translated as time delta with respect to
            this new origin.
            Default is None
            Supported time string: 'HH:MM:SS'

        Returns
        -------
        auc : float
            Area under the curve.
        """
        # Check if requested channel is available
        if channel not in self.data.columns:
            raise ValueError(
                'The light channel you tried to access ({}) '.format(channel)
                + 'is not available.\nAvailable channels:\n-{}'.format(
                    '\n- '.join(self.data.columns)
                )
            )

        # Binarize (+resample) data, if required.
        if binarize:
            data = self.binarized_data(
                threshold,
                rsfreq=None,
                agg='sum'
            )
        else:
            data = self.data

        # Select requested channel
        data = data.loc[:, channel]

        # Compute average daily profile
        avgdaily = _average_daily_activity(data, cyclic=False)

        if time_origin is not None:

            if isinstance(time_origin, str):
                # Regex pattern for HH:MM:SS time string
                pattern = re.compile(
                    r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$"
                )

                if pattern.match(time_origin):
                    time_origin = pd.Timedelta(time_origin)
                else:
                    raise ValueError(
                        'Time origin format ({}) not supported.\n'.format(
                            time_origin
                        ) + 'Supported format: HH:MM:SS.'
                    )

            elif not isinstance(time_origin, pd.Timedelta):
                raise ValueError(
                    'Time origin is neither a time string with a supported '
                    'format, nor a pd.Timedelta.'
                )

            # Round time origin to the required frequency
            time_origin = time_origin.round(data.index.freq)

            shift = int((pd.Timedelta('12h')-time_origin)/data.index.freq)

            avgdaily = _shift_time_axis(avgdaily, shift)

        # Restrict profile to start/stop times
        if start_time is not None:
            start_time = pd.Timedelta(start_time)
        if stop_time is not None:
            stop_time = pd.Timedelta(stop_time)

        return avgdaily.loc[start_time:stop_time].sum()

    def _light_exposure(self, threshold=None, start_time=None, stop_time=None):
        r"""Light exposure

        Calculate the light exposure level and time

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.

        Returns
        -------
        mask : pandas.DataFrame
            A DataFrame where the original data are set to Nan if below
            threshold and/or outside time window.
        """
        if threshold is not None:
            data_mask = self.data.mask(self.data < threshold)
        else:
            data_mask = self.data

        if start_time is stop_time is None:
            return data_mask
        elif (start_time is None) or (stop_time is None):
            raise ValueError(
                'Both start and stop times have to be specified, if any.'
            )
        else:
            return data_mask.between_time(
                start_time=start_time, end_time=stop_time, include_end=False
            )

    def light_exposure_level(
        self, threshold=None, start_time=None, stop_time=None, agg='mean'
    ):
        r"""Light exposure level

        Calculate the aggregated (mean, median, etc) light exposure level
        per epoch.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        agg: str, optional
            Aggregating function used to summarize exposure levels.
            Available functions: 'mean', 'median', 'std', etc.
            Default is 'mean'.

        Returns
        -------
        levels : pd.Series
            A pandas Series with aggreagted light exposure levels per channel
        """
        light_exposure = self._light_exposure(
            threshold=threshold,
            start_time=start_time,
            stop_time=stop_time
        )

        levels = getattr(light_exposure, agg)

        return levels()

    def summary_statistics_per_time_bin(
        self,
        bins='24h',
        agg_func=['mean', 'median', 'sum', 'std', 'min', 'max']
    ):
        r"""Summary statistics.

        Calculate summary statistics (ex: mean, median, etc) according to a
        user-defined (regular or arbitrary) binning.

        Parameters
        ----------
        bins: str or list of tuples, optional
            If set to a string, bins is used to define a regular binning where
            every bin is of length "bins". Ex: "2h".
            Otherwise, the list of 2-tuples is used to define an arbitrary
            binning. Ex: [('2000-01-01 00:00:00','2000-01-01 11:59:00'),
                          ('2000-01-01 12:00:00','2000-01-02 23:59:00')]
            Default is '24h'.
        agg_func: list, optional
            List of aggregation functions to be used on every bin.
            Default is ['mean', 'median', 'sum', 'std', 'min', 'max'].

        Returns
        -------
        ss : pd.DataFrame
            A pandas DataFrame with summary statistics per channel.
        """
        if isinstance(bins, str):
            summary_stats = self.data.resample(bins).agg(agg_func)
        elif isinstance(bins, list):
            df_col = []
            for idx, (start, end) in enumerate(bins):
                df_bins = self.data.loc[start:end, :].apply(
                    agg_func
                ).pivot_table(columns=agg_func)
                channels = {}
                for ch in df_bins.index:
                    channels[ch] = df_bins.loc[df_bins.index == ch]
                    channels[ch] = channels[ch].rename(
                        index={ch: idx},
                        inplace=False
                    )
                    channels[ch] = channels[ch].loc[:, agg_func]
                df_col.append(
                    pd.concat(
                        channels,
                        axis=1
                    )
                )
            summary_stats = pd.concat(df_col)

        return summary_stats

    def TAT(
        self, threshold=None, start_time=None, stop_time=None, oformat=None
    ):
        r"""Time above light threshold.

        Calculate the total light exposure time above the threshold.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        oformat: str, optional
            Output format. Available formats: 'minute' or 'timedelta'.
            If set to 'minute', the result is in number of minutes.
            If set to 'timedelta', the result is a pd.Timedelta.
            If set to None, the result is in number of epochs.
            Default is None.

        Returns
        -------
        tat : pd.Series
            A pandas Series with aggreagted light exposure levels per channel
        """
        available_formats = [None, 'minute', 'timedelta']
        if oformat not in available_formats:
            raise ValueError(
                'Specified output format ({}) not supported. '.format(oformat)
                + 'Available formats are: {}'.format(str(available_formats))
            )

        light_exposure_counts = self._light_exposure(
            threshold=threshold,
            start_time=start_time,
            stop_time=stop_time
        ).count()

        if oformat == 'minute':
            tat = light_exposure_counts * \
                self.data.index.freq.delta/pd.Timedelta('1min')
        elif oformat == 'timedelta':
            tat = light_exposure_counts * self.data.index.freq.delta
        else:
            tat = light_exposure_counts

        return tat

    def VAT(self, threshold=None):
        r"""Values above light threshold.

        Returns the light exposure values above the threshold.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.

        Returns
        -------
        vat : pd.Series
            A pandas Series with light exposure levels per channel
        """

        return self._light_exposure(
            threshold=threshold,
            start_time=None,
            stop_time=None
        )

    def get_light_extremum(self, extremum):
        r"""Light extremum.

        Return the index and the value of the requested extremum (min or max).

        Parameters
        ----------
        extremum: str
            Name of the extremum.
            Available: 'min' or 'max'.

        Returns
        -------
        ext : pd.DataFrame
            A pandas DataFrame with extremum info per channel.
        """
        extremum_list = ['min', 'max']
        if extremum not in extremum_list:
            raise ValueError(
                'Requested extremum ({}) not available.'.format(extremum)
                + ' Available options are:\n- min\n- max'
            )
        extremum_att = 'idxmax' if extremum == 'max' else 'idxmin'

        extremum_per_ch = []
        for ch in self.data.columns:
            index_ext = getattr(self.data.loc[:, ch], extremum_att)()
            extremum_per_ch.append(
                pd.Series(
                    {
                        'channel': ch,
                        'index': index_ext,
                        'value': self.data.loc[index_ext, ch]
                    }
                )
            )

        return pd.concat(extremum_per_ch, axis=1).T
