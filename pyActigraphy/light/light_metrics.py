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
        channel=None,
        freq='5min',
        cyclic=False,
        binarize=True,
        threshold=4,
        time_origin=None
    ):
        r"""Average daily light profile

        Calculate the daily profile of light exposure. Data are averaged over
        all the days.

        Parameters
        ----------
        freq: str, optional
            Data resampling frequency.
            Cf. #timeseries-offset-aliases in
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        cyclic: bool, optional
            If set to True, two daily profiles are concatenated to ensure
            continuity between the last point of the day and the first one.
            Default is False.
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
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
        data = self.resampled_light(channel, freq, binarize, threshold)

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
                    'Time origin is neither a time string with a supported'
                    'format, nor a pd.Timedelta.'
                )

            # Round time origin to the required frequency
            time_origin = time_origin.round(data.index.freq)

            shift = int((pd.Timedelta('12h')-time_origin)/data.index.freq)

            return _shift_time_axis(avgdaily, shift)
