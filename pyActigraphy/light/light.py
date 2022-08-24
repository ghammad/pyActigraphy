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
# import pandas as pd
import warnings
from ..filters import FiltersMixin
from ..recording import BaseRecording
from .light_metrics import LightMetricsMixin


class LightRecording(LightMetricsMixin, FiltersMixin, BaseRecording):
    """ Base class for log files containing time stamps.

    Parameters
    ----------
    name: str
        Name of the light recording.
    data: pandas.DataFrame
        Dataframe containing the light data found in the recording.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        r"""Name of the light recording."""
        return self.__name

    def get_channel(self, channel):
        r"""Light channel accessor"""
        if channel not in self.data.columns:
            raise ValueError(
                'The light channel you tried to access ({}) '.format(channel)
                + 'is not available.\n Available channels:{}'.format(
                    '\n- {}'.join(self.data.columns)
                )
            )

        return self.data.loc[:, channel]

    def get_channels(self, channels=None):
        r"""Light channel accessor

        Get access of the requested channels.

        Parameters
        ----------
        channels: list of str, optional.
            Channel list. If set to None, use all available channels.
            Default is None.

        Returns
        -------
        light: pd.DataFrame
            Dataframe with the requested channels.

        """

        # Select channels of interest
        if channels is None:
            channels_sel = self.data.columns
        else:
            # Check if some required channels are not available:
            channels_unavail = set(channels)-set(self.data.columns)
            if channels_unavail == set(channels):
                raise ValueError(
                    'None of the requested channels ({}) is available.'.format(
                        ', '.join(channels)
                    )
                )
            elif len(channels_unavail) > 0:
                warnings.warn(
                    'Required but unavailable channel(s): {}'.format(
                        ', '.join(channels_unavail)
                    )
                )
            channels_sel = [ch for ch in self.data.columns if ch in channels]

        return self.data.loc[:, channels_sel]

    def get_channel_list(self):
        r"""List of light channels"""
        return self.data.columns
