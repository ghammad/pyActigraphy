#############################################################################
# Copyright (c) 2022, Daylight Academy
# Author: Grégory Hammad
# Owner: Daylight Academy (https://daylight.academy)
# Maintainer: Grégory Hammad
# Email: gregory.hammad@uliege.be
# Status: development
#############################################################################
# The development of a module for analysing light exposure
# data was led and financially supported by members of the Daylight Academy
# Project “The role of daylight for humans” (led by Mirjam Münch, Manuel
# Spitschan). The module is part of the Human Light Exposure Database. For
# more information about the project, please see
# https://daylight.academy/projects/state-of-light-in-humans/.
#
# This module is also part of the pyActigraphy software.
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
import os

from .light import LightRecording


class GenLightDevice(LightRecording):
    r"""Generic light acquisition device

    Parameters
    ----------
    input_fname: str
        Path to the file.
    channels: list of str, optional
        Select channels to read from the input file.
        If the list is empty, all channels are read.
        Default is \[\].
    rsfreq: str, optional
        Resampling frequency. Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None.
    agg: str, optional
        Aggregation function to use when resampling.
        Default is 'mean'.
    log10_transform: bool, optional
        If set to True, data are (log10\+1)-transformed.
        Default is True.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    dayfirst: bool, optional
        If set to True, the timestamps are parsed as DD/MM/YYYY
    """

    def __init__(
        self,
        input_fname,
        channels=[],
        rsfreq=None,
        agg='mean',
        log10_transform=True,
        start_time=None,
        period=None,
        dayfirst=True
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .awd

        # Extracting data
        data = pd.read_csv(
            input_fname,
            index_col='UTC Timestamp',
            parse_dates=True,
            infer_datetime_format=True,
            dayfirst=dayfirst
        )
        # Extracting UUID
        uuid = data.loc[:, 'Device ID'].unique()
        if uuid.size != 1:
            raise ValueError(
                'The UUID retrieved from the input file is {}.'.format(
                    'missing' if uuid.size == 0 else 'not unique: {}'.format(
                        ', '.join(uuid)
                    )
                )
            )
        else:
            uuid = uuid[0]
            # Drop UUID column once it has been extracted
            data.drop(columns=['Device ID'], inplace=True)

        # Resampling, if required and possible.
        if rsfreq is None:
            if data.index.inferred_freq is not None:
                data = data.asfreq(data.index.inferred_freq)
            else:
                raise ValueError(
                    "The acquisition frequency could not be retrieved from the"
                    " data and no resampling frequency was not provided by the"
                    " user.\nPlease specify the input parameter 'rsfrq' in"
                    " order to overcome this issue."
                )
        else:
            data = data.resample(rsfreq).agg(agg)

        # Restricting data to start/stop times, if required.
        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = data.index[0]

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = data.index[-1]
            period = stop_time - start_time

        data = data.loc[start_time:stop_time]

        # Extracting other metadata
        self.__cct = self.__extract_from_data(
            data, 'CCT in K'
        )
        self.__duv = self.__extract_from_data(
            data, 'Duv'
        )
        self.__tilt = self.__extract_from_data(
            data, 'Tilt in °'
        )
        self.__usertriggered = self.__extract_from_data(
            data, 'TriggeredByUser'
        )

        # Drop metadata
        data.drop(
            columns=['CCT in K', 'Duv', 'Tilt in °', 'TriggeredByUser'],
            inplace=True,
            errors='raise'  # Change to ignore optional columns
        )

        # call __init__ function of the base class
        super().__init__(
            name=os.path.basename(input_fname),
            uuid=uuid,
            data=data[
                [col for col in data.columns
                 if ((col in channels) if channels else True)]
            ],
            frequency=data.index.freq.delta,
            log10_transform=log10_transform
        )
        self.start_time = start_time
        self.period = period

    @classmethod
    def __extract_from_data(cls, data, key):
        if key in data.columns:
            return data[key]
        else:
            return None

    @property
    def cct(self):
        r"""Value of the CCT (in K)."""
        return self.__cct

    @property
    def duv(self):
        r"""Value of the delta u,v."""
        return self.__duv

    @property
    def tilt(self):
        r"""Value of the tilt (in °)."""
        return self.__tilt

    @property
    def triggered_by_user(self):
        r"""Value of the marker 'TriggeredByUser'."""
        return self.__usertriggered.round(0).astype(bool)


def read_raw_gld(
    input_fname,
    channels=[],
    rsfreq=None,
    agg='mean',
    log10_transform=True,
    start_time=None,
    period=None,
    dayfirst=True


):
    r"""Reader function for generic light device file.

    Parameters
    ----------
    input_fname: str
        Path to the file.
    channels: list of str, optional
        Select channels to read from the input file.
        If the list is empty, all channels are read.
        Default is \[\].
    rsfreq: str, optional
        Resampling frequency. Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None.
    agg: str, optional
        Aggregation function to use when resampling.
        Default is 'mean'.
    log10_transform: bool, optional
        If set to True, data are (log10\+1)-transformed.
        Default is True.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    dayfirst: bool, optional
        If set to True, the timestamps are parsed as DD/MM/YYYY

    Returns
    -------
    raw : Instance of GenLightDevice
        An object containing raw GLD data
    """

    return GenLightDevice(
        input_fname,
        channels=channels,
        rsfreq=rsfreq,
        agg=agg,
        log10_transform=log10_transform,
        start_time=start_time,
        period=period,
        dayfirst=dayfirst
    )
