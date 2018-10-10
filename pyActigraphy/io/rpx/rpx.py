import datetime
import pandas as pd
import numpy as np
import os
import re
import warnings

from ..base import BaseRaw


class RawRPX(BaseRaw):
    """Raw object from .CSV file (recorded by Respironics)

    Parameters
    ----------
    input_fname: str
        Path to the rpx file.
    header_offset: int
        Offset (i.e. number of lines) between the end of the header and
        the data. Default is 15.
    delimiter: str
        Delimiter to use when reading the input file.
    """

    def __init__(
        self,
        input_fname,
        header_offset=15,
        start_time=None,
        period=None,
        delimiter=','
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .rpx

        # extract header and size
        header = []
        with open(input_fname, encoding='utf-8') as file:
            for num, line in enumerate(file, 1):
                if "Données période par période" in line:
                    break
                else:
                    header.append(line)

        # extract informations from the header
        name = self.__extract_rpx_name(header, delimiter)
        uuid = self.__extract_rpx_uuid(header, delimiter)
        start = self.__extract_rpx_start_time(header, delimiter)
        frequency = self.__extract_rpx_frequency(header, delimiter)
        axial_mode = 'DUMMY [TODO]: extract from header if possible'

        # read data file
        index_data = pd.read_csv(
            input_fname,
            encoding='utf-8',
            skiprows=num+header_offset,
            header=0,
            delimiter=delimiter,
            # infer_datetime_format=True,
            index_col=0,
            parse_dates=[[0, 1]],
            dayfirst=True,
            usecols=[
                'Date', 'Heure', 'Activité', 'Marqueur', 'Lumière blanche'
            ],
            na_values='NAN',
            dtype={'Activité': np.float32, 'Marqueur': np.float32}
        ).dropna(subset=['Activité'])

        # verify that the start time and the first date index matches
        self.__check_rpx_start_time(index_data, start)

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = start

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = index_data.index[-1]
            period = stop_time - start_time

        index_data = index_data[start_time:stop_time]

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='RPX',
            axial_mode=axial_mode,
            start_time=start_time,
            period=period,
            frequency=pd.Timedelta(frequency),
            data=index_data['Activité'],
            light=index_data['Lumière blanche']
        )

    def __extract_rpx_name(self, header, delimiter):
        for line in header:
            if 'Identité' in line:
                name = re.sub(
                    r'[^\w\s]', '', line.split(delimiter)[1]
                ).strip()
                break
        return name

    def __extract_rpx_uuid(self, header, delimiter):
        for line in header:
            if 'Numéro de série de l\'Actiwatch' in line:
                uuid = re.sub(r'[\W_]+', '', line.split(delimiter)[1])
                break
        return uuid

    def __extract_rpx_start_time(self, header, delimiter):
        start_time = []
        for line in header:
            if 'Date de début de la collecte des données' in line:
                start_time.append(
                    re.sub(r'[^\d./]+', '', line.split(delimiter)[1])
                )
            elif 'Heure de début de la collecte des données' in line:
                start_time.append(
                    re.sub(r'[^\d.:]+', '', line.split(delimiter)[1])
                )
        return pd.to_datetime(' '.join(start_time), dayfirst=True)

    def __extract_rpx_frequency(self, header, delimiter):
        for line in header:
            if 'Longueur de la période' in line:
                frequency = pd.Timedelta(
                    re.sub(r'([^\s\w])+', '', line.split(delimiter)[3])
                    .replace('\xa0', ' ').strip()
                )
                break
        return frequency

    def __check_rpx_start_time(
        self, data, start_time, tolerance=datetime.timedelta(minutes=1)
    ):
        warning_msg = """
- Start time extracted from the header_size: {0}
- Datetime index of the first data points : {1}
do not match.
Please verify your input file.
"""
        if abs(data.index[0] - start_time) > tolerance:
            warnings.warn(
                warning_msg.format(start_time, data.index[0])
            )


def read_raw_rpx(input_fname, header_offset=15, delimiter=','):
    """Reader function for raw Respironics file.

    Parameters
    ----------
    input_fname: str
        Path to the Respironics file.
    header_offset: int
        Offset (i.e. number of lines) between the end of the header and
        the data. Default is 15.
    delimiter: str
        Delimiter to use when reading the input file.

    Returns
    -------
    raw : Instance of RawRPX
        An object containing raw RPX data
    """

    return RawRPX(
        input_fname=input_fname,
        header_offset=header_offset,
        delimiter=delimiter
    )
