import datetime
import io
import pandas as pd
# import numpy as np
import os
import re
import warnings

from .multilang import fields, columns, day_first
from ..base import BaseRaw


class RawRPX(BaseRaw):
    """Raw object from .CSV file (recorded by Respironics)

    Parameters
    ----------
    input_fname: str
        Path to the rpx file.
    language: str, optional
        Language of the input csv file.
        Available options are: 'ENG_UK', 'ENG_US', 'FR'.
        Default is 'ENG_US'.
    dayfirst: bool, optional
        Whether to interpret the first value of a date as the day.
        If None, rely on the laguage:
        * ENG_US: False
        * ENG_UK or FR: True
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    data_dtype: dtype, optional
        The dtype of the raw data.
        Default is 'float'.
    light_dtype: dtype, optional
        The dtype of the raw light data.
        Default is 'float'.
    delimiter: str, optional
        Delimiter to use when reading the input file.
        Default is '.'
    """

    def __init__(
        self,
        input_fname,
        language='ENG_US',
        dayfirst=None,
        start_time=None,
        period=None,
        data_dtype='float',
        light_dtype='float',
        delimiter=','
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .rpx

        if language not in fields.keys():
            raise ValueError(
                'Language {0} not supported. Supported languages: {1}'.format(
                    language, '" or "'.join(fields.keys())
                )
            )
        else:
            self.__language = language
        # [TO-DO] check language is supported

        # Unless specified otherwise,
        # set dayfirst as a function of the language
        if dayfirst is None:
            dayfirst = day_first[language]

        # extract file header and data header
        header = []
        data_available_cols = []
        with open(input_fname, mode='rb') as file:
            data = file.readlines()
        for header_offset, line in enumerate(data, 1):
            if fields[self.language]['Data'] in line.decode('utf-8'):
                break
            else:
                header.append(line.decode('utf-8'))
        # Read file until the next blank line
        # First, skip blank line after section title
        # next(file)
        for data_offset, line in enumerate(data[header_offset+1:]):
            line_clean = line.replace(b'\r\r\n', b'\r\n')
            if line_clean == b'\r\n':
                break
            else:
                data_available_cols.append(
                    line_clean.decode(
                        'utf-8'
                    ).split(',')[0].strip('"').rstrip(':')
                )
        # Verify that the input file contains the needed informations
        try:
            assert (
                set(columns[self.language].values()) <=
                set(data_available_cols[2:])
            )
        except AssertionError:
            print(
                'The data section of the input file {}'.format(input_fname) +
                'does not contain the required columns.\n' +
                'Required columns: {}'.format('" or "'.join(
                    columns[self.language].values())
                ) +
                'Available columns: {}'.format('" or "'.join(
                    data_available_cols[2:])
                )
            )

        # extract informations from the header
        name = self.__extract_rpx_name(header, delimiter)
        uuid = self.__extract_rpx_uuid(header, delimiter)
        start = self.__extract_rpx_start_time(header, delimiter, dayfirst)
        frequency = self.__extract_rpx_frequency(header, delimiter)
        axial_mode = 'Unknown'

        # read actigraphy data
        with open(input_fname, mode='rb') as file:
            data = file.read()
        data = data.replace(b'\r\r\n', b'\r\n')

        index_data = pd.read_csv(
            # input_fname,
            io.StringIO(data.decode('utf-8')),
            encoding='utf-8',
            skiprows=header_offset+data_offset+1,
            header=0,
            delimiter=delimiter,
            # infer_datetime_format=True,
            index_col=0,
            parse_dates={
                'Date_Time': [
                    columns[self.language]['Date'],
                    columns[self.language]['Time']
                ]
            },
            dayfirst=dayfirst,  # (self.language in day_first),
            usecols=list(columns[self.language].values()),
            na_values='NAN',
            dtype={
                columns[self.language]['Activity']: data_dtype,
                columns[self.language]['White_light']: light_dtype
                # columns[self.language]['Marker']: light_dtype
            }
        ).dropna(subset=[columns[self.language]['Activity']])

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
            data=index_data[columns[self.language]['Activity']].asfreq(
                freq=pd.Timedelta(frequency)
            ),
            light=index_data[columns[self.language]['White_light']].asfreq(
                freq=pd.Timedelta(frequency)
            )
        )

    @property
    def language(self):
        return self.__language

    def __extract_rpx_name(self, header, delimiter):
        for line in header:
            if fields[self.language]['Name'] in line:
                name = re.sub(
                    r'[^\w\s]', '', line.split(delimiter)[1]
                ).strip()
                break
        return name

    def __extract_rpx_uuid(self, header, delimiter):
        for line in header:
            if fields[self.language]['Device_id'] in line:
                uuid = re.sub(r'[\W_]+', '', line.split(delimiter)[1])
                break
        return uuid

    def __extract_rpx_start_time(self, header, delimiter, dayfirst):
        start_time = []
        for line in header:
            if fields[self.language]['Start_date'] in line:
                start_time.append(
                    re.sub(r'[^\d./]+', '', line.split(delimiter)[1])
                )
            elif fields[self.language]['Start_time'] in line:
                start_time.append(
                    re.sub(r'[^\d.:AMP]+', '', line.split(delimiter)[1])
                )
        return pd.to_datetime(
            ' '.join(start_time),
            dayfirst=dayfirst  # (self.language in day_first)
        )

    def __extract_rpx_frequency(self, header, delimiter):
        for line in header:
            if fields[self.language]['Period'] in line:
                frequency = pd.Timedelta(
                    int(re.sub(r'([^\s\w])+', '', line.split(delimiter)[1])
                        .replace('\xa0', ' ').strip()),
                    unit='second'
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


def read_raw_rpx(
    input_fname,
    language='ENG_US',
    dayfirst=None,
    start_time=None,
    period=None,
    data_dtype='float',
    light_dtype='float',
    delimiter=','
):
    """Reader function for raw Respironics file.

    Parameters
    ----------
    input_fname: str
        Path to the rpx file.
    language: str, optional
        Language of the input csv file.
        Available options are: 'ENG_UK', 'ENG_US', 'FR'.
        Default is 'ENG_US'.
    dayfirst: bool, optional
        Whether to interpret the first value of a date as the day.
        If None, rely on the laguage:
        * ENG_US: False
        * ENG_UK or FR: True
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    data_dtype: dtype, optional
        The dtype of the raw data.
        Default is 'float'.
    light_dtype: dtype, optional
        The dtype of the raw light data.
        Default is 'float'.
    delimiter: str, optional
        Delimiter to use when reading the input file.
        Default is '.'

    Returns
    -------
    raw : Instance of RawRPX
        An object containing raw RPX data
    """

    return RawRPX(
        input_fname=input_fname,
        language=language,
        dayfirst=dayfirst,
        start_time=start_time,
        period=period,
        data_dtype=data_dtype,
        light_dtype=light_dtype,
        delimiter=delimiter
    )
