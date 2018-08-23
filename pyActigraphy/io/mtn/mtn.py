import pandas as pd
import numpy as np
import xmltodict
import os

from ..base import BaseRaw


class RawMTN(BaseRaw):

    """Raw object from .MTN file (recorded by ActiWatches)

    Parameters
    ----------
    input_fname: str
        Path to the MTN file.
    header_size: int
        Header size (i.e. number of lines) of the raw data file. Default is 16.
    frequency: str
        Data acquisition frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is '1T'.
    data_dtype: dtype
        The dtype of the raw data. Default is np.int.
    light_dtype: dtype
        The dtype of the raw light data. Default is np.float.
    """

    """-----------------------------Constructeur----------------------------"""

    def __init__(
        self,
        input_fname,
        header_size=16,
        start_time=None,
        period=None,
        data_dtype=np.int,
        light_dtype=np.float
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .rpx

        # read xml file
        raw_data = self.__reading_and_parsing_file(input_fname)

        # extract informations from the header
        name = self.__extract_mtn_name(raw_data)
        uuid = self.__extract_mtn_uuid(raw_data)
        start = self.__extract_mtn_start_time(raw_data)
        frequency = self.__extract_frequency(raw_data, header_size)
        axial_mode = self.__extract_axial_mode(raw_data, header_size)

        motion = self.__extract_motion(raw_data, header_size, data_dtype)
        light = self.__extract_light(raw_data, header_size, light_dtype)

        # index the motion time serie
        index_data = pd.Series(
            data=motion,
            index=pd.date_range(
                start=start,
                periods=len(motion),
                freq=frequency
            ),
            dtype=data_dtype
        )

        # index the light time serie
        if light is not None:
            index_light = pd.Series(
                data=light,
                index=pd.date_range(
                    start=start,
                    periods=len(light),
                    freq=frequency
                    ),
                dtype=light_dtype
                )
        else:
            index_light = None

        if start_time is not None and period is not None:
            start_time = pd.to_datetime(start_time)
            period = pd.Timedelta(period)
            index_data = index_data[start_time:start_time+period]
            if index_light is not None:
                index_light[start_time:start_time+period]
        else:
            start_time = start

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='MTN',
            axial_mode=axial_mode,
            start_time=start_time,
            frequency=pd.Timedelta(frequency),
            data=index_data,
            light=index_light
        )

    def __reading_and_parsing_file(self, input_fname):
        with open(input_fname, encoding='utf-8') as fd:
            doc = xmltodict.parse(fd.read())
            return doc['motionfile']['log2']['change']

    def __extract_mtn_name(self, raw_data):
        """ Extract name from the MTN file"""
        return raw_data[0]['property']['content']

    def __extract_mtn_uuid(self, raw_data):
        """ Extract device name and serial number (white space separation)"""
        return "{} {}".format(
            raw_data[6]['property']['content'],
            raw_data[7]['property']['content']
        )

    def __extract_mtn_start_time(self, raw_data):
        """ Extract start time from the MTN file"""
        return pd.to_datetime(raw_data[13]['property']['content'])

    def __extract_frequency(self, raw_data, header_size):
        """ Return acquisition frequency after time conversion in second"""
        return pd.Timedelta(
            int(raw_data[header_size]['channel']['epoch']),
            unit='s'
        )

    def __extract_frequency_light(self, raw_data, header_size):
        """ Return light frequency after time conversion in second"""
        # Use a try statement as there might be no light measurement.
        try:
            freq = pd.Timedelta(
                int(raw_data[header_size+1]['channel']['epoch']),
                unit='s'
            )
        except IndexError:
            print('Could not find light frequency.')
            return None
        else:
            return freq

    def __extract_axial_mode(self, raw_data, header_size):
        """ Extract axial mode (mono-axial or tri-axial)"""
        return raw_data[header_size]['channel']['units']

    def __extract_motion(self, raw_data, header_size, dtype):
        """ Extract motion measurement from the MTN file"""
        motion = np.fromstring(
            raw_data[header_size]['channel']['data']['#text'].replace(
                '\n',
                ''
            ),
            dtype=dtype,
            sep=','
        )
        return motion

    def __extract_light(self, raw_data, header_size, dtype):
        """ Extract light measurement from the MTN file (if any)"""
        # Use a try statement as there might be no light measurement.
        try:
            light = np.fromstring(
                raw_data[header_size+1]['channel']['data']['#text'].replace(
                    '\n',
                    ''
                ),
                dtype=dtype,
                sep=','
            )
        except IndexError:
            print('Could not find light measurement.')
            return None
        else:
            return light


def read_raw_mtn(
    input_fname,
    header_size=16,
    start_time=None,
    period=None,
    data_dtype=np.int,
    light_dtype=np.float
):
    """Reader function for raw MTN file.

    Parameters
    ----------
    input_fname: str
        Path to the MTN file.
    header_size: int
        Header size (i.e. number of lines) of the raw data file. Default is 16.
    start_time: datetime-like str
        If not None, the start_time will be used to slice the data.
        Default is None.
    period: str
        Default is None.
    data_dtype: dtype
        The dtype of the raw data. Default is np.int.
    light_dtype: dtype
        The dtype of the raw light data. Default is np.float.

    Returns
    -------
    raw : Instance of RawMTN
        An object containing raw MTN data
    """

    return RawMTN(
        input_fname=input_fname,
        header_size=header_size,
        start_time=start_time,
        period=period,
        data_dtype=data_dtype,
        light_dtype=light_dtype
    )
