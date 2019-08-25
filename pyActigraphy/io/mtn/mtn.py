import pandas as pd
import numpy as np
import os

from lxml import etree
from ..base import BaseRaw


class RawMTN(BaseRaw):

    """Raw object from .MTN file (recorded by ActiWatches)

    Parameters
    ----------
    input_fname: str
        Path to the MTN file.
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
        Default is np.int.
    light_dtype: dtype, optional
        The dtype of the raw light data.
        Default is np.float.
    """

    def __init__(
        self,
        input_fname,
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
        frequency = self.__extract_frequency(raw_data)
        axial_mode = self.__extract_axial_mode(raw_data)

        motion = self.__extract_motion(raw_data, data_dtype)
        light = self.__extract_light(raw_data, light_dtype)

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
        if index_light is not None:
            index_light[start_time:stop_time]

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='MTN',
            axial_mode=axial_mode,
            start_time=start_time,
            period=period,
            frequency=pd.Timedelta(frequency),
            data=index_data,
            light=index_light
        )

    def __reading_and_parsing_file(self, input_fname):
        return etree.parse(input_fname).getroot()

    def __extract_mtn_name(self, tree):
        """ Extract name from the MTN file"""
        elem = tree.xpath(
            ("//log2/change/property[name = '+UserID']"
             "//following-sibling::content")
        )
        return elem[0].text
        # return raw_data[0]['property']['content']

    def __extract_mtn_uuid(self, tree):
        """ Extract device name and serial number (white space separation)"""
        device = tree.xpath(
            ("//log2/change/property[name = '=Device']"
             "//following-sibling::content")
        )
        serial = tree.xpath(
            ("//log2/change/property[name = '=SerialNo']"
             "//following-sibling::content")
        )
        return "{} {}".format(
            device[0].text,
            serial[0].text
        )

    def __extract_mtn_start_time(self, tree):
        """ Extract start time from the MTN file"""
        elem = tree.xpath(
            ("//log2/change/property[name = '=StartTime']"
             "//following-sibling::content")
        )
        return pd.to_datetime(elem[0].text)
        # return pd.to_datetime(raw_data[13]['property']['content'])

    def __extract_frequency(self, tree):
        """ Return acquisition frequency after time conversion in second"""
        elem = tree.xpath(
            "//log2/change/channel[name = 'motion']//following-sibling::epoch"
        )
        return pd.Timedelta(
            # int(raw_data[header_size]['channel']['epoch']),
            int(elem[0].text),
            unit='s'
        )

    def __extract_frequency_light(self, tree):
        """ Return light frequency after time conversion in second"""
        # Use a try statement as there might be no light measurement.
        elem = tree.xpath(
            "//log2/change/channel[name = 'Light']//following-sibling::epoch"
        )
        freq = None
        try:
            freq = pd.Timedelta(
                # int(raw_data[header_size+1]['channel']['epoch']),
                int(elem[0].text),
                unit='s'
            )
        except IndexError:
            print('Could not find light frequency.')

        return freq

    def __extract_axial_mode(self, tree):
        """ Extract axial mode (mono-axial or tri-axial)"""
        elem = tree.xpath(
            "//log2/change/channel[name = 'motion']//following-sibling::units"
        )
        # return raw_data[header_size]['channel']['units']
        return elem[0].text

    def __extract_motion(self, tree, dtype):
        """ Extract motion measurement from the MTN file"""
        elem = tree.xpath(
            "//log2/change/channel[name = 'motion']//following-sibling::data"
        )
        motion = np.fromstring(
            # raw_data[header_size]['channel']['data']['#text'].replace(
            elem[0].text.replace(
                '\n',
                ''
            ),
            dtype=dtype,
            sep=','
        )
        return motion

    def __extract_light(self, tree, dtype):
        """ Extract light measurement from the MTN file (if any)"""
        # Use a try statement as there might be no light measurement.
        elem = tree.xpath(
            "//log2/change/channel[name = 'Light']//following-sibling::data"
        )
        try:
            light = np.fromstring(
                elem[0].text.replace(
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
        start_time=start_time,
        period=period,
        data_dtype=data_dtype,
        light_dtype=light_dtype
    )
