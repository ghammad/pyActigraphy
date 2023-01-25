import pandas as pd
import os
import re

from ..base import BaseRaw
from pyActigraphy.light import LightRecording


class RawATR(BaseRaw):
    r"""Raw object from .txt file recorded by ActTrust (Condor Instruments)

    Parameters
    ----------
    input_fname: str
        Path to the ActTrust file.
    mode: str, optional
        Activity sampling mode.
        Available modes are: Proportional Integral Mode (PIM),  Time Above
        Threshold (TAT) and Zero Crossing Mode (ZCM).
        Default is PIM.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    """

    __default_modes = ["PIM", "PIMn", "TAT", "TATn", "ZCM", "ZCMn"]

    def __init__(
        self,
        input_fname,
        mode='PIM',
        start_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # extract header and data size
        header = {}
        with open(input_fname) as fp:
            first_line = fp.readline()
            if not re.match(r"\+-*\+ \w+ \w+ \w+ \+-*\+", first_line):
                raise ValueError(
                    "The input file ({}) does not ".format(input_fname)
                    + "seem to contain the usual header.\n Aborting."
                )
            for line in fp:
                if '+-------------------' in line:
                    break
                else:
                    chunks = line.strip().split(' : ')
                    if chunks:
                        header[chunks[0]] = chunks[1:]
        if not header:
            raise ValueError(
                "The input file ({}) does not ".format(input_fname)
                + "contain a header.\n Aborting."
            )

        # extract informations from the header
        uuid = header['DEVICE_ID'][0]
        name = header['SUBJECT_NAME'][0]
        freq = pd.Timedelta(int(header['INTERVAL'][0]), unit='s')
        self.__tat_thr = self.__extract_from_header(header, 'TAT_THRESHOLD')

        index_data = pd.read_csv(
            input_fname,
            skiprows=len(header)+2,
            sep=';',
            parse_dates=True,
            infer_datetime_format=True,
            dayfirst=True,
            index_col=[0]
        ).resample(freq).sum()

        self.__available_modes = sorted(list(
            set(index_data.columns.values).intersection(
                set(self.__default_modes))))

        # Check requested sampling mode is available:
        if mode not in self.__available_modes:  # header['MODE'][0].split('/'):
            raise ValueError(
                "The requested mode ({}) is not available".format(mode)
                + " for this recording.\n"
                + "Available modes are {}.".format(
                    self.__available_modes  # header['MODE'][0]
                )
            )

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = index_data.index[0]

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = index_data.index[-1]
            period = stop_time - start_time

        index_data = index_data[start_time:stop_time]

        # ACTIVITY
        self.__activity = index_data[self.__available_modes]

        # TEMP
        self.__temperature = self.__extract_from_data(
            index_data, 'TEMPERATURE'
        )
        self.__temperature_ext = self.__extract_from_data(
            index_data, 'EXT TEMPERATURE'
        )

        # LIGHT
        index_light = index_data.filter(like="LIGHT")

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='ATR',
            axial_mode='tri-axial',
            start_time=start_time,
            period=period,
            frequency=freq,
            data=index_data[mode],
            light=LightRecording(
                name=name,
                uuid=uuid,
                data=index_light,
                frequency=index_light.index.freq
            ) if index_light is not None else None
            # self.__extract_from_data(index_data, 'LIGHT')
        )

    @property
    def available_modes(self):
        r"""Available acquistion modes (PIM, ZCM, etc)"""
        return self.__available_modes

    @property
    def PIM(self):
        r"""Activity (in PIM mode)."""
        return self.__extract_from_data(self.__activity, 'PIM')

    @property
    def PIMn(self):
        r"""Activity (in normalized PIM mode)."""
        return self.__extract_from_data(self.__activity, 'PIMn')

    @property
    def TAT(self):
        r"""Activity (in TAT mode)."""
        return self.__extract_from_data(self.__activity, 'TAT')

    @property
    def TATn(self):
        r"""Activity (in normalized PIM mode)."""
        return self.__extract_from_data(self.__activity, 'TATn')

    @property
    def ZCM(self):
        r"""Activity (in ZCM mode)."""
        return self.__extract_from_data(self.__activity, 'ZCM')

    @property
    def ZCMn(self):
        r"""Activity (in normalized ZCM mode)."""
        return self.__extract_from_data(self.__activity, 'ZCMn')

    @property
    def temperature(self):
        r"""Value of the temperature (in ° C)."""
        return self.__temperature

    @property
    def temperature_ext(self):
        r"""Value of the external temperature (in ° C)."""
        return self.__temperature_ext

    @property
    def amb_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("AMB LIGHT")

    @property
    def white_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("LIGHT")

    @property
    def red_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("RED LIGHT")

    @property
    def green_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("GREEN LIGHT")

    @property
    def blue_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("BLUE LIGHT")

    @property
    def ir_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("IR LIGHT")

    @property
    def uva_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("UVA LIGHT")

    @property
    def uvb_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__extract_light_channel("UVB LIGHT")

    @property
    def tat_threshold(self):
        r"""Threshold used in the TAT mode."""
        return self.__tat_thr

    @classmethod
    def __extract_from_header(cls, header, key):
        if header.get(key, None) is not None:
            return header[key][0]

    @classmethod
    def __extract_from_data(cls, data, key):
        if key in data.columns:
            return data[key]
        else:
            return None

    def __extract_light_channel(self, channel):
        if self.light is None:
            return None
        else:
            return self.light.get_channel(channel)


def read_raw_atr(
    input_fname,
    mode='PIM',
    start_time=None,
    period=None
):
    r"""Reader function for .txt file recorded by ActTrust (Condor Instruments)

    Parameters
    ----------
    input_fname: str
        Path to the ActTrust file.
    mode: str, optional
        Activity sampling mode.
        Available modes are: Proportional Integral Mode (PIM),  Time Above
        Threshold (TAT) and Zero Crossing Mode (ZCM).
        Default is PIM.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).

    Returns
    -------
    raw : Instance of RawATR
        An object containing raw ATR data
    """

    return RawATR(
        input_fname=input_fname,
        mode=mode,
        start_time=start_time,
        period=period
    )
