import pandas as pd
import numpy as np
import os
import warnings

from ..base import BaseRaw


class RawDQT(BaseRaw):
    r"""Raw object from .csv file recorded by Daqtometers (Daqtix, Germany)

    Parameters
    ----------
    input_fname: str
        Path to the Daqtometer file.
    name: str, optional
        Name of the recording. If None, the device UUID is used instead.
        Default is None.
    header_size: int, optional
        Header size (i.e. number of lines) of the raw data file.
        Default is 15.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    """

    def __init__(
        self,
        input_fname,
        name=None,
        header_size=15,
        start_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .awd

        # extract header and data size
        with open(input_fname) as f:
            header = [next(f) for x in range(header_size)]

        # extract informations from the header
        uuid = self.__extract_dqt_uuid(header)
        if name is None:
            name = uuid
        freq = self.__extract_dqt_freq(header)

        if freq > self.__extract_dqt_sample_freq(header):
            warnings.warn(
                "The store rate of the DQT data is greater than the sampling "
                "rate.\nData are thus aggregated with the following settigs:\n"
                " - Binning mode: {}".format(
                    self.__extract_dqt_bin_mode(header)
                ),
                UserWarning
            )

        index_data = pd.read_csv(
            input_fname,
            delimiter=',',
            skiprows=(header_size-1),
            header=None,
            names=['activity', 'light'],
            index_col=0,
            parse_dates=[0],
            infer_datetime_format=True,
            dtype=np.float,
            na_values='x'
        ).asfreq(freq)

        # Convert activity from string to float
        index_data['activity'] = index_data['activity'].astype(np.float)

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

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='DQT',
            axial_mode='bi-axial',
            start_time=start_time,
            period=period,
            frequency=freq,
            data=index_data['activity'],
            light=index_data['light']
        )

    @classmethod
    def __match_string(cls, header, match):
        matchings = [s for s in header if match in s]
        if len(matchings) == 0:
            print('No match found for the string: {}.'.format(match))
            return None
        if len(matchings) > 1:
            print('Found multiple matchings for the string: {}'.format(match))
        else:
            return matchings[0]

    @classmethod
    def __extract_dqt_uuid(cls, header):
        uuidstr = cls.__match_string(header=header, match='Serial number')
        return uuidstr.split(',')[1]

    @classmethod
    def __extract_dqt_freq(cls, header):
        freqstr = cls.__match_string(header=header, match='Store rate')
        return pd.Timedelta(int(freqstr.split(',')[1]), unit='s')

    @classmethod
    def __extract_dqt_sample_freq(cls, header):
        freqstr = cls.__match_string(header=header, match='Sample rate')
        return pd.Timedelta(int(freqstr.split(',')[1])/0.1, unit='s')

    @classmethod
    def __extract_dqt_bin_mode(cls, header):
        modestr = cls.__match_string(header=header, match='Binning mode')
        return modestr.split(',')[1]


def read_raw_dqt(
    input_fname,
    name=None,
    header_size=15,
    start_time=None,
    period=None
):
    r"""Raw object from .csv file recorded by Daqtometers (Daqtix, Germany)

    Parameters
    ----------
    input_fname: str
        Path to the DQT file.
    name: str, optional
        Name of the recording. If None, the device UUID is used instead.
        Default is None.
    header_size: int
        Header size (i.e. number of lines) of the raw data file. Default is 15.
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
    raw : Instance of RawDQT
        An object containing raw DQT data
    """

    return RawDQT(
        input_fname=input_fname,
        name=name,
        header_size=header_size,
        start_time=start_time,
        period=period
    )
