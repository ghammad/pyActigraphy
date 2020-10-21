import pandas as pd
import os

from ..base import BaseRaw


class RawAWD(BaseRaw):
    r"""Raw object from .AWD file (recorded by ActiWatches)

    Parameters
    ----------
    input_fname: str
        Path to the AWD file.
    header_size: int, optional
        Header size (i.e. number of lines) of the raw data file. Default is 7.
    frequency: str, optional
        Data acquisition frequency to use if it cannot be infered from the
        header. Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    dtype: dtype
        The dtype of the raw data. Default is np.int.
    """

    frequency_code = {
        '1': '15s',
        '2': '30s',
        '4': '60s',
        '8': '2min',
        '20': '5min',
        '81': '2s',
        'C1': '5s',
        'C2': '10s'
    }

    def __init__(
        self,
        input_fname,
        header_size=7,
        frequency=None,
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
            data = [int(line.split(' ')[0]) for line in f]

        # extract informations from the header
        name = RawAWD.__extract_awd_name(header)
        freq = RawAWD.__extract_awd_frequency(header)
        uuid = RawAWD.__extract_awd_uuid(header)
        start = RawAWD.__extract_awd_start_time(header)

        if freq is None:
            if frequency is not None:
                freq = frequency
            else:
                raise ValueError(
                    "The acquisition frequency could not be retrieved from the"
                    " header and was not provided by the user. Please specify"
                    " the input parameter 'frequency' in order to overcome"
                    " this issue."
                )

        index_data = pd.Series(
            data=data,
            index=pd.date_range(
                start=start,
                periods=len(data),
                freq=freq
            )
        )

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

        index_data = index_data.loc[start_time:stop_time]

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='AWD',
            axial_mode='mono-axial',
            start_time=start_time,
            period=period,
            frequency=pd.Timedelta(freq),
            data=index_data,
            light=None
        )

    @staticmethod
    def __extract_awd_name(header):
        return header[0].replace('\n', '')

    @staticmethod
    def __extract_awd_frequency(header):
        freq = header[3].replace('\n', '').strip()
        if freq not in RawAWD.frequency_code.keys():
            print("Could not find acquisition frequency in header info.")
            return None
        else:
            return RawAWD.frequency_code[freq]

    @staticmethod
    def __extract_awd_uuid(header):
        return header[5].replace('\n', '')

    @staticmethod
    def __extract_awd_start_time(header):
        return pd.to_datetime(header[1] + ' ' + header[2])


def read_raw_awd(
    input_fname,
    header_size=7,
    frequency=None,
    start_time=None,
    period=None
):
    r"""Reader function for raw AWD file.

    Parameters
    ----------
    input_fname: str
        Path to the AWD file.
    header_size: int, optional
        Header size (i.e. number of lines) of the raw data file. Default is 7.
    frequency: str, optional
        Data acquisition frequency to use if it cannot be infered from the
        header. Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None.
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
    raw : Instance of RawAWD
        An object containing raw AWD data
    """

    return RawAWD(
        input_fname=input_fname,
        header_size=header_size,
        frequency=frequency,
        start_time=start_time,
        period=period
    )
