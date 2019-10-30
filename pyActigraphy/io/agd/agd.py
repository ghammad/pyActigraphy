import pandas as pd
import os
import sqlite3

from ..base import BaseRaw


class RawAGD(BaseRaw):
    r"""Raw object from .agd file (recorded by Actigraph)

    Parameters
    ----------
    input_fname: str
        Path to the AWD file.
    header_size: int
        Header size (i.e. number of lines) of the raw data file. Default is 7.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    end-time: datetime-like, optional
        Read data up to this time.
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
        start_time=None,
        end_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .agd

        # create connection to the SQLITE3 .agd file in read-only mode
        connection = sqlite3.connect('file:'+input_fname+'?mode=ro', uri=True)

        # extract header and data size
        settings = pd.read_sql_query(
            "SELECT * FROM settings",
            connection,
            index_col='settingName'
        )

        # extract informations from the header
        name = settings.at['subjectname', 'settingValue']
        uuid = settings.at['devicename', 'settingValue']
        start = self.__to_timestamps(
            int(settings.at['startdatetime', 'settingValue'])
        )
        freq = pd.to_timedelta(
            int(settings.at['epochlength', 'settingValue']),
            unit='s'
        )

        data = pd.read_sql_query(
            "SELECT * FROM data",
            connection,
            index_col='dataTimestamp'
        )

        index_data = pd.Series(
            data=data,
            index=pd.date_range(
                start=start,
                periods=len(data),
                freq='1min'
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
            frequency=freq,
            data=index_data,
            light=None
        )

    @staticmethod
    def __to_timestamps(ticks):
        return pd.to_datetime(
            (ticks/10000000) - 62135596800,
            unit='s'
        )

def read_raw_awd(
    input_fname,
    header_size=7,
    frequency='1min',
    start_time=None,
    period=None
):
    r"""Reader function for raw AWD file.

    Parameters
    ----------
    input_fname: str
        Path to the AWD file.
    header_size: int
        Header size (i.e. number of lines) of the raw data file. Default is 7.
    frequency: str
        Data acquisition frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is '1T'.
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
