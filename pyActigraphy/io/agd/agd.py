import pandas as pd
import numpy as np
import os
import sqlite3
import warnings

from ..base import BaseRaw


class RawAGD(BaseRaw):
    r"""Raw object from .agd file (recorded by Actigraph)

    Parameters
    ----------
    input_fname: str
        Path to the AGD file.
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
        start_time=None,
        # end_time=None,
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
        uuid = settings.at['deviceserial', 'settingValue']
        start = self.__to_timestamps(
            int(settings.at['startdatetime', 'settingValue'])
        )
        freq = pd.to_timedelta(
            int(settings.at['epochlength', 'settingValue']),
            unit='s'
        )

        # extract proximity (wear/no-wear) informations
        capsense = pd.read_sql_query(
            "SELECT state, timeStamp FROM capsense",
            connection,
            index_col='timeStamp'
        ).squeeze()
        # convert index to a date time
        capsense.index = self.__to_timestamps(capsense.index)
        # set index frequency
        if 'proximityIntervalInSeconds' in settings.index:
            self.capsense = capsense.asfreq(
                pd.to_timedelta(
                    int(
                        settings.at[
                            'proximityIntervalInSeconds',
                            'settingValue'
                        ]
                    ),
                    unit='s'
                )
            )
        elif capsense.index.inferred_freq is not None:
            self.capsense = capsense.asfreq(capsense.index.inferred_freq)
        else:
            warnings.warn(
                'Acquisition frequency for the wearing sensor data ' +
                '(capsense) could neither be retrieved nor inferred from ' +
                'the data.\n Use this information at your own risk',
                UserWarning
            )
            self.capsense = capsense

        # extract acceleration and light data
        data = pd.read_sql_query(
            "SELECT * FROM data",
            connection,
            index_col='dataTimestamp'
        )
        # convert index to a date time
        data.index = self.__to_timestamps(data.index)
        # set index frequency
        data = data.asfreq(freq=freq)

        # calculate the magnitude of the acceleration vector.
        data['mag'] = np.sqrt(
            data['axis1']*data['axis1'] +
            data['axis2']*data['axis2'] +
            data['axis3']*data['axis3']
        )

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = start

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = data.index[-1]
            period = stop_time - start_time

        data = data.loc[start_time:stop_time]

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='AGD',
            axial_mode='tri-axial',
            start_time=start_time,
            period=period,
            frequency=freq,
            data=data['mag'],
            light=data['lux'] if 'lux' in data.columns else None
        )

        # Close sqlite3 connection
        connection.close()

    @staticmethod
    def __to_timestamps(ticks):
        return pd.to_datetime(
            (ticks/10000000) - 62135596800,
            unit='s'
        )


def read_raw_agd(
    input_fname,
    start_time=None,
    period=None
):
    r"""Reader function for raw AWD file.

    Parameters
    ----------
    input_fname: str
        Path to the AGD file.
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

    return RawAGD(
        input_fname=input_fname,
        start_time=start_time,
        period=period
    )
