import pandas as pd
import os

from ..base import BaseRaw


class RawMESA(BaseRaw):
    r"""Raw object from MESA files

    Parameters
    ----------
    input_fname: str
        Path to the MESA file.
    time_origin: datetime-like
        Time origin of the timestamps.
        Required as the MESA files do not contain date informations.
        Default is '2000-01-01'
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
        time_origin='2000-01-01',
        start_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # read file
        data = pd.read_csv(input_fname, index_col='line')

        # extract informations from the header
        name = data.loc[1, 'mesaid']

        # set additional informations manually
        uuid = None
        freq = pd.Timedelta(30, unit='s')

        # reconstruct MESA datetime index
        date = pd.to_datetime(
            data['daybymidnight'] - 1 + data.loc[1, 'dayofweek'],
            unit='D',
            origin=time_origin
        ).astype(str)
        time = data['linetime']

        index = pd.DatetimeIndex(date + ' ' + time, freq='infer')

        data.set_index(index, inplace=True)

        # set start and stop times
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

        data = data[start_time:stop_time]

        # LIGHT
        self.__red_light = data['redlight']
        self.__green_light = data['greenlight']
        self.__blue_light = data['bluelight']

        # wake indicator
        self.__wake = data['wake']

        # no wear indicator
        self.__nowear = data['offwrist']

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='MESA',
            axial_mode='tri-axial',
            start_time=start_time,
            period=period,
            frequency=freq,
            data=data['activity'],
            light=data['whitelight']
        )

    @property
    def wake(self):
        r"""Awake indicator."""
        return self.__wake

    @property
    def nowear(self):
        r"""Off-wrist indicator."""
        return self.__nowear

    @property
    def red_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__red_light

    @property
    def green_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__green_light

    @property
    def blue_light(self):
        r"""Value of the light intensity in µw/cm²."""
        return self.__blue_light


def read_raw_mesa(
    input_fname,
    time_origin='2000-01-01',
    start_time=None,
    period=None
):
    r"""Reader function for MESA files

    Parameters
    ----------
    input_fname: str
        Path to the ActTrust file.
    time_origin: datetime-like
        Time origin of the timestamps.
        Required as the MESA files do not contain date informations.
        Default is '2000-01-01'
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
    raw : Instance of RawMESA
        An object containing raw MESA data
    """

    return RawMESA(
        input_fname=input_fname,
        time_origin=time_origin,
        start_time=start_time,
        period=period
    )
