import pandas as pd
import os

from ..base import BaseRaw


class RawBBA(BaseRaw):
    r"""Raw object from files produced by the
    [biobankanalysis](
        https://biobankaccanalysis.readthedocs.io/en/latest/index.html
    ) package.

    Parameters
    ----------
    input_fname: str
        Path to the .csv(.gz) file.
    name: str, optional
        Name of the recording.
        Default is None.
    uuid: str, optional
        Device UUID.
        Default is None.
    frequency: str, optional
        Sampling frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        If None, the sampling frequency is inferred from the data. Otherwise,
        the data are resampled to the specified frequency.
        Default is None.
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
        uuid=None,
        frequency=None,
        start_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # read file
        data = pd.read_csv(
            input_fname,
            index_col=['time'],
            date_parser=lambda x: pd.to_datetime(
                x, format='%Y-%m-%d %H:%M:%S.%f%z', exact=False
            )
        )

        if frequency is not None:
            data = data.resample(frequency).mean()
            freq = pd.Timedelta(frequency)
        elif not data.index.inferred_freq:
            raise ValueError(
                'The sampling frequency:\n'
                '- cannot be inferred from the data\n'
                'AND\n'
                '- is NOT explicity passed to the reader function.\n'
            )
        else:
            data = data.asfreq(data.index.inferred_freq)
            freq = pd.Timedelta(data.index.freq)

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

        data = data.loc[start_time:stop_time]

        # LIGHT
        self.__white_light = self.__extract_baa_data(
            data, 'light'
        )

        # MVPA
        self.__mvpa = self.__extract_baa_data(
            data, 'moderate-vigorous'
        )

        # Sedentary
        self.__sedentary = self.__extract_baa_data(
            data, 'sedentary'
        )

        # Sleep
        self.__sleep = self.__extract_baa_data(
            data, 'sleep'
        )

        # MET
        self.__met = self.__extract_baa_data(
            data, 'MET'
        )

        # call __init__ function of the base class
        super().__init__(
            name=name,
            uuid=uuid,
            format='BAA',
            axial_mode='tri-axial',
            start_time=start_time,
            period=period,
            frequency=freq,
            data=data.loc[:, 'acc'],
            light=None
        )

    @property
    def white_light(self):
        r"""Value of the white light illuminance in lux."""
        return self.__white_light

    @property
    def mvpa(self):
        r"""Value of the moderate-vigorous physical activity binary index."""
        return self.__mvpa

    @property
    def sedentary(self):
        r"""Value of the sedentary physical activity binary index."""
        return self.__sedentary

    @property
    def sleep(self):
        r"""Value of the sleep binary index."""
        return self.__sleep

    @property
    def met(self):
        r"""Value of the MET index."""
        return self.__met

    @staticmethod
    def __extract_baa_data(data, column):

        return data.loc[:, column] if column in data.columns else None


def read_raw_bba(
    input_fname,
    name=None,
    uuid=None,
    frequency=None,
    start_time=None,
    period=None,
):
    r"""Reader function for files produced by the biobankAccelerometerAnalysis
    package.

    Parameters
    ----------
    input_fname: str
        Path to the BAA file.
    name: str, optional
        Name of the recording.
        Default is None.
    uuid: str, optional
        Device UUID.
        Default is None.
    frequency: str, optional
        Sampling frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        If None, the sampling frequency is inferred from the data. Otherwise,
        the data are resampled to the specified frequency.
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
    raw : Instance of RawBBA
        An object containing preprocessed data from raw accelerometers.
    """

    return RawBBA(
        input_fname=input_fname,
        name=name,
        uuid=uuid,
        frequency=frequency,
        start_time=start_time,
        period=period,
    )
