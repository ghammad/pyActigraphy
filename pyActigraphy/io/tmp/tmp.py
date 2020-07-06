import pandas as pd
import os
import re

from ..base import BaseRaw


class RawTMP(BaseRaw):
    r"""Raw object from .txt file recorded by Tempatilumi (CE Brasil)

    Parameters
    ----------
    input_fname: str
        Path to the Tempatilumi file.
    name: str, optional
        Name of the recording. If None, the device UUID is used instead.
        Default is None.
    header_size: int, optional
        Header size (i.e. number of lines) of the raw data file.
        Default is 5.
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
    encoding: str, optional
        Encoding to use for UTF when reading the file.
        Default is "latin-1".
    """

    def __init__(
        self,
        input_fname,
        name=None,
        header_size=5,
        frequency=None,
        start_time=None,
        period=None,
        encoding='latin-1'
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .awd

        # extract header and data size
        with open(input_fname, encoding=encoding) as f:
            header = [next(f) for x in range(header_size)]

        # extract informations from the header
        uuid = self.__extract_tmp_uuid(header)
        if name is None:
            name = uuid

        index_data = pd.read_csv(
            # input_fname,
            filepath_or_buffer=input_fname,
            encoding=encoding,
            skipinitialspace=True,
            skiprows=len(header),
            delimiter='\t',
            infer_datetime_format=True,
            index_col=0,
            parse_dates={
                'Date_Time': [
                    'Data',
                    'Hora'
                ]
            },
            dayfirst=True
        )

        # Check column names
        # Evento	 Temperatura	 Luminosidade	 Atividade
        if 'Atividade' not in index_data.columns:
            raise ValueError(
                'The activity counts cannot be found in the data.\n'
                'Column name in file header should be "Atividade".'
            )

        self.__temperature = self.__extract_from_data(
            index_data, 'Temperatura'
        )

        self.__events = self.__extract_from_data(
            index_data, 'Evento'
        )

        if frequency is not None:
            index_data = index_data.resample(frequency).sum()
            freq = pd.Timedelta(frequency)
        elif not index_data.index.inferred_freq:
            raise ValueError(
                'The sampling frequency:\n'
                '- cannot be inferred from the data\n'
                'AND\n'
                '- is NOT explicity passed to the reader function.\n'
            )
        else:
            index_data = index_data.asfreq(index_data.index.inferred_freq)
            freq = pd.Timedelta(index_data.index.freq)

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
            format='TMP',
            axial_mode='tri-axial',
            start_time=start_time,
            period=period,
            frequency=freq,
            data=index_data['Atividade'],
            light=self.__extract_from_data(index_data, 'Luminosidade')
        )

    @property
    def temperature(self):
        r"""Value of the temperature (in ° C)."""
        return self.__temperature

    @property
    def events(self):
        r"""Events markers."""
        return self.__events

    @classmethod
    def __extract_tmp_uuid(cls, header):
        match = re.search(r'\d+', header[0])
        if not match:
            raise ValueError('UUID cannot be extracted from the file header.')
        return match[0]

    @classmethod
    def __extract_from_data(cls, data, key):
        if key in data.columns:
            return data[key]
        else:
            return None


def read_raw_tmp(
    input_fname,
    name=None,
    header_size=5,
    frequency=None,
    start_time=None,
    period=None,
    encoding='latin-1'
):
    r"""Raw object from .txt file recorded by Tempatilumi (CE Brasil)

    Parameters
    ----------
    input_fname: str
        Path to the Tempatilumi file.
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
    raw : Instance of RawTMP
        An object containing raw TMP data
    """

    return RawTMP(
        input_fname=input_fname,
        name=name,
        header_size=header_size,
        frequency=frequency,
        start_time=start_time,
        period=period,
        encoding=encoding
    )
