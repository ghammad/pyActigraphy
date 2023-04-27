import pandas as pd
import os
import re
import warnings

from ..base import BaseRaw
from pyActigraphy.light import LightRecording


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
    engine: str, optional
        Parser engine to use.
        Default is 'python'.
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

    device_code = {
        'D': 'Actiwatch-7',
        'I': 'Actiwatch-Insomnia (pressure sens.)',
        'L': 'Actiwatch-L (amb. light)',
        'M': 'Actiwatch-Mini',
        'P': 'Actiwatch-L-Plus (amb. light)',
        'S': 'Actiwatch-S (env. sound)',
        'T': 'Actiwatch-T (temp.)',
        'V': 'Actiwatch-4'
    }

    device_default_channels = ['Activity', 'Marker']

    device_additional_channel = {
        'D': 'Light',
        'I': 'Pressure',
        'L': 'Light',
        # 'M': None,
        'P': 'Light',
        'S': 'Sound',
        'T': 'Temp.',
        # 'V': None,
        # 'U': None,
        # 'X': None
    }

    def __init__(
        self,
        input_fname,
        header_size=7,
        frequency=None,
        start_time=None,
        period=None,
        engine='python'
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # [TO-DO] check if file exists
        # [TO-DO] check it is has the right file extension .awd

        # extract header and data size
        with open(input_fname) as f:
            header = [next(f) for x in range(header_size)]

        # extract informations from the header
        name = RawAWD.__extract_awd_name(header)
        freq = RawAWD.__extract_awd_frequency(header)
        uuid = RawAWD.__extract_awd_uuid(header)
        start = RawAWD.__extract_awd_start_time(header)
        if uuid:
            # extract model from UUID:
            self.__device_type = RawAWD.__extract_awd_model(uuid)

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

        # set up channel configuration as a function of the device
        all_channels = RawAWD.device_default_channels.copy()
        use_channels = RawAWD.device_default_channels.copy()
        if self.__device_type in RawAWD.device_additional_channel.keys():
            # whitespace delimiter introduces the comma as an additional col.
            all_channels[1:1] = [
                'sep', RawAWD.device_additional_channel[self.__device_type]
            ]
            use_channels[1:1] = [
                RawAWD.device_additional_channel[self.__device_type]
            ]

        # extract data
        data = pd.read_csv(
            filepath_or_buffer=input_fname,
            encoding='utf-8',
            engine=engine,
            header=None,
            delim_whitespace=True,
            names=all_channels,
            index_col=False,
            usecols=use_channels,
            skiprows=header_size,
            dtype={
                'Activity': int,
                'Light': float,
                'Pressure': float,
                'Sound': float,
                'Temp.': float,
                'Marker': str
                }
        )

        data.index = pd.date_range(
            start=start,
            periods=len(data),
            freq=freq
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
            fpath=input_fname,
            name=name,
            uuid=uuid,
            format='AWD',
            axial_mode='mono-axial',
            start_time=start_time,
            period=period,
            frequency=pd.Timedelta(freq),
            data=data.loc[:, 'Activity'],
            light=LightRecording(
                name=name,
                uuid=uuid,
                data=data.loc[:, 'Light'].to_frame(name='whitelight'),
                frequency=freq
            ) if 'Light' in data.columns else None
        )

    @property
    def model(self):
        """Actiwatch Model as inferred from the header info."""
        return RawAWD.device_code[self.__device_type]

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

    @staticmethod
    def __extract_awd_model(uuid):
        # extract model from UUID:
        wrn_msg = (
            'Only the first data column will be used, assuming it corresponds '
            'to activity counts.'
        )
        match = re.match(pattern=r'^([A-Za-z])[0-9a-fA-F]+', string=uuid)
        if match:  # check if UUID matches the expected pattern
            dcode = match.groups()[0].upper()
            if dcode in RawAWD.device_code.keys():
                return dcode
            else:
                warnings.warn(
                    'The model specified in the UUID ({})'.format(dcode)
                    + ' is not supported at the moment.\n'
                    + 'List of supported Actiwatch models:\n'
                    + '\n'.join(
                        [
                            '- {}: {}'.format(k, dev)
                            for k, dev in RawAWD.device_code.items()
                        ]
                    )
                    + '\n'
                    + wrn_msg
                )
                return 'U'
        else:
            warnings.warn(
                'Cannot detect from the header info (UUID) '
                + 'which Actiwatch model was used to acquire the data.'
                + '\n'
                + wrn_msg
            )
            return 'X'


def read_raw_awd(
    input_fname,
    header_size=7,
    frequency=None,
    start_time=None,
    period=None,
    engine='python'
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
    engine: str, optional
        Parser engine to use.
        Default is 'python'.

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
        period=period,
        engine=engine
    )
