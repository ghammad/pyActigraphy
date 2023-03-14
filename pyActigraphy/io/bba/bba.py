import json
import os
import pandas as pd

from ..base import BaseRaw
from accelerometer.utils import date_parser
from accelerometer.summarisation import imputeMissing


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
    engine: str, optional
        Parser engine to use. Argument passed to Pandas.
        Default is 'c'.
    impute_missing: bool, optional
        If set to True, use missing data imputation from the biobankanalysis
        package.
        Default is False.
    use_metadata_json: bool, optional.
        If set to True, extract meta-data from summary json file.
        Default is True.
    metadata_fname: str, optional
        Path to the summary json file.
        If None, the path to the summary json file is inferred from the input
        file (/path/to/XXX-timeSeries.csv.gz -> /path/to/XXX-summary.json).
        Default is None.
    """

    def __init__(
        self,
        input_fname,
        name=None,
        uuid=None,
        frequency=None,
        start_time=None,
        period=None,
        engine='c',
        impute_missing=False,
        use_metadata_json=True,
        metadata_fname=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # read file
        data = pd.read_csv(
            input_fname,
            engine=engine,
            index_col=['time'],
            parse_dates=['time'],
            date_parser=date_parser
        )

        # read meta-data file (if found):
        if use_metadata_json:
            meta_data = self.__read_baa_metadata_json(
                input_fname, metadata_fname
            )

            # extract UUID if not provided
            if uuid is None:
                uuid = self.__extract_baa_metadata(meta_data, 'file-deviceID')

            # extract QC calibration marker
            self.__qc_calibrated_on_own_data = bool(
                self.__extract_baa_metadata(
                    meta_data,
                    'quality-calibratedOnOwnData'
                )
            )

            # extract QC DST cross-over marker
            self.__qc_daylight_savings_crossover = bool(
                self.__extract_baa_metadata(
                    meta_data,
                    'quality-daylightSavingsCrossover'
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

        # Impute missing data (if required)
        if impute_missing:
            data = imputeMissing(data)

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
            name=name if name is not None else os.path.basename(input_fname),
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

    @property
    def isCalibratedOnOwnData(self):
        r"""Boolean indicating if the data have been calibrated."""
        return self.__qc_calibrated_on_own_data

    @property
    def isDSTCrossing(self):
        r"""Boolean indicating if the data have been acquired during a DST."""
        return self.__qc_daylight_savings_crossover

    @staticmethod
    def __extract_baa_data(data, column):
        """ Data reader

        Read requested data column.

        Parameters
        ----------
        data: pd.DataFrame
            Input data frame.
        column: str
            Column name.

        Returns
        -------
        ts : pd.Series
            Data contained in the requested column.
        """

        return data.loc[:, column] if column in data.columns else None

    @staticmethod
    def __read_baa_metadata_json(input_fname, metadata_fname):
        """ Meta-data reader

        Read meta data summary json file produced when processing .CWA file
        into X-timeSeries.csv.gz file with the biobank acc. package.

        Parameters
        ----------
        input_fname: str
            Path to the .csv(.gz) file.
        metadata_fname: str
            Path to the meta-data (summary.json) file.

        Returns
        -------
        metadata : dict
            Dictionnary containing the meta-data.
        """

        # read meta-data file (if found):
        if metadata_fname is None:
            input_metadata = input_fname.replace(
                '-timeSeries.csv.gz',
                '-summary.json'
            )
        else:
            input_metadata = os.path.abspath(metadata_fname)

        # load meta data json file
        with open(input_metadata) as file:
            meta_data = json.load(file)

        # check filename consistency:
        if meta_data['file-name'] != os.path.basename(input_fname):
            raise ValueError(
                'Attempting to read a metadata file referring to another '
                + 'input file.\n'
                + '- Input file: {}\n'.format(os.path.basename(input_fname))
                + '- Metadata ref: {}\n'.format(
                    os.path.basename(meta_data['file-name'])
                )
                + '- Metadata path: {}\n'.format(input_metadata)
            )
        return meta_data

    @staticmethod
    def __extract_baa_metadata(meta_data, field):
        """ Meta-data extractor

        Extract meta data from summary json file.

        Parameters
        ----------
        meta_data : dict
            Dictionnary containing the meta-data.
        field: str
            Field (key) to extract.

        Returns
        -------
        value: str or int
            Requested field extracted from the meta-data dict.
        """

        if field in meta_data.keys():
            return meta_data[field]
        else:
            raise KeyError(
                'Information ({}) not found in meta-data file.'.format(field)
            )


def read_raw_bba(
    input_fname,
    name=None,
    uuid=None,
    frequency=None,
    start_time=None,
    period=None,
    engine='c',
    impute_missing=False,
    use_metadata_json=True,
    metadata_fname=None
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
    engine: str, optional
        Parser engine to use. Argument passed to Pandas.
        Default is 'c'.
    impute_missing: bool, optional
        If set to True, use missing data imputation from the biobankanalysis
        package.
        Default is False.
    use_metadata_json: bool, optional.
        If set to True, extract meta-data from summary json file.
        Default is True.
    metadata_fname: str, optional
        Path to the summary json file.
        If None, the path to the summary json file is inferred from the input
        file (/path/to/XXX-timeSeries.csv.gz -> /path/to/XXX-summary.json).
        Default is None.

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
        engine=engine,
        impute_missing=impute_missing,
        use_metadata_json=use_metadata_json,
        metadata_fname=metadata_fname
    )
