import glob
import numpy as np
import warnings

from pandas import Timedelta
from pandas.tseries.frequencies import to_offset
from pyActigraphy.metrics import ForwardMetricsMixin
from joblib import Parallel, delayed
from ..awd import read_raw_awd
from ..mtn import read_raw_mtn
from ..rpx import read_raw_rpx
from pyActigraphy.filters import SSTLog


class RawReader(ForwardMetricsMixin):
    r"""Reader for multiple Raw files

    Parameters
    ----------
    readers: list
        List of instances of RawXXX classes
    """

    def __init__(self, reader_type, readers=[]):

        # store reader_type
        self.__reader_type = reader_type

        # store list of readers
        self.__readers = readers

        self.__sst_log = None

    @property
    def reader_type(self):
        r"""The type of RawXXX readers."""
        return self.__reader_type

    @property
    def readers(self):
        r"""The list of RawXXX readers."""
        return np.asarray(self.__readers)

    def append(self, raw_reader):
        if raw_reader.format != self.__reader_type:
            # [TODO]: use isinstance instead
            # and allow multiple types for the same reader
            raise TypeError('Wrong reader type: {}. Should be: {}'.format(
                raw_reader.format,
                self.__reader_type
            ))
        else:
            self.__readers.append(raw_reader)

    def mask_fraction(self):

        return {
            read.display_name: read.mask_fraction() for read in self.__readers
        }

    def start_time(self):

        return {
            iread.display_name: iread.start_time for iread in self.__readers
        }

    def duration(self):

        return {
            iread.display_name: iread.duration() for iread in self.__readers
        }

    def resampled_data(
        self, freq,
        binarize=False, threshold=0,
        n_jobs=1, prefer=None, verbose=0
    ):
        r"""Data resampled at the specified frequency.
        If mask_inactivity is True, inactive data (0 count) are masked.
        """

        # Check if resampling is possible

        # 1. check if resampling freq is lower than th lowest acquistion freq.
        freq = to_offset(freq).delta
        freqs = [reader.frequency for reader in self.readers]
        if freq <= min(freqs):
            warnings.warn(
                'Resampling frequency equal to or lower than the lowest' +
                ' acquisition frequency found in the list of readers.' +
                ' Returning original data.',
                UserWarning
            )
            return self.readers

        # 2. check if resampling freq is a multiple of the acquistion freq.
        if False in [(freq % ifreq) == Timedelta(0) for ifreq in freqs]:
            warnings.warn(
                'Resampling frequency is *not* a multiple of the' +
                ' acquisition frequencies found in the list of readers.' +
                ' Returning original data.',
                UserWarning
            )
            return self.readers

        def parallel_resampling(rawReader, freq, binarize, threshold):
            return (
                rawReader.display_name,
                rawReader.resampled_data(freq, binarize, threshold)
            )

        return dict(Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose)(
            delayed(parallel_resampling)(
                rawReader=reader,
                freq=freq,
                binarize=binarize,
                threshold=threshold
            ) for reader in self.readers
        ))

    def read_sst_log(
            self,
            input_fname,
            header_size=1
    ):
        r"""Reader function for start/stop-time log files.

        Parameters
        ----------
        input_fname: str
            Path to the start/stop-time log file.
        header_size: int
            Header size (i.e. number of lines) of the sst log file.
            Default is 1.
        """

        self.__sst_log = SSTLog(
            input_fname=input_fname,
            header_size=header_size
        )

    @property
    def sst_log(self):
        r""":class:`SSTLog` class instanciation."""
        return self.__sst_log

    def apply_sst(self, verbose=False):
        r"""Set start time and duration in all readers """
        if self.sst_log is not None:
            for reader in self.readers:
                if reader.display_name in self.sst_log.log.index:
                    if verbose:
                        print('Found an entry in SST log file for '
                              '{}'.format(reader.display_name))
                    # Retrieve start_time and duration from sst_log by position
                    # in order to avoid mis-spelling of the index names.
                    start_time = self.sst_log.log.loc[reader.display_name][0]
                    period = self.sst_log.log.loc[reader.display_name][2]
                    reader.start_time = start_time
                    reader.period = period
                else:
                    if verbose:
                        print('Could not find an entry in SST log file for '
                              '{}'.format(reader.display_name))
        else:
            print('Could not find a SST log file. Please run the read_sst_log'
                  ' function before using the `apply_sst` function.')


def read_raw(input_path, reader_type, n_jobs=1, prefer=None, verbose=0):
        r"""Reader function for multiple raw files.

        Parameters
        ----------
        input_path: str
            Path to the files. Accept wild cards.
            E.g. '/path/to/my/files/*.csv'
        reader_type: str
            Reader type.
            Supported types: AWD (ActiWatch), MTN (MotionWatch8)
            and RPX (Respironics)
        n_jobs: int
            Number of CPU to use for parallel reading
        prefer: str
            Soft hint to choose the default backendself.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose: int
            Display a progress meter if set to a value > 0.
            Default is 0.

        Returns
        -------
        raw : list
            A list of instances of RawAWD, RawMTN or RawRPX
        """

        supported_types = ['AWD', 'MTN', 'RPX']
        if reader_type not in supported_types:
            raise ValueError(
                'Type {0} unsupported. Supported types: {1}'.format(
                    reader_type, supported_types
                )
            )

        files = glob.glob(input_path)

        def parallel_reader(
            n_jobs, read_func, file_list, prefer=None, verbose=0
        ):
            return Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose)(
                delayed(read_func)(file) for file in file_list
            )

        readers = {
            'AWD': lambda files: parallel_reader(
                n_jobs, read_raw_awd, files, prefer, verbose
            ),
            'MTN': lambda files: parallel_reader(
                n_jobs, read_raw_mtn, files, prefer, verbose
            ),
            'RPX': lambda files: parallel_reader(
                n_jobs, read_raw_rpx, files, prefer, verbose
            )
        }[reader_type](files)

        return RawReader(reader_type, readers)
