import glob

from pyActigraphy.metrics import ForwardMetricsMixin
from joblib import Parallel, delayed
from ..awd import read_raw_awd
from ..mtn import read_raw_mtn
from ..rpx import read_raw_rpx


class RawReader(ForwardMetricsMixin):
    """Reader for multiple Raw files

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

    @property
    def reader_type(self):
        """The type of RawXXX readers."""
        return self.__reader_type

    @property
    def readers(self):
        """The list of RawXXX readers."""
        return self.__readers

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


def read_raw(input_path, reader_type, n_jobs=1):
        """Reader function for multiple raw files.

        Parameters
        ----------
        input_path: str
            Path to the files. Accept wild cards.
            E.g. '/path/to/my/files/*.csv'
        reader_type: str
            Reader type.
            Supported types: AWD (ActiWatch), MTN (MotionWatch8)
            and RPX (Respironics)

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

        def parallel_reader(n_jobs, read_func, file_list):
            return Parallel(n_jobs=n_jobs)(
                delayed(read_func)(file) for file in file_list
            )

        readers = {
            'AWD': lambda files: parallel_reader(n_jobs, read_raw_awd, files),
            'MTN': lambda files: parallel_reader(n_jobs, read_raw_mtn, files),
            'RPX': lambda files: parallel_reader(n_jobs, read_raw_rpx, files)
        }[reader_type](files)

        return RawReader(readers, reader_type)
