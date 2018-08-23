import glob

from pyActigraphy.metrics import ForwardMetricsMixin
from joblib import Parallel, delayed
from ..awd import read_raw_awd
from ..rpx import read_raw_rpx


class RawReader(ForwardMetricsMixin):
    """Reader for multiple Raw files

    Parameters
    ----------
    readers: list
        List of instances of RawXXX classes
    """

    def __init__(self, readers, reader_type):

        # store list of readers
        self.__readers = readers

        # store reader_type
        self.__reader_type = reader_type

    @property
    def readers(self):
        """The list of RawXXX readers."""
        return self.__readers

    @property
    def reader_type(self):
        """The type of RawXXX readers."""
        return self.__reader_type

    # def IS(self):
    #     return [reader.IS() for reader in self.__readers]


def read_raw(input_path, reader_type, n_jobs=1):
        """Reader function for multiple raw files.

        Parameters
        ----------
        input_path: str
            Path to the files. Accept wild cards.
            E.g. '/path/to/my/files/*.csv'
        reader_type: str
            Reader type.
            Supported types: AWD (MotionWatch) and RPX (Respironics)

        Returns
        -------
        raw : list
            A list of instances of RawAWD or RawRPX
        """

        supported_types = ['AWD', 'RPX']
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
            'RPX': lambda files: parallel_reader(n_jobs, read_raw_rpx, files)
            # 'AWD': lambda files: [read_raw_awd(ifile) for ifile in files],
            # 'RPX': lambda files: [read_raw_rpx(ifile) for ifile in files]
        }[reader_type](files)

        return RawReader(readers, reader_type)
