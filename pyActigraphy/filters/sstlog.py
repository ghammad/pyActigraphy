import os
import numpy as np
import pandas as pd
import pyexcel as pxl


class SSTLog():
    """ Class for reading start/stop-time log files

    Data structure for start and stop times.

    Parameters
    ----------
    fname: str
        Absolute filepath of the input log file.
    log: pandas.DataFrame
        Dataframe containing the data found in the SST log file.

    """

    def __init__(
        self,
        input_fname,
        log
    ):

        # get absolute file path
        self.__fname = os.path.abspath(input_fname)

        # Add `duration` column
        log['Duration'] = log['Stop_time'] - log['Start_time']

        # Inplace drop of NA
        log.dropna(inplace=True)

        self.__log = log

    @classmethod
    def from_file(cls, input_fname, *args, **kwargs):
        """ Read start/stop-times from log files.

        Generic function to read start and stop times from log files. Supports
        different file format (.ods, .xls(x), .csv).

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        *args
            Variable length argument list passed to the subsequent reader
            function.
        **kwargs
            Arbitrary keyword arguments passed to the subsequent reader
            function.

        Returns
        -------
        sstlog : SSTLog
            An instance of the SSTLog class

        """

        # get absolute file path
        absname = os.path.abspath(input_fname)

        # get basename and split it into base and extension
        basename = os.path.basename(absname)
        name, ext = os.path.splitext(basename)

        if(ext == '.csv'):
            log = cls.from_csv(absname, *args, **kwargs)
        elif((ext == '.xlsx') or (ext == '.xls') or (ext == '.ods')):
            log = cls.from_excel(absname, *args, **kwargs)
        else:
            raise ValueError(
                (
                    'File format for the input file {}'.format(basename) +
                    'is not currently supported.' +
                    'Supported file format:\n' +
                    '.csv (text),\n' +
                    '.ods (OpenOffice spreadsheet),\n' +
                    '.xls (Excel spreadsheet).'
                )
            )
        return SSTLog(absname, log)

    @classmethod
    def from_csv(cls, input_fname, sep=',', dayfirst=False):
        """ Read start/stop-times from .csv files.

        Specific function to read start and stop times from csv files.

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        sep: str, optional
            Delimiter to use.
            Default is ','.
        dayfirst: bool, optional
            If set to True, use DD/MM/YYYY format dates.
            Default is False.

        Returns
        -------
        log : a pandas.DataFrame
            A dataframe with the start and stop times (columns)

        """

        # Read data from the csv file into a dataframe
        log = pd.read_csv(
            input_fname,
            sep=sep,
            dayfirst=dayfirst,
            header=0,
            index_col=[0],
            usecols=[0, 1, 2],
            names=['Subject_id', 'Start_time', 'Stop_time'],
            parse_dates=[1, 2],
            infer_datetime_format=False
        )
        return log

    @classmethod
    def from_excel(cls, input_fname):
        """ Read start/stop-times from excel-like files.

        Specific function to read start and stop times from .ods/.xls(x) files.

        Parameters
        ----------
        input_fname: str
            Path to the log file.

        Returns
        -------
        log : a pandas.DataFrame
            A dataframe with the start and stop times (columns)

        """

        # Read data from the log file into a np array
        sst_narray = np.array(pxl.get_array(file_name=input_fname))

        # Create a DF with columns: subject_id, start_time, stop time
        log = pd.DataFrame(
            sst_narray[1:, 1:3],
            index=sst_narray[1:, 0],
            columns=['Start_time', 'Stop_time'],
            dtype='datetime64[ns]'
        )
        log.index.name = 'Subject_id'

        return log

    @property
    def fname(self):
        """The absolute filepath of the input log file."""
        return self.__fname

    @property
    def log(self):
        """The dataframe containing the data found in the SST log file."""
        return self.__log

    def summary(self):
        """ Returns a dataframe of summary statistics."""
        return self.__log['Duration'].describe()
