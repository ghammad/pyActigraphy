# import os
# import numpy as np
# import pandas as pd
# import pyexcel as pxl
from .baselog import BaseLog


class SSTLog(BaseLog):
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

        # call __init__ function of the base class
        super().__init__(
            input_fname=input_fname,
            log=log
        )

    def summary(self):
        """ Returns a dataframe of summary statistics."""
        return super(SSTLog, self).summary('Duration')


def read_sst_log(input_fname, *args, **kwargs):
    """ Read start/stop-times from SST log files.

    Function to read start and stop times from SST log files. Supports
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

    input_fname, log = BaseLog.from_file(
        input_fname=input_fname,
        index_name='Subject_id',
        *args, **kwargs
    )

    return SSTLog(input_fname, log)
