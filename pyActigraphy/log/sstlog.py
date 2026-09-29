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
    time_zone: str, optional
        Time zone of time stamps. Ex: 'Europe/Brussels'.
        If set to None, the time stamps in the log file are considered as tz-naive.
        In this case, checking for overlaps between date range and DST times is not possible.
        Default is None.

    """

    def __init__(
        self,
        input_fname,
        log,
        time_zone=None
    ):

        # call __init__ function of the base class
        super().__init__(
            input_fname=input_fname,
            log=log,
            time_zone=time_zone
        )

    def summary(self):
        """ Returns a dataframe of summary statistics."""
        return super(SSTLog, self).summary('Duration')


def read_sst_log(input_fname, time_zone=None, *args, **kwargs):
    """ Read start/stop-times from SST log files.

    Function to read start and stop times from SST log files. Supports
    different file format (.ods, .xls(x), .csv).

    Parameters
    ----------
    input_fname: str
        Path to the log file.
    time_zone: str, optional
        Time zone of time stamps. Ex: 'Europe/Brussels'.
        If set to None, the time stamps in the log file are considered as tz-naive.
        In this case, checking for overlaps between date range and DST times is not possible.
        Default is None.
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

    return SSTLog(input_fname, log, time_zone)
