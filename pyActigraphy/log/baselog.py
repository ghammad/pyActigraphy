import os
import numpy as np
import pandas as pd
import pyexcel as pxl
from pytz import timezone


class BaseLog():
    """ Base class for log files containing time stamps.

    Parameters
    ----------
    fname: str
        Absolute filepath of the input log file.
    log: pandas.DataFrame
        Dataframe containing the data found in the log file.
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

        # get absolute file path
        self.__fname = os.path.abspath(input_fname)

        # Add `duration` column
        log['Duration'] = log['Stop_time'] - log['Start_time']

        # Add DST check
        if time_zone is not None:
            from pytz import timezone
            tz = timezone(time_zone)
            # Localize the log timestamps
            log.loc[:,'Start_time'] = log.Start_time.dt.tz_localize(tz)
            log.loc[:,'Stop_time'] = log.Stop_time.dt.tz_localize(tz)
            # compute the DST transition times according to the specified time zone
            transition_times = [
                t.replace(tzinfo=timezone('UTC')).astimezone(tz)
                for t in tz._utc_transition_times[1:]
            ]
            # Dictionary mapping the transition times to the corresponding year                                                                                                                                                    
            transition_times_by_year = {
                start_time.year: [start_time, stop_time]
                for start_time, stop_time in zip(
                    transition_times[::2], transition_times[1::2]
                    )
            }

            log.loc[:,'DST_crossover'] = log.apply(
                self.__is_straddling_dst_transitions,
                dst_transition_times_dict=transition_times_by_year,
                axis=1
            )


        # Inplace drop of NA
        log.dropna(inplace=True)

        # add dataframe
        self.__log = log

    @classmethod
    def from_file(cls, input_fname, index_name, *args, **kwargs):
        """ Read start/stop-times from log files.

        Generic function to read start and stop times from log files. Supports
        different file format (.ods, .xls(x), .csv).

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        index_name: str
            Name of the index.
        *args
            Variable length argument list passed to the subsequent reader
            function.
        **kwargs
            Arbitrary keyword arguments passed to the subsequent reader
            function.

        Returns
        -------
        absname: str
            Absolute filepath of the input log file.
        log: pandas.DataFrame
            Dataframe containing the data found in the log file.

        """

        # get absolute file path
        absname = os.path.abspath(input_fname)

        # get basename and split it into base and extension
        basename = os.path.basename(absname)
        _, ext = os.path.splitext(basename)

        if(ext == '.csv'):
            log = cls.__from_csv(absname, index_name, *args, **kwargs)
        elif((ext == '.xlsx') or (ext == '.xls') or (ext == '.ods')):
            log = cls.__from_excel(absname, index_name, *args, **kwargs)
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
        return absname, log

    @classmethod
    def __from_csv(cls, input_fname, index_name, sep=',', dayfirst=False):
        """ Read start/stop-times from .csv files.

        Specific function to read start and stop times from csv files.

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        index_name: str
            Name of the index.
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
            names=[index_name, 'Start_time', 'Stop_time'],
            parse_dates=[1, 2],
            infer_datetime_format=False
        )
        return log

    @classmethod
    def __from_excel(cls, input_fname, index_name):
        """ Read start/stop-times from excel-like files.

        Specific function to read start and stop times from .ods/.xls(x) files.

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        index_name: str
            Name of the index.

        Returns
        -------
        log : a pandas.DataFrame
            A dataframe with the start and stop times (columns)

        """

        # Read data from the log file into a np array
        sst_narray = np.array(pxl.get_array(file_name=input_fname))

        # Create a DF with columns: index_name, start_time, stop time
        log = pd.DataFrame(
            sst_narray[1:, 1:3],
            index=sst_narray[1:, 0],
            columns=['Start_time', 'Stop_time']
            # dtype='datetime64[ns]'
        )
        log.index.name = index_name

        return log

    @staticmethod
    def __is_straddling_dst_transitions(times, dst_transition_times_dict):
        """ Verify if the date range includes a DST transition time

        Function designed to be applied to 'log' dataframe in order to check
        if the date range between the specified start and stop times
        do include a DST (either CET->CEST or CEST->CET) transition.

        Parameters
        ----------
        times: pd.Series
            Series containing the following indices: 'Start_time' and 'Stop_time'.
        dst_transition_times_dict: dict
            Dictionary with years as keys and the corresponding DST transition datetimes as values (2-tuple).

        Returns
        -------
        dst: bool
            True if the date range straddles a DST. False otherwise.

        """
    
        # Extract the year of the start AND stop times
        year_start = times['Start_time'].year
        year_stop = times['Stop_time'].year
        
        # Check if start-stop range spans over new year's eve
        span_nye = (year_start!=year_stop)
        
        # Shortcut: if the recording is longer than a year, then
        # it has crossed DST transition times at least once.
        # NB: actually it is already the case for 7-month long recordings
        if (times['Stop_time']-times['Start_time']) >= pd.Timedelta('365D'):
            return True
        
        if not span_nye:
            # Simple case: 
            # the date range should only be tested with DST
            # transition times of the current (i.e start) year
            cet2cest, cest2cet = dst_transition_times_dict[year_start]

            # Does the date range contain CET->CEST:
            isCET2CEST = (times['Start_time'] <= cet2cest <= times['Stop_time'])

            # Does the date range contain CEST->CET:
            isCEST2CET = (times['Start_time'] <= cest2cet <= times['Stop_time'])

            return (isCET2CEST or isCEST2CET)
        else:
            # Not so simple case:
            # For (very) long recordings, spanning a New Year's Eve,
            # the date range should also be tested with DST transition times
            # of the next year (year stop)
            cet2cest_start, cest2cet_start = transition_times_by_year[year_start]
            cet2cest_stop, _ = transition_times_by_year[year_stop]

            # Does the date range contain CET->CEST (Year=N):
            isCET2CEST_start = (times['Start_time'] <= cet2cest_start <= times['Stop_time'])

            # Does the date range contain CEST->CET (Year=N):
            isCEST2CET_start = (times['Start_time'] <= cest2cet_start <= times['Stop_time'])

            # Does the date range contain CET->CEST (Year=N+1):
            isCET2CEST_stop = (times['Start_time'] <= cet2cest_stop <= times['Stop_time'])

            return (isCET2CEST_start or isCEST2CET_start or isCET2CEST_stop)

            year = times['Start_time'].year
            cet2cest, cest2cet = transition_times_by_year[year]
            
            # Does the date range contain CET->CEST:
            isCET2CEST = (times['Start_time'] <= cet2cest <= times['Stop_time'])
            
            # Does the date range contain CEST->CET:
            isCEST2CET = (times['Start_time'] <= cest2cet <= times['Stop_time'])
            
            return (isCET2CEST or isCEST2CET)        

    @property
    def fname(self):
        """The absolute filepath of the input log file."""
        return self.__fname

    @property
    def log(self):
        """The dataframe containing the data found in the log file."""
        return self.__log

    def summary(self, colname):
        """ Returns a dataframe of summary statistics."""
        return self.__log[colname].describe()
