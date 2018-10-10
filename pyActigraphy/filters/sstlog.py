import os
import numpy as np
import pandas as pd
import pyexcel as pxl


class SSTLog():
    """Class for reading start/stop-time log files."""
    def __init__(
        self,
        input_fname,
        header_size=1
    ):

        # get absolute file path
        self.__fname = os.path.abspath(input_fname)

        # Read data from the log file into a np array
        sst_narray = np.array(pxl.get_array(file_name=self.__fname))

        # TODO: check if the log file is valid
        # i.e at least 3 columns with the correct types

        # Create a DF with columns: subject_id, start_time, stop time
        log = pd.DataFrame(
            sst_narray[1:, 1:3],
            index=sst_narray[1:, 0],
            columns=sst_narray[0, 1:3]).astype({
                sst_narray[0, 1]: 'datetime64[ns]',
                sst_narray[0, 2]: 'datetime64[ns]'
                })

        # Add `duration` column
        log['Duration'] = log[sst_narray[0, 2]] - log[sst_narray[0, 1]]

        # Inplace drop of NA
        log.dropna(inplace=True)

        self.__log = log

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
