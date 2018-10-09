import os
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

        # Read data from the log file
        sst_array = pxl.get_array(file_name=self.__fname)

        # TODO: check if the log file is valid
        # i.e at least 3 columns with the correct types

        # Discard columns beyond the subject_id, start and stop time.
        fsst_array = [subarray[:3] for subarray in sst_array]

        log = pd.DataFrame(
            fsst_array[header_size:],
            columns=fsst_array[header_size-1]).astype({
                fsst_array[0][0]: 'str',
                fsst_array[0][1]: 'datetime64[ns]',
                fsst_array[0][2]: 'datetime64[ns]'
            })

        # Add `duration` column
        log['duration'] = log[fsst_array[0][2]] - log[fsst_array[0][1]]

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
        return self.__log['duration'].describe()
