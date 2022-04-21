import pandas as pd
import warnings
# from pandas.tseries.frequencies import to_offset
from .utils import _binarized_data
from .utils import _resampled_data


class BaseRecording():
    """ Base class for any type of recording.

    Parameters
    ----------
    name: str
        Name of the recording.
    data: pandas.DataFrame
        Dataframe containing the data found in the recording.

    """

    def __init__(
        self,
        name,
        uuid,
        data,
        frequency,
        start_time=None,
        stop_time=None,
        period=None,
        mask=None
    ):

        # Mandatory fields
        self.__name = name
        self.__display_name = name
        self.__uuid = uuid
        self.__data = data
        self.__frequency = frequency

        # Optional fields
        # User-specified start/stop/period
        self.__start_time = start_time
        self.__stop_time = stop_time
        self.__period = period

        # Mask-related fields
        self.__mask = mask
        self.__mask_inactivity = False
        self.__inactivity_length = None
        self.__exclude_if_mask = True

    @property
    def name(self):
        r"""Name of the recording."""
        return self.__name

    @property
    def display_name(self):
        r"""Name to be used for display."""
        return self.__display_name

    @display_name.setter
    def display_name(self, value):
        self.__display_name = value

    @property
    def uuid(self):
        r"""UUID of the recording."""
        return self.__uuid

    @property
    def mask_inactivity(self):
        r"""Inactivity mask indicator."""
        return self.__mask_inactivity

    @mask_inactivity.setter
    def mask_inactivity(self, value):
        self.__mask_inactivity = value

    @property
    def start_time(self):
        r"""Start time of the recording."""
        return self.__start_time

    @start_time.setter
    def start_time(self, value):
        if (self.__stop_time is not None) and (self.__period is not None):
            raise ValueError(
                'Stop time and period fields have already been set.\n'
                + 'Use the reset_times function first.'
            )
        self.__start_time = pd.Timestamp(value)

    @property
    def stop_time(self):
        r"""Stop time of the recording."""
        return self.__stop_time

    @stop_time.setter
    def stop_time(self, value):
        if (self.__start_time is not None) and (self.__period is not None):
            raise ValueError(
                'Start time and period fields have already been set.\n'
                + 'Use the reset_times function first.'
            )
        self.__stop_time = pd.Timestamp(value)

    @property
    def period(self):
        r"""Time period of the recording."""
        return self.__period

    @period.setter
    def period(self, value):
        if (self.__start_time is not None) and (self.__stop_time is not None):
            raise ValueError(
                'Start and stop time fields have already been set.\n'
                + 'Use the reset_times function first.'
            )
        self.__period = pd.Timedelta(value)
        # Optionally compute start or stop time
        if self.__start_time is not None:
            self.stop_time = self.__start_time + self.__period
        elif self.__stop_time is not None:
            self.start_time = self.__stop_time - self.__period

    def reset_times(self):
        r"""Reset start and stop times, as well as the period of the recording.
        """
        self.__start_time = None
        self.__stop_time = None
        self.__period = None

    # def check_times(self):
    #     r"""Check the start time, the stop time and the period of the
    #     recording.
    #     If, at least, two out of the three parameters are present,
    #     it calculates the third one and returns True. False otherwise.
    #     """
    #     if (self.__start_time is not None) and ()
    #     self.__start_time = None
    #     self.__stop_time = None
    #     self.__period = None

    @property
    def frequency(self):
        r"""Acquisition frequency of the recording."""
        return self.__frequency

    @property
    def raw_data(self):
        r"""Indexed data extracted from the raw file."""
        return self.__data

    # TODO: @lru_cache(maxsize=6) ???
    @property
    def data(self):
        r"""Data of the recording.

        If mask_inactivity is set to true, the `mask` is used
        to filter out data.
        """
        if self.__data is None:
            return self.__data

        if self.mask_inactivity is True:
            if self.mask is not None:
                data = self.raw_data.where(self.mask > 0)
            else:
                warnings.warn(
                    (
                        'Mask inactivity set to True but no mask could be'
                        ' found.\n Please create a mask by using the '
                        '"create_inactivity_mask" function.'
                    ),
                    UserWarning
                )
                data = self.raw_data
        else:
            data = self.raw_data
        return data[self.start_time:self.stop_time]

    def resampled_data(self, rsfreq, agg='sum'):

        return _resampled_data(self.data, rsfreq=rsfreq, agg=agg)

    def binarized_data(self, threshold, rsfreq=None, agg='sum'):

        if rsfreq is None:
            rsdata = self.data
        else:
            rsdata = _resampled_data(self.data, rsfreq=rsfreq, agg=agg)

        return _binarized_data(rsdata, threshold)

    # TODO: @lru_cache(maxsize=6) ???
    # def resampled_data(self, freq, binarize=False, threshold=0):
    #     r"""Data resampled at the specified frequency.
    #     If mask_inactivity is True, the `mask` is used to filter inactive data.
    #     """
    #     if binarize is False:
    #         data = self.data
    #     else:
    #         data = self.binarized_data(threshold)
    #
    #     if freq is None:
    #         return data
    #     elif to_offset(freq).delta < self.frequency:
    #         warnings.warn(
    #             'Resampling frequency lower than the acquisition'
    #             + ' frequency. Returning original data.',
    #             UserWarning
    #         )
    #         return data
    #     elif to_offset(freq).delta == self.frequency:
    #         return data
    #
    #     resampled_data = data.resample(freq).sum()
    #     if self.mask_inactivity is True:
    #         if self.mask is None:
    #             warnings.warn(
    #                 (
    #                     'Mask inactivity set to True but no mask could be'
    #                     ' found.\n Please create a mask by using the '
    #                     '"create_inactivity_mask" function.'
    #                 ),
    #                 UserWarning
    #             )
    #             return resampled_data
    #         elif self.exclude_if_mask:
    #             resampled_mask = self.mask.resample(freq).min()
    #         else:
    #             resampled_mask = self.mask.resample(freq).max()
    #         return resampled_data.where(resampled_mask > 0)
    #     else:
    #         return resampled_data
