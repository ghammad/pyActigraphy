import numpy as np
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
        log10_transform=True,
        mask=None
    ):

        # Mandatory fields
        self.__name = name
        self.__display_name = name
        self.__uuid = uuid
        self.__data = np.log10(data+1) if log10_transform else data
        self.__frequency = frequency

        # Optional fields
        # User-specified start/stop/period
        self.__start_time = start_time
        self.__stop_time = stop_time
        self.__period = period

        # Mask-related fields
        self.__mask = mask
        self.__apply_mask = False
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
    def apply_mask(self):
        r"""Mask indicator."""
        return self.__apply_mask

    @apply_mask.setter
    def apply_mask(self, value):
        self.__apply_mask = value

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
        # NB one epoch has to be suppressed as accessing data with .loc[]
        # includes the last epoch.
        if self.__start_time is not None:
            self.__stop_time = self.__start_time + \
                (self.__period - self.frequency)
        elif self.__stop_time is not None:
            self.__start_time = self.__stop_time - \
                (self.__period - self.frequency)

    def reset_times(self):
        r"""Reset start and stop times, as well as the period of the recording.
        """
        self.__start_time = None
        self.__stop_time = None
        self.__period = None

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

        If apply_mask is set to true, the `mask` is used
        to filter out data.
        """
        if self.__data is None:
            return self.__data

        if self.apply_mask is True:
            if self.mask is not None:
                data = self.raw_data.where(self.mask > 0)
            else:
                warnings.warn(
                    (
                        'Apply_mask set to True but no mask could be'
                        ' found.\n Please create a mask by using the'
                        ' appropriate "create_mask" function.'
                    ),
                    UserWarning
                )
                data = self.raw_data
        else:
            data = self.raw_data
        return data.loc[self.start_time:self.stop_time]

    # TODO: @lru_cache(maxsize=6) ???
    def resampled_data(self, rsfreq, agg='sum'):

        return _resampled_data(self.data, rsfreq=rsfreq, agg=agg)

    # TODO: @lru_cache(maxsize=6) ???
    def binarized_data(self, threshold, rsfreq=None, agg='sum'):

        if rsfreq is None:
            rsdata = self.data
        else:
            rsdata = _resampled_data(self.data, rsfreq=rsfreq, agg=agg)

        return _binarized_data(rsdata, threshold)

    @property
    def mask(self):
        r"""Mask used to filter out data."""
        return self.__mask

    @mask.setter
    def mask(self, value):
        self.__mask = value
