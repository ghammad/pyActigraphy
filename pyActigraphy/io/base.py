import pandas as pd
import numpy as np
import warnings

from pandas.tseries.frequencies import to_offset
from ..filters import FiltersMixin
from ..metrics import MetricsMixin
from ..sleep import SleepDiary, ScoringMixin


class BaseRaw(ScoringMixin, MetricsMixin, FiltersMixin):
    """Base class for raw data."""
    def __init__(
        self,
        name,
        uuid,
        format,
        axial_mode,
        start_time,
        frequency,
        data,
        light
    ):

        self.__name = name
        self.__display_name = name
        self.__uuid = uuid
        self.__format = format
        self.__axial_mode = axial_mode
        self.__start_time = start_time
        self.__frequency = frequency
        self.__data = data

        self.__light = light

        self.__mask_inactivity = False
        self.__inactivity_length = 120
        self.__mask = None
        self.__exclude_if_mask = True

        self.__sleep_diary = None

    @property
    def name(self):
        """The study name as extracted from the raw file."""
        return self.__name

    @property
    def display_name(self):
        """The name to be used for display."""
        return self.__display_name

    @display_name.setter
    def display_name(self, value):
        self.__display_name = value

    @property
    def uuid(self):
        """The UUID of the device used to acquire the data"""
        return self.__uuid

    @property
    def format(self):
        """The format of the raw data file (AWD,RPX,MTN,...)"""
        return self.__format

    @property
    def axial_mode(self):
        """The acquistion mode (mono-axial or tri-axial)"""
        return self.__axial_mode

    @property
    def start_time(self):
        """The start time of data acquistion as extracted from the raw file."""
        return self.__start_time

    @property
    def frequency(self):
        """The acquisition frequency as extracted from the raw file."""
        return self.__frequency

    @property
    def raw_data(self):
        """The indexed data extracted from the raw file."""
        return self.__data

    # TODO: @lru_cache(maxsize=6) ???
    @property
    def data(self):
        """The indexed data extracted from the raw file.
        If mask_inactivity is set to true, the `mask` is used
        to filter out inactive data.
        """
        if self.mask_inactivity is True:
            data = self.raw_data.where(self.mask > 0)
        else:
            data = self.raw_data
        return data

    @property
    def raw_light(self):
        """The light measurement performed by the device"""
        return self.__light

    # TODO: @lru_cache(maxsize=6) ???
    @property
    def light(self):
        """The indexed light extracted from the raw file.
        If mask_inactivity is set to true, the `mask` is used
        to filter out inactive data.
        """
        if self.mask_inactivity is True:
            return self.raw_light.where(self.mask > 0)
        else:
            return self.raw_light

    @property
    def mask_inactivity(self):
        """ The switch to mask inactive data."""
        return self.__mask_inactivity

    @mask_inactivity.setter
    def mask_inactivity(self, value):
        self.__mask_inactivity = value

    @property
    def inactivity_length(self):
        """ The length of the inactivity mask."""
        return self.__inactivity_length

    @inactivity_length.setter
    def inactivity_length(self, value):
        self.__inactivity_length = value

    @property
    def mask(self):
        """ The mask used to filter out inactive data."""
        if self.__mask is None:
            # Create a mask if it does not exist
            if self.inactivity_length is not None:
                self.create_inactivity_mask(self.inactivity_length)
            else:
                warnings.warn(
                    'Inactivity length set to None. Could not create a mask.',
                    UserWarning
                )

        return self.__mask

    @mask.setter
    def mask(self, value):
        self.__mask = value

    @property
    def exclude_if_mask(self):
        return self.__exclude_if_mask

    @exclude_if_mask.setter
    def exclude_if_mask(self, value):
        self.__exclude_if_mask = value

    def mask_fraction(self):
        return 1.-(self.mask.sum()/len(self.mask))

    def length(self):
        """ Number of data acquisition points"""
        return len(self.data)

    def time_range(self):
        """ Range (in days, hours, etc) of the data acquistion period"""
        return (self.data.index[-1]-self.data.index[0])

    def duration(self):
        """ Duration (in days, hours, etc) of the data acquistion period"""
        return self.frequency * self.length()

    def binarized_data(self, threshold):
        """Boolean thresholding of Pandas Series"""
        return pd.Series(
            np.where(self.data > threshold, 1, 0),
            index=self.data.index
        )

    # TODO: @lru_cache(maxsize=6) ???
    def resampled_data(self, freq, binarize=False, threshold=0):
        """The data resampled at the specified frequency.
        If mask_inactivity is True, the `mask` is used to filter inactive data.
        """
        if binarize is False:
            data = self.data
        else:
            data = self.binarized_data(threshold)

        if to_offset(freq).delta <= self.frequency:
            warnings.warn(
                'Resampling frequency equal to or lower than the acquisition' +
                ' frequency. Returning original data.',
                UserWarning
            )
            return data

        resampled_data = data.resample(freq).sum()
        if self.mask_inactivity is True:
            if self.exclude_if_mask:
                resampled_mask = self.mask.resample(freq).min()
            else:
                resampled_mask = self.mask.resample(freq).max()
            return resampled_data.where(resampled_mask > 0)
        else:
            return resampled_data

    # TODO: @lru_cache(maxsize=6) ???
    def resampled_light(self, freq):
        """The light meeasurement, resampled at the specified frequency.
        """
        light = self.light

        if to_offset(freq).delta <= self.frequency:
            warnings.warn(
                'Resampling frequency equal to or lower than the acquisition' +
                ' frequency. Returning original data.',
                UserWarning
            )
            return light
        else:
            return light.resample(freq).sum()

    def read_sleep_diary(
            self,
            input_fname,
            header_size=2,
            state_index=dict(ACTIVE=2, NAP=1, NIGHT=0, NOWEAR=-1),
            state_colour=dict(
                NAP='#7bc043',
                NIGHT='#d3d3d3',
                NOWEAR='#ee4035'
            )
    ):
        """Reader function for sleep diaries.

        Parameters
        ----------
        input_fname: str
            Path to the sleep diary file.
        header_size: int
            Header size (i.e. number of lines) of the sleep diary.
            Default is 2.
        state_index: dict
            The dictionnary of state's indices.
            Default is ACTIVE=2, NAP=1, NIGHT=0, NOWEAR=-1.
        state_color: dict
            The dictionnary of state's colours.
            Default is NAP='#7bc043', NIGHT='#d3d3d3', NOWEAR='#ee4035'.
        """

        self.__sleep_diary = SleepDiary(
            input_fname=input_fname,
            start_time=self.start_time,
            periods=self.length(),
            frequency=self.frequency,
            header_size=header_size,
            state_index=state_index,
            state_colour=state_colour
        )

    @property
    def sleep_diary(self):
        """ The SleepDiary class instanciation."""
        return self.__sleep_diary
