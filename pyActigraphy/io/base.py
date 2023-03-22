import pandas as pd
import numpy as np
import warnings

from pandas.tseries.frequencies import to_offset
from ..filters import FiltersMixin
from ..metrics import MetricsMixin, _interval_maker
from ..reports import ActivityReport, create_sleep_report
from ..sleep import SleepDiary, ScoringMixin, SleepBoutMixin


class BaseRaw(SleepBoutMixin, ScoringMixin, MetricsMixin, FiltersMixin):
    """Base class for raw data."""

    def __init__(
        self,
        name,
        uuid,
        format,
        axial_mode,
        start_time,
        period,
        frequency,
        data,
        light,
        fpath=None
    ):

        self.__fpath = fpath
        self.__name = name
        self.__display_name = name
        self.__uuid = uuid
        self.__format = format
        self.__axial_mode = axial_mode
        self.__start_time = start_time
        self.__period = period
        self.__frequency = frequency
        self.__data = data

        self.__light = light

        self.__mask_inactivity = False
        self.__inactivity_length = None
        self.__mask = None
        self.__exclude_if_mask = True

        self.__sleep_diary = None

    @property
    def fpath(self):
        r"""Absolute path of the raw input file."""
        return self.__fpath

    @property
    def name(self):
        r"""Study name as extracted from the raw file."""
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
        r"""UUID of the device used to acquire the data"""
        return self.__uuid

    @property
    def format(self):
        r"""Format of the raw data file (AWD,RPX,MTN,...)"""
        return self.__format

    @property
    def axial_mode(self):
        r"""Acquistion mode (mono-axial or tri-axial)"""
        return self.__axial_mode

    @property
    def start_time(self):
        r"""Start time of data acquistion as extracted from the raw file or
        specified by the user."""
        return self.__start_time

    @start_time.setter
    def start_time(self, value):
        self.__start_time = value

    @property
    def period(self):
        r"""Period of data acquistion as extracted from the raw file or
        specified by the user."""
        return self.__period

    @period.setter
    def period(self, value):
        self.__period = value

    @property
    def frequency(self):
        r"""Acquisition frequency as extracted from the raw file."""
        return self.__frequency

    @property
    def raw_data(self):
        r"""Indexed data extracted from the raw file."""
        return self.__data

    # TODO: @lru_cache(maxsize=6) ???
    @property
    def data(self):
        r"""Indexed data extracted from the raw file.
        If mask_inactivity is set to true, the `mask` is used
        to filter out inactive data.
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
        return data.loc[self.start_time:self.start_time+self.period]

    @property
    def raw_light(self):
        r"""Light measurement performed by the device"""
        return self.__light

    # TODO: @lru_cache(maxsize=6) ???
    @property
    def light(self):
        r"""Light measurement performed by the device"""
        return self.__light

    @property
    def mask_inactivity(self):
        r"""Switch to mask inactive data."""
        return self.__mask_inactivity

    @mask_inactivity.setter
    def mask_inactivity(self, value):
        self.__mask_inactivity = value

    @property
    def inactivity_length(self):
        r"""Length of the inactivity mask."""
        return self.__inactivity_length

    @inactivity_length.setter
    def inactivity_length(self, value):
        self.__inactivity_length = value
        # Discard current mask (will be recreated upon access if needed)
        self.mask = None
        # Set switch to False if None
        if value is None:
            self.mask_inactivity = False

    @property
    def mask(self):
        r"""Mask used to filter out inactive data."""
        if self.__mask is None:
            # Create a mask if it does not exist
            if self.inactivity_length is not None:
                self.create_inactivity_mask(self.inactivity_length)
                return self.__mask.loc[
                    self.start_time:self.start_time+self.period
                ]
            else:
                warnings.warn(
                    'Inactivity length set to None. Could not create a mask.',
                    UserWarning
                )
        else:
            return self.__mask.loc[self.start_time:self.start_time+self.period]

    @mask.setter
    def mask(self, value):
        self.__mask = value

    @property
    def exclude_if_mask(self):
        r"""Boolean to exclude partially masked data when resampling"""
        return self.__exclude_if_mask

    @exclude_if_mask.setter
    def exclude_if_mask(self, value):
        self.__exclude_if_mask = value

    def mask_fraction(self, start=None, stop=None):
        r"""Fraction of masked data"""
        return 1.-(
            self.mask.loc[start:stop].sum()/len(self.mask.loc[start:stop])
        )

    def mask_fraction_period(self, period='7D', verbose=False):
        r"""Mask fraction per consecutive periods"""

        # Compute consecutive intervals
        intervals = _interval_maker(self.data.index, period, verbose)

        results = [
            self.mask_fraction(start=time[0], stop=time[1])
            for time in intervals
        ]
        return results

    def length(self):
        r"""Number of data acquisition points"""
        return len(self.data)

    def time_range(self):
        r"""Range (in days, hours, etc) of the data acquistion period"""
        return (self.data.index[-1]-self.data.index[0])

    def duration(self):
        r"""Duration (in days, hours, etc) of the data acquistion period"""
        return self.frequency * self.length()

    def binarized_data(self, threshold):
        """Boolean thresholding of Pandas Series"""
        return pd.Series(
            np.where(self.data > threshold, 1, 0),
            index=self.data.index
        ).where(self.data.notna(), np.nan)

    # TODO: @lru_cache(maxsize=6) ???
    def resampled_data(self, freq, binarize=False, threshold=0):
        r"""Data resampled at the specified frequency.
        If mask_inactivity is True, the `mask` is used to filter inactive data.
        """
        if binarize is False:
            data = self.data
        else:
            data = self.binarized_data(threshold)

        if freq is None:
            return data
        elif to_offset(freq).delta < self.frequency:
            warnings.warn(
                'Resampling frequency lower than the acquisition'
                + ' frequency. Returning original data.',
                UserWarning
            )
            return data
        elif to_offset(freq).delta == self.frequency:
            return data

        resampled_data = data.resample(freq, origin='start').sum()
        if self.mask_inactivity is True:
            if self.mask is None:
                warnings.warn(
                    (
                        'Mask inactivity set to True but no mask could be'
                        ' found.\n Please create a mask by using the '
                        '"create_inactivity_mask" function.'
                    ),
                    UserWarning
                )
                return resampled_data
            elif self.exclude_if_mask:
                resampled_mask = self.mask.resample(freq, origin='start').min()
            else:
                resampled_mask = self.mask.resample(freq, origin='start').max()
            return resampled_data.where(resampled_mask > 0)
        else:
            return resampled_data

    # TODO: @lru_cache(maxsize=6) ???
    def resampled_light(self, freq):
        """Light measurement, resampled at the specified frequency.
        """
        light = self.light

        if to_offset(freq).delta <= self.frequency:
            warnings.warn(
                'Resampling frequency equal to or lower than the acquisition'
                + ' frequency. Returning original data.',
                UserWarning
            )
            return light
        else:
            return light.resample(freq, origin='start').sum()

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
        r"""Reader function for sleep diaries.

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
        """ :class:`SleepDiary` class instanciation."""
        return self.__sleep_diary

    @sleep_diary.setter
    def sleep_diary(self, value):
        self.__sleep_diary = value

    def create_activity_report(
        self,
        cut_points,
        labels=None,
        threshold=None,
        start_time=None,
        stop_time=None,
        oformat=None,
        verbose=False
    ):
        r"""Activity report.

        Create an activity report with aggregated statistics on activity levels
        and time spent between the specified cut-points.

        Parameters
        ----------
        cut_points: array
            Activity cut-points. If all the values are below 1, they are
            interpreted as percentiles of the activity counts. Lower
            (i.e -infinity count) and upper (i.e infinity count) boundaries are
            automatically added.
        labels: array, optional
            Labels for the intervals defined by the cut points.
            The number of labels should be N+1 for N cut-points.
            If set to None, the cut points are used to define the labels.
            Default is None.
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            activity ranges.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        oformat: str, optional
            Output format. Available formats: 'minute' or 'timedelta'.
            If set to 'minute', the result is in number of minutes.
            If set to 'timedelta', the result is a pd.Timedelta.
            If set to None, the result is in number of epochs.
            Default is None.
        verbose: bool, optional
            If set to True, print out info about the cut points.
            Default is False.
        """
        # Create activity report
        self.__activity_report = ActivityReport(self.data, cut_points, labels)
        # Fill the activity report
        self.__activity_report.fit(
            threshold=threshold,
            start_time=start_time,
            stop_time=stop_time,
            oformat=oformat,
            verbose=verbose
        )
        # Reset and rename index
        self.__activity_report.results.reset_index(inplace=True)
        self.__activity_report.results.rename(
            columns={'index': 'activity level'}, inplace=True
        )

        # Add recording name
        self.__activity_report.results.loc[:, 'ID'] = self.display_name

    @property
    def activity_report(self):
        r"""Activity report accessor"""
        # self.__activity_report.results.name = self.name
        return self.__activity_report.pretty_results()

    def create_sleep_report(
        self,
        states=['NIGHT'],
        state_scoring={'NIGHT': 1},
        convert_td_to_num_min=True,
        verbose=False,
        scoring_algo='Scripps',
        *args,
        **kwargs
    ):
        r"""Sleep report.

        Create an sleep report using the periods reported in the sleep diary as
        periods of interest.

        Parameters
        ----------
        states: list
            List of types of periods of interest. Should match the types
            reported in the sleep diary file.
        state_scoring: dict
            Expected scores from the sleep algorithm for the states of
            interest.
        convert_dt_to_num_min: bool, optional
            If set to True, all durations are reported in minutes instead of
            pd.Timedelta.
        verbose: bool, optional
            If set to True, print out info about periods found in the sleep
            diary.
            Default is False.
        scoring_algo: str, optional
            Sleep/wake scoring algorithm to use.
            Default is 'Scripps'.
        *args
            Variable length argument list passed to the scoring algorithm.
        **kwargs
            Arbitrary keyword arguments passed to the scoring algorithm.
        """

        # Check if sleep diary is available
        if self.sleep_diary is None:
            warning_msg = (
                'The sleep diary is None. '
                'Please run the "read_sleep_diary" function.'
            )
            print(warning_msg)
            return None

        # Retrieve sleep scoring function dynamically by name
        sleep_algo = getattr(self, scoring_algo)

        # Sleep scoring
        scoring = sleep_algo(*args, **kwargs)

        # Create sleep report
        self.__sleep_report = create_sleep_report(
            self.sleep_diary,
            scoring,
            states=states,
            state_scoring=state_scoring,
            convert_td_to_num_min=convert_td_to_num_min,
            verbose=verbose,
        )

    @property
    def sleep_report(self):
        r"""Sleep report accessor"""
        self.__sleep_report.name = self.name
        return self.__sleep_report
