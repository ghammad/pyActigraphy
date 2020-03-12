import numpy as np
import pandas as pd
import warnings
from ..log import BaseLog


def _create_inactivity_mask(data, duration, threshold):

    if duration is None:
        return None

    # Create the mask filled iwith ones by default.
    mask = np.ones_like(data)

    # If duration is -1, return a mask with 1s for later manual edition.
    if duration == -1:
        return pd.Series(mask, index=data.index)

    # Binary data
    binary_data = np.where(data >= threshold, 1, 0)

    # The first order diff Series indicates the indices of the transitions
    # between series of zeroes and series of ones.
    # Add zero at the beginning of this series to mark the beginning of the
    # first sequence found in the data.
    edges = np.concatenate([[0], np.diff(binary_data)])

    # Test if there is no edge (i.e. no consecutive zeroes).
    if all(e == 0 for e in edges):
        return pd.Series(mask, index=data.index)

    # Indices of upper transitions (zero to one).
    idx_plus_one = (edges > 0).nonzero()[0]
    # Indices of lower transitions (one to zero).
    idx_minus_one = (edges < 0).nonzero()[0]

    # Even number of transitions.
    if idx_plus_one.size == idx_minus_one.size:

        # Start with zeros
        if idx_plus_one[0] < idx_minus_one[0]:
            starts = np.concatenate([[0], idx_minus_one])
            ends = np.concatenate([idx_plus_one, [edges.size]])
        else:
            starts = idx_minus_one
            ends = idx_plus_one
    # Odd number of transitions
    # starting with an upper transition
    elif idx_plus_one.size > idx_minus_one.size:
        starts = np.concatenate([[0], idx_minus_one])
        ends = idx_plus_one
    # starting with an lower transition
    else:
        starts = idx_minus_one
        ends = np.concatenate([idx_plus_one, [edges.size]])

    # Index pairs (start,end) of the sequences of zeroes
    seq_idx = np.c_[starts, ends]
    # Length of the aforementioned sequences
    seq_len = ends - starts

    for i in seq_idx[np.where(seq_len >= duration)]:
        mask[i[0]:i[1]] = 0

    return pd.Series(mask, index=data.index)


class FiltersMixin(object):
    """ Mixin Class """

    def create_inactivity_mask(self, duration):
        """Create a mask for inactivity (count equal to zero) periods.

        This mask has the same length as its underlying data and can be used
        to offuscate inactive periods where the actimeter has most likely been
        removed.
        Warning: use a sufficiently long duration in order not to mask sleep
        periods.
        A minimal duration corresponding to two hours seems reasonable.

        Parameters
        ----------
        duration: int or str
            Minimal number of consecutive zeroes for an inactive period.
            Time offset strings (ex: '90min') can also be used.
        """

        if isinstance(duration, int):
            nepochs = duration
        elif isinstance(duration, str):
            nepochs = int(pd.Timedelta(duration)/self.frequency)
        else:
            nepochs = None
            warnings.warn(
                'Inactivity length must be a int and time offset string (ex: '
                '\'90min\'). Could not create a mask.',
                UserWarning
            )

        # Store requested mask duration (and discard current mask)
        self.inactivity_length = nepochs

        # Create actual mask
        self.mask = _create_inactivity_mask(self.raw_data, nepochs, 1)

    def add_mask_period(self, start, stop):
        """ Add a period to the inactivity mask

        Parameters
        ----------
        start: str
            Start time (YYYY-MM-DD HH:MM:SS) of the inactivity period.
        stop: str
            Stop time (YYYY-MM-DD HH:MM:SS) of the inactivity period.
        """

        # Check if a mask has already been created
        # NB : if the inactivity_length is not None, accessing the mask will
        # trigger its creation.
        if self.inactivity_length is None:
            self.inactivity_length = -1
            # self.mask = pd.Series(
            #     np.ones(self.length()),
            #     index=self.data.index
            # )

        # Check if start and stop are within the index range
        if (pd.Timestamp(start) < self.mask.index[0]):
            raise ValueError((
                "Attempting to set the start time of a mask period before " +
                "the actual start time of the data.\n" +
                "Mask start time: {}".format(start) +
                "Data start time: {}".format(self.mask.index[0])
            ))
        if (pd.Timestamp(stop) > self.mask.index[-1]):
            raise ValueError((
                "Attempting to set the stop time of a mask period after " +
                "the actual stop time of the data.\n" +
                "Mask stop time: {}".format(stop) +
                "Data stop time: {}".format(self.mask.index[-1])
            ))

        # Set mask values between start and stop to zeros
        self.mask.loc[start:stop] = 0

    def add_mask_periods(self, input_fname, *args, **kwargs):
        """ Add periods to the inactivity mask

        Function to read start and stop times from a Mask log file. Supports
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
        """

        # Convert the log file into a DataFrame
        absname, log = BaseLog.from_file(input_fname, 'Mask', *args, **kwargs)

        # Iterate over the rows of the DataFrame
        for _, row in log.iterrows():
            self.add_mask_period(row['Start_time'], row['Stop_time'])
