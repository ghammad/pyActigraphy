import numpy as np
import pandas as pd


def _create_inactivity_mask(data, duration, threshold):

    # Binary data
    binary_data = np.where(data >= threshold, 1, 0)

    # The first order diff Series indicates the indices of the transitions
    # between series of zeroes and series of ones.
    # Add zero at the beginning of this series to mark the beginning of the
    # first sequence found in the data.
    edges = np.concatenate([[0], np.diff(binary_data)])

    # Create the mask filled iwith ones by default.
    mask = np.ones_like(data)

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
        duration: int
            Minimal number of consecutive zeroes for an inactive period
        """
        self.mask = _create_inactivity_mask(self.raw_data, duration, 1)
