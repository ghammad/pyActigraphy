import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import warnings


def _binarized_data(data, threshold=0):
    """Boolean thresholding of Pandas Series

    Compute a binary time series with the orignal NaNs.

    Parameters
    ----------
    data: pd.DataFrame
        The data to binarize.
    threshold: int, optional
        Threshold used to binarize the data. Set output value to 1 if initial
        value > threshold. Set to 0 othertwise.

    Returns
    -------
    bin : pd.DataFrame
        Binarized time series
    """

    return (data > threshold).where(data.notna(), np.nan).astype(float)


def _resampled_data(data, rsfreq, agg='sum'):
    r"""Data resampling

    Resample input data at the specified frequency.

    Parameters
    ----------
    data: pd.Series
        The data to binarize.
    rsfreq: str
        Resampling frequency.
    agg: str, optional
        Aggregation function applied to resampled data.
        Default is 'sum'.

    Returns
    -------
    res : pd.Series
        Resampled time series
    """

    # Short-cut: returns data if specified freq is lower than the original freq
    if rsfreq is None:
        return data
    elif to_offset(rsfreq).delta < data.index.freq:
        warnings.warn(
            'Resampling frequency lower than the acquisition'
            + ' frequency. Returning original data.',
            UserWarning
        )
        return data
    elif to_offset(rsfreq).delta == data.index.freq:
        return data

    # Resample data to specified frequency)
    # TODO: secure function against NaN when agg=mean
    resampled_data = data.resample(rsfreq).agg(
        getattr(pd.Series, agg),
        skipna=False
    )

    return resampled_data

    # if self.mask_inactivity is True:
    #     if self.mask is None:
    #         warnings.warn(
    #             (
    #                 'Mask inactivity set to True but no mask could be'
    #                 ' found.\n Please create a mask by using the '
    #                 '"create_inactivity_mask" function.'
    #             ),
    #             UserWarning
    #         )
    #         return resampled_data
    #     elif self.exclude_if_mask:
    #         resampled_mask = self.mask.resample(freq).min()
    #     else:
    #         resampled_mask = self.mask.resample(freq).max()
    #     return resampled_data.where(resampled_mask > 0)
    # else:
    #     return resampled_data
