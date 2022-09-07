import numpy as np
import pandas as pd
from ..log import BaseLog


def _create_dummy_mask(data):
    """ Create a dummy mask

    Parameters
    ----------
    data: ndarray
        Input data.

    Returns
    -------
    mask: pd.DataFrame
        Dummy mask with the shape of the input data.

    """

    # Create a mask filled with ones by default.
    return pd.DataFrame(
        data=np.ones_like(data),
        index=data.index,
        columns=data.columns,
        dtype='int'
    )


def _add_mask_period(mask, start, stop, channel=None):
    """ Add a masking period to an existing mask

    Parameters
    ----------
    mask: pd.DataFrame or pd.Series
        Mask to which a masking period is added.
    start: str
        Start time (YYYY-MM-DD HH:MM:SS) of the masking period.
    stop: str
        Stop time (YYYY-MM-DD HH:MM:SS) of the masking period.
    channel: str, optional
        Set masking period to a specific channel (i.e. column).
        If set to None, the period is set on all channels.
        Default is None.
    """

    start_ts = pd.Timestamp(start)
    stop_ts = pd.Timestamp(stop)
    # Check if start and stop are within the index range
    if (start_ts < mask.index[0]):
        raise ValueError((
            "Attempting to set the start time of a mask period before "
            + "the actual start time of the data.\n"
            + "Mask start time: {}".format(start)
            + "Data start time: {}".format(mask.index[0])
        ))
    if (stop_ts > mask.index[-1]):
        raise ValueError((
            "Attempting to set the stop time of a mask period after "
            + "the actual stop time of the data.\n"
            + "Mask stop time: {}".format(stop)
            + "Data stop time: {}".format(mask.index[-1])
        ))

    # Set mask values between start and stop to zeros
    if channel is None:
        mask.loc[start_ts:stop_ts] = 0
    else:
        mask.loc[start_ts:stop_ts, channel] = 0


def _add_mask_periods(input_fname, mask, channel=None, *args, **kwargs):
    """ Add masking periods to an existing mask

    Function to read start and stop times from a Mask log file. Supports
    different file format (.ods, .xls(x), .csv) throught the BaseLog class.

    Parameters
    ----------
    input_fname: str
        Path to the log file.
    mask: pd.DataFrame or pd.Series
        Mask to which masking periods are added.
    channel: str, optional
        Set masking period to a specific channel (i.e. column).
        If set to None, the period is set on all channels.
        Default is None.
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
        _add_mask_period(mask, row['Start_time'], row['Stop_time'], channel)
