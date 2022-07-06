import numpy as np
import pandas as pd


def troiano(
    data,
    win_size=60,
    activity_threshold=0,
    spike_threshold=100,
    spike_max_nr=2,
):
    r'''Wear and non wear period detection algorithm.

    Identification of wear and non wear episodes using the
    algorithm developed by Troiano et al. [1]_ in the context of the NHANES
    study.

    Parameters
    ----------
    data: array-like
        Activity time series.
    win_size: int, optional
        Number of consecutive zero epochs needed to initiate an off-wrist
        period.
        Default value is 60.
    activity_threshold: int, optional
        Activity equal or above this threshold is considered nonzero.
        Default value is 0.
    spike_threshold: int, optional
        One epoch of activity at or above this threshold will restart counting
        of consecutive zero epochs, and the window will not be considered
        'off-wrist'.
        Default is 100.
    spike_max_nr: int, optional
        Maximum allowable number of nonzero epochs occuring during the
        window of consecutive zeros to consider the window as "off-wrist".
        Default value is 2.

    Return
    ----------
    mask : np.narray
        Binary time series containing the estimated nonwear/wear periods (0/1).

    References
    ----------

    .. [1] Troiano RP, Berrigan D, Dodd KW, Mâsse LC, Tilert T, McDowell M.
           Physical activity in the United States measured by accelerometer.
           Med Sci Sports Exerc. 2008 Jan;40(1):181-8.
           Doi: 10.1249/mss.0b013e31815a51b3 PMID: 18091006
    '''
    if activity_threshold > spike_threshold:
        raise ValueError(
            "Setting an activity threshold higher than the spike threshold "
            "does not make sense. Aborting."
        )

    # Classify data as:
    # 0, below threshold;
    # 1, above activity threshold and below spike threshold
    # spike_max_nr+1, above spike threshold
    data_classified = pd.Series(
        np.where(
            data <= activity_threshold,
            0,
            np.where(
                data < spike_threshold,
                1,
                spike_max_nr+1
            )
        )
    )

    # Count the number of spikes in each window of size 'win_size'
    data_nr_spikes = data_classified.rolling(win_size).sum()

    # Create the mask filled with ones by default.
    mask = np.ones_like(data)

    # Set mask to zero if the number of spikes is below the maximum allowed nr.
    # NB: the rolling window display the resulting number of spikes at the
    # right end of the window. Indices are thus shifted to the left.
    for idx in data_nr_spikes[data_nr_spikes <= spike_max_nr].index:
        mask[idx-win_size+1:idx+1] = 0

    return mask


def choi(
    data,
    win_size=60,
    activity_threshold=0,
    spike_threshold=100,
    spike_max_nr=2,
):
    r'''Wear and non wear period detection algorithm.

    Identification of wear and non wear episodes using the
    algorithm developed by Choi et al. [1]_ by improving upon
    Troiano's algorithm.

    Parameters
    ----------
    data: array-like
        Activity time series.
    win_size: int, optional
        Number of consecutive zero epochs needed to initiate an off-wrist
        period.
        Default value is 60.
    activity_threshold: int, optional
        Activity equal or above this threshold is considered nonzero.
        Default value is 0.
    spike_threshold: int, optional
        One epoch of activity at or above this threshold will restart counting
        of consecutive zero epochs, and the window will not be considered
        'off-wrist'.
        Default is 100.
    spike_max_nr: int, optional
        Maximum allowable number of nonzero epochs occuring during the
        window of consecutive zeros to consider the window as "off-wrist".
        Default value is 2.

    Return
    ----------
    mask : np.narray
        Binary time series containing the estimated nonwear/wear periods (0/1).

    References
    ----------

    .. [1] Choi L, Liu Z, Matthews CE, Buchowski MS. Validation of
           accelerometer wear and nonwear time classification algorithm.
           Med Sci Sports Exerc. 2011 Feb;43(2):357-64.
           Doi: 10.1249/MSS.0b013e3181ed61a3. PMID: 20581716; PMCID: PMC3184184
    '''
    # Excerpt from Choi et al.:
    # The recommended elements in the new algorithm are as follows:
    # 1) zero-count threshold during a nonwear time interval,
    # 2) 90-min time window for consecutive zero or nonzero counts, and
    # 3) allowance of a 2-min interval of nonzero counts with upstream or
    # downstream 30-min consecutive zero-count windows for artifactual movement
    # detection.

    # Transform the data to do the following:
    # Create 0's when the data is <= activity_threshold and > spike_stoplevel
    # Create 1's when the data is > activity_threshold and < spike_stoplevel
    # Create 1 more than the spike_tolerance when data > spike_tolerance

    # data_series = pd.Series(
    #     np.where(
    #         data <= activity_threshold, 0, np.where(
    #             (data > activity_threshold) & (data < spike_stoplevel),
    #             1,
    #             spike_tolerance+1
    #         )
    #     )
    # )

    # # Get the rolling sum of the previous consecutive epochs of length
    # "minlength". The default for choi is 90 epochs.
    # # This will be the rolling number of epochs above "activity_threshold"
    # for the previous epochs of length "minlength"
    #
    # data_rolling_sum = data_series.rolling(minlength).sum()
    #
    # # The choi algorithm has another criteria where any spikes in activity
    # (i.e., nonzero epochs) must have a certain number of consecutive zero
    # epochs prior and after without any nonzero epochs
    # # We will create another data series, with binary 0's and 1's where 1 is
    # nonzero activity. Currently, the 'data_series' variable, contains the
    # value 3 for cells above spike_stoplevel, which isn't ideal for this
    # calculation
    #
    # data_series_binary = pd.Series(np.where(data_series > 0, 1, 0))
    #
    # # We will subtract the data series from the sum of the rolling window,
    # because we do not want to include the value at the center of the window
    # (i.e, the spike in movement).
    #
    # data_rolling_sum_at_spike_value = np.where((data_series_binary == 1) & (
    #     (data_series_binary.rolling((min_window_length*2), center=True).sum()
    # - data_series_binary) > 0), 1, 0)
    #
    # # If the rolling sum is less than the spike tolerance and any spikes have
    # "minlength" consecutive prior/following epochs of values <=
    # activity_threshold,  return 0 (off-wrist), else, return 1 (on-wrist)
    #
    # off_wrist_values = np.where((data_rolling_sum < spike_tolerance) & (
    #     data_rolling_sum_at_spike_value == 0), 0, 1)
    #
    raise NotImplementedError
