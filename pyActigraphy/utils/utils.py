import numpy as np
import pandas as pd


def _average_daily_activity(data, cyclic=False):
    """Calculate the average daily activity distribution"""

    avgdaily = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ]).mean()

    if cyclic:
        avgdaily = pd.concat([avgdaily, avgdaily])
        avgdaily.index = pd.timedelta_range(
            start='0 day',
            end='2 days',
            freq=data.index.freq,
            closed='left'
        )
    else:
        avgdaily.index = pd.timedelta_range(
            start='0 day',
            end='1 day',
            freq=data.index.freq,
            closed='left'
        )

    return avgdaily


def _shift_time_axis(avgdaily, shift):

    avgdaily_shifted = avgdaily.reindex(
        index=np.roll(avgdaily.index, shift)
    )
    avgdaily_shifted.index = pd.timedelta_range(
        start='-12h',
        end='12h',
        freq=avgdaily.index.freq,
        closed='left'
    )

    return avgdaily_shifted


def _onset_detection(x, whs):
    return np.mean(x[whs:])/np.mean(x[0:whs])-1


def _offset_detection(x, whs):
    return np.mean(x[0:whs])/np.mean(x[whs:])-1


def _activity_inflexion_time(data, fct, whs):

    r = data.rolling(whs*2, center=True)

    aot = r.apply(fct, kwargs={'whs': whs}, raw=True).idxmax()

    return aot


def _activity_onset_time(data, whs=4):

    return _activity_inflexion_time(data, _onset_detection, whs)


def _activity_offset_time(data, whs=4):

    return _activity_inflexion_time(data, _offset_detection, whs)
