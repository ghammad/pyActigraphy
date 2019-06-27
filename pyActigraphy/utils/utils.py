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
        index = pd.timedelta_range(
            start='0 day',
            end='2 days',
            freq=data.index.freq,
            closed='left'
        )
    else:
        index = pd.timedelta_range(
            start='0 day',
            end='1 day',
            freq=data.index.freq,
            closed='left'
        )

    avgdaily.index = index

    return avgdaily
