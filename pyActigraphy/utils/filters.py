import pandas as pd


def filter_ts_duration(ts, duration_min='3H', duration_max='12H'):
    r'''Filter time series according to their duration

    '''

    def duration(s):
        return s.index[-1]-s.index[0]

    td_min = pd.Timedelta(duration_min)
    td_max = pd.Timedelta(duration_max)

    from itertools import filterfalse
    filtered = []
    filtered[:] = filterfalse(
        lambda x: duration(x) < td_min or duration(x) > td_max,
        ts
    )
    return filtered
