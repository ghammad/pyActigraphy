import os.path as op

import pyActigraphy
import inspect
import pandas as pd

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')

sstlog_eu = pyActigraphy.log.read_sst_log(
    op.join(data_dir, 'example_sstlog_dst.csv'),
    sep=',',
    time_zone='Europe/Brussels'
)
sstlog_us = pyActigraphy.log.read_sst_log(
    op.join(data_dir, 'example_sstlog_dst.csv'),
    sep=',',
    time_zone='America/Los_Angeles'
)

grd_truth_eu = pd.Series(
    index=pd.Index(
        [
            'example_nodst',  # False:No DST
            'example_dst_1',  # True:DST in Europe but not in US
            'example_dst_2',  # True:DST everywhere
            'example_dst_3',  # True:Recording over New Year's Eve
            'example_dst_4',  # True:Year long recording
            'example_dst_5'   # False:DST in US but not in Europe
        ],
        name='Subject_id'
    ),
    data=[False, True, True, True, True, False],
    name='DST_crossover'
)

grd_truth_us = pd.Series(
    index=pd.Index(
        [
            'example_nodst',  # False:No DST
            'example_dst_1',  # False:DST in Europe but not in US
            'example_dst_2',  # True:DST everywhere
            'example_dst_3',  # True:Recording over New Year's Eve
            'example_dst_4',  # True:Year long recording
            'example_dst_5'   # True:DST in US but not in Europe
        ],
        name='Subject_id'
    ),
    data=[False, False, True, True, True, True],
    name='DST_crossover'
)


def test_sstlog_dst_eu():

    pd.testing.assert_series_equal(
        sstlog_eu.log['DST_crossover'],
        grd_truth_eu
    )


def test_sstlog_dst_us():

    pd.testing.assert_series_equal(
        sstlog_us.log['DST_crossover'],
        grd_truth_us
    )
