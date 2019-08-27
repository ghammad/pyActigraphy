import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
rpx_path = op.join(data_dir, 'test_sample.csv')

# read RPX with default parameters
rawRPX = pyActigraphy.io.read_raw_rpx(rpx_path, language='FR')


def test_instance_rpx():

    assert isinstance(rawRPX, pyActigraphy.io.rpx.RawRPX)


def test_read_raw_rpx_name():

    assert rawRPX.name == 'TEST_SAMPLE'


def test_read_raw_rpx_uuid():

    assert rawRPX.uuid == 'A18272'


def test_read_raw_rpx_start_time():

    assert rawRPX.start_time == pd.Timestamp('2015-02-04 11:45:00')


def test_read_raw_rpx_frequency():

    assert rawRPX.frequency == pd.Timedelta('00:01:00')


def test_read_raw_rpx_data():

    assert len(rawRPX.data) == 11555
