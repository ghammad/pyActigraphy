import os.path as op

import actimetry
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
rpx_path = op.join(data_dir, 'test_sample.csv')

# read RPX with default parameters
rawRPX = actimetry.io.read_raw_rpx(rpx_path)


def test_instance_rpx():
    """is rawRPX an instance of : actimetry.io.read_raw_rpx(rpx_path ?
    if true return True, else AssertionError"""
    assert isinstance(rawRPX, actimetry.io.rpx.RawRPX)


def test_read_raw_rpx_name():
    """Is the name of the file "text_sample" ?
    if true continue, else AssertionError"""
    assert rawRPX.name == 'TEST_SAMPLE'


def test_read_raw_rpx_uuid():
    """Are the name and serial number of the device "MW8 007565" ?
    if true continue, else AssertionError"""
    assert rawRPX.uuid == 'A18272'


def test_read_raw_rpx_start_time():
    """Is the start_time of the file "2018-05-24 10:49:07"
    and have the right format ?
    if true continue, else AssertionError"""
    assert rawRPX.start_time == pd.Timestamp('2018-12-01 00:00:00')


def test_read_raw_rpx_frequency():
    """Test the extraction of the acquisition frequency of the file """
    assert rawRPX.frequency == pd.Timedelta('00:01:00')


def test_read_raw_rpx_data():
    """Is the length of the data is 5978 (like the real file content)?
    if true continue, else AssertionError"""
    assert len(rawRPX.data) == 11555


# def test_mixin():
#
#     assert rawRPX.length() == 11718
