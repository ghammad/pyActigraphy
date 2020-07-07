import os.path as op

import pyActigraphy
# import inspect
import pandas as pd


# FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(pyActigraphy.__file__), 'tests/data/')
tmp_path = op.join(data_dir, 'test_sample_tmp.txt')

# read TMP with default parameters
rawTMP = pyActigraphy.io.read_raw_tmp(tmp_path, name='TEST_SAMPLE')


def test_instance_tmp():

    assert isinstance(rawTMP, pyActigraphy.io.tmp.RawTMP)


def test_read_raw_tmp_name():

    assert rawTMP.name == 'TEST_SAMPLE'


def test_read_raw_tmp_uuid():

    assert rawTMP.uuid == '00000'


def test_read_raw_tmp_start_time():

    assert rawTMP.start_time == pd.Timestamp('2000-01-01 00:00:00')


def test_read_raw_tmp_duration():

    assert rawTMP.duration() == pd.Timedelta('7D')


def test_read_raw_tmp_length():

    assert rawTMP.length() == 10080
