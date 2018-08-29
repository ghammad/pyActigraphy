import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
awd_path = op.join(data_dir, 'test_sample.AWD')

# read AWD with default parameters
rawAWD = pyActigraphy.io.read_raw_awd(awd_path)


def test_instance_awd():

    assert isinstance(rawAWD, pyActigraphy.io.awd.RawAWD)


def test_read_raw_awd_name():

    assert rawAWD.name == 'TEST_SAMPLE'


def test_read_raw_awd_uuid():

    assert rawAWD.uuid == 'myUUID'


def test_read_raw_awd_start_time():

    assert rawAWD.start_time == pd.Timestamp('2018-01-01 08:30:00')


def test_read_raw_awd_data():

    assert len(rawAWD.data) == 11718


# def test_mixin():
#
#     assert rawAWD.length() == 11718
