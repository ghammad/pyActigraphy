import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
atr_path = op.join(data_dir, 'test_sample_atr.txt')

# read AWD with default parameters
rawATR = pyActigraphy.io.read_raw_atr(atr_path)


def test_instance_atr():

    assert isinstance(rawATR, pyActigraphy.io.atr.RawATR)


def test_read_raw_atr_name():

    assert rawATR.name == 'TEST_SAMPLE'


def test_read_raw_atr_uuid():

    assert rawATR.uuid == 'myUUID'


def test_read_raw_atr_start_time():

    assert rawATR.start_time == pd.Timestamp('1918-01-01 09:00:00')


def test_read_raw_atr_data():

    assert len(rawATR.data) == 4*1440
