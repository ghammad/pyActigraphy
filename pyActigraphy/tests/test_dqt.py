import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
dqt_path = op.join(data_dir, 'test_sample_dqt.csv')

# read AWD with default parameters
rawDQT = pyActigraphy.io.read_raw_dqt(dqt_path, name='TEST_SAMPLE')


def test_instance_dqt():

    assert isinstance(rawDQT, pyActigraphy.io.dqt.RawDQT)


def test_read_raw_dqt_name():

    assert rawDQT.name == 'TEST_SAMPLE'


def test_read_raw_dqt_uuid():

    assert rawDQT.uuid == 'myUUID'


def test_read_raw_dqt_start_time():

    assert rawDQT.start_time == pd.Timestamp('2000-01-27 19:00:00')


def test_read_raw_dqt_duration():

    assert rawDQT.duration() == pd.Timedelta('12h')


def test_read_raw_dqt_length():

    assert rawDQT.length() == 43200


def test_read_raw_dqt_light():

    assert len(rawDQT.light.get_channel('whitelight')) == 43200


def test_read_raw_dqt_white_light():

    assert (
        len(rawDQT.light.get_channel('whitelight')) == len(rawDQT.white_light)
    )
