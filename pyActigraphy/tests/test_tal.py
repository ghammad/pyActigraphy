import os.path as op

import pyActigraphy
# import inspect
import pandas as pd


# FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(pyActigraphy.__file__), 'tests/data/')
tal_path = op.join(data_dir, 'test_sample_tal.txt')

# read TAL with default parameters
rawTAL = pyActigraphy.io.read_raw_tal(
    tal_path,
    name='TEST_SAMPLE',
    # sep='\t',
    encoding='utf-8',  # test file is encoded in UTF-8 (cf pytest issue)
    # frequency='1min'
)


def test_instance_tal():

    assert isinstance(rawTAL, pyActigraphy.io.tal.RawTAL)


def test_read_raw_tal_name():

    assert rawTAL.name == 'TEST_SAMPLE'


def test_read_raw_tal_uuid():

    assert rawTAL.uuid == '00000'


def test_read_raw_tal_start_time():

    assert rawTAL.start_time == pd.Timestamp('2000-01-01 00:00:00')


def test_read_raw_tal_duration():

    assert rawTAL.duration() == pd.Timedelta('7D')


def test_read_raw_tal_length():

    assert rawTAL.length() == 10080


def test_read_raw_tal_light():

    assert rawTAL.length() == len(rawTAL.light.get_channel('whitelight'))


def test_read_raw_tal_whitelight():

    assert rawTAL.length() == len(rawTAL.white_light)
