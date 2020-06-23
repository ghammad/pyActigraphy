import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
agd_path = op.join(data_dir, 'test_sample.agd')

# read AWD with default parameters
rawAGD = pyActigraphy.io.read_raw_agd(agd_path)


def test_instance_agd():

    assert isinstance(rawAGD, pyActigraphy.io.agd.RawAGD)


def test_read_raw_agd_name():

    assert rawAGD.name == 'TEST_SAMPLE'


def test_read_raw_agd_uuid():

    assert rawAGD.uuid == 'myUUID'


def test_read_raw_agd_start_time():

    assert rawAGD.start_time == pd.Timestamp('2019-04-15 15:00:00')


def test_read_raw_agd_frequency_header():

    assert rawAGD.frequency == pd.Timedelta('00:00:10')


def test_read_raw_agd_data():

    assert len(rawAGD.data) == 5394


def test_read_raw_agd_data_frequency():

    assert pd.Timedelta(rawAGD.data.index.freq) == rawAGD.frequency


# def test_mixin():
#
#     assert rawAWD.length() == 11718
