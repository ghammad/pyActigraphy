import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
mtn_path = op.join(data_dir, 'test_sample.mtn')

# read MTN with default parameters
rawMTN = pyActigraphy.io.read_raw_mtn(mtn_path)


def test_instance_mtn():
    """is rawMTN an instance of : actimetry.io.read_raw_mtn(mtn_path ?
    if true return True, else AssertionError"""
    assert isinstance(rawMTN, pyActigraphy.io.mtn.RawMTN)


def test_read_raw_mtn_name():
    """Is the name of the file "text_sample" ?
    if true continue, else AssertionError"""
    assert rawMTN.name == 'TEST_SAMPLE'


def test_read_raw_mtn_uuid():
    """Are the name and serial number of the device "MW8 007565" ?
    if true continue, else AssertionError"""
    assert rawMTN.uuid == 'MW8 007565'


def test_read_raw_mtn_start_time():
    """Is the start_time of the file "2018-05-24 10:49:07"
    and have the right format ?
    if true continue, else AssertionError"""
    assert rawMTN.start_time == pd.Timestamp('2018-05-23 17:30:00')


def test_read_raw_mtn_frequency():
    """Test the extraction of the acquisition frequency of the file """
    assert rawMTN.frequency == pd.Timedelta('00:00:05')


def test_read_raw_mtn_data():
    """Is the length of the data is 5978 (like the real file content)?
    if true continue, else AssertionError"""
    assert len(rawMTN.data) == 5978


def test_read_raw_mtn_light():
    """Is the length of the data equal to the length of the light data?
    if true continue, else AssertionError"""
    assert len(rawMTN.data) == len(rawMTN.light.data)


def test_read_raw_mtn_white_light():
    """Is the length of the data equal to the length of the white light data?
    if true continue, else AssertionError"""
    assert len(rawMTN.data) == len(rawMTN.white_light)
