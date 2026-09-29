import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data/ftb')

# read FTB with default parameters
rawFTB = pyActigraphy.io.read_raw_ftb(data_dir)


def test_instance_ftb():
    """is rawFTB an instance of : actimetry.io.read_raw_ftb()
    if true return True, else AssertionError"""
    assert isinstance(rawFTB, pyActigraphy.io.ftb.ftb.RawFTB)


def test_read_raw_ftb_start_time():
    """Is the start_time of the file "2023-09-25 00:00:00"
    and have the right format ?
    if true continue, else AssertionError"""
    assert rawFTB.start_time == pd.Timestamp('2023-09-25 00:00:00')


def test_read_raw_ftb_frequency():
    """Test the extraction of the acquisition frequency of the file """
    assert rawFTB.frequency == pd.Timedelta('00:01:00')


def test_read_raw_ftb_data():
    """Is the length of the data is 1440 (like the real file content)?
    if true continue, else AssertionError"""
    assert len(rawFTB.data) == 1440


def test_read_raw_ftb_light():
    """Is the length of the data equal to the length of the light data?
    if true continue, else AssertionError"""
    assert len(rawFTB.data) == len(rawFTB.light.data)


def test_read_raw_ftb_white_light():
    """Is the length of the data equal to the length of the white light data?
    if true continue, else AssertionError"""
    assert len(rawFTB.data) == len(rawFTB.white_light)