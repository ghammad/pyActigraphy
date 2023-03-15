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


def test_read_raw_atr_format():

    assert rawATR.format == 'ATR'


def test_read_raw_atr_available_modes():

    assert rawATR.available_modes == [
        'PIM', 'PIMn', 'TAT', 'TATn', 'ZCM', 'ZCMn'
    ]


def test_read_raw_atr_available_light_channels():

    assert list(rawATR.light.get_channel_list()) == [
        'LIGHT', 'AMB LIGHT', 'RED LIGHT', 'GREEN LIGHT',
        'BLUE LIGHT', 'IR LIGHT', 'UVA LIGHT', 'UVB LIGHT'
    ]


def test_read_raw_atr_start_time():

    assert rawATR.start_time == pd.Timestamp('1918-01-01 09:00:00')


def test_read_raw_atr_data():

    assert len(rawATR.data) == 4*1440


def test_read_raw_atr_light():

    assert (
        len(rawATR.light.get_channel('LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('AMB LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('RED LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('GREEN LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('BLUE LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('IR LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('UVA LIGHT')) == 4*1440
        & len(rawATR.light.get_channel('UVB LIGHT')) == 4*1440
    )
