import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
alu_path = op.join(data_dir, 'test_sample_alu.txt')

# read AWD with default parameters
rawALU = pyActigraphy.io.read_raw_alu(alu_path)


def test_instance_alu():

    assert isinstance(rawALU, pyActigraphy.io.alu.RawALU)


def test_read_raw_alu_name():

    assert rawALU.name == 'TEST_SAMPLE'


def test_read_raw_alu_uuid():

    assert rawALU.uuid == 'myUUID'


def test_read_raw_alu_format():

    assert rawALU.format == 'ALU'


def test_read_raw_alu_available_modes():

    assert rawALU.available_modes == [
        'PIM', 'PIMn', 'TAT', 'TATn', 'ZCM', 'ZCMn'
    ]


def test_read_raw_alu_available_light_channels():

    assert list(rawALU.light.get_channel_list()) == [
        'LIGHT', 'AMB LIGHT', 'RED LIGHT', 'GREEN LIGHT',
        'BLUE LIGHT', 'IR LIGHT', 'UVA LIGHT', 'UVB LIGHT',
        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'MELANOPIC_LUX', 'CLEAR'
    ]


def test_read_raw_alu_start_time():

    assert rawALU.start_time == pd.Timestamp('1918-01-01 09:00:00')


def test_read_raw_alu_data():

    assert len(rawALU.data) == 4*1440


def test_read_raw_alu_light():

    assert (
        len(rawALU.light.get_channel('LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('AMB LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('RED LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('GREEN LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('BLUE LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('IR LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('UVA LIGHT')) == 4*1440
        & len(rawALU.light.get_channel('F1')) == 4*1440
        & len(rawALU.light.get_channel('F2')) == 4*1440
        & len(rawALU.light.get_channel('F3')) == 4*1440
        & len(rawALU.light.get_channel('F4')) == 4*1440
        & len(rawALU.light.get_channel('F5')) == 4*1440
        & len(rawALU.light.get_channel('F6')) == 4*1440
        & len(rawALU.light.get_channel('F7')) == 4*1440
        & len(rawALU.light.get_channel('F8')) == 4*1440
        & len(rawALU.light.get_channel('CLEAR')) == 4*1440
        & len(rawALU.light.get_channel('MELANOPIC_LUX')) == 4*1440
    )

def test_read_raw_alu_period():

    assert (rawALU.period) == pd.Timedelta('3 days 23:59:00')