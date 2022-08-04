import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
rpx_path_eng = op.join(data_dir, 'test_sample_rpx_eng.csv')
rpx_path_fr = op.join(data_dir, 'test_sample_rpx_fr.csv')
rpx_path_ger = op.join(data_dir, 'test_sample_rpx_ger_no_light.csv')
rpx_path_ger_with_light = op.join(
    data_dir, 'test_sample_rpx_ger_with_light.csv'
)

# read RPX ENG with default parameters
rawRPX_ENG = pyActigraphy.io.read_raw_rpx(
    rpx_path_eng, language='ENG_UK')

# read RPX FR with default parameters
rawRPX_FR = pyActigraphy.io.read_raw_rpx(
    rpx_path_fr, language='FR')

# read RPX FR with modified parameters
rawRPX_FR_NaN = pyActigraphy.io.read_raw_rpx(
    rpx_path_fr, language='FR', drop_na=False)

# read RPX GER with default parameters
rawRPX_GER = pyActigraphy.io.read_raw_rpx(
    rpx_path_ger,
    language='GER',
    drop_na=False
)
rawRPX_GER_with_light = pyActigraphy.io.read_raw_rpx(
    rpx_path_ger_with_light,
    language='GER',
    delimiter=',',
    decimal=',',
    drop_na=False
)


def test_read_raw_rpx_eng_name():

    assert rawRPX_ENG.name == 'TEST_SAMPLE_UK'


def test_read_raw_rpx_eng_uuid():

    assert rawRPX_ENG.uuid == 'AXXXUK'


def test_read_raw_rpx_eng_start_time():

    assert rawRPX_ENG.start_time == pd.Timestamp('2015-07-04 09:45:00')


def test_read_raw_rpx_eng_frequency():

    assert rawRPX_ENG.frequency == pd.Timedelta('00:00:30')


def test_instance_rpx_fr():

    assert isinstance(rawRPX_FR, pyActigraphy.io.rpx.RawRPX)


def test_read_raw_rpx_fr_name():

    assert rawRPX_FR.name == 'TEST_SAMPLE_FR'


def test_read_raw_rpx_fr_uuid():

    assert rawRPX_FR.uuid == 'AXXXFR'


def test_read_raw_rpx_fr_start_time():

    assert rawRPX_FR.start_time == pd.Timestamp('2015-02-04 11:45:00')


def test_read_raw_rpx_fr_frequency():

    assert rawRPX_FR.frequency == pd.Timedelta('00:01:00')


def test_read_raw_rpx_fr_data():

    assert len(rawRPX_FR.data) == 11555


def test_read_raw_rpx_fr_nan_data():

    assert len(rawRPX_FR_NaN.data) == 11581


def test_read_raw_rpx_fr_sleep_wake_score():

    assert rawRPX_FR.sleep_wake.sum() == 2260


def test_read_raw_rpx_fr_mobility_score():

    assert rawRPX_FR.mobility.sum() == 3104


def test_read_raw_rpx_fr_interval_status():

    assert list(rawRPX_FR.interval_status.unique()) == \
        ['ACTIVITÉ', 'REPOS', 'S-REPOS']


def test_read_raw_rpx_ger_name():

    assert rawRPX_GER.name == 'TEST_SAMPLE_GER'


def test_read_raw_rpx_ger_uuid():

    assert rawRPX_GER.uuid == 'PXXGER'


def test_read_raw_rpx_ger_start_time():

    assert rawRPX_GER.start_time == pd.Timestamp('2020-01-24 09:05:00')


def test_read_raw_rpx_ger_frequency():

    assert rawRPX_GER.frequency == pd.Timedelta('00:00:30')


def test_read_raw_rpx_ger_data():

    assert len(rawRPX_GER.data) == 20160


def test_read_raw_rpx_ger_no_light():

    assert rawRPX_GER.light is None


def test_read_raw_rpx_ger_with_light_shape():

    assert rawRPX_GER_with_light.light.data.shape == (20160, 4)


def test_read_raw_rpx_ger_with_light():

    assert rawRPX_GER_with_light.white_light.shape == \
           rawRPX_GER_with_light.red_light.shape == \
           rawRPX_GER_with_light.blue_light.shape == \
           rawRPX_GER_with_light.green_light.shape
