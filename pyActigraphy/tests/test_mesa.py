import os.path as op

import pyActigraphy
import pytest
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
mesa_path = op.join(data_dir, 'test_sample_mesa.csv')
warn_msg = (
    'Specified time_origin is such that the day of the week in the '
    'reconstructed time index is *not* aligned with the day of the week '
    'reported in the recording.'
)

# read AWD with default parameters
rawMESA = pyActigraphy.io.read_raw_mesa(mesa_path)

# time indices
index = pd.to_datetime([
    '2000-01-06 00:00:00',
    '2000-01-06 00:00:30',
    '2000-01-06 00:01:00',
    '2000-01-06 00:01:30'
])

# light intensities
light_intensities = [pd.NA, 5.00, 2.50, 0.50]


def test_instance_mesa():

    assert isinstance(rawMESA, pyActigraphy.io.mesa.RawMESA)


def test_read_raw_mesa_name():

    assert rawMESA.name == 'X'


def test_read_raw_mesa_uuid():

    assert rawMESA.uuid is None


def test_read_raw_mesa_start_time():

    assert rawMESA.start_time == pd.Timestamp(index[0])


def test_read_raw_mesa_frequency_header():

    assert rawMESA.frequency == pd.Timedelta('00:00:30')


def test_read_raw_mesa_duration():

    assert rawMESA.duration() == pd.Timedelta('00:02:00')


def test_read_raw_mesa_data_frequency():

    assert pd.Timedelta(rawMESA.data.index.freq) == rawMESA.frequency


def test_mesa_intervals():

    cmp = (rawMESA.intervals == pd.Series(index=index, data=[-1, 1, 0.5, 0.0]))
    assert all(cmp)


def test_mesa_white_light():

    cmp = (rawMESA.white_light.dropna() == pd.Series(
        index=index,
        data=[2*li for li in light_intensities]).dropna()
    )
    assert all(cmp)


def test_mesa_red_light():

    # Dropna because NaNs cannot be compared.
    cmp = (rawMESA.red_light.dropna() == pd.Series(
        index=index,
        data=light_intensities).dropna()
    )
    assert all(cmp)


def test_mesa_green_light():

    # Dropna because NaNs cannot be compared.
    cmp = (rawMESA.green_light.dropna() == pd.Series(
        index=index,
        data=light_intensities).dropna()
    )
    assert all(cmp)


def test_mesa_blue_light():

    # Dropna because NaNs cannot be compared.
    cmp = (rawMESA.blue_light.dropna() == pd.Series(
        index=index,
        data=light_intensities).dropna()
    )
    assert all(cmp)


def test_mesa_check_dayofweek():

    with pytest.warns(UserWarning) as record:
        pyActigraphy.io.read_raw_mesa(
            mesa_path,
            check_dayofweek=True,
            time_origin="2000-01-05"  # Origin not aligned with reported day
        )
    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == warn_msg
