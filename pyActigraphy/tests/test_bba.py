import os.path as op

import pyActigraphy
import inspect
import json
import pytest
# import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
bba_path_ts = op.join(data_dir, 'sample-timeSeries.csv.gz')
bba_path_json = op.join(data_dir, 'sample-summary.json')
bba_path_json_wrong_name = op.join(data_dir, 'sample-summary-wrong-name.json')
bba_path_json_no_calib = op.join(
    data_dir, 'sample-summary-no-calibration.json'
)

# read BBA file with default metadata
rawBBA = pyActigraphy.io.read_raw_bba(bba_path_ts)

# metadata
with open(bba_path_json) as file:
    meta_data_default = json.load(file)


def test_read_raw_bba_metadata_default():

    assert (
        (rawBBA.name == op.basename(bba_path_ts))
        & (rawBBA.uuid == meta_data_default['file-deviceID'])
    )


def test_read_raw_bba_metadata_wrong_name():

    with pytest.raises(ValueError):
        # read BBA file with wrong metadata: name
        pyActigraphy.io.read_raw_bba(
            bba_path_ts, metadata_fname=bba_path_json_wrong_name
        )


def test_read_raw_bba_metadata_no_calibration():

    # read BBA file with metadata specifying uncalibrated data
    rawBBA_nocalib = pyActigraphy.io.read_raw_bba(
            bba_path_ts, metadata_fname=bba_path_json_no_calib
        )
    assert (rawBBA_nocalib.isCalibratedOnOwnData is False)


def test_read_raw_bba_data_mean_std():

    assert (
        (rawBBA.data.mean() == pytest.approx(33.034, abs=0.001))
        & (rawBBA.data.std() == pytest.approx(88.173, abs=0.001))
    )


def test_read_raw_bba_data_imputation():

    # read BBA file with imputed missing data
    rawBBA_imp = pyActigraphy.io.read_raw_bba(
        bba_path_ts, use_metadata_json=False, impute_missing=True
    )
    assert (
        (rawBBA.data[rawBBA.data.isna()].size == 125)
        & (rawBBA_imp.data[rawBBA_imp.data.isna()].size == 0)
    )


def test_read_raw_bba_data_physical_activity():

    assert (
        rawBBA.sleep.shape
        == rawBBA.sedentary.shape
        == rawBBA.mvpa.shape
        == rawBBA.met.shape
        == (16841,)
    )


def test_read_raw_bba_data_white_light():

    assert (rawBBA.white_light.shape == (16841,))
