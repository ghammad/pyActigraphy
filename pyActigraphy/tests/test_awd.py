import os.path as op

import pyActigraphy
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
awd_path = op.join(data_dir, 'test_sample.AWD')
aw4_path = op.join(data_dir, 'test_sample_aw4.AWD')
aw7_path = op.join(data_dir, 'test_sample_aw7.AWD')
awi_path = op.join(data_dir, 'test_sample_awi.AWD')
awl_path = op.join(data_dir, 'test_sample_awl.AWD')
awlp_path = op.join(data_dir, 'test_sample_awlp.AWD')
awmk2_path = op.join(data_dir, 'test_sample_awmk2.AWD')
aws_path = op.join(data_dir, 'test_sample_aws.AWD')
awt_path = op.join(data_dir, 'test_sample_awt.AWD')

# read AWD with default parameters
rawAWD = pyActigraphy.io.read_raw_awd(awd_path)


def test_instance_awd():

    assert isinstance(rawAWD, pyActigraphy.io.awd.RawAWD)


def test_read_raw_awd_name():

    assert rawAWD.name == 'TEST_SAMPLE'


def test_read_raw_awd_uuid():

    assert rawAWD.uuid == 'V00000'


def test_read_raw_awd_frequency():

    assert rawAWD.frequency == pd.Timedelta('1min')


def test_read_raw_awd_start_time():

    assert rawAWD.start_time == pd.Timestamp('2018-01-01 08:30:00')


def test_read_raw_awd_data():

    assert len(rawAWD.data) == 11718


def test_read_raw_aw4():

    # read AW4 with default parameters
    rawAW4 = pyActigraphy.io.read_raw_awd(aw4_path)
    assert (
        (rawAW4.model == 'Actiwatch-4')
        & (rawAW4.frequency == pd.Timedelta('1min'))
        & (rawAW4.light is None)
    )


def test_read_raw_aw4_engine():

    # read AW4 with default parameters
    rawAW4 = pyActigraphy.io.read_raw_awd(aw4_path, engine='c')
    assert (
        (rawAW4.model == 'Actiwatch-4')
        & (rawAW4.frequency == pd.Timedelta('1min'))
        & (rawAW4.light is None)
    )


def test_read_raw_aw7():

    # read AW7 with default parameters
    rawAW7 = pyActigraphy.io.read_raw_awd(aw7_path)
    assert (
        (rawAW7.model == 'Actiwatch-7')
        & (rawAW7.frequency == pd.Timedelta('15s'))
        & (rawAW7.light.get_channel_list() == ['whitelight'])
    )


def test_read_raw_awi():

    # read AW-Insomnia with default parameters
    rawAWI = pyActigraphy.io.read_raw_awd(awi_path)
    assert (
        (rawAWI.model == 'Actiwatch-Insomnia (pressure sens.)')
        & (rawAWI.frequency == pd.Timedelta('1min'))
        & (rawAWI.light is None)
    )


def test_read_raw_awl():

    # read AWL file with default parameters
    rawAWL = pyActigraphy.io.read_raw_awd(awl_path)
    assert (
        (rawAWL.model == 'Actiwatch-L (amb. light)')
        & (rawAWL.frequency == pd.Timedelta('1min'))
        & (rawAWL.light.get_channel_list() == ['whitelight'])
    )


def test_read_raw_awlp():

    # read AWL-Plus file with default parameters
    rawAWLP = pyActigraphy.io.read_raw_awd(awlp_path)
    assert (
        (rawAWLP.model == 'Actiwatch-L-Plus (amb. light)')
        & (rawAWLP.frequency == pd.Timedelta('1min'))
        & (rawAWLP.light.get_channel_list() == ['whitelight'])
    )


def test_read_raw_awmk2():

    # read AW-Insomnia with default parameters
    rawAWMk2 = pyActigraphy.io.read_raw_awd(awmk2_path)
    assert (
        (rawAWMk2.model == 'Actiwatch-Mini')
        & (rawAWMk2.frequency == pd.Timedelta('30s'))
        & (rawAWMk2.light is None)
    )


def test_read_raw_aws():

    # read AW-Insomnia with default parameters
    rawAWS = pyActigraphy.io.read_raw_awd(aws_path)
    assert (
        (rawAWS.model == 'Actiwatch-S (env. sound)')
        & (rawAWS.frequency == pd.Timedelta('1min'))
        & (rawAWS.light is None)
    )


def test_read_raw_awt():

    # read AW-Insomnia with default parameters
    rawAWT = pyActigraphy.io.read_raw_awd(awt_path)
    assert (
        (rawAWT.model == 'Actiwatch-T (temp.)')
        & (rawAWT.frequency == pd.Timedelta('1min'))
        & (rawAWT.light is None)
    )
