import numpy as np
import pandas as pd
from pyActigraphy.light import LightRecording
from pytest import approx

# Synthetic data characteristics
sf = '30s'  # Sampling frequency
nperiods = 10

A = 1  # Amplitude

x = np.linspace(0, nperiods-1, nperiods)

# Synthetic data
f_ch1 = nperiods*[A]
f_ch2 = nperiods*[10*A]

data_df = pd.DataFrame(
    data={
        'channel_1': f_ch1,
        'channel_2': f_ch2
    },
    index=pd.date_range(
        start='01/01/2000 00:00:00',
        periods=nperiods,
        freq=sf
    )
)

# Set abnormal data points
abnval = 10000000
abnval_start = pd.to_datetime('01/01/2000 00:02:00')
abnval_stop = pd.to_datetime('01/01/2000 00:04:00')
abnval_nperiods = ((abnval_stop-abnval_start)/pd.Timedelta(sf)) + 1
data_df.loc[abnval_start:abnval_stop] = abnval

# Data mask
data_dummy_mask = pd.DataFrame(
    data={
        'channel_1': nperiods*[1],
        'channel_2': nperiods*[1]
    },
    index=pd.date_range(
        start='01/01/2000 00:00:00',
        periods=nperiods,
        freq=sf
    ),
    dtype='int'
)

data_mask = data_dummy_mask.copy(deep=True)
data_mask.loc[abnval_start:abnval_stop, 'channel_1'] = 0

# LightRecording instanciation
myLightRecording = LightRecording(
    name='LightRecording',
    uuid='MultiChanUX20',
    data=data_df,
    frequency=data_df.index.freq,
    log10_transform=False
)

# # Set up data mask
# myLightRecording.mask = data_mask


def test_light_mask_creation():

    myLightRecording.create_light_mask()
    assert myLightRecording.mask.equals(data_dummy_mask)


def test_light_mask_add_period():

    myLightRecording.add_light_mask_period(
        abnval_start, abnval_stop, channel='channel_1'
    )
    assert myLightRecording.mask.equals(data_mask)


def test_light_mask():

    myLightRecording.create_light_mask()
    myLightRecording.add_light_mask_period(
        abnval_start, abnval_stop, channel='channel_1'
    )
    myLightRecording.apply_mask = True
    assert (
        myLightRecording.get_channel('channel_1').mean() == approx(1.0)
        and myLightRecording.get_channel('channel_2').mean() == approx(
            abnval_nperiods*abnval/nperiods
        )
    )


def test_light_unmask():

    myLightRecording.apply_mask = False

    assert myLightRecording.get_channel(
        'channel_1').mean() == approx(abnval_nperiods*abnval/nperiods)
