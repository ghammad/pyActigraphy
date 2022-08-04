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

data_df = pd.DataFrame(
    data={'channel_1': f_ch1},
    index=pd.date_range(
        start='01/01/2000 00:00:00',
        periods=nperiods,
        freq=sf
    )
)

# Data mask
data_mask = pd.Series(
    data=nperiods*[1],
    index=pd.date_range(
        start='01/01/2000 00:00:00',
        periods=nperiods,
        freq=sf
    ),
    dtype=pd.Int16Dtype
)

# Set abnormal data points
abnval = 10000000
data_df.iloc[4:5] = abnval
data_mask.iloc[4:5] = 0

# LightRecording instanciation
myLightRecording = LightRecording(
    name='LightRecording',
    uuid='MultiChanUX20',
    data=data_df,
    frequency=data_df.index.freq,
    log10_transform=False
)

# Set up data mask
myLightRecording.mask = data_mask


def test_light_mask():

    myLightRecording.mask_inactivity = True

    assert myLightRecording.get_channel('channel_1').mean() == approx(1.0)


def test_light_unmask():

    myLightRecording.mask_inactivity = False

    assert myLightRecording.get_channel(
        'channel_1').mean() == approx(abnval/nperiods)
