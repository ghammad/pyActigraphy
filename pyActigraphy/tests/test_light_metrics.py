import numpy as np
import pandas as pd
from pyActigraphy.light import LightRecording
from pytest import approx

# Synthetic data characteristics
sf = '30s'  # Sampling frequency
nperiods = int(1440*(pd.Timedelta('1min')/pd.Timedelta(sf)))  # nr epoch/day
ndays = 3

A = 1000  # Amplitude
mu = nperiods/2  # Noon
sigma = 360  # 3h

x = np.linspace(0, nperiods-1, nperiods)

# Synthetic data
f_ch1 = np.tile(
    A/(sigma*np.sqrt(2*np.pi))*np.exp(-np.square((x-mu)/(np.sqrt(2)*sigma))),
    ndays
)
f_ch2 = 10 * f_ch1

data_df = pd.DataFrame(
    data={'channel_1': f_ch1, 'channel_2': f_ch2},
    index=pd.date_range(
        start='01/01/2000 00:00:00',
        periods=nperiods*ndays,
        freq=sf
    )
)

# LightRecroding instanciation
myLightRecording = LightRecording(
    name='LightRecording',
    uuid='MultiChanXYZ',
    data=data_df,
    frequency=data_df.index.freq,
    log10_transform=False
)


def test_light_average_daily_profile_auc():

    assert myLightRecording.average_daily_profile_auc(
        channel='channel_1') == approx(A, rel=0.0001)
