import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import json

from ..base import BaseRaw
from pyActigraphy.light import LightRecording


class RawFTB(BaseRaw):

    """Raw object from JSON file (recorded by Fitbit)

    Parameters
    ----------
    path_to_fitbit: str, optional
        Path to the folder structure from Fitbit download (.../Fitbit/...). Must be provided if no preloaded data is provided.
        Default is None.
    preloaded_data: pd.DataFrame, optional
        Preloaded Fitbit data. Must contain columns 'calories' and 'heart'.
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    data_dtype: dtype, optional
        The dtype of the raw data.
        Default is np.int.
    light_dtype: dtype, optional
        The dtype of the raw light data. Important: Fitbit does not record light! The heart rate is taken as surrogate.
        Default is np.float.
    """

    def __init__(
        self,
        path_to_fitbit=None,
        preloaded_data=None,
        start_time=None,
        period=None,
        name=None
    ):
        
        # read files
        self._url = path_to_fitbit
        self.__assert_input(path_to_fitbit, preloaded_data)
        raw_data = self.__reading_and_parsing_file(preloaded_data)
        raw_data = self.__preprocess_raw_data(raw_data)

        # extract informations from the header
        start = self.__extract_ftb_start_time(raw_data)
        frequency = self.__extract_frequency(raw_data)
        frequency_light = frequency # same frequency as "activity"

        activity_data = self.__extract_activity_data(raw_data)
        light_data = self.__extract_light_data(raw_data)
        
        # index the motion time serie
        index_data = pd.Series(
            data=activity_data.values,
            index=pd.date_range(
                start=start,
                periods=len(activity_data),
                freq=frequency
            ),
            dtype=float
        )
            
        # index the light time serie
        if light_data is not None:
            index_light = pd.Series(
                data=light_data.values,
                index=pd.date_range(
                    start=start,
                    periods=len(light_data),
                    freq=frequency_light
                    ),
                dtype=float
        )

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = start

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = index_data.index[-1]
            period = stop_time - start_time
    
        # Create a new index covering the entire desired period
        full_index = pd.date_range(start=start_time, end=stop_time, freq=frequency)
        index_data = index_data.reindex(full_index, fill_value=pd.NA)
        index_light = index_light.reindex(full_index, fill_value=pd.NA)

        # call __init__ function of the base class
        super().__init__(
            fpath=path_to_fitbit,
            name=name,
            uuid=None, # not available for Fitbit
            format='FTB',
            axial_mode=None,
            start_time=start_time,
            period=period,
            frequency=frequency,
            data=index_data,
            light=LightRecording(
                name=name,
                uuid=None,
                data=index_light.to_frame(name='whitelight'),
                frequency=frequency_light
            ) if index_light is not None else None
        )

    @property
    def white_light(self):
        r"""Value of the light intensity in µw/cm²."""
        if self.light is None:
            return None
        else:
            return self.light.get_channel("whitelight")
    
    def __assert_input(self, path_to_fitbit, preloaded_data):
        if path_to_fitbit is None and preloaded_data is None:
            raise ValueError("Must provide a path to the Fitbit data 'path_to_fitbit' if no preloaded data is provided.")
    
    def __reading_and_parsing_file(self, preloaded_data):
        """ Load the data from the JSON file(s) or return the preloaded data. """
        if preloaded_data is not None:
            return preloaded_data
        elif os.path.isdir(self._url):
            return self.__reading_and_parsing_from_json(self._url)
        else:
            raise ValueError(f" Invalid path {self._url}.")

    def __reading_and_parsing_from_json(self, path_to_data: str) -> pd.DataFrame:
        """ Load the raw data from JSON file """    
        df_calories = self.__load_fitbit_json(path_to_data, 'calories')
        df_heart = self.__load_fitbit_json(path_to_data, 'heart')
        
        # identify the slowest frequency and resample the dataframes
        frequency = max(pd.infer_freq(df_calories.index[:3]), 
                        pd.infer_freq(df_heart.index[:3]))
        df_calories = df_calories.resample(frequency).mean(numeric_only=True)
        df_heart = df_heart.resample(frequency).mean(numeric_only=True)
        
        df = pd.concat([df_calories, df_heart], axis=1)
        df = df.sort_index() # sort chronologically
        return df

    def __load_fitbit_json(self, path_to_data, prefix):
        """ Open and read the JSON file. Save as pandas.DataFrame. """
        fnames = self.__get_fnames(path_to_data, prefix)
        data_list = self.__read_json_fast(path_to_data, fnames)
        df = pd.concat([pd.DataFrame(data) for data in data_list], ignore_index=True) 
        
        if prefix == 'heart':
            df['value'] = df['value'].apply(lambda x: x['bpm'])
        
        df['dateTime'] = pd.to_datetime(df['dateTime'], format='%m/%d/%y %H:%M:%S')
        df['value'] = df['value'].astype('float32')
        
        df = df.set_index('dateTime').rename(columns={'value': prefix})
        return df      
            
    def __get_fnames(self, path_to_data, prefix):
        """ Get a list of all available JSON files for the given sensor type ('calories' | 'heart'). """
        file_list = os.listdir(path_to_data)
        fnames = [filename for filename in file_list if filename.startswith(prefix) and filename.endswith('.json')]
        if len(fnames) == 0:
            raise FileNotFoundError(f"No files found with prefix '{prefix}' in '{path_to_data}'")
        return fnames  
           
    def __read_json_fast(self, path_to_data, fnames):
        """ Read JSON files in parallel. """
        def process_file(fname):
            file_path = os.path.join(path_to_data, fname)
            with open(file_path, 'r') as file:
                return json.load(file)
        with ThreadPoolExecutor() as executor:
            data_list = list(executor.map(process_file, fnames))
        return data_list      

    def __extract_ftb_start_time(self, df):
        """ Extract start time from the raw dataframe"""
        return df.index[0]

    def __extract_frequency(self, df):
        """ Extract frequency from the raw dataframe"""
        return pd.Timedelta(1, unit=pd.infer_freq(df.index[:3]))

    def __extract_activity_data(self, df):
        """ Extract calories as surrogate activity measurement from the raw dataframe """
        return df['calories']

    def __extract_light_data(self, df):
        """ Extract heart rate measurement from the raw dataframe as surrogate 'light' """
        return df['heart']
    
    def __preprocess_raw_data(self, df):
        """ Normalize the calories and heart rate. """
        df['heart'] = self.__zeros_to_nan(df, 'heart')
        df['heart'] = self.__minmax_scaling(df, 'heart')
        df['calories'] = self.__zeros_to_nan(df, 'calories') 
        df['calories'] = self.__preprocess_calories(df, 'calories')
        return df
    
    def __zeros_to_nan(self, df, var):
        """ set zero values to nan """
        to_nan = (df[var] <= 0)  
        col_idx = df.columns.get_loc(var)
        df.iloc[to_nan, col_idx] = np.nan
        return df[var]
        
    def __minmax_scaling(self, df, var):      
        """ Normalize the data by subtracting the minimum and dividing by the maximum. """
        df[var] -= df[var].min()
        df[var] /= df[var].max()
        return df[var]
    
    def __preprocess_calories(self, df, var):
        """ Normalize the calories in segments."""
        baseline = self.__get_baseline(df, var)
        df[var] = df[var] - baseline.values
        df[var] = self.__minmax_scaling(df, var)
        return df[var]    
    
    def __get_baseline(self, df, var):
        """ estimate the baseline. """
        # get daily baseline
        daily_offset = df.groupby(df.index.date)[var].min()
        baseline = pd.Series(df.index.date).map(daily_offset)
        return baseline

def read_raw_ftb(
    path_to_fitbit=None,
    preloaded_data=None,
    start_time=None,
    period=None,
    name=None
):
    """Reader function for raw .json file recorded by Fitbit.

    Parameters
    ----------
    path_to_fitbit: str, optional
        Path to the folder structure from Fitbit download (.../Fitbit/...). Must be provided if no preloaded data is provided.
        Default is None.
    preloaded_data: pd.DataFrame, optional
        Preloaded Fitbit data. Must contain columns 'calories' and 'heart'.
        Default is None.
    start_time: datetime-like str
        If not None, the start_time will be used to slice the data.
        Default is None.
    period: str
        Default is None.
    data_dtype: dtype
        The dtype of the raw data. Default is np.int.
    light_dtype: dtype
        The dtype of the raw light data. Default is np.float.

    Returns
    -------
    raw : Instance of RawFTB
        An object containing raw FTB data
    """

    return RawFTB(
        path_to_fitbit=path_to_fitbit,
        preloaded_data=preloaded_data,
        start_time=start_time,
        period=period,
        name=name
    )
