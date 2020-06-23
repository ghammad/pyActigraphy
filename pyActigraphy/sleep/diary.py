import os
import pandas as pd
import pyexcel as pxl


class SleepDiary():
    """Class for reading sleep diaries."""
    def __init__(
        self,
        input_fname,
        start_time,
        periods,
        frequency,
        header_size=2,
        state_index=dict(ACTIVE=2, NAP=1, NIGHT=0, NOWEAR=-1),
        state_colour=dict(
            NAP='#7bc043',
            NIGHT='#d3d3d3',
            NOWEAR='#ee4035'
        )
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        sd_array = pxl.get_array(file_name=input_fname)

        self.__name = sd_array[0][1]
        self.__diary = pd.DataFrame(
            sd_array[header_size+1:],
            columns=sd_array[header_size]).astype({
                'TYPE': 'str',
                'START': 'datetime64[ns]',
                'END': 'datetime64[ns]'
            })  # .drop(['DURATION (min)'], axis=1).dropna()

        # Inplace drop of useless columns
        # TODO: find a way to drop all the useless cols in a programmatical way
        self.__diary.drop(
            columns=['DURATION (min)'],
            inplace=True,
            errors='ignore'
        )

        # Inplace drop of NA
        self.__diary.dropna(inplace=True)

        self.__state_index = state_index
        self.__state_colour = state_colour

        # Create a time series with ACTIVE as default value.
        self.__raw_data = pd.Series(
            data=self.__state_index['ACTIVE'],
            index=pd.date_range(
                start_time,
                periods=periods,
                freq=frequency
            ),
            dtype=int
        )

        # Replace the default value with the ones found in the sleep diary.
        for index, row in self.__diary.iterrows():
            self.__raw_data[
                row['START']:row['END']
            ] = self.__state_index[row['TYPE']]

        # Create a template shape to overlay over a plotly plot
        self.__shaded_area = dict(
            type='rect',
            xref='x',
            yref='paper',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            fillcolor='',
            opacity=0.5,
            layer='below',
            line=dict(width=0)
        )

    @property
    def name(self):
        """The name of the subject."""
        return self.__name

    @property
    def diary(self):
        """The dataframe containing the data found in the sleep diary."""
        return self.__diary

    @property
    def state_index(self):
        """The indices assigned to the states found in the sleep diary."""
        return self.__state_index

    @state_index.setter
    def state_index(self, value):
        self.__state_index = value

    @property
    def state_colour(self):
        """The colours assigned to the states found in the sleep diary."""
        return self.__state_colour

    @state_colour.setter
    def state_colour(self, value):
        self.__state_colour = value

    @property
    def raw_data(self):
        """The time series related to the states found in the sleep diary."""
        return self.__raw_data

    @property
    def shaded_area(self):
        """The template shape which can be overlaid over a plotly plot of the
        associated actimetry time series."""
        return self.__shaded_area

    @shaded_area.setter
    def shaded_area(self, value):
        self.__shaded_area = value

    def shapes(self):
        """ """
        shapes = []
        for index, row in self.__diary.iterrows():
            shape = self.__shaded_area.copy()
            shape['x0'] = row['START']
            shape['x1'] = row['END']
            shape['fillcolor'] = self.__state_colour[row['TYPE']]
            shapes.append(shape)
        return shapes

    def summary(self):
        """ Returns a dataframe of summary statistics."""
        if 'DURATION' not in self.__diary.columns:
            self.__diary['DURATION'] = self.__diary['END']\
                - self.__diary['START']
        return self.__diary.groupby(['TYPE'])['DURATION'].describe()

    def state_infos(self, state):
        """ Returns summary statistics for a given state

        Parameters
        ----------
        state: str
            State of interest
        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.
        """

        # Re-use the summary function
        summary = self.summary()

        # Verify that the state is present in the summary object
        if state not in summary.index:
            raise KeyError(
                "{} is not a valid state. Valid states are {}".format(
                    state, '" or "'.join(summary.index)
                )
            )

        # Access the summary object to get the mean
        mean = summary.loc[state, 'mean']
        # Access the summary object to get the std
        std = summary.loc[state, 'std']

        return mean, std

    def total_bed_time(self, state='NIGHT'):
        """ Returns the total in-bed time

        Parameters
        ----------
        state : str, optional
            State of interest.
            Default is 'NIGHT'.

        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.

        """

        return self.state_infos(state)

    def total_nap_time(self, state='NAP'):
        """ Returns the total nap time

        Parameters
        ----------
        state : str, optional
            State of interest.
            Default is 'NAP'.

        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.

        """

        return self.state_infos(state)

    def total_nowear_time(self, state='NOWEAR'):
        """ Returns the total 'no-wear' time

        Parameters
        ----------
        state : str, optional
            State of interest.
            Default is 'NOWEAR'.

        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.

        """

        return self.state_infos(state)
