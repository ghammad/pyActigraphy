from .report import Reports
import numpy as np
import pandas as pd


class SleepReports(Reports):
    r"""Class for sleep reports"""

    def __init__(self, data, sleep_state=1, label=None):

        # call __init__ function of the base class
        super().__init__(data=data)

        # store sleep state
        self.__sleep_state = sleep_state

        # store label for the sleep state
        self.__label = label

        # Current results
        self.__results = None

    @property
    def sleep_state(self):
        r'''Sleep state accessor'''
        return self.__sleep_state

    @sleep_state.setter
    def sleep_state(self, value):
        self.__sleep_state = value

    @property
    def label(self):
        r'''Label accessor'''
        return self.__label

    @property
    def results(self):
        r'''Result accessor'''
        return self.__results

    def fit(self, verbose=False):
        r'''DESCRIPTION'''
        pass
