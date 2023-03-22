import pandas as pd


class Report():
    r"""Base class for report

    Parameters
    ----------
    data: pd.Series
        Time series
    """

    def __init__(self, data):

        # store data
        self.__data = data

        # store results
        self.__results = None

    # Overload add operator?
    def __add__(self, other):
        pass

    @property
    def data(self):
        r"""Data accessor"""
        return self.__data

    @property
    def results(self):
        r"""Results accessor"""
        return self.__results

    def pretty_results(self, transpose=False):
        r'''DESCRIPTION'''

        if self.results is None:
            print(
                "Results are empty."
                + " Please run the fit method before accessing the resuls."
            )
        else:
            pretty_results = pd.DataFrame(self.results)
            return pretty_results.T if transpose else pretty_results
