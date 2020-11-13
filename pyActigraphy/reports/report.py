# from pyActigraphy.io.base import BaseRaw
import numpy as np
import pandas as pd


def _is_percentile(q_arr):
    return all(0 <= qs <= 1 for qs in q_arr)


class Reports():
    r"""Base class for reports

    Parameters
    ----------
    data: pd.Series
        Time series
    """

    def __init__(self, data):

        # store data
        self.__data = data

    # Overload add operator?
    def __add__(self, other):
        pass

    @property
    def data(self):
        r"""Data accessor"""
        return self.__data

    # def time_in_state(self, state_nr):
    #     return pd.Timedelta(
    #         len(self.data[self.data == state_nr])*self.data.index.freq
    #     )
    #
    # def activity_in_state(self, state_nr):
    #     return pd.Timedelta(
    #         len(self.data[self.data == state_nr])*self.data.index.freq
    #     )


class ActivityReports(Reports):
    r"""Class for activity reports"""

    def __init__(self, data, cut_points=None, labels=None):

        if (cut_points is not None) and (labels is not None):
            if len(labels) != len(cut_points)+1:
                raise ValueError(
                    "The number of labels should match the number of (N+1)" +
                    " intervals defined by the N specified cut points.\n" +
                    "Number of labels: {}\n".format(len(labels)) +
                    "Number of cut points: {}\n".format(len(cut_points))
                )
        # call __init__ function of the base class
        super().__init__(data=data)

        # store cut points
        self.__cut_points = cut_points

        # store labels for the cut points
        self.__labels = labels

        # Current results
        self.__results = None

    @property
    def cut_points(self):
        r'''Cut point accessor'''
        return self.__cut_points

    @cut_points.setter
    def cut_points(self, value):
        self.__cut_points = value

    @property
    def labels(self):
        r'''Label accessor'''
        return self.__labels

    @property
    def results(self):
        r'''Result accessor'''
        return self.__results

    def fit(self, verbose=False):
        r'''DESCRIPTION'''
        if not self.cut_points:
            print('Set cut points before running the fit method.')
            return None

        # if cut-points are percentiles, determine associated activity counts
        if _is_percentile(self.cut_points):

            # Convert percentiles into activity thresholds
            activity_thr = self.data.quantile(self.cut_points).values
            if verbose:
                print('Cut-points are interpreted as percentiles.')
                print('Corresponding activity counts:')
                print('\n'.join(
                    ['- {}: {}'.format(k, v) for k, v in zip(
                        self.cut_points, activity_thr)]
                    )
                )
        else:
            activity_thr = self.cut_points
            if verbose:
                print('Cut-points are interpreted as activity thresholds.')

        # Add min/max activity counts to form boundaries
        activity_thr = np.concatenate([[-1], activity_thr, [np.infty]])

        out, bins = pd.cut(
            self.data.values,
            activity_thr,
            labels=self.labels,
            retbins=True,
            duplicates='drop'
        )

        self.__results = out.describe()['freqs']
        # Reset index name
        self.__results.index.name = None

    def pretty_results(self):
        r'''DESCRIPTION'''

        if self.results is None:
            print(
                "Results are empty." +
                " Please run the fit method before accessing the resuls."
            )
        else:
            return pd.DataFrame(self.results).T
