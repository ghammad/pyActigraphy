import numpy as np
import pandas as pd
from .report import Report


def _is_percentile(q_arr):
    return all(0 <= qs <= 1 for qs in q_arr)


class ActivityReport(Report):
    r"""Class for activity report"""

    def __init__(self, data, cut_points=None, labels=None):

        if (cut_points is not None) and (labels is not None):
            if len(labels) != len(cut_points)+1:
                raise ValueError(
                    "The number of labels should match the number of (N+1)"
                    + " intervals defined by the N specified cut points.\n"
                    + "Number of labels: {}\n".format(len(labels))
                    + "Number of cut points: {}\n".format(len(cut_points))
                )
        # call __init__ function of the base class
        super(ActivityReport, self).__init__(data=data)

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

        return super(ActivityReport, self).pretty_results(transpose=True)
    # pd.DataFrame(self.results).T
