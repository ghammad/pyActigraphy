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

    def fit(
        self,
        threshold=None,
        start_time=None,
        stop_time=None,
        oformat=None,
        verbose=False
    ):
        r"""Compute time spent above thresholds.

        Calculate the total time spent within the specified activity ranges.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            activity ranges.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        oformat: str, optional
            Output format. Available formats: 'minute' or 'timedelta'.
            If set to 'minute', the result is in number of minutes.
            If set to 'timedelta', the result is a pd.Timedelta.
            If set to None, the result is in number of epochs.
            Default is None.
        Returns
        -------
        tat : pd.Series
            Report on the time spent within activity ranges.
        """

        available_formats = [None, 'minute', 'timedelta']
        if oformat not in available_formats:
            raise ValueError(
                'Specified output format ({}) not supported. '.format(oformat)
                + 'Available formats are: {}'.format(str(available_formats))
            )

        if oformat == 'minute':
            scale = self.data.index.freq.delta/pd.Timedelta('1min')
            time_label = 'Time elapsed (min.)'
        elif oformat == 'timedelta':
            scale = self.data.index.freq.delta
            time_label = 'Time elapsed'
        else:
            scale = None

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
        activity_thr = np.concatenate(
            [[-np.infty], activity_thr, [np.infty]]
        )

        if threshold is not None:
            data_mask = self.data.mask(self.data < threshold)
        else:
            data_mask = self.data

        if start_time is stop_time is None:
            data_mask_in = data_mask
        elif (start_time is None) or (stop_time is None):
            raise ValueError(
                'Both start and stop times have to be specified, if any.'
            )
        else:
            data_mask_in = data_mask.between_time(
                start_time=start_time, end_time=stop_time, include_end=False
            )

        out, bins = pd.cut(
            data_mask_in,  # self.data.values,
            bins=activity_thr,
            labels=self.labels,
            retbins=True,
            include_lowest=True,
            duplicates='drop'
        )

        results = data_mask_in.groupby(out).agg(
            ['sum', 'mean', 'median', 'std', 'count']
        )

        if scale:
            results.loc[:, 'count'] *= scale
            results.rename(columns={'count': time_label}, inplace=True)

        self.__results = results
        # Reset index name
        self.__results.index.name = None

    def pretty_results(self):
        r'''DESCRIPTION'''

        return super(ActivityReport, self).pretty_results(transpose=False)
    # pd.DataFrame(self.results).T
