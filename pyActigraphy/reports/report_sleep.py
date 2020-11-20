from .report import Report
from .utils import ScoringDescriptor
import numpy as np
import pandas as pd


class SleepReport(Report):
    r"""Class for sleep report"""

    def __init__(
        self,
        sleep_periods,
        sleep_score=1,
        scoring=None,
        target_score=None,
        labels=None
    ):

        # call __init__ function of the base class
        super(SleepReport, self).__init__(data=sleep_periods)

        # store sleep score
        self.__sleep_score = sleep_score

        # store target score
        self.__target_score = target_score

        # store labels for the sleep state
        self.__labels = labels

        # store scoring
        self.__scoring = scoring

        # Current results
        self.__results = None

    @property
    def sleep_score(self):
        r'''Sleep score accessor'''
        return self.__sleep_score

    @sleep_score.setter
    def sleep_score(self, value):
        self.__sleep_score = value

    @property
    def target_score(self):
        r'''Target score accessor'''
        return self.__target_score

    @target_score.setter
    def target_score(self, value):
        self.__target_score = value

    @property
    def labels(self):
        r'''Label accessor'''
        return self.__labels

    @property
    def scoring(self):
        r'''Scoring accessor'''
        return self.__scoring

    @property
    def results(self):
        r'''Result accessor'''
        return self.__results

    @classmethod
    def onset(cls, bout):
        return bout.index[0]

    @classmethod
    def offset(cls, bout):
        return bout.index[-1]

    @classmethod
    def duration(cls, bout, convert_to_num_min=False):
        length = cls.offset(bout) - cls.onset(bout)
        # Add one extra period to account for the last epoch
        length += bout.index.freq
        if convert_to_num_min:
            length = length.total_seconds()/60
        return length

    def fit(self, convert_to_num_min=False, min_length=None, verbose=False):
        r'''DESCRIPTION'''

        self.__results = []

        for idx, sleep_period in enumerate(self.data):
            report = {}
            report['Label'] = self.labels[idx]
            report['OnsetTime'] = self.onset(sleep_period)
            report['OffsetTime'] = self.offset(sleep_period)
            report['Duration'] = self.duration(
                sleep_period, convert_to_num_min
            )
            if self.scoring is not None:
                sd = ScoringDescriptor(
                    truth=sleep_period,
                    scoring=self.scoring,
                    truth_target=self.sleep_score,
                    scoring_target=self.target_score
                )
                min_length_in_epoch = (
                    pd.Timedelta(min_length)//sleep_period.index.freq
                ) if min_length is not None else 0

                fragments = sd.non_overlap_fragments(
                    inner=True, min_length=min_length_in_epoch
                )
                # Remove first and last fragments if the start/stop coincide
                # with the onset/offet times
                if fragments[0].index[0] == sleep_period.index[0]:
                    del fragments[0]
                if fragments[-1].index[-1] == sleep_period.index[-1]:
                    del fragments[-1]

                report['SOL'] = sd.distance_to_overlap(convert_to_num_min)
                report['WASO_PCT'] = 1 - sd.overlap_pct(inner=True)
                report['NoAwakening'] = len(fragments)
                report['AwakeningMeanTime'] = np.mean(
                    [self.duration(frag, convert_to_num_min)
                     for frag in fragments]
                )

                self.__results.append(report)

    def pretty_results(self):
        r'''DESCRIPTION'''

        return super(SleepReport, self).pretty_results(transpose=False)
