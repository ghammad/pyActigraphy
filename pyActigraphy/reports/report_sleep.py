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
                if (len(fragments) > 0):

                    if (fragments[0].index[0] == sleep_period.index[0]):
                        del fragments[0]

                    if (fragments[-1].index[-1] == sleep_period.index[-1]):
                        del fragments[-1]

                report['SOL'] = sd.distance_to_overlap(convert_to_num_min)
                report['WASO_PCT'] = 1 - sd.overlap_pct(inner=True)
                report['NrAwakening'] = len(fragments)
                report['AwakeningMeanTime'] = np.mean(
                    [self.duration(frag, convert_to_num_min)
                     for frag in fragments]
                )

                self.__results.append(report)

    def pretty_results(self):
        r'''DESCRIPTION'''

        return super(SleepReport, self).pretty_results(transpose=False)


def create_sleep_report(
    sleep_diary,
    scoring,
    states=['NIGHT'],
    state_scoring={'NIGHT': 1},
    convert_td_to_num_min=True,
    verbose=False,
):

    # Check all states have an associated score
    if states != list(state_scoring.keys()):
        warning_msg = (
            'Could not find an associated score for the following states:\n'
            '\n'.join(
                '- {}'.format(list(set(states)-set(state_scoring.keys())))
            )
        )
        print(warning_msg)
        return None

    # Extract periods reported in the sleep diary
    reported_periods = {k: [] for k in states}  # initialize with empty lists

    for idx, row in sleep_diary.diary.iterrows():
        # Skip periods if type is not in the requested list of states
        if row['TYPE'] not in states:
            if verbose:
                print("Skipping reported period nr {} of type: {}".format(
                    idx, row['TYPE']))
            continue
        else:
            # Extract chunk
            chunk = sleep_diary.raw_data.loc[row['START']:row['END']]
            if verbose:
                print(
                    "Found reported period nr {} of type: {}.".format(
                        idx, row['TYPE']
                    )
                    + " START:{} / END:{}".format(
                        row['START'], row['END']
                    )
                )
            if chunk.empty:
                if verbose:
                    print(
                        "-> Skipped since outside of the "
                        "range of the recording."
                    )
                continue

            reported_periods[row['TYPE']].append(chunk)

    sleep_reports = []
    # Create a sleep report for each state:
    for state in states:
        sleep_report = SleepReport(
            sleep_periods=reported_periods[state],
            scoring=scoring,
            sleep_score=sleep_diary.state_index[state],
            target_score=state_scoring[state],
            labels=[state]*len(reported_periods[state]))
        # Fit the current sleep report
        sleep_report.fit(convert_to_num_min=convert_td_to_num_min)

        # Append results
        sleep_reports.append(sleep_report.pretty_results())

    result = pd.concat(
        sleep_reports
    ).sort_values('OnsetTime').reset_index(drop=True)

    return result
