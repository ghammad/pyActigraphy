from ..sleep.scoring.utils import consecutive_values


class ScoringDescriptor():

    def __init__(self, truth, scoring, truth_score=1, sleep_score=1):
        self.truth = (truth == truth_score).astype(int)
        self.scoring = (scoring == sleep_score).astype(int)
        self.truth_score = truth_score
        self.sleep_score = sleep_score

    def overlap(self, inner=False, cast_to_int=False):

        overlap_series = (self.truth & self.scoring)
        if inner:
            overlap_series = overlap_series.loc[
                self.truth.index[0]:self.truth.index[-1]
            ]

        return overlap_series.astype(int) if cast_to_int else overlap_series

    def overlap_pct(self, inner=False):

        return self.overlap(inner=inner).mean()

    def overlap_fragments(self, inner=False, min_length=0):

        overlap_series = self.overlap(inner=inner, cast_to_int=True)

        fragment_indices = consecutive_values(
            overlap_series.values, target=1, min_length=min_length
        )

        return [overlap_series.iloc[indices[0]:indices[1]]
                for indices in fragment_indices]

    def distance_to_overlap(self):

        # Search for the position of the first epoch where truth and
        # scoring overlap
        pos = (self.truth & self.scores).idxmax()

        return pos - self.truth.index[0]
