from ..sleep.scoring.utils import consecutive_values


class ScoringDescriptor():

    def __init__(self, truth, scoring, truth_target=1, scoring_target=1):
        self.truth = (truth == truth_target).astype(int)
        self.scoring = (scoring == scoring_target).astype(int)
        self.truth_target = truth_target
        self.scoring_target = scoring_target

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

    def non_overlap_fragments(self, inner=False, min_length=0):

        overlap_series = self.overlap(inner=inner, cast_to_int=True)

        fragment_indices = consecutive_values(
            overlap_series.values, target=0, min_length=min_length
        )

        return [overlap_series.iloc[indices[0]:indices[1]]
                for indices in fragment_indices]

    def distance_to_overlap(self, convert_to_num_min=False):

        # Search for the position of the first epoch where truth and
        # scoring overlap
        pos = self.overlap(inner=True, cast_to_int=True).idxmax()

        d = pos - self.truth.index[0]

        return d.total_seconds()/60 if convert_to_num_min else d
