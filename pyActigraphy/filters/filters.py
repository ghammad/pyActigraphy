def _create_inactivity_mask(data, duration, threshold):
    """Create a mask for periods of a given duration below a given threshold"""

    return data.groupby(
        # create identical 'labels' for identical consecutive numbers
        [data.diff().ne(0).cumsum()]
    ).transform(
        # 0: windows of length 'duration" with null activity
        # 1: otherwise
        lambda x: not((x.size > duration) & (x.sum() < threshold))
    ).astype('int')


class FiltersMixin(object):
    """ Mixin Class """

    def create_inactivity_mask(self, duration):
        """Create a mask for inactivity periods"""
        self.mask = _create_inactivity_mask(self.raw_data, duration, 1)
