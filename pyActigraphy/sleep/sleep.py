from ..utils.filters import filter_ts_duration


class SleepBoutMixin(object):
    """ Mixin Class for identifying sleep bouts"""

    def sleep_bouts(
        self,
        duration_min=None,
        duration_max=None,
        algo='Roenneberg',
        *args, **kwargs
    ):
        r"""Sleep bouts.

        Activity periods identified as sleep.

        Parameters
        ----------
        duration_min: str,optional
            Minimal time duration for a sleep period.
            Default is None (no filtering).
        duration_max: str,optional
            Maximal time duration for a sleep period.
            Default is None (no filtering).
        algo: str, optional
            Sleep scoring algorithm to use.
            Default is 'Roenneberg'.
        *args
            Variable length argument list passed to the scoring algorithm.
        **kwargs
            Arbitrary keyword arguements passed to the scoring algorithm.

        Returns
        -------
        sleep_bouts: a list of pandas.Series


        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.sleep_bouts(duration_min='2h', algo='Roenneberg')
            XXX

        """

        # Retrieve sleep scoring function dynamically by name
        sleep_algo = getattr(self, algo+'_AoT')

        # Detect activity onset and offset times
        onsets, offsets = sleep_algo(*args, **kwargs)

        # For each inactivity period (from offset to onset times)
        sleep_bouts = []
        for onset, offset in zip(onsets, offsets):
            sleep_bout = self.data[offset:onset]
            sleep_bouts.append(sleep_bout)

        return filter_ts_duration(sleep_bouts, duration_min, duration_max)

    def sleep_durations(
        self,
        duration_min=None,
        duration_max=None,
        algo='Roenneberg',
        *args, **kwargs
    ):
        r"""Duration of the sleep bouts.

        Duration of the activity periods identified as sleep.

        Parameters
        ----------
        duration_min: str,optional
            Minimal time duration for a sleep period.
            Default is None (no filtering).
        duration_max: str,optional
            Maximal time duration for a sleep period.
            Default is None (no filtering).
        algo: str, optional
            Sleep scoring algorithm to use.
            Default is 'Roenneberg'.
        *args
            Variable length argument list passed to the scoring algorithm.
        **kwargs
            Arbitrary keyword arguements passed to the scoring algorithm.

        Returns
        -------
        sleep_durations: a list of pandas.TimeDelta


        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.sleep_durations(duration_min='2h', algo='Roenneberg')
            XXX

        """

        # Retrieve sleep bouts
        filtered_bouts = self.sleep_bouts(
            duration_min=duration_min,
            duration_max=duration_max,
            algo=algo,
            *args, **kwargs
        )

        return [s.index[-1]-s.index[0] for s in filtered_bouts]
