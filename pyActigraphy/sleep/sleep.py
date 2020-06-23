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
            Sleep/wake scoring algorithm to use.
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

    def active_bouts(
        self,
        duration_min=None,
        duration_max=None,
        algo='Roenneberg',
        *args, **kwargs
    ):
        r"""Active bouts.

        Activity periods identified as active.

        Parameters
        ----------
        duration_min: str,optional
            Minimal time duration for an active period.
            Default is None (no filtering).
        duration_max: str,optional
            Maximal time duration for an active period.
            Default is None (no filtering).
        algo: str, optional
            Sleep/wake scoring algorithm to use.
            Default is 'Roenneberg'.
        *args
            Variable length argument list passed to the scoring algorithm.
        **kwargs
            Arbitrary keyword arguements passed to the scoring algorithm.

        Returns
        -------
        active_bouts: a list of pandas.Series


        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.active_bouts(duration_min='2h', algo='Roenneberg')
            XXX

        """

        # Retrieve sleep scoring function dynamically by name
        sleep_algo = getattr(self, algo+'_AoT')

        # Detect activity onset and offset times
        onsets, offsets = sleep_algo(*args, **kwargs)

        # Check if first onset occurs after the first offset
        assert offsets[0] < onsets[0]

        # For each activity period (from onset to offset times)
        #  - Deal with first and last active periods manually

        active_bouts = []

        # First active bout (from the beginning of recording to first offset)
        active_bouts.append(self.data[:offsets[0]])

        for onset, offset in zip(onsets[:-1], offsets[1:]):
            active_bout = self.data[onset:offset]
            active_bouts.append(active_bout)

        # Last active bout (from last onset to the end of the recording)
        active_bouts.append(self.data[onsets[-1]:])

        return filter_ts_duration(active_bouts, duration_min, duration_max)

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
            Sleep/wake scoring algorithm to use.
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

    def active_durations(
        self,
        duration_min=None,
        duration_max=None,
        algo='Roenneberg',
        *args, **kwargs
    ):
        r"""Duration of the active bouts.

        Duration of the activity periods identified as active.

        Parameters
        ----------
        duration_min: str,optional
            Minimal time duration for an active period.
            Default is None (no filtering).
        duration_max: str,optional
            Maximal time duration for an active period.
            Default is None (no filtering).
        algo: str, optional
            Sleep/wake scoring algorithm to use.
            Default is 'Roenneberg'.
        *args
            Variable length argument list passed to the scoring algorithm.
        **kwargs
            Arbitrary keyword arguements passed to the scoring algorithm.

        Returns
        -------
        active_durations: a list of pandas.TimeDelta


        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.active_durations(duration_min='2h', algo='Roenneberg')
            XXX

        """

        # Retrieve sleep bouts
        filtered_bouts = self.active_bouts(
            duration_min=duration_min,
            duration_max=duration_max,
            algo=algo,
            *args, **kwargs
        )

        return [s.index[-1]-s.index[0] for s in filtered_bouts]
