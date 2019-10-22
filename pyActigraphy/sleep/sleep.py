# from ..io import BaseRaw, RawReader
# from ..io.base import BaseRaw


class SleepBoutMixin(object):
    """ Mixin Class for identifying sleep bouts"""

    def sleep_bouts(self, method='crespo'):
        r"""Sleep bouts.

        Activity periods identified as sleep.

        Parameters
        ----------
        method: str
            Method used to identify continuous sleep bouts.
            Available methods are: 'crespo', 'roenneberg'
            Default is 'crespo'.

        Returns
        -------
        sleep_bouts: a list of pandas.Series


        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.sleep_bouts()
            XXX

        """

        methods = ('crespo', 'roenneberg')
        if method not in methods:
            raise ValueError(
                '`method` must be "%s". You passed: "%s"' %
                ('" or "'.join(methods), method)
            )

        # Create a RawReader instance
        sleep_bouts = []  # RawReader(reader_type)

        # Identify sleep bouts
        if method == 'crespo':
            # retrieve activity onset and offset times
            onsets, offsets = self.Crespo_AoT()
        elif method == 'roenneberg':
            # retrieve activity onset and offset times
            onsets, offsets = self.Roenneberg_AoT()

        # For each inactivity period (from offset to onset times)
        # create a BaseRaw object
        for onset, offset in zip(onsets, offsets):
            sleep_bout = self.data[offset:onset]

            sleep_bouts.append(sleep_bout)

        return sleep_bouts
