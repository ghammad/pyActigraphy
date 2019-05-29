# from ..io import BaseRaw, RawReader
# from ..io.base import BaseRaw


class SleepBoutMixin(object):
    """ Mixin Class for identifying sleep bouts"""

    def sleep_bouts(raw, method='crespo'):
        r"""Sleep bouts.

        Activity periods identified as sleep.

        Parameters
        ----------
        method: str
            Method used to identify contiuous sleep bouts.
            Available methods are: 'crespo', 'chrono'
            Default is 'crespo'.

        Returns
        -------
        sbs: a list of pandas.Series


        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> raw.sleepbouts()
            XXX

        """

        methods = ('crespo', 'chrono')
        if method not in methods:
            raise ValueError(
                '`method` must be "%s". You passed: "%s"' %
                ('" or "'.join(methods), method)
            )

        # identify reader type
        # reader_type = raw.format

        # create a RawReader instance
        sleep_bouts = []  # RawReader(reader_type)

        # identify sleep bouts
        if method == 'crespo':
            # retrieve activity onset and offset times
            onsets, offsets = raw.Crespo_AoT()

            # for each inactivity period (from offset to onset times)
            # create a BaseRaw object
            for onset, offset in zip(onsets, offsets):
                sleep_bout = raw.data[offset:onset]
                # BaseRaw(
                #     name='{} (night: {})'.format(raw.name, 0),
                #     uuid=raw.uuid,
                #     format=raw.format,
                #     axial_mode=raw.axial_mode,
                #     start_time=offset,
                #     period=onset - offset,
                #     frequency=raw.frequency,
                #     data=raw.data[offset:onset],
                #     light=(raw.light[offset:onset] if raw.light is not None
                #            else None)
                # )
                # sleep_bout.mask_inactivity = raw.mask_inactivity
                # sleep_bout.inactivity_length = raw.inactivity_length
                # sleep_bout.exclude_if_mask = raw.exclude_if_mask
                # sleep_bout.sleep_diary = raw.sleep_diary

                sleep_bouts.append(sleep_bout)

        return sleep_bouts
