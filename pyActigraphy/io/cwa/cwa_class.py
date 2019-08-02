from datetime import datetime
from struct import Struct

"""
general structure proposal

class CWA(baseRaw)
    def __init__(self, filename)
    def IsValidFile(filename) # checks if file exists and of correct format
    def readHeader(self, data) sets values
    def readBlock(self, data) checks values, returns dataframe
    def normalize(df) returns serie

"""

class CWA(object):
    __slots__ = ["__endianess", "__stream", "__meta", 
                 "_data", "_offset"]
    __hardware = {0x00: "AX3",
                  0xff: "AX3",
                  0x17: "AX3",
                  0x64: "AX6"}

    @classmethod
    def IsValidFile(filename):
        """
        Checks if given file is a valid cwa file.
        Checks is performed by name (do it have .cwa)
        and by leading bits, which must be 'MD'

        Parameters
        ----------
        filename: str
            path to an existing file

        Returns
        -------
        bool
            True if given file is valid cwa file
            False otherwise

        Raises
        ------
        TypeError
            if passed parameters are of incompatible type
        """

        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        ext = os.path.splitext(filename)
        if ext != ".cwa":
            return False

        f = open(filename, "rb")
        tag = f.read(2).decode("ASCII")
        if tag == "MD" or tag == "DM":
            f.close()
            return True
        f.close()
        return False

    def __init__(self, filename):
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        self.__stream = open(filename, "rb")
        self.__aux = dict()
        self._ReadHeader()

    def _ReadHeader(self, offset=None):
        """
        reads and parce the cwa header block

        Parameters
        ----------
        offset: int, optional
            if given, will read from given position
            if ommited, will read from current position
        """
        self._data = self.__stream.read(1024)
        tag = self._data[:2].decode("ASCII")
        if tag == "MD":
            self.__endianess = "<"
        elif tag == "DM":
            self.__endianess = ">"
        else:
            self.__stream.close()
            raise ValueError("{} not a CWA file"
                             .format(self.__stream.__name__))
        self._offset = 2
        lenght = self.__unpack("H")
        if length != 1020:
            raise ValueError("Packet length must be 1020, but {} is read"
                             .format(length))
        type = self.__unpack("B") 
        if type not in self.__hardware:
            raise ValueError("Unknown hardware tag: {:x}".format(type))
        self.__meta["hardwareType"] = self.__hardware[type]
        self.__meta["deviceId"] = self.__unpack("H")
        self.__meta["sessionId"] = self.__unpack("L")
        self.__meta["upperDeviceId"] = self.__unpack("H")
        if self.__meta["upperDeviceId"] == 0xffff:
            self.__meta["upperDeviceId"] = 0
        self.__meta["loggingStartTime"] = self._conertTime(self.__unpack("L"))
        self.__meta["loggingEndTime"] = self._conertTime(self.__unpack("L"))
        self.__meta["loggingCapacity"] = self.__unpack("L")
        self._offset += 1
        self.__meta["flashLed"] = self.__unpack("B")
        self._offset += 8
        self.__meta["sensorConfig"] = self.__unpack("B")
        self.__meta["samplingRate"] = self.__unpack("B")
        self.__meta["lastChangeTime"] = self._conertTime(self.__unpack("L"))
        self.__meta["firmwareRevision"] = self.__unpack("B")
        self.__meta["timeZone"] = self.__unpack("b")
        self._offset += 20
        # self.__meta["annotation"] = 448 ASCII ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs
        # self.__meta["reserved"] = +512 Reserved for device-specific meta-data (512 bytes, ASCII characters, ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs, leading '&' if present?)

    def __unpack(self, type):
        res = struct.unpack_from(self.__endianess + type,
                                 self._data, 
                                 offset=self._offset)
        self._offset += struct.calcsize(type)
        return res

class CwaMixin(object):
    """
    Mixin class for the description of the CWA binary file:
    https://raw.githubusercontent.com/digitalinteraction/openmovement/master
    /Docs/ax3/cwa.h
    """
    def _unpack(self, type, offset):
        return Struct(self._endianness+type).unpack_from(
            buffer=self.data,
            offset=offset
        )[0]

    def _unpack_uchar(self, offset):
        # return Struct('<B').unpack_from(buffer=self.data, offset=offset)[0]
        return self._unpack('B', offset)

    def _unpack_short(self, offset):
        # return Struct('<h').unpack_from(buffer=self.data, offset=offset)[0]
        return self._unpack('h', offset)

    def _unpack_ushort(self, offset):
        return Struct('<H').unpack_from(buffer=self.data, offset=offset)[0]

    def _unpack_uint(self, offset):
        return Struct('<I').unpack_from(buffer=self.data, offset=offset)[0]

    def _unpack_ulong(self, offset):
        return Struct('<L').unpack_from(buffer=self.data, offset=offset)[0]

    def _unpack_timestamp(self, offset):

        ts = Struct('<I').unpack_from(buffer=self.data, offset=offset)[0]
        # bit pattern:  YYYYYYMM MMDDDDDh hhhhmmmm mmssssss
        year = ((ts >> 26) & 0x3f) + 2000
        month = (ts >> 22) & 0x0f
        day = (ts >> 17) & 0x1f
        hour = (ts >> 12) & 0x1f
        minute = (ts >> 6) & 0x3f
        second = ts & 0x3f

        return datetime(year, month, day, hour, minute, second)

    @staticmethod
    def _sampling_frequency(rate):
        "Sampling frequency in Hz"
        return 3200 / (1 << (15 - rate & 15))

    @staticmethod
    def _sampling_range(rate):
        "Sampling range (+/- g)"
        return 16 >> (rate >> 6)

    @staticmethod
    def _num_axes_bps(byte):
        high, low = byte >> 4, byte & 0x0F
        return (high, low)


class CwaHeader(CwaMixin):
    """
    Class for the description of the header of the CWA binary file:
    https://raw.githubusercontent.com/digitalinteraction/openmovement/master
    /Docs/ax3/cwa.h
    """

    hardware_dict = {0: 'AX3', 23: 'AX3', 255: 'AX3', 100: 'AX6'}

    @property
    def data(self):
        return self._data

    @property
    def fname(self):
        return self._fname

    def __init__(self, fname, data, endianness='<'):

        self._fname = fname
        self._data = data

        self._endianness = endianness

        # Position the stream at the start of the file
        # self._stream.seek(0)

        self.header = self.data[:2].decode('ASCII')  # self._unpack_ushort(0)
        self.length = self._unpack_ushort(2)
        self.hardware_type = CwaHeader.hardware_dict[self._unpack_uchar(4)]
        self.device_id = self._unpack_ushort(5)
        self.session_id = self._unpack_ulong(7)
        self.upper_device_id = self._unpack_ushort(11)
        self.logging_start_time = self._unpack_timestamp(13)
        self.logging_stop_time = self._unpack_timestamp(17)
        self.logging_capacity = self._unpack_ulong(21)
        # self.reserved1 = self._unpack_uchar(25)
        self.flash_led = self._unpack_uchar(26)
        # self.reserved2 = unpack('B', self._stream.read(8))[0]
        self.sensor_config = self._unpack_uchar(35)
        self.sampling_rate = self._unpack_uchar(36)
        self.last_change_time = self._unpack_timestamp(37)
        self.firmware_revision = self._unpack_uchar(41)
        self.time_zone = self._unpack_short(42)
        # uint8_t  reserved3[20];                     ///< @44  +20
        # uint8_t  annotation[OM_METADATA_SIZE];      ///< @64  +448
        # Scratch buffer / meta-data (448 ASCII characters, ignore trailing
        # 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs)
        # uint8_t  reserved[512];                     ///< @512 +512
        # Reserved for device-specific meta-data (512 bytes, ASCII characters,
        # ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value
        # pairs, leading '&' if present?)

        # Additional parameters
        self.sampling_freq = CwaHeader._sampling_frequency(self.sampling_rate)
        self.sampling_range = CwaHeader._sampling_range(self.sampling_rate)

    def __str__(self):
        return r"""
        File name: {}
        Header content:
        - header: {}
        - length: {}
        - hardware type: {}
        - device id: {}
        - session id: {}
        - upper device id: {}
        - logging start time: {}
        - logging stop time: {}
        - logging capacity: {}
        - flash LED: {}
        - sensor config: {}
        - sampling rate: {}
        - sampling frequency (Hz): {}
        - sampling range (+/- g): {}
        - last change time: {}
        - firmware revision: {}
        - time zone: {}
        """.format(
            self.fname,
            self.header,
            self.length,
            self.hardware_type,
            self.device_id,
            self.session_id,
            self.upper_device_id,
            self.logging_start_time,
            self.logging_stop_time,
            self.logging_capacity,
            self.flash_led,
            self.sensor_config,
            self.sampling_rate,
            self.sampling_freq,
            self.sampling_range,
            self.last_change_time,
            self.firmware_revision,
            self.time_zone
        )


class CwaBlock(CwaMixin):
    """
    Class for the description of the 512-byte long CWA data block:
    https://raw.githubusercontent.com/digitalinteraction/openmovement/master
    /Docs/ax3/cwa.h
    """
    @property
    def data(self):
        return self._data

    def __init__(self, data, endianness='<'):

        self._data = data

        self._endianness = endianness
        # Position the stream at the start of the file
        # self._stream.seek(0)

        self.header = self.data[:2].decode('ASCII')  # self._unpack_ushort(0)
        self.length = self._unpack_ushort(2)

        self.sampling_rate = self._unpack_uchar(24)
        self.num_axes_bps = self._unpack_uchar(25)
        self.sample_count = self._unpack_ushort(28)

        # Additional parameters
        self.sampling_freq = CwaBlock._sampling_frequency(self.sampling_rate)
        self.sampling_range = CwaBlock._sampling_range(self.sampling_rate)
        self.num_axes, self.bps = CwaBlock._num_axes_bps(self.num_axes_bps)

    def __str__(self):
        return r"""
        Header content:
        - header: {}
        - length: {}
        - sampling rate: {}
        - sampling frequency (Hz): {}
        - sampling range (+/- g): {}
        - number of axes: {}
        - bps: {}
        - sample count: {}
        """.format(
            self.header,
            self.length,
            self.sampling_rate,
            self.sampling_freq,
            self.sampling_range,
            self.num_axes,
            self.bps,
            self.sample_count
        )


def main(argv):

    stream = open(argv, 'rb')
    data = stream.read(1024)

    import CWA
    cwa = CWA.CwaHeader(argv, data)
    print(cwa)

    first_data_block = stream.read(512)
    cwa_first_block = CWA.CwaBlock(first_data_block)
    print(cwa_first_block)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('cwa.py <inputfile>')
        sys.exit(2)

    main(sys.argv[1])
