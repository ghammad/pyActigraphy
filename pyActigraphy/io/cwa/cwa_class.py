import os
from datetime import datetime
from struct import Struct
from tools import DataReader


class CWA(object):
    __slots__ = ["__endianess", "__stream", "__meta",
                 "__lastBlock", "__axesTag"]
    __hardware = {0x00: "AX3",
                  0xff: "AX3",
                  0x17: "AX3",
                  0x64: "AX6"}
    __axes = {1: "Axyz",
              2: "Gxyz/Axyz",
              3: "Gxyz/Axyz/Mxyz"
              }
              

    @classmethod
    def IsValidFile(cls, filename):
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
        if ext[1] != ".cwa":
            return False

        f = open(filename, "rb")
        tag = f.read(2).decode("ASCII")
        if tag == "MD" or tag == "DM":
            f.close()
            return True
        f.close()
        return False

    def __init__(self, filename, ):
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        self.__stream = open(filename, "rb")
        self.__endianess = "<"
        self.__meta = dict()
        self.__lastBlock = -1
        self._ReadHeader()
        print(self.__meta)
        self._ReadBlock()
        self._ReadBlock()
        self._ReadBlock()

    def _ReadHeader(self, offset=None):
        """
        reads and parce the cwa header block

        Parameters
        ----------
        offset: int, optional
            if given, will read from given position
            if ommited, will read from current position
        """
        if offset is not None:
            self.__stream.seek(offset)

        dr = DataReader(self.__stream.read(1024), self.__endianess)
        tag = dr.unpack("", 2).decode("ASCII")
        if tag != "MD":
            self.__stream.close()
            raise ValueError("{} not a CWA file"
                             .format(self.__stream.__name__))
        length = dr.unpack("uint16")
        if length != 1020:
            raise ValueError("Packet length must be 1020, but {} is read"
                             .format(length))
        type = dr.unpack("uint8")
        if type not in self.__hardware:
            raise ValueError("Unknown hardware tag: {:x}".format(type))
        self.__meta["hardwareType"] = self.__hardware[type]
        self.__meta["deviceId"] = dr.unpack("uint16") 
        self.__meta["sessionId"] = dr.unpack("uint32")
        device = dr.unpack("uint16") 
        if device == 0xffff:
            device = 0
        self.__meta["deviceId"] += device << 16
        self.__meta["loggingStartTime"] = self.DecodeTime(dr.unpack("uint32"))
        self.__meta["loggingEndTime"] = self.DecodeTime(dr.unpack("uint32"))
        self.__meta["loggingCapacity"] = dr.unpack("uint32")
        dr >> 1
        self.__meta["flashLed"] = dr.unpack("uint8")
        dr >> 8
        self.__meta["sensorConfig"] = dr.unpack("uint8")
        self.__meta["samplingRate"] = self.DecodeRate(dr.unpack("uint8"))
        self.__meta["lastChangeTime"] = self.DecodeTime(dr.unpack("uint32"))
        self.__meta["firmwareRevision"] = dr.unpack("uint8")
        self.__meta["timeZone"] = dr.unpack("int16")
        dr >> 20
        self.__meta["annotation"] = dr.unpack("", 448)\
            .rstrip(b'\x20\x00\xff')
        # 448 ASCII ignore trailing 0x20/0x00/0xff bytes,
        # url-encoded UTF-8 name-value pairs
        self.__meta["reserved"] = dr.unpack("", 512).rstrip(b'\x20\x00\xff')
        # 512 Reserved for device-specific meta-data
        # (512 bytes, ASCII characters,
        # name-value pairs, leading '&' if present?)

    def _ReadBlock(self, offset=None):
        """
        reads and parce the cwa data block

        Parameters
        ----------
        offset: int, optional
            if given, will read from given position
            if ommited, will read from current position
        """
        if offset is not None:
            self.__stream.seek(offset)
        offset = self.__stream.tell()
        dr = DataReader(self.__stream.read(512), self.__endianess)
        tag = dr.unpack("", 2).decode("ASCII")
        if tag != "AX":
            raise ValueError("Corrupted block at {}"
                             .format(offset))
        lenght = dr.unpack("uint16")
        if lenght != 508:
            raise ValueError("Corrupted block at {}"
                             .format(offset))

        # Top bit set: 15-bit fraction of a second for the time stamp,
        # the timestampOffset was already adjusted to minimize
        # this assuming ideal sample rate;
        # Top bit clear: 15-bit device identifier, 0 = unknown;
        deviceFractional = dr.unpack("uint16")
        microseconds = 0
        deviceId = 0
        if deviceFractional >> 15 :
            microseconds = deviceFractional & 0x7fff 
        else :
            deviceId = deviceFractional

        if dr.unpack("uint32") != self.__meta["sessionId"]:
            raise Exception("Block at {}: mismach session id"
                            .format(offset))
        seqId = dr.unpack("uint32")
        timestamp = self.DecodeTime(dr.unpack("uint32"))

        # AAAGGGLLLLLLLLLL
        # Bottom 10 bits is last recorded light sensor value in raw units,
        # 0 = none;#
        # top three bits are unpacked accel scale (1/2^(8+n) g);
        # next three bits are gyro scale  (8000/2^n dps)
        scales = dr.unpack("uint16")
        accelScale, gyroScale, light = self.DecodeScale(scales) 

        temperature = dr.unpack("uint16")
        # Event flags since last packet,
        # b0 = resume logging,
        # b1 = reserved for single-tap event,
        # b2 = reserved for double-tap event,
        # b3 = reserved,
        # b4 = reserved for diagnostic hardware buffer,
        # b5 = reserved for diagnostic software buffer,
        # b6 = reserved for diagnostic internal flag,
        # b7 = reserved)
        events = dr.unpack("uint8")
        battery = dr.unpack("uint8")
        sampleRate = self.DecodeRate(dr.unpack("uint8"))

        # 0x32
        # top nibble: number of axes, 3=Axyz, 6=Gxyz/Axyz, 9=Gxyz/Axyz/Mxyz;
        # bottom nibble: packing format -
        #    2 = 3x 16-bit signed,
        #    0 = 3x 10-bit signed + 2-bit exponent
        numAxesBPS = dr.unpack("uint8")
        numAxes = numAxesBPS >> 4
        if numAxes % 3 != 0:
            raise ValueError("Number of axes must be multiple of 3")
        numAxes /= 3
        enc_style = numAxesBPS & 0xf
        

        # Relative sample index from the start of the buffer where
        # the whole-second timestamp is valid
        timestampOffset = dr.unpack("int16")

        # Number of sensor samples (if this sector is full --
        # Axyz: 80 or 120 samples, Gxyz/Axyz: 40 samples
        sampleCount = dr.unpack("uint16")

        # Raw sample data.
        # Each sample is either 3x/6x/9x 16-bit signed values (x, y, z)
        # or one 32-bit packed value (The bits in bytes [3][2][1][0]:
        #   eezzzzzz zzzzyyyy yyyyyyxx xxxxxxxx, e = binary exponent,
        #   lsb on right)
        # data = dr.unpack("uint32", 120)
        for i in range(0,sampleCount):
            data = [b""] * numAxes
            for i,d in enumerate(data):
                if enc_style == 2:
                    data[i] = dr.unpack("", 3)
                else:
                    data[i] = dr.unpack("", 4)
            values = [None] * numAxes
            if numAxes > 1:
                Gxyz = self.DecodeValue(data enc_style, gyroScale)
            Axyz = self.DecodeValue(data, enc_style, accelScale)
            if numAxes == 9:
                Mxyz = self.DecodeValue(data, enc_style, 1)


        # Checksum of packet (16-bit word-wise sum of
        # the whole packet should be zero
        # checksum = dr.unpack("uint16")

        print("Block ", seqId)
        print("\tmicroseconds ", microseconds)
        print("\tdeviceId ", deviceId)
        print("\tTimestamp {}".format(timestamp))
        print("\tScales: A:1/{}\tG:{}\tL:{}".format(accelScale,
                                                  gyroScale,
                                                  lightScale))
        print("\tTemperature ", temperature)
        print("\tEvent flags ", bin(events))
        print("\tBattery ", battery)
        print("\tSamplerate ", sampleRate)
        print("\tAxes {} ({})".format(self.__axes[numAxes], enc_style))
        print("\ttimestampOffset ", timestampOffset)
        print("\tSamples ", sampleCount)
        print("\n")

    @staticmethod
    def DecodeTime(time):
        # bit pattern:  YYYYYYMM MMDDDDDh hhhhmmmm mmssssss
        year = ((time >> 26) & 0x3f) + 2000
        month = (time >> 22) & 0x0f
        day = (time >> 17) & 0x1f
        hour = (time >> 12) & 0x1f
        minute = (time >> 6) & 0x3f
        second = time & 0x3f
        return datetime(year, month, day, hour, minute, second)

    @staticmethod
    def DecodeRate(rate):
        "Sampling frequency in Hz"
        return 3200 / (1 << (15 - rate & 15)), 16 >> (rate >> 6)

    @staticmethod
    def DecodeScale(data):
        accel = data >> 13
        gyro = (data >> 10) & 0b111  # 0x7
        light = data & 0b1111111111   # 0x3ff
        light = ((light + 512.0)*6000./1024)
        light = pow(10.0, light/1000.0)
        return 2**(8+accel), 4000/(2**gyro), light 

    @staticmethod
    def DecodeValue(data, style, scale):
        """
        decodes a 3x16 bit or 32 data bits into tuple of (x,y,z)
        measurement. If data lenght is 3x16, then values are reocvered
        as int16, if data is 32, the packed values are decoded
        """
        # print(data)
        if style == 2:
            if len(data) != 3:
                raise ValueError("Expected 3 bytes, {} recieved"
                                 .format(len(data)))
            x,y,z =  struct.unpack("<hhh",data)
        elif style == 0:
            print(bin(data[0]))
            if len(data) != 4:
                raise ValueError("Expected 4 bytes, {} recieved"
                                 .format(len(data)))
            exp = data[0] >> 30
            # x = ((data[0] << 6) & 0xffc0 ) >> (6 - exp)
            # y = ((data[0] >> 4) & 0xffc0 ) >> (6 - exp)
            # z = ((data[0] >> 14) & 0xffc0 ) >> (6 - exp)
            # left most bit == sign : x & 0x8000
            # everything elese value: x & 0x7fc0
            # the 2 first bites exponents :  >> (6 - exp)
            x = data[0] << 6
            x = (1 - 2 * (x & 0x8000) )*((x & 0x7fc0 ) >> (6 - exp))
            y = data[0] >> 4
            y = (1 - 2 * (x & 0x8000) )*((y & 0x7fc0 ) >> (6 - exp))
            z = data[0] >> 14
            z = (1 - 2 * (x & 0x8000) )*((z & 0x7fc0 ) >> (6 - exp))
            print ("\t\t", exp, bin(x), x)
            
        else:
            raise ValueError("incorrect triplet lenght")
        return x/scale, y/scale, z/scale

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
