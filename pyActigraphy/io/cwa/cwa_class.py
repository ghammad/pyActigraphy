import os
from datetime import datetime
from datetime import timedelta
from tools import DataReader
import pandas
import numpy
import struct


class CWA(object):
    __slots__ = ["__endianess", "__data", "__meta",
                 "__nblocks", "__nsamples",
                 "__device",
                 "__name", "__uuid", "__startTime",
                 "__duration", "__period",
                 "__Accel", "__Gyro", "__Mag",
                 "__long_index_array", "__accel_array",
                 "__gyro_array", "__mag_array",
                 "__short_index_array", "__others_array",
                 "__Ascale", "__Gscale", "__Mscale",
                 "__Others"]
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
        if tag == "MD":
            f.close()
            return True
        f.close()
        return False

    def __init__(self, filename, ):
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        with open(filename, "rb") as f:
            self.__data = f.read()
        self.__endianess = "<"
        self.__meta = dict()
        self.__device = 0
        self.__name = 0
        self.__uuid = 0
        self.__startTime = datetime.min
        self.__duration = timedelta()
        self.__period = 0
        self.__Accel = None  # pandas.DataFrame(columns=["x","y", "z","norm"])
        self.__Gyro = None   # pandas.DataFrame(columns=["x","y", "z","norm"])
        self.__Mag = None    # pandas.DataFrame(columns=["x","y", "z","norm"])
        self.__Ascale = 0
        self.__Gscale = 0
        self.__Mscale = 0
        self.__Others = None

        self.__nblocks = len(self.__data) - 1024
        if self.__nblocks % 512 != 0:
            raise ValueError("Data file corrupted")
        self.__nblocks = self.__nblocks // 512

        self.__nsamples = 0
        for blk in range(0, self.__nblocks):
            offset = 1024 + 512 * blk + 28
            self.__nsamples += struct.unpack("<H", self.__data[offset:offset + 2])[0]

        self.__long_index_array = numpy.zeros([self.__nsamples], dtype='datetime64[us]')
        self.__accel_array = numpy.full([self.__nsamples,4], numpy.nan)
        self.__gyro_array = numpy.full([self.__nsamples,4], numpy.nan)
        self.__mag_array = numpy.full([self.__nsamples,4], numpy.nan)

        self.__short_index_array = numpy.zeros([self.__nblocks], dtype='datetime64[us]')
        self.__others_array = numpy.full([self.__nblocks, 3], numpy.nan)

        self._ReadHeader()
        self.printHeader()

        t = datetime.now()
        sample = 0
        for blk in range(0, self.__nblocks):
            rem = blk % 1000
            if rem == 0:
                print("Block {} of {} ({}%) {}"
                      .format(blk,self.__nblocks,
                              100. * blk / self.__nblocks,
                              datetime.now() - t)
                      )
                t = datetime.now()
            sample += self._ReadBlock(blk, sample)

        self.__Accel = pandas.DataFrame(self.__accel_array,
                                        index=self.__long_index_array,
                                        columns=["x","y","z","norm"])
        self.__Gyro = pandas.DataFrame(self.__gyro_array,
                                       index=self.__long_index_array,
                                       columns=["x","y","z","norm"])
        self.__Mag = pandas.DataFrame(self.__mag_array,
                                      index=self.__long_index_array,
                                      columns=["x","y","z","norm"])
        self.__Others = pandas.DataFrame(self.__others_array,
                                         index=self.__short_index_array,
                                         columns=["light","temperature", "battery"])
        print("Accelorimeter:")
        print(self.__Accel.head())
        print(self.__Accel.memory_usage())
        print("Gyroscope")
        print(self.__Gyro.head())
        print(self.__Gyro.memory_usage())
        print("Magnetoscope")
        print(self.__Mag.head())
        print(self.__Mag.memory_usage())
        print("Others")
        print(self.__Others.head())
        print(self.__Others.memory_usage())

    def _ReadHeader(self):
        """
        reads and parce the cwa header block

        Parameters
        ----------
        offset: int, optional
            if given, will read from given position
            if ommited, will read from current position
        """
        dr = DataReader(self.__data[0:1024], self.__endianess)

        # @ 0  +2   ASCII "MD", little-endian (0x444D)
        tag = dr.unpack("", 2).decode("ASCII")
        if tag != "MD":
            self.__stream.close()
            raise ValueError("{} not a CWA file")

        # @ 2  +2   Packet length (1020 bytes, 
        # with header (4) = 1024 bytes total)
        length = dr.unpack("uint16")
        if length != 1020:
            raise ValueError("Packet length must be 1020, but {} is read"
                             .format(length))

        # @ 4  +1 * Hardware type (0x00/0xff/0x17 = AX3, 0x64 = AX6)
        type = dr.unpack("uint8")
        if type not in self.__hardware:
            raise ValueError("Unknown hardware tag: {:x}".format(type))
        self.__device = self.__hardware[type]

        # @ 5  +2   Device identifier (lower 16-bits)
        self.__uuid = dr.unpack("uint16")
        # @ 7  +4   Unique session identifier
        self.__name = dr.unpack("uint32")
        # @11  +2 * Upper word of device id
        # (if 0xffff is read, treat as 0x0000)
        device = dr.unpack("uint16") 
        if device == 0xffff:
            device = 0
        self.__uuid += (device << 16)

        # @13  +4   Start time for delayed logging
        self.__startTime = self.DecodeTime(dr.unpack("uint32"))
        # @17  +4   Stop time for delayed logging
        self.__duration = self.DecodeTime(dr.unpack("uint32"))\
            - self.__startTime
        # @21  +4   (Deprecated: preset maximum number of samples to collect,
        # 0 = unlimited)
        self.__meta["loggingCapacity"] = dr.unpack("uint32")
        # @25  +1   (1 byte reserved)
        dr >> 1
        # @26  +1   Flash LED during recording
        self.__meta["flashLed"] = dr.unpack("uint8")
        # @27  +8   (8 bytes reserved)
        dr >> 8
        # @35  +1 * Fixed rate sensor configuration,
        # 0x00 or 0xff means accel only,
        # otherwise bottom nibble is gyro range (8000/2^n dps): 
        #   2=2000, 3=1000, 4=500, 5=250, 6=125, 
        # top nibble non-zero is magnetometer enabled.
        sensor = dr.unpack("uint8")
        if sensor == 0 or sensor == 0xff:
            self.__Gscale = 0
            self.__Mscale = 0
        else:
            self.__Gscale = 1 << (sensor & 0x0f)
            self.__Mscale = 1 if (sensor >> 4) > 0 else 0
        # @36  +1   Sampling rate code, 
        # frequency (3200/(1<<(15-(rate & 0x0f)))) Hz, 
        # range (+/-g) (16 >> (rate >> 6))
        freq, scale = self.DecodeRate(dr.unpack("uint8"))
        self.__period = 1. / freq
        self.__Ascale = (1 << scale)
        # @37  +4   Last change metadata time
        self.__meta["lastChangeTime"] = self.DecodeTime(dr.unpack("uint32"))
        # @41  +1   Firmware revision number
        self.__meta["firmwareRevision"] = dr.unpack("uint8")
        # @42  +2   (Unused: originally reserved for a "Time Zone offset from
        # UTC in minutes", 0xffff = -1 = unknown)
        self.__meta["timeZone"] = dr.unpack("int16")
        dr >> 20
        # @64  +448 Scratch buffer / meta-data (448 ASCII characters,
        # ignore trailing 0x20/0x00/0xff bytes, 
        # url-encoded UTF-8 name-value pairs)
        self.__meta["annotation"] = dr.unpack("", 448)\
            .rstrip(b'\x20\x00\xff')
        # 512 Reserved for device-specific meta-data
        # (512 bytes, ASCII characters,
        # name-value pairs, leading '&' if present?)
        self.__meta["reserved"] = dr.unpack("", 512).rstrip(b'\x20\x00\xff')

    def _ReadBlock(self, block, sample):
        """
        reads and parce the cwa data block

        Parameters
        ----------
        offset: int, optional
            if given, will read from given position
            if ommited, will read from current position
        """
        offset = 1024 + 512 * block
        data = self.__data[offset: offset + 512]
        if len(data) == 0:
            return None
        if len(data) != 512:
            raise ValueError("Corrupted block at {}: unexpected EOF"
                             .format(offset))
        dr = DataReader(data, self.__endianess)
        # checksum
        # ch = dr.checksum("uint16")
        # if (ch & 0xffff) != 0:
        #    raise ValueError("Corrupted block at {}: checksum failed"
        #                     .format(offset))

        # @ 0  +2   ASCII "AX", little-endian (0x5841)
        tag = dr.unpack("", 2).decode("ASCII")
        if tag != "AX":
            raise ValueError("Corrupted block at {}: incorrect tag"
                             .format(offset))
        # @ 2  +2   Packet length (508 bytes,
        # with header (4) = 512 bytes total)
        lenght = dr.unpack("uint16")
        if lenght != 508:
            raise ValueError("Corrupted block at {}: incorrect lenght"
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
        # @ 6  +4   Unique session identifier, 0 = unknown
        sesId = dr.unpack("uint32")
        if sesId != 0 and sesId != self.__name:
            raise Exception("Block at {}: mismach session id"
                            .format(offset))
        # @10  +4   Sequence counter (0-indexed), 
        # each packet has a new number (reset if restarted)
        seqId = dr.unpack("uint32")
        # @14  +4   Last reported RTC value, 0 = unknown
        timestamp = self.DecodeTime(dr.unpack("uint32"))

        # @18  +2   AAAGGGLLLLLLLLLL
        # Bottom 10 bits is last recorded light sensor value in raw units,
        # 0 = none;#
        # top three bits are unpacked accel scale (1/2^(8+n) g);
        # next three bits are gyro scale  (8000/2^n dps)
        scales = dr.unpack("uint16")
        accelScale, gyroScale, light = self.DecodeScale(scales) 
        if accelScale != 0 and accelScale != self.__Ascale:
            self.__Ascale = accelScale
            print("Block at {}: new Accel scale - {}"
                  .format(offset, self.__Ascale))
        if gyroScale != 0 and gyroScale != self.__Gscale:
            self.__Gscale = gyroScale
            print("Block at {}: new Gyro scale - {}"
                  .format(offset, self.__Gscale))
        # @20  +2   Last recorded temperature sensor value
        # in raw units, 0 = none
        temperature = dr.unpack("uint16")
        if temperature == 0:
            temperature = float("NaN")
        else:
            temperature = (temperature * 150.0 - 20500) / 1000
        # @22  +1   Event flags since last packet,
        # b0 = resume logging,
        # b1 = reserved for single-tap event,
        # b2 = reserved for double-tap event,
        # b3 = reserved,
        # b4 = reserved for diagnostic hardware buffer,
        # b5 = reserved for diagnostic software buffer,
        # b6 = reserved for diagnostic internal flag,
        # b7 = reserved)
        events = dr.unpack("uint8")
        # @23  +1   Last recorded battery level in scaled/cropped 
        # raw units (double and add 512 for 10-bit ADC value),
        # 0 = unknown
        battery = dr.unpack("uint8")
        if battery == 0:
            battery = float("NaN")
        else:
            battery = (battery + 512.0) * 6 / 1024
        # @24  +1   Sample rate code, 
        # frequency (3200/(1<<(15-(rate & 0x0f)))) Hz, 
        # range (+/-g) (16 >> (rate >> 6))
        freq, scale = self.DecodeRate(dr.unpack("uint8"))
        if self.__period != 1. / freq:
            raise Exception("Block at {}: mismach frequency"
                            .format(offset))
        if self.__Ascale != (1 << scale):
            raise Exception("Block at {}: mismach accelometer scale {}"
                            .format(offset, 1 << scale))

        # 0x32
        # top nibble: number of axes, 3=Axyz, 6=Gxyz/Axyz, 9=Gxyz/Axyz/Mxyz;
        # bottom nibble: packing format -
        #    2 = 3x 16-bit signed,
        #    0 = 3x 10-bit signed + 2-bit exponent
        numAxesBPS = dr.unpack("uint8")
        numAxes = numAxesBPS >> 4
        if numAxes % 3 != 0:
            raise ValueError("Number of axes must be multiple of 3")
        numAxes = numAxes // 3
        enc_style = numAxesBPS & 0xf

        if numAxes < 1 and self.__Ascale != 0:
            print("Block {}: missing accelometer data".format(offset))
        if numAxes < 2 and self.__Gscale != 0:
            print("Block {}: missing gyrometer data".format(offset))
        if numAxes < 3 and self.__Mscale != 0:
            print("Block {}: missing magnetometer data".format(offset))

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

        self.__short_index_array[block] = numpy.datetime64(timestamp)
        self.__others_array[block] = [light, temperature, battery]

        if enc_style == 2:
            values = dr.unpack("int16", 3 * numAxes * sampleCount)
        else:
            values = [self.DecodeValue(v) 
                      for v in dr.unpack("uint32", numAxes * sampleCount)]

        for s in range(0, sampleCount):
            self.__long_index_array[sample + s] \
                    = numpy.datetime64(timestamp + 
                                       timedelta(seconds=self.__period))
            if numAxes == 0:
                break
            if numAxes == 1:
                vector = [v / self.__Ascale 
                          for v in values[s]]
                self.__accel_array[sample + s] \
                    = vector + [numpy.linalg.norm(vector)] 
            elif numAxes == 2:
                vector = [v / self.__Gscale 
                          for v in values[s]]
                self.__gyro_array[sample + s] \
                    = vector + [numpy.linalg.norm(vector)] 
                vector = [v / self.__Gscale 
                          for v in values[s + 1]]
                self.__accel_array[sample + s] \
                    = vector + [numpy.linalg.norm(vector)] 
            else:
                vector = [v / self.__Gscale 
                          for v in values[s]]
                self.__gyro_array[sample + s] \
                    = vector + [numpy.linalg.norm(vector)] 
                vector = [v / self.__Gscale 
                          for v in values[s + 1]]
                self.__accel_array[sample + s] \
                    = vector + [numpy.linalg.norm(vector)] 
                vector = [v / self.__Mscale 
                          for v in values[s + 2]]
                self.__mag_array[sample + s] \
                    = vector + [numpy.linalg.norm(vector)] 

        return sampleCount

    def printHeader(self):
        print("Header content:\n"
              "- device id: {} ({})\n"
              "- session: {}\n"
              "- sampling period (s): {}\n"
              "- sampling frequency (Hz): {}\n"
              "- Accel scale: {}\n"
              "- Gyro scale: {}\n"
              "- Magneto scale: {}:\n"
              "- StartTime: {}\n"
              "- Duration: {}\n"
              "- Nmb. blocks: {}\n"
              "- Nmb. samples: {}\n"
              .format(
                    self.__uuid, self.__device,
                    self.__name,
                    self.__period,
                    1. / self.__period,
                    self.AccelScale(),
                    self.GyroScale(),
                    self.MagnetoScale(),
                    self.__startTime.isoformat(),
                    self.__duration,
                    self.__nblocks,
                    self.__nsamples
                     )
              )

        print("Metadata:")
        for meta, data in self.__meta.items():
            print("- {}: {}".format(meta, data))

    def AccelRange(self):
        if self.__Ascale > 0:
            return 2 / self.__Ascale
        else:
            return None

    def AccelScale(self):
        return self.__Ascale

    def GyroRange(self):
        if self.__Gscale > 0:
            return 4000 / self.__Gscale
        else:
            return None

    def GyroScale(self):
        return self.__Gscale

    def MagnetoRange(self):
        if self.__Mscale > 0:
            return 1. / self.__Mscale
        else:
            return None

    def MagnetoScale(self):
        return self.__Mscale

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
        if accel > 0:
            accel = 1 << accel
        gyro = (data >> 10) & 0b111  # 0x7
        if gyro != 0:
            gyro = 1 << gyro

        light = data & 0b1111111111   # 0x3ff
        light = ((light + 512.0) * 6000. / 1024)
        light = pow(10.0, light / 1000.0)
        return accel, gyro, light 

    @classmethod
    def DecodeValue(cls, data):
        """
        decodes a uint32 packed data into tuple of (x,y,z)
        measurement.
        Unpacking follows:
        [byte-3] [byte-2] [byte-1] [byte-0]
        eezzzzzz zzzzyyyy yyyyyyxx xxxxxxxx
        where x/y/z represents the 3 10-bits values.
        The left most bit of each value is sign bit.
        The e represents the binary exponent value (0-3)
        number of bits to left-shift all axis values
        """
        exp = data >> 30
        x = cls.__decode(data) << exp
        y = cls.__decode(data >> 10) << exp
        z = cls.__decode(data >> 20) << exp
        return x, y, z

    @staticmethod
    def __decode(value):
        # taking last 10 bits
        value = value & 0x3ff
        # left bit is sign
        sign = (value & 0x200) >> 9 
        value = (value & 0x1ff)
        return (1 - 2 * sign) * value
