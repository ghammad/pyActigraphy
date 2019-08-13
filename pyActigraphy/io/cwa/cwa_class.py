import os
from datetime import datetime
from datetime import timedelta
from ..tools import DataReader
import pandas
import numpy
import math

from ..base import BaseRaw


class CWA(BaseRaw):
    r"""
    Raw object from .CWA file (recorded by AX3, AX6 actiwatches)

    Parameters
    ----------
    input_fname: str
        Path to CWA file
    time_correction: str
        type of timestamps correction:
        "" : no correction, times as in file
        "time" : start time of each block is corrected, 
        but frequency remain as in file
        "freq" : both start times and frequency are affected
    time_fractional : bool
        to use or not fractional time
    raw : bool
        retrieved values are raw (unscaled)
    dtype: str
        numpy data tipe to store values actimetry values in dataframe
    """
    __slots__ = [
                 "__filename", 
                 "__endianess", 
                 "__data", 

                 "__device_type",  # type of devise (AX3, AX6)
                 "__device_id",    # 16 bits unique device id
                 "__session_id",   # id of session, 
                                   # interpreted as __name for base class 
                 "__start_time",   # logging start time, as in header
                 "__duration",     # logging duration, as in header
                 "__period",       # period between 2 succesive measurements

                 "__nblocks",      # total number of blocks
                 "__nsamples",     # total number of samples over blocks
                 "__nevents",      # total number of events
                 "__naxes",        # number of axes (accel, gyro, mag)
                 "__encoding",     # type of data encoding 
                                   #   0: uint32, x,y,z packed
                                   #   2: 3x int16 for x,y,z

                 "__accel_scale",  # scale for accelorimeter
                 "__accel_scale_def",  # default scale for accelorimeter
                 "__accel_df",     # dataframe for accelorimeter
                 "__accel_np",     # np array for accelorimeter

                 "__gyro_scale",   # scale for gyrometer
                 "__gyro_scale_def",   # default scale for gyrometer
                 "__gyro_df",      # dataframe for gyrometer
                 "__gyro_np",      # np array for gyrometer

                 "__mag_scale",    # scale for magnetometer     
                 "__mag_scale_def",  # default scale for magnetometer
                 "__mag_df",       # dataframe for magnetometer
                 "__mag_np",       # np array for magnetometer

                 "__aux_df",       # dataframe for auxiliary data
                 "__aux_np",       # np array for auxiliary data
                 "__evt_df",       # dataframe for events
                 "__evt_np",       # np array for events

                 "__index",        # time index for measurements
                 "__aux_index",    # time index for auxiliary data
                 "__evt_index",    # time index for events

                 "__meta",         # a dict containing non-standard info 
                                   # about recording
                 "__time_correction",  # type of correction for timestamps
                                       # 0 -- no correction
                                       # 1 -- timestamps corrected
                                       # 2 -- frequency corrcted
                 "__time_fractional",  # to use or not fractional time

                 "__last_start_time",  # Starting time of previous block
                 "__last_end_time",    # End time of previous block
                 "__sample_count"      # Sample count of previous block 
                 ]
    __hardware = {0x00: "AX3", 
                  0xff: "AX3",
                  0x17: "AX3",
                  0x64: "AX6"}

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

    def __init__(self, 
                 input_fname, 
                 time_correction="", 
                 time_fractional=True, 
                 raw=False, 
                 dtype="float32"
                 ):
        times = [datetime.now()]
        if not isinstance(input_fname, str):
            raise TypeError("input_fname must be a string")
        with open(input_fname, "rb") as f:
            self.__data = f.read()
        if len(self.__data) < 1024:
            raise ValueError("{}: file less than 1024 bytes"
                             .format(input_fname))
        self.__filename = input_fname
        self.__endianess = "<"
        self.__device_type = ""
        self.__device_id = 0
        self.__session_id = 0
        self.__start_time = datetime.min
        self.__duration = timedelta()
        self.__period = 0
        self.__nblocks = 0
        self.__nsamples = 0
        self.__nevents = 0
        self.__naxes = 1
        self.__encoding = None
        self.__accel_scale = None
        self.__accel_scale_def = 256  # 32768 = 2000dps
        self.__accel_df = None
        self.__accel_np = None
        self.__gyro_scale = None
        self.__gyro_scale_def = 2000  # 32768 = 2000dps
        self.__gyro_df = None
        self.__gyro_np = None
        self.__mag_scale = None
        self.__mag_scale_def = 16  # 1uT = 16
        self.__mag_df = None
        self.__mag_np = None

        self.__aux_df = None
        self.__aux_np = None
        self.__evt_df = None
        self.__evt_np = None

        self.__index = None
        self.__aux_index = None
        self.__evt_index = None

        self.__meta = dict()

        self.__last_start_time = None
        self.__last_end_time = None
        self.__sample_count = None

        if time_correction == "":
            self.__time_correction = 0
        elif time_correction == "time":
            self.__time_correction = 1
        elif time_correction == "freq":
            self.__time_correction = 2
        else:
            raise ValueError("Unknown methode for time correction")
        self.__time_fractional = time_fractional

        self.__read_header()
        self.print_header()
        times.append(datetime.now())
        print("Header dtime:", times[-1] - times[-2])

        self.__nblocks = len(self.__data) - 1024
        if self.__nblocks % 512 != 0:
            print("{}: data lenght not multiple of 512 bytes, "
                  "last block might be corrupted"
                  .format(self.__filename))
        self.__nblocks = self.__nblocks // 512

        for blk in range(0, self.__nblocks):
            if not self.__scan_block(blk):
                self.__nblocks = blk + 1
                break

        times.append(datetime.now())
        print("Block scan dtime: {} ({} per block)"
              .format(times[-1] - times[-2],
                      (times[-1] - times[-2]) / self.__nblocks)
              )

        self.__index = numpy.empty([self.__nsamples],
                                   dtype='datetime64[ns]')
        self.__aux_index = numpy.empty([self.__nblocks],
                                       dtype='datetime64[ns]')
        self.__evt_index = numpy.empty([self.__nevents],
                                       dtype='datetime64[ns]')
        self.__aux_np = numpy.empty([self.__nblocks, 3], 
                                    dtype="float32")
        self.__evt_np = numpy.empty([self.__nevents, 1], 
                                    dtype="uint8")

        if raw:
            self.__accel_scale = 1.
            self.__gyro_scale = 1.
            self.__mag_scale = 1.
            dtype = "int16"

        if self.__naxes > 0:
            if self.__accel_scale is None:
                raise ValueError("Having accelorimeter "
                                 "but scale is not defined")
            self.__accel_np = numpy.empty([self.__nsamples,4], 
                                          dtype=dtype)
        if self.__naxes > 1:
            if self.__gyro_scale is None:
                raise ValueError("Having gyrometer "
                                 "but scale is not defined")
            self.__gyro_np = numpy.empty([self.__nsamples,4],
                                         dtype=dtype)
        if self.__naxes > 2:
            if self.__mag_scale is None:
                raise ValueError("Having magnetometer "
                                 "but scale is not defined")
            self.__mag_np = numpy.empty([self.__nsamples,4],
                                        dtype=dtype)
        times.append(datetime.now())
        print("Array dtime:", times[-1] - times[-2])

        sample = 0
        event = 0
        times.append(datetime.now())
        for blk in range(0, self.__nblocks):
            rem = blk % 10000
            if rem == 0:
                print("Block {} of {} ({}%) {}"
                      .format(blk,self.__nblocks,
                              100. * blk / self.__nblocks,
                              datetime.now() - times[-1])
                      )
                times[-1] = datetime.now()
            s, e = self.__read_block(blk, sample, event, raw)
            sample += s
            event += e

        if self.__accel_np is not None:
            self.__accel_df = pandas.DataFrame(self.__accel_np,
                                               index=self.__index,
                                               columns=["x","y","z","norm"])
            print("Accelorimeter:")
            print(self.__accel_df.head())
            print(self.__accel_df.memory_usage())
        if self.__gyro_np is not None:
            self.__gyro_df = pandas.DataFrame(self.__gyro_np,
                                              index=self.__index,
                                              columns=["x","y","z","norm"])
            print("Gyroscope")
            print(self.__gyro_df.head())
            print(self.__gyro_df.memory_usage())
        if self.__mag_np is not None:
            self.__mag_df = pandas.DataFrame(self.__mag_np,
                                             index=self.__index,
                                             columns=["x","y","z","norm"])
            print("Magnetoscope")
            print(self.__mag_df.head())
            print(self.__mag_df.memory_usage())

        self.__aux_df = pandas.DataFrame(self.__aux_np,
                                         index=self.__aux_index,
                                         columns=["light",
                                                  "temperature",
                                                  "battery"])
        print("Auxiliary")
        print(self.__aux_df.head())
        print(self.__aux_df.memory_usage())
        self.__evt_df = pandas.DataFrame(self.__evt_np,
                                         index=self.__evt_index,
                                         columns=["events"])
        print("Events")
        print(self.__evt_df.head())
        print(self.__evt_df.memory_usage())

        times.append(datetime.now())
        print("DataFrame creation dtime: {}"
              .format(times[-1] - times[-2]))
        print("Total user time: ", times[-1] - times[0])

        super().__init__(
                name=self.__session_id,
                uuid=self.__device_id,
                format="CWA",
                axial_mode='tri-axial',
                start_time=pandas.datetime64(self.__start_time),
                period=pandas.timedelta64(self.__duration),
                frequency=pandas.timedelta64(self.__period),
                data=self.accelometry_norm,
                light=self.auxiliary_light
                )

    def __read_header(self):
        """
        reads and parce the cwa header block
        """
        dr = DataReader(self.__data[0:1024], self.__endianess)

        tag = dr.unpack("", 2).decode("ASCII")
        if tag != "MD":
            self.__stream.close()
            raise ValueError("{} not a CWA file")

        length = dr.unpack("uint16")
        if length != 1020:
            raise ValueError("Packet length must be 1020, but {} is read"
                             .format(length))

        type = dr.unpack("uint8")
        if type not in self.__hardware:
            raise ValueError("Unknown hardware tag: {:x}".format(type))
        self.__device_type = self.__hardware[type]

        self.__device_id = dr.unpack("uint16")
        self.__session_id = dr.unpack("uint32")
        device = dr.unpack("uint16") 
        if device == 0xffff:
            device = 0
        self.__device_id += (device << 16)

        self.__start_time = self.DecodeTime(dr.unpack("uint32"))
        self.__duration = self.DecodeTime(dr.unpack("uint32"))\
            - self.__start_time
        dr >> 4
        dr >> 1
        self.__meta["flashLed"] = dr.unpack("uint8")
        dr >> 8
        sensor = dr.unpack("uint8")
        if sensor != 0 and sensor != 0xff:
            self.__naxes = 2
            self.__gyro_scale = 8000 / (1 << (sensor & 0x0f))
            if (sensor >> 4) > 0:
                self.__naxes = 3
                self.__mag_scale = self.__mag_scale_def
        freq, scale = self.DecodeRate(dr.unpack("uint8"))
        self.__period = 1. / freq
        self.__accel_scale = (1 << scale)
        self.__meta["lastChangeTime"] = self.DecodeTime(dr.unpack("uint32"))
        self.__meta["firmwareRevision"] = dr.unpack("uint8")
        self.__meta["timeZone"] = dr.unpack("int16")
        dr >> 20
        self.__meta["annotation"] = dr.unpack("", 448)\
            .rstrip(b'\x20\x00\xff')
        self.__meta["reserved"] = dr.unpack("", 512).rstrip(b'\x20\x00\xff')

    def __scan_block(self, block):
        offset = 1024 + 512 * block
        dr = DataReader(self.__data[offset:offset + 512], "<")
        if dr.unpack("",2) != b'AX':
            print("Block at {}: header is not 'AX'".format(offset))
            return False
        if dr.unpack("uint16") != 508:
            print("Block at {}: block size is not 508".format(offset))
            return False
        if dr.unpack_at("uint8",22) > 0:
            self.__nevents += 1
        numAxes = dr.unpack_at("uint8", 25)
        if numAxes >> 4 != 3 * self.__naxes:
            print("Block at {}: mismatch number of axis".format(offset))
            return False
        if self.__encoding is None:
            self.__encoding = numAxes & 0x0f
        elif self.__encoding != numAxes & 0x0f:
            print("Block at {}: mismatch data format".format(offset))
            return False

        if (dr.checksum("uint16") & 0xffff) != 0:
            print("Block at {}: checksum failed".format(offset))
            return False
        self.__nsamples += dr.unpack_at("uint16", 28)
        return True

    def __read_block(self, block, sample, event, raw=False):
        """
        reads and parce the cwa data block

        Parameters
        ----------
        block: int
            index to block to read
        sample: int
            index to sample in arrays corresponding
            to first entry of block
        """
        offset = 1024 + 512 * block
        data = self.__data[offset: offset + 512]
        if len(data) == 0:
            return None
        if len(data) != 512:
            raise ValueError("Corrupted block at {}: unexpected EOF"
                             .format(offset))
        dr = DataReader(data, self.__endianess)
        deviceFractional = dr.unpack_at("uint16",4)
        seqId = dr.unpack_at("uint32",10)

        # Top bit set: 15-bit fraction of a second for the time stamp,
        # the timestampOffset was already adjusted to minimize
        # this assuming ideal sample rate;
        # Top bit clear: 15-bit device identifier, 0 = unknown;
        fractional = 0
        if self.__time_fractional and deviceFractional >> 15 :
            # use 15 bits number as 16 bits
            # represents 1/65536 fraction of second
            fractional = (deviceFractional & 0x7fff) * 2 / 65536
        timestamp = self.DecodeTime(dr.unpack_at("uint32", 14))

        # @18  +2   AAAGGGLLLLLLLLLL
        # Bottom 10 bits is last recorded light sensor value in raw units,
        # 0 = none;#
        # top three bits are unpacked accel scale (1/2^(8+n) g);
        # next three bits are gyro scale  (8000/2^n dps)
        scales, temp = dr.unpack_at("uint16", 18, 2)
        accelScale, gyroScale, light = self.DecodeScale(scales) 
        if not raw and accelScale != 0 and accelScale != self.__accel_scale:
            self.__accel_scale = accelScale
            print("Block at {}: new accelorimeter scale - {}"
                  .format(offset, self.__accel_scale))
        if not raw and gyroScale != 0 and gyroScale != self.__gyro_scale:
            self.__gyro_scale = gyroScale
            print("Block at {}: new gyrometer scale - {}"
                  .format(offset, self.__gyro_scale))
        light = self.scale_light(light)
        temp = self.scale_temp(temp)
        evt, battery, rate = dr.unpack_at("uint8", 22, 3)
        battery = self.scale_battery(battery)
        freq, scale = self.DecodeRate(rate)
        if self.__period != 1. / freq:
            self.__period = 1. / freq
            print("Block at {}: new sampling period - {}"
                  .format(offset, self.__period))
        if not raw and self.__accel_scale != (1 << scale):
            raise Exception("Block at {}: mismach accelometer scale {}"
                            .format(offset, 1 << scale))

        timestampOffset, sampleCount = dr.unpack_at("int16", 26, 2)

        if self.__time_correction == 0:
            fractional = fractional % self.__period 
            timeoffset = fractional - timestampOffset * self.__period
            timestamp = timestamp + timedelta(seconds=timeoffset)
            period = timedelta(seconds=self.__period)
        elif self.__time_correction == 1:
            fractional = fractional % self.__period 
            timeoffset = fractional - timestampOffset * self.__period
            timestamp = timestamp + timedelta(seconds=timeoffset)
            period = timedelta(seconds=self.__period)
            if self.__last_start_time is None or seqId == 0:
                self.__last_start_time = timestamp
                self.__last_end_time = timestamp\
                    + timedelta(seconds=self.__period * sampleCount)
                self.__sample_count = sampleCount
            else:
                t0 = self.__last_end_time
                self.__last_start_time = timestamp
                self.__last_end_time = timestamp\
                    + timedelta(seconds=self.__period * sampleCount)
                if (timestamp - t0).total_seconds() < 1:
                    timestamp = t0
        elif self.__time_correction == 2:
            timeoffset = fractional 
            timestampOffset += int(fractional // self.__period)
            timestamp = timestamp + timedelta(seconds=timeoffset)

            if self.__last_start_time is None or seqId == 0:
                self.__last_start_time = timestamp
                self.__sample_count = sampleCount
                self.__last_end_time = timestampOffset - sampleCount
                period = timedelta(seconds=self.__period)
            else:
                period = (timestamp - self.__last_start_time)\
                        / (timestampOffset - self.__last_end_time)
                self.__last_start_time = timestamp
                self.__sample_count = sampleCount
                self.__last_end_time = timestampOffset - sampleCount
            timestamp -= timestampOffset * period
        else:
            raise ValueError("Unknown type of correction")

        self.__aux_index[block] = numpy.datetime64(timestamp)
        self.__aux_np[block] = [light, temp, battery]

        if evt > 0:
            self.__evt_index[block] = numpy.datetime64(timestamp)
            self.__evt_np[block] = [evt]
            event += 1

        if self.__encoding == 2:
            values = dr.unpack_at("int16", 30, 3 * self.__naxes * sampleCount)
        elif self.__encoding == 0:
            values = [self.DecodeValue(v) 
                      for v in dr.unpack_at("uint32", 30, 
                                            self.__naxes * sampleCount)]
            values = [val for sublist in values for val in sublist]
        else:
            raise Exception("Invalid encoding")

        for s in range(0, sampleCount):
            self.__index[sample + s] \
                    = numpy.datetime64(timestamp)
            timestamp += period
            if self.__naxes == 0:
                break
            if self.__naxes == 1:
                self.__accel_np[sample + s] = \
                        self.__create_vector(values[3 * s: 3 * s + 3], 
                                             self.__accel_scale, raw)
            elif self.__naxes == 2:
                self.__gyro_np[sample + s] = \
                        self.__create_vector(values[3 * s: 3 * s + 3],
                                             self.__gyro_scale, raw)
                self.__accel_np[sample + s] = \
                    self.__create_vector(values[3 * (s + 1): 3 * (s + 2)],
                                         self.__accel_scale, raw)
            else:
                self.__gyro_np[sample + s] = \
                    self.__create_vector(values[3 * s: 3 * s + 3], 
                                         self.__gyro_scale, raw)
                self.__accel_np[sample + s] = \
                    self.__create_vector(values[3 * (s + 1): 3 * (s + 2)],
                                         self.__accel_scale, raw)
                self.__mag_np[sample + s] = \
                    self.__create_vector(values[3 * (s + 2): 3 * (s + 3)],
                                         self.__mag_scale, raw)

        return sampleCount, event

    def print_header(self, meta=True):
        """
        printout the information retrieved from header

        Parameters
        ----------
        meta: bool
            printout additional metadata
        """
        print("Header content:\n"
              "- file: {}\n"
              "- device id: {} ({})\n"
              "- session: {}\n"
              "- sampling period: {}s ({}Hz)\n"
              "- Accelometer scale: {}\n"
              "- Gyrometer scale: {}\n"
              "- Magnetometer scale: {}:\n"
              "- StartTime: {}\n"
              "- Duration: {}\n"
              .format(
                    self.__filename,
                    self.__device_id, self.__device_type,
                    self.__session_id,
                    self.__period, 1. / self.__period,
                    self.__accel_scale,
                    self.__gyro_scale,
                    self.__mag_scale,
                    self.__start_time.isoformat(),
                    self.__duration,
                     )
              )
        if meta:
            print("Metadata:")
            for meta, data in self.__meta.items():
                print("- {}: {}".format(meta, data))

    def AccelRange(self):
        if self.__accel_scale > 0:
            return 512 * self.__accel_scale
        else:
            return None

    def GyroRange(self):
        if self.__gyro_scale is not None:
            return 512 * self.__gyro_scale
        else:
            return None

    def MagnetoRange(self):
        if self.__mag_scale is not None:
            return 512 * self.__mag_scale
        else:
            return None

    def MagnetoScale(self):
        return self.__mag_scale

    @property
    def accelometry(self):
        return self.__accel_df

    @property
    def accelometry_norm(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["norm"]
        else:
            return None

    @property
    def accelometry_x(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["x"]
        else:
            return None

    @property
    def accelometry_y(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["y"]
        else:
            return None

    @property
    def accelometry_z(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["z"]
        else:
            return None

    @property
    def gyrometry(self):
        return self.__accel_df

    @property
    def gyrometry_norm(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["norm"]
        else:
            return None

    @property
    def gyrometry_x(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["x"]
        else:
            return None

    @property
    def gyrometry_y(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["y"]
        else:
            return None

    @property
    def gyrometry_z(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["z"]
        else:
            return None

    @property
    def magnetometry(self):
        return self.__accel_df

    @property
    def magnetometry_norm(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["norm"]
        else:
            return None

    @property
    def magnetometry_x(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["x"]
        else:
            return None

    @property
    def magnetometry_y(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["y"]
        else:
            return None

    @property
    def magnetometry_z(self):
        if self.__accel_df is not None:
            return self.__accel_df.loc["z"]
        else:
            return None

    @property
    def auxiliary(self):
        return self.__aux_df

    @property
    def auxiliary_light(self):
        return self.__aux_df.loc["light"]

    @property
    def auxiliary_temperature(self):
        return self.__aux_df.loc["temperature"]

    @property
    def auxiliary_battery(self):
        return self.__aux_df.loc["battery"]

    @property
    def events(self):
        return self.__evt_df

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
        freq = 3200. / (1 << (15 - rate & 15))
        if freq <= 0:
            freq = 1
        return freq, 16 >> (rate >> 6)

    @staticmethod
    def DecodeScale(data):
        accel = data >> 13
        if accel > 0:
            accel = 1 << accel
        gyro = (data >> 10) & 0b111  # 0x7
        if gyro != 0:
            gyro = 1 << gyro

        light = data & 0b1111111111   # 0x3ff
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
        # initial formula: (value + 2 ** 15) % 2 ** 16 - 2 ** 15
        # 2**15 = 0x8000
        # 2**16 = 0x10000
        # 10 bits adapted: (value + 2 ** 9) % 2 ** 10 - 2 ** 9
        # 2**9 = 0x200
        # 2**10 = 0x400
        return (value + 0x200) % 0x400 - 0x200

    @staticmethod
    def scale_light(value):
        if value == 0:
            return numpy.nan
        log10LuxTimes10Power3 = ((value + 512.0) * 6 / 1024)
        return pow(10.0, log10LuxTimes10Power3)

    @staticmethod
    def scale_temp(value):
        if value == 0:
            return numpy.nan
        return (value * 150.0 - 20500) / 1000

    @staticmethod
    def scale_battery(value):
        if value == 0:
            return numpy.nan
        return (value + 512.0) * 6 / 1024

    @staticmethod
    def __create_vector(vec, scale, raw=False):
        if raw:
            norm = math.sqrt(vec[0] * vec[0]
                             + vec[1] * vec[1]
                             + vec[2] * vec[2]
                             )
            return vec + [norm]

        norm = math.sqrt(vec[0] * vec[0]
                         + vec[1] * vec[1]
                         + vec[2] * vec[2]
                         ) / scale
        return [v / scale for v in vec] + [norm]
