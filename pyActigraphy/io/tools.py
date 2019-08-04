import struct


"""
package collecting various tools for handeling imported
Actigraphy files
"""


class DataReader(object):
    """
    class allowing to read and interpret generic binary data
    """

    __slots__ = ["__data", "__endianess", "__offset"]
    __types__ = {"none": "x",
                 "int8": "b",
                 "uint8": "B",
                 "bool": "?",
                 "int16": "h",
                 "uint16": "H",
                 "int32": "i",
                 "uint32": "I",
                 "int64": "q",
                 "uint64": "Q",
                 "float32": "f",
                 "float64": "d"
                 }

    def __init__(self, data, endianess):
        """
        initialize the reader class.
        Each instance of calss coressponds to a block of
        binary data.

        Parameter
        ---------
        data: bytes
            data to interprete
        endianess: string
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")

        self.__data = data
        self.__offset = 0
        self.endianess = endianess

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, offset):
        if not isinstance(offset, int):
            raise TypeError("offset must be int")
        if offset < 0:
            offset = self.size() + offset
        if offset >= self.size() or offset < 0:
            raise IndexError("offset out of range")
        self.__offset = offset

    @property
    def endianess(self):
        return self.__endianess

    @endianess.setter
    def endianess(self, endianess):
        if not isinstance(endianess, str):
            raise TypeError("endianess must be str")
        if len(endianess) != 1 or endianess not in "@=<>!":
            raise ValueError("endianess must be one of struct character")
        self.__endianess = endianess

    def __rshift__(self, increment):
        """
        increases offset by increment
        """
        if not isinstance(increment, int):
            raise TypeError("increment must be int")
        if self.__offset + increment > len(self.__data):
            raise IndexError("offset out of range")
        self.__offset += increment

    def __lshift__(self, increment):
        """
        decrease offset by increment
        """
        if not isinstance(increment, int):
            raise TypeError("increment must be int")
        if self.__offset < increment:
            raise IndexError("offset out of range")
        self.__offset -= increment

    def __iconcat__(self, data):
        """
        appends new byte array to data
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        self.__data += data

    def size(self):
        """
        returns size of data
        """
        return len(self.__data)

    def unpack(self, type, size=1, offset=None):
        """
        unpack and returns corresponding data, increasing
        the offset

        Parameters
        ----------
        type: str
            type of data to unpack
            if type is char, standard struct types used
            if type is string, type is interpreted
            if type is empty, raw data is returned
        size: int
            number of data to interpret
            if equal to 1, returns one instance
            if >1 returns list of instances
        offset: int, optional
            if precised, data readed at given offset, without
            advancing internal one
        """
        if not isinstance(type, str):
            raise TypeError("type must be str")
        if not isinstance(size, int):
            raise TypeError("size must be int")
        if offset is not None and not isinstance(offset, int):
            raise TypeError("offset must be int")
        if size <= 0:
            raise ValueError("invalid size to read")
        if offset is None:
            off = self.__offset
        else:
            if offset < 0:
                off = self.size() + offset
        if off < 0 or off >= self.size():
            raise IndexError("offset out of range")

        if len(type) > 1:
            type = self.__types__[type]

        if type == "":
            res = self.__data[off:off+size]
        else:
            type = self.endianess + type*size
            size = struct.calcsize(type)
            res = struct.unpack_from(type,
                                     self.__data,
                                     offset=off)
            if len(res) == 1:
                res = res[0]
            else:
                res = list(res)

        if offset is None:
            self.__rshift__(size)

        return res
