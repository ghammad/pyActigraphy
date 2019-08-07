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
            offset = self.size + offset
        if offset >= self.size or offset < 0:
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

    @property
    def size(self):
        return len(self.__data)

    def __rshift__(self, increment):
        """
        increases offset by increment
        """
        self.__offset += increment

    def __lshift__(self, increment):
        """
        decrease offset by increment
        """
        self.__offset -= increment

    def __iconcat__(self, data):
        """
        appends new byte array to data
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        self.__data += data

    def unpack_at(self, type, offset, size=1):
        """
        unpack data at offset, and returns corresponding 
        value.
        Do not advance internal data pointer
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
        offset: int
            place to read data from
        """
        if len(type) > 1:
            type = self.__types__[type]
        if type == "":
            dsize = size
            res = self.__data[offset:offset + dsize]
        else:
            type = self.endianess + type * size
            dsize = struct.calcsize(type)
            data = self.__data[offset:offset + dsize]
            res = struct.unpack(type, data)
        self.__rshift__(dsize)
        if size == 1:
            return res[0]
        else:
            return res

    def unpack(self, type, size=1):
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
        """
        if len(type) > 1:
            type = self.__types__[type]
        if type == "":
            dsize = size
            res = self.__data[self.__offset:self.__offset + dsize]
        else:
            type = self.endianess + type * size
            dsize = struct.calcsize(type)
            data = self.__data[self.__offset:self.__offset + dsize]
            res = struct.unpack(type, data)

        self.__rshift__(dsize)
        if size == 1:
            return res[0]
        else:
            return res

    def checksum(self, type):
        """
        Calculates the checksum of data block, based on passed type

        Parameters
        ----------
        type: str
            type of data to unpack
            if type is char, standard struct types used
            if type is string, type is interpreted

        Returns
        -------
        float, or int:
            sum of all unpacked data
        """
        if not isinstance(type, str):
            raise TypeError("type must be str")
        if len(type) > 1:
            type = self.__types__[type]
        dsize = struct.calcsize(type)
        if self.size % dsize != 0:
            raise ValueError("Unable to split data to words of size {}"
                             .format(dsize))
        type = self.endianess + type * (self.size // dsize)
        return sum(struct.unpack(type, self.__data))
