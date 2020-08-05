"""Module to read Tempatilumi files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .tal import RawTAL

from .tal import read_raw_tal

__all__ = ["RawTAL", "read_raw_tal"]
