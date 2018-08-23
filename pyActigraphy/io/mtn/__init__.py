"""Module to read MTN files."""

# Author: Aubin Ardois <aubin.ardois@gmail.com>
#
# License: BSD (3-clause)

from .mtn import RawMTN

from .mtn import read_raw_mtn

__all__ = ["RawMTN", "read_raw_mtn"]
