"""Module to read MESA files."""

# Author: Aubin Ardois <aubin.ardois@gmail.com>
#
# License: BSD (3-clause)

from .mesa import RawMESA

from .mesa import read_raw_mesa

__all__ = ["RawMESA", "read_raw_mesa"]
