"""Module to read Tempatilumi files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .tmp import RawTMP

from .tmp import read_raw_tmp

__all__ = ["RawTMP", "read_raw_tmp"]
