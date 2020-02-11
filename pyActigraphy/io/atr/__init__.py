"""Module to read ActTrust files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .atr import RawATR

from .atr import read_raw_atr

__all__ = ["RawATR", "read_raw_atr"]
