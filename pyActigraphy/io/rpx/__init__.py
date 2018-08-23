"""Module to read Respironics files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .rpx import RawRPX

from .rpx import read_raw_rpx

__all__ = ["RawRPX", "read_raw_rpx"]
