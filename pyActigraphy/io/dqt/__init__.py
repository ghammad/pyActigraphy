"""Module to read AWD files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .dqt import RawDQT

from .dqt import read_raw_dqt

__all__ = ["RawDQT", "read_raw_dqt"]
