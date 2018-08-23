"""Module to read AWD files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .awd import RawAWD

from .awd import read_raw_awd

__all__ = ["RawAWD", "read_raw_awd"]
