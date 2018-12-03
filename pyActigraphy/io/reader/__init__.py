"""Module to read multiple files at once."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .reader import RawReader

from .reader import read_raw

__all__ = ["RawReader", "read_raw"]
