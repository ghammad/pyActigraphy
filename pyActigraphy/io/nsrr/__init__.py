"""Module to read generic NSRR files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .nsrr import RawNSRR

from .nsrr import read_raw_nsrr

__all__ = ["RawNSRR", "read_raw_nsrr"]
