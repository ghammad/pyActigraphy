"""Module to read .agd files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .agd import RawAGD

from .agd import read_raw_agd

__all__ = ["RawAGD", "read_raw_agd"]
