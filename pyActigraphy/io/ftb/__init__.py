"""Module to read Fitbit JSON files."""

# Author: Rafael Morand <rafael.morand@unibe.com>
#
# License: BSD (3-clause)

from .ftb import RawFTB

from .ftb import read_raw_ftb

__all__ = ["RawFTB", "read_raw_ftb"]
