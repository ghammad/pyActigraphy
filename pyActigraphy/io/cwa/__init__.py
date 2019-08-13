"""Module to read AWD files."""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .cwa_class import CWA

from .cwa_class import read_raw_cwa

__all__ = ["CWA", "read_raw_cwa"]
