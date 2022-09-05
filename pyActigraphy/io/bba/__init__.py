"""Module to read files produced by the
[biobankanalysis](
    https://biobankaccanalysis.readthedocs.io/en/latest/index.html
) package.
"""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .bba import RawBBA

from .bba import read_raw_bba

__all__ = ["RawBBA", "read_raw_bba"]
