"""Module to read files produced by the
[biobankanalysis](
    https://biobankaccanalysis.readthedocs.io/en/latest/index.html
) package.
"""

# Author: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

from .baa import RawBAA

from .baa import read_raw_baa

__all__ = ["RawBAA", "read_raw_baa"]
