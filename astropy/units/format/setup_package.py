"""setup_package.py module for generating the parser optimization files.

This does not implement any of the functions typically called from
setup_package modules, but simply by importing it at setup time the
appropriate files will be generated.
"""

import sys

if sys.version_info[0] < 3:
    from astropy.units.format.generic import Generic
    from astropy.units.format.cds import CDS

    Generic()
    CDS()
