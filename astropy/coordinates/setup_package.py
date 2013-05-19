# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys

if sys.version_info[0] < 3:
    from astropy.coordinates.angle_utilities import _AngleParser

    _AngleParser()

def get_package_data():
    return {'astropy.coordinates.tests.accuracy': ['*.csv']}
