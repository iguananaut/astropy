# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Statistic functions used in `~astropy.modeling.fitting.py`.
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np

__all__ = ['residuals', 'least_square']


def residuals(measured_values, approximated_values, weights=None):
    """
    Returns the residuals with optional weights.

    measured_values : `~numpy.ndarray`
        The measured data values being optimized against.

    approximated_values : `~numpy.ndarray`
        The approximate values predicted by the fitted model.

    weights : `~numpy.ndarray` (optional)
        Optional weights to give to each residual.
    """

    if weights is not None:
        return weights * (measured_values - approximated_values)
    else:
        return measured_values - approximated_values


def least_square(measured_values, approximated_values, weights=None):
    """
    Least square statistic with optional weights.

    measured_values : `~numpy.ndarray`
        The measured data values being optimized against.

    approximated_values : `~numpy.ndarray`
        The approximate values predicted by the fitted model.

    weights : `~numpy.ndarray` (optional)
        Optional weights to give to each residual.
    """

    return np.sum(residuals(measured_values, approximated_values,
                            weights=weights) ** 2)
