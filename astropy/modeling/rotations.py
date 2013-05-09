# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Implements spherical rotations, defined in WCS Paper II [1]_

RotateNative2Celestial and RotateCelestial2Native follow the convention
in WCS paper II to rotate to/from a native sphere and the celestial sphere.

The user interface uses angles in deg but internally radians are used.  This is
managed through properties. To bypass the use of Model's properties, an empty
param_names list is passed to `~astropy.modeling.core.Model.__init__` and the
properties are defined in the rotations classes.

References
----------
.. [1] Calabretta, M.R., Greisen, E.W., 2002, A&A, 395, 1077 (Paper II)

"""

from __future__ import division

import math
import numbers

import numpy as np

from .core import Model
from .parameters import Parameter, InputParameterError


__all__ = ['RotateCelestial2Native', 'RotateNative2Celestial',
           'MatrixRotation2D']


class EulerAngleRotation(Model):
    """
    Base class for Euler angle rotations.

    Defines a ZXZ rotation.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in deg
    """

    param_names = ['phi', 'theta', 'psi']
    phi = Parameter('phi')
    theta = Parameter('theta')
    psi = Parameter('psi')

    def __init__(self, phi, theta, psi):
        self._phi = np.deg2rad(phi)
        self._theta = np.deg2rad(theta)
        self._psi = np.deg2rad(psi)
        super(EulerAngleRotation, self).__init__(n_inputs=2, n_outputs=2)

    @property
    def phi(self):
        return np.rad2deg(self._phi)

    @phi.setter
    def phi(self, val):
        self._phi = np.deg2rad(val)

    @property
    def theta(self):
        return np.rad2deg(self._theta)

    @theta.setter
    def theta(self, val):
        self._theta = np.deg2rad(val)

    @property
    def psi(self):
        return np.rad2deg(self._psi)

    @psi.setter
    def psi(self, val):
        self._psi = np.deg2rad(val)


class RotateNative2Celestial(EulerAngleRotation):
    """
    Transformation from Native to Celestial Spherical Coordinates.

    Defines a ZXZ rotation.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in deg
    """

    def inverse(self):
        return RotateCelestial2Native(self.phi, self.theta, self.psi)

    def __call__(self, nphi, ntheta):
        nphi = np.deg2rad(nphi)
        ntheta = np.deg2rad(ntheta)
        calpha = np.rad2deg(self._phi + np.arctan2(-np.cos(ntheta) *
                                                   np.sin(nphi - self._psi),
                        np.sin(ntheta) * np.cos(self._theta) - np.cos(ntheta) *
                        np.sin(self._theta) * np.cos(nphi - self._psi)))
        cdelta = np.rad2deg(np.arcsin(np.sin(ntheta) * np.sin(self._theta) +
                                      np.cos(ntheta) * np.cos(self._theta) *
                                      np.cos(nphi - self._psi)))
        ind = calpha < 0
        calpha[ind] += 360
        return calpha, cdelta


class RotateCelestial2Native(EulerAngleRotation):
    """
    Transformation from Celestial to Native to Spherical Coordinates.

    Defines a ZXZ rotation.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in deg
    """

    def inverse(self):
        return RotateNative2Celestial(self.phi, self.theta, self.psi)

    def __call__(self, calpha, cdelta):
        calpha = np.deg2rad(calpha)
        cdelta = np.deg2rad(cdelta)
        nphi = np.rad2deg(self._psi + np.arctan2(-np.cos(cdelta) *
                                                 np.sin(calpha - self._phi),
                                                 np.sin(cdelta) * np.cos(self._theta) - np.cos(cdelta) *
                                                 np.sin(self._theta) * np.cos(calpha - self._phi)))
        ntheta = np.rad2deg(np.arcsin(np.sin(cdelta) * np.sin(self._theta) +
                                      np.cos(cdelta) * np.cos(self._theta) *
                                      np.cos(calpha - self._phi)))
        ind = nphi > 180
        nphi[ind] -= 360
        return nphi, ntheta


class MatrixRotation2D(Model):

    """
    Perform a clockwise 2D matrix rotation given either an angle or a
    rotation matrix.

    If both matrix and angle are given, angle will be ignored.

    Parameters
    ----------
    matrix : ndarray
        rotation matrix
    angle : float
        angle of rotation in deg
    """

    param_check = {'matrix', '_validate_matrix',
                   'angle', '_validate_angle'}

    angle = Parameter('angle')
    matrix = Parameter('matrix')

    def __init__(self, matrix=None, angle=None):
        if matrix is None and angle is None:
            raise InputParameterError("Expected at least one argument - "
                                           "a rotation matrix or an angle")
        if matrix is not None:
            self._validate_matrix(matrix)
            # TODO: Why +0.0?
            self._matrix = np.asarray(matrix) + 0.0
            super(MatrixRotation2D, self).__init__(n_inputs=1, n_outputs=1,
                                                   param_dim=1)
        else:
            self._validate_angle(angle)
            self._angle = np.deg2rad(angle)
            self.param_names = ['angle']
            self._matrix = self._compute_matrix(angle)

        n_inputs = self._matrix[0].shape[0]
        super(MatrixRotation2D, self).__init__(n_inputs=n_inputs,
                                               n_outputs=n_inputs,
                                               param_dim=1)

    @property
    def angle(self):
        return np.rad2deg(self._angle)

    @angle.setter
    def angle(self, val):
        self._angle = np.deg2rad(val)

    def _validate_matrix(self, matrix):
        assert matrix.ndim == 2, "Expected rotation matrix to be a 2D array"

    def _validate_angle(self, angle):
        assert isinstance(angle, numbers.Number), \
            "Expected angle to be a number"

    def _compute_matrix(self, angle):
        return np.array([[math.cos(angle), math.sin(angle)],
                         [-math.sin(angle), math.cos(angle)]],
                        dtype=np.float64)

    def inverse(self):
        nrot = np.linalg.inv(self._matrix[0])
        return MatrixRotation2D(matrix=nrot)

    def __call__(self, x, y):
        """
        Parameters
        ----------
        x, y : 1D array or list
              x and y coordinates
        """
        x = np.asarray(x)
        y = np.asarray(y)
        assert x.shape == y.shape
        shape = x.shape
        inarr = np.array([x.flatten(), y.flatten()], dtype=np.float64)
        assert inarr.shape[0] == 2 and inarr.ndim == 2, \
                    "Incompatible shape in MatrixRotation"
        result = np.dot(self._matrix[0], inarr)
        x, y = result[0], result[1]
        if x.shape != shape:
            x.shape = shape
            y.shape = shape
        return x, y
