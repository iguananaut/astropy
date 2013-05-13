# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains predefined polynomial models.
"""
from __future__ import division
import collections

import numpy as np

from .core import (ParametricModel, Model, SCompositeModel, _convert_input,
                   _convert_output)
from .functional_models import ShiftModel
from .parameters import Parameter, Parameters
from .utils import poly_map_domain, comb
from ..logger import log
from ..utils import lazyproperty


__all__ = ['Chebyshev1DModel', 'Chebyshev2DModel', 'Legendre2DModel',
           'Legendre1DModel', 'Poly1DModel', 'Poly2DModel', 'SIPModel',
           'OrthogonalPolynomialModel', 'PolynomialModel']


class PolynomialBase(ParametricModel):
    """
    Base class for all polynomial-like models with an arbitrary number of
    parameters in the form of coeffecients.

    In this case Parameter instances are returned through the class's
    ``__getattr__`` rather than through class descriptors.
    """

    # Default _param_names list; this will be filled in by the implementation's
    # __init__
    _param_names = []


    @lazyproperty
    def param_names(self):
        """Coefficient names generated based on the model's polynomial degree
        and number of dimensions.

        Subclasses should implement this to return parameter names in the
        desired format.

        On most `Model` classes this is a class attribute, but for polynomial
        models it is an instance attribute since each polynomial model instance
        can have different parameters depending on the degree of the polynomial
        and the number of dimensions, for example.
        """

        return self._param_names

    def __getattr__(self, attr):
        if self._param_names and attr in self._param_names:
            return Parameter(attr, model=self)
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        # TODO: Support a means of specifying default values for coefficients
        # Check for self._ndim first--if it hasn't been defined then the
        # instance hasn't been initialized yet and self.param_names probably
        # won't work.
        # This has to vaguely duplicate the functionality of
        # Parameter.__set__.
        # TODO: I wonder if there might be a way around that though...
        if attr[0] != '_' and self._param_names and attr in self._param_names:
            param = Parameter(attr, model=self)
            # This is a little hackish, but we can actually reuse the
            # Parameter.__set__ method here
            param.__set__(self, value)
            # Rebuild the internal Parameters list
            self._parameters = Parameters(self)
        else:
            super(PolynomialBase, self).__setattr__(attr, value)

    def _set_default_coeffs(self, param_dim):
        """
        Set default values for coefficients
        """

        for name in self.param_names:
            if param_dim == 1:
                value = 0.0
            else:
                value = np.zeros(param_dim, dtype=float)
            # This is just setting the default values, so set straight to the
            # internal attributes instead of going through the __setattr__
            # magic
            setattr(self, '_' + name, value)


class PolynomialModel(PolynomialBase):
    """
    Base class for polynomial models.

    Its main purpose is to determine how many coefficients are needed
    based on the polynomial order and dimension and to provide their
    default values, names and ordering.

    """

    def __init__(self, degree, n_inputs=1, n_outputs=1, param_dim=1,
                 **parameters):
        self._degree = degree
        self._order = self.get_num_coeff(n_inputs)
        self._param_names = self._generate_coeff_names(n_inputs)

        if parameters:
            p = parameters.get('c0', parameters.get('c0_0'))
            if isinstance(p, collections.Sequence):
                lenpars = len(p)
            else:
                lenpars = 1
            if param_dim != lenpars:
                if param_dim == 1:
                    log.info("Inferred {0} dimensions when creating a {1} model. "
                             "Resetting param_dim to {2}".format(
                                 lenpars, self.__class__.__name__, lenpars))
                    param_dim = lenpars
                else:
                    raise ValueError("Number of coefficient sets ({0}) does "
                                     "not match number of parameter sets "
                                     "({1}).".format(lenpars, param_dim))

            self._validate_pars(**parameters)

        self._set_default_coeffs(param_dim)

        super(PolynomialModel, self).__init__(n_inputs=n_inputs,
                                              n_outputs=n_outputs,
                                              param_dim=param_dim,
                                              **parameters)

    @property
    def degree(self):
        """TODO: Docstring for me"""

        return self._degree

    def get_num_coeff(self, ndim):
        """
        Return the number of coefficients in one parameter set
        """

        if self.degree < 1  or self.degree > 16:
            raise ValueError("Degree of polynomial must be 1< deg < 16")
        # deg+1 is used to account for the difference between iraf using
        # degree and numpy using exact degree
        if ndim != 1:
            nmixed = comb(self.degree, ndim)
        else:
            nmixed = 0
        numc = self.degree * ndim + nmixed + 1
        return numc

    def _invlex(self):
        c = []
        lencoeff = self.degree + 1
        for i in range(lencoeff):
            for j in range(lencoeff):
                if i + j <= self.degree:
                    c.append((j, i))
        return c[::-1]

    def _validate_pars(self, **pars):
        numcoeff = self._order
        assert(len(pars) == numcoeff)

    def _generate_coeff_names(self, ndim):
        names = []
        if ndim == 1:
            for n in range(self._order):
                names.append('c{0}'.format(n))
        else:
            for i in range(self.degree + 1):
                names.append('c{0}_{1}'.format(i, 0))
            for i in range(1, self.degree + 1):
                names.append('c{0}_{1}'.format(0, i))
            for i in range(1, self.degree):
                for j in range(1, self.degree):
                    if i + j < self.degree + 1:
                        names.append('c{0}_{1}'.format(i, j))
        return names


class OrthogonalPolynomialModel(PolynomialBase):
    """

    This is a base class for the 2D Chebyshev and Legendre models.

    The polynomials implemented here require a maximum degree in x and y.

    Parameters
    ----------

    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : list or None
        domain of the x independent variable
    y_domain : list or None
        domain of the y independent variable
    x_window : list or None
        range of the x independent variable
    y_window : list or None
        range of the y independent variable
    param_dim : int
        number of parameter sets
    **parameters : dict
        {keyword: value} pairs, representing {parameter_name: value}
    """

    def __init__(self, x_degree, y_degree, x_domain=None, x_window=None,
                 y_domain=None, y_window=None, param_dim=1, **parameters):
        # TODO: Perhaps some of these other parameters should be properties?
        self.x_degree = x_degree
        self.y_degree = y_degree
        self._order = self.get_num_coeff()
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.x_window = x_window
        self.y_window = y_window
        self._param_names = self._generate_coeff_names()

        if parameters:
            p = parameters.get('c0_0')
            if isinstance(p, collections.Sequence):
                lenparameters = len(p)
            else:
                lenparameters = 1
            if param_dim != lenpars:
                if param_dim == 1:
                    log.info("Inferred {0} dimensions when creating a {1} "
                             "model. Resetting param_dim to {2}".format(
                                 lenpars, self.__class__.__name__, lenpars))
                    param_dim = lenpars
                else:
                    raise ValueError("Number of coefficient sets {0} does "
                                     "not match number of parameter sets "
                                     "({1})".format(lenpars, param_dim))

            self._validate_pars(**parameters)

        self._set_default_coeffs(param_dim)

        super(OrthogonalPolynomialModel, self).__init__(n_inputs=2,
                                                        n_outputs=1,
                                                        param_dim=param_dim,
                                                        **parameters)

    def get_num_coeff(self):
        """
        Determine how many coefficients are needed

        Returns
        -------
        numc : int
            number of coefficients
        """

        return (self.x_degree + 1) * (self.y_degree + 1)

    def _validate_pars(self, **pars):
        numcoeff = self.get_num_coeff()
        assert(len(pars) == numcoeff)

    def _invlex(self):
        # TODO: This is a very slow way to do this; fix it and related methods
        # like _alpha
        c = []
        xvar = np.arange(self.x_degree + 1)
        yvar = np.arange(self.y_degree + 1)
        for j in yvar:
            for i in xvar:
                c.append((i, j))
        return np.array(c[::-1])

    def invlex_coeff(self):
        coeff = []
        xvar = np.arange(self.x_degree + 1)
        yvar = np.arange(self.y_degree + 1)
        for j in yvar:
            for i in xvar:
                name = 'c{0}_{1}'.format(i, j)
                coeff.append(getattr(self, name))
        return np.array(coeff[::-1])

    def _alpha(self):
        invlexdeg = self._invlex()
        invlexdeg[:, 1] = invlexdeg[:, 1] + self.x_degree + 1
        nx = self.x_degree + 1
        ny = self.y_degree + 1
        alpha = np.zeros((ny * nx + 3, ny + nx))
        for n in range(len(invlexdeg)):
            alpha[n][invlexdeg[n]] = [1, 1]
            alpha[-2, 0] = 1
            alpha[-3, nx] = 1
        return alpha

    def imhorner(self, x, y, coeff):
        _coeff = list(coeff[:])
        _coeff.extend([0, 0, 0])
        alpha = self._alpha()
        r0 = _coeff[0]
        nalpha = len(alpha)

        karr = np.diff(alpha, axis=0)
        kfunc = self._fcache(x, y)
        x_terms = self.x_degree + 1
        y_terms = self.y_degree + 1
        nterms = x_terms + y_terms
        for n in range(1, nterms + 1 + 3):
            setattr(self, 'r' + str(n), 0.)

        for n in range(1, nalpha):
            k = karr[n - 1].nonzero()[0].max() + 1
            rsum = 0
            for i in range(1, k + 1):
                rsum = rsum + getattr(self, 'r' + str(i))
            val = kfunc[k - 1] * (r0 + rsum)
            setattr(self, 'r' + str(k), val)
            r0 = _coeff[n]
            for i in range(1, k):
                setattr(self, 'r' + str(i), 0.)
        result = r0
        for i in range(1, nterms + 1 + 3):
            result = result + getattr(self, 'r' + str(i))
        return result

    def _generate_coeff_names(self):
        names = []
        for j in range(self.y_degree + 1):
            for i in range(self.x_degree + 1):
                names.append('c{0}_{1}'.format(i, j))
        return names

    def _fcache(self, x, y):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses should implement this")

    def __call__(self, x, y):
        """
        Transforms data using this model.

        Parameters
        --------------
        x : scalar, list or array
        y : scalar, lis or array

        """
        x, _ = _convert_input(x, self.param_dim)
        y, fmt = _convert_input(y, self.param_dim)
        assert x.shape == y.shape, \
            "Expected input arrays to have the same shape"
        if self.x_domain is not None:
            x = poly_map_domain(x, self.x_domain, self.x_window)
        if self.y_domain is not None:
            y = poly_map_domain(y, self.y_domain, self.y_window)
        invcoeff = self.invlex_coeff()

        result = self.imhorner(x, y, invcoeff)
        return _convert_output(result, fmt)


class Chebyshev1DModel(PolynomialModel):
    """

    1D Chebyshev polynomial of the 1st kind.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : list or None
    window : list or None
        If None, it is set to [-1,1]
        Fitters will remap the domain to this window
    param_dim : int
        number of parameter sets
    **parameters : dict
        keyword : value pairs, representing parameter_name: value

    """

    def __init__(self, degree, domain=None, window=[-1, 1], param_dim=1,
                 **parameters):
        self.domain = domain
        self.window = window
        super(Chebyshev1DModel, self).__init__(degree, n_inputs=1, n_outputs=1,
                                               param_dim=param_dim,
                                               **parameters)

    def clenshaw(self, x, coeff):
        """
        Evaluates the polynomial using Clenshaw's algorithm.
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = np.asarray(x)
        if len(coeff) == 1:
            c0 = coeff[0]
            c1 = 0
        elif len(coeff) == 2:
            c0 = coeff[0]
            c1 = coeff[1]
        else:
            x2 = 2 * x
            c0 = coeff[-2]
            c1 = coeff[-1]
            for i in range(3, len(coeff) + 1):
                tmp = c0
                c0 = coeff[-i] - c1
                c1 = tmp + c1 * x2
        return c0 + c1 * x

    def deriv(self, pars=None, x=None, y=None):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        pars : throw away parameter
            parameter list returned by non-linear fitters
        x : ndarray
            input
        y : throw away parameter
            Present here so that the non-linear fitting algorithms can work

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
        x = np.array(x, dtype=np.float, copy=False, ndmin=1)
        v = np.empty((self.degree + 1,) + x.shape, dtype=x.dtype)
        v[0] = x * 0 + 1
        x2 = 2 * x
        v[1] = x
        for i in range(2, self.degree + 1):
            v[i] = v[i - 1] * x2 - v[i - 2]
        return np.rollaxis(v, 0, v.ndim)

    def __call__(self, x):
        """
        Transforms data using this model.

        Parameters
        --------------
        x : scalar, list or array
            input

        """
        if self.domain is not None:
            x = poly_map_domain(x, self.domain, self.window)
        x, fmt = _convert_input(x, self.param_dim)
        result = self.clenshaw(x, self.param_sets)
        return _convert_output(result, fmt)


class Legendre1DModel(PolynomialModel):

    """

    1D Legendre polynomial.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : list or None
    window : list or None
        If None, it is set to [-1,1]
        Fitters will remap the domain to this window
    param_dim : int
        number of parameter sets
    **parameters : dict
        keyword: value pairs, representing parameter_name: value

    """
    def __init__(self, degree, domain=None, window=[-1, 1], param_dim=1,
                 **parameters):
        self.domain = domain
        self.window = window
        super(Legendre1DModel, self).__init__(degree, n_inputs=1, n_outputs=1,
                                              param_dim=param_dim,
                                              **parameters)

    def clenshaw(self, x, coeff):
        if isinstance(x, tuple) or isinstance(x, list):
            x = np.asarray(x)
        if len(coeff) == 1:
            c0 = coeff[0]
            c1 = 0
        elif len(coeff) == 2:
            c0 = coeff[0]
            c1 = coeff[1]
        else:
            nd = len(coeff)
            c0 = coeff[-2]
            c1 = coeff[-1]
            for i in range(3, len(coeff) + 1):
                tmp = c0
                nd = nd - 1
                c0 = coeff[-i] - (c1 * (nd - 1)) / nd
                c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
        return c0 + c1 * x

    def deriv(self, pars=None, x=None, y=None):
        """
        Parameters
        ----------
        pars : throw away parameter
            parameter list returned by non-linear fitters
        x : ndarray
            input
        y : throw away parameter
            Present here so that the non-linear fitting algorithms can work

        Returns
        -------
        result : ndarray
            The Vandermonde matrix

        """
        x = np.array(x, dtype=np.float, copy=False, ndmin=1)
        v = np.empty((self.degree + 1,) + x.shape, dtype=x.dtype)
        v[0] = x * 0 + 1
        v[1] = x
        for i in range(2, self.degree + 1):
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
        return np.rollaxis(v, 0, v.ndim)

    def __call__(self, x):
        """
        Transforms data using this model.

        Parameters
        --------------
        x : scalar, list or array
            input

        """
        if self.domain is not None:
            x = poly_map_domain(x, self.domain, self.window)
        x, fmt = _convert_input(x, self.param_dim)
        result = self.clenshaw(x, self.param_sets)
        return _convert_output(result, fmt)


class Poly1DModel(PolynomialModel):

    """

    1D Polynomial model.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : list or None
    window : list or None
        If None, it is set to [-1,1]
        Fitters will remap the domain to this window
    param_dim : int
        number of parameter sets
    **parameters : dict
        keyword: value pairs, representing parameter_name: value
    """

    def __init__(self, degree, domain=[-1, 1], window=[-1, 1], param_dim=1,
                 **parameters):
        self.domain = domain
        self.window = window
        super(Poly1DModel, self).__init__(degree, n_inputs=1, n_outputs=1,
                                          param_dim=param_dim, **parameters)

    def deriv(self, pars=None, x=None, y=None):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        pars : throw away parameter
            parameter list returned by non-linear fitters
        x : ndarray
            input
        y : throw away parameter
            Present here so that the non-linear fitting algorithms can work

        Returns
        -------
        result : ndarray
            The Vandermonde matrix

        """
        x = np.array(x, dtype=np.float, copy=False, ndmin=1)
        v = np.empty((self.degree + 1,) + x.shape, dtype=np.float)
        v[0] = x * 0 + 1
        v[1] = x
        for i in range(2, self.degree + 1):
            v[i] = v[i - 1] * x
        return np.rollaxis(v, 0, v.ndim)

    def horner(self, x, coef):
        c0 = coef[-1] + x * 0
        for i in range(2, len(coef) + 1):
            c0 = coef[-i] + c0 * x
        return c0

    def __call__(self, x):
        """
        Transforms data using this model.

        Parameters
        --------------
        x : scalar, list or array
            input

        """
        x, fmt = _convert_input(x, self.param_dim)
        result = self.horner(x, self.param_sets)
        return _convert_output(result, fmt)


class Poly2DModel(PolynomialModel):

    """
    2D Polynomial  model.

    Represents a general polynomial of degree n:

    .. math::
        P(x,y) = c_{0_0} + c_{1_0}*x + ...+ c_{n_0}*x^n + c_{0_1}*y + ...+ c_{0_n}*y^n
        + c_{1_1}*x*y + c_{1_2}*x*y^2 + ... + c_{1_(n-1)}*x*y^{n-1}+ ... + c_{(n-1)_1}*x^{n-1}*y

    Parameters
    ----------
    degree : int
        highest power of the polynomial,
        the number of terms is degree+1
    x_domain : list or None
        domain of the x independent variable
    y_domain : list or None
        domain of the y independent variable
    x_window : list or None
        range of the x independent variable
    y_window : list or None
        range of the y independent variable
    param_dim : int
        number of parameter sets
    **parameters : dict
        keyword: value pairs, representing parameter_name: value

    """

    def __init__(self, degree, x_domain=[-1, 1], y_domain=[-1, 1],
                 x_window=[-1, 1], y_window=[-1, 1],
                 param_dim=1, **parameters):
        super(Poly2DModel, self).__init__(degree, n_inputs=2, n_outputs=1,
                                          param_dim=param_dim, **parameters)
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.x_window = x_window
        self.y_window = y_window

    def mhorner(self, x, y, coeff):
        """
        Multivariate Horner's scheme

        Parameters
        --------------
        x, y : array
        coeff : array of coefficients in inverse lexical order
        """
        alpha = np.array(self._invlex())
        r0 = coeff[0]
        r1 = r0 * 0.0
        r2 = r0 * 0.0
        karr = np.diff(alpha, axis=0)
        for n in range(len(karr)):
            if karr[n, 1] != 0:
                r2 = y * (r0 + r1 + r2)
                r1 = coeff[0] * 0.
            else:
                r1 = x * (r0 + r1)
            r0 = coeff[n + 1]
        return r0 + r1 + r2

    def deriv(self, pars=None, x=None, y=None, z=None):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        pars : throw away parameter
            parameter list returned by non-linear fitters
        x : ndarray
            input
        y : ndarray
            input
        z : throw away parameter
            Present here so that the non-linear fitting algorithms can work

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
        if x.ndim == 2:
            x = x.flatten()
        if y.ndim == 2:
            y = y.flatten()
        if x.size != y.size:
            raise ValueError('Expected x and y to be of equal size')

        designx = x[:, None] ** np.arange(self.degree + 1)
        designy = y[:, None] ** np.arange(1, self.degree + 1)

        designmixed = []
        for i in range(1, self.degree):
            for j in range(1, self.degree):
                if i + j <= self.degree:
                    designmixed.append((x ** i) * (y ** j))
        designmixed = np.array(designmixed).T
        if designmixed.any():
            v = np.hstack([designx, designy, designmixed])
        else:
            v = np.hstack([designx, designy])
        return v

    def invlex_coeff(self):
        coeff = []
        lencoeff = range(self.degree + 1)
        for i in lencoeff:
            for j in lencoeff:
                if i + j <= self.degree:
                    name = 'c{0}_{1}'.format(j, i)
                    coeff.append(getattr(self, name))
        return np.array(coeff[::-1])

    def __call__(self, x, y):
        """
        Transforms data using this model.

        Parameters
        --------------
        x : scalar, list or array
            input
        y : scalar, list or array
            input

        """
        invcoeff = self.invlex_coeff()
        x, _ = _convert_input(x, self.param_dim)
        y, fmt = _convert_input(y, self.param_dim)
        assert x.shape == y.shape, \
            "Expected input arrays to have the same shape"

        result = self.mhorner(x, y, invcoeff)
        return _convert_output(result, fmt)


class Chebyshev2DModel(OrthogonalPolynomialModel):
    """
    2D Chebyshev polynomial of the 1st kind.

    It is defined as

    .. math:: P_{n_m}(x,y) = \sum C_{n_m}  T_n(x) T_m(y)

    Parameters
    ----------

    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : list or None
        domain of the x independent variable
    y_domain : list or None
        domain of the y independent variable
    x_window : list or None
        range of the x independent variable
    y_window : list or None
        range of the y independent variable
    param_dim : int
        number of parameter sets
    **parameters : dict
        keyword: value pairs, representing parameter_name: value

    """

    def __init__(self, x_degree, y_degree, x_domain=None, x_window=[-1, 1],
                 y_domain=None, y_window=[-1,1], param_dim=1, **parameters):
        super(Chebyshev2DModel, self).__init__(x_degree, y_degree,
                                               x_domain=x_domain,
                                               y_domain=y_domain,
                                               x_window=x_window,
                                               y_window=y_window,
                                               param_dim=param_dim,
                                               **parameters)

    def _fcache(self, x, y):
        """
        Calculate the individual Chebyshev functions once
        and store them in a dictionary to be reused.
        """

        x_terms = self.x_degree + 1
        y_terms = self.y_degree + 1
        kfunc = {}
        kfunc[0] = np.ones(x.shape)
        kfunc[1] = x.copy()
        kfunc[x_terms] = np.ones(y.shape)
        kfunc[x_terms + 1] = y.copy()
        for n in range(2, x_terms):
            kfunc[n] = 2 * x * kfunc[n - 1] - kfunc[n - 2]
        for n in range(x_terms + 2, x_terms + y_terms):
            kfunc[n] = 2 * y * kfunc[n - 1] - kfunc[n - 2]
        return kfunc

    def deriv(self, pars=None, x=None, y=None, z=None):
        """
        Derivatives with respect to the coefficients.
        This is an array with Chebyshev polynomials:

        Tx0Ty0  Tx1Ty0...TxnTy0...TxnTym

        Parameters
        ----------
        pars : throw away parameter
            parameter list returned by non-linear fitters
        x : ndarray
            input
        y : ndarray
            input
        z : throw away parameter
            Present here so that the non-linear fitting algorithms can work

        Returns
        -------
        result : ndarray
            The Vandermonde matrix

        """

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        x = x.flatten()
        y = y.flatten()
        x_deriv = self._chebderiv1d(x, self.x_degree + 1).T
        y_deriv = self._chebderiv1d(y, self.y_degree + 1).T

        ij = []
        for i in range(self.y_degree + 1):
            for j in range(self.x_degree + 1):
                ij.append(x_deriv[j] * y_deriv[i])

        v = np.array(ij)
        return v.T

    def _chebderiv1d(self, x, deg):
        """
        Derivative of 1D Chebyshev series
        """

        x = np.array(x, dtype=np.float, copy=False, ndmin=1)
        d = np.empty((deg + 1, len(x)), dtype=x.dtype)
        d[0] = x * 0 + 1
        if deg > 0:
            x2 = 2 * x
            d[1] = x
            for i in range(2, deg + 1):
                d[i] = d[i - 1] * x2 - d[i - 2]
        return np.rollaxis(d, 0, d.ndim)



class Legendre2DModel(OrthogonalPolynomialModel):
    """
    Legendre 2D polynomial.

    Defined as:

    .. math:: P_{nm}(x,y) = C_{n_m}  L_n(x ) L_m(y)


    Parameters
    ----------

    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : list or None
        domain of the x independent variable
    y_domain : list or None
        domain of the y independent variable
    x_window : list or None
        range of the x independent variable
    y_window : list or None
        range of the y independent variable
    param_dim : int
        number of parameter sets
    **parameters : dict
        keyword: value pairs, representing parameter_name: value
    """

    def __init__(self, x_degree, y_degree, x_domain=None, x_window=[-1, 1],
                 y_domain=None, y_window=[-1, 1], param_dim=1, **parameters):
        super(Legendre2DModel, self).__init__(x_degree, y_degree,
                                              x_domain=x_domain,
                                              y_domain=y_domain,
                                              x_window=x_window,
                                              y_window=y_window,
                                              param_dim=param_dim,
                                              **parameters)

    def _fcache(self, x, y):
        """
        Calculate the individual Legendre functions once
        and store them in a dictionary to be reused.
        """

        x_terms = self.x_degree + 1
        y_terms = self.y_degree + 1
        kfunc = {}
        kfunc[0] = np.ones(x.shape)
        kfunc[1] = x.copy()
        kfunc[x_terms] = np.ones(y.shape)
        kfunc[x_terms + 1] = y.copy()
        for n in range(2, x_terms):
            kfunc[n] = ((2 * (n - 1) + 1) * x * kfunc[n - 1] - (n - 1) * kfunc[n - 2]) / n
        for n in range(2, y_terms):
            kfunc[n + x_terms] = ((2 * (n - 1) + 1) * y * kfunc[n + x_terms - 1] -
                                  (n - 1) * kfunc[n + x_terms - 2]) / (n)
        return kfunc

    def deriv(self, pars=None, x=None, y=None, z=None):
        """
        Derivatives with repect to the coefficients.
        This is an array with Legendre polynomials:

        Lx0Ly0  Lx1Ly0...LxnLy0...LxnLym

        Parameters
        ----------
        pars : throw away parameter
            parameter list returned by non-linear fitters
        x : ndarray
            input
        y : ndarray
            input
        z : throw away parameter
            Present here so that the non-linear fitting algorithms can work

        Returns
        -------
        result : ndarray
            The Vandermonde matrix

        """
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        x = x.flatten()
        y = y.flatten()
        x_deriv = self._legendderiv1d(x, self.x_degree + 1).T
        y_deriv = self._legendderiv1d(y, self.y_degree + 1).T

        ij = []
        for i in range(self.y_degree + 1):
            for j in range(self.x_degree + 1):
                ij.append(x_deriv[j] * y_deriv[i])

        v = np.array(ij)
        return v.T

    def _legendderiv1d(self, x, deg):
        """
        Derivative of 1D Legendre polynomial
        """

        x = np.array(x, dtype=np.float, copy=False, ndmin=1)
        d = np.empty((deg + 1,) + x.shape, dtype=x.dtype)
        d[0] = x * 0 + 1
        if deg > 0:
            d[1] = x
            for i in range(2, deg + 1):
                d[i] = (d[i - 1] * x * (2 * i - 1) - d[i - 2] * (i - 1)) / i
        return np.rollaxis(d, 0, d.ndim)


class _SIP1D(PolynomialBase):
    """
    This implements the Simple Imaging Polynomial Model (SIP) in 1D.

    It's unlikely it will be used in 1D so this class is private
    and SIPModel should be used instead.
    """

    def __init__(self, order, coeff_prefix, param_dim=1, **parameters):
        self.order = order
        self.coeff_prefix = coeff_prefix
        self._param_names = self._generate_coeff_names(coeff_prefix)
        coeffname = '{0}0_2'.format(coeff_prefix)

        if parameters:
            p = parameters.get(coeffname, None)
            if isinstance(p, collections.Sequence):
                lenpars = len(p)
            else:
                lenpars = 1
            # TODO: This pattern is repeated in so many models and seems like a
            # check that could be moved into a method
            if param_dim != lenpars:
                if param_dim == 1:
                    log.info("Inferred {0} dimensions when creating a {1} "
                             "model. Resetting param_dim to {2}".format(
                                 lenpars, self.__class__.__name__, lenpars))
                    param_dim = lenpars
                else:
                    raise ValueError("Number of coefficient sets ({0}) does "
                                     "not match number of parameter sets "
                                     "({1}).".format(lenpars, param_dim))

        if parameters:
            self._validate_pars(ndim=2, **parameters)

        self._set_default_coeffs(param_dim)

        super(_SIP1D, self).__init__(n_inputs=2, n_outputs=1,
                                     param_dim=param_dim, **parameters)

    def __repr__(self):
        fmt = """
        Model: {0}
        order: {1}
        coeff_prefix: {2}
        param_dim: {3}
        pars: {4}
        """.format(
              self.__class__.__name__,
              self.order,
              self.coeff_prefix,
              self._param_dim,
              indent('\n'.join('{0}: {1}'.format(n, getattr(self, n))
                     for n in self.param_names),
                     width=19))

        return dedent(fmt[1:])

    def __str__(self):
        fmt = """
        Model: {0}
        order: {1}
        coeff_prefix: {2}
        Parameter sets: {3}
        Parameters:
                   {3}
        """.format(
              self.__class__.__name__,
              self.order,
              self.coeff_prefix,
              self._param_dim,
              indent('\n'.join('{0}: {1}'.format(n, getattr(self, n))
                     for n in self.param_names),
                     width=19))

        return dedent(fmt[1:])

    def get_num_coeff(self, ndim):
        """
        Return the number of coefficients in one parset
        """
        if self.order < 2 or self.order > 9:
            raise ValueError("Degree of polynomial must be 2< deg < 9")
        nmixed = comb(self.order - 1, ndim)
        numc = self.order * ndim + nmixed + 1
        return numc

    def _generate_coeff_names(self, coeff_prefix):
        names = []
        for i in range(2, self.order + 1):
            names.append('{0}{1}_{2}'.format(coeff_prefix, i, 0))
        for i in range(2, self.order + 1):
            names.append('{0}{1}_{2}'.format(coeff_prefix, 0, i))
        for i in range(1, self.order):
            for j in range(1, self.order):
                if i + j < self.order + 1:
                    names.append('{0}{1}_{2}'.format(coeff_prefix, i, j))
        return names

    def _validate_pars(self, ndim, **pars):
        numcoeff = self.get_num_coeff(ndim)
        assert(len(pars) == numcoeff)

    def _coef_matrix(self, coeff_prefix):
        mat = np.zeros((self.order + 1, self.order + 1))
        for i in range(2, self.order + 1):
            mat[i, 0] = getattr(self, '{0}{1}_{2}'.format(coeff_prefix, i, 0))[0]
        for i in range(2, self.order + 1):
            mat[0, i] = getattr(self, '{0}{1}_{2}'.format(coeff_prefix, 0, i))[0]
        for i in range(1, self.order):
            for j in range(1, self.order):
                if i + j < self.order + 1:
                    mat[i, j] = getattr(self, '{0}{1}_{2}'.format(coeff_prefix, i, j))[0]
        return mat

    def _eval_sip(self, x, y, coef):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.coeff_prefix == 'a':
            result = np.zeros(x.shape)
        else:
            result = np.zeros(y.shape)

        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                if i + j > 1 and i + j < self.order + 1:
                    result = result + coef[i, j] * x ** i * y ** j
        return result

    def __call__(self, x, y):
        mcoef = self._coef_matrix(self.coeff_prefix)
        return self._eval_sip(x, y, mcoef)


class SIPModel(SCompositeModel):
    """
    Simple Imaging Polynomial (SIP) model.

    The SIP convention is used to represent distortions in FITS image headers.
    See [1]_ for a description of the SIP convention.

    Parameters
    ----------
    crpix : list or ndarray of length(2)
        CRPIX values
    a_order : int
        SIP polynomial order for first axis
    a_coeff : dict
        SIP coefficients for first axis
    coeff_prefix : string: 'a', 'b', 'A' or 'B'
        SIP coefficient preffix
    b_order : int
        SIP order for second axis
    b_coeff : dict
        SIP coefficients for the second axis
    a_inv_order : int
        order for the inverse transformation (AP coefficients)
    a_inv_coeff : dict
        coefficients for the inverse transform
    b_inv_order : int
        order for the inverse transformation (BP coefficients)
    b_inv_coeff : dict
        coefficients for the inverse transform
    param_dim : int
        number of parameter sets

    References
    ----------
    .. [1] `David Shupe, et al, ADASS, ASP Conference Series, Vol. 347, 2005 <http://adsabs.harvard.edu/abs/2005ASPC..347..491S>`_

    """

    def __init__(self, crpix, a_order, a_coeff, b_order, b_coeff,
                 a_inv_order=None, a_inv_coeff=None, b_inv_order=None,
                 b_inv_coeff=None, param_dim=1):
        self.shifta = ShiftModel(-crpix[0])
        self.shiftb = ShiftModel(-crpix[1])
        self.sip1da = _SIP1D(a_order, coeff_prefix='A',
                             param_dim=param_dim, **a_coeff)
        self.sip1db = _SIP1D(b_order, coeff_prefix='B',
                             param_dim=param_dim, **b_coeff)
        if a_inv_order is not None and a_inv_coeff is not None and \
                b_inv_order is not None and b_inv_coeff is not None:
            self.inversea = _SIP1D(a_inv_order, coeff_prefix='A',
                                   **a_inv_coeff)
            self.inverseb = _SIP1D(b_inv_order, coeff_prefix='BP',
                                   **b_inv_coeff)
            self.inverse = True
        else:
            self.inverse = None
        super(SIPModel, self).__init__([self.shifta, self.shiftb, self.sip1da,
                                        self.sip1db],
                                       inmap=[['x'], ['y'], ['x', 'y'],
                                              ['x', 'y']],
                                       outmap=[['x'], ['y'], ['x1'], ['y1']],
                                       n_inputs=2, n_outputs=2)

    def __repr__(self):
        models = [self.shifta, self.shiftb, self.sip1da, self.sip1db]
        fmt = """
            Model:  {0}
            """.format(self.__class__.__name__, self.sip1d.coeff_prefix)
        fmt1 = " %s  " * len(models) % tuple([repr(model) for model in models])
        fmt = fmt + fmt1
        return fmt

    def __str__(self):
        models = [self.shifta, self.shiftb, self.sip1da, sip1db]
        fmt = """
            Model:  {0}
            """.format(self.__class__.__name__, self.sip1d.coeff_prefix)
        fmt1 = " %s  " * len(models) % tuple([str(model) for model in models])
        fmt = fmt + fmt1
        return fmt

    def __call__(self, x, y):
        """
        Transforms data using this model.

        Parameters
        ----------
        x : scalar, list ot array
            input
        y : scalar, list or array
            input

        """
        labeled_input = LabeledInput([x, y], ['x', 'y'])
        result = SCompositeModel.__call__(self, labeled_input)
        return result.x1, result.y1
