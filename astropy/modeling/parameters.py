# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines two classes that deal with parameters.
It is unlikely users will need to work with these classes directly,
unless they define their own models.
"""

from __future__ import division

import numbers

import numpy as np

from ..utils import misc


__all__ = ['Parameters', 'Parameter']


def _tofloat(value):
    """
    Convert a parameter to float or float array

    """

    if misc.isiterable(value):
        try:
            value = np.array(value, dtype=np.float)
            shape = value.shape
        except (TypeError, ValueError):
            # catch arrays with strings or user errors like different
            # types of parameters in a parameter set
            raise InputParameterError(
                "Parameter of {0} could not be converted to "
                "float".format(type(value)))
    elif isinstance(value, bool):
        raise InputParameterError(
            "Expected parameter to be of numerical type, not boolean")
    elif isinstance(value, numbers.Number):
        value = float(value)
        shape = ()
    else:
        raise InputParameterError(
            "Don't know how to convert parameter of {0} to "
            "float".format(type(value)))
    return value, shape


# TODO: Is there really any need for this special exception over just, say, a
# ValueError?
class InputParameterError(Exception):
    """
    Called when there's a problem with input parameters.
    """


class Parameter(object):
    """
    Wraps individual parameters.

    This class represents a model's parameter (in a somewhat broad sense). To
    support multiple parameter sets, a parameter has a dimension (param_dim).
    Parameter objects behave like numbers.

    Parameters
    ----------
    name : string
        parameter name
    val :  number or an iterable of numbers
    mclass : object
        an instance of a Model class
    param_dim : int
        parameter dimension
    fixed: boolean
        if True the parameter is not varied during fitting
    tied: callable or False
        if callable is suplplied it provides a way to link to another parameter
    min: float
        the lower bound of a parameter
    max: float
        the upper bound of a parameter
    """

    def __init__(self, name, fixed=False, tied=False, minvalue=None,
                 maxvalue=None, model=None):
        super(Parameter, self).__init__()
        self._name = name
        self._attr = '_' + name
        self._fixed = fixed
        self._tied = tied
        self._min = min
        self._max = max

        self._model = model

        if model is not None:
            value = getattr(model, self._attr)
            _, self._shape = self._validate_value(model, value)

    def __get__(self, obj, objtype):
        return self.__class__(self._name, model=obj)

    def __set__(self, obj, value):
        value, shape = self._validate_value(obj, value)
        # Compare the shape against the previous value's shape, if it exists
        if hasattr(obj, self._attr):
            oldvalue = getattr(obj, self._attr)
            oldvalue, oldshape = self._validate_value(obj, oldvalue)
            if shape != oldshape:
                raise InputParameterError(
                    "Input value for parameter {0!r} does not have the "
                    "required shape {1}".format(self.name, oldshape))

        if (hasattr(obj, '_parameters') and
                isinstance(obj._parameters, Parameters)):
            # Model instance has a Parameters list (in general this means it's
            # a ParametricModel)
            setattr(obj, self._attr, value)
            # Rebuild the Parameters list
            # TODO: Is this really necessary? Could we get away with just
            # updating this parameter's value in the Parameters list?
            obj._parameters = Parameters(obj)
        else:
            setattr(obj, self._attr, value)

    def __len__(self):
        if self._model is None:
            raise TypeError('Parameter definitions do not have a length.')
        return self._model.param_dim

    def __getitem__(self, key):
        value = getattr(self._model, self._attr)
        if self._model.param_dim == 1:
            # Wrap the value in a list so that getitem can work for sensible
            # indcies like [0] and [-1]
            value = [value]
        return value[key]

    def __setitem__(self, key, value):
        # Get the existing value and check whether it even makes sense to
        # apply this index
        oldvalue = getattr(self._model, self._attr)
        param_dim = self._model.param_dim

        if param_dim == 1:
            # Convert the single-dimension value to a list to allow some slices
            # that would be compatible with a length-1 array like [:] and [0:]
            oldvalue = [oldvalue]

        if isinstance(key, slice):
            if not oldvalue[key]:
                raise InputParameterError(
                    "Slice assignment outside the parameter dimensions for "
                    "{0!r}".format(self.name))
            for idx, val in zip(range(*key.indices(len(self))), value):
                self.__setitem__(idx, val)
        else:
            try:
                oldvalue[key] = value
                if param_dim == 1:
                    setattr(self._model, self._attr, value)
            except IndexError:
                raise InputParameterError(
                    "Input dimension {0} invalid for {1!r} parameter with "
                    "dimension {2}".format(key, self.name, param_dim))

    def __repr__(self):
        if self._model is None:
            return 'Parameter({0!r})'.format(self._name)
        else:
            return 'Parameter({0!r}, value={1!r})'.format(
                self._name, self.value)

    @property
    def name(self):
        """
        Parameter name
        """

        return self._name

    @property
    def value(self):
        if self._model is not None:
            return getattr(self._model, self._attr)
        raise AttributeError('Parameter definition does not have a value')

    @property
    def shape(self):
        return self._shape

    @property
    def fixed(self):
        """
        Boolean indicating if the parameter is kept fixed during fitting.
        """

        return self._fixed

    @fixed.setter
    def fixed(self, value):
        if self._model is not None:
            assert isinstance(value, bool), "Fixed can be True or False"
            self._fixed = value
            self._model.constraints._fixed.update({self.name: value})
            self._model.constraints._update()
        else:
            raise AttributeError("can't set attribute 'fixed' on Parameter "
                                 "definition")

    @property
    def tied(self):
        """
        Indicates that this parameter is linked to another one.
        A callable which provides the relationship of the two parameters.
        """

        return self._tied

    @tied.setter
    def tied(self, value):
        if self._model is not None:
            assert callable(value) or value in (False, None), \
                    "Tied must be a callable"
            self._tied = value
            self._model.constraints._tied.update({self.name:value})
            self._model.constraints._update()
        else:
            raise AttributeError("can't set attribute 'tied' on Parameter "
                                 "definition")

    @property
    def min(self):
        """
        A value used as a lower bound when fitting a parameter.
        """

        return self._min

    @min.setter
    def min(self, value):
        if self._model is not None:
            assert isinstance(value, numbers.Number), \
                    "Min value must be a number"
            self._min = float(value)
            self._model.constraints.set_range(
                    {self.name: (value, self.max)})
        else:
            raise AttributeError("can't set attribute 'min' on Parameter "
                                 "definition")

    @property
    def max(self):
        """
        A value used as an upper bound when fitting a parameter.
        """
        return self._max

    @max.setter
    def max(self, value):
        if self._model is not None:
            assert isinstance(value, numbers.Number), \
                    "Max value must be a number"
            self._max = float(value)
            self._model.constraints.set_range(
                    {self.name: (self.min, value)})
        else:
            raise AttributeError("can't set attribute 'max' on Parameter "
                                 "definition")

    def _validate_value(self, model, value):
        if model is None:
            return

        param_dim = model.param_dim
        if param_dim == 1:
            # Just validate the value with _tofloat
            return _tofloat(value)
        else:
            # If there are more parameter dimensions the value should
            # be a sequence with at least one item
            try:
                if len(value) != param_dim:
                    raise InputParameterError(
                        "Expected parameter {0!r} to be of dimension "
                        "{1}".format(self.name, param_dim))
                # Validate each value
                values = [_tofloat(v) for v in value]
            except (TypeError, IndexError):
                raise InputParameterError(
                    "Expected a multivalued input of dimension {0} "
                    "for parameter {1!r}".format(param_dim, self.name))

            # Check that the value for each dimension has the same shape
            shapes = set(v[1] for v in values)
            if len(shapes) != 1:
                raise InputParameterError(
                    "The value for parameter {0!r} does not have the same "
                    "shape for every dimension".format(self.name))

            # Return the value for each dimension as a list, along with the
            # shape
            return [v[0] for v in values], shapes.pop()

    def __add__(self, value):
        return np.asarray(self) + value

    def __radd__(self, value):
        return np.asarray(self) + value

    def __sub__(self, value):
        return np.asarray(self) - value

    def __rsub__(self, value):
        return value - np.asarray(self)

    def __mul__(self, value):
        return np.asarray(self) * value

    def __rmul__(self, value):
        return np.asarray(self) * value

    def __pow__(self, val):
        return np.asarray(self) ** val

    def __div__(self, value):
        return np.asarray(self) / value

    def __rdiv__(self, value):
        return value / np.asarray(self)

    def __truediv__(self, value):
        return np.asarray(self) / value

    def __rtruediv__(self, value):
        return value / np.asarray(self)

    def __eq__(self, value):
        return (np.asarray(self) == np.asarray(value)).all()

    def __ne__(self, value):
        return not (np.asarray(self) == np.asarray(value)).all()

    def __lt__(self, value):
        return (np.asarray(self) < np.asarray(value)).all()

    def __gt__(self, value):
        return (np.asarray(self) > np.asarray(value)).all()

    def __le__(self, value):
        return (np.asarray(self) <= np.asarray(value)).all()

    def __ge__(self, value):
        return (np.asarray(self) >= np.asarray(value)).all()

    def __neg__(self):
        return np.asarray(self) * (-1)

    def __abs__(self):
        return np.abs(np.asarray(self))


class Parameters(list):

    """
    Store model parameters as a flat list of floats.

    This is a list-like object which stores model parameters. Only  instances
    of `~astropy.modeling.core.ParametricModel` keep an instance of this class
    as an attribute. The list of parameters can be modified by the user or by
    an instance of `~astropy.modeling.fitting.Fitter`.
    This list of parameters is kept in sync with single model parameter attributes.
    When more than one dimensional, a `~astropy.modeling.fitting.Fitter` treats each
    set of parameters as belonging to the same model but different set of data.

    Parameters
    ----------
    instance : object
        an instance of a subclass of `~astropy.modeling.core.ParametricModel`
    """

    def __init__(self, model):
        self._model = model
        # A flag set to True by a fitter to indicate that the flat
        # list of parameters has been changed.
        self._modified = False
        self.parinfo = {}
        flat = self._flatten()
        super(Parameters, self).__init__(flat)

    # Although deprecated the list class implements __setslice__ so we must
    # do the same here
    def __setslice__(self, i, j, value):
        self.__setitem__(slice(i, j), value)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            for idx, val in zip(range(*key.indices(len(self))), value):
                self.__setitem__(idx, val)
        else:
            _value = _tofloat(value)[0]
            super(Parameters, self).__setitem__(key, _value)
            self._modified = True
            self._update_model_pars()

    def _update_model_pars(self):
        """
        Update single parameters
        """

        for key, value in self.parinfo.items():
            sl = value[0]
            par = self[sl]
            if len(par) == 1:
                par = par[0]
            setattr(self._model, key, par)
        self._modified = False

    def _is_same_length(self, newpars):
        """
        Checks if the user supplied value of
        `~astropy.modeling.core.ParametricModel.parameters`
        has the same length as the original parameters list.

        """

        # TODO: Would len(newpars) == len(parsize) really not be sufficient?
        parsize = _tofloat(newpars)[0].size
        return parsize == len(self)

    def _flatten(self):
        """
        Create a list of model parameters
        """

        param_names = self._model.param_names
        parlist = [getattr(self._model, attr) for attr in param_names]
        flatpars = []
        start = 0
        for (name, par) in zip(param_names, parlist):
            pararr = np.array(par)
            fpararr = pararr.flatten()

            stop = start + len(fpararr)
            self.parinfo[name] = (slice(start, stop, 1), pararr.shape)
            start = stop
            flatpars.extend(list(fpararr))
        return flatpars
