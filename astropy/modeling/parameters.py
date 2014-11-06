# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module defines two classes that deal with parameters.

It is unlikely users will need to work with these classes directly, unless they
define their own models.
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import collections
import inspect
import functools
import numbers

import numpy as np

from ..utils import isiterable
from ..utils.compat import ignored
from ..extern import six
from .utils import check_broadcast, IncompatibleShapeError, asarray

__all__ = ['Parameter', 'InputParameterError']


class InputParameterError(ValueError):
    """Used for incorrect input parameter values and definitions."""


class Parameter(object):
    """
    Wraps individual parameters.

    This class represents a model's parameter (in a somewhat broad sense).  It
    acts as both a descriptor that can be assigned to a class attribute to
    describe the parameters accepted by an individual model (this is called an
    "unbound parameter"), or it can act as a proxy for the parameter values on
    an individual model instance (called a "bound parameter").

    Parameter instances never store the actual value of the parameter
    directly.  Rather, each instance of a model stores its own parameters
    as either hidden attributes or (in the case of
    `~astropy.modeling.FittableModel`) in an array.  A *bound*
    Parameter simply wraps the value in a Parameter proxy which provides some
    additional information about the parameter such as its constraints.

    *Unbound* Parameters are not associated with any specific model instance,
    and are merely used by model classes to determine the names of their
    parameters and other information about each parameter such as their default
    values and default constraints.

    Parameters
    ----------
    name : str
        parameter name
    description : str
        parameter description
    default : float or array
        default value to use for this parameter
    getter : callable
        a function that wraps the raw (internal) value of the parameter
        when returning the value through the parameter proxy (eg. a
        parameter may be stored internally as radians but returned to the
        user as degrees)
    setter : callable
        a function that wraps any values assigned to this parameter; should
        be the inverse of getter
    fixed : bool
        if True the parameter is not varied during fitting
    tied : callable or False
        if callable is supplied it provides a way to link the value of this
        parameter to another parameter (or some other arbitrary function)
    min : float
        the lower bound of a parameter
    max : float
        the upper bound of a parameter
    model : object
        an instance of a Model class; this should only be used internally for
        creating bound Parameters
    """

    # See the _nextid classmethod
    _nextid = 1

    def __init__(self, name='', description='', default=None, getter=None,
                 setter=None, fixed=False, tied=False, min=None, max=None,
                 model=None):
        super(Parameter, self).__init__()

        if model is not None and not name:
            raise TypeError('Bound parameters must have a name specified.')

        self._name = name
        self.__doc__ = description.strip()
        self._default = default

        self._default_fixed = fixed
        self._default_tied = tied
        self._default_min = min
        self._default_max = max

        self._order = None
        self._model = model

        self._getter = getter
        self._wrapped_getter = None
        self._setter = setter
        self._wrapped_setter = None

        if model is None:
            # Only Parameters declared as class-level descriptors require
            # and ordering ID
            self._order = self._get_nextid()

    def __get__(self, obj, objtype):
        if obj is None:
            return self

        return self.__class__(self._name, default=self._default,
                              getter=self._getter, setter=self._setter,
                              fixed=self._default_fixed,
                              tied=self._default_tied, min=self._default_min,
                              max=self._default_max, model=obj)

    def __set__(self, obj, value):
        try:
            value = asarray(value)
        except ValueError:
            raise InputParameterError(
                "Value {0} for parameter '{1}' could not be converted to a "
                "floating point (or complex) array.".format(
                    value, self._name))

        if self._setter is not None:
            setter = self._create_value_wrapper(self._setter, obj)
            value = setter(value)

        obj._parameters[self._name] = value

    def __len__(self):
        if self._model is None:
            raise TypeError('Parameter definitions do not have a length.')
        return len(self._model)

    def __getitem__(self, key):
        value = self.value
        if len(self._model) == 1:
            # Wrap the value in a list so that getitem can work for sensible
            # indices like [0] and [-1]
            value = [value]
        return value[key]

    def __setitem__(self, key, value):
        # Get the existing value and check whether it even makes sense to
        # apply this index
        oldvalue = self.value
        n_models = len(self._model)

        #if n_models == 1:
        #    # Convert the single-dimension value to a list to allow some slices
        #    # that would be compatible with a length-1 array like [:] and [0:]
        #    oldvalue = [oldvalue]

        if isinstance(key, slice):
            if len(oldvalue[key]) == 0:
                raise InputParameterError(
                    "Slice assignment outside the parameter dimensions for "
                    "'{0}'".format(self.name))
            for idx, val in zip(range(*key.indices(len(self))), value):
                self.__setitem__(idx, val)
        else:
            try:
                oldvalue[key] = value
            except IndexError:
                raise InputParameterError(
                    "Input dimension {0} invalid for {1!r} parameter with "
                    "dimension {2}".format(key, self.name, n_models))

    def __repr__(self):
        if self._model is None:
            return "Parameter('{0}')".format(self._name)
        else:
            return "Parameter('{0}', value={1})".format(
                self._name, self.value)

    @property
    def name(self):
        """Parameter name"""

        return self._name

    @property
    def default(self):
        """Parameter default value"""

        return self._default  # Just for now...

        if (self._model is None or self._default is None or
                len(self._model) == 1):
            return self._default

        # Otherwise the model we are providing for has more than one parameter
        # sets, so ensure that the default is repeated the correct number of
        # times along the model_set_axis if necessary
        n_models = len(self._model)
        model_set_axis = self._model._model_set_axis
        default = self._default
        new_shape = (np.shape(default) +
                     (1,) * (model_set_axis + 1 - np.ndim(default)))
        default = np.reshape(default, new_shape)
        # Now roll the new axis into its correct position if necessary
        default = np.rollaxis(default, -1, model_set_axis)
        # Finally repeat the last newly-added axis to match n_models
        default = np.repeat(default, n_models, axis=-1)

        # NOTE: Regardless of what order the last two steps are performed in,
        # the resulting array will *look* the same, but only if the repeat is
        # performed last will it result in a *contiguous* array

        return default

    @property
    def value(self):
        """The unadorned value proxied by this parameter"""

        if self._model is None:
            raise AttributeError('Parameter definition does not have a value')

        if not hasattr(self._model, '_parameters'):
            # The _parameters array hasn't been initialized yet; just translate
            # this to an AttributeError
            raise AttributeError(self._name)

        value = self._model._parameters[self._name]

        if self._getter is None:
            return value
        else:
            return self._getter(value)

    @value.setter
    def value(self, value):
        if self._model is None:
            raise AttributeError('Cannot set a value on a parameter '
                                 'definition')

        if self._setter is not None:
            val = self._setter(value)

        self._model._parameters[self._name] = value

    @property
    def getter(self):
        if self._getter is None:
            return None

        # The getter/setter functions take one or two arguments: The first
        # argument is always the value itself (either the value returned or the
        # value being set).  The second argument is optional, but if present
        # will contain a reference to the model object tied to a parameter (if
        # this is a bound parameter)
        if self._wrapped_getter is None:
            self._wrapped_getter = self._create_value_wrapper(self._getter,
                                                              self._model)

        return self._wrapped_getter

    @property
    def setter(self):
        if self._setter is None:
            return None

        if self._wrapped_setter is None:
            self._wrapped_setter = self._create_value_wrapper(self._setter,
                                                              self._model)

        return self._wrapped_setter

    @property
    def shape(self):
        """The shape of this parameter's value array."""

        if self._model._parameters.n_models == 1:
            return np.shape(self.value)
        else:
            shape = np.shape(self.value)
            model_axis = self._model._parameters._model_set_axis
            if model_axis < 0:
                model_axis = len(shape) + model_axis
            shape = shape[:model_axis] + shape[model_axis + 1:]

            return shape

    @property
    def size(self):
        """The size of this parameter's value array."""

        return np.size(self.value)

    @property
    def fixed(self):
        """
        Boolean indicating if the parameter is kept fixed during fitting.
        """

        if self._model is not None:
            fixed = self._model._constraints['fixed']
            return fixed.get(self._name, self._default_fixed)
        else:
            return self._default_fixed

    @fixed.setter
    def fixed(self, value):
        """Fix a parameter"""
        if self._model is not None:
            if not isinstance(value, bool):
                raise TypeError("Fixed can be True or False")
            self._model._constraints['fixed'][self._name] = value
        else:
            raise AttributeError("can't set attribute 'fixed' on Parameter "
                                 "definition")

    @property
    def tied(self):
        """
        Indicates that this parameter is linked to another one.

        A callable which provides the relationship of the two parameters.
        """

        if self._model is not None:
            tied = self._model._constraints['tied']
            return tied.get(self._name, self._default_tied)
        else:
            return self._default_tied

    @tied.setter
    def tied(self, value):
        """Tie a parameter"""

        if self._model is not None:
            if not six.callable(value) and value not in (False, None):
                    raise TypeError("Tied must be a callable")
            self._model._constraints['tied'][self._name] = value
        else:
            raise AttributeError("can't set attribute 'tied' on Parameter "
                                 "definition")

    @property
    def bounds(self):
        """The minimum and maximum values of a parameter as a tuple"""

        if self._model is not None:
            bounds = self._model._constraints['bounds']
            default_bounds = (self._default_min, self._default_max)
            return bounds.get(self._name, default_bounds)
        else:
            return (self._default_min, self._default_max)

    @bounds.setter
    def bounds(self, value):
        """Set the minimum and maximum values of a parameter from a tuple"""

        if self._model is not None:
            _min, _max = value
            if _min is not None:
                if not isinstance(_min, numbers.Number):
                        raise TypeError("Min value must be a number")
                _min = float(_min)

            if _max is not None:
                if not isinstance(_max, numbers.Number):
                        raise TypeError("Max value must be a number")
                _max = float(_max)

            bounds = self._model._constraints.setdefault('bounds', {})
            self._model._constraints['bounds'][self._name] = (_min, _max)
        else:
            raise AttributeError("can't set attribute 'bounds' on Parameter "
                                 "definition")

    @property
    def min(self):
        """A value used as a lower bound when fitting a parameter"""

        return self.bounds[0]

    @min.setter
    def min(self, value):
        """Set a minimum value of a parameter"""

        if self._model is not None:
            self.bounds = (value, self.max)
        else:
            raise AttributeError("can't set attribute 'min' on Parameter "
                                 "definition")

    @property
    def max(self):
        """A value used as an upper bound when fitting a parameter"""

        return self.bounds[1]

    @max.setter
    def max(self, value):
        """Set a maximum value of a parameter."""

        if self._model is not None:
            self.bounds = (self.min, value)
        else:
            raise AttributeError("can't set attribute 'max' on Parameter "
                                 "definition")

    @classmethod
    def _get_nextid(cls):
        """Returns a monotonically increasing ID used to order Parameter
        descriptors declared at the class-level of Model subclasses.

        This allows the desired parameter order to be determined without
        having to list it manually in the param_names class attribute.
        """

        nextid = cls._nextid
        cls._nextid += 1
        return nextid

    def _create_value_wrapper(self, wrapper, model):
        """Wraps a getter/setter function to support optionally passing in
        a reference to the model object as the second argument.

        If a model is tied to this parameter and its getter/setter supports
        a second argument then this creates a partial function using the model
        instance as the second argument.
        """

        if isinstance(wrapper, np.ufunc):
            if wrapper.nin != 1:
                raise TypeError("A numpy.ufunc used for Parameter "
                                "getter/setter may only take one input "
                                "argument")
        else:
            wrapper_args = inspect.getargspec(wrapper)
            nargs = len(wrapper_args.args)

            if nargs == 1:
                pass
            elif nargs == 2:
                if model is not None:
                    # Don't make a partial function unless we're tied to a
                    # specific model instance
                    model_arg = wrapper_args.args[1]
                    wrapper = functools.partial(wrapper, **{model_arg: model})
            else:
                raise TypeError("Parameter getter/setter must be a function "
                                "of either one or two arguments")

        return wrapper

    def __array__(self, dtype=None):
        # Make np.asarray(self) work a little more straightforwardly
        if self._model is None:
            return np.array([], dtype=np.float)
        else:
            return np.asarray(self.value, dtype=dtype)

    def __nonzero__(self):
        if self._model is None:
            return True
        else:
            return bool(self.value)

    __bool__ = __nonzero__

    def __add__(self, val):
        return self.value + val

    def __radd__(self, val):
        return val + self.value

    def __sub__(self, val):
        return self.value - val

    def __rsub__(self, val):
        return val - self.value

    def __mul__(self, val):
        return self.value * val

    def __rmul__(self, val):
        return val * self.value

    def __pow__(self, val):
        return self.value ** val

    def __rpow__(self, val):
        return val ** self.value

    def __div__(self, val):
        return self.value / val

    def __rdiv__(self, val):
        return val / self.value

    def __truediv__(self, val):
        return self.value / val

    def __rtruediv__(self, val):
        return val / self.value

    def __eq__(self, val):
        return (np.asarray(self) == np.asarray(val)).all()

    def __ne__(self, val):
        return not (np.asarray(self) == np.asarray(val)).all()

    def __lt__(self, val):
        return (np.asarray(self) < np.asarray(val)).all()

    def __gt__(self, val):
        return (np.asarray(self) > np.asarray(val)).all()

    def __le__(self, val):
        return (np.asarray(self) <= np.asarray(val)).all()

    def __ge__(self, val):
        return (np.asarray(self) >= np.asarray(val)).all()

    def __neg__(self):
        return -self.value

    def __abs__(self):
        return np.abs(self.value)


class Parameters(object):
    def __init__(self, parameters, param_names=None, n_models=None,
                 model_set_axis=None):

        if param_names is None:
            param_names = tuple(parameters.keys())

        if model_set_axis is None:
            if n_models is not None and n_models > 1:
                # Default to zero
                model_set_axis = 0
            else:
                # Otherwise disable
                model_set_axis = False

        self._param_names = param_names
        self._model_set_axis = model_set_axis

        total_size = 0
        dtype = np.float
        param_values = {}
        param_metrics = collections.defaultdict(lambda: {})

        for name in param_names:
            value = asarray(parameters[name]['value'])
            # TODO: Keep track of parameters with complex values
            param_values[name] = value
            param_size = np.size(value)
            param_slice = slice(total_size, total_size + param_size)
            total_size += param_size
            param_metrics[name] = {'slice': param_slice,
                                   'shape': np.shape(value)}

        self._param_metrics = param_metrics

        # Determine the number of model sets: If the model_set_axis is
        # None then there is just one parameter set; otherwise it is determined
        # by the size of that axis on the first parameter--if the other
        # parameters don't have the right number of axes or the sizes of their
        # model_set_axis don't match an error is raised
        if model_set_axis is not False and n_models != 1:
            max_ndim = 0
            if model_set_axis < 0:
                min_ndim = abs(model_set_axis)
            else:
                min_ndim = model_set_axis + 1

            for name, value in six.iteritems(param_values):
                param_ndim = np.ndim(value)
                if param_ndim < min_ndim:
                    raise InputParameterError(
                        "All parameter values must be arrays of dimension "
                        "at least {0} for model_set_axis={1} (the value "
                        "given for {2!r} is only {3}-dimensional)".format(
                            min_ndim, model_set_axis, name, param_ndim))

                max_ndim = max(max_ndim, param_ndim)

                if n_models is None:
                    # Use the dimensions of the first parameter to determine
                    # the number of model sets
                    n_models = value.shape[model_set_axis]
                elif value.shape[model_set_axis] != n_models:
                    raise InputParameterError(
                        "Inconsistent dimensions for parameter {0!r} for "
                        "{1} model sets.  The length of axis {2} must be the "
                        "same for all input parameter values".format(
                        name, n_models, model_set_axis))

            self._check_param_broadcast(param_values, max_ndim,
                                        model_set_axis)
        else:
            if n_models is None:
                n_models = 1

            self._check_param_broadcast(param_values, None, None)

        self._n_models = n_models

        offset = 0
        array = np.empty(total_size, dtype=dtype)
        for name in param_names:
            value = param_values[name]
            size = np.size(value)
            array[offset:offset + size] = value.ravel()
            offset += size

        self._array = array

    @property
    def param_names(self):
        return self._param_names

    @property
    def array(self):
        return self._array

    @property
    def n_models(self):
        return self._n_models

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._param_names[index]

        return self._get_parameter_value(index)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            self._array[index] = value
            return

        if isinstance(index, int):
            index = self._param_names[index]

        self._set_parameter_value(index, value)

    def __iter__(self):
        for name in self._param_names:
            yield self._get_parameter_value(name)

    def __array__(self):
        return self._array

    def __len__(self):
        return len(self._array)

    def __eq__(self, other):
        return self._array == other

    def iter_broadcasted(self):
        param_metrics = self._param_metrics
        get_value = self._get_parameter_value
        for name in self._param_names:
            value = get_value(name)
            broadcast_shape = param_metrics[name].get('broadcast_shape')
            if broadcast_shape is not None:
                value = value.reshape(broadcast_shape)
            if self._n_models == 1:
                # Always add an additional dimension representing the "model
                # set axis" even when there is only one parameter set, just for
                # consistency's sake
                value = np.expand_dims(value, 0)
            yield value

    def _get_parameter_value(self, name):
        """
        This method implements how to retrieve the value of this parameter from
        the model instance.  See also `Parameter._set_model_value`.

        These methods take an explicit model argument rather than using
        self._model so that they can be used from unbound `Parameter`
        instances.
        """

        # Use the _param_metrics to extract the parameter value from the
        # _parameters array
        param_metrics = self._param_metrics[name]
        param_slice = param_metrics['slice']
        param_shape = param_metrics['shape']
        value = self._array[param_slice]
        if param_shape:
            value = value.reshape(param_shape)
        else:
            value = value[0]
        return value

    def _set_parameter_value(self, name, value):
        """
        This method implements how to store the value of a parameter on the
        model instance.

        Currently there is only one storage mechanism (via the ._parameters
        array) but other mechanisms may be desireable, in which case really the
        model class itself should dictate this and *not* `Parameter` itself.
        """

        # TODO: Maybe handle exception on invalid input shape
        param_metrics = self._param_metrics[name]
        param_slice = param_metrics['slice']
        param_shape = param_metrics['shape']
        param_size = np.prod(param_shape)

        value = asarray(value)

        if np.size(value) != param_size:
            raise InputParameterError(
                "Input value for parameter {0!r} does not have {1} elements "
                "as the current value does".format(name, param_size))

        self._array[param_slice] = value.ravel()

    def _check_param_broadcast(self, params, max_ndim, model_set_axis):
        """
        This subroutine checks that all parameter arrays can be broadcast
        against each other, and determimes the shapes parameters must have in
        order to broadcast correctly.

        If model_set_axis is None this merely checks that the parameters
        broadcast and returns an empty dict if so.  This mode is only used for
        single model sets.
        """

        param_metrics = self._param_metrics
        all_shapes = []
        param_names = []

        for name in self._param_names:
            # Previously this just used iteritems(params), but we loop over all
            # param_names instead just to ensure some determinism in the
            # ordering behavior
            if name not in params:
                continue

            value = params[name]
            param_names.append(name)
            # We've already checked that each parameter array is compatible in
            # the model_set_axis dimension, but now we need to check the
            # dimensions excluding that axis
            # Split the array dimensions into the axes before model_set_axis
            # and after model_set_axis
            param_shape = np.shape(value)

            param_ndim = len(param_shape)
            if max_ndim is not None and param_ndim < max_ndim:
                # All arrays have the same number of dimensions up to the
                # model_set_axis dimension, but after that they may have a
                # different number of trailing axes.  The number of trailing
                # axes must be extended for mutual compatibility.  For example
                # if max_ndim = 3 and model_set_axis = 0, an array with the
                # shape (2, 2) must be extended to (2, 1, 2).  However, an
                # array with shape (2,) is extended to (2, 1).
                new_axes = (1,) * (max_ndim - param_ndim)

                if self._model_set_axis < 0:
                    # Just need to prepend axes to make up the difference
                    broadcast_shape = new_axes + param_shape
                else:
                    broadcast_shape = (
                        param_shape[:self._model_set_axis + 1] + new_axes +
                        param_shape[self._model_set_axis + 1:])
                param_metrics[name]['broadcast_shape'] = broadcast_shape
                all_shapes.append(broadcast_shape)
            else:
                all_shapes.append(param_shape)

        # Now check mutual broadcastability of all shapes
        try:
            check_broadcast(*all_shapes)
        except IncompatibleShapeError as exc:
            shape_a, shape_a_idx, shape_b, shape_b_idx = exc.args
            param_a = param_names[shape_a_idx]
            param_b = param_names[shape_b_idx]

            raise InputParameterError(
                "Parameter {0!r} of shape {1!r} cannot be broadcast with "
                "parameter {2!r} of shape {3!r}.  All parameter arrays "
                "must have shapes that are mutually compatible according "
                "to the broadcasting rules.".format(param_a, shape_a,
                                                    param_b, shape_b))
