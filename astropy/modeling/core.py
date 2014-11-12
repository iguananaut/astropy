# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module defines base classes for all models.  The base class of all
models is `~astropy.modeling.Model`. `~astropy.modeling.FittableModel` is
the base class for all fittable models. Fittable models can be linear or
nonlinear in a regression analysis sense.

All models provide a `__call__` method which performs the transformation in
a purely mathematical way, i.e. the models are unitless.  Model instances can
represent either a single model, or a "model set" representing multiple copies
of the same type of model, but with potentially different values of the
parameters in each model making up the set.
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import abc
import copy
import inspect
import itertools
import functools
import operator
import warnings

from collections import defaultdict

import numpy as np

from ..utils import indent, isiterable, isinstancemethod
from ..extern import six
from ..extern.six.moves import zip as izip
from ..extern.six.moves import range
from ..table import Table
from ..utils import deprecated, sharedmethod, find_current_module
from ..utils.codegen import make_function_with_signature
from ..utils.exceptions import AstropyDeprecationWarning
from .utils import (array_repr_oneline, check_broadcast, combine_labels,
                    make_binary_operator_eval, ExpressionTree,
                    IncompatibleShapeError)

from .parameters import Parameter, InputParameterError


__all__ = ['Model', 'FittableModel', 'Fittable1DModel', 'Fittable2DModel',
           'custom_model', 'ModelDefinitionError']


class ModelDefinitionError(Exception):
    """Used for incorrect models definitions"""


class _ModelMeta(abc.ABCMeta):
    """
    Metaclass for Model.

    Currently just handles auto-generating the param_names list based on
    Parameter descriptors declared at the class-level of Model subclasses.
    """

    registry = set()
    """
    A registry of all known concrete (non-abstract) Model subclasses.
    """

    def __new__(mcls, name, bases, members):
        parameters = mcls._handle_parameters(members, name)
        mcls._handle_backwards_compat(members, name)
        mcls._create_inverse_property(members)

        cls = super(_ModelMeta, mcls).__new__(mcls, name, bases, members)

        mcls._handle_special_methods(members, cls, parameters)

        if not inspect.isabstract(cls) and not name.startswith('_'):
            mcls.registry.add(cls)

        return cls

    def __repr__(cls):
        """
        Custom repr for Model subclasses.
        """

        return cls._format_cls_repr()

    def _repr_pretty_(cls, p, cycle):
        """
        Repr for IPython's pretty printer.

        By default IPython "pretty prints" classes, so we need to implement
        this so that IPython displays the custom repr for Models.
        """

        p.text(repr(cls))

    @property
    def name(cls):
        """
        The name of this model class--equivalent to ``cls.__name__``.

        This attribute is provided for symmetry with the `Model.name` attribute
        of model instances.
        """

        return cls.__name__

    @property
    def n_inputs(cls):
        return len(cls.inputs)

    @property
    def n_outputs(cls):
        return len(cls.outputs)

    def rename(cls, name):
        """
        Creates a copy of this model class with a new name.

        The new class is technically a subclass of the original class, so that
        instance and type checks will still work.  For example::

            >>> from astropy.modeling.models import Rotation2D
            >>> SkyRotation = Rotation2D.rename('SkyRotation')
            >>> SkyRotation
            <class '__main__.SkyRotation'>
            Name: SkyRotation (Rotation2D)
            Inputs: ('x', 'y')
            Outputs: ('x', 'y')
            Fittable parameters: ('angle',)
            >>> issubclass(SkyRotation, Rotation2D)
            True
            >>> r = SkyRotation(90)
            >>> isinstance(r, Rotation2D)
            True
        """

        if six.PY2 and isinstance(name, six.text_type):
            # Unicode names are not allowed in Python 2, so just convert to
            # ASCII.  As such, for cross-compatibility all model names should
            # just be ASCII for now.
            name = name.encode('ascii')

        mod = find_current_module(2)
        if mod:
            modname = mod.__name__
        else:
            modname = '__main__'

        new_cls = type(name, (cls,), {})
        # On Python 2 __module__ must be a str, not unicode
        new_cls.__module__ = str(modname)

        if hasattr(cls, '__qualname__'):
            if new_cls.__module__ == '__main__':
                # __main__ is not added to a class's qualified name
                new_cls.__qualname__ = name
            else:
                new_cls.__qualname__ = '{0}.{1}'.format(modname, name)

        return new_cls

    @classmethod
    def _handle_parameters(mcls, members, name):
        # Handle parameters
        param_names = members.get('param_names', ())
        parameters = {}
        for key, value in members.items():
            if not isinstance(value, Parameter):
                continue
            if not value.name:
                # Name not explicitly given in the constructor; add the name
                # automatically via the attribute name
                value._name = key
                value._attr = '_' + key
            if value.name != key:
                raise ModelDefinitionError(
                    "Parameters must be defined with the same name as the "
                    "class attribute they are assigned to.  Parameters may "
                    "take their name from the class attribute automatically "
                    "if the name argument is not given when initializing "
                    "them.")
            parameters[value.name] = value

        # If no parameters were defined get out early--this is especially
        # important for PolynomialModels which take a different approach to
        # parameters, since they can have a variable number of them
        if parameters:
            mcls._check_parameters(name, members, param_names, parameters)

        return parameters

    @staticmethod
    def _check_parameters(name, members, param_names, parameters):
        # If param_names was declared explicitly we use only the parameters
        # listed manually in param_names, but still check that all listed
        # parameters were declared
        if param_names and isiterable(param_names):
            for param_name in param_names:
                if param_name not in parameters:
                    raise RuntimeError(
                        "Parameter {0!r} listed in {1}.param_names was not "
                        "declared in the class body.".format(param_name, name))
        else:
            param_names = tuple(param.name for param in
                                sorted(parameters.values(),
                                       key=lambda p: p._order))
            members['param_names'] = param_names
            members['_param_orders'] = \
                    dict((name, idx) for idx, name in enumerate(param_names))

    @staticmethod
    def _create_inverse_property(members):
        inverse = members.get('inverse', None)
        if inverse is None:
            return

        if isinstance(inverse, property):
            fget = inverse.fget
        else:
            # We allow the @property decoratore to be ommitted entirely from
            # the class definition, though its use should be encouraged for
            # clarity
            fget = inverse

        def wrapped_fget(self):
            if self._custom_inverse is not None:
                return self._custom_inverse

            return fget(self)

        def fset(self, value):
            if not isinstance(value, (Model, type(None))):
                raise ValueError(
                    "The ``inverse`` attribute may be assigned a `Model` "
                    "instance or `None` (where `None` restores the default "
                    "inverse for this model if one is defined.")

            self._custom_inverse = value

        members['inverse'] = property(wrapped_fget, fset,
                                      doc=inverse.__doc__)

    @classmethod
    def _handle_backwards_compat(mcls, members, name):
        # Backwards compatibility check for 'eval' -> 'evaluate'
        # TODO: Remove sometime after Astropy 1.0 release.
        if 'eval' in members and 'evaluate' not in members:
            warnings.warn(
                "Use of an 'eval' method when defining subclasses of "
                "FittableModel is deprecated; please rename this method to "
                "'evaluate'.  Otherwise its semantics remain the same.",
                AstropyDeprecationWarning)
            members['evaluate'] = members['eval']
        elif 'evaluate' in members:
            alt = '.'.join((name, 'evaluate'))
            deprecate = deprecated('1.0', alternative=alt, name='eval')
            members['eval'] = deprecate(members['evaluate'])

    @classmethod
    def _handle_special_methods(mcls, members, cls, parameters):
        # Handle init creation from inputs
        if '__call__' not in members and 'inputs' in members:
            inputs = members['inputs']
            # Done create a custom __call__ for classes that already have one
            # explicitly defined (this includes the Model base class, and any
            # other classes that manually override __call__
            def __call__(self, *inputs, **kwargs):
                """Evaluate this model on the supplied inputs."""

                return super(cls, self).__call__(*inputs, **kwargs)

            args = ('self',) + inputs
            cls.__call__ = make_function_with_signature(
                    __call__, args, [('model_set_axis', None)])

        if '__init__' not in members and parameters:
            # If *all* the parameters have default values we can make them
            # keyword arguments; otherwise they must all be positional
            # arguments
            if all(p.default is not None
                   for p in six.itervalues(parameters)):
                args = ('self',)
                kwargs = [(name, parameters[name].default)
                          for name in cls.param_names]
            else:
                args = ('self',) + cls.param_names
                kwargs = {}

            def __init__(self, *params, **kwargs):
                return super(cls, self).__init__(*params, **kwargs)

            cls.__init__ = make_function_with_signature(
                    __init__, args, kwargs, varkwargs='kwargs')

    # *** Arithmetic operators for creating compound models ***
    def __add__(cls, other):
        return _make_compound_model(cls, other, '+')

    def __sub__(cls, other):
        return _make_compound_model(cls, other, '-')

    def __mul__(cls, other):
        return _make_compound_model(cls, other, '*')

    def __truediv__(cls, other):
        return _make_compound_model(cls, other, '/')

    def __pow__(cls, other):
        return _make_compound_model(cls, other, '**')

    def __or__(cls, other):
        return _make_compound_model(cls, other, '|')

    def __and__(cls, other):
        return _make_compound_model(cls, other, '&')

    # *** Other utilities ***

    def _format_cls_repr(cls, keywords=[]):
        """
        Internal implementation of ``__repr__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__repr__`` while keeping the same basic
        formatting.
        """

        # For the sake of familiarity start the output with the standard class
        # __repr__
        parts = [super(_ModelMeta, cls).__repr__()]

        if cls.__name__.startswith('_') or inspect.isabstract(cls):
            return parts[0]

        def format_inheritance(cls):
            bases = []
            for base in cls.mro()[1:]:
                if not issubclass(base, Model):
                    continue
                elif (inspect.isabstract(base) or
                        base.__name__.startswith('_')):
                    break
                bases.append(base.name)
            if bases:
                return '{0} ({1})'.format(cls.name, ' -> '.join(bases))
            else:
                return cls.name

        try:
            default_keywords = [
                ('Name', format_inheritance(cls)),
                ('Inputs', cls.inputs),
                ('Outputs', cls.outputs),
            ]

            if cls.param_names:
                default_keywords.append(('Fittable parameters',
                                         cls.param_names))

            for keyword, value in default_keywords + keywords:
                if value is not None:
                    parts.append('{0}: {1}'.format(keyword, value))

            return '\n'.join(parts)
        except:
            # If any of the above formatting fails fall back on the basic repr
            # (this is particularly useful in debugging)
            return parts[0]


@six.add_metaclass(_ModelMeta)
class Model(object):
    """
    Base class for all models.

    This is an abstract class and should not be instantiated directly.

    This class sets the constraints and other properties for all individual
    parameters and performs parameter validation.

    Parameters
    ----------
    name : str, optional
        A human-friendly name associated with this model instance
        (particularly useful for identifying the individual components of a
        compound model).

    fixed : dict, optional
        Dictionary ``{parameter_name: bool}`` setting the fixed constraint
        for one or more parameters.  `True` means the parameter is held fixed
        during fitting and is prevented from updates once an instance of the
        model has been created.

        Alternatively the `~astropy.modeling.Parameter.fixed` property of a
        parameter may be used to lock or unlock individual parameters.

    tied : dict, optional
        Dictionary ``{parameter_name: callable}`` of parameters which are
        linked to some other parameter. The dictionary values are callables
        providing the linking relationship.

        Alternatively the `~astropy.modeling.Parameter.tied` property of a
        parameter may be used to set the ``tied`` constraint on individual
        parameters.

    bounds : dict, optional
        Dictionary ``{parameter_name: value}`` of lower and upper bounds of
        parameters. Keys are parameter names. Values are a list of length 2
        giving the desired range for the parameter.

        Alternatively the `~astropy.modeling.Parameter.min` and
        `~astropy.modeling.Parameter.max` or
        ~astropy.modeling.Parameter.bounds` properties of a parameter may be
        used to set bounds on individual parameters.

    eqcons : list, optional
        List of functions of length n such that ``eqcons[j](x0, *args) == 0.0``
        in a successfully optimized problem.

    ineqcons : list, optional
        List of functions of length n such that ``ieqcons[j](x0, *args) >=
        0.0`` is a successfully optimized problem.

    Examples
    --------
    >>> from astropy.modeling import models
    >>> def tie_center(model):
    ...         mean = 50 * model.stddev
    ...         return mean
    >>> tied_parameters = {'mean': tie_center}

    Specify that ``'mean'`` is a tied parameter in one of two ways:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        tied=tied_parameters)

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.mean.tied
    False
    >>> g1.mean.tied = tie_center
    >>> g1.mean.tied
    <function tie_center at 0x...>

    Fixed parameters:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        fixed={'stddev': True})
    >>> g1.stddev.fixed
    True

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.stddev.fixed
    False
    >>> g1.stddev.fixed = True
    >>> g1.stddev.fixed
    True
    """

    parameter_constraints = ('fixed', 'tied', 'bounds')
    model_constraints = ('eqcons', 'ineqcons')

    param_names = ()
    """
    Names of the parameters that describe models of this type.

    The parameters in this tuple are in the same order they should be passed in
    when initializing a model of a specific type.  Some types of models, such
    as polynomial models, have a different number of parameters depending on
    some other property of the model, such as the degree.
    """

    inputs = ()
    outputs = ()

    standard_broadcasting = True
    fittable = False
    linear = True

    _custom_inverse = None

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

        self._name = kwargs.pop('name', None)

        self._initialize_constraints(kwargs)
        # Remaining keyword args are either parameter values or invalid
        # Parameter values must be passed in as keyword arguments in order to
        # distinguish them
        self._initialize_parameters(args, kwargs)

    def __repr__(self):
        return self._format_repr()

    def __str__(self):
        return self._format_str()

    def __len__(self):
        return self._n_models

    def __call__(self, *inputs, **kwargs):
        inputs, format_info = self.prepare_inputs(*inputs, **kwargs)

        outputs = self.evaluate(*itertools.chain(inputs, self.param_sets))

        if self.n_outputs == 1:
            outputs = (outputs,)

        return self.prepare_outputs(format_info, *outputs, **kwargs)

    # *** Arithmetic operators for creating compound models ***
    def __add__(self, other):
        return _make_compound_model(self, other, '+')

    def __sub__(self, other):
        return _make_compound_model(self, other, '-')

    def __mul__(self, other):
        return _make_compound_model(self, other, '*')

    def __truediv__(self, other):
        return _make_compound_model(self, other, '/')

    def __pow__(self, other):
        return _make_compound_model(self, other, '**')

    def __or__(self, other):
        return _make_compound_model(self, other, '|')

    def __and__(self, other):
        return _make_compound_model(self, other, '&')

    # *** Properties ***
    @property
    def name(self):
        """User-provided name for this model instance."""

        return self._name

    @property
    @deprecated('0.4', alternative='len(model)')
    def param_dim(self):
        return self._n_models

    @property
    def n_inputs(self):
        return len(self.inputs)

    @property
    def n_outputs(self):
        return len(self.outputs)

    @property
    def model_set_axis(self):
        return self._model_set_axis

    @property
    def param_sets(self):
        """
        Return parameters as a pset.

        This is a list with one item per parameter set, which is an array of
        that parameter's values across all parameter sets, with the last axis
        associated with the parameter set.
        """

        values = [getattr(self, name).value for name in self.param_names]

        # Ensure parameter values are broadcastable
        for name, shape in six.iteritems(self._param_broadcast_shapes):
            idx = self._param_orders[name]
            values[idx] = values[idx].reshape(shape)

        shapes = [np.shape(value) for value in values]

        if len(self) == 1:
            # Add a single param set axis to the parameter's value (thus
            # converting scalars to shape (1,) array values) for consistency
            values = [np.array([value]) for value in values]

        if len(set(shapes)) != 1:
            # If the parameters are not all the same shape, converting to an
            # array is going to produce an object array
            # However the way Numpy creates object arrays is tricky in that it
            # will recurse into array objects in the list and break them up
            # into separate objects.  Doing things this way ensures a 1-D
            # object array the elements of which are the individual parameter
            # arrays.  There's not much reason to do this over returning a list
            # except for consistency
            psets = np.empty(len(values), dtype=object)
            psets[:] = values
            return psets

        return np.array(values)

    @property
    def parameters(self):
        """
        A flattened array of all parameter values in all parameter sets.

        Fittable parameters maintain this list and fitters modify it.
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):
        """
        Assigning to this attribute updates the parameters array rather than
        replacing it.
        """

        try:
            value = np.array(value).reshape(self._parameters.shape)
        except ValueError as e:
            raise InputParameterError(
                "Input parameter values not compatible with the model "
                "parameters array: {0}".format(e))

        self._parameters[:] = value

    @property
    def fixed(self):
        """
        A `dict` mapping parameter names to their fixed constraint.
        """

        return self._constraints['fixed']

    @property
    def tied(self):
        """
        A `dict` mapping parameter names to their tied constraint.
        """

        return self._constraints['tied']

    @property
    def bounds(self):
        """
        A `dict` mapping parameter names to their upper and lower bounds as
        ``(min, max)`` tuples.
        """

        return self._constraints['bounds']

    @property
    def eqcons(self):
        """List of parameter equality constraints."""

        return self._constraints['eqcons']

    @property
    def ineqcons(self):
        """List of parameter inequality constraints."""

        return self._constraints['ineqcons']

    # *** Public methods ***

    @property
    def inverse(self):
        """
        Returns a new `Model` instance which performs the inverse
        transform, if an analytic inverse is defined for this model.

        Even on models that don't have an inverse defined, this property can be
        set with a manually-defined inverse, such a pre-computed or
        experimentally determined inverse (often given as a
        `~astropy.modeling.polynomial.PolynomialModel`, but not by
        requirement).

        Note to authors of `Model` subclasses:  To define an inverse for a
        model simply override this property to return the appropriate model
        representing the inverse.  The machinery that will make the inverse
        manually-overridable is added automatically by the base class.
        """

        raise NotImplementedError("An analytical inverse transform has not "
                                  "been implemented for this model.")

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the model on some input variables."""

    def prepare_inputs(self, *inputs, **kwargs):
        """
        This method is used in `Model.__call__` to ensure that all the inputs
        to the model can be broadcast into compatible shapes (if one or both of
        them are input as arrays), particularly if there are more than one
        parameter sets.
        """

        model_set_axis = kwargs.pop('model_set_axis', None)

        if model_set_axis is None:
            # By default the model_set_axis for the input is assumed to be the
            # same as that for the parameters the model was defined with
            # TODO: Ensure that negative model_set_axis arguments are respected
            model_set_axis = self.model_set_axis

        n_models = len(self)

        params = [getattr(self, name) for name in self.param_names]
        inputs = [np.asanyarray(_input, dtype=float) for _input in inputs]

        scalar_params = all(not param.shape for param in params)
        scalar_inputs = all(not np.shape(_input) for _input in inputs)

        if n_models == 1 and scalar_params and scalar_inputs:
            # Simplest case is either a parameterless models (currently I don't
            # think we have any but they could exist in principle) or a single
            # model (not a model set) with all scalar paramaters and all scalar
            # inputs
            return inputs, ()

        _validate_input_shapes(inputs, self.inputs, n_models,
                               model_set_axis, self.standard_broadcasting)

        # The input formatting required for single models versus a multiple
        # model set are different enough that they've been split into separate
        # subroutines
        if n_models == 1:
            return _prepare_inputs_single_model(self, params, inputs,
                                                **kwargs)
        else:
            return _prepare_inputs_model_set(self, params, inputs, n_models,
                                             model_set_axis, **kwargs)

    def prepare_outputs(self, format_info, *outputs, **kwargs):
        if len(self) == 1:
            return _prepare_outputs_single_model(self, outputs, format_info)
        else:
            return _prepare_outputs_model_set(self, outputs, format_info)

    @deprecated('1.0',
                alternative='Use Model operators (TODO: link to compound '
                            'model docs')
    def add_model(self, model, mode):
        """
        Create a CompositeModel by chaining the current model with the new one
        using the specified mode.

        Parameters
        ----------
        model : an instance of a subclass of Model
        mode :  string
               'parallel', 'serial', 'p' or 's'
               a flag indicating whether to combine the models
               in series or in parallel

        Returns
        -------
        model : CompositeModel
            an instance of CompositeModel
        """

        from ._compound_deprecated import (SummedCompositeModel,
                                           SerialCompositeModel)

        if mode in ['parallel', 'p']:
            return SummedCompositeModel([self, model])
        elif mode in ['serial', 's']:
            return SerialCompositeModel([self, model])
        else:
            raise InputParameterError("Unrecognized mode {0}".format(mode))

    def copy(self):
        """
        Return a copy of this model.

        Uses a deep copy so that all model attributes, including parameter
        values, are copied as well.
        """

        return copy.deepcopy(self)

    @sharedmethod
    def rename(self, name):
        """
        Return a copy of this model with a new name.
        """

        new_model = self.copy()
        new_model._name = name
        return new_model

    # *** Internal methods ***

    def _initialize_constraints(self, kwargs):
        """
        Pop parameter constraint values off the keyword arguments passed to
        `Model.__init__` and store them in private instance attributes.
        """

        self._constraints = {}
        # Pop any constraints off the keyword arguments
        for constraint in self.parameter_constraints:
            values = kwargs.pop(constraint, {})
            self._constraints[constraint] = values

            # Update with default parameter constraints
            for param_name in self.param_names:
                param = getattr(self, param_name)

                # Parameters don't have all constraint types
                value = getattr(param, constraint)
                if value is not None:
                    self._constraints[constraint][param_name] = value

        for constraint in self.model_constraints:
            values = kwargs.pop(constraint, [])
            self._constraints[constraint] = values

    def _initialize_parameters(self, args, kwargs):
        """
        Initialize the _parameters array that stores raw parameter values for
        all parameter sets for use with vectorized fitting algorithms; on
        FittableModels the _param_name attributes actually just reference
        slices of this array.
        """

        n_models = None
        # Pop off param_dim and handle backwards compatibility
        if 'param_dim' in kwargs:
            n_models = kwargs.pop('param_dim')
            warnings.warn(
                'The param_dim argument to {0}.__init__ is deprecated; '
                'use n_models instead.  See also the model_set_axis argument '
                'and related discussion in the docstring for Model.'.format(
                    self.__class__.__name__), AstropyDeprecationWarning)
            if 'n_models' in kwargs:
                raise TypeError(
                    "param_dim and n_models cannot both be specified; use "
                    "n_models, as param_dim is deprecated")
        else:
            n_models = kwargs.pop('n_models', None)

        if not (n_models is None or
                    (isinstance(n_models, int) and n_models >=1)):
            raise ValueError(
                "n_models must be either None (in which case it is "
                "determined from the model_set_axis of the parameter initial "
                "values) or it must be a positive integer "
                "(got {0!r})".format(n_models))

        model_set_axis = kwargs.pop('model_set_axis', None)
        if model_set_axis is None:
            if n_models is not None and n_models > 1:
                # Default to zero
                model_set_axis = 0
            else:
                # Otherwise disable
                model_set_axis = False
        else:
            if not (model_set_axis is False or
                    (isinstance(model_set_axis, int) and
                        not isinstance(model_set_axis, bool))):
                raise ValueError(
                    "model_set_axis must be either False or an integer "
                    "specifying the parameter array axis to map to each "
                    "model in a set of models (got {0!r}).".format(
                        model_set_axis))

        # Process positional arguments by matching them up with the
        # corresponding parameters in self.param_names--if any also appear as
        # keyword arguments this presents a conflict
        params = {}
        if len(args) > len(self.param_names):
            raise TypeError(
                "{0}.__init__() takes at most {1} positional arguments ({2} "
                "given)".format(self.__class__.__name__, len(self.param_names),
                                len(args)))

        for idx, arg in enumerate(args):
            if arg is None:
                # A value of None implies using the default value, if exists
                continue
            params[self.param_names[idx]] = np.asanyarray(arg, dtype=np.float)

        # At this point the only remaining keyword arguments should be
        # parameter names; any others are in error.
        for param_name in self.param_names:
            if param_name in kwargs:
                if param_name in params:
                    raise TypeError(
                        "{0}.__init__() got multiple values for parameter "
                        "{1!r}".format(self.__class__.__name__, param_name))
                value = kwargs.pop(param_name)
                if value is None:
                    continue
                params[param_name] = np.asanyarray(value, dtype=np.float)

        if kwargs:
            # If any keyword arguments were left over at this point they are
            # invalid--the base class should only be passed the parameter
            # values, constraints, and param_dim
            for kwarg in kwargs:
                # Just raise an error on the first unrecognized argument
                raise TypeError(
                    '{0}.__init__() got an unrecognized parameter '
                    '{1!r}'.format(self.__class__.__name__, kwarg))

        # Determine the number of model sets: If the model_set_axis is
        # None then there is just one parameter set; otherwise it is determined
        # by the size of that axis on the first parameter--if the other
        # parameters don't have the right number of axes or the sizes of their
        # model_set_axis don't match an error is raised
        if model_set_axis is not False and n_models != 1 and params:
            max_ndim = 0
            if model_set_axis < 0:
                min_ndim = abs(model_set_axis)
            else:
                min_ndim = model_set_axis + 1

            for name, value in six.iteritems(params):
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

            self._param_broadcast_shapes = self._check_param_broadcast(
                    params, max_ndim, model_set_axis)
        else:
            if n_models is None:
                n_models = 1

            self._param_broadcast_shapes = self._check_param_broadcast(
                    params, None, None)

        # First we need to determine how much array space is needed by all the
        # parameters based on the number of parameters, the shape each input
        # parameter, and the param_dim
        self._n_models = n_models
        self._model_set_axis = model_set_axis

        self._initialize_parameter_values(params)

    def _initialize_parameter_values(self, params):
        self._param_metrics = {}
        total_size = 0

        for name in self.param_names:
            if params.get(name) is None:
                default = getattr(self, name).default

                if default is None:
                    # No value was supplied for the parameter, and the
                    # parameter does not have a default--therefor the model is
                    # underspecified
                    raise TypeError(
                        "{0}.__init__() requires a value for parameter "
                        "{1!r}".format(self.__class__.__name__, name))

                value = params[name] = default
            else:
                value = params[name]

            param_size = np.size(value)
            param_shape = np.shape(value)

            param_slice = slice(total_size, total_size + param_size)
            self._param_metrics[name] = (param_slice, param_shape)
            total_size += param_size

        self._parameters = np.empty(total_size, dtype=np.float64)
        # Now set the parameter values (this will also fill
        # self._parameters)
        for name, value in params.items():
            setattr(self, name, value)

    def _check_param_broadcast(self, params, max_ndim, model_set_axis):
        """
        This subroutine checks that all parameter arrays can be broadcast
        against each other, and determimes the shapes parameters must have in
        order to broadcast correctly.

        If model_set_axis is None this merely checks that the parameters
        broadcast and returns an empty dict if so.  This mode is only used for
        single model sets.
        """

        broadcast_shapes = {}
        all_shapes = []
        param_names = []

        for name in self.param_names:
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

                if model_set_axis < 0:
                    # Just need to prepend axes to make up the difference
                    broadcast_shape = new_axes + param_shape
                else:
                    broadcast_shape = (param_shape[:model_set_axis + 1] +
                                       new_axes +
                                       param_shape[model_set_axis + 1:])
                broadcast_shapes[name] = broadcast_shape
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

        return broadcast_shapes

    def _format_repr(self, args=[], kwargs={}, defaults={}):
        """
        Internal implementation of ``__repr__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__repr__`` while keeping the same basic
        formatting.
        """

        # TODO: I think this could be reworked to preset model sets better

        parts = [repr(a) for a in args]

        parts.extend(
            "{0}={1}".format(name,
                             array_repr_oneline(getattr(self, name).value))
            for name in self.param_names)

        if self.name is not None:
            parts.append('name={0!r}'.format(self.name))

        for kwarg, value in kwargs.items():
            if kwarg in defaults and defaults[kwarg] != value:
                continue
            parts.append('{0}={1!r}'.format(kwarg, value))

        if len(self) > 1:
            parts.append("n_models={0}".format(len(self)))

        return '<{0}({1})>'.format(self.__class__.__name__, ', '.join(parts))

    def _format_str(self, keywords=[]):
        """
        Internal implementation of ``__str__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__str__`` while keeping the same basic
        formatting.
        """

        default_keywords = [
            ('Model', self.__class__.__name__),
            ('Name', self.name),
            ('Inputs', self.inputs),
            ('Outputs', self.outputs),
            ('Model set size', len(self))
        ]

        parts = ['{0}: {1}'.format(keyword, value)
                 for keyword, value in default_keywords + keywords
                 if value is not None]

        parts.append('Parameters:')

        if len(self) == 1:
            columns = [[getattr(self, name).value]
                       for name in self.param_names]
        else:
            columns = [getattr(self, name).value
                       for name in self.param_names]

        param_table = Table(columns, names=self.param_names)

        parts.append(indent(str(param_table), width=4))

        return '\n'.join(parts)


class FittableModel(Model):
    """
    Base class for models that can be fitted using the built-in fitting
    algorithms.
    """

    linear = False
    # derivative with respect to parameters
    fit_deriv = None
    """
    Function (similar to the model's ``eval``) to compute the derivatives of
    the model with respect to its parameters, for use by fitting algorithms.
    """
    # Flag that indicates if the model derivatives with respect to parameters
    # are given in columns or rows
    col_fit_deriv = True
    fittable = True


class Fittable1DModel(FittableModel):
    """
    Base class for one-dimensional fittable models.

    This class provides an easier interface to defining new models.
    Examples can be found in `astropy.modeling.functional_models`.
    """

    inputs = ('x',)
    outputs = ('y',)


class Fittable2DModel(FittableModel):
    """
    Base class for one-dimensional fittable models.

    This class provides an easier interface to defining new models.
    Examples can be found in `astropy.modeling.functional_models`.
    """

    inputs = ('x', 'y')
    outputs = ('z',)


class Mapping(Model):
    def __init__(self, mapping):
        self._inputs = tuple('x' + str(idx)
                             for idx in range(max(mapping) + 1))
        self._outputs = tuple('x' + str(idx) for idx in range(len(mapping)))
        self._mapping = mapping
        super(Mapping, self).__init__()

    def __call__(self, *args):
        return self.evaluate(*args)

    @property
    def name(self):
        return 'Mapping({0})'.format(self.mapping)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def mapping(self):
        return self._mapping

    def evaluate(self, *args):
        if len(args) < self.n_inputs:
            raise TypeError('{0} expects at most {1} inputs; got {2}'.format(
                self.name, self.n_inputs, len(args)))

        result = tuple(args[idx] for idx in self._mapping)

        if self.n_outputs == 1:
            return result[0]

        return result


class Identity(Mapping):
    def __init__(self, n_inputs):
        mapping = tuple(range(n_inputs))
        super(Identity, self).__init__(mapping)

    @property
    def name(self):
        return 'Identity({0})'.format(self.n_inputs)


def _make_compound_model(left, right, operator):
    name = str('CompoundModel{0}'.format(_CompoundModelMeta._nextid))
    _CompoundModelMeta._nextid += 1

    children = []
    for child in (left, right):
        if isinstance(child, _CompoundModel):
            children.append(child._tree)
        elif isinstance(child, _CompoundModelMeta):
            children.append(child._tree)
        else:
            children.append(ExpressionTree(child))

    tree = ExpressionTree(operator, left=children[0], right=children[1])

    mod = find_current_module(3)
    if mod:
        modname = mod.__name__
    else:
        modname = '__main__'

    # TODO: These aren't the full rules for handling inputs and outputs, but
    # this will handle most basic cases correctly
    if operator == '|':
        inputs = left.inputs
        outputs = right.outputs
    elif operator == '&':
        inputs = combine_labels(left.inputs, right.inputs)
        outputs = combine_labels(left.outputs, right.outputs)
    else:
        # Without loss of generality
        inputs = left.inputs
        outputs = left.outputs


    members = {'_tree': tree,
               # TODO: These are temporary until we implement the full rules
               # for handling inputs/outputs
               'inputs': inputs,
               'outputs': outputs,
               '__module__': str(modname)}

    new_cls = _CompoundModelMeta(name, (_CompoundModel,), members)

    if isinstance(left, Model) and isinstance(right, Model):
        # Both models used in the operator were already instantiated models,
        # not model *classes*.  As such it's not particularly useful to return
        # the class itself, but to instead produce a new instance:
        return new_cls()

    # Otherwise return the new uninstantiated class itself
    return new_cls


def _make_arithmetic_operator(oper):
    def op(f, g):
        f, f_in, f_out = f
        g = g[0]

        return (make_binary_operator_eval(oper, f, g), f_in, f_out)

    return op


def _composition_operator(f, g):
    f, f_in, _ = f
    g, _, g_out = g
    return (lambda *args: g(*f(*args)), f_in, g_out)


def _join_operator(f, g):
    f, f_in, f_out = f
    g, g_in, g_out = g

    h = lambda *args: f(*args[:f_in]) + g(*args[f_in:])

    return (h, f_in + g_in, f_out + g_out)


# TODO: Support a couple unary operators--at least negation?
BINARY_OPERATORS = {
    '+': _make_arithmetic_operator(operator.add),
    '-': _make_arithmetic_operator(operator.sub),
    '*': _make_arithmetic_operator(operator.mul),
    '/': _make_arithmetic_operator(operator.truediv),
    '**': _make_arithmetic_operator(operator.pow),
    '|': _composition_operator,
    '&': _join_operator
}


_ORDER_OF_OPERATORS = [('|',), ('&',), ('+', '-'), ('*', '/'), ('**',)]
OPERATOR_PRECEDENCE = {}
for idx, ops in enumerate(_ORDER_OF_OPERATORS):
    for op in ops:
        OPERATOR_PRECEDENCE[op] = idx


class _CompoundModelMeta(_ModelMeta):
    _tree = None
    _submodels = None
    _submodel_names = None
    _nextid = 0

    _param_names = None
    _param_map = None

    def __getitem__(cls, index):
        if isinstance(index, int):
            return cls._get_submodels()[index]
        elif isinstance(index, str):
            return cls._get_submodels()[cls.submodel_names.index(index)]

        raise TypeError(
            'Submodels can be indexed either by their integer order or '
            'their name (got {0!r}).'.format(index))

    def __getattr__(cls, attr):
        if attr in cls.param_names:
            cls._init_param_descriptors()
            return getattr(cls, attr)

        raise AttributeError(attr)

    def __repr__(cls):
        if cls._tree is None:
            # This case is mostly for debugging purposes
            return cls._format_cls_repr()

        expression = cls._format_expression()
        components = '\n\n'.join('[{0}]: {1!r}'.format(idx, m)
                                 for idx, m in enumerate(cls._get_submodels()))
        keywords = [
            ('Expression', expression),
            ('Components', '\n' + indent(components))
        ]

        return cls._format_cls_repr(keywords=keywords)

    def __dir__(cls):
        basedir = super(_CompoundModelMeta, cls).__dir__()
        if cls._tree is not None:
            for name in cls.param_names:
                basedir.append(name)

            basedir.sort()

        return basedir

    @property
    def submodel_names(cls):
        if cls._submodel_names is not None:
            return cls._submodel_names

        by_name = defaultdict(lambda: [])

        for idx, submodel in enumerate(cls._get_submodels()):
            # Keep track of the original sort order of the submodels
            by_name[submodel.name].append(idx)

        names = []
        for basename, indices in six.iteritems(by_name):
            if len(indices) == 1:
                # There is only one model with this name, so it doesn't need an
                # index appended to its name
                names.append((basename, indices[0]))
            else:
                for idx in indices:
                    names.append(('{0}_{1}'.format(basename, idx), idx))

        # Sort according to the models' original sort orders
        names.sort(key=lambda k: k[1])

        names = tuple(k[0] for k in names)

        cls._submodels_names = names
        return names

    @property
    def param_names(cls):
        if cls._param_names is None:
            cls._init_param_names()

        return cls._param_names

    # TODO: Perhaps, just perhaps, the post-order (or ???-order) ordering of
    # leaf nodes is something the ExpressionTree class itself could just know
    def _get_submodels(cls):
        # Would make this a lazyproperty but those don't currently work with
        # type objects
        if cls._submodels is not None:
            return cls._submodels

        submodels = [c.value for c in cls._tree.traverse_postorder()
                     if c.isleaf]
        cls._submodels = submodels
        return submodels

    def _init_param_descriptors(cls):
        """
        This routine sets up the names for all the parameters on a compound
        model, including figuring out unique names for those parameters and
        also mapping them back to their associated parameters of the underlying
        submodels.

        Setting this all up is costly, and only necessary for compound models
        that a user will directly interact with.  For example when building an
        expression like::

            >>> M = (Model1 + Model2) * Model3  # doctest: +SKIP

        the user will generally never interact directly with the temporary
        result of the subexpression ``(Model1 + Model2)``.  So there's no need
        to setup all the parameters for that temporary throwaway.  Only once
        the full expression is built and the user initializes or introspects
        ``M`` is it necessary to determine its full parameterization.
        """

        # Accessing cls.param_names will implicitly call _init_param_names if
        # needed and thus also set up the _param_map; I'm not crazy about that
        # design but it stands for now
        for param_name in cls.param_names:
            submodel_idx, submodel_param = cls._param_map[param_name]
            submodel = cls[submodel_idx]

            orig_param = getattr(submodel, submodel_param)

            # Note: Parameter.copy() returns a new unbound Parameter, never a
            # bound Parameter even if submodel is a Model instance (as opposed
            # to a Model subclass)
            new_param = orig_param.copy(name=param_name)
            setattr(cls, param_name, new_param)

    def _init_param_names(cls):
        """
        This subroutine is solely for setting up the ``param_names`` attribute
        itself.

        See ``_init_param_descriptors`` for the full parameter setup.
        """

        # Currently this skips over Model *instances* in the expression tree;
        # basically these are treated as constants and do not add
        # fittable/tunable parameters to the compound model.
        # TODO: I'm not 100% happy with this design, and maybe we need some
        # interface for distinguishing fittable/settable parameters with
        # *constant* parameters (which would be distinct from parameters with
        # fixed constraints since they're permanently locked in place). But I'm
        # not sure if this is really the best way to treat the issue.

        names = []
        param_map = {}

        for idx, model in enumerate(model for model in cls._get_submodels()
                                    if isinstance(model, type) and
                                    model.param_names):
            for param_name in model.param_names:
                name = '{0}_{1}'.format(param_name, idx)
                names.append(name)
                param_map[name] = (idx, param_name)

        cls._param_names = tuple(names)
        cls._param_map = param_map

    def _format_expression(cls):
        # TODO: At some point might be useful to make a public version of this,
        # albeit with more formatting options
        return cls._tree.format_expression(OPERATOR_PRECEDENCE)


@six.add_metaclass(_CompoundModelMeta)
class _CompoundModel(Model):
    _submodels = None

    def __getattr__(self, attr):
        return getattr(self.__class__, attr)

    @property
    def submodel_names(self):
        return self.__class__.submodel_names

    @property
    def param_names(self):
        return self.__class__.param_names

    def evaluate(self, *args):
        inputs = args[:self.n_inputs]
        params = iter(args[self.n_inputs:])

        def getter(model, params=params):
            # Returns partial evaluation of the model's evaluate() method witht
            # the parameter values already applied
            nparams = len(model.param_names)

            if isinstance(model, Model):
                # Use the fixed model instance's parameters in the evaluation
                param_values = tuple(model.param_sets)
            else:
                param_values = tuple(itertools.islice(params, nparams))

                # There is currently an unfortunate iconsistency in some
                # models, which requires them to be instantiated for their
                # evaluate to work.  I think that needs to be reconsidered and
                # fixed somehow, but in the meantime we need to check for that
                # case
                if isinstancemethod(model, model.evaluate):
                    # Where previously model was a class, now make an instance
                    model = model(*param_values)

            if model.n_outputs == 1:
                f = lambda *inputs: \
                        (model.evaluate(*(inputs + param_values)),)
            else:
                f = lambda *inputs: \
                        model.evaluate(*(inputs + param_values))

            return (f, model.n_inputs, model.n_outputs)

        func = self._tree.evaluate(BINARY_OPERATORS, getter=getter)[0]
        result = func(*inputs)

        if self.n_outputs == 1:
            return result[0]
        else:
            return result

    _get_submodels = sharedmethod(_CompoundModelMeta._get_submodels)


def custom_model(*args, **kwargs):
    """
    Create a model from a user defined function. The inputs and parameters of
    the model will be inferred from the arguments of the function.

    This can be used either as a function or as a decorator.  See below for
    examples of both usages.

    .. note::

        All model parameters have to be defined as keyword arguments with
        default values in the model function.  Use `None` as a default argument
        value if you do not want to have a default value for that parameter.

    Parameters
    ----------
    func : function
        Function which defines the model.  It should take N positional
        arguments where ``N`` is dimensions of the model (the number of
        independent variable in the model), and any number of keyword arguments
        (the parameters).  It must return the value of the model (typically as
        an array, but can also be a scalar for scalar inputs).  This
        corresponds to the `~astropy.modeling.Model.evaluate` method.
    fit_deriv : function, optional
        Function which defines the Jacobian derivative of the model. I.e., the
        derivive with respect to the *parameters* of the model.  It should
        have the same argument signature as ``func``, but should return a
        sequence where each element of the sequence is the derivative
        with respect to the correseponding argument. This corresponds to the
        :meth:`~astropy.modeling.FittableModel.fit_deriv` method.

    Examples
    --------
    Define a sinusoidal model function as a custom 1D model::

        >>> from astropy.modeling.models import custom_model
        >>> import numpy as np
        >>> def sine_model(x, amplitude=1., frequency=1.):
        ...     return amplitude * np.sin(2 * np.pi * frequency * x)
        >>> def sine_deriv(x, amplitude=1., frequency=1.):
        ...     return 2 * np.pi * amplitude * np.cos(2 * np.pi * frequency * x)
        >>> SineModel = custom_model(sine_model, fit_deriv=sine_deriv)

    Create an instance of the custom model and evaluate it::

        >>> model = SineModel()
        >>> model(0.25)
        1.0

    This model instance can now be used like a usual astropy model.

    The next example demonstrates a 2D beta function model, and also
    demonstrates the support for docstrings (this example could also include
    a derivative, but it has been ommitted for simplicity)::

        >>> @custom_model
        ... def Beta2D(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, gamma=1.0,
        ...            alpha=1.0):
        ...     \"\"\"Two dimensional beta function.\"\"\"
        ...     rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2
        ...     return amplitude * (1 + rr_gg) ** (-alpha)
        ...
        >>> print(Beta2D.__doc__)
        Two dimensional beta function.
        >>> model = Beta2D()
        >>> model(1, 1)  # doctest: +FLOAT_CMP
        0.3333333333333333
    """

    fit_deriv = kwargs.get('fit_deriv', None)

    if len(args) == 1 and six.callable(args[0]):
        return _custom_model_wrapper(args[0], fit_deriv=fit_deriv)
    elif not args:
        return functools.partial(_custom_model_wrapper, fit_deriv=fit_deriv)
    else:
        raise TypeError(
            "{0} takes at most one positional argument (the callable/"
            "function to be turned into a model.  When used as a decorator "
            "it should be passed keyword arguments only (if "
            "any).".format(__name__))


def _custom_model_wrapper(func, fit_deriv=None):
    """
    Internal implementation `custom_model`.

    When `custom_model` is called as a function its arguments are passed to
    this function, and the result of this function is returned.

    When `custom_model` is used as a decorator a partial evaluation of this
    function is returned by `custom_model`.
    """

    if not six.callable(func):
        raise ModelDefinitionError(
            "func is not callable; it must be a function or other callable "
            "object")

    if fit_deriv is not None and not six.callable(fit_deriv):
        raise ModelDefinitionError(
            "fit_deriv not callable; it must be a function or other "
            "callable object")

    model_name = func.__name__
    argspec = inspect.getargspec(func)
    param_values = argspec.defaults or ()

    nparams = len(param_values)
    param_names = argspec.args[-nparams:]

    if (fit_deriv is not None and
            len(six.get_function_defaults(fit_deriv)) != nparams):
        raise ModelDefinitionError("derivative function should accept "
                                   "same number of parameters as func.")

    if nparams:
        input_names = argspec.args[:-nparams]
    else:
        input_names = argspec.args

    # TODO: Maybe have a clever scheme for default output name?
    if input_names:
        output_names = (input_names[0],)
    else:
        output_names = ('x',)

    params = dict((name, Parameter(name, default=default))
                  for name, default in zip(param_names, param_values))

    mod = find_current_module(2)
    if mod:
        modname = mod.__name__
    else:
        modname = '__main__'

    members = {
        '__module__': str(modname),
        '__doc__': func.__doc__,
        'inputs': tuple(input_names),
        'outputs': output_names,
        'evaluate': staticmethod(func),
    }

    if fit_deriv is not None:
        members['fit_deriv'] = staticmethod(fit_deriv)

    members.update(params)

    return type(model_name, (FittableModel,), members)


def _prepare_inputs_single_model(model, params, inputs, **kwargs):
    broadcasts = []

    for idx, _input in enumerate(inputs):
        _input = np.asanyarray(_input, dtype=np.float)
        input_shape = _input.shape
        max_broadcast = ()

        for param in params:
            try:
                if model.standard_broadcasting:
                    broadcast = check_broadcast(input_shape, param.shape)
                else:
                    broadcast = input_shape
            except IncompatibleShapeError:
                raise ValueError(
                    "Model input argument {0!r} of shape {1!r} cannot be "
                    "broadcast with parameter {2!r} of shape "
                    "{3!r}.".format(model.inputs[idx], input_shape,
                                    param.name, param.shape))

            if len(broadcast) > len(max_broadcast):
                max_broadcast = broadcast
            elif len(broadcast) == len(max_broadcast):
                max_broadcast = max(max_broadcast, broadcast)

        broadcasts.append(max_broadcast)

    return inputs, (broadcasts,)


def _prepare_outputs_single_model(model, outputs, format_info):
    if not format_info:
        # This is the shortcut for models with all scalar inputs/parameters
        if model.n_outputs == 1:
            return np.asscalar(outputs[0])
        else:
            return tuple(np.asscalar(output) for output in outputs)

    broadcasts = format_info[0]

    outputs = list(outputs)

    for idx, output in enumerate(outputs):
        if broadcasts[idx]:
            outputs[idx] = output.reshape(broadcasts[idx])

    if model.n_outputs == 1:
        return outputs[0]
    else:
        return tuple(outputs)


def _prepare_inputs_model_set(model, params, inputs, n_models, model_set_axis,
                              **kwargs):
    reshaped = []
    pivots = []

    for idx, _input in enumerate(inputs):
        _input = np.asanyarray(_input, dtype=np.float)

        max_param_shape = ()

        if n_models > 1 and model_set_axis is not False:
            # Use the shape of the input *excluding* the model axis
            input_shape = (_input.shape[:model_set_axis] +
                           _input.shape[model_set_axis + 1:])
        else:
            input_shape = _input.shape

        for param in params:
            try:
                check_broadcast(input_shape, param.shape)
            except IncompatibleShapeError:
                raise ValueError(
                    "Model input argument {0!r} of shape {1!r} cannot be "
                    "broadcast with parameter {2!r} of shape "
                    "{3!r}.".format(model.inputs[idx], input_shape,
                                    param.name, param.shape))

            if len(param.shape) > len(max_param_shape):
                max_param_shape = param.shape

        # We've now determined that, excluding the model_set_axis, the
        # input can broadcast with all the parameters
        input_ndim = len(input_shape)
        if model_set_axis is False:
            if len(max_param_shape) > input_ndim:
                # Just needs to prepend new axes to the input
                n_new_axes = 1 + len(max_param_shape) - input_ndim
                new_axes = (1,) * n_new_axes
                new_shape = new_axes + _input.shape
                pivot = model.model_set_axis
            else:
                pivot = input_ndim - len(max_param_shape)
                new_shape = (_input.shape[:pivot] + (1,) +
                             _input.shape[pivot:])
            new_input = _input.reshape(new_shape)
        else:
            if len(max_param_shape) >= input_ndim:
                n_new_axes = len(max_param_shape) - input_ndim
                pivot = model.model_set_axis
                new_axes = (1,) * n_new_axes
                new_shape = (_input.shape[:pivot + 1] + new_axes +
                             _input.shape[pivot + 1:])
                new_input = _input.reshape(new_shape)
            else:
                pivot = _input.ndim - len(max_param_shape) - 1
                new_input = np.rollaxis(_input, model_set_axis,
                                        pivot + 1)

        pivots.append(pivot)
        reshaped.append(new_input)

    return reshaped, (pivots,)


def _prepare_outputs_model_set(model, outputs, format_info):
    pivots = format_info[0]

    outputs = list(outputs)

    for idx, output in enumerate(outputs):
        pivot = pivots[idx]
        if pivot < output.ndim and pivot != model.model_set_axis:
            outputs[idx] = np.rollaxis(output, pivot,
                                       model.model_set_axis)

    if model.n_outputs == 1:
        return outputs[0]
    else:
        return tuple(outputs)


def _validate_input_shapes(inputs, argnames, n_models, model_set_axis,
                           validate_broadcasting):
    """
    Perform basic validation of model inputs--that they are mutually
    broadcastable and that they have the minimum dimensions for the given
    model_set_axis.

    If validation succeeds, returns the total shape that will result from
    broadcasting the input arrays with each other.
    """

    check_model_set_axis = n_models > 1 and model_set_axis is not False

    if not (validate_broadcasting or check_model_set_axis):
        # Nothing else needed here
        return

    all_shapes = []

    for idx, _input in enumerate(inputs):
        input_shape = np.shape(_input)
        # Ensure that the input's model_set_axis matches the model's
        # n_models
        if input_shape and check_model_set_axis:
            # Note: Scalar inputs *only* get a pass on this
            if len(input_shape) < model_set_axis + 1:
                raise ValueError(
                    "For model_set_axis={0}, all inputs must be at "
                    "least {1}-dimensional.".format(
                        model_set_axis, model_set_axis + 1))
            elif input_shape[model_set_axis] != n_models:
                raise ValueError(
                    "Input argument {0!r} does not have the correct "
                    "dimensions in model_set_axis={1} for a model set with "
                    "n_models={2}.".format(argnames[idx], model_set_axis,
                                           n_models))
        all_shapes.append(input_shape)

    if not validate_broadcasting:
        return

    try:
        input_broadcast = check_broadcast(*all_shapes)
    except IncompatibleShapesError as exc:
        shape_a, shape_a_idx, shape_b, shape_b_idx = exc.args
        arg_a = argnames[shape_a_idx]
        arg_b = argnames[shape_b_idx]

        raise ValueError(
            "Model input argument {0!r} of shape {1!r} cannot "
            "be broadcast with input {2!r} of shape {3!r}".format(
                arg_a, shape_a, arg_b, shape_b))

    return input_broadcast
