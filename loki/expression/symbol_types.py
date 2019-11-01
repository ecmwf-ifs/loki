import pymbolic.primitives as pmbl
from pymbolic.mapper.stringifier import (StringifyMapper, PREC_NONE, PREC_CALL)
from six.moves import intern

from loki.tools import as_tuple


__all__ = ['Scalar', 'Array', 'Variable', 'Literal', 'IntLiteral', 'FloatLiteral', 'LogicLiteral',
           'LiteralList', 'RangeIndex', 'InlineCall', 'LokiStringifyMapper', 'Cast']


class LokiStringifyMapper(StringifyMapper):
    """
    A class derived from the default :class:`StringifyMapper` that adds mappings for nodes of the
    expression tree that we added ourselves.
    """

    def __init__(self, constant_mapper=repr):
        super(LokiStringifyMapper, self).__init__(constant_mapper)

    def map_logic_literal(self, expr, *args, **kwargs):
        return str(expr.value)

    map_float_literal = map_logic_literal
    map_int_literal = map_logic_literal

    def map_string_literal(self, expr, *args, **kwargs):
        return "'%s'" % expr.value

    def map_scalar(self, expr, *args, **kwargs):
        parent = ''
        if expr.parent:
            parent = self.rec(expr.parent, *args, **kwargs) + '%'
        return parent + expr.name

    def map_array(self, expr, enclosing_prec, *args, **kwargs):
        dims = ''
        if expr.dimensions:
            dims = ','.join(self.rec(d, PREC_NONE, *args, **kwargs) for d in expr.dimensions)
            dims = '(' + dims + ')'
        parent = ''
        if expr.parent:
            parent = self.rec(expr.parent, PREC_NONE, *args, **kwargs) + '%'
        initial = ''
        if expr.initial:
            ' = %s' % self.rec(expr.initial, PREC_NONE, *args, **kwargs)
        return parent + expr.name + dims + initial

    map_inline_call = StringifyMapper.map_call_with_kwargs

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        name = self.rec(expr.function, PREC_CALL, *args, **kwargs)
        expression = self.rec(expr.parameters[0], PREC_NONE, *args, **kwargs)
        if expr.kind:
            if isinstance(expr.kind, pmbl.Expression):
                kind = ', kind=' + self.rec(expr.kind, PREC_NONE, *args, **kwargs)
            else:
                kind = ', kind=' + str(expr.kind)
        else:
            kind = ''
        return self.format('%s(%s%s)', name, expression, kind)

    def map_range_index(self, expr, *args, **kwargs):
        lower = self.rec(expr.lower, *args, **kwargs) if expr.lower else ''
        upper = self.rec(expr.upper, *args, **kwargs) if expr.upper else ''
        if expr.step:
            return '%s:%s:%s' % (lower, upper, self.rec(expr.step, *args, **kwargs))
        else:
            return '%s:%s' % (lower, upper)

    def map_parenthesised_add(self, *args, **kwargs):
        return self.parenthesize(self.map_sum(*args, **kwargs))

    def map_parenthesised_mul(self, *args, **kwargs):
        return self.parenthesize(self.map_product(*args, **kwargs))

    def map_parenthesised_pow(self, *args, **kwargs):
        return self.parenthesize(self.map_power(*args, **kwargs))

    def map_literal_list(self, expr, *args, **kwargs):
        return '[' + ','.join(str(c) for c in expr.elements) + ']'


class Scalar(pmbl.Variable):
    """
    Expression node for scalar variables (and other algebraic leaves).

    It can have an associated type or a parent (when member of a struct/derived type).
    """

    def __init__(self, name, type=None, parent=None, initial=None, _source=None):
        super(Scalar, self).__init__(name)

        self._type = type
        self.parent = parent
        self.initial = initial
        self._source = _source

    def __getinitargs__(self):
        args = [self.name]
        if self._type:
            args += [('type', self._type)]
        if self.parent:
            args += [('parent', self.parent)]
        if self.initial:
            args += [('initial', self.initial)]
        return tuple(args)

    mapper_method = intern('map_scalar')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    def clone(self, **kwargs):
        """
        Replicate the :class:`Scalar` variable with the provided overrides.
        """
        # Add existing meta-info to the clone arguments, only if we have them.
        if self.name and 'name' not in kwargs:
            kwargs['name'] = self.name
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent
        if self.type and 'type' not in kwargs:
            kwargs['type'] = self.type
        if self.initial and 'initial' not in kwargs:
            kwargs['initial'] = self.initial

        return Variable(**kwargs)

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        return self._type


class Array(pmbl.Variable):
    """
    Expression node for array variables.

    It usually has an associated shape (i.e., the size of the array) but can be instantiated
    without (e.g., when shape is later deferred from an allocate statement) and possibly some
    dimensions (which can be a :class:`RangeIndex` or an expression or a :class:`Literal` or
    a :class:`Scalar`).
    """

    def __init__(self, name, type=None, shape=None, dimensions=None, parent=None, initial=None, _source=None):
        super(Array, self).__init__(name)

        self._type = type
        self.parent = parent
        self.initial = initial
        self._source = _source
        self._shape = shape
        self.args = dimensions

    def __getinitargs__(self):
        args = [self.name]
        if self._type:
            args += [('type', self._type)]
        if self.shape:
            args += [('shape', self.shape)]
        if self.args:
            args += [('dimensions', self.args)]
        if self.parent:
            args += [('parent', self.parent)]
        if self.initial:
            args += [('initial', self.initial)]
        return tuple(args)

    mapper_method = intern('map_array')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    def clone(self, **kwargs):
        """
        Replicate the :class:`Array` variable with the provided overrides.

        Note, if :param dimensions: is provided as ``None``, a
        :class:`Scalar` variable will be created.
        """
        # Add existing meta-info to the clone arguments, only if we have them.
        if self.name and 'name' not in kwargs:
            kwargs['name'] = self.name
        if self.dimensions and 'dimensions' not in kwargs:
            kwargs['dimensions'] = self.dimensions
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent
        if self.type and 'type' not in kwargs:
            kwargs['type'] = self.type
        if self.shape and 'shape' not in kwargs:
            kwargs['shape'] = self.shape
        if self.initial and 'initial' not in kwargs:
            kwargs['initial'] = self.initial

        return Variable(**kwargs)

    @property
    def dimensions(self):
        """
        Symbolic representation of the dimensions or indices.
        """
        return self.args

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        return self._type

    @property
    def shape(self):
        """
        Original allocated shape of the variable as a tuple of dimensions.
        """
        return self._shape


class Variable:
    """
    A symbolic object representing either a :class:`Scalar` or a :class:`Array`
    variable in arithmetic expressions.

    Note, that this is only a convenience constructor that always returns either
    a :class:`Scalar` or :class:`Array` variable object.
    """

    def __new__(cls, *args, **kwargs):
        """
        1st-level variables creation with name injection via the object class
        """
        name = kwargs.pop('name')
        dimensions = kwargs.pop('dimensions', None)
        parent = kwargs.pop('parent', None)
        _type = kwargs.get('type', None)
        shape = kwargs.get('shape', None)
        initial = kwargs.get('initial', None)
        source = kwargs.get('source', None)

        if dimensions is None and (shape is None or len(shape) == 0):
            return Scalar(name=name, parent=parent, type=_type, initial=initial, _source=source)
        else:
            return Array(name=name, shape=shape, dimensions=dimensions, parent=parent, type=_type,
                         initial=initial, _source=source)


class FloatLiteral(pmbl.Leaf):
    """
    A floating point constant in an expression.

    It can have a specific type associated, which can be used to cast the constant to that
    type in the output of the backend.
    """

    def __init__(self, value, **kwargs):
        super(FloatLiteral, self).__init__()

        self.value = value
        self._type = kwargs.get('type', None)
        self._kind = kwargs.get('kind', None)

    def __getinitargs__(self):
        args = [self.value]
        if self._type:
            args += [('type', self._type)]
        if self._kind:
            args += [('kind', self._kind)]
        return tuple(args)

    mapper_method = intern('map_float_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class IntLiteral(pmbl.Leaf):
    """
    An integer constant in an expression.

    It can have a specific type associated, which can be used to cast the constant to that
    type in the output of the backend.
    """

    def __init__(self, value, **kwargs):
        super(IntLiteral, self).__init__()

        self.value = value
        self._type = kwargs.get('type', None)
        self._kind = kwargs.get('kind', None)

    def __getinitargs__(self):
        args = [self.value]
        if self._type:
            args += [('type', self._type)]
        if self._kind:
            args += [('kind', self._kind)]
        return tuple(args)

    mapper_method = intern('map_int_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class LogicLiteral(pmbl.Leaf):
    """
    A boolean constant in an expression.
    """

    def __init__(self, value, **kwargs):
        super(LogicLiteral, self).__init__()

        self.value = value.lower() in ('true', '.true.')

    def __getinitargs__(self):
        return (self.value,)

    mapper_method = intern('map_logic_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class StringLiteral(pmbl.Leaf):
    """
    A string.
    """

    def __init__(self, value, **kwargs):
        super(StringLiteral, self).__init__()

        if value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]

        self.value = value

    def __getinitargs__(self):
        return (self.value,)

    mapper_method = intern('map_string_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class Literal(object):
    """
    A factory class that instantiates the appropriate :class:`*Literal` type for
    a given value.

    This always returns a :class:`IntLiteral`, :class:`FloatLiteral`, :class:`StringLiteral`,
    or :class:`LogicLiteral`.
    """

    @classmethod
    def _from_literal(cls, value, **kwargs):
        if isinstance(value, int):
            obj = IntLiteral(value, **kwargs)
        elif isinstance(value, float):
            obj = FloatLiteral(value, **kwargs)
        elif isinstance(value, str) and len(value) >= 2 and value[0] == value[-1] \
                and value[0] in '"\'':
            obj = StringLiteral(value, **kwargs)
        elif str(value).lower() in ['.true.', 'true', '.false.', 'false']:
            # Ensure we capture booleans
            obj = LogicLiteral(value, **kwargs)
        else:
            raise TypeError('Unknown literal: %s' % value)

        return obj

    def __new__(cls, value, **kwargs):
        try:
            obj = cls._from_literal(value, **kwargs)
        except TypeError:
            # Let Pymbolic figure our what we're dealing with
            from pymbolic import parse
            obj = parse(value)

            # Make sure we catch elementary literals
            if not isinstance(obj, pmbl.Expression):
                obj = cls._from_literal(obj, **kwargs)

        # And attach our own meta-data
        if hasattr(obj, '_type'):
            obj._type = kwargs.get('type', None)
        if hasattr(obj, '_kind'):
            obj._kind = kwargs.get('kind', None)
        return obj


class LiteralList(pmbl.AlgebraicLeaf):
    """
    A list of constant literals, e.g., as used in Array Initialization Lists.
    """

    def __init__(self, values):
        super(LiteralList, self).__init__()

        self.elements = values

    mapper_method = intern('map_literal_list')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    def __getinitargs__(self):
        return ('[%s]' % (','.join(repr(c) for c in self.elements)),)


class InlineCall(pmbl.CallWithKwargs):
    """
    Internal representation of an in-line function call.
    """

    def __init__(self, function, parameters=None, kw_parameters=None):
        function = pmbl.make_variable(function)
        parameters = parameters or tuple()
        kw_parameters = kw_parameters or {}

        super(InlineCall, self).__init__(function, parameters, kw_parameters)

    mapper_method = intern('map_inline_call')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    @property
    def name(self):
        return self.function.name


class Cast(pmbl.Call):
    """
    Internal representation of a data type cast.
    """

    def __init__(self, name, expression, kind=None):
        super(Cast, self).__init__(pmbl.make_variable(name), as_tuple(expression))
        self.kind = kind

    mapper_method = intern('map_cast')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    @property
    def name(self):
        return self.function.name


class RangeIndex(pmbl.AlgebraicLeaf):
    """
    Internal representation of a subscript range.
    """

    @classmethod
    def _args2bounds(cls, *args, **kwargs):
        lower, upper, step = None, None, None
        if len(args) == 1:
            upper = args[0]
        elif len(args) == 2:
            lower = args[0]
            upper = args[1]
        elif len(args) == 3:
            lower = args[0]
            upper = args[1]
            step = args[2]

        lower = kwargs.get('lower', lower)
        upper = kwargs.get('upper', upper)
        step = kwargs.get('step', step)

        return lower, upper, step

    def __new__(cls, *args, **kwargs):
        lower, upper, step = RangeIndex._args2bounds(*args, **kwargs)

        # Short-circuit for direct indices
        if upper is not None and lower is None and step is None:
            return IntLiteral(upper)

        obj = object.__new__(cls)
        obj._lower = lower
        obj._upper = upper
        obj._step = step

        return obj

    def __init__(self, *args, **kwargs):
        super(RangeIndex, self).__init__()

        lower, upper, step = RangeIndex._args2bounds(*args, **kwargs)
        self._lower = lower
        self._upper = upper
        self._step = step

    def __getinitargs__(self):
        if self._step:
            return (self._lower, self._upper, self._step)
        elif self._lower:
            return (self._lower, self._upper)
        else:
            return (self._upper,)

    mapper_method = intern('map_range_index')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def step(self):
        return self._step
