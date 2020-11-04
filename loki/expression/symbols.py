import weakref
import pymbolic.primitives as pmbl
from six.moves import intern

from loki.tools import as_tuple
from loki.types import BasicType, DerivedType, SymbolType, Scope
from loki.expression.mappers import LokiStringifyMapper


__all__ = ['ExprMetadataMixin', 'Scalar', 'Array', 'Variable',
           'FloatLiteral', 'IntLiteral', 'LogicLiteral', 'StringLiteral', 'IntrinsicLiteral',
           'Literal', 'LiteralList',
           'Sum', 'Product', 'Quotient', 'Power', 'Comparison', 'LogicalAnd', 'LogicalOr',
           'LogicalNot', 'InlineCall', 'Cast', 'Range', 'LoopRange', 'RangeIndex', 'ArraySubscript',
           'ProcedureSymbol']


# pylint: disable=abstract-method


class ExprMetadataMixin:
    """
    Meta-data annotations for expression tree nodes.
    """

    def __init__(self, *args, **kwargs):
        self._metadata = {
            'source': kwargs.pop('source', None)
        }
        super().__init__(*args, **kwargs)

    def get_metadata(self):
        return self._metadata.copy()

    def update_metadata(self, data):
        self._metadata.update(data)

    @property
    def source(self):
        return self._metadata['source']

    @staticmethod
    def make_stringifier(originating_stringifier=None):  # pylint:disable=unused-argument
        return LokiStringifyMapper()


class StrCompareMixin:
    """
    String comparison overrides to reliably and flexibly identify
    expression symbols from equivalent strings.
    """

    def __hash__(self):
        return hash(super().__str__().lower().replace(' ', ''))

    def __eq__(self, other):
        if isinstance(other, str):
            # Do comparsion based on canonical string representations (lower-case, no spaces)
            sexpr = super().__str__().lower().replace(' ', '')
            other = other.lower().replace(' ', '')
            return sexpr == other

        return super().__eq__(other)

class TypedSymbol:
    """
    Base class for symbols that carry type information from an associated scope.


    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name')
        scope = kwargs.pop('scope')
        type = kwargs.pop('type', None)

        super().__init__(*args, **kwargs)

        assert isinstance(scope, Scope)
        self._scope = weakref.ref(scope)

        if type is None:
            # Insert the deferred type in the type table only if it does not exist
            # yet (necessary for deferred type definitions, e.g., derived types in header or
            # parameters from other modules)
            self.scope.symbols.setdefault(self.name, SymbolType(BasicType.DEFERRED))
        elif type is not self.scope.symbols.lookup(self.name):
            # If the type information does already exist and is identical (not just
            # equal) we don't update it. This makes sure that we don't create double
            # entries for variables inherited from a parent scope
            self.type = type.clone()

    @property
    def scope(self):
        """
        The object corresponding to the symbols scope.
        """
        return self._scope()

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        return self.scope.symbols.lookup(self.name)

    @type.setter
    def type(self, value):
        self.scope.symbols[self.name] = value


class Scalar(TypedSymbol, ExprMetadataMixin, StrCompareMixin, pmbl.Variable):
    """
    Expression node for scalar variables (and other algebraic leaves).

    It is always associated with a given class:``Scope`` (typically held by a
    class:``Subroutine``) where the corresponding symbol table is found with its type.

    Warning: Providing a type overwrites the corresponding entry in the symbol table.
    This is due to the assumption that we might have encountered a variable name before
    knowing about its declaration and thus treat the latest given type information as
    the one that is most up-to-date.
    """

    def __init__(self, name, scope, type=None, parent=None, **kwargs):
        # Stop complaints about `type` in this function
        # pylint: disable=redefined-builtin
        super().__init__(name=name, scope=scope, type=type, **kwargs)

        self.parent = parent

    @property
    def basename(self):
        """
        The symbol name without the qualifier from the parent.
        """
        idx = self.name.rfind('%')
        return self.name[idx+1:]

    @property
    def initial(self):
        """
        Initial value of the variable in declaration.
        """
        return self.type.initial

    @initial.setter
    def initial(self, value):
        self.type.initial = value

    def __getinitargs__(self):
        args = [('scope', self.scope)]
        if self.parent:
            args += [('parent', self.parent)]
        return super().__getinitargs__() + tuple(args)

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
        if self.scope and 'scope' not in kwargs:
            kwargs['scope'] = self.scope
        if self.type and 'type' not in kwargs:
            kwargs['type'] = self.type
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent

        return Variable(**kwargs)


class Array(TypedSymbol, ExprMetadataMixin, StrCompareMixin, pmbl.Variable):
    """
    Expression node for array variables.

    It can have associated dimensions (i.e., the indexing/slicing when accessing entries),
    which can be a :class:`RangeIndex` or an expression or a :class:`Literal` or
    a :class:`Scalar`

    Shape, data type and parent information are part of the type.

    Warning: Providing a type overwrites the corresponding entry in the symbol table.
    This is due to the assumption that we might have encountered a variable name before
    knowing about its declaration and thus treat the latest given type information as
    the one that is most up-to-date.
    """

    def __init__(self, name, scope, type=None, parent=None, dimensions=None, **kwargs):
        # Stop complaints about `type` in this function
        # pylint: disable=redefined-builtin
        super().__init__(name=name, scope=scope, type=type, **kwargs)

        self.parent = parent
        # Ensure dimensions are treated via ArraySubscript objects
        if dimensions is not None and not isinstance(dimensions, ArraySubscript):
            dimensions = ArraySubscript(dimensions)
        self.dimensions = dimensions

    @property
    def basename(self):
        """
        The symbol name without the qualifier from the parent.
        """
        idx = self.name.rfind('%')
        return self.name[idx+1:]

    @property
    def initial(self):
        """
        Initial value of the variable in declaration.
        """
        return self.type.initial

    @initial.setter
    def initial(self, value):
        self.type.initial = value

    @property
    def dimensions(self):
        """
        Symbolic representation of the dimensions or indices.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._dimensions = value

    @property
    def shape(self):
        """
        Original allocated shape of the variable as a tuple of dimensions.
        """
        return self.type.shape

    @shape.setter
    def shape(self, value):
        self.type.shape = value

    def __getinitargs__(self):
        args = [('scope', self.scope)]
        if self.dimensions:
            args += [('dimensions', self.dimensions)]
        if self.parent:
            args += [('parent', self.parent)]
        return super().__getinitargs__() + tuple(args)

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
        if self.scope and 'scope' not in kwargs:
            kwargs['scope'] = self.scope
        if self.dimensions and 'dimensions' not in kwargs:
            kwargs['dimensions'] = self.dimensions
        if self.type and 'type' not in kwargs:
            kwargs['type'] = self.type
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent

        return Variable(**kwargs)


class Variable:
    """
    A symbolic object representing either a :class:`Scalar` or a :class:`Array`
    variable in arithmetic expressions.

    Note, that this is only a convenience constructor that always returns either
    a :class:`Scalar` or :class:`Array` variable object.

    Warning: Providing a type overwrites the corresponding entry in the symbol table.
    This is due to the assumption that we might have encountered a variable name before
    knowing about its declaration and thus treat the latest given type information as
    the one that is most up-to-date.
    """

    def __new__(cls, **kwargs):
        """
        1st-level variables creation with name injection via the object class
        """
        name = kwargs['name']
        scope = kwargs['scope']
        _type = kwargs.setdefault('type', scope.symbols.lookup(name))

        dimensions = kwargs.pop('dimensions', None)
        shape = _type.shape if _type is not None else None

        if dimensions is None and not shape:
            obj = Scalar(**kwargs)
        else:
            obj = Array(dimensions=dimensions, **kwargs)

        obj = cls.instantiate_derived_type_variables(obj)
        return obj

    @classmethod
    def instantiate_derived_type_variables(cls, obj):
        """
        If the type of obj is a derived type then its list of variables is possibly from
        the declarations inside a TypeDef and as such, the variables are referring to a
        different scope. Thus, we must re-create these variables in the correct scope.
        For the actual instantiation of a variable with that type, we need to create a dedicated
        copy of that type and replace its parent by this object and its list of variables (which
        is an OrderedDict of SymbolTypes) by a list of Variable instances.
        """
        if obj.type is not None and isinstance(obj.type.dtype, DerivedType):
            if obj.type.dtype.typedef is not BasicType.DEFERRED:
                obj.type.dtype.variables = tuple(v.clone(name='%s%%%s' % (obj.name, v.basename),
                                                         type=v.type.clone(parent=obj),
                                                         scope=obj.scope)
                                                 for v in obj.type.dtype.typedef.variables)
        return obj


class _Literal(pmbl.Leaf):
    """
    Helper base class for literals to overcome the problem of a disfunctional
    __getinitargs__ in py:class:`pymbolic.primitives.Leaf`.
    """

    def __getinitargs__(self):
        return ()


class FloatLiteral(ExprMetadataMixin, _Literal):
    """
    A floating point constant in an expression.

    It can have a specific type associated, which can be used to cast the constant to that
    type in the output of the backend.
    """

    def __init__(self, value, **kwargs):
        # We store float literals as strings to make sure no information gets
        # lost in the conversion
        self.value = str(value)
        self.kind = kwargs.pop('kind', None)
        super().__init__(**kwargs)

    def __hash__(self):
        return hash((self.value, self.kind))

    def __eq__(self, other):
        if isinstance(other, FloatLiteral):
            return self.value == other.value and self.kind == other.kind

        try:
            return float(self.value) == float(other)
        except:
            return False

    def __getinitargs__(self):
        args = [self.value]
        if self.kind:
            args += [('kind', self.kind)]
        return tuple(args) + super().__getinitargs__()

    mapper_method = intern('map_float_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class IntLiteral(ExprMetadataMixin, _Literal):
    """
    An integer constant in an expression.

    It can have a specific type associated, which can be used to cast the constant to that
    type in the output of the backend.
    """

    def __init__(self, value, **kwargs):
        self.value = int(value)
        self.kind = kwargs.pop('kind', None)
        super().__init__(**kwargs)

    def __hash__(self):
        return hash((self.value, self.kind))

    def __eq__(self, other):
        if isinstance(other, IntLiteral):
            return self.value == other.value and self.kind == other.kind

        try:
            return self.value == int(other)
        except:
            return False

    def __getinitargs__(self):
        args = [self.value]
        if self.kind:
            args += [('kind', self.kind)]
        return tuple(args) + super().__getinitargs__()

    mapper_method = intern('map_int_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class LogicLiteral(ExprMetadataMixin, _Literal):
    """
    A boolean constant in an expression.
    """

    def __init__(self, value, **kwargs):
        self.value = value.lower() in ('true', '.true.')
        super().__init__(**kwargs)

    def __getinitargs__(self):
        return (self.value,) + super().__getinitargs__()

    mapper_method = intern('map_logic_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class StringLiteral(ExprMetadataMixin, _Literal):
    """
    A string.
    """

    def __init__(self, value, **kwargs):
        # Remove quotation marks
        if value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]

        self.value = value

        super().__init__(**kwargs)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, StringLiteral):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

    def __getinitargs__(self):
        return (self.value,) + super().__getinitargs__()

    mapper_method = intern('map_string_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class IntrinsicLiteral(ExprMetadataMixin, _Literal):
    """
    Any literal not represented by a dedicated class. Its value is stored as
    string and returned unaltered.

    This is currently used for complex and BOZ constants.
    """

    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def __getinitargs__(self):
        return (self.value,) + super().__getinitargs__()

    mapper_method = intern('map_intrinsic_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class Literal:
    """
    A factory class that instantiates the appropriate :class:`*Literal` type for
    a given value.

    This always returns a :class:`IntLiteral`, :class:`FloatLiteral`, :class:`StringLiteral`,
    :class:`LogicLiteral` or, as a fallback, :class:`IntrinsicLiteral`.
    """

    @staticmethod
    def _from_literal(value, **kwargs):

        cls_map = {BasicType.INTEGER: IntLiteral, BasicType.REAL: FloatLiteral,
                   BasicType.LOGICAL: LogicLiteral, BasicType.CHARACTER: StringLiteral}

        _type = kwargs.pop('type', None)
        if _type is None:
            if isinstance(value, int):
                _type = BasicType.INTEGER
            elif isinstance(value, float):
                _type = BasicType.REAL
            elif isinstance(value, str):
                if str(value).lower() in ('.true.', 'true', '.false.', 'false'):
                    _type = BasicType.LOGICAL
                else:
                    _type = BasicType.CHARACTER

        return cls_map[_type](value, **kwargs)

    def __new__(cls, value, **kwargs):
        try:
            obj = cls._from_literal(value, **kwargs)
        except KeyError:
            obj = IntrinsicLiteral(value, **kwargs)

        # And attach our own meta-data
        if hasattr(obj, 'kind'):
            obj.kind = kwargs.get('kind', None)
        return obj


class LiteralList(ExprMetadataMixin, pmbl.AlgebraicLeaf):
    """
    A list of constant literals, e.g., as used in Array Initialization Lists.
    """

    def __init__(self, values, **kwargs):
        self.elements = values
        super().__init__(**kwargs)

    mapper_method = intern('map_literal_list')

    def __getinitargs__(self):
        return ('[%s]' % (','.join(repr(c) for c in self.elements)),)


class Sum(ExprMetadataMixin, StrCompareMixin, pmbl.Sum):
    """Representation of a sum."""


class Product(ExprMetadataMixin, StrCompareMixin, pmbl.Product):
    """Representation of a product."""


class Quotient(ExprMetadataMixin, StrCompareMixin, pmbl.Quotient):
    """Representation of a quotient."""


class Power(ExprMetadataMixin, StrCompareMixin, pmbl.Power):
    """Representation of a power."""


class Comparison(ExprMetadataMixin, StrCompareMixin, pmbl.Comparison):
    """Representation of a comparison operation."""


class LogicalAnd(ExprMetadataMixin, StrCompareMixin, pmbl.LogicalAnd):
    """Representation of an 'and' in a logical expression."""


class LogicalOr(ExprMetadataMixin, StrCompareMixin, pmbl.LogicalOr):
    """Representation of an 'or' in a logical expression."""


class LogicalNot(ExprMetadataMixin, StrCompareMixin, pmbl.LogicalNot):
    """Representation of a negation in a logical expression."""


class InlineCall(ExprMetadataMixin, pmbl.CallWithKwargs):
    """
    Internal representation of an in-line function call.
    """

    def __init__(self, function, parameters=None, kw_parameters=None, **kwargs):
        assert isinstance(function, ProcedureSymbol)
        parameters = parameters or ()
        kw_parameters = kw_parameters or {}

        super().__init__(function=function, parameters=parameters,
                         kw_parameters=kw_parameters, **kwargs)

    mapper_method = intern('map_inline_call')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    @property
    def name(self):
        return self.function.name


class Cast(ExprMetadataMixin, pmbl.Call):
    """
    Internal representation of a data type cast.
    """

    def __init__(self, name, expression, kind=None, **kwargs):
        self.kind = kind
        super().__init__(pmbl.make_variable(name), as_tuple(expression), **kwargs)

    mapper_method = intern('map_cast')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    @property
    def name(self):
        return self.function.name


class Range(ExprMetadataMixin, StrCompareMixin, pmbl.Slice):
    """
    Internal representation of a loop or index range.
    """

    def __init__(self, children, **kwargs):
        assert len(children) in (2, 3)
        if len(children) == 2:
            children += (None,)
        super().__init__(children, **kwargs)

    mapper_method = intern('map_range')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    @property
    def lower(self):
        return self.start

    @property
    def upper(self):
        return self.stop


class RangeIndex(Range):
    """
    Internal representation of a subscript range.
    """

    mapper_method = intern('map_range_index')


class LoopRange(Range):
    """
    Internal representation of a loop range.
    """

    mapper_method = intern('map_loop_range')


class ArraySubscript(ExprMetadataMixin, StrCompareMixin, pmbl.Subscript):
    """
    Internal representation of an array subscript.
    """

    def __init__(self, index, **kwargs):
        # TODO: have aggregate here?
        super().__init__(None, index, **kwargs)

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    mapper_method = intern('map_array_subscript')


class ProcedureSymbol(ExprMetadataMixin, pmbl.FunctionSymbol):
    """
    Internal representation of a callable subroutine or function.
    """

    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def __getinitargs__(self):
        args = [('name', self.name)]
        return super().__getinitargs__() + tuple(args)

    mapper_method = intern('map_procedure_symbol')
