"""
Expression tree node classes for
:ref:`internal_representation:Expression tree`.
"""
import weakref
from sys import intern
import pymbolic.primitives as pmbl

from loki.tools import as_tuple
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes
from loki.scope import Scope
from loki.expression.mappers import LokiStringifyMapper, ExpressionRetriever


__all__ = [
    # Mix-ins
    'ExprMetadataMixin', 'StrCompareMixin',
    # Typed leaf nodes
    'TypedSymbol', 'DeferredTypeSymbol', 'VariableSymbol', 'ProcedureSymbol',
    'MetaSymbol', 'Scalar', 'Array', 'Variable',
    # Non-typed leaf nodes
    'FloatLiteral', 'IntLiteral', 'LogicLiteral', 'StringLiteral',
    'IntrinsicLiteral', 'Literal', 'LiteralList',
    # Internal nodes
    'Sum', 'Product', 'Quotient', 'Power', 'Comparison', 'LogicalAnd', 'LogicalOr',
    'LogicalNot', 'InlineCall', 'Cast', 'Range', 'LoopRange', 'RangeIndex', 'ArraySubscript',
]


# pylint: disable=abstract-method


class ExprMetadataMixin:
    """
    Mixin to store metadata annotations for expression tree nodes.

    Currently, this stores only source information for expression nodes.

    Parameters
    ----------
    source : :any:`Source`, optional
        Raw source string and line information from original source code.
    """

    def __init__(self, *args, **kwargs):
        self._metadata = {
            'source': kwargs.pop('source', None)
        }
        super().__init__(*args, **kwargs)

    def get_metadata(self):
        """All metadata as a dict."""
        return self._metadata.copy()

    def update_metadata(self, data):
        """Update the metadata for this expression node."""
        self._metadata.update(data)

    @property
    def source(self):
        """The :any:`Source` object for this expression node."""
        return self._metadata['source']

    @staticmethod
    def make_stringifier(originating_stringifier=None):  # pylint:disable=unused-argument,missing-function-docstring
        return LokiStringifyMapper()

    def clone(self, **kwargs):
        if self.source and 'source' not in kwargs:
            kwargs['source'] = self.source
        return super().clone(**kwargs)


class StrCompareMixin:
    """
    Mixin to enable comparing expressions to strings.

    The purpose of the string comparison override is to reliably and flexibly
    identify expression symbols from equivalent strings.
    """

    @staticmethod
    def _canonical(s):
        """ Define canonical string representations (lower-case, no spaces) """
        return s.__str__().lower().replace(' ', '')

    def __hash__(self):
        return hash(self._canonical(self))

    def __eq__(self, other):
        if isinstance(other, str):
            # Do comparsion based on canonical string representations
            return self._canonical(self) == self._canonical(other)

        return super().__eq__(other)

    def __contains__(self, other):
        # Assess containment via a retriver with node-wise string comparison
        return len(ExpressionRetriever(lambda x: x == other).retrieve(self)) > 0


class TypedSymbol:
    """
    Base class for symbols that carry type information.

    :class:`TypedSymbol` can be associated with a specific :any:`Scope` in
    which it is declared. In that case, all type information is cached in that
    scope's :any:`SymbolTable`. Creating :class:`TypedSymbol` without attaching
    it to a scope stores the type information locally.

    .. note::
        Providing :attr:`scope` and :attr:`type` overwrites the corresponding
        entry in the scope's symbol table. To not modify the type information
        omit :attr:`type` or use ``type=None``.

    Objects should always be created via the factory class :any:`Variable`.


    Parameters
    ----------
    name : str
        The identifier of that symbol (e.g., variable name).
    scope : :any:`Scope`
        The scope in which that symbol is declared.
    type : optional
        The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
    parent : :any:`Scalar` or :any:`Array`, optional
        The derived type variable this variable belongs to.
    *args : optional
        Any other positional arguments for other parent classes
    **kwargs : optional
        Any other keyword arguments for other parent classes
    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs['name']

        self.parent = kwargs.pop('parent', None)
        assert self.parent is None or isinstance(self.parent, (TypedSymbol, MetaSymbol))

        scope = kwargs.pop('scope', None)
        assert scope is None or isinstance(scope, Scope)
        self._scope = None if scope is None else weakref.ref(scope)

        # Use provided type or try to determine from scope
        self._type = None
        _type = kwargs.pop('type', None) or self.type

        # Update the stored type information
        if self._scope is None:
            # Store locally if not attached to a scope
            self._type = _type
        elif _type is None:
            # Store deferred type if unknown
            self.scope.symbols[self.name] = SymbolAttributes(BasicType.DEFERRED)
        elif _type is not self.type:
            # Update type if differs from stored type
            self.scope.symbols[self.name] = _type

        super().__init__(*args, **kwargs)

    def __getinitargs__(self):
        args = [self.name, ('scope', self.scope)]
        return tuple(args)

    @property
    def scope(self):
        """
        The object corresponding to the symbol's scope.
        """
        if self._scope is None:
            return None
        return self._scope()

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        if self.scope is None:
            return self._type or SymbolAttributes(BasicType.DEFERRED)
        return self.scope.symbols.lookup(self.name)

    @property
    def basename(self):
        """
        The symbol name without the qualifier from the parent.
        """
        idx = self.name.rfind('%')
        return self.name[idx+1:]

    def clone(self, **kwargs):
        """
        Replicate the object with the provided overrides.
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


class DeferredTypeSymbol(ExprMetadataMixin, StrCompareMixin, TypedSymbol, pmbl.Variable):  # pylint: disable=too-many-ancestors
    """
    Internal representation of symbols with deferred type

    This is used, for example, in the symbol list of :any:`Import` if a symbol's
    definition is not available.

    Note that symbols with deferred type are assumed to be variables, which
    implies they are included in the result from visitors such as
    :any:`FindVariables`.

    Parameters
    ----------
    name : str
        The name of the symbol
    scope : :any:`Scope`
        The scope in which the symbol is declared
    """

    def __init__(self, name, scope=None, **kwargs):
        if kwargs.get('type') is None:
            kwargs['type'] = SymbolAttributes(BasicType.DEFERRED)
        assert kwargs['type'].dtype is BasicType.DEFERRED
        super().__init__(name=name, scope=scope, **kwargs)

    mapper_method = intern('map_deferred_type_symbol')


class VariableSymbol(ExprMetadataMixin, StrCompareMixin, TypedSymbol, pmbl.Variable):  # pylint: disable=too-many-ancestors
    """
    Expression node to represent a variable symbol

    Note that this node should not be used directly to represent variables
    but instead meta nodes :any:`Scalar` or :any:`Array` (via their factory
    :any:`Variable`) should be used.

    The purpose of this is to align Loki's "convenience layer" for expressions
    with Pymbolic's expression tree structure. Loki makes variable use
    (especially for arrays) with or without properties (such as subscript
    dimensions) directly accessible from a single object, whereas Pymbolic
    represents array subscripts as an operation applied to a variable.

    Furthermore, it adds type information via :any:`TypedSymbol`.

    Parameters
    ----------
    name : str
        The name of the variable.
    scope : :any:`Scope`, optional
        The scope in which the variable is declared.
    type : :any:`SymbolAttributes`, optional
        The type of that symbol. Defaults to :any:`SymbolAttributes` with
        :any:`BasicType.DEFERRED`.
    """

    def __init__(self, name, scope=None, type=None, **kwargs):
        # Stop complaints about `type` in this function
        # pylint: disable=redefined-builtin
        super().__init__(name=name, scope=scope, type=type, **kwargs)

    @property
    def initial(self):
        """
        Initial value of the variable in declaration.
        """
        return self.type.initial

    @initial.setter
    def initial(self, value):
        self.type.initial = value

    mapper_method = intern('map_variable_symbol')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class _FunctionSymbol(pmbl.FunctionSymbol):
    """
    Adapter class for :any:`pymbolic.primitives.FunctionSymbol` that intercepts
    constructor arguments

    This is needed since the original symbol does not like having a :data:`name`
    parameter handed down in the constructor.
    """

    def __init__(self, *args, **kwargs):  # pylint:disable=unused-argument
        super().__init__()


class ProcedureSymbol(ExprMetadataMixin, StrCompareMixin, TypedSymbol, _FunctionSymbol):  # pylint: disable=too-many-ancestors
    """
    Internal representation of a symbol that represents a callable
    subroutine or function

    Parameters
    ----------
    name : str
        The name of the symbol.
    scope : :any:`Scope`
        The scope in which the symbol is declared.
    type : optional
        The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
    """

    def __init__(self, name, scope=None, type=None, **kwargs):
        # pylint: disable=redefined-builtin
        assert type is None or isinstance(type.dtype, ProcedureType) or \
                (isinstance(type.dtype, DerivedType) and name.lower() == type.dtype.name.lower())
        super().__init__(name=name, scope=scope, type=type, **kwargs)

    mapper_method = intern('map_procedure_symbol')


class MetaSymbol(StrCompareMixin, pmbl.AlgebraicLeaf):
    """
    Base class for meta symbols to encapsulate a symbol node with optional
    enclosing operations in a unifying interface

    The motivation for this class is that Loki strives to make variables
    and their use accessible via uniform interfaces :any:`Scalar` or
    :any:`Array`. Pymbolic's representation of array subscripts or access
    to members of a derived type are represented as operations on a symbol,
    thus resulting in a inside-out view that has the symbol innermost.

    To make it more convenient to find symbols and apply transformations on
    them, Loki wraps these compositions of expression tree nodes into meta
    nodes that store these compositions and provide direct access to properties
    of the contained nodes from a single object.

    In the simplest case, an instance of a :any:`TypedSymbol` subclass is
    stored as :attr:`symbol` and accessible via this property. Typical
    properties of this symbol (such as :attr:`name`, :attr:`type`, etc.) are
    directly accessible as properties that are redirected to the actual symbol.

    For arrays, not just the :any:`TypedSymbol` subclass but also an enclosing
    :any:`ArraySubscript` may be stored inside the meta symbol, providing
    additionally access to the subscript dimensions. The properties are then
    dynamically redirected to the relevant expression tree node.
    """

    def __init__(self, symbol, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._symbol = symbol

    @property
    def symbol(self):
        """
        The underlying :any:`TypedSymbol` node encapsulated by this meta node
        """
        return self._symbol

    @property
    def name(self):
        """
        The fully qualifying symbol name

        For derived type members this yields parent and basename
        """
        return self.symbol.name

    @property
    def basename(self):
        """
        For derived type members this yields the declared member name without
        the parent's name
        """
        return self.symbol.basename

    @property
    def parent(self):
        """
        For derived type members this yields the declared parent symbol to
        which it belongs
        """
        return self.symbol.parent

    @property
    def scope(self):
        """
        The scope in which the symbol was declared

        Note: for imported symbols this refers to the scope into which it is
        imported, _not_ where it was declared.
        """
        return self.symbol.scope

    @property
    def type(self):
        """
        The :any:`SymbolAttributes` declared for this symbol

        This includes data type as well as additional properties, such as
        ``INTENT``, ``KIND`` etc.
        """
        return self.symbol.type

    @property
    def initial(self):
        """
        Initial value of the variable in a declaration, if given
        """
        return self.type.initial

    @property
    def source(self):
        return self.symbol.source

    mapper_method = intern('map_meta_symbol')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    def __getinitargs__(self):
        return self.symbol.__getinitargs__()

    def clone(self, **kwargs):
        return self.symbol.clone(**kwargs)


class Scalar(MetaSymbol):  # pylint: disable=too-many-ancestors
    """
    Expression node for scalar variables.

    See :any:`MetaSymbol` for a description of meta symbols.

    Parameters
    ----------
    name : str
        The name of the variable.
    scope : :any:`Scope`
        The scope in which the variable is declared.
    type : optional
        The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
    """

    def __init__(self, name, scope=None, type=None, **kwargs):
        # Stop complaints about `type` in this function
        # pylint: disable=redefined-builtin
        symbol = VariableSymbol(name=name, scope=scope, type=type, **kwargs)
        super().__init__(symbol=symbol)

    def __getinitargs__(self):
        args = []
        if self.parent:
            args += [('parent', self.parent)]
        return super().__getinitargs__() + tuple(args)

    mapper_method = intern('map_scalar')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class Array(MetaSymbol):
    """
    Expression node for array variables.

    Similar to :any:`Scalar` with the notable difference that it has
    a shape (stored in :data:`type`) and can have associated
    :data:`dimensions` (i.e., the array subscript for indexing/slicing
    when accessing entries).

    See :any:`MetaSymbol` for a description of meta symbols.

    Parameters
    ----------
    name : str
        The name of the variable.
    scope : :any:`Scope`
        The scope in which the variable is declared.
    type : optional
        The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
    dimensions : :any:`ArraySubscript`, optional
        The array subscript expression.
    """

    def __init__(self, name, scope=None, type=None, dimensions=None, **kwargs):
        # Stop complaints about `type` in this function
        # pylint: disable=redefined-builtin

        symbol = VariableSymbol(name=name, scope=scope, type=type, **kwargs)
        if dimensions:
            symbol = ArraySubscript(symbol, dimensions)
        super().__init__(symbol=symbol)

    @property
    def symbol(self):
        if isinstance(self._symbol, ArraySubscript):
            return self._symbol.aggregate
        return self._symbol

    @property
    def dimensions(self):
        """
        Symbolic representation of the dimensions or indices.
        """
        if isinstance(self._symbol, ArraySubscript):
            return self._symbol.index_tuple
        return ()

    @property
    def shape(self):
        """
        Original allocated shape of the variable as a tuple of dimensions.
        """
        return self.type.shape

    def __getinitargs__(self):
        args = super().__getinitargs__()
        if self.dimensions:
            args += (('dimensions', self.dimensions),)
        return args

    mapper_method = intern('map_array')

    def clone(self, **kwargs):
        """
        Replicate the :class:`Array` variable with the provided overrides.

        Note, if :data:`dimensions` is set to ``None`` and :data:`type` updated
        to have no shape, this will create a :any:`Scalar` variable.
        """
        # Add existing meta-info to the clone arguments, only if we have them.
        if self.dimensions and 'dimensions' not in kwargs:
            kwargs['dimensions'] = self.dimensions
        return super().clone(**kwargs)


class Variable:
    """
    Factory class for :any:`TypedSymbol` or :any:`MetaSymbol` classes

    This is a convenience constructor to provide a uniform interface for
    instantiating different symbol types. It checks the symbol's type
    (either the provided :data:`type` or via a lookup in :data:`scope`)
    and :data:`dimensions` and dispatches the relevant class constructor.

    The tier algorithm is as follows:

    1. `type.dtype` is :any:`ProcedureType`: Instantiate a
       :any:`ProcedureSymbol`;
    2. :data:`dimensions` is not `None` or `type.shape` is not `None`:
       Instantiate an :any:`Array`;
    3. `type.dtype` is not :any:`BasicType.DEFERRED`:
       Instantiate a :any:`Scalar`;
    4. None of the above: Instantiate a :any:`DeferredTypeSymbol`

    All objects created by this factory implement :class:`TypedSymbol`. A
    :class:`TypedSymbol` object can be associated with a specific :any:`Scope` in
    which it is declared. In that case, all type information is cached in that
    scope's :any:`SymbolTable`. Creating :class:`TypedSymbol` without attaching
    it to a scope stores the type information locally.

    .. note::
        Providing :attr:`scope` and :attr:`type` overwrites the corresponding
        entry in the scope's symbol table. To not modify the type information
        omit :attr:`type` or use ``type=None``.

    Note that all :class:`TypedSymbol` and :class:`MetaSymbol` classes are
    intentionally quasi-immutable:
    Changing any of their attributes, including attaching them to a scope or
    modifying their type, should always be done via the :meth:`clone` method:

    .. codeblock::
        var = Variable(name='foo')
        var = var.clone(scope=scope, type=SymbolAttributes(BasicType.INTEGER))
        var = var.clone(type=var.type.clone(dtype=BasicType.REAL))

    Parameters
    ----------
    name : str
        The name of the variable.
    scope : :any:`Scope`
        The scope in which the variable is declared.
    type : optional
        The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
    parent : :any:`Scalar` or :any:`Array`, optional
        The derived type variable this variable belongs to.
    dimensions : :any:`ArraySubscript`, optional
        The array subscript expression.
    """

    def __new__(cls, **kwargs):
        name = kwargs['name']
        scope = kwargs.get('scope')
        _type = kwargs.get('type')

        if scope is not None and (_type is None or _type.dtype is BasicType.DEFERRED):
            # Try to determine stored type information if we have no or only deferred type
            stored_type = scope.symbols.lookup(name)
            if _type is None:
                _type = stored_type
            elif stored_type is not None:
                if stored_type.dtype is not BasicType.DEFERRED or not _type.attributes:
                    # If provided and stored are deferred but attributes given, we use provided
                    _type = stored_type
            kwargs['type'] = _type

        if _type and isinstance(_type.dtype, ProcedureType):
            # This is the name in a function/subroutine call
            return ProcedureSymbol(**kwargs)

        if _type and isinstance(_type.dtype, DerivedType) and name.lower() == _type.dtype.name.lower():
            # This is a constructor call
            return ProcedureSymbol(**kwargs)

        if 'dimensions' in kwargs and kwargs['dimensions'] is None:
            # Convenience: This way we can construct Scalar variables with `dimensions=None`
            kwargs.pop('dimensions')

        if kwargs.get('dimensions') is not None or (_type and _type.shape):
            obj = Array(**kwargs)
        elif _type and _type.dtype is not BasicType.DEFERRED:
            obj = Scalar(**kwargs)
        else:
            obj = DeferredTypeSymbol(**kwargs)

        obj = cls.instantiate_derived_type_variables(obj)
        return obj

    @classmethod
    def instantiate_derived_type_variables(cls, obj):
        """
        Helper routine to instantiate derived type variables.

        This method is called from the class constructor

        For members of a derived type we store type information in the scope
        the derived type variable is declared. To make sure type information
        exists for all members of the derived type, this helper routine
        instantiates these member variables.

        The reason this has to be done is that the list of variables stored in
        the type object of :data:`obj` potentially stems from a :any:`TypeDef`
        and as such, the variables are referring to a different scope.
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
    Base class for literals.

    This exists to overcome the problem of a disfunctional
    :meth:`__getinitargs__` in any:`pymbolic.primitives.Leaf`.
    """

    def __getinitargs__(self):
        return ()


class FloatLiteral(ExprMetadataMixin, _Literal):
    """
    A floating point constant in an expression.

    Note that its :data:`value` is stored as a string to avoid any
    representation issues that could stem from converting it to a
    Python floating point number.

    It can have a specific type associated, which backends can use to cast
    or annotate the constant to make sure the specified type is used.

    Parameters
    ----------
    value : str
        The value of that literal.
    kind : optional
        The kind information for that literal.
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
        except (TypeError, ValueError):
            return False

    def __lt__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) < float(other.value)
        try:
            return float(self.value) < float(other)
        except ValueError:
            return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) <= float(other.value)
        try:
            return float(self.value) <= float(other)
        except ValueError:
            return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) > float(other.value)
        try:
            return float(self.value) > float(other)
        except ValueError:
            return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) >= float(other.value)
        try:
            return float(self.value) >= float(other)
        except ValueError:
            return super().__ge__(other)

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

    It can have a specific type associated, which backends can use to cast
    or annotate the constant to make sure the specified type is used.

    Parameters
    ----------
    value : int
        The value of that literal.
    kind : optional
        The kind information for that literal.
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
        if isinstance(other, (int, float, complex)):
            return self.value == other

        try:
            return self.value == int(other)
        except (TypeError, ValueError):
            return False

    def __lt__(self, other):
        if isinstance(other, IntLiteral):
            return self.value < other.value
        if isinstance(other, int):
            return self.value < other
        return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, IntLiteral):
            return self.value <= other.value
        if isinstance(other, int):
            return self.value <= other
        return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, IntLiteral):
            return self.value > other.value
        if isinstance(other, int):
            return self.value > other
        return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, IntLiteral):
            return self.value >= other.value
        if isinstance(other, int):
            return self.value >= other
        return super().__ge__(other)

    def __getinitargs__(self):
        args = [self.value]
        if self.kind:
            args += [('kind', self.kind)]
        return tuple(args) + super().__getinitargs__()

    def __int__(self):
        return self.value

    def __bool__(self):
        return bool(self.value)

    mapper_method = intern('map_int_literal')

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


# Register IntLiteral as a constant class in Pymbolic
pmbl.register_constant_class(IntLiteral)


class LogicLiteral(ExprMetadataMixin, _Literal):
    """
    A boolean constant in an expression.

    Parameters
    ----------
    value : bool
        The value of that literal.
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
    A string constant in an expression.

    Parameters
    ----------
    value : str
        The value of that literal. Enclosing quotes are removed.
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
    Any literal not represented by a dedicated class.

    Its value is stored as string and returned unaltered.
    This is currently used for complex and BOZ constants.

    Parameters
    ----------
    value : str
        The value of that literal.
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
    Factory class to instantiate the best-matching literal node.

    This always returns a :class:`IntLiteral`, :class:`FloatLiteral`,
    :class:`StringLiteral`, :class:`LogicLiteral` or, as a fallback,
    :class:`IntrinsicLiteral`, selected by using any provided :data:`type`
    information or inspecting the Python data type of :data: value.

    Parameters
    ----------
    value :
        The value of that literal.
    kind : optional
        The kind information for that literal.
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
        assert isinstance(function, (ProcedureSymbol, DeferredTypeSymbol))
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

    @property
    def procedure_type(self):
        """
        Returns the underpinning procedure type if the type is know,
        ``BasicType.DEFFERED`` otherwise.
        """
        return self.function.type.dtype


class Cast(ExprMetadataMixin, pmbl.Call):
    """
    Internal representation of a data type cast.
    """

    def __init__(self, name, expression, kind=None, **kwargs):
        assert kind is None or isinstance(kind, pmbl.Expression)
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

    def __hash__(self):
        """ Need custom hashing function if we sepcialise :meth:`__eq__` """
        return hash(super().__str__().lower().replace(' ', ''))

    def __eq__(self, other):
        """ Specialization to capture ``a(1:n) == a(n)`` """
        if self.children[0] == 1 and self.children[2] is None:
            return self.children[1] == other or super().__eq__(other)
        return super().__eq__(other)

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

    def __hash__(self):
        """ Need custom hashing function if we specialise :meth:`__eq__` """
        return hash(super().__str__().lower().replace(' ', ''))

    def __eq__(self, other):
        """ Specialization to capture `a(1:n) == a(n)` """
        if self.children[0] == 1 and self.children[2] is None:
            return self.children[1] == other or super().__eq__(other)
        return super().__eq__(other)

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
    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()

    mapper_method = intern('map_array_subscript')
