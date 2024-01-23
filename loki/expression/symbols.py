# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

"""
Expression tree node classes for
:ref:`internal_representation:Expression tree`.
"""

from itertools import chain
import weakref
from sys import intern
import pymbolic.primitives as pmbl
from pymbolic.mapper.evaluator import UnknownVariableError

from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes
from loki.scope import Scope
from loki.expression.mappers import ExpressionRetriever
from loki.config import config


__all__ = [
    'loki_make_stringifier',
    # Mix-ins
    'StrCompareMixin',
    # Typed leaf nodes
    'TypedSymbol', 'DeferredTypeSymbol', 'VariableSymbol', 'ProcedureSymbol',
    'MetaSymbol', 'Scalar', 'Array', 'Variable',
    # Non-typed leaf nodes
    'FloatLiteral', 'IntLiteral', 'LogicLiteral', 'StringLiteral',
    'IntrinsicLiteral', 'Literal', 'LiteralList', 'InlineDo',
    # Internal nodes
    'Sum', 'Product', 'Quotient', 'Power', 'Comparison', 'LogicalAnd', 'LogicalOr',
    'LogicalNot', 'InlineCall', 'Cast', 'Range', 'LoopRange', 'RangeIndex', 'ArraySubscript',
    'StringSubscript',
    # C/C++ concepts
    'Reference', 'Dereference',
]

# pylint: disable=abstract-method,too-many-lines

def loki_make_stringifier(self, originating_stringifier=None):  # pylint: disable=unused-argument
    """
    Return a :any:`LokiStringifyMapper` instance that can be used to generate a
    human-readable representation of :data:`self`.

    This is used as common abstraction for the :meth:`make_stringifier` method in
    Pymbolic expression nodes.
    """
    from loki.expression.mappers import LokiStringifyMapper  # pylint: disable=import-outside-toplevel
    return LokiStringifyMapper()


class StrCompareMixin:
    """
    Mixin to enable comparing expressions to strings.

    The purpose of the string comparison override is to reliably and flexibly
    identify expression symbols from equivalent strings.
    """

    @staticmethod
    def _canonical(s):
        """ Define canonical string representations (lower-case, no spaces) """
        if config['case-sensitive']:
            return str(s).replace(' ', '')
        return str(s).lower().replace(' ', '')

    def __hash__(self):
        return hash(self._canonical(self))

    def __eq__(self, other):
        if isinstance(other, (str, type(self))):
            # Do comparsion based on canonical string representations
            return self._canonical(self) == self._canonical(other)

        return super().__eq__(other)

    def __contains__(self, other):
        # Assess containment via a retriver with node-wise string comparison
        return len(ExpressionRetriever(lambda x: x == other).retrieve(self)) > 0

    make_stringifier = loki_make_stringifier


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

    init_arg_names = ('name', 'scope', 'parent', 'type', )

    def __init__(self, *args, **kwargs):
        self.name = kwargs['name']
        self.parent = kwargs.pop('parent', None)
        self.scope = kwargs.pop('scope', None)

        # Use provided type or try to determine from scope
        self._type = None
        self.type = kwargs.pop('type', None) or self.type

        super().__init__(*args, **kwargs)

    @property
    def name(self):
        if self.parent:
            return f'{self.parent.name}%{self._name}'
        return self._name

    @name.setter
    def name(self, name):
        self._name = name.split('%')[-1]

    def __getinitargs__(self):
        """
        Fixed tuple of initialisation arguments, corresponding to
        ``init_arg_names`` above.

        Note that this defines the pickling behaviour of pymbolic
        symbol objects. We do not recurse here, since we own the
        "name" attribute, which pymbolic will otherwise replicate.
        """
        return (self.name, None, self._parent, self._type, )

    @property
    def scope(self):
        """
        The object corresponding to the symbol's scope.
        """
        if self._scope is None:
            return None
        return self._scope()

    @scope.setter
    def scope(self, scope):
        assert scope is None or isinstance(scope, Scope)
        self._scope = None if scope is None else weakref.ref(scope)

    def _lookup_type(self, scope):
        """
        Helper method to look-up type information in any :data:`scope`

        Note that this is useful when trying to discover type information
        without putting the variable in :data:`scope` first. Combined with
        the recursive lookup of type information via the parent, this allows
        e.g. to distinguish between procedure calls and array subscripts for
        ambiguous derived type components.
        """
        _type = scope.symbol_attrs.lookup(self.name)
        if _type and _type.dtype is not BasicType.DEFERRED:
            # We have a clean entry in the symbol table which is not deferred
            return _type

        # Try a look-up via parent
        if self.parent:
            tdef_var = self.parent.variable_map.get(self.basename)
            if not tdef_var and self.parent.scope is not scope:
                # If the parent isn't delivering straight away (may happen e.g. for nested derived types)
                # we'll try discovering its parent's type via the provided scope
                parent = self._lookup_parent(scope)
                if parent:
                    tdef_var = parent.variable_map.get(self.basename)
            if tdef_var:
                return tdef_var.type

        return _type

    def _lookup_parent(self, scope):
        """
        Helper method to look-up parent variable using provided :data:`scope`
        """
        # Start at the root, i.e. the declared derived type object
        parent_name = self.name_parts[0]
        parent_type = scope.symbol_attrs.lookup(parent_name)
        parent_var = Variable(name=parent_name, scope=scope, type=parent_type)
        # Walk through nested derived types (if any)...
        for name in self.name_parts[1:-1]:
            if not parent_var:
                # If the look-up fails somewhere we have to bail out
                return None
            parent_var = parent_var.variable_map.get(name)  # pylint: disable=no-member
        # ...until we are at the actual parent
        return parent_var

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        if self.scope is None:
            return self._type
        return self._lookup_type(self.scope)

    @type.setter
    def type(self, _type):
        """
        Update the stored type information
        """
        if self._scope is None:
            # Store locally if not attached to a scope
            self._type = _type
        elif _type is None:
            # Store deferred type if unknown
            self.scope.symbol_attrs[self.name] = SymbolAttributes(BasicType.DEFERRED)
        elif _type is not self.scope.symbol_attrs.lookup(self.name):
            # Update type if it differs from stored type
            self.scope.symbol_attrs[self.name] = _type

    @property
    def parent(self):
        """
        Parent variable for derived type members

        Returns
        -------
        :any:`TypedSymbol` or :any:`MetaSymbol` or `NoneType`
            The parent variable or None
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        assert parent is None or isinstance(parent, (TypedSymbol, MetaSymbol))
        self._parent = parent

    @property
    def parents(self):
        """
        Variables nodes for all parents

        Returns
        -------
        tuple
            The list of parent variables, e.g., for a variable ``a%b%c%d`` this
            yields the nodes corresponding to ``(a, a%b, a%b%c)``
        """
        parent = self.parent
        if parent:
            return parent.parents + (parent,)
        return ()

    @property
    def variables(self):
        """
        List of member variables in a derived type

        Returns
        -------
        tuple of :any:`TypedSymbol` or :any:`MetaSymbol` if derived type variable, else `None`
            List of member variables in a derived type
        """
        _type = self.type
        if _type and isinstance(_type.dtype, DerivedType):
            if _type.dtype.typedef is BasicType.DEFERRED:
                return ()
            return tuple(
                v.clone(name=f'{self.name}%{v.name}', scope=self.scope, type=v.type, parent=self)
                for v in _type.dtype.typedef.variables
            )
        return None

    @property
    def variable_map(self):
        """
        Member variables in a derived type variable as a map

        Returns
        -------
        dict of (str, :any:`TypedSymbol` or :any:`MetaSymbol`)
            Map of member variable basenames to variable objects
        """
        return CaseInsensitiveDict((v.basename, v) for v in self.variables or ())

    @property
    def basename(self):
        """
        The symbol name without the qualifier from the parent.
        """
        return self._name

    @property
    def name_parts(self):
        """
        All name parts with parent qualifiers separated
        """
        if self.parent:
            return self.parent.name_parts + [self.basename]
        return [self.basename]

    def clone(self, **kwargs):
        """
        Replicate the object with the provided overrides.
        """
        # Add existing meta-info to the clone arguments, only if we have them.
        if 'name' not in kwargs and self.name:
            kwargs['name'] = self.name
        if 'scope' not in kwargs and self.scope:
            kwargs['scope'] = self.scope
        if 'type' not in kwargs:
            # If no type is given, check new scope
            if 'scope' in kwargs and kwargs['scope'] and kwargs['name'] in kwargs['scope'].symbol_attrs:
                kwargs['type'] = kwargs['scope'].symbol_attrs[kwargs['name']]
            else:
                kwargs['type'] = self.type
        if 'parent' not in kwargs and self.parent:
            kwargs['parent'] = self.parent

        return Variable(**kwargs)

    def rescope(self, scope):
        """
        Replicate the object with a new scope

        This is a bespoke variant of :meth:`clone` for rescoping
        symbols. The difference lies in the handling of the
        type information, making sure not to overwrite any existing
        symbol table entry in the provided scope.
        """
        if self.type:
            existing_type = self._lookup_type(scope)
            if existing_type:
                return self.clone(scope=scope, type=existing_type)
        return self.clone(scope=scope)


class DeferredTypeSymbol(StrCompareMixin, TypedSymbol, pmbl.Variable):  # pylint: disable=too-many-ancestors
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


class VariableSymbol(StrCompareMixin, TypedSymbol, pmbl.Variable):  # pylint: disable=too-many-ancestors
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


class _FunctionSymbol(pmbl.FunctionSymbol):
    """
    Adapter class for :any:`pymbolic.primitives.FunctionSymbol` that intercepts
    constructor arguments

    This is needed since the original symbol does not like having a :data:`name`
    parameter handed down in the constructor.
    """

    def __init__(self, *args, **kwargs):  # pylint:disable=unused-argument
        super().__init__()


class ProcedureSymbol(StrCompareMixin, TypedSymbol, _FunctionSymbol):  # pylint: disable=too-many-ancestors
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

    def __getstate__(self):
        return self._symbol

    def __setstate__(self, state):
        self._symbol = state

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
    def name_parts(self):
        return self.symbol.name_parts

    @property
    def parent(self):
        """
        For derived type members this yields the declared parent symbol to
        which it belongs
        """
        return self.symbol.parent

    @property
    def parents(self):
        """
        Yield all parent symbols for derived type members
        """
        return self.symbol.parents

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

    @type.setter
    def type(self, _type):
        """
        Update the :any:`SymbolAttributes` declared for this symbol
        """
        self.symbol.type = _type

    @property
    def variables(self):
        """
        List of member variables in a derived type

        Returns
        -------
        tuple of :any:`TypedSymbol` or :any:`MetaSymbol` if derived type variable, else `None`
            List of member variables in a derived type
        """
        return self.symbol.variables

    @property
    def variable_map(self):
        """
        Member variables in a derived type variable as a map

        Returns
        -------
        dict of (str, :any:`TypedSymbol` or :any:`MetaSymbol`)
            Map of member variable basenames to variable objects
        """
        return self.symbol.variable_map

    @property
    def initial(self):
        """
        Initial value of the variable in a declaration, if given
        """
        return self.type.initial

    mapper_method = intern('map_meta_symbol')
    make_stringifier = loki_make_stringifier

    def __getinitargs__(self):
        return self.symbol.__getinitargs__()

    @property
    def init_arg_names(self):
        return self.symbol.init_arg_names

    def clone(self, **kwargs):
        """
        Replicate the object with the provided overrides.
        """
        return self.symbol.clone(**kwargs)

    def rescope(self, scope):
        """
        Replicate the object with a new scope

        This is a bespoke variant of :meth:`clone` for rescoping
        symbols. The difference lies in the handling of the
        type information, making sure not to overwrite any existing
        symbol table entry in the provided scope.
        """
        return self.symbol.rescope(scope)


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

    mapper_method = intern('map_scalar')


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
    def name_parts(self):
        return self.symbol.name_parts

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
        return super().__getinitargs__() + (self.dimensions, )

    @property
    def init_arg_names(self):
        return super().init_arg_names + ('dimensions', )

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

    def rescope(self, scope):
        """
        Replicate the object with a new scope

        This is a bespoke variant of :meth:`clone` for rescoping
        symbols. The difference lies in the handling of the
        type information, making sure not to overwrite any existing
        symbol table entry in the provided scope.
        """
        if self.type:
            existing_type = scope.symbol_attrs.lookup(self.name)
            if existing_type:
                return self.clone(scope=scope, type=existing_type, dimensions=self.dimensions)
        return self.clone(scope=scope, dimensions=self.dimensions)


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
    Changing any of their attributes, including attaching them to a scope and
    modifying their type, should always be done via the :meth:`clone` method:

    .. code-block::

        var = Variable(name='foo')
        var = var.clone(scope=scope, type=SymbolAttributes(BasicType.INTEGER))
        var = var.clone(type=var.type.clone(dtype=BasicType.REAL))

    Attaching a symbol to a new scope without updating any stored type information
    (but still inserting type information if it doesn't exist, yet), can be done
    via the dedicated :meth:`rescope` method. This is essentially a :meth:`clone`
    invocation but without the type update:

    .. code-block::

        var = Variable(name='foo', type=SymbolAttributes(BasicType.INTEGER), scope=scope)
        unscoped_var = Variable(name='foo', type=SymbolAttributes(BasicType.REAL))
        scoped_var = unscoped_var.rescope(scope)  # scoped_var will have INTEGER type

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

        if scope is not None and _type is None:
            # Determine type information from scope if not provided explicitly
            _type = cls._get_type_from_scope(name, scope, kwargs.get('parent'))
        kwargs['type'] = _type

        if _type and isinstance(_type.dtype, ProcedureType):
            # This is the name in a function/subroutine call
            return ProcedureSymbol(**kwargs)

        if _type and isinstance(_type.dtype, DerivedType) and name.lower() == _type.dtype.name.lower():
            # This is a constructor call (or a type imported in an ``IMPORT`` statement, in which
            # case this is classified wrong...)
            return ProcedureSymbol(**kwargs)

        if 'dimensions' in kwargs and kwargs['dimensions'] is None:
            # Convenience: This way we can construct Scalar variables with `dimensions=None`
            kwargs.pop('dimensions')

        if kwargs.get('dimensions') is not None or (_type and _type.shape):
            return Array(**kwargs)
        if _type and _type.dtype is not BasicType.DEFERRED:
            return Scalar(**kwargs)
        return DeferredTypeSymbol(**kwargs)

    @classmethod
    def _get_type_from_scope(cls, name, scope, parent=None):
        """
        Helper method to determine the type of a symbol

        If no entry is found in the scope's symbol table, a lookup via
        the parent is attempted to construct the type for derived type
        members.

        Parameters
        ----------
        name : str
            The symbol's name
        scope : :any:`Scope`
            The scope in which to search for the symbol's type
        parent : :any:`MetaSymbol` or :any:`TypedSymbol`, optional
            The symbol's parent (for derived type members)

        Returns
        -------
        :any:`SymbolAttributes` or `None`
        """
        # 1. Try to find symbol in scope
        stored_type = scope.symbol_attrs.lookup(name)

        # 2. For derived type members, we can try to find it via the parent instead
        if '%' in name and (not stored_type or stored_type.dtype is BasicType.DEFERRED):
            name_parts = name.split('%')
            if not parent:
                # Build the parent if not given
                parent_type = scope.symbol_attrs.lookup(name_parts[0])
                parent = Variable(name=name_parts[0], scope=scope, type=parent_type)
                for pname in name_parts[1:-1]:
                    if not parent:
                        return None
                    parent = parent.variable_map.get(pname)  # pylint: disable=no-member
            if parent:
                # Lookup type in parent's typedef
                tdef_var = parent.variable_map.get(name_parts[-1])
                if tdef_var:
                    return tdef_var.type

        return stored_type


class _Literal(pmbl.Leaf):
    """
    Base class for literals.

    This exists to overcome the problem of a disfunctional
    :meth:`__getinitargs__` in any:`pymbolic.primitives.Leaf`.
    """

    def __getinitargs__(self):
        return ()


class FloatLiteral(StrCompareMixin, _Literal):
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
        except (TypeError, ValueError, UnknownVariableError):
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

    init_arg_names = ('value', 'kind')

    def __getinitargs__(self):
        return (self.value, self.kind)

    mapper_method = intern('map_float_literal')


class IntLiteral(StrCompareMixin, _Literal):
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

    init_arg_names = ('value', 'kind')

    def __getinitargs__(self):
        return (self.value, self.kind)

    def __int__(self):
        return self.value

    def __bool__(self):
        return bool(self.value)

    mapper_method = intern('map_int_literal')


# Register IntLiteral as a constant class in Pymbolic
pmbl.register_constant_class(IntLiteral)


class LogicLiteral(StrCompareMixin, _Literal):
    """
    A boolean constant in an expression.

    Parameters
    ----------
    value : bool
        The value of that literal.
    """

    def __init__(self, value, **kwargs):
        self.value = str(value).lower() in ('true', '.true.')
        super().__init__(**kwargs)

    init_arg_names = ('value', )

    def __getinitargs__(self):
        return (self.value, )

    mapper_method = intern('map_logic_literal')


class StringLiteral(StrCompareMixin, _Literal):
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

    init_arg_names = ('value', )

    def __getinitargs__(self):
        return (self.value, )

    mapper_method = intern('map_string_literal')


class IntrinsicLiteral(StrCompareMixin, _Literal):
    """
    Any literal not represented by a dedicated class.

    Its value is stored as string and returned unaltered.
    This is currently used for complex and BOZ constants and to retain
    array constructor expressions with type spec or implied-do.

    Parameters
    ----------
    value : str
        The value of that literal.
    """

    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    init_arg_names = ('value', )

    def __getinitargs__(self):
        return (self.value, )

    mapper_method = intern('map_intrinsic_literal')


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


class LiteralList(pmbl.AlgebraicLeaf):
    """
    A list of constant literals, e.g., as used in Array Initialization Lists.
    """

    def __init__(self, values, dtype=None, **kwargs):
        self.elements = values
        self.dtype = dtype
        super().__init__(**kwargs)

    mapper_method = intern('map_literal_list')
    make_stringifier = loki_make_stringifier

    def __getinitargs__(self):
        return (self.elements, self.dtype)


class InlineDo(pmbl.AlgebraicLeaf):
    """
    An inlined do, e.g., implied-do as used in array constructors
    """

    def __init__(self, values, variable, bounds, **kwargs):
        self.values = values
        self.variable = variable
        self.bounds = bounds
        super().__init__(**kwargs)

    mapper_method = intern('map_inline_do')
    make_stringifier = loki_make_stringifier

    def __getinitargs__(self):
        return (self.values, self.variable, self.bounds)


class Sum(StrCompareMixin, pmbl.Sum):
    """Representation of a sum."""


class Product(StrCompareMixin, pmbl.Product):
    """Representation of a product."""


class Quotient(StrCompareMixin, pmbl.Quotient):
    """Representation of a quotient."""


class Power(StrCompareMixin, pmbl.Power):
    """Representation of a power."""


class Comparison(StrCompareMixin, pmbl.Comparison):
    """Representation of a comparison operation."""


class LogicalAnd(StrCompareMixin, pmbl.LogicalAnd):
    """Representation of an 'and' in a logical expression."""


class LogicalOr(StrCompareMixin, pmbl.LogicalOr):
    """Representation of an 'or' in a logical expression."""


class LogicalNot(StrCompareMixin, pmbl.LogicalNot):
    """Representation of a negation in a logical expression."""


class InlineCall(pmbl.CallWithKwargs):
    """
    Internal representation of an in-line function call.
    """

    init_arg_names = ('function', 'parameters', 'kw_parameters')

    def __getinitargs__(self):
        return (self.function, self.parameters, self.kw_parameters)


    def __init__(self, function, parameters=None, kw_parameters=None, **kwargs):
        # Unfortunately, have to accept MetaSymbol here for the time being as
        # rescoping before injecting statement functions may create InlineCalls
        # with Scalar/Variable function names.
        assert isinstance(function, (ProcedureSymbol, DeferredTypeSymbol, MetaSymbol))
        parameters = parameters or ()
        kw_parameters = kw_parameters or {}

        super().__init__(function=function, parameters=parameters,
                         kw_parameters=kw_parameters, **kwargs)

    mapper_method = intern('map_inline_call')
    make_stringifier = loki_make_stringifier

    @property
    def _canonical(self):
        return (self.function, self.parameters, as_tuple(self.kw_parameters))

    def __hash__(self):
        # A custom `__hash__` function to protect us from unhashasble
        # dicts that `pmbl.CallWithKwargs` uses internally
        return hash(self._canonical)

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

    @property
    def arguments(self):
        """
        Alias for :attr:`parameters`
        """
        return self.parameters

    @property
    def kwarguments(self):
        """
        Alias for :attr:`kw_parameters`
        """
        return as_tuple(self.kw_parameters.items())

    @property
    def routine(self):
        """
        The :any:`Subroutine` object of the called routine

        Shorthand for ``call.function.type.dtype.procedure``

        Returns
        -------
        :any:`Subroutine` or :any:`BasicType.DEFERRED`
            If the :any:`ProcedureType` object of the :any:`ProcedureSymbol`
            in :attr:`function` is linked up to the target routine, this returns
            the corresponding :any:`Subroutine` object, otherwise `None`.
        """
        procedure_type = self.procedure_type
        if procedure_type is BasicType.DEFERRED:
            return BasicType.DEFERRED
        return procedure_type.procedure

    def arg_iter(self):
        """
        Iterator that maps argument definitions in the target :any:`Subroutine`
        to arguments and keyword arguments in the call.

        Returns
        -------
        iterator
            An iterator that traverses the mapping ``(arg name, call arg)`` for
            all positional and then keyword arguments.
        """
        routine = self.routine
        assert routine is not BasicType.DEFERRED
        r_args = CaseInsensitiveDict((arg.name, arg) for arg in routine.arguments)
        args = zip(routine.arguments, self.arguments)
        kwargs = ((r_args[kw], arg) for kw, arg in as_tuple(self.kwarguments))
        return chain(args, kwargs)

    def clone(self, **kwargs):
        """
        Replicate the object with the provided overrides.
        """
        function = kwargs.get('function', self.function)
        parameters = kwargs.get('parameters', self.parameters)
        kw_parameters = kwargs.get('kw_parameters', self.kw_parameters)
        return InlineCall(function, parameters, kw_parameters)


class Cast(pmbl.Call):
    """
    Internal representation of a data type cast.
    """

    def __init__(self, name, expression, kind=None, **kwargs):
        assert kind is None or isinstance(kind, pmbl.Expression)
        self.kind = kind
        super().__init__(pmbl.make_variable(name), as_tuple(expression), **kwargs)

    mapper_method = intern('map_cast')

    @property
    def name(self):
        return self.function.name


class Range(StrCompareMixin, pmbl.Slice):
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


class ArraySubscript(StrCompareMixin, pmbl.Subscript):
    """
    Internal representation of an array subscript.
    """
    mapper_method = intern('map_array_subscript')


class StringSubscript(StrCompareMixin, pmbl.Subscript):
    """
    Internal representation of a substring subscript operator.
    """
    mapper_method = intern('map_string_subscript')

    @property
    def symbol(self):
        return self.aggregate

class Reference(pmbl.Expression):
    """
    Internal representation of a Reference.

    **C/C++ only**, no corresponding concept in Fortran.
    Referencing refers to taking the address of an
    existing variable (to set a pointer variable).
    """
    init_arg_names = ('expression',)

    def __getinitargs__(self):
        return self.expression,

    def __init__(self, expression):
        assert isinstance(expression, pmbl.Expression)
        self.expression = expression

    """
    @property
    def name(self):
        return self.expression.name

    @property
    def type(self):
        return self.expression.type

    @property
    def scope(self):
        return self.expression.scope

    @property
    def initial(self):
        return self.expression.initial

    mapper_method = intern('map_c_reference')
    make_stringifier = loki_make_stringifier
    """


class Dereference(pmbl.Expression):
    """
    Internal representation of a Dereference.

    **C/C++ only**, no corresponding concept in Fortran.
    Dereferencing (a pointer) refers to retrieving the value
    from a memory address (that is pointed by the pointer).
    """
    init_arg_names = ('expression', )

    def __getinitargs__(self):
        return (self.expression, )

    def __init__(self, expression):
        assert isinstance(expression, pmbl.Expression)
        self.expression = expression

    """
    @property
    def name(self):
        return self.expression.name

    @property
    def type(self):
        return self.expression.type

    @property
    def scope(self):
        return self.expression.scope

    @property
    def initial(self):
        return self.expression.initial
    """

    mapper_method = intern('map_c_dereference')
    make_stringifier = loki_make_stringifier
