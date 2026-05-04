# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Intermediate node classes for nested node definitions in the Loki IR. """

from typing import Any, Tuple, Union, Optional

from pymbolic.primitives import Expression
from pydantic import field_validator

from loki.ir.nodes.abstract_nodes import (
    Node, InternalNode, ScopedNode
)
from loki.expression import (
    symbols as sym, AttachScopesMapper, ExpressionDimensionsMapper
)
from loki.tools import (
    as_tuple, dataclass_strict, flatten, sanitize_tuple, CaseInsensitiveDict
)
from loki.types import BasicType, SymbolAttributes


__all__ = [
    'Section', 'Associate', 'Loop', 'WhileLoop', 'Conditional',
    'PragmaRegion', 'Interface',
]


@dataclass_strict(frozen=True)
class _SectionBase():
    """ Type definitions for :any:`Section` node type. """


@dataclass_strict(frozen=True)
class Section(InternalNode, _SectionBase):
    """
    Internal representation of a single code region.
    """

    def __iter__(self):
        return iter(self.body)

    def __getitem__(self, index):
        return self.body[index]

    def __len__(self):
        return len(self.body)

    def __contains__(self, item):
        return item in self.body

    def index(self, item, start=0, stop=None):
        if stop is None:
            return self.body.index(item, start)
        return self.body.index(item, start, stop)

    def append(self, node):
        """
        Append the given node(s) to the section's body.

        Parameters
        ----------
        node : :any:`Node` or tuple of :any:`Node`
            The node(s) to append to the section.
        """
        self._update(body=self.body + as_tuple(node))

    def insert(self, pos, node):
        """
        Insert the given node(s) into the section's body at a specific
        position.

        Parameters
        ----------
        pos : int
            The position at which the node(s) should be inserted. Any existing
            nodes at this or after this position are shifted back.
        node : :any:`Node` or tuple of :any:`Node`
            The node(s) to append to the section.
        """
        self._update(body=self.body[:pos] + as_tuple(node) + self.body[pos:])  # pylint: disable=unsubscriptable-object

    def prepend(self, node):
        """
        Insert the given node(s) at the beginning of the section's body.

        Parameters
        ----------
        node : :any:`Node` or tuple of :any:`Node`
            The node(s) to insert into the section.
        """
        self._update(body=as_tuple(node) + self.body)

    def __repr__(self):
        if self.label is not None:
            return f'Section:: {self.label}'
        return 'Section::'


@dataclass_strict(frozen=True)
class _AssociateBase():
    """ Type definitions for :any:`Associate` node type. """

    associations: Tuple[Tuple[Expression, Expression], ...]


@dataclass_strict(frozen=True)
class Associate(ScopedNode, Section, _AssociateBase):  # pylint: disable=too-many-ancestors
    """
    Internal representation of a code region in which names are associated
    with expressions or variables.

    Parameters
    ----------
    body : tuple
        The associate's body.
    associations : dict or collections.OrderedDict
        The mapping of names to expressions or variables valid inside the
        associate's body.
    parent : :any:`Scope`, optional
        The parent scope in which the associate appears
    symbol_attrs : :any:`SymbolTable`, optional
        An existing symbol table to use
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['body', 'associations']

    def __bool__(self):
        return True

    def __post_init__(self, parent=None):
        super(ScopedNode, self).__post_init__(parent=parent)
        super(Section, self).__post_init__()

        assert self.associations is None or isinstance(self.associations, tuple)

    @property
    def association_map(self):
        """
        An :any:`collections.OrderedDict` of associated expressions.
        """
        return CaseInsensitiveDict((k, v) for k, v in self.associations)

    @property
    def inverse_map(self):
        """
        An :any:`collections.OrderedDict` of associated expressions.
        """
        return CaseInsensitiveDict((v, k) for k, v in self.associations)

    @property
    def variables(self):
        return tuple(v for _, v in self.associations)

    def _derive_local_symbol_types(self, parent_scope):
        """ Derive the types of locally defined symbols from their associations. """

        rescoped_associations = ()
        for expr, name in self.associations:
            # Put symbols in associated expression into the right scope
            expr = AttachScopesMapper()(expr, scope=parent_scope)

            # Determine type of new names
            if isinstance(expr, (sym.TypedSymbol, sym.MetaSymbol)):
                # Use the type of the associated variable
                _type = expr.type.clone(parent=None)
                if isinstance(expr, sym.Array) and expr.dimensions is not None:
                    shape = ExpressionDimensionsMapper()(expr)
                    if shape == (sym.IntLiteral(1),):
                        # For a scalar expression, we remove the shape
                        shape = None
                    _type = _type.clone(shape=shape)
            else:
                # TODO: Handle data type and shape of complex expressions
                shape = ExpressionDimensionsMapper()(expr)
                if shape == (sym.IntLiteral(1),):
                    # For a scalar expression, we remove the shape
                    shape = None
                _type = SymbolAttributes(BasicType.DEFERRED, shape=shape)
            name = name.clone(scope=self, type=_type)
            rescoped_associations += ((expr, name),)

        self._update(associations=rescoped_associations)

    def __repr__(self):
        if self.associations:
            associations = ', '.join(f'{str(var)}={str(expr)}'
                                     for var, expr in self.associations)
            return f'Associate:: {associations}'
        return 'Associate::'


@dataclass_strict(frozen=True)
class _LoopBase():
    """ Type definitions for :any:`Loop` node type. """

    variable: Expression
    bounds: Expression
    body: Tuple[Node, ...]
    pragma: Optional[Tuple[Node, ...]] = None
    pragma_post: Optional[Tuple[Node, ...]] = None
    loop_label: Optional[Any] = None
    name: Optional[str] = None
    has_end_do: Optional[bool] = True


@dataclass_strict(frozen=True)
class Loop(InternalNode, _LoopBase):
    """
    Internal representation of a loop with induction variable and range.

    Parameters
    ----------
    variable : :any:`Scalar`
        The induction variable of the loop.
    bounds : :any:`LoopRange`
        The range of the loop, defining the iteration space.
    body : tuple
        The loop body.
    pragma : tuple of :any:`Pragma`, optional
        Pragma(s) that appear in front of the loop. By default :any:`Pragma`
        nodes appear as standalone nodes in the IR before the :any:`Loop` node.
        Only a bespoke context created by :py:func:`pragmas_attached`
        attaches them for convenience.
    pragma_post : tuple of :any:`Pragma`, optional
        Pragma(s) that appear after the loop. The same applies as for `pragma`.
    loop_label : str, optional
        The Fortran label for that loop. Importantly, this is an intrinsic
        Fortran feature and different from the statement label that can be
        attached to other nodes.
    name : str, optional
        The Fortran construct name for that loop.
    has_end_do : bool, optional
        In Fortran, loop blocks can be closed off by a ``CONTINUE`` statement
        (which we retain as an :any:`Intrinsic` node) and therefore ``END DO``
        can be omitted. For string reproducibility this parameter can be set
        `False` to indicate that this loop did not have an ``END DO``
        statement in the original source.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variable', 'bounds', 'body']

    def __post_init__(self):
        super().__post_init__()
        assert self.variable is not None

    def __repr__(self):
        label = ', '.join(l for l in [self.name, self.loop_label] if l is not None)
        if label:
            label = ' ' + label
        control = f'{str(self.variable)}={str(self.bounds)}'
        return f'Loop::{label} {control}'


@dataclass_strict(frozen=True)
class _WhileLoopBase():
    """ Type definitions for :any:`WhileLoop` node type. """

    condition: Optional[Expression]
    body: Tuple[Node, ...]
    pragma: Optional[Node] = None
    pragma_post: Optional[Node] = None
    loop_label: Optional[Any] = None
    name: Optional[str] = None
    has_end_do: Optional[bool] = True


@dataclass_strict(frozen=True)
class WhileLoop(InternalNode, _WhileLoopBase):
    """
    Internal representation of a while loop in source code.

    Importantly, this is different from a ``DO`` (Fortran) or ``for`` (C) loop,
    as we do not have a specified induction variable with explicit iteration
    range.

    Parameters
    ----------
    condition : :any:`pymbolic.primitives.Expression`
        The condition evaluated before executing the loop body.
    body : tuple
        The loop body.
    pragma : tuple of :any:`Pragma`, optional
        Pragma(s) that appear in front of the loop. By default :any:`Pragma`
        nodes appear as standalone nodes in the IR before the :any:`Loop` node.
        Only a bespoke context created by :py:func:`pragmas_attached`
        attaches them for convenience.
    pragma_post : tuple of :any:`Pragma`, optional
        Pragma(s) that appear after the loop. The same applies as for `pragma`.
    loop_label : str, optional
        The Fortran label for that loop. Importantly, this is an intrinsic
        Fortran feature and different from the statement label that can be
        attached to other nodes.
    name : str, optional
        The Fortran construct name for that loop.
    has_end_do : bool, optional
        In Fortran, loop blocks can be closed off by a ``CONTINUE`` statement
        (which we retain as an :any:`Intrinsic` node) and therefore ``END DO``
        can be omitted. For string reproducibility this parameter can be set
        `False` to indicate that this loop did not have an ``END DO``
        statement in the original source.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['condition', 'body']

    def __repr__(self):
        label = ', '.join(l for l in [self.name, self.loop_label] if l is not None)
        if label:
            label = ' ' + label
        control = str(self.condition) if self.condition else ''
        return f'WhileLoop::{label} {control}'


@dataclass_strict(frozen=True)
class _ConditionalBase():
    """ Type definitions for :any:`Conditional` node type. """

    condition: Expression
    body: Tuple[Node, ...]
    else_body: Optional[Tuple[Node, ...]] = ()
    inline: bool = False
    has_elseif: bool = False
    name: Optional[str] = None


@dataclass_strict(frozen=True)
class Conditional(InternalNode, _ConditionalBase):
    """
    Internal representation of a conditional branching construct.

    Parameters
    ----------
    condition : :any:`pymbolic.primitives.Expression`
        The condition evaluated before executing the body.
    body : tuple
        The conditional's body.
    else_body : tuple
        The body of the else branch. Can be empty.
    inline : bool, optional
        Flag that marks this conditional as inline, i.e., it s body consists
        only of a single statement that appeared immediately after the
        ``IF`` statement and it does not have an ``else_body``.
    has_elseif : bool, optional
        Flag that indicates that this conditional has an ``ELSE IF`` branch
        in the original source. In Loki's IR these are represented as a chain
        of :any:`Conditional` but for string reproducibility this flag can be
        provided to enable backends to reproduce the original appearance.
    name : str, optional
        The Fortran construct name for that conditional.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['condition', 'body', 'else_body']

    @field_validator('body', 'else_body', mode='before')
    @classmethod
    def ensure_tuple(cls, value):
        return sanitize_tuple(value)

    def __post_init__(self):
        super().__post_init__()
        assert self.condition is not None

        if self.has_elseif:
            assert len(self.else_body) == 1
            assert isinstance(self.else_body[0], Conditional)  # pylint: disable=unsubscriptable-object

    def __repr__(self):
        if self.name:
            return f'Conditional:: {self.name}'
        return 'Conditional::'

    @property
    def else_bodies(self):
        """
        Return all nested node tuples in the ``ELSEIF``/``ELSE`` part
        of the conditional chain.
        """
        if self.has_elseif:
            return (self.else_body[0].body,) + self.else_body[0].else_bodies
        return (self.else_body,) if self.else_body else ()


@dataclass_strict(frozen=True)
class _PragmaRegionBase():
    """ Type definitions for :any:`PragmaRegion` node type. """

    body: Tuple[Node, ...]
    pragma: Node = None
    pragma_post: Node = None


@dataclass_strict(frozen=True)
class PragmaRegion(InternalNode, _PragmaRegionBase):
    """
    Internal representation of a block of code defined by two matching pragmas.

    Generally, the pair of pragmas are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.

    This node type is injected into the IR within a context created by
    :py:func:`pragma_regions_attached`.

    Parameters
    ----------
    body : tuple
        The statements appearing between opening and closing pragma.
    pragma : :any:`Pragma`
        The opening pragma declaring that region.
    pragma_post : :any:`Pragma`
        The closing pragma for that region.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['body']

    def append(self, node):
        self._update(body=self.body + as_tuple(node))

    def insert(self, pos, node):
        '''Insert at given position'''
        self._update(body=self.body[:pos] + as_tuple(node) + self.body[pos:])  # pylint: disable=unsubscriptable-object

    def prepend(self, node):
        self._update(body=as_tuple(node) + self.body)

    def __repr__(self):
        return 'PragmaRegion::'


@dataclass_strict(frozen=True)
class _InterfaceBase():
    """ Type definitions for :any:`Interface` node type. """

    body: Tuple[Any, ...]
    abstract: bool = False
    spec: Optional[Union[Expression, str]] = None


@dataclass_strict(frozen=True)
class Interface(InternalNode, _InterfaceBase):
    """
    Internal representation of a Fortran interface block.

    Parameters
    ----------
    body : tuple
        The body of the interface block, containing function and subroutine
        specifications or procedure statements
    abstract : bool, optional
        Flag to indicate that this is an abstract interface
    spec : str, optional
        A generic name, operator, assignment, or I/O specification
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['body']

    def __post_init__(self):
        super().__post_init__()
        assert not (self.abstract and self.spec)

    @property
    def symbols(self):
        """
        The list of symbol names declared by this interface
        """
        symbols = as_tuple(flatten(
            getattr(node, 'procedure_symbol', getattr(node, 'symbols', ()))
            for node in self.body  # pylint: disable=not-an-iterable
        ))
        if self.spec:
            return (self.spec,) + symbols
        return symbols

    @property
    def symbol_map(self):
        """
        Map symbol name to symbol declared by this interface
        """
        return CaseInsensitiveDict(
            (s.name.lower(), s) for s in self.symbols
        )

    def __contains__(self, name):
        return name in self.symbol_map

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        if self.abstract:
            return f'Abstract Interface:: {symbols}'
        if self.spec:
            return f'Interface {self.spec}:: {symbols}'
        return f'Interface:: {symbols}'
