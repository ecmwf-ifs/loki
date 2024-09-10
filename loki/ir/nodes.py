# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines
"""
Control flow node classes for
:ref:`internal_representation:Control flow tree`
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Tuple, Union, Optional

from pymbolic.primitives import Expression

from pydantic.dataclasses import dataclass as dataclass_validated
from pydantic import model_validator

from loki.expression import Variable, parse_expr
from loki.frontend.source import Source
from loki.scope import Scope
from loki.tools import flatten, as_tuple, is_iterable, truncate_string, CaseInsensitiveDict
from loki.types import DataType, BasicType, DerivedType, SymbolAttributes


__all__ = [
    # Abstract base classes
    'Node', 'InternalNode', 'LeafNode', 'ScopedNode',
    # Internal node classes
    'Section', 'Associate', 'Loop', 'WhileLoop', 'Conditional',
    'PragmaRegion', 'Interface',
    # Leaf node classes
    'Assignment', 'ConditionalAssignment', 'CallStatement',
    'Allocation', 'Deallocation', 'Nullify',
    'Comment', 'CommentBlock', 'Pragma', 'PreprocessorDirective',
    'Import', 'VariableDeclaration', 'ProcedureDeclaration', 'DataDeclaration',
    'StatementFunction', 'TypeDef', 'MultiConditional', 'Forall', 'MaskedStatement',
    'Intrinsic', 'Enumeration', 'RawSource',
]

# Configuration for validation mechanism via pydantic
dataclass_validation_config  = {
    'arbitrary_types_allowed': True,
}

# Using this decorator, we can force strict validation
dataclass_strict = partial(dataclass_validated, config=dataclass_validation_config)


def _sanitize_tuple(t):
    """
    Small helper method to ensure non-nested tuples without ``None``.
    """
    return tuple(n for n in flatten(as_tuple(t)) if n is not None)


# Abstract base classes

@dataclass_strict(frozen=True)
class Node:
    """
    Base class for all node types in Loki's internal representation.

    Provides the common functionality shared by all node types; specifically,
    this comprises functionality to update or rebuild a node, and source
    metadata.

    Attributes
    ----------
    traversable : list of str
        The traversable fields of the Node; that is, fields walked over by
        a :any:`Visitor`. All arguments in :py:meth:`__init__` whose
        name appear in this list are treated as traversable fields.

    Parameters
    ----------
    source : :any:`Source`, optional
        the information about the original source for the Node.
    label : str, optional
        the label assigned to the statement in the original source
        corresponding to the Node.

    """

    source: Optional[Union[Source, str]] = None
    label: Optional[str] = None

    _traversable = []

    def __post_init__(self):
        # Create private placeholders for dataflow analysis fields that
        # do not show up in the dataclass field definitions, as these
        # are entirely transient.
        self._update(_live_symbols=None, _defines_symbols=None, _uses_symbols=None)

    @property
    def children(self):
        """
        The traversable children of the node.
        """
        return tuple(getattr(self, i) for i in self._traversable)

    def _rebuild(self, *args, **kwargs):
        """
        Rebuild the node.

        Constructs an identical copy of the node from when it was first
        created. Optionally, some or all of the arguments for it can
        be overwritten.

        Parameters
        ----------
        *args : optional
            The traversable arguments used to create the node. By default,
            ``args`` are used.
        **kwargs : optional
            The non-traversable arguments used to create the node, By
            default, ``args_frozen`` are used.
        """
        handle = self.args
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict(zip(argnames, args)))
        handle.update(kwargs)
        return type(self)(**handle)

    clone = _rebuild

    def _update(self, *args, **kwargs):
        """
        In-place update that modifies (re-initializes) the node
        without rebuilding it. Use with care!

        Parameters
        ----------
        *args : optional
            The traversable arguments used to create the node. By default,
            ``args`` are used.
        **kwargs : optional
            The non-traversable arguments used to create the node, By
            default, ``args_frozen`` are used.

        """
        argnames = [i for i in self._traversable if i not in kwargs]
        kwargs.update(zip(argnames, args))
        self.__dict__.update(kwargs)

    @property
    def args(self):
        """
        Arguments used to construct the Node.
        """
        return {k: v for k, v in self.__dict__.items() if k in self.__dataclass_fields__.keys()}  # pylint: disable=no-member

    @property
    def args_frozen(self):
        """
        Arguments used to construct the Node that cannot be traversed.
        """
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    def __repr__(self):
        raise NotImplementedError

    def view(self):
        """
        Pretty-print the node hierachy under this node.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from loki.ir.pprint import pprint
        pprint(self)

    def ir_graph(self, show_comments=False, show_expressions=False, linewidth=40, symgen=str):
        """
        Get the IR graph to visualize the node hierachy under this node.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from loki.ir.ir_graph import ir_graph

        return ir_graph(self, show_comments, show_expressions,linewidth, symgen)

    @property
    def live_symbols(self):
        """
        Yield the list of live symbols at this node, i.e., variables that
        have been defined (potentially) prior to this point in the control flow
        graph.

        This property is attached to the Node by
        :py:func:`loki.analyse.analyse_dataflow.attach_dataflow_analysis` or
        when using the
        :py:func:`loki.analyse.analyse_dataflow.dataflow_analysis_attached`
        context manager.
        """
        if self.__dict__['_live_symbols'] is None:
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self.__dict__['_live_symbols']

    @property
    def defines_symbols(self):
        """
        Yield the list of symbols (potentially) defined by this node.

        This property is attached to the Node by
        :py:func:`loki.analyse.analyse_dataflow.attach_dataflow_analysis` or
        when using the
        :py:func:`loki.analyse.analyse_dataflow.dataflow_analysis_attached`
        context manager.
        """
        if self.__dict__['_defines_symbols'] is None:
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self.__dict__['_defines_symbols']

    @property
    def uses_symbols(self):
        """

        Yield the list of symbols used by this node before defining it.

        This property is attached to the Node by
        :py:func:`loki.analyse.analyse_dataflow.attach_dataflow_analysis` or
        when using the
        :py:func:`loki.analyse.analyse_dataflow.dataflow_analysis_attached`
        context manager.
        """
        if self.__dict__['_uses_symbols'] is None:
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self.__dict__['_uses_symbols']


@dataclass_strict(frozen=True)
class _InternalNode():
    """ Type definitions for :any:`InternalNode` node type. """

    body: Tuple[Union[Node, Scope], ...] = ()


@dataclass_strict(frozen=True)
class InternalNode(Node, _InternalNode):
    """
    Internal representation of a control flow node that has a traversable
    `body` property.

    Parameters
    ----------
    body : tuple
        The nodes that make up the body.
    """

    _traversable = ['body']

    @model_validator(mode='before')
    @classmethod
    def pre_init(cls, values):
        """ Ensure non-nested tuples for body. """
        if values.kwargs and 'body' in values.kwargs:
            values.kwargs['body'] = _sanitize_tuple(values.kwargs['body'])
        if values.args:
            # ArgsKwargs are immutable, so we need to force it a little
            new_args = (_sanitize_tuple(values.args[0]),) + values.args[1:]
            values = type(values)(args=new_args, kwargs=values.kwargs)
        return values

    def __repr__(self):
        raise NotImplementedError


@dataclass_strict(frozen=True)
class LeafNode(Node):
    """
    Internal representation of a control flow node without a `body`.
    """

    def __repr__(self):
        raise NotImplementedError


# Mix-ins

class ScopedNode(Scope):
    """
    Mix-in to attache a scope to an IR :any:`Node`

    Additionally, this specializes the node's :meth:`_update` and
    :meth:`_rebuild` methods to make sure that an existing symbol table
    is carried over correctly.
    """

    @property
    def args(self):
        """
        Arguments used to construct the :any:`ScopedNode`, excluding
        the symbol table.
        """
        keys = tuple(k for k in self.__dataclass_fields__.keys() if k not in ('symbol_attrs', ))  # pylint: disable=no-member
        return {k: v for k, v in self.__dict__.items() if k in keys}

    def _update(self, *args, **kwargs):
        if 'symbol_attrs' not in kwargs:
            # Retain the symbol table (unless given explicitly)
            kwargs['symbol_attrs'] = self.symbol_attrs
        super()._update(*args, **kwargs)  # pylint: disable=no-member

    def _rebuild(self, *args, **kwargs):
        # Retain the symbol table (unless given explicitly)
        symbol_attrs = kwargs.pop('symbol_attrs', self.symbol_attrs)
        rescope_symbols = kwargs.pop('rescope_symbols', False)

        # Ensure 'parent' is always explicitly set
        kwargs['parent'] = kwargs.get('parent', None)

        new_obj = super()._rebuild(*args, **kwargs)  # pylint: disable=no-member
        new_obj.symbol_attrs.update(symbol_attrs)

        if rescope_symbols:
            new_obj.rescope_symbols()
        return new_obj

    def __getstate__(self):
        s = self.args
        s['symbol_attrs'] = self.symbol_attrs
        return s

    def __setstate__(self, s):
        symbol_attrs = s.pop('symbol_attrs', None)
        self._update(**s, symbol_attrs=symbol_attrs, rescope_symbols=True)

    def get_symbol(self, name):
        """
        Returns the symbol for a given name as defined in its declaration.

        The returned symbol might include dimension symbols if it was
        declared as an array.

        Parameters
        ----------
        name : str
            Base name of the symbol to be retrieved
        """
        return self.get_symbol_scope(name).variable_map.get(name)  # pylint: disable=no-member

    def Variable(self, **kwargs):
        """
        Factory method for :any:`TypedSymbol` or :any:`MetaSymbol` classes.

        This invokes the :any:`Variable` with this node as the scope.

        Parameters
        ----------
        name : str
            The name of the variable.
        type : optional
            The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
        parent : :any:`Scalar` or :any:`Array`, optional
            The derived type variable this variable belongs to.
        dimensions : :any:`ArraySubscript`, optional
            The array subscript expression.
        """
        kwargs['scope'] = self
        return Variable(**kwargs)

    def parse_expr(self, expr_str, strict=False, evaluate=False, context=None):
        """
        Uses :meth:`parse_expr` to convert expression(s) represented
        in a string to Loki expression(s)/IR.

        Parameters
        ----------
        expr_str : str
            The expression as a string
        strict : bool, optional
            Whether to raise exception for unknown variables/symbols when
            evaluating an expression (default: `False`)
        evaluate : bool, optional
            Whether to evaluate the expression or not (default: `False`)
        context : dict, optional
            Symbol context, defining variables/symbols/procedures to help/support
            evaluating an expression

        Returns
        -------
        :any:`Expression`
            The expression tree corresponding to the expression
        """
        return parse_expr(expr_str, scope=self, strict=strict, evaluate=evaluate, context=context)


# Intermediate node types


@dataclass_strict(frozen=True)
class _SectionBase():
    """ Type definitions for :any:`Section` node type. """


@dataclass_strict(frozen=True)
class Section(InternalNode, _SectionBase):
    """
    Internal representation of a single code region.
    """

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

    def __post_init__(self, parent=None):
        super(ScopedNode, self).__post_init__(parent=parent)
        super(Section, self).__post_init__()

        assert self.associations is None or isinstance(self.associations, tuple)

    @property
    def association_map(self):
        """
        An :any:`collections.OrderedDict` of associated expressions.
        """
        return OrderedDict(self.associations)

    @property
    def variables(self):
        return tuple(v for _, v in self.associations)

    @property
    def variable_map(self):
        """
        Map of variable names to :any:`Variable` objects
        """
        return CaseInsensitiveDict((v.name, v) for v in self.variables)

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

    @model_validator(mode='before')
    @classmethod
    def pre_init(cls, values):
        values = super().pre_init(values)
        # Ensure non-nested tuples for else_body
        if 'else_body' in values.kwargs:
            values.kwargs['else_body'] = _sanitize_tuple(values.kwargs['else_body'])
        return values

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

# Leaf node types

@dataclass_strict(frozen=True)
class _AssignmentBase():
    """ Type definitions for :any:`Assignment` node type. """

    lhs: Expression
    rhs: Expression
    ptr: bool = False
    comment: Optional[Node] = None


@dataclass_strict(frozen=True)
class Assignment(LeafNode, _AssignmentBase):
    """
    Internal representation of a variable assignment.

    Parameters
    ----------
    lhs : :any:`pymbolic.primitives.Expression`
        The left-hand side of the assignment.
    rhs : :any:`pymbolic.primitives.Expression`
        The right-hand side expression of the assignment.
    ptr : bool, optional
        Flag to indicate pointer assignment (``=>``). Defaults to ``False``.
    comment : :py:class:`Comment`, optional
        Inline comment that appears in-line after the right-hand side in the
        original source.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['lhs', 'rhs']

    def __post_init__(self):
        super().__post_init__()
        assert self.lhs is not None
        assert self.rhs is not None

    def __repr__(self):
        return f'Assignment:: {str(self.lhs)} = {str(self.rhs)}'


@dataclass_strict(frozen=True)
class _ConditionalAssignmentBase():
    """ Type definitions for :any:`ConditionalAssignment` node type. """

    lhs: Optional[Expression] = None
    condition: Optional[Expression] = None
    rhs: Optional[Expression] = None
    else_rhs: Optional[Expression] = None


@dataclass_strict(frozen=True)
class ConditionalAssignment(LeafNode, _ConditionalAssignmentBase):
    """
    Internal representation of an inline conditional assignment using a
    ternary operator.

    There is no Fortran-equivalent to this. In C, this takes the following form:

    .. code-block:: C

        lhs = condition ? rhs : else_rhs;

    Parameters
    ----------
    lhs : :any:`pymbolic.primitives.Expression`
        The left-hand side of the assignment.
    condition : :any:`pymbolic.primitives.Expression`
        The condition of the ternary operator.
    rhs : :any:`pymbolic.primitives.Expression`
        The right-hand side expression of the assignment that is assigned when
        the condition applies.
    else_rhs : :any:`pymbolic.primitives.Expression`
        The right-hand side expression of the assignment that is assigned when
        the condition does not apply.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['condition', 'lhs', 'rhs', 'else_rhs']

    def __repr__(self):
        return f'CondAssign:: {self.lhs} = {self.condition} ? {self.rhs} : {self.else_rhs}'


@dataclass_strict(frozen=True)
class _CallStatementBase():
    """ Type definitions for :any:`CallStatement` node type. """

    name: Expression
    arguments: Optional[Tuple[Expression, ...]] = None
    kwarguments: Optional[Tuple[Tuple[str, Expression], ...]] = None
    pragma: Optional[Tuple[Node, ...]] = None
    not_active: Optional[bool] = None
    chevron: Optional[Tuple[Expression, ...]] = None


@dataclass_strict(frozen=True)
class CallStatement(LeafNode, _CallStatementBase):
    """
    Internal representation of a subroutine call.

    Parameters
    ----------
    name : :any:`pymbolic.primitives.Expression`
        The name of the subroutine to call.
    arguments : tuple of :any:`pymbolic.primitives.Expression`
        The list of positional arguments.
    kwarguments : tuple of tuple
        The list of keyword arguments, provided as pairs of `(name, value)`.
    pragma : tuple of :any:`Pragma`, optional
        Pragma(s) that appear in front of the statement. By default
        :any:`Pragma` nodes appear as standalone nodes in the IR before.
        Only a bespoke context created by :py:func:`pragmas_attached`
        attaches them for convenience.
    not_active : bool, optional
        Flag to indicate that this call has explicitly been marked as inactive for
        the purpose of processing call trees (Default: `None`)
    chevron : tuple of :any:`pymbolic.primitives.Expression`
        Launch configuration for CUDA Fortran Kernels.
        See [CUDA Fortran programming guide](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/).
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['name', 'arguments', 'kwarguments']

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.arguments, tuple)
        assert all(isinstance(arg, Expression) for arg in as_tuple(self.arguments))

        if self.kwarguments is not None:
            assert isinstance(self.kwarguments, tuple)
            assert all(
                isinstance(a, tuple) and len(a) == 2 and isinstance(a[1], Expression)
                for a in self.kwarguments  # pylint: disable=not-an-iterable
            )

        if self.chevron is not None:
            assert isinstance(self.chevron, tuple)
            assert all(isinstance(a, Expression) for a in self.chevron)  # pylint: disable=not-an-iterable
            assert 2 <= len(self.chevron) <= 4

    def __repr__(self):
        return f'Call:: {self.name}'

    @property
    def procedure_type(self):
        """
        The :any:`ProcedureType` of the :any:`Subroutine` object of the called routine

        For a :class:`CallStatement` node called ``call``, this is shorthand for ``call.name.type.dtype``.

        If the procedure type object has been linked up with the corresponding
        :any:`Subroutine` object, then it is available via ``call.procedure_type.procedure``.

        Returns
        -------
        :any:`ProcedureType` or :any:`BasicType.DEFERRED`
            The type of the called procedure. If the symbol type of the called routine
            has not been identified correctly, this may yield :any:`BasicType.DEFERRED`.
        """
        return self.name.type.dtype

    @property
    def routine(self):
        """
        The :any:`Subroutine` object of the called routine

        Shorthand for ``call.name.type.dtype.procedure``

        Returns
        -------
        :any:`Subroutine` or :any:`BasicType.DEFERRED`
            If the :any:`ProcedureType` object of the :any:`ProcedureSymbol`
            in :attr:`name` is linked up to the target routine, this returns
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

    @property
    def arg_map(self):
        """
        A full map of all qualified argument matches from arguments
        and keyword arguments.
        """
        return dict(self.arg_iter())

    def _sort_kwarguments(self):
        """
        Helper routine to sort the kwarguments according to the order of the
        arguments (``self.routine.arguments``)`.
        """
        routine = self.routine
        assert routine is not BasicType.DEFERRED
        kwargs = CaseInsensitiveDict(self.kwarguments)
        r_arg_names = [arg.name for arg in routine.arguments if arg.name in kwargs]
        new_kwarguments = tuple((arg_name, kwargs[arg_name]) for arg_name in r_arg_names)
        return new_kwarguments

    def check_kwarguments_order(self):
        """
        Check whether kwarguments are correctly ordered
        in respect to the arguments (``self.routine.arguments``).
        """
        return self.kwarguments == self._sort_kwarguments()

    def sort_kwarguments(self):
        """
        Sort and update the kwarguments according to the order of the
        arguments (``self.routine.arguments``).
        """
        new_kwarguments = self._sort_kwarguments()
        self._update(kwarguments=new_kwarguments)

    def convert_kwargs_to_args(self):
        """
        Convert all kwarguments to arguments and update the call accordingly.
        """
        new_kwarguments = self._sort_kwarguments()
        new_args = tuple(arg[1] for arg in new_kwarguments)
        self._update(arguments=self.arguments + new_args, kwarguments=())


@dataclass_strict(frozen=True)
class _AllocationBase():
    """ Type definitions for :any:`Allocation` node type. """

    variables: Tuple[Expression, ...]
    data_source: Optional[Expression] = None
    status_var: Optional[Expression] = None


@dataclass_strict(frozen=True)
class Allocation(LeafNode, _AllocationBase):
    """
    Internal representation of a variable allocation.

    Parameters
    ----------
    variables : tuple of :any:`pymbolic.primitives.Expression`
        The list of variables that are allocated.
    data_source : :any:`pymbolic.primitives.Expression` or str
        Fortran's ``SOURCE`` allocation option.
    status_var : :any:`pymbolic.primitives.Expression`
        Fortran's ``STAT`` allocation option.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variables', 'data_source', 'status_var']

    def __post_init__(self):
        super().__post_init__()
        assert is_iterable(self.variables)
        assert all(isinstance(var, Expression) for var in self.variables)
        assert self.data_source is None or isinstance(self.data_source, Expression)
        assert self.status_var is None or isinstance(self.status_var, Expression)

    def __repr__(self):
        return f'Allocation:: {", ".join(str(var) for var in self.variables)}'


@dataclass_strict(frozen=True)
class _DeallocationBase():
    """ Type definitions for :any:`Deallocation` node type. """

    variables: Tuple[Expression, ...]
    status_var: Optional[Expression] = None


@dataclass_strict(frozen=True)
class Deallocation(LeafNode, _DeallocationBase):
    """
    Internal representation of a variable deallocation.

    Parameters
    ----------
    variables : tuple of :any:`pymbolic.primitives.Expression`
        The list of variables that are deallocated.
    status_var : :any:`pymbolic.primitives.Expression`
        Fortran's ``STAT`` deallocation option.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variables', 'status_var']

    def __post_init__(self):
        super().__post_init__()
        assert is_iterable(self.variables)
        assert all(isinstance(var, Expression) for var in self.variables)
        assert self.status_var is None or isinstance(self.status_var, Expression)

    def __repr__(self):
        return f'Deallocation:: {", ".join(str(var) for var in self.variables)}'


@dataclass_strict(frozen=True)
class _NullifyBase():
    """ Type definitions for :any:`Nullify` node type. """

    variables: Tuple[Expression, ...]


@dataclass_strict(frozen=True)
class Nullify(LeafNode, _NullifyBase):
    """
    Internal representation of a pointer nullification.

    Parameters
    ----------
    variables : tuple of :any:`pymbolic.primitives.Expression`
        The list of pointer variables that are nullified.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variables']

    def __post_init__(self):
        super().__post_init__()
        assert is_iterable(self.variables)
        assert all(isinstance(var, Expression) for var in self.variables)

    def __repr__(self):
        return f'Nullify:: {", ".join(str(var) for var in self.variables)}'


@dataclass_strict(frozen=True)
class _CommentBase():
    """ Type definitions for :any:`Comment` node type. """

    text: str


@dataclass_strict(frozen=True)
class Comment(LeafNode, _CommentBase):
    """
    Internal representation of a single comment.

    Parameters
    ----------
    text : str, optional
        The content of the comment. Can be empty to represent empty lines
        in the original source.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __post_init__(self):
        assert isinstance(self.text, str)

    def __repr__(self):
        return f'Comment:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class _CommentBlockBase():
    """ Type definitions for :any:`CommentBlock` node type. """

    comments: Tuple[Node, ...]


@dataclass_strict(frozen=True)
class CommentBlock(LeafNode, _CommentBlockBase):
    """
    Internal representation of a block comment that is formed from
    multiple single-line comments.

    Parameters
    ----------
    comments: tuple of :any:`Comment`
        The individual (subsequent) comments that make up the block.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.comments is not None
        assert is_iterable(self.comments)

    @property
    def text(self):
        """The combined string of all comments in this block"""
        return ''.join(comment.text for comment in self.comments)

    def __repr__(self):
        return f'CommentBlock:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class _PragmaBase():
    """ Type definitions for :any:`Pragma` node type. """

    keyword: str
    content: Optional[str] = None


@dataclass_strict(frozen=True)
class Pragma(LeafNode, _PragmaBase):
    """
    Internal representation of a pragma.

    Pragmas are assumed to appear in Fortran source code in the form of
    `!$<keyword> <content>`.

    Parameters
    ----------
    keyword : str
        The keyword of the pragma.
    content : str, optional
        The content of the pragma after the keyword.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.keyword and isinstance(self.keyword, str)

    def __repr__(self):
        return f'Pragma:: {self.keyword} {truncate_string(self.content)}'


@dataclass_strict(frozen=True)
class _PreprocessorDirectiveBase():
    """ Type definitions for :any:`PreprocessorDirective` node type. """

    text: str = None


@dataclass_strict(frozen=True)
class PreprocessorDirective(LeafNode, _PreprocessorDirectiveBase):
    """
    Internal representation of a preprocessor directive.

    Preprocessor directives are typically assumed to start at the beginning of
    a line with the letter ``#`` in the original source.

    Parameters
    ----------
    text : str, optional
        The content of the directive.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __repr__(self):
        return f'PreprocessorDirective:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class _ImportBase():
    """ Type definitions for :any:`Import` node type. """

    module: Optional[str]
    symbols: Tuple[Expression, ...] = ()
    nature: Optional[str] = None
    c_import: bool = False
    f_include: bool = False
    f_import: bool = False
    rename_list: Optional[Tuple[Any, ...]] = None


@dataclass_strict(frozen=True)
class Import(LeafNode, _ImportBase):
    """
    Internal representation of an import.

    Parameters
    ----------
    module : str
        The name of the module or header file to import from.
    symbols : tuple of :any:`Expression` or :any:`DataType`, optional
        The list of names imported. Can be empty when importing all.
    nature : str, optional
        The module nature (``INTRINSIC`` or ``NON_INTRINSIC``)
    c_import : bool, optional
        Flag to indicate that this is a C-style include. Defaults to `False`.
    f_include : bool, optional
        Flag to indicate that this is a preprocessor-style include in
        Fortran source code.
    f_import : bool, optional
        Flag to indicate that this is a Fortran ``IMPORT``.
    rename_list: tuple of tuples (`str`, :any:`Expression`), optional
        Rename list with pairs of `(use name, local name)` entries
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['symbols', 'rename_list']

    def __post_init__(self):
        super().__post_init__()
        assert self.module is None or isinstance(self.module, str)
        assert isinstance(self.symbols, tuple)
        assert all(isinstance(s, (Expression, DataType)) for s in self.symbols)
        assert self.nature is None or (
            isinstance(self.nature, str) and
            self.nature.lower() in ('intrinsic', 'non_intrinsic') and
            not (self.c_import or self.f_include or self.f_import)
        )
        if self.c_import + self.f_include + self.f_import not in (0, 1):
            raise ValueError('Import can only be either C include, F include or F import')
        if self.rename_list and (self.symbols or self.c_import or self.f_include or self.f_import):
            raise ValueError('Import cannot have rename and only lists or be an include')

    def __repr__(self):
        if self.f_import:
            return f'Import:: {self.symbols}'
        _c = 'C-' if self.c_import else 'F-' if self.f_include else ''
        return f'{_c}Import:: {self.module} => {self.symbols}'


@dataclass_strict(frozen=True)
class _VariableDeclarationBase():
    """ Type definitions for :any:`VariableDeclaration` node type. """

    symbols: Tuple[Expression, ...]
    dimensions: Optional[Tuple[Expression, ...]] = None
    comment: Optional[Node] = None
    pragma: Optional[Node] = None


@dataclass_strict(frozen=True)
class VariableDeclaration(LeafNode, _VariableDeclarationBase):
    """
    Internal representation of a variable declaration.

    Parameters
    ----------
    symbols : tuple of :any:`pymbolic.primitives.Expression`
        The list of variables declared by this declaration.
    dimensions : tuple of :any:`pymbolic.primitives.Expression`, optional
        The declared allocation size if given as part of the declaration
        attributes.
    comment : :py:class:`Comment`, optional
        Inline comment that appears in-line after the declaration in the
        original source.
    pragma : tuple of :any:`Pragma`, optional
        Pragma(s) that appear before the declaration. By default
        :any:`Pragma` nodes appear as standalone nodes in the IR.
        Only a bespoke context created by :py:func:`pragmas_attached`
        attaches them for convenience.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['symbols', 'dimensions']

    def __post_init__(self):
        super().__post_init__()
        assert self.symbols is not None
        assert is_iterable(self.symbols)
        assert all(isinstance(s, Expression) for s in self.symbols)

        if self.dimensions is not None:
            assert is_iterable(self.dimensions)
            assert all(isinstance(d, Expression) for d in self.dimensions)  # pylint: disable=not-an-iterable

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        return f'VariableDeclaration:: {symbols}'


@dataclass_strict(frozen=True)
class _ProcedureDeclarationBase():
    """ Type definitions for :any:`ProcedureDeclaration` node type. """

    symbols: Tuple[Expression, ...]
    interface: Optional[Union[Expression, DataType]] = None
    external: bool = False
    module: bool = False
    generic: bool = False
    final: bool = False
    comment: Optional[Node] = None
    pragma: Optional[Tuple[Node, ...]] = None


@dataclass_strict(frozen=True)
class ProcedureDeclaration(LeafNode, _ProcedureDeclarationBase):
    """
    Internal representation of a procedure declaration.

    Parameters
    ----------
    symbols : tuple of :any:`pymbolic.primitives.Expression`
        The list of procedure symbols declared by this declaration.
    interface : :any:`pymbolic.primitives.Expression` or :any:`DataType`, optional
        The procedure interface of the declared procedure entity names.
    external : bool, optional
        This is a Fortran ``EXTERNAL`` declaration.
    module : bool, optional
        This is a Fortran ``MODULE PROCEDURE`` declaration in an interface
        (i.e. includes the keyword ``MODULE``)
    generic : bool,  optional
        This is a generic binding procedure statement in a derived type.
    final : bool, optional
        This is a declaration to mark a subroutine for clean-up of a
        derived type.
    comment : :py:class:`Comment`, optional
        Inline comment that appears in-line after the declaration in the
        original source.
    pragma : tuple of :any:`Pragma`, optional
        Pragma(s) that appear before the declaration. By default
        :any:`Pragma` nodes appear as standalone nodes in the IR.
        Only a bespoke context created by :py:func:`pragmas_attached`
        attaches them for convenience.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['symbols', 'interface']

    def __post_init__(self):
        super().__post_init__()
        assert is_iterable(self.symbols)
        assert all(isinstance(var, Expression) for var in self.symbols)
        assert self.interface is None or isinstance(self.interface, (Expression, DataType))

        assert self.external + self.module + self.generic + self.final in (0, 1)

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        return f'ProcedureDeclaration:: {symbols}'


@dataclass_strict(frozen=True)
class _DataDeclarationBase():
    """ Type definitions for :any:`DataDeclaration` node type. """

    # TODO: This should only allow Expression instances but needs frontend changes
    # TODO: Support complex statements (LOKI-23)
    variable: Any
    values: Tuple[Expression, ...]


@dataclass_strict(frozen=True)
class DataDeclaration(LeafNode, _DataDeclarationBase):
    """
    Internal representation of a ``DATA`` declaration for explicit array
    value lists.

    Parameters
    ----------
    variable : :any:`pymbolic.primitives.Expression`
        The left-hand side of the data declaration.
    values : tuple of :any:`pymbolic.primitives.Expression`
        The right-hand side of the data declaration.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variable', 'values']

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.variable, (Expression, str, tuple))
        assert is_iterable(self.values)
        assert all(isinstance(val, Expression) for val in self.values)

    def __repr__(self):
        return f'DataDeclaration:: {str(self.variable)}'


@dataclass_strict(frozen=True)
class _StatementFunctionBase():
    """ Type definitions for :any:`StatementFunction` node type. """

    variable: Expression
    arguments: Tuple[Expression, ...]
    rhs: Expression
    return_type: SymbolAttributes


@dataclass_strict(frozen=True)
class StatementFunction(LeafNode, _StatementFunctionBase):
    """
    Internal representation of Fortran statement function statements

    Parameters
    ----------
    variable : :any:`pymbolic.primitives.Expression`
        The name of the statement function
    arguments : tuple of :any:`pymbolic.primitives.Expression`
        The list of dummy arguments
    rhs : :any:`pymbolic.primitives.Expression`
        The expression defining the statement function
    return_type : :any:`SymbolAttributes`
        The return type of the statement function
    """

    _traversable = ['variable', 'arguments', 'rhs']

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.variable, Expression)
        assert is_iterable(self.arguments) and all(isinstance(a, Expression) for a in self.arguments)
        assert isinstance(self.return_type, SymbolAttributes)

    @property
    def name(self):
        return str(self.variable)

    @property
    def is_function(self):
        return True

    def __repr__(self):
        return f'StatementFunction:: {self.variable}({" ,".join(str(a) for a in self.arguments)})'


@dataclass_strict(frozen=True)
class _TypeDefBase():
    """ Type definitions for :any:`TypeDef` node type. """

    name: Optional[str] = None
    body: Optional[Tuple[Node, ...]] = None
    abstract: bool = False
    extends: Optional[str] = None
    bind_c: bool = False
    private: bool = False
    public: bool = False


@dataclass_strict(frozen=True)
class TypeDef(ScopedNode, InternalNode, _TypeDefBase):
    """
    Internal representation of a derived type definition.

    Similar to :py:class:`Sourcefile`, :py:class:`Module`, and
    :py:class:`Subroutine`, it forms its own scope for symbols and types.
    This is required to instantiate :py:class:`TypedSymbol` instances in
    declarations, imports etc. without having them show up in the enclosing
    scope.

    Parameters
    ----------
    name : str
        The name of the type.
    body : tuple
        The body of the type definition.
    abstract : bool, optional
        Flag to indicate that this is an abstract type definition.
    extends : str, optional
        The parent type name
    bind_c : bool, optional
        Flag to indicate that this contains a ``BIND(C)`` attribute.
    private : bool, optional
        Flag to indicate that this has been declared explicitly as ``PRIVATE``
    public : bool, optional
        Flag to indicate that this has been declared explicitly as ``PUBLIC``
    parent : :any:`Scope`, optional
        The parent scope in which the type definition appears
    symbol_attrs : :any:`SymbolTable`, optional
        An existing symbol table to use
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['body']

    def __post_init__(self, parent=None):
        super(ScopedNode, self).__post_init__(parent=parent)
        super(InternalNode, self).__post_init__()

        # Register this typedef in the parent scope
        if self.parent:
            self.parent.symbol_attrs[self.name] = SymbolAttributes(self.dtype)

    @property
    def ir(self):
        return self.body

    @property
    def parent_type(self):
        if not self.extends:
            return None
        if not self.parent:
            return BasicType.DEFERRED
        parent_type = self.parent.symbol_attrs.lookup(self.extends)
        if not (parent_type and isinstance(parent_type.dtype, DerivedType)):
            return BasicType.DEFERRED
        return parent_type.dtype.typedef

    @property
    def declarations(self):
        decls = tuple(
            c for c in as_tuple(self.body)
            if isinstance(c, (VariableDeclaration, ProcedureDeclaration))
        )

        # Inherit non-overriden symbols from parent type
        if (parent_type := self.parent_type) and parent_type is not BasicType.DEFERRED:
            local_symbols = [s for decl in decls for s in decl.symbols]
            for decl in parent_type.declarations:
                decl_symbols = tuple(s.clone(scope=self) for s in decl.symbols if s not in local_symbols)
                if decl_symbols:
                    decls += (decl.clone(symbols=decl_symbols),)

        return decls

    @property
    def comments(self):
        return tuple(c for c in as_tuple(self.body) if isinstance(c, Comment))

    @property
    def variables(self):
        return tuple(flatten([decl.symbols for decl in self.declarations]))

    @property
    def variable_map(self):
        return CaseInsensitiveDict((s.name, s) for s in self.variables)

    @property
    def imported_symbols(self):
        """
        Return the symbols imported in this typedef
        """
        return tuple(flatten(c.symbols for c in as_tuple(self.body) if isinstance(c, Import)))

    @property
    def imported_symbol_map(self):
        """
        Map of imported symbol names to objects
        """
        return CaseInsensitiveDict((s.name, s) for s in self.imported_symbols)

    def __contains__(self, name):
        """
        Check if a symbol with the given name is declared in this type
        """
        return name in self.variables

    @property
    def interface_symbols(self):
        """
        Return the list of symbols declared via interfaces in this unit

        This returns always an empty tuple since there are no interface declarations
        allowed in typedefs.
        """
        return ()

    @property
    def dtype(self):
        """
        Return the :any:`DerivedType` representing this type
        """
        return DerivedType(name=self.name, typedef=self)

    def __repr__(self):
        return f'TypeDef:: {self.name}'

    def clone(self, **kwargs):
        from loki.ir.transformer import Transformer  # pylint: disable=import-outside-toplevel,cyclic-import
        if 'body' not in kwargs:
            kwargs['body'] = Transformer().visit(self.body)
        return super().clone(**kwargs)


@dataclass_strict(frozen=True)
class _MultiConditionalBase():
    """ Type definitions for :any:`MultiConditional` node type. """

    expr: Expression
    values: Tuple[Any, ...]
    bodies: Tuple[Any, ...]
    else_body: Tuple[Node, ...]
    name: Optional[str] = None


@dataclass_strict(frozen=True)
class MultiConditional(LeafNode, _MultiConditionalBase):
    """
    Internal representation of a multi-value conditional (eg. ``SELECT``).

    Parameters
    ----------
    expr : :any:`pymbolic.primitives.Expression`
        The expression that is evaluated to choose the appropriate case.
    values : tuple of tuple of :any:`pymbolic.primitives.Expression`
        The list of values, a tuple for each case.
    bodies : tuple of tuple
        The corresponding bodies for each case.
    else_body : tuple
        The body for the ``DEFAULT`` case.
    name : str, optional
        The construct-name of the multi conditional in the original source.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['expr', 'values', 'bodies', 'else_body']

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.expr, Expression)
        assert is_iterable(self.values)
        assert all(isinstance(v, tuple) and all(isinstance(c, Expression) for c in v)
                                           for v in self.values)
        assert is_iterable(self.bodies) and all(is_iterable(b) for b in self.bodies)
        assert is_iterable(self.else_body)

    def __repr__(self):
        label = f' {self.name}' if self.name else ''
        return f'MultiConditional::{label} {str(self.expr)}'


@dataclass_strict(frozen=True)
class _ForallBase():
    """ Type definition for :any:`Forall` node type. """

    named_bounds: Tuple[Tuple[Expression, Expression], ...]
    body: Tuple[Node, ...]
    mask: Optional[Expression] = None
    name: Optional[str] = None
    inline: bool = False


@dataclass_strict(frozen=True)
class Forall(InternalNode, _ForallBase):
    """
    Internal representation of a FORALL statement or construct.

    Parameters
    ----------
    named_bounds : tuple of pairs (<variable>, <range>) of type :any:`pymbolic.primitives.Expression`
        The collection of named variables with bounds (ranges).
    body : tuple of :any:`Node`
        The collection of assignment statements, nested FORALLs, and/or comments.
    mask : :any:`pymbolic.primitives.Expression`, optional
        The condition that define the mask.
    name : str, optional
        The name of the multi-line FORALL construct in the original source.
    inline : bool, optional
        Flag to indicate a single-line FORALL statement.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """
    _traversable = ['named_bounds', 'mask', 'body']

    def __post_init__(self):
        super().__post_init__()
        assert is_iterable(self.named_bounds) and all(isinstance(c, tuple) for c in self.named_bounds), \
            "FORALL named bounds must be tuples of <variable, range>"
        assert is_iterable(self.body), "FORALL body must be iterable"
        if self.inline:
            assert len(self.body) == 1, "FORALL statement must contain exactly one assignment"
            assert self.name is None, "FORALL statement cannot have a name label"

    def __repr__(self):
        return f"Forall:: {', '.join([e[0].name for e in self.named_bounds])}"


@dataclass_strict(frozen=True)
class _MaskedStatementBase():
    """ Type definitions for :any:`MaskedStatement` node type. """

    conditions: Tuple[Expression, ...]
    bodies: Tuple[Tuple[Node, ...], ...]
    default: Optional[Tuple[Node, ...]] = None
    inline: bool = False


@dataclass_strict(frozen=True)
class MaskedStatement(LeafNode, _MaskedStatementBase):
    """
    Internal representation of a masked array assignment (``WHERE`` clause).

    Parameters
    ----------
    conditions : tuple of :any:`pymbolic.primitives.Expression`
        The conditions that define the mask
    bodies : tuple of tuple of :any:`Node`
        The conditional assignment statements corresponding to each condition.
    default : tuple of :any:`Node`, optional
        The assignment statements to be executed for array entries not
        captured by the mask (``ELSEWHERE`` statement).
    inline : bool, optional
        Flag to indicate this is a one-line where-stmt
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['conditions', 'bodies', 'default']

    def __post_init__(self):
        super().__post_init__()
        assert is_iterable(self.conditions) and all(isinstance(c, Expression) for c in self.conditions)
        assert is_iterable(self.bodies) and all(isinstance(c, tuple) for c in self.bodies)
        assert len(self.conditions) == len(self.bodies)
        assert is_iterable(self.default)

        if self.inline:
            assert len(self.bodies) == 1 and len(self.bodies[0]) == 1 and not self.default

    def __repr__(self):
        return f'MaskedStatement:: {str(self.conditions[0])}'


@dataclass(frozen=True)
class _IntrinsicBase():
    """ Type definitions for :any:`Intrinsic` node type. """

    text: str


@dataclass_strict(frozen=True)
class Intrinsic(LeafNode, _IntrinsicBase):
    """
    Catch-all generic node for corner-cases.

    This is provided as a fallback for any statements that do not have
    an appropriate representation in the IR. These can either be language
    features for which support was not yet added, or statements that are not
    relevant in Loki's scope of applications. This node retains the text of
    the statement in the original source as-is.

    Parameters
    ----------
    text : str
        The statement as a string.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.text, str)

    def __repr__(self):
        return f'Intrinsic:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class _EnumerationBase():
    """ Type definitions for :any:`Enumeration` node type. """

    symbols: Tuple[Expression, ...]


@dataclass_strict(frozen=True)
class Enumeration(LeafNode, _EnumerationBase):
    """
    Internal representation of an ``ENUM``

    The constants declared by this are represented as :any:`Variable`
    objects with their value (if specified explicitly) stored as the
    ``initial`` property in the symbol's type.

    Parameters
    ----------
    symbols : list of :any:`Expression`
        The named constants declared in this enum
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.symbols is not None:
            assert all(isinstance(s, Expression) for s in self.symbols)  # pylint: disable=not-an-iterable

    def __repr__(self):
        symbols = ', '.join(str(var) for var in as_tuple(self.symbols))
        return f'Enumeration:: {symbols}'


@dataclass_strict(frozen=True)
class _RawSourceBase():
    """ Type definitions for :any:`RawSource` node type. """

    text: str


@dataclass_strict(frozen=True)
class RawSource(LeafNode, _RawSourceBase):
    """
    Generic node for unparsed source code sections

    This is used by the :any:`REGEX` frontend to store unparsed code sections
    in the IR. Currently, they don't serve any other purpose than making sure
    the entire string content of the original Fortran source is retained.

    Parameters
    ----------
    text : str
        The source code as a string.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __repr__(self):
        return f'RawSource:: {truncate_string(self.text.strip())}'
