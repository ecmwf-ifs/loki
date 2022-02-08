# pylint: disable=too-many-lines
"""
Control flow node classes for
:ref:`internal_representation:Control flow tree`
"""

from collections import OrderedDict
from itertools import chain
import inspect

from pymbolic.primitives import Expression

from loki.tools import flatten, as_tuple, is_iterable, truncate_string
from loki.types import DataType, DerivedType, SymbolAttributes
from loki.scope import Scope
from loki.tools import CaseInsensitiveDict


__all__ = [
    # Abstract base classes
    'Node', 'InternalNode', 'LeafNode',
    # Internal node classes
    'Section', 'Associate', 'Loop', 'WhileLoop', 'Conditional',
    'PragmaRegion', 'Interface',
    # Leaf node classes
    'Assignment', 'ConditionalAssignment', 'CallStatement',
    'CallContext', 'Allocation', 'Deallocation', 'Nullify',
    'Comment', 'CommentBlock', 'Pragma', 'PreprocessorDirective',
    'Import', 'VariableDeclaration', 'ProcedureDeclaration', 'DataDeclaration',
    'StatementFunction', 'TypeDef', 'MultiConditional', 'MaskedStatement',
    'Intrinsic', 'Enumeration'
]


# Abstract base classes

class Node:
    """
    Base class for all node types in Loki's internal representation.

    Provides the common functionality shared by all node types; specifically,
    this comprises functionality to update or rebuild a node, and source
    metadata.

    Attributes
    -----------
    traversable : list of str
        The traversable fields of the Node; that is, fields walked over by
        a :any:`Visitor`. All arguments in :py:meth:`__init__` whose
        name appear in this list are treated as traversable fields.

    Parameters
    -----------
    source : :any:`Source`, optional
        the information about the original source for the Node.
    label : str, optional
        the label assigned to the statement in the original source
        corresponding to the Node.

    """
    # pylint: disable=no-member  # Stop reports about _args

    _traversable = []

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames = inspect.getfullargspec(cls.__init__).args
        obj._args = dict(zip(argnames[1:], args))
        obj._args.update(kwargs.items())
        obj._args.update({k: None for k in argnames[1:] if k not in obj._args})
        return obj

    def __init__(self, source=None, label=None):
        super().__init__()
        self._source = source
        self._label = label

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
        handle = self._args.copy()  # Original constructor arguments
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
        self._args.update(OrderedDict(zip(argnames, args)))
        self._args.update(kwargs)
        self.__init__(**self._args)

    @property
    def args(self):
        """
        Arguments used to construct the Node.
        """
        return self._args.copy()

    @property
    def args_frozen(self):
        """
        Arguments used to construct the Node that cannot be traversed.
        """
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    @property
    def source(self):
        """
        The :py:class:`loki.frontend.source.Source` object with information
        about the original source for that Node.
        """
        return self._source

    @property
    def label(self):
        """
        Return the statement label of this node.
        """
        return self._label

    def __repr__(self):
        raise NotImplementedError

    def view(self):
        """
        Pretty-print the node hierachy under this node.
        """
        # pylint: disable=import-outside-toplevel
        from loki.visitors import pprint
        pprint(self)

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
        if not hasattr(self, '_live_symbols'):
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self._live_symbols

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
        if not hasattr(self, '_defines_symbols'):
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self._defines_symbols

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
        if not hasattr(self, '_uses_symbols'):
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self._uses_symbols


class InternalNode(Node):
    """
    Internal representation of a control flow node that has a traversable
    `body` property.

    Parameters
    ----------
    body : tuple
        The nodes that make up the body.
    """

    _traversable = ['body']

    def __init__(self, body=None, **kwargs):
        super().__init__(**kwargs)
        self.body = as_tuple(flatten(as_tuple(body)))

    def __repr__(self):
        raise NotImplementedError


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

    def _update(self, *args, **kwargs):
        if 'symbol_attrs' not in kwargs:
            # Retain the symbol table (unless given explicitly)
            kwargs['symbol_attrs'] = self.symbol_attrs
        super()._update(*args, **kwargs)  # pylint: disable=no-member

    def _rebuild(self, *args, **kwargs):
        if 'symbol_attrs' not in kwargs:
            # Retain the symbol table (unless given explicitly)
            kwargs['symbol_attrs'] = self.symbol_attrs
        kwargs['rescope_symbols'] = True
        return super()._rebuild(*args, **kwargs)  # pylint: disable=no-member


# Intermediate node types


class Section(InternalNode):
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
        self._update(body=self.body[:pos] + as_tuple(node) + self.body[pos:])

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


class Associate(ScopedNode, Section):
    """
    Internal representation of a code region in which names are associated
    with expressions or variables.

    Parameters
    ----------
    body : tuple
        The associate's body.
    associations : dict or OrderedDict
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

    def __init__(self, body=None, associations=None, parent=None, symbol_attrs=None, **kwargs):
        if not isinstance(associations, tuple):
            assert isinstance(associations, (dict, OrderedDict)) or associations is None
            self.associations = as_tuple(associations.items())
        else:
            self.associations = associations

        super().__init__(body=body, parent=parent, symbol_attrs=symbol_attrs, **kwargs)

    @property
    def association_map(self):
        """
        An :any:`OrderedDict` of associated expressions.
        """
        return OrderedDict(self.associations)

    @property
    def variables(self):
        return tuple(v for _, v in self.associations)

    def __repr__(self):
        if self.associations:
            associations = ', '.join(f'{str(var)}={str(expr)}'
                                     for var, expr in self.associations)
            return f'Associate:: {associations}'
        return 'Associate::'


class Loop(InternalNode):
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

    def __init__(self, variable, bounds=None, body=None, pragma=None, pragma_post=None,
                 loop_label=None, name=None, has_end_do=None, **kwargs):
        super().__init__(body, **kwargs)

        assert isinstance(variable, Expression)
        assert isinstance(bounds, Expression)

        self.variable = variable
        self.bounds = bounds
        self.pragma = pragma
        self.pragma_post = pragma_post
        self.loop_label = loop_label
        self.name = name
        self.has_end_do = has_end_do if has_end_do is not None else True

    def __repr__(self):
        label = ', '.join(l for l in [self.name, self.loop_label] if l is not None)
        if label:
            label = ' ' + label
        control = f'{str(self.variable)}={str(self.bounds)}'
        return f'Loop::{label} {control}'


class WhileLoop(InternalNode):
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

    def __init__(self, condition, body=None, pragma=None, pragma_post=None,
                 loop_label=None, name=None, has_end_do=None, **kwargs):
        super().__init__(body=body, **kwargs)

        # Unfortunately, unbounded DO ... END DO loops exist and we capture
        # those in this class
        assert isinstance(condition, Expression) or condition is None

        self.condition = condition
        self.pragma = pragma
        self.pragma_post = pragma_post
        self.loop_label = loop_label
        self.name = name
        self.has_end_do = has_end_do if has_end_do is not None else True

    def __repr__(self):
        label = ', '.join(l for l in [self.name, self.loop_label] if l is not None)
        if label:
            label = ' ' + label
        control = str(self.condition) if self.condition else ''
        return f'WhileLoop::{label} {control}'


class Conditional(InternalNode):
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

    def __init__(self, condition, body, else_body, inline=False,
                 has_elseif=False, name=None, **kwargs):
        super().__init__(body=body, **kwargs)

        assert isinstance(condition, Expression)

        else_body = as_tuple(else_body)
        if has_elseif:
            assert len(else_body) == 1 and isinstance(else_body[0], Conditional)

        self.condition = condition
        self.else_body = else_body
        self.inline = inline
        self.has_elseif = has_elseif
        self.name = name

    def __repr__(self):
        if self.name:
            return f'Conditional:: {self.name}'
        return 'Conditional::'


class PragmaRegion(InternalNode):
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

    def __init__(self, body=None, pragma=None, pragma_post=None, **kwargs):
        super().__init__(body=body, **kwargs)

        self.pragma = pragma
        self.pragma_post = pragma_post

    def append(self, node):
        self._update(body=self.body + as_tuple(node))

    def insert(self, pos, node):
        '''Insert at given position'''
        self._update(body=self.body[:pos] + as_tuple(node) + self.body[pos:])

    def prepend(self, node):
        self._update(body=as_tuple(node) + self.body)

    def __repr__(self):
        return 'PragmaRegion::'


class Interface(InternalNode):
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

    def __init__(self, body=None, abstract=False, spec=None, **kwargs):
        super().__init__(body=body, **kwargs)
        self.abstract = abstract
        self.spec = spec
        assert not (self.abstract and self.spec)

    @property
    def symbols(self):
        """
        The list of symbol names declared by this interface
        """
        symbols = as_tuple(flatten(
            getattr(node, 'procedure_symbol', getattr(node, 'symbols', ()))
            for node in self.body
        ))
        if self.spec:
            return (self.spec,) + symbols
        return symbols

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        if self.abstract:
            return f'Abstract Interface:: {symbols}'
        if self.spec:
            return f'Interface {self.spec}:: {symbols}'
        return f'Interface:: {symbols}'

# Leaf node types

class Assignment(LeafNode):
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

    def __init__(self, lhs, rhs, ptr=False, comment=None, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(lhs, Expression)
        assert isinstance(rhs, Expression)

        self.lhs = lhs
        self.rhs = rhs
        self.ptr = ptr  # Marks pointer assignment '=>'
        self.comment = comment

    def __repr__(self):
        return f'Assignment:: {str(self.lhs)} = {str(self.rhs)}'


class ConditionalAssignment(LeafNode):
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

    def __init__(self, lhs, condition, rhs, else_rhs, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(lhs, Expression)
        assert isinstance(condition, Expression)
        assert isinstance(rhs, Expression)
        assert isinstance(else_rhs, Expression)

        self.lhs = lhs
        self.condition = condition
        self.rhs = rhs
        self.else_rhs = else_rhs

    def __repr__(self):
        return f'CondAssign:: {self.lhs} = {self.condition} ? {self.rhs} : {self.else_rhs}'


class CallStatement(LeafNode):
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
    context : :any:`CallContext`
        The information about the called subroutine.
    pragma : tuple of :any:`Pragma`, optional
        Pragma(s) that appear in front of the statement. By default
        :any:`Pragma` nodes appear as standalone nodes in the IR before.
        Only a bespoke context created by :py:func:`pragmas_attached`
        attaches them for convenience.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['name', 'arguments', 'kwarguments']

    def __init__(self, name, arguments, kwarguments=None, context=None, pragma=None, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(name, Expression)
        assert is_iterable(arguments) and all(isinstance(arg, Expression) for arg in arguments)
        assert kwarguments is None or (
            is_iterable(kwarguments) and all(isinstance(a, tuple) and len(a) == 2 and
                                             isinstance(a[1], Expression) for a in kwarguments)
        )

        self.name = name
        self.arguments = as_tuple(arguments)
        # kwarguments is kept as a list of tuples!
        self.kwarguments = as_tuple(kwarguments) if kwarguments else ()
        self.context = context
        self.pragma = pragma

    def __repr__(self):
        return f'Call:: {self.name}'


class CallContext(LeafNode):
    """
    Special node type to encapsulate the target of a :any:`CallStatement`
    node (usually a :any:`Subroutine`) alongside context-specific
    meta-information. This is required for transformations requiring
    context-sensitive inter-procedural analysis (IPA).

    Parameters
    ----------
    routine : :any:`Subroutine`
        The target of the call.
    active : bool
        Flag to indicate if this context is valid.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __init__(self, routine, active, **kwargs):
        super().__init__(**kwargs)
        self.routine = routine
        self.active = active

    def arg_iter(self, call):
        """
        Iterator that maps argument definitions in the target :any:`Subroutine`
        to arguments and keyword arguments in the `call` provided.

        Parameters
        ----------
        call : :any:`CallStatement`
            The call statement to map.

        Returns
        -------
        iterator
            An iterator that traverses the mapping `(arg name, call arg)` for
            all positional and then keyword arguments.
        """
        r_args = {arg.name: arg for arg in self.routine.arguments}
        args = zip(self.routine.arguments, call.arguments)
        kwargs = ((r_args[kw], arg) for kw, arg in call.kwarguments)
        return chain(args, kwargs)

    def __repr__(self):
        return f'CallContext:: {self.routine.name}'


class Allocation(LeafNode):
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

    def __init__(self, variables, data_source=None, status_var=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        assert data_source is None or isinstance(data_source, Expression)
        assert status_var is None or isinstance(status_var, Expression)

        self.variables = as_tuple(variables)
        self.data_source = data_source  # Argh, Fortran...!
        self.status_var = status_var

    def __repr__(self):
        return f'Allocation:: {", ".join(str(var) for var in self.variables)}'


class Deallocation(LeafNode):
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

    def __init__(self, variables, status_var=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        assert status_var is None or isinstance(status_var, Expression)

        self.variables = as_tuple(variables)
        self.status_var = status_var

    def __repr__(self):
        return f'Deallocation:: {", ".join(str(var) for var in self.variables)}'


class Nullify(LeafNode):
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

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        self.variables = as_tuple(variables)

    def __repr__(self):
        return f'Nullify:: {", ".join(str(var) for var in self.variables)}'


class Comment(LeafNode):
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
    def __init__(self, text=None, **kwargs):
        super().__init__(**kwargs)

        self.text = text

    def __repr__(self):
        return f'Comment:: {truncate_string(self.text)}'


class CommentBlock(LeafNode):
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

    def __init__(self, comments, **kwargs):
        super().__init__(**kwargs)

        self.comments = comments

    def __repr__(self):
        string = ''.join(comment.text for comment in self.comments)
        return f'CommentBlock:: {truncate_string(string)}'


class Pragma(LeafNode):
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

    def __init__(self, keyword, content=None, **kwargs):
        super().__init__(**kwargs)

        self.keyword = keyword
        self.content = content

    def __repr__(self):
        return f'Pragma:: {self.keyword} {truncate_string(self.content)}'


class PreprocessorDirective(LeafNode):
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

    def __init__(self, text, **kwargs):
        super().__init__(**kwargs)

        self.text = text

    def __repr__(self):
        return f'PreprocessorDirective:: {truncate_string(self.text)}'


class Import(LeafNode):
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

    def __init__(self, module, symbols=None, nature=None, c_import=False, f_include=False, f_import=False,
                 rename_list=False, **kwargs):
        super().__init__(**kwargs)

        self.module = module
        self.symbols = symbols or ()
        self.nature = nature
        self.c_import = c_import or False
        self.f_include = f_include or False
        self.f_import = f_import or False
        self.rename_list = rename_list

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


class VariableDeclaration(LeafNode):
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

    def __init__(self, symbols, dimensions=None, comment=None, pragma=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(symbols) and all(isinstance(var, Expression) for var in symbols)
        assert dimensions is None or (is_iterable(dimensions) and
                                      all(isinstance(d, Expression) for d in dimensions))

        self.symbols = as_tuple(symbols)
        self.dimensions = as_tuple(dimensions) if dimensions else None

        self.comment = comment
        self.pragma = pragma

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        return f'VariableDeclaration:: {symbols}'


class ProcedureDeclaration(LeafNode):
    """
    Internal representation of a procedure declaration.

    Parameters
    ----------
    symbols : tuple of :any:`pymbolic.primitives.Expression`
        The list of procedure symbols declared by this declaration.
    interface : :any:`pymbolic.primitves.Expression` or :any:`DataType`, optional
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

    def __init__(self, symbols, interface=None, external=False, module=False,
                 generic=False, final=False, comment=None, pragma=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(symbols) and all(isinstance(var, Expression) for var in symbols)
        assert interface is None or isinstance(interface, (Expression, DataType))

        self.symbols = as_tuple(symbols)
        self.interface = interface
        self.external = external or False
        self.module = module or False
        self.generic = generic or False
        self.final = final or False
        self.comment = comment
        self.pragma = pragma

        assert self.external + self.module + self.generic + self.final in (0, 1)

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        return f'ProcedureDeclaration:: {symbols}'


class DataDeclaration(LeafNode):
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

    def __init__(self, variable, values, **kwargs):
        super().__init__(**kwargs)

        # TODO: This should only allow Expression instances but needs frontend changes
        # TODO: Support complex statements (LOKI-23)
        assert isinstance(variable, (Expression, str, tuple))
        assert is_iterable(values) and all(isinstance(val, Expression) for val in values)

        self.variable = variable
        self.values = as_tuple(values)

    def __repr__(self):
        return f'DataDeclaration:: {str(self.variable)}'


class StatementFunction(LeafNode):
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

    def __init__(self, variable, arguments, rhs, return_type, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(variable, Expression)
        assert is_iterable(arguments) and all(isinstance(a, Expression) for a in arguments)
        assert isinstance(return_type, SymbolAttributes)

        self.variable = variable
        self.arguments = as_tuple(arguments)
        self.rhs = rhs
        self.return_type = return_type

    @property
    def name(self):
        return str(self.variable)

    @property
    def is_function(self):
        return True

    def __repr__(self):
        return f'StatementFunction:: {self.variable}({" ,".join(str(a) for a in self.arguments)})'


class TypeDef(ScopedNode, LeafNode):
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

    def __init__(self, name, body, abstract=False, extends=None, bind_c=False,
                 private=False, public=False, parent=None, symbol_attrs=None, **kwargs):
        assert is_iterable(body)
        assert extends is None or isinstance(extends, str)
        assert not (private and public)

        # First, store the local properties
        self.name = name
        self.body = as_tuple(body)
        self.abstract = abstract
        self.extends = extends
        self.bind_c = bind_c
        self.private = private
        self.public = public

        # Then, call the parent constructors to take care of any generic
        # properties and handle the scope information
        super().__init__(parent=parent, symbol_attrs=symbol_attrs, **kwargs)

        # Finally, register this typedef in the parent scope
        if self.parent:
            self.parent.symbol_attrs[self.name] = SymbolAttributes(self.dtype)

    @property
    def declarations(self):
        return as_tuple(c for c in self.body if isinstance(c, VariableDeclaration))

    @property
    def comments(self):
        return as_tuple(c for c in self.body if isinstance(c, Comment))

    @property
    def variables(self):
        return tuple(flatten([decl.symbols for decl in self.declarations]))

    @property
    def imported_symbols(self):
        """
        Return the symbols imported in this typedef
        """
        return as_tuple(flatten(c.symbols for c in self.body if isinstance(c, Import)))

    @property
    def imported_symbol_map(self):
        """
        Map of imported symbol names to objects
        """
        return CaseInsensitiveDict((s.name, s) for s in self.imported_symbols)

    @property
    def dtype(self):
        """
        Return the :any:`DerivedType` representing this type
        """
        return DerivedType(name=self.name, typedef=self)

    def __repr__(self):
        return f'TypeDef:: {self.name}'

    def clone(self, **kwargs):
        from loki.visitors import Transformer  # pylint: disable=import-outside-toplevel
        if 'body' not in kwargs:
            kwargs['body'] = Transformer().visit(self.body)
        return super().clone(**kwargs)


class MultiConditional(LeafNode):
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

    def __init__(self, expr, values, bodies, else_body, name=None, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(expr, Expression)
        assert is_iterable(values) and all(isinstance(v, tuple) and all(isinstance(c, Expression) for c in v)
                                           for v in values)
        assert is_iterable(bodies) and all(isinstance(b, tuple) for b in bodies)
        assert is_iterable(else_body)

        self.expr = expr
        self.values = as_tuple(values)
        self.bodies = as_tuple(bodies)
        self.else_body = as_tuple(else_body)
        self.name = name

    def __repr__(self):
        label = f' {self.name}' if self.name else ''
        return f'MultiConditional::{label} {str(self.expr)}'


class MaskedStatement(LeafNode):
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

    def __init__(self, conditions, bodies, default, inline=False, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(conditions) and all(isinstance(c, Expression) for c in conditions)
        assert is_iterable(bodies) and all(isinstance(c, tuple) for c in bodies)
        assert len(conditions) == len(bodies)
        assert is_iterable(default)

        if inline:
            assert len(bodies) == 1 and len(bodies[0]) == 1 and not default

        self.conditions = as_tuple(conditions)
        self.bodies = as_tuple(bodies)
        self.default = as_tuple(default)
        self.inline = inline or False

    def __repr__(self):
        return f'MaskedStatement:: {str(self.conditions[0])}'


class Intrinsic(LeafNode):
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
    def __init__(self, text=None, **kwargs):
        super().__init__(**kwargs)

        self.text = text

    def __repr__(self):
        return f'Intrinsic:: {truncate_string(self.text)}'


class Enumeration(LeafNode):
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
    def __init__(self, symbols, **kwargs):
        super().__init__(**kwargs)

        self.symbols = as_tuple(symbols)
        assert all(isinstance(s, Expression) for s in self.symbols)

    def __repr__(self):
        symbols = ', '.join(str(var) for var in self.symbols)
        return f'Enumeration:: {symbols}'
