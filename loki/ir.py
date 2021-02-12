"""
Control flow node classes for :ref:`control-flow-ir`.
"""

from collections import OrderedDict
from itertools import chain
import inspect

from pymbolic.primitives import Expression

from loki.tools import flatten, as_tuple, is_iterable, truncate_string
from loki.types import Scope


__all__ = [
    # Abstract base classes
    'Node', 'InternalNode', 'LeafNode',
    # Internal node classes
    'Section', 'Associate', 'Loop', 'WhileLoop', 'Conditional',
    'MaskedStatement', 'PragmaRegion', 'Interface',
    # Leaf node classes
    'Assignment', 'ConditionalAssignment', 'CallStatement',
    'CallContext', 'Allocation', 'Deallocation', 'Nullify',
    'Comment', 'CommentBlock', 'Pragma', 'PreprocessorDirective',
    'Import', 'Declaration', 'DataDeclaration', 'TypeDef',
    'MultiConditional', 'Intrinsic'
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
        from loki.visitors import pprint  # pylint: disable=import-outside-toplevel
        return pprint(self)

    @property
    def defined_symbols(self):
        """
        Yield the list of symbol names defined at this node.

        This property is attached to the Node by
        :py:func:`loki.analyse.analyse_dataflow.attach_defined_symbols` or
        when using the
        :py:func:`loki.analyse.analyse_dataflow.defined_symbols_attached`
        context manager.
        """
        if not hasattr(self, '_defined_symbols'):
            raise RuntimeError('Need to run "defined_symbols" for the IR first.')
        return self._defined_symbols


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
        self.body = as_tuple(body)

    def __repr__(self):
        raise NotImplementedError


class LeafNode(Node):
    """
    Internal representation of a control flow node without a `body`.
    """

    def __repr__(self):
        raise NotImplementedError


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


class Associate(Section):
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
    """

    _traversable = ['body', 'associations']

    def __init__(self, body=None, associations=None, **kwargs):
        super().__init__(body=body, **kwargs)

        if not isinstance(associations, tuple):
            assert isinstance(associations, (dict, OrderedDict)) or associations is None
            self.associations = as_tuple(associations.items())
        else:
            self.associations = associations

    @property
    def association_map(self):
        """
        An ``OrderedDict`` of associated expressions.
        """
        return OrderedDict(self.associations)

    def __repr__(self):
        if self.associations:
            associations = ', '.join('{}={}'.format(str(var), str(expr))
                                     for var, expr in self.associations)
            return 'Associate:: {}'.format(associations)
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
        control = '{}={}'.format(str(self.variable), str(self.bounds))
        return 'Loop::{} {}'.format(label, control)


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
        return 'WhileLoop::{} {}'.format(label, control)


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
            return 'Conditional:: {}'.format(self.name)
        return 'Conditional::'


class MaskedStatement(InternalNode):
    """
    Internal representation of a masked array assignment (``WHERE`` clause).

    Parameters
    ----------
    condition : :any:`pymbolic.primitives.Expression`
        The condition that defines the mask.
    body : tuple
        The assignment statements.
    default : tuple
        The assignment statements to be executed for array entries not
        captured by the mask (``ELSEWHERE`` statement).
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['condition', 'body', 'default']

    def __init__(self, condition, body, default, **kwargs):
        super().__init__(body=body, **kwargs)

        assert isinstance(condition, Expression)
        assert is_iterable(default)

        self.condition = condition
        self.default = as_tuple(default)  # The ELSEWHERE stmt

    def __repr__(self):
        return 'MaskedStatement:: {}'.format(str(self.condition))


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
    spec : tuple
        The imports, declarations, etc. of the interface block.
    body : tuple
        The body of the interface block.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['body']

    def __init__(self, spec=None, body=None, **kwargs):
        super().__init__(body=body, **kwargs)

        self.spec = spec

    def __repr__(self):
        return 'Interface::'

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
        return 'Assignment:: {} = {}'.format(str(self.lhs), str(self.rhs))


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
        return 'CondAssign:: %s = %s ? %s : %s' % (self.lhs, self.condition, self.rhs,
                                                   self.else_rhs)


class CallStatement(LeafNode):
    """
    Internal representation of a subroutine call.

    Parameters
    ----------
    name : str
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

    _traversable = ['arguments', 'kwarguments']

    def __init__(self, name, arguments, kwarguments=None, context=None, pragma=None, **kwargs):
        super().__init__(**kwargs)

        # TODO: Currently, also simple strings are allowed as arguments. This should be expressions
        arg_types = (Expression, str)
        assert is_iterable(arguments) and all(isinstance(arg, arg_types) for arg in arguments)
        assert kwarguments is None or (
            is_iterable(kwarguments) and all(isinstance(a, tuple) and len(a) == 2 and
                                             isinstance(a[1], arg_types) for a in kwarguments))

        self.name = name
        self.arguments = as_tuple(arguments)
        # kwarguments is kept as a list of tuples!
        self.kwarguments = as_tuple(kwarguments) if kwarguments else ()
        self.context = context
        self.pragma = pragma

    def __repr__(self):
        return 'Call:: {}'.format(self.name)


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
        return 'CallContext:: {}'.format(self.routine.name)


class Allocation(LeafNode):
    """
    Internal representation of a variable allocation.

    Parameters
    ----------
    variables : tuple of :any:`pymbolic.primitives.Expression`
        The list of variables that are allocated.
    data_source : :any:`pymbolic.primitives.Expression` or str
        Fortran's ``SOURCE`` allocation option.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variables']

    def __init__(self, variables, data_source=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)

        self.variables = as_tuple(variables)
        self.data_source = data_source  # Argh, Fortran...!

    def __repr__(self):
        return 'Allocation:: {}'.format(', '.join(str(var) for var in self.variables))


class Deallocation(LeafNode):
    """
    Internal representation of a variable deallocation.

    Parameters
    ----------
    variables : tuple of :any:`pymbolic.primitives.Expression`
        The list of variables that are deallocated.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['variables']

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        self.variables = as_tuple(variables)

    def __repr__(self):
        return 'Deallocation:: {}'.format(', '.join(str(var) for var in self.variables))


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
        return 'Nullify:: {}'.format(', '.join(str(var) for var in self.variables))


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
        return 'Comment:: {}'.format(truncate_string(self.text))


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
        return 'CommentBlock:: {}'.format(truncate_string(string))


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
        return 'Pragma:: {} {}'.format(self.keyword, truncate_string(self.content))


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
        return 'PreprocessorDirective:: {}'.format(truncate_string(self.text))


class Import(LeafNode):
    """
    Internal representation of an import.

    Parameters
    ----------
    module : str
        The name of the module or header file to import from.
    symbols : tuple of :any:`pymbolic.primitives.Expression`, optional
        The list of names imported. Can be empty when importing all.
    c_import : bool, optional
        Flag to indicate that this is a C-style include. Defaults to `False`.
    f_include : bool, optional
        Flag to indicate that this is a preprocessor-style include in
        Fortran source code.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['symbols']

    def __init__(self, module, symbols=None, c_import=False, f_include=False, **kwargs):
        super().__init__(**kwargs)

        self.module = module
        self.symbols = symbols or ()
        self.c_import = c_import
        self.f_include = f_include

        if c_import and f_include:
            raise ValueError('Import cannot be C include and Fortran include')

    def __repr__(self):
        _c = 'C-' if self.c_import else 'F-' if self.f_include else ''
        return '{}Import:: {} => {}'.format(_c, self.module, self.symbols)


class Declaration(LeafNode):
    """
    Internal representation of a variable declaration.

    Parameters
    ----------
    variables : tuple of :any:`pymbolic.primitives.Expression`
        The list of variables declared by this declaration.
    dimensions : tuple of :any:`pymbolic.primitives.Expression`, optional
        The declared allocation size if given as part of the declaration
        attributes.
    external : bool, optional
        ???
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

    _traversable = ['variables', 'dimensions']

    def __init__(self, variables, dimensions=None, external=False,
                 comment=None, pragma=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        assert dimensions is None or (is_iterable(dimensions) and
                                      all(isinstance(d, Expression) for d in dimensions))

        self.variables = as_tuple(variables)
        self.dimensions = as_tuple(dimensions) if dimensions else None
        self.external = external

        self.comment = comment
        self.pragma = pragma

    def __repr__(self):
        variables = ', '.join(str(var) for var in self.variables)
        return 'Declaration:: {}'.format(variables)


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
        return 'DataDeclaration:: {}'.format(str(self.variable))


class TypeDef(LeafNode):
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
    bind_c : bool, optional
        Flag to indicate that this contains a ``BIND(C)`` attribute.
    scope : :any:`Scope`, optional
        An existing scope to use for this type definition. Useful when
        having inserted symbols into the scope already, e.g., while building
        the body.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    _traversable = ['body']

    def __init__(self, name, body, bind_c=False, scope=None, **kwargs):
        super().__init__(**kwargs)
        assert is_iterable(body)

        self.name = name
        self.body = as_tuple(body)
        self.bind_c = bind_c
        self._scope = Scope() if scope is None else scope
        self.scope.defined_by = self

    @property
    def children(self):
        # We do not traverse into the TypeDef.body at present
        return ()

    @property
    def scope(self):
        return self._scope

    @property
    def symbols(self):
        return self.scope.symbols

    @property
    def declarations(self):
        return as_tuple(c for c in self.body if isinstance(c, Declaration))

    @property
    def comments(self):
        return as_tuple(c for c in self.body if isinstance(c, Comment))

    @property
    def variables(self):
        return tuple(flatten([decl.variables for decl in self.declarations]))

    def __repr__(self):
        return 'TypeDef:: {}'.format(self.name)


class MultiConditional(LeafNode):
    """
    Internal representation of a multi-value conditional (eg. ``SELECT``).

    Parameters
    ----------
    expr : :any:`pymbolic.primitives.Expression`
        The expression that is evaluated to choose the appropriate case.
    values : tuple of :any:`pymbolic.primitives.Expression`
        The list of values, one for each case.
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
        assert is_iterable(values) and all(isinstance(v, Expression) for v in flatten(values))
        assert is_iterable(bodies)
        assert is_iterable(else_body)

        self.expr = expr
        self.values = as_tuple(values)
        self.bodies = as_tuple(bodies)
        self.else_body = as_tuple(else_body)
        self.name = name

    def __repr__(self):
        label = ' {}'.format(self.name) if self.name else ''
        return 'MultiConditional::{} {}'.format(label, str(self.expr))


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
        return 'Intrinsic:: {}'.format(truncate_string(self.text))
