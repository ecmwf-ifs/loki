from collections import OrderedDict
from itertools import chain
import inspect

from pymbolic.primitives import Expression

from loki.tools import flatten, as_tuple, is_iterable, truncate_string
from loki.types import Scope


__all__ = [
    'Node', 'Loop', 'Assignment', 'Conditional', 'CallStatement',
    'CallContext', 'Comment', 'CommentBlock', 'Pragma', 'Declaration',
    'TypeDef', 'Section', 'Associate', 'Import', 'Allocation',
    'Deallocation', 'Nullify', 'MaskedStatement', 'MultiConditional',
    'Interface', 'Intrinsic', 'PreprocessorDirective',
    'ConditionalAssignment'
]


class Node:
    # pylint: disable=no-member  # Stop reports about _args

    """
    :attr:`_traversable`. The traversable fields of the Node; that is, fields
    walked over by a :class:`Visitor`. All arguments in __init__ whose name
    appears in this list are treated as traversable fields.
    """
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
        """
        argnames = [i for i in self._traversable if i not in kwargs]
        self._args.update(OrderedDict(zip(argnames, args)))
        self._args.update(kwargs)
        self.__init__(**self._args)

    @property
    def args(self):
        """Arguments used to construct the Node."""
        return self._args.copy()

    @property
    def args_frozen(self):
        """Arguments used to construct the Node that cannot be traversed."""
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    @property
    def source(self):
        return self._source

    @property
    def label(self):
        """Return the statement label of this node."""
        return self._label

    def __repr__(self):
        return 'Node::'


class Intrinsic(Node):
    """
    Catch-all generic node for corner-cases.
    """
    def __init__(self, text=None, **kwargs):
        super().__init__(**kwargs)

        self.text = text

    def __repr__(self):
        return 'Intrinsic:: {}'.format(truncate_string(self.text))


class Comment(Node):
    """
    Internal representation of a single comment line.
    """
    def __init__(self, text=None, **kwargs):
        super().__init__(**kwargs)

        self.text = text

    def __repr__(self):
        return 'Comment:: {}'.format(truncate_string(self.text))


class CommentBlock(Node):
    """
    Internal representation of a block comment.
    """

    def __init__(self, comments, **kwargs):
        super().__init__(**kwargs)

        self.comments = comments

    def __repr__(self):
        string = ''.join(comment.text for comment in self.comments)
        return 'CommentBlock:: {}'.format(truncate_string(string))


class Pragma(Node):
    """
    Internal representation of a pragma
    """

    def __init__(self, keyword, content=None, **kwargs):
        super().__init__(**kwargs)

        self.keyword = keyword
        self.content = content

    def __repr__(self):
        return 'Pragma:: {} {}'.format(self.keyword, truncate_string(self.content))


class PreprocessorDirective(Node):
    """
    Internal representation of a preprocessor directive.
    """

    def __init__(self, text, **kwargs):
        super().__init__(**kwargs)

        self.text = text

    def __repr__(self):
        return 'PreprocessorDirective:: {}'.format(truncate_string(self.text))


class Loop(Node):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines its body.
    """

    _traversable = ['variable', 'bounds', 'body']

    def __init__(self, variable, body=None, bounds=None, pragma=None, pragma_post=None,
                 loop_label=None, name=None, has_end_do=None, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(variable, Expression)
        assert isinstance(bounds, Expression)
        assert is_iterable(body)

        self.variable = variable
        self.body = as_tuple(body)
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


class WhileLoop(Node):
    """
    Internal representation of a while loop in source code.

    Importantly, this is different from a DO or FOR loop, as we don't
    have a dedicated loop variable with bounds.
    """

    _traversable = ['condition', 'body']

    def __init__(self, condition, body=None, pragma=None, pragma_post=None,
                 loop_label=None, name=None, has_end_do=None, **kwargs):
        super().__init__(**kwargs)

        # Unfortunately, unbounded DO ... END DO loops exist and we capture
        # those in this class
        assert isinstance(condition, Expression) or condition is None

        self.condition = condition
        self.body = as_tuple(body)
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


class Conditional(Node):
    """
    Internal representation of a conditional branching construct.
    """

    _traversable = ['conditions', 'bodies', 'else_body']

    def __init__(self, conditions, bodies, else_body, inline=False, name=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(conditions) and all(isinstance(c, Expression) for c in conditions)
        assert is_iterable(bodies) and len(bodies) == len(conditions)

        self.conditions = as_tuple(conditions)
        self.bodies = as_tuple(bodies)
        self.else_body = as_tuple(else_body)
        self.inline = inline
        self.name = name

    def __repr__(self):
        if self.name:
            return 'Conditional:: {}'.format(self.name)
        return 'Conditional::'


class ConditionalAssignment(Node):
    """
    Internal representation of an inline conditional
    """

    _traversable = ['condition', 'lhs', 'rhs', 'else_rhs']

    def __init__(self, lhs, condition, rhs, else_rhs, source=None):
        super().__init__(source=source)

        self.lhs = lhs
        self.condition = condition
        self.rhs = rhs
        self.else_rhs = else_rhs

    def __repr__(self):
        return 'CondAssign:: %s = %s ? %s : %s' % (self.lhs, self.condition, self.rhs,
                                                   self.else_rhs)


class MultiConditional(Node):
    """
    Internal representation of a multi-value conditional (eg. SELECT)
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


class Assignment(Node):
    """
    Internal representation of a variable assignment
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


class MaskedStatement(Node):
    """
    Internal representation of a masked array assignment (WHERE clause).
    """

    _traversable = ['condition', 'body', 'default']

    def __init__(self, condition, body, default, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(condition, Expression)
        assert is_iterable(body)
        assert is_iterable(default)

        self.condition = condition
        self.body = as_tuple(body)
        self.default = as_tuple(default)  # The ELSEWHERE stmt

    def __repr__(self):
        return 'MaskedStatement:: {}'.format(str(self.condition))


class Section(Node):
    """
    Internal representation of a single code region.
    """

    _traversable = ['body']

    def __init__(self, body=None, **kwargs):
        super().__init__(**kwargs)

        self.body = as_tuple(body)

    def append(self, node):
        self._update(body=self.body + as_tuple(node))

    def insert(self, pos, node):
        '''Insert at given position'''
        self._update(body=self.body[:pos] + as_tuple(node) + self.body[pos:])

    def prepend(self, node):
        self._update(body=as_tuple(node) + self.body)

    def __repr__(self):
        return 'Section::'


class Associate(Section):
    """
    Internal representation of a code region with specific properties,
    eg. variable associations.
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


class Declaration(Node):
    """
    Internal representation of a variable declaration
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


class DataDeclaration(Node):
    """
    Internal representation of a DATA declaration for explicit array value lists.
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


class Import(Node):
    """
    Internal representation of a module import.
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


class Interface(Node):
    """
    Internal representation of a Fortran interface block.
    """

    _traversable = ['body']

    def __init__(self, spec=None, body=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(body)

        self.spec = spec
        self.body = as_tuple(body)

    def __repr__(self):
        return 'Interface::'


class Allocation(Node):
    """
    Internal representation of a variable allocation
    """

    _traversable = ['variables']

    def __init__(self, variables, data_source=None, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)

        self.variables = as_tuple(variables)
        self.data_source = data_source  # Argh, Fortran...!

    def __repr__(self):
        return 'Allocation:: {}'.format(', '.join(str(var) for var in self.variables))


class Deallocation(Node):
    """
    Internal representation of a variable deallocation
    """

    _traversable = ['variables']

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        self.variables = as_tuple(variables)

    def __repr__(self):
        return 'Deallocation:: {}'.format(', '.join(str(var) for var in self.variables))


class Nullify(Node):
    """
    Internal representation of a pointer nullification
    """

    _traversable = ['variables']

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        self.variables = as_tuple(variables)

    def __repr__(self):
        return 'Nullify:: {}'.format(', '.join(str(var) for var in self.variables))


class CallStatement(Node):
    """
    Internal representation of a function call
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


class CallContext(Node):
    """
    Special node type to encapsulate the target of a :class:`CallStatement`
    node (usually a :call:`Subroutine`) alongside context-specific
    meta-information. This is required for transformations requiring
    context-sensitive inter-procedural analysis (IPA).
    """

    def __init__(self, routine, active):
        super().__init__()
        self.routine = routine
        self.active = active

    def arg_iter(self, call):
        """
        Iterator that maps argument definitions in the target :class:`Subroutine`
        to arguments and keyword arguments in the :param:`CallStatement` provided.
        """
        r_args = {arg.name: arg for arg in self.routine.arguments}
        args = zip(self.routine.arguments, call.arguments)
        kwargs = ((r_args[kw], arg) for kw, arg in call.kwarguments)
        return chain(args, kwargs)


class TypeDef(Node):
    """
    Internal representation of derived type definition

    Similar to class:`Sourcefile`, class:`Module`,  and class:`Subroutine`, it forms its
    own scope for symbols and types. This is required to instantiate class:`Variable` in
    declarations without having them show up in the enclosing scope.
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
