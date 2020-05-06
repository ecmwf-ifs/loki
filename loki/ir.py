from collections import OrderedDict
from itertools import chain
import inspect

from pymbolic.primitives import Expression

from loki.tools import flatten, as_tuple, is_iterable
from loki.types import TypeTable


__all__ = ['Node', 'Loop', 'Statement', 'Conditional', 'CallStatement', 'CallContext',
           'Comment', 'CommentBlock', 'Pragma', 'Declaration', 'TypeDef',
           'Import', 'Allocation', 'Deallocation', 'Nullify', 'MaskedStatement',
           'MultiConditional', 'Interface', 'Intrinsic']


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

    def __init__(self, source=None):
        self._source = source

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
    def children(self):
        return ()


class Intrinsic(Node):
    """
    Catch-all generic node for corner-cases.
    """
    def __init__(self, text=None, source=None):
        super(Intrinsic, self).__init__(source=source)

        self.text = text

    def __repr__(self):
        return 'Intrinsic:: %s' % self.text


class Comment(Node):
    """
    Internal representation of a single comment line.
    """
    def __init__(self, text=None, source=None):
        super(Comment, self).__init__(source=source)

        self.text = text

    def __repr__(self):
        return 'Comment:: ... '


class CommentBlock(Node):
    """
    Internal representation of a block comment.
    """

    def __init__(self, comments, source=None):
        super(CommentBlock, self).__init__(source=source)

        self.comments = comments

    def __repr__(self):
        return 'CommentBlock::'


class Pragma(Node):
    """
    Internal representation of a pragma
    """

    def __init__(self, keyword, content=None, source=None):
        super(Pragma, self).__init__(source=source)

        self.keyword = keyword
        self.content = content


class Loop(Node):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines its body.
    """

    _traversable = ['variable', 'bounds', 'body']

    def __init__(self, variable, body=None, bounds=None, pragma=None,
                 pragma_post=None, label=None, source=None):
        super(Loop, self).__init__(source=source)

        assert isinstance(variable, Expression)
        assert isinstance(bounds, Expression)
        assert is_iterable(body)

        self.variable = variable
        self.body = as_tuple(body)
        self.bounds = bounds
        self.pragma = pragma
        self.pragma_post = pragma_post
        self.label = label

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple((self.variable,) + (self.bounds,) + (self.body,))


class WhileLoop(Node):
    """
    Internal representation of a while loop in source code.

    Importantly, this is different from a DO or FOR loop, as we don't
    have a dedicated loop variable with bounds.
    """

    _traversable = ['condition', 'body']

    def __init__(self, condition, body=None, pragma=None, pragma_post=None,
                 label=None, source=None):
        super(WhileLoop, self).__init__(source=source)

        assert isinstance(condition, Expression)

        self.condition = condition
        self.body = as_tuple(body)
        self.pragma = pragma
        self.pragma_post = pragma_post
        self.label = label

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple((self.condition,) + (self.body,))


class Conditional(Node):
    """
    Internal representation of a conditional branching construct.
    """

    _traversable = ['conditions', 'bodies', 'else_body']

    def __init__(self, conditions, bodies, else_body, inline=False, source=None):
        super(Conditional, self).__init__(source=source)

        assert is_iterable(conditions) and all(isinstance(c, Expression) for c in conditions)
        assert is_iterable(bodies) and len(bodies) == len(conditions)

        self.conditions = as_tuple(conditions)
        self.bodies = as_tuple(bodies)
        self.else_body = as_tuple(else_body)
        self.inline = inline

    @property
    def children(self):
        # Note that we currently ignore the condition itself
        return tuple((self.conditions, ) + (self.bodies, ) + (self.else_body, ))


class MultiConditional(Node):
    """
    Internal representation of a multi-value conditional (eg. SELECT)
    """

    _traversable = ['expr', 'values', 'bodies', 'else_body']

    def __init__(self, expr, values, bodies, else_body, source=None):
        super(MultiConditional, self).__init__(source=source)

        assert isinstance(expr, Expression)
        assert is_iterable(values) and all(isinstance(v, Expression) for v in flatten(values))
        assert is_iterable(bodies)
        assert is_iterable(else_body)

        self.expr = expr
        self.values = as_tuple(values)
        self.bodies = as_tuple(bodies)
        self.else_body = as_tuple(else_body)

    @property
    def children(self):
        return tuple((self.expr,) + (self.values,) + (self.bodies,) + (self.else_body,))


class Statement(Node):
    """
    Internal representation of a variable assignment
    """

    _traversable = ['target', 'expr']

    def __init__(self, target, expr, ptr=False, comment=None, source=None):
        super(Statement, self).__init__(source=source)

        assert isinstance(target, Expression)
        assert isinstance(expr, Expression)

        self.target = target
        self.expr = expr
        self.ptr = ptr  # Marks pointer assignment '=>'
        self.comment = comment

    @property
    def children(self):
        return tuple((self.target,) + (self.expr,))

    def __repr__(self):
        return 'Stmt:: %s = %s' % (self.target, self.expr)


class MaskedStatement(Node):
    """
    Internal representation of a masked array assignment (WHERE clause).
    """

    _traversable = ['condition', 'body', 'default']

    def __init__(self, condition, body, default, source=None):
        super(MaskedStatement, self).__init__(source=source)

        assert isinstance(condition, Expression)
        assert is_iterable(body)
        assert is_iterable(default)

        self.condition = condition
        self.body = as_tuple(body)
        self.default = as_tuple(default)  # The ELSEWHERE stmt

    @property
    def children(self):
        return tuple((self.condition,) + (self.body,) + (self.default,))


class Section(Node):
    """
    Internal representation of a single code region.
    """

    _traversable = ['body']

    def __init__(self, body=None, source=None):
        super(Section, self).__init__(source=source)

        self.body = as_tuple(body)

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple([self.body])

    def append(self, node):
        self._update(body=self.body + as_tuple(node))

    def insert(self, pos, node):
        '''Insert at given position'''
        self._update(body=self.body[:pos] + as_tuple(node) + self.body[pos:])

    def prepend(self, node):
        self._update(body=as_tuple(node) + self.body)


class Scope(Section):
    """
    Internal representation of a code region with specific properties,
    eg. variable associations.
    """

    def __init__(self, body=None, associations=None, source=None):
        super(Scope, self).__init__(body=body, source=source)

        self.associations = associations


class Declaration(Node):
    """
    Internal representation of a variable declaration
    """

    _traversable = ['variables', 'dimensions']

    def __init__(self, variables, dimensions=None, type=None,
                 comment=None, pragma=None, source=None):
        # Stop complaints about `type` in this function
        # pylint: disable=redefined-builtin
        super(Declaration, self).__init__(source=source)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        assert dimensions is None or (is_iterable(dimensions) and
                                      all(isinstance(d, Expression) for d in dimensions))

        self.variables = as_tuple(variables)
        self.dimensions = as_tuple(dimensions) if dimensions else None
        self.type = type

        self.comment = comment
        self.pragma = pragma

    @property
    def children(self):
        return tuple((self.variables,) + (self.dimensions or [],))


class DataDeclaration(Node):
    """
    Internal representation of a DATA declaration for explicit array value lists.
    """

    _traversable = ['variable', 'values']

    def __init__(self, variable, values, source=None):
        super(DataDeclaration, self).__init__(source=source)

        assert isinstance(variable, Expression)
        assert is_iterable(values) and all(isinstance(val, Expression) for val in values)

        self.variable = variable
        self.values = as_tuple(values)

    @property
    def children(self):
        return tuple((self.variable,) + (self.values,))


class Import(Node):
    """
    Internal representation of a module import.
    """
    def __init__(self, module, symbols=None, c_import=False, f_include=False, source=None):
        super(Import, self).__init__(source=source)

        self.module = module
        self.symbols = symbols or ()
        self.c_import = c_import
        self.f_include = f_include

        if c_import and f_include:
            raise ValueError('Import cannot be C include and Fortran include')

    def __repr__(self):
        _c = 'C-' if self.c_import else 'F-' if self.f_include else ''
        return '%sImport:: %s => %s' % (_c, self.module, self.symbols)


class Interface(Node):
    """
    Internal representation of a Fortran interface block.
    """

    _traversable = ['body']

    def __init__(self, spec=None, body=None, source=None):
        super(Interface, self).__init__(source=source)

        assert is_iterable(body)

        self.spec = spec
        self.body = as_tuple(body)

    @property
    def children(self):
        return tuple((self.body,))


class Allocation(Node):
    """
    Internal representation of a variable allocation
    """

    _traversable = ['variables']

    def __init__(self, variables, data_source=None, source=None):
        super(Allocation, self).__init__(source=source)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)

        self.variables = as_tuple(variables)
        self.data_source = data_source  # Argh, Fortran...!

    @property
    def children(self):
        return tuple([self.variables])


class Deallocation(Node):
    """
    Internal representation of a variable deallocation
    """

    _traversable = ['variables']

    def __init__(self, variables, source=None):
        super(Deallocation, self).__init__(source=source)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        self.variables = as_tuple(variables)

    @property
    def children(self):
        return tuple([self.variables])


class Nullify(Node):
    """
    Internal representation of a pointer nullification
    """

    _traversable = ['variables']

    def __init__(self, variables, source=None):
        super(Nullify, self).__init__(source=source)

        assert is_iterable(variables) and all(isinstance(var, Expression) for var in variables)
        self.variables = as_tuple(variables)

    @property
    def children(self):
        return tuple([self.variables])


class CallStatement(Node):
    """
    Internal representation of a function call
    """

    _traversable = ['arguments', 'kwarguments']

    def __init__(self, name, arguments, kwarguments=None, context=None,
                 pragma=None, source=None):
        super(CallStatement, self).__init__(source=source)

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

    @property
    def children(self):
        return tuple((self.arguments,) + as_tuple(a for _, a in self.kwarguments or []))


class CallContext(Node):
    """
    Special node type to encapsulate the target of a :class:`CallStatement`
    node (usually a :call:`Subroutine`) alongside context-specific
    meta-information. This is required for transformations requiring
    context-sensitive inter-procedural analysis (IPA).
    """

    def __init__(self, routine, active):
        super(CallContext, self).__init__()
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

    def __init__(self, name, declarations, bind_c=False, comments=None, pragmas=None, source=None,
                 symbols=None):
        super(TypeDef, self).__init__(source=source)

        assert is_iterable(declarations) and all(isinstance(d, Declaration) for d in declarations)
        assert comments is None or (is_iterable(comments) and
                                    all(isinstance(c, (Comment, CommentBlock)) for c in comments))

        self.name = name
        self.declarations = as_tuple(declarations)
        self.comments = as_tuple(comments) if comments else None
        self.pragmas = pragmas
        self.bind_c = bind_c

        self.symbols = symbols if symbols is not None else TypeTable()

    @property
    def variables(self):
        return tuple(flatten([decl.variables for decl in self.declarations]))
