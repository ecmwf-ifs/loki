from collections import OrderedDict
import inspect

from loki.tools import flatten


__all__ = ['Node', 'Loop', 'Statement', 'Conditional', 'Call', 'Comment',
           'CommentBlock', 'Pragma', 'Declaration', 'TypeDef', 'Import']


class Node(object):

    """
    :attr:`_traversable`. The traversable fields of the Node; that is, fields
    walked over by a :class:`Visitor`. All arguments in __init__ whose name
    appears in this list are treated as traversable fields.
    """
    _traversable = []

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames = inspect.getargspec(cls.__init__).args
        obj._args = {k: v for k, v in zip(argnames[1:], args)}
        obj._args.update(kwargs.items())
        obj._args.update({k: None for k in argnames[1:] if k not in obj._args})
        return obj

    def __init__(self, source=None):
        self._source = source

    def _rebuild(self, *args, **kwargs):
        handle = self._args.copy()  # Original constructor arguments
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict([(k, v) for k, v in zip(argnames, args)]))
        handle.update(kwargs)
        return type(self)(**handle)

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
    pass


class Comment(Node):
    """
    Internal representation of a single comment line.
    """
    pass


class CommentBlock(Node):
    """
    Internal representation of a block comment.
    """

    def __init__(self, comments, source=None):
        super(CommentBlock, self).__init__(source=source)

        self.comments = comments


class Pragma(Node):
    """
    Internal representation of a EcIR-specific pragma
    """

    def __init__(self, keyword, source=None):
        super(Pragma, self).__init__(source=source)

        self.keyword = keyword


class Loop(Node):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines it's body.
    """

    _traversable = ['body']

    def __init__(self, variable, body=None, bounds=None, pragma=None,
                 source=None):
        super(Loop, self).__init__(source=source)

        self.variable = variable
        self.body = body
        # Ensure three-entry tuple
        self.bounds = tuple(bounds[i] if len(bounds) > i else None
                            for i in range(3))
        self.pragma = pragma

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple([self.body])


class Conditional(Node):
    """
    Internal representation of a conditional branching construct.
    """

    _traversable = ['bodies', 'else_body']

    def __init__(self, conditions, bodies, else_body, source=None):
        super(Conditional, self).__init__(source=source)

        self.conditions = conditions
        self.bodies = bodies
        self.else_body = else_body

    @property
    def children(self):
        # Note that we currently ignore the condition itself
        return tuple(tuple([self.bodies]) + tuple([self.else_body]))


class Statement(Node):
    """
    Internal representation of a variable assignment
    """
    def __init__(self, target, expr, ptr=False, comment=None, source=None):
        super(Statement, self).__init__(source=source)

        self.target = target
        self.expr = expr
        self.ptr = ptr  # Marks pointer assignment '=>'
        self.comment = comment


class Scope(Node):
    """
    Internal representation of a code region with specific properties,
    eg. variable associations.
    """

    _traversable = ['body']

    def __init__(self, body=None, associations=None, source=None):
        super(Scope, self).__init__(source=source)

        self.body = body
        self.associations = associations

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple([self.body])


class Declaration(Node):
    """
    Internal representation of a variable declaration
    """
    def __init__(self, variables, comment=None, pragma=None, source=None):
        super(Declaration, self).__init__(source=source)

        self.variables = variables
        self.comment = comment
        self.pragma = pragma


class Import(Node):
    """
    Internal representation of a module import.
    """
    def __init__(self, module, symbols, source=None):
        super(Import, self).__init__(source=source)

        self.module = module
        self.symbols = symbols


class Allocation(Node):
    """
    Internal representation of a variable allocation
    """
    def __init__(self, variable, source=None):
        super(Allocation, self).__init__(source=source)

        self.variable = variable


class Deallocation(Node):
    """
    Internal representation of a variable deallocation
    """
    def __init__(self, variable, source=None):
        super(Deallocation, self).__init__(source=source)

        self.variable = variable


class Call(Node):
    """
    Internal representation of a function call
    """
    def __init__(self, name, arguments, kwarguments=None, pragma=None, source=None):
        super(Call, self).__init__(source=source)

        self.name = name
        self.arguments = arguments
        self.kwarguments = kwarguments
        self.pragma = pragma


class TypeDef(Node):
    """
    Internal representation of derived type definition
    """

    def __init__(self, name, declarations, comments=None, pragmas=None, source=None):
        super(TypeDef, self).__init__(source=source)

        self.name = name
        self.declarations = declarations
        self.comments = comments
        self.pragmas = pragmas

    @property
    def variables(self):
        return tuple(flatten([decl.variables for decl in self.declarations]))
