from collections import OrderedDict
import inspect


__all__ = ['Node', 'Loop', 'Statement', 'Conditional', 'Call',
           'Comment', 'CommentBlock', 'Pragma', 'Declaration', 'Type',
           'DerivedType', 'Variable', 'Expression', 'Index']

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

    def __init__(self, source=None, line=None):
        self._source = source
        self._line = line

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


class Comment(Node):
    """
    Internal representation of a single comment line.
    """
    pass


class CommentBlock(Node):
    """
    Internal representation of a block comment.
    """

    def __init__(self, comments, source=None, line=None):
        super(CommentBlock, self).__init__(source=source, line=line)

        self.comments = comments

class Pragma(Node):
    """
    Internal representation of a EcIR-specific pragma
    """

    def __init__(self, keyword, source=None, line=None):
        super(Pragma, self).__init__(source=source, line=line)

        self.keyword = keyword


class Loop(Node):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines it's body.
    """

    _traversable = ['body']

    def __init__(self, variable, body=None, bounds=None, source=None, line=None):
        super(Loop, self).__init__(source=source, line=line)

        self.variable = variable
        self.body = body
        self.bounds = bounds

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple([self.body])


class Conditional(Node):
    """
    Internal representation of a conditional branching construct.
    """

    _traversable = ['bodies', 'else_body']

    def __init__(self, conditions, bodies, else_body, source=None, line=None):
        super(Conditional, self).__init__(source=source, line=line)

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
    def __init__(self, target, expr, comment=None, source=None, line=None):
        super(Statement, self).__init__(source=source, line=line)

        self.target = target
        self.expr = expr
        self.comment = comment


class Declaration(Node):
    """
    Internal representation of a variable declaration
    """
    def __init__(self, variables, comment=None, pragma=None, source=None, line=None):
        super(Declaration, self).__init__(source=source, line=line)

        self.variables = variables
        self.comment = comment
        self.pragma = pragma


class Allocation(Node):
    """
    Internal representation of a variable allocation
    """
    def __init__(self, variable, source=None, line=None):
        super(Allocation, self).__init__(source=source, line=line)

        self.variable = variable


class Call(Node):
    """
    Internal representation of a function call
    """
    def __init__(self, name, arguments, source=None, line=None):
        super(Call, self).__init__(source=source, line=line)

        self.name = name
        self.arguments = arguments


############################################################
## Utility classes that are not (yet) part of the hierachy
############################################################

class Variable(object):

    def __init__(self, name, type=None, dimensions=None, source=None, line=None):
        self._source = source
        self._line = line

        self.name = name
        self.type = type
        self.dimensions = dimensions

    def __repr__(self):
        idx = '(%s)' % ','.join([str(i) for i in self.dimensions]) if len(self.dimensions) > 0 else ''
        return '%s%s' % (self.name, idx)

    def __key(self):
        return (self.name, self.type, self.dimensions)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        # Allow direct comparison to string and other Variable objects
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Variable):
            return self.__key() == other.__key()
        else:
            self == other


class Type(object):
    """
    Basic type of a variable with type, kind, intent, allocatable, etc.
    """

    def __init__(self, name, kind=None, intent=None, allocatable=False,
                 pointer=False, optional=None, source=None):
        self._source = source

        self.name = name
        self.kind = kind
        self.intent = intent
        self.allocatable = allocatable
        self.pointer = pointer
        self.optional = optional

    def __repr__(self):
        return '<Type %s%s%s%s%s%s>' % (self.name, '(kind=%s)' % self.kind if self.kind else '',
                                        ', intent=%s' % self.intent if self.intent else '',
                                        ', all' if self.allocatable else '',
                                        ', ptr' if self.pointer else '',
                                        ', opt' if self.optional else '')

    def __key(self):
        return (self.name, self.kind, self.intent, self.allocatable, self.pointer, self.optional)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        # Allow direct comparison to string and other Variable objects
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Type):
            return self.__key() == other.__key()
        else:
            self == other


class DerivedType(object):

    def __init__(self, name, variables, comments=None, pragmas=None, source=None):
        self._source = source

        self.name = name
        self.variables = variables
        self.comments = comments
        self.pragmas = pragmas


class Index(object):

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        # Allow direct comparisong to string and other Index objects
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Index):
            return self.name == other.name
        else:
            self == other

    def __repr__(self):
        return '%s' % self.name


class Expression(object):

    def __init__(self, source):
        self.expr = source

    def __repr__(self):
        return '%s' % (self.expr)


class Import(object):

    def __init__(self, module, symbols, source=None):
        self._source = source

        self.module = module
        self.symbols = symbols
