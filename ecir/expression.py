from abc import ABCMeta, abstractproperty

__all__ = ['Expression', 'Operation', 'Literal', 'Variable', 'Index', 'Type', 'DerivedType']


class Expression(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def expr(self):
        """Symbolic representation - might be used for cde generation"""
        pass

    @abstractproperty
    def type(self):
        """Data type of (sub-)expression"""
        pass

    def __repr__(self):
        return self.expr


class Operation(Expression):

    def __init__(self, op, operands, parenthesis=False):
        self.op = op
        self.operands = operands
        self.parenthesis = parenthesis

    @property
    def expr(self):
        operands = (' %s ' % self.op).join(self.operands)
        return '(%s)' % operands if self.parenthesis else operands

    @property
    def type(self):
        # TODO
        return None


class Literal(Expression):

    def __init__(self, value, type=None):
        self.value = value
        self._type = type

    @property
    def expr(self):
        return self.value

    @property
    def type(self):
        return self._value


class Variable(Expression):

    def __init__(self, name, type=None, dimensions=None, source=None, line=None):
        self._source = source
        self._line = line

        self.name = name
        self._type = type
        self.dimensions = dimensions or ()

    @property
    def expr(self):
        idx = '(%s)' % ','.join([str(i) for i in self.dimensions]) if len(self.dimensions) > 0 else ''
        return '%s%s' % (self.name, idx)

    @property
    def type(self):
        return self._type

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


class Index(Expression):

    def __init__(self, name):
        self.name = name

    @property
    def expr(self):
        return '%s' % self.name

    @property
    def type(self):
        # TODO: Some common form of `INT`, maybe?
        return None

    def __eq__(self, other):
        # Allow direct comparisong to string and other Index objects
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Index):
            return self.name == other.name
        else:
            self == other

############################################################
####  Type hiearchy
############################################################

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
