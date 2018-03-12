from abc import ABCMeta, abstractproperty
from enum import Enum

__all__ = ['Expression', 'Operation', 'Literal', 'Variable', 'Index', 'FType', 'DerivedType']


class Expression(object):

    __metaclass__ = ABCMeta

    _traversable = []

    @abstractproperty
    def expr(self):
        """
        Symbolic representation - might be used in this raw form
        for code generation.
        """
        pass

    @abstractproperty
    def type(self):
        """
        Data type of (sub-)expressions.

        Note, that this is the pure data type (eg. int32, float64),
        not the full variable declaration type (allocatable, pointer,
        etc.). This is so that we may reason about it recursively.
        """
        pass

    def __repr__(self):
        return self.expr


class Operation(Expression):

    _traversable = ['operands']

    def __init__(self, op, operands, parenthesis=False):
        self.op = op
        self.operands = operands
        self.parenthesis = parenthesis

    @property
    def expr(self):
        operands = ('%s' % self.op).join(str(o) for o in self.operands)
        return '(%s)' % operands if self.parenthesis else operands

    @property
    def type(self):
        types = [o.type for o in self.operands]
        assert(all(types == types[0]))
        return types[0]


class Literal(Expression):

    def __init__(self, value, type=None):
        self.value = value
        self._type = type

    @property
    def expr(self):
        return self.value

    @property
    def type(self):
        return self._type


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

class DataType(Enum):

    LOGICAL = ('LOGICAL', None)  # bool
    JPRM = ('REAL', 'JPRM')  # float32
    JPRB = ('REAL', 'JPRB')  # float64
    JPIM = ('INTEGER', 'JPIM')  # int32

    def __init__(self, type, kind):
        self.type = type
        self.kind = kind


class FType(object):
    """
    Basic Fortran variable type with data type, kind, intent, allocatable, etc.
    """

    _base_types = ['REAL', 'INTEGER', 'LOGICAL', 'COMPLEX']

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
        return '<Type %s%s%s%s%s%s>' % (self.type.type, '(kind=%s)' % self.type.kind if self.type.kind else '',
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

    @property
    def dtype(self):
        return DataType((self.name, self.kind))


class DerivedType(object):

    def __init__(self, name, variables, comments=None, pragmas=None, source=None):
        self._source = source

        self.name = name
        self.variables = variables
        self.comments = comments
        self.pragmas = pragmas
