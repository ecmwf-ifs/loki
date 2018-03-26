from enum import Enum

__all__ = ['DataType', 'BaseType', 'DerivedType']


class DataType(Enum):

    LOGICAL = ('LOGICAL', None)  # bool
    JPRM = ('REAL', 'JPRM')  # float32
    JPRB = ('REAL', 'JPRB')  # float64
    JPIM = ('INTEGER', 'JPIM')  # int32

    def __init__(self, type, kind):
        self.type = type
        self.kind = kind


class BaseType(object):
    """
    Basic Fortran variable type with data type, kind, intent, allocatable, etc.
    """

    _base_types = ['REAL', 'INTEGER', 'LOGICAL', 'COMPLEX']

    def __init__(self, name, kind=None, intent=None, allocatable=False, pointer=False,
                 optional=None, parameter=None, target=None, source=None):
        self._source = source

        self.name = name
        self.kind = kind
        self.intent = intent
        self.allocatable = allocatable
        self.pointer = pointer
        self.optional = optional
        self.parameter = parameter
        self.target = target

    def __repr__(self):
        return '<Type %s%s%s%s%s%s%s>' % (
            self.name, '(kind=%s)' % self.kind if self.kind else '',
            ', intent=%s' % self.intent if self.intent else '',
            ', all' if self.allocatable else '',
            ', ptr' if self.pointer else '',
            ', opt' if self.optional else '',
            ', tgt' if self.target else '',
            ', param' if self.parameter else '')

    def __key(self):
        return (self.name, self.kind, self.intent, self.allocatable, self.pointer,
                self.optional, self.parameter)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        # Allow direct comparison to string and other Variable objects
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, BaseType):
            return self.__key() == other.__key()
        else:
            return super(BaseType, self).__eq__(other)

    @property
    def dtype(self):
        return DataType((self.name, self.kind))


class DerivedType(BaseType):
    """
    A compound/derived type that includes base type definitions for
    multiple variables.
    """

    def __init__(self, name, variables, **kwargs):
        super(DerivedType, self).__init__(name=name, **kwargs)
        self.name = name
        self._variables = variables

    def __key(self):
        return (self.name, (v.__key for v in self.variables), self.intent,
                self.allocatable, self.pointer, self.optional)

    def __repr__(self):
        return '<DerivedType %s%s%s%s%s>' % (self.name,
                                             ', intent=%s' % self.intent if self.intent else '',
                                             ', all' if self.allocatable else '',
                                             ', ptr' if self.pointer else '',
                                             ', opt' if self.optional else '')

    @property
    def variables(self):
        """
        Map of variable names to variable objects.
        """
        return {v.name: v for v in self._variables}
