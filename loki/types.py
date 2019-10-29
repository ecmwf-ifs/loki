from enum import IntEnum

__all__ = ['BaseType', 'DerivedType', 'DataType']


class DataType(IntEnum):
    """
    Raw data type with conversion and detection mechanisms.
    """

    BOOL = 1
    INT32 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    CHARACTER = 5

    @classmethod
    def from_type_kind(cls, type, kind):
        """
        Detect raw data type from OMNI XcodeML node.
        """
        type_kind_map = {
            ('logical', None): cls.BOOL,

            ('integer', None): cls.INT32,
            ('integer', '4'): cls.INT32,
            ('integer', 'jpim'): cls.INT32,
            ('integer', 'c_int'): cls.INT32,

            ('real', 'real32'): cls.FLOAT32,
            ('real', 'c_float'): cls.FLOAT32,

            ('real', 'real64'): cls.FLOAT64,
            ('real', 'c_double'): cls.FLOAT64,
            ('real', 'jprb'): cls.FLOAT64,
            ('real', 'selected_real_kind(13,300)'): cls.FLOAT64,
            ('real', 'selected_real_kind(13, 300)'): cls.FLOAT64,

            ('character', None): cls.CHARACTER,
        }
        type = type if type is None else str(type).lower()
        kind = kind if kind is None else str(kind).lower()
        return type_kind_map.get((type, kind), None)

    @classmethod
    def from_omni(cls, node):
        """
        Detect raw data type from OMNI XcodeML node.
        """
        raise NotImplementedError()

    @property
    def ctype(self):
        """
        String representing the C equivalent of this data type.
        """
        map = {
            self.BOOL: 'int', self.INT32: 'int',
            self.FLOAT32: 'float', self.FLOAT64: 'double',
            self.CHARACTER: 'char'
        }
        return map.get(self, None)

    @property
    def isoctype(self):
        """
        String representing the C equivalent of this data type.
        """
        map = {
            self.BOOL: BaseType('logical', kind='4'),
            self.INT32: BaseType('integer', kind='c_int'),
            self.FLOAT32: BaseType('real', kind='c_float'),
            self.FLOAT64: BaseType('real', kind='c_double'),
            self.CHARACTER: BaseType('character', kind='c_char')
        }
        return map.get(self, None)

    @property
    def ftype(self):
        """
        String representing the Fortran aequivalent of this data type.
        """
        raise NotImplementedError()


class BaseType(object):
    """
    Basic variable type with raw data type and Fortran attributes like
    ``intent``, ``allocatable``, ``pointer``, etc.
    """

    # TODO: Funnel this through the raw data type above
    _base_types = ['REAL', 'INTEGER', 'LOGICAL', 'COMPLEX', 'CHARACTER']

    # TODO: Funnel this through the raw data type above
    _omni_types = {
        'Fint': 'INTEGER',
        'Freal': 'REAL',
        'Flogical': 'LOGICAL',
        'Fcharacter': 'CHARACTER',
    }

    def __init__(self, name, kind=None, intent=None, allocatable=False, pointer=False,
                 optional=None, parameter=None, target=None, contiguous=None, value=None,
                 source=None):
        self._source = source

        self.name = name
        self.kind = kind
        self.intent = intent
        self.allocatable = allocatable
        self.pointer = pointer
        self.optional = optional
        self.parameter = parameter
        self.target = target
        self.contiguous = contiguous
        self.value = value

    def __repr__(self):
        return '<Type %s%s%s%s%s%s%s%s%s>' % (
            self.name,
            '(kind=%s)' % self.kind if self.kind else '',
            ', all' if self.allocatable else '',
            ', ptr' if self.pointer else '',
            ', opt' if self.optional else '',
            ', tgt' if self.target else '',
            ', contig' if self.contiguous else '',
            ', param' if self.parameter else '',
            ', intent=%s' % self.intent if self.intent else '',
        )

    def __key(self):
        return (self.name, self.kind, self.intent, self.allocatable, self.pointer,
                self.optional, self.target, self.contiguous, self.parameter)

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
        return DataType.from_type_kind(self.name, self.kind)


class DerivedType(BaseType):
    """
    A compound/derived type that includes base type definitions for
    multiple variables.
    """

    def __init__(self, name, variables, **kwargs):
        super(DerivedType, self).__init__(name=name, **kwargs)
        self.name = name
        self.variables = variables

    def __key(self):
        return (self.name, (v.__key for v in self.variables), self.intent,
                self.allocatable, self.pointer, self.optional)

    def __repr__(self):
        return '<DerivedType %s%s%s%s%s>' % (self.name,
                                             ', intent=%s' % self.intent if self.intent else '',
                                             ', all' if self.allocatable else '',
                                             ', ptr' if self.pointer else '',
                                             ', opt' if self.optional else '')
