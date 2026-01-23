# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of classes to represent type information for symbols used throughout
:doc:`internal_representation`
"""

from enum import Enum

from loki.tools import flatten


__all__ = [
    'DataType', 'BasicType',
    'DEFERRED', 'LOGICAL', 'INTEGER', 'REAL', 'CHARACTER', 'COMPLEX'
]


class DataType:
    """
    Base class for data types a symbol may have
    """


class BasicType(DataType, int, Enum):
    """
    Representation of intrinsic data types, names taken from the FORTRAN convention.

    Currently, there are

    - :any:`LOGICAL`
    - :any:`INTEGER`
    - :any:`REAL`
    - :any:`CHARACTER`
    - :any:`COMPLEX`

    and, to indicate an undefined data type (e.g., for imported
    symbols whose definition is not available), :any:`DEFERRED`.

    For convenience, string representations of FORTRAN and C99 types can be
    heuristically converted.
    """

    DEFERRED = -1
    LOGICAL = 1
    INTEGER = 2
    REAL = 3
    CHARACTER = 4
    COMPLEX = 5

    @classmethod
    def from_str(cls, value):
        """
        Try to convert the given string using one of the `from_*` methods.
        """
        lookup_methods = (cls.from_name, cls.from_fortran_type, cls.from_c99_type)
        for meth in lookup_methods:
            try:
                return meth(value)
            except KeyError:
                pass
        raise ValueError(f'Unknown data type: {value}')

    @classmethod
    def from_name(cls, value):
        """
        Convert the given string representation of the :any:`BasicType`.
        """
        return {t.name: t for t in cls}[value]

    @classmethod
    def from_fortran_type(cls, value):
        """
        Convert the given string representation of a FORTRAN type.
        """
        type_map = {'logical': cls.LOGICAL, 'integer': cls.INTEGER, 'real': cls.REAL,
                    'double precision': cls.REAL, 'double complex': cls.COMPLEX,
                    'character': cls.CHARACTER, 'complex': cls.COMPLEX}
        return type_map[value.lower()]

    @classmethod
    def from_c99_type(cls, value):
        """
        Convert the given string representation of a C99 type.
        """
        logical_types = ['bool', '_Bool']
        integer_types = ['short', 'int', 'long', 'long long']
        integer_types += flatten([(f'signed {t}', f'unsigned {t}') for t in integer_types])
        real_types = ['float', 'double', 'long double']
        character_types = ['char']
        complex_types = ['float _Complex', 'double _Complex', 'long double _Complex']

        type_map = {t: cls.LOGICAL for t in logical_types}
        type_map.update({t: cls.INTEGER for t in integer_types})
        type_map.update({t: cls.REAL for t in real_types})
        type_map.update({t: cls.CHARACTER for t in character_types})
        type_map.update({t: cls.COMPLEX for t in complex_types})

        return type_map[value]


DEFERRED = BasicType.DEFERRED
LOGICAL = BasicType.LOGICAL
INTEGER = BasicType.INTEGER
REAL = BasicType.REAL
CHARACTER = BasicType.CHARACTER
COMPLEX = BasicType.COMPLEX
