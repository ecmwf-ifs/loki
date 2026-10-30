# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Generic Fortran intrinsic procedures and the read-only scope in which they
are declared
"""

from dataclasses import dataclass, field

from loki.types.procedure_type import ProcedureType
from loki.types.scope import Scope
from loki.types.symbol_table import SymbolTable, SymbolAttributes


__all__ = [
    'GENERIC_INTRINSIC_FUNCTIONS', 'GENERIC_INTRINSIC_SUBROUTINES',
    'GENERIC_INTRINSIC_PROCEDURES', 'IntrinsicSymbolTable', 'IntrinsicScope',
    'INTRINSIC_SCOPE'
]


GENERIC_INTRINSIC_FUNCTIONS = (
    'ABS', 'ACHAR', 'ACOS', 'ACOSH', 'ADJUSTL', 'ADJUSTR', 'AIMAG', 'AINT', 'ALL',
    'ALLOCATED', 'ANINT', 'ANY', 'ASIN', 'ASINH', 'ASSOCIATED', 'ATAN', 'ATAN2',
    'ATANH', 'BESSEL_J0', 'BESSEL_J1', 'BESSEL_JN', 'BESSEL_Y0', 'BESSEL_Y1',
    'BESSEL_YN', 'BGE', 'BGT', 'BIT_SIZE', 'BLE', 'BLT', 'BTEST', 'CEILING', 'CHAR',
    'CMPLX', 'COMMAND_ARGUMENT_COUNT', 'CONJG', 'COS', 'COSH', 'COUNT', 'CSHIFT',
    'DBLE', 'DIGITS', 'DIM', 'DOT_PRODUCT', 'DPROD', 'DSHIFTL', 'DSHIFTR',
    'EOSHIFT', 'EPSILON', 'ERF', 'ERFC', 'ERFC_SCALED', 'EXP', 'EXPONENT',
    'EXTENDS_TYPE_OF', 'FINDLOC', 'FLOOR', 'FRACTION', 'GAMMA', 'HUGE', 'HYPOT',
    'IACHAR', 'IALL', 'IAND', 'IANY', 'IBCLR', 'IBITS', 'IBSET', 'ICHAR', 'IEOR',
    'IMAGE_INDEX', 'INDEX', 'INT', 'IOR', 'IPARITY', 'ISHFT', 'ISHFTC',
    'IS_CONTIGUOUS', 'IS_IOSTAT_END', 'IS_IOSTAT_EOR', 'KIND', 'LBOUND',
    'LCOBOUND', 'LEADZ', 'LEN', 'LEN_TRIM', 'LGE', 'LGT', 'LLE', 'LLT', 'LOG',
    'LOG_GAMMA', 'LOG10', 'LOGICAL', 'MASKL', 'MASKR', 'MATMUL', 'MAX',
    'MAXEXPONENT', 'MAXLOC', 'MAXVAL', 'MERGE', 'MERGE_BITS', 'MIN',
    'MINEXPONENT', 'MINLOC', 'MINVAL', 'MOD', 'MODULO', 'NEAREST', 'NEW_LINE',
    'NINT', 'NORM2', 'NOT', 'NULL', 'NUM_IMAGES', 'PACK', 'PARITY', 'POPCNT',
    'POPPAR', 'PRECISION', 'PRESENT', 'PRODUCT', 'RADIX', 'RANGE', 'REAL',
    'REPEAT', 'RESHAPE', 'RRSPACING', 'SAME_TYPE_AS', 'SCALE', 'SCAN',
    'SELECTED_CHAR_KIND', 'SELECTED_INT_KIND', 'SELECTED_REAL_KIND',
    'SET_EXPONENT', 'SHAPE', 'SHIFTA', 'SHIFTL', 'SHIFTR', 'SIGN', 'SIN', 'SINH',
    'SIZE', 'SPACING', 'SPREAD', 'SQRT', 'STORAGE_SIZE', 'SUM', 'TAN', 'TANH',
    'THIS_IMAGE', 'TINY', 'TRAILZ', 'TRANSFER', 'TRANSPOSE', 'TRIM', 'UBOUND',
    'UCOBOUND', 'UNPACK', 'VERIFY',
)
"""Names of the generic intrinsic functions in the Fortran 2008 standard"""

GENERIC_INTRINSIC_SUBROUTINES = (
    'ATOMIC_DEFINE', 'ATOMIC_REF', 'CPU_TIME', 'DATE_AND_TIME',
    'EXECUTE_COMMAND_LINE', 'GET_COMMAND', 'GET_COMMAND_ARGUMENT',
    'GET_ENVIRONMENT_VARIABLE', 'MOVE_ALLOC', 'MVBITS', 'RANDOM_NUMBER',
    'RANDOM_SEED', 'SYSTEM_CLOCK',
)
"""Names of the intrinsic subroutines in the Fortran 2008 standard"""

GENERIC_INTRINSIC_PROCEDURES = GENERIC_INTRINSIC_FUNCTIONS + GENERIC_INTRINSIC_SUBROUTINES
"""Names of all generic intrinsic procedures in the Fortran 2008 standard"""


class IntrinsicSymbolTable(SymbolTable):
    """
    Read-only :any:`SymbolTable` that declares the generic Fortran intrinsic
    procedures

    It serves as the final fallback for recursive look-ups in any
    :any:`SymbolTable`, so that intrinsic procedures are known in every
    scope without being declared there. Declarations in user scopes take
    precedence over intrinsic procedures of the same name.
    """

    def _populate(self):
        """
        Declare the intrinsic procedures on first access

        This is deferred until first use, because :any:`ProcedureType`
        cannot be instantiated while the type system is being imported.
        """
        if self.__dict__.get('_populated'):
            return
        self.__dict__['_populated'] = True
        for name in GENERIC_INTRINSIC_PROCEDURES:
            proc_type = ProcedureType(
                name=name, is_function=name in GENERIC_INTRINSIC_FUNCTIONS,
                is_intrinsic=True, procedure=None
            )
            dict.__setitem__(
                self, self.format_lookup_name(name),
                SymbolAttributes(dtype=proc_type, is_intrinsic=True)
            )

    def _lookup_formatted_name(self, name, recursive):
        self._populate()
        value = dict.get(self, self.format_lookup_name(name))
        return value.clone() if value is not None else None

    def __contains__(self, key):
        self._populate()
        return super().__contains__(key)

    def __iter__(self):
        self._populate()
        return super().__iter__()

    def __len__(self):
        self._populate()
        return super().__len__()

    def keys(self):
        self._populate()
        return super().keys()

    def items(self):
        self._populate()
        return super().items()

    def values(self):
        self._populate()
        return super().values()

    def _read_only(self, *args, **kwargs):
        raise TypeError('[Loki::IntrinsicSymbolTable] The intrinsic symbol table is read-only')

    __setitem__ = _read_only
    __delitem__ = _read_only
    setdefault = _read_only
    update = _read_only
    pop = _read_only
    popitem = _read_only
    clear = _read_only

    def __reduce__(self):
        return 'INTRINSIC_SYMBOL_TABLE'


@dataclass(frozen=True)
class IntrinsicScope(Scope):
    """
    Read-only :any:`Scope` in which the generic Fortran intrinsic procedures
    are declared

    There is a single instance :data:`INTRINSIC_SCOPE`, which is used as the
    final fallback when looking up symbols in any other scope.
    """

    symbol_attrs: SymbolTable = field(default_factory=IntrinsicSymbolTable, init=False)

    def __repr__(self):
        return 'IntrinsicScope'

    def __reduce__(self):
        return 'INTRINSIC_SCOPE'


INTRINSIC_SCOPE = IntrinsicScope()
"""The read-only :any:`IntrinsicScope` for generic Fortran intrinsic procedures"""

INTRINSIC_SYMBOL_TABLE = INTRINSIC_SCOPE.symbol_attrs
"""The read-only :any:`IntrinsicSymbolTable` of :data:`INTRINSIC_SCOPE`"""
