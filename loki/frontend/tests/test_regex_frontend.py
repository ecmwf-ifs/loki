# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct parsing behaviour of the REGEX frontend
"""

from loki.frontend import REGEX
from loki.types import BasicType, DerivedType
from loki.subroutine import Subroutine

def test_declaration_whitespace_attributes():
    """
    Test correct behaviour with/without white space inside declaration attributes
    (reported in #318).
    """
    fcode = """
subroutine my_whitespace_declaration_routine(kdim, state_t0, paux)
    use type_header, only: dimension_type, STATE_TYPE, aux_type, jprb
    implicit none
    TYPE( DIMENSION_TYPE) , INTENT (IN) :: KDIM
    type (state_type  ) , intent ( in ) :: state_t0
    TYPE (AUX_TYPE) , InteNT( In) :: PAUX
    CHARACTER  ( LEN=10) :: STR
    REAL(  KIND = JPRB  ) :: VAR
end subroutine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=REGEX)

    # Verify that variables and dtype information has been extracted correctly
    assert routine.variables == ('kdim', 'state_t0', 'paux', 'str', 'var')
    assert isinstance(routine.variable_map['kdim'].type.dtype, DerivedType)
    assert routine.variable_map['kdim'].type.dtype.name.lower() == 'dimension_type'
    assert isinstance(routine.variable_map['state_t0'].type.dtype, DerivedType)
    assert routine.variable_map['state_t0'].type.dtype.name.lower() == 'state_type'
    assert isinstance(routine.variable_map['paux'].type.dtype, DerivedType)
    assert routine.variable_map['paux'].type.dtype.name.lower() == 'aux_type'
    assert routine.variable_map['str'].type.dtype == BasicType.CHARACTER
    assert routine.variable_map['var'].type.dtype == BasicType.REAL

    routine.make_complete()

    # Verify that additional type attributes are correct after full parse
    assert routine.variables == ('kdim', 'state_t0', 'paux', 'str', 'var')
    assert isinstance(routine.variable_map['kdim'].type.dtype, DerivedType)
    assert routine.variable_map['kdim'].type.dtype.name.lower() == 'dimension_type'
    assert routine.variable_map['kdim'].type.intent == 'in'
    assert isinstance(routine.variable_map['state_t0'].type.dtype, DerivedType)
    assert routine.variable_map['state_t0'].type.dtype.name.lower() == 'state_type'
    assert routine.variable_map['state_t0'].type.intent == 'in'
    assert isinstance(routine.variable_map['paux'].type.dtype, DerivedType)
    assert routine.variable_map['paux'].type.dtype.name.lower() == 'aux_type'
    assert routine.variable_map['paux'].type.intent == 'in'
    assert routine.variable_map['str'].type.dtype == BasicType.CHARACTER
    assert routine.variable_map['str'].type.length == 10
    assert routine.variable_map['var'].type.dtype == BasicType.REAL
    assert routine.variable_map['var'].type.kind == 'jprb'
