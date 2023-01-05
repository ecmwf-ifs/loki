# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki.transform import single_variable_declaration, single_variable_declarations
from loki import Subroutine, OMNI, FindNodes, VariableDeclaration
from loki.expression import symbols as sym


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Makes variable declaration already unique')]))
def test_transform_utilities_single_variable_declaration(frontend):
    """
    Test correct inlining of elemental functions.
    """
    fcode = """
subroutine foo(a, x, y)
    integer, intent(in) :: a
    real, intent(inout):: x(a), y(a, a)
    integer :: i1, i2, i3, i4
    real :: r1, r2, r3, r4
    x = a
    y = a
end subroutine foo
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declaration(routine=routine, variables=('y', 'i1', 'i3', 'r1', 'r2', 'r3', 'r4'))

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert declarations[0].symbols == ('a',)
    assert [smbl.name for smbl in declarations[1].symbols] == ['x']
    assert [smbl.name for smbl in declarations[2].symbols] == ['y']
    assert declarations[3].symbols == ('i2', 'i4')
    assert declarations[4].symbols == ('i1',)
    assert declarations[5].symbols == ('i3',)
    assert declarations[6].symbols == ('r1',)
    assert declarations[7].symbols == ('r2',)
    assert declarations[8].symbols == ('r3',)
    assert declarations[9].symbols == ('r4',)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Makes variable declaration already unique')]))
def test_transform_utilities_single_variable_declarations(frontend):
    """
    Test correct inlining of elemental functions.
    """
    fcode = """
subroutine foo(a, x, y)
    integer, intent(in) :: a
    real, intent(inout):: x(a), y(a, a)
    integer :: i1, i2, i3, i4
    real :: r1, r2, r3, r4
    real :: x1, x2(a), x3(a), x4(a, a)
    x = a
    y = a
end subroutine foo
"""
    # strict = True, meaning all variable declarations to be unique
    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declarations(routine=routine, strict=True)

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert len(declarations) == 15
    for decl in declarations:
        assert len(decl.symbols) == 1

    # strict = False, meaning only non-similar variable declarations unique
    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declarations(routine=routine, strict=False)
    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert len(declarations) == 8
    for decl in declarations:
        types = [smbl.type for smbl in decl.symbols]
        _ = [type == types[0] for type in types]
        assert all(_)
        if isinstance(decl.symbols[0], sym.Array):
            shapes = [smbl.shape for smbl in decl.symbols]
            _ = [shape == shapes[0] for shape in shapes]
            assert all(_)
