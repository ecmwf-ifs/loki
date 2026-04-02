# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for control-flow IR nodes.
"""

import pytest

from loki import Module
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI fails to read without full module')]))
def test_select_type(frontend, tmp_path):
    fcode = """
module select_type_mod
    use imported_type_mod, only: imported_type
    implicit none
    type, abstract :: base
    end type base
    type, extends(base) :: derived1
        real :: val
    end type derived1
    type, extends(base) :: derived2
        integer :: val
    end type derived2
contains
    subroutine select_type_routine(arg, arg2)
        class(base), intent(inout) :: arg
        class(imported_type), intent(inout) :: arg2
        select type( arg )
            class is(derived1)
                arg%val = 1.0
            class is(derived2)
                arg%val = 1
            class default
                print *, 'error'
        end select
        ! Some comment before the second select
        select type( arg )
            type is(base)
                write(*,*) 'default'
        end select
        select type( arg2 )
            ! inline comment
            type is(imported_type)
                print *, 'imported type'
        end select
    end subroutine select_type_routine
end module select_type_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    tconds = FindNodes(ir.TypeConditional).visit(module['select_type_routine'].body)
    assert len(tconds) == 3

    assert tconds[0].expr == 'arg'
    assert tconds[0].values == (
        ('derived1', True), ('derived2', True)
    )
    assert len(tconds[0].bodies) == 2
    assert len(tconds[0].else_body) == 1

    assert tconds[1].expr == 'arg'
    assert tconds[1].values == (('base', False),)
    assert not tconds[1].else_body

    assert tconds[2].expr == 'arg2'
    assert tconds[2].values == (('imported_type', False),)
    assert not tconds[2].else_body

    comments = FindNodes(ir.Comment).visit(module['select_type_routine'].body)
    assert len(comments) == 2
    assert 'Some comment' in comments[0].text
    assert 'inline comment' in comments[1].text
