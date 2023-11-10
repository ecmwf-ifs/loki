# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends

from loki.backend import fgen
from loki.transform import transform_sequence_association
from loki.module import Module
from loki.ir import CallStatement
from loki.visitors import FindNodes

@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_scalar_notation(frontend):
    fcode = """
module mod_a
    implicit none

    type type_b
        integer :: c
        integer :: d
    end type type_b

    type type_a
        type(type_b) :: b
    end type type_a

contains

    subroutine main()

        type(type_a) :: a
        integer :: k, m, n

        real    :: array(10,10)

        call sub_x(array(1, 1), 1)
        call sub_x(array(2, 2), 2)
        call sub_x(array(m, 1), k)
        call sub_x(array(m-1, 1), k-1)
        call sub_x(array(a%b%c, 1), a%b%d)

    contains

        subroutine sub_x(array, k)
    
            integer, intent(in) :: k
            real, intent(in)    :: array(k:n)
    
        end subroutine sub_x

    end subroutine main

end module mod_a
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    routine = module['main']

    transform_sequence_association(routine)

    calls = FindNodes(CallStatement).visit(routine.body)

    assert fgen(calls[0]).lower() == 'call sub_x(array(1:10, 1), 1)'
    assert fgen(calls[1]).lower() == 'call sub_x(array(2:10, 2), 2)'
    assert fgen(calls[2]).lower() == 'call sub_x(array(m:10, 1), k)'
    assert fgen(calls[3]).lower() == 'call sub_x(array(m - 1:10, 1), k - 1)'
    assert fgen(calls[4]).lower() == 'call sub_x(array(a%b%c:10, 1), a%b%d)'
