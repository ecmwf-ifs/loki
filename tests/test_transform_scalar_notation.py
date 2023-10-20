# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki.transform import fix_scalar_syntax
from loki.module import Module
from loki.ir import CallStatement
from loki.visitors import FindNodes
from loki.expression import Sum, IntLiteral, Scalar, Product


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

        call sub_a(array(1, 1), k)
        call sub_a(array(2, 2), k)
        call sub_a(array(m, m), k)
        call sub_a(array(m-1, m-1), k)
        call sub_a(array(a%b%c, a%b%c), k)

        call sub_b(array(1, 1))
        call sub_b(array(2, 2))
        call sub_b(array(m, 2))
        call sub_b(array(m-1, m), k)
        call sub_b(array(a%b%c, 2))

        call sub_c(array(1, 1), k)
        call sub_c(array(2, 2), k)
        call sub_c(array(m, 1), k)
        call sub_c(array(m-1, m), k)
        call sub_c(array(a%b%c, 1), k)

        call sub_d(array(1, 1), 1, n)
        call sub_d(array(2, 2), 1, n)
        call sub_d(array(m, 1), k, n)
        call sub_d(array(m-1, 1), k, n-1)
        call sub_d(array(a%b%c, 1), 1, n)

        call sub_e(array(1, 1), a%b)
        call sub_e(array(2, 2), a%b)
        call sub_e(array(m, 1), a%b)
        call sub_e(array(m-1, 1), a%b)
        call sub_e(array(a%b%c, 1), a%b)

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

    subroutine sub_a(array, k)

        integer, intent(in) :: k
        real, intent(in)    :: array(k)

    end subroutine sub_a

    subroutine sub_b(array)

        real, intent(in)    :: array(1:3)

    end subroutine sub_b

    subroutine sub_c(array, k)

        integer, intent(in) :: k
        real, intent(in)    :: array(2:k)

    end subroutine sub_c

    subroutine sub_d(array, k, n)

        integer, intent(in) :: k, n
        real, intent(in)    :: array(k:n)

    end subroutine sub_d

    subroutine sub_e(array, x)

        type(type_b), intent(in) :: x
        real, intent(in)         :: array(x%d)

    end subroutine sub_e

end module mod_a
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    routine = module['main']

    fix_scalar_syntax(routine)

    calls = FindNodes(CallStatement).visit(routine.body)

    one     = IntLiteral(1)
    two     = IntLiteral(2)
    three   = IntLiteral(3)
    four    = IntLiteral(4)
    m_one   = Product((-1,one))
    m_two   = Product((-1,two))
    m_three = Product((-1,three))
    m       = Scalar('m')
    n       = Scalar('n')
    k       = Scalar('k')
    m_k     = Product((-1,k))
    abc     = Scalar(name='a%b%c', parent=Scalar(name='a%b', parent=Scalar('a')))
    abd     = Scalar(name='a%b%d', parent=Scalar(name='a%b', parent=Scalar('a')))
    m_abd   = Product((-1,abd))

    #Check that second dimension is properly added
    assert calls[0].arguments[0].dimensions[1] == one
    assert calls[1].arguments[0].dimensions[1] == two
    assert calls[2].arguments[0].dimensions[1] == m
    assert calls[3].arguments[0].dimensions[1] == Sum((m,m_one))
    assert calls[4].arguments[0].dimensions[1] == abc

    #Check that start of ranges is correct
    assert calls[0].arguments[0].dimensions[0].start == one
    assert calls[1].arguments[0].dimensions[0].start == two
    assert calls[2].arguments[0].dimensions[0].start == m
    assert calls[3].arguments[0].dimensions[0].start == Sum((m,m_one))
    assert calls[4].arguments[0].dimensions[0].start == abc

    #Check that stop of ranges is correct
    #sub_a
    assert calls[0].arguments[0].dimensions[0].stop == k
    assert calls[1].arguments[0].dimensions[0].stop == Sum((k,one))
    assert calls[2].arguments[0].dimensions[0].stop == Sum((k,m,m_one))
    assert calls[3].arguments[0].dimensions[0].stop == Sum((k,m,m_two))
    assert calls[4].arguments[0].dimensions[0].stop == Sum((k,abc,m_one))

    #sub_b
    assert calls[5].arguments[0].dimensions[0].stop == three
    assert calls[6].arguments[0].dimensions[0].stop == four
    assert calls[7].arguments[0].dimensions[0].stop == Sum((m,two))
    assert calls[8].arguments[0].dimensions[0].stop == Sum((m,one))
    assert calls[9].arguments[0].dimensions[0].stop == Sum((abc,two))

    #sub_c
    assert calls[10].arguments[0].dimensions[0].stop == Sum((k,m_one))
    assert calls[11].arguments[0].dimensions[0].stop == k
    assert calls[12].arguments[0].dimensions[0].stop == Sum((k,m,m_two))
    assert calls[13].arguments[0].dimensions[0].stop == Sum((k,m,m_three))
    assert calls[14].arguments[0].dimensions[0].stop == Sum((k,abc,m_two))

    #sub_d
    assert calls[15].arguments[0].dimensions[0].stop == n
    assert calls[16].arguments[0].dimensions[0].stop == Sum((n,one))
    assert calls[17].arguments[0].dimensions[0].stop == Sum((n,m_k,m))
    assert calls[18].arguments[0].dimensions[0].stop == Sum((n,m_k,m,m_two))
    assert calls[19].arguments[0].dimensions[0].stop == Sum((n,abc,m_one))

    #sub_e
    assert calls[20].arguments[0].dimensions[0].stop == abd
    assert calls[21].arguments[0].dimensions[0].stop == Sum((abd,one))
    assert calls[22].arguments[0].dimensions[0].stop == Sum((abd,m,m_one))
    assert calls[23].arguments[0].dimensions[0].stop == Sum((abd,m,m_two))
    assert calls[24].arguments[0].dimensions[0].stop == Sum((abd,abc,m_one))

    #sub_x
    assert calls[25].arguments[0].dimensions[0].stop == n
    assert calls[26].arguments[0].dimensions[0].stop == n
    assert calls[27].arguments[0].dimensions[0].stop == Sum((n,m_k,m))
    assert calls[28].arguments[0].dimensions[0].stop == Sum((n,Product((-1,Sum((k, m_one)))),m,m_one))
    assert calls[29].arguments[0].dimensions[0].stop == Sum((n,m_abd,abc))
