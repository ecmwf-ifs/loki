# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes

from loki.transformations.sanitise import (
    SequenceAssociationTransformation, do_resolve_sequence_association
)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_trafo', [False, True])
def test_resolve_sequence_assocaition_scalar_notation(tmp_path, frontend, use_trafo):
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

    real :: array(10,10)
    real :: another(1:8, 2)

    ! Test array with scalar dimension
    call sub_x(array(1, 1), 1)
    call sub_x(array(2, 2), 2)
    call sub_x(array(m, 1), k)
    call sub_x(array(m-1, 1), k-1)
    call sub_x(array(a%b%c, 1), a%b%d)

    ! Test array with range dimension
    call sub_x(another(1, 1), 1)
    call sub_x(another(2, 2), 2)
    call sub_x(another(m, 1), k)

  contains

    subroutine sub_x(array, k)
      integer, intent(in) :: k
      real, intent(in)    :: array(k:n)

    end subroutine sub_x

  end subroutine main

end module mod_a
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['main']

    if use_trafo:
        SequenceAssociationTransformation(
            resolve_sequence_associations=True
        ).apply(routine)
    else:
        do_resolve_sequence_association(routine)

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert all(c.name == 'sub_x' for c in calls)

    assert calls[0].arguments == ('array(1:10, 1)', 1)
    assert calls[1].arguments == ('array(2:10, 2)', 2)
    assert calls[2].arguments == ('array(m:10, 1)', 'k')
    assert calls[3].arguments == ('array(m - 1:10, 1)', 'k - 1')
    assert calls[4].arguments == ('array(a%b%c:10, 1)', 'a%b%d')

    assert calls[5].arguments == ('another(1:8, 1)', 1)
    assert calls[6].arguments == ('another(2:8, 2)', 2)
    assert calls[7].arguments == ('another(m:8, 1)', 'k')


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not like not knowing shapes!')]
))
def test_resolve_sequence_assocaition_missing_shape(frontend):
    fcode = """
subroutine test_resolve_seq_assoc_no_shape(a, n)
  use ricks_module, only: my_type
  implicit none

  type(my_type), intent(inout) :: a
  integer, intent(in) :: n

  ! Test array with no known shape
  call sub_x(a%a(1, 1), 1)
  call sub_x(a%b(1), 1)
  call sub_x(a%c, 1)

contains

  subroutine sub_x(array, k)
    integer, intent(in) :: k
    real, intent(in)    :: array(k:n)

  end subroutine sub_x
end subroutine test_resolve_seq_assoc_no_shape
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_resolve_sequence_association(routine)

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert all(c.name == 'sub_x' for c in calls)

    assert calls[0].arguments == ('a%a(:, 1)', 1)
    assert calls[1].arguments == ('a%b(:)', 1)
    assert calls[2].arguments == ('a%c', 1)
