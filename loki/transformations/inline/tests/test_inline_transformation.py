# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module, Subroutine
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.batch import Scheduler, SchedulerConfig

from loki.transformations.inline import InlineTransformation


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('pass_as_kwarg', (False, True))
def test_inline_transformation(tmp_path, frontend, pass_as_kwarg):
    """Test combining recursive inlining via :any:`InliningTransformation`."""

    fcode_module = """
module one_mod
  real(kind=8), parameter :: one = 1.0
end module one_mod
"""

    fcode_inner = """
subroutine add_one_and_two(a)
  use one_mod, only: one
  implicit none

  real(kind=8), intent(inout) :: a

  a = a + one

  a = add_two(a)

contains
  elemental function add_two(x)
    real(kind=8), intent(in) :: x
    real(kind=8) :: add_two

    add_two = x + 2.0
  end function add_two
end subroutine add_one_and_two
"""

    fcode = f"""
subroutine test_inline_pragma(a, b)
  implicit none
  real(kind=8), intent(inout) :: a(3), b(3)
  integer, parameter :: n = 3
  integer :: i
  real :: stmt_arg
  real :: some_stmt_func
  some_stmt_func ( stmt_arg ) = stmt_arg + 3.1415

#include "add_one_and_two.intfb.h"

  do i=1, n
    !$loki inline
    call add_one_and_two({'a=' if pass_as_kwarg else ''}a(i))
  end do

  do i=1, n
    !$loki inline
    call add_one_and_two({'a=' if pass_as_kwarg else ''}b(i))
  end do

  a(1) = some_stmt_func({'stmt_arg=' if pass_as_kwarg else ''}a(2))

end subroutine test_inline_pragma
"""
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    inner = Subroutine.from_source(fcode_inner, definitions=module, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine.enrich(inner)

    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_stmt_funcs=True
    )

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert all(c.routine == inner for c in calls)

    # Apply to the inner subroutine first to resolve parameter and calls
    trafo.apply(inner)

    assigns = FindNodes(ir.Assignment).visit(inner.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'a' and assigns[0].rhs == 'a + 1.0'
    assert assigns[1].lhs == 'result_add_two' and assigns[1].rhs == 'a + 2.0'
    assert assigns[2].lhs == 'a' and assigns[2].rhs == 'result_add_two'

    # Apply to the outer routine, but with resolved body of the inner
    trafo.apply(routine)

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 0
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 7
    assert assigns[0].lhs == 'a(i)' and assigns[0].rhs == 'a(i) + 1.0'
    assert assigns[1].lhs == 'result_add_two' and assigns[1].rhs == 'a(i) + 2.0'
    assert assigns[2].lhs == 'a(i)' and assigns[2].rhs == 'result_add_two'
    assert assigns[3].lhs == 'b(i)' and assigns[3].rhs == 'b(i) + 1.0'
    assert assigns[4].lhs == 'result_add_two' and assigns[4].rhs == 'b(i) + 2.0'
    assert assigns[5].lhs == 'b(i)' and assigns[5].rhs == 'result_add_two'
    assert assigns[6].lhs == 'a(1)' and assigns[6].rhs == 'a(2) + 3.1415'



@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_local_seq_assoc(frontend, tmp_path):
    fcode = """
module somemod
    implicit none
    contains

    subroutine minusone_second(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x(3)
        output = x(2) - 1
    end subroutine minusone_second

    subroutine plusone(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x
        output = x + 1
    end subroutine plusone

    subroutine outer()
      implicit none
      real :: x(3, 3)
      real :: y
      x = 10.0

      call inner(y, x(1, 1)) ! Sequence association tmp_path for member routine.

      !$loki inline
      call plusone(y, x(3, 3)) ! Marked for inlining.

      call minusone_second(y, x(1, 3)) ! Standard call with sequence association (never processed).

      contains

      subroutine inner(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x(3)

        output = x(2) + 2.0
      end subroutine inner
    end subroutine outer

end module somemod
"""
    # Test case that nothing happens if `resolve_sequence_association=True`
    # but inlining "marked" and "internals" is disabled.
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=False, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' in callnames
    assert 'inner' in callnames
    assert 'minusone_second' in callnames

    # Test case that only marked processed if
    # `resolve_sequence_association=True`
    # `inline_marked=True`,
    # `inline_internals=False`
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' not in callnames
    assert 'inner' in callnames
    assert 'minusone_second' in callnames

    # Test case that a crash occurs if sequence association is not enabled even if it is needed.
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=True, resolve_sequence_association=False
    )
    outer = module["outer"]
    with pytest.raises(RuntimeError):
        trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]

    # Test case that sequence association is run and corresponding call inlined, avoiding crash.
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=False, inline_internals=True, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' in callnames
    assert 'inner' not in callnames
    assert 'minusone_second' in callnames

    # Test case that everything is enabled.
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=True, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' not in callnames
    assert 'inner' not in callnames
    assert 'minusone_second' in callnames


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_local_seq_assoc_crash_marked_no_seq_assoc(frontend, tmp_path):
    # Test case that a crash occurs if marked routine with sequence association is
    # attempted to inline without sequence association enabled.
    fcode = """
module somemod
    implicit none
    contains

    subroutine inner(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x(3)

        output = x(2) + 2.0
    end subroutine inner

    subroutine outer()
      real :: x(3, 3)
      real :: y
      x = 10.0

      !$loki inline
      call inner(y, x(1, 1)) ! Sequence association tmp_path for marked routine.
    end subroutine outer

end module somemod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=False
    )
    outer = module["outer"]
    with pytest.raises(RuntimeError):
        trafo.apply(outer)

    # Test case that crash is avoided by activating sequence association.
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    assert len(FindNodes(ir.CallStatement).visit(outer.body)) == 0

@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_local_seq_assoc_crash_value_err_no_source(frontend, tmp_path):
    # Testing that ValueError is thrown if sequence association is requested with inlining,
    # but source code behind call is missing (not enough type information).
    fcode = """
module somemod
    implicit none
    contains

    subroutine outer()
      real :: x(3, 3)
      real :: y
      x = 10.0

      !$loki inline
      call inner(y, x(1, 1)) ! Sequence association tmp_path for marked routine.
    end subroutine outer

end module somemod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    with pytest.raises(ValueError):
        trafo.apply(outer)


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_adjust_imports(frontend, tmp_path):
    fcode_module = """
module bnds_module
  integer :: m
  integer :: n
  integer :: l
end module bnds_module
    """

    fcode_another = """
module another_module
  integer :: x
end module another_module
    """

    fcode_outer = """
subroutine test_inline_outer(a, b)
  use bnds_module, only: n
  use test_inline_mod, only: test_inline_inner
  use test_inline_another_mod, only: test_inline_another_inner
  implicit none

  real(kind=8), intent(inout) :: a(n), b(n)

  !$loki inline
  call test_inline_another_inner()
  !$loki inline
  call test_inline_inner(a, b)
end subroutine test_inline_outer
    """

    fcode_inner = """
module test_inline_mod
  implicit none
  contains

subroutine test_inline_inner(a, b)
  use BNDS_module, only: n, m
  use another_module, only: x

  real(kind=8), intent(inout) :: a(n), b(n)
  real(kind=8) :: tmp(m)
  integer :: i

  tmp(1:m) = x
  do i=1, n
    a(i) = b(i) + sum(tmp)
  end do
end subroutine test_inline_inner
end module test_inline_mod
    """

    fcode_another_inner = """
module test_inline_another_mod
  implicit none
  contains

subroutine test_inline_another_inner()
  use BNDS_module, only: n, m, l

end subroutine test_inline_another_inner
end module test_inline_another_mod
    """

    _ = Module.from_source(fcode_another, frontend=frontend, xmods=[tmp_path])
    _ = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    inner = Module.from_source(fcode_inner, frontend=frontend, xmods=[tmp_path])
    another_inner = Module.from_source(fcode_another_inner, frontend=frontend, xmods=[tmp_path])
    outer = Subroutine.from_source(
        fcode_outer, definitions=(inner, another_inner), frontend=frontend, xmods=[tmp_path]
    )

    trafo = InlineTransformation(
        inline_elementals=False, inline_marked=True, adjust_imports=True
    )
    trafo.apply(outer)

    # Check that the inlining has happened
    assign = FindNodes(ir.Assignment).visit(outer.body)
    assert len(assign) == 2
    assert assign[0].lhs == 'tmp(1:m)'
    assert assign[0].rhs == 'x'
    assert assign[1].lhs == 'a(i)'
    assert assign[1].rhs == 'b(i) + sum(tmp)'

    # Now check that the right modules have been moved,
    # and the import of the call has been removed
    imports = FindNodes(ir.Import).visit(outer.spec)
    assert len(imports) == 2
    assert imports[0].module == 'another_module'
    assert imports[0].symbols == ('x',)
    assert imports[1].module == 'bnds_module'
    assert all(_ in imports[1].symbols for _ in ['l', 'm', 'n'])


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_intermediate(tmp_path, frontend):
    fcode_outermost = """
module outermost_mod
implicit none
contains
subroutine outermost()
use intermediate_mod, only: intermediate

!$loki inline
call intermediate()

end subroutine outermost
end module outermost_mod
"""

    fcode_intermediate = """
module intermediate_mod
implicit none
contains
subroutine intermediate()
use innermost_mod, only: innermost

call innermost()

end subroutine intermediate
end module intermediate_mod
"""

    fcode_innermost = """
module innermost_mod
implicit none
contains
subroutine innermost()

end subroutine innermost
end module innermost_mod
"""

    (tmp_path/'outermost_mod.F90').write_text(fcode_outermost)
    (tmp_path/'intermediate_mod.F90').write_text(fcode_intermediate)
    (tmp_path/'innermost_mod.F90').write_text(fcode_innermost)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routines': {
            'outermost': {'role': 'kernel'}
        }
    }

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path]
    )

    def _get_successors(item):
        return scheduler.sgraph.successors(scheduler[item])

    # check graph edges before transformation
    assert len(scheduler.items) == 3
    assert len(_get_successors('outermost_mod#outermost')) == 1
    assert scheduler['intermediate_mod#intermediate'] in _get_successors('outermost_mod#outermost')
    assert len(_get_successors('intermediate_mod#intermediate')) == 1
    assert scheduler['innermost_mod#innermost'] in _get_successors('intermediate_mod#intermediate')

    scheduler.process( transformation=InlineTransformation() )

    # check graph edges were updated correctly
    assert len(scheduler.items) == 2
    assert len(_get_successors('outermost_mod#outermost')) == 1
    assert scheduler['innermost_mod#innermost'] in _get_successors('outermost_mod#outermost')
