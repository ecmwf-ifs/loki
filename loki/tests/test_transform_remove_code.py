# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import shutil
import pytest

from loki import (
    Subroutine, Module, Sourcefile, FindNodes, OMNI, gettempdir
)
from loki.batch import Scheduler, SchedulerConfig
from loki.frontend import available_frontends
from loki.ir import nodes as ir
from loki.transform import (
    do_remove_dead_code, do_remove_marked_regions, do_remove_calls,
    RemoveCodeTransformation
)


@pytest.fixture(scope='module', name='srcdir')
def fixture_srcdir():
    """
    Create a src directory in the temp directory
    """
    srcdir = gettempdir()/'test_remove_code'
    if srcdir.exists():
        shutil.rmtree(srcdir)
    srcdir.mkdir()
    yield srcdir
    shutil.rmtree(srcdir)


@pytest.fixture(scope='module', name='source')
def fixture_source(srcdir):
    """
    Write some source files to use in the test
    """
    fcode_driver = """
subroutine rick_astley
    use parkind1, only: jprb
    use yomhook, only : lhook, dr_hook
    use rick_rolled, only : never_gonna_give
    implicit none

    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('rick_astley',0,zhook_handle)
    call never_gonna_give()
    if (lhook) call dr_hook('rick_astley',1,zhook_handle)
end subroutine
    """.strip()

    fcode_kernel = """
module rick_rolled
contains
subroutine never_gonna_give
    use parkind1, only: jprb
    use yomhook, only : lhook, dr_hook
    implicit none

    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('never_gonna_give',0,zhook_handle)

    CALL ABOR1('[SUBROUTINE CALL]')

    print *, 'never gonna let you down'

    if (dave) call abor1('[INLINE CONDITIONAL]')

    call never_gonna_run_around()

    WRITE(NULOUT,*) "[WRITE INTRINSIC]"
    if (.not. dave) WRITE(NULOUT, *) "[WRITE INTRINSIC]"

    if (lhook) call dr_hook('never_gonna_give',1,zhook_handle)

contains

subroutine never_gonna_run_around

    implicit none

    if (lhook) call dr_hook('never_gonna_run_around',0,zhook_handle)

    if (dave) call abor1('[INLINE CONDITIONAL]')
    WRITE(NULOUT,*) "[WRITE INTRINSIC]"
    if (.not. dave) WRITE(NULOUT, *) "[WRITE INTRINSIC]"

    if (lhook) call dr_hook('never_gonna_run_around',1,zhook_handle)

end subroutine never_gonna_run_around

end subroutine
subroutine i_hope_you_havent_let_me_down
    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('i_hope_you_havent_let_me_down',0,zhook_handle)

    if (lhook) call dr_hook('i_hope_you_havent_let_me_down',1,zhook_handle)
end subroutine i_hope_you_havent_let_me_down
end module rick_rolled
    """.strip()

    (srcdir/'rick_astley.F90').write_text(fcode_driver)
    (srcdir/'never_gonna_give.F90').write_text(fcode_kernel)

    yield srcdir

    (srcdir/'rick_astley.F90').unlink()
    (srcdir/'never_gonna_give.F90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_dead_code_conditional(frontend):
    """
    Test correct elimination of unreachable conditional branches.
    """
    fcode = """
subroutine test_dead_code_conditional(a, b, flag)
  real(kind=8), intent(inout) :: a, b
  logical, intent(in) :: flag

  if (flag) then
    if (1 == 6) then
      a = a + b
    else
      b = b + 2.0
    end if

    if (2 == 2) then
      b = b + a
    else
      a = a + 3.0
    end if

    if (1 == 2) then
      b = b + a
    elseif (3 == 3) then
      a = a + b
    else
      a = a + 6.0
    end if

  end if
end subroutine test_dead_code_conditional
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Please note that nested conditionals (elseif) counts as two
    assert len(FindNodes(ir.Conditional).visit(routine.body)) == 5
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 7

    do_remove_dead_code(routine)

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == 1
    assert conditionals[0].condition == 'flag'
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'b' and assigns[0].rhs == 'b + 2.0'
    assert assigns[1].lhs == 'b' and assigns[1].rhs == 'b + a'
    assert assigns[2].lhs == 'a' and assigns[2].rhs == 'a + b'


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_dead_code_conditional_nested(frontend):
    """
    Test correct elimination of unreachable branches in nested conditionals.
    """
    fcode = """
subroutine test_dead_code_conditional(a, b, flag)
  real(kind=8), intent(inout) :: a, b
  logical, intent(in) :: flag

  if (1 == 2) then
    a = a + 5
  elseif (flag) then
    b = b + 4
  else
    b = a + 3
  end if

  if (a > 2.0) then
    a = a + 5.0
  elseif (2 == 3) then
    a = a + 3.0
  else
    a = a + 1.0
  endif

  if (a > 2.0) then
    a = a + 5.0
  elseif (2 == 3) then
    a = a + 3.0
  elseif (a > 1.0) then
    a = a + 2.0
  else
    a = a + 1.0
  endif
end subroutine test_dead_code_conditional
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Please note that nested conditionals (elseif) counts as two
    assert len(FindNodes(ir.Conditional).visit(routine.body)) == 7
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 10

    do_remove_dead_code(routine)

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == 4
    assert conditionals[0].condition == 'flag'
    assert not conditionals[0].has_elseif
    assert conditionals[1].condition == 'a > 2.0'
    assert not conditionals[1].has_elseif
    assert conditionals[2].condition == 'a > 2.0'
    if not frontend == OMNI:  # OMNI does not get elseifs right
        assert conditionals[2].has_elseif
    assert conditionals[3].condition == 'a > 1.0'
    assert not conditionals[3].has_elseif
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 7
    assert assigns[0].lhs == 'b' and assigns[0].rhs == 'b + 4'
    assert assigns[1].lhs == 'b' and assigns[1].rhs == 'a + 3'
    assert assigns[2].lhs == 'a' and assigns[2].rhs == 'a + 5.0'
    assert assigns[3].lhs == 'a' and assigns[3].rhs == 'a + 1.0'
    assert assigns[4].lhs == 'a' and assigns[4].rhs == 'a + 5.0'
    assert assigns[5].lhs == 'a' and assigns[5].rhs == 'a + 2.0'
    assert assigns[6].lhs == 'a' and assigns[6].rhs == 'a + 1.0'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('mark_with_comment', [True, False])
def test_transform_remove_code_pragma_region(frontend, mark_with_comment):
    """
    Test correct removal of pragma-marked code regions.
    """
    fcode = """
subroutine test_remove_code(a, b, n, flag)
  real(kind=8), intent(inout) :: a, b(n)
  integer, intent(in) :: n
  logical, intent(in) :: flag
  integer :: i

  if (flag) then
    a = a + 1.0
  end if

  !$loki remove
  do i=1, n
    !$loki rick-roll
    a = a + 3.0
    !$loki end rick-roll
  end do
  !$loki end remove

  b(:) = 1.0

  !$acc parallel
  do i=1, n
    b(i) = b(i) + a

    !$loki remove
    a = b(i) + 42.
    !$loki end remove
  end do
end subroutine test_remove_code
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_remove_marked_regions(routine, mark_with_comment=mark_with_comment)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'a' and assigns[0].rhs == 'a + 1.0'
    assert assigns[1].lhs == 'b(:)' and assigns[1].rhs == '1.0'
    assert assigns[2].lhs == 'b(i)' and assigns[2].rhs == 'b(i) + a'

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1
    assert assigns[2] in loops[0].body

    comments = [
        c for c in FindNodes(ir.Comment).visit(routine.body)
        if '[Loki] Removed content' in c.text
    ]
    assert len(comments) == (2 if mark_with_comment else 0)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('remove_imports', [True, False])
def test_transform_remove_calls(frontend, remove_imports):
    """
    Test removal of utility calls and intrinsics with custom patterns.
    """

    fcode_yomhook = """
module yomhook
  logical lhook
contains
  subroutine dr_hook(name, id, handle)
    character(len=*), intent(in) :: name
    integer(kind=8), intent(in) :: id, handle
  end subroutine dr_hook
end module yomhook
    """

    fcode_abor1 = """
module abor1_mod
implicit none
contains
  subroutine abor1(msg)
    character(len=*), intent(in) :: msg
    write(*,*) msg
  end subroutine abor1
end module abor1_mod
    """

    fcode = """
subroutine never_gonna_give(dave)
    use yomhook, only : lhook, dr_hook
    use abor1_mod, only : abor1
    implicit none

    integer(kind=8), parameter :: NULOUT = 6
    integer, parameter :: jprb = 8
    logical, intent(in) :: dave
    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('never_gonna_give',0,zhook_handle)

    CALL ABOR1('[SUBROUTINE CALL]')

    print *, 'never gonna let you down'

    if (dave) call abor1('[INLINE CONDITIONAL]')

    call never_gonna_run_around()

    WRITE(NULOUT,*) "[WRITE INTRINSIC]"
    if (.not. dave) WRITE(NULOUT, *) "[WRITE INTRINSIC]"

    if (lhook) call dr_hook('never_gonna_give',1,zhook_handle)

end subroutine
    """

    # Parse utility module first, to get type info for OMNI
    _ = Module.from_source(fcode_yomhook, frontend=frontend)
    _ = Module.from_source(fcode_abor1, frontend=frontend)

    # Parse the main test function and remove calls
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Note that OMNI enforces keyword-arg passing for intrinsic
    # call to ``write``, so we match both conventions.
    do_remove_calls(
        routine, call_names=('ABOR1', 'DR_HOOK'),
        intrinsic_names=('WRITE(NULOUT', 'write(unit=nulout'),
        remove_imports=remove_imports
    )

    # Check that all but one specific call have been removed
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].name == 'never_gonna_run_around'

    # OMNI resolves inline-conditionals and expands the keyword-args,
    # so neither the inline-conditional removal, nor the intrinsic
    # matching works with it.
    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == (4 if frontend == OMNI else 0)

    # Check that all intrinsic calls to WRITE have been removed
    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)
    assert len(intrinsics) == 1
    assert 'never gonna let you down' in intrinsics[0].text

    # Check that the repsective imports have also been stripped
    imports = FindNodes(ir.Import).visit(routine.spec)
    assert len(imports) == 1 if remove_imports else 2
    assert imports[0].module == 'yomhook'
    if remove_imports:
        assert imports[0].symbols == ('lhook',)
    else:
        assert imports[0].symbols == ('lhook', 'dr_hook')
        assert imports[1].module == 'abor1_mod'
        assert imports[1].symbols == ('abor1',)


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'Incomplete source tree impossible with OMNI')]
))
@pytest.mark.parametrize('include_intrinsics', (True, False))
@pytest.mark.parametrize('kernel_only', (True, False))
def test_remove_code_transformation(frontend, source, include_intrinsics, kernel_only):
    """
    Test the use of code removal utilities, in particular the call
    removal, via the scheduler.
    """

    config = {
        'default': {
            'role': 'kernel', 'expand': True, 'strict': False,
            'disable': ['dr_hook', 'abor1']
        },
        'routines': {
            'rick_astley': {'role': 'driver'},
        }
    }
    scheduler_config = SchedulerConfig.from_dict(config)
    scheduler = Scheduler(paths=source, config=scheduler_config, frontend=frontend)

    # Apply the transformation to the call tree
    transformation = RemoveCodeTransformation(
        call_names=('ABOR1', 'DR_HOOK'),
        intrinsic_names=('WRITE(NULOUT',) if include_intrinsics else (),
        kernel_only=kernel_only
    )
    scheduler.process(transformation=transformation)

    routine = scheduler['rick_rolled#never_gonna_give'].ir
    transformed = routine.to_fortran()

    assert '[SUBROUTINE CALL]' not in transformed
    assert '[INLINE CONDITIONAL]' not in transformed
    assert ('dave' not in transformed) == include_intrinsics
    assert ('[WRITE INTRINSIC]' not in transformed) == include_intrinsics

    for r in routine.members:
        transformed = r.to_fortran()
        assert '[SUBROUTINE CALL]' not in transformed
        assert '[INLINE CONDITIONAL]' not in transformed
        assert ('dave' not in transformed) == include_intrinsics

    routine = Sourcefile.from_file(
        source/'never_gonna_give.F90', frontend=frontend
    )['i_hope_you_havent_let_me_down']
    assert 'zhook_handle' in routine.variables
    assert len([call for call in FindNodes(ir.CallStatement).visit(routine.body) if call.name == 'dr_hook']) == 2

    driver = scheduler['#rick_astley'].ir
    drhook_calls = [call for call in FindNodes(ir.CallStatement).visit(driver.body) if call.name == 'dr_hook']
    assert len(drhook_calls) == (2 if kernel_only else 0)
