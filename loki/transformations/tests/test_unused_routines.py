# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module, Sourcefile
from loki.frontend import OMNI, available_frontends
from loki.ir import FindNodes, nodes as ir
from loki.transformations import SanitiseUnusedRoutineTransformation


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_sanitise_unused_routine_transformation(frontend, tmp_path):
    """Defer kept array shapes, drop local arrays, and replace the body with an error-stop stub."""
    fcode = """
module legacy_unused_mod
  implicit none

contains

  subroutine legacy_unused(nblocks, pout)
    integer, intent(in) :: nblocks
    real, optional, intent(out) :: pout(10, nblocks, 2)
    real, pointer, contiguous :: zoper(:, :, :)
    real, allocatable :: zbuffer(:, :)
    real :: zouts(10, nblocks)

    nullify(zoper)
    zouts = 0.0
    if (present(pout)) pout(1, 1, 1) = 0.0
  end subroutine legacy_unused

  subroutine still_used(a)
    real, intent(inout) :: a(:)
    a = 0.0
  end subroutine still_used
end module legacy_unused_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['legacy_unused']
    untouched = module['still_used']

    trafo = SanitiseUnusedRoutineTransformation(routines=('legacy_unused',), stub_kind='error_stop')
    trafo.apply(routine)

    # Scalar is kept, local array ``zouts`` is dropped, kept arrays are fully deferred
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    assert len(decls) == 4
    assert decls[0].symbols == ('nblocks',)

    assert decls[1].symbols == ('pout(:, :, :)',)
    assert decls[1].symbols[0].type.shape == (':', ':', ':')
    assert decls[1].symbols[0].type.intent == 'out'

    assert decls[2].symbols == ('zoper(:, :, :)',)
    assert decls[2].symbols[0].type.shape == (':', ':', ':')
    assert decls[2].symbols[0].type.pointer

    assert decls[3].symbols == ('zbuffer(:, :)',)
    assert decls[3].symbols[0].type.shape == (':', ':')
    assert decls[3].symbols[0].type.allocatable

    # The body is replaced by a single error-stop stub
    stmts = FindNodes(ir.GenericStmt).visit(routine.body)
    assert len(stmts) == 1
    assert 'error stop "sanitised unused routine legacy_unused was called"' in stmts[0].text.lower()

    # Routines that are not configured are left untouched
    assert not FindNodes(ir.GenericStmt).visit(untouched.body)
    assert len(FindNodes(ir.Assignment).visit(untouched.body)) == 1


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_sanitise_unused_routine_by_qualified_name(frontend, tmp_path):
    """Match configured routines by fully qualified module-and-routine name."""
    fcode = """
module another_legacy_mod
contains
  subroutine keep_me(a)
    real :: a(5)
    a = 1.0
  end subroutine keep_me
end module another_legacy_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['keep_me']

    trafo = SanitiseUnusedRoutineTransformation(
        routines=('another_legacy_mod#keep_me',), stub_kind='error_stop'
    )
    trafo.apply(routine)

    stmts = FindNodes(ir.GenericStmt).visit(routine.body)
    assert len(stmts) == 1
    assert 'error stop "sanitised unused routine keep_me was called"' in stmts[0].text.lower()


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_sanitise_unused_routine_empty_stub(frontend, tmp_path):
    """Allow sanitised routines to use an empty executable section instead of an error-stop stub."""
    fcode = """
module empty_stub_mod
contains
  subroutine legacy_noop(a)
    real, intent(inout) :: a(:)
    a = 2.0
  end subroutine legacy_noop
end module empty_stub_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['legacy_noop']

    trafo = SanitiseUnusedRoutineTransformation(routines=('legacy_noop',), stub_kind='empty')
    trafo.apply(routine)

    assert routine.body.body == ()
    assert not FindNodes(ir.GenericStmt).visit(routine.body)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_sanitise_unused_routine_no_match(frontend, tmp_path):
    """Leave routines unchanged when they are not configured for sanitisation."""
    fcode = """
module no_match_mod
contains
  subroutine active_kernel(a)
    real, intent(inout) :: a(:)
    a = 3.0
  end subroutine active_kernel
end module no_match_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['active_kernel']

    trafo = SanitiseUnusedRoutineTransformation(routines=('other_kernel',), stub_kind='error_stop')
    trafo.apply(routine)

    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1
    assert not FindNodes(ir.GenericStmt).visit(routine.body)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI skips C imports in the frontend')]))
def test_sanitise_unused_routine_removes_c_imports(frontend, tmp_path):
    """Drop C-style include imports while preserving ordinary Fortran imports."""
    filepath = tmp_path/'unused_c_import.F90'
    include_path = tmp_path/'legacy_unused.intfb.h'
    include_path.write_text('! interface intentionally empty\n')
    filepath.write_text(
        """
module helper_mod
  implicit none
  integer, parameter :: rk = kind(1.0)
end module helper_mod

module c_import_unused_mod
contains
  subroutine legacy_unused(a)
    use helper_mod, only: rk
    real(kind=rk), intent(inout) :: a(:)
#include "legacy_unused.intfb.h"
    a = 0.0_rk
  end subroutine legacy_unused
end module c_import_unused_mod
""".strip()
    )
    source = Sourcefile.from_file(filepath, frontend=frontend, includes=[tmp_path], xmods=[tmp_path])
    routine = source['c_import_unused_mod']['legacy_unused']

    before_imports = FindNodes(ir.Import).visit((routine.spec, routine.body))
    assert any(imp.c_import for imp in before_imports)
    assert any(not imp.c_import and imp.module == 'helper_mod' for imp in before_imports)

    trafo = SanitiseUnusedRoutineTransformation(routines=('legacy_unused',), stub_kind='error_stop')
    trafo.apply(routine)

    imports = FindNodes(ir.Import).visit((routine.spec, routine.body))
    assert not any(imp.c_import for imp in imports)
    assert any(not imp.c_import and imp.module == 'helper_mod' for imp in imports)

    stmts = FindNodes(ir.GenericStmt).visit(routine.body)
    assert len(stmts) == 1
    assert 'error stop "sanitised unused routine legacy_unused was called"' in stmts[0].text.lower()


def test_sanitise_unused_routine_rejects_unknown_stub_kind():
    """Reject unsupported stub kinds early during transformation construction."""
    with pytest.raises(ValueError, match='Invalid stub_kind'):
        SanitiseUnusedRoutineTransformation(stub_kind='warn')
