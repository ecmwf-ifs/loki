# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module
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

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    pout_decl = next(decl for decl in decls if any(sym.name.lower() == 'pout' for sym in decl.symbols))
    zoper_decl = next(decl for decl in decls if any(sym.name.lower() == 'zoper' for sym in decl.symbols))
    zbuffer_decl = next(decl for decl in decls if any(sym.name.lower() == 'zbuffer' for sym in decl.symbols))
    pout = next(sym for sym in pout_decl.symbols if sym.name.lower() == 'pout')
    zoper = next(sym for sym in zoper_decl.symbols if sym.name.lower() == 'zoper')
    zbuffer = next(sym for sym in zbuffer_decl.symbols if sym.name.lower() == 'zbuffer')
    declared_names = {sym.name.lower() for decl in decls for sym in decl.symbols}
    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)
    untouched_intrinsics = FindNodes(ir.Intrinsic).visit(untouched.body)

    assert pout.type.shape == (':', ':', ':')
    assert pout.dimensions == (':', ':', ':')
    assert zoper.type.shape == (':', ':', ':')
    assert zoper.dimensions == (':', ':', ':')
    assert zbuffer.type.shape == (':', ':')
    assert zbuffer.dimensions == (':', ':')
    assert 'zouts' not in declared_names
    assert len(intrinsics) == 1
    assert 'error stop "sanitised unused routine legacy_unused was called"' in intrinsics[0].text.lower()
    assert not untouched_intrinsics
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

    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)
    assert len(intrinsics) == 1
    assert 'error stop "sanitised unused routine keep_me was called"' in intrinsics[0].text.lower()


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
    assert not FindNodes(ir.Intrinsic).visit(routine.body)


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

    assignments = FindNodes(ir.Assignment).visit(routine.body)
    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)

    assert len(assignments) == 1
    assert not intrinsics


def test_sanitise_unused_routine_rejects_unknown_stub_kind():
    """Reject unsupported stub kinds early during transformation construction."""
    with pytest.raises(ValueError, match='Invalid stub_kind'):
        SanitiseUnusedRoutineTransformation(stub_kind='warn')
