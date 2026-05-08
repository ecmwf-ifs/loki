# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pytest

from loki import Module
from loki.frontend import OMNI, available_frontends
from loki.ir import FindNodes, nodes as ir
from loki.transformations import SanitiseUnusedRoutineTransformation


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_sanitise_unused_routine_transformation(frontend, tmp_path):
    fcode = """
module legacy_unused_mod
  implicit none

contains

  subroutine legacy_unused(nblocks, pout)
    integer, intent(in) :: nblocks
    real, optional, intent(out) :: pout(10, nblocks, 2)
    real, pointer, contiguous :: zoper(:, :, :)
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

    trafo = SanitiseUnusedRoutineTransformation(routines=('legacy_unused',), raise_error=True)
    trafo.apply(routine)

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    pout_decl = next(decl for decl in decls if any(sym.name.lower() == 'pout' for sym in decl.symbols))
    zoper_decl = next(decl for decl in decls if any(sym.name.lower() == 'zoper' for sym in decl.symbols))
    pout = next(sym for sym in pout_decl.symbols if sym.name.lower() == 'pout')
    zoper = next(sym for sym in zoper_decl.symbols if sym.name.lower() == 'zoper')
    declared_names = {sym.name.lower() for decl in decls for sym in decl.symbols}
    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)
    untouched_intrinsics = FindNodes(ir.Intrinsic).visit(untouched.body)

    assert pout.type.shape == (':', ':', ':')
    assert pout.dimensions == (':', ':', ':')
    assert zoper.type.shape == (':', ':', ':')
    assert zoper.dimensions == (':', ':', ':')
    assert 'zouts' not in declared_names
    assert len(intrinsics) == 1
    assert 'error stop "sanitised unused routine legacy_unused was called"' in intrinsics[0].text.lower()
    assert not untouched_intrinsics
    assert len(FindNodes(ir.Assignment).visit(untouched.body)) == 1


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_sanitise_unused_routine_by_qualified_name(frontend, tmp_path):
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

    trafo = SanitiseUnusedRoutineTransformation(routines=('another_legacy_mod#keep_me',), raise_error=True)
    trafo.apply(routine)

    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)
    assert len(intrinsics) == 1
    assert 'error stop "sanitised unused routine keep_me was called"' in intrinsics[0].text.lower()
