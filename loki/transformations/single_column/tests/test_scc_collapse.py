# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine
from loki.frontend import available_frontends
from loki.ir import (
    nodes as ir, FindNodes, get_pragma_parameters, pragmas_attached
)
from loki.transformations.single_column import SCCCollapseTransformation


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('directive', [False, 'openacc', 'omp-gpu', 'openmp'])
@pytest.mark.parametrize('enabled,role', [(False, 'driver'), (True, 'driver'), (True, 'kernel')])
def test_scc_collapse(frontend, tmp_path, directive, enabled, role):
    """Collapse only enabled driver nests for supported programming models."""
    routine = Subroutine.from_source('''
subroutine driver(a, n, nb)
  integer, intent(in) :: n, nb
  real, intent(inout) :: a(n, nb)
  integer :: ib, jl
  !$loki loop gang vlength(32) async(1)
  do ib = 1, nb
    ! Before the vector loop
    !$loki loop vector
    do jl = 1, n, 1
      a(jl, ib) = jl + ib
    end do
    !$loki end loop vector
    ! After the vector loop
  end do
  !$loki end loop gang
end subroutine driver
''', frontend=frontend, xmods=[tmp_path])
    before = FindNodes(ir.Loop).visit(routine.body)
    assignments = FindNodes(ir.Assignment).visit(routine.body)
    comments = FindNodes(ir.Comment).visit(routine.body)
    transform = SCCCollapseTransformation(collapse_gang_vector=enabled, directive=directive)
    transform.apply(routine, role=role)
    transform.apply(routine, role=role)
    collapsed = enabled and role == 'driver' and directive in (False, 'openacc', 'omp-gpu')
    with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
        outer, inner = FindNodes(ir.Loop).visit(routine.body)
        assert (outer.variable, outer.bounds) == (before[0].variable, before[0].bounds)
        assert (inner.variable, inner.bounds) == (before[1].variable, before[1].bounds)
        assert inner in outer.body
        params = get_pragma_parameters(outer.pragma, starts_with='loop')
        assert params == ({'gang': None, 'vector': None, 'collapse': '2', 'vlength': '32', 'async': '1'}
                          if collapsed else {'gang': None, 'vlength': '32', 'async': '1'})
        assert get_pragma_parameters(outer.pragma_post, starts_with='end loop') == (
            {'gang': None, 'vector': None} if collapsed else {'gang': None}
        )
        if collapsed:
            assert not inner.pragma
            assert not inner.pragma_post
        else:
            assert get_pragma_parameters(inner.pragma, starts_with='loop') == {'vector': None}
            assert get_pragma_parameters(inner.pragma_post, starts_with='end loop') == {'vector': None}
    assert FindNodes(ir.Assignment).visit(routine.body) == assignments
    assert FindNodes(ir.Comment).visit(routine.body) == comments


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('change', [
    'setup', 'trailing', 'siblings', 'dependent', 'modified', 'call-modified', 'call-bound',
    'step', 'same-index', 'private', 'firstprivate', 'reduction', 'conflicting',
    'exit', 'cycle', 'return'
])
def test_scc_collapse_ineligible(frontend, tmp_path, change):
    """Leave unsupported structure, bounds, sharing and control flow intact."""
    setup = 'stack_ptr = ib' if change == 'setup' else ''
    trailing = 'stack_ptr = ib' if change == 'trailing' else ''
    inner_bounds = {'dependent': '1, ib', 'call-bound': '1, min(n, 4)', 'step': '1, n, 2'}.get(change, '1, n')
    gang_clause = {'private': 'private(stack_ptr)', 'firstprivate': 'firstprivate(n)'}.get(change, '')
    vector_clause = 'reduction(+:stack_ptr)' if change == 'reduction' else ''
    extra_pragma = '!$acc loop seq' if change == 'conflicting' else ''
    extra_statement = {
        'modified': 'n = n - 1', 'call-modified': 'call touch(n)',
        'exit': 'exit', 'cycle': 'cycle', 'return': 'return'
    }.get(change, '')
    sibling = '''
    !$loki loop vector
    do jl = 1, n
      a(jl, ib) = 0
    end do
    ''' if change == 'siblings' else ''
    routine = Subroutine.from_source(f'''
subroutine driver(a, n, nb)
  integer :: n, nb, ib, jl, stack_ptr
  real :: a(n, nb)
  !$loki loop gang {gang_clause}
  do ib = 1, nb
    {setup}
    {extra_pragma}
    !$loki loop vector {vector_clause}
    do jl = {inner_bounds}
      a(jl, ib) = jl + ib
      {extra_statement}
    end do
    {trailing}
    {sibling}
  end do
  !$loki end loop gang
end subroutine driver
''', frontend=frontend, xmods=[tmp_path])
    if change == 'same-index':
        outer, inner = FindNodes(ir.Loop).visit(routine.body)
        inner._update(variable=outer.variable)
    loops = tuple((loop.variable, loop.bounds) for loop in FindNodes(ir.Loop).visit(routine.body))
    pragmas = tuple((p.keyword, p.content) for p in FindNodes(ir.Pragma).visit(routine.body))
    assignments = tuple((a.lhs, a.rhs) for a in FindNodes(ir.Assignment).visit(routine.body))
    SCCCollapseTransformation(collapse_gang_vector=True).apply(routine, role='driver')
    assert tuple((loop.variable, loop.bounds) for loop in FindNodes(ir.Loop).visit(routine.body)) == loops
    assert tuple((p.keyword, p.content) for p in FindNodes(ir.Pragma).visit(routine.body)) == pragmas
    assert tuple((a.lhs, a.rhs) for a in FindNodes(ir.Assignment).visit(routine.body)) == assignments
