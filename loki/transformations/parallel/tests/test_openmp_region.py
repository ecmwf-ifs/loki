# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine
from loki.frontend import available_frontends
from loki.ir import (
    nodes as ir, FindNodes, pragma_regions_attached, is_loki_pragma
)

from loki.transformations.parallel import remove_openmp_regions


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('insert_loki_parallel', (True, False))
def test_remove_openmp_regions(frontend, insert_loki_parallel):
    """
    A simple test for :any:`remove_openmp_regions`
    """
    fcode = """
subroutine test_driver_openmp(n, arr)
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: arr(n)
  integer :: i

  !$omp parallel private(i)
  !$omp do schedule dynamic(1)
  do i=1, n
    arr(i) = arr(i) + 1.0
  end do
  !$omp end do
  !$omp end parallel


  !$OMP PARALLEL PRIVATE(i)
  !$OMP DO SCHEDULE DYNAMIC(1)
  do i=1, n
    arr(i) = arr(i) + 1.0
  end do
  !$OMP END DO
  !$OMP END PARALLEL


  !$omp parallel do private(i)
  do i=1, n
    !$omp simd
    arr(i) = arr(i) + 1.0
  end do
  !$omp end parallel
end subroutine test_driver_openmp
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3
    assert len(FindNodes(ir.Pragma).visit(routine.body)) == 11

    with pragma_regions_attached(routine):
        # Without attaching Loop-pragmas, all are recognised as regions
        assert len(FindNodes(ir.PragmaRegion).visit(routine.body)) == 5

    remove_openmp_regions(routine, insert_loki_parallel=insert_loki_parallel)

    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == (6 if insert_loki_parallel else 0)

    if insert_loki_parallel:
        with pragma_regions_attached(routine):
            pragma_regions = FindNodes(ir.PragmaRegion).visit(routine.body)
            assert len(pragma_regions) == 3
            for region in pragma_regions:
                assert is_loki_pragma(region.pragma, starts_with='parallel')
                assert is_loki_pragma(region.pragma_post, starts_with='end parallel')
