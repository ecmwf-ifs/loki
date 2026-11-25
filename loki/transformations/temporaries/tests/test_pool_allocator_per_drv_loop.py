# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`TemporariesPoolAllocatorPerDrvLoopTransformation`.
"""

import pytest

from loki import Dimension, Subroutine
from loki.batch import ProcedureItem
from loki.expression import IntLiteral, InlineCall, Literal, Variable
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes

from loki.transformations.temporaries import TemporariesPoolAllocatorPerDrvLoopTransformation


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(name='block_dim', size='nb', index='b', aliases=('nb',))


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('klon', 'columns')
    )


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_per_drv_loop_multiple_loops(tmp_path, frontend, horizontal):
    """
    Test per-driver-loop pool allocator with multiple independent driver
    loops calling different kernels.

    The first driver loop is identified through its call to a target
    kernel and resolves the block index from an assignment in the loop
    body, while the second one is marked with ``!$loki driver-loop``.
    """
    block_dim = Dimension(name='block_dim', size='nb', index=('ibl', 'bnds%kbl'))

    fcode_kernel1 = """
subroutine kernel1(nlon, nz, start, end, field1)
  implicit none
  integer, intent(in) :: nlon, nz, start, end
  real, intent(inout) :: field1(nlon, nz)
  real :: tmp1(nlon, nz)
  integer :: jl, jk

  do jk = 1, nz
    do jl = start, end
      tmp1(jl, jk) = field1(jl, jk) * 2.0
      field1(jl, jk) = tmp1(jl, jk)
    end do
  end do
end subroutine kernel1
    """.strip()

    fcode_kernel2 = """
subroutine kernel2(nlon, nz, start, end, field2)
  implicit none
  integer, intent(in) :: nlon, nz, start, end
  real, intent(inout) :: field2(nlon, nz)
  real :: tmp2(nlon, nz)
  real :: tmp3(nlon)
  integer :: jl, jk

  do jk = 1, nz
    do jl = start, end
      tmp2(jl, jk) = field2(jl, jk) + 1.0
      field2(jl, jk) = tmp2(jl, jk)
    end do
  end do
  do jl = start, end
    tmp3(jl) = field2(jl, 1)
    field2(jl, 1) = tmp3(jl)
  end do
end subroutine kernel2
    """.strip()

    fcode_driver = """
subroutine driver(nlon, nz, nproma, nb, kgpcomp, field1, field2)
  implicit none
  integer, intent(in) :: nlon, nz, nproma, nb, kgpcomp
  real, intent(inout) :: field1(nlon, nz, nb)
  real, intent(inout) :: field2(nlon, nz, nb)
  integer :: jkglo, kst, kend, ibl

  do jkglo = 1, kgpcomp, nproma
    kst = 1
    kend = min(nproma, kgpcomp - jkglo + 1)
    ibl = (jkglo - 1) / nproma + 1
    call kernel1(nlon, nz, kst, kend, field1(:,:,ibl))
  end do

  !$loki driver-loop
  do ibl = 1, nb
    call kernel2(nlon, nz, 1, nlon, field2(:,:,ibl))
  end do
end subroutine driver
    """.strip()

    kernel1 = Subroutine.from_source(fcode_kernel1, frontend=frontend, xmods=[tmp_path])
    kernel2 = Subroutine.from_source(fcode_kernel2, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path])

    driver.enrich(kernel1)
    driver.enrich(kernel2)

    transformation = TemporariesPoolAllocatorPerDrvLoopTransformation(
        block_dim=block_dim, horizontal=horizontal, check_bounds=True
    )

    kernel1_item = ProcedureItem(name='#kernel1', source=kernel1, config={'role': 'kernel'})
    kernel2_item = ProcedureItem(name='#kernel2', source=kernel2, config={'role': 'kernel'})
    driver_item = ProcedureItem(name='#driver', source=driver, config={'role': 'driver'})

    class MockSGraph:
        def successors(self, item):
            if item is driver_item:
                return (kernel1_item, kernel2_item)
            return ()

    sgraph = MockSGraph()

    # Transform kernels first
    transformation.transform_subroutine(
        kernel1, role='kernel', item=kernel1_item, targets=('kernel1',),
        sub_sgraph=sgraph
    )
    transformation.transform_subroutine(
        kernel2, role='kernel', item=kernel2_item, targets=('kernel2',),
        sub_sgraph=sgraph
    )

    # Transform driver
    transformation.transform_subroutine(
        driver, role='driver', item=driver_item, targets=('kernel1', 'kernel2'),
        sub_sgraph=sgraph
    )

    # Verify: single ISTSZ and (allocatable) ZSTACK
    assert 'istsz' in driver.variable_map
    assert 'zstack' in driver.variable_map
    assert driver.variable_map['zstack'].type.allocatable

    # Verify: the ISTSZ assignment uses MAX of the two different kernel sizes
    istsz_assigns = [
        a for a in FindNodes(ir.Assignment).visit(driver.body) if a.lhs == 'istsz'
    ]
    assert len(istsz_assigns) == 1
    rhs = istsz_assigns[0].rhs
    assert isinstance(rhs, InlineCall) and rhs.function == 'MAX'

    # Verify: both driver loops have stack pointer assignments
    loops = FindNodes(ir.Loop).visit(driver.body)
    driver_loops = [l for l in loops if l.variable in ('jkglo', 'ibl')]
    assert len(driver_loops) == 2

    for loop in driver_loops:
        assignments = FindNodes(ir.Assignment).visit(loop.body)
        stack_assigns = [a for a in assignments if 'ylstack_l' in str(a.lhs).lower()]
        assert len(stack_assigns) >= 1
        assert 'loc(zstack(1, ibl))' in str(stack_assigns[0].rhs).lower()

    # Verify: the stack pointer is inserted after the resolved block index
    # assignment in the first loop and at the top of the second loop
    body = driver_loops[0].body
    ibl_pos = next(i for i, n in enumerate(body) if isinstance(n, ir.Assignment) and n.lhs == 'ibl')
    stack_pos = next(i for i, n in enumerate(body) if isinstance(n, ir.Assignment) and n.lhs == 'ylstack_l')
    assert stack_pos > ibl_pos
    assert isinstance(driver_loops[1].body[0], ir.Assignment)
    assert driver_loops[1].body[0].lhs == 'ylstack_l'

    # Verify: kernels have stack arguments and Cray pointers for their temporaries
    for kernel, num_temporaries in ((kernel1, 1), (kernel2, 2)):
        assert 'ydstack_l' in kernel.variable_map
        assert 'ydstack_u' in kernel.variable_map
        pointer_decls = [
            i for i in FindNodes(ir.GenericStmt).visit(kernel.spec) if 'POINTER' in i.text
        ]
        assert len(pointer_decls) == num_temporaries

    # Verify: both kernel calls have stack kwargs
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 2
    for call in calls:
        kwarg_names = [kw[0].lower() for kw in call.kwarguments]
        assert 'ydstack_l' in kwarg_names
        assert 'ydstack_u' in kwarg_names


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_per_drv_loop_nested_driver_kernel(tmp_path, frontend, block_dim, horizontal):
    """
    Test that a kernel with LowerBlockIndex trafo_data (nested driver)
    also gets pool allocator infrastructure in its block loops.
    """
    fcode_inner = """
subroutine inner_kernel(nlon, nz, start, end, field)
  implicit none
  integer, intent(in) :: nlon, nz, start, end
  real, intent(inout) :: field(nlon, nz)
  real :: work(nlon, nz)
  integer :: jl, jk

  do jk = 1, nz
    do jl = start, end
      work(jl, jk) = field(jl, jk) * 3.0
      field(jl, jk) = work(jl, jk)
    end do
  end do
end subroutine inner_kernel
    """.strip()

    fcode_outer = """
subroutine outer_kernel(nlon, nz, nb, start, end, field)
  implicit none
  integer, intent(in) :: nlon, nz, nb, start, end
  real, intent(inout) :: field(nlon, nz, nb)
  integer :: b

  !$loki driver-loop
  do b = 1, nb
    call inner_kernel(nlon, nz, start, end, field(:,:,b))
  end do
end subroutine outer_kernel
    """.strip()

    inner = Subroutine.from_source(fcode_inner, frontend=frontend, xmods=[tmp_path])
    outer = Subroutine.from_source(fcode_outer, frontend=frontend, xmods=[tmp_path])

    outer.enrich(inner)

    transformation = TemporariesPoolAllocatorPerDrvLoopTransformation(
        block_dim=block_dim, horizontal=horizontal, check_bounds=True
    )

    inner_item = ProcedureItem(name='#inner_kernel', source=inner, config={'role': 'kernel'})
    outer_item = ProcedureItem(name='#outer_kernel', source=outer, config={'role': 'kernel'})
    # Simulate LowerBlockIndex trafo_data presence
    outer_item.trafo_data['LowerBlockIndex'] = {'driver_loop': None}

    class MockSGraph:
        def successors(self, item):
            if item is outer_item:
                return (inner_item,)
            return ()

    sgraph = MockSGraph()

    transformation.transform_subroutine(
        inner, role='kernel', item=inner_item, targets=('inner_kernel',),
        sub_sgraph=sgraph
    )
    transformation.transform_subroutine(
        outer, role='kernel', item=outer_item, targets=('inner_kernel',),
        sub_sgraph=sgraph
    )

    # Verify: outer kernel has pool allocator infrastructure
    assert 'istsz' in outer.variable_map
    assert 'zstack' in outer.variable_map

    # Verify: stack pointer assignment in driver loop
    loops = FindNodes(ir.Loop).visit(outer.body)
    block_loops = [l for l in loops if l.variable == 'b']
    assert len(block_loops) == 1
    assignments = FindNodes(ir.Assignment).visit(block_loops[0].body)
    stack_assigns = [a for a in assignments if 'ylstack_l' in str(a.lhs).lower()]
    assert len(stack_assigns) >= 1


def test_aggregate_stack_sizes_unit():
    """
    Unit test for _aggregate_stack_sizes static method.
    """
    aggregate = TemporariesPoolAllocatorPerDrvLoopTransformation._aggregate_stack_sizes

    # All zeros
    result = aggregate([Literal(0), Literal(0)])
    assert int(result) == 0

    # Single non-zero
    expr = IntLiteral(42)
    result = aggregate([Literal(0), expr])
    assert result is expr

    # Multiple non-zero -> MAX
    expr1 = IntLiteral(10)
    expr2 = IntLiteral(20)
    result = aggregate([expr1, expr2])
    assert isinstance(result, InlineCall) and result.function == 'MAX'
    assert len(result.parameters) == 2

    # Deduplication
    result = aggregate([expr1, expr1])
    assert result is expr1  # Only one unique value

    # Unwraps nested MAX
    inner_max = InlineCall(
        function=Variable(name='MAX'),
        parameters=(IntLiteral(5), IntLiteral(15)),
        kw_parameters=()
    )
    result = aggregate([inner_max, IntLiteral(25)])
    assert isinstance(result, InlineCall) and result.function == 'MAX'
    assert len(result.parameters) == 3  # 5, 15, 25 flattened


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_per_drv_loop_cray_ptr_loc_rhs(tmp_path, frontend, block_dim, horizontal):
    """
    Test per-driver-loop pool allocator with cray_ptr_loc_rhs=True mode,
    where the stack array is passed directly and LOC() calls happen in kernels.
    """
    fcode_kernel = """
subroutine kernel_loc(nlon, nz, start, end, field1)
  implicit none
  integer, intent(in) :: nlon, nz, start, end
  real, intent(inout) :: field1(nlon, nz)
  real :: tmp1(nlon, nz)
  integer :: jl, jk

  do jk = 1, nz
    do jl = start, end
      tmp1(jl, jk) = field1(jl, jk) * 2.0
      field1(jl, jk) = tmp1(jl, jk)
    end do
  end do
end subroutine kernel_loc
    """.strip()

    fcode_driver = """
subroutine driver_loc(nlon, nz, nb, field1)
  implicit none
  integer, intent(in) :: nlon, nz, nb
  real, intent(inout) :: field1(nlon, nz, nb)
  integer :: b

  !$loki driver-loop
  do b = 1, nb
    call kernel_loc(nlon, nz, 1, nlon, field1(:,:,b))
  end do
end subroutine driver_loc
    """.strip()

    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path])

    driver.enrich(kernel)

    transformation = TemporariesPoolAllocatorPerDrvLoopTransformation(
        block_dim=block_dim, horizontal=horizontal, check_bounds=True,
        cray_ptr_loc_rhs=True
    )

    kernel_item = ProcedureItem(name='#kernel_loc', source=kernel, config={'role': 'kernel'})
    driver_item = ProcedureItem(name='#driver_loc', source=driver, config={'role': 'driver'})

    class MockSGraph:
        def successors(self, item):
            if item is driver_item:
                return (kernel_item,)
            return ()

    sgraph = MockSGraph()

    transformation.transform_subroutine(
        kernel, role='kernel', item=kernel_item, targets=('kernel_loc',),
        sub_sgraph=sgraph
    )
    transformation.transform_subroutine(
        driver, role='driver', item=driver_item, targets=('kernel_loc',),
        sub_sgraph=sgraph
    )

    # Verify: kernel has ZSTACK argument (contiguous real array)
    assert 'zstack' in kernel.variable_map
    kernel_zstack = kernel.variable_map['zstack']
    assert kernel_zstack.type.contiguous
    assert kernel_zstack.type.intent == 'inout'

    # Verify: driver stack pointer uses integer 1 (not LOC)
    loops = FindNodes(ir.Loop).visit(driver.body)
    driver_loop = [l for l in loops if l.variable == 'b'][0]
    assignments = FindNodes(ir.Assignment).visit(driver_loop.body)
    stack_ptr_assigns = [a for a in assignments if 'ylstack_l' in str(a.lhs).lower()]
    assert len(stack_ptr_assigns) >= 1
    # In cray_ptr_loc_rhs mode, YLSTACK_L = 1
    assert stack_ptr_assigns[0].rhs == 1

    # Verify: kernel call has ZSTACK kwarg
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    kernel_calls = [c for c in calls if c.name == 'kernel_loc']
    assert len(kernel_calls) == 1
    kwarg_names = [kw[0].lower() for kw in kernel_calls[0].kwarguments]
    assert 'zstack' in kwarg_names
    assert 'ydstack_l' in kwarg_names

    # Verify: kernel uses LOC(ZSTACK(...)) for pointer assignment
    kernel_assigns = FindNodes(ir.Assignment).visit(kernel.body)
    loc_assigns = [a for a in kernel_assigns if 'LOC' in str(a.rhs)]
    assert len(loc_assigns) >= 1
