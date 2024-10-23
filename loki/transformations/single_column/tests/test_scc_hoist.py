# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Sourcefile, Dimension, fgen
from loki.batch import (
    Scheduler, SchedulerConfig, ProcedureItem, ModuleItem
)
from loki.frontend import available_frontends
from loki.ir import (
    FindNodes, Assignment, CallStatement, Loop, Pragma,
    pragmas_attached
)

from loki.transformations import (
    InlineTransformation
)
from loki.transformations.single_column import (
    SCCBaseTransformation, SCCDevectorTransformation,
    SCCDemoteTransformation, SCCRevectorTransformation,
    SCCAnnotateTransformation, SCCHoistPipeline,
)


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',)
    )

@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk', aliases=('nlev',))


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_hoist_multiple_kernels(frontend, horizontal, blocking):
    """
    Test hoisting of column temporaries to "driver" level.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b))

      ! A second call, to check multiple calls are honored
      call compute_column(start, end, nlon, nz, q(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    kernel_source = Sourcefile.from_source(fcode_kernel, frontend=frontend)
    driver_source = Sourcefile.from_source(fcode_driver, frontend=frontend)
    driver = driver_source['column_driver']
    kernel = kernel_source['compute_column']
    driver.enrich(kernel)  # Attach kernel source to driver call

    driver_item = ProcedureItem(name='#column_driver', source=driver_source)
    kernel_item = ProcedureItem(name='#compute_column', source=kernel_source)

    scc_hoist = SCCHoistPipeline(
        horizontal=horizontal, block_dim=blocking, directive='openacc'
    )

    # Apply pipeline in reverse order to ensure analysis runs before hoisting
    scc_hoist.apply(kernel, role='kernel', item=kernel_item)
    scc_hoist.apply(
        driver, role='driver', item=driver_item,
        successors=(kernel_item,), targets=['compute_column']
    )

    # Ensure we two loops left in kernel
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[0].bounds == 'start:end'
    assert kernel_loops[1].variable == 'jk'
    assert kernel_loops[1].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure we have only one driver block loop
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 1
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'

    # Ensure we have two kernel calls in the driver loop
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 2
    assert kernel_calls[0].name == 'compute_column'
    assert kernel_calls[1].name == 'compute_column'
    assert 'compute_column_t(:,:,b)' in kernel_calls[0].arguments
    assert 'compute_column_t(:,:,b)' in kernel_calls[1].arguments

    # Ensure that column local `t(nlon,nz)` has been hoisted
    assert 't' in kernel.argnames
    assert kernel.variable_map['t'].type.intent.lower() == 'inout'
    assert kernel.variable_map['t'].type.shape == ('nlon', 'nz')
    assert driver.variable_map['compute_column_t'].dimensions == ('nlon', 'nz', 'nb')


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [True, False])
def test_scc_hoist_multiple_kernels_loops(tmp_path, frontend, trim_vector_sections, horizontal, blocking):
    """
    Test hoisting of column temporaries to "driver" level.
    """

    fcode_driver = """
  SUBROUTINE driver(nlon, nz, q, nb)
    use kernel_mod, only: kernel
    implicit none
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    REAL                  :: c, tmp(nlon,nz,nb)
    INTEGER :: b, jk, jl, start, end

    tmp = 0.0

    !$loki driver-loop
    do b=1, nb
      end = nlon - nb
      do jk = 2, nz
        do jl = start, end
          q(jl, jk, b) = 2.0 * jk * jl
        end do
      end do
    end do

    do b=2, nb
      end = nlon - nb
      call kernel(start, end, nlon, nz, q(:,:,b))

      DO jk = 2, nz
        DO jl = start, end
          tmp(jl, jk, b) = 2.0 * jk * jl
          q(jl, jk, b) = q(jl, jk-1, b) * c + tmp(jl, jk, b)
        END DO
      END DO

      ! A second call, to check multiple calls are honored
      call kernel(start, end, nlon, nz, q(:,:,b))

      DO jk = 2, nz
        DO jl = start, end
          q(jl, jk, b) = (-1.0) * q(jl, jk, b)
        END DO
      END DO
    end do

    !$loki driver-loop
    do b=3, nb
      end = nlon - nb
      !$loki separator
      do jk = 2, nz
        do jl = start, end
          q(jl, jk, b) = 2.0 * jk * jl
        end do
      end do
    end do
  END SUBROUTINE driver
""".strip()

    fcode_kernel = """
MODULE kernel_mod
implicit none
CONTAINS
  SUBROUTINE kernel(start, end, nlon, nz, q)
    implicit none
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * k
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE kernel
END MODULE kernel_mod
""".strip()

    (tmp_path / 'driver.F90').write_text(fcode_driver)
    (tmp_path / 'kernel.F90').write_text(fcode_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }
    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config), frontend=frontend, xmods=[tmp_path]
    )

    driver = scheduler["#driver"].ir
    kernel = scheduler["kernel_mod#kernel"].ir

    transformation = (SCCBaseTransformation(horizontal=horizontal, directive='openacc'),)
    transformation += (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
    transformation += (SCCDemoteTransformation(horizontal=horizontal),)
    transformation += (SCCRevectorTransformation(horizontal=horizontal),)
    transformation += (SCCAnnotateTransformation(directive='openacc', block_dim=blocking),)
    for transform in transformation:
        scheduler.process(transformation=transform)

    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[0].bounds == 'start:end'
    assert kernel_loops[1].variable == 'jk'
    assert kernel_loops[1].bounds == '2:nz'

    driver_loops = FindNodes(Loop).visit(driver.body)
    driver_loop_pragmas = [pragma for pragma in FindNodes(Pragma).visit(driver.body) if pragma.keyword.lower() == 'acc']
    assert len(driver_loops) == 11
    assert len(driver_loop_pragmas) == 14
    assert "parallel loop gang vector_length" in driver_loop_pragmas[0].content.lower()
    assert "loop vector" in driver_loop_pragmas[1].content.lower()
    assert "loop seq" in driver_loop_pragmas[2].content.lower()
    assert "end parallel loop" in driver_loop_pragmas[3].content.lower()
    assert "parallel loop gang vector_length" in driver_loop_pragmas[4].content.lower()
    assert "loop vector" in driver_loop_pragmas[5].content.lower()
    assert "loop seq" in driver_loop_pragmas[6].content.lower()
    assert "loop vector" in driver_loop_pragmas[7].content.lower()
    assert "loop seq" in driver_loop_pragmas[8].content.lower()
    assert "end parallel loop" in driver_loop_pragmas[9].content.lower()
    assert "parallel loop gang vector_length" in driver_loop_pragmas[10].content.lower()
    assert "loop vector" in driver_loop_pragmas[11].content.lower()
    assert "loop seq" in driver_loop_pragmas[12].content.lower()
    assert "end parallel loop" in driver_loop_pragmas[13].content.lower()

    assert driver_loops[1] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[2] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'
    assert driver_loops[1].variable == 'jl'
    assert driver_loops[1].bounds == 'start:end'
    assert driver_loops[2].variable == 'jk'
    assert driver_loops[2].bounds == '2:nz'

    # check location of loop-bound assignment
    assign = FindNodes(Assignment).visit(driver_loops[0])[0]
    assert assign.lhs == 'end'
    assert assign.rhs == 'nlon-nb'
    assigns = FindNodes(Assignment).visit(driver_loops[1])
    if trim_vector_sections:
        assert not assign in assigns
    else:
        assert assign in assigns

    assert driver_loops[4] in FindNodes(Loop).visit(driver_loops[3].body)
    assert driver_loops[5] in FindNodes(Loop).visit(driver_loops[3].body)
    assert driver_loops[6] in FindNodes(Loop).visit(driver_loops[3].body)
    assert driver_loops[7] in FindNodes(Loop).visit(driver_loops[3].body)
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[3])
    assert len(kernel_calls) == 2
    assert kernel_calls[0].name == 'kernel'
    assert kernel_calls[1].name == 'kernel'

    assert driver_loops[3].variable == 'b'
    assert driver_loops[3].bounds == '2:nb'
    assert driver_loops[4].variable == 'jl'
    assert driver_loops[4].bounds == 'start:end'
    assert driver_loops[5].variable == 'jk'
    assert driver_loops[5].bounds == '2:nz'
    assert driver_loops[6].variable == 'jl'
    assert driver_loops[6].bounds == 'start:end'
    assert driver_loops[7].variable == 'jk'
    assert driver_loops[7].bounds == '2:nz'

    assert driver_loops[9] in FindNodes(Loop).visit(driver_loops[8].body)
    assert driver_loops[10] in FindNodes(Loop).visit(driver_loops[8].body)
    assert driver_loops[8].variable == 'b'
    assert driver_loops[8].bounds == '3:nb'
    assert driver_loops[9].variable == 'jl'
    assert driver_loops[9].bounds == 'start:end'
    assert driver_loops[10].variable == 'jk'
    assert driver_loops[10].bounds == '2:nz'

    # check location of loop-bound assignment
    assign = FindNodes(Assignment).visit(driver_loops[8])[0]
    assert assign.lhs == 'end'
    assert assign.rhs == 'nlon-nb'
    assigns = FindNodes(Assignment).visit(driver_loops[9])
    assert not assign in assigns


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_hoist_openacc(frontend, horizontal, vertical, blocking, tmp_path):
    """
    Test the correct addition of OpenACC pragmas to SCC format code
    when hoisting array temporaries to driver.
    """

    fcode_mod = """
MODULE BLOCK_DIM_MOD
    INTEGER :: nb
END MODULE BLOCK_DIM_MOD
    """.strip()

    fcode_driver = """
SUBROUTINE column_driver(nlon, nz, q)
    USE BLOCK_DIM_MOD, ONLY : nb
    INTEGER, INTENT(IN)   :: nlon, nz  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b))
    end do
END SUBROUTINE column_driver
    """.strip()

    fcode_kernel = """
SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)
    REAL :: a(nlon)
    REAL :: b(nlon,psize)
    INTEGER, PARAMETER :: psize = 3
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      a(jl) = Q(JL, 1)
      b(jl, 1) = Q(JL, 2)
      b(jl, 2) = Q(JL, 3)
      b(jl, 3) = a(jl) * (b(jl, 1) + b(jl, 2))

      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
END SUBROUTINE compute_column
    """.strip()

    fcode_module = """
module my_scaling_value_mod
    implicit none
    REAL :: c = 5.345
end module my_scaling_value_mod
    """.strip()

    # Mimic the scheduler internal mechanis to apply the transformation cascade
    mod_source = Sourcefile.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(fcode_kernel, frontend=frontend, xmods=[tmp_path])
    driver_source = Sourcefile.from_source(
        fcode_driver, frontend=frontend, definitions=mod_source.modules, xmods=[tmp_path]
    )
    module_source = Sourcefile.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    driver = driver_source['column_driver']
    kernel = kernel_source['compute_column']
    module = module_source['my_scaling_value_mod']
    kernel.enrich(module)
    driver.enrich(kernel)  # Attach kernel source to driver call

    driver_item = ProcedureItem(name='#column_driver', source=driver_source)
    kernel_item = ProcedureItem(name='#compute_column', source=kernel_source)
    module_item = ModuleItem(name='my_scaling_value_mod', source=module_source)

    scc_hoist = SCCHoistPipeline(
        horizontal=horizontal, block_dim=blocking,
        directive='openacc', dim_vars=vertical.sizes
    )

    # Apply in reverse order to ensure hoisting analysis gets run on kernel first
    scc_hoist.apply(kernel, role='kernel', item=kernel_item, successors=(module_item,))
    scc_hoist.apply(
        driver, role='driver', item=driver_item, successors=(kernel_item,), targets=['compute_column']
    )

    # Check that blocking size has not been redefined
    assert driver.symbol_map[blocking.size].type.module.name.lower() == 'block_dim_mod'

    with pragmas_attached(kernel, Loop):
        # Ensure kernel routine is anntoated at vector level
        kernel_pragmas = FindNodes(Pragma).visit(kernel.ir)
        assert len(kernel_pragmas) == 3
        assert kernel_pragmas[0].keyword == 'acc'
        assert kernel_pragmas[0].content == 'routine vector'
        assert kernel_pragmas[1].keyword == 'acc'
        assert kernel_pragmas[1].content == 'data present(q, t)'
        assert kernel_pragmas[2].keyword == 'acc'
        assert kernel_pragmas[2].content == 'end data'

        # Ensure `seq` and `vector` loops in kernel
        kernel_loops = FindNodes(Loop).visit(kernel.body)
        assert len(kernel_loops) == 2
        assert kernel_loops[0].pragma[0].keyword == 'acc'
        assert kernel_loops[0].pragma[0].content == 'loop vector private(b)'
        assert kernel_loops[1].pragma[0].keyword == 'acc'
        assert kernel_loops[1].pragma[0].content == 'loop seq'

    # Ensure two levels of blocked parallel loops in driver
    with pragmas_attached(driver, Loop):
        driver_loops = FindNodes(Loop).visit(driver.body)
        assert len(driver_loops) == 1
        assert driver_loops[0].pragma[0].keyword == 'acc'
        assert driver_loops[0].pragma[0].content == 'parallel loop gang vector_length(nlon)'

        # Ensure device allocation and teardown via `!$acc enter/exit data`
        driver_pragmas = FindNodes(Pragma).visit(driver.body)
        assert len(driver_pragmas) == 2
        assert driver_pragmas[0].keyword == 'acc'
        assert driver_pragmas[0].content == 'enter data create(compute_column_t)'
        assert driver_pragmas[1].keyword == 'acc'
        assert driver_pragmas[1].content == 'exit data delete(compute_column_t)'

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_scc_hoist_nested_openacc(frontend, horizontal, vertical, blocking,
        as_kwarguments):
    """
    Test the correct addition of OpenACC pragmas to SCC format code
    when hoisting array temporaries to driver.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_outer_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + 1.0
    END DO

    call update_q(start, end, nlon, nz, q, c)

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""

    fcode_inner_kernel = """
  SUBROUTINE update_q(start, end, nlon, nz, q, c)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL, INTENT(IN)    :: c
    REAL :: t(nlon,nz)
    INTEGER :: jl, jk

    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO
  END SUBROUTINE update_q
"""

    # Mimic the scheduler internal mechanis to apply the transformation cascade
    outer_kernel_source = Sourcefile.from_source(fcode_outer_kernel, frontend=frontend)
    inner_kernel_source = Sourcefile.from_source(fcode_inner_kernel, frontend=frontend)
    driver_source = Sourcefile.from_source(fcode_driver, frontend=frontend)
    driver = driver_source['column_driver']
    outer_kernel = outer_kernel_source['compute_column']
    inner_kernel = inner_kernel_source['update_q']
    outer_kernel.enrich(inner_kernel)  # Attach kernel source to driver call
    driver.enrich(outer_kernel)  # Attach kernel source to driver call

    driver_item = ProcedureItem(name='#column_driver', source=driver)
    outer_kernel_item = ProcedureItem(name='#compute_column', source=outer_kernel)
    inner_kernel_item = ProcedureItem(name='#update_q', source=inner_kernel)

    scc_hoist = SCCHoistPipeline(
        horizontal=horizontal, block_dim=blocking,
        dim_vars=vertical.sizes, as_kwarguments=as_kwarguments, directive='openacc'
    )

    # Apply in reverse order to ensure hoisting analysis gets run on kernel first
    scc_hoist.apply(inner_kernel, role='kernel', item=inner_kernel_item)
    scc_hoist.apply(
        outer_kernel, role='kernel', item=outer_kernel_item,
        targets=['compute_q'], successors=(inner_kernel_item,)
    )
    scc_hoist.apply(
        driver, role='driver', item=driver_item,
        targets=['compute_column'], successors=(outer_kernel_item,)
    )

    # Ensure calls have correct arguments
    # driver
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    if not as_kwarguments:
        assert calls[0].arguments == ('start', 'end', 'nlon', 'nz', 'q(:, :, b)',
                'update_q_t(:, :, b)')
        assert calls[0].kwarguments == ()
    else:
        assert calls[0].arguments == ('start', 'end', 'nlon', 'nz', 'q(:, :, b)')
        assert calls[0].kwarguments == (('update_q_t', 'update_q_t(:, :, b)'),)
    # outer kernel
    calls = FindNodes(CallStatement).visit(outer_kernel.body)
    assert len(calls) == 1
    if not as_kwarguments:
        assert calls[0].arguments == ('start', 'end', 'nlon', 'nz', 'q', 'c', 'update_q_t')
        assert calls[0].kwarguments == ()
    else:
        assert calls[0].arguments == ('start', 'end', 'nlon', 'nz', 'q', 'c')
        assert calls[0].kwarguments == (('t', 'update_q_t'),)

    # Ensure a single outer parallel loop in driver
    with pragmas_attached(driver, Loop):
        driver_loops = FindNodes(Loop).visit(driver.body)
        assert len(driver_loops) == 1
        assert driver_loops[0].variable == 'b'
        assert driver_loops[0].bounds == '1:nb'
        assert driver_loops[0].pragma[0].keyword == 'acc'
        assert driver_loops[0].pragma[0].content == 'parallel loop gang vector_length(nlon)'

        # Ensure we have a kernel call in the driver loop
        kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
        assert len(kernel_calls) == 1
        assert kernel_calls[0].name == 'compute_column'

    # Ensure that the intermediate kernel contains two wrapped loops and an unwrapped call statement
    with pragmas_attached(outer_kernel, Loop):
        outer_kernel_loops = FindNodes(Loop).visit(outer_kernel.body)
        assert len(outer_kernel_loops) == 2
        assert outer_kernel_loops[0].variable == 'jl'
        assert outer_kernel_loops[0].bounds == 'start:end'
        assert outer_kernel_loops[0].pragma[0].keyword == 'acc'
        assert outer_kernel_loops[0].pragma[0].content == 'loop vector'
        assert outer_kernel_loops[1].variable == 'jl'
        assert outer_kernel_loops[1].bounds == 'start:end'
        assert outer_kernel_loops[1].pragma[0].keyword == 'acc'
        assert outer_kernel_loops[1].pragma[0].content == 'loop vector'

        # Ensure we still have a call, but not in the loops
        assert len(FindNodes(CallStatement).visit(outer_kernel_loops[0])) == 0
        assert len(FindNodes(CallStatement).visit(outer_kernel_loops[1])) == 0
        assert len(FindNodes(CallStatement).visit(outer_kernel.body)) == 1

        # Ensure the routine has been marked properly
        outer_kernel_pragmas = FindNodes(Pragma).visit(outer_kernel.ir)
        assert len(outer_kernel_pragmas) == 3
        assert outer_kernel_pragmas[0].keyword == 'acc'
        assert outer_kernel_pragmas[0].content == 'routine vector'
        assert outer_kernel_pragmas[1].keyword == 'acc'
        assert outer_kernel_pragmas[1].content == 'data present(q, update_q_t)'
        assert outer_kernel_pragmas[2].keyword == 'acc'
        assert outer_kernel_pragmas[2].content == 'end data'

    # Ensure that the leaf kernel contains two nested loops
    with pragmas_attached(inner_kernel, Loop):
        inner_kernel_loops = FindNodes(Loop).visit(inner_kernel.body)
        assert len(inner_kernel_loops) == 2
        assert inner_kernel_loops[1] in FindNodes(Loop).visit(inner_kernel_loops[0].body)
        assert inner_kernel_loops[0].variable == 'jl'
        assert inner_kernel_loops[0].bounds == 'start:end'
        assert inner_kernel_loops[0].pragma[0].keyword == 'acc'
        assert inner_kernel_loops[0].pragma[0].content == 'loop vector'
        assert inner_kernel_loops[1].variable == 'jk'
        assert inner_kernel_loops[1].bounds == '2:nz'
        assert inner_kernel_loops[1].pragma[0].keyword == 'acc'
        assert inner_kernel_loops[1].pragma[0].content == 'loop seq'

        # Ensure the routine has been marked properly
        inner_kernel_pragmas = FindNodes(Pragma).visit(inner_kernel.ir)
        assert len(inner_kernel_pragmas) == 3
        assert inner_kernel_pragmas[0].keyword == 'acc'
        assert inner_kernel_pragmas[0].content == 'routine vector'
        assert outer_kernel_pragmas[1].keyword == 'acc'
        assert outer_kernel_pragmas[1].content == 'data present(q, update_q_t)'
        assert outer_kernel_pragmas[2].keyword == 'acc'
        assert outer_kernel_pragmas[2].content == 'end data'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_hoist_nested_inline_openacc(frontend, horizontal, vertical, blocking):
    """
    Test the correct addition of OpenACC pragmas to SCC format code
    when hoisting array temporaries to driver.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_outer_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + 1.0
    END DO

    !$loki inline
    call update_q(start, end, nlon, nz, q, c)

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""

    fcode_inner_kernel = """
  SUBROUTINE update_q(start, end, nlon, nz, q, c)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL, INTENT(IN)    :: c
    REAL :: t(nlon,nz)
    INTEGER :: jl, jk

    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO
  END SUBROUTINE update_q
"""

    # Mimic the scheduler internal mechanis to apply the transformation cascade
    outer_kernel_source = Sourcefile.from_source(fcode_outer_kernel, frontend=frontend)
    inner_kernel_source = Sourcefile.from_source(fcode_inner_kernel, frontend=frontend)
    driver_source = Sourcefile.from_source(fcode_driver, frontend=frontend)
    driver = driver_source['column_driver']
    outer_kernel = outer_kernel_source['compute_column']
    inner_kernel = inner_kernel_source['update_q']
    outer_kernel.enrich(inner_kernel)  # Attach kernel source to driver call
    driver.enrich(outer_kernel)  # Attach kernel source to driver call

    driver_item = ProcedureItem(name='#column_driver', source=driver)
    outer_kernel_item = ProcedureItem(name='#compute_column', source=outer_kernel)
    inner_kernel_item = ProcedureItem(name='#update_q', source=inner_kernel)

    scc_hoist = SCCHoistPipeline(
        horizontal=horizontal, block_dim=blocking,
        dim_vars=vertical.sizes, directive='openacc'
    )

    InlineTransformation(allowed_aliases=horizontal.index).apply(outer_kernel)

    # Apply in reverse order to ensure hoisting analysis gets run on kernel first
    scc_hoist.apply(inner_kernel, role='kernel', item=inner_kernel_item)
    scc_hoist.apply(
        outer_kernel, role='kernel', item=outer_kernel_item,
        targets=['compute_q'], successors=()
    )
    scc_hoist.apply(
        driver, role='driver', item=driver_item,
        targets=['compute_column'], successors=(outer_kernel_item,)
    )

    # Ensure calls have correct arguments
    # driver
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('start', 'end', 'nlon', 'nz', 'q(:, :, b)',
            'compute_column_t(:, :, b)')

    # Ensure a single outer parallel loop in driver
    with pragmas_attached(driver, Loop):
        driver_loops = FindNodes(Loop).visit(driver.body)
        assert len(driver_loops) == 1
        assert driver_loops[0].variable == 'b'
        assert driver_loops[0].bounds == '1:nb'
        assert driver_loops[0].pragma[0].keyword == 'acc'
        assert driver_loops[0].pragma[0].content == 'parallel loop gang vector_length(nlon)'

        # Ensure we have a kernel call in the driver loop
        kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
        assert len(kernel_calls) == 1
        assert kernel_calls[0].name == 'compute_column'

    # Ensure that the intermediate kernel contains two wrapped loops and an unwrapped call statement
    with pragmas_attached(outer_kernel, Loop):
        outer_kernel_loops = FindNodes(Loop).visit(outer_kernel.body)
        assert len(outer_kernel_loops) == 2
        assert outer_kernel_loops[0].variable == 'jl'
        assert outer_kernel_loops[0].bounds == 'start:end'
        assert outer_kernel_loops[0].pragma[0].keyword == 'acc'
        assert outer_kernel_loops[0].pragma[0].content == 'loop vector'

        # check correctly nested vertical loop from inlined routine
        assert outer_kernel_loops[1] in FindNodes(Loop).visit(outer_kernel_loops[0].body)

        # Ensure the call was inlined
        assert not FindNodes(CallStatement).visit(outer_kernel.body)

        # Ensure the routine has been marked properly
        outer_kernel_pragmas = FindNodes(Pragma).visit(outer_kernel.ir)
        assert len(outer_kernel_pragmas) == 3
        assert outer_kernel_pragmas[0].keyword == 'acc'
        assert outer_kernel_pragmas[0].content == 'routine vector'
        assert outer_kernel_pragmas[1].keyword == 'acc'
        assert outer_kernel_pragmas[1].content == 'data present(q, t)'
        assert outer_kernel_pragmas[2].keyword == 'acc'
        assert outer_kernel_pragmas[2].content == 'end data'
