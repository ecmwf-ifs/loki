# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree

import pytest

from loki import (
    OMNI, OFP, Subroutine, Dimension, FindNodes, Loop, Assignment,
    CallStatement, Conditional, Scalar, Array, Pragma, pragmas_attached,
    fgen, Sourcefile, Section, ProcedureItem, ModuleItem, pragma_regions_attached, PragmaRegion,
    is_loki_pragma, IntLiteral, RangeIndex, Comment, HoistTemporaryArraysAnalysis,
    gettempdir, Scheduler, SchedulerConfig, SanitiseTransformation, InlineTransformation
)
from conftest import available_frontends
from transformations import (
    DataOffloadTransformation, SCCBaseTransformation, SCCDevectorTransformation,
    SCCDemoteTransformation, SCCRevectorTransformation, SCCAnnotateTransformation,
    SCCHoistTemporaryArraysTransformation, SCCVectorPipeline
)
#pylint: disable=too-many-lines

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'), aliases=('nproma',))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_revector_transformation(frontend, horizontal):
    """
    Test removal of vector loops in kernel and re-insertion of a single
    hoisted horizontal loop in the kernel.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, t, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b), t(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
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
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 3

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(driver, role='driver')
        transform.apply(kernel, role='kernel')

    # Ensure we have two nested loops in the kernel
    # (the hoisted horizontal and the native vertical)
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[1] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[0].bounds == 'start:end'
    assert kernel_loops[1].variable == 'jk'
    assert kernel_loops[1].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure driver remains unaffected
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 1
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'

    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    assert kernel_calls[0].name == 'compute_column'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_base_resolve_vector_notation(frontend, horizontal):
    """
    Test resolving of vector notation in kernel.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      t(start:end, jk) = c * jk
      q(start:end, jk) = q(start:end, jk-1) + t(start:end, jk) * c
    END DO
  END SUBROUTINE compute_column
"""

    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_transform = SCCBaseTransformation(horizontal=horizontal)
    scc_transform.apply(kernel, role='kernel')

    # Ensure horizontal loop variable has been declared
    assert 'jl' in kernel.variables

    # Ensure we have three loops in the kernel,
    # horizontal loops should be nested within vertical
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 3
    assert kernel_loops[1] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[2] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[1].variable == 'jl'
    assert kernel_loops[1].bounds == 'start:end'
    assert kernel_loops[2].variable == 'jl'
    assert kernel_loops[2].bounds == 'start:end'
    assert kernel_loops[0].variable == 'jk'
    assert kernel_loops[0].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_base_masked_statement(frontend, horizontal):
    """
    Test resolving of masked statements in kernel.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      WHERE (q(start:end, jk) > 1.234)
        q(start:end, jk) = q(start:end, jk-1) + t(start:end, jk) * c
      ELSEWHERE
        q(start:end, jk) = t(start:end, jk)
      END WHERE
    END DO
  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_transform = SCCBaseTransformation(horizontal=horizontal)
    scc_transform.apply(kernel, role='kernel')

    # Ensure horizontal loop variable has been declared
    assert 'jl' in kernel.variables

    # Ensure we have three loops in the kernel,
    # horizontal loops should be nested within vertical
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[1] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[1].variable == 'jl'
    assert kernel_loops[1].bounds == 'start:end'
    assert kernel_loops[0].variable == 'jk'
    assert kernel_loops[0].bounds == '2:nz'

    # Ensure that the respective conditional has been inserted correctly
    kernel_conds = FindNodes(Conditional).visit(kernel.body)
    assert len(kernel_conds) == 1
    assert kernel_conds[0] in FindNodes(Conditional).visit(kernel_loops[1])
    assert kernel_conds[0].condition == 'q(jl, jk) > 1.234'
    assert fgen(kernel_conds[0].body) == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(kernel_conds[0].else_body) == 'q(jl, jk) = t(jl, jk)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_demote_transformation(frontend, horizontal):
    """
    Test that local array variables that do not buffer values
    between vector sections and whose size is known at compile-time
    are demoted.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)
    REAL :: a(nlon)
    REAL :: b(nlon,psize)
    REAL :: unused(nlon)
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

      Q(JL, NZ) = Q(JL, NZ) * C + b(jl, 3)
    END DO
  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Must run SCCDevector first because demotion relies on knowledge
    # of vector sections
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')

    # Ensure correct array variables shapes
    assert isinstance(kernel.variable_map['a'], Scalar)
    assert isinstance(kernel.variable_map['b'], Array)
    assert isinstance(kernel.variable_map['c'], Scalar)
    assert isinstance(kernel.variable_map['t'], Array)
    assert isinstance(kernel.variable_map['q'], Array)
    assert isinstance(kernel.variable_map['unused'], Scalar)

    # Ensure that parameter-sized array b got demoted only
    assert kernel.variable_map['b'].shape == ((3,) if frontend is OMNI else ('psize',))
    assert kernel.variable_map['t'].shape == ('nlon', 'nz')
    assert kernel.variable_map['q'].shape == ('nlon', 'nz')

    # Ensure relevant expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'a = q(jl, 1)'
    assert fgen(assigns[4]).lower() == 'b(1) = q(jl, 2)'
    assert fgen(assigns[5]).lower() == 'b(2) = q(jl, 3)'
    assert fgen(assigns[6]).lower() == 'b(3) = a*(b(1) + b(2))'
    assert fgen(assigns[7]).lower() == 'q(jl, nz) = q(jl, nz)*c + b(3)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_hoist_multiple_kernels(frontend, horizontal, vertical, blocking):
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

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(driver, role='driver', item=driver_item, targets=['compute_column'])
        transform.apply(kernel, role='kernel', item=kernel_item)

    # Now apply the hoisting passes (anaylisis in reverse order)
    analysis = HoistTemporaryArraysAnalysis(dim_vars=(vertical.size,))
    synthesis = SCCHoistTemporaryArraysTransformation(block_dim=blocking)
    analysis.apply(kernel, role='kernel', item=kernel_item)
    analysis.apply(driver, role='driver', item=driver_item, successors=(kernel_item,))
    synthesis.apply(driver, role='driver', item=driver_item, successors=(kernel_item,))
    synthesis.apply(kernel, role='kernel', item=kernel_item)

    annotate = SCCAnnotateTransformation(
        horizontal=horizontal, vertical=vertical, directive='openacc', block_dim=blocking
    )
    annotate.apply(driver, role='driver', item=driver_item, targets=['compute_column'])
    annotate.apply(kernel, role='kernel', item=kernel_item)

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
def test_scc_hoist_multiple_kernels_loops(frontend, trim_vector_sections, horizontal, vertical, blocking):
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

    basedir = gettempdir() / 'test_scc_hoist_multiple_kernels_loops'
    basedir.mkdir(exist_ok=True)
    (basedir / 'driver.F90').write_text(fcode_driver)
    (basedir / 'kernel.F90').write_text(fcode_kernel)

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
    scheduler = Scheduler(paths=[basedir], config=SchedulerConfig.from_dict(config), frontend=frontend)

    driver = scheduler["#driver"].ir
    kernel = scheduler["kernel_mod#kernel"].ir

    transformation = (SCCBaseTransformation(horizontal=horizontal, directive='openacc'),)
    transformation += (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
    transformation += (SCCDemoteTransformation(horizontal=horizontal),)
    transformation += (SCCRevectorTransformation(horizontal=horizontal),)
    transformation += (SCCAnnotateTransformation(
        horizontal=horizontal, vertical=vertical, directive='openacc', block_dim=blocking,
    ),)
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

    rmtree(basedir)


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_annotate_openacc(frontend, horizontal, vertical, blocking):
    """
    Test the correct addition of OpenACC pragmas to SCC format code (no hoisting).
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
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(kernel)  # Attach kernel source to driver call

    # Test OpenACC annotations on non-hoisted version
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical, directive='openacc',
                                              block_dim=blocking),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column'])
        transform.apply(kernel, role='kernel')

    # Ensure routine is anntoated at vector level
    pragmas = FindNodes(Pragma).visit(kernel.ir)
    assert len(pragmas) == 5
    assert pragmas[0].keyword == 'acc'
    assert pragmas[0].content == 'routine vector'
    assert pragmas[1].keyword == 'acc'
    assert pragmas[1].content == 'data present(q)'
    assert pragmas[-1].keyword == 'acc'
    assert pragmas[-1].content == 'end data'

    # Ensure vector and seq loops are annotated, including privatized variable `b`
    with pragmas_attached(kernel, Loop):
        kernel_loops = FindNodes(Loop).visit(kernel.ir)
        assert len(kernel_loops) == 2
        assert kernel_loops[0].pragma[0].keyword == 'acc'
        assert kernel_loops[0].pragma[0].content == 'loop vector private(b)'
        assert kernel_loops[1].pragma[0].keyword == 'acc'
        assert kernel_loops[1].pragma[0].content == 'loop seq'

    # Ensure a single outer parallel loop in driver
    with pragmas_attached(driver, Loop):
        driver_loops = FindNodes(Loop).visit(driver.body)
        assert len(driver_loops) == 1
        assert driver_loops[0].pragma[0].keyword == 'acc'
        assert driver_loops[0].pragma[0].content == 'parallel loop gang vector_length(nlon)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_hoist_openacc(frontend, horizontal, vertical, blocking):
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
"""

    fcode_module = """
module my_scaling_value_mod
    implicit none
    REAL :: c = 5.345
end module my_scaling_value_mod
""".strip()

    # Mimic the scheduler internal mechanis to apply the transformation cascade
    kernel_source = Sourcefile.from_source(fcode_kernel, frontend=frontend)
    driver_source = Sourcefile.from_source(fcode_driver, frontend=frontend)
    module_source = Sourcefile.from_source(fcode_module, frontend=frontend)
    driver = driver_source['column_driver']
    kernel = kernel_source['compute_column']
    module = module_source['my_scaling_value_mod']
    kernel.enrich(module)
    driver.enrich(kernel)  # Attach kernel source to driver call

    driver_item = ProcedureItem(name='#column_driver', source=driver_source)
    kernel_item = ProcedureItem(name='#compute_column', source=kernel_source)
    module_item = ModuleItem(name='my_scaling_value_mod', source=module_source)

    # Test OpenACC annotations on hoisted version
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(driver, role='driver', item=driver_item, targets=['compute_column'], successors=[kernel_item])
        transform.apply(kernel, role='kernel', item=kernel_item, successors=[module_item])

    # Now apply the hoisting passes (anaylisis in reverse order)
    analysis = HoistTemporaryArraysAnalysis(dim_vars=(vertical.size,))
    synthesis = SCCHoistTemporaryArraysTransformation(block_dim=blocking)

    # The try-except is for checking a bug where HoistTemporaryArraysAnalysis would
    # access a GlobalVarImportItem, which should not happen. Note that in case of a KeyError (which signifies
    # the issue occurring), an explicit pytest failure is thrown to signify that there is no bug in the test itself.
    try:
        analysis.apply(kernel, role='kernel', item=kernel_item, successors=(module_item,))
    except KeyError:
        pytest.fail('`HoistTemporaryArraysAnalysis` should not attempt to access `GlobalVarImportItem`s')
    analysis.apply(driver, role='driver', item=driver_item, successors=(kernel_item,))
    synthesis.apply(driver, role='driver', item=driver_item, successors=(kernel_item,))
    synthesis.apply(kernel, role='kernel', item=kernel_item, successors=(module_item,))

    annotate = SCCAnnotateTransformation(
        horizontal=horizontal, vertical=vertical, directive='openacc', block_dim=blocking
    )
    annotate.apply(driver, role='driver', item=driver_item, targets=['compute_column'])
    annotate.apply(kernel, role='kernel', item=kernel_item)

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
def test_single_column_coalesced_hoist_nested_openacc(frontend, horizontal, vertical, blocking,
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

    # Test OpenACC annotations on hoisted version
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(driver, role='driver', item=driver_item, targets=['compute_column'])
        transform.apply(outer_kernel, role='kernel', item=outer_kernel_item, targets=['compute_q'])
        transform.apply(inner_kernel, role='kernel', item=inner_kernel_item)

    # Now apply the hoisting passes (anaylisis in reverse order)
    analysis = HoistTemporaryArraysAnalysis(dim_vars=(vertical.size,))
    synthesis = SCCHoistTemporaryArraysTransformation(block_dim=blocking, as_kwarguments=as_kwarguments)
    # analysis reverse order
    analysis.apply(inner_kernel, role='kernel', item=inner_kernel_item)
    analysis.apply(outer_kernel, role='kernel', item=outer_kernel_item, successors=(inner_kernel_item,))
    analysis.apply(driver, role='driver', item=driver_item, successors=(outer_kernel_item,))
    # synthesis
    synthesis.apply(driver, role='driver', item=driver_item, successors=(outer_kernel_item,))
    synthesis.apply(outer_kernel, role='kernel', item=outer_kernel_item, successors=(inner_kernel_item,))
    synthesis.apply(inner_kernel, role='kernel', item=outer_kernel_item)

    annotate = SCCAnnotateTransformation(
        horizontal=horizontal, vertical=vertical, directive='openacc', block_dim=blocking
    )
    annotate.apply(driver, role='driver', item=driver_item, targets=['compute_column'])
    annotate.apply(outer_kernel, role='kernel', item=outer_kernel_item, targets=['update_q'])
    annotate.apply(inner_kernel, role='kernel', item=outer_kernel_item)

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
def test_single_column_coalesced_nested(frontend, horizontal, vertical, blocking):
    """
    Test the correct handling of nested vector-level routines in SCC.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    associate(x => q)
    do b=1, nb
      call compute_column(start, end, nlon, nz, x(:,:,b))
    end do
    end associate
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

    outer_kernel = Subroutine.from_source(fcode_outer_kernel, frontend=frontend)
    inner_kernel = Subroutine.from_source(fcode_inner_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    outer_kernel.enrich(inner_kernel)  # Attach kernel source to driver call
    driver.enrich(outer_kernel)  # Attach kernel source to driver call

    # Instantial SCCVector pipeline and apply
    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking, directive='openacc'
    )
    scc_pipeline.apply(driver, role='driver', targets=['compute_column'])
    scc_pipeline.apply(outer_kernel, role='kernel', targets=['compute_q'])
    scc_pipeline.apply(inner_kernel, role='kernel')

    # Apply annotate twice to test bailing out mechanism
    scc_annotate = SCCAnnotateTransformation(
        horizontal=horizontal, vertical=vertical, directive='openacc', block_dim=blocking
    )
    scc_annotate.apply(driver, role='driver', targets=['compute_column'])
    scc_annotate.apply(outer_kernel, role='kernel', targets=['compute_q'])
    scc_annotate.apply(inner_kernel, role='kernel')

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
        assert outer_kernel_pragmas[1].content == 'data present(q)'
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
        assert outer_kernel_pragmas[1].content == 'data present(q)'
        assert outer_kernel_pragmas[2].keyword == 'acc'
        assert outer_kernel_pragmas[2].content == 'end data'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_outer_loop(frontend, horizontal, vertical, blocking):
    """
    Test the correct handling of an outer loop that breaks scoping.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk, niter
    LOGICAL :: maybe
    REAL :: c

    if (maybe)  call logger()

    c = 5.345
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + 3.0
    END DO

    DO niter = 1, 3

      DO JL = START, END
        Q(JL, NZ) = Q(JL, NZ) + 1.0
      END DO

      call update_q(start, end, nlon, nz, q, c)

    END DO

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO

    IF (.not. maybe) THEN
      call update_q(start, end, nlon, nz, q, c)
    END IF

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + C * 3.
    END DO

    IF (maybe)  call logger()

  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Test SCC transform for kernel with scope-splitting outer loop
    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking, directive='openacc'
    )
    scc_pipeline.apply(kernel, role='kernel')

    # Ensure that we capture vector loops outside the outer vertical loop, as well as the one vector loop inside it.
    with pragmas_attached(kernel, Loop):
        kernel_loops = FindNodes(Loop).visit(kernel.body)
        assert len(kernel_loops) == 5
        assert kernel_loops[2] in kernel_loops[1].body

        assert kernel_loops[0].variable == 'jl'
        assert kernel_loops[0].bounds == 'start:end'
        assert kernel_loops[0].pragma[0].keyword == 'acc'
        assert kernel_loops[0].pragma[0].content == 'loop vector'
        assert kernel_loops[1].variable == 'niter'
        assert kernel_loops[1].bounds == '1:3'
        assert kernel_loops[1].pragma[0].keyword == 'acc'
        assert kernel_loops[1].pragma[0].content == 'loop seq'
        assert kernel_loops[2].variable == 'jl'
        assert kernel_loops[2].bounds == 'start:end'
        assert kernel_loops[2].pragma[0].keyword == 'acc'
        assert kernel_loops[2].pragma[0].content == 'loop vector'
        assert kernel_loops[3].variable == 'jl'
        assert kernel_loops[3].bounds == 'start:end'
        assert kernel_loops[3].pragma[0].keyword == 'acc'
        assert kernel_loops[3].pragma[0].content == 'loop vector'
        assert kernel_loops[4].variable == 'jl'
        assert kernel_loops[4].bounds == 'start:end'
        assert kernel_loops[4].pragma[0].keyword == 'acc'
        assert kernel_loops[4].pragma[0].content == 'loop vector'

        # Ensure we still have a call, but only in the outer counter loop
        assert len(FindNodes(CallStatement).visit(kernel_loops[0])) == 0
        assert len(FindNodes(CallStatement).visit(kernel_loops[1])) == 1
        assert len(FindNodes(CallStatement).visit(kernel_loops[2])) == 0
        assert len(FindNodes(CallStatement).visit(kernel_loops[3])) == 0
        assert len(FindNodes(CallStatement).visit(kernel_loops[4])) == 0
        assert len(FindNodes(CallStatement).visit(kernel.body)) == 4


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_devector_transformation(frontend, horizontal):
    """
    Test the correct identification of vector sections and removal of vector loops.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk, niter
    LOGICAL :: maybe
    REAL :: c

    if (maybe)  call logger()

    c = 5.345
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + 3.0
    END DO

    DO niter = 1, 3

      DO JL = START, END
        Q(JL, NZ) = Q(JL, NZ) + 1.0
      END DO

      call update_q(start, end, nlon, nz, q, c)

    END DO

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO

    IF (.not. maybe) THEN
      call update_q(start, end, nlon, nz, q, c)
    END IF

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + C * 3.
    END DO

    IF (maybe)  call logger()

  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Check number of horizontal loops prior to transformation
    loops = [l for l in FindNodes(Loop).visit(kernel.body) if l.variable == 'jl']
    assert len(loops) == 4

    # Test SCCDevector transform for kernel with scope-splitting outer loop
    scc_transform = SCCDevectorTransformation(horizontal=horizontal)
    scc_transform.apply(kernel, role='kernel')

    # Check removal of horizontal loops
    loops = [l for l in FindNodes(Loop).visit(kernel.body) if l.variable == 'jl']
    assert not loops

    # Check number and content of vector sections
    sections = [s for s in FindNodes(Section).visit(kernel.body) if s.label == 'vector_section']
    assert len(sections) == 4

    assigns = FindNodes(Assignment).visit(sections[0])
    assert len(assigns) == 2
    assigns = FindNodes(Assignment).visit(sections[1])
    assert len(assigns) == 1
    assigns = FindNodes(Assignment).visit(sections[2])
    assert len(assigns) == 1
    assigns = FindNodes(Assignment).visit(sections[3])
    assert len(assigns) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_variable_demotion(frontend, horizontal):
    """
    Test the correct demotion of an outer loop that breaks scoping.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL :: a(nlon), b(nlon), c(nlon)
    INTEGER :: jl, jk, niter

    DO JL = START, END
      A(JL) = A(JL) + 3.0
      B(JL) = B(JL) + 1.0
    END DO

    DO niter = 1, 3

      DO JL = START, END
        B(JL) = B(JL) + 1.0
      END DO

    END DO

    call update_q(start, end, nlon, nz)

    DO JL = START, END
      A(JL) = A(JL) + 3.0
      C(JL) = C(JL) + 1.0
    END DO

  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Test SCC transform for kernel with scope-splitting outer loop
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')

    # Ensure that only a has not been demoted, as it buffers information across the subroutine call.
    assert isinstance(kernel.variable_map['a'], Array)
    assert isinstance(kernel.variable_map['b'], Scalar)
    assert isinstance(kernel.variable_map['c'], Scalar)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP,
                         'OFP fails to parse multiconditional with embedded call.')]))
def test_single_column_coalesced_multicond(frontend, horizontal, vertical, blocking):
    """
    Test if horizontal loops in multiconditionals with CallStatements are
    correctly transformed.
    """

    fcode = """
    subroutine test(icase, start, end, work)
    implicit none

      integer, intent(in) :: icase, start, end
      real, dimension(start:end), intent(inout) :: work
      integer :: jl

      select case(icase)
      case(1)
        work(start:end) = 1.
      case(2)
        do jl = start,end
           work(jl) = work(jl) + 2.
        enddo
      case(3)
        do jl = start,end
           work(jl) = work(jl) + 3.
        enddo
        call some_kernel(start, end, work)
      case default
        work(start:end) = 0.
      end select

    end subroutine test
    """

    kernel = Subroutine.from_source(fcode, frontend=frontend)

    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking, directive='openacc'
    )
    scc_pipeline.apply(kernel, role='kernel')

    # Ensure we have three vector loops in the kernel
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 4
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[1].variable == 'jl'
    assert kernel_loops[2].variable == 'jl'
    assert kernel_loops[3].variable == 'jl'

    # Check acc pragmas of newly created vector loops
    pragmas = FindNodes(Pragma).visit(kernel.ir)
    assert len(pragmas) == 7
    assert pragmas[2].keyword == 'acc'
    assert pragmas[2].content == 'loop vector'
    assert pragmas[3].keyword == 'acc'
    assert pragmas[3].content == 'loop vector'
    assert pragmas[4].keyword == 'acc'
    assert pragmas[4].content == 'loop vector'
    assert pragmas[5].keyword == 'acc'
    assert pragmas[5].content == 'loop vector'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_multiple_acc_pragmas(frontend, horizontal, vertical, blocking):
    """
    Test that both '!$acc data' and '!$acc parallel loop gang' pragmas are created at the
    driver layer.
    """

    fcode = """
    subroutine test(work, nlon, nb)
    implicit none

      integer, intent(in) :: nb, nlon
      real, dimension(nlon, nb), intent(inout) :: work
      integer :: b

      !$loki data
      !$omp parallel do private(b) shared(work, nproma)
        do b=1, nb
           call some_kernel(nlon, work(:,b))
        enddo
      !$omp end parallel do
      !$loki end data

    end subroutine test

    subroutine some_kernel(nlon, work)
    implicit none

      integer, intent(in) :: nlon
      real, dimension(nlon), intent(inout) :: work
      integer :: jl

      do jl=1,nlon
         work(jl) = work(jl) + 1.
      enddo

    end subroutine some_kernel
    """

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test']
    routine.enrich(source.all_subroutines)

    data_offload = DataOffloadTransformation(remove_openmp=True)
    data_offload.transform_subroutine(routine, role='driver', targets=['some_kernel',])

    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking, directive='openacc'
    )
    scc_pipeline.apply(routine, role='driver', targets=['some_kernel',])

    # Check that both acc pragmas are created
    pragmas = FindNodes(Pragma).visit(routine.ir)
    assert len(pragmas) == 4
    assert pragmas[0].keyword == 'acc'
    assert pragmas[1].keyword == 'acc'
    assert pragmas[2].keyword == 'acc'
    assert pragmas[3].keyword == 'acc'

    assert 'data' in pragmas[0].content
    assert 'copy' in pragmas[0].content
    assert '(work)' in pragmas[0].content
    assert pragmas[1].content == 'parallel loop gang vector_length(nlon)'
    assert pragmas[2].content == 'end parallel loop'
    assert pragmas[3].content == 'end data'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_base_routine_seq_pragma(frontend, horizontal):
    """
    Test that `!$loki routine seq` pragmas are replaced correctly by `!$acc routine seq` pragmas.
    """

    fcode = """
    subroutine some_kernel(work, nang)
       implicit none

       integer, intent(in) :: nang
       real, dimension(nang), intent(inout) :: work
!$loki routine seq
       integer :: k

       do k=1,nang
          work(k) = 1.
       enddo

    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)

    pragmas = FindNodes(Pragma).visit(routine.spec)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'loki'
    assert pragmas[0].content == 'routine seq'

    transformation = SCCBaseTransformation(horizontal=horizontal, directive='openacc')
    transformation.transform_subroutine(routine, role='kernel', targets=['some_kernel',])

    pragmas = FindNodes(Pragma).visit(routine.spec)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'acc'
    assert pragmas[0].content == 'routine seq'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_vector_inlined_call(frontend, horizontal):
    """
    Test that calls targeted for inlining inside a vector region are not treated as separators
    """

    fcode = """
    subroutine some_inlined_kernel(work)
    !$loki routine seq
       real, intent(inout) :: work

       work = work*2.
    end subroutine some_inlined_kernel

    subroutine some_kernel(start, end, nlon, work, cond)
       logical, intent(in) :: cond
       integer, intent(in) :: nlon, start, end
       real, dimension(nlon), intent(inout) :: work

       integer :: jl

       do jl=start,end
          if(cond)then
             call some_inlined_kernel(work(jl))
          endif
          work(jl) = work(jl) + 1.
       enddo

       call some_other_kernel()

    end subroutine some_kernel
    """

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['some_kernel']
    inlined_routine = source['some_inlined_kernel']
    routine.enrich((inlined_routine,))

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel', 'some_inlined_kernel'])

    # Check loki pragma has been removed
    assert not FindNodes(Pragma).visit(routine.body)

    # Check that 'some_inlined_kernel' remains within vector-parallel region
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    calls = FindNodes(CallStatement).visit(loops[0].body)
    assert len(calls) == 1
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_vector_reduction(frontend, horizontal, vertical, blocking):
    """
    Test for the insertion of OpenACC vector reduction directives.
    """

    fcode = """
    subroutine some_kernel(start, end, nlon, mij)
       integer, intent(in) :: nlon, start, end
       integer, dimension(nlon), intent(in) :: mij

       integer :: jl, maxij

       maxij = -1
       !$loki vector-reduction( mAx:maXij )
       do jl=start,end
          maxij = max(maxij, mij(jl))
       enddo
       !$loki end vector-reduction( mAx:maXij )

    end subroutine some_kernel
    """

    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking, directive='openacc'
    )

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['some_kernel']

    with pragma_regions_attached(routine):
        region = FindNodes(PragmaRegion).visit(routine.body)
        assert is_loki_pragma(region[0].pragma, starts_with = 'vector-reduction')


    scc_pipeline.apply(routine, role='kernel', targets=['some_kernel',])

    pragmas = FindNodes(Pragma).visit(routine.body)
    assert len(pragmas) == 3
    assert all(p.keyword == 'acc' for p in pragmas)

    # Check OpenACC directives have been inserted
    with pragmas_attached(routine, Loop):
        loops = FindNodes(Loop).visit(routine.body)
        assert len(loops) == 1
        assert loops[0].pragma[0].content == 'loop vector reduction( mAx:maXij )'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_demotion_parameter(frontend, horizontal):
    """
    Test that temporary arrays with compile-time constants are marked for demotion.
    """

    fcode_mod = """
    module YOWPARAM
       integer, parameter :: nang_param = 36
    end module YOWPARAM
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon, nang)
       use yowparam, only: nang_param
       implicit none

       integer, intent(in) :: nlon, start, end, nang
       real, dimension(nlon, nang_param, 2) :: work

       integer :: jl, k

       do jl=start,end
          do k=1,nang
             work(jl,k,1) = 1.
             work(jl,k,2) = 1.
          enddo
       enddo

    end subroutine some_kernel
    """

    source = Sourcefile.from_source(fcode_mod, frontend=frontend)
    routine = Subroutine.from_source(fcode_kernel, definitions=source.definitions,
                                     frontend=frontend)

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    assert len(routine.symbol_map['work'].shape) == 2
    if frontend == OMNI:
        assert routine.symbol_map['work'].shape == (RangeIndex(children=(IntLiteral(1), IntLiteral(36))),
                                                    IntLiteral(2))
    else:
        assert routine.symbol_map['work'].shape == ('nang_param', IntLiteral(2))


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_base_horizontal_bounds_checks(frontend, horizontal):
    """
    Test the SCCBaseTransformation checks for horizontal loop bounds.
    """

    fcode_no_start = """
    subroutine kernel(end, work)
      real, intent(inout) :: work
      integer, intent(in) :: end

    end subroutine kernel
"""

    fcode_no_end = """
    subroutine kernel(start, work)
      real, intent(inout) :: work
      integer, intent(in) :: start

    end subroutine kernel
"""

    no_start = Subroutine.from_source(fcode_no_start, frontend=frontend)
    no_end = Subroutine.from_source(fcode_no_end, frontend=frontend)

    transform = SCCBaseTransformation(horizontal=horizontal)
    with pytest.raises(RuntimeError):
        transform.apply(no_start, role='kernel')
    with pytest.raises(RuntimeError):
        transform.apply(no_end, role='kernel')


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_single_column_coalesced_vector_section_trim_simple(frontend, horizontal, trim_vector_sections):
    """
    Test the trimming of vector-sections to exclude scalar assignments.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical :: flag0
       real, dimension(nlon) :: work
       integer :: jl

       flag0 = .true.

       do jl=start,end
          work(jl) = 1.
       enddo
       ! random comment
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    assign = FindNodes(Assignment).visit(routine.body)[0]
    loop = FindNodes(Loop).visit(routine.body)[0]
    comment = [c for c in FindNodes(Comment).visit(routine.body) if c.text == '! random comment'][0]

    # check we found the right assignment
    assert assign.lhs.name.lower() == 'flag0'

    # check we found the right comment
    assert comment.text == '! random comment'

    if trim_vector_sections:
        assert assign not in loop.body
        assert assign in routine.body.body

        assert comment not in loop.body
        assert comment in routine.body.body
    else:
        assert assign in loop.body
        assert assign not in routine.body.body

        assert comment in loop.body
        assert comment not in routine.body.body


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_single_column_coalesced_vector_section_trim_nested(frontend, horizontal, trim_vector_sections):
    """
    Test the trimming of vector-sections to exclude nested scalar assignments.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon, flag0)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical, intent(in) :: flag0
       logical :: flag1, flag2
       real, dimension(nlon) :: work

       integer :: jl

       if(flag0)then
         flag1 = .true.
         flag2 = .false.
       else
         flag1 = .false.
         flag2 = .true.
       endif

       do jl=start,end
          work(jl) = 1.
       enddo
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    cond = FindNodes(Conditional).visit(routine.body)[0]
    loop = FindNodes(Loop).visit(routine.body)[0]

    if trim_vector_sections:
        assert cond not in loop.body
        assert cond in routine.body.body
    else:
        assert cond in loop.body
        assert cond not in routine.body.body


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_single_column_coalesced_vector_section_trim_complex(
        frontend, horizontal, vertical, blocking, trim_vector_sections
):
    """
    Test to highlight the limitations of vector-section trimming.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon, flag0)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical, intent(in) :: flag0
       logical :: flag1, flag2
       real, dimension(nlon) :: work, work1

       integer :: jl

       flag1 = .true.
       if(flag0)then
         flag2 = .false.
       else
         work1(start:end) = 1.
       endif

       do jl=start,end
          work(jl) = 1.
       enddo
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        directive='openacc', trim_vector_sections=trim_vector_sections
    )
    scc_pipeline.apply(routine, role='kernel', targets=['some_kernel',])

    assign = FindNodes(Assignment).visit(routine.body)[0]

    # check we found the right assignment
    assert assign.lhs.name.lower() == 'flag1'

    cond = FindNodes(Conditional).visit(routine.body)[0]
    loop = FindNodes(Loop).visit(routine.body)[0]

    assert cond in loop.body
    assert cond not in routine.body.body
    if trim_vector_sections:
        assert assign not in loop.body
        assert(len(FindNodes(Assignment).visit(loop.body)) == 3)
    else:
        assert assign in loop.body
        assert(len(FindNodes(Assignment).visit(loop.body)) == 4)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('inline_internals', [False, True])
@pytest.mark.parametrize('resolve_sequence_association', [False, True])
def test_single_column_coalesced_inline_and_sequence_association(
        frontend, horizontal, inline_internals, resolve_sequence_association
):
    """
    Test the combinations of routine inlining and sequence association
    """

    fcode_kernel = """
    subroutine some_kernel(nlon, start, end)
       implicit none

       integer, intent(in) :: nlon, start, end
       real, dimension(nlon) :: work

       call contained_kernel(work(1))

     contains

       subroutine contained_kernel(work)
          implicit none

          real, dimension(nlon) :: work
          integer :: jl

          do jl = start, end
             work(jl) = 1.
          enddo

       end subroutine contained_kernel
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Remove sequence association via SanitiseTransform
    sanitise_transform = SanitiseTransformation(
        resolve_sequence_association=resolve_sequence_association
    )
    sanitise_transform.apply(routine, role='kernel')

    # Create member inlining transformation to go along SCC
    inline_transform = InlineTransformation(inline_internals=inline_internals)

    scc_transform = SCCBaseTransformation(horizontal=horizontal)

    #Not really doing anything for contained routines
    if (not inline_internals and not resolve_sequence_association):
        inline_transform.apply(routine, role='kernel')
        scc_transform.apply(routine, role='kernel')

        assert len(routine.members) == 1
        assert not FindNodes(Loop).visit(routine.body)

    #Should fail because it can't resolve sequence association
    elif (inline_internals and not resolve_sequence_association):
        with pytest.raises(RuntimeError) as e_info:
            inline_transform.apply(routine, role='kernel')
            scc_transform.apply(routine, role='kernel')
        assert(e_info.exconly() ==
               'RuntimeError: [Loki::TransformInline] Cannot resolve procedure call to contained_kernel')

    #Check that the call is properly modified
    elif (not inline_internals and resolve_sequence_association):
        inline_transform.apply(routine, role='kernel')
        scc_transform.apply(routine, role='kernel')

        assert len(routine.members) == 1
        call = FindNodes(CallStatement).visit(routine.body)[0]
        assert fgen(call).lower() == 'call contained_kernel(work(1:nlon))'

    #Check that the contained subroutine has been inlined
    else:
        inline_transform.apply(routine, role='kernel')
        scc_transform.apply(routine, role='kernel')

        assert len(routine.members) == 0

        loop = FindNodes(Loop).visit(routine.body)[0]
        assert loop.variable == 'jl'
        assert loop.bounds == 'start:end'

        assign = FindNodes(Assignment).visit(loop.body)[0]
        assert fgen(assign).lower() == 'work(jl) = 1.'
