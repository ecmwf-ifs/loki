# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Sourcefile, Dimension, fgen
from loki.batch import ProcedureItem
from loki.expression import Scalar, Array, IntLiteral
from loki.frontend import available_frontends, OMNI, OFP
from loki.ir import (
    FindNodes, Assignment, CallStatement, Conditional, Loop,
    Pragma, PragmaRegion, pragmas_attached, is_loki_pragma,
    pragma_regions_attached
)

from loki.transformations import (
    DataOffloadTransformation, SanitiseTransformation,
    InlineTransformation, get_loop_bounds
)
from loki.transformations.single_column import (
    SCCBaseTransformation, SCCDevectorTransformation,
    SCCDemoteTransformation, SCCRevectorTransformation,
    SCCAnnotateTransformation, SCCVectorPipeline
)


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',)
    )

@pytest.fixture(scope='module', name='horizontal_bounds_aliases')
def fixture_horizontal_bounds_aliases():
    return Dimension(
        name='horizontal_bounds_aliases', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',),
        bounds_aliases=('bnds%start', 'bnds%end')
    )

@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


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
  SUBROUTINE compute_column(start, end, nlon, nproma, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    INTEGER, INTENT(IN) :: nproma      ! Horizontal size alias
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)
    REAL :: a(nproma)
    REAL :: b(nlon,psize)
    REAL :: unused(nlon)
    REAL :: d(nlon,psize)
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

      d(jl, 1) = b(jl, 1)
      d(jl, 2) = b(jl, 2)
      d(jl, 3) = b(jl, 3)

      Q(JL, NZ) = Q(JL, NZ) * C + b(jl, 3)
    END DO
  END SUBROUTINE compute_column
"""
    kernel_source = Sourcefile.from_source(fcode_kernel, frontend=frontend)
    kernel_item = ProcedureItem(name='#compute_column', source=kernel_source, config={'preserve_arrays': ['d',]})
    kernel = kernel_source.subroutines[0]

    # Must run SCCDevector first because demotion relies on knowledge
    # of vector sections
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel', item=kernel_item)

    # Ensure correct array variables shapes
    assert isinstance(kernel.variable_map['a'], Scalar)
    assert isinstance(kernel.variable_map['b'], Array)
    assert isinstance(kernel.variable_map['c'], Scalar)
    assert isinstance(kernel.variable_map['t'], Array)
    assert isinstance(kernel.variable_map['q'], Array)
    assert isinstance(kernel.variable_map['unused'], Scalar)
    assert isinstance(kernel.variable_map['d'], Array)

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
    assert fgen(assigns[7]).lower() == 'd(jl, 1) = b(1)'
    assert fgen(assigns[8]).lower() == 'd(jl, 2) = b(2)'
    assert fgen(assigns[9]).lower() == 'd(jl, 3) = b(3)'
    assert fgen(assigns[10]).lower() == 'q(jl, nz) = q(jl, nz)*c + b(3)'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('acc_data', ['default', 'copyin', None])
def test_scc_annotate_openacc(frontend, horizontal, blocking, acc_data):
    """
    Test the correct addition of OpenACC pragmas to SCC format code (no hoisting).
    """

    fcode_driver = f"""
  SUBROUTINE column_driver(nlon, nproma, nlev, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    INTEGER, INTENT(IN)   :: nproma, nlev  ! Aliases of horizontal and vertical sizes
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    REAL :: other_var(nlon)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    {'!$acc data default(present)' if acc_data == 'default' else ''}
    {'!$acc data copyin(other_var)' if acc_data == 'copyin' else ''}
    !
    do b=1, nb
      call compute_column(start, end, nlon, nproma, nz, q(:,:,b), other_var)
    end do
    !
    {'!$acc end data' if acc_data else ''}
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nproma, nlev, nz, q, other_var)
    INTEGER, INTENT(IN) :: start, end   ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz     ! Size of the horizontal and vertical
    INTEGER, INTENT(IN) :: nproma, nlev ! Aliases of horizontal and vertical sizes
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL, INTENT(IN) :: other_var
    REAL :: t(nlon,nz)
    REAL :: a(nlon)
    REAL :: d(nproma)
    REAL :: e(nlev)
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
    scc_transform += (SCCAnnotateTransformation(directive='openacc', block_dim=blocking),)
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
        assert driver_loops[0].pragma[0].keyword.lower() == 'acc'
        if acc_data:
            assert driver_loops[0].pragma[0].content == 'parallel loop gang vector_length(nlon)'
        else:
            assert driver_loops[0].pragma[0].content == 'parallel loop gang private(other_var) vector_length(nlon)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_nested(frontend, horizontal, blocking):
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
        horizontal=horizontal, block_dim=blocking, directive='openacc'
    )
    scc_pipeline.apply(driver, role='driver', targets=['compute_column'])
    scc_pipeline.apply(outer_kernel, role='kernel', targets=['compute_q'])
    scc_pipeline.apply(inner_kernel, role='kernel')

    # Apply annotate twice to test bailing out mechanism
    scc_annotate = SCCAnnotateTransformation(directive='openacc', block_dim=blocking)
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
def test_scc_outer_loop(frontend, horizontal, blocking):
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
        horizontal=horizontal, block_dim=blocking, directive='openacc'
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
def test_scc_variable_demotion(frontend, horizontal):
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
def test_scc_multicond(frontend, horizontal, blocking):
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
        horizontal=horizontal, block_dim=blocking, directive='openacc'
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
def test_scc_multiple_acc_pragmas(frontend, horizontal, blocking):
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
        horizontal=horizontal, block_dim=blocking, directive='openacc'
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
def test_scc_annotate_routine_seq_pragma(frontend, blocking):
    """
    Test that `!$loki routine seq` pragmas are replaced correctly by
    `!$acc routine seq` pragmas.
    """

    fcode = """
    subroutine some_kernel(work, nang)
       implicit none

       integer, intent(in) :: nang
       real, dimension(nang), intent(inout) :: work
       integer :: k
!$loki routine seq

       do k=1,nang
          work(k) = 1.
       enddo

    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)

    pragmas = FindNodes(Pragma).visit(routine.ir)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'loki'
    assert pragmas[0].content == 'routine seq'

    transformation = SCCAnnotateTransformation(directive='openacc', block_dim=blocking)
    transformation.transform_subroutine(routine, role='kernel', targets=['some_kernel',])

    # Ensure the routine pragma is in the first pragma in the spec
    pragmas = FindNodes(Pragma).visit(routine.spec)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'acc'
    assert pragmas[0].content == 'routine seq'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_annotate_empty_data_clause(frontend, blocking):
    """
    Test that we do not generate empty `!$acc data` clauses.
    """

    fcode = """
    subroutine some_kernel(n)
       implicit none
       ! Scalars should not show up in `!$acc data` clause
       integer, intent(inout) :: n
!$loki routine seq
       integer :: k

       k = n
       do k=1, 3
          n = k + 1
       enddo
    end subroutine some_kernel
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    pragmas = FindNodes(Pragma).visit(routine.ir)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'loki'
    assert pragmas[0].content == 'routine seq'

    transformation = SCCAnnotateTransformation(directive='openacc', block_dim=blocking)
    transformation.transform_subroutine(routine, role='kernel', targets=['some_kernel',])

    # Ensure the routine pragma is in the first pragma in the spec
    pragmas = FindNodes(Pragma).visit(routine.ir)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'acc'
    assert pragmas[0].content == 'routine seq'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_vector_reduction(frontend, horizontal, blocking):
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
        horizontal=horizontal, block_dim=blocking, directive='openacc'
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
def test_scc_demotion_parameter(frontend, horizontal, tmp_path):
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

    source = Sourcefile.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode_kernel, definitions=source.definitions,
                                     frontend=frontend, xmods=[tmp_path])

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    assert len(routine.symbol_map['work'].shape) == 2
    if frontend == OMNI:
        assert routine.symbol_map['work'].shape == (IntLiteral(36), IntLiteral(2))
    else:
        assert routine.symbol_map['work'].shape == ('nang_param', IntLiteral(2))


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_base_horizontal_bounds_checks(frontend, horizontal, horizontal_bounds_aliases, tmp_path):
    """
    Test the SCCBaseTransformation checks for horizontal loop bounds.
    """

    fcode = """
subroutine kernel(start, end, work)
    real, intent(inout) :: work
    integer, intent(in) :: start, end

end subroutine kernel
    """.strip()

    fcode_no_start = """
subroutine kernel(end, work)
    real, intent(inout) :: work
    integer, intent(in) :: end

end subroutine kernel
    """.strip()

    fcode_no_end = """
subroutine kernel(start, work)
    real, intent(inout) :: work
    integer, intent(in) :: start

end subroutine kernel
    """.strip()

    fcode_alias = """
module bnds_type_mod
    implicit none
    type bnds_type
        integer :: start
        integer :: end
    end type bnds_type
end module bnds_type_mod

subroutine kernel(bnds, work)
    use bnds_type_mod, only : bnds_type
    type(bnds_type), intent(in) :: bnds
    real, intent(inout) :: work

end subroutine kernel
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    no_start = Subroutine.from_source(fcode_no_start, frontend=frontend, xmods=[tmp_path])
    no_end = Subroutine.from_source(fcode_no_end, frontend=frontend, xmods=[tmp_path])
    alias = Sourcefile.from_source(fcode_alias, frontend=frontend, xmods=[tmp_path]).subroutines[0]

    transform = SCCBaseTransformation(horizontal=horizontal)
    with pytest.raises(RuntimeError):
        transform.apply(no_start, role='kernel')
    with pytest.raises(RuntimeError):
        transform.apply(no_end, role='kernel')

    transform = SCCBaseTransformation(horizontal=horizontal_bounds_aliases)
    transform.apply(alias, role='kernel')

    bounds = get_loop_bounds(routine, dimension=horizontal_bounds_aliases)
    assert bounds[0] == 'start'
    assert bounds[1] == 'end'

    bounds = get_loop_bounds(alias, dimension=horizontal_bounds_aliases)
    assert bounds[0] == 'bnds%start'
    assert bounds[1] == 'bnds%end'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('inline_internals', [False, True])
@pytest.mark.parametrize('resolve_sequence_association', [False, True])
def test_scc_inline_and_sequence_association(
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
