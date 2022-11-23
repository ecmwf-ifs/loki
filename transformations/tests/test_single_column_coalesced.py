# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    OMNI, Subroutine, Dimension, FindNodes, Loop, Assignment,
    CallStatement, Scalar, Array, Pragma, pragmas_attached, fgen
)
from conftest import available_frontends
from transformations import SingleColumnCoalescedTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_simple(frontend, horizontal, vertical):
    """
    Test removal of vector loops in kernel and re-insertion of the
    horizontal loop in the "driver".
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
        t(jl, jk) = c * k
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
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical,
        hoist_column_arrays=False
    )
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(kernel, role='kernel')

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
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*k'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure only one loop in the driver
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 1
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'

    # Ensure we have a kernel call in the driver loop
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    assert kernel_calls[0].name == 'compute_column'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_demote(frontend, horizontal, vertical):
    """
    Test that local array variables that do not contain the
    vertical dimension are demoted and privativised.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, nb, q)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Array dimensions
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
        t(jl, jk) = c * k
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
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical,
        hoist_column_arrays=False
    )
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(kernel, role='kernel')

    # Ensure correct array variables shapes
    assert isinstance(kernel.variable_map['a'], Scalar)
    assert isinstance(kernel.variable_map['b'], Array)
    assert isinstance(kernel.variable_map['c'], Scalar)
    assert isinstance(kernel.variable_map['t'], Array)
    assert isinstance(kernel.variable_map['q'], Array)

    # Ensure that parameter-sized array b got demoted only
    assert kernel.variable_map['b'].shape == ((3,) if frontend is OMNI else ('psize',))
    assert kernel.variable_map['t'].shape == ('nlon', 'nz')
    assert kernel.variable_map['q'].shape == ('nlon', 'nz')

    # Ensure relevant expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*k'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'a = q(jl, 1)'
    assert fgen(assigns[4]).lower() == 'b(1) = q(jl, 2)'
    assert fgen(assigns[5]).lower() == 'b(2) = q(jl, 3)'
    assert fgen(assigns[6]).lower() == 'b(3) = a*(b(1) + b(2))'
    assert fgen(assigns[7]).lower() == 'q(jl, nz) = q(jl, nz)*c + b(3)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_hoist(frontend, horizontal, vertical, blocking):
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
        t(jl, jk) = c * k
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
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        hoist_column_arrays=True
    )
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(kernel, role='kernel')

    # Ensure we have only one loop left in kernel
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 1
    assert kernel_loops[0].variable == 'jk'
    assert kernel_loops[0].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*k'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure we have two nested driver loops
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 2
    assert driver_loops[1] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'
    assert driver_loops[1].variable == 'jl'
    assert driver_loops[1].bounds == 'start:end'

    # Ensure we have a kernel call in the driver loop
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    assert kernel_calls[0].name == 'compute_column'
    assert ('jl', 'jl') in kernel_calls[0].kwarguments
    assert 't(:,:,b)' in kernel_calls[0].arguments

    # Ensure that column local `t(nlon,nz)` has been hoisted
    assert 't' in kernel.argnames
    assert kernel.variable_map['t'].type.intent.lower() == 'inout'
    # TODO: Shape doesn't translate correctly yet.
    assert driver.variable_map['t'].dimensions == ('nlon', 'nz', 'nb')
    # assert driver.variable_map['t'].shape == ('nlon', 'nz', 'nb')

    # Ensure that the loop index variable is correctly promoted
    assert 'jl' in kernel.argnames
    assert kernel.variable_map['jl'].type.intent.lower() == 'in'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_openacc(frontend, horizontal, vertical, blocking):
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
        t(jl, jk) = c * k
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
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    # Test OpenACC annotations on non-hoisted version
    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        hoist_column_arrays=False, directive='openacc'
    )
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(kernel, role='kernel')

    # Ensure routine is anntoated at vector level
    pragmas = FindNodes(Pragma).visit(kernel.body)
    assert len(pragmas) == 5
    assert pragmas[0].keyword == 'acc'
    assert pragmas[0].content == 'routine vector'
    assert pragmas[1].keyword == 'acc'
    assert pragmas[1].content == 'data present(q)'
    assert pragmas[-1].keyword == 'acc'
    assert pragmas[-1].content == 'end data'

    # Ensure vector and seq loops are annotated, including
    # privatized variable `b`
    with pragmas_attached(kernel, Loop):
        kernel_loops = FindNodes(Loop).visit(kernel.body)
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
        assert driver_loops[0].pragma[0].content == 'parallel loop gang'


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_hoist_openacc(frontend, horizontal, vertical, blocking):
    """
    Test the correct addition of OpenACC pragmas to SCC format code
    when hoisting column array temporaries to driver.
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
        t(jl, jk) = c * k
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
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    # Test OpenACC annotations on non-hoisted version
    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        hoist_column_arrays=True, directive='openacc'
    )
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(kernel, role='kernel')

    with pragmas_attached(kernel, Loop):
        # Ensure routine is anntoated at vector level
        kernel_pragmas = FindNodes(Pragma).visit(kernel.body)
        assert len(kernel_pragmas) == 3
        assert kernel_pragmas[0].keyword == 'acc'
        assert kernel_pragmas[0].content == 'routine seq'
        assert kernel_pragmas[1].keyword == 'acc'
        assert kernel_pragmas[1].content == 'data present(q, t)'
        assert kernel_pragmas[2].keyword == 'acc'
        assert kernel_pragmas[2].content == 'end data'

        # Ensure only a single `seq` loop is left
        kernel_loops = FindNodes(Loop).visit(kernel.body)
        assert len(kernel_loops) == 1
        assert kernel_loops[0].pragma[0].keyword == 'acc'
        assert kernel_loops[0].pragma[0].content == 'loop seq'

    # Ensure two levels of blocked parallel loops in driver
    with pragmas_attached(driver, Loop):
        driver_loops = FindNodes(Loop).visit(driver.body)
        assert len(driver_loops) == 2
        assert driver_loops[0].pragma[0].keyword == 'acc'
        assert driver_loops[0].pragma[0].content == 'parallel loop gang'
        assert driver_loops[1].pragma[0].keyword == 'acc'
        assert driver_loops[1].pragma[0].content == 'loop vector'

        # Ensure deviece allocation and teardown via `!$acc enter/exit data`
        driver_pragmas = FindNodes(Pragma).visit(driver.body)
        assert len(driver_pragmas) == 2
        assert driver_pragmas[0].keyword == 'acc'
        assert driver_pragmas[0].content == 'enter data create(t)'
        assert driver_pragmas[1].keyword == 'acc'
        assert driver_pragmas[1].content == 'exit data delete(t)'


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
        t(jl, jk) = c * k
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO
  END SUBROUTINE update_q
"""

    outer_kernel = Subroutine.from_source(fcode_outer_kernel, frontend=frontend)
    inner_kernel = Subroutine.from_source(fcode_inner_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    outer_kernel.enrich_calls(inner_kernel)  # Attach kernel source to driver call
    driver.enrich_calls(outer_kernel)  # Attach kernel source to driver call

    # Test SCC transform for plain nested kernel
    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        hoist_column_arrays=False, directive='openacc'
    )
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(outer_kernel, role='kernel', targets=['compute_q'])
    scc_transform.apply(inner_kernel, role='kernel')

    # Ensure a single outer parallel loop in driver
    with pragmas_attached(driver, Loop):
        driver_loops = FindNodes(Loop).visit(driver.body)
        assert len(driver_loops) == 1
        assert driver_loops[0].variable == 'b'
        assert driver_loops[0].bounds == '1:nb'
        assert driver_loops[0].pragma[0].keyword == 'acc'
        assert driver_loops[0].pragma[0].content == 'parallel loop gang'

        # Ensure we have a kernel call in the driver loop
        kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
        assert len(kernel_calls) == 1
        assert kernel_calls[0].name == 'compute_column'

    # Ensure that the intermediate kernel contains two wrapped loops
    # and an unwrapped call statement
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
        outer_kernel_pragmas = FindNodes(Pragma).visit(outer_kernel.body)
        assert len(outer_kernel_pragmas) == 2
        assert outer_kernel_pragmas[0].keyword == 'acc'
        assert outer_kernel_pragmas[0].content == 'routine vector'
        assert outer_kernel_pragmas[1].keyword == 'acc'
        assert outer_kernel_pragmas[1].content == 'data present(q)'

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
        inner_kernel_pragmas = FindNodes(Pragma).visit(inner_kernel.body)
        assert len(inner_kernel_pragmas) == 2
        assert inner_kernel_pragmas[0].keyword == 'acc'
        assert inner_kernel_pragmas[0].content == 'routine vector'
        assert outer_kernel_pragmas[1].keyword == 'acc'
        assert outer_kernel_pragmas[1].content == 'data present(q)'


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
    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        hoist_column_arrays=False, directive='openacc'
    )
    scc_transform.apply(kernel, role='kernel')

    # Ensure that we capture vector loops outside the outer vertical
    # loop, as well as the one vector loop inside it.
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
def test_single_column_coalesced_variable_demotion(frontend, horizontal, vertical, blocking):
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
    scc_transform = SingleColumnCoalescedTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        hoist_column_arrays=False, directive='openacc'
    )
    scc_transform.apply(kernel, role='kernel')

    # Ensure that only a has not been demoted, as it buffers
    # information across the subroutine call.
    assert isinstance(kernel.variable_map['a'], Array)
    assert isinstance(kernel.variable_map['b'], Scalar)
    assert isinstance(kernel.variable_map['c'], Scalar)
