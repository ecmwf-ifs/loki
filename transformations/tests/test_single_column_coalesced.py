# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    OMNI, OFP, Subroutine, Dimension, FindNodes, Loop, Assignment,
    CallStatement, Conditional, Scalar, Array, Pragma, pragmas_attached,
    fgen, Sourcefile, Section, SubroutineItem
)
from conftest import available_frontends
from transformations import (
     DataOffloadTransformation, SCCBaseTransformation, SCCDevectorTransformation, SCCDemoteTransformation,
     SCCRevectorTransformation, SCCHoistTransformation, SCCAnnotateTransformation, SingleColumnCoalescedTransformation
)


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
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*k'
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
def test_scc_base_resolve_vector_notation(frontend, horizontal, vertical):
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
      t(start:end, jk) = c * k
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
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*k'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_base_masked_statement(frontend, horizontal, vertical):
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
def test_scc_wrapper_masked_statement(frontend, horizontal, vertical):
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

    scc_transform = SingleColumnCoalescedTransformation(horizontal=horizontal, vertical=vertical,
                                                        directive='openacc', hoist_column_arrays=False)
    scc_transform.apply(kernel, role='kernel')

    # Ensure horizontal loop variable has been declared
    assert 'jl' in kernel.variables

    # Ensure we have three loops in the kernel,
    # horizontal loops should be nested within vertical
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[1] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[0].bounds == 'start:end'
    assert kernel_loops[1].variable == 'jk'
    assert kernel_loops[1].bounds == '2:nz'

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

    # Must run SCCDevector first because demotion relies on knowledge
    # of vector sections
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')
        transform.apply(kernel, role='kernel')

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

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCHoistTransformation(horizontal=horizontal, vertical=vertical,
                                           block_dim=blocking),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column'])
        transform.apply(kernel, role='kernel')

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

    # Ensure we have two vector loops, nested in one driver loop
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 3
    assert driver_loops[1] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[2] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'
    assert driver_loops[1].variable == 'jl'
    assert driver_loops[1].bounds == 'start:end'
    assert driver_loops[2].variable == 'jl'
    assert driver_loops[2].bounds == 'start:end'

    # Ensure we have two kernel calls in the driver loop
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 2
    assert kernel_calls[0].name == 'compute_column'
    assert kernel_calls[1].name == 'compute_column'
    assert ('jl', 'jl') in kernel_calls[0].kwarguments
    assert 't(:,:,b)' in kernel_calls[0].arguments
    assert ('jl', 'jl') in kernel_calls[1].kwarguments
    assert 't(:,:,b)' in kernel_calls[1].arguments

    # Ensure that column local `t(nlon,nz)` has been hoisted
    assert 't' in kernel.argnames
    assert kernel.variable_map['t'].type.intent.lower() == 'inout'
    # TODO: Shape doesn't translate correctly yet.
    assert driver.variable_map['t'].dimensions == ('nlon', 'nz', 'nb')

    # Ensure that the loop index variable is correctly promoted
    assert 'jl' in kernel.argnames
    assert kernel.variable_map['jl'].type.intent.lower() == 'in'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_wrapper_multiple_kernels(frontend, horizontal, vertical, blocking):
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

    scc_transform = (SingleColumnCoalescedTransformation(horizontal=horizontal, vertical=vertical,
                                                         block_dim=blocking, hoist_column_arrays=True),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column'])
        transform.apply(kernel, role='kernel')

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

    # Ensure we have two vector loops, nested in one driver loop
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 3
    assert driver_loops[1] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[2] in FindNodes(Loop).visit(driver_loops[0].body)
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'
    assert driver_loops[1].variable == 'jl'
    assert driver_loops[1].bounds == 'start:end'
    assert driver_loops[2].variable == 'jl'
    assert driver_loops[2].bounds == 'start:end'

    # Ensure we have two kernel calls in the driver loop
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 2
    assert kernel_calls[0].name == 'compute_column'
    assert kernel_calls[1].name == 'compute_column'
    assert ('jl', 'jl') in kernel_calls[0].kwarguments
    assert 't(:,:,b)' in kernel_calls[0].arguments
    assert ('jl', 'jl') in kernel_calls[1].kwarguments
    assert 't(:,:,b)' in kernel_calls[1].arguments

    # Ensure that column local `t(nlon,nz)` has been hoisted
    assert 't' in kernel.argnames
    assert kernel.variable_map['t'].type.intent.lower() == 'inout'
    # TODO: Shape doesn't translate correctly yet.
    assert driver.variable_map['t'].dimensions == ('nlon', 'nz', 'nb')

    # Ensure that the loop index variable is correctly promoted
    assert 'jl' in kernel.argnames
    assert kernel.variable_map['jl'].type.intent.lower() == 'in'

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

    # Ensure vector and seq loops are annotated, including
    # privatized variable `b`
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

    item_driver = SubroutineItem(name='#column_driver', source=fcode_driver)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    # Test OpenACC annotations on hoisted version
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    scc_transform += (SCCHoistTransformation(horizontal=horizontal, vertical=vertical, block_dim=blocking),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical, directive='openacc',
                                              block_dim=blocking, hoist_column_arrays=True),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column'], item=item_driver)
        transform.apply(kernel, role='kernel')


    with pragmas_attached(kernel, Loop):
        # Ensure routine is anntoated at vector level
        kernel_pragmas = FindNodes(Pragma).visit(kernel.ir)
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
def test_scc_wrapper_hoist_openacc(frontend, horizontal, vertical, blocking):
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

    item_driver = SubroutineItem(name='#column_driver', source=fcode_driver)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    # Test OpenACC annotations on hoisted version
    scc_transform = (SingleColumnCoalescedTransformation(horizontal=horizontal, vertical=vertical,
                                                         block_dim=blocking, directive='openacc',
                                                         hoist_column_arrays=True),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column'], item=item_driver)
        transform.apply(kernel, role='kernel')

    with pragmas_attached(kernel, Loop):
        # Ensure routine is anntoated at vector level
        kernel_pragmas = FindNodes(Pragma).visit(kernel.ir)
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
def test_single_column_coalesced_hoist_empty(frontend, horizontal, vertical, blocking):
    """
    Test the correct addition of OpenACC pragmas in SCC code with
    hoisting, if only one of two kernels contains locals to hoist.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column1(start, end, nlon, nz, q(:,:,b))

      call compute_column2(start, end, nlon, nz, q(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel1 = """
  SUBROUTINE compute_column1(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)  ! <= temporary array to be hoisted

    DO jl = start, end
      t(jl, nz) = q(jl, nz) + 0.2
      q(jl, nz) = q(jl, nz) * q(jl, nz)
    END DO
  END SUBROUTINE compute_column1
"""

    fcode_kernel2 = """
  SUBROUTINE compute_column2(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    ! No temporary arrays to be hoisted!

    DO jl = start, end
      q(jl, nz) = q(jl, nz) * 1.2
    END DO
  END SUBROUTINE compute_column2
"""

    item_driver = SubroutineItem(name='#column_driver', source=fcode_driver)
    kernel1 = Subroutine.from_source(fcode_kernel1, frontend=frontend)
    kernel2 = Subroutine.from_source(fcode_kernel2, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(routines=(kernel1, kernel2))  # Attach kernel source to driver call

    # Test OpenACC annotations on non-hoisted version
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    scc_transform += (SCCHoistTransformation(horizontal=horizontal, vertical=vertical, block_dim=blocking),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical, directive='openacc',
                                              block_dim=blocking, hoist_column_arrays=True),)

    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column1', 'compute_column2'], item=item_driver)
        transform.apply(kernel1, role='kernel')
        transform.apply(kernel2, role='kernel')

    # Ensure only one of the kernel calls caused device allocations
    # for hoisted variables
    with pragmas_attached(driver, Loop):

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
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical,
                                                directive='openacc', block_dim=blocking),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=['compute_column'])
        transform.apply(outer_kernel, role='kernel', targets=['compute_q'])
        transform.apply(inner_kernel, role='kernel')

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
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical,
                                                directive='openacc', block_dim=blocking),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')

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
    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCDemoteTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')

    # Ensure that only a has not been demoted, as it buffers
    # information across the subroutine call.
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

    scc_transform = (SCCBaseTransformation(horizontal=horizontal),)
    scc_transform += (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical,
                                                directive='openacc', block_dim=blocking),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')

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


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP,
                         'OFP fails to parse multiconditional with embedded call.')]))
def test_scc_wrapper_multicond(frontend, horizontal, vertical, blocking):
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
    scc_transform = (SingleColumnCoalescedTransformation(horizontal=horizontal, vertical=vertical,
                                                directive='openacc', block_dim=blocking,
                                                hoist_column_arrays=False),)
    for transform in scc_transform:
        transform.apply(kernel, role='kernel')

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
    routine.enrich_calls(source.all_subroutines)

    data_offload = DataOffloadTransformation(remove_openmp=True)
    data_offload.transform_subroutine(routine, role='driver', targets=['some_kernel',])


    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCAnnotateTransformation(horizontal=horizontal, vertical=vertical,
                                                directive='openacc', block_dim=blocking),)
    for transform in scc_transform:
        transform.apply(routine, role='driver', targets=['some_kernel',])

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
    assert pragmas[1].content == 'parallel loop gang'
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
