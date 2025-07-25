# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Sourcefile, Dimension, fgen, Module
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, pragmas_attached, is_loki_pragma
)
from loki.transformations.single_column import (
    SCCDevectorTransformation, SCCRevectorTransformation, SCCVectorPipeline,
    SCCVecRevectorTransformation, SCCSeqRevectorTransformation,
    SCCVVectorPipeline, SCCSVectorPipeline
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

@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk', aliases=('nlev',))

@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('revector_trafo', [SCCSeqRevectorTransformation, SCCVecRevectorTransformation])
@pytest.mark.parametrize('ignore_nested_kernel', [False, True])
def test_scc_revector_transformation(frontend, horizontal, revector_trafo, ignore_nested_kernel, tmp_path):
    """
    Test removal of vector loops in kernel and re-insertion of a single
    hoisted horizontal loop in the kernel.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, t, nb)
    use compute_mod, only: compute_column
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
  MODULE compute_mod
  use compute_ctl_mod, only: compute_ctl
  use compute_ctl2_mod, only: compute_ctl2
  contains
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

    CALL COMPUTE_CTL(start, end, nlon, nz, q, t)
    CALL COMPUTE_CTL2(start, end, nlon, nz, q, t)

  END SUBROUTINE compute_column
  END MODULE compute_mod
"""

    fcode_intermediate_kernel = """
  MODULE compute_ctl_mod
  use compute2_mod, only: compute_another_column
  contains
  SUBROUTINE compute_ctl(start, end, nlon, nz, q1, t1)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t1(nlon,nz)
    REAL, INTENT(INOUT) :: q1(nlon,nz)
    CALL COMPUTE_ANOTHER_COLUMN(start, end, nlon, nz, q, t)
  END SUBROUTINE compute_ctl
  END MODULE compute_ctl_mod
"""

    fcode_intermediate2_kernel = """
  MODULE compute_ctl2_mod
  contains
  SUBROUTINE compute_ctl2(start, end, nlon, nz, q1, t1)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t1(nlon,nz)
    REAL, INTENT(INOUT) :: q1(nlon,nz)
  END SUBROUTINE compute_ctl2
  END MODULE compute_ctl2_mod
"""

    fcode_nested_kernel = """
  MODULE compute2_mod
  contains
  SUBROUTINE compute_another_column(start, end, nlon, nz, q1, t1)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t1(nlon,nz)
    REAL, INTENT(INOUT) :: q1(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t1(jl, jk) = c * jk
        q1(jl, jk) = q1(jl, jk-1) + t1(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q1(JL, NZ) = Q1(JL, NZ) * C
    END DO
  END SUBROUTINE compute_another_column
  END MODULE compute2_mod
"""

    # kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    nested_kernel_mod = Module.from_source(
        fcode_nested_kernel, frontend=frontend, xmods=[tmp_path]
    )
    intermediate_kernel_mod = Module.from_source(
        fcode_intermediate_kernel, frontend=frontend, xmods=[tmp_path],
        definitions=nested_kernel_mod
    )
    intermediate_kernel2_mod = Module.from_source(
        fcode_intermediate2_kernel, frontend=frontend, xmods=[tmp_path]
    )
    kernel_mod = Module.from_source(
        fcode_kernel, frontend=frontend, xmods=[tmp_path],
        definitions=[intermediate_kernel_mod, intermediate_kernel2_mod]
    )
    driver = Subroutine.from_source(
        fcode_driver, frontend=frontend, xmods=[tmp_path],
        definitions=kernel_mod
    )
    kernel = kernel_mod.subroutines[0]
    intermediate_kernel = intermediate_kernel_mod.subroutines[0]
    nested_kernel = nested_kernel_mod.subroutines[0]

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(ir.Loop).visit(kernel.body)
    assert len(kernel_loops) == 3

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (revector_trafo(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=('compute_column',))
        transform.apply(kernel, role='kernel', targets=('compute_Ctl',), ignore=('compute_Ctl2',))
        if ignore_nested_kernel:
            transform.apply(intermediate_kernel, role='kernel', ignore=('compute_Another_column',))
        else:
            transform.apply(intermediate_kernel, role='kernel', targets=('compute_Another_column',))
        if not ignore_nested_kernel:
            transform.apply(nested_kernel, role='kernel')

    # Ensure we have two nested loops in the kernel
    # (the hoisted horizontal and the native vertical)
    with pragmas_attached(kernel, node_type=ir.Loop):
        kernel_loops = FindNodes(ir.Loop).visit(kernel.body)
        calls = FindNodes(ir.CallStatement).visit(kernel.body)
        if revector_trafo == SCCSeqRevectorTransformation:
            assert len(kernel_loops) == 1
            assert kernel_loops[0].variable == 'jk'
            assert kernel_loops[0].bounds == '2:nz'
            assert kernel_loops[0].pragma
            assert is_loki_pragma(kernel_loops[0].pragma, starts_with='loop seq')
            for call in calls:
                assert 'jl' in call.arg_map
                assert call.routine.variable_map['jl'].type.intent.lower() == 'in'
        else:
            assert len(kernel_loops) == 2
            assert kernel_loops[1] in FindNodes(ir.Loop).visit(kernel_loops[0].body)
            assert kernel_loops[0].variable == 'jl'
            assert kernel_loops[0].bounds == 'start:end'
            assert kernel_loops[1].variable == 'jk'
            assert kernel_loops[1].bounds == '2:nz'

            # Check internal loop pragma annotations
            assert kernel_loops[0].pragma
            assert is_loki_pragma(kernel_loops[0].pragma, starts_with='loop vector')
            assert kernel_loops[1].pragma
            assert is_loki_pragma(kernel_loops[1].pragma, starts_with='loop seq')
            for call in calls:
                assert 'jl' not in call.arg_map

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(ir.Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure that vector-section labels have been removed
    sections = FindNodes(ir.Section).visit(kernel.body)
    assert all(not s.label for s in sections)

    # Ensure driver remains unaffected and is marked
    with pragmas_attached(driver, node_type=ir.Loop):
        driver_loops = FindNodes(ir.Loop).visit(driver.body)
        if revector_trafo == SCCSeqRevectorTransformation:
            assert len(driver_loops) == 2
            assert driver_loops[1].variable == 'jl'
            assert driver_loops[1].bounds == 'start:end'
            assert driver_loops[1].pragma and len(driver_loops[1].pragma) == 1
            assert is_loki_pragma(driver_loops[1].pragma, starts_with='loop vector')
        else:
            assert len(driver_loops) == 1
        assert driver_loops[0].variable == 'b'
        assert driver_loops[0].bounds == '1:nb'
        assert driver_loops[0].pragma and len(driver_loops[0].pragma) == 1
        assert is_loki_pragma(driver_loops[0].pragma[0], starts_with='loop driver')
        assert 'vector_length(nlon)' in driver_loops[0].pragma[0].content

    kernel_calls = FindNodes(ir.CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    if revector_trafo == SCCSeqRevectorTransformation:
        assert 'jl' in kernel_calls[0].arg_map
        assert 'jl' in kernel_calls[0].routine.arguments
    else:
        assert 'jl' not in kernel_calls[0].arg_map

    assert kernel_calls[0].name == 'compute_column'
    if revector_trafo == SCCSeqRevectorTransformation:
        # make sure call to nested kernel gets horizontal.index as argument
        #  no matter whether it is a target or within ignore
        nested_kernel_call = FindNodes(ir.CallStatement).visit(kernel.body)[0]
        assert 'jl' in nested_kernel_call.arg_map

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('revector_trafo', [SCCSeqRevectorTransformation, SCCVecRevectorTransformation])
def test_scc_revector_transformation_aliased_bounds(frontend, horizontal_bounds_aliases, revector_trafo, tmp_path):
    """
    Test removal of vector loops in kernel and re-insertion of a single
    hoisted horizontal loop in the kernel with aliased loop bounds.
    """

    fcode_bnds_type_mod = """
module bnds_type_mod
implicit none
    type bnds_type
        integer :: start
        integer :: end
    end type bnds_type
end module bnds_type_mod
    """.strip()

    fcode_driver = """
SUBROUTINE column_driver(nlon, nz, q, t, nb)
    USE bnds_type_mod, only : bnds_type
    USE compute_mod, only : compute_column
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end
    TYPE(bnds_type) :: bnds

    bnds%start = 1
    bnds%end = nlon
    do b=1, nb
      call compute_column(bnds, nlon, nz, q(:,:,b), t(:,:,b))
    end do
END SUBROUTINE column_driver
    """.strip()

    fcode_kernel = """
SUBROUTINE compute_column(bnds, nlon, nz, q, t)
    USE bnds_type_mod, only : bnds_type
    TYPE(bnds_type), INTENT(IN) :: bnds
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = bnds%start, bnds%end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = BNDS%START, BNDS%END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
END SUBROUTINE compute_column
    """.strip()
    fcode_kernel = """
MODULE compute_mod
contains
SUBROUTINE compute_column(bnds, nlon, nz, q, t)
    USE bnds_type_mod, only : bnds_type
    TYPE(bnds_type), INTENT(IN) :: bnds
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = bnds%start, bnds%end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = BNDS%START, BNDS%END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
END SUBROUTINE compute_column
END MODULE compute_mod
    """.strip()

    bnds_type_mod = Module.from_source(fcode_bnds_type_mod, frontend=frontend, xmods=[tmp_path])
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, xmods=[tmp_path],
                                    definitions=bnds_type_mod.definitions)
    kernel = kernel_mod.subroutines[0]
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path],
                                    definitions=(bnds_type_mod, kernel_mod))

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(ir.Loop).visit(kernel.body)
    assert len(kernel_loops) == 3

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal_bounds_aliases),)
    scc_transform += (revector_trafo(horizontal=horizontal_bounds_aliases),)
    for transform in scc_transform:
        transform.apply(driver, role='driver', targets=('compute_column',))
        transform.apply(kernel, role='kernel')

    # Ensure we have two nested loops in the kernel
    # (the hoisted horizontal and the native vertical)
    with pragmas_attached(kernel, node_type=ir.Loop):
        kernel_loops = FindNodes(ir.Loop).visit(kernel.body)
        if revector_trafo == SCCSeqRevectorTransformation:
            assert len(kernel_loops) == 1
            assert kernel_loops[0].variable == 'jk'
            assert kernel_loops[0].bounds == '2:nz'
            assert kernel_loops[0].pragma
            assert is_loki_pragma(kernel_loops[0].pragma, starts_with='loop seq')
        else:
            assert len(kernel_loops) == 2
            assert kernel_loops[1] in FindNodes(ir.Loop).visit(kernel_loops[0].body)
            assert kernel_loops[0].variable == 'jl'
            assert kernel_loops[0].bounds == 'bnds%start:bnds%end'
            assert kernel_loops[1].variable == 'jk'
            assert kernel_loops[1].bounds == '2:nz'

            # Check internal loop pragma annotations
            assert kernel_loops[0].pragma
            assert is_loki_pragma(kernel_loops[0].pragma, starts_with='loop vector')
            assert kernel_loops[1].pragma
            assert is_loki_pragma(kernel_loops[1].pragma, starts_with='loop seq')

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(ir.Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure that vector-section labels have been removed
    sections = FindNodes(ir.Section).visit(kernel.body)
    assert all(not s.label for s in sections)

    # Ensure driver remains unaffected and is marked
    with pragmas_attached(driver, node_type=ir.Loop):
        driver_loops = FindNodes(ir.Loop).visit(driver.body)
        if revector_trafo == SCCSeqRevectorTransformation:
            assert len(driver_loops) == 2
            assert driver_loops[1].variable == 'jl'
            assert driver_loops[1].bounds == 'bnds%start:bnds%end'
            assert driver_loops[1].pragma and len(driver_loops[1].pragma) == 1
            assert is_loki_pragma(driver_loops[1].pragma, starts_with='loop vector')
        else:
            assert len(driver_loops) == 1
        assert driver_loops[0].variable == 'b'
        assert driver_loops[0].bounds == '1:nb'
        assert driver_loops[0].pragma and len(driver_loops[0].pragma) == 1
        assert is_loki_pragma(driver_loops[0].pragma[0], starts_with='loop driver')
        assert 'vector_length(nlon)' in driver_loops[0].pragma[0].content

    kernel_calls = FindNodes(ir.CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    if revector_trafo == SCCSeqRevectorTransformation:
        assert 'jl' in kernel_calls[0].arg_map
        assert 'jl' in kernel_calls[0].routine.arguments
    else:
        assert 'jl' not in kernel_calls[0].arg_map

    assert kernel_calls[0].name == 'compute_column'


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
    loops = [l for l in FindNodes(ir.Loop).visit(kernel.body) if l.variable == 'jl']
    assert len(loops) == 4

    # Test SCCDevector transform for kernel with scope-splitting outer loop
    scc_transform = SCCDevectorTransformation(horizontal=horizontal)
    scc_transform.apply(kernel, role='kernel')

    # Check removal of horizontal loops
    loops = [l for l in FindNodes(ir.Loop).visit(kernel.body) if l.variable == 'jl']
    assert not loops

    # Check number and content of vector sections
    sections = [
        s for s in FindNodes(ir.Section).visit(kernel.body)
        if s.label == 'vector_section'
    ]
    assert len(sections) == 4

    assigns = FindNodes(ir.Assignment).visit(sections[0])
    assert len(assigns) == 2
    assigns = FindNodes(ir.Assignment).visit(sections[1])
    assert len(assigns) == 1
    assigns = FindNodes(ir.Assignment).visit(sections[2])
    assert len(assigns) == 1
    assigns = FindNodes(ir.Assignment).visit(sections[3])
    assert len(assigns) == 1


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('revector_trafo', [SCCSeqRevectorTransformation, SCCVecRevectorTransformation])
def test_scc_vector_inlined_call(frontend, horizontal, revector_trafo):
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
    scc_transform += (revector_trafo(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel', 'some_inlined_kernel'])

    # Check only `!$loki loop vector` pragma has been inserted
    if revector_trafo == SCCVecRevectorTransformation:
        pragmas = FindNodes(ir.Pragma).visit(routine.ir)
        assert len(pragmas) == 2
        assert is_loki_pragma(pragmas[0], starts_with='routine vector')
        assert is_loki_pragma(pragmas[1], starts_with='loop vector')

        # Check that 'some_inlined_kernel' remains within vector-parallel region
        loops = FindNodes(ir.Loop).visit(routine.body)
        assert len(loops) == 1
        calls = FindNodes(ir.CallStatement).visit(loops[0].body)
        assert len(calls) == 1
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        assert len(calls) == 2
    else:
        assert horizontal.index in routine.arguments
        assert routine.variable_map[horizontal.index].type.intent == 'in'
        pragmas = FindNodes(ir.Pragma).visit(routine.ir)
        assert len(pragmas) == 1
        assert is_loki_pragma(pragmas[0], starts_with='routine seq')
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        assert len(calls) == 2


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
@pytest.mark.parametrize('revector_trafo', [SCCSeqRevectorTransformation, SCCVecRevectorTransformation])
def test_scc_vector_section_trim_simple(frontend, horizontal, trim_vector_sections, revector_trafo):
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
    scc_transform += (revector_trafo(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    assign = FindNodes(ir.Assignment).visit(routine.body)[0]
    loops = FindNodes(ir.Loop).visit(routine.body)
    if revector_trafo == SCCSeqRevectorTransformation:
        assert len(loops) == 0
    else:
        loop = loops[0]
        comment = [
            c for c in FindNodes(ir.Comment).visit(routine.body)
            if c.text == '! random comment'
        ][0]

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
def test_scc_vector_section_trim_nested(frontend, horizontal, trim_vector_sections):
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

    cond = FindNodes(ir.Conditional).visit(routine.body)[0]
    loop = FindNodes(ir.Loop).visit(routine.body)[0]

    if trim_vector_sections:
        assert cond not in loop.body
        assert cond in routine.body.body
    else:
        assert cond in loop.body
        assert cond not in routine.body.body


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_scc_vector_section_trim_complex(
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

    assign = FindNodes(ir.Assignment).visit(routine.body)[0]

    # check we found the right assignment
    assert assign.lhs.name.lower() == 'flag1'

    cond = FindNodes(ir.Conditional).visit(routine.body)[0]
    loop = FindNodes(ir.Loop).visit(routine.body)[0]

    assert cond in loop.body
    assert cond not in routine.body.body
    if trim_vector_sections:
        assert assign not in loop.body
        assert(len(FindNodes(ir.Assignment).visit(loop.body)) == 3)
    else:
        assert assign in loop.body
        assert(len(FindNodes(ir.Assignment).visit(loop.body)) == 4)

@pytest.mark.parametrize('frontend', available_frontends(
    skip={OMNI: 'OMNI automatically expands ELSEIF into a nested ELSE=>IF.'}
))
@pytest.mark.parametrize('trim_vector_sections', [False, True])
@pytest.mark.parametrize('vector_pipeline', [SCCVVectorPipeline, SCCSVectorPipeline])
def test_scc_devector_section_special_case(frontend, horizontal, vertical, blocking, trim_vector_sections,
        vector_pipeline):
    """
    Test to highlight the limitations of vector-section trimming.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon, flag0, flag1, flag2)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical, intent(in) :: flag0, flag1, flag2
       real, dimension(nlon) :: work

       integer :: jl

       if (flag0) then
         call some_other_kernel()
       elseif (flag1) then
         do jl=start,end
            work(jl) = 1.
         enddo
       elseif (flag2) then
         do jl=start,end
            work(jl) = 1.
            work(jl) = 2.
         enddo
       else
         do jl=start,end
            work(jl) = 41.
            work(jl) = 42.
         enddo
       endif

    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # check whether pipeline can be applied and works as expected
    scc_pipeline = vector_pipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        directive='openacc', trim_vector_sections=trim_vector_sections
    )
    scc_pipeline.apply(routine, role='kernel', targets=['some_kernel',])

    with pragmas_attached(routine, node_type=ir.Loop):
        conditional = FindNodes(ir.Conditional).visit(routine.body)[0]
        assert isinstance(conditional.body[0], ir.CallStatement)
        assert len(conditional.body) == 1
        assert isinstance(conditional.else_body[0], ir.Conditional)
        assert len(conditional.else_body) == 1
        if vector_pipeline == SCCVVectorPipeline:
            assert isinstance(conditional.else_body[0].body[0], ir.Comment)
            assert isinstance(conditional.else_body[0].body[1], ir.Loop)
            assert conditional.else_body[0].body[1].pragma[0].content.lower() == 'loop vector'

            # Check that all else-bodies have been wrapped
            else_bodies = conditional.else_bodies
            assert len(else_bodies) == 3
            for body in else_bodies:
                assert isinstance(body[0], ir.Comment)
                assert isinstance(body[1], ir.Loop)
                assert body[1].pragma[0].content.lower() == 'loop vector'
        else:
            assert isinstance(conditional.else_body[0].body[0], ir.Assignment)
            # Check that all else-bodies have been wrapped
            else_bodies = conditional.else_bodies
            assert len(else_bodies) == 3
            for body in else_bodies:
                assert isinstance(body[0], ir.Assignment)
