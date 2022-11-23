# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki import CallStatement, FindNodes, OMNI, Subroutine
from transformations import ArgumentArrayShapeAnalysis, ExplicitArgumentArrayShapeTransformation


@pytest.mark.parametrize('frontend', available_frontends())
def test_argument_shape_simple(frontend):
    """
    Test to ensure that implicit array argument shapes are correctly derived
    from the calling context, so that the driver-level shapes are propagated
    into the kernel routines.
    """

    fcode_driver = """
  SUBROUTINE trafo_driver(nlon, nlev, a, b, c)
    ! Driver routine with explicit array shapes
    INTEGER, INTENT(IN)   :: nlon, nlev  ! Dimension sizes
    INTEGER, PARAMETER    :: n = 5
    REAL, INTENT(INOUT)   :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,n)

    call trafo_kernel(a, b, c)
  END SUBROUTINE trafo_driver
    """

    fcode_kernel = """
  SUBROUTINE trafo_kernel(a, b, c)
    ! Kernel routine with implicit shape array arguments
    REAL, INTENT(INOUT)   :: a(:)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE trafo_kernel
    """

    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    # Ensure initial call uses implicit argument shapes
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1 and calls[0].routine
    assert len(calls[0].routine.arguments) == 3
    assert calls[0].routine.arguments[0].shape == (':', )
    assert calls[0].routine.arguments[1].shape == (':', ':')
    assert calls[0].routine.arguments[2].shape == (':', ':')

    arg_shape_trafo = ArgumentArrayShapeAnalysis()
    arg_shape_trafo.apply(driver, role='driver')

    assert kernel.arguments[0].shape == ('nlon',)
    assert kernel.arguments[1].shape == ('nlon', 'nlev')
    assert kernel.arguments[2].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')


@pytest.mark.parametrize('frontend', available_frontends())
def test_argument_shape_nested(frontend):
    """
    Test to ensure that implicit array argument shapes are propagated
    through multiple subroutine calls.
    """

    fcode_driver = """
  SUBROUTINE trafo_driver(nlon, nlev, a, b, c)
    ! Driver routine with explicit array shapes
    INTEGER, INTENT(IN)   :: nlon, nlev  ! Dimension sizes
    INTEGER, PARAMETER    :: n = 5
    REAL, INTENT(INOUT)   :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,n)

    call trafo_kernel_a(a, b, c)
  END SUBROUTINE trafo_driver
    """

    fcode_kernel_a = """
  SUBROUTINE trafo_kernel_a(a, b, c)
    REAL, INTENT(INOUT)   :: a(:)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

    CALL trafo_kernel_b(b, c)
  END SUBROUTINE trafo_kernel_a
    """

    fcode_kernel_b = """
  SUBROUTINE trafo_kernel_b(b, c)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE trafo_kernel_b
    """

    kernel_b = Subroutine.from_source(fcode_kernel_b, frontend=frontend)
    kernel_a = Subroutine.from_source(fcode_kernel_a, frontend=frontend)
    kernel_a.enrich_calls(kernel_b)  # Attach kernel source to call
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel_a)  # Attach kernel source to call

    # Ensure initial call uses implicit argument shapes
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1 and calls[0].routine
    assert len(calls[0].routine.arguments) == 3
    assert tuple(a.shape for a in calls[0].routine.arguments) == ((':', ), (':', ':'), (':', ':'))

    calls = FindNodes(CallStatement).visit(kernel_a.body)
    assert len(calls) == 1 and calls[0].routine
    assert len(calls[0].routine.arguments) == 2
    assert tuple(a.shape for a in calls[0].routine.arguments) == ((':', ':'), (':', ':'))

    # Apply the shape propagation in a manual forward pass
    arg_shape_trafo = ArgumentArrayShapeAnalysis()
    arg_shape_trafo.apply(driver, role='driver')
    arg_shape_trafo.apply(kernel_a, role='kernel')

    assert kernel_a.arguments[0].shape == ('nlon',)
    assert kernel_a.arguments[1].shape == ('nlon', 'nlev')
    assert kernel_a.arguments[2].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')

    assert kernel_b.arguments[0].shape == ('nlon', 'nlev')
    assert kernel_b.arguments[1].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')


@pytest.mark.parametrize('frontend', available_frontends())
def test_argument_shape_multiple(frontend):
    """
    Test to ensure that multiple call paths are also honoured correctly.


    Note that conflicting array shape information is currently not
    detected, since the trnasformation only replaces deferred array
    dimensions (":" ).
    """

    fcode_driver = """
  SUBROUTINE trafo_driver(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev  ! Dimension sizes
    INTEGER, PARAMETER    :: n = 5
    REAL, INTENT(INOUT)   :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,n)

    call trafo_kernel_a1(a, b, c)

    call trafo_kernel_a2(b, c)

    call trafo_kernel_a3(nlon, nlev, b, c)
  END SUBROUTINE trafo_driver
    """

    fcode_kernel_a1 = """
  SUBROUTINE trafo_kernel_a1(a, b, c)
    ! First-level kernel call, as before
    REAL, INTENT(INOUT)   :: a(:)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

    CALL trafo_kernel_b(b, c)
  END SUBROUTINE trafo_kernel_a1
    """

    fcode_kernel_a2 = """
  SUBROUTINE trafo_kernel_a2(b, c)
    ! First-level kernel call that agrees with kernel_a1
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

    CALL trafo_kernel_b(b, c)
  END SUBROUTINE trafo_kernel_a2
    """

    fcode_kernel_a3 = """
  SUBROUTINE trafo_kernel_a3(nlon, nlev, b, c)
    ! First-level kernel call that disagrees with kernel_a1
    INTEGER, INTENT(IN) :: nlon, nlev
    REAL :: b(nlev, nlon), c(nlev, nlev)

    CALL trafo_kernel_b(b, c)
  END SUBROUTINE trafo_kernel_a3
    """

    fcode_kernel_b = """
  SUBROUTINE trafo_kernel_b(b, c)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE trafo_kernel_b
    """

    kernel_b = Subroutine.from_source(fcode_kernel_b, frontend=frontend)
    kernel_a1 = Subroutine.from_source(fcode_kernel_a1, frontend=frontend)
    kernel_a1.enrich_calls(kernel_b)  # Attach kernel source to call
    kernel_a2 = Subroutine.from_source(fcode_kernel_a2, frontend=frontend)
    kernel_a2.enrich_calls(kernel_b)  # Attach kernel source to call
    kernel_a3 = Subroutine.from_source(fcode_kernel_a3, frontend=frontend)
    kernel_a3.enrich_calls(kernel_b)  # Attach kernel source to call
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel_a1)  # Attach kernel source to call
    driver.enrich_calls(kernel_a2)  # Attach kernel source to call
    driver.enrich_calls(kernel_a3)  # Attach kernel source to call

    # Ensure initial call uses implicit argument shapes
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 3 and all(c.routine for c in calls)
    assert tuple(a.shape for a in calls[0].routine.arguments) == ((':', ), (':', ':'), (':', ':'))
    assert tuple(a.shape for a in calls[1].routine.arguments) == ((':', ':'), (':', ':'))
    assert tuple(a.shape for a in calls[2].routine.arguments[2:]) == (('nlev', 'nlon'), ('nlev', 'nlev'))

    # Apply the legal shape propagation in a manual forward pass
    arg_shape_trafo = ArgumentArrayShapeAnalysis()
    arg_shape_trafo.apply(driver, role='driver')
    arg_shape_trafo.apply(kernel_a1, role='kernel')
    arg_shape_trafo.apply(kernel_a2, role='kernel')
    arg_shape_trafo.apply(kernel_b, role='kernel')

    # Check that the agreeable argument shapes indeed propagate
    assert kernel_a1.arguments[0].shape == ('nlon',)
    assert kernel_a1.arguments[1].shape == ('nlon', 'nlev')
    assert kernel_a1.arguments[2].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')

    assert kernel_a2.arguments[0].shape == ('nlon', 'nlev')
    assert kernel_a2.arguments[1].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')

    assert kernel_b.arguments[0].shape == ('nlon', 'nlev')
    assert kernel_b.arguments[1].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')

    # Now we apply conflicting information and ensure that it completes
    # and does not override the derived shape.
    # TODO: We should eventually provide an option to fail here, so that
    # conflicting shape info can be detected and dealt with, but that's
    # for the future. A failure condition can then be inserted here.

    arg_shape_trafo.apply(kernel_a3, role='kernel')
    assert kernel_b.arguments[0].shape == ('nlon', 'nlev')
    assert kernel_b.arguments[1].shape == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')


@pytest.mark.parametrize('frontend', available_frontends())
def test_argument_shape_transformation(frontend):
    """
    Test that ensures that explicit argument shapes are indeed inserted
    in a multi-layered call tree.
    """

    fcode_driver = """
  SUBROUTINE trafo_driver(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev  ! Dimension sizes
    INTEGER, PARAMETER    :: n = 5
    REAL, INTENT(INOUT)   :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,n)

    call trafo_kernel_a1(a, b, c)

    call trafo_kernel_a2(b, c)
  END SUBROUTINE trafo_driver
    """

    fcode_kernel_a1 = """
  SUBROUTINE trafo_kernel_a1(a, b, c)
    ! First-level kernel call, as before
    REAL, INTENT(INOUT)   :: a(:)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

    CALL trafo_kernel_b(b, c)
  END SUBROUTINE trafo_kernel_a1
    """

    fcode_kernel_a2 = """
  SUBROUTINE trafo_kernel_a2(b, c)
    ! First-level kernel call that agrees with kernel_a1
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

    CALL trafo_kernel_b(b, c)
  END SUBROUTINE trafo_kernel_a2
    """

    fcode_kernel_b = """
  SUBROUTINE trafo_kernel_b(b, c)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE trafo_kernel_b
    """

    # Manually create subroutines and attach call-signature info
    kernel_b = Subroutine.from_source(fcode_kernel_b, frontend=frontend)
    kernel_a1 = Subroutine.from_source(fcode_kernel_a1, frontend=frontend)
    kernel_a1.enrich_calls(kernel_b)  # Attach kernel source to call
    kernel_a2 = Subroutine.from_source(fcode_kernel_a2, frontend=frontend)
    kernel_a2.enrich_calls(kernel_b)  # Attach kernel source to call
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel_a1)  # Attach kernel source to call
    driver.enrich_calls(kernel_a2)  # Attach kernel source to call

    # Ensure initial call uses implicit argument shapes
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 2 and all(c.routine for c in calls)
    assert tuple(a.shape for a in calls[0].routine.arguments) == ((':', ), (':', ':'), (':', ':'))
    assert tuple(a.shape for a in calls[1].routine.arguments) == ((':', ':'), (':', ':'))

    # Apply the legal shape propagation in a manual forward pass
    arg_shape_analysis = ArgumentArrayShapeAnalysis()
    arg_shape_analysis.apply(driver)
    arg_shape_analysis.apply(kernel_a1)
    arg_shape_analysis.apply(kernel_a2)
    arg_shape_analysis.apply(kernel_b)

    # Apply the insertion of explicit array argument shapes in a backward pass
    arg_shape_trafo = ExplicitArgumentArrayShapeTransformation()
    arg_shape_trafo.apply(kernel_b)
    arg_shape_trafo.apply(kernel_a2)
    arg_shape_trafo.apply(kernel_a1)
    arg_shape_trafo.apply(driver)

    # Check that argument shapes have been applied
    assert kernel_a1.arguments[0].dimensions == ('nlon',)
    assert kernel_a1.arguments[1].dimensions == ('nlon', 'nlev')
    assert kernel_a1.arguments[2].dimensions == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')
    assert 'nlon' in kernel_a1.arguments
    assert 'nlon' in kernel_a1.arguments
    assert 'n' in kernel_a1.arguments or frontend == OMNI

    assert kernel_a2.arguments[0].dimensions == ('nlon', 'nlev')
    assert kernel_a2.arguments[1].dimensions == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')
    assert 'nlon' in kernel_a2.arguments
    assert 'nlon' in kernel_a2.arguments
    assert 'n' in kernel_a2.arguments or frontend == OMNI

    assert kernel_b.arguments[0].dimensions == ('nlon', 'nlev')
    assert kernel_b.arguments[1].dimensions == ('nlon', 5) if frontend == OMNI else ('nlon', 'n')
    assert 'nlon' in kernel_b.arguments
    assert 'nlon' in kernel_b.arguments
    assert 'n' in kernel_b.arguments or frontend == OMNI

    # And finally, check that scalar dimension size variables have been added to calls
    for v  in ('nlon', 'nlev') if frontend == OMNI else ('nlon', 'nlev', 'n'):
        assert (v, v) in FindNodes(CallStatement).visit(kernel_a1.body)[0].kwarguments
        assert (v, v) in FindNodes(CallStatement).visit(kernel_a2.body)[0].kwarguments
        assert (v, v) in FindNodes(CallStatement).visit(driver.body)[0].kwarguments
        assert (v, v) in FindNodes(CallStatement).visit(driver.body)[1].kwarguments
