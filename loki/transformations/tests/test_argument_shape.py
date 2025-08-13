# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from loki import  Subroutine, Scheduler, Sourcefile, flatten
from loki.frontend import available_frontends, OMNI, HAVE_FP, FP
from loki.ir import nodes as ir, CallStatement, FindNodes
from loki.transformations import (
    ArgumentArrayShapeAnalysis, ExplicitArgumentArrayShapeTransformation,
    infer_array_shape_caller
)
from loki.expression import symbols as sym
from loki.types import BasicType


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


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
    driver.enrich(kernel)  # Attach kernel source to driver call

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
    kernel_a.enrich(kernel_b)  # Attach kernel source to call
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(kernel_a)  # Attach kernel source to call

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
    kernel_a1.enrich(kernel_b)  # Attach kernel source to call
    kernel_a2 = Subroutine.from_source(fcode_kernel_a2, frontend=frontend)
    kernel_a2.enrich(kernel_b)  # Attach kernel source to call
    kernel_a3 = Subroutine.from_source(fcode_kernel_a3, frontend=frontend)
    kernel_a3.enrich(kernel_b)  # Attach kernel source to call
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(kernel_a1)  # Attach kernel source to call
    driver.enrich(kernel_a2)  # Attach kernel source to call
    driver.enrich(kernel_a3)  # Attach kernel source to call

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
    kernel_a1.enrich(kernel_b)  # Attach kernel source to call
    kernel_a2 = Subroutine.from_source(fcode_kernel_a2, frontend=frontend)
    kernel_a2.enrich(kernel_b)  # Attach kernel source to call
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(kernel_a1)  # Attach kernel source to call
    driver.enrich(kernel_a2)  # Attach kernel source to call

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


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_argument_shape_transformation_import(frontend, here, tmp_path):
    """
    Test that ensures that explicit argument shapes are indeed inserted
    in a multi-layered call tree.
    """

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

    header = [here/'sources/projArgShape/var_module_mod.F90']
    frontend_type = frontend
    headers = [Sourcefile.from_file(filename=h, frontend=frontend_type) for h in header]
    definitions = flatten(h.modules for h in headers)
    scheduler = Scheduler(paths=here/'sources/projArgShape', config=config, frontend=frontend,
                          definitions=definitions, xmods=[tmp_path])
    scheduler.process(transformation=ArgumentArrayShapeAnalysis())
    scheduler.process(transformation=ExplicitArgumentArrayShapeTransformation())

    item_map = {item.name: item for item in scheduler.items}
    driver = item_map['driver_mod#driver'].source['driver']
    kernel_a = item_map['kernel_a_mod#kernel_a'].source['kernel_a']
    kernel_a1 = item_map['kernel_a1_mod#kernel_a1'].source['kernel_a1']
    kernel_b = item_map['kernel_b_mod#kernel_b'].source['kernel_b']

    # Check that argument shapes have been applied
    assert kernel_a.arguments[0].dimensions == ('nlon',)
    assert kernel_a.arguments[1].dimensions == ('nlon', 'nlev')
    assert kernel_a.arguments[2].dimensions == ('nlon', 'n')
    assert 'nlon' in kernel_a.arguments
    assert 'nlon' in kernel_a.arguments
    assert 'n' not in kernel_a.arguments

    assert kernel_b.arguments[0].dimensions == ('nlon', 'nlev')
    assert kernel_b.arguments[1].dimensions == ('nlon', 'n')
    assert 'nlon' in kernel_b.arguments
    assert 'nlon' in kernel_b.arguments
    assert 'n' not in kernel_b.arguments

    assert kernel_a1.arguments[0].dimensions == ('nlon', 'nlev')
    assert kernel_a1.arguments[1].dimensions == ('nlon', 'n')
    assert 'nlon' in kernel_a1.arguments
    assert 'nlon' in kernel_a1.arguments
    assert 'n' in kernel_a1.arguments

    # And finally, check that scalar dimension size variables have been added to calls
    for v in ('nlon', 'nlev'):
        assert (v, v) in FindNodes(CallStatement).visit(driver.body)[0].kwarguments
        assert (v, v) in FindNodes(CallStatement).visit(driver.body)[1].kwarguments
    for v in ('nlon', 'nlev', 'n'):
        assert (v, v) in FindNodes(CallStatement).visit(kernel_a.body)[0].kwarguments


@pytest.mark.skipif(not HAVE_FP, reason="Assumed size declarations only supported for FP")
@pytest.mark.parametrize('transform', [True, False])
def test_argument_size_assumed_size(transform):
    """
    Test to ensure that assumed size arguments are correctly sized
    from the calling context, so that the driver-level sizes are propagated
    into the kernel routines.
    """

    fcode_driver = """
  SUBROUTINE trafo_driver(nlon, nlev, a, b, c, d, e)
    ! Driver routine with explicit array shapes
    INTEGER, INTENT(IN)   :: nlon, nlev  ! Dimension sizes
    INTEGER, PARAMETER    :: n = 5
    REAL, INTENT(INOUT)   :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon,nlev,n)
    REAL, INTENT(INOUT)   :: c(nlon,nlev,n)
    REAL, INTENT(INOUT)   :: d(nlon,nlev)
    REAL, INTENT(INOUT)   :: e(2,4,nlon,nlev)

    call trafo_kernel(nlon, a, b, c, d(:,1:2), e)
  END SUBROUTINE trafo_driver
    """

    fcode_kernel = """
  SUBROUTINE trafo_kernel(nlon, a, b, c, d, e)
    ! Kernel routine with implicit shape array arguments
    INTEGER, INTENT(IN)   :: nlon
    REAL, INTENT(INOUT)   :: a(*)
    REAL, INTENT(INOUT)   :: b(nlon,*)
    REAL, INTENT(INOUT)   :: c(nlon,0:*)
    REAL, INTENT(INOUT)   :: d(*)
    REAL, INTENT(INOUT)   :: e(2,4,3:*)

  END SUBROUTINE trafo_kernel
    """

    kernel = Subroutine.from_source(fcode_kernel, frontend=FP)
    driver = Subroutine.from_source(fcode_driver, frontend=FP)
    driver.enrich(kernel)  # Attach kernel source to driver call

    # Ensure initial call uses assumed size declarations
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1 and calls[0].routine
    assert len(calls[0].routine.arguments) == 6
    assert calls[0].routine.arguments[0] == 'nlon'
    assert calls[0].routine.arguments[1].shape == ('*', )
    assert calls[0].routine.arguments[2].shape == ('nlon', '*')
    assert calls[0].routine.arguments[3].shape == ('nlon', '0:*')
    assert calls[0].routine.arguments[4].shape == ('*',)
    assert calls[0].routine.arguments[5].shape == (2, 4, '3:*',)

    arg_shape_trafo = ArgumentArrayShapeAnalysis()
    arg_shape_trafo.apply(driver, role='driver')

    assert kernel.arguments[1].shape == ('nlon',)
    assert kernel.arguments[2].shape == ('nlon', 'nlev * n')

    assert kernel.arguments[3].shape[0] == 'nlon'
    assert isinstance(kernel.arguments[3].shape[1], sym.RangeIndex)
    assert kernel.arguments[3].shape[1].lower == 0
    assert kernel.arguments[3].shape[1].upper == '-1 + n*nlev'

    assert kernel.arguments[4].shape == ('2*nlon',)

    assert kernel.arguments[5].shape[0] == 2
    assert kernel.arguments[5].shape[1] == 4
    assert isinstance(kernel.arguments[5].shape[2], sym.RangeIndex)
    assert kernel.arguments[5].shape[2].lower == 3
    assert kernel.arguments[5].shape[2].upper == '2 + nlev*nlon'

    if transform:
        arg_shape_trafo = ExplicitArgumentArrayShapeTransformation()
        arg_shape_trafo.apply(kernel)
        arg_shape_trafo.apply(driver)

        # check that the driver side call was updated
        calls = FindNodes(CallStatement).visit(driver.body)
        assert len(calls) == 1
        assert len(calls[0].arguments) + len(calls[0].kwarguments) == 8
        assert calls[0].kwarguments[0][1] in ['n', 'nlev']
        assert calls[0].kwarguments[1][1] in ['n', 'nlev']
        assert calls[0].kwarguments[0][1] != calls[0].kwarguments[1][1]

        # check that the kernel argument declarations were updated
        arguments = kernel.arguments

        assert len(arguments) == 8
        assert arguments[6] in ['n', 'nlev']
        assert arguments[7] in ['n', 'nlev']
        assert arguments[6] != arguments[7]

        assert arguments[6].type.dtype == BasicType.INTEGER
        assert arguments[7].type.dtype == BasicType.INTEGER

        # check array argument declarations were updated correctly
        assert arguments[1].dimensions == ('nlon',)
        assert arguments[2].dimensions == ('nlon', 'nlev * n')

        assert arguments[3].dimensions[0] == 'nlon'
        assert isinstance(arguments[3].dimensions[1], sym.RangeIndex)
        assert arguments[3].dimensions[1].lower == 0
        assert arguments[3].dimensions[1].upper == '-1 + n*nlev'

        assert arguments[4].dimensions == ('2*nlon',)

        assert arguments[5].dimensions[0] == 2
        assert arguments[5].dimensions[1] == 4
        assert isinstance(arguments[5].dimensions[2], sym.RangeIndex)
        assert arguments[5].dimensions[2].lower == 3
        assert arguments[5].dimensions[2].upper == '2 + nlev*nlon'


@pytest.mark.parametrize('frontend', available_frontends())
def test_argument_shape_caller(frontend):
    """
    Test backward propagation of shape info from callee to caller.
    """

    fcode_driver = """
subroutine test_driver(n, m, a)
  use my_mod, only: my_type
  implicit none
  integer, intent(in) :: n, m
  real, intent(inout) :: a(:)
  type(my_type), intent(inout) :: obj

  call test_kernel(n, m, a, obj%b(:,:), c=obj%c)
end subroutine test_driver
    """

    fcode_kernel = """
subroutine test_kernel(n, m, a, b, c)
  implicit none
  integer, intent(in) :: n, m
  real, intent(inout) :: a(n)
  real, intent(inout) :: b(n,m)
  real, intent(inout) :: c(n,m,m)

end subroutine test_kernel
    """
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver.enrich(kernel)

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].name.type.dtype.procedure == kernel
    assert driver.variable_map['a'].shape == (':',)

    infer_array_shape_caller(driver.body)

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1

    assert len(calls[0].arguments) == 4
    assert calls[0].arguments[0] == 'n'
    assert calls[0].arguments[1] == 'm'
    assert calls[0].arguments[2] == 'a(:)'
    assert calls[0].arguments[2].shape == ('n',)
    assert calls[0].arguments[3] == 'obj%b(:,:)'
    assert calls[0].arguments[3].shape == ('n', 'm')
    assert len(calls[0].kwarguments) == 1
    assert calls[0].kwarguments[0] == ('c', 'obj%c(:,:,:)')
    assert calls[0].kwarguments[0][1].shape == ('n', 'm', 'm')
