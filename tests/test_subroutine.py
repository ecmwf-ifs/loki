from pathlib import Path
import pytest
import numpy as np

from loki import (
    clean, compile_and_load, SourceFile, Subroutine, OFP, OMNI, FP, FindVariables, FindNodes,
    Intrinsic, CallStatement, DataType, Array, Scalar, fgen, FCodeMapper
)


@pytest.fixture(scope='module', name='header_path')
def fixture_header_path():
    return Path(__file__).parent/'sources/header.f90'


@pytest.fixture(scope='module', name='header_mod')
def fixture_header_mod(header_path):
    """
    Compile and load the reference solution
    """
    clean(filename=header_path)  # Delete parser cache
    return compile_and_load(header_path, cwd=str(header_path.parent))


@pytest.fixture(scope='module', name='testpath')
def fixture_testpath():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_simple(testpath, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_simple
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)'])  # OMNI

    # Generate code, compile and load
    filename = 'routine_simple_%s.f90' % frontend
    source = SourceFile(routines=[routine], path=testpath/filename)
    source.write(source=fgen(routine))
    function = compile_and_load(source.path, cwd=testpath).routine_simple

    # Test the generated identity results
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments(testpath, frontend):
    """
    A set of test to test internalisation and handling of arguments.
    """

    fcode = """
subroutine routine_arguments (x, y, vector, matrix)
  ! Test internal argument handling
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), dimension(x), intent(inout) :: vector
  real(kind=jprb), intent(inout) :: matrix(x, y)

  integer :: i, j
  real(kind=jprb), dimension(x) :: local_vector
  real(kind=jprb) :: local_matrix(x, y)

  do i=1, x
     local_vector(i) = i * 10.
     do j=1, y
        local_matrix(i, j) = local_vector(i) + j * 2.
     end do
  end do

  vector(:) = local_vector(:)
  matrix(:, :) = local_matrix(:, :)

end subroutine routine_arguments
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars in (['jprb', 'x', 'y', 'vector(x)', 'matrix(x,y)',
                             'i', 'j', 'local_vector(x)', 'local_matrix(x,y)'],
                            ['jprb', 'x', 'y', 'vector(1:x)', 'matrix(1:x,1:y)',
                             'i', 'j', 'local_vector(1:x)', 'local_matrix(1:x,1:y)'])
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'vector(x)', 'matrix(x,y)'],
                            ['x', 'y', 'vector(1:x)', 'matrix(1:x,1:y)'])

    # Generate code, compile and load
    filename = 'routine_arguments_%s.f90' % frontend
    source = SourceFile(routines=[routine], path=testpath/filename)
    source.write(source=fgen(routine))
    function = compile_and_load(source.path, cwd=testpath).routine_arguments

    # Test results of the generated and compiled code
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, vector=vector, matrix=matrix)
    assert np.all(vector == [10., 20.])
    assert np.all(matrix == [[12., 14., 16.],
                             [22., 24., 26.]])


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments_multiline(testpath, frontend):
    """
    Test argument declarations with comments interjectected between dummies.
    """
    fcode = """
subroutine routine_arguments_multiline &
 ! Test multiline dummy arguments with comments
 & (x, y, scalar, &
 ! Of course, not one...
 ! but two comment lines
 & vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_arguments_multiline
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)'])

    # Generate code, compile and load
    filename = 'routine_arguments_multiline_%s.f90' % frontend
    source = SourceFile(routines=[routine], path=testpath/filename)
    source.write(source=fgen(routine))
    function = compile_and_load(source.path, cwd=testpath).routine_arguments_multiline

    # Test results of the generated and compiled code
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_local(testpath, frontend):
    """
    Test local variables and types
    """
    fcode = """
subroutine routine_variables_local (x, y, maximum)
  ! Test local variables and types
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(out) :: maximum

  integer :: i, j
  real(kind=jprb), dimension(x) :: vector
  real(kind=jprb) :: matrix(x, y)

  do i=1, x
     vector(i) = i * 10.
     do j=1, y
        matrix(i, j) = vector(i) + j * 2.
     end do
  end do
  maximum = matrix(x, y)
end subroutine routine_variables_local
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars in (
        ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(x)', 'matrix(x,y)'],
        ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(1:x)', 'matrix(1:x,1:y)'])

    # Generate code, compile and load
    filename = 'routine_variables_local_%s.f90' % frontend
    source = SourceFile(routines=[routine], path=testpath/filename)
    source.write(source=fgen(routine))
    function = compile_and_load(source.path, cwd=testpath).routine_variables_local

    # Test results of the generated and compiled code
    maximum = function(x=3, y=4)
    assert np.all(maximum == 38.)  # 10*x + 2*y


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variable_caching(frontend):
    """
    Test that equivalent names in distinct routines don't cache.
    """
    fcode_real = """
subroutine routine_real (x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_real
"""

    fcode_int = """
subroutine routine_simple_caching (x, y, scalar, vector, matrix)
  ! A simple standard looking routine to test variable caching.
  integer, parameter :: jpim = selected_int_kind(9)
  integer, intent(in) :: x, y
  ! The next two share names with `routine_simple`, but have different
  ! dimensions or types, so that we can test variable caching.
  integer(kind=jpim), intent(in) :: scalar
  integer(kind=jpim), intent(inout) :: vector(y), matrix(x, y)
  integer :: i

  do i=1, y
     vector(i) = vector(i) + scalar
     matrix(:, i) = i * vector(i)
  end do
end subroutine routine_simple_caching
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode_real, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)'])
    assert routine.arguments[2].type.dtype == DataType.REAL
    assert routine.arguments[3].type.dtype == DataType.REAL

    routine = Subroutine.from_source(fcode_int, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(y)', 'matrix(x,y)'],
                            ['x', 'y', 'scalar', 'vector(1:y)', 'matrix(1:x,1:y)'])
    # Ensure that the types in the second routine have been picked up
    assert routine.arguments[2].type.dtype == DataType.INTEGER
    assert routine.arguments[3].type.dtype == DataType.INTEGER


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_find(frontend):
    """
    Tests the `FindVariables` utility (not the best place to put this).
    """
    fcode = """
subroutine routine_variables_find (x, y, maximum)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(out) :: maximum
  integer :: i, j
  real(kind=jprb), dimension(x) :: vector
  real(kind=jprb) :: matrix(x, y)

  do i=1, x
     vector(i) = i * 10.
  end do
  do i=1, x
     do j=1, y
        matrix(i, j) = vector(i) + j * 2.
     end do
  end do
  maximum = matrix(x, y)
end subroutine routine_variables_find
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    vars_all = FindVariables(unique=False).visit(routine.ir)
    # Note, we are not counting declarations here
    assert sum(1 for s in vars_all if str(s) == 'i') == 6
    assert sum(1 for s in vars_all if str(s) == 'j') == 3
    assert sum(1 for s in vars_all if str(s) == 'matrix(i,j)') == 1
    assert sum(1 for s in vars_all if str(s) == 'matrix(x,y)') == 1
    assert sum(1 for s in vars_all if str(s) == 'maximum') == 1
    assert sum(1 for s in vars_all if str(s) == 'vector(i)') == 2
    assert sum(1 for s in vars_all if str(s) == 'x') == 3
    assert sum(1 for s in vars_all if str(s) == 'y') == 2

    vars_unique = FindVariables(unique=True).visit(routine.ir)
    assert sum(1 for s in vars_unique if str(s) == 'i') == 1
    assert sum(1 for s in vars_unique if str(s) == 'j') == 1
    assert sum(1 for s in vars_unique if str(s) == 'matrix(i,j)') == 1
    assert sum(1 for s in vars_unique if str(s) == 'matrix(x,y)') == 1
    assert sum(1 for s in vars_unique if str(s) == 'maximum') == 1
    assert sum(1 for s in vars_unique if str(s) == 'vector(i)') == 1
    assert sum(1 for s in vars_unique if str(s) == 'x') == 1
    assert sum(1 for s in vars_unique if str(s) == 'y') == 1


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_dim_shapes(frontend):
    """
    A set of test to ensure matching different dimension and shape
    expressions against strings and other expressions works as expected.
    """
    fcode = """
subroutine routine_dim_shapes(v1, v2, v3, v4, v5)
  ! Simple variable assignments with non-trivial sizes and indices
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: v1, v2
  real(kind=jprb), allocatable, intent(out) :: v3(:)
  real(kind=jprb), intent(out) :: v4(v1,v2), v5(1:v1,v2-1)

  allocate(v3(v1))
  v3(v1-v2+1) = 1.
  v4(3:v1,1:v2-3) = 2.
  v5(:,:) = 3.

end subroutine routine_dim_shapes
"""
    fsymgen = FCodeMapper()

    # TODO: Need a named subroutine lookup
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_args = [fsymgen(arg) for arg in routine.arguments]
    assert routine_args in (['v1', 'v2', 'v3(:)', 'v4(v1,v2)', 'v5(1:v1,v2 - 1)'],
                            ['v1', 'v2', 'v3(:)', 'v4(1:v1,1:v2)', 'v5(1:v1,1:v2 - 1)'])

    # Make sure variable/argument shapes on the routine work
    shapes = [fsymgen(v.shape) for v in routine.arguments if isinstance(v, Array)]
    assert shapes in (['(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'],
                      ['(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)'])

    # Ensure shapes of body variables are ok
    b_shapes = [fsymgen(v.shape) for v in FindVariables(unique=False).visit(routine.ir)
                if isinstance(v, Array)]
    assert b_shapes in (['(v1,)', '(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'],
                        ['(v1,)', '(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)'])


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_shape_propagation(header_path, frontend):
    """
    Test for the correct identification and forward propagation of variable shapes
    from the subroutine declaration.
    """
    fsymgen = FCodeMapper()

    # Parse simple kernel routine to check plain array arguments
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_shape(x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_shape
""")

    # Check shapes on the internalized variable and argument lists
    # x, y, = routine.arguments[0], routine.arguments[1]
    # TODO: The string comparison here is due to the fact that shapes are actually
    # `RangeIndex(upper=Scalar)` objects, instead of the raw dimension variables.
    # This needs some more thorough conceptualisation of dimensions and indices!
    assert fsymgen(routine.arguments[3].shape) in ['(x,)', '(1:x,)']
    assert fsymgen(routine.arguments[4].shape) in ['(x, y)', '(1:x, 1:y)']

    # Verify that all variable instances have type and shape information
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    vmap = {v.name: v for v in variables}
    assert fsymgen(vmap['vector'].shape) in ['(x,)', '(1:x,)']
    assert fsymgen(vmap['matrix'].shape) in ['(x, y)', '(1:x, 1:y)']

    # Parse kernel with external typedefs to test shape inferred from
    # external derived type definition
    fcode = """
subroutine routine_typedefs_simple(item)
  ! simple vector/matrix arithmetic with a derived type
  ! imported from an external header module
  use header, only: derived_type
  implicit none

  type(derived_type), intent(inout) :: item
  integer :: i, j, n

  n = 3
  do i=1, n
    item%vector(i) = item%vector(i) + item%scalar
  end do

  do j=1, n
    do i=1, n
      item%matrix(i, j) = item%matrix(i, j) + item%scalar
    end do
  end do

end subroutine routine_typedefs_simple
"""
    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    routine = Subroutine.from_source(fcode, frontend=frontend, typedefs=header.typedefs)

    # Verify that all derived type variables have shape info
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    # Verify shape info from imported derived type is propagated
    vmap = {v.name: v for v in variables}
    assert fsymgen(vmap['item%vector'].shape) in ['(3,)', '(1:3,)']
    assert fsymgen(vmap['item%matrix'].shape) in ['(3, 3)', '(1:3, 1:3)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_type_propagation(header_path, frontend):
    """
    Test for the forward propagation of derived-type information from
    a standalone module to a foreign subroutine via the :param typedef:
    argument.
    """
    # TODO: Note, if we wanted to test the reference solution with
    # typedefs, we need to extend compile_and_load to use multiple
    # source files/paths, so that the header can be compiled alongside
    # the subroutine in the same f90wrap execution.

    # Parse simple kernel routine to check plain array arguments
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_simple (x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_simple
""")

    # Check types on the internalized variable and argument lists
    assert routine.arguments[0].type.dtype == DataType.INTEGER
    assert routine.arguments[1].type.dtype == DataType.INTEGER
    assert routine.arguments[2].type.dtype == DataType.REAL
    assert str(routine.arguments[2].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert routine.arguments[3].type.dtype == DataType.REAL
    assert str(routine.arguments[3].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert routine.arguments[4].type.dtype == DataType.REAL
    assert str(routine.arguments[4].type.kind) in ('jprb', 'selected_real_kind(13, 300)')

    # Verify that all variable instances have type information
    variables = FindVariables().visit(routine.body)
    assert all(v.type is not None for v in variables if isinstance(v, (Scalar, Array)))

    vmap = {v.name: v for v in variables}
    assert vmap['x'].type.dtype == DataType.INTEGER
    assert vmap['scalar'].type.dtype == DataType.REAL
    assert str(vmap['scalar'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['vector'].type.dtype == DataType.REAL
    assert str(vmap['vector'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['matrix'].type.dtype == DataType.REAL
    assert str(vmap['matrix'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')

    # Parse kernel routine and provide external typedefs
    fcode = """
subroutine routine_typedefs_simple(item)
  ! simple vector/matrix arithmetic with a derived type
  ! imported from an external header module
  use header, only: derived_type
  implicit none

  type(derived_type), intent(inout) :: item
  integer :: i, j, n

  n = 3
  do i=1, n
    item%vector(i) = item%vector(i) + item%scalar
  end do

  do j=1, n
    do i=1, n
      item%matrix(i, j) = item%matrix(i, j) + item%scalar
    end do
  end do

end subroutine routine_typedefs_simple
"""
    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    routine = Subroutine.from_source(fcode, frontend=frontend, typedefs=header.typedefs)

    # Check that external typedefs have been propagated to kernel variables
    # First check that the declared parent variable has the correct type
    assert routine.arguments[0].name == 'item'
    assert routine.arguments[0].type.name == 'derived_type'

    # Verify that all variable instances have type and shape information
    variables = FindVariables().visit(routine.body)
    assert all(v.type is not None for v in variables)

    # Verify imported derived type info explicitly
    vmap = {v.name: v for v in variables}
    assert vmap['item%scalar'].type.dtype == DataType.REAL
    assert str(vmap['item%scalar'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['item%vector'].type.dtype == DataType.REAL
    assert str(vmap['item%vector'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['item%matrix'].type.dtype == DataType.REAL
    assert str(vmap['item%matrix'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_call_arrays(header_path, frontend):
    """
    Test that arrays passed down a subroutine call are treated as arrays.
    """
    fcode = """
subroutine routine_call_caller(x, y, vector, matrix, item)
  ! Simple routine calling another routine
  use header, only: derived_type
  implicit none

  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  type(derived_type), intent(inout) :: item

  ! To a parser, these arrays look like scalarst!
  call routine_call_callee(x, y, vector, matrix, item%matrix)

end subroutine routine_call_caller
"""
    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    routine = Subroutine.from_source(fcode, frontend=frontend, typedefs=header.typedefs)
    call = FindNodes(CallStatement).visit(routine.body)[0]

    assert str(call.arguments[0]) == 'x'
    assert str(call.arguments[1]) == 'y'
    assert str(call.arguments[2]) == 'vector'
    assert str(call.arguments[3]) == 'matrix'
    assert str(call.arguments[4]) == 'item%matrix'

    assert isinstance(call.arguments[0], Scalar)
    assert isinstance(call.arguments[1], Scalar)
    assert isinstance(call.arguments[2], Array)
    assert isinstance(call.arguments[3], Array)
    assert isinstance(call.arguments[4], Array)

    fsymgen = FCodeMapper()
    assert fsymgen(call.arguments[2].shape) in ['(x,)', '(1:x,)']
    assert fsymgen(call.arguments[3].shape) in ['(x, y)', '(1:x, 1:y)']
#    assert fsymgen(call.arguments[4].shape) in ['(3, 3)', '(1:3, 1:3)']

    assert fgen(call) == 'CALL routine_call_callee(x, y, vector, &\n     & matrix, item%matrix)'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_call_no_arg(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_call_no_arg()
  implicit none

  call abort
end subroutine routine_call_no_arg
""")
    assert isinstance(routine.body[0], CallStatement)
    assert routine.body[0].arguments == ()
    assert routine.body[0].kwarguments == ()


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Files are preprocessed')),
    FP
])
def test_pp_macros(testpath, frontend):
    refpath = testpath/'sources/subroutine_pp_macros.F90'
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_pp_macros']
    visitor = FindNodes(Intrinsic)
    # We need to collect the intrinsics in multiple places because different frontends
    # make the cut between parts of a routine in different places
    intrinsics = visitor.visit(routine.docstring)
    intrinsics += visitor.visit(routine.spec)
    intrinsics += visitor.visit(routine.body)
    assert len(intrinsics) == 9
    assert all(node.text.startswith('#') or 'implicit none' in node.text.lower()
               for node in intrinsics)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_empty_spec(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_empty_spec
write(*,*) 'Hello world!'
end subroutine routine_empty_spec
""")
    if frontend == OMNI:
        # OMNI inserts IMPLICIT NONE into spec
        assert len(routine.spec.body) == 1
    else:
        assert not routine.spec.body
    assert len(routine.body) == 1


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_member_procedures(testpath, frontend):
    """
    Test member subroutine and function
    """
    fcode = """
subroutine routine_member_procedures(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
  ! out2 = member_function(out1)
  out2 = 3 * out1 + 2
contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar
  end subroutine member_procedure

  ! Below is disabled because f90wrap (wrongly) exhibits that
  ! symbol to the public, which causes double defined symbols
  ! upon compilation.

  ! function member_function(a) result(b)
  !   ! This function is just included to test that functions
  !   ! are also possible
  !   implicit none
  !   integer, intent(in) :: a
  !   integer :: b

  !   b = 3 * a + 2
  ! end function member_function
end subroutine routine_member_procedures
"""
    # Check that member procedures are parsed correctly
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(routine.members) == 1
    assert routine.members[0].name == 'member_procedure'
    assert routine.members[0].symbols.lookup('localvar', recursive=False) is None
    assert routine.members[0].symbols.lookup('localvar') is not None
    assert routine.members[0].symbols.lookup('localvar') is routine.symbols.lookup('localvar')
    assert routine.members[0].symbols.lookup('in1') is not None
    assert routine.symbols.lookup('in1') is not None
    assert routine.members[0].symbols.lookup('in1') is not routine.symbols.lookup('in1')

    # Generate code, compile and load
    filename = 'routine_member_procedures_%s.f90' % frontend
    source = SourceFile(routines=[routine], path=testpath/filename)
    source.write(source=fgen(routine))
    function = compile_and_load(source.path, cwd=testpath).routine_member_procedures

    # Test results of the generated and compiled code
    out1, out2 = function(1, 2)
    assert out1 == 7
    assert out2 == 23
