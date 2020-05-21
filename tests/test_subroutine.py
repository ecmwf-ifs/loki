from pathlib import Path
import pytest
import numpy as np

from loki import (
    SourceFile, Subroutine, OFP, OMNI, FP, FindVariables, FindNodes,
    Intrinsic, CallStatement, DataType, Array, Scalar, Variable,
    SymbolType, StringLiteral, fgen, fexprgen
)
from conftest import jit_compile, clean_test


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='header_path')
def fixture_header_path(here):
    return here/'sources/header.f90'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_simple(here, frontend):
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
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)'])  # OMNI

    # Generate code, compile and load
    filepath = here/('routine_simple_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_simple')

    # Test the generated identity results
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments(here, frontend):
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
    assert routine_vars in (['jprb', 'x', 'y', 'vector(x)', 'matrix(x, y)',
                             'i', 'j', 'local_vector(x)', 'local_matrix(x, y)'],
                            ['jprb', 'x', 'y', 'vector(1:x)', 'matrix(1:x, 1:y)',
                             'i', 'j', 'local_vector(1:x)', 'local_matrix(1:x, 1:y)'])
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'vector(1:x)', 'matrix(1:x, 1:y)'])

    # Generate code, compile and load
    filepath = here/('routine_arguments_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_arguments')

    # Test results of the generated and compiled code
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, vector=vector, matrix=matrix)
    assert np.all(vector == [10., 20.])
    assert np.all(matrix == [[12., 14., 16.],
                             [22., 24., 26.]])
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments_multiline(here, frontend):
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
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)'])

    # Generate code, compile and load
    filepath = here/('routine_arguments_multiline_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_arguments_multiline')

    # Test results of the generated and compiled code
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments_order(frontend):
    """
    Test argument ordering honours singateu (dummy list) instead of
    order of apearance in spec declarations.
    """
    fcode = """
subroutine routine_arguments_order(x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x
  real(kind=jprb), intent(inout) :: matrix(x, y)
  real(kind=jprb), intent(in) :: scalar
  integer, intent(in) :: y
  real(kind=jprb), intent(inout) :: vector(x)
  integer :: i
end subroutine routine_arguments_order
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)'])


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments_add_remove(frontend):
    """
    Test addition and removal of subroutine arguments.
    """
    fcode = """
subroutine routine_arguments_add_remove(x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13, 300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
end subroutine routine_arguments_add_remove
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)'])

    # Create a new set of variables and add to local routine variables
    x = routine.variables[1]  # That's the symbol for variable 'x'
    real_type = routine.symbols['scalar']  # Type of variable 'maximum'
    a = Scalar(name='a', type=real_type, scope=routine.symbols)
    b = Array(name='b', dimensions=(x, ), type=real_type, scope=routine.symbols)
    c = Variable(name='c', type=x.type, scope=routine.symbols)

    # Add new variables and check that they are all in the routine spec
    routine.arguments += (a, b, c)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (
        ['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)', 'a', 'b(x)', 'c'],
        ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)', 'a', 'b(x)', 'c', ]
    )
    if frontend == OMNI:
        assert fgen(routine.spec).lower() == """
implicit none
integer, parameter :: jprb = selected_real_kind(13, 300)
integer, intent(in) :: x
integer, intent(in) :: y
real(kind=selected_real_kind(13, 300)), intent(in) :: scalar
real(kind=selected_real_kind(13, 300)), intent(inout) :: vector(1:x)
real(kind=selected_real_kind(13, 300)), intent(inout) :: matrix(1:x, 1:y)
real(kind=selected_real_kind(13, 300)), intent(in) :: a
real(kind=selected_real_kind(13, 300)), intent(in) :: b(x)
integer, intent(in) :: c
""".strip().lower()
    else:
        assert fgen(routine.spec).lower() == """
integer, parameter :: jprb = selected_real_kind(13, 300)
integer, intent(in) :: x, y
real(kind=jprb), intent(in) :: scalar
real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
real(kind=jprb), intent(in) :: a
real(kind=jprb), intent(in) :: b(x)
integer, intent(in) :: c
""".strip().lower()

    # Remove a select number of arguments
    routine.arguments = [arg for arg in routine.arguments if 'x' not in str(arg)]
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['y', 'scalar', 'a', 'c', ]


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_local(here, frontend):
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
        ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(x)', 'matrix(x, y)'],
        ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(1:x)', 'matrix(1:x, 1:y)'])

    # Generate code, compile and load
    filepath = here/('routine_variables_local_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_variables_local')

    # Test results of the generated and compiled code
    maximum = function(x=3, y=4)
    assert np.all(maximum == 38.)  # 10*x + 2*y
    clean_test(filepath)


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
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)'])
    assert routine.arguments[2].type.dtype == DataType.REAL
    assert routine.arguments[3].type.dtype == DataType.REAL

    routine = Subroutine.from_source(fcode_int, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(y)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:y)', 'matrix(1:x, 1:y)'])
    # Ensure that the types in the second routine have been picked up
    assert routine.arguments[2].type.dtype == DataType.INTEGER
    assert routine.arguments[3].type.dtype == DataType.INTEGER


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_add_remove(frontend):
    """
    Test local variable addition and removal.
    """
    fcode = """
subroutine routine_variables_add_remove(x, y, maximum)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(out) :: maximum
  real(kind=jprb) :: vector(x), matrix(x, y)
end subroutine routine_variables_add_remove
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars in (
        ['jprb', 'x', 'y', 'maximum', 'vector(x)', 'matrix(x, y)'],
        ['jprb', 'x', 'y', 'maximum', 'vector(1:x)', 'matrix(1:x, 1:y)']
    )

    # Create a new set of variables and add to local routine variables
    x = routine.variables[1]  # That's the symbol for variable 'x'
    real_type = SymbolType('real', kind='jprb')
    int_type = SymbolType('integer')
    a = Scalar(name='a', type=real_type, scope=routine.symbols)
    b = Array(name='b', dimensions=(x, ), type=real_type, scope=routine.symbols)
    c = Variable(name='c', type=int_type, scope=routine.symbols)

    # Add new variables and check that they are all in the routine spec
    routine.variables += (a, b, c)
    if frontend == OMNI:
        # OMNI frontend inserts a few peculiarities
        assert fgen(routine.spec).lower() == """
implicit none
integer, parameter :: jprb = selected_real_kind(13, 300)
integer, intent(in) :: x
integer, intent(in) :: y
real(kind=selected_real_kind(13, 300)), intent(out) :: maximum
real(kind=selected_real_kind(13, 300)) :: vector(1:x)
real(kind=selected_real_kind(13, 300)) :: matrix(1:x, 1:y)
real(kind=jprb) :: a
real(kind=jprb) :: b(x)
integer :: c
""".strip().lower()

    else:
        assert fgen(routine.spec).lower() == """
integer, parameter :: jprb = selected_real_kind(13, 300)
integer, intent(in) :: x, y
real(kind=jprb), intent(out) :: maximum
real(kind=jprb) :: vector(x), matrix(x, y)
real(kind=jprb) :: a
real(kind=jprb) :: b(x)
integer :: c
""".strip().lower()

    # Now remove the `maximum` variable and make sure it's gone
    routine.variables = [v for v in routine.variables if v.name != 'maximum']
    assert 'maximum' not in fgen(routine.spec).lower()
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars in (
        ['jprb', 'x', 'y', 'vector(x)', 'matrix(x, y)', 'a', 'b(x)', 'c'],
        ['jprb', 'x', 'y', 'vector(1:x)', 'matrix(1:x, 1:y)', 'a', 'b(x)', 'c']
    )


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

    vars_all = FindVariables(unique=False).visit(routine.body)
    # Note, we are not counting declarations here
    assert sum(1 for s in vars_all if str(s) == 'i') == 6
    assert sum(1 for s in vars_all if str(s) == 'j') == 3
    assert sum(1 for s in vars_all if str(s) == 'matrix(i, j)') == 1
    assert sum(1 for s in vars_all if str(s) == 'matrix(x, y)') == 1
    assert sum(1 for s in vars_all if str(s) == 'maximum') == 1
    assert sum(1 for s in vars_all if str(s) == 'vector(i)') == 2
    assert sum(1 for s in vars_all if str(s) == 'x') == 3
    assert sum(1 for s in vars_all if str(s) == 'y') == 2

    vars_unique = FindVariables(unique=True).visit(routine.ir)
    assert sum(1 for s in vars_unique if str(s) == 'i') == 1
    assert sum(1 for s in vars_unique if str(s) == 'j') == 1
    assert sum(1 for s in vars_unique if str(s) == 'matrix(i, j)') == 1
    assert sum(1 for s in vars_unique if str(s) == 'matrix(x, y)') == 1
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
    # TODO: Need a named subroutine lookup
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_args = [fexprgen(arg) for arg in routine.arguments]
    assert routine_args in (['v1', 'v2', 'v3(:)', 'v4(v1, v2)', 'v5(1:v1, v2 - 1)'],
                            ['v1', 'v2', 'v3(:)', 'v4(1:v1, 1:v2)', 'v5(1:v1, 1:v2 - 1)'])

    # Make sure variable/argument shapes on the routine work
    shapes = [fexprgen(v.shape) for v in routine.arguments if isinstance(v, Array)]
    assert shapes in (['(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'],
                      ['(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)'])

    # Ensure shapes of body variables are ok
    b_shapes = [fexprgen(v.shape) for v in FindVariables(unique=False).visit(routine.body)
                if isinstance(v, Array)]
    assert b_shapes in (['(v1,)', '(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'],
                        ['(v1,)', '(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)'])


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_variables_shape_propagation(header_path, frontend):
    """
    Test for the correct identification and forward propagation of variable shapes
    from the subroutine declaration.
    """

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
    assert fexprgen(routine.arguments[3].shape) in ['(x,)', '(1:x,)']
    assert fexprgen(routine.arguments[4].shape) in ['(x, y)', '(1:x, 1:y)']

    # Verify that all variable instances have type and shape information
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    vmap = {v.name: v for v in variables}
    assert fexprgen(vmap['vector'].shape) in ['(x,)', '(1:x,)']
    assert fexprgen(vmap['matrix'].shape) in ['(x, y)', '(1:x, 1:y)']

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
    assert fexprgen(vmap['item%vector'].shape) in ['(3,)', '(1:3,)']
    assert fexprgen(vmap['item%matrix'].shape) in ['(3, 3)', '(1:3, 1:3)']


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

    assert fexprgen(call.arguments[2].shape) in ['(x,)', '(1:x,)']
    assert fexprgen(call.arguments[3].shape) in ['(x, y)', '(1:x, 1:y)']
#    assert fexprgen(call.arguments[4].shape) in ['(3, 3)', '(1:3, 1:3)']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_call_kwargs(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_call_kwargs()
  implicit none
  integer :: kprocs

  call mpl_init(kprocs=kprocs, cdstring='routine_call_kwargs')
end subroutine routine_call_kwargs
""")
    assert isinstance(routine.body[0], CallStatement)
    assert routine.body[0].name == 'mpl_init'

    assert routine.body[0].arguments == ()
    assert len(routine.body[0].kwarguments) == 2
    assert all(isinstance(arg, tuple) and len(arg) == 2 for arg in routine.body[0].kwarguments)

    assert routine.body[0].kwarguments[0][0] == 'kprocs'
    assert (isinstance(routine.body[0].kwarguments[0][1], Scalar) and
            routine.body[0].kwarguments[0][1].name == 'kprocs')

    assert routine.body[0].kwarguments[1] == ('cdstring', StringLiteral('routine_call_kwargs'))


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_call_args_kwargs(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_call_args_kwargs(pbuf, ktag, kdest)
  implicit none
  integer, intent(in) :: pbuf(:), ktag, kdest

  call mpl_send(pbuf, ktag, kdest, cdstring='routine_call_args_kwargs')
end subroutine routine_call_args_kwargs
""")
    assert isinstance(routine.body[0], CallStatement)
    assert routine.body[0].name == 'mpl_send'
    assert len(routine.body[0].arguments) == 3
    assert all(a.name == b.name for a, b in zip(routine.body[0].arguments, routine.arguments))
    assert routine.body[0].kwarguments == (('cdstring', StringLiteral('routine_call_args_kwargs')),)


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Files are preprocessed')),
    FP
])
def test_pp_macros(here, frontend):
    refpath = here/'sources/subroutine_pp_macros.F90'
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
def test_member_procedures(here, frontend):
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
    filepath = here/('routine_member_procedures_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_member_procedures')

    # Test results of the generated and compiled code
    out1, out2 = function(1, 2)
    assert out1 == 7
    assert out2 == 23
    clean_test(filepath)
