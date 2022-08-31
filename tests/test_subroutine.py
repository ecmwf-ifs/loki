# pylint: disable=too-many-lines
from pathlib import Path
import pytest
import numpy as np

from conftest import available_frontends, jit_compile, jit_compile_lib, clean_test
from loki import (
    Sourcefile, Subroutine, OFP, OMNI, FindVariables, FindNodes,
    Section, CallStatement, BasicType, Array, Scalar, Variable,
    SymbolAttributes, StringLiteral, fgen, fexprgen, VariableDeclaration,
    Transformer, FindTypedSymbols, ProcedureSymbol, ProcedureType,
    StatementFunction
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='header_path')
def fixture_header_path(here):
    return here/'sources/header.f90'


@pytest.mark.parametrize('frontend', available_frontends())
def test_routine_simple(here, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
  ! This is the docstring
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
    assert isinstance(routine.body, Section)
    assert isinstance(routine.spec, Section)
    assert len(routine.docstring) == 1
    assert routine.docstring[0].text == '! This is the docstring'

    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x, 1:y)'])  # OMNI

    # Generate code, compile and load
    filepath = here/(f'routine_simple_{frontend}.f90')
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


@pytest.mark.parametrize('frontend', available_frontends())
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
    filepath = here/(f'routine_arguments_{frontend}.f90')
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


@pytest.mark.parametrize('frontend', available_frontends())
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
    filepath = here/(f'routine_arguments_multiline_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='routine_arguments_multiline')

    # Test results of the generated and compiled code
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
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


@pytest.mark.parametrize('frontend', available_frontends())
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
    real_type = routine.symbol_attrs['scalar']  # Type of variable 'maximum'
    a = Scalar(name='a', type=real_type, scope=routine)
    b = Array(name='b', dimensions=(x, ), type=real_type, scope=routine)
    c = Variable(name='c', type=x.type, scope=routine)

    # Add new arguments and check that they are all in the routine spec
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

    # Check that removed args still exist as variables
    routine_vars = [str(arg) for arg in routine.variables]
    assert 'vector(x)' in routine_vars or 'vector(1:x)' in routine_vars
    assert 'matrix(x, y)' in routine_vars or 'matrix(1:x, 1:y)' in routine_vars
    assert 'b(x)' in routine_vars


@pytest.mark.parametrize('frontend', available_frontends())
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
    filepath = here/(f'routine_variables_local_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='routine_variables_local')

    # Test results of the generated and compiled code
    maximum = function(x=3, y=4)
    assert np.all(maximum == 38.)  # 10*x + 2*y
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
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
    assert routine.arguments[2].type.dtype == BasicType.REAL
    assert routine.arguments[3].type.dtype == BasicType.REAL

    routine = Subroutine.from_source(fcode_int, frontend=frontend)
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args in (['x', 'y', 'scalar', 'vector(y)', 'matrix(x, y)'],
                            ['x', 'y', 'scalar', 'vector(1:y)', 'matrix(1:x, 1:y)'])
    # Ensure that the types in the second routine have been picked up
    assert routine.arguments[2].type.dtype == BasicType.INTEGER
    assert routine.arguments[3].type.dtype == BasicType.INTEGER


@pytest.mark.parametrize('frontend', available_frontends())
def test_routine_variables_add_remove(frontend):
    """
    Test local variable addition and removal.
    """
    fcode = """
subroutine routine_variables_add_remove(x, y, maximum, vector)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(out) :: maximum
  real(kind=jprb), intent(inout) :: vector(x)
  real(kind=jprb) :: matrix(x, y)
end subroutine routine_variables_add_remove
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars in (
        ['jprb', 'x', 'y', 'maximum', 'vector(x)', 'matrix(x, y)'],
        ['jprb', 'x', 'y', 'maximum', 'vector(1:x)', 'matrix(1:x, 1:y)']
    )

    # Create a new set of variables and add to local routine variables
    x = routine.variable_map['x']  # That's the symbol for variable 'x'
    real_type = SymbolAttributes('real', kind=routine.variable_map['jprb'])
    int_type = SymbolAttributes('integer')
    a = Scalar(name='a', type=real_type, scope=routine)
    b = Array(name='b', dimensions=(x, ), type=real_type, scope=routine)
    c = Variable(name='c', type=int_type, scope=routine)

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
real(kind=selected_real_kind(13, 300)), intent(inout) :: vector(1:x)
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
real(kind=jprb), intent(inout) :: vector(x)
real(kind=jprb) :: matrix(x, y)
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
    # Ensure `maximum` has been removed from arguments, but they are otherwise unharmed
    assert [str(arg) for arg in routine.arguments] in (
        ['x', 'y', 'vector(x)'],
        ['x', 'y', 'vector(1:x)']
    )


@pytest.mark.parametrize('frontend', available_frontends())
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


@pytest.mark.parametrize('frontend', available_frontends())
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


@pytest.mark.parametrize('frontend', available_frontends())
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
    header = Sourcefile.from_file(header_path, frontend=frontend)['header']
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=header)

    # Verify that all derived type variables have shape info
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    # Verify shape info from imported derived type is propagated
    vmap = {v.name: v for v in variables}
    assert fexprgen(vmap['item%vector'].shape) in ['(3,)', '(1:3,)']
    assert fexprgen(vmap['item%matrix'].shape) in ['(3, 3)', '(1:3, 1:3)']


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI does not like Loki pragmas, yet!')]))
def test_routine_variables_dimension_pragmas(frontend):
    """
    Test that `!$loki dimension` pragmas can be used to verride the
    conceptual `.shape` of local and argument variables.
    """
    fcode = """
subroutine routine_variables_dimensions(x, y, v1, v2, v3, v4)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  !$loki dimension(x,:)
  real(kind=jprb), intent(inout) :: v1(:,:)
  !$loki dimension(x,y,:)
  real(kind=jprb), dimension(:,:,:), intent(inout) :: v2, v3
  !$loki dimension(x,y)
  real(kind=jprb), pointer, intent(inout) :: v4(:,:)
  !$loki dimension(y,:)
  real(kind=jprb), allocatable :: v5(:,:)
  !$loki dimension(x+y)
  real(kind=jprb), dimension(:), pointer :: v6

end subroutine routine_variables_dimensions
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert fexprgen(routine.variable_map['v1'].shape) == '(x, :)'
    assert fexprgen(routine.variable_map['v2'].shape) == '(x, y, :)'
    assert fexprgen(routine.variable_map['v3'].shape) == '(x, y, :)'
    assert fexprgen(routine.variable_map['v4'].shape) == '(x, y)'
    assert fexprgen(routine.variable_map['v5'].shape) == '(y, :)'
    assert fexprgen(routine.variable_map['v6'].shape) == '(x+y,)'


@pytest.mark.parametrize('frontend', available_frontends())
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
    assert routine.arguments[0].type.dtype == BasicType.INTEGER
    assert routine.arguments[1].type.dtype == BasicType.INTEGER
    assert routine.arguments[2].type.dtype == BasicType.REAL
    assert str(routine.arguments[2].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert routine.arguments[3].type.dtype == BasicType.REAL
    assert str(routine.arguments[3].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert routine.arguments[4].type.dtype == BasicType.REAL
    assert str(routine.arguments[4].type.kind) in ('jprb', 'selected_real_kind(13, 300)')

    # Verify that all variable instances have type information
    variables = FindVariables().visit(routine.body)
    assert all(v.type is not None for v in variables if isinstance(v, (Scalar, Array)))

    vmap = {v.name: v for v in variables}
    assert vmap['x'].type.dtype == BasicType.INTEGER
    assert vmap['scalar'].type.dtype == BasicType.REAL
    assert str(vmap['scalar'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['vector'].type.dtype == BasicType.REAL
    assert str(vmap['vector'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['matrix'].type.dtype == BasicType.REAL
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
    header = Sourcefile.from_file(header_path, frontend=frontend)['header']
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=header)

    # Check that external typedefs have been propagated to kernel variables
    # First check that the declared parent variable has the correct type
    assert routine.arguments[0].name == 'item'
    assert routine.arguments[0].type.dtype.name == 'derived_type'

    # Verify that all variable instances have type and shape information
    variables = FindVariables().visit(routine.body)
    assert all(v.type is not None for v in variables)

    # Verify imported derived type info explicitly
    vmap = {v.name: v for v in variables}
    assert vmap['item%scalar'].type.dtype == BasicType.REAL
    assert str(vmap['item%scalar'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['item%vector'].type.dtype == BasicType.REAL
    assert str(vmap['item%vector'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['item%matrix'].type.dtype == BasicType.REAL
    assert str(vmap['item%matrix'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')


@pytest.mark.parametrize('frontend', available_frontends())
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
    header = Sourcefile.from_file(header_path, frontend=frontend)['header']
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=header)
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

    assert fgen(call) == 'CALL routine_call_callee(x, y, vector, matrix, item%matrix)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_call_no_arg(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_call_no_arg()
  implicit none

  call abort
end subroutine routine_call_no_arg
""")
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments == ()
    assert calls[0].kwarguments == ()


@pytest.mark.parametrize('frontend', available_frontends())
def test_call_kwargs(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_call_kwargs()
  implicit none
  integer :: kprocs

  call mpl_init(kprocs=kprocs, cdstring='routine_call_kwargs')
end subroutine routine_call_kwargs
""")
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].name == 'mpl_init'

    assert calls[0].arguments == ()
    assert len(calls[0].kwarguments) == 2
    assert all(isinstance(arg, tuple) and len(arg) == 2 for arg in calls[0].kwarguments)

    assert calls[0].kwarguments[0][0] == 'kprocs'
    assert (isinstance(calls[0].kwarguments[0][1], Scalar) and
            calls[0].kwarguments[0][1].name == 'kprocs')

    assert calls[0].kwarguments[1] == ('cdstring', StringLiteral('routine_call_kwargs'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_call_args_kwargs(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_call_args_kwargs(pbuf, ktag, kdest)
  implicit none
  integer, intent(in) :: pbuf(:), ktag, kdest

  call mpl_send(pbuf, ktag, kdest, cdstring='routine_call_args_kwargs')
end subroutine routine_call_args_kwargs
""")
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].name == 'mpl_send'
    assert len(calls[0].arguments) == 3
    assert all(a.name == b.name for a, b in zip(calls[0].arguments, routine.arguments))
    assert calls[0].kwarguments == (('cdstring', StringLiteral('routine_call_args_kwargs')),)


@pytest.mark.parametrize('frontend', available_frontends())
def test_convert_endian(here, frontend):
    pre = """
SUBROUTINE ROUTINE_CONVERT_ENDIAN()
  INTEGER :: IUNIT
  CHARACTER(LEN=100) :: CL_CFILE
"""
    body = """
IUNIT = 61
OPEN(IUNIT, FILE=TRIM(CL_CFILE), FORM="UNFORMATTED", CONVERT='BIG_ENDIAN')
IUNIT = 62
OPEN(IUNIT, FILE=TRIM(CL_CFILE), CONVERT="LITTLE_ENDIAN", &
  & FORM="UNFORMATTED")
"""
    post = """
END SUBROUTINE ROUTINE_CONVERT_ENDIAN
"""
    fcode = pre + body + post

    filepath = here/(f'routine_convert_endian_{frontend}.f90')
    Sourcefile.to_file(fcode, filepath)
    routine = Sourcefile.from_file(filepath, frontend=frontend, preprocess=True)['routine_convert_endian']

    if frontend == OMNI:
        # F... OMNI
        body = body.replace('OPEN(IUNIT', 'OPEN(UNIT=IUNIT')
        body = body.replace('"', "'")
        body = body.replace('&\n  & ', '')
    # TODO: This is hacky as the fgen backend is still pretty much WIP
    assert fgen(routine.body).upper().strip() == body.strip()
    filepath.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_open_newunit(here, frontend):
    pre = """
SUBROUTINE ROUTINE_OPEN_NEWUNIT()
  INTEGER :: IUNIT
  CHARACTER(LEN=100) :: CL_CFILE
"""
    body = """
OPEN(NEWUNIT=IUNIT, FILE=TRIM(CL_CFILE), FORM="UNFORMATTED")
OPEN(FILE=TRIM(CL_CFILE), FORM="UNFORMATTED", NEWUNIT=IUNIT)
OPEN(FILE=TRIM(CL_CFILE), NEWUNIT=IUNIT, &
  & FORM="UNFORMATTED")
OPEN(FILE=TRIM(CL_CFILE), NEWUNIT=IUNIT&
  & , FORM="UNFORMATTED")
"""
    post = """
END SUBROUTINE ROUTINE_OPEN_NEWUNIT
"""
    fcode = pre + body + post

    filepath = here/(f'routine_open_newunit_{frontend}.f90')
    Sourcefile.to_file(fcode, filepath)
    routine = Sourcefile.from_file(filepath, frontend=frontend, preprocess=True)['routine_open_newunit']

    if frontend == OMNI:
        # F... OMNI
        body = body.replace('"', "'")
        body = body.replace('&\n  & ', '')
    # TODO: This is hacky as the fgen backend is still pretty much WIP
    assert fgen(routine.body).upper().strip() == body.strip()
    filepath.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
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
    assert len(routine.body.body) == 1


@pytest.mark.parametrize('frontend', available_frontends())
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
  out2 = member_function(out1)
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

  function member_function(in2)
    ! This function is just included to test that functions
    ! are also possible
    implicit none
    integer, intent(in) :: in2
    integer :: member_function

    member_function = 3 * in2 + 2
  end function member_function
end subroutine routine_member_procedures
"""
    # Check that member procedures are parsed correctly
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(routine.members) == 2

    assert routine.members[0].name == 'member_procedure'
    assert routine.members[0].symbol_attrs.lookup('localvar', recursive=False) is None
    assert routine.members[0].symbol_attrs.lookup('localvar') is not None
    assert routine.members[0].get_symbol_scope('localvar') is routine
    assert routine.members[0].symbol_attrs.lookup('in1') is not None
    assert routine.symbol_attrs.lookup('in1') is not None
    assert routine.members[0].get_symbol_scope('in1') is routine.members[0]

    assert routine.members[1].name == 'member_function'
    assert routine.members[1].symbol_attrs.lookup('in2') is not None
    assert routine.members[1].get_symbol_scope('in2') is routine.members[1]
    assert routine.symbol_attrs.lookup('in2') is not None
    assert routine.get_symbol_scope('in2') is routine

    # Generate code, compile and load
    filepath = here/(f'routine_member_procedures_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='routine_member_procedures')

    # Test results of the generated and compiled code
    out1, out2 = function(1, 2)
    assert out1 == 7
    assert out2 == 23
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_member_routine_clone(frontend):
    """
    Test that member subroutine scopes get cloned correctly.
    """
    fcode = """
subroutine member_routine_clone(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
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
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    new_routine = routine.clone()

    # Ensure we have cloned routine and member
    assert routine is not new_routine
    assert routine.members[0] is not new_routine.members[0]
    assert fgen(routine) == fgen(new_routine)
    assert fgen(routine.members[0]) == fgen(new_routine.members[0])

    # Check that the scopes are linked correctly
    assert routine.members[0].parent is routine
    assert new_routine.members[0].parent is new_routine

    # Check that variables are in the right scope everywhere
    assert all(v.scope is routine for v in FindVariables().visit(routine.ir))
    assert all(v.scope in (routine, routine.members[0]) for v in FindVariables().visit(routine.members[0].ir))
    assert all(v.scope is new_routine for v in FindVariables().visit(new_routine.ir))
    assert all(
        v.scope in (new_routine, new_routine.members[0])
        for v in FindVariables().visit(new_routine.members[0].ir)
    )


@pytest.mark.parametrize('frontend', available_frontends())
def test_member_routine_clone_inplace(frontend):
    """
    Test that member subroutine scopes get cloned correctly.
    """
    fcode = """
subroutine member_routine_clone(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
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

  subroutine other_member(inout1)
    ! Another member that uses a parent symbol
    implicit none
    integer, intent(inout) :: inout1

    inout1 = 2 * inout1 + localvar
  end subroutine other_member
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Make sure the initial state is as expected
    member = routine['member_procedure']
    assert member.parent is routine
    assert member.symbol_attrs.parent is routine.symbol_attrs
    other_member = routine['other_member']
    assert other_member.parent is routine
    assert other_member.symbol_attrs.parent is routine.symbol_attrs

    # Put the inherited symbol in the local scope, first with a clean clone...
    member.variables += (routine.variable_map['localvar'].clone(scope=member),)
    member = member.clone(parent=None)
    # ...and then with a clone that preserves the symbol table
    other_member.variables += (routine.variable_map['localvar'].clone(scope=other_member),)
    other_member = other_member.clone(parent=None, symbol_attrs=other_member.symbol_attrs)
    # Ultimately, remove the member routines
    routine = routine.clone(contains=None)

    # Check that variables are in the right scope everywhere
    assert all(v.scope is routine for v in FindVariables().visit(routine.ir))
    assert all(v.scope is member for v in FindVariables().visit(member.ir))

    # Check that we aren't looking somewhere above anymore
    assert member.parent is None
    assert member.symbol_attrs.parent is None
    assert member._parent is None
    assert member.symbol_attrs._parent is None
    assert other_member.parent is None
    assert other_member.symbol_attrs.parent is None
    assert other_member._parent is None
    assert other_member.symbol_attrs._parent is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_external_stmt(here, frontend):
    """
    Tests procedures passed as dummy arguments and declared as EXTERNAL.
    """
    fcode_external = """
! This should be tested as well with interface statements in the caller
! routine, and the subprogram definitions outside (to have "truly external"
! procedures, however, we need to make the INTERFACE support more robust first

subroutine other_external_subroutine(outvar)
  implicit none
  integer, intent(out) :: outvar
  outvar = 4
end subroutine other_external_subroutine

function other_external_function() result(outvar)
  implicit none
  integer :: outvar
  outvar = 6
end function other_external_function
    """.strip()

    fcode = """
subroutine routine_external_stmt(invar, sub1, sub2, sub3, outvar, func1, func2, func3)
  implicit none
  integer, intent(in) :: invar
  external sub1
  external :: sub2, sub3
  integer, intent(out) :: outvar
  integer, external :: func1, func2
  integer, external :: func3
  integer tmp

  call sub1(tmp)
  outvar = invar + tmp  ! invar + 1
  call sub2(tmp)
  outvar = outvar + tmp + func1()  ! (invar + 1) + 1 + 6
  call sub3(tmp)
  outvar = outvar + tmp + func2()  ! (invar + 8) + 4 + 2
  tmp = func3()
  outvar = outvar + tmp  ! (invar + 14) + 2
end subroutine routine_external_stmt

subroutine routine_call_external_stmt(invar, outvar)
  implicit none
  integer, intent(in) :: invar
  integer, intent(out) :: outvar

  interface
    subroutine other_external_subroutine(outvar)
      integer, intent(out) :: outvar
    end subroutine other_external_subroutine
  end interface

  interface
    function other_external_function()
      integer :: other_external_function
    end function other_external_function
  end interface

  call routine_external_stmt(invar, external_subroutine, external_subroutine, other_external_subroutine, &
                            &outvar, other_external_function, external_function, external_function)

contains

  subroutine external_subroutine(outvar)
    implicit none
    integer, intent(out) :: outvar
    outvar = 1
  end subroutine external_subroutine

  function external_function()
    implicit none
    integer :: external_function
    external_function = 2
  end function external_function

end subroutine routine_call_external_stmt
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['routine_external_stmt']
    assert len(routine.arguments) == 8

    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        # Skip local variables
        if decl.symbols[0].name in ('invar', 'outvar', 'tmp'):
            continue
        # Is the EXTERNAL attribute set?
        assert decl.external
        for v in decl.symbols:
            # Are procedure names represented as Scalar objects?
            assert isinstance(v, ProcedureSymbol)
            assert isinstance(v.type.dtype, ProcedureType)
            assert v.type.external is True
            assert v.type.dtype.procedure == BasicType.DEFERRED
            if 'sub' in v.name:
                assert not v.type.dtype.is_function
                assert v.type.dtype.return_type is None
            else:
                assert v.type.dtype.is_function
                assert v.type.dtype.return_type.compare(SymbolAttributes(BasicType.INTEGER))

    # Generate code, compile and load
    extpath = here/(f'subroutine_routine_external_{frontend}.f90')
    with extpath.open('w') as f:
        f.write(fcode_external)
    filepath = here/(f'subroutine_routine_external_stmt_{frontend}.f90')
    source.path = filepath
    lib = jit_compile_lib([source, extpath], path=here, name='subroutine_external')
    function = lib.routine_call_external_stmt

    outvar = function(7)
    assert outvar == 23
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_subroutine_interface(here, frontend):
    """
    Test auto-generation of an interface block for a given subroutine.
    """
    fcode = """
subroutine test_subroutine_interface (in1, in2, out1, out2)
  use header, only: jprb
  IMPLICIT NONE
  integer, intent(in) :: in1, in2
  real(kind=jprb), intent(out) :: out1, out2
  integer :: localvar
  localvar = in1 + in2
  out1 = real(localvar, kind=jprb)
  out2 = out1 + 2.
end subroutine
"""
    routine = Subroutine.from_source(fcode, xmods=[here/'source/xmod'], frontend=frontend)

    if frontend == OMNI:
        assert fgen(routine.interface).strip() == """
INTERFACE
  SUBROUTINE test_subroutine_interface (in1, in2, out1, out2)
    USE header, ONLY: jprb
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: in1
    INTEGER, INTENT(IN) :: in2
    REAL(KIND=selected_real_kind(13, 300)), INTENT(OUT) :: out1
    REAL(KIND=selected_real_kind(13, 300)), INTENT(OUT) :: out2
  END SUBROUTINE test_subroutine_interface
END INTERFACE
""".strip()
    else:
        assert fgen(routine.interface).strip() == """
INTERFACE
  SUBROUTINE test_subroutine_interface (in1, in2, out1, out2)
    USE header, ONLY: jprb
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: in1, in2
    REAL(KIND=jprb), INTENT(OUT) :: out1, out2
  END SUBROUTINE test_subroutine_interface
END INTERFACE
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails without dummy module provided')]))
def test_subroutine_rescope_symbols(frontend):
    """
    Test the rescoping of variables.
    """
    fcode = """
subroutine test_subroutine_rescope(a, b, n)
  use some_mod, only: ext1
  implicit none
  integer, intent(in) :: a(n)
  integer, intent(out) :: b(n)
  integer, intent(in) :: n
  integer :: j

  b(:) = 0

  do j=1,n
    b(j) = a(j)
  end do

  call nested_routine(b, n)
contains

  subroutine nested_routine(a, n)
    use some_mod, only: ext2
    integer, parameter :: jpim = selected_int_kind(4)
    integer, intent(inout) :: a
    integer, intent(in) :: n
    integer(kind=jpim) :: j

    do j=1,n
      a(j) = a(j) + 1
    end do

    call ext1(a)
    call ext2(a)
  end subroutine nested_routine
end subroutine test_subroutine_rescope
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(routine)

    # Create a copy of the nested subroutine with rescoping and
    # make sure all symbols are in the right scope
    nested_spec = Transformer().visit(routine.members[0].spec)
    nested_body = Transformer().visit(routine.members[0].body)
    nested_routine = Subroutine(name=routine.members[0].name, args=routine.members[0]._dummies,
                                spec=nested_spec, body=nested_body, parent=routine,
                                rescope_symbols=True)

    for var in FindTypedSymbols().visit(nested_routine.ir):
        if var.name == 'ext1':
            assert var.scope is routine
        else:
            assert var.scope is nested_routine

    # Make sure the KIND parameter symbol in the variable's type is also correctly rescoped
    assert routine.members[0].variable_map['j'].type.kind.scope is routine.members[0]
    assert nested_routine.variable_map['j'].type.kind.scope is nested_routine

    # Create another copy of the nested subroutine without rescoping
    nested_spec = Transformer().visit(routine.members[0].spec)
    nested_body = Transformer().visit(routine.members[0].body)
    other_routine = Subroutine(name=routine.members[0].name, args=routine.members[0].argnames,
                               spec=nested_spec, body=nested_body, parent=routine)

    # Save the kind symbol for later
    other_kind_var = other_routine.variable_map['j'].type.kind
    assert other_kind_var.scope is routine.members[0]

    # Explicitly throw away type information from original nested routine
    routine.members[0]._parent = None
    routine.members[0].symbol_attrs.clear()
    routine.members[0].symbol_attrs._parent = None
    assert all(var.type is None for var in other_routine.variables)
    assert all(var.scope is not None for var in other_routine.variables)

    # Replace member routine by copied routine
    contains = [nested_routine if isinstance(c, Subroutine) else c for c in routine.contains.body]
    routine.contains = routine.contains.clone(body=contains)

    # Now, all variables should still be well-defined and fgen should produce the same string
    assert all(var.scope is not None for var in nested_routine.variables)
    assert fgen(routine) == ref_fgen

    # accessing any local type information should fail because either the scope got garbage
    # collected or its types are gonee
    assert all(var.scope is None or var.type is None for var in other_routine.variables)

    # Make sure changes apply also to the KIND attribute
    assert routine.members[0].variable_map['j'].type.kind.scope is routine.members[0]

    # This points (weakly) to an entry in routine.members[0].symbols which may or may not
    # have been garbage collected at this point
    assert other_kind_var.scope is not other_routine

    # fgen of the not rescoped routine should lack some type information and thus either fail or
    # produce a different output, depending on whether GC has already happened
    try:
        other_fgen = fgen(other_routine)
        assert other_fgen != ref_fgen
        assert len(other_fgen) < len(ref_fgen)
    except AttributeError as e:
        assert str(e) in (
            "'NoneType' object has no attribute 'compare'",
            "'NoneType' object has no attribute 'dtype'",
            "'NoneType' object has no attribute 'use_name'"
        )


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parser fails without dummy module provided')]))
def test_subroutine_rescope_clone(frontend):
    """
    Test the rescoping of variables in clone.
    """
    fcode = """
subroutine test_subroutine_rescope_clone(a, b, n)
  use some_mod, only: ext1
  implicit none
  integer, intent(in) :: a(n)
  integer, intent(out) :: b(n)
  integer, intent(in) :: n
  integer :: j

  b(:) = 0

  do j=1,n
    b(j) = a(j)
  end do

  call nested_routine(b, n)
contains

  subroutine nested_routine(a, n)
    use some_mod, only: ext2
    integer, intent(inout) :: a
    integer, intent(in) :: n
    integer :: j

    do j=1,n
      a(j) = a(j) + 1
    end do

    call ext1(a)
    call ext2(a)
  end subroutine nested_routine
end subroutine test_subroutine_rescope_clone
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(routine)

    # Create a copy of the nested subroutine with rescoping and
    # make sure all symbols are in the right scope
    nested_routine = routine.members[0].clone()

    for var in FindTypedSymbols().visit(nested_routine.ir):
        if var.name == 'ext1':
            assert var.scope is routine
        else:
            assert var.scope is nested_routine

    # Create another copy of the nested subroutine without rescoping (this breaks
    # things on purpose and should never be done in practice, but hey, for the lolz)
    other_routine = routine.members[0].clone(symbol_attrs=routine.symbol_attrs.clone(), rescope_symbols=False)

    # Explicitly throw away type information from original nested routine
    routine.members[0]._parent = None
    routine.members[0].symbol_attrs.clear()
    routine.members[0].symbol_attrs._parent = None
    assert all(var.type is None for var in other_routine.variables)
    assert all(var.scope is not None for var in other_routine.variables)

    # Replace member routine by copied routine
    contains = [nested_routine if isinstance(c, Subroutine) else c for c in routine.contains.body]
    routine.contains = routine.contains.clone(body=contains)

    # Now, all variables should still be well-defined and fgen should produce the same string
    assert all(var.scope is not None for var in nested_routine.variables)
    assert fgen(routine) == ref_fgen

    # accessing any local type information should fail because either the scope got garbage
    # collected or its types are gonee
    assert all(var.scope is None or var.type is None for var in other_routine.variables)

    # fgen of the not rescoped routine should lack some type information and thus either fail or
    # produce a different output, depending on whether GC has already happened
    try:
        other_fgen = fgen(other_routine)
        assert other_fgen != ref_fgen
        assert len(other_fgen) < len(ref_fgen)
    except AttributeError as e:
        assert str(e) in (
            "'NoneType' object has no attribute 'compare'",
            "'NoneType' object has no attribute 'dtype'",
            "'NoneType' object has no attribute 'use_name'"
        )


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP, 'No support for statement functions')]))
def test_subroutine_stmt_func(here, frontend):
    """
    Test the correct identification of statement functions
    """
    fcode = """
subroutine subroutine_stmt_func(a, b)
    implicit none
    integer, intent(in) :: a
    integer, intent(out) :: b
    integer :: i, j
    integer :: plus, minus
    plus(i, j) = i + j
    minus(i, j) = i - j
    integer :: mult
    mult(i, j) = i * j
    integer :: tmp

    tmp = plus(a, 5)
    tmp = minus(tmp, 1)
    b = mult(2, tmp)
end subroutine subroutine_stmt_func
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine.name += f'_{frontend!s}'

    # OMNI inlines statement functions, so we can only check correct representation
    # for fparser
    if frontend != OMNI:
        stmt_func_decls = {d.variable: d for d in FindNodes(StatementFunction).visit(routine.spec)}
        assert len(stmt_func_decls) == 3

        for name in ('plus', 'minus', 'mult'):
            var = routine.variable_map[name]
            assert isinstance(var, ProcedureSymbol)
            assert isinstance(var.type.dtype, ProcedureType)
            assert var.type.dtype.procedure is stmt_func_decls[var]

    # Make sure this produces the correct result
    filepath = here/f'{routine.name}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    assert function(3) == 14
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_mixed_declaration_interface(frontend):
    """
    A simple test to catch and shame mixed declarations.
    """
    fcode = """
subroutine valid_fortran(i, m)
   integer :: i, j, m
   integer :: k,l
end subroutine valid_fortran
"""

    with pytest.raises(AssertionError) as error:
        routine = Subroutine.from_source(fcode, frontend=frontend)
        assert isinstance(routine.body, Section)
        assert isinstance(routine.spec, Section)
        _ = routine.interface

    assert "Declarations must have intents" in str(error.value)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP, 'Prefix support not implemented')]))
def test_subroutine_prefix(frontend):
    """
    Test various prefixes that can occur in function/subroutine definitions
    """
    fcode = """
pure elemental real function f_elem(a)
    real, intent(in) :: a
    f_elem = a
end function f_elem
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert 'PURE' in routine.prefix
    assert 'ELEMENTAL' in routine.prefix
    assert routine.is_function is True
    assert routine.return_type.dtype is BasicType.REAL

    assert routine.name in routine.symbol_map
    decl = [d for d in FindNodes(VariableDeclaration).visit(routine.spec) if routine.name in d.symbols]
    assert len(decl) == 1
    decl = decl[0]

    assert routine.procedure_type.is_function is True
    assert routine.procedure_type.return_type.dtype is BasicType.REAL
    assert routine.procedure_type.procedure is routine

    assert routine.procedure_symbol.type.dtype.is_function is True
    assert routine.procedure_symbol.type.dtype.return_type.dtype is BasicType.REAL
    assert routine.procedure_symbol.type.dtype.procedure is routine

    code = fgen(routine)
    assert 'PURE' in code
    assert 'ELEMENTAL' in code
    assert fgen(decl) in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_subroutine_comparison(frontend):
    """
    Test that string-equivalence works on relevant components.
    """

    fcode = """
subroutine my_routine(n, a, b, d)
  integer, intent(in) :: n
  real, intent(in) :: a(n), b(n)
  real, intent(out) :: d(n)
  integer :: i

  do i=1, n
    d(i) = a(i) + b(i)
  end do
end subroutine my_routine
"""
    # Two distinct string-equivalent subroutine objects
    r1 = Subroutine.from_source(fcode)
    r2 = Subroutine.from_source(fcode)

    assert r1.symbol_attrs == r2.symbol_attrs
    assert r1.spec == r2.spec
    assert r1.body == r2.body
    assert r1 == r2

    # Counter example: Change the semantic meaning by adding an index
    # offset, so that symbol table and declaration spec are identical.
    r3 = Subroutine.from_source(fcode.replace('d(i)', 'd(i+1)'))
    assert r1.symbol_attrs == r3.symbol_attrs
    assert r1.spec == r3.spec
    assert not r1.body == r3.body
    assert not r1 == r3


@pytest.mark.parametrize('frontend', available_frontends())
def test_subroutine_comparison_case_sensitive(frontend):
    """
    Test that string-equivalence works and is case sensitive.
    """
    from loki import config, config_override  # pylint: disable=import-outside-toplevel

    fcode = """
subroutine my_routine(n, a, b, d)
  integer, intent(in) :: n
  real, intent(in) :: a(n), b(n)
  real, intent(out) :: d(n)
  integer :: i

  do i=1, n
    d(i) = a(i) + b(i)
  end do
end subroutine my_routine
"""
    # Create two subroutine objects, but capitalize a variable in one
    r1 = Subroutine.from_source(fcode)
    r2 = Subroutine.from_source(fcode.replace('d(i)', 'D(I)'))

    assert not 'D(I)' in fgen(r1)
    assert 'D(I)' in fgen(r2)

    # Ensure basic non-case-sensitive equivalence
    assert r1.symbol_attrs == r2.symbol_attrs
    assert r1.spec == r2.spec
    assert r1.body == r2.body
    assert r1 == r2

    with config_override({'case-sensitive': True}):
        assert r1.symbol_attrs == r2.symbol_attrs
        assert r1.spec == r2.spec
        assert not r1.body == r2.body
        assert not r1 == r2
