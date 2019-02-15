import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, OFP, OMNI, FindVariables, DataType, Array, Scalar
from conftest import generate_identity


@pytest.fixture(scope='module')
def header_path():
    return Path(__file__).parent / 'header.f90'


@pytest.fixture(scope='module')
def header_mod(header_path):
    """
    Compile and load the reference solution
    """
    clean(filename=header_path)  # Delete parser cache
    return compile_and_load(header_path, cwd=str(header_path.parent))


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'subroutine.f90'


@pytest.fixture(scope='module')
def reference(refpath, header_mod):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath, cwd=str(refpath.parent))


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_simple(refpath, reference, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_simple']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)']  # OMNI

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_simple', frontend=frontend)
    function = getattr(test, 'routine_simple_%s' % frontend)
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_simple_caching(refpath, reference, frontend):
    """
    A simple standard looking routine to test variable caching.
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_simple']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)']
    assert routine.arguments[2].type.name.lower() == 'real'
    assert routine.arguments[3].type.name.lower() == 'real'

    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_simple_caching']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(y)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'scalar', 'vector(1:y)', 'matrix(1:x,1:y)']
    # Ensure that the types in the second routine have been picked up
    assert routine.arguments[2].type.name.lower() == 'integer'
    assert routine.arguments[3].type.name.lower() == 'integer'


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_multiline_args(refpath, reference, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_multiline_args']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)']

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_multiline_args', frontend=frontend)
    function = getattr(test, 'routine_multiline_args_%s' % frontend)
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0, :] == 5.)
    assert np.all(matrix[1, :] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_local_variables(refpath, reference, frontend):
    """
    Test local variables and types
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_local_variables']
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars == ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(x)', 'matrix(x,y)'] \
        or routine_vars == ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(1:x)', 'matrix(1:x,1:y)']

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_local_variables', frontend=frontend)
    function = getattr(test, 'routine_local_variables_%s' % frontend)
    maximum = function(x=3, y=4)
    assert np.all(maximum == 38.)  # 10*x + 2*y


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_arguments(refpath, reference, frontend):
    """
    A set of test to test internalisation and handling of arguments.
    """

    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_arguments']
    routine_vars = [str(arg) for arg in routine.variables]
    assert routine_vars == ['jprb', 'x', 'y', 'vector(x)', 'matrix(x,y)',
                            'i', 'j', 'local_vector(x)', 'local_matrix(x,y)'] \
        or routine_vars == ['jprb', 'x', 'y', 'vector(1:x)', 'matrix(1:x,1:y)',
                            'i', 'j', 'local_vector(1:x)', 'local_matrix(1:x,1:y)']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'vector(x)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'vector(1:x)', 'matrix(1:x,1:y)']

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_arguments', frontend=frontend)
    function = getattr(test, 'routine_arguments_%s' % frontend)
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, vector=vector, matrix=matrix)
    assert np.all(vector == [10., 20.])
    assert np.all(matrix == [[12., 14., 16.],
                             [22., 24., 26.]])


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_find_variables(refpath, reference, frontend):
    """
    Tests the `FindVariables` utility (not the best place to put this).
    """
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_local_variables']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_dim_shapes(refpath, reference, frontend):
    """
    A set of test to ensure matching different dimension and shape
    expressions against strings and other expressions works as expected.
    """
    # TODO: Need a named subroutine lookup
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_dim_shapes']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['v1', 'v2', 'v3(:)', 'v4(v1,v2)', 'v5(1:v1,v2 - 1)'] \
        or routine_args == ['v1', 'v2', 'v3(:)', 'v4(1:v1,1:v2)', 'v5(1:v1,1:v2 - 1)'] \

    # Make sure variable/argument shapes on the routine work
    shapes = [str(v.shape) for v in routine.arguments if v.is_Array]
    assert shapes == ['(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'] \
        or shapes == ['(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)']

    # Ensure shapes of body variables are ok
    b_shapes = [str(v.shape) for v in FindVariables(unique=False).visit(routine.ir)
                if v.is_Function]
    assert b_shapes == ['(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'] \
        or b_shapes == ['(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)']


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_shape_propagation(refpath, reference, header_path, header_mod, frontend):
    """
    Test for the correct identification and forward propagation of variable shapes
    from the subroutine declaration.
    """
    # Parse simple kernel routine to check plain array arguments
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['routine_simple']

    # Check shapes on the internalized variable and argument lists
    x, y, = routine.arguments[0], routine.arguments[1]
    # TODO: The string comparison here is due to the fact that shapes are actually
    # `RangeIndex(upper=Scalar)` objects, instead of the raw dimension variables.
    # This needs some more thorough conceptualisation of dimensions and indices!
    assert str(routine.arguments[3].shape) in ['(x,)', '(1:x,)']
    assert str(routine.arguments[4].shape) in ['(x, y)', '(1:x, 1:y)']

    # Verify that all variable instances have type and shape information
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    vmap = {v.name: v for v in variables}
    assert str(vmap['vector'].shape) in ['(x,)', '(1:x,)']
    assert str(vmap['matrix'].shape) in ['(x, y)', '(1:x, 1:y)']

    # Parse kernel with external typedefs to test shape inferred from
    # external derived type definition
    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    source = SourceFile.from_file(refpath, frontend=frontend, typedefs=header.typedefs)
    routine = source['routine_typedefs_simple']

    # Verify that all derived type variables have shape info
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    # Verify shape info from imported derived type is propagated
    vmap = {v.name: v for v in variables}
    assert str(vmap['item%vector'].shape) in ['(3,)', '(1:3,)']
    assert str(vmap['item%matrix'].shape) in ['(3, 3)', '(1:3, 1:3)']


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_type_propagation(refpath, reference, header_path, header_mod, frontend):
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
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['routine_simple']

    # Check types on the internalized variable and argument lists
    assert routine.arguments[0].type.dtype == DataType.INT32
    assert routine.arguments[1].type.dtype == DataType.INT32
    assert routine.arguments[2].type.dtype == DataType.FLOAT64
    assert routine.arguments[3].type.dtype == DataType.FLOAT64
    assert routine.arguments[4].type.dtype == DataType.FLOAT64

    # Verify that all variable instances have type information
    variables = FindVariables().visit(routine.body)
    assert all(v.type is not None for v in variables
               if isinstance(v, Scalar) or isinstance(v, Array))

    vmap = {v.name: v for v in variables}
    assert vmap['x'].type.dtype == DataType.INT32
    assert vmap['scalar'].type.dtype == DataType.FLOAT64
    assert vmap['vector'].type.dtype == DataType.FLOAT64
    assert vmap['matrix'].type.dtype == DataType.FLOAT64

    # Parse kernel routine and provide external typedefs
    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    source = SourceFile.from_file(refpath, frontend=frontend, typedefs=header.typedefs)
    routine = source['routine_typedefs_simple']

    # Check that external typedefs have been propagated to kernel variables
    # First check that the declared parent variable has the correct type
    assert routine.arguments[0].name == 'item'
    assert routine.arguments[0].type.name == 'derived_type'

    # Verify that all variable instances have type and shape information
    variables = FindVariables().visit(routine.body)
    assert all(v.type is not None for v in variables)

    # Verify imported derived type info explicitly
    vmap = {v.name: v for v in variables}
    assert vmap['item%scalar'].type.dtype == DataType.FLOAT64
    assert vmap['item%vector'].type.dtype == DataType.FLOAT64
    assert vmap['item%matrix'].type.dtype == DataType.FLOAT64


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_call_arrays(refpath, reference, header_path, header_mod, frontend):
    """
    Test that arrays passed down a subroutine call are treated as arrays.
    """
    from loki import FindNodes, Call, fgen

    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    source = SourceFile.from_file(refpath, frontend=frontend, typedefs=header.typedefs)
    routine = source['routine_call_caller']
    call = FindNodes(Call).visit(routine.body)[0]

    assert str(call.arguments[0]) == 'x'
    assert str(call.arguments[1]) == 'y'
    assert str(call.arguments[2]) == 'vector'
    assert str(call.arguments[3]) == 'matrix'
    assert str(call.arguments[4]) == 'item%matrix'

    assert call.arguments[0].is_Scalar
    assert call.arguments[1].is_Scalar
    assert call.arguments[2].is_Array
    assert call.arguments[3].is_Array
    assert call.arguments[4].is_Array

    assert str(call.arguments[2].shape) in ['(x,)', '(1:x,)']
    assert str(call.arguments[3].shape) in ['(x, y)', '(1:x, 1:y)']
    assert str(call.arguments[4].shape) in ['(3, 3)', '(1:3, 1:3)']

    assert fgen(call) == 'CALL routine_call_callee(x, y, vector, &\n     & matrix, item%matrix)'
