import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, OFP, OMNI, FP, FindVariables, DataType, Array, Scalar
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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_simple(refpath, reference, header_path, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_simple_caching(refpath, reference, header_path, frontend):
    """
    A simple standard looking routine to test variable caching.
    """

    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_simple']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'scalar', 'vector(1:x)', 'matrix(1:x,1:y)']
    assert routine.arguments[2].type.dtype == DataType.REAL
    assert routine.arguments[3].type.dtype == DataType.REAL

    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_simple_caching']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(y)', 'matrix(x,y)'] \
        or routine_args == ['x', 'y', 'scalar', 'vector(1:y)', 'matrix(1:x,1:y)']
    # Ensure that the types in the second routine have been picked up
    assert routine.arguments[2].type.dtype == DataType.INTEGER
    assert routine.arguments[3].type.dtype == DataType.INTEGER


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_multiline_args(refpath, reference, header_path, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_local_variables(refpath, reference, header_path, frontend):
    """
    Test local variables and types
    """
    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_arguments(refpath, reference, header_path, frontend):
    """
    A set of test to test internalisation and handling of arguments.
    """
    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_find_variables(refpath, reference, header_path, frontend):
    """
    Tests the `FindVariables` utility (not the best place to put this).
    """
    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_dim_shapes(refpath, reference, header_path, frontend):
    """
    A set of test to ensure matching different dimension and shape
    expressions against strings and other expressions works as expected.
    """
    from loki import FCodeMapper
    fsymgen = FCodeMapper()

    # FIXME: Forces pre-parsing of header module for OMNI parser to generate .xmod!
    _ = SourceFile.from_file(header_path, frontend=frontend)['header']

    # TODO: Need a named subroutine lookup
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_dim_shapes']
    routine_args = [fsymgen(arg) for arg in routine.arguments]
    assert routine_args == ['v1', 'v2', 'v3(:)', 'v4(v1,v2)', 'v5(1:v1,v2 - 1)'] \
        or routine_args == ['v1', 'v2', 'v3(:)', 'v4(1:v1,1:v2)', 'v5(1:v1,1:v2 - 1)'] \

    # Make sure variable/argument shapes on the routine work
    shapes = [fsymgen(v.shape) for v in routine.arguments if isinstance(v, Array)]
    assert shapes == ['(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'] \
        or shapes == ['(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)']

    # Ensure shapes of body variables are ok
    b_shapes = [fsymgen(v.shape) for v in FindVariables(unique=False).visit(routine.ir)
                if isinstance(v, Array)]
    assert b_shapes == ['(v1,)', '(v1,)', '(v1, v2)', '(1:v1, v2 - 1)'] \
        or b_shapes == ['(v1,)', '(v1,)', '(1:v1, 1:v2)', '(1:v1, 1:v2 - 1)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_shape_propagation(refpath, reference, header_path, header_mod, frontend):
    """
    Test for the correct identification and forward propagation of variable shapes
    from the subroutine declaration.
    """
    from loki import FCodeMapper
    fsymgen = FCodeMapper()

    # Parse simple kernel routine to check plain array arguments
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['routine_simple']

    # Check shapes on the internalized variable and argument lists
    x, y, = routine.arguments[0], routine.arguments[1]
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
    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    source = SourceFile.from_file(refpath, frontend=frontend, typedefs=header.typedefs)
    routine = source['routine_typedefs_simple']

    # Verify that all derived type variables have shape info
    variables = FindVariables().visit(routine.body)
    assert all(v.shape is not None for v in variables if isinstance(v, Array))

    # Verify shape info from imported derived type is propagated
    vmap = {v.name: v for v in variables}
    assert fsymgen(vmap['item%vector'].shape) in ['(3,)', '(1:3,)']
    assert fsymgen(vmap['item%matrix'].shape) in ['(3, 3)', '(1:3, 1:3)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    assert all(v.type is not None for v in variables
               if isinstance(v, Scalar) or isinstance(v, Array))

    vmap = {v.name: v for v in variables}
    assert vmap['x'].type.dtype == DataType.INTEGER
    assert vmap['scalar'].type.dtype == DataType.REAL
    assert str(vmap['scalar'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['vector'].type.dtype == DataType.REAL
    assert str(vmap['vector'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['matrix'].type.dtype == DataType.REAL
    assert str(vmap['matrix'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')

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
    assert vmap['item%scalar'].type.dtype == DataType.REAL
    assert str(vmap['item%scalar'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['item%vector'].type.dtype == DataType.REAL
    assert str(vmap['item%vector'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')
    assert vmap['item%matrix'].type.dtype == DataType.REAL
    assert str(vmap['item%matrix'].type.kind) in ('jprb', 'selected_real_kind(13, 300)')


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_routine_call_arrays(refpath, reference, header_path, header_mod, frontend):
    """
    Test that arrays passed down a subroutine call are treated as arrays.
    """
    from loki import FindNodes, CallStatement, FCodeMapper, fgen

    header = SourceFile.from_file(header_path, frontend=frontend)['header']
    source = SourceFile.from_file(refpath, frontend=frontend, typedefs=header.typedefs)
    routine = source['routine_call_caller']
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


@pytest.mark.parametrize('frontend', [OFP, FP])  # OMNI applies preproc and doesn't keep directives
def test_pp_macros(refpath, reference, frontend):
    from loki import FindNodes, Intrinsic
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
