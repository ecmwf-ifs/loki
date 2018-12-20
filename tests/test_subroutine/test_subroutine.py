import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, OFP, OMNI, FindVariables
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'subroutine.f90'


@pytest.fixture(scope='module')
def reference(refpath):
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
    assert routine_args == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)']

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
def test_routine_multiline_args(refpath, reference, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_multiline_args']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)']

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
    assert routine_vars == ['jprb', 'x', 'y', 'maximum', 'i', 'j', 'vector(x)', 'matrix(x, y)']

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
    assert routine_vars == ['jprb', 'x', 'y', 'vector(x)', 'matrix(x, y)',
                            'i', 'j', 'local_vector(x)', 'local_matrix(x, y)']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['x', 'y', 'vector(x)', 'matrix(x, y)']

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


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_dim_shapes(refpath, reference, frontend):
    """
    A set of test to ensure matching different dimension and shape
    expressions against strings and other expressions works as expected.
    """
    # TODO: Need a named subroutine lookup
    routine = SourceFile.from_file(refpath, frontend=frontend)['routine_dim_shapes']
    routine_args = [str(arg) for arg in routine.arguments]
    assert routine_args == ['v1', 'v2', 'v3(:)', 'v4(v1, v2)', 'v5(v1, v2 - 1)']

    # Make sure variable/argument shapes on the routine work
    shapes = [str(v.shape) for v in routine.arguments if v.is_Array]
    assert shapes == ['(v1,)', '(v1, v2)', '(v1, v2 - 1)']

    # Ensure shapes of body variables are ok
    b_shapes = [str(v.shape) for v in FindVariables(unique=False).visit(routine.ir)
                if v.is_Function]
    assert b_shapes == ['(v1,)', '(v1, v2)', '(v1, v2 - 1)']
