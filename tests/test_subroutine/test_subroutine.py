import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, fgen, OFP, OMNI
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
    routine = SourceFile.from_file(refpath, frontend=frontend).subroutines[0]
    assert routine.arguments == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)']
    assert routine.arguments == ['X', 'Y', 'SCALAR', 'VECTOR(X)', 'MATRIX(X,Y)']

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_simple', frontend=frontend)
    function = getattr(test, 'routine_simple_%s' % frontend)
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0,:] == 5.)
    assert np.all(matrix[1,:] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_multiline_args(refpath, reference, frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend).subroutines[0]
    assert routine.arguments == ['x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)']
    assert routine.arguments == ['X', 'Y', 'SCALAR', 'VECTOR(X)', 'MATRIX(X,Y)']

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_multiline_args', frontend=frontend)
    function = getattr(test, 'routine_multiline_args_%s' % frontend)
    x, y = 2, 3
    vector = np.zeros(x, order='F')
    matrix = np.zeros((x, y), order='F')
    function(x=x, y=y, scalar=5., vector=vector, matrix=matrix)
    assert np.all(vector == 5.)
    assert np.all(matrix[0,:] == 5.)
    assert np.all(matrix[1,:] == 10.)


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_local_variables(refpath, reference, frontend):
    """
    Test local variables and types
    """
    # Test the internals of the :class:`Subroutine`
    routine = SourceFile.from_file(refpath, frontend=frontend).subroutines[0]
    assert routine.variables == ['jprb', 'x', 'y', 'scalar', 'vector(x)', 'matrix(x,y)', 'i']
    assert routine.variables == ['JPRB', 'X', 'Y', 'SCALAR', 'VECTOR(X)', 'MATRIX(X,Y)', 'I']

    # Test the generated identity results
    test = generate_identity(refpath, 'routine_local_variables', frontend=frontend)
    function = getattr(test, 'routine_local_variables_%s' % frontend)
    maximum = function(x=3, y=4)
    assert np.all(maximum == 38.)  # 10*x + 2*y


@pytest.mark.parametrize('frontend', [OFP, OMNI])
def test_routine_dim_shapes(refpath, reference, frontend):
    """
    A set of test to ensure matching different dimension and shape
    expressions against strings and other expressions works as expected.
    """
    # TODO: Need a named subroutine lookup
    routine = SourceFile.from_file(refpath, frontend=frontend).routines[3]
    assert routine.arguments == ['v1', 'v2', 'v3(:)', 'v4(v1,v2)', 'v5(v1,v2-1)']

    # Make sure variable/argument shapes work
    shapes = [v.shape for v in routine.arguments]
    assert shapes == [None, None, ('v1',), ('v1','v2'), ('v1','v2-1')]

    # TODO: More in-depth (compoenent-wise) equivalence against strings
