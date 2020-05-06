from pathlib import Path
import pytest
import numpy as np

from loki import clean, compile_and_load, OFP, OMNI, FP, SourceFile
from conftest import generate_identity


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent/'sources/derived_types.f90'


@pytest.fixture(scope='module', name='reference')
def fixture_reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    pymod = compile_and_load(refpath, cwd=str(refpath.parent))
    return getattr(pymod, refpath.stem)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_simple_loops(refpath, reference, frontend):
    """
    item%vector = item%vector + vec
    item%matrix = item%matrix + item%scalar
    """
    # Test the reference solution
    item = reference.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    reference.simple_loops(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='simple_loops', frontend=frontend)
    item = test.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    getattr(test, 'simple_loops_%s' % frontend)(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_array_indexing_explicit(refpath, reference, frontend):
    """
    item.a(:, :) = 666.

    do i=1, 3
       item%b(:, i) = vals(i)
    end do
    """
    # Test the reference solution
    item = reference.explicit()
    reference.array_indexing_explicit(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='array_indexing_explicit', frontend=frontend)
    item = test.explicit()
    getattr(test, 'array_indexing_explicit_%s' % frontend)(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_array_indexing_deferred(refpath, reference, frontend):
    """
    item.a(:, :) = 666.

    do i=1, 3
       item%b(:, i) = vals(i)
    end do
    """
    # Test the reference solution
    item = reference.deferred()
    reference.alloc_deferred(item)
    reference.array_indexing_deferred(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    reference.free_deferred(item)

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='array_indexing_deferred', frontend=frontend)
    item = test.deferred()
    reference.alloc_deferred(item)
    getattr(test, 'array_indexing_deferred_%s' % frontend)(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    reference.free_deferred(item)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_array_indexing_nested(refpath, reference, frontend):
    """
    item%a_vector(:) = 666.
    item%another_item%a_vector(:) = 999.

    do i=1, 3
       item%another_item%matrix(:, i) = vals(i)
    end do
    """
    # Test the reference solution
    item = reference.nested()
    reference.array_indexing_nested(item)
    assert (item.a_vector == 666.).all()
    assert (item.another_item.vector == 999.).all()
    assert (item.another_item.matrix == np.array([[1., 2., 3.],
                                                  [1., 2., 3.],
                                                  [1., 2., 3.]])).all()

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='array_indexing_nested', frontend=frontend)
    item = test.nested()
    getattr(test, 'array_indexing_nested_%s' % frontend)(item)
    assert (item.a_vector == 666.).all()
    assert (item.another_item.vector == 999.).all()
    assert (item.another_item.matrix == np.array([[1., 2., 3.],
                                                  [1., 2., 3.],
                                                  [1., 2., 3.]])).all()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_deferred_array(refpath, reference, frontend):
    """
    item2%vector(:) = 666.

    do i=1, 3
       item2%matrix(:, i) = vals(i)
    end do

    ----

    item%vector = item%vector + item2(:)%vector
    item%matrix = item%matrix + item2(:)%matrix
    """
    # Test the reference solution
    item = reference.deferred()
    reference.alloc_deferred(item)
    reference.deferred_array(item)
    assert (item.vector == 4 * 666.).all()
    assert (item.matrix == 4 * np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    reference.free_deferred(item)

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='deferred_array', frontend=frontend)
    item = test.deferred()
    reference.alloc_deferred(item)
    getattr(test, 'deferred_array_%s' % frontend)(item)
    assert (item.vector == 4 * 666.).all()
    assert (item.matrix == 4 * np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    reference.free_deferred(item)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_derived_type_caller(refpath, reference, frontend):
    """
    item%vector = item%vector + item%scalar
    item%matrix = item%matrix + item%scalar
    item%red_herring = 42.
    """
    # Test the reference solution
    item = reference.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    item.red_herring = -1.
    reference.derived_type_caller(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all() and item.red_herring == 42.

    test = generate_identity(refpath, modulename='derived_types',
                             routinename='derived_type_caller', frontend=frontend)

    # Test the generated identity
    item = test.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    item.red_herring = -1.
    getattr(test, 'derived_type_caller_%s' % frontend)(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all() and item.red_herring == 42.


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_associates(refpath, reference, frontend):
    """
    associate(vector=>item%vector)
    item%vector(2) = vector(1)
    vector(3) = item%vector(1) + vector(2)
    """
    from loki import FindVariables, IntLiteral, RangeIndex  # pylint: disable=import-outside-toplevel

    # Test the reference solution
    item = reference.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    item.matrix = 0.
    reference.associates(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()
    assert (item.matrix[:, 0::2] == 3.).all()

    # Test the internals
    routine = SourceFile.from_file(refpath, frontend=frontend)['associates']
    variables = FindVariables().visit(routine.body)
    if frontend == OMNI:
        assert all([v.shape == (RangeIndex(IntLiteral(1), IntLiteral(3)),)
                    for v in variables if v.name in ['vector', 'vector2']])
    else:
        assert all([isinstance(v.shape, tuple) and len(v.shape) == 1 and
                    isinstance(v.shape[0], IntLiteral) and v.shape[0].value == 3
                    for v in variables if v.name in ['vector', 'vector2']])

    test = generate_identity(refpath, modulename='derived_types',
                             routinename='associates', frontend=frontend)

    # Test the generated identity
    item = reference.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    getattr(test, 'associates_%s' % frontend)(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_associates_deferred(refpath, frontend):
    """
    Verify that reading in subroutines with deferred external type definitions
    and associates working on that are supported.
    """

    code = '''
SUBROUTINE ASSOCIATES_DEFERRED(ITEM, IDX)
USE SOME_MOD, ONLY: SOME_TYPE
IMPLICIT NONE
TYPE(SOME_TYPE), INTENT(IN) :: ITEM
INTEGER, INTENT(IN) :: IDX
ASSOCIATE(SOME_VAR=>ITEM%SOME_VAR(IDX))
SOME_VAR = 5
END ASSOCIATE
END SUBROUTINE
    '''
    from loki import FindVariables, Scalar, DataType  # pylint: disable=import-outside-toplevel

    filename = refpath.parent / ('associates_deferred_%s.f90' % frontend)
    with open(filename, 'w') as f:
        f.write(code)

    routine = SourceFile.from_file(filename, frontend=frontend)['associates_deferred']
    some_var = FindVariables().visit(routine.body).pop()
    assert isinstance(some_var, Scalar)
    assert some_var.name.upper() == 'SOME_VAR'
    assert some_var.type.dtype == DataType.DEFERRED


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_case_sensitivity(refpath, reference, frontend):
    """
    Some abuse of the case agnostic behaviour of Fortran
    """
    # Test the reference solution
    item = reference.case_sensitive()
    item.u = 0.
    item.v = 0.
    item.t = 0.
    item.q = 0.
    item.a = 0.
    reference.check_case(item)
    assert item.u == 1.0 and item.v == 2.0 and item.t == 3.0
    assert item.q == -1.0 and item.a == -5.0

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='check_case', frontend=frontend)
    item = test.case_sensitive()
    item.u = 0.
    item.v = 0.
    item.t = 0.
    item.q = 0.
    item.a = 0.
    function = getattr(test, 'check_case_%s' % frontend)
    function(item)
    assert item.u == 1.0 and item.v == 2.0 and item.t == 3.0
    assert item.q == -1.0 and item.a == -5.0
