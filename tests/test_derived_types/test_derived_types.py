import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, OFP, OMNI, FP
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'derived_types.f90'


@pytest.fixture(scope='module')
def reference(refpath):
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
    # Test the reference solution
    item = reference.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    item.matrix = 0.
    reference.associates(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()
    assert (item.matrix[:, 0::2] == 3.).all()

    test = generate_identity(refpath, modulename='derived_types',
                             routinename='associates', frontend=frontend)

    # Test the generated identity
    item = reference.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    getattr(test, 'associates_%s' % frontend)(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()
