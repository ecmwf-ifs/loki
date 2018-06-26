import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, fgen, OFP, OMNI


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'derived_types.f90'


@pytest.fixture(scope='module')
def reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath, cwd=str(refpath.parent), use_f90wrap=True)


def generate_identity(refpath, routinename, modulename=None, frontend=OFP):
    """
    Generate the "identity" of a single subroutine with a frontend-specific suffix.
    """
    testname = refpath.parent/('%s_%s_%s.f90' % (refpath.stem, routinename, frontend))
    source = SourceFile.from_file(refpath, frontend=frontend)
    for routine in source.subroutines:
        routine.name += '_%s' % frontend
    if modulename:
        module = [m for m in source.modules if m.name == modulename][0]
        module.name += '_%s_%s' % (routinename, frontend)
        source.write(source=fgen(module), filename=testname)
    else:
        routine = [r for r in source.subroutines if r.name == routinename][0]
        source.write(source=fgen(routine), filename=testname)

    return compile_and_load(testname, cwd=str(refpath.parent),
                            use_f90wrap=modulename is not None)


@pytest.mark.parametrize('frontend', [OFP])
def test_simple_loops(refpath, reference, frontend):
    """
    item%vector = item%vector + vec
    item%matrix = item%matrix + item%scalar
    """
    # Test the reference solution
    item = reference.Explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    reference.simple_loops(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='simple_loops', frontend=frontend)
    item = test.Explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    getattr(test, 'simple_loops_%s' % frontend)(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()


@pytest.mark.parametrize('frontend', [OFP])
def test_array_indexing_explicit(refpath, reference, frontend):
    """
    item.a(:, :) = 666.

    do i=1, 3
       item%b(:, i) = vals(i)
    end do
    """
    # Test the reference solution
    item = reference.Explicit()
    reference.array_indexing_explicit(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='array_indexing_explicit', frontend=frontend)
    item = test.Explicit()
    getattr(test, 'array_indexing_explicit_%s' % frontend)(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()


@pytest.mark.parametrize('frontend', [OFP])
def test_array_indexing_deferred(refpath, reference, frontend):
    """
    item.a(:, :) = 666.

    do i=1, 3
       item%b(:, i) = vals(i)
    end do
    """
    # Test the reference solution
    item = reference.Deferred()
    reference.alloc_deferred(item)
    reference.array_indexing_deferred(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    reference.free_deferred(item)

    # Test the generated identity
    test = generate_identity(refpath, modulename='derived_types',
                             routinename='array_indexing_deferred', frontend=frontend)
    item = test.Deferred()
    reference.alloc_deferred(item)
    getattr(test, 'array_indexing_deferred_%s' % frontend)(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    reference.free_deferred(item)
