import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, FortranSourceFile, fgen


@pytest.fixture
def reference():
    return Path(__file__).parent / 'derived_types.f90'


def generate_identity(reference, routinename, modulename=None, suffix=None):
    """
    Generate the "identity" of a single subroutine with a specific suffix.
    """
    refpath = Path(reference)
    testname = refpath.parent/('%s_%s_%s.f90' % (reference.stem, routinename, suffix))
    source = FortranSourceFile(reference)
    if suffix:
        for routine in source.subroutines:
            routine.name += '_%s' % suffix
    if modulename:
        module = [m for m in source.modules if m.name == modulename][0]
        module.name += '_%s_%s' % (routinename, suffix)
        source.write(source=fgen(module), filename=testname)
    else:
        routine = [r for r in source.subroutines if r.name == routinename][0]
        source.write(source=fgen(routine), filename=testname)

    return compile_and_load(testname, use_f90wrap=modulename is not None)


def test_simple_loops(reference):
    """
    item%vector = item%vector + vec
    item%matrix = item%matrix + item%scalar
    """
    clean(filename=reference)  # Delete parser cache

    # Test the reference solution
    ref = compile_and_load(reference, use_f90wrap=True)
    item = ref.Structure()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    ref.simple_loops(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()

    # Test the generated identity
    test = generate_identity(reference, modulename='derived_types',
                             routinename='simple_loops', suffix='test')

    item = test.Structure()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    test.simple_loops_test(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()


def test_array_indexing(reference):
    """
    item.a(:, :) = 666.

    do i=1, 3
       item%b(:, i) = vals(i)
    end do
    """
    from loki import logger, DEBUG; logger.setLevel(DEBUG)

    clean(filename=reference)  # Delete parser cache

    # Test the reference solution
    ref = compile_and_load(reference, use_f90wrap=True)
    item = ref.Structure()
    ref.array_indexing(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()

    # Test the generated identity
    test = generate_identity(reference, modulename='derived_types',
                             routinename='array_indexing', suffix='test')

    item = test.Structure()
    test.array_indexing_test(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
