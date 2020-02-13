import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, OFP, OMNI, FP, SourceFile
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'control_flow.f90'


@pytest.fixture(scope='module')
def reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath, cwd=str(refpath.parent), use_f90wrap=True)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_loop_nest_fixed(refpath, reference, frontend):
    """
    Basic loop nest loop:
        out1(i, j) = in1(i, j) + in2(i, j)

    Basic reduction:
        out2(j) = out2(j) + in1(i, j) * in1(i, j)
    """
    # Test the reference solution
    in1 = np.array([[1., 2.], [2., 3.], [3., 4.]], order='F')
    in2 = np.array([[2., 3.], [3., 4.], [4., 5.]], order='F')
    out1 = np.zeros((3, 2), order='F')
    out2 = np.zeros(2, order='F')

    reference.loop_nest_fixed(in1, in2, out1, out2)
    assert (out1 == [[3, 5], [5, 7], [7, 9]]).all()
    assert (out2 == [20, 38]).all()

    # Test the generated identity
    in1 = np.array([[1., 2.], [2., 3.], [3., 4.]], order='F')
    in2 = np.array([[2., 3.], [3., 4.], [4., 5.]], order='F')
    out1 = np.zeros((3, 2), order='F')
    out2 = np.zeros(2, order='F')

    test = generate_identity(refpath, 'loop_nest_fixed', frontend=frontend)
    function = getattr(test, 'loop_nest_fixed_%s' % frontend)
    function(in1, in2, out1, out2)
    assert (out1 == [[3, 5], [5, 7], [7, 9]]).all()
    assert (out2 == [20, 38]).all()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_loop_nest_variable(refpath, reference, frontend):
    """
    Basic loop nest loop:
        out1(i, j) = in1(i, j) + in2(i, j)

    Basic reduction:
        out2(j) = out2(j) + in1(i, j) * in1(i, j)
    """
    # Test the reference solution
    in1 = np.array([[1., 2.], [2., 3.], [3., 4.]], order='F')
    in2 = np.array([[2., 3.], [3., 4.], [4., 5.]], order='F')
    out1 = np.zeros((3, 2), order='F')
    out2 = np.zeros(2, order='F')

    reference.loop_nest_variable(3, 2, in1, in2, out1, out2)
    assert (out1 == [[3, 5], [5, 7], [7, 9]]).all()
    assert (out2 == [20, 38]).all()

    # Test the generated identity
    in1 = np.array([[1., 2.], [2., 3.], [3., 4.]], order='F')
    in2 = np.array([[2., 3.], [3., 4.], [4., 5.]], order='F')
    out1 = np.zeros((3, 2), order='F')
    out2 = np.zeros(2, order='F')

    test = generate_identity(refpath, 'loop_nest_variable', frontend=frontend)
    function = getattr(test, 'loop_nest_variable_%s' % frontend)
    function(3, 2, in1, in2, out1, out2)
    assert (out1 == [[3, 5], [5, 7], [7, 9]]).all()
    assert (out2 == [20, 38]).all()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_inline_conditionals(refpath, reference, frontend):
    in1, in2 = 2, 2
    out1, out2 = reference.inline_conditionals(in1, in2)
    assert out1 == 2 and out2 == 2

    in1, in2 = -2, 10
    out1, out2 = reference.inline_conditionals(in1, in2)
    assert out1 == 0 and out2 == 5

    test = generate_identity(refpath, 'inline_conditionals', frontend=frontend)
    function = getattr(test, 'inline_conditionals_%s' % frontend)

    in1, in2 = 2, 2
    out1, out2 = function(in1, in2)
    assert out1 == 2 and out2 == 2

    in1, in2 = -2, 10
    out1, out2 = function(in1, in2)
    assert out1 == 0 and out2 == 5


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_multi_body_conditionals(refpath, reference, frontend):
    out1, out2 = reference.multi_body_conditionals(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = reference.multi_body_conditionals(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = reference.multi_body_conditionals(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = reference.multi_body_conditionals(10)
    assert out1 == 5 and out2 == 5

    test = generate_identity(refpath, 'multi_body_conditionals', frontend=frontend)
    function = getattr(test, 'multi_body_conditionals_%s' % frontend)

    out1, out2 = function(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = function(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = function(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = function(10)
    assert out1 == 5 and out2 == 5


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_goto_stmt(refpath, reference, frontend):
    ref = reference.goto_stmt()
    assert ref == 3

    test = generate_identity(refpath, 'goto_stmt', frontend=frontend)
    function = getattr(test, 'goto_stmt_%s' % frontend)
    result = function()
    assert result == ref
