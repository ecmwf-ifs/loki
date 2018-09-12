from sympy import symbols

from loki import Variable, Subroutine


def test_symbolic_equivalence():
    """
    Test symbolic equivalence for internal expressions.
    """
    x, y = symbols('x y')
    f = Variable(name='f', dimensions=(x, y))

    f_plus_1 = f + 1
    assert f + 1 == 1 + f
    assert f_plus_1 + 1 == 2 + f


def test_symbol_caching_global():
    """
    Test symbol caching via the global symbol cache.
    """
    x, y, a = symbols('x y a')
    f0 = Variable(name='f', dimensions=(x, y))
    f1 = Variable(name='f', dimensions=(x, y))
    # Symbol with different parents are not equivalent
    f2 = Variable(name='f', dimensions=(x, y), parent=a)
    f3 = Variable(name='f', dimensions=(x, y), parent=a)

    f0_plus_1 = f0 + 1
    assert f0 + 1 == 1 + f1
    assert f0_plus_1 + 1 == 2 + f1
    assert f0 != f2
    assert f1 != f2
    assert f2 + 1 == 1 + f3


def test_symbol_caching_kernel():
    """
    Test symbol caching via a symbol cache on a local kernel.
    """
    kernel1 = Subroutine(name='test_kernel1')
    kernel2 = Subroutine(name='test_kernel2')
    x, y, a = symbols('x y a')

    f0 = kernel1.Variable(name='f', dimensions=(x, y))
    f1 = kernel1.Variable(name='f', dimensions=(x, y))
    # Symbol with different parents are not equivalent
    f2 = kernel2.Variable(name='f', dimensions=(x, y))
    f3 = kernel2.Variable(name='f', dimensions=(x, y))

    f0_plus_1 = f0 + 1
    assert f0 + 1 == 1 + f1
    assert f0_plus_1 + 1 == 2 + f1
    assert f0 != f2 and f0 != f3
    assert f1 != f2 and f1 != f3
    assert f2 + 1 == 1 + f3
