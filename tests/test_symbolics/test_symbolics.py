from sympy import symbols, simplify

from loki import Variable, Scalar, Subroutine, InlineCall, Cast, fsymgen, BaseType


def test_symbolic_equivalence():
    """
    Test symbolic equivalence for internal expressions.
    """
    x = Variable(name='x')
    y = Variable(name='y')
    f = Variable(name='f', dimensions=(x, y))

    # Sanity check the SymPy-style class markers
    assert x.is_Symbol and x.is_Scalar
    assert y.is_Symbol and y.is_Scalar
    assert f.is_Function and f.is_Array

    f_plus_1 = f + 1
    assert f + 1 == 1 + f
    assert f_plus_1 + 1 == 2 + f


def test_symbol_caching_global():
    """
    Test symbol caching via the global symbol cache.
    """
    x = Scalar(name='x')
    y = Scalar(name='y')
    a = Variable(name='a')
    assert x.is_Symbol and x.is_Scalar
    assert y.is_Symbol and y.is_Scalar
    assert a.is_Symbol and a.is_Scalar

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

    x = Variable(name='x')
    y = Variable(name='y')

    f0 = kernel1.Variable(name='f', dimensions=(x, y))
    f1 = kernel1.Variable(name='f', dimensions=(x, y))
    # Symbol with different parents are not equivalent
    f2 = kernel2.Variable(name='f', dimensions=(x, y))
    f3 = kernel2.Variable(name='f', dimensions=(x, y))

    # from IPython import embed; embed()

    f0_plus_1 = f0 + 1
    assert f0 + 1 == 1 + f1
    assert f0_plus_1 + 1 == 2 + f1
    assert f0 != f2 and f0 != f3
    assert f1 != f2 and f1 != f3
    assert f2 + 1 == 1 + f3


def test_symbol_regenerate_variable():
    """
    Test symbols can be re-created by SymPy for simplification.
    """
    x = Variable(name='x')
    y = Variable(name='y')
    f = Variable(name='f', dimensions=(x, y))
    g = Variable(name='g')

    # Sanity check the SymPy-style class markers
    assert x.is_Symbol and x.is_Scalar
    assert y.is_Symbol and y.is_Scalar
    assert f.is_Function and f.is_Array
    assert g.is_Symbol and g.is_Scalar

    # Force simplification to trigger symbol re-generation
    assert 3*f == simplify(f + 2*f)
    assert 3*g == simplify(g + 2*g)


def test_symbol_regenerate_inlinecall():
    """
    Test custom symbols can be re-created by SymPy for simplification.
    """
    x = Variable(name='x')
    y = Variable(name='y')
    f = Variable(name='f', dimensions=(x, y))
    g = Variable(name='g')

    a = InlineCall('a', arguments=(f, g))
    b = InlineCall(name='b', arguments=(f, g), kwarguments=(('x', x), ('y', y)))

    # Sanity check the SymPy-style class markers
    assert not a.is_Symbol and not a.is_Function
    assert not b.is_Symbol and not b.is_Function

    # Force simplification to trigger symbol re-generation
    assert 3*a == simplify(a + 2*a)
    assert 3*b == simplify(b + 2*b)

    # And finally check correct Fortran printing
    assert fsymgen(a) == 'a(f(x, y), g)'
    assert fsymgen(b) == 'b(f(x, y), g, x=x, y=y)'


def test_symbol_regenerate_cast():
    """
    Test custom symbols can be re-created by SymPy for simplification.
    """
    x = Variable(name='x')
    y = Variable(name='y')
    f = Variable(name='f', dimensions=(x, y))
    g = Variable(name='g')

    a = Cast(name='real', expression=x + y)
    b = Cast(name='real', expression=x + y, kind='JPRB')
    kind = InlineCall(name='selected_real_kind', arguments=(13, 300))
    c = Cast(name='real', expression=x + y, kind=kind)

    # Sanity check the SymPy-style class markers
    assert not a.is_Symbol and not a.is_Function
    assert not b.is_Symbol and not b.is_Function
    assert not c.is_Symbol and not c.is_Function

    # Force simplification to trigger symbol re-generation
    assert 3*a == simplify(a + 2*a)
    assert 3*b == simplify(b + 2*b)
    assert 3*c == simplify(c + 2*c)

    # Check correct pretty-printing
    assert str(a) == 'real(x + y)'
    assert str(b) == 'real(x + y, kind=JPRB)'
    assert str(c) == 'real(x + y, kind=selected_real_kind(13, 300))'

    # And finally check correct Fortran printing
    assert fsymgen(a) == 'real(x + y)'
    assert fsymgen(b) == 'real(x + y, kind=JPRB)'
    assert fsymgen(c) == 'real(x + y, kind=selected_real_kind(13, 300))'
