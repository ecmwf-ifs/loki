from sympy import symbols, simplify

from loki import (Variable, Scalar, Array, Subroutine, InlineCall, Cast, fsymgen,
                  BaseType, indexify, SymbolCache)


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


def test_symbol_regenerate_meta_info():
    """
    Test symbols can be re-created by SymPy for simplification.
    """
    x = Variable(name='x')
    y = Variable(name='y')
    i = Variable(name='i')
    j = Variable(name='j')
    f = Variable(name='f', dimensions=(i, j), shape=(x, y))

    # Sanity check meta-info before simplification
    assert f.shape == (x, y)

    # Force simplification, but leave variable intact
    g = simplify(f + 2*f)
    assert g == 3*f
    assert g.args[1] == f
    assert g.args[1].shape == (x, y)

    # Replace index on variable and checl shape
    h = f.xreplace({j: i})
    assert h == f.func(i, i)
    assert h.shape == (x, y)


def test_symbol_regenerate_meta_from_cache():
    """
    Test symbols can be re-created by SymPy for simplification.
    """
    cache = SymbolCache()
    x = cache.Variable(name='x')
    y = cache.Variable(name='y')
    i = cache.Variable(name='i')
    j = cache.Variable(name='j')
    f = cache.Variable(name='f', dimensions=(i, j), shape=(x, y))

    # Sanity check meta-info before simplification
    assert f.shape == (x, y)

    # Force simplification, but leave variable intact
    g = simplify(f + 2*f)
    assert g == 3*f
    assert g.args[1] == f
    assert g.args[1].shape == (x, y)

    # Replace index on variable and checl shape
    h = f.xreplace({j: i})
    assert h == f.func(i, i)
    assert h.shape == (x, y)


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
    assert fsymgen(a) == 'a(f(x,y), g)'
    assert fsymgen(b) == 'b(f(x,y), g, x=x, y=y)'


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


def test_boolean_arrays():
    """
    SymPy does not like boolean logic symbols to be mixed with regular
    symbols, but of course, Fortran does do ``logical :: array(dim)``.
    This test ensure that our basic variable and expressions can deal
    with expressions that contain boolean array symbols.
    """

    x = Variable(name='x')
    y = Variable(name='y')
    booltype = BaseType(name='LOGICAL')
    f = Variable(name='f', dimensions=(x, y))
    g = Variable(name='g', dimensions=(x, y), type=booltype)
    h = Array(name='h', dimensions=(x, y), type=booltype)

    # This can causes all kinds of trouble in sympy-land
    # if we were to use plain Arrays instead of BoolArrays.
    expr = g & (f > 1) | h
    indexed = indexify(expr)
    code = fsymgen(expr)

    assert code == 'h(x,y) .or. g(x,y) .and. f(x,y) > 1'


def test_range_index():
    """
    Ensure proper sybolic behaviour for array ranges, like variable
    substitution on bounds.
    """
    from loki import RangeIndex

    i = Variable(name='i')
    j = Variable(name='j')
    k = Variable(name='k')

    assert str(RangeIndex(i)) == 'i'
    assert str(RangeIndex(lower=i)) == 'i:'
    assert str(RangeIndex(step=i)) == '::i'

    idx = RangeIndex(lower=i, upper=j, step=k)
    assert str(idx) == 'i:j:k'
    assert str(idx.xreplace({i: k})) == 'k:j:k'
