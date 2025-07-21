# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.scope import Scope
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes


def test_variable_symbols():
    """ Test the default symbol creation and attributes of variable symbols """

    # pylint: disable=no-member

    scope = Scope()
    int_type = SymbolAttributes(BasicType.INTEGER, parameter=True, initial=42)
    real_type = SymbolAttributes(BasicType.REAL, kind='rick')
    deferred_type = SymbolAttributes(BasicType.DEFERRED)
    dtype = DerivedType(name='DerDieDas', typedef=BasicType.DEFERRED)
    dtype_type = SymbolAttributes(dtype)

    # A simple integer scalar parameter with inital value
    i = sym.Variable(name='i', type=int_type, scope=scope)
    assert isinstance(i, sym.Scalar) and i == 'i'
    assert isinstance(i.symbol, sym.VariableSymbol)
    assert i.type.parameter and i.initial == 42

    # A derived-type scalar variable
    r = sym.Variable(name='r', type=dtype_type)
    assert isinstance(r, sym.Scalar) and r == 'r'
    assert isinstance(r.symbol, sym.VariableSymbol)  # Why not DerivedTypeSymbol?
    assert isinstance(r.type.dtype, DerivedType) and r.type.dtype.name == 'DerDieDas'
    assert r.type.dtype.typedef == BasicType.DEFERRED

    # A scalar variable with an unknown type
    x = sym.Variable(name='x', type=deferred_type)
    assert isinstance(x, sym.DeferredTypeSymbol) and x == 'x'
    # For unknown-type, we don't create a meta-symbol - should we though?
    # assert isinstance(x.symbol, sym.VariableSymbol)

    # A simple array variable with a single dimension access
    a = sym.Variable(name='a', dimensions=(i,), type=real_type, scope=scope)
    assert isinstance(a, sym.Array) and a == 'a(i)'
    assert isinstance(a.symbol, sym.VariableSymbol)
    assert a.type.kind == 'rick' and a.dimensions == (i,)

    # Update the shape and create another instance of "a" that shares the shape
    assert not a.type.shape
    a.clone(type=a.type.clone(shape=(i,)))
    assert a.type.shape == ('i',)
    j = sym.Variable(name='j', type=int_type, scope=scope)
    a_j = sym.Variable(name='a', dimensions=(j,), scope=scope)
    assert a_j == 'a(j)' and a.shape == ('i',)

    # A scalar variable access on a derived-type parent variable
    b = sym.Variable(name='b', parent=r, type=real_type, scope=scope)
    assert isinstance(b, sym.Scalar) and b == 'r%b'
    assert isinstance(b.symbol, sym.VariableSymbol)
    assert b.parent == 'r' and b.parent.type.dtype.name == 'DerDieDas'
    assert b.type.kind == 'rick'

    # A two-dimensional array access on a derived-type parent variable
    c = sym.Variable(name='c', dimensions=(i, i), parent=r, type=real_type, scope=scope)
    assert isinstance(c, sym.Array) and c == 'r%c(i, i)'
    assert isinstance(c.symbol, sym.VariableSymbol)
    assert c.parent == 'r' and c.parent.type.dtype.name == 'DerDieDas'
    assert c.type.kind == 'rick' and c.dimensions == ('i', 'i')


def test_symbol_recreation():
    """ 
    Test the correct construction and re-construction of our symbol objects.
    """
    scope = Scope()
    int_type = SymbolAttributes(BasicType.INTEGER, parameter=True)
    real_type = SymbolAttributes(BasicType.REAL, kind='rick')
    log_type = SymbolAttributes(BasicType.LOGICAL)
    proc_type = SymbolAttributes(
        ProcedureType(name='f', is_function=True, return_type=real_type)
    )

    i = sym.Scalar(name='i', type=int_type, scope=scope)
    a = sym.Array(name='a', type=real_type, scope=scope)
    b = sym.Variable(
        name='b', dimensions=(i,), type=int_type, scope=scope
    )
    t = sym.Scalar(name='t', type=log_type, scope=scope)
    f = sym.ProcedureSymbol(name='f', type=proc_type, scope=scope)

    # Basic variables and symbols
    exprs = [i, a, b, t, f]

    # Literals
    exprs.append( sym.FloatLiteral(66.6) )
    exprs.append( sym.IntLiteral(42) )
    exprs.append( sym.LogicLiteral(True) )
    exprs.append( sym.StringLiteral('Dave') )
    exprs.append( sym.LiteralList(
        values=(sym.Literal(1), sym.IntLiteral(2)), dtype=int_type
    ) )

    # Operations
    exprs.append( sym.Sum((b, a)) )  # b(i) + a
    exprs.append( sym.Product((b, a)) )  # b(i) +* a
    exprs.append( sym.Sum((b, sym.Product((-1, a)))))  # b(i) - a
    exprs.append( sym.Quotient(numerator=b, denominator=a) )  # b(i) / a

    exprs.append( sym.Comparison(b, '==', a) )  # b(i) == a
    exprs.append( sym.LogicalNot(t) )
    exprs.append( sym.LogicalAnd((t, sym.LogicalNot(t))) )
    exprs.append( sym.LogicalOr((t, sym.LogicalNot(t))) )

    # Slightly special symbol types
    exprs.append( sym.InlineCall(function=f, parameters=(a, b)) )
    exprs.append( sym.Range((sym.IntLiteral(1), i)) )
    exprs.append( sym.LoopRange((sym.IntLiteral(1), i)) )
    exprs.append( sym.RangeIndex((sym.IntLiteral(1), i)) )

    exprs.append( sym.Cast(name='int', expression=b, kind=i) )
    exprs.append( sym.Reference(expression=b) )
    exprs.append( sym.Dereference(expression=b) )

    for expr in exprs:
        # Check that Pymbolic-style re-generation works for all
        # TODO: Should we introduce a Mixin "Cloneable" to makes these sane?
        cargs = dict(zip(expr.init_arg_names, expr.__getinitargs__()))
        clone = type(expr)(**cargs)
        assert clone == expr
        assert clone is not expr

        if isinstance(expr, sym.TypedSymbol):
            # Check that TypedSymbols replicate scope via .clone()
            scoped_clone = expr.clone()
            assert scoped_clone == expr
            assert scoped_clone is not expr
            assert scoped_clone.scope is expr.scope
