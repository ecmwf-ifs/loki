# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.scope import Scope
from loki.types import BasicType, ProcedureType, SymbolAttributes


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
