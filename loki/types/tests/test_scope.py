# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A collection of tests for :any:`SymbolAttrs`, :any:`SymbolTable` and :any:`Scope`.
"""

import pytest

from loki import Module
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.types import (
    INTEGER, REAL, BasicType, DerivedType, ProcedureType, Scope, SymbolAttributes
)


def test_symbol_attributes():
    """
    Tests the attachment, lookup and deletion of arbitrary attributes from
    :any:`SymbolAttributes`
    """
    _type = SymbolAttributes('integer', a='a', b=True, c=None)
    assert _type.dtype == INTEGER
    assert _type.a == 'a'
    assert _type.b
    assert _type.c is None
    assert _type.foofoo is None

    _type.foofoo = 'bar'
    assert _type.foofoo == 'bar'

    delattr(_type, 'foofoo')
    assert _type.foofoo is None

    _type.b = None
    assert _type.b is None


def test_symbol_attributes_compare():
    """
    Test dedicated `type.compare` methods that allows certain
    attributes to be excluded from comparison.
    """
    someint = SymbolAttributes('integer', a='a', b=True, c=None)
    another = SymbolAttributes('integer', a='a', b=False, c=None)
    somereal = SymbolAttributes('real', a='a', b=True, c=None)

    assert not someint.compare(another)
    assert not another.compare(someint)
    assert someint.compare(another, ignore='b')
    assert another.compare(someint, ignore=['b'])
    assert not someint.compare(somereal)


def test_scope_setter():
    """ Test basic declaration and update behaviour of :any:`Scope` """
    scope = Scope()

    # Check basic type declaration
    scope.declare('a', dtype='integer', kind=4, intent='in')
    assert 'a' in scope.symbol_attrs
    assert scope.symbol_attrs['a'].dtype == INTEGER
    assert scope.symbol_attrs['a'].kind == 4
    assert scope.symbol_attrs['a'].intent == 'in'

    # Test erroneous and intentional re-declaration
    with pytest.raises(ValueError):
        scope.declare('a', dtype='real', kind=8)

    scope.declare('a', dtype='real', kind=8, fail=False)
    assert 'a' in scope.symbol_attrs
    assert scope.symbol_attrs['a'].dtype == REAL
    assert scope.symbol_attrs['a'].kind == 8
    assert not scope.symbol_attrs['a'].intent  # Wiped previous value

    # Check type declaration updates
    scope.update('a', dtype='integer', intent='inout')
    assert 'a' in scope.symbol_attrs
    assert scope.symbol_attrs['a'].dtype == INTEGER
    assert scope.symbol_attrs['a'].kind == 8  # Previous not wiped
    assert scope.symbol_attrs['a'].intent == 'inout'

    with pytest.raises(ValueError):
        scope.update('b', dtype='integer', intent='inout')

    # Override fail-safe, acts as another `declare()` call
    scope.update('b', dtype='integer', intent='inout', fail=False)
    assert 'b' in scope.symbol_attrs
    assert scope.symbol_attrs['b'].dtype == INTEGER
    assert scope.symbol_attrs['b'].intent == 'inout'


def test_scope_getter():
    """ Test basic :method:`get_type`/:method:`get_dtype` behaviour of :any:`Scope` """
    parent = Scope()
    scope = Scope(parent=parent)

    scope.declare('a', dtype='real', kind=8, intent='inout')
    parent.declare('b', dtype='integer', kind=4, intent='in')

    assert scope.get_type('a').dtype == REAL
    assert scope.get_type('a').kind == 8
    assert scope.get_type('a').intent == 'inout'

    # Non-recursive and recursive lookups through parent
    with pytest.raises(KeyError):
        scope.get_type('b', recursive=False)

    assert scope.get_type('b', recursive=False, fail=False) is None


def test_scope_get_variable():
    """
    Test the class dispatch and recursive lookup of :method:`get_variable`
    of :any:`Scope`
    """
    parent = Scope()
    scope = Scope(parent=parent)

    scope.declare('a', dtype='real', kind=8)
    scope.declare('b', dtype='integer', shape=(5,))
    scope.declare('f', dtype=ProcedureType('f', is_function=False))
    scope.declare('my_type', dtype=DerivedType('my_type'))
    scope.declare('ext', dtype=BasicType.DEFERRED)
    parent.declare('c', dtype='integer')
    symbol_attrs = dict(scope.symbol_attrs)

    a = scope.get_variable('a')
    assert isinstance(a, sym.Scalar)
    assert a == 'a'
    assert a.scope is scope
    assert a.type.dtype == REAL
    assert a.type.kind == 8

    b = scope.get_variable('b')
    assert isinstance(b, sym.Array)
    assert b.shape == (5,)
    assert not b.dimensions

    b_i = scope.get_variable('b', dimensions=(sym.IntLiteral(1),))
    assert isinstance(b_i, sym.Array)
    assert b_i == 'b(1)'

    f = scope.get_variable('f')
    assert isinstance(f, sym.ProcedureSymbol)
    assert not f.type.dtype.is_function

    my_type = scope.get_variable('my_type')
    assert isinstance(my_type, sym.DerivedTypeSymbol)

    ext = scope.get_variable('ext')
    assert isinstance(ext, sym.DeferredTypeSymbol)
    assert ext.type.dtype is BasicType.DEFERRED

    # Recursive lookup through the parent scope
    c = scope.get_variable('c')
    assert isinstance(c, sym.Scalar)
    assert c.scope is scope
    assert c.type.dtype == INTEGER
    with pytest.raises(KeyError):
        parent.get_variable('a')

    # Undeclared symbols fail, unless requested otherwise
    with pytest.raises(KeyError):
        scope.get_variable('d')
    d = scope.get_variable('d', fail=False)
    assert isinstance(d, sym.DeferredTypeSymbol)
    assert d.scope is scope
    assert d.type.dtype is BasicType.DEFERRED

    # None of the look-ups modify the symbol table
    assert dict(scope.symbol_attrs) == symbol_attrs
    assert 'd' not in scope.symbol_attrs
    assert 'd' not in parent.symbol_attrs


@pytest.mark.parametrize('frontend', available_frontends())
def test_scope_get_variable_derived_type_members(tmp_path, frontend):
    """
    Test the lookup of derived type members via :method:`get_variable`
    of :any:`Scope`
    """
    fcode = """
module scope_get_variable_mod
    implicit none
    type inner_type
        real :: x
        real, allocatable :: y(:)
    end type inner_type

    type outer_type
        type(inner_type) :: inner
        integer :: n
    end type outer_type
contains
    subroutine routine(outer)
        type(outer_type), intent(inout) :: outer
    end subroutine routine
end module scope_get_variable_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['routine']
    symbol_attrs = dict(routine.symbol_attrs)

    # Members via a given parent
    outer = routine.get_variable('outer')
    assert isinstance(outer, sym.Scalar)
    assert isinstance(outer.type.dtype, DerivedType)

    n = routine.get_variable('n', parent=outer)
    assert isinstance(n, sym.Scalar)
    assert n == 'outer%n'
    assert n.parent is outer
    assert n.type.dtype == INTEGER

    # Nested members with parents created from the qualified name
    x = routine.get_variable('outer%inner%x')
    assert isinstance(x, sym.Scalar)
    assert x == 'outer%inner%x'
    assert x.parent == 'outer%inner'
    assert x.parent.parent == 'outer'
    assert x.type.dtype == REAL

    y = routine.get_variable('outer%inner%y')
    assert isinstance(y, sym.Array)
    assert y == 'outer%inner%y'
    assert len(y.shape) == 1

    # Unknown members fail, unless requested otherwise
    with pytest.raises(KeyError):
        routine.get_variable('outer%z')
    z = routine.get_variable('outer%z', fail=False)
    assert isinstance(z, sym.DeferredTypeSymbol)
    assert z == 'outer%z'
    assert z.parent == 'outer'

    # None of the look-ups modify the symbol table
    assert dict(routine.symbol_attrs) == symbol_attrs


def test_scope_declare_variable():
    """
    Test the declaration of variables via :method:`declare_variable`
    of :any:`Scope`
    """
    scope = Scope()

    # Declaration via dtype and attributes
    a = scope.declare_variable('a', dtype='real', kind=8)
    assert isinstance(a, sym.Scalar)
    assert a.scope is scope
    assert a.type.dtype == REAL
    assert a.type.kind == 8
    assert scope.get_type('a').kind == 8

    # Declaration via type, with attribute overrides and array shape
    int_type = SymbolAttributes('integer', intent='in')
    b = scope.declare_variable('b', type=int_type, shape=(5,))
    assert isinstance(b, sym.Array)
    assert b.type.dtype == INTEGER
    assert b.type.intent == 'in'
    assert b.shape == (5,)
    assert not b.dimensions
    assert int_type.shape is None

    b_i = scope.declare_variable('b_i', type=int_type, dimensions=(sym.IntLiteral(1),))
    assert isinstance(b_i, sym.Array)
    assert b_i == 'b_i(1)'

    # Procedure and derived type symbols
    f = scope.declare_variable('f', dtype=ProcedureType('f', is_function=False))
    assert isinstance(f, sym.ProcedureSymbol)
    t = scope.declare_variable('t', dtype=DerivedType('my_type'))
    assert isinstance(t, sym.Scalar)
    assert isinstance(t.type.dtype, DerivedType)

    # Round trip with get_variable
    assert scope.get_variable('a') == a
    assert scope.get_variable('b').shape == (5,)

    # Re-declaration fails, unless requested otherwise
    with pytest.raises(ValueError):
        scope.declare_variable('a', dtype='integer')
    a = scope.declare_variable('a', dtype='integer', fail=False)
    assert isinstance(a, sym.Scalar)
    assert a.type.dtype == INTEGER
    assert a.type.kind is None

    # Either type or dtype must be given
    with pytest.raises(ValueError):
        scope.declare_variable('c')
    with pytest.raises(ValueError):
        scope.declare_variable('c', type=int_type, dtype='integer')
    assert 'c' not in scope.symbol_attrs

    # Declarations in a child scope do not affect the parent
    child = Scope(parent=scope)
    child.declare_variable('a', dtype='logical')
    assert child.get_type('a').dtype == BasicType.LOGICAL
    assert scope.get_type('a').dtype == INTEGER
