# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A collection of tests for :any:`SymbolAttrs`, :any:`SymbolTable` and :any:`Scope`.
"""

from copy import deepcopy
import pickle
import pytest

from loki import Module, Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import FindInlineCalls, FindNodes, nodes as ir
from loki.types import (
    INTEGER, REAL, BasicType, DerivedType, ProcedureType, Scope, SymbolAttributes,
    GENERIC_INTRINSIC_FUNCTIONS, GENERIC_INTRINSIC_SUBROUTINES, INTRINSIC_SCOPE
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

    # Member enumeration resolves types without modifying the symbol table
    inner = outer.variable_map['inner']
    assert inner.scope is routine
    assert inner.variable_map['x'].type.dtype == REAL
    assert isinstance(inner.variable_map['y'], sym.Array)
    assert dict(routine.symbol_attrs) == symbol_attrs

    # Explicit entries for members in the scope take precedence
    routine.symbol_attrs['outer%inner%x'] = SymbolAttributes(REAL, shape=(sym.IntLiteral(3),))
    assert isinstance(routine.get_variable('outer%inner%x'), sym.Array)
    assert isinstance(outer.variable_map['inner'].variable_map['x'], sym.Array)
    assert outer.variable_map['inner'].variable_map['x'].type.shape == (3,)


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


def test_intrinsic_scope():
    """
    Test the read-only :any:`IntrinsicScope` of generic intrinsic procedures
    """
    table = INTRINSIC_SCOPE.symbol_attrs
    assert len(table) == len(GENERIC_INTRINSIC_FUNCTIONS) + len(GENERIC_INTRINSIC_SUBROUTINES)
    assert 'min' in table and 'MIN' in table
    assert 'dsqrt' not in table

    min_type = table.lookup('min')
    assert isinstance(min_type.dtype, ProcedureType)
    assert min_type.is_intrinsic and min_type.dtype.is_intrinsic
    assert min_type.dtype.is_function
    assert not table.lookup('random_number').dtype.is_function

    # Look-ups hand out copies that do not modify the table
    min_type.foo = 'bar'
    assert table.lookup('min').foo is None

    # The table is read-only
    with pytest.raises(TypeError):
        table['min'] = SymbolAttributes(BasicType.REAL)
    with pytest.raises(TypeError):
        del table['min']
    with pytest.raises(TypeError):
        table.update({'foo': SymbolAttributes(BasicType.REAL)})
    with pytest.raises(TypeError):
        table.setdefault('foo')
    with pytest.raises(TypeError):
        table.clear()
    with pytest.raises(TypeError):
        INTRINSIC_SCOPE.declare_variable('foo', dtype=BasicType.REAL)
    assert 'foo' not in table

    # The scope and its table are singletons
    assert pickle.loads(pickle.dumps(INTRINSIC_SCOPE)) is INTRINSIC_SCOPE
    assert pickle.loads(pickle.dumps(table)) is table
    assert deepcopy(INTRINSIC_SCOPE) is INTRINSIC_SCOPE


def test_scope_intrinsic_lookup():
    """
    Test the look-up of generic intrinsic procedures from any :any:`Scope`
    """
    parent = Scope()
    scope = Scope(parent=parent)

    # Intrinsic procedures are found without table entries
    assert scope.get_type('min').dtype.is_intrinsic
    assert scope.get_symbol_scope('min') is INTRINSIC_SCOPE
    assert scope.get_type('min', recursive=False, fail=False) is None
    size = scope.get_variable('size')
    assert isinstance(size, sym.ProcedureSymbol)
    assert size.scope is scope
    assert size.type.is_intrinsic
    assert not scope.symbol_attrs and not parent.symbol_attrs

    # The deprecated factory does not declare intrinsic procedures either
    size = sym.Variable(name='size', scope=scope)
    assert isinstance(size, sym.ProcedureSymbol)
    assert not scope.symbol_attrs

    # Declarations in user scopes shadow intrinsic procedures
    parent.declare('sum', dtype='real')
    assert scope.get_symbol_scope('sum') is parent
    assert isinstance(scope.get_variable('sum'), sym.Scalar)
    assert scope.get_type('sum').dtype == REAL

    # Unknown names are still undeclared
    with pytest.raises(KeyError):
        scope.get_type('dsqrt')
    assert scope.get_symbol_scope('dsqrt') is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_intrinsic_scope(tmp_path, frontend):
    """
    Test that frontends bind generic intrinsic procedures to the
    :any:`IntrinsicScope` without declaring them
    """
    fcode = """
module frontend_intrinsic_mod
    implicit none
    type geom_type
        integer :: dim
    end type geom_type
contains
    subroutine frontend_intrinsic(a, n, geom)
        integer, intent(in) :: n
        real(kind=8), intent(inout) :: a(n)
        type(geom_type), intent(in) :: geom
        real(kind=8) :: sum
        sum = min(a(1), maxval(a)) + size(a) + dsqrt(a(1)) + geom%dim
        call random_number(a)
        a(1) = sum
    end subroutine frontend_intrinsic
end module frontend_intrinsic_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['frontend_intrinsic']

    # Only user symbols and specific intrinsics are declared in the routine
    for name in ('min', 'maxval', 'size', 'random_number'):
        assert name not in routine.symbol_attrs
        assert name not in module.symbol_attrs
    assert 'dsqrt' in routine.symbol_attrs

    calls = {str(c.function).lower(): c.function for c in FindInlineCalls().visit(routine.body)}
    assert set(calls) == {'min', 'maxval', 'size', 'dsqrt'}
    for name in ('min', 'maxval', 'size'):
        assert isinstance(calls[name], sym.ProcedureSymbol)
        assert calls[name].scope is INTRINSIC_SCOPE
        assert calls[name].type.is_intrinsic
    assert calls['dsqrt'].scope is routine
    assert calls['dsqrt'].type.is_intrinsic

    call = FindNodes(ir.CallStatement).visit(routine.body)[0]
    assert call.name == 'random_number'
    assert call.name.scope is INTRINSIC_SCOPE
    assert not call.name.type.dtype.is_function

    # The local variable shadows the intrinsic, and the member is not an intrinsic
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert isinstance(assigns[0].lhs, sym.Scalar)
    assert assigns[0].lhs.scope is routine
    assert assigns[0].lhs.type.dtype == REAL
    member = [v for v in assigns[0].rhs.children if v == 'geom%dim'][0]
    assert isinstance(member, sym.Scalar)
    assert member.scope is routine
    assert member.type.dtype == INTEGER

    # Unchanged output
    assert routine.to_fortran().lower().count('min(a(1), maxval(a))') == 1


def test_intrinsic_scope_subroutine_pickle(tmp_path):
    """
    Test that intrinsic procedures are rebound after pickling a routine
    """
    fcode = """
subroutine intrinsic_pickle(a, n)
    integer, intent(in) :: n
    real, intent(inout) :: a(n)
    a(1) = max(a(1), sum(a))
end subroutine intrinsic_pickle
"""
    routine = Subroutine.from_source(fcode, xmods=[tmp_path])
    other = pickle.loads(pickle.dumps(routine))
    calls = FindInlineCalls().visit(other.body)
    assert len(calls) == 2
    assert all(c.function.scope is INTRINSIC_SCOPE for c in calls)
    assert all(c.function.type.is_intrinsic for c in calls)
    assert 'max' not in other.symbol_attrs
