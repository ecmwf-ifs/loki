# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A collection of tests for :any:`SymbolAttrs`, :any:`SymbolTable` and :any:`Scope`.
"""

import pytest

from loki.types import INTEGER, REAL, Scope, SymbolAttributes


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
