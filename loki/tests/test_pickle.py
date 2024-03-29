# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A set of tests that ensures that serialisation/deserialisation via
pickle works and creates equivalent objects of various types.
"""
from pathlib import Path
from pickle import dumps, loads
import pytest

from conftest import available_frontends

from loki import (
    Subroutine, Module, Sourcefile, SymbolAttributes, BasicType,
    Scope, AttachScopes, OMNI
)
from loki.expression import symbols
from loki.batch import Item


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def test_pickle_expression():
    """
    Ensure pickle-replication of Pymbolic-backed expressions.
    """
    # pylint: disable=no-member

    # Ensure basic variable components are picklable
    t = SymbolAttributes(BasicType.INTEGER)
    v1 = symbols.Variable(name='v1', type=t)
    assert v1.symbol == loads(dumps(v1.symbol))
    assert v1.type == loads(dumps(v1.type))
    assert v1 == loads(dumps(v1))

    # Now we add a scope to the expression and replicate both
    scope = Scope()
    v2 = symbols.Variable(name='v2', scope=scope, type=t)
    scope_new = loads(dumps(scope))
    v2_new = loads(dumps(v2))

    # Re-attach the new expression to the new scope
    v2_new = AttachScopes().visit(v2_new, scope=scope_new)

    assert len(scope_new.symbol_attrs) == 1
    assert 'v2' in scope_new.symbol_attrs
    assert scope_new.symbol_attrs['v2'] == t
    assert v2_new == v2

    # And now, one more time but with arrays!
    scope = Scope()
    i = symbols.Variable(name='i', scope=scope, type=t)
    v3 = symbols.Variable(name='v3', dimensions=(i,), scope=scope, type=t)
    scope_new = loads(dumps(scope))
    v3_new = loads(dumps(v3))
    v3_new = AttachScopes().visit(v3_new, scope=scope_new)

    assert len(scope_new.symbol_attrs) == 2
    assert 'v3' in scope_new.symbol_attrs
    assert 'i' in scope_new.symbol_attrs
    assert scope_new.symbol_attrs['v3'] == t
    assert v3_new == v3

    # Check that Literals are trivial replicated
    i = symbols.IntLiteral(value=1., kind='jpim')
    assert loads(dumps(i)) == i


@pytest.mark.parametrize('frontend', available_frontends())
def test_pickle_subroutine(frontend):
    """
    Ensure that :any:`Subroutine` and its components are picklable.
    """

    fcode = """
subroutine my_routine(n, a, b, d)
  integer, intent(in) :: n
  real, intent(in) :: a(n), b(n)
  real, intent(out) :: d(n)
  integer :: i

  do i=1, n
    d(i) = a(i) + b(i)
  end do
end subroutine my_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # First, replicate the scope individually, ...
    scope_new = Scope()
    scope_new.symbol_attrs.update(loads(dumps(routine.symbol_attrs)))

    # Replicate spec and body independently...
    spec_new = loads(dumps(routine.spec))
    spec_new = AttachScopes().visit(spec_new, scope=scope_new)
    assert spec_new == routine.spec

    body_new = loads(dumps(routine.body))
    body_new = AttachScopes().visit(body_new, scope=scope_new)
    assert body_new == routine.body

    # Ensure equivalence after pickle-cyle
    assert routine == loads(dumps(routine))


@pytest.mark.parametrize('frontend', available_frontends())
def test_pickle_module(frontend):
    """
    Ensure that serialisation/deserialisation via pickling works as expected.
    """

    fcode = """
module my_type_mod

  real(8) :: a, b
  integer :: s

end module my_type_mod
"""
    module = Module.from_source(fcode, frontend=frontend)

    # Ensure equivalence after pickle-cyle
    assert module.symbol_attrs == loads(dumps(module.symbol_attrs))
    assert module.spec == loads(dumps(module.spec))
    assert module.contains == loads(dumps(module.contains))
    assert module == loads(dumps(module))


@pytest.mark.parametrize('frontend', available_frontends())
def test_pickle_module_with_typedef(frontend):
    """
    Ensure that a type definition in a module is pickle-safe.
    """

    fcode = """
module my_type_mod

  type a_type
    real(kind=8) :: scalar
    real(kind=8) :: vector(3)
  end type a_type

  type(a_type) :: some_numbers

end module my_type_mod
"""
    module = Module.from_source(fcode, frontend=frontend)

    # Replicate the TypeDef individually
    typedef = module['a_type']
    typedef_new = loads(dumps(typedef))
    assert typedef_new == typedef

    # Replicate the scope individually
    scope_new = Scope()
    scope_new.symbol_attrs.update(loads(dumps(module.symbol_attrs)))

    # Replicate the spec independently...
    spec_new = loads(dumps(module.spec))
    spec_new = AttachScopes().visit(spec_new, scope=scope_new)
    assert spec_new == module.spec

    # Replicate the member type
    contains_new = loads(dumps(module.contains))
    contains_new = AttachScopes().visit(contains_new, scope=scope_new)
    assert contains_new == module.contains

    # Ensure equivalence after pickle-cyle
    assert module.symbol_attrs == loads(dumps(module.symbol_attrs))
    assert module.spec == loads(dumps(module.spec))
    assert module.contains == loads(dumps(module.contains))
    assert module == loads(dumps(module))


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'No external module available')]))
def test_pickle_subroutine_with_member(frontend):
    """
    Ensure that :any:`Subroutine` and its components are picklable.
    """

    fcode = """
subroutine my_routine(n, a, b, d)
  use another_module, only: some_routine

  integer, intent(in) :: n
  real, intent(in) :: a(n), b(n)
  real, intent(out) :: d(n)
  integer :: i

  call member_routine(a, b)

  contains

  subroutine member_routine(n, a, b)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)

  end subroutine member_routine
end subroutine my_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # First, replicate the scope individually, ...
    scope_new = Scope()
    scope_new.symbol_attrs.update(loads(dumps(routine.symbol_attrs)))

    # Replicate spec and body independently...
    spec_new = loads(dumps(routine.spec))
    spec_new = AttachScopes().visit(spec_new, scope=scope_new)
    assert spec_new == routine.spec

    body_new = loads(dumps(routine.body))
    body_new = AttachScopes().visit(body_new, scope=scope_new)
    assert body_new == routine.body

    # Replicate the member routine
    contains_new = loads(dumps(routine.contains))
    body_new = AttachScopes().visit(contains_new, scope=scope_new)
    assert contains_new == routine.contains

    # Ensure equivalence after pickle-cyle with scope-level replication
    routine_new = loads(dumps(routine))
    assert routine_new.spec == routine.spec
    assert routine_new.body == routine.body
    assert routine_new.contains == routine.contains
    assert routine_new.symbol_attrs == routine.symbol_attrs
    assert routine_new == routine


@pytest.mark.parametrize('frontend', available_frontends())
def test_pickle_module_with_routines(frontend):
    """
    Ensure that :any:`Module` object with cross-calling subroutines
    pickle cleanly, including the procedure type symbols.
    """

    fcode = """
module my_module
  implicit none

  contains
  subroutine my_routine(n, a, b, d)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)
    real, intent(out) :: d(n)
    integer :: i

    call other_routine(a, b)
  end subroutine my_routine

  subroutine other_routine(n, a, b)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)

  end subroutine other_routine
end module my_module
"""
    module = Module.from_source(fcode, frontend=frontend)

    # First, replicate the scope individually, ...
    scope_new = Scope()
    scope_new.symbol_attrs.update(loads(dumps(module.symbol_attrs)))

    # Replicate spec and body independently...
    spec_new = loads(dumps(module.spec))
    spec_new = AttachScopes().visit(spec_new, scope=scope_new)
    assert spec_new == module.spec

    contains_new = loads(dumps(module.contains))
    # We need to attach the parent here first, so that the deferred
    # procedure type symbol in the call can be resolved
    contains_new.body[1]._reset_parent(scope_new)
    contains_new.body[-1]._reset_parent(scope_new)
    contains_new = AttachScopes().visit(contains_new, scope=scope_new)
    assert contains_new == module.contains

    # Ensure equivalence after pickle-cyle with scope-level replication
    module_new = loads(dumps(module))
    assert module_new.spec == module.spec
    assert module_new.contains == module.contains
    assert module_new.symbol_attrs == module.symbol_attrs
    assert module_new == module


@pytest.mark.parametrize('frontend', available_frontends())
def test_pickle_scheduler_item(here, frontend):
    """
    Test that :any:`Item` objects are picklable, so that we may use
    them with parallel processes.
    """
    filepath = here/'sources/sourcefile_item.f90'
    source = Sourcefile.from_file(filename=filepath, frontend=frontend)
    item_a = Item(name='#routine_a', source=source)

    # Check the individual routines and modules in the parsed source file
    for node in item_a.source.ir.body:
        assert loads(dumps(node)) == node

    assert loads(dumps(item_a.source.ir)) == item_a.source.ir
    assert loads(dumps(item_a.source)) == item_a.source
    assert loads(dumps(item_a)) == item_a
