# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# A set of tests for the symbol accessort and management API built into `ScopedNode`.

import pytest

from loki import Module
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.expression import symbols as sym
from loki.types import BasicType


@pytest.mark.parametrize('frontend', available_frontends())
def test_scoped_node_get_symbols(frontend, tmp_path):
    """ Test :method:`get_symbol` functionality on scoped nodes. """
    fcode = """
module test_scoped_node_symbols_mod
implicit none
integer, parameter :: jprb = 8

contains
  subroutine test_scoped_node_symbols(n, a, b, c)
    integer, intent(in) :: n
    real(kind=jprb), intent(inout) :: a(n), b(n), c
    integer :: i

    a(1) = 42.0_jprb

    associate(d => a)
    do i=1, n
      b(i) = a(i) + c
    end do
    end associate
  end subroutine test_scoped_node_symbols
end module test_scoped_node_symbols_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['test_scoped_node_symbols']
    associate = FindNodes(ir.Associate).visit(routine.body)[0]

    # Check symbol lookup from subroutine
    assert routine.get_symbol('a') == 'a(n)'
    assert routine.get_symbol('a').scope == routine
    assert routine.get_symbol('b') == 'b(n)'
    assert routine.get_symbol('b').scope == routine
    assert routine.get_symbol('c') == 'c'
    assert routine.get_symbol('c').scope == routine    
    assert routine.get_symbol('jprb') == 'jprb'
    assert routine.get_symbol('jprb').scope == module
    assert routine.get_symbol('jprb').initial == 8

    # Check passthrough from the Associate (ScopedNode)
    assert associate.get_symbol('a') == 'a(n)'
    assert associate.get_symbol('a').scope == routine
    assert associate.get_symbol('b') == 'b(n)'
    assert associate.get_symbol('b').scope == routine
    assert associate.get_symbol('c') == 'c'
    assert associate.get_symbol('c').scope == routine
    assert associate.get_symbol('d') == 'd'
    assert associate.get_symbol('d').scope == associate
    assert associate.get_symbol('jprb') == 'jprb'
    assert associate.get_symbol('jprb').scope == module
    assert associate.get_symbol('jprb').initial == 8


@pytest.mark.parametrize('frontend', available_frontends())
def test_scoped_node_variable_constructor(frontend, tmp_path):
    """ Test :any:`Variable` constrcutore on scoped nodes. """
    fcode = """
module test_scoped_nodes_mod
implicit none
integer, parameter :: jprb = 8

contains
  subroutine test_scoped_nodes(n, a, b, c)
    integer, intent(in) :: n
    real(kind=jprb), intent(inout) :: a(n), b(n), c
    integer :: i

    a(1) = 42.0_jprb

    associate(d => a)
    do i=1, n
      b(i) = a(i) + c
    end do
    end associate
  end subroutine test_scoped_nodes
end module test_scoped_nodes_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['test_scoped_nodes']
    associate = FindNodes(ir.Associate).visit(routine.body)[0]

    # Build some symbiols and check their type
    c = routine.Variable(name='c')
    assert c.type.dtype == BasicType.REAL
    assert c.type.kind in ('jprb', 8)
    i = routine.Variable(name='i')
    assert i.type.dtype == BasicType.INTEGER
    jprb = routine.Variable(name='jprb')
    assert jprb.type.dtype == BasicType.INTEGER
    assert jprb.type.initial == 8

    a_i = routine.Variable(name='a', dimensions=(i,))
    assert a_i == 'a(i)'
    assert isinstance(a_i, sym.Array)
    assert a_i.dimensions == (i,)
    assert a_i.type.dtype == BasicType.REAL
    assert a_i.type.kind in ('jprb', 8)

    # Build another, but from the associate node
    b_i = associate.Variable(name='b', dimensions=(i,))
    assert b_i == 'b(i)'
    assert isinstance(b_i, sym.Array)
    assert b_i.dimensions == (i,)
    assert b_i.type.dtype == BasicType.REAL
    assert b_i.type.kind in ('jprb', 8)


@pytest.mark.parametrize('frontend', available_frontends())
def test_scoped_node_parse_expr(frontend, tmp_path):
    """ Test :any:`Variable` constrcutore on scoped nodes. """
    fcode = """
module test_scoped_nodes_mod
implicit none
integer, parameter :: jprb = 8

contains
  subroutine test_scoped_nodes(n, a, b, c)
    integer, intent(in) :: n
    real(kind=jprb), intent(inout) :: a(n), b(n), c
    integer :: i

    a(1) = 42.0_jprb

    associate(d => a)
    do i=1, n
      b(i) = a(i) + c
    end do
    end associate
  end subroutine test_scoped_nodes
end module test_scoped_nodes_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['test_scoped_nodes']
    associate = FindNodes(ir.Associate).visit(routine.body)[0]

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert assigns[0].lhs == 'a(1)'
    assert assigns[1].rhs == 'a(i) + c'

    # Check that all variables are identified
    ai_c = routine.parse_expr('a(i)   +   c')
    assert ai_c == assigns[1].rhs
    assert isinstance(ai_c, sym.Sum)
    assert isinstance(ai_c.children[0], sym.Array)
    assert isinstance(ai_c.children[0].dimensions[0], sym.Scalar)
    assert isinstance(ai_c.children[1], sym.Scalar)
    assert ai_c.children[0].scope == routine
    assert ai_c.children[0].dimensions[0].scope == routine
    assert ai_c.children[1].scope == routine

    # Check that k is deferred
    ai_k = routine.parse_expr('a(i) + k')
    assert isinstance(ai_k, sym.Sum)
    assert isinstance(ai_k.children[0], sym.Array)
    assert isinstance(ai_k.children[0].dimensions[0], sym.Scalar)
    assert isinstance(ai_k.children[1], sym.DeferredTypeSymbol)
    assert ai_c.children[0].scope == routine
    assert ai_c.children[0].dimensions[0].scope == routine
    assert ai_c.children[1].scope == routine

    # Check that all variables are identified
    ai_c = associate.parse_expr('a(i)   +   c')
    assert ai_c == assigns[1].rhs
    assert isinstance(ai_c, sym.Sum)
    assert isinstance(ai_c.children[0], sym.Array)
    assert isinstance(ai_c.children[0].dimensions[0], sym.Scalar)
    assert isinstance(ai_c.children[1], sym.Scalar)
    assert ai_c.children[0].scope == routine
    assert ai_c.children[0].dimensions[0].scope == routine
    assert ai_c.children[1].scope == routine

    # Check that k is deferred
    ai_k = associate.parse_expr('a(i) + k')
    assert isinstance(ai_k, sym.Sum)
    assert isinstance(ai_k.children[0], sym.Array)
    assert isinstance(ai_k.children[0].dimensions[0], sym.Scalar)
    assert isinstance(ai_k.children[1], sym.DeferredTypeSymbol)
    assert ai_c.children[0].scope == routine
    assert ai_c.children[0].dimensions[0].scope == routine
    assert ai_c.children[1].scope == routine
