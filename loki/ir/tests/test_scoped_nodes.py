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
