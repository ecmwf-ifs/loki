# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Specific test battery for the OMNI parser frontend.
"""

import pytest

from loki import Module, Subroutine
from loki.expression import FindVariables
from loki.frontend import OMNI, HAVE_OMNI
from loki.ir import nodes as ir, FindNodes


@pytest.mark.skipif(not HAVE_OMNI, reason='Test tequires OMNI frontend.')
def test_derived_type_definitions(tmp_path):
    """ Test correct parsing of derived type declarations. """
    fcode = """
module omni_derived_type_mod
  type explicit
      real(kind=8) :: scalar, vector(3), matrix(3, 3)
  end type explicit

  type deferred
      real(kind=8), allocatable :: scalar, vector(:), matrix(:, :)
  end type deferred

  type ranged
      real(kind=8) :: scalar, vector(1:3), matrix(0:3, 0:3)
  end type ranged
end module omni_derived_type_mod
"""
    # Parse the source and validate the IR
    module = Module.from_source(fcode, frontend=OMNI, xmods=[tmp_path])

    assert len(module.typedefs) == 3
    explicit_symbols = FindVariables(unique=False).visit(module['explicit'].body)
    assert explicit_symbols == ('scalar', 'vector(3)', 'matrix(3, 3)')

    deferred_symbols = FindVariables(unique=False).visit(module['deferred'].body)
    assert deferred_symbols == ('scalar', 'vector(:)', 'matrix(:, :)')

    ranged_symbols = FindVariables(unique=False).visit(module['ranged'].body)
    assert ranged_symbols == ('scalar', 'vector(3)', 'matrix(0:3, 0:3)')


@pytest.mark.skipif(not HAVE_OMNI, reason='Test tequires OMNI frontend.')
def test_array_dimensions(tmp_path):
    """ Test correct parsing of derived type declarations. """
    fcode = """
subroutine omni_array_indexing(n, a, b)
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a(3), b(n)
  real(kind=8) :: c(n, n)
  real(kind=8) :: d(1:n, 0:n)

  a(:) = 11.
  b(1:n) = 42.
  c(2:n, 0:n) = 66.
  d(:, 0:n) = 68.
end subroutine omni_array_indexing
"""
    # Parse the source and validate the IR
    routine = Subroutine.from_source(fcode, frontend=OMNI, xmods=[tmp_path])

    # OMNI separate declarations per variable
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    assert len(decls) == 5
    assert decls[0].symbols == ('n',)
    assert decls[1].symbols == ('a(3)',)
    assert decls[2].symbols == ('b(n)',)
    assert decls[3].symbols == ('c(n, n)',)
    assert decls[4].symbols == ('d(n, 0:n)',)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].lhs == 'a(:)'
    assert assigns[1].lhs == 'b(1:n)'
    assert assigns[2].lhs == 'c(2:n, 0:n)'
    assert assigns[3].lhs == 'd(:, 0:n)'
