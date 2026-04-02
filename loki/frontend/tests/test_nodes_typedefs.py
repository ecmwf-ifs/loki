# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for derived type definitions.
"""

import pytest

from loki import Module
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes, FindVariables


@pytest.mark.parametrize('frontend', available_frontends())
def test_typedef_components(frontend, tmp_path):
    """
    Test parsing of derived type component declarations.
    """
    fcode = """
module derived_type_mod
  implicit none
  type explicit
      real(kind=8) :: scalar, vector(3), matrix(3, 3)
  end type explicit

  type deferred
      real(kind=8), allocatable :: scalar, vector(:), matrix(:, :)
  end type deferred

  type ranged
      real(kind=8) :: scalar, vector(1:3), matrix(0:3, 0:3)
  end type ranged
end module derived_type_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert len(module.typedefs) == 3

    explicit_symbols = FindVariables(unique=False).visit(module['explicit'].body)
    assert explicit_symbols == ('scalar', 'vector(3)', 'matrix(3, 3)')

    deferred_symbols = FindVariables(unique=False).visit(module['deferred'].body)
    assert deferred_symbols == ('scalar', 'vector(:)', 'matrix(:, :)')

    ranged_symbols = FindVariables(unique=False).visit(module['ranged'].body)
    assert ranged_symbols[0] == 'scalar'
    assert ranged_symbols[1] in ('vector(3)', 'vector(1:3)')
    assert ranged_symbols[2] == 'matrix(0:3, 0:3)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_typedef_nested_components(frontend, tmp_path):
    """
    Test nested derived type definitions and component typing.
    """
    fcode = """
module nested_type_mod
  implicit none
  type leaf_type
    integer :: value
  end type leaf_type

  type inner_type
    type(leaf_type) :: leaf
    real :: coords(2)
  end type inner_type

  type outer_type
    type(inner_type) :: inner
    integer :: flags(3)
  end type outer_type
end module nested_type_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert len(module.typedefs) == 3
    outer = module['outer_type']
    decls = FindNodes(ir.VariableDeclaration).visit(outer.body)
    assert len(decls) == 2
    assert decls[0].symbols == ('inner',)
    assert decls[1].symbols == ('flags(3)',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_typedef_component_shapes(frontend, tmp_path):
    """
    Test component shape handling inside derived type definitions.
    """
    fcode = """
module typedef_shape_mod
  implicit none
  type grid_type
    integer :: nx, ny
    real :: field(4, 5)
    real, allocatable :: work(:)
  end type grid_type
end module typedef_shape_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    grid_type = module['grid_type']
    variable_map = grid_type.variable_map
    assert variable_map['field'].shape == ('4', '5')
    assert variable_map['work'].shape == (':',)
