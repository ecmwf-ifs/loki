# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import FindNodes, Subroutine
from loki.expression import IntLiteral, symbols as sym
from loki.frontend import available_frontends
from loki.ir import nodes as ir
from loki.transformations import ConstantPropagationTransformer


def test_constant_propagation_transformer_subtree_visit():
    assignment = ir.Assignment(lhs=sym.Variable(name='b'), rhs=sym.Variable(name='a'))
    transformed = ConstantPropagationTransformer().visit(
        assignment, constants_map={('a', ()): IntLiteral(4)}
    )

    assert transformed._constants_map is None


def test_constant_propagation_transformer_export():
    assert ConstantPropagationTransformer is not None


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_transformer_routine_keeps_structure(frontend):
    fcode = """
subroutine const_prop_transform(a)
  integer, intent(out) :: a
  integer :: b = 1
  a = b
end subroutine const_prop_transform
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    transformed = ConstantPropagationTransformer().visit(routine)
    assignments = FindNodes(ir.Assignment).visit(transformed.body)

    assert len(assignments) == 1
    assert assignments[0]._constants_map is None
