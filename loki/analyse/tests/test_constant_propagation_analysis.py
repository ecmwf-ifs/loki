# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import FindNodes, Subroutine
from loki.analyse import ConstantPropagationAnalysis
from loki.expression import FloatLiteral, IntLiteral, LogicLiteral, StringLiteral
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import nodes as ir


def test_constant_propagation_analysis_declarations_map():
    fcode = """
subroutine const_prop_decls
  integer :: a = 1
  integer :: b(2) = (/2, 3/)
  logical :: l = .true.
end subroutine const_prop_decls
    """.strip()
    routine = Subroutine.from_source(fcode)

    declarations_map = ConstantPropagationAnalysis().generate_declarations_map(routine)

    assert declarations_map[('a', ())] == IntLiteral(1)
    assert declarations_map[('l', ())] == LogicLiteral(True)
    assert declarations_map[('b', (IntLiteral(1),))] == IntLiteral(2)
    assert declarations_map[('b', (IntLiteral(2),))] == IntLiteral(3)


def test_constant_propagation_mapper_folds_expressions():
    mapper = ConstantPropagationAnalysis.ConstPropMapper()

    assert mapper(sym.Sum((IntLiteral(1), IntLiteral(2)))) == IntLiteral(3)
    assert mapper(sym.Quotient(IntLiteral(7), IntLiteral(2))) == IntLiteral(3)
    assert mapper(sym.Power(IntLiteral(2), IntLiteral(3))) == IntLiteral(8)
    assert mapper(sym.LogicalAnd((LogicLiteral(True), LogicLiteral(False)))) == LogicLiteral(False)
    assert mapper(sym.StringConcat((StringLiteral('foo'), StringLiteral('bar')))) == StringLiteral('foobar')
    assert mapper(sym.Sum((FloatLiteral('1.5'), FloatLiteral('2.5')))) == FloatLiteral('4.0')


def test_constant_propagation_mapper_short_circuits_boolean_ops():
    mapper = ConstantPropagationAnalysis.ConstPropMapper()
    dyn = sym.Variable(name='dyn')

    assert mapper(sym.LogicalOr((LogicLiteral(True), dyn))) == LogicLiteral(True)
    assert mapper(sym.LogicalAnd((LogicLiteral(False), dyn))) == LogicLiteral(False)


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_analysis_attaches_maps(frontend):
    fcode = """
subroutine const_prop_attach
  integer :: a = 1
  integer :: b
  b = a
end subroutine const_prop_attach
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    analysis = ConstantPropagationAnalysis()
    analysis.attach_dataflow_analysis(routine)

    assert assignments[0]._constants_map == {('a', ()): IntLiteral(1)}

    analysis.detach_dataflow_analysis(routine)
    assert assignments[0]._constants_map is None
