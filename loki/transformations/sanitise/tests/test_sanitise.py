# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes

from loki.transformations.sanitise import (
    SanitiseTransformation, SanitisePipeline
)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('resolve_associate', [True, False])
@pytest.mark.parametrize('resolve_sequence', [True, False])
def test_transformation_sanitise(frontend, resolve_associate, resolve_sequence, tmp_path):
    """
    Test that the selective dispatch of the sanitisations works.
    """

    fcode = """
module test_transformation_sanitise_mod
  implicit none

  type rick
    real :: scalar
  end type rick
contains

  subroutine test_transformation_sanitise(a, dave)
    real, intent(inout) :: a(3)
    type(rick), intent(inout) :: dave

    associate(scalar => dave%scalar)
      scalar = a(1) + a(2)

      call vadd(a(1), 2.0, 3)
    end associate

  contains
    subroutine vadd(x, y, n)
      real, intent(inout) :: x(n)
      real, intent(inout) :: y
      integer, intent(in) :: n

      x = x + 2.0
    end subroutine vadd
  end subroutine test_transformation_sanitise
end module test_transformation_sanitise_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['test_transformation_sanitise']

    assoc = FindNodes(ir.Associate).visit(routine.body)
    assert len(assoc) == 1
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments[0] == 'a(1)'

    trafo = SanitiseTransformation(
        resolve_associate_mappings=resolve_associate,
        resolve_sequence_association=resolve_sequence,
    )
    trafo.apply(routine)

    assoc = FindNodes(ir.Associate).visit(routine.body)
    assert len(assoc) == 0 if resolve_associate else 1

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments[0] == 'a(1:3)' if resolve_sequence else 'a(1)'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('substitute_expressions', [True, False])
@pytest.mark.parametrize('resolve_associates', [True, False])
@pytest.mark.parametrize('resolve_sequence_associations', [True, False])
def test_sanitise_pipeline(
        tmp_path, frontend, substitute_expressions,
        resolve_associates, resolve_sequence_associations
):
    """
    Test the agglomerated :any:`SanitisePipeline` with different settings.
    """
    fcode = """
module test_sanitise_pipeline_mod
  implicit none

  type rick
    real :: scalar
  end type rick
contains

  subroutine test_pipeline_sanitise(n, a, dave)
    integer, intent(in) :: n
    real, intent(inout) :: a(n+1)
    type(rick), intent(inout) :: dave

    associate(scalar => dave%scalar)
      scalar = a(1) + a(2)

      call vadd(a(1), 2.0, n+1)
    end associate

  contains
    subroutine vadd(x, y, n)
      real, intent(inout) :: x(n)
      real, intent(inout) :: y
      integer, intent(in) :: n

      x = x + 2.0
    end subroutine vadd
  end subroutine test_pipeline_sanitise
end module test_sanitise_pipeline_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['test_pipeline_sanitise']

    assoc = FindNodes(ir.Associate).visit(routine.body)
    assert len(assoc) == 1
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1

    pipeline = SanitisePipeline(
        substitute_expressions=substitute_expressions,
        expression_map={'n + 1': 'n'},
        resolve_associates=resolve_associates,
        resolve_sequence_associations=resolve_sequence_associations
    )
    pipeline.apply(routine)

    assoc = FindNodes(ir.Associate).visit(routine.body)
    assert len(assoc) == 0 if resolve_associates else 1

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    if resolve_sequence_associations:
        assert calls[0].arguments[0] == 'a(1:n)' if substitute_expressions else 'a(1:n+1)'
    else:
        assert calls[0].arguments[0] == 'a(1)'
