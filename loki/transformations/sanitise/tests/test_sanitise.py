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

from loki.transformations.sanitise import SanitiseTransformation


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
