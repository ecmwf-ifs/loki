# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from conftest import available_frontends
from loki import (
    Subroutine, FortranCTransformation, cgen, cppgen, cudagen,
    CallStatement, FindNodes
)
from loki.build import Builder
from loki.transform import normalize_range_indexing
from loki.expression import symbols as sym

@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    return Builder(source_dirs=here, build_dir=here/'build')


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_c_like_gen_intrinsic_type(here, frontend, use_c_ptr):
    """
    A simple test routine to test C transpilation of loops
    """

    fcode = """
subroutine c_like_gen_intrinsic_type(len, arr_in, arr_inout)
  integer, intent(in) :: len
  real, intent(in) :: arr_in(len, len)
  real, intent(inout) :: arr_inout(len, len)
  integer, parameter :: param = 4

  call some_func(len, arr_inout)

end subroutine c_like_gen_intrinsic_type
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine) # Fix OMNI nonsense

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=here)
    print("\nC    -------------------------------")
    print(cgen(routine, header=True))
    print('----')
    print(cgen(routine, header=True, guards=True))
    # print('----')
    # print(cgen(routine, extern=True))
    print("\nCPP  -------------------------------")
    print(cppgen(routine))
    print('----')
    print(cppgen(routine, extern=True))
    print("\nCUDA -------------------------------")
    print(cudagen(routine))
    print('----')
    calls = FindNodes(CallStatement).visit(routine.body)
    for call in calls:
        call._update(chevron=(sym.IntLiteral(1), sym.IntLiteral(1)))
    print(cudagen(routine))
