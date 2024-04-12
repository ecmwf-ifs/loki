# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from loki import Sourcefile, fexprgen
from loki.frontend import available_frontends, OMNI


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Loki annotations break frontend parser')]))
def test_nested_types(frontend):
    """
    Regression test that ensures that nested types are correctly
    propagated through manual construction from source files.
    """
    here = Path(__file__).parent

    # First, get the sub_type and check that the dimension annotation is honoured
    subtypes = Sourcefile.from_file(here/'sub_types.f90', frontend=frontend)['sub_types']
    child = subtypes.typedef_map['sub_type']
    assert fexprgen(child.variables[0].shape) == '(size,)'

    # Check that dimension in sub_type has propagated to parent_type
    types = Sourcefile.from_file(here/'types.f90', definitions=subtypes,
                                 frontend=frontend)['types']
    parent = types.typedef_map['parent_type']
    x = parent.variables[1].variable_map['x']
    assert fexprgen(x.shape) == '(size,)'

    # Ensure that the driver has the correct shape info for pt%type_member%x
    driver = Sourcefile.from_file(here/'driver.f90', definitions=types, frontend=frontend)['driver']
    pt_d = driver.routines[0].variables[0]
    x_d = pt_d.variable_map['type_member'].variable_map['x']
    assert fexprgen(x_d.shape) == '(size,)'

    kernel = Sourcefile.from_file(here/'kernel.f90', definitions=types, frontend=frontend)['kernel']
    pt_k = kernel.routines[0].variables[1]
    x_k = pt_k.variable_map['type_member'].variable_map['x']
    assert fexprgen(x_k.shape) == '(size,)'
