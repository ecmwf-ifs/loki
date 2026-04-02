# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for source-level model handling.
"""

import pytest

from loki import Module, Sourcefile, config, config_override
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.usefixtures('reset_frontend_mode')
def test_frontend_strict_mode(frontend, tmp_path):
    """
    Verify that frontends fail on unsupported features if strict mode is enabled
    """
    # Parameterized derived types currently not implemented
    fcode = """
module frontend_strict_mode
    implicit none
    TYPE matrix ( k, b )
      INTEGER,     KIND :: k = 4
      INTEGER (8), LEN  :: b
      REAL (k)          :: element (b,b)
    END TYPE matrix
end module frontend_strict_mode
    """
    config['frontend-strict-mode'] = True
    with pytest.raises(NotImplementedError):
        Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    config['frontend-strict-mode'] = False
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert 'matrix' in module.symbol_attrs
    assert 'matrix' in module.typedef_map


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_main_program(frontend):
    """
    Loki can't handle PROGRAM blocks and the frontends should throw an exception
    """
    fcode = """
program hello
    print *, "Hello World!"
end program
    """.strip()

    with config_override({'frontend-strict-mode': True}):
        with pytest.raises(NotImplementedError):
            Sourcefile.from_source(fcode, frontend=frontend)

    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert source.ir.body == ()


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_source_lineno(frontend):
    """
    ...
    """
    fcode = """
    subroutine driver
        call kernel()
        call kernel()
        call kernel()
    end subroutine driver
    """

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['driver']
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert calls[0] != calls[1]
    assert calls[1] != calls[2]
    assert calls[0].source.lines[0] < calls[1].source.lines[0] < calls[2].source.lines[0]


@pytest.mark.parametrize(
    'frontend',
    available_frontends(include_regex=True, xfail=[(OMNI, 'OMNI may segfault on empty files')])
)
@pytest.mark.parametrize('fcode', ['', '\n', '\n\n\n\n'])
def test_frontend_empty_file(frontend, fcode):
    """Ensure that all frontends can handle empty source files correctly (#186)"""
    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert isinstance(source.ir, ir.Section)
    assert not source.to_fortran().strip()
