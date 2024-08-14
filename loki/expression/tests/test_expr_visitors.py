# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Sourcefile
from loki.expression import FindVariables, FindTypedSymbols
from loki.frontend import available_frontends


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_finder_retrieval_function(frontend, tmp_path):
    """
    Verify that expression finder visitors work as intended and remain
    functional if re-used
    """
    fcode = """
module some_mod
    implicit none
contains
    function some_func() result(ret)
        integer :: ret
        ret = 1
    end function some_func

    subroutine other_routine
        integer :: var, tmp
        var = 5 + some_func()
    end subroutine other_routine
end module some_mod
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    expected_ts = {'var', 'some_func'}
    expected_vars = ('var',)

    # Instantiate the first expression finder and make sure it works as expected
    find_ts = FindTypedSymbols()
    assert find_ts.visit(source['other_routine'].body) == expected_ts

    # Verify that it works also on a repeated invocation
    assert find_ts.visit(source['other_routine'].body) == expected_ts

    # Instantiate the second expression finder and make sure it works as expected
    find_vars = FindVariables(unique=False)
    assert find_vars.visit(source['other_routine'].body) == expected_vars

    # Make sure the first expression finder still works
    assert find_ts.visit(source['other_routine'].body) == expected_ts
