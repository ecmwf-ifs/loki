# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest

from loki.frontend import available_frontends, OMNI
from loki import Sourcefile

from transformations.parallel_routine_dispatch import ParallelRoutineDispatchTransformation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_parallel_regions(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    assert transformation.dummy_return_value == ['dispatch_routine']
