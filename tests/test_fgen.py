# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends

from loki import Subroutine, fgen

@pytest.mark.parametrize('frontend', available_frontends())
def test_fgen_literal_list_linebreak(frontend):
    """
    Test correct handling of linebreaks for LiteralList expression nodes
    """
    fcode = """
subroutine literal_list_linebreak
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    call config_gas_optics_sw_spectral_def_allocate_bands_only( &
         &  [2600.0_jprb, 3250.0_jprb, 4000.0_jprb, 4650.0_jprb, 5150.0_jprb, 6150.0_jprb, 7700.0_jprb, &
         &   8050.0_jprb, 12850.0_jprb, 16000.0_jprb, 22650.0_jprb, 29000.0_jprb, 38000.0_jprb, 820.0_jprb], &
         &  [3250.0_jprb, 4000.0_jprb, 4650.0_jprb, 5150.0_jprb, 6150.0_jprb, 7700.0_jprb, 8050.0_jprb, &
         &   12850.0_jprb, 16000.0_jprb, 22650.0_jprb, 29000.0_jprb, 38000.0_jprb, 50000.0_jprb, 2600.0_jprb])
end subroutine literal_list_linebreak
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    body_code = fgen(routine.body)
    assert body_code.count(',') == 27
    assert body_code.count('(/') == 2
    assert body_code.count('/)') == 2
    body_lines = body_code.splitlines()
    assert all(len(line) < 132 for line in body_lines)
