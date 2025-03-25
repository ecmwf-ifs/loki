# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour with regards to source parsing and sanitisation.
"""

import pytest

from loki import Module, Subroutine, Sourcefile
from loki.frontend import FP


@pytest.mark.parametrize('from_file', (True, False))
@pytest.mark.parametrize('preprocess', (True, False))
def test_source_sanitize_fp_source(tmp_path, from_file, preprocess):
    """
    Test that source sanitizing works as expected and postprocessing
    rules are correctly applied
    """
    fcode = """
subroutine some_routine(input_path)
    implicit none
    character(len=255), intent(in) :: input_path
    integer :: ios, fu
    write(*,*) "we print CPP value ", MY_VAR
    ! In the following line the PP definition should be replace by '0'
    ! or the actual line number
    write(*,*) "We are in line ",__LINE__
    open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
end subroutine some_routine
""".strip()

    if from_file:
        filepath = tmp_path/'some_routine.F90'
        filepath.write_text(fcode)
        obj = Sourcefile.from_file(filepath, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))
    else:
        obj = Sourcefile.from_source(fcode, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))

    if preprocess:
        # CPP takes care of that
        assert '"We are in line ", 8' in obj.to_fortran()
        assert '"we print CPP value ", 5' in obj.to_fortran()
    else:
        # source sanitisation takes care of that
        assert '"We are in line ", 0' in obj.to_fortran()
        assert '"we print CPP value ", MY_VAR' in obj.to_fortran()

    assert 'newunit=fu' in obj.to_fortran()


@pytest.mark.parametrize('preprocess', (True, False))
def test_source_sanitize_fp_subroutine(preprocess):
    """
    Test that source sanitizing works as expected and postprocessing
    rules are correctly applied
    """
    fcode = """
subroutine some_routine(input_path)
    implicit none
    character(len=255), intent(in) :: input_path
    integer :: ios, fu
    write(*,*) "we print CPP value ", MY_VAR
    ! In the following line the PP definition should be replace by '0'
    ! or the actual line number
    write(*,*) "We are in line ",__LINE__
    open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
end subroutine some_routine
""".strip()

    obj = Subroutine.from_source(fcode, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))

    if preprocess:
        # CPP takes care of that
        assert '"We are in line ", 8' in obj.to_fortran()
        assert '"we print CPP value ", 5' in obj.to_fortran()
    else:
        # source sanitisation takes care of that
        assert '"We are in line ", 0' in obj.to_fortran()
        assert '"we print CPP value ", MY_VAR' in obj.to_fortran()

    assert 'newunit=fu' in obj.to_fortran()


@pytest.mark.parametrize('preprocess', (True, False))
def test_source_sanitize_fp_module(preprocess):
    """
    Test that source sanitizing works as expected and postprocessing
    rules are correctly applied
    """
    fcode = """
module some_mod
    implicit none
    integer line = __LINE__ + MY_VAR
contains
subroutine some_routine(input_path)
    implicit none
    character(len=255), intent(in) :: input_path
    integer :: ios, fu
    write(*,*) "we print CPP value ", MY_VAR
    ! In the following line the PP definition should be replace by '0'
    ! or the actual line number
    write(*,*) "We are in line ",__LINE__
    open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
end subroutine some_routine
end module some_mod
""".strip()

    obj = Module.from_source(fcode, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))

    if preprocess:
        # CPP takes care of that
        assert 'line = 3 + 5' in obj.to_fortran()
        assert '"We are in line ", 12' in obj.to_fortran()
        assert '"we print CPP value ", 5' in obj.to_fortran()
    else:
        # source sanitisation takes care of that
        assert 'line = 0 + MY_VAR' in obj.to_fortran()
        assert '"We are in line ", 0' in obj.to_fortran()
        assert '"we print CPP value ", MY_VAR' in obj.to_fortran()

    assert 'newunit=fu' in obj.to_fortran()


# TODO: Add tests for source sanitizer with other frontends
