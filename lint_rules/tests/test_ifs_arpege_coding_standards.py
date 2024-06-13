# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import pytest

from conftest import run_linter, available_frontends
from loki import Sourcefile
from loki.lint import DefaultHandler


pytestmark = pytest.mark.skipif(not available_frontends(),
                                reason='Suitable frontend not available')


@pytest.fixture(scope='module', name='rules')
def fixture_rules():
    rules = importlib.import_module('lint_rules.ifs_arpege_coding_standards')
    return rules


@pytest.mark.parametrize('frontend', available_frontends())
def test_implicit_none(rules, frontend):
    fcode = """
subroutine routine_okay
implicit none
integer :: a
a = 5
contains
subroutine contained_routine_not_okay
! This should report
integer :: b
b = 5
end subroutine contained_routine_not_okay
end subroutine routine_okay

module mod_okay
implicit none
contains
subroutine contained_mod_routine_okay
integer :: a
a = 5
contains
subroutine contained_mod_routine_contained_routine_okay
integer :: b
b = 2
end subroutine contained_mod_routine_contained_routine_okay
end subroutine contained_mod_routine_okay
end module mod_okay

subroutine routine_not_okay
! This should report
integer :: a
a = 5
contains
subroutine contained_not_okay_routine_okay
implicit none
integer :: b
b = 5
end subroutine contained_not_okay_routine_okay
end subroutine routine_not_okay

module mod_not_okay
contains
subroutine contained_mod_not_okay_routine_okay
implicit none
integer :: a
a = 5
end subroutine contained_mod_not_okay_routine_okay
end module mod_not_okay

subroutine routine_also_not_okay
! This should report
integer :: a
a = 5
contains
subroutine contained_routine_not_okay
! This should report
integer :: b
b = 5
end subroutine contained_routine_not_okay
end subroutine routine_also_not_okay

module mod_also_not_okay
contains
subroutine contained_mod_routine_not_okay
! This should report
integer :: a
a = 5
contains
subroutine contained_contained_routine_not_okay
! This should report
integer :: b
b = 5
end subroutine contained_contained_routine_not_okay
end subroutine contained_mod_routine_not_okay
end module mod_also_not_okay
    """
    source = Sourcefile.from_source(fcode, frontend=frontend)
    messages = []
    handler = DefaultHandler(target=messages.append)
    run_linter(source, [rules.MissingImplicitNoneRule], handlers=[handler])

    expected_messages = (
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'mod_not_okay', '(l. 40)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'mod_also_not_okay', '(l. 61)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'contained_mod_routine_not_okay', '(l. 63)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'contained_contained_routine_not_okay', '(l. 68)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'contained_routine_not_okay', '(l. 7)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'routine_not_okay', '(l. 28)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'routine_also_not_okay', '(l. 49)']),
        (['[L1]', 'MissingImplicitNoneRule', '`IMPLICIT NONE`', 'contained_routine_not_okay', '(l. 54)']),
    )

    assert len(messages) == len(expected_messages)
    for msg, keywords in zip(messages, expected_messages):
        for keyword in keywords:
            assert keyword in msg


@pytest.mark.parametrize('frontend', available_frontends())
def test_only_param_global_var_rule(rules, frontend):
    fcode = """
module some_mod
use other_mod, only: some_type
implicit none

integer, parameter :: param_ok = 123
integer, parameter :: arr_param_ok(:) = (/ 1, 2, 3 /)
integer :: var_not_ok
integer, allocatable :: arr_not_ok(:), other_arr_not_ok(:,:)
integer, pointer :: ptr_not_ok
real, parameter :: rparam_ok = -42.
type(some_type) :: dt_var_not_ok
type(some_type) :: dt_arr_not_ok(2)
end module some_mod
    """
    source = Sourcefile.from_source(fcode, frontend=frontend)
    messages = []
    handler = DefaultHandler(target=messages.append)
    run_linter(source, [rules.OnlyParameterGlobalVarRule], handlers=[handler])

    expected_messages = (
        (['L3', 'OnlyParameterGlobalVarRule', 'var_not_ok', '(l. 8)']),
        (['L3', 'OnlyParameterGlobalVarRule', 'arr_not_ok', 'other_arr_not_ok', '(l. 9)']),
        (['L3', 'OnlyParameterGlobalVarRule', 'ptr_not_ok', '(l. 10)']),
        (['L3', 'OnlyParameterGlobalVarRule', 'dt_var_not_ok', '(l. 12)']),
        (['L3', 'OnlyParameterGlobalVarRule', 'dt_arr_not_ok', '(l. 13)']),
    )

    assert len(messages) == len(expected_messages)
    for msg, keywords in zip(messages, expected_messages):
        for keyword in keywords:
            assert keyword in msg


def test_missing_intfb_rule_subroutine(rules):
    fcode = """
subroutine missing_intfb_rule(a, b, dt)
    use some_mod, only: imported_routine
    use other_mod, only: imported_func
    use type_mod, only: imported_type
    implicit none
    integer, intent(in) :: a, b
    type(imported_type), intent(in) :: dt
#include "included_routine.intfb.h"
    integer :: local_var
    interface
        subroutine local_intf_routine(X)
            integer, intent(in) :: x
        end subroutine local_intf_routine
    end interface
#include "included_func.intfb.h"
#include "other_inc_func.func.h"

    CALL IMPORTED_ROUTINE(A)
    CALL INCLUDED_ROUTINE(B)
    CALL MISSING_ROUTINE(A, B)
    CALL LOCAL_INTF_ROUTINE(A)
    CALL DT%PROC(A+B)
    LOCAL_VAR = IMPORTED_FUNC(A)
    LOCAL_VAR = LOCAL_VAR + MIN(INCLUDED_FUNC(B), 1)
    LOCAL_VAR = LOCAL_VAR + MISSING_FUNC(A, B)
    LOCAL_VAR = LOCAL_VAR + DT%FUNC(A+B)
    LOCAL_VAR = LOCAL_VAR + OTHER_INC_FUNC(A, 'STR VAL')
    LOCAL_VAR = LOCAL_VAR + MISSING_INC_FUNC(A, 'STR VAL')
end subroutine missing_intfb_rule
""".strip()
    source = Sourcefile.from_source(fcode)
    messages = []
    handler = DefaultHandler(target=messages.append)
    run_linter(source, [rules.MissingIntfbRule], handlers=[handler])

    expected_messages = (
        (['[L9]', 'MissingIntfbRule', '`missing_routine`', '(l. 20)']),
        # (['[L9]', 'MissingIntfbRule', 'MISSING_FUNC', '(l. 25)']),
        (['[L9]', 'MissingIntfbRule', '`missing_inc_func`', '(l. 28)'])
        # NB:
        #     - The `missing_func` is not discovered because it is syntactically
        #       indistinguishable from an Array subscript
        #     - The `missing_inc_func` has a string argument and can therefore be
        #       identified as an inline call by fparser
        #     - Calls to type-bound procedures are not reported
    )

    assert len(messages) == len(expected_messages)
    for msg, keywords in zip(messages, expected_messages):
        for keyword in keywords:
            assert keyword in msg


def test_missing_intfb_rule_module(rules):
    fcode = """
module missing_intfb_rule_mod
    use external_mod, only: module_imported_routine, module_imported_func
    implicit none
    interface
        function local_intf_func()
            integer local_intf_func
        end function local_intf_func
    end interface
#include "included_parent.intfb.h"
contains
    subroutine missing_intfb_rule(a, b)
        use some_mod, only: imported_routine
        use other_mod, only: imported_func
        implicit none
        integer, intent(in) :: a, b
#include "included_routine.intfb.h"
        integer :: local_var
#include "included_func.intfb.h"


        CALL IMPORTED_ROUTINE(A)
        CALL INCLUDED_ROUTINE(B)
        CALL MODULE_IMPORTED_ROUTINE(A, B)
        CALL MISSING_ROUTINE(A, B)
        CALL INCLUDED_PARENT(A)
        call missing_routine(a, b)
        LOCAL_VAR = IMPORTED_FUNC(A)
        LOCAL_VAR = LOCAL_VAR + INCLUDED_FUNC(B)
        LOCAL_VAR = LOCAL_VAR + MISSING_FUNC(A, KEY=B)
        LOCAL_VAR = LOCAL_VAR + MAX(MODULE_IMPORTED_FUNC(KEY=A), -1)
        LOCAL_VAR = LOCAL_VAR + LOCAL_INTF_FUNC()
    end subroutine missing_intfb_rule
end module missing_intfb_rule_mod
""".strip()
    source = Sourcefile.from_source(fcode)
    messages = []
    handler = DefaultHandler(target=messages.append)
    run_linter(source, [rules.MissingIntfbRule], handlers=[handler])

    expected_messages = (
        (['[L9]', 'MissingIntfbRule', '`missing_routine`', '(l. 24)']),
        (['[L9]', 'MissingIntfbRule', '`missing_func`', '(l. 29)']),
        # NB:
        #   - The missing function is discovered here because
        #     the use of a keyword-argument makes it syntactically
        #     clear to be an inline call rather than an Array subscript
        #   - MISSING_ROUTINE is only imported once for the first occurence
        #   - We are not reporting the intrinsic Fortran routine MAX
    )

    assert len(messages) == len(expected_messages)
    for msg, keywords in zip(messages, expected_messages):
        for keyword in keywords:
            assert keyword in msg
