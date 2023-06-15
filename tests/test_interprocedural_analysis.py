# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Various tests for interprocedural analysis features in Loki
"""

import pytest

from conftest import available_frontends
from loki import (
    Sourcefile, FindNodes, FindInlineCalls,
    CallStatement, IntLiteral
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_ipa_call_statement_arg_iter(frontend):
    """
    Test that :any:`CallStatement.arg_iter` works as expected
    """
    fcode_caller = """
subroutine caller
    use some_mod, only: callee
    implicit none
    integer :: arg1, arg3(10)
    real :: arg2
    call callee(arg1, arg2, arg3, 4)
end subroutine caller
    """.strip()

    fcode_callee = """
module some_mod
    implicit none
contains
    subroutine callee(var, VAR2, arr, val)
        integer, intent(inout) :: var
        real, intent(in) :: var2
        integer, intent(in) :: arr(:)
        integer, intent(in) :: val
    end subroutine callee
end module some_mod
    """.strip()

    callee_source = Sourcefile.from_source(fcode_callee, frontend=frontend)
    caller_source = Sourcefile.from_source(fcode_caller, frontend=frontend, definitions=callee_source.definitions)

    callee = callee_source['callee']
    caller = caller_source['caller']

    calls = FindNodes(CallStatement).visit(caller.body)
    assert len(calls) == 1
    arg_iter = list(calls[0].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('arr(:)', 'arg3'), ('val', '4')
    ]

    for kernel_arg, caller_arg in calls[0].arg_iter():
        assert kernel_arg.scope is callee
        assert isinstance(caller_arg, IntLiteral) or caller_arg.scope is caller


@pytest.mark.parametrize('frontend', available_frontends())
def test_ipa_call_statement_arg_iter_optional(frontend):
    """
    Test that :any:`CallStatement.arg_iter` works as expected with optional arguments
    """
    fcode_caller = """
subroutine caller
    use some_mod, only: callee
    implicit none
    integer :: arg1, arg3(10)
    real :: arg2
    call callee(arg1, arg2, arg3, 4)
    call callee(arg1, arg2, VAL=4, arr=arg3, opt2=1)
    call callee(arg1, arg2, arg3, 4, 1, 2)
end subroutine caller
    """.strip()

    fcode_callee = """
module some_mod
    implicit none
contains
    subroutine callee(var, VAR2, arr, val, OPT1, opt2)
        integer, intent(inout) :: var
        real, intent(in) :: var2
        integer, intent(in) :: arr(:)
        integer, intent(in) :: val
        integer, intent(in), optional :: OPT1, opt2
    end subroutine callee
end module some_mod
    """.strip()

    callee_source = Sourcefile.from_source(fcode_callee, frontend=frontend)
    caller_source = Sourcefile.from_source(fcode_caller, frontend=frontend, definitions=callee_source.definitions)

    callee = callee_source['callee']
    caller = caller_source['caller']

    calls = FindNodes(CallStatement).visit(caller.body)
    assert len(calls) == 3
    arg_iter = list(calls[0].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('arr(:)', 'arg3'), ('val', '4')
    ]
    arg_iter = list(calls[1].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('val', '4'), ('arr(:)', 'arg3'), ('opt2', '1')
    ]
    arg_iter = list(calls[2].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('arr(:)', 'arg3'), ('val', '4'), ('opt1', '1'), ('opt2', '2')
    ]

    for call in calls:
        for kernel_arg, caller_arg in call.arg_iter():
            assert kernel_arg.scope is callee
            assert isinstance(caller_arg, IntLiteral) or caller_arg.scope is caller


@pytest.mark.parametrize('frontend', available_frontends())
def test_ipa_inline_call_arg_iter(frontend):
    """
    Test that :any:`InlineCall.arg_iter` works as expected
    """
    fcode_caller = """
subroutine caller
    use some_mod, only: callee
    implicit none
    integer :: arg1, arg3(10)
    real :: arg2, ret
    ret = callee(arg1, arg2, arg3, 4)
end subroutine caller
    """.strip()

    fcode_callee = """
module some_mod
    implicit none
contains
    function callee(var, VAR2, arr, val)
        integer, intent(inout) :: var
        real, intent(in) :: var2
        integer, intent(in) :: arr(:)
        integer, intent(in) :: val
        real :: callee
    end function callee
end module some_mod
    """.strip()

    callee_source = Sourcefile.from_source(fcode_callee, frontend=frontend)
    caller_source = Sourcefile.from_source(fcode_caller, frontend=frontend, definitions=callee_source.definitions)

    callee = callee_source['callee']
    caller = caller_source['caller']

    calls = list(FindInlineCalls().visit(caller.body))
    assert len(calls) == 1
    arg_iter = list(calls[0].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('arr(:)', 'arg3'), ('val', '4')
    ]

    for kernel_arg, caller_arg in calls[0].arg_iter():
        assert kernel_arg.scope is callee
        assert isinstance(caller_arg, IntLiteral) or caller_arg.scope is caller


@pytest.mark.parametrize('frontend', available_frontends())
def test_ipa_inline_call_arg_iter_optional(frontend):
    """
    Test that :any:`InlineCall.arg_iter` works as expected
    """
    fcode_caller = """
subroutine caller
    use some_mod, only: callee
    implicit none
    integer :: arg1, arg3(10)
    real :: arg2, ret
    ret = callee(arg1, arg2, arg3, 4)
    ret = ret + callee(arg1, ARR=arg3, var2=arg2, opt2=2, val=4)
    ret = ret + callee(arg1, arg2, arg3, 4, 1, 2)
end subroutine caller
    """.strip()

    fcode_callee = """
module some_mod
    implicit none
contains
    function callee(var, VAR2, arr, val, opt1, OPT2)
        integer, intent(inout) :: var
        real, intent(in) :: var2
        integer, intent(in) :: arr(:)
        integer, intent(in) :: val
        integer, intent(in) :: opt1, OPT2
        real :: callee
    end function callee
end module some_mod
    """.strip()

    callee_source = Sourcefile.from_source(fcode_callee, frontend=frontend)
    caller_source = Sourcefile.from_source(fcode_caller, frontend=frontend, definitions=callee_source.definitions)

    callee = callee_source['callee']
    caller = caller_source['caller']

    calls = list(FindInlineCalls(unique=False).visit(caller.body))
    assert len(calls) == 3
    arg_iter = list(calls[0].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('arr(:)', 'arg3'), ('val', '4')
    ]
    arg_iter = list(calls[1].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('arr(:)', 'arg3'), ('var2', 'arg2'), ('opt2', '2'), ('val', '4')
    ]
    arg_iter = list(calls[2].arg_iter())
    assert arg_iter == [
        ('var', 'arg1'), ('var2', 'arg2'), ('arr(:)', 'arg3'), ('val', '4'), ('opt1', '1'), ('opt2', '2')
    ]

    for call in calls:
        for kernel_arg, caller_arg in call.arg_iter():
            assert kernel_arg.scope is callee
            assert isinstance(caller_arg, IntLiteral) or caller_arg.scope is caller
