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

from loki import Module, Subroutine, Sourcefile, config_override
from loki.frontend import FP
from loki.ir import nodes as ir, FindNodes


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


@pytest.mark.parametrize('store_source', (True, False))
def test_fparser_source_parsing(store_source):
    fcode = """
module test_source_mod
  use my_kind_mod, only: akind
  implicit none

  type my_type
    real(kind=akind) :: scalar, vector(3)
    integer :: asize
  end type my_type

contains

  subroutine my_test_routine(n, rick, dave)
    integer, intent(in) :: n
    real(kind=akind), intent(inout) :: rick, dave(n)
    integer :: i

    do i=1, n
      if (dave(i) > 0.5) then
        dave(i) = dave(i) + rick
      end if
    end do

    forall(i=1:n)
      dave(i) = dave(i) + 2.0
    end forall
  end subroutine my_test_routine
end module test_source_mod
"""
    with config_override({'frontend-store-source': store_source}):
        source = Sourcefile.from_source(fcode, frontend=FP)
        module = source['test_source_mod']
        routine = module['my_test_routine']

    if store_source:
        assert routine.spec.source and routine.spec.source.lines == (14, 16)
        assert routine.body.source and routine.body.source.lines == (17, 26)
    else:
        assert not routine.spec.source
        assert not routine.body.source

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    loops = FindNodes(ir.Loop).visit(routine.body)
    conds = FindNodes(ir.Conditional).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    foralls = FindNodes(ir.Forall).visit(routine.body)
    assert len(decls) == 3 and len(loops) == 1 and len(conds) == 1
    assert len(assigns) == 2 and len(foralls) == 1

    if store_source:
        assert decls[0].source and decls[0].source.lines == (14, 14)
        assert decls[1].source and decls[1].source.lines == (15, 15)
        assert decls[2].source and decls[2].source.lines == (16, 16)
        assert loops[0].source and loops[0].source.lines == (18, 22)
        assert conds[0].source and conds[0].source.lines == (19, 21)
        assert assigns[0].source and assigns[0].source.lines == (20, 20)
        assert assigns[1].source and assigns[1].source.lines == (25, 25)
        assert foralls[0].source and foralls[0].source.lines == (24, 26)
    else:
        assert not decls[0].source and not decls[1].source and not decls[2].source
        assert not loops[0].source
        assert not conds[0].source
        assert not assigns[0].source

    imprts = FindNodes(ir.Import).visit(module.spec)
    intrs = FindNodes(ir.Intrinsic).visit(module.spec)
    tdefs = FindNodes(ir.TypeDef).visit(module.spec)
    assert len(imprts) == 1 and len(tdefs) == 1 and len(intrs) == 1
    tdecls = FindNodes(ir.VariableDeclaration).visit(tdefs[0].body)
    assert len(tdecls) == 2

    if store_source:
        assert imprts[0].source and imprts[0].source.lines == (3, 3)
        assert intrs[0].source and intrs[0].source.lines == (4, 4)
        assert tdefs[0].source and tdefs[0].source.lines == (6, 9)
        assert tdecls[0].source and tdecls[0].source.lines == (7, 7)
        assert tdecls[1].source and tdecls[1].source.lines == (8, 8)
    else:
        assert not imprts[0].source
        assert not intrs[0].source
        assert not tdefs[0].source
        assert not tdecls[0].source
        assert not tdecls[1].source


def test_fparser_sanitize_fypp_line_annotations():
    """
    Test that fypp line number annotations are sanitized correctly.
    """

    fcode = """
module some_templated_mod

# 1 "/path-to-hypp-macro/macro.hypp" 1
# 2 "/path-to-hypp-macro/macro.hypp"
# 3 "/path-to-hypp-macro/macro.hypp"
# 5 "/path-to-fypp-template/template.fypp" 2

integer :: a0
integer :: a1
integer :: a2
integer :: a3
integer :: a4

end module some_templated_mod
"""

    module = Module.from_source(fcode, frontend=FP)
    decls = FindNodes(ir.VariableDeclaration).visit(module.spec)

    assert len(decls) == 5
