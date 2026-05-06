# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for annotation-like IR nodes.
"""

import pytest

from loki import Module, Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, FP
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_pragma_vs_comment(frontend, tmp_path):
    """
    Make sure pragmas and comments are identified correctly
    """
    fcode = """
module frontend_pragma_vs_comment
    implicit none
!$some pragma
    integer :: var1
!!$some comment
    integer :: var2
!some comment
    integer :: var3
    !$some pragma
    integer :: var4
    ! !$some comment
    integer :: var5
end module frontend_pragma_vs_comment
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    pragmas = FindNodes(ir.Pragma).visit(module.ir)
    comments = FindNodes(ir.Comment).visit(module.ir)
    assert len(pragmas) == 2
    assert len(comments) == 3
    assert all(pragma.keyword == 'some' for pragma in pragmas)
    assert all(pragma.content == 'pragma' for pragma in pragmas)
    assert all('some comment' in comment.text for comment in comments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_pragma_line_continuation(frontend):
    """
    Test that multi-line pragmas are parsed and dealt with correctly.
    """
    fcode = """
SUBROUTINE TOTO(A,B)

IMPLICIT NONE
REAL, INTENT(IN) :: A
REAL, INTENT(INOUT) :: B

!$ACC PARALLEL LOOP GANG &
!$ACC& PRESENT(ZRDG_LCVQ,ZFLU_QSATS,ZRDG_CVGQ) &
!$ACC& PRIVATE (JBLK) &
!$ACC& VECTOR_LENGTH (YDCPG_OPTS%KLON)
!$ACC SEQUENTIAL

END SUBROUTINE TOTO
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    pragmas = FindNodes(ir.Pragma).visit(routine.body)

    assert len(pragmas) == 2
    assert pragmas[0].keyword == 'ACC'
    assert 'PARALLEL' in pragmas[0].content
    assert 'PRESENT' in pragmas[0].content
    assert 'PRIVATE' in pragmas[0].content
    assert 'VECTOR_LENGTH' in pragmas[0].content
    assert pragmas[1].content == 'SEQUENTIAL'

    # Check that source object was generated right
    assert pragmas[0].source
    assert pragmas[0].source.lines == (8, 8) if frontend == OMNI else (8, 11)
    assert pragmas[1].source
    assert pragmas[1].source.lines == (12, 12)


@pytest.mark.parametrize('frontend', available_frontends())
def test_comment_block_clustering(frontend):
    """
    Test that multiple :any:`Comment` nodes into a :any:`CommentBlock`.
    """
    fcode = """
subroutine test_comment_block(a, b)
  ! What is this?
  ! Ohhh, ... a docstring?
  real, intent(inout) :: a, b

  a = a + 1.0
  ! Never gonna
  b = b + 2
  ! give you
  ! up...

  a = a + b
  ! Shut up, ...
  ! Rick!
end subroutine test_comment_block
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    comments = FindNodes(ir.Comment).visit(routine.spec)
    assert len(comments) == 0
    blocks = FindNodes(ir.CommentBlock).visit(routine.spec)
    assert len(blocks) == 0

    assert isinstance(routine.docstring[0], ir.CommentBlock)
    assert len(routine.docstring[0].comments) == 2
    assert routine.docstring[0].comments[0].text == '! What is this?'
    assert routine.docstring[0].comments[1].text == '! Ohhh, ... a docstring?'

    comments = FindNodes(ir.Comment).visit(routine.body)
    assert len(comments) == 2 if frontend == FP else 1
    assert comments[-1].text == '! Never gonna'

    blocks = FindNodes(ir.CommentBlock).visit(routine.body)
    assert len(blocks) == 2
    assert len(blocks[0].comments) == 3 if frontend == FP else 2
    assert blocks[0].comments[0].text == '! give you'
    assert blocks[0].comments[1].text == '! up...'

    assert len(blocks[1].comments) == 2
    assert blocks[1].comments[0].text == '! Shut up, ...'
    assert blocks[1].comments[1].text == '! Rick!'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI strips comments during parse')]
))
def test_inline_comments(frontend):
    """
    Test that multiple :any:`Comment` nodes into a :any:`CommentBlock`.
    """
    fcode = """
subroutine test_inline_comments(a, b)
  real, intent(inout) :: a, b  ! We don't need no education
  real, external :: alien_func ! We don't need no thought control
  integer :: i

  a = a + 1.0
  ! Who said that?
  b = b + 2             ! All in all it's just another

  do i=1, 10
    b = b + 2           ! Brick in the ...
  enddo

  a = a + alien_func()  ! wall !
end subroutine test_inline_comments
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    assert len(decls) == 2
    assert decls[0].comment.text == "! We don't need no education"
    assert decls[1].comment is None

    proc_decls = FindNodes(ir.ProcedureDeclaration).visit(routine.spec)
    assert len(proc_decls) == 1
    assert proc_decls[0].comment.text == "! We don't need no thought control"

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].comment is None
    assert assigns[1].comment.text == "! All in all it's just another"
    assert assigns[2].comment.text == '! Brick in the ...'
    assert assigns[3].comment.text == '! wall !'

    comments = FindNodes(ir.Comment).visit(routine.body)
    assert len(comments) == 4
    assert comments[1].text == '! Who said that?'
    assert comments[0].text == comments[2].text == comments[3].text == ''


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI does not like Loki pragmas, yet!')]))
def test_routine_variables_dimension_pragmas(frontend):
    """
    Test that `!$loki dimension` pragmas can be used to override the
    conceptual `.shape` of local and argument variables.
    """
    fcode = """
subroutine routine_variables_dimensions(x, y, v0, v1, v2, v3, v4)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y

  !$loki dimension(10)
  real(kind=jprb), intent(inout) :: v0(:)
  !$loki dimension(x)
  real(kind=jprb), intent(inout) :: v1(:)
  !$loki dimension(x,y,:)
  real(kind=jprb), dimension(:,:,:), intent(inout) :: v2, v3
  !$loki dimension(x,y)
  real(kind=jprb), pointer, intent(inout) :: v4(:,:)
  !$loki dimension(x+y,2*x)
  real(kind=jprb), allocatable :: v5(:,:)
  !$loki dimension(x/2, x**2, (x+y)/x)
  real(kind=jprb), dimension(:, :, :), pointer :: v6

end subroutine routine_variables_dimensions
"""

    def to_str(expr):
        return str(expr).lower().replace(' ', '')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert routine.variable_map['v0'].shape[0] == 10
    assert isinstance(routine.variable_map['v0'].shape[0], sym.IntLiteral)
    assert isinstance(routine.variable_map['v1'].shape[0], sym.Scalar)
    assert routine.variable_map['v2'].shape[0] == 'x'
    assert routine.variable_map['v2'].shape[1] == 'y'
    assert routine.variable_map['v2'].shape[2] == ':'
    assert isinstance(routine.variable_map['v2'].shape[0], sym.Scalar)
    assert isinstance(routine.variable_map['v2'].shape[1], sym.Scalar)
    assert isinstance(routine.variable_map['v2'].shape[2], sym.RangeIndex)
    assert routine.variable_map['v3'].shape[0] == 'x'
    assert routine.variable_map['v3'].shape[1] == 'y'
    assert routine.variable_map['v3'].shape[2] == ':'
    assert isinstance(routine.variable_map['v3'].shape[0], sym.Scalar)
    assert isinstance(routine.variable_map['v3'].shape[1], sym.Scalar)
    assert isinstance(routine.variable_map['v3'].shape[2], sym.RangeIndex)
    assert routine.variable_map['v4'].shape[0] == 'x'
    assert routine.variable_map['v4'].shape[1] == 'y'
    assert isinstance(routine.variable_map['v4'].shape[0], sym.Scalar)
    assert isinstance(routine.variable_map['v4'].shape[1], sym.Scalar)
    assert to_str(routine.variable_map['v5'].shape[0]) == 'x+y'
    assert to_str(routine.variable_map['v5'].shape[1]) == '2*x'
    assert isinstance(routine.variable_map['v5'].shape[0], sym.Sum)
    assert isinstance(routine.variable_map['v5'].shape[1], sym.Product)
    assert to_str(routine.variable_map['v6'].shape[0]) == 'x/2'
    assert to_str(routine.variable_map['v6'].shape[1]) == 'x**2'
    assert to_str(routine.variable_map['v6'].shape[2]) == '(x+y)/x'
    assert isinstance(routine.variable_map['v6'].shape[0], sym.Quotient)
    assert isinstance(routine.variable_map['v6'].shape[1], sym.Power)
    assert isinstance(routine.variable_map['v6'].shape[2], sym.Quotient)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI does not like Loki pragmas, yet!')]))
def test_module_variables_dimension_pragmas(frontend, tmp_path):
    """
    Test that `!$loki dimension` pragmas can be used to override the
    conceptual `.shape` of module variables.
    """
    code_mod = """
module mod_variable_dimensions

  integer, parameter :: jprb = selected_real_kind(13,300)
  integer :: x, y

  !$loki dimension(10)
  real(kind=jprb), intent(inout) :: v0(:)
  !$loki dimension(x)
  real(kind=jprb), intent(inout) :: v1(:)
  !$loki dimension(x,y,:)
  real(kind=jprb), dimension(:,:,:), intent(inout) :: v2, v3
  !$loki dimension(x,y)
  real(kind=jprb), pointer, intent(inout) :: v4(:,:)
  !$loki dimension(x+y,2*x)
  real(kind=jprb), allocatable :: v5(:,:)
  !$loki dimension(x/2, x**2, (x+y)/x)
  real(kind=jprb), dimension(:, :, :), pointer :: v6
end module mod_variable_dimensions
    """

    def to_str(expr):
        return str(expr).lower().replace(' ', '')

    mod = Module.from_source(code_mod, frontend=frontend, xmods=[tmp_path])
    variable_map = mod.variable_map
    assert variable_map['v0'].shape[0] == 10
    assert isinstance(variable_map['v0'].shape[0], sym.IntLiteral)
    assert isinstance(variable_map['v1'].shape[0], sym.Scalar)
    assert variable_map['v2'].shape[0] == 'x'
    assert variable_map['v2'].shape[1] == 'y'
    assert variable_map['v2'].shape[2] == ':'
    assert isinstance(variable_map['v2'].shape[0], sym.Scalar)
    assert isinstance(variable_map['v2'].shape[1], sym.Scalar)
    assert isinstance(variable_map['v2'].shape[2], sym.RangeIndex)
    assert variable_map['v3'].shape[0] == 'x'
    assert variable_map['v3'].shape[1] == 'y'
    assert variable_map['v3'].shape[2] == ':'
    assert isinstance(variable_map['v3'].shape[0], sym.Scalar)
    assert isinstance(variable_map['v3'].shape[1], sym.Scalar)
    assert isinstance(variable_map['v3'].shape[2], sym.RangeIndex)
    assert variable_map['v4'].shape[0] == 'x'
    assert variable_map['v4'].shape[1] == 'y'
    assert isinstance(variable_map['v4'].shape[0], sym.Scalar)
    assert isinstance(variable_map['v4'].shape[1], sym.Scalar)
    assert to_str(variable_map['v5'].shape[0]) == 'x+y'
    assert to_str(variable_map['v5'].shape[1]) == '2*x'
    assert isinstance(variable_map['v5'].shape[0], sym.Sum)
    assert isinstance(variable_map['v5'].shape[1], sym.Product)
    assert to_str(variable_map['v6'].shape[0]) == 'x/2'
    assert to_str(variable_map['v6'].shape[1]) == 'x**2'
    assert to_str(variable_map['v6'].shape[2]) == '(x+y)/x'
    assert isinstance(variable_map['v6'].shape[0], sym.Quotient)
    assert isinstance(variable_map['v6'].shape[1], sym.Power)
    assert isinstance(variable_map['v6'].shape[2], sym.Quotient)
