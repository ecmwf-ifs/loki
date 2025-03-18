# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour and correct parsing of certain Fortran
language features.
"""

import numpy as np
import pytest

from loki import (
    Module, Subroutine, Sourcefile, BasicType, config, config_override
)
from loki.jit_build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, FP, HAVE_FP
from loki.ir import nodes as ir, FindNodes, FindVariables

from conftest import XFAIL_DERIVED_TYPE_JIT_TESTS


@pytest.fixture(name='reset_frontend_mode')
def fixture_reset_frontend_mode():
    original_frontend_mode = config['frontend-strict-mode']
    yield
    config['frontend-strict-mode'] = original_frontend_mode


@pytest.mark.xfail(
    XFAIL_DERIVED_TYPE_JIT_TESTS,
    reason='Support for user-defined derived type arguments is broken in JIT compile'
)
@pytest.mark.parametrize('frontend', available_frontends())
def test_check_alloc_opts(tmp_path, frontend):
    """
    Test the use of SOURCE and STAT in allocate
    """

    fcode = """
module alloc_mod
    integer, parameter :: jprb = selected_real_kind(13,300)

    type explicit
        real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
        real(kind=jprb) :: red_herring
    end type explicit

    type deferred
        real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
        real(kind=jprb), allocatable :: red_herring
    end type deferred

contains

    subroutine alloc_deferred(item)
        type(deferred), intent(inout) :: item
        integer :: stat
        allocate(item%vector(3), stat=stat)
        allocate(item%matrix(3, 3))
    end subroutine alloc_deferred

    subroutine free_deferred(item)
        type(deferred), intent(inout) :: item
        integer :: stat
        deallocate(item%vector, stat=stat)
        deallocate(item%matrix)
    end subroutine free_deferred

    subroutine check_alloc_source(item, item2)
        type(explicit), intent(inout) :: item
        type(deferred), intent(inout) :: item2
        real(kind=jprb), allocatable :: vector(:), vector2(:)

        allocate(vector, source=item%vector)
        vector(:) = vector(:) + item%scalar
        item%vector(:) = vector(:)

        allocate(vector2, source=item2%vector)  ! Try mold here when supported by fparser
        vector2(:) = item2%scalar
        item2%vector(:) = vector2(:)
    end subroutine check_alloc_source
end module alloc_mod
    """.strip()

    # Parse the source and validate the IR
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    allocations = FindNodes(ir.Allocation).visit(module['check_alloc_source'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is not None for alloc in allocations)
    assert all(alloc.status_var is None for alloc in allocations)

    allocations = FindNodes(ir.Allocation).visit(module['alloc_deferred'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is None for alloc in allocations)
    assert allocations[0].status_var is not None
    assert allocations[1].status_var is None

    deallocs = FindNodes(ir.Deallocation).visit(module['free_deferred'].body)
    assert len(deallocs) == 2
    assert deallocs[0].status_var is not None
    assert deallocs[1].status_var is None

    # Sanity check for the backend
    assert module.to_fortran().lower().count(', stat=stat') == 2

    # Generate Fortran and test it
    filepath = tmp_path/(f'frontends_check_alloc_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='alloc_mod')

    item = mod.explicit()
    item.scalar = 1.
    item.vector[:] = 1.

    item2 = mod.deferred()
    mod.alloc_deferred(item2)
    item2.scalar = 2.
    item2.vector[:] = -1.

    mod.check_alloc_source(item, item2)
    assert (item.vector == 2.).all()
    assert (item2.vector == 2.).all()
    mod.free_deferred(item2)


@pytest.mark.xfail(
    XFAIL_DERIVED_TYPE_JIT_TESTS,
    reason='Support for user-defined derived type arguments is broken in JIT compile'
)
@pytest.mark.parametrize('frontend', available_frontends())
def test_associates(tmp_path, frontend):
    """Test the use of associate to access and modify other items"""

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine associates(item)
    type(explicit), intent(inout) :: item
    type(deferred) :: item2

    item%scalar = 17.0

    associate(vector2=>item%matrix(:,1))
        vector2(:) = 3.
        item%matrix(:,3) = vector2(:)
    end associate

    associate(vector=>item%vector)
        item%vector(2) = vector(1)
        vector(3) = item%vector(1) + vector(2)
        vector(1) = 1.
    end associate

    call alloc_deferred(item2)

    associate(vec=>item2%vector(2))
        vec = 1.
    end associate

    call free_deferred(item2)
  end subroutine associates
end module
"""
    # Test the internals
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['associates']
    variables = FindVariables().visit(routine.body)
    assert all(
        v.shape == ('3',) for v in variables if v.name in ['vector', 'vector2']
    )

    for assoc in FindNodes(ir.Associate).visit(routine.body):
        for var in FindVariables().visit(assoc.body):
            if var.name in assoc.variables:
                assert var.scope is assoc
                assert var.type.parent is None
            else:
                assert var.scope is routine

    # Test the generated module
    filepath = tmp_path/(f'derived_types_associates_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    mod.associates(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI fails to read without full module')]))
def test_associates_deferred(frontend):
    """
    Verify that reading in subroutines with deferred external type definitions
    and associates working on that are supported.
    """

    fcode = """
SUBROUTINE ASSOCIATES_DEFERRED(ITEM, IDX)
USE SOME_MOD, ONLY: SOME_TYPE
IMPLICIT NONE
TYPE(SOME_TYPE), INTENT(IN) :: ITEM
INTEGER, INTENT(IN) :: IDX
ASSOCIATE(SOME_VAR=>ITEM%SOME_VAR(IDX), SOME_OTHER_VAR=>ITEM%SOME_VAR(ITEM%OFFSET))
SOME_VAR = 5
END ASSOCIATE
END SUBROUTINE
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 6
    some_var = variables['SOME_VAR']
    assert isinstance(some_var, sym.DeferredTypeSymbol)
    assert some_var.name.upper() == 'SOME_VAR'
    assert some_var.type.dtype == BasicType.DEFERRED
    associate = FindNodes(ir.Associate).visit(routine.body)[0]
    assert some_var.scope is associate

    some_other_var = variables['SOME_OTHER_VAR']
    assert isinstance(some_var, sym.DeferredTypeSymbol)
    assert some_other_var.name.upper() == 'SOME_OTHER_VAR'
    assert some_other_var.type.dtype == BasicType.DEFERRED
    assert some_other_var.type.shape == ('ITEM%OFFSET',)
    assert some_other_var.scope is associate


@pytest.mark.parametrize('frontend', available_frontends())
def test_associates_expr(tmp_path, frontend):
    """Verify that associates with expressions are supported"""
    fcode = """
subroutine associates_expr(in, out)
  implicit none
  integer, intent(in) :: in(3)
  integer, intent(out) :: out(3)

  out(:) = 0

  associate(a=>1+3)
    out(:) = out(:) + a
  end associate

  associate(b=>2*in(:) + in(:))
    out(:) = out(:) + b(:)
  end associate
end subroutine associates_expr
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 4
    assert isinstance(variables['a'], sym.DeferredTypeSymbol)
    assert variables['a'].type.dtype is BasicType.DEFERRED  # TODO: support type derivation for expressions
    assert isinstance(variables['b'], sym.Array)  # Note: this is an array because we have a shape
    assert variables['b'].type.dtype is BasicType.DEFERRED  # TODO: support type derivation for expressions
    assert variables['b'].type.shape == ('3',)

    filepath = tmp_path/(f'associates_expr_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    a = np.array([1, 2, 3], dtype='i')
    b = np.zeros(3, dtype='i')
    function(a, b)
    assert np.all(b == [7, 10, 13])


@pytest.mark.parametrize('frontend', available_frontends())
def test_enum(tmp_path, frontend):
    """Verify that enums are represented correctly"""
    # F2008, Note 4.67
    fcode = """
subroutine test_enum (out)
    implicit none

    ! Comment 1
    ENUM, BIND(C)
        ENUMERATOR :: RED = 4, BLUE = 9
        ! Comment 2
        ENUMERATOR YELLOW
    END ENUM
    ! Comment 3

    integer, intent(out) :: out

    out = RED + BLUE + YELLOW
end subroutine test_enum
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check Enum exists
    enums = FindNodes(ir.Enumeration).visit(routine.spec)
    assert len(enums) == 1

    # Check symbols are available
    assert enums[0].symbols == ('red', 'blue', 'yellow')
    assert all(name in routine.symbols for name in ('red', 'blue', 'yellow'))
    assert all(s.scope is routine for s in enums[0].symbols)

    # Check assigned values
    assert routine.symbol_map['red'].type.initial == '4'
    assert routine.symbol_map['blue'].type.initial == '9'
    assert routine.symbol_map['yellow'].type.initial is None

    # Verify comments are preserved (don't care about the actual place)
    code = routine.to_fortran()
    for i in range(1, 4):
        assert f'! Comment {i}' in code

    # Check fgen produces valid code and runs
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    out = function()
    assert out == 23


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
def test_frontend_pragma_vs_comment(frontend, tmp_path):
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
    xfail=[(OMNI, 'OMNI strips comments during parse')]
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


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI does not like Loki pragmas, yet!')]))
def test_frontend_routine_variables_dimension_pragmas(frontend):
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

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI does not like Loki pragmas, yet!')]))
def test_frontend_module_variables_dimension_pragmas(frontend, tmp_path):
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_import_of_private_symbols(tmp_path, frontend):
    """
    Verify that only public symbols are imported from other modules.
    """
    code_mod_private = """
module mod_private
    private
    integer :: var
end module mod_private
    """
    code_mod_public = """
module mod_public
    public
    integer:: var
end module mod_public
    """
    code_mod_main = """
module mod_main
    use mod_public
    use mod_private
contains

    subroutine test_routine()
        integer :: result
        result = var
    end subroutine test_routine

end module mod_main
    """

    mod_private = Module.from_source(code_mod_private, frontend=frontend, xmods=[tmp_path])
    mod_public = Module.from_source(code_mod_public, frontend=frontend, xmods=[tmp_path])
    mod_main = Module.from_source(
        code_mod_main, frontend=frontend, definitions=[mod_private, mod_public], xmods=[tmp_path]
    )
    var = mod_main.subroutines[0].body.body[0].rhs
    # Check if this is really our symbol
    assert var.name == "var"
    assert var.scope is mod_main
    # Check if the symbol is imported
    assert var.type.imported is True
    # Check if the symbol comes from the mod_public module
    assert var.type.module is mod_public


@pytest.mark.parametrize('frontend', available_frontends())
def test_access_spec(tmp_path, frontend):
    """
    Check that access-spec statements are dealt with correctly.
    """
    code_mod_private_var_public = """
module mod_private_var_public
    private
    integer :: var
    public :: var
end module mod_private_var_public
    """
    code_mod_public_var_private = """
module mod_public_var_private
    public
    integer :: var
    private :: var
end module mod_public_var_private
    """
    code_mod_main = """
module mod_main
    use mod_private_var_public
    use mod_public_var_private
contains

    subroutine test_routine()
        integer :: result
        result = var
    end subroutine test_routine

end module mod_main
    """

    mod_private_var_public = Module.from_source(code_mod_private_var_public, frontend=frontend, xmods=[tmp_path])
    mod_public_var_private = Module.from_source(code_mod_public_var_private, frontend=frontend, xmods=[tmp_path])
    mod_main = Module.from_source(
        code_mod_main, frontend=frontend, definitions=[mod_private_var_public, mod_public_var_private], xmods=[tmp_path]
    )
    var = mod_main.subroutines[0].body.body[0].rhs
    # Check if this is really our symbol
    assert var.name == "var"
    assert var.scope is mod_main
    # Check if the symbol is imported
    assert var.type.imported is True
    # Check if the symbol comes from the mod_private_var_public module
    assert var.type.module is mod_private_var_public


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not like intrinsic shading for member functions!')]
))
def test_intrinsic_shadowing(tmp_path, frontend):
    """
    Test that locally defined functions that shadow intrinsics are handled.
    """
    fcode_algebra = """
module algebra_mod
implicit none
contains
  function dot_product(a, b) result(c)
    real(kind=8), intent(inout) :: a(:), b(:)
    real(kind=8) :: c
  end function dot_product

  function min(x, y)
    real(kind=8), intent(in) :: x, y
    real(kind=8) :: min

    min = y
    if (x < y) min = x
  end function min
end module algebra_mod
"""

    fcode = """
module test_intrinsics_mod
use algebra_mod, only: dot_product
implicit none

contains

  subroutine test_intrinsics(a, b, c, d)
    use algebra_mod, only: min
    implicit none
    real(kind=8), intent(inout) :: a(:), b(:)
    real(kind=8) :: c, d, e

    c = dot_product(a, b)
    d = max(c, a(1))
    e = min(c, a(1))

  contains

    function max(x, y)
      real(kind=8), intent(in) :: x, y
      real(kind=8) :: max

      max = y
      if (x > y) max = x
    end function max
  end subroutine test_intrinsics
end module test_intrinsics_mod
"""
    algebra = Module.from_source(fcode_algebra, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(
        fcode, definitions=algebra, frontend=frontend, xmods=[tmp_path]
    )
    routine = module['test_intrinsics']

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3

    assert isinstance(assigns[0].rhs.function, sym.ProcedureSymbol)
    assert not assigns[0].rhs.function.type.is_intrinsic
    assert assigns[0].rhs.function.type.dtype.procedure == algebra['dot_product']

    assert isinstance(assigns[1].rhs.function, sym.ProcedureSymbol)
    assert not assigns[1].rhs.function.type.is_intrinsic
    assert assigns[1].rhs.function.type.dtype.procedure == routine.members[0]

    assert isinstance(assigns[2].rhs.function, sym.ProcedureSymbol)
    assert not assigns[2].rhs.function.type.is_intrinsic
    assert assigns[2].rhs.function.type.dtype.procedure == algebra['min']


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_symbol_scoping(frontend):
    """ Check that the return symbol of a function has the right scope """
    fcode = """
real function double_real(i)
  implicit none
  integer, intent(in) :: i

  double_real =  dble(i*2)
end function double_real
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    rsym = routine.variable_map['double_real']
    assert isinstance(rsym, sym.Scalar)
    assert rsym.type.dtype == BasicType.REAL
    assert rsym.scope == routine

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 1
    assert assigns[0].lhs == 'double_real'
    assert isinstance(assigns[0].lhs, sym.Scalar)
    assert assigns[0].lhs.type.dtype == BasicType.REAL
    assert assigns[0].lhs.scope == routine


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_derived_type_imports(tmp_path, frontend):
    """ Checks that provided module and type info is attached during parse """
    fcode_module = """
module my_type_mod
  type my_type
    real(kind=8) :: a, b(:)
  end type my_type
end module my_type_mod
"""

    fcode = """
subroutine test_derived_type_parse
  use my_type_mod, only: my_type
  implicit none
  type(my_type) :: obj

  obj%a = 42.0
  obj%b = 66.6
end subroutine test_derived_type_parse
"""
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(
        fcode, definitions=module, frontend=frontend, xmods=[tmp_path]
    )

    assert len(module.typedefs) == 1
    assert module.typedefs[0].name == 'my_type'

    # Ensure that the imported type is recognised as such
    assert len(routine.imports) == 1
    assert routine.imports[0].module == 'my_type_mod'
    assert len(routine.imports[0].symbols) == 1
    assert routine.imports[0].symbols[0] == 'my_type'
    assert isinstance(routine.imports[0].symbols[0], sym.DerivedTypeSymbol)

    # Ensure that the declared variable and its components are recognised
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert isinstance(assigns[0].lhs, sym.Scalar)
    assert assigns[0].lhs.type.dtype == BasicType.REAL
    assert isinstance(assigns[1].lhs, sym.Array)
    assert assigns[1].lhs.type.dtype == BasicType.REAL
    assert assigns[1].lhs.type.shape == (':',)


@pytest.mark.skipif(not HAVE_FP, reason="Assumed size declarations only supported for FP")
def test_assumed_size_declarations():
    """
    Test if assumed size declarations are correctly parsed.
    """

    fcode = """
subroutine kernel(a, b, c)
  implicit none
  real, intent(in) :: a(*)
  real, intent(in) :: b(8,*)
  real, intent(in) :: c(8,0:*)

end subroutine kernel
"""

    kernel = Subroutine.from_source(fcode, frontend=FP)

    variable_map = kernel.variable_map
    a = variable_map['a']
    b = variable_map['b']
    c = variable_map['c']

    assert len(a.shape) == 1

    assert len(b.shape) == 2
    assert b.shape[0] == 8

    assert len(c.shape) == 2
    assert c.shape[0] == 8
    assert c.shape[1].lower == 0

    assert all('*' in str(shape) for shape in [a.shape, b.shape, c.shape])


@pytest.mark.parametrize('frontend', available_frontends())
def test_empty_print_statement(frontend):
    """
    Test if an empty print statement (PRINT *) is parsed correctly.
    """
    fcode = """
SUBROUTINE test_routine()
    IMPLICIT NONE
    print *
    ! Using single quotes to simplify the test comparison (see below)
    print *, 'test_text'
END SUBROUTINE test_routine
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    print_stmts = [
        intr for intr in FindNodes(ir.Intrinsic).visit(routine.ir)
        if 'print' in intr.text.lower()
    ]
    assert print_stmts[0].text.lower() == "print *"
    # NOTE: OMNI always uses single quotes ('') to represent string data in PRINT statements
    #       while fparser will mimic the quotes used in the parsed source code
    assert print_stmts[1].text.lower() == "print *, 'test_text'"
