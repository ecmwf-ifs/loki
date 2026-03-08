# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for subroutine parse shape.
"""

import pytest

from loki import Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, REGEX
from loki.ir import nodes as ir, FindNodes
from loki.types import BasicType


@pytest.mark.parametrize('frontend', available_frontends())
def test_routine_simple(frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
  ! This is the docstring ...

  ! It spans multiple intersected lines ...
  ! ... and is followed by a ...

  !$loki routine fun

  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_simple
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert routine.arguments == ('x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)')
    assert routine.variables == ('jprb', 'x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)', 'i')

    assert len(routine.docstring) == 1
    assert isinstance(routine.docstring[0], ir.CommentBlock)
    if frontend == OMNI:
        assert len(routine.docstring[0].comments) == 3
        assert routine.docstring[0].comments[0].text == '! This is the docstring ...'
        assert routine.docstring[0].comments[1].text == '! It spans multiple intersected lines ...'
        assert routine.docstring[0].comments[2].text == '! ... and is followed by a ...'
    else:
        assert len(routine.docstring[0].comments) == 5
        assert routine.docstring[0].comments[0].text == '! This is the docstring ...'
        assert routine.docstring[0].comments[2].text == '! It spans multiple intersected lines ...'
        assert routine.docstring[0].comments[3].text == '! ... and is followed by a ...'
    assert routine.definitions == ()

    assert isinstance(routine.body, ir.Section)
    if frontend == OMNI:
        assert len(routine.spec) == 9
        assert isinstance(routine.spec[0], ir.ImplicitStmt)
        assert isinstance(routine.spec[1], ir.Pragma)
        assert all(isinstance(node, ir.VariableDeclaration) for node in routine.spec[2:])
        assert routine.spec[2].symbols == ('jprb',)
        assert routine.spec[3].symbols == ('x',)
        assert routine.spec[4].symbols == ('y',)
        assert routine.spec[5].symbols == ('scalar',)
        assert routine.spec[6].symbols == ('vector(x)',)
        assert routine.spec[7].symbols == ('matrix(x, y)',)
        assert routine.spec[8].symbols == ('i',)
    else:
        assert len(routine.spec) == 7
        assert isinstance(routine.spec[0], ir.Pragma)
        assert isinstance(routine.spec[1], ir.Comment)
        assert all(isinstance(node, ir.VariableDeclaration) for node in routine.spec[2:])
        assert routine.spec[2].symbols == ('jprb',)
        assert routine.spec[3].symbols == ('x', 'y')
        assert routine.spec[4].symbols == ('scalar',)
        assert routine.spec[5].symbols == ('vector(x)', 'matrix(x, y)')
        assert routine.spec[6].symbols == ('i',)

    assert isinstance(routine.spec, ir.Section)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1 and loops[0].variable == 'i'
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0] in loops[0].body and assigns[1] in loops[0].body


@pytest.mark.parametrize('frontend', available_frontends())
def test_routine_arguments(frontend):
    """
    A set of test to test internalisation and handling of arguments.
    """

    fcode = """
subroutine routine_arguments &
 ! Test multiline dummy arguments with comments
 & (x, y, scalar, &
 ! Of course, not one...
 ! but two comment lines
 & vector, matrix)
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)
  ! The order below is intentioanlly inverted
  real(kind=jprb), intent(inout) :: matrix(x, y)
  real(kind=jprb), intent(in)    :: scalar
  real(kind=jprb), dimension(x)  :: local_vector
  real(kind=jprb), dimension(x), intent(out) :: vector
  integer, intent(in) :: x, y

  integer :: i, j
  real(kind=jprb) :: local_matrix(x, y)

  do i=1, x
     local_vector(i) = i * 10.
     do j=1, y
        local_matrix(i, j) = local_vector(i) + j * scalar
     end do
  end do

  vector(:) = local_vector(:)
  matrix(:, :) = local_matrix(:, :)

end subroutine routine_arguments
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    if frontend == OMNI:
        assert not routine.docstring
    else:
        assert len(routine.docstring) == 1
        assert len(routine.docstring[0].comments) == 3
        assert routine.docstring[0].comments[0].text == '! Test multiline dummy arguments with comments'
        assert routine.docstring[0].comments[1].text == '! Of course, not one...'
        assert routine.docstring[0].comments[2].text == '! but two comment lines'

    assert routine.arguments == ('x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)')
    assert all(isinstance(arg, sym.Scalar) for arg in routine.arguments[0:3])
    assert all(arg.type.intent == 'in' for arg in routine.arguments[0:3])
    assert all(isinstance(arg, sym.Array) for arg in routine.arguments[3:])
    assert all(arg.type.dtype == BasicType.INTEGER for arg in routine.arguments[0:2])
    assert all(arg.type.dtype == BasicType.REAL for arg in routine.arguments[2:5])
    if frontend == OMNI:
        assert all(isinstance(arg.type.kind, sym.InlineCall) for arg in routine.arguments[2:5])
    else:
        assert all(arg.type.kind == 'jprb' for arg in routine.arguments[2:5])
    assert routine.arguments[3].shape == ('x',)
    assert routine.arguments[4].shape == ('x', 'y')
    assert routine.arguments[3].type.intent == 'out'
    assert routine.arguments[4].type.intent == 'inout'

    assert routine.variables == (
        'jprb', 'matrix(x, y)', 'scalar', 'local_vector(x)',
        'vector(x)', 'x', 'y', 'i', 'j', 'local_matrix(x, y)'
    )
    assert routine.variables[0].type.parameter
    assert isinstance(routine.variables[0].type.initial, sym.InlineCall)
    assert routine.variables[0].type.initial.function == 'selected_real_kind'
    assert routine.variables[1].type.dtype == BasicType.REAL
    assert routine.variables[1].shape == ('x', 'y')
    assert routine.variables[2].type.dtype == BasicType.REAL
    assert routine.variables[3].type.dtype == BasicType.REAL
    assert routine.variables[3].shape == ('x',)
    assert routine.variables[4].type.dtype == BasicType.REAL
    assert routine.variables[4].shape == ('x',)
    assert routine.variables[5].type.dtype == BasicType.INTEGER
    assert routine.variables[6].type.dtype == BasicType.INTEGER
    assert routine.variables[7].type.dtype == BasicType.INTEGER
    assert routine.variables[8].type.dtype == BasicType.INTEGER
    assert routine.variables[9].type.dtype == BasicType.REAL
    assert routine.variables[9].shape == ('x', 'y')


@pytest.mark.parametrize('frontend', available_frontends())
def test_empty_spec(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_empty_spec
write(*,*) 'Hello world!'
end subroutine routine_empty_spec
""")
    if frontend == OMNI:
        assert len(routine.spec) == 1
    else:
        assert not routine.spec
    assert len(routine.body.body) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_subroutine_lazy_arguments_incomplete1(frontend):
    """
    Test that argument lists for subroutines are correctly captured when the object is made
    complete.

    The rationale for this test is that for dummy argument lists with interleaved comments and line
    breaks, matching is non-trivial and, since we don't currently need the argument list
    in the incomplete REGEX-parsed IR, we accept that this information is incomplete initially.
    tmp_path, we make sure this information is captured correctly after completing the full frontend
    parse.
    """
    fcode = """
subroutine my_routine(n, a, b, d)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)
    real, intent(out) :: d(n)
    integer :: i

    do i=1, n
        d(i) = a(i) + b(i)
    end do
end subroutine my_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=REGEX)
    assert routine._incomplete
    assert routine.arguments == ()
    assert routine.argnames == []
    assert routine._dummies == ()
    assert all(isinstance(arg, sym.DeferredTypeSymbol) for arg in routine.arguments)

    routine.make_complete(frontend=frontend)
    assert not routine._incomplete
    assert routine.arguments == ('n', 'a(n)', 'b(n)', 'd(n)')
    assert routine.argnames == ['n', 'a', 'b', 'd']
    assert routine._dummies == ('n', 'a', 'b', 'd')
    assert isinstance(routine.arguments[0], sym.Scalar)
    assert all(isinstance(arg, sym.Array) for arg in routine.arguments[1:])


@pytest.mark.parametrize('frontend', available_frontends())
def test_subroutine_lazy_arguments_incomplete2(frontend):
    """
    Test that argument lists for subroutines are correctly captured when the object is made
    complete.

    The rationale for this test is that for dummy argument lists with interleaved comments and line
    breaks, matching is non-trivial and, since we don't currently need the argument list
    in the incomplete REGEX-parsed IR, we accept that this information is not available initially.
    tmp_path, we make sure this information is captured correctly after completing the full frontend
    parse.
    """
    fcode = """
SUBROUTINE CLOUDSC &
 !---input
 & (KIDIA,    KFDIA,    KLON,    KLEV,&
 & PT, PQ, &
 !---prognostic fields
 & PA,&
 & PCLV,  &
 & PSUPSAT,&
!-- arrays for aerosol-cloud interactions
!!! & PQAER,    KAER, &
 & PRE_ICE,&
 & PCCN,     PNICE,&
 !---diagnostic output
 & PCOVPTOT, PRAINFRAC_TOPRFZ,&
 !---resulting fluxes
 & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG&
 & )
IMPLICIT NONE
INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
INTEGER(KIND=JPIM),PARAMETER :: NCLV=5      ! number of microphysics variables
INTEGER(KIND=JPIM),INTENT(IN)    :: KLON             ! Number of grid points
INTEGER(KIND=JPIM),INTENT(IN)    :: KLEV             ! Number of levels
INTEGER(KIND=JPIM),INTENT(IN)    :: KIDIA
INTEGER(KIND=JPIM),INTENT(IN)    :: KFDIA
REAL(KIND=JPRB)   ,INTENT(IN)    :: PT(KLON,KLEV)    ! T at start of callpar
REAL(KIND=JPRB)   ,INTENT(IN)    :: PQ(KLON,KLEV)    ! Q at start of callpar
REAL(KIND=JPRB)   ,INTENT(IN)    :: PA(KLON,KLEV)    ! Original Cloud fraction (t)
REAL(KIND=JPRB)   ,INTENT(IN)    :: PCLV(KLON,KLEV,NCLV)
REAL(KIND=JPRB)   ,INTENT(IN)    :: PSUPSAT(KLON,KLEV)
REAL(KIND=JPRB)   ,INTENT(IN)    :: PRE_ICE(KLON,KLEV)
REAL(KIND=JPRB)   ,INTENT(IN)    :: PCCN(KLON,KLEV)     ! liquid cloud condensation nuclei
REAL(KIND=JPRB)   ,INTENT(IN)    :: PNICE(KLON,KLEV)    ! ice number concentration (cf. CCN)
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PCOVPTOT(KLON,KLEV) ! Precip fraction
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PRAINFRAC_TOPRFZ(KLON)
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQLF(KLON,KLEV+1)  ! Flux of liquid
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQIF(KLON,KLEV+1)  ! Flux of ice
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFCQLNG(KLON,KLEV+1) ! -ve corr for liq
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFCQNNG(KLON,KLEV+1) ! -ve corr for ice
END SUBROUTINE CLOUDSC
    """.strip()

    argnames = (
        'kidia', 'kfdia', 'klon', 'klev', 'pt', 'pq',
        'pa', 'pclv', 'psupsat',
        'pre_ice', 'pccn', 'pnice',
        'pcovptot', 'prainfrac_toprfz',
        'pfsqlf', 'pfsqif', 'pfcqnng', 'pfcqlng'
    )
    argnames_with_dim = (
        'kidia', 'kfdia', 'klon', 'klev', 'pt(klon, klev)', 'pq(klon, klev)',
        'pa(klon, klev)', 'pclv(klon, klev, nclv)', 'psupsat(klon, klev)',
        'pre_ice(klon, klev)', 'pccn(klon, klev)', 'pnice(klon, klev)',
        'pcovptot(klon, klev)', 'prainfrac_toprfz(klon)',
        'pfsqlf(klon, klev + 1)', 'pfsqif(klon, klev + 1)', 'pfcqnng(klon, klev + 1)', 'pfcqlng(klon, klev + 1)'
    )

    routine = Subroutine.from_source(fcode, frontend=REGEX)
    assert routine._incomplete
    # NOTE: This represents the current capabilities of the REGEX frontend. If this test
    # suddenly fails because the argument list happens to be captured correctly:
    # Nice one! Go ahead and change the test.
    assert routine.arguments == ()
    assert routine.argnames == []
    assert routine._dummies == ()
    assert all(isinstance(arg, sym.DeferredTypeSymbol) for arg in routine.arguments)

    routine.make_complete(frontend=frontend)
    assert not routine._incomplete
    assert routine.arguments == argnames_with_dim
    assert [arg.upper() for arg in routine.argnames] == [arg.upper() for arg in argnames]
    assert routine._dummies == argnames
    assert all(isinstance(arg, sym.Scalar) for arg in routine.arguments[:4])
    assert all(isinstance(arg, sym.Array) for arg in routine.arguments[4:])
