# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify REGEX frontend behaviour for incomplete subroutine parsing.
"""

import pytest

from loki import Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, REGEX


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
