# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends

from loki import (
    Module, Subroutine, fgen, OMNI, OFP, Intrinsic, DataDeclaration,
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_fgen_literal_list_linebreak(frontend):
    """
    Test correct handling of linebreaks for LiteralList expression nodes
    """
    fcode = """
module some_mod
  implicit none
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
  interface
    subroutine config_gas_optics_sw_spectral_def_allocate_bands_only(a, b)
        INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
        real(kind=jprb), intent(in) :: a(:), b(:)
    end subroutine config_gas_optics_sw_spectral_def_allocate_bands_only
  end interface
contains
  subroutine literal_list_linebreak
    implicit none
    real(jprb), parameter, dimension(1,140) :: frac &
        = reshape( (/ 0.21227E+00, 0.18897E+00, 0.25491E+00, 0.17864E+00, 0.11735E+00, 0.38298E-01, 0.57871E-02, &
        &    0.31753E-02, 0.53169E-03, 0.76476E-04, 0.16388E+00, 0.15241E+00, 0.14290E+00, 0.12864E+00, &
        &    0.11615E+00, 0.10047E+00, 0.80013E-01, 0.60445E-01, 0.44918E-01, 0.63395E-02, 0.32942E-02, &
        &    0.54541E-03, 0.15380E+00, 0.15194E+00, 0.14339E+00, 0.13138E+00, 0.11701E+00, 0.10081E+00, &
        &    0.82296E-01, 0.61735E-01, 0.41918E-01, 0.45918E-02, 0.37743E-02, 0.30121E-02, 0.22500E-02, &
        &    0.14490E-02, 0.55410E-03, 0.78364E-04, 0.15938E+00, 0.15146E+00, 0.14213E+00, 0.13079E+00, &
        &    0.11672E+00, 0.10053E+00, 0.81566E-01, 0.61126E-01, 0.41150E-01, 0.44488E-02, 0.36950E-02, &
        &    0.29101E-02, 0.21357E-02, 0.19609E-02, 0.14134E+00, 0.14390E+00, 0.13913E+00, 0.13246E+00, &
        &    0.12185E+00, 0.10596E+00, 0.87518E-01, 0.66164E-01, 0.44862E-01, 0.49402E-02, 0.40857E-02, &
        &    0.32288E-02, 0.23613E-02, 0.15406E-02, 0.58258E-03, 0.82171E-04, 0.29127E+00, 0.28252E+00, &
        &    0.22590E+00, 0.14314E+00, 0.45494E-01, 0.71792E-02, 0.38483E-02, 0.65712E-03, 0.29810E+00, &
        &    0.27559E+00, 0.11997E+00, 0.10351E+00, 0.84515E-01, 0.62253E-01, 0.41050E-01, 0.44217E-02, &
        &    0.36946E-02, 0.29113E-02, 0.34290E-02, 0.55993E-03, 0.31441E+00, 0.27586E+00, 0.21297E+00, &
        &    0.14064E+00, 0.45588E-01, 0.65665E-02, 0.34232E-02, 0.53199E-03, 0.19811E+00, 0.16833E+00, &
        &    0.13536E+00, 0.11549E+00, 0.10649E+00, 0.93264E-01, 0.75720E-01, 0.56405E-01, 0.41865E-01, &
        &    0.59331E-02, 0.26510E-02, 0.40040E-03, 0.32328E+00, 0.26636E+00, 0.21397E+00, 0.14038E+00, &
        &    0.52142E-01, 0.38852E-02, 0.14601E+00, 0.13824E+00, 0.27703E+00, 0.22388E+00, 0.15446E+00, &
        &    0.48687E-01, 0.98054E-02, 0.18870E-02, 0.11961E+00, 0.12106E+00, 0.13215E+00, 0.13516E+00, &
        &    0.25249E+00, 0.16542E+00, 0.68157E-01, 0.59725E-02, 0.49258E+00, 0.33651E+00, 0.16182E+00, &
        &    0.90984E-02, 0.95202E+00, 0.47978E-01, 0.91716E+00, 0.82857E-01, 0.77464E+00, 0.22536E+00 /), (/ 1,140 /) )
    call config_gas_optics_sw_spectral_def_allocate_bands_only( &
         &  [2600.0_jprb, 3250.0_jprb, 4000.0_jprb, 4650.0_jprb, 5150.0_jprb, 6150.0_jprb, 7700.0_jprb, &
         &   8050.0_jprb, 12850.0_jprb, 16000.0_jprb, 22650.0_jprb, 29000.0_jprb, 38000.0_jprb, 820.0_jprb], &
         &  [3250.0_jprb, 4000.0_jprb, 4650.0_jprb, 5150.0_jprb, 6150.0_jprb, 7700.0_jprb, 8050.0_jprb, &
         &   12850.0_jprb, 16000.0_jprb, 22650.0_jprb, 29000.0_jprb, 38000.0_jprb, 50000.0_jprb, 2600.0_jprb])
  end subroutine literal_list_linebreak
end module some_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    routine = module['literal_list_linebreak']

    # Make sure all lines are continued correctly
    code = module.to_fortran()
    code_lines = code.splitlines()
    assert len(code_lines) in (35, 36) # OMNI produces an extra line
    assert all(line.strip(' &\n') for line in code_lines)
    assert all(len(line) < 132 for line in code_lines)

    # Make sure it works also with less indentation
    spec_code = fgen(routine.spec)
    assert spec_code.count('&') == 32
    spec_lines = spec_code.splitlines()
    assert len(spec_lines) == 18
    assert all(len(line) < 132 for line in spec_code.splitlines())

    body_code = fgen(routine.body)
    assert body_code.count(',') == 27
    assert body_code.count('(/') == 2
    assert body_code.count('/)') == 2
    assert body_code.count('&') == 6
    body_lines = body_code.splitlines()
    assert len(body_lines) == 4
    assert all(len(line) < 132 for line in body_lines)


@pytest.mark.parametrize('frontend', available_frontends())
def test_fgen_data_stmt(frontend):
    """
    Test correct formatting of data declaration statements
    """
    fcode = """
subroutine data_stmt
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    REAL(KIND=JPRB) :: ZAMD
    INTEGER :: KXINDX(35)
    data ZAMD   /  28.970_JPRB    /
    DATA KXINDX /0,2,3,0,31*0/
end subroutine data_stmt
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    if frontend == OFP:
        assert isinstance(routine.spec.body[-1], Intrinsic)
    else:
        assert isinstance(routine.spec.body[-1], DataDeclaration)
    spec_code = fgen(routine.spec)
    assert spec_code.lower().count('data ') == 2
    assert spec_code.count('/') == 4
    if frontend != OMNI:
        # OMNI seems to evaluate constant expressions, replacing 31*0 by 0,
        # although it's not a product here but a repeat specifier (great job, Fortran!)
        assert '31*0' in spec_code


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Loki likes only valid code')]))
def test_multiline_inline_conditional(frontend):
    """
    Test correct formatting of an inline :any:`Conditional` that
    contains a multi-line :any:`CallStatement`.
    """
    fcode = """
subroutine test_fgen(DIMS, ZSURF_LOCAL)
contains
subroutine test_inline_multiline(KDIMS, LBUD23)

  DO JKGLO=1,NGPTOT,NPROMA
    ! Add saturation adjustment tendencies to cloud scheme (LBUD23)
    IF (LBUD23) CALL UPDATE_FIELDS(YDPHY2,1,DIMS%KIDIA,DIMS%KFDIA,DIMS%KLON,DIMS%KLEV,&
     & PTA1=TENDENCY_LOC%T, PO1=ZSURF_LOCAL%GSD_XA%PGROUP(:,:,19),&
     & PTA2=TENDENCY_LOC%Q, PO2=ZSURF_LOCAL%GSD_XA%PGROUP(:,:,20),&
     & LDV3=YGFL%YL%LT1, PTA3=TENDENCY_LOC%CLD(:,:,NCLDQL), PO3=ZSURF_LOCAL%GSD_XA%PGROUP(:,:,21),&
     & LDV4=YGFL%YI%LT1, PTA4=TENDENCY_LOC%CLD(:,:,NCLDQI), PO4=ZSURF_LOCAL%GSD_XA%PGROUP(:,:,22))
  ENDDO
end subroutine test_inline_multiline
end subroutine test_fgen
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    out = fgen(routine, linewidth=132)
    for line in out.splitlines():
        assert line.count('&') <= 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_fgen_save_attribute(frontend):
    """
    Make sure the SAVE attribute on declarations is preserved (#164)
    """
    fcode = """
MODULE test
    INTEGER, SAVE :: variable
END MODULE test
    """.strip()
    module = Module.from_source(fcode, frontend=frontend)
    assert module['variable'].type.save is True
    assert len(module.declarations) == 1
    assert 'SAVE' in fgen(module.declarations[0])
    assert 'SAVE' in module.to_fortran()
