# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki.backend import fgen
from loki.batch import Scheduler, SchedulerConfig, ProcedureItem
from loki.dimension import Dimension
from loki.expression import DeferredTypeSymbol, InlineCall, IntLiteral
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, CallStatement, Assignment, Pragma
from loki.sourcefile import Sourcefile
from loki.types import BasicType

from loki.transformations.raw_stack_allocator import TemporariesRawStackTransformation


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(name='block_dim', size='nb', index='b')

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('jstart', 'jend'))

@pytest.mark.parametrize('directive', ['openacc', 'openmp'])
@pytest.mark.parametrize('frontend', available_frontends())
def test_raw_stack_allocator_temporaries(frontend, block_dim, horizontal, directive, tmp_path):

    fcode_parkind_mod = """
module parkind1
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, parameter :: jpim = selected_int_kind(9)
  integer, parameter :: jplm = jpim
end module parkind1
    """.strip()

    fcode_yomphy_mod = """
module yomphy
  use parkind1, only: jpim
  implicit none
  type tphy
    integer(kind=jpim) :: n_spband
  end type tphy
end module yomphy
    """.strip()

    fcode_mf_phys_mod = """
module model_physics_mf_mod
  use yomphy, only: tphy
  implicit none
  type model_physics_mf_type
    type(tphy) :: yrphy
  end type model_physics_mf_type
end module model_physics_mf_mod
    """.strip()

    fcode_driver = """
module driver_mod
  contains
  subroutine driver(nlon, klev, nb, ydml_phy_mf)

    use parkind1, only: jpim, jprb

    use model_physics_mf_mod, only: model_physics_mf_type
    use kernel1_mod, only: kernel1

    implicit none

    type(model_physics_mf_type), intent(in) :: ydml_phy_mf

    integer(kind=jpim), intent(in) :: nlon
    integer(kind=jpim), intent(in) :: klev
    integer(kind=jpim), intent(in) :: nb

    integer(kind=jpim) :: jstart
    integer(kind=jpim) :: jend

    integer(kind=jpim) :: b

    real(kind=jprb), dimension(nlon, klev) :: zzz

    jstart = 1
    jend = nlon

    do b = 1, nb

        call kernel1(ydml_phy_mf, nlon, klev, jstart, jend, zzz)

    enddo

  end subroutine driver
end module driver_mod
    """.strip()

    fcode_kernel1 = """
module kernel1_mod
  contains
  subroutine kernel1(ydml_phy_mf, nlon, klev, jstart, jend, pzz)

    use parkind1, only: jpim, jprb

    use model_physics_mf_mod, only: model_physics_mf_type
    use kernel2_mod, only: kernel2
    use kernel3_mod, only: kernel3

    implicit none

    type(model_physics_mf_type), intent(in) :: ydml_phy_mf

    integer(kind=jpim), intent(in) :: nlon
    integer(kind=jpim), intent(in) :: klev

    integer(kind=jpim), intent(in) :: jstart
    integer(kind=jpim), intent(in) :: jend

    real(kind=jprb), intent(in), dimension(nlon, klev) :: pzz

    real(kind=jprb), dimension(nlon, klev) :: zzx
    real(kind=selected_real_kind(13,300)), dimension(nlon, klev) :: zzy
    logical, dimension(nlon, klev) :: zzl

    integer(kind=jpim) :: testint
    integer(kind=jpim) :: jl, jlev

    zzl = .false.
    do jl =1, nlon
      do jlev = 1, klev
        zzx(jl, jlev) = pzz(jl, jlev)
        zzy(jl, jlev) = pzz(jl, jlev)
      enddo
    enddo

    call kernel2(ydml_phy_mf%yrphy, nlon, klev, jstart, jend, testint)
    call kernel3(ydml_phy_mf%yrphy, nlon, klev, jstart, jend, pzz)

  end subroutine kernel1
end module kernel1_mod
    """.strip()

    fcode_kernel2 = """
module kernel2_mod
  contains
  subroutine kernel2(ydphy, nlon, klev, jstart, jend, testint)

      use parkind1, only: jpim, jprb

      use yomphy, only:  tphy

      implicit none

      type(tphy), intent(in) :: ydphy

      integer(kind=jpim), intent(in) :: nlon
      integer(kind=jpim), intent(in) :: klev
      integer(kind=jpim), intent(in) :: jstart
      integer(kind=jpim), intent(in) :: jend
      integer(kind=jpim), optional, intent(in) :: testint

      integer(kind=jpim) :: jb, jlev, jl

      real(kind=jprb) :: zde1(nlon, 0:klev, ydphy%n_spband)
      real(kind=jprb) :: zde2(nlon, klev, ydphy%n_spband)

      do jb = 1, ydphy%n_spband
        do jlev = 1, klev
          do jl = jstart, jend

            zde1(jl, jlev, jb) = 0._jprb
            zde2(jl, jlev, jb) = 0._jprb

          enddo
        enddo
      enddo

  end subroutine kernel2
end module kernel2_mod
    """.strip()

    fcode_kernel3 = """
module kernel3_mod
  contains
  subroutine kernel3(ydphy, nlon, klev, jstart, jend, pzz)

      use parkind1, only: jpim, jprb

      use yomphy, only:  tphy

      implicit none

      type(tphy), intent(in) :: ydphy

      integer(kind=jpim), intent(in) :: nlon
      integer(kind=jpim), intent(in) :: klev
      integer(kind=jpim), intent(in) :: jstart
      integer(kind=jpim), intent(in) :: jend

      real(kind=jprb), intent(in), dimension(nlon, klev) :: pzz

      integer(kind=jpim) :: jb, jlev, jl

      real(kind=jprb) :: zde1(nlon, 0:klev, ydphy%n_spband)
      real(kind=jprb) :: zde2(nlon, klev, ydphy%n_spband)
      real(kind=jprb) :: zde3(nlon, 1:klev)

!$acc data present(pzz)

      do jb = 1, ydphy%n_spband
        zde1(:, 0, jb) = 0._jprb
        zde2(:, :, jb) = 0._jprb
        do jlev = 1, klev
          do jl = jstart, jend

            zde1(jl, jlev, jb) = 1._jprb
            zde2(jl, jlev, jb) = 0._jprb

          enddo
        enddo
      enddo

      zde3 = pzz
      zde3(1:nlon,1:klev) = pzz

!$acc end data

  end subroutine kernel3
end module kernel3_mod
    """.strip()

    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel1_mod.F90').write_text(fcode_kernel1)
    (tmp_path/'kernel2_mod.F90').write_text(fcode_kernel2)
    (tmp_path/'kernel3_mod.F90').write_text(fcode_kernel3)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

    if frontend == OMNI:
        (tmp_path/'parkind_mod.F90').write_text(fcode_parkind_mod)
        parkind_mod = Sourcefile.from_file(tmp_path/'parkind_mod.F90', frontend=frontend, xmods=[tmp_path])
        (tmp_path/'yomphy_mod.F90').write_text(fcode_yomphy_mod)
        yomphy_mod = Sourcefile.from_file(tmp_path/'yomphy_mod.F90', frontend=frontend, xmods=[tmp_path])
        (tmp_path/'mf_phys_mod.F90').write_text(fcode_mf_phys_mod)
        mf_phys_mod = Sourcefile.from_file(tmp_path/'mf_phys_mod.F90', frontend=frontend, xmods=[tmp_path])
        definitions = parkind_mod.definitions + yomphy_mod.definitions + mf_phys_mod.definitions
    else:
        definitions = ()

    scheduler = Scheduler(paths=[tmp_path], config=SchedulerConfig.from_dict(config), frontend=frontend,
                          definitions=definitions, xmods=[tmp_path])

    transformation = TemporariesRawStackTransformation(block_dim=block_dim, horizontal=horizontal, directive=directive)
    scheduler.process(transformation=transformation)

    driver_item  = scheduler['driver_mod#driver']
    kernel1_item = scheduler['kernel1_mod#kernel1']
    kernel2_item = scheduler['kernel2_mod#kernel2']
    kernel3_item = scheduler['kernel3_mod#kernel3']

    assert transformation._key in kernel1_item.trafo_data

    jprb_stack_size = 'MAX(klev + ydml_phy_mf%yrphy%n_spband + 2*klev*ydml_phy_mf%yrphy%n_spband, '\
                        '2*klev + ydml_phy_mf%yrphy%n_spband + 2*klev*ydml_phy_mf%yrphy%n_spband)'
    srk_stack_size = 'MAX(2*klev + ydml_phy_mf%yrphy%n_spband + 2*klev*ydml_phy_mf%yrphy%n_spband, '\
                         '3*klev + ydml_phy_mf%yrphy%n_spband + 2*klev*ydml_phy_mf%yrphy%n_spband)'
    klev_stack_size = 'klev'

    real = BasicType.REAL
    logical = BasicType.LOGICAL
    jprb = DeferredTypeSymbol('JPRB')
    srk = InlineCall(function = DeferredTypeSymbol(name = 'SELECTED_REAL_KIND'),
                     parameters = (IntLiteral(13), IntLiteral(300)))

    stack_dict = kernel1_item.trafo_data[transformation._key]['stack_dict']

    assert real in stack_dict

    if frontend == OMNI:
        assert srk in stack_dict[real]
        assert fgen(stack_dict[real][srk]) == srk_stack_size
    else:
        assert jprb in stack_dict[real]
        assert fgen(stack_dict[real][jprb]) == jprb_stack_size
        assert srk in stack_dict[real]
        assert fgen(stack_dict[real][srk]) == klev_stack_size

    assert logical in stack_dict
    assert None in stack_dict[logical]
    assert fgen(stack_dict[logical][None]) == klev_stack_size

    driver = driver_item.ir
    kernel1 = kernel1_item.ir
    kernel2 = kernel2_item.ir
    kernel3 = kernel3_item.ir

    assert 'j_ll_stack_size' in driver.variable_map
    assert 'll_stack' in driver.variable_map

    assert 'j_z_selected_real_kind_13_300_stack_size' in driver.variable_map
    assert 'z_selected_real_kind_13_300_stack' in driver.variable_map

    if not frontend == OMNI:
        assert 'j_z_jprb_stack_size' in driver.variable_map
        assert 'z_jprb_stack' in driver.variable_map

    assert 'j_p_selected_real_kind_13_300_stack_used' in kernel1.variable_map
    assert 'k_p_selected_real_kind_13_300_stack_size' in kernel1.variable_map
    assert 'p_selected_real_kind_13_300_stack' in kernel1.variable_map

    assert 'j_ld_stack_used' in kernel1.variable_map
    assert 'k_ld_stack_size' in kernel1.variable_map
    assert 'ld_stack' in kernel1.variable_map

    if not frontend == OMNI:
        assert 'j_p_jprb_stack_used' in kernel1.variable_map
        assert 'k_p_jprb_stack_size' in kernel1.variable_map
        assert 'p_jprb_stack' in kernel1.variable_map

    assert 'jd_zzx' in kernel1.variable_map
    assert 'jd_zzy' in kernel1.variable_map
    assert 'jd_zzl' in kernel1.variable_map

    calls = FindNodes(CallStatement).visit(driver.body)

    if frontend == OMNI:
        assert fgen(calls[0].arguments).lower() == 'ydml_phy_mf\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'zzz\n'\
        'j_z_selected_real_kind_13_300_stack_size\n'\
        'z_selected_real_kind_13_300_stack(:, :, b)\n'\
        'j_ll_stack_size\n'\
        'll_stack(:, :, b)'
    else:
        assert fgen(calls[0].arguments).lower() == 'ydml_phy_mf\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'zzz\n'\
        'j_z_jprb_stack_size\n'\
        'z_jprb_stack(:, :, b)\n'\
        'j_z_selected_real_kind_13_300_stack_size\n'\
        'z_selected_real_kind_13_300_stack(:, :, b)\n'\
        'j_ll_stack_size\n'\
        'll_stack(:, :, b)'

    if frontend == OMNI:
        assert fgen(kernel1.arguments).lower() == 'ydml_phy_mf\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'pzz(nlon, klev)\n'\
        'k_p_selected_real_kind_13_300_stack_size\n'\
        'p_selected_real_kind_13_300_stack(nlon, k_p_selected_real_kind_13_300_stack_size)\n'\
        'k_ld_stack_size\n'\
        'ld_stack(nlon, k_ld_stack_size)'
    else:
        assert fgen(kernel1.arguments).lower() == 'ydml_phy_mf\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'pzz(nlon, klev)\n'\
        'k_p_jprb_stack_size\n'\
        'p_jprb_stack(nlon, k_p_jprb_stack_size)\n'\
        'k_p_selected_real_kind_13_300_stack_size\n'\
        'p_selected_real_kind_13_300_stack(nlon, k_p_selected_real_kind_13_300_stack_size)\n'\
        'k_ld_stack_size\n'\
        'ld_stack(nlon, k_ld_stack_size)'

    calls = FindNodes(CallStatement).visit(kernel1.body)

    if frontend == OMNI:
        assert fgen(calls[0].arguments).lower() == 'ydml_phy_mf%yrphy\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'k_p_selected_real_kind_13_300_stack_size - j_p_selected_real_kind_13_300_stack_used\n'\
        'p_selected_real_kind_13_300_stack'\
        '(1:nlon, j_p_selected_real_kind_13_300_stack_used + 1:k_p_selected_real_kind_13_300_stack_size)\n'\
        'testint'
    else:
        assert fgen(calls[0].arguments).lower() == 'ydml_phy_mf%yrphy\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'k_p_jprb_stack_size - j_p_jprb_stack_used\n'\
        'p_jprb_stack(1:nlon, j_p_jprb_stack_used + 1:k_p_jprb_stack_size)\n'\
        'testint'

    if frontend == OMNI:
        assert fgen(kernel2.arguments).lower() == 'ydphy\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'k_p_selected_real_kind_13_300_stack_size\n'\
        'p_selected_real_kind_13_300_stack(nlon, k_p_selected_real_kind_13_300_stack_size)\n'\
        'testint'
    else:
        assert fgen(kernel2.arguments).lower() == 'ydphy\n'\
        'nlon\n'\
        'klev\n'\
        'jstart\n'\
        'jend\n'\
        'k_p_jprb_stack_size\n'\
        'p_jprb_stack(nlon, k_p_jprb_stack_size)\n'\
        'testint'

    assignments = FindNodes(Assignment).visit(driver.body)

    lhs = [fgen(a.lhs).lower() for a in assignments]

    assert 'j_z_selected_real_kind_13_300_stack_size' in lhs
    assert 'j_ll_stack_size' in lhs
    if not frontend == OMNI:
        assert 'j_z_jprb_stack_size' in lhs

    for a in assignments:

        if fgen(a.lhs).lower() == 'j_z_selected_real_kind_13_300_stack_size':
            if frontend == OMNI:
                assert fgen(a.rhs).lower() == srk_stack_size.lower()
            else:
                assert fgen(a.rhs).lower() == klev_stack_size.lower()

        if fgen(a.lhs).lower() == 'j_ll_stack_size':
            assert fgen(a.rhs).lower() == klev_stack_size.lower()

        if fgen(a.lhs).lower() == 'j_z_jprb_stack_size':
            assert fgen(a.rhs).lower() == jprb_stack_size.lower()

    assignments = FindNodes(Assignment).visit(kernel3.body)

    assert assignments[0].lhs == 'jd_zde1'
    assert assignments[0].rhs == '0'

    assert assignments[1].lhs == 'jd_zde2'
    assert assignments[1].rhs == 'jd_zde1 + ydphy%n_spband + klev*ydphy%n_spband'

    assert assignments[2].lhs == 'jd_zde3'
    assert assignments[2].rhs == 'jd_zde2 + klev*ydphy%n_spband'

    if frontend == OMNI:
        assert assignments[3].lhs == 'j_p_selected_real_kind_13_300_stack_used'
        assert assignments[3].rhs == 'jd_zde3 + klev'

        assert assignments[4].lhs == 'p_selected_real_kind_13_300_stack(:, jd_zde1 + jb - klev + jb*klev)'
        assert fgen(assignments[4].rhs) == '0._jprb'  # Need fgen for kind specified

        assert assignments[5].lhs == 'p_selected_real_kind_13_300_stack'\
            '(:, jd_zde2 + 1 - klev + jb*klev:jd_zde2 + jb*klev)'
        assert fgen(assignments[5].rhs) == '0._jprb'

        assert assignments[6].lhs == 'p_selected_real_kind_13_300_stack'\
            '(jl, jd_zde1 + jlev + jb - klev + jb*klev)'
        assert fgen(assignments[6].rhs) == '1._jprb'

        assert assignments[7].lhs == 'p_selected_real_kind_13_300_stack'\
            '(jl, jd_zde2 + jlev - klev + jb*klev)'
        assert fgen(assignments[7].rhs) == '0._jprb'

        assert assignments[8].lhs == 'p_selected_real_kind_13_300_stack'\
            '(1:nlon, jd_zde3 + 1:jd_zde3 + klev)'
        assert assignments[8].rhs == 'pzz'

        assert assignments[9].lhs == 'p_selected_real_kind_13_300_stack'\
            '(1:nlon, jd_zde3 + 1:jd_zde3 + klev)'
        assert assignments[9].rhs == 'pzz'
    else:
        assert assignments[3].lhs == 'j_p_jprb_stack_used'
        assert assignments[3].rhs == 'jd_zde3 + klev'

        assert assignments[4].lhs == 'p_jprb_stack(:, jd_zde1 + jb - klev + jb*klev)'
        assert fgen(assignments[4].rhs) == '0._jprb'  # Need fgen for kind specified

        assert assignments[5].lhs == 'p_jprb_stack(:, jd_zde2 + 1 - klev + jb*klev:jd_zde2 + jb*klev)'
        assert fgen(assignments[5].rhs) == '0._jprb'

        assert assignments[6].lhs == 'p_jprb_stack(jl, jd_zde1 + jlev + jb - klev + jb*klev)'
        assert fgen(assignments[6].rhs) == '1._jprb'

        assert assignments[7].lhs == 'p_jprb_stack(jl, jd_zde2 + jlev - klev + jb*klev)'
        assert fgen(assignments[7].rhs) == '0._jprb'

        assert assignments[8].lhs == 'p_jprb_stack(1:nlon, jd_zde3 + 1:jd_zde3 + klev)'
        assert assignments[8].rhs == 'pzz'

        assert assignments[9].lhs == 'p_jprb_stack(1:nlon, jd_zde3 + 1:jd_zde3 + klev)'
        assert assignments[9].rhs == 'pzz'

    if directive in ['openacc', 'openmp']:
        pragmas = FindNodes(Pragma).visit(driver.body)

        if directive == 'openacc':
            if frontend == OMNI:
                assert pragmas[0].content.lower() == 'data create(z_selected_real_kind_13_300_stack, ll_stack)'
            else:
                assert pragmas[0].content.lower() == 'data create(z_jprb_stack, '\
                                                     'z_selected_real_kind_13_300_stack, ll_stack)'

        if directive == 'openmp':
            if frontend == OMNI:
                assert pragmas[0].content.lower() == 'target allocate(z_selected_real_kind_13_300_stack, ll_stack)'
            else:
                assert pragmas[0].content.lower() == 'target allocate(z_jprb_stack, '\
                                                     'z_selected_real_kind_13_300_stack, ll_stack)'

    if directive == 'openacc':
        pragmas = FindNodes(Pragma).visit(kernel1.body)
        if frontend == OMNI:
            assert pragmas[0].content.lower() == 'data present(p_selected_real_kind_13_300_stack, ld_stack)'
        else:
            assert pragmas[0].content.lower() == 'data present(p_jprb_stack, '\
                                                 'p_selected_real_kind_13_300_stack, ld_stack)'

        pragmas = FindNodes(Pragma).visit(kernel3.body)
        if frontend == OMNI:
            assert pragmas[0].content.lower() == 'data present(p_selected_real_kind_13_300_stack, pzz)'
        else:
            assert pragmas[0].content.lower() == 'data present(p_jprb_stack, pzz)'
