# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki.batch import Scheduler, SchedulerConfig
from loki.dimension import Dimension
from loki.expression import parse_expr
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, nodes as ir
from loki.sourcefile import Sourcefile
from loki.transformations.pragma_model import PragmaModelTransformation

from loki.transformations.temporaries import PoolAllocatorFtrPtrTransformation, PoolAllocatorRawTransformation

@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(name='block_dim', size='nb', index='b')

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('jstart', 'jend'))

@pytest.mark.parametrize('directive', ['openacc', 'omp-gpu'])
@pytest.mark.parametrize('stack_trafo', [PoolAllocatorFtrPtrTransformation, PoolAllocatorRawTransformation])
@pytest.mark.parametrize('frontend', available_frontends())
def test_raw_stack_allocator_temporaries(frontend, block_dim, horizontal, directive, stack_trafo, tmp_path):

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

!$loki device-present vars(pzz)

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

!$loki end device-present

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
            'strict': True,
            'ignore': ['parkind1', 'model_physics_mf_mod', 'yomphy'],
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

    transformation = stack_trafo(block_dim=block_dim, horizontal=horizontal)
    scheduler.process(transformation=transformation)
    pragma_model_trafo = PragmaModelTransformation(directive=directive)
    scheduler.process(transformation=pragma_model_trafo)

    driver_item  = scheduler['driver_mod#driver']
    kernel1_item = scheduler['kernel1_mod#kernel1']
    kernel2_item = scheduler['kernel2_mod#kernel2']
    kernel3_item = scheduler['kernel3_mod#kernel3']

    driver = driver_item.ir
    kernel1 = kernel1_item.ir
    kernel2 = kernel2_item.ir
    kernel3 = kernel3_item.ir

    directive_keyword_map = {'openacc': 'acc', 'omp-gpu': 'omp'}
    driver_var_map = driver.variable_map
    stack_size_vars = ['j_z_jprb_stack_size', 'j_z_selected_real_kind_13_300_stack_size', 'j_ll_stack_size']
    stack_used_vars = ['j_z_jprb_stack_used', 'j_z_selected_real_kind_13_300_stack_used', 'j_ll_stack_used']
    stack_vars = ['z_jprb_stack', 'll_stack', 'z_selected_real_kind_13_300_stack']
    stack_vars_size = {
        'z_jprb_stack': ('MAX(klev*nlon + nlon*ydml_phy_mf%yrphy%n_spband + '
                         '2*klev*nlon*ydml_phy_mf%yrphy%n_spband, 2*klev*nlon + '
                         'nlon*ydml_phy_mf%yrphy%n_spband + 2*klev*nlon*ydml_phy_mf%yrphy%n_spband)'),
        'll_stack': 'klev*nlon',
        'z_selected_real_kind_13_300_stack': 'klev*nlon'
    }

    for stack_var in stack_vars:
        assert stack_var in driver_var_map
        assert driver_var_map[stack_var].type.allocatable
        assert len(driver_var_map[stack_var].dimensions) == 2

    driver_allocs = FindNodes(ir.Allocation).visit(driver.body)
    for driver_alloc in driver_allocs:
        var = driver_alloc.variables[0]
        assert var.name.lower() in stack_vars
        assert var.dimensions[1] == 'nb'
        assert var.dimensions[0] == parse_expr(stack_vars_size[var.name.lower()])

    driver_deallocs = FindNodes(ir.Deallocation).visit(driver.body)
    for driver_dealloc in driver_deallocs:
        var = driver_dealloc.variables[0]
        assert var.name.lower() in stack_vars
        assert var.dimensions == ()

    driver_pragmas = FindNodes(ir.Pragma).visit(driver.body)
    assert len(driver_pragmas) == 2
    assert driver_pragmas[0].keyword.lower() == directive_keyword_map[directive]
    # target enter data map(alloc: z_jprb_stack, z_selected_real_kind_13_300_stack, ll_stack)
    assert driver_pragmas[1].keyword.lower() == directive_keyword_map[directive]
    if directive == 'openacc':
        assert 'enter data create' in driver_pragmas[0].content.lower()
        assert 'exit data delete' in driver_pragmas[1].content.lower()
    if directive == 'omp-gpu':
        assert 'target enter data map(alloc:' in driver_pragmas[0].content.lower()
        assert 'target exit data map(delete:' in driver_pragmas[1].content.lower()
    for stack_var in stack_vars:
        assert stack_var in driver_pragmas[0].content.lower()
        assert stack_var in driver_pragmas[1].content.lower()

    driver_calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(driver_calls) == 1
    driver_arg_map = {v.name.lower(): k for k,v in driver_calls[0].arg_map.items()}
    for stack_var in stack_vars:
        assert stack_var in driver_arg_map
    for stack_size_var in stack_size_vars:
        assert stack_size_var in driver_arg_map
    for stack_used_var in stack_used_vars:
        assert stack_used_var in driver_arg_map

    jprb_stack = {'size': 'k_p_jprb_stack_size', 'stack': 'p_jprb_stack', 'used': 'jd_p_jprb_stack_used'}
    selected_real_kind_stack = {'size': 'k_p_selected_real_kind_13_300_stack_size',
                                'stack': 'p_selected_real_kind_13_300_stack',
                                'used': 'jd_p_selected_real_kind_13_300_stack_used'}
    l_stack = {'size': 'k_ld_stack_size', 'stack': 'ld_stack', 'used': 'jd_ld_stack_used'}

    kernel1_args = [arg.name.lower() for arg in kernel1.arguments]
    kernel2_args = [arg.name.lower() for arg in kernel2.arguments]
    kernel3_args = [arg.name.lower() for arg in kernel3.arguments]
    for var in list(jprb_stack.values()) + list(selected_real_kind_stack.values()) + list(l_stack.values()):
        assert var in kernel1_args
    for var in jprb_stack.values():
        assert var in kernel2_args
        assert var in kernel3_args
    for var in list(selected_real_kind_stack.values()) + list(l_stack.values()):
        assert var not in kernel2_args
        assert var not in kernel3_args

    if stack_trafo == PoolAllocatorFtrPtrTransformation:
        kernel1_incr_vars = {'jprb': ('jd_incr_jprb',),
                             'selected_real': ('jd_incr_selected_real_kind_13_300',), 'l': ('jd_incr',)}
        kernel2_incr_vars = {'jprb': ('jd_incr_jprb',)}
        kernel3_incr_vars = {'jprb': ('jd_incr_jprb',)}
    else: # PoolAllocatorRawTransformation
        kernel1_incr_vars = {'jprb': ('jd_zzx',), 'selected_real': ('jd_zzy',), 'l': ('jd_zzl',)}
        kernel2_incr_vars = {'jprb': ('jd_zde1', 'jd_zde2')}
        kernel3_incr_vars = {'jprb': ('jd_zde1', 'jd_zde2', 'jd_zde3')}
    for kernel1_incr_var in kernel1_incr_vars['jprb'] + kernel1_incr_vars['selected_real'] + kernel1_incr_vars['l']:
        assert kernel1_incr_var in kernel1.variables
    for var in kernel2_incr_vars['jprb']:
        assert var in kernel2.variables
    for var in kernel3_incr_vars['jprb']:
        assert var in kernel3.variables
    kernels_stack_used_args = {'jprb': 'jd_p_jprb_stack_used',
                               'selected_real': 'jd_p_selected_real_kind_13_300_stack_used',
                               'l': 'jd_ld_stack_used'}
    kernels_stack_used_vars = {'jprb': 'j_p_jprb_stack_used',
                               'selected_real': 'j_p_selected_real_kind_13_300_stack_used',
                               'l': 'j_ld_stack_used'}
    kernel1_assignments_map = {}
    for assign in FindNodes(ir.Assignment).visit(kernel1.body):
        kernel1_assignments_map.setdefault(assign.lhs.name.lower(), []).append(assign.rhs)
    for key, kernels_stack_used_var in kernels_stack_used_vars.items():
        assert kernel1_assignments_map[kernels_stack_used_var][0] == kernels_stack_used_args[key]
        assert kernel1_assignments_map[kernel1_incr_vars[key][0]][0] == kernels_stack_used_var
