# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree
import pytest

from loki import (
    Dimension, gettempdir, Scheduler, OMNI, FindNodes, Assignment, FindVariables, CallStatement, Subroutine,
    Item, available_frontends
)
from loki.transformations import BlockViewToFieldViewTransformation, InjectBlockIndexTransformation

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'),
                     aliases=('nproma',), bounds_aliases=('bnds%start', 'bnds%end'))


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='ibl', index_aliases='bnds%kbl')


@pytest.fixture(scope='function', name='config')
def fixture_config():
    """
    Default configuration dict with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': True,
            'disable': ['*%init', '*%final']
        },
    }


@pytest.fixture(scope='module', name='blockview_to_fieldview_code', params=[True, False])
def fixture_blockview_to_fieldview_code(request):
    fcode = {
        #-------------
        'variable_mod': (
        #-------------
"""
module variable_mod
  implicit none

  type variable_3d
      real, pointer :: p(:,:) => null()
      real, pointer :: p_field(:,:,:) => null()
  end type variable_3d

  type variable_3d_ptr
      integer :: comp
      type(variable_3d), pointer :: ptr => null()
  end type variable_3d_ptr

end module variable_mod
"""
        ).strip(),
        #--------------------
        'field_variables_mod': (
        #--------------------
"""
module field_variables_mod
  use variable_mod, only: variable_3d, variable_3d_ptr
  implicit none

  type field_variables
      type(variable_3d_ptr), allocatable :: gfl_ptr_g(:)
      type(variable_3d_ptr), pointer :: gfl_ptr(:) => null()
      type(variable_3d) :: var
  end type field_variables

end module field_variables_mod
"""
        ).strip(),
        #-------------------
        'container_type_mod': (
        #-------------------
"""
module container_type_mod
  implicit none

  type container_3d_var
    real, pointer :: p(:,:) => null()
    real, pointer :: p_field(:,:,:) => null()
  end type container_3d_var

  type container_type
    type(container_3d_var), allocatable :: vars(:)
  end type container_type

end module container_type_mod
"""
        ).strip(),
        #--------------
        'dims_type_mod': (
        #--------------
"""
module dims_type_mod
   type dims_type
      integer :: start, end, kbl, nb
   end type dims_type
end module dims_type_mod
"""
        ).strip(),
        #-------
        'driver': (
        #-------
f"""
subroutine driver(data, ydvars, container, nlon, nlev, {'start, end, nb' if request.param else 'bnds'})
   use field_array_module, only: field_3rb_array
   use container_type_mod, only: container_type
   use field_variables_mod, only: field_variables
   {'use dims_type_mod, only: dims_type' if not request.param else ''}
   implicit none

   #include "kernel.intfb.h"

   real, intent(inout) :: data(:,:,:)
   integer, intent(in) :: nlon, nlev
   type(field_variables), intent(inout) :: ydvars
   type(container_type), intent(inout) :: container
   {'integer, intent(in) :: start, end, nb' if request.param else 'type(dims_type), intent(in) :: bnds'}

   integer :: ibl
   type(field_3rb_array) :: yla_data

   call yla_data%init(data)

   do ibl=1,{'nb' if request.param else 'bnds%nb'}
      {'bnds%kbl = ibl' if not request.param else ''}
      call kernel(nlon, nlev, {'start, end, ibl' if request.param else 'bnds'}, ydvars, container, yla_data)
   enddo

   call yla_data%final()

end subroutine driver
"""
        ).strip(),
        #-------
        'kernel': (
        #-------
f"""
subroutine kernel(nlon, nlev, {'start, end, ibl' if request.param else 'bnds'}, ydvars, container, yla_data)
   use field_array_module, only: field_3rb_array
   use container_type_mod, only: container_type
   use field_variables_mod, only: field_variables
   {'use dims_type_mod, only: dims_type' if not request.param else ''}
   implicit none

   #include "another_kernel.intfb.h"

   integer, intent(in) :: nlon, nlev
   type(field_variables), intent(inout) :: ydvars
   type(container_type), intent(inout) :: container
   {'integer, intent(in) :: start, end, ibl' if request.param else 'type(dims_type), intent(in) :: bnds'}
   type(field_3rb_array), intent(inout) :: yda_data

   integer :: jl, jfld
   {'associate(start=>bnds%start, end=>bnds%end, ibl=>bnds%kbl)' if not request.param else ''}

   ydvars%var%p_field(:,:) = 0. !... this should only get the block-index
   ydvars%var%p_field(:,:,ibl) = 0. !... this should be untouched

   yda_data%p(start:end,:) = 1
   ydvars%var%p(start:end,:) = 1

   do jfld=1,size(ydvars%gfl_ptr)
      do jl=start,end
         ydvars%gfl_ptr(jfld)%ptr%p(jl,:) = yda_data%p(jl,:)
         container%vars(ydvars%gfl_ptr(jfld)%comp)%p(jl,:) = 0.
      enddo
   enddo

   call another_kernel(nlon, nlev, data=yda_data%p)

   {'end associate' if not request.param else ''}
end subroutine kernel
"""
        ).strip(),
        #-------
        'another_kernel': (
        #-------
"""
subroutine another_kernel(nproma, nlev, data)
   implicit none
   !... not a sequential routine but still labelling it as one to test the
   !... bail-out mechanism
   !$loki routine seq
   integer, intent(in) :: nproma, nlev
   real, intent(inout) :: data(nproma, nlev)
end subroutine another_kernel
"""
        ).strip()
    }

    workdir = gettempdir()/'test_blockview_to_fieldview'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    for name, code in fcode.items():
        (workdir/f'{name}.F90').write_text(code)

    yield workdir, request.param

    rmtree(workdir)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI fails to import undefined module.')]))
def test_blockview_to_fieldview_pipeline(horizontal, blocking, config, frontend, blockview_to_fieldview_code, tmp_path):

    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(
        paths=(blockview_to_fieldview_code[0],), config=config, seed_routines='driver', frontend=frontend,
        xmods=[tmp_path]
    )
    scheduler.process(BlockViewToFieldViewTransformation(horizontal, global_gfl_ptr=True))
    scheduler.process(InjectBlockIndexTransformation(blocking))

    kernel = scheduler['#kernel'].ir
    aliased_bounds = not blockview_to_fieldview_code[1]
    ibl_expr = blocking.index
    if aliased_bounds:
        ibl_expr = blocking.index_expressions[1]

    assigns = FindNodes(Assignment).visit(kernel.body)

    # check that access pointers for arrays without horizontal index in dimensions were not updated
    assert assigns[0].lhs == f'ydvars%var%p_field(:,:,{ibl_expr})'
    assert assigns[1].lhs == f'ydvars%var%p_field(:,:,{ibl_expr})'

    # check that vector notation was resolved correctly
    assert assigns[2].lhs == f'yda_data%p_field(jl, :, {ibl_expr})'
    assert assigns[3].lhs == f'ydvars%var%p_field(jl, :, {ibl_expr})'

    # check thread-local ydvars%gfl_ptr was replaced with its global equivalent
    gfl_ptr_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr' in v.name.lower()}
    gfl_ptr_g_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr_g' in v.name.lower()}
    assert gfl_ptr_g_vars
    assert not gfl_ptr_g_vars - gfl_ptr_vars

    assert assigns[4].lhs == f'ydvars%gfl_ptr_g(jfld)%ptr%p_field(jl,:,{ibl_expr})'
    assert assigns[4].rhs == f'yda_data%p_field(jl,:,{ibl_expr})'
    assert assigns[5].lhs == f'container%vars(ydvars%gfl_ptr_g(jfld)%comp)%p_field(jl,:,{ibl_expr})'

    # check callstatement was updated correctly
    call = FindNodes(CallStatement).visit(kernel.body)[0]
    assert f'yda_data%p_field(:,:,{ibl_expr})' in call.arg_map.values()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI fails to import undefined module.')]))
@pytest.mark.parametrize('global_gfl_ptr', [False, True])
def test_blockview_to_fieldview_only(horizontal, blocking, config, frontend, blockview_to_fieldview_code,
                                     global_gfl_ptr, tmp_path):

    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(
        paths=(blockview_to_fieldview_code[0],), config=config, seed_routines='driver', frontend=frontend,
        xmods=[tmp_path]
    )
    scheduler.process(BlockViewToFieldViewTransformation(horizontal, global_gfl_ptr=global_gfl_ptr))

    kernel = scheduler['#kernel'].ir
    aliased_bounds = not blockview_to_fieldview_code[1]
    ibl_expr = blocking.index
    if aliased_bounds:
        ibl_expr = blocking.index_expressions[1]

    assigns = FindNodes(Assignment).visit(kernel.body)

    # check that access pointers for arrays without horizontal index in dimensions were not updated
    assert assigns[0].lhs == 'ydvars%var%p_field(:,:)'
    assert assigns[1].lhs == f'ydvars%var%p_field(:,:,{ibl_expr})'

    # check that vector notation was resolved correctly
    assert assigns[2].lhs == 'yda_data%p_field(jl, :)'
    assert assigns[3].lhs == 'ydvars%var%p_field(jl, :)'

    # check thread-local ydvars%gfl_ptr was replaced with its global equivalent
    if global_gfl_ptr:
        gfl_ptr_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr' in v.name.lower()}
        gfl_ptr_g_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr_g' in v.name.lower()}
        assert gfl_ptr_g_vars
        assert not gfl_ptr_g_vars - gfl_ptr_vars
    else:
        assert not {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr_g' in v.name.lower()}

    assert assigns[4].rhs == 'yda_data%p_field(jl,:)'
    if global_gfl_ptr:
        assert assigns[4].lhs == 'ydvars%gfl_ptr_g(jfld)%ptr%p_field(jl,:)'
        assert assigns[5].lhs == 'container%vars(ydvars%gfl_ptr_g(jfld)%comp)%p_field(jl,:)'
    else:
        assert assigns[4].lhs == 'ydvars%gfl_ptr(jfld)%ptr%p_field(jl,:)'
        assert assigns[5].lhs == 'container%vars(ydvars%gfl_ptr(jfld)%comp)%p_field(jl,:)'

    # check callstatement was updated correctly
    call = FindNodes(CallStatement).visit(kernel.body)[0]
    assert 'yda_data%p_field' in call.arg_map.values()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI correctly complains about rank mismatch in assignment.')]))
def test_simple_blockindex_inject(blocking, frontend):
    fcode = """
subroutine kernel(nlon,nlev,nb,var)
  implicit none

  interface
    subroutine compute(nlon,nlev,var)
      implicit none
      integer, intent(in) :: nlon,nlev
      real, intent(inout) :: var(nlon,nlev)
    end subroutine compute
  end interface

  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,nlev,4,nb) !... this dummy arg was potentially promoted by a previous transformation

  integer :: ibl

  do ibl=1,nb !... this loop was potentially lowered by a previous transformation
     var(:,:,:) = 0.
     call compute(nlon,nlev,var(:,:,1))
  enddo

end subroutine kernel
"""

    kernel = Subroutine.from_source(fcode, frontend=frontend)
    InjectBlockIndexTransformation(blocking).apply(kernel, role='kernel', targets=('compute',))

    assigns = FindNodes(Assignment).visit(kernel.body)
    assert assigns[0].lhs == 'var(:,:,:,ibl)'

    calls = FindNodes(CallStatement).visit(kernel.body)
    assert 'var(:,:,1,ibl)' in calls[0].arguments


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI complains about undefined type.')]))
def test_blockview_to_fieldview_exception(frontend, horizontal):
    fcode = """
subroutine kernel(nlon,nlev,start,end,var)
  implicit none

  interface
    subroutine compute(nlon,nlev,var)
      implicit none
      integer, intent(in) :: nlon,nlev
      real, intent(inout) :: var(nlon,nlev)
    end subroutine compute
  end interface

  integer, intent(in) :: nlon,nlev,start,end
  type(wrapped_field) :: var

  call compute(nlon,nlev,var%p)

end subroutine kernel
"""

    kernel = Subroutine.from_source(fcode, frontend=frontend)
    item = Item(name='#kernel', source=kernel)
    item.trafo_data['BlockViewToFieldViewTransformation'] = {'definitions': []}
    with pytest.raises(KeyError):
        BlockViewToFieldViewTransformation(horizontal).apply(kernel, item=item, role='kernel',
                                           targets=('compute',))

    with pytest.raises(RuntimeError):
        BlockViewToFieldViewTransformation(horizontal).apply(kernel, role='kernel',
                                           targets=('compute',))
