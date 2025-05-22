# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree
import pytest
import yaml

from loki import (
        gettempdir, available_frontends, Scheduler, DataOffloadDeepcopyAnalysis,
        find_driver_loops, log_levels
)


@pytest.fixture(scope='module', name='deepcopy_code')
def fixture_deepcopy_code():
    fcode = {
        #----- field_module -----
        'field_module' : (
            """
module field_module
type :: field_3d
end type
end module field_module
            """.strip()
        ),

        #----- type_mod -----
        'type_def_mod' : (
            """
module type_def_mod
   use field_module, only : field_3d
   type :: variable_type
      class(field_3d), pointer :: fp => null()
      real, pointer, contiguous :: p(:,:,:) => null()
   end type

   type :: superfluous_type
      type(variable_type) :: var
   end type

   type :: struct_type
      type(variable_type) :: a
      type(variable_type) :: b
      type(variable_type) :: c
      type(variable_type) :: d
      type(variable_type) :: e
      type(superfluous_type), allocatable :: var_ptr(:)
   end type

   type :: opts_type
      logical :: one_flag
      logical :: another_flag
   end type

   type :: dims_type
      integer :: kst
      integer :: kend
      integer :: kbl
      integer :: ngpblks
   end type
end module type_def_mod
            """.strip()
        ),

        #----- nested_kernel_write -----
        'nested_kernel_write' : (
            """
module nested_kernel_write_mod
contains
subroutine nested_kernel_write(p)
    !... intent(out) can be dangerous with pointers, so we make this intent(inout)
    real, intent(inout) :: p(:,:,:)

    p = 0.
end subroutine nested_kernel_write
end module nested_kernel_write_mod
            """.strip()
        ),

        #----- nested_kernel_read -----
        'nested_kernel_read' : (
            """
module nested_kernel_read_mod
contains
subroutine nested_kernel_read(p)
    real, intent(in) :: p(:,:,:)
    real, allocatable :: b(:,:,:)

    allocate(b, mold=p)
    b = p
    deallocate(b)
end subroutine nested_kernel_read
end module nested_kernel_read_mod
            """.strip()
        ),

        #----- other_kernel -----
        'other_kernel' : (
            """
module other_kernel_mod
contains
subroutine other_kernel(struct)
   use type_def_mod, only : variable_type
   type(variable_type), intent(inout) :: struct

   struct%p = struct%p + 1.
end subroutine other_kernel
end module other_kernel_mod
            """.strip()
        ),

        #----- kernel -----
        'kernel' : (
            """
module kernel_mod
contains
subroutine kernel(bnds, struct)
   use nested_kernel_write_mod, only: nested_kernel_write
   use nested_kernel_read_mod, only: nested_kernel_read
   use other_kernel_mod, only : other_kernel
   use type_def_mod, only: struct_type, dims_type
   implicit none

   type(dims_type), intent(in) :: bnds
   type(struct_type), intent(inout) :: struct

   integer :: jrof, jfld
   real, pointer :: tmp(:,:,:) => null()

   call nested_kernel_write(struct%a%p(:,:,bnds%kbl))
   call nested_kernel_read(struct%b%p(:,:,bnds%kbl))

   tmp => struct%c%p !... yes this completely breaks the dataflow analysis
   tmp = 0.

   do jrof = bnds%kst, bnds%kend
     struct%b%p(jrof,:,bnds%kbl) = struct%a%p(jrof,:,bnds%kbl)
   enddo 

   do jfld = 1, size(struct%var_ptr)
     call other_kernel(struct%var_ptr(jfld)%var)
   enddo

   do jrof = bnds%kst, bnds%kend
     struct%d%p(jrof,:,bnds%kbl) = struct%e%p(jrof,:,bnds%kbl)
   enddo 

end subroutine kernel
end module kernel_mod
            """.strip()
        ),

        #----- driver -----
        'driver' : (
            """
subroutine driver(dims, struct)
   use kernel_mod, only : kernel
   use nested_kernel_write_mod, only: nested_kernel_write
   use type_def_mod, only: struct_type, dims_type
   implicit none

   type(dims_type), intent(in) :: dims
   type(struct_type), intent(inout) :: struct
   type(dims_type) :: local_dims
   integer :: ibl

   local_dims = dims

   do ibl=1,local_dims%ngpblks
     local_dims%kbl = ibl

     call kernel(local_dims, struct)
     call nested_kernel_write(struct%e%p(:,:,local_dims%kbl))
   enddo

end subroutine driver
            """.strip()
        )
    }

    workdir = gettempdir()/'test_offload_deepcopy'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    for name, code in fcode.items():
        (workdir/f'{name}.F90').write_text(code)

    yield workdir

    rmtree(workdir)


@pytest.fixture(scope='module', name='expected_analysis')
def fixture_expected_analysis():
    return {
        'local_dims': {
            'kbl': 'read',
            'kend': 'read',
            'kst': 'read'
        },
        'struct': {
            'a': {
                'p': 'write'
            },
            'b': {
                'p': 'readwrite'
            },
            'c': {
                'p': 'readwrite'
            },
            'd': {
                'p': 'write'
            },
            'e': {
                'p': 'readwrite'
            },
            'var_ptr': {
                'var': {
                    'p': 'readwrite'
                }
            }
        }
    }


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
        },
    }


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('output_analysis', [True, False])
def test_offload_deepcopy_analysis(frontend, config, deepcopy_code, expected_analysis,
                                   output_analysis, tmp_path, caplog):
    """
    Test the analysis for the data offload deepcopy generation.
    """

    def _nested_sort(_dict):
        sorted_dict = {}
        for k, v in _dict.items():
            if isinstance(v, dict):
                sorted_dict[k] = _nested_sort(v)
            else:
                sorted_dict[k] = k

        return dict(sorted(sorted_dict.items()))

    config['routines'] = {
        'driver': {'role': 'driver'},
        'variable_type': {'field_prefix': 'f'}
    }

    scheduler = Scheduler(
        paths=deepcopy_code, config=config, frontend=frontend, xmods=[tmp_path],
        output_dir=tmp_path
    )

    with caplog.at_level(log_levels['WARNING']):
        transformation = DataOffloadDeepcopyAnalysis(output_analysis=output_analysis)
        scheduler.process(transformation=transformation)

        # check that the warning for pointer associations is produced
        messages = [log.message for log in caplog.records]
        assert '[Loki::DataOffloadDeepcopyAnalysis] Pointer associations found in kernel' in messages

    # The analysis is tied to driver loops
    trafo_data_key = transformation._key
    driver_item = scheduler['#driver']
    driver_loop = find_driver_loops(driver_item.ir.body, targets=['kernel', 'nested_kernel_write'])[0]

    #stringify dict for comparison
    stringified_dict = transformation.stringify_dict(driver_item.trafo_data[trafo_data_key]['analysis'][driver_loop])
    sorted_expected_analysis = _nested_sort(expected_analysis)
    assert _nested_sort(stringified_dict) == sorted_expected_analysis

    # check that the typedef config was also collected
    assert driver_item.trafo_data[trafo_data_key]['typedef_configs']['variable_type']['field_prefix'] == 'f'

    if output_analysis:
        with open(tmp_path/'driver_kernel_dataoffload_analysis.yaml', 'r') as file:
            _dict = yaml.safe_load(file)
        assert _nested_sort(_dict) == sorted_expected_analysis
