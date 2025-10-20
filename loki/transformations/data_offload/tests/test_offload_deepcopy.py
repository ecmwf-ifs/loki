# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree
import pytest
import yaml

from loki.backend import fgen
from loki.batch import Scheduler
from loki.ir import nodes as ir, FindNodes, is_loki_pragma, pragma_regions_attached, get_pragma_parameters
from loki.expression import Variable, RangeIndex, IntLiteral
from loki.frontend import available_frontends
from loki.logging import log_levels
from loki.scope import Scope
from loki.subroutine import Subroutine
from loki.tools import gettempdir, flatten, as_tuple
from loki.transformations import DataOffloadDeepcopyAnalysis, DataOffloadDeepcopyTransformation, find_driver_loops
from loki.types import BasicType, DerivedType, SymbolAttributes


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

   type :: other_variable_type
      class(field_3d), pointer :: f_t0 => null()
      real, pointer, contiguous :: vt0_field(:,:,:) => null()
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
      integer :: m
   end type

   type :: geom_dims
      integer :: nproma
   end type

   type :: geom_type
      type(geom_dims) :: dim
      integer, pointer :: metadata(:)
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
subroutine kernel(geometry, bnds, struct, variable)
   use nested_kernel_write_mod, only: nested_kernel_write
   use nested_kernel_read_mod, only: nested_kernel_read
   use other_kernel_mod, only : other_kernel
   use type_def_mod, only: struct_type, dims_type, geom_type, other_variable_type
   implicit none

   type(geom_type), intent(in) :: geometry
   type(dims_type), intent(in) :: bnds
   type(struct_type), intent(inout) :: struct
   type(other_variable_type), intent(inout) :: variable

   integer :: jrof, jfld, j
   real, pointer :: tmp(:,:,:) => null()
   integer :: a(geometry%dim%nproma)

   call nested_kernel_write(struct%a%p(:,:,bnds%kbl))
   call nested_kernel_read(struct%b%p(:,:,bnds%kbl))
   call nested_kernel_read(variable%vt0_field(:,:,bnds%kbl))

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

   j = geometry%metadata(1)

end subroutine kernel
end module kernel_mod
            """.strip()
        ),

        #----- driver -----
        'driver' : (
            """
subroutine driver(dims, struct, array_arg, geometry, variable)
   use kernel_mod, only : kernel
   use nested_kernel_write_mod, only: nested_kernel_write
   use type_def_mod, only: struct_type, dims_type, geom_type, other_variable_type
   use iso_fortran_env, only : real64
   implicit none

   type(dims_type), intent(in) :: dims
   type(struct_type), intent(inout) :: struct
   integer, intent(out) :: array_arg(:,:,:)
   type(geom_type), intent(in) :: geometry
   type(other_variable_type), intent(inout) :: variable
   type(dims_type) :: local_dims
   integer :: ibl, ij

   local_dims = dims

#ifdef geometry_present
!$loki data private(local_dims) present(geometry) write(struct%c%p)
   do ibl=1,local_dims%ngpblks
     local_dims%kbl = ibl
     ij = 0

     variable%vt0_field(:,:,ibl) = 0._real64

     call kernel(geometry, local_dims, struct, variable)
     call nested_kernel_write(struct%e%p(:,local_dims%m,local_dims%kbl))
     call nested_kernel_write(array_arg)
   enddo
!$loki end data
#else
!$loki data private(local_dims) write(struct%c%p)
   do ibl=1,local_dims%ngpblks
     local_dims%kbl = ibl
     ij = 0

     variable%vt0_field(:,:,ibl) = 0._real64

     call kernel(geometry, local_dims, struct, variable)
     call nested_kernel_write(struct%e%p(:,local_dims%m,local_dims%kbl))
     call nested_kernel_write(array_arg)
   enddo
!$loki end data
#endif

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
            'kbl': 'write',
            'kend': 'read',
            'kst': 'read',
            'ngpblks': 'read',
            'm': 'read'
        },
        'geometry': {
            'dim': {
                'nproma': 'read'
            },
            'metadata' : 'read'
        },
        'array_arg': 'write',
        'variable' : {
            'vt0_field' : 'write'
        },
        'struct': {
            'a': {
                'p': 'write'
            },
            'b': {
                'p': 'readwrite'
            },
            'c': {
                'p': 'read'
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
        },
        'ij': 'write'
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
            'field_ptr_suffix': '_field',
            'field_ptr_map': {}
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
                sorted_dict[k] = v

        return dict(sorted(sorted_dict.items()))

    config['routines'] = {
        'driver': {'role': 'driver'},
        'variable_type': {'field_prefix': 'f'}
    }

    scheduler = Scheduler(
        paths=deepcopy_code, config=config, frontend=frontend, xmods=[tmp_path],
        output_dir=tmp_path, preprocess=True
    )

    with caplog.at_level(log_levels['WARNING']):
        transformation = DataOffloadDeepcopyAnalysis(output_analysis=output_analysis)
        scheduler.process(transformation=transformation)

        # check that the warning for pointer associations is produced
        messages = [log.message for log in caplog.records]
        assert '[Loki::DataOffloadDeepcopyAnalysis] Pointer associations found in kernel' in messages[-1]

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


def check_array_arg(mode, pragmas):
    """Check the correct generation of deepcopy for `array_arg`."""

    if mode == 'offload':
        copy_pragma = [(p, loc) for loc, p in enumerate(pragmas)
                       if 'unstructured-data create' in p.content and '(array_arg)' in p.content for p in pragmas]
        host_pragma = [(p, loc) for loc, p in enumerate(pragmas)
                       if 'update host' in p.content and '(array_arg)' in p.content for p in pragmas]
        assert copy_pragma
        assert host_pragma
        assert host_pragma[0][1] > copy_pragma[0][1]
    else:
        assert not any('array_arg' in p.content for p in pragmas)


def check_variable_type_host(var, conds):
    """Check generated host pull-back for `type(variable_type)` variables."""

    conds = [c for c in conds
             if c.condition.name.lower() == 'associated' and f'{var}%fp' in c.condition.parameters]
    calls = FindNodes(ir.CallStatement).visit(conds)
    assert any(call.name.name.lower() == f'{var}%fp%get_host_data_rdwr' and f'{var}%p' in call.arguments
               for call in calls)

def check_variable_type_device(var, conds, access):
    """Check generated copy to device for `type(variable_type)` variables."""

    conds = [c for c in conds
             if c.condition.name.lower() == 'associated' and f'{var}%fp' in c.condition.parameters]

    _pass = 0
    for cond in conds:
        calls = FindNodes(ir.CallStatement).visit(cond.body)
        pragmas = FindNodes(ir.Pragma).visit(cond.body)

        calls = [call for call in calls
                 if call.name.name.lower() == f'{var}%fp%get_device_data_{access}' and f'{var}%p' in call.arguments]
        pragmas = [pragma for pragma in pragmas
                   if 'unstructured-data attach' in pragma.content and f'{var}%p' in pragma.content]
        if calls and pragmas:
            assert cond.body.index(calls[0]) < cond.body.index(pragmas[0])
            _pass += 1

    assert _pass == 1


def check_variable_type_wipe(var, conds):
    """Check generated wipe for `type(variable_type)` variables."""

    conds = [c for c in conds
             if c.condition.name.lower() == 'associated' and f'{var}%fp' in c.condition.parameters]
    _pass = 0
    for cond in conds:
        calls = FindNodes(ir.CallStatement).visit(cond.body)
        pragmas = FindNodes(ir.Pragma).visit(cond.body)

        calls = [call for call in calls
                 if call.name.name.lower() == f'{var}%fp%delete_device_data']
        pragmas = [pragma for pragma in pragmas
                   if 'end unstructured-data detach' in pragma.content and f'{var}%p' in pragma.content
                   and 'finalize' in pragma.content]
        if calls and pragmas:
            assert cond.body.index(calls[0]) > cond.body.index(pragmas[0])
            _pass += 1

    assert _pass == 1


def check_other_variable_type(mode, conds, pragmas, routine):
    """Check the generated deepcopy for `type(other_variable_type) :: variable`."""

    # Check pullback to host
    conds = [c for c in conds
             if c.condition.name.lower() == 'associated' and 'variable%f_t0' in c.condition.parameters]
    calls = FindNodes(ir.CallStatement).visit(conds)

    assert any(call.name.name.lower() == 'variable%f_t0%get_host_data_rdwr' and 'variable%vt0_field' in call.arguments
               for call in calls)

    if mode == 'offload':
        # Check copy to device of struct
        pragma = [p for p in pragmas if
                  'unstructured-data in' in p.content and '(variable)' in p.content][0]
        assert routine.body.body.index(pragma) < routine.body.body.index(conds[0])

        # Check deletion of struct from device
        pragma = [p for p in pragmas if
                  'end unstructured-data delete' in p.content and '(variable)' in p.content][0]
        assert routine.body.body.index(pragma) > routine.body.body.index(conds[-1])

        # Check FIELD_API boilerplate for copying to device and wiping device
        _pass = 0
        for cond in conds:
            calls = FindNodes(ir.CallStatement).visit(cond.body)
            pragmas = FindNodes(ir.Pragma).visit(cond.body)

            calls = [call for call in calls
                     if call.name.name.lower() == 'variable%f_t0%delete_device_data']
            pragmas = [pragma for pragma in pragmas
                       if 'end unstructured-data detach' in pragma.content and 'variable%vt0_field' in pragma.content
                       and 'finalize' in pragma.content]
            if calls and pragmas:
                assert cond.body.index(calls[0]) > cond.body.index(pragmas[0])
                _pass += 1

        for cond in conds:
            calls = FindNodes(ir.CallStatement).visit(cond.body)
            pragmas = FindNodes(ir.Pragma).visit(cond.body)

            calls = [call for call in calls
                     if call.name.name.lower() == 'variable%f_t0%get_device_data_wronly'
                     and 'variable%vt0_field' in call.arguments]
            pragmas = [pragma for pragma in pragmas
                       if 'unstructured-data attach' in pragma.content and 'variable%vt0_field' in pragma.content]
            if calls and pragmas:
                assert cond.body.index(calls[0]) < cond.body.index(pragmas[0])
                _pass += 1

        assert _pass == 2


def check_geometry(conds, pragmas, routine):
    """Check the generated deepcopy for `type(geom_type) :: geometry`."""

    conds = [c for c in conds
             if 'geometry%metadata' in c.condition.parameters]

    assert all(c.condition.name.lower() == 'associated' for c in conds)

    # geometry should only have copy and wipe related instructions
    assert len(conds) == 2

    # Check copy to device of struct
    pragma = [p for p in pragmas if
              'unstructured-data in' in p.content and '(geometry)' in p.content][0]
    assert routine.body.body.index(pragma) < routine.body.body.index(conds[0])

    # Check copy to device of member
    assert 'unstructured-data in' in conds[0].body[0].content
    assert '(geometry%metadata)' in conds[0].body[0].content

    # Check deletion of struct from device
    pragma = [p for p in pragmas if 'end unstructured-data delete' in p.content
              and '(geometry)' in p.content and 'finalize' in p.content][0]
    assert routine.body.body.index(pragma) > routine.body.body.index(conds[-1])

    # Check deletion of member from device
    assert 'end unstructured-data delete' in conds[-1].body[0].content
    assert '(geometry%metadata)' in conds[-1].body[0].content
    assert 'finalize' in conds[-1].body[0].content


def check_struct(mode, conds, pragmas, routine):
    """Check the generated deepcopy for `type(struct_type)`."""

    # Filter out conditions on type(struct_type) :: struct
    struct_conds = []
    for cond in conds:
        parameters = flatten([p.name.lower() for p in cond.condition.parameters])
        if any('struct' in p for p in parameters):
            struct_conds.append(cond)

    # Check var_ptr member
    check_struct_var_ptr(mode, struct_conds)

    # Check host pull-back
    check_variable_type_host('struct%a', struct_conds)
    check_variable_type_host('struct%b', struct_conds)
    check_variable_type_host('struct%c', struct_conds)
    check_variable_type_host('struct%d', struct_conds)
    check_variable_type_host('struct%e', struct_conds)

    if mode == 'offload':
        # Check copy to device of struct
        pragma = [p for p in pragmas if
                  'unstructured-data in' in p.content and '(struct)' in p.content][0]
        assert routine.body.body.index(pragma) < routine.body.body.index(struct_conds[0])

        # Check deletion of struct from device
        pragma = [p for p in pragmas if
                  'end unstructured-data delete' in p.content and '(struct)' in p.content][0]
        assert routine.body.body.index(pragma) > routine.body.body.index(struct_conds[-1])

        check_variable_type_device('struct%a', struct_conds, 'wronly')
        check_variable_type_device('struct%b', struct_conds, 'rdwr')
        check_variable_type_device('struct%c', struct_conds, 'wronly')
        check_variable_type_device('struct%d', struct_conds, 'wronly')
        check_variable_type_device('struct%e', struct_conds, 'rdwr')

        check_variable_type_wipe('struct%a', struct_conds)
        check_variable_type_wipe('struct%b', struct_conds)
        check_variable_type_wipe('struct%c', struct_conds)
        check_variable_type_wipe('struct%d', struct_conds)
        check_variable_type_wipe('struct%e', struct_conds)


def check_struct_var_ptr(mode, conds):
    """Check the `var_ptr` member of `type(struct_type) :: struct`."""

    # First check host pull-back FIELD_API calls
    conds = [c for c in conds
             if c.condition.name.lower() == 'allocated' and 'struct%var_ptr' in c.condition.parameters]
    loops = FindNodes(ir.Loop).visit(conds)
    calls = FindNodes(ir.CallStatement).visit(loops)

    assert any(fgen(call.name).lower() == 'struct%var_ptr(j1)%var%fp%get_host_data_rdwr'
               and 'struct%var_ptr(J1)%var%p' in call.arguments for call in calls)

    if mode == 'offload':
        _pass = 0
        for cond in conds:
            calls = FindNodes(ir.CallStatement).visit(cond.body)

            if any('get_device_data' in call.name.name.lower() for call in calls):
                # first entry in conditional body should be copyin pragma
                assert 'unstructured-data in' in cond.body[0].content
                assert '(struct%var_ptr)' in cond.body[0].content

                # then we have a loop and an association check for a field object
                loop = FindNodes(ir.Loop).visit(cond.body)[0]
                _cond = FindNodes(ir.Conditional).visit(loop.body)[0]
                assert _cond.condition.name.lower() == 'associated'
                assert fgen(_cond.condition.parameters[0]).lower() == 'struct%var_ptr(j1)%var%fp'

                # finally inside the conditional body we have a FIELD_API GET_DEVICE_DATA call
                # and an attach statement
                assert fgen(_cond.body[0].name).lower() == 'struct%var_ptr(j1)%var%fp%get_device_data_rdwr'
                assert _cond.body[0].arguments[0] == 'struct%var_ptr(j1)%var%p'
                assert 'unstructured-data attach' in _cond.body[1].content
                assert 'struct%var_ptr(J1)%var%p' in _cond.body[1].content
                _pass += 1

            elif any('delete_device_data' in call.name.name.lower() for call in calls):
                # last entry in conditional body should be delete pragma
                assert 'end unstructured-data delete' in cond.body[-1].content
                assert 'finalize' in cond.body[-1].content
                assert '(struct%var_ptr)' in cond.body[-1].content

                # then we have a loop and an association check for a field object
                loop = FindNodes(ir.Loop).visit(cond.body)[0]
                _cond = FindNodes(ir.Conditional).visit(loop.body)[0]
                assert _cond.condition.name.lower() == 'associated'
                assert fgen(_cond.condition.parameters[0]).lower() == 'struct%var_ptr(j1)%var%fp'

                # finally inside the conditional body we have a FIELD_API DELETE_DEVICE_DATA call
                # preceded by an detach statement
                assert 'end unstructured-data detach' in _cond.body[0].content
                assert 'finalize' in _cond.body[0].content
                assert 'struct%var_ptr(J1)%var%p' in _cond.body[0].content
                assert fgen(_cond.body[1].name).lower() == 'struct%var_ptr(j1)%var%fp%delete_device_data'
                _pass += 1

        assert _pass == 2


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('present', [True, False])
@pytest.mark.parametrize('mode', ['offload', 'set_pointers'])
def test_offload_deepcopy_transformation(frontend, config, deepcopy_code, present, mode, tmp_path):
    """
    Test the generation of host-device deepcopy.
    """

    config['routines'] = {
        'driver': {'role': 'driver'},
        'variable_type': {
            'field_prefix': 'f',
            'field_ptrs': ['p'],
        },
        'other_variable_type': {
            'field_prefix': 'f_',
            'view_prefix': 'v',
            'field_ptr_map': {
                'vt0_field': 'f_t0'
            }
        }
    }

    defines = ['geometry_present'] if present else []
    scheduler = Scheduler(
        paths=deepcopy_code, config=config, frontend=frontend, xmods=[tmp_path],
        output_dir=tmp_path, preprocess=True, defines=defines
    )

    scheduler.process(transformation=DataOffloadDeepcopyAnalysis())
    transformation = DataOffloadDeepcopyTransformation(mode=mode)
    scheduler.process(transformation=transformation)

    driver = scheduler['#driver'].ir
    pragmas = FindNodes(ir.Pragma).visit(driver.body)
    conds = FindNodes(ir.Conditional, greedy=True).visit(driver.body)

    # check array_arg
    check_array_arg(mode, pragmas)

    # check other_variable_type
    check_other_variable_type(mode, conds, pragmas, driver)

    # check struct
    check_struct(mode, conds, pragmas, driver)

    if not present and mode == 'offload':
        check_geometry(conds, pragmas, driver)

    if mode == 'offload':
        # check data present region
        with pragma_regions_attached(driver):

            assert not any('update host' in p.content and '(ij)' in p.content for p in pragmas)

            region = FindNodes(ir.PragmaRegion).visit(driver.body)[-1]
            assert is_loki_pragma(region.pragma, starts_with='structured-data')

            parameters = get_pragma_parameters(region.pragma)
            present_vars = [v.strip().lower() for v in parameters['present'].split(',')]
            assert all(v in present_vars for v in ['geometry', 'struct', 'variable'])


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_loop_nest_wrapping(frontend, depth):
    """Test the utility to wrap a given body in a loop nest."""

    fcode = """
subroutine kernel()
  implicit none
end subroutine kernel
    """

    def _check_loops(loop, depth, loop_level):
        assert isinstance(loop, ir.Loop)

        # check loop bounds
        assert fgen(loop.bounds.lower).lower() == f'lbound(var, {loop_level})'
        assert fgen(loop.bounds.upper).lower() == f'ubound(var, {loop_level})'

        # check loop index
        assert loop.variable.name.lower() == f'j{loop_level}'

        if loop_level > 1:
            _check_loops(loop.body[0], depth, loop_level - 1)
        else:
            assert isinstance(loop.body[0], ir.Assignment)
            assert fgen(loop.body[0].lhs).lower() == f"var({', '.join(f'j{d+1}' for d in range(depth))})"
            assert loop.body[0].rhs == '0'
            assert isinstance(loop.body[1], ir.Pragma)
            assert loop.body[1].keyword == 'loki'
            assert loop.body[1].content.lower() == f"update device( var({', '.join(f'j{d+1}' for d in range(depth))}) )"


    routine = Subroutine.from_source(fcode, frontend=frontend)
    shape = as_tuple([RangeIndex((None, None))] * depth)
    var = Variable(name='var', type=SymbolAttributes(BasicType.INTEGER, shape=shape), dimensions=shape, scope=routine)

    routine.variables += (var,)
    assign = ir.Assignment(lhs=var.clone(dimensions=None), rhs=IntLiteral(0)) # pylint:disable=no-member
    pragma = ir.Pragma(keyword='loki', content='update device( var )')
    body = (assign, pragma)

    trafo = DataOffloadDeepcopyTransformation(mode='offload')
    loopnest = trafo.wrap_in_loopnest(var.clone(dimensions=None), body, routine) # pylint:disable=no-member

    # check loop variable was added to routine
    variables = routine.variables
    assert all(f'J{d+1}' in variables for d in range(depth))

    # check loops are correctly nested
    _check_loops(loopnest[0], depth, depth)


@pytest.mark.parametrize('rank', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('suff', ['im', 'rb', 'rd', 'lm'])
def test_dummy_field_array_typdef_config(rank, suff):
    """Test the creation of a typedef config for ``FIELD_RANKSUFF_ARRAY`` types."""

    scope = Scope()
    typedef = ir.TypeDef(name=f'field_{rank}{suff}_array', parent=scope) # pylint: disable=unexpected-keyword-arg
    _type = SymbolAttributes(DerivedType(typedef=typedef))
    var = Variable(name='field_array', scope=scope, type=_type)

    trafo = DataOffloadDeepcopyTransformation(mode='offload')
    typedef_config = trafo.create_dummy_field_array_typedef_config(var)

    # check typedef config was created correctly
    ref_config = {
        'field_prefix': 'F_',
        'field_ptr_suffix': '_FIELD',
        'field_ptr_map': {}
    }
    assert typedef_config == ref_config
