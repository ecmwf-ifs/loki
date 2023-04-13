# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree

import pytest
from loki import (
    gettempdir, Scheduler, SchedulerConfig, Dimension, simplify, Sourcefile,
    FindNodes, FindVariables, normalize_range_indexing, OMNI, FP, get_pragma_parameters,
    CallStatement, Assignment, Allocation, Deallocation, Loop, InlineCall, Pragma
)
from conftest import available_frontends

from transformations.pool_allocator import TemporariesPoolAllocatorTransformation


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(name='block_dim', size='nb', index='b')


def check_stack_module_import(routine):
    assert any(import_.module.lower() == 'stack_mod' for import_ in routine.imports)
    assert 'stack' in routine.imported_symbols


def check_stack_created_in_driver(driver, stack_size, first_kernel_call, num_block_loops, generate_driver_stack=True):
    # Are stack size, storage and stack derived type declared?
    assert 'istsz' in driver.variables
    assert 'zstack(:,:)' in driver.variables
    assert 'ylstack' in driver.variables

    # Is there an allocation and deallocation for the stack storage?
    allocations = FindNodes(Allocation).visit(driver.body)
    assert len(allocations) == 1 and 'zstack(istsz,nb)' in allocations[0].variables
    deallocations = FindNodes(Deallocation).visit(driver.body)
    assert len(deallocations) == 1 and 'zstack' in deallocations[0].variables

    # Check the stack size
    assignments = FindNodes(Assignment).visit(driver.body)
    for assignment in assignments:
        if assignment.lhs == 'istsz':
            assert str(simplify(assignment.rhs)).lower().replace(' ', '') == str(stack_size).lower().replace(' ', '')
            break

    # Check for stack assignment inside loop
    loops = FindNodes(Loop).visit(driver.body)
    assert len(loops) == num_block_loops
    assignments = FindNodes(Assignment).visit(loops[0].body)
    assert len(assignments) == 2
    assert assignments[0].lhs == 'ylstack%l'
    assert isinstance(assignments[0].rhs, InlineCall) and assignments[0].rhs.function == 'loc'
    assert 'zstack(1, b)' in assignments[0].rhs.parameters
    if generate_driver_stack:
        assert assignments[1].lhs == 'ylstack%u' and assignments[1].rhs == 'ylstack%l + istsz * 8'
    else:
        assert assignments[1].lhs == 'ylstack%u' and assignments[1].rhs == 'ylstack%l + 8 * istsz'

    # Check that stack assignment happens before kernel call
    assert all(loops[0].body.index(a) < loops[0].body.index(first_kernel_call) for a in assignments)


@pytest.mark.parametrize('generate_driver_stack', [False, True])
@pytest.mark.parametrize('frontend', available_frontends())
def test_pool_allocator_temporaries(frontend, generate_driver_stack, block_dim):
    fcode_stack_mod = """
MODULE STACK_MOD
IMPLICIT NONE
TYPE STACK
  INTEGER*8 :: L, U
END TYPE
PRIVATE
PUBLIC :: STACK
END MODULE
    """.strip()

    fcode_stack_import = "use stack_mod, only: stack"
    fcode_stack_decl = """
    integer :: istsz
    REAL(KIND=JPRB), ALLOCATABLE :: ZSTACK(:, :)
    type(stack) :: ylstack

    istsz = (nz+1) * nlon
    ALLOCATE(ZSTACK(ISTSZ, nb))
    """
    fcode_stack_assign = """
        ylstack%l = loc(zstack(1, b))
        ylstack%u = ylstack%l + 8 * istsz
    """
    fcode_stack_dealloc = "DEALLOCATE(ZSTACK)"

    fcode_driver = f"""
subroutine driver(NLON, NZ, NB, FIELD1, FIELD2)
    {fcode_stack_import if not generate_driver_stack else ''}
    use kernel_mod, only: kernel
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    INTEGER, INTENT(IN) :: NLON, NZ, NB
    real(kind=jprb), intent(inout) :: field1(nlon, nb)
    real(kind=jprb), intent(inout) :: field2(nlon, nz, nb)
    integer :: b
    {fcode_stack_decl if not generate_driver_stack else ''}
    do b=1,nb
        {fcode_stack_assign if not generate_driver_stack else ''}
        call KERNEL(1, nlon, nlon, nz, field1(:,b), field2(:,:,b))
    end do
    {fcode_stack_dealloc if not generate_driver_stack else ''}
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(start, end, klon, klev, field1, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: start, end, klon, klev
        real(kind=jprb), intent(inout) :: field1(klon)
        real(kind=jprb), intent(inout) :: field2(klon,klev)
        real(kind=jprb) :: tmp1(klon)
        real(kind=jprb) :: tmp2(klon, klev)
        integer :: jk, jl

        do jk=1,klev
            tmp1(jl) = 0.0_jprb
            do jl=start,end
                tmp2(jl, jk) = field2(jl, jk)
                tmp1(jl) = field2(jl, jk)
            end do
            field1(jl) = tmp1(jl)
        end do
    end subroutine kernel
end module kernel_mod
    """.strip()

    basedir = gettempdir()/'test_pool_allocator_temporaries'
    basedir.mkdir(exist_ok=True)
    (basedir/'driver.F90').write_text(fcode_driver)
    (basedir/'kernel_mod.F90').write_text(fcode_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routine': [{
            'name': 'driver',
            'role': 'driver',
        }]
    }

    if frontend == FP and not generate_driver_stack:
        # Patch "LOC" intrinsic into fparser. This is not strictly needed (it will just represent it
        # as Array instead of an InlineCall) but makes for a more coherent check further down
        from fparser.two import Fortran2003  # pylint: disable=import-outside-toplevel
        Fortran2003.Intrinsic_Name.other_inquiry_names.update({"LOC": {'min': 1, 'max': 1}})
        Fortran2003.Intrinsic_Name.generic_function_names.update({"LOC": {'min': 1, 'max': 1}})
        Fortran2003.Intrinsic_Name.function_names += ["LOC"]

    if frontend == OMNI:
        (basedir/'stack_mod.F90').write_text(fcode_stack_mod)
        stack_mod = Sourcefile.from_file(basedir/'stack_mod.F90', frontend=frontend)
        definitions = stack_mod.definitions
    else:
        definitions = ()

    scheduler = Scheduler(
        paths=[basedir], config=SchedulerConfig.from_dict(config),
        definitions=definitions, frontend=frontend
    )

    if frontend == OMNI:
        for item in scheduler.items:
            normalize_range_indexing(item.routine)

    transformation = TemporariesPoolAllocatorTransformation(block_dim=block_dim)
    scheduler.process(transformation=transformation, reverse=True)
    kernel_item = scheduler['kernel_mod#kernel']

    assert transformation._key in kernel_item.trafo_data
    assert kernel_item.trafo_data[transformation._key] == 'klon + klev * klon'
    assert all(v.scope is None for v in FindVariables().visit(kernel_item.trafo_data[transformation._key]))

    #
    # A few checks on the driver
    #
    driver = scheduler['#driver'].routine

    # Has the stack module been imported?
    check_stack_module_import(driver)

    # Has the stack been added to the call statement?
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('1', 'nlon', 'nlon', 'nz', 'field1(:,b)', 'field2(:,:,b)', 'ylstack')

    check_stack_created_in_driver(driver, 'nlon + nlon * nz', calls[0], 1, generate_driver_stack)

    #
    # A few checks on the kernel
    #
    kernel = kernel_item.routine

    # Has the stack module been imported?
    check_stack_module_import(kernel)

    # Has the stack been added to the arguments?
    assert 'ydstack' in kernel.arguments

    # Is it being assigned to a local variable?
    assert 'ylstack' in kernel.variables

    # Let's check for the relevant "allocations" happening in the right order
    assign_idx = {}
    for idx, assign in enumerate(FindNodes(Assignment).visit(kernel.body)):
        if assign.lhs == 'ylstack' and assign.rhs == 'ydstack':
            # Local copy of stack status
            assign_idx['stack_assign'] = idx
        elif assign.lhs == 'ip_tmp1' and assign.rhs == 'ylstack%l':
            # Assign Cray pointer for tmp1
            assign_idx['tmp1_ptr_assign'] = idx
        elif assign.lhs == 'ip_tmp2' and assign.rhs == 'ylstack%l':
            # Assign Cray pointer for tmp2
            assign_idx['tmp2_ptr_assign'] = idx
        elif assign.lhs == 'ylstack%l' and 'ylstack%l' in assign.rhs and 'size' in assign.rhs and 'tmp1' in assign.rhs:
            # Stack increment for tmp1
            assign_idx['tmp1_stack_incr'] = idx
        elif assign.lhs == 'ylstack%l' and 'ylstack%l' in assign.rhs and 'tmp2' in assign.rhs and 'tmp2' in assign.rhs:
            # Stack increment for tmp2
            assign_idx['tmp2_stack_incr'] = idx

    expected_assign_in_order = [
        'stack_assign', 'tmp1_ptr_assign', 'tmp1_stack_incr', 'tmp2_ptr_assign', 'tmp2_stack_incr'
    ]
    assert set(expected_assign_in_order) == set(assign_idx.keys())

    for assign1, assign2 in zip(expected_assign_in_order, expected_assign_in_order[1:]):
        assert assign_idx[assign2] > assign_idx[assign1]

    # Check for pointer declarations in generated code
    fcode = kernel.to_fortran()
    assert 'pointer(ip_tmp1, tmp1)' in fcode.lower()
    assert 'pointer(ip_tmp2, tmp2)' in fcode.lower()

    # Check for stack size safegurads in generated code
    assert fcode.lower().count('if (ylstack%l > ylstack%u)') == 2
    assert fcode.lower().count('stop') == 2

    rmtree(basedir)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('directive', [None, 'openmp', 'openacc'])
def test_pool_allocator_temporaries_kernel_sequence(frontend, block_dim, directive):
    if directive == 'openmp':
        driver_loop_pragma1 = '!$omp parallel default(shared) private(b)\n    !$omp do'
        driver_end_loop_pragma1 = '!$omp end do\n    !$omp end parallel'
        driver_loop_pragma2 = '!$omp parallel do'
        driver_end_loop_pragma2 = '!$omp end parallel do'
        kernel_pragma = ''
    elif directive == 'openacc':
        driver_loop_pragma1 = '!$acc parallel loop gang private(b)'
        driver_end_loop_pragma1 = '!$acc end parallel loop'
        driver_loop_pragma2 = '!$acc parallel loop gang'
        driver_end_loop_pragma2 = '!$acc end parallel loop'
        kernel_pragma = '!$acc routine vector'
    else:
        driver_loop_pragma1 = ''
        driver_end_loop_pragma1 = ''
        driver_loop_pragma2 = ''
        driver_end_loop_pragma2 = ''
        kernel_pragma = ''

    fcode_driver = f"""
subroutine driver(NLON, NZ, NB, FIELD1, FIELD2)
    use kernel_mod, only: kernel, kernel2
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    INTEGER, INTENT(IN) :: NLON, NZ, NB
    real(kind=jprb), intent(inout) :: field1(nlon, nb)
    real(kind=jprb), intent(inout) :: field2(nlon, nz, nb)
    integer :: b

    {driver_loop_pragma1}
    do b=1,nb
        call KERNEL(1, nlon, nlon, nz, field1(:,b), field2(:,:,b))
    end do
    {driver_end_loop_pragma1}

    {driver_loop_pragma2}
    do b=1,nb
        call KERNEL2(1, nlon, nlon, nz, field2(:,:,b))
    end do
    {driver_end_loop_pragma2}
end subroutine driver
    """.strip()

    fcode_kernel = f"""
module kernel_mod
    implicit none
contains
    subroutine kernel(start, end, klon, klev, field1, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: start, end, klon, klev
        real(kind=jprb), intent(inout) :: field1(klon)
        real(kind=jprb), intent(inout) :: field2(klon,klev)
        real(kind=jprb) :: tmp1(klon)
        real(kind=jprb) :: tmp2(klon, klev)
        integer :: jk, jl
        {kernel_pragma}

        do jk=1,klev
            tmp1(jl) = 0.0_jprb
            do jl=start,end
                tmp2(jl, jk) = field2(jl, jk)
                tmp1(jl) = field2(jl, jk)
            end do
            field1(jl) = tmp1(jl)
        end do
    end subroutine kernel

    subroutine kernel2(start, end, klon, klev, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: start, end, klon, klev
        real(kind=jprb), intent(inout) :: field2(klon,klev)
        real(kind=jprb) :: tmp1(klon, klev), tmp2(klon, klev)
        integer :: jk, jl

        do jk=1,klev
            do jl=start,end
                tmp1(jl, jk) = field2(jl, jk)
                tmp2(jl, jk) = tmp1(jl, jk) + 1._jprb
                field2(jl, jk) = tmp2(jl, jk)
            end do
        end do
    end subroutine kernel2

end module kernel_mod
    """.strip()

    basedir = gettempdir()/'test_pool_allocator_temporaries_kernel_sequence'
    basedir.mkdir(exist_ok=True)
    (basedir/'driver.F90').write_text(fcode_driver)
    (basedir/'kernel_mod.F90').write_text(fcode_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routine': [{
            'name': 'driver',
            'role': 'driver',
        }]
    }
    scheduler = Scheduler(paths=[basedir], config=SchedulerConfig.from_dict(config), frontend=frontend)
    if frontend == OMNI:
        for item in scheduler.items:
            normalize_range_indexing(item.routine)

    transformation = TemporariesPoolAllocatorTransformation(block_dim=block_dim, directive=directive)
    scheduler.process(transformation=transformation, reverse=True)
    kernel_item = scheduler['kernel_mod#kernel']
    kernel2_item = scheduler['kernel_mod#kernel2']

    assert transformation._key in kernel_item.trafo_data
    assert kernel_item.trafo_data[transformation._key] == 'klon + klev * klon'
    assert kernel2_item.trafo_data[transformation._key] == '2 * klev * klon'
    assert all(v.scope is None for v in FindVariables().visit(kernel_item.trafo_data[transformation._key]))
    assert all(v.scope is None for v in FindVariables().visit(kernel2_item.trafo_data[transformation._key]))

    #
    # A few checks on the driver
    #
    driver = scheduler['#driver'].routine

    # Has the stack module been imported?
    check_stack_module_import(driver)

    # Has the stack been added to the call statements?
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 2
    assert calls[0].arguments == ('1', 'nlon', 'nlon', 'nz', 'field1(:,b)', 'field2(:,:,b)', 'ylstack')
    assert calls[1].arguments == ('1', 'nlon', 'nlon', 'nz', 'field2(:,:,b)', 'ylstack')

    check_stack_created_in_driver(driver, 'max(nlon + nlon * nz, 2 * nz * nlon)', calls[0], 2)

    # Has the data sharing been updated?
    if directive in ['openmp', 'openacc']:
        keyword = {'openmp': 'omp', 'openacc': 'acc'}[directive]
        pragmas = [
            p for p in FindNodes(Pragma).visit(driver.body)
            if p.keyword.lower() == keyword and p.content.startswith('parallel')
        ]
        assert len(pragmas) == 2
        for pragma in pragmas:
            parameters = get_pragma_parameters(pragma, starts_with='parallel', only_loki_pragmas=False)
            assert 'private' in parameters and 'ylstack' in parameters['private'].lower()

    # Are there data regions for the stack?
    if directive == ['openacc']:
        pragmas = [
            p for p in FindNodes(Pragma).visit(driver.body)
            if p.keyword.lower() == 'acc' and 'data' in p.content
        ]
        assert len(pragmas) == 2
        parameters = get_pragma_parameters(pragmas[0], starts_with='data', only_loki_pragmas=False)
        assert parameters['create'] == 'zstack'

    #
    # A few checks on the kernel
    #
    for item in [kernel_item, kernel2_item]:
        kernel = item.routine

        # Has the stack module been imported?
        check_stack_module_import(kernel)

        # Has the stack been added to the arguments?
        assert 'ydstack' in kernel.arguments

        # Is it being assigned to a local variable?
        assert 'ylstack' in kernel.variables

        # Let's check for the relevant "allocations" happening in the right order
        assign_idx = {}
        for idx, ass in enumerate(FindNodes(Assignment).visit(kernel.body)):
            if ass.lhs == 'ylstack' and ass.rhs == 'ydstack':
                # Local copy of stack status
                assign_idx['stack_assign'] = idx
            elif ass.lhs == 'ip_tmp1' and ass.rhs == 'ylstack%l':
                # ass Cray pointer for tmp1
                assign_idx['tmp1_ptr_assign'] = idx
            elif ass.lhs == 'ip_tmp2' and ass.rhs == 'ylstack%l':
                # ass Cray pointer for tmp2
                assign_idx['tmp2_ptr_assign'] = idx
            elif ass.lhs == 'ylstack%l' and 'ylstack%l' in ass.rhs and 'size' in ass.rhs and 'tmp1' in ass.rhs:
                # Stack increment for tmp1
                assign_idx['tmp1_stack_incr'] = idx
            elif ass.lhs == 'ylstack%l' and 'ylstack%l' in ass.rhs and 'tmp2' in ass.rhs and 'tmp2' in ass.rhs:
                # Stack increment for tmp2
                assign_idx['tmp2_stack_incr'] = idx

        expected_assign_in_order = [
            'stack_assign', 'tmp1_ptr_assign', 'tmp1_stack_incr', 'tmp2_ptr_assign', 'tmp2_stack_incr'
        ]
        assert set(expected_assign_in_order) == set(assign_idx.keys())

        for assign1, assign2 in zip(expected_assign_in_order, expected_assign_in_order[1:]):
            assert assign_idx[assign2] > assign_idx[assign1]

        # Check for pointer declarations in generated code
        fcode = kernel.to_fortran()
        assert 'pointer(ip_tmp1, tmp1)' in fcode.lower()
        assert 'pointer(ip_tmp2, tmp2)' in fcode.lower()

        # Check for stack size safegurads in generated code
        assert fcode.lower().count('if (ylstack%l > ylstack%u)') == 2
        assert fcode.lower().count('stop') == 2

    rmtree(basedir)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('directive', [None, 'openmp', 'openacc'])
def test_pool_allocator_temporaries_kernel_nested(frontend, block_dim, directive):
    if directive == 'openmp':
        driver_pragma = '!$omp parallel do'
        driver_end_pragma = '!$omp end parallel do'
        kernel_pragma = ''
    elif directive == 'openacc':
        driver_pragma = '!$acc parallel loop gang'
        driver_end_pragma = '!$acc end parallel loop'
        kernel_pragma = '!$acc routine vector'
    else:
        driver_pragma = ''
        driver_end_pragma = ''
        kernel_pragma = ''

    fcode_driver = f"""
subroutine driver(NLON, NZ, NB, FIELD1, FIELD2)
    use kernel_mod, only: kernel
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    INTEGER, INTENT(IN) :: NLON, NZ, NB
    real(kind=jprb), intent(inout) :: field1(nlon, nb)
    real(kind=jprb), intent(inout) :: field2(nlon, nz, nb)
    integer :: b
    {driver_pragma}
    do b=1,nb
        call KERNEL(1, nlon, nlon, nz, field1(:,b), field2(:,:,b))
    end do
    {driver_end_pragma}
end subroutine driver
    """.strip()
    fcode_kernel = f"""
module kernel_mod
    implicit none
contains
    subroutine kernel(start, end, klon, klev, field1, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: start, end, klon, klev
        real(kind=jprb), intent(inout) :: field1(klon)
        real(kind=jprb), intent(inout) :: field2(klon,klev)
        real(kind=jprb) :: tmp1(klon)
        real(kind=jprb) :: tmp2(klon, klev)
        integer :: jk, jl
        {kernel_pragma}

        do jk=1,klev
            tmp1(jl) = 0.0_jprb
            do jl=start,end
                tmp2(jl, jk) = field2(jl, jk)
                tmp1(jl) = field2(jl, jk)
            end do
            field1(jl) = tmp1(jl)
        end do

        call kernel2(start, end, klon, klev, field2)
    end subroutine kernel

    subroutine kernel2(start, end, columns, levels, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: start, end, columns, levels
        real(kind=jprb), intent(inout) :: field2(columns,levels)
        real(kind=jprb) :: tmp1(columns, levels), tmp2(columns, levels)
        integer :: jk, jl
        {kernel_pragma}

        do jk=1,levels
            do jl=start,end
                tmp1(jl, jk) = field2(jl, jk)
                tmp2(jl, jk) = tmp1(jl, jk) + 1._jprb
                field2(jl, jk) = tmp2(jl, jk)
            end do
        end do
    end subroutine kernel2

end module kernel_mod
    """.strip()

    basedir = gettempdir()/'test_pool_allocator_temporaries_kernel_nested'
    basedir.mkdir(exist_ok=True)
    (basedir/'driver.F90').write_text(fcode_driver)
    (basedir/'kernel_mod.F90').write_text(fcode_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routine': [{
            'name': 'driver',
            'role': 'driver',
        }]
    }
    scheduler = Scheduler(paths=[basedir], config=SchedulerConfig.from_dict(config), frontend=frontend)
    if frontend == OMNI:
        for item in scheduler.items:
            normalize_range_indexing(item.routine)

    transformation = TemporariesPoolAllocatorTransformation(block_dim=block_dim, directive=directive)
    scheduler.process(transformation=transformation, reverse=True)
    kernel_item = scheduler['kernel_mod#kernel']
    kernel2_item = scheduler['kernel_mod#kernel2']

    assert transformation._key in kernel_item.trafo_data
    assert kernel_item.trafo_data[transformation._key] == 'klon + 3 * klev * klon'
    assert kernel2_item.trafo_data[transformation._key] == '2 * columns * levels'
    assert all(v.scope is None for v in FindVariables().visit(kernel_item.trafo_data[transformation._key]))
    assert all(v.scope is None for v in FindVariables().visit(kernel2_item.trafo_data[transformation._key]))

    #
    # A few checks on the driver
    #
    driver = scheduler['#driver'].routine

    # Has the stack module been imported?
    check_stack_module_import(driver)

    # Has the stack been added to the call statements?
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('1', 'nlon', 'nlon', 'nz', 'field1(:,b)', 'field2(:,:,b)', 'ylstack')

    check_stack_created_in_driver(driver, 'nlon + 3 * nlon * nz', calls[0], 1)

    # Has the data sharing been updated?
    if directive in ['openmp', 'openacc']:
        keyword = {'openmp': 'omp', 'openacc': 'acc'}[directive]
        pragmas = [
            p for p in FindNodes(Pragma).visit(driver.body)
            if p.keyword.lower() == keyword and p.content.startswith('parallel')
        ]
        assert len(pragmas) == 1
        for pragma in pragmas:
            parameters = get_pragma_parameters(pragma, starts_with='parallel', only_loki_pragmas=False)
            assert 'private' in parameters and 'ylstack' in parameters['private'].lower()

    # Are there data regions for the stack?
    if directive == ['openacc']:
        pragmas = [
            p for p in FindNodes(Pragma).visit(driver.body)
            if p.keyword.lower() == 'acc' and 'data' in p.content
        ]
        assert len(pragmas) == 2
        parameters = get_pragma_parameters(pragmas[0], starts_with='data', only_loki_pragmas=False)
        assert parameters['create'] == 'zstack'

    #
    # A few checks on the kernels
    #
    calls = FindNodes(CallStatement).visit(kernel_item.routine.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('start', 'end', 'klon', 'klev', 'field2', 'ylstack')

    for item in [kernel_item, kernel2_item]:
        kernel = item.routine

        # Has the stack module been imported?
        check_stack_module_import(kernel)

        # Has the stack been added to the arguments?
        assert 'ydstack' in kernel.arguments

        # Is it being assigned to a local variable?
        assert 'ylstack' in kernel.variables

        # Let's check for the relevant "allocations" happening in the right order
        assign_idx = {}
        for idx, ass in enumerate(FindNodes(Assignment).visit(kernel.body)):
            if ass.lhs == 'ylstack' and ass.rhs == 'ydstack':
                # Local copy of stack status
                assign_idx['stack_assign'] = idx
            elif ass.lhs == 'ip_tmp1' and ass.rhs == 'ylstack%l':
                # ass Cray pointer for tmp1
                assign_idx['tmp1_ptr_assign'] = idx
            elif ass.lhs == 'ip_tmp2' and ass.rhs == 'ylstack%l':
                # ass Cray pointer for tmp2
                assign_idx['tmp2_ptr_assign'] = idx
            elif ass.lhs == 'ylstack%l' and 'ylstack%l' in ass.rhs and 'size' in ass.rhs and 'tmp1' in ass.rhs:
                # Stack increment for tmp1
                assign_idx['tmp1_stack_incr'] = idx
            elif ass.lhs == 'ylstack%l' and 'ylstack%l' in ass.rhs and 'tmp2' in ass.rhs and 'tmp2' in ass.rhs:
                # Stack increment for tmp2
                assign_idx['tmp2_stack_incr'] = idx

        expected_assign_in_order = [
            'stack_assign', 'tmp1_ptr_assign', 'tmp1_stack_incr', 'tmp2_ptr_assign', 'tmp2_stack_incr'
        ]
        assert set(expected_assign_in_order) == set(assign_idx.keys())

        for assign1, assign2 in zip(expected_assign_in_order, expected_assign_in_order[1:]):
            assert assign_idx[assign2] > assign_idx[assign1]

        # Check for pointer declarations in generated code
        fcode = kernel.to_fortran()
        assert 'pointer(ip_tmp1, tmp1)' in fcode.lower()
        assert 'pointer(ip_tmp2, tmp2)' in fcode.lower()

        # Check for stack size safegurads in generated code
        assert fcode.lower().count('if (ylstack%l > ylstack%u)') == 2
        assert fcode.lower().count('stop') == 2

    rmtree(basedir)
