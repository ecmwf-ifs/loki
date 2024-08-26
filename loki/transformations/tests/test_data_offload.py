# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from loki import (
    Sourcefile, Scheduler,  FindInlineCalls
)
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    FindNodes, Pragma, PragmaRegion, Loop, CallStatement, Import,
    pragma_regions_attached, get_pragma_parameters
)
from loki.transformations import (
    DataOffloadTransformation, GlobalVariableAnalysis,
    GlobalVarOffloadTransformation, GlobalVarHoistTransformation
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(name='config')
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
@pytest.mark.parametrize('assume_deviceptr', [True, False])
def test_data_offload_region_openacc(frontend, assume_deviceptr):
    """
    Test the creation of a simple device data offload region
    (`!$acc update`) from a `!$loki data` region with a single
    kernel call.
    """

    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)

    !$loki data
    call kernel_routine(nlon, nlev, a, b, c)
    !$loki end data

  END SUBROUTINE driver_routine
"""
    fcode_kernel = """
  SUBROUTINE kernel_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(IN)      :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(OUT)     :: c(nlon,nlev)
    INTEGER :: i, j

    do j=1, nlon
      do i=1, nlev
        b(i,j) = a(i,j) + 0.1
        c(i,j) = 0.1
      end do
    end do
  END SUBROUTINE kernel_routine
"""
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend)['driver_routine']
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend)['kernel_routine']
    driver.enrich(kernel)

    driver.apply(DataOffloadTransformation(assume_deviceptr=assume_deviceptr), role='driver',
                 targets=['kernel_routine'])

    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 2
    assert all(p.keyword == 'acc' for p in pragmas)
    if assume_deviceptr:
        assert 'deviceptr' in pragmas[0].content
        params = get_pragma_parameters(pragmas[0], only_loki_pragmas=False)
        assert all(var in params['deviceptr'] for var in ('a', 'b', 'c'))
    else:
        transformed = driver.to_fortran()
        assert 'copyin( a )' in transformed
        assert 'copy( b )' in transformed
        assert 'copyout( c )' in transformed


@pytest.mark.parametrize('frontend', available_frontends())
def test_data_offload_region_complex_remove_openmp(frontend):
    """
    Test the creation of a data offload region (OpenACC) with
    driver-side loops and CPU-style OpenMP pragmas to be removed.
    """

    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b, c, flag)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)
    logical, intent(in) :: flag
    INTEGER :: j

    !$loki data
    call my_custom_timer()

    if(flag)then
       !$omp parallel do private(j)
       do j=1, nlev
         call kernel_routine(nlon, j, a(:,j), b(:,j), c(:,j))
       end do
       !$omp end parallel do
    else
       !$omp parallel do private(j)
       do j=1, nlev
          a(:,j) = 0.
          b(:,j) = 0.
          c(:,j) = 0.
       end do
       !$omp end parallel do
    endif
    call my_custom_timer()

    !$loki end data
  END SUBROUTINE driver_routine
"""
    fcode_kernel = """
  SUBROUTINE kernel_routine(nlon, j, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, j
    REAL, INTENT(IN)      :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon)
    REAL, INTENT(INOUT)   :: c(nlon)
    INTEGER :: i

    do j=1, nlon
      b(i) = a(i) + 0.1
      c(i) = 0.1
    end do
  END SUBROUTINE kernel_routine
"""
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend)['driver_routine']
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend)['kernel_routine']
    driver.enrich(kernel)

    offload_transform = DataOffloadTransformation(remove_openmp=True)
    driver.apply(offload_transform, role='driver', targets=['kernel_routine'])

    assert len(FindNodes(Pragma).visit(driver.body)) == 2
    assert all(p.keyword == 'acc' for p in FindNodes(Pragma).visit(driver.body))

    with pragma_regions_attached(driver):
        # Ensure that loops in the region are preserved
        regions = FindNodes(PragmaRegion).visit(driver.body)
        assert len(regions) == 1
        assert len(FindNodes(Loop).visit(regions[0])) == 2

        # Ensure all activa and inactive calls are there
        calls = FindNodes(CallStatement).visit(regions[0])
        assert len(calls) == 3
        assert calls[0].name == 'my_custom_timer'
        assert calls[1].name == 'kernel_routine'
        assert calls[2].name == 'my_custom_timer'

        # Ensure OpenMP loop pragma is taken out
        assert len(FindNodes(Pragma).visit(regions[0])) == 0

    transformed = driver.to_fortran()
    assert 'copyin( a )' in transformed
    assert 'copy( b, c )' in transformed
    assert '!$omp' not in transformed


@pytest.mark.parametrize('frontend', available_frontends())
def test_data_offload_region_multiple(frontend):
    """
    Test the creation of a device data offload region (`!$acc update`)
    from a `!$loki data` region with multiple kernel calls.
    """

    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b, c, d)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)
    REAL, INTENT(INOUT)   :: d(nlon,nlev)

    !$loki data
    call kernel_routine(nlon, nlev, a, b, c)

    call kernel_routine(nlon, nlev, d, b, a)
    !$loki end data

  END SUBROUTINE driver_routine
"""
    fcode_kernel = """
  SUBROUTINE kernel_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(IN)      :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(OUT)     :: c(nlon,nlev)
    INTEGER :: i, j

    do j=1, nlon
      do i=1, nlev
        b(i,j) = a(i,j) + 0.1
        c(i,j) = 0.1
      end do
    end do
  END SUBROUTINE kernel_routine
"""
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend)['driver_routine']
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend)['kernel_routine']
    driver.enrich(kernel)

    driver.apply(DataOffloadTransformation(), role='driver', targets=['kernel_routine'])

    assert len(FindNodes(Pragma).visit(driver.body)) == 2
    assert all(p.keyword == 'acc' for p in FindNodes(Pragma).visit(driver.body))

    # Ensure that the copy direction is the union of the two calls, ie.
    # "a" is "copyin" in first call and "copyout" in second, so it should be "copy"
    transformed = driver.to_fortran()
    assert 'copyin( d )' in transformed
    assert 'copy( b, a )' in transformed
    assert 'copyout( c )' in transformed


@pytest.fixture(name='global_variable_analysis_code')
def fixture_global_variable_analysis_code(tmp_path):
    fcode = {
        #------------------------------
        'global_var_analysis_header_mod': (
        #------------------------------
"""
module global_var_analysis_header_mod
    implicit none

    integer, parameter :: nval = 5
    integer, parameter :: nfld = 3

    integer :: n

    integer :: iarr(nfld)
    real :: rarr(nval, nfld)
end module global_var_analysis_header_mod
"""
        ).strip(),
        #----------------------------
        'global_var_analysis_data_mod': (
        #----------------------------
"""
module global_var_analysis_data_mod
    implicit none

    real, allocatable :: rdata(:,:,:)

    type some_type
        real :: val
        real, allocatable :: vals(:,:)
    end type some_type

    type(some_type) :: tt

contains
    subroutine some_routine(i)
        integer, intent(inout) :: i
        i = i + 1
    end subroutine some_routine
end module global_var_analysis_data_mod
"""
        ).strip(),
        #------------------------------
        'global_var_analysis_kernel_mod': (
        #------------------------------
"""
module global_var_analysis_kernel_mod
    use global_var_analysis_header_mod, only: rarr
    use global_var_analysis_data_mod, only: some_routine, some_type

    implicit none

contains
    subroutine kernel_a(arg, tt)
        use global_var_analysis_header_mod, only: iarr, nval, nfld, n

        real, intent(inout) :: arg(:,:)
        type(some_type), intent(in) :: tt
        real :: tmp(n)
        integer :: i, j

        do i=1,nfld
            if (iarr(i) > 0) then
                do j=1,nval
                    arg(j,i) = rarr(j, i) + tt%val
                    call some_routine(arg(j,i))
                enddo
            endif
        enddo
    end subroutine kernel_a

    subroutine kernel_b(arg)
        use global_var_analysis_header_mod, only: iarr, nfld
        use global_var_analysis_data_mod, only: rdata, tt

        real, intent(inout) :: arg(:,:)
        integer :: i

        do i=1,nfld
            if (iarr(i) .ne. 0) then
                rdata(:,:,i) = arg(:,:) + rdata(:,:,i)
            else
                arg(:,:) = tt%vals(:,:)
            endif
        enddo
    end subroutine kernel_b
end module global_var_analysis_kernel_mod
"""
        ).strip(),
        #-------
        'driver': (
        #-------
"""
subroutine driver(arg)
    use global_var_analysis_kernel_mod, only: kernel_a, kernel_b
    use global_var_analysis_data_mod, only: tt
    implicit none

    real, intent(inout) :: arg(:,:)

    !$loki update_device

    call kernel_a(arg, tt)

    call kernel_b(arg)

    !$loki update_host
end subroutine driver
"""
        ).strip()
    }

    for name, code in fcode.items():
        (tmp_path/f'{name}.F90').write_text(code)
    return tmp_path


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('key', (None, 'foobar'))
def test_global_variable_analysis(frontend, key, config, global_variable_analysis_code):
    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(
        paths=(global_variable_analysis_code,), config=config, seed_routines='driver',
        frontend=frontend, xmods=(global_variable_analysis_code,)
    )
    scheduler.process(GlobalVariableAnalysis(key=key))
    if key is None:
        key = GlobalVariableAnalysis._key

    # Validate the analysis trafo_data

    # OMNI handles array indices and parameters differently
    if frontend == OMNI:
        nfld_dim = '3'
        nval_dim = '5'
        nfld_data = set()
        nval_data = set()
    else:
        nfld_dim = 'nfld'
        nval_dim = 'nval'
        nfld_data = {('nfld', 'global_var_analysis_header_mod')}
        nval_data = {('nval', 'global_var_analysis_header_mod')}

    expected_trafo_data = {
        'global_var_analysis_header_mod': {
            'declares': {f'iarr({nfld_dim})', f'rarr({nval_dim}, {nfld_dim})', 'n'},
            'offload': {}
        },
        'global_var_analysis_data_mod': {
            'declares': {'rdata(:, :, :)', 'tt'},
            'offload': {}
        },
        'global_var_analysis_data_mod#some_routine': {'defines_symbols': set(), 'uses_symbols': set()},
        'global_var_analysis_kernel_mod#kernel_a': {
            'defines_symbols': set(),
            'uses_symbols': nval_data | nfld_data | {
                (f'iarr({nfld_dim})', 'global_var_analysis_header_mod'),
                ('n', 'global_var_analysis_header_mod'),
                (f'rarr({nval_dim}, {nfld_dim})', 'global_var_analysis_header_mod')
            }
        },
        'global_var_analysis_kernel_mod#kernel_b': {
            'defines_symbols': {('rdata(:, :, :)', 'global_var_analysis_data_mod')},
            'uses_symbols': nfld_data | {
                ('rdata(:, :, :)', 'global_var_analysis_data_mod'), ('tt', 'global_var_analysis_data_mod'),
                ('tt%vals', 'global_var_analysis_data_mod'), (f'iarr({nfld_dim})', 'global_var_analysis_header_mod')
            }
        },
        '#driver': {
            'defines_symbols': {('rdata(:, :, :)', 'global_var_analysis_data_mod')},
            'uses_symbols': nval_data | nfld_data | {
                ('rdata(:, :, :)', 'global_var_analysis_data_mod'),
                ('n', 'global_var_analysis_header_mod'),
                ('tt', 'global_var_analysis_data_mod'), ('tt%vals', 'global_var_analysis_data_mod'),
                (f'iarr({nfld_dim})', 'global_var_analysis_header_mod'),
                (f'rarr({nval_dim}, {nfld_dim})', 'global_var_analysis_header_mod')
            }
        }
    }

    assert set(scheduler.items) == set(expected_trafo_data) | {'global_var_analysis_data_mod#some_type'}
    for item in scheduler.items:
        if item == 'global_var_analysis_data_mod#some_type':
            continue
        for trafo_data_key, trafo_data_value in item.trafo_data[key].items():
            assert (
                sorted(
                    tuple(str(vv) for vv in v) if isinstance(v, tuple) else str(v)
                    for v in trafo_data_value
                ) == sorted(expected_trafo_data[item.name][trafo_data_key])
            )


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('key', (None, 'foobar'))
def test_global_variable_offload(frontend, key, config, global_variable_analysis_code):

    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    # OMNI handles array indices and parameters differently
    if frontend == OMNI:
        nfld_dim = '3'
        nval_dim = '5'
    else:
        nfld_dim = 'nfld'
        nval_dim = 'nval'

    scheduler = Scheduler(
        paths=(global_variable_analysis_code,), config=config, seed_routines='driver',
        frontend=frontend, xmods=(global_variable_analysis_code,)
    )
    scheduler.process(GlobalVariableAnalysis(key=key))
    scheduler.process(GlobalVarOffloadTransformation(key=key))
    driver = scheduler['#driver'].ir

    if key is None:
        key = GlobalVariableAnalysis._key

    expected_trafo_data = {
        'global_var_analysis_header_mod': {
            'declares': {f'iarr({nfld_dim})', f'rarr({nval_dim}, {nfld_dim})', 'n'},
            'offload': {f'iarr({nfld_dim})', f'rarr({nval_dim}, {nfld_dim})', 'n'}
        },
        'global_var_analysis_data_mod': {
            'declares': {'rdata(:, :, :)', 'tt'},
            'offload': {'rdata(:, :, :)', 'tt', 'tt%vals'}
        },
    }

    # Verify module offload sets
    for item in [scheduler['global_var_analysis_header_mod'], scheduler['global_var_analysis_data_mod']]:
        for trafo_data_key, trafo_data_value in item.trafo_data[key].items():
            assert (
                sorted(
                    tuple(str(vv) for vv in v) if isinstance(v, tuple) else str(v)
                    for v in trafo_data_value
                ) == sorted(expected_trafo_data[item.name][trafo_data_key])
            )

    # Verify imports have been added to the driver
    expected_imports = {
        'global_var_analysis_header_mod': {'iarr', 'rarr', 'n'},
        'global_var_analysis_data_mod': {'rdata'}
    }

    # We need to check only the first imports as they have to be prepended
    for import_ in driver.imports[:len(expected_imports)]:
        assert {var.name.lower() for var in import_.symbols} == expected_imports[import_.module.lower()]

    expected_h2d_pragmas = {
        'update device': {'iarr', 'rdata', 'rarr', 'n'},
        'enter data copyin': {'tt%vals'}
    }
    expected_d2h_pragmas = {
        'update self': {'rdata'}
    }

    acc_pragmas = [p for p in FindNodes(Pragma).visit(driver.ir) if p.keyword.lower() == 'acc']
    assert len(acc_pragmas) == len(expected_h2d_pragmas) + len(expected_d2h_pragmas)
    for pragma in acc_pragmas[:len(expected_h2d_pragmas)]:
        command, variables = pragma.content.lower().split('(')
        assert command.strip() in expected_h2d_pragmas
        assert set(variables.strip()[:-1].strip().split(',')) == expected_h2d_pragmas[command.strip()]
    for pragma in acc_pragmas[len(expected_h2d_pragmas):]:
        command, variables = pragma.content.lower().split('(')
        assert command.strip() in expected_d2h_pragmas
        assert set(variables.strip()[:-1].strip().split(',')) == expected_d2h_pragmas[command.strip()]

    # Verify declarations have been added to the header modules
    expected_declarations = {
        'global_var_analysis_header_mod': {'iarr', 'rarr', 'n'},
        'global_var_analysis_data_mod': {'rdata', 'tt'}
    }

    modules = {
        name: scheduler[name].ir for name in expected_declarations
    }

    for name, module in modules.items():
        acc_pragmas = [p for p in FindNodes(Pragma).visit(module.spec) if p.keyword.lower() == 'acc']
        variables = {
            v.strip()
            for pragma in acc_pragmas
            for v in pragma.content.lower().split('(')[-1].strip()[:-1].split(',')
        }
        assert variables == expected_declarations[name]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transformation_global_var_import(here, config, frontend, tmp_path):
    """
    Test the generation of offload instructions of global variable imports.
    """
    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(paths=here/'sources/projGlobalVarImports', config=config, frontend=frontend, xmods=[tmp_path])
    scheduler.process(transformation=GlobalVariableAnalysis())
    scheduler.process(transformation=GlobalVarOffloadTransformation())

    driver = scheduler['#driver'].ir
    moduleA = scheduler['modulea'].ir
    moduleB = scheduler['moduleb'].ir
    moduleC = scheduler['modulec'].ir

    # check that global variables have been added to driver symbol table
    imports = FindNodes(Import).visit(driver.spec)
    assert len(imports) == 2
    assert imports[0].module != imports[1].module
    assert imports[0].symbols != imports[1].symbols
    for i in imports:
        assert len(i.symbols) == 2
        assert i.module.lower() in ('moduleb', 'modulec')
        assert set(s.name for s in i.symbols) in ({'var2', 'var3'}, {'var4', 'var5'})

    # check that existing acc pragmas have not been stripped and update device/update self added correctly
    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 4
    assert all(p.keyword.lower() == 'acc' for p in pragmas)

    assert 'update device' in pragmas[0].content
    assert 'var2' in pragmas[0].content
    assert 'var3' in pragmas[0].content

    assert pragmas[1].content == 'serial'
    assert pragmas[2].content == 'end serial'

    assert 'update self' in pragmas[3].content
    assert 'var4' in pragmas[3].content
    assert 'var5' in pragmas[3].content

    # check that no declarations have been added for parameters
    pragmas = FindNodes(Pragma).visit(moduleA.spec)
    assert not pragmas

    # check for device-side declarations where appropriate
    pragmas = FindNodes(Pragma).visit(moduleB.spec)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'acc'
    assert 'declare create' in pragmas[0].content
    assert 'var2' in pragmas[0].content
    assert 'var3' in pragmas[0].content

    pragmas = FindNodes(Pragma).visit(moduleC.spec)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'acc'
    assert 'declare create' in pragmas[0].content
    assert 'var4' in pragmas[0].content
    assert 'var5' in pragmas[0].content


@pytest.mark.parametrize('frontend', available_frontends())
def test_transformation_global_var_import_derived_type(here, config, frontend, tmp_path):
    """
    Test the generation of offload instructions of derived-type global variable imports.
    """

    config['default']['enable_imports'] = True
    config['routines'] = {
        'driver_derived_type': {'role': 'driver'}
    }

    scheduler = Scheduler(paths=here/'sources/projGlobalVarImports', config=config, frontend=frontend, xmods=[tmp_path])
    scheduler.process(transformation=GlobalVariableAnalysis())
    scheduler.process(transformation=GlobalVarOffloadTransformation())

    driver = scheduler['#driver_derived_type'].ir
    module = scheduler['module_derived_type'].ir

    # check that global variables have been added to driver symbol table
    imports = FindNodes(Import).visit(driver.spec)
    assert len(imports) == 1
    assert len(imports[0].symbols) == 2
    assert imports[0].module.lower() == 'module_derived_type'
    assert set(s.name for s in imports[0].symbols) == {'p', 'p0'}

    # check that existing acc pragmas have not been stripped and update device/update self added correctly
    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 5
    assert all(p.keyword.lower() == 'acc' for p in pragmas)

    assert 'enter data copyin' in pragmas[0].content
    assert 'p0%x' in pragmas[0].content
    assert 'p0%y' in pragmas[0].content
    assert 'p0%z' in pragmas[0].content
    assert 'p%n' in pragmas[0].content

    assert 'enter data create' in pragmas[1].content
    assert 'p%x' in pragmas[1].content
    assert 'p%y' in pragmas[1].content
    assert 'p%z' in pragmas[1].content

    assert pragmas[2].content == 'serial'
    assert pragmas[3].content == 'end serial'

    assert 'exit data copyout' in pragmas[4].content
    assert 'p%x' in pragmas[4].content
    assert 'p%y' in pragmas[4].content
    assert 'p%z' in pragmas[4].content

    # check for device-side declarations
    pragmas = FindNodes(Pragma).visit(module.spec)
    assert len(pragmas) == 1
    assert pragmas[0].keyword == 'acc'
    assert 'declare create' in pragmas[0].content
    assert 'p' in pragmas[0].content
    assert 'p0' in pragmas[0].content
    assert 'p_array' in pragmas[0].content
    # Note: g is not offloaded because it is not used by the kernel (albeit imported)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('hoist_parameters', (False, True))
@pytest.mark.parametrize('ignore_modules', (None, ('moduleb',)))
def test_transformation_global_var_hoist(here, config, frontend, hoist_parameters, ignore_modules, tmp_path):
    """
    Test hoisting of global variable imports.
    """
    config['default']['enable_imports'] = True
    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(paths=here/'sources/projGlobalVarImports', config=config, frontend=frontend, xmods=[tmp_path])
    scheduler.process(transformation=GlobalVariableAnalysis())
    scheduler.process(transformation=GlobalVarHoistTransformation(hoist_parameters=hoist_parameters,
        ignore_modules=ignore_modules))

    driver = scheduler['#driver'].ir
    kernel0 = scheduler['#kernel0'].ir
    kernel_map = {key: scheduler[f'#{key}'].ir for key in ['kernel1', 'kernel2', 'kernel3']}
    some_func = scheduler['func_mod#some_func'].ir

    # symbols within each module
    expected_symbols = {'modulea': ['var0', 'var1'], 'moduleb': ['var2', 'var3'],
            'modulec': ['var4', 'var5']}
    # expected intent of those variables (if hoisted)
    var_intent_map = {'var0': 'in', 'var1': 'in', 'var2': 'in',
            'var3': 'in', 'var4': 'inout', 'var5': 'inout', 'tmp': None}
    # DRIVER
    imports = FindNodes(Import).visit(driver.spec)
    import_names = [_import.module.lower() for _import in imports]
    # check driver imports
    expected_driver_modules = ['modulec']
    expected_driver_modules += ['moduleb'] if ignore_modules is None else []
    # OMNI handles parameters differently, ModuleA only contains parameters
    if frontend != OMNI:
        expected_driver_modules += ['modulea'] if hoist_parameters else []
    assert len(imports) == len(expected_driver_modules)
    assert sorted(expected_driver_modules) == sorted(import_names)
    for _import in imports:
        assert sorted([sym.name for sym in _import.symbols]) == expected_symbols[_import.module.lower()]
    # check driver call
    driver_calls = FindNodes(CallStatement).visit(driver.body)
    expected_args = []
    for module in expected_driver_modules:
        expected_args.extend(expected_symbols[module])
    assert [arg.name for arg in driver_calls[0].arguments] == sorted(expected_args)

    originally = {'kernel1': ['modulea'], 'kernel2': ['moduleb'],
            'kernel3': ['moduleb', 'modulec']}
    # KERNEL0
    expected_vars = expected_args.copy()
    expected_vars.append('a')
    assert [arg.name for arg in kernel0.arguments] == sorted(expected_args)
    assert [arg.name for arg in kernel0.variables] == sorted(expected_vars)
    for var in kernel0.arguments:
        assert kernel0.variable_map[var.name.lower()].type.intent == var_intent_map[var.name.lower()]
        assert var.scope == kernel0
    kernel0_inline_calls = FindInlineCalls().visit(kernel0.body)
    for inline_call in kernel0_inline_calls:
        if ignore_modules is None:
            assert len(inline_call.arguments) == 1
            assert [arg.name for arg in inline_call.arguments] == ['var2']
            assert [arg.name for arg in some_func.arguments] == ['var2']
        else:
            assert len(inline_call.arguments) == 0
            assert len(some_func.arguments) == 0
    kernel0_calls = FindNodes(CallStatement).visit(kernel0.body)
    # KERNEL1 & KERNEL2 & KERNEL3
    for call in kernel0_calls:
        expected_args = []
        expected_imports = []
        kernel_expected_symbols = []
        for module in originally[call.routine.name]:
            # always, since at least 'some_func' is imported
            if call.routine.name == 'kernel1' and module == 'modulea':
                expected_imports.append(module)
                kernel_expected_symbols.append('some_func')
            if module in expected_driver_modules:
                expected_args.extend(expected_symbols[module])
            else:
                # already added
                if module != 'modulea':
                    expected_imports.append(module)
                kernel_expected_symbols.extend(expected_symbols[module])
        assert len(expected_args) == len(call.arguments)
        assert [arg.name for arg in call.arguments] == expected_args
        assert [arg.name for arg in kernel_map[call.routine.name].arguments] == expected_args
        for var in kernel_map[call.routine.name].variables:
            var_intent = kernel_map[call.routine.name].variable_map[var.name.lower()].type.intent
            assert var.scope == kernel_map[call.routine.name]
            assert var_intent == var_intent_map[var.name.lower()]
        if call.routine.name in ['kernel1', 'kernel2']:
            expected_args = ['tmp'] + expected_args
        assert [arg.name for arg in kernel_map[call.routine.name].variables] == expected_args
        kernel_imports = FindNodes(Import).visit(call.routine.spec)
        assert sorted([_import.module.lower() for _import in kernel_imports]) == sorted(expected_imports)
        imported_symbols = [] # _import.symbols for _import in kernel_imports]
        for _import in kernel_imports:
            imported_symbols.extend([sym.name.lower() for sym in _import.symbols])
        assert sorted(imported_symbols) == sorted(kernel_expected_symbols)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('hoist_parameters', (False, True))
def test_transformation_global_var_derived_type_hoist(here, config, frontend, hoist_parameters, tmp_path):
    """
    Test hoisting of derived-type global variable imports.
    """

    config['default']['enable_imports'] = True
    config['routines'] = {
        'driver_derived_type': {'role': 'driver'}
    }

    scheduler = Scheduler(paths=here/'sources/projGlobalVarImports', config=config, frontend=frontend, xmods=[tmp_path])
    scheduler.process(transformation=GlobalVariableAnalysis())
    scheduler.process(transformation=GlobalVarHoistTransformation(hoist_parameters))

    driver = scheduler['#driver_derived_type'].ir
    kernel = scheduler['#kernel_derived_type'].ir

    # DRIVER
    imports = FindNodes(Import).visit(driver.spec)
    assert len(imports) == 1
    assert imports[0].module.lower() == 'module_derived_type'
    assert sorted([sym.name.lower() for sym in imports[0].symbols]) == sorted(['p', 'p_array', 'p0'])
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    # KERNEL
    assert [arg.name for arg in calls[0].arguments] == ['p', 'p0', 'p_array']
    assert [arg.name for arg in kernel.arguments] == ['p', 'p0', 'p_array']
    kernel_imports = FindNodes(Import).visit(kernel.spec)
    assert len(kernel_imports) == 1
    assert [sym.name.lower() for sym in kernel_imports[0].symbols] == ['g']
    assert sorted([var.name for var in kernel.variables]) == ['i', 'j', 'p', 'p0', 'p_array']
    assert kernel.variable_map['p_array'].type.allocatable
    assert kernel.variable_map['p_array'].type.intent == 'inout'
    assert kernel.variable_map['p_array'].type.dtype.name == 'point'
    assert kernel.variable_map['p'].type.intent == 'inout'
    assert kernel.variable_map['p'].type.dtype.name == 'point'
    assert kernel.variable_map['p0'].type.intent == 'in'
    assert kernel.variable_map['p0'].type.dtype.name == 'point'
