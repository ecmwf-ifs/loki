# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
# from shutil import rmtree
import pytest
import numpy as np

from loki import Subroutine, Module, cgen, cppgen, cudagen, FindNodes, Dimension, Scheduler, read_file
from loki.build import jit_compile, jit_compile_lib, clean_test, Builder, Obj
import loki.expression.symbols as sym
from loki.frontend import available_frontends, OFP
from loki import ir, fgen

from loki.transformations.array_indexing import normalize_range_indexing
from loki.transformations.transpile import FortranCTransformation
from loki.transformations.single_column import SCCLowLevelHoist, SCCLowLevelParametrise

@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='klon', index='jl', bounds=('kidia', 'kfdia'))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='klev', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='ngpblks', index='ibl', index_aliases=['JKGLO'])


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
            'strict': False,  # cudafor import
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

def remove_whitespace_linebreaks(text):
    # return text.replace(' ', '').replace('\n', ' ').replace('\r', '').replace('\t', '').lower()
    return text

@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuda_hoist_extended(tmp_path, here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / '../../tests/sources/projSccCufExtended/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuda_transform = SCCLowLevelHoist(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='hoist', demote_local_arrays=False, hoist_parameters=True,
        all_derived_types=True, skip_driver_imports=True, recurse_to_kernels=True,
        # dim_vars=(vertical.size,),
        as_kwarguments=True, remove_vector_section=True,
        use_c_ptr=True, path=here, mode='cuda'
    )
    # dependency = DependencyTransformation(suffix='_FC', module_suffix='_MOD')
    # scheduler.process(dependency)
    # # Write out all modified source files into the build directory
    # scheduler.process(transformation=FileWriteTransformation(
    #     builddir=build, mode=mode, cuf='cuf' in mode,
    #     include_module_var_imports=global_var_offload
    # ))

    # transformation_type='hoist', derived_types = ['TECLDP'], block_dim=block_dim, mode='cuda',
    ## dim_vars=(vertical.size, horizontal.size),
    # demote_local_arrays=False,
    # as_kwarguments=True, hoist_parameters=True,
    # ignore_modules=['parkind1'], all_derived_types=True,
    # dic2p=dic2p, skip_driver_imports=True,
    # recurse_to_kernels=True)

    scheduler.process(transformation=cuda_transform)

    if True:
        f2c_transformation = FortranCTransformation(path=tmp_path, language='cuda', use_c_ptr=True)
        scheduler.process(transformation=f2c_transformation)

        f_driver = remove_whitespace_linebreaks(fgen(scheduler["driver_mod#driver"].ir))
        # f_driver = remove_whitespace_linebreaks(read_file(tmp_path/'driver.cuda_hoist.f90'))
        fc_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_fc.F90'))
        c_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.c'))
        c_kernel_header = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.h'))
        c_kernel_launch = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c_launch.h'))
        c_device = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.c'))
        c_nested_device = remove_whitespace_linebreaks(read_file(tmp_path/'nested_device_c.c'))
        # c_device_header = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.h'))
        c_elemental_device = remove_whitespace_linebreaks(read_file(tmp_path/'elemental_device_c.c'))
        print(f"driver\n----------------")
        print(f"{f_driver}")
        print(f"fc_kernel\n----------------")
        print(f"{fc_kernel}")
        print(f"kernel\n----------------")
        print(f"{c_kernel}")
        print(f"device\n----------------")
        print(f"{c_device}")
        print(f"nested_device\n----------------")
        print(f"{c_nested_device}")
        print(f"elemental_device\n----------------")
        print(f"{c_elemental_device}")
    else:
        f_driver = remove_whitespace_linebreaks(fgen(scheduler["driver_mod#driver"].ir))
        f_kernel = remove_whitespace_linebreaks(fgen(scheduler["kernel_mod#kernel"].ir))
        f_device = remove_whitespace_linebreaks(fgen(scheduler["device_mod#device"].ir))
        f_nested_device = remove_whitespace_linebreaks(fgen(scheduler["nested_device_mod#nested_device"].ir))

        print(f"driver\n----------------")
        print(f"{f_driver}")
        print(f"kernel\n----------------")
        print(f"{f_kernel}")
        print(f"device\n----------------")
        print(f"{f_device}")
        print(f"f_nested_device\n----------------")
        print(f"{f_nested_device}")

