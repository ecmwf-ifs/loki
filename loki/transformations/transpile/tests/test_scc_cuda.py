# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from loki import Scheduler, Dimension, read_file
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes

from loki.transformations.transpile import (
    FortranCTransformation, FortranISOCWrapperTransformation
)
from loki.transformations.single_column import (
    SCCLowLevelHoist, SCCLowLevelParametrise
)


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'iend'))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


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
    return text.replace(' ', '').replace('\n', ' ').replace('\r', '').replace('\t', '').lower()

@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuda_parametrise(tmp_path, here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / '../../tests/sources/projSccCuf/module'

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver'],
        output_dir=tmp_path, frontend=frontend, xmods=[tmp_path]
    )

    dic2p = {'nz': 137}
    cuda_transform = SCCLowLevelParametrise(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='parametrise',
        dim_vars=(vertical.size,), as_kwarguments=True, remove_vector_section=True,
        use_c_ptr=True, dic2p=dic2p, path=here, mode='cuda'
    )
    scheduler.process(transformation=cuda_transform)
    f2c_transformation = FortranCTransformation(language='cuda')
    scheduler.process(transformation=f2c_transformation)
    f2cwrap = FortranISOCWrapperTransformation(language='cuda', use_c_ptr=True)
    scheduler.process(transformation=f2cwrap)

    kernel = scheduler['kernel_mod#kernel'].ir
    kernel_variable_map = kernel.variable_map
    assert kernel_variable_map[horizontal.index].type.intent is None
    assert kernel_variable_map[horizontal.index].scope == kernel
    device = scheduler['kernel_mod#device'].ir
    device_variable_map = device.variable_map
    assert device_variable_map[horizontal.index].type.intent.lower() == 'in'
    assert device_variable_map[horizontal.index].scope == device

    fc_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_fc.F90'))
    c_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.c'))
    c_kernel_header = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.h'))
    c_kernel_launch = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c_launch.h'))
    c_device = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.c'))
    c_elemental_device = remove_whitespace_linebreaks(read_file(tmp_path/'elemental_device_c.c'))
    c_some_func = remove_whitespace_linebreaks(read_file(tmp_path/'some_func_c.c'))
    c_some_func_header = remove_whitespace_linebreaks(read_file(tmp_path/'some_func_c.h'))

    calls = FindNodes(ir.CallStatement).visit(scheduler["driver_mod#driver"].ir.body)
    assert len(calls) == 3
    for call in calls:
        assert str(call.name).lower() == 'kernel'
        assert call.pragma[0].keyword == 'loki'
        assert 'removed_loop' in call.pragma[0].content
    # kernel_fc.F90
    assert '!$acchost_datause_device(q,t,z)' in fc_kernel
    assert 'kernel_iso_c(start,nlon,c_loc(q),c_loc(t),c_loc(z),nb,tot,iend)' in fc_kernel
    assert 'bind(c,name="kernel_c_launch")' in fc_kernel
    assert 'useiso_c_binding' in fc_kernel
    # kernel_c.c
    assert '#include<cuda.h>' in c_kernel
    assert '#include<cuda_runtime.h>' in c_kernel
    assert '#include"kernel_c.h"' in c_kernel
    assert '#include"kernel_c_launch.h"' in c_kernel
    assert 'include"elemental_device_c.h"' in c_kernel
    assert 'include"device_c.h"' in c_kernel
    assert 'include"some_func_c.h"' in c_kernel
    assert '__global__voidkernel_c' in c_kernel
    assert 'jl=threadidx.x;' in c_kernel
    assert 'b=blockidx.x;' in c_kernel
    assert 'device_c(' in c_kernel
    assert 'elemental_device_c(' in c_kernel
    assert '=some_func_c(' in c_kernel
    # kernel_c.h
    assert '__global__voidkernel_c' in c_kernel_header
    assert 'jl=threadidx.x;' not in c_kernel_header
    assert 'b=blockidx.x;' not in c_kernel_header
    # kernel_c_launch.h
    assert 'extern"c"' in c_kernel_launch
    assert 'voidkernel_c_launch(' in c_kernel_launch
    assert 'structdim3blockdim;' in c_kernel_launch
    assert 'structdim3griddim;' in c_kernel_launch
    assert 'griddim=dim3(' in c_kernel_launch
    assert 'blockdim=dim3(' in c_kernel_launch
    assert 'kernel_c<<<griddim,blockdim>>>(' in c_kernel_launch
    assert 'cudadevicesynchronize();' in c_kernel_launch
    # device_c.c
    assert '#include<cuda.h>' in c_device
    assert '#include<cuda_runtime.h>' in c_device
    assert '#include"device_c.h"' in c_device
    # elemental_device_c.c
    assert '__device__voiddevice_c(' in c_device
    assert '#include<cuda.h>' in c_elemental_device
    assert '#include<cuda_runtime.h>' in c_elemental_device
    assert '#include"elemental_device_c.h"' in c_elemental_device
    # some_func_c.c
    assert 'doublesome_func_c(doublea)' in c_some_func
    assert 'returnsome_func' in c_some_func
    # some_func_c.h
    assert 'doublesome_func_c(doublea);' in c_some_func_header


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuda_hoist(tmp_path, here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / '../../tests/sources/projSccCuf/module'

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver'],
        output_dir=tmp_path, frontend=frontend, xmods=[tmp_path]
    )

    cuda_transform = SCCLowLevelHoist(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='parametrise',
        dim_vars=(vertical.size,), as_kwarguments=True, remove_vector_section=True,
        use_c_ptr=True, path=here, mode='cuda'
    )
    scheduler.process(transformation=cuda_transform)
    f2c_transformation = FortranCTransformation(language='cuda')
    scheduler.process(transformation=f2c_transformation)
    f2cwrap = FortranISOCWrapperTransformation(language='cuda', use_c_ptr=True)
    scheduler.process(transformation=f2cwrap)

    kernel = scheduler['kernel_mod#kernel'].ir
    kernel_variable_map = kernel.variable_map
    assert kernel_variable_map[horizontal.index].type.intent is None
    assert kernel_variable_map[horizontal.index].scope == kernel
    device = scheduler['kernel_mod#device'].ir
    device_variable_map = device.variable_map
    assert device_variable_map[horizontal.index].type.intent.lower() == 'in'
    assert device_variable_map[horizontal.index].scope == device

    fc_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_fc.F90'))
    c_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.c'))
    c_kernel_header = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.h'))
    c_kernel_launch = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c_launch.h'))
    c_device = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.c'))
    c_elemental_device = remove_whitespace_linebreaks(read_file(tmp_path/'elemental_device_c.c'))
    c_some_func = remove_whitespace_linebreaks(read_file(tmp_path/'some_func_c.c'))
    c_some_func_header = remove_whitespace_linebreaks(read_file(tmp_path/'some_func_c.h'))

    calls = FindNodes(ir.CallStatement).visit(scheduler["driver_mod#driver"].ir.body)
    assert len(calls) == 3
    for call in calls:
        assert str(call.name).lower() == 'kernel'
        assert call.pragma[0].keyword == 'loki'
        assert 'removed_loop' in call.pragma[0].content
    # kernel_fc.F90
    assert '!$acchost_datause_device(q,t,z,local_z,device_local_x)' in fc_kernel
    assert 'kernel_iso_c(start,nlon,nz,c_loc(q),c_loc(t),c_loc(z)' in fc_kernel
    assert 'c_loc(z),nb,tot,iend,c_loc(local_z),c_loc(device_local_x))' in fc_kernel
    assert 'bind(c,name="kernel_c_launch")' in fc_kernel
    assert 'useiso_c_binding' in fc_kernel
    # kernel_c.c
    assert '#include<cuda.h>' in c_kernel
    assert '#include<cuda_runtime.h>' in c_kernel
    assert '#include"kernel_c.h"' in c_kernel
    assert '#include"kernel_c_launch.h"' in c_kernel
    assert '#include"elemental_device_c.h"' in c_kernel
    assert '#include"device_c.h"' in c_kernel
    assert 'include"some_func_c.h"' in c_kernel
    assert '__global__voidkernel_c' in c_kernel
    assert 'jl=threadidx.x;' in c_kernel
    assert 'b=blockidx.x;' in c_kernel
    assert 'device_c(' in c_kernel
    assert 'elemental_device_c(' in c_kernel
    assert '=some_func_c(' in c_kernel
    # kernel_c.h
    assert '__global__voidkernel_c' in c_kernel_header
    assert 'jl=threadidx.x;' not in c_kernel_header
    assert 'b=blockidx.x;' not in c_kernel_header
    # kernel_c_launch.h
    assert 'extern"c"' in c_kernel_launch
    assert 'voidkernel_c_launch(' in c_kernel_launch
    assert 'structdim3blockdim;' in c_kernel_launch
    assert 'structdim3griddim;' in c_kernel_launch
    assert 'griddim=dim3(' in c_kernel_launch
    assert 'blockdim=dim3(' in c_kernel_launch
    assert 'kernel_c<<<griddim,blockdim>>>(' in c_kernel_launch
    assert 'cudadevicesynchronize();' in c_kernel_launch
    # device_c.c
    assert '#include<cuda.h>' in c_device
    assert '#include<cuda_runtime.h>' in c_device
    assert '#include"device_c.h"' in c_device
    assert '__device__voiddevice_c(' in c_device
    # elemental_device_c.c
    assert '#include<cuda.h>' in c_elemental_device
    assert '#include<cuda_runtime.h>' in c_elemental_device
    assert '#include"elemental_device_c.h"' in c_elemental_device
    # some_func_c.c
    assert 'doublesome_func_c(doublea)' in c_some_func
    assert 'returnsome_func' in c_some_func
    # some_func_c.h
    assert 'doublesome_func_c(doublea);' in c_some_func_header
