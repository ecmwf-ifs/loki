# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pathlib import Path
from conftest import available_frontends
from transformations import SccCuf
from loki import (
    Scheduler, OMNI, Subroutine, Dimension, FindNodes, Loop, Assignment,
    CallStatement, Scalar, Array, Pragma, pragmas_attached, fgen, frontend, cufgen,
    as_tuple, Allocation, Deallocation
)
from loki.expression import symbols as sym
from loki.transform import HoistTemporaryArraysAnalysis, HoistVariablesTransformation, ParametriseTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'))


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
            'strict': True,
        },
        'routine': [
            {
                'name': 'driver',
                'role': 'driver',
            },
        ]
    }


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_simple(frontend, horizontal, vertical, blocking):

    fcode_driver = """
  SUBROUTINE driver(nlon, nz, nb, tot, q, t)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    INTEGER, INTENT(IN)   :: tot
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: z(nlon,nz+1,nb)
    INTEGER :: b, start, end, ibl, icend

    start = 1
    end = tot
    do b=1,end,nlon
      ibl = (b-1)/nlon+1
      icend = MIN(nlon,tot-b+1)
      call kernel(start, icend, nlon, nz, q(:,:,b), t(:,:,b), z(:,:,b))
    end do
  END SUBROUTINE driver
"""

    fcode_kernel = """
  SUBROUTINE kernel(start, end, nlon, nz, q, t, z)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL, INTENT(INOUT) :: z(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * k
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO
    
    DO jk = 2, nz
      DO jl = start, end
        z(jl, jk) = 0.0 
      END DO
    END DO
    
    ! DO JL = START, END
    !   Q(JL, NZ) = Q(JL, NZ) * C
    ! END DO
  END SUBROUTINE kernel
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    cuf_transform = SccCuf(
        horizontal=horizontal, vertical=vertical, block_dim=blocking #  , hoist_column_arrays=False
    )

    cuf_transform.apply(driver, role='driver', targets=['kernel'])
    cuf_transform.apply(kernel, role='kernel')

    print("driver\n-------------------------------------------------------")
    print(cufgen(driver))
    print('\n')
    print("kernel\n-------------------------------------------------------")
    print(cufgen(kernel))

    assert False


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_parametrise(here, frontend, config, horizontal, vertical, blocking):

    proj = here / 'sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuf_transform = SccCuf(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type=0
    )
    scheduler.process(transformation=cuf_transform)

    dic2p = {'nz': 137}
    scheduler.process(transformation=ParametriseTransformation(dic2p=dic2p))

    for item in scheduler.items:
        suffix = f'.scc_cuf.parametrise.F90'
        sourcefile = item.source
        sourcefile.write(path=Path('testing/parametrise')/sourcefile.path.with_suffix(suffix).name, cuf=True)

    assert False


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_hoist(here, frontend, config, horizontal, vertical, blocking):

    class HoistTemporaryArraysTransformationDeviceAllocatable(HoistVariablesTransformation):

        def __init__(self, key=None, **kwargs):
            super().__init__(key=key, **kwargs)

        def driver_variable_declaration(self, routine, var):
            type = var.type.clone(device=True, allocatable=True)
            routine.variables += tuple([var.clone(scope=routine, dimensions=as_tuple(
                [sym.RangeIndex((None, None))] * (len(var.dimensions))), type=type)])

            # EITHER
            # routine.body.prepend(Allocation((var.clone(),)))
            # routine.body.append(Deallocation((var.clone(dimensions=None),)))

            # OR: just for better formatting ...
            allocations = FindNodes(Allocation).visit(routine.body)
            if allocations:
                insert_index = routine.body.body.index(allocations[-1])
                routine.body.insert(insert_index + 1, Allocation((var.clone(),)))
            else:
                routine.body.prepend(Allocation((var.clone(),)))
            de_allocations = FindNodes(Deallocation).visit(routine.body)
            if allocations:
                insert_index = routine.body.body.index(de_allocations[-1])
                routine.body.insert(insert_index + 1, Deallocation((var.clone(dimensions=None),)))
            else:
                routine.body.append(Deallocation((var.clone(dimensions=None),)))


    proj = here / 'sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuf_transform = SccCuf(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type=1
    )
    scheduler.process(transformation=cuf_transform)

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis(), reverse=True)
    # Transformation: Synthesis
    scheduler.process(transformation=HoistTemporaryArraysTransformationDeviceAllocatable())

    for item in scheduler.items:
        suffix = f'.scc_cuf.hoist.F90'
        sourcefile = item.source
        sourcefile.write(path=Path('testing/hoist')/sourcefile.path.with_suffix(suffix).name, cuf=True)

    assert False


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_dynamic_memory(here, frontend, config, horizontal, vertical, blocking):

    proj = here / 'sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuf_transform = SccCuf(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type=2
    )
    scheduler.process(transformation=cuf_transform)

    for item in scheduler.items:
        suffix = f'.scc_cuf.dynamic_memory.F90'
        sourcefile = item.source
        sourcefile.write(path=Path('testing/dynamic')/sourcefile.path.with_suffix(suffix).name, cuf=True)

    assert False
