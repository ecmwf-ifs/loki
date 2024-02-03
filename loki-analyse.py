#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A small set of utilities for offline analysis of control flow in IFS.


"""
from enum import Enum, IntEnum, Flag, auto
from pathlib import Path
import click

from loki import (
    Sourcefile, FindNodes, CallStatement, Loop, flatten, symbols as sym,
    dataflow_analysis_attached, info, warning
)


class Access(Flag):
    RD = auto()
    WR = auto()
    # RDWR = auto()

RD = Access.RD
WR = Access.WR


def driver_analyse_field_offload_accesses(routine):
    """
    Check that offload accessors of FIELD API objects match the call intents.

    
    """
    calls = FindNodes(CallStatement).visit(routine.body)

    # Get FIELD API arrays and implied access descriptors
    accessor_calls = [c for c in calls if 'GET_DEVICE_DATA' in str(c).upper()]
    field_map = {}
    field_map.update(
        {c.arguments[0]: RD for c in accessor_calls if str(c.name).endswith('RDONLY')}
    )
    field_map.update(
        {c.arguments[0]: WR | RD for c in accessor_calls if str(c.name).endswith('RDWR')}
    )
    field_map.update({
        c.arguments[-1]: WR
        for c in accessor_calls if len(c.arguments) == 2 and c.arguments[0] == 'WR'
    })

    # Get arrays that are CALL arguments and derive access descriptors from callee
    kernel_calls = [c for c in calls if 'GET_DEVICE_DATA' not in str(c.name)]
    kernel_calls = [c for c in kernel_calls if 'GET_HOST_DATA' not in str(c.name)]
    kernel_calls = [c for c in kernel_calls if 'DR_HOOK' not in str(c.name)]
    kernel_calls = [c for c in kernel_calls if 'UPDATE_FIELDS' not in str(c.name)]

    argument_map ={}
    _intent2access = {
        'in' : RD, 'inout' : RD | WR, 'out' : WR
    }
    for call in kernel_calls:
        arg_map = {
            var.name : _intent2access[arg.type.intent]
            for arg, var in call.arg_iter() if isinstance(arg, sym.Array)
        }
        # Update full map, but check for existing entries
        for v, acc in arg_map.items():
            if v in argument_map:
                argument_map[v] |= acc
            else:
                argument_map[v] = acc

    # Get the local accesses in the driver loops
    with dataflow_analysis_attached(routine):
        loops = [
            l for l in FindNodes(Loop).visit(routine.body)
            if l.variable == 'JKGLO'
        ]
        assert len(loops) == 1
        driver_loop = loops[0]

        for v in driver_loop.uses_symbols:
            if not isinstance(v, sym.Array):
                continue

            if v not in field_map:
                continue

            if v.name not in argument_map:
                argument_map[v.name] = RD
            else:
                argument_map[v.name] |= RD

        for v in driver_loop.defines_symbols:
            if not isinstance(v, sym.Array):
                continue

            if v not in field_map:
                continue

            if v.name not in argument_map:
                argument_map[v.name] = WR
            else:
                argument_map[v.name] |= WR

    info('[Loki-anaylse] ===================================================')
    info(f'[Loki-anaylse] ====    Field accesses:  {routine.name}      ====')
    info('[Loki-anaylse] ===================================================')
    
    # Compare declared access to analysis and complain
    for arr, acc in field_map.items():
        if not isinstance(arr, sym.Array):
            continue

        var = routine.variable_map[arr.name]
        arg_acc = argument_map.get(var.name, None)
        if not arg_acc:
            continue

        if not arg_acc == acc:
            if arg_acc == RD:
                warning(f'[Loki-anaylse] Field {var.name:<22} :: declared {acc:<12}  =>  usage {arg_acc} !!!')
            else:
                info(f'[Loki-anaylse] Field {var.name:<22} :: declared {acc:<12}  =>  usage {arg_acc}')

    info('[Loki-anaylse] ===================================================')
    info('[Loki-anaylse] ')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), multiple=True,
              help='Path to search during source exploration.')
def offload(source):

    root = Path(source[0])
    drivers = []

    headers = [
        Sourcefile.from_file(root/'arpifs/module/ecphys_state_type_mod.F90'),
    ]

    # ec_phys_fc = Sourcefile.from_file(root/'arpifs/phys_ec/ec_phys_fc_mod.F90')

    # GW-Drag parametrisation
    gwdrag = Sourcefile.from_file(root/'arpifs/phys_ec/gwdrag.F90')
    gwdrag_layer = Sourcefile.from_file(root/'arpifs/phys_ec/gwdrag_layer.F90')
    gwdrag_driver = gwdrag_layer['gwdrag_layer_loki']
    gwdrag_driver.enrich([gwdrag['gwdrag']])
    driver_analyse_field_offload_accesses(gwdrag_driver)

    # Turbulence parametrisation
    vdfouter = Sourcefile.from_file(root/'arpifs/phys_ec/vdfouter.F90')
    turbulence_layer = Sourcefile.from_file(root/'arpifs/phys_ec/turbulence_layer.F90')
    turbulence_driver = turbulence_layer['turbulence_layer_loki']
    turbulence_driver.enrich([vdfouter['vdfouter']])
    driver_analyse_field_offload_accesses(turbulence_driver)

    # Convection parametrisation
    cucalln = Sourcefile.from_file(root/'arpifs/phys_ec/cucalln.F90')
    cuancape2 = Sourcefile.from_file(root/'arpifs/phys_ec/cuancape2.F90')
    convection_layer = Sourcefile.from_file(root/'arpifs/phys_ec/convection_layer.F90')
    convection_driver = convection_layer['convection_layer_loki']
    convection_driver.enrich([cucalln['cucalln'], cuancape2['cuancape2']])
    driver_analyse_field_offload_accesses(convection_driver)

     # Cloud parametrisation
    cloudsc = Sourcefile.from_file(root/'arpifs/phys_ec/cloudsc.F90')
    cloud_satadj = Sourcefile.from_file(root/'arpifs/phys_ec/cloud_satadj.F90')
    cloud_layer = Sourcefile.from_file(root/'arpifs/phys_ec/cloud_layer.F90')
    cloud_driver = cloud_layer['cloud_layer_loki']
    cloud_driver.enrich([cloudsc['cloudsc'], cloud_satadj['cloud_satadj']])
    # driver.enrich(definitions=flatten([h.definitions for h in headers]))
    driver_analyse_field_offload_accesses(cloud_driver)
    

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
