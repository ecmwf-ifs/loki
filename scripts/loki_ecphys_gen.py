#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
An offline script to generate the complex control-flow patterns of the
EC-physics drivers.
"""

import click
from pathlib import Path

from loki import Sourcefile


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
def inline(source, build):
    """
    Inlines EC_PHYS and CALLPAR into EC_PHYS_DRV to expose the parallel loop.
    """
    source = Path(source)

    ec_phys_drv = Sourcefile.from_file(source/'ec_phys_drv.F90')['EC_PHYS_DRV']
    ec_phys = Sourcefile.from_file(source/'ec_phys.F90')['EC_PHYS']
    callpar = Sourcefile.from_file(source/'callpar.F90')['CALLPAR']

    ec_phys_drv.enrich(ec_phys)
    ec_phys.enrich(callpar)

    # Clone original and change subroutine name
    ec_phys_fc = ec_phys_drv.clone(name='EC_PHYS_FC')

    # And write the generated subroutine to file
    Sourcefile(path=build/'ec_phys_fc.F90', ir=(ec_phys_fc,)).write()
