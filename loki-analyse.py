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

from pathlib import Path
import click

from loki import (
    Sourcefile, FindNodes, CallStatement, flatten
)


def driver_analyse_field_offload_accesses(routine):
    """
    Check that offload accessors of FIELD API objects match the call intents.

    
    """

    calls = FindNodes(CallStatement).visit(routine.body)

    # Now what...?
    
    from IPython import embed; embed()


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
    
    cloudsc = Sourcefile.from_file(root/'arpifs/phys_ec/cloudsc.F90')
    cloud_satadj = Sourcefile.from_file(root/'arpifs/phys_ec/cloud_satadj.F90')
    cloud_layer = Sourcefile.from_file(root/'arpifs/phys_ec/cloud_layer.F90')

    driver = cloud_layer['cloud_layer_loki']
    driver.enrich([cloudsc['cloudsc'], cloud_satadj['cloud_satadj']])

    driver.enrich(definitions=flatten([h.definitions for h in headers]))

    driver_analyse_field_offload_accesses(driver)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
