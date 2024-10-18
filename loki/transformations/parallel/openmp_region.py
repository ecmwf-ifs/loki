# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Sub-package with utilities to remove and manipulate parallel OpenMP regions.
"""

from loki.ir import (
    nodes as ir, FindNodes, Transformer, pragma_regions_attached
)


__all__ = ['remove_openmp_regions']


def remove_openmp_regions(routine):
    """
    Remove any OpenMP annotations and replace with `!$loki parallel` pragmas
    """
    with pragma_regions_attached(routine):
        for region in FindNodes(ir.PragmaRegion).visit(routine.body):
            if region.pragma.keyword.lower() == 'omp':

                if 'PARALLEL' in region.pragma.content:
                    region._update(
                        pragma=ir.Pragma(keyword='loki', content='parallel'),
                        pragma_post=ir.Pragma(keyword='loki', content='end parallel')
                    )

    # Now remove all other pragmas
    pragma_map = {
        pragma: None for pragma in FindNodes(ir.Pragma).visit(routine.body)
        if pragma.keyword.lower() == 'omp'
    }
    routine.body = Transformer(pragma_map).visit(routine.body)

    # Note: This is slightly hacky, as some of the "OMP PARALLEL DO" regions
    # are not detected correctly! So instead we hook on the "OMP DO SCHEDULE"
    # and remove all other OMP pragmas.

    pragma_map = {
        pragma: None for pragma in FindNodes(ir.Pragma).visit(routine.body)
        if pragma.keyword == 'OMP'
    }
    routine.body = Transformer(pragma_map).visit(routine.body)
