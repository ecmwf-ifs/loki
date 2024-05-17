# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Sub-package with utilities to remove and manipulate parallel OpenMP regions.
"""

from loki.analyse import dataflow_analysis_attached
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, Transformer, is_loki_pragma,
    pragmas_attached, pragma_regions_attached
)
from loki.tools import dict_override, flatten


__all__ = ['remove_openmp_regions', 'add_openmp_regions']


def remove_openmp_regions(routine, insert_loki_parallel=False):
    """
    Remove any OpenMP parallel annotations (``!$omp parallel``).

    Optionally, this can replace ``!$omp parallel`` with ``!$loki
    parallel`` pragmas.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The routine from which to strip all OpenMP annotations.
    insert_loki_parallel : bool
        Flag for the optional insertion of ``!$loki parallel` pragmas
    """

    class RemoveOpenMPRegionTransformer(Transformer):
        """
        Remove OpenMP pragmas from "parallel" regions and remove all
        contained OpenMP pragmas and pragma regions.

        Optionally replaces outer ``!$omp parallel`` region with
        ``!$loki parallel`` region.
        """

        def visit_PragmaRegion(self, region, **kwargs):
            """
            Perform the fileterin and removeal of OpenMP pragma regions.

            Parameters
            ----------
            active : tuple
                Flag to indicate whether we're actively traversing an
                outer OpenMP region.
            """
            if not region.pragma.keyword.lower() == 'omp':
                return region

            if kwargs['active'] and region.pragma.keyword.lower() == 'omp':
                # Remove other OpenMP pragam regions when in active mode
                region._update(pragma=None, pragma_post=None)
                return region

            if 'parallel' in region.pragma.content.lower():
                # Replace or remove pragmas
                pragma = None
                pragma_post = None
                if insert_loki_parallel:
                    pragma = ir.Pragma(keyword='loki', content='parallel')
                    pragma_post = ir.Pragma(keyword='loki', content='end parallel')

                with dict_override(kwargs, {'active': True}):
                    body = self.visit(region.body, **kwargs)

                region._update(body=body, pragma=pragma, pragma_post=pragma_post)

            return region

        def visit_Pragma(self, pragma, **kwargs):
            """ Remove other OpenMP pragmas if in active region """

            if kwargs['active'] and pragma.keyword.lower() == 'omp':
                return None

            return pragma

    with pragma_regions_attached(routine):
        routine.body = RemoveOpenMPRegionTransformer().visit(routine.body, active=False)


def add_openmp_regions(routine, global_variables=None, field_group_types=None):
    """
    Add the OpenMP directives for a parallel driver region with an
    outer block loop.
    """
    block_dim_size = 'YDGEOMETRY%YRDIM%NGPBLKS'

    global_variables = global_variables or {}
    field_group_types = field_group_types or {}

    # First get local variables and separate scalars and arrays
    routine_arguments = routine.arguments
    local_variables = tuple(
        v for v in routine.variables if v not in routine_arguments
    )
    local_scalars = tuple(
        v for v in local_variables if isinstance(v, sym.Scalar)
    )
    # Filter arrays by block-dim size, as these are global
    local_arrays = tuple(
        v for v in local_variables
        if isinstance(v, sym.Array) and not v.dimensions[-1] == block_dim_size
    )

    with pragma_regions_attached(routine):
        with dataflow_analysis_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='parallel'):
                    return

                # Accumulate the set of locally used symbols and chase parents
                symbols = tuple(region.uses_symbols | region.defines_symbols)
                symbols = tuple(dict.fromkeys(flatten(
                    s.parents if s.parent else s for s in symbols
                )))

                # Start with loop variables and add local scalars and arrays
                local_vars = tuple(dict.fromkeys(flatten(
                    loop.variable for loop in FindNodes(ir.Loop).visit(region.body)
                )))

                local_vars += tuple(v for v in local_scalars if v.name in symbols)
                local_vars += tuple(v for v in local_arrays if v.name in symbols )

                # Also add used symbols that might be field groups
                local_vars += tuple(dict.fromkeys(
                    v for v in routine_arguments
                    if v.name in symbols and str(v.type.dtype) in field_group_types
                ))

                # Filter out known global variables
                local_vars = tuple(v for v in local_vars if v.name not in global_variables)

                # Make field group types firstprivate
                firstprivates = tuple(dict.fromkeys(
                    v.name for v in local_vars if v.type.dtype.name in field_group_types
                ))
                # Also make values that have an initial value firstprivate
                firstprivates += tuple(v.name for v in local_vars if v.type.initial)

                # Mark all other variables as private
                privates = tuple(dict.fromkeys(
                    v.name for v in local_vars if v.name not in firstprivates
                ))

                s_fp_vars = ", ".join(str(v) for v in firstprivates)
                s_firstprivate = f'FIRSTPRIVATE({s_fp_vars})' if firstprivates else ''
                s_private = f'PRIVATE({", ".join(str(v) for v in privates)})' if privates else ''
                pragma_parallel = ir.Pragma(
                    keyword='OMP', content=f'PARALLEL {s_private} {s_firstprivate}'
                )
                region._update(
                    pragma=pragma_parallel,
                    pragma_post=ir.Pragma(keyword='OMP', content='END PARALLEL')
                )

                # And finally mark all block-dimension loops as parallel
                with pragmas_attached(routine, node_type=ir.Loop):
                    for loop in FindNodes(ir.Loop).visit(region.body):
                        # Add OpenMP DO directives onto block loops
                        if loop.variable == 'JKGLO':
                            loop._update(
                                pragma=ir.Pragma(keyword='OMP', content='DO SCHEDULE(DYNAMIC,1)'),
                                pragma_post=ir.Pragma(keyword='OMP', content='END DO'),
                            )
