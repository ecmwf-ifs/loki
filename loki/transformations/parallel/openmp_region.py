# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Sub-package with utilities to remove and manipulate parallel OpenMP regions.
"""

from loki.ir import nodes as ir, Transformer, pragma_regions_attached
from loki.tools import dict_override


__all__ = ['remove_openmp_regions']


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
