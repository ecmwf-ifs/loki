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
    nodes as ir, FindNodes, Transformer, SubstituteStringExpressions,
    pragma_regions_attached, is_loki_pragma, FindVariables
)
from loki.types import DerivedType


__all__ = [
    'remove_openmp_regions', 'remove_explicit_firstprivatisation',
    'create_explicit_firstprivatisation'
]


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


def remove_explicit_firstprivatisation(region, fprivate_map, routine):
    """
    Removes an IFS-specific workaround, where complex derived-type
    objects are explicitly copied into a local copy of the object to
    avoid erroneous firstprivatisation in OpenMP loops.

    Parameters
    ----------
    region : tuple of :any:`Node`
        The code region from which to remove firstprivate copies
    fprivate_map : dict of (str, str)
        String mapping of local-to-global names for explicitly
        privatised objects
    routine : :any:`Subroutine`
        Subroutine to look up variable symbols in
    """

    class RemoveExplicitCopyTransformer(Transformer):
        """ Remove assignments that match the firstprivatisation map """

        def visit_Assignment(self, assign, **kwargs):
            if not isinstance(assign.lhs.type.dtype, DerivedType):
                return assign

            lhs = assign.lhs.name
            if lhs in fprivate_map and assign.rhs == fprivate_map[lhs]:
                return None
            return assign

    # Strip assignments of local copies
    region = RemoveExplicitCopyTransformer().visit(region)

    # Invert the local use of the private copy
    return SubstituteStringExpressions(fprivate_map, scope=routine).visit(region)


def create_explicit_firstprivatisation(routine, fprivate_map):
    """
    Injects IFS-specific thread-local copies of named complex derived
    type objects in parallel regions. This is to prevent issues with
    firstprivate variables in OpenMP loops.

    Parameters
    ----------
    routine : :any:`Subroutine`
        Subroutine in which to insert privatisation copies
    fprivate_map : dict of (str, str)
        String mapping of local-to-global names for explicitly
        privatised objects
    """
    inverse_map = {v: k for k, v in fprivate_map.items()}

    class InjectExplicitCopyTransformer(Transformer):
        """" Inject assignments that match the firstprivate map in parallel regions """

        def visit_PragmaRegion(self, region, **kwargs):
            # Apply to pragma-marked "parallel" regions only
            if not is_loki_pragma(region.pragma, starts_with='parallel'):
                return region

            # Collect the explicit privatisation copies
            lvars = FindVariables(unique=True).visit(region.body)
            assigns = ()
            for lcl, gbl in fprivate_map.items():
                lhs = routine.get_symbol(lcl)
                rhs = routine.get_symbol(gbl)
                if rhs in lvars:
                    assigns += (ir.Assignment(lhs=lhs, rhs=rhs),)

            # Remap from global to local name in marked regions
            region = SubstituteStringExpressions(inverse_map, scope=routine).visit(region)

            # Add the copies and return
            region.prepend(assigns)
            return region

    with pragma_regions_attached(routine):
        # Inject assignments of local copies
        routine.body = InjectExplicitCopyTransformer().visit(routine.body)
