# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes, pragma_regions_attached, is_loki_pragma
)
from loki.types import BasicType, SymbolAttributes

from loki.transformations.parallel.openmp_region import (
    do_remove_openmp_regions, do_remove_firstprivate_copies,
    add_openmp_parallel_region, InjectFirstprivateCopyTransformer
)
from loki.transformations.parallel.block_loop import (
    do_remove_block_loops, InsertBlockLoopTransformer
)
from loki.transformations.parallel.field_views import (
    do_remove_field_api_view_updates, InsertFieldAPIViewsTransformer
)
from loki.transformations.utilities import ensure_imported_symbols


__all__ = [
    'RemoveViewDriverLoopTransformation', 'AddViewDriverLoopTransformation'
]


class RemoveViewDriverLoopTransformation(Transformation):
    """

    """

    def __init__(
            self, remove_block_loops=True,
            remove_openmp_regions=True,
            remove_field_api_view_updates=True,
            remove_firstprivate_copies=True,
            dimension=None, dim_object=None,
            fprivate_map=None, field_group_types=None,
            insert_loki_parallel=False
    ):
        self.remove_block_loops = remove_block_loops
        self.remove_openmp_regions = remove_openmp_regions
        self.remove_field_api_view_updates = remove_field_api_view_updates
        self.remove_firstprivate_copies = remove_firstprivate_copies

        self.dimension = dimension
        self.dim_object = dim_object
        self.fprivate_map = fprivate_map
        self.field_group_types = field_group_types
        self.insert_loki_parallel = insert_loki_parallel

    def transform_subroutine(self, routine, **kwargs):

        if self.remove_openmp_regions:
            do_remove_openmp_regions(
                routine, insert_loki_parallel=self.insert_loki_parallel
            )

        if self.remove_firstprivate_copies:
            do_remove_firstprivate_copies(
                routine.body, fprivate_map=self.fprivate_map, scope=routine
            )

        if self.remove_block_loops:
            do_remove_block_loops(routine, dimension=self.dimension)

        if self.remove_field_api_view_updates:
            do_remove_field_api_view_updates(
                routine, dim_object=self.dim_object,
                field_group_types=self.field_group_types
            )


class AddViewDriverLoopTransformation(Transformation):
    """

    """

    def __init__(
            self, add_block_loops=True, add_openmp_regions=True,
            add_field_api_view_updates=True,
            add_firstprivate_copies=True, dimension=None,
            dim_object=None, fprivate_map=None,
            field_group_types=None, shared_variables=None,
            fprivate_variables=None
    ):
        self.add_block_loops = add_block_loops
        self.add_openmp_regions = add_openmp_regions
        self.add_field_api_view_updates = add_field_api_view_updates
        self.add_firstprivate_copies = add_firstprivate_copies

        self.dimension = dimension
        self.dim_object = dim_object
        self.fprivate_map = fprivate_map
        self.field_group_types = field_group_types
        self.shared_variables = shared_variables
        self.fprivate_variables = fprivate_variables

    def transform_subroutine(self, routine, **kwargs):

        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):

                if not is_loki_pragma(region.pragma, starts_with='parallel'):
                    continue

                # Ensure and derive default integer type
                ensure_imported_symbols(
                    routine, symbols='JPIM', module='PARKIND1'
                )
                default_type = SymbolAttributes(
                    BasicType.INTEGER, kind=routine.Variable(name='JPIM')
                )

                if self.add_block_loops:
                    # Insert the driver block loop
                    InsertBlockLoopTransformer(
                        inplace=True, dimension=self.dimension,
                        default_type=default_type
                    ).visit(region, scope=routine)

                if self.add_field_api_view_updates:
                    # Add field group view update calls
                    InsertFieldAPIViewsTransformer(
                        inplace=True, dimension=self.dimension,
                        dim_object=self.dim_object,
                        field_group_types=self.field_group_types,
                    ).visit(region, scope=routine)

                if self.add_firstprivate_copies:
                    # Add explicit firstprivatisation copies
                    InjectFirstprivateCopyTransformer(
                        inplace=True, fprivate_map=self.fprivate_map
                    ).visit(region, scope=routine)

                if self.add_openmp_regions:
                    # Need to re-generate dataflow info here, as prior
                    # transformers might have added new symbols.
                    with dataflow_analysis_attached(routine):

                        # Add OpenMP parallel region
                        add_openmp_parallel_region(
                            region=region, routine=routine,
                            dimension=self.dimension,
                            shared_variables=self.shared_variables,
                            fprivate_variables=self.fprivate_variables
                        )
