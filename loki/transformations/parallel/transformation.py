# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation

from loki.transformations.parallel.openmp_region import (
    do_remove_openmp_regions, do_remove_firstprivate_copies
)
from loki.transformations.parallel.block_loop import do_remove_block_loops
from loki.transformations.parallel.field_views import do_remove_field_api_view_updates


__all__ = ['RemoveViewDriverLoopTransformation']


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
