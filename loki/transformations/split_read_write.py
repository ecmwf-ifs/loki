# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from loki.batch import Transformation, ProcedureItem
from loki.expression import FindVariables, Array, SubstituteExpressions
from loki.tools import as_tuple, flatten
from loki.ir import (
    pragma_regions_attached, is_loki_pragma, nodes as ir, FindNodes,
    Transformer, FindScopes
)

__all__ = ['SplitReadWriteTransformation']

class SplitReadWriteTransformation(Transformation):
    """
    When accumulating values to multiple components of an array, a compiler cannot rule out
    the possibility that the indices alias the same address. Consider for example the following
    code:

    .. code-block:: fortran

        !$loki split read-write
        do jlon=1,nproma
           var(jlon, n1) = var(jlon, n1) + 1.
           var(jlon, n2) = var(jlon, n2) + 1.
        enddo
        !$loki end split read-write

    In the above example, there is no guarantee that ``n1`` and ``n2`` do not in fact point to the same location.
    Therefore the load and store instructions for ``var`` have to be executed in order.

    For cases where the user knows ``n1`` and ``n2`` indeed represent distinct locations, this transformation
    provides a pragma assisted mechanism to split the reads and writes, and therefore make the loads independent
    from the stores. The above code would therefore be transformed to:

    .. code-block:: fortran

        !$loki split read-write
        do jlon=1,nproma
           loki_temp_0(jlon) = var(jlon, n1) + 1.
           loki_temp_1(jlon) = var(jlon, n2) + 1.
        enddo

        do jlon=1,nproma
           var(jlon, n1) = loki_temp_0(jlon)
           var(jlon, n2) = loki_temp_1(jlon)
        enddo
        !$loki end split read-write
    """

    item_filter = (ProcedureItem,)

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def transform_subroutine(self, routine, **kwargs):

        # initialise working vars, lists and maps
        temp_vars = []
        region_map = {}
        temp_counter = 0

        # cache variable_map for fast lookup later
        variable_map = routine.variable_map

        # find split read-write pragmas
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma, starts_with='split read-write'):

                    # find assignments inside pragma region
                    assigns = FindNodes(ir.Assignment).visit(region.body)

                    # filter-out non read-write assignments
                    assigns = [a for a in assigns if a.lhs in FindVariables().visit(a.rhs)]

                    # filter-out scalars
                    assigns = [a for a in assigns if isinstance(a.lhs, Array)]

                    # delete all leafnodes in second copy of region
                    assign_read_map = {}
                    assign_write_map = {leaf: None for leaf in FindNodes(ir.LeafNode).visit(region.body)}

                    lhs_var_map = {}
                    lhs_vars = set(a.lhs for a in assigns)
                    lhs_var_read_map = {var: False for var in lhs_vars}
                    temp_counter_map = {var: count + temp_counter for count, var in enumerate(lhs_vars)}
                    temp_counter += len(temp_counter_map)

                    for assign in assigns:

                        # determine all ancestor loops of assignment
                        parent_loop_dims = []
                        ancestors = flatten(FindScopes(assign).visit(region.body))
                        for a in ancestors:
                            if isinstance(a, ir.Loop):
                                dim = [dim for dim in self.dimensions if a.variable.name.lower() == dim.index.lower()]
                                assert dim
                                parent_loop_dims += [dim[0]]

                        # determine shape of temporary declaration and assignment
                        _shape = []
                        _dims = []
                        for s in assign.lhs.type.shape:
                            if (dim := [dim for dim in self.dimensions if s in dim.size_expressions]):
                                if dim[0] in parent_loop_dims:
                                    _shape += [variable_map[dim[0].size]]
                                    _dims += [variable_map[dim[0].index]]

                        # define vars to store temporary assignment
                        _type = assign.lhs.type.clone(shape=as_tuple(_shape), intent=None)
                        temp_vars += [assign.lhs.clone(name=f'loki_temp_{temp_counter_map[assign.lhs]}',
                                                       dimensions=as_tuple(_dims), type=_type),]

                        # split reads and writes
                        rhs = SubstituteExpressions(lhs_var_map).visit(assign.rhs)
                        if not lhs_var_read_map[assign.lhs]:
                            lhs_var_map.update({assign.lhs: temp_vars[-1]})
                            lhs_var_read_map[assign.lhs] = True

                            new_write = ir.Assignment(lhs=assign.lhs, rhs=temp_vars[-1])
                            assign_write_map[assign] = as_tuple(new_write)

                        new_read = ir.Assignment(lhs=temp_vars[-1], rhs=rhs)
                        assign_read_map[assign] = as_tuple(new_read)

                    # create two copies of the pragma region, the second containing
                    # only the newly split writes
                    new_reads = Transformer(assign_read_map).visit(region.body)
                    new_writes = Transformer(assign_write_map).visit(region.body)
                    region_map[region.body] = (new_reads, new_writes)

        # add declarations for temporaries
        if temp_vars:
            temp_vars = set(var.clone(dimensions=var.type.shape) for var in temp_vars)
            routine.variables += as_tuple(temp_vars)

        routine.body = Transformer(region_map).visit(routine.body)
