# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from loki.batch import Transformation, ProcedureItem
from loki.expression import Array
from loki.tools import as_tuple
from loki.ir import (
    nodes as ir, pragma_regions_attached, is_loki_pragma, FindNodes,
    Transformer, SubstituteExpressions
)

__all__ = ['SplitReadWriteTransformation']

class SplitReadWriteWalk(Transformer):
    """
    A :any:`Transformer` class to traverse the IR, in-place replace read-write
    assignments with reads, and build a transformer map for the corresponding writes.

    Parameters
    ----------
    dimensions : list
       A list of :any:`Dimension` objects corresponding to all :any:`Loop`s in the ``!$loki split-read-write`` region.
    variable_map : dict
       The variable_map of the parent :any:`Subroutine`.
    count : int
       A running count of the newly created temporaries in the parent :any:`Subroutine` so that
       temporaries created by previous ``!$loki split-read-write`` regions are not redefined.
    """

    def __init__(self, dimensions, variable_map, count=-1, **kwargs):
        self.write_map = {}
        self.temp_count = count
        self.lhs_var_map = {}
        self.dimensions = dimensions
        self.tmp_vars = []

        # parent subroutine variable_map
        self.variable_map = variable_map

        kwargs['inplace'] = True
        super().__init__(**kwargs)

    def visit_Loop(self, o, **kwargs):

        dim = [d for d in self.dimensions if d.index == o.variable]
        dim_nest = kwargs.pop('dim_nest', [])
        return super().visit_Node(o, dim_nest=dim_nest + dim, **kwargs)

    def visit_Assignment(self, o, **kwargs):

        dim_nest = kwargs.pop('dim_nest', [])
        write = None

        # filter out non read-write assignments and scalars
        if isinstance(o.lhs, Array) and o.lhs.name in o.rhs:

            rhs = SubstituteExpressions(self.lhs_var_map).visit(o.rhs)
            if not o.lhs in self.lhs_var_map:
                _dims = []
                _shape = []

                # determine shape of temporary declaration and assignment
                for s in o.lhs.type.shape:
                    if (dim := [dim for dim in self.dimensions
                                if s in dim.size_expressions]):
                        if dim[0] in dim_nest:
                            _shape += [self.variable_map[dim[0].size]]
                            _dims += [self.variable_map[dim[0].index]]

                # define var to store temporary assignment
                self.temp_count += 1
                _type = o.lhs.type.clone(shape=as_tuple(_shape), intent=None)
                tmp_var = o.lhs.clone(name=f'loki_temp_{self.temp_count}',
                                      dimensions=as_tuple(_dims), type=_type)
                self.lhs_var_map[o.lhs] = tmp_var
                self.tmp_vars += [tmp_var,]

                write = as_tuple(ir.Assignment(lhs=o.lhs, rhs=tmp_var))

            o._update(lhs=self.lhs_var_map[o.lhs], rhs=rhs)

        self.write_map[o] = write
        return o

    def visit_LeafNode(self, o, **kwargs):
        # remove all other leaf nodes from second copy of region
        self.write_map[o] = None
        return super().visit_Node(o, **kwargs)

class SplitReadWriteTransformation(Transformation):
    """
    When accumulating values to multiple components of an array, a compiler cannot rule out
    the possibility that the indices alias the same address. Consider for example the following
    code:

    .. code-block:: fortran

        !$loki split-read-write
        do jlon=1,nproma
           var(jlon, n1) = var(jlon, n1) + 1.
           var(jlon, n2) = var(jlon, n2) + 1.
        enddo
        !$loki end split-read-write

    In the above example, there is no guarantee that ``n1`` and ``n2`` do not in fact point to the same location.
    Therefore the load and store instructions for ``var`` have to be executed in order.

    For cases where the user knows ``n1`` and ``n2`` indeed represent distinct locations, this transformation
    provides a pragma assisted mechanism to split the reads and writes, and therefore make the loads independent
    from the stores. The above code would therefore be transformed to:

    .. code-block:: fortran

        !$loki split-read-write
        do jlon=1,nproma
           loki_temp_0(jlon) = var(jlon, n1) + 1.
           loki_temp_1(jlon) = var(jlon, n2) + 1.
        enddo

        do jlon=1,nproma
           var(jlon, n1) = loki_temp_0(jlon)
           var(jlon, n2) = loki_temp_1(jlon)
        enddo
        !$loki end split-read-write

    Parameters
    ----------
    dimensions : list
       A list of :any:`Dimension` objects corresponding to all :any:`Loop`s in the ``!$loki split-read-write`` region.
    """

    item_filter = (ProcedureItem,)

    def __init__(self, dimensions):
        self.dimensions = as_tuple(dimensions)

    def transform_subroutine(self, routine, **kwargs):

        # cache variable_map for fast lookup later
        variable_map = routine.variable_map
        temp_counter = -1
        tmp_vars = []

        # find split read-write pragmas
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma, starts_with='split-read-write'):

                    transformer = SplitReadWriteWalk(self.dimensions, variable_map, count=temp_counter)
                    transformer.visit(region.body)

                    temp_counter += (transformer.temp_count + 1)
                    tmp_vars += transformer.tmp_vars

                    if transformer.write_map:
                        new_writes = Transformer(transformer.write_map).visit(region.body)
                        region.append(new_writes)

        # add declarations for temporaries
        if tmp_vars:
            tmp_vars = set(var.clone(dimensions=var.type.shape) for var in tmp_vars)
            routine.variables += as_tuple(tmp_vars)
