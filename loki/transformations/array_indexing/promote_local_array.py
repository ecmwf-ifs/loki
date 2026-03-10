# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import symbols as sym, is_dimension_constant, RangeIndex
from loki.ir import nodes as ir, FindNodes, Transformer, FindVariables, SubstituteExpressions
from loki.tools import as_tuple, OrderedSet
from loki import warning
from loki.transformations.array_indexing.promote import promote_variable_declarations
from loki.transformations.utilities import get_local_arrays, get_integer_variable, _is_deferred_shape

__all__ = ['PromoteLocalArrayTransformation']


class PromoteLocalArrayTransformation(Transformation):
    """
    Transformation to promote threadlocal arrays to vector dimension in kernels.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal, promote_local_arrays=True):
        self.horizontal = horizontal
        self.promote_local_arrays = promote_local_arrays

    @classmethod
    def get_locals_to_promote(cls, routine, sections, horizontal):
        """
        Create a list of local arrays to promote.

        Local arrays get promoted if they:
        * Are used within at least one vector section
        * Do not already contain the horizontal dimension
        * Have at least one non-constant dimension (i.e. are not
          compile-time-constant-sized arrays)
        * Do not have an entirely deferred shape (all ``:`` dimensions),
          since the actual shape is not known at compile time for
          pointer/allocatable locals
        """
        # Create a list of local temporary arrays to filter down
        candidates = get_local_arrays(routine, routine.spec)

        # Filter out arrays that already have the horizontal dimension or
        # are entirely compile-time-constant-sized
        candidates = [
            a for a in candidates if a.shape and
            all(s not in horizontal.size_expressions for s in a.shape) and
            not all(is_dimension_constant(d) for d in a.shape)
        ]

        # Filter out arrays with entirely deferred shapes (all ':' dimensions).
        # These are pointer or allocatable locals whose actual shape is not
        # known at compile time, so we cannot determine whether the horizontal
        # dimension is already present.
        skipped = [a for a in candidates if _is_deferred_shape(a.shape)]
        for a in skipped:
            warning(
                '[PromoteLocalArrayTransformation] Skipping %s in %s'
                ' — deferred shape (pointer/allocatable), cannot determine'
                ' if horizontal dimension is present',
                a.name, routine.name
            )
        candidates = [a for a in candidates if not _is_deferred_shape(a.shape)]

        # Create an index of variable names used per vector section
        vector_sec = {
            s: OrderedSet(
                v.name.lower() for v in get_local_arrays(routine, s, unique=False)
            ) for s in sections
        }

        # Only promote arrays that appear in at least one vector section
        candidates = [
            a for a in candidates
            if any(a.name.lower() in vs for vs in vector_sec.values())
        ]

        return candidates


    @classmethod
    def promote_variables_vector_sections(cls, routine, variable_names, pos, index, sections):
        """
        Promote variable uses by inserting a new index dimension, scoped
        to the given vector sections.

        This mirrors the index-insertion logic in :any:`promote_variables`
        but instead of using dataflow liveness to decide the index
        expression per node, it uses vector-section membership: uses
        inside the given *sections* get the provided *index*; remaining
        uses get ``:``.

        Only the variable **uses** in ``routine.body`` are updated; the
        declarations are not touched. Call
        :any:`promote_variable_declarations` separately to update the spec.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine whose body should be updated.
        variable_names : list of str
            The names of variables to promote. Matching is case-insensitive.
        pos : int
            The position of the new array dimension using Python indexing
            convention (count from 0, negative counts from the end).
        index : :py:class:`pymbolic.primitives.Expression`
            The indexing expression (or tuple for multi-dimension) to use
            inside vector sections, e.g. the horizontal loop variable.
        sections : list of :any:`Section`
            The vector-section IR nodes that define the "inside" region.
        """
        variable_names = {name.lower() for name in variable_names}
        if not variable_names:
            return

        index = as_tuple(index)

        # Create a copy of the tree and apply promotion in-place
        routine.body = Transformer().visit(routine.body)

        # Re-find vector sections from the cloned tree
        section_label = sections[0].label
        sections = [
            s for s in FindNodes(ir.Section).visit(routine.body)
            if s.label == section_label
        ]

        # Pass 1: walk vector sections — insert the index expression
        for section in sections:
            for node, var_list in FindVariables(unique=False, with_ir_node=True).visit(section):
                var_list = [v for v in var_list if v.name.lower() in variable_names]
                if not var_list:
                    continue

                var_map = {}
                for var in var_list:
                    var_dim = getattr(var, 'dimensions', ()) or ()
                    # If the variable is declared as an array but used without
                    # subscripts, fill in the existing dimensions with ':'.
                    # Note: declarations are already promoted at this point, so
                    # var.shape includes the new dimension; subtract len(index)
                    # to get the original number of dimensions.
                    if not var_dim and getattr(var, 'shape', None):
                        orig_ndim = len(var.shape) - len(index)
                        var_dim = tuple(sym.RangeIndex((None, None)) for _ in range(orig_ndim))
                    if pos < 0:
                        var_pos = len(var_dim) - pos + 1
                    else:
                        var_pos = pos
                    dimensions = as_tuple(var_dim[:var_pos] + index + var_dim[var_pos:])
                    var_map[var] = var.clone(dimensions=dimensions)

                # Apply immediately: identical variable uses in other nodes
                # may yield the same hash but need different substitutions
                SubstituteExpressions(var_map, inplace=True).visit(node)

        # Pass 2: walk the whole body — insert ':' for remaining un-indexed uses
        range_index = tuple(sym.RangeIndex((None, None)) for _ in index)
        for node, var_list in FindVariables(unique=False, with_ir_node=True).visit(routine.body):
            var_list = [v for v in var_list if v.name.lower() in variable_names]
            if not var_list:
                continue

            var_map = {}
            for var in var_list:
                var_dim = getattr(var, 'dimensions', ()) or ()
                # Skip variables already updated in pass 1
                if any(i in var_dim for i in index):
                    continue
                # If the variable is declared as an array but used without
                # subscripts, fill in the existing dimensions with ':'.
                # Note: declarations are already promoted at this point, so
                # var.shape includes the new dimension; subtract len(index)
                # to get the original number of dimensions.
                if not var_dim and getattr(var, 'shape', None):
                    orig_ndim = len(var.shape) - len(index)
                    var_dim = tuple(sym.RangeIndex((None, None)) for _ in range(orig_ndim))
                if pos < 0:
                    var_pos = len(var_dim) - pos + 1
                else:
                    var_pos = pos
                dimensions = as_tuple(var_dim[:var_pos] + range_index + var_dim[var_pos:])
                var_map[var] = var.clone(dimensions=dimensions)

            if var_map:
                SubstituteExpressions(var_map, inplace=True).visit(node)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply Promote utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)
        if role == 'kernel':
            promote_locals = self.promote_local_arrays
            preserve_arrays = []
            if item:
                promote_locals = item.config.get('promote_locals', self.promote_local_arrays)
                preserve_arrays = item.config.get('preserve_arrays', [])
            self.process_kernel(routine, promote_locals=promote_locals, preserve_arrays=preserve_arrays, item=item)

    def process_kernel(self, routine, promote_locals=True, preserve_arrays=None, item=None):
        """
        Applies the Promote utilities to a "kernel" and promotes all suitable local arrays.

        This does two things:

        1. Promotes the **declarations** of local arrays by appending a new
           horizontal dimension (using :any:`promote_variable_declarations`).
        2. Inserts the horizontal index variable into every **use** of those
           arrays that appears inside a vector section (using
           :any:`promote_variables_vector_sections`).

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        promote_locals : bool, optional
            Whether to promote local arrays; default: ``True``
        preserve_arrays : list of str, optional
            Names of arrays to exclude from promotion.
        item : optional
            Scheduler item for configuration overrides.
        """
        if not promote_locals:
            return

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [
            s for s in FindNodes(ir.Section).visit(routine.body)
            if s.label == 'vector_section'
        ]

        # Determine which local arrays to promote
        to_promote = self.get_locals_to_promote(routine, sections, self.horizontal)

        # Filter out arrays marked explicitly for preservation
        if preserve_arrays:
            to_promote = [v for v in to_promote if v.name not in preserve_arrays]

        if not to_promote:
            return

        variable_names = [var.name for var in to_promote]

        # Identify the horizontal bounds for the promoted dimension
        lower = None
        upper = None
        for l, u in zip(self.horizontal.lower, self.horizontal.upper):
            _lower = routine.variable_map.get(l, None)
            _upper = routine.variable_map.get(u, None)
            if _lower and _upper:
                if lower or upper:
                    warning(
                        f"[SCC::PromoteLocalArrayTransformation] Multiple horizontal "
                        f"loop variables in subroutine: {routine.name}"
                    )
                lower = _lower
                upper = _upper

        horizontal_var = get_integer_variable(routine, self.horizontal.index)

        # Step 1: Promote declarations
        promote_variable_declarations(
            routine, variable_names, pos=-1,
            size=RangeIndex((lower, upper))
        )

        # Step 2: Insert horizontal index into variable uses
        self.promote_variables_vector_sections(
            routine, variable_names, pos=-1,
            index=horizontal_var, sections=sections
        )

