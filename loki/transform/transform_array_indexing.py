"""
Collection of utility routines to deal with common array indexing conversions.


"""
from itertools import count

from loki.expression import symbols as sym, FindVariables, SubstituteExpressions
from loki.visitors import FindNodes
from loki.ir import Assignment, Loop
from loki.types import SymbolType, BasicType
from loki.visitors import Transformer
from loki.tools import as_tuple


__all__ = [
    'shift_to_zero_indexing', 'invert_array_indices',
    'resolve_vector_notation', 'normalize_range_indexing'
]


def shift_to_zero_indexing(routine):
    """
    Shift all array indices to adjust to 0-based indexing conventions (eg. for C or Python)
    """
    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            new_dims = []
            for d in v.dimensions.index_tuple:
                if isinstance(d, sym.RangeIndex):
                    start = d.start - sym.Literal(1) if d.start is not None else None
                    # no shift for stop because Python ranges are [start, stop)
                    new_dims += [sym.RangeIndex((start, d.stop, d.step))]
                else:
                    new_dims += [d - sym.Literal(1)]
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(as_tuple(new_dims)))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def invert_array_indices(routine):
    """
    Invert data/loop accesses from column to row-major

    TODO: Take care of the indexing shift between C and Fortran.
    Basically, we are relying on the CGen to shift the iteration
    indices and dearly hope that nobody uses the index's value.
    """
    # Invert array indices in the routine body
    vmap = {}
    for v in FindVariables(unique=True).visit(routine.body):
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions.index_tuple))
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(rdim))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Invert variable and argument dimensions for the automatic cast generation
    for v in routine.variables:
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions.index_tuple))
            if v.shape:
                rshape = as_tuple(reversed(v.shape))
                vmap[v] = v.clone(dimensions=sym.ArraySubscript(rdim),
                                  type=v.type.clone(shape=rshape))
            else:
                vmap[v] = v.clone(dimensions=sym.ArraySubscript(rdim))
    routine.variables = [vmap.get(v, v) for v in routine.variables]


def resolve_vector_notation(routine):
    """
    Resolve implicit vector notation by inserting explicit loops
    """
    loop_map = {}
    index_vars = set()
    vmap = {}
    for stmt in FindNodes(Assignment).visit(routine.body):
        # Loop over all variables and replace them with loop indices
        vdims = []
        shape_index_map = {}
        index_range_map = {}
        for v in FindVariables(unique=False).visit(stmt):
            if not isinstance(v, sym.Array):
                continue

            ivar_basename = 'i_%s' % stmt.lhs.basename
            for i, dim, s in zip(count(), v.dimensions.index_tuple, as_tuple(v.shape)):
                if isinstance(dim, sym.RangeIndex):
                    # Create new index variable
                    vtype = SymbolType(BasicType.INTEGER)
                    ivar = sym.Variable(name='%s_%s' % (ivar_basename, i), type=vtype,
                                        scope=routine.symbols)
                    shape_index_map[s] = ivar
                    index_range_map[ivar] = s

                    if ivar not in vdims:
                        vdims.append(ivar)

            # Add index variable to range replacement
            new_dims = as_tuple(shape_index_map.get(s, d)
                                for d, s in zip(v.dimensions.index_tuple, as_tuple(v.shape)))
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(new_dims))

        index_vars.update(list(vdims))

        # Recursively build new loop nest over all implicit dims
        if len(vdims) > 0:
            loop = None
            body = stmt
            for ivar in vdims:
                irange = index_range_map[ivar]
                if isinstance(irange, sym.RangeIndex):
                    bounds = sym.LoopRange(irange.children)
                else:
                    bounds = sym.LoopRange((sym.Literal(1), irange, sym.Literal(1)))
                loop = Loop(variable=ivar, body=as_tuple(body), bounds=bounds)
                body = loop

            loop_map[stmt] = loop

    if len(loop_map) > 0:
        routine.body = Transformer(loop_map).visit(routine.body)
    routine.variables += tuple(set(index_vars))

    # Apply variable substitution
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def normalize_range_indexing(routine):
    """
    Replace the ``(1:size)`` indexing in array sizes that OMNI introduces.
    """
    def is_one_index(dim):
        return isinstance(dim, sym.RangeIndex) and dim.lower == 1 and dim.step is None

    vmap = {}
    for v in routine.variables:
        if isinstance(v, sym.Array):
            new_dims = [d.upper if is_one_index(d) else d for d in v.dimensions.index_tuple]
            new_shape = [d.upper if is_one_index(d) else d for d in v.shape]
            new_type = v.type.clone(shape=as_tuple(new_shape))
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(as_tuple(new_dims)), type=new_type)
    routine.variables = [vmap.get(v, v) for v in routine.variables]
