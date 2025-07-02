# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Utilities to manipulate vector notation in array expressions. """

from itertools import count

from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, Assignment, Loop, FindNodes,
    Transformer, FindVariables, SubstituteExpressions, FindInlineCalls
)
from loki.tools import as_tuple
from loki.types import SymbolAttributes, BasicType


__all__ = [
    'remove_explicit_array_dimensions', 'add_explicit_array_dimensions',
    'resolve_vector_notation', 'ResolveVectorNotationTransformer'
]


def remove_explicit_array_dimensions(routine, calls_only=False):
    """
    Remove colon notation from array dimensions within :any:`Subroutine` ``routine``.
    E.g., convert two-dimensional array ``arr2d(:,:)`` to ``arr2d`` or
    ``arr3d(:,:,:)`` to ``arr3d``, but NOT e.g., ``arr(1,:,:)``.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check
    calls_only: bool
        Whether to remove colon notation from array dimensions only
        from arrays within (inline) calls or all arrays (default: False)
    """
    if calls_only:
        # handle calls (to subroutines) and inline calls (to functions)
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        inline_calls = FindInlineCalls().visit(routine.body)
        inline_call_map = {}
        for call in as_tuple(calls) + as_tuple(inline_calls):
            # handle arguments
            arguments = ()
            for arg in call.arguments:
                if isinstance(arg, sym.Array) and all(dim == sym.RangeIndex((None, None)) for dim in arg.dimensions):
                    new_dimensions = None
                    arguments += (arg.clone(dimensions=new_dimensions),)
                else:
                    arguments += (arg,)
            # handle kwargs
            kwarguments = ()
            for (kwarg_name, kwarg) in call.kwarguments:
                if isinstance(kwarg, sym.Array) and all(dim==sym.RangeIndex((None, None)) for dim in kwarg.dimensions):
                    kwarguments += ((kwarg_name, kwarg.clone(dimensions=None)),)
                else:
                    kwarguments += ((kwarg_name, kwarg),)
            # distinguish calls and inline calls
            if isinstance(call, sym.InlineCall):
                inline_call_map[call] = call.clone(parameters=arguments, kw_parameters=kwarguments)
            else:
                # directly update calls
                call._update(arguments=arguments, kwarguments=kwarguments)
        if inline_call_map:
            # update inline calls via expression substitution
            routine.body = SubstituteExpressions(inline_call_map).visit(routine.body)
    else:
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if all(dim == sym.RangeIndex((None, None)) for dim in array.dimensions):
                new_dimensions = None
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)


def add_explicit_array_dimensions(routine):
    """
    Make dimensions of arrays explicit within :any:`Subroutine` ``routine``.
    E.g., convert two-dimensional array ``arr2d`` to ``arr2d(:,:)`` or
    ``arr3d`` to ``arr3d(:,:,:)``.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check
    """
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
    array_map = {}
    for array in arrays:
        if not array.dimensions:
            new_dimensions = (sym.RangeIndex((None, None)),) * len(array.shape)
            array_map[array] = array.clone(dimensions=new_dimensions)
    routine.body = SubstituteExpressions(array_map).visit(routine.body)


def resolve_vector_notation(routine):
    """
    Resolve implicit vector notation by inserting explicit loops
    """

    # Find available loops and create map {(lower, upper, step): loop_variable}
    loops = FindNodes(Loop).visit(routine.body)
    loop_map = {(loop.bounds.lower, loop.bounds.upper, loop.bounds.step or 1):
                loop.variable for loop in loops}

    transformer = ResolveVectorNotationTransformer(
        loop_map=loop_map, scope=routine, inplace=True
    )
    routine.body = transformer.visit(routine.body)

    # Add declarations for all newly create loop index variables
    routine.variables += tuple(set(transformer.index_vars))


class ResolveVectorNotationTransformer(Transformer):
    """
    A :any:`Transformer` that resolves implicit vector notation by
    inserting explicit loops.

    Parameters
    ----------
    loop_map : dict of tuple to :any:`Variable`
        A dict mapping the tuple ``(lower, upper, step)`` to
        a known variable symbol to use as loop index.
    scope : :any:`Subroutine` or :any:`Module`
        The scope in which to create new loop index variables
    """

    def __init__(self, *args, loop_map={}, scope=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.scope = scope
        self.loop_map = loop_map
        self.index_vars = set()

    def visit_Assignment(self, stmt):

        vmap = {}
        vdims = []
        shape_index_map = {}
        index_range_map = {}

        # Loop over all variables and replace them with loop indices
        for v in FindVariables(unique=False).visit(stmt):
            if not isinstance(v, sym.Array):
                continue

            # Skip if the entire array is used implicitly
            if not v.dimensions:
                continue

            ivar_basename = f'i_{stmt.lhs.basename}'
            for i, dim, s in zip(count(), v.dimensions, as_tuple(v.shape)):
                if isinstance(dim, sym.RangeIndex):
                    # use the shape for e.g., `ARR(:)`, but use the dimension for e.g., `ARR(2:5)`
                    _s = dim if dim.lower is not None else s
                    # create tuple to test whether an appropriate loop is already available
                    test_range = (sym.IntLiteral(1), _s, 1) if not isinstance(_s, sym.RangeIndex)\
                            else (_s.lower, _s.upper, 1)
                    # actually test for it
                    if test_range in self.loop_map:
                        # Use index variable of available matching loop
                        ivar = self.loop_map[test_range]
                    else:
                        # Create new index variable
                        vtype = SymbolAttributes(BasicType.INTEGER)
                        ivar = sym.Variable(name=f'{ivar_basename}_{i}', type=vtype, scope=self.scope)
                    shape_index_map[(i, s)] = ivar
                    index_range_map[ivar] = _s

                    if ivar not in vdims:
                        vdims.append(ivar)

            # Add index variable to range replacement
            new_dims = as_tuple(shape_index_map.get((i, s), d)
                                for i, d, s in zip(count(), v.dimensions, as_tuple(v.shape)))
            vmap[v] = v.clone(dimensions=new_dims)

        self.index_vars.update(list(vdims))

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

            # Return the loop nest to replace the statement
            return SubstituteExpressions(vmap).visit(loop)

        # No vector dimensions encountered, return unchanged
        return stmt
