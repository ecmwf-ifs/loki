# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""
A small selection of utility transformations that resolve certain code
constructs to unify code structure and make reasoning about Fortran
code easier.
"""

from loki.batch import Transformation
from loki.expression import symbols as sym,  LokiIdentityMapper
from loki.ir import nodes as ir, Transformer, NestedTransformer
from loki.logging import warning
from loki.tools import dict_override
from loki.types import SymbolTable


__all__ = [
    'AssociatesTransformation', 'do_resolve_associates',
    'ResolveAssociatesTransformer', 'do_merge_associates'
]


class AssociatesTransformation(Transformation):
    """
    :any:`Transformation` object to apply code sanitisation steps
    specific to :any:`Associate` nodes.

    It allows merging in nested :any:`Associate` scopes to move
    independent assocation pairs to the outermost scope, optionally
    restricted by a number of ``max_parents`` symbols.

    It also provides partial or full resolution of :any:`Associate`
    nodes by replacing ``identifier`` symbols with the corresponding
    ``selector`` in the node's body.

    Parameters
    ----------
    resolve_associates : bool, default: True
        Enable full or partial resolution of only :any:`Associate`
        scopes.
    merge_associates : bool, default: False
        Enable merging :any:`Associate` to the outermost possible
        scope in nested associate blocks.
    start_depth : int, optional
        Starting depth for partial resolution of :any:`Associate`
        after merging.
    max_parents : int, optional
        Maximum number of parent symbols for valid selector to have
        when merging :any:`Associate` nodes.
    """

    def __init__(
            self, resolve_associates=True, merge_associates=False,
            start_depth=0, max_parents=None
    ):
        self.resolve_associates = resolve_associates
        self.merge_associates = merge_associates

        self.start_depth = start_depth
        self.max_parents = max_parents

    def transform_subroutine(self, routine, **kwargs):

        # Merge associates first so that remainig ones can be resolved
        if self.merge_associates:
            do_merge_associates(routine, max_parents=self.max_parents)

        # Resolve remaining associates depending on start_depth
        if self.resolve_associates:
            do_resolve_associates(routine, start_depth=self.start_depth)


def do_resolve_associates(routine, start_depth=0):
    """
    Resolve :any:`Associate` mappings in the body of a given routine.

    Optionally, partial resolution of only inner :any:`Associate`
    mappings is supported when a ``start_depth`` is specified.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to resolve all associate blocks.
    start_depth : int, optional
        Starting depth for partial resolution of :any:`Associate`
    """
    transformer = ResolveAssociatesTransformer(start_depth=start_depth)
    routine.body = transformer.visit(routine.body)

    # Ensure that all symbols have the appropriate scope attached.
    # This is needed, as the parent of a symbol might have changed,
    # which affects the symbol's type-defining scope.
    routine.rescope_symbols()


class ResolveAssociateMapper(LokiIdentityMapper):
    """
    Exppression mapper that will resolve symbol associations due
    :any:`Associate` scopes.

    The mapper will inspect the associated scope of each symbol
    and replace it with the inverse of the associate mapping.
    """

    def __init__(self, *args, start_depth=0, **kwargs):
        self.start_depth = start_depth
        super().__init__(*args, **kwargs)

    @staticmethod
    def _match_range_indices(expressions, indices):
        """ Map :data:`indices` to free ranges in :data:`expressions` """
        assert isinstance(expressions, tuple)
        assert isinstance(indices, tuple)

        free_symbols = tuple(e for e in expressions if isinstance(e, sym.RangeIndex))
        if any(s.lower not in (None, 1) for s in free_symbols):
            warning('WARNING: Bounds shifts through association is currently not supported')

        if len(free_symbols) == len(indices):
            # If the provided indices are enough to bind free symbols,
            # we match them in sequence.
            it = iter(indices)
            return tuple(
                next(it) if isinstance(e, sym.RangeIndex) else e
                for e in expressions
            )

        return expressions

    def map_scalar(self, expr, *args, **kwargs):
        # Skip unscoped expressions
        if not hasattr(expr, 'scope'):
            return self.rec(expr, *args, **kwargs)

        # Stop if scope is not an associate
        if not isinstance(expr.scope, ir.Associate):
            return expr

        scope = expr.scope

        # Determine the depth of the symbol-defining associate
        depth = len(tuple(
            p for p in scope.parents if isinstance(p, ir.Associate)
        )) + 1
        if depth <= self.start_depth:
            return expr

        # Recurse on parent first and propagate scope changes
        parent = self.rec(expr.parent, *args, **kwargs)
        if parent != expr.parent:
            expr = expr.clone(parent=parent, scope=parent.scope)

        # Find a match in the given inverse map
        if expr.basename in scope.inverse_map:
            expr = scope.inverse_map[expr.basename]
            return self.rec(expr, *args, **kwargs)

        # Update the scope, as any inner associates will be removed.
        # For this we count backwards the nested scopes, the tail of
        # which will the (innermost) associates.
        new_scope = scope.parents[::-1][depth-self.start_depth-1]
        return expr.clone(scope=new_scope)

    def map_array(self, expr, *args, **kwargs):
        """ Partially resolve dimension indices and handle shape """

        # Recurse over existing array dimensions
        expr_dims = self.rec(expr.dimensions, *args, **kwargs)

        # Recurse over the type's shape
        _type = expr.type
        if expr.type.shape:
            new_shape = self.rec(expr.type.shape, *args, **kwargs)
            _type = expr.type.clone(shape=new_shape)

        # Stop if scope is not an associate
        if not isinstance(expr.scope, ir.Associate):
            return expr.clone(dimensions=expr_dims, type=_type)

        new = self.map_scalar(expr, *args, **kwargs)

        # Recurse over array dimensions
        if isinstance(new, sym.Array) and new.dimensions:
            # Resolve unbound range symbols form existing indices
            new_dims = self.rec(new.dimensions, *args, **kwargs)
            new_dims = self._match_range_indices(new_dims, expr_dims)
        else:
            new_dims = expr_dims

        return new.clone(dimensions=new_dims, type=_type)

    map_variable_symbol = map_scalar
    map_deferred_type_symbol = map_scalar
    map_procedure_symbol = map_scalar


class ResolveAssociatesTransformer(Transformer):
    """
    :any:`Transformer` class to resolve :any:`Associate` nodes in IR trees.

    This will replace each :any:`Associate` node with its own body,
    where all ``identifier`` symbols have been replaced with the
    corresponding ``selector`` expression defined in ``associations``.

    Importantly, this :any:`Transformer` can also be applied over partial
    bodies of :any:`Associate` bodies.

    Optionally, partial resolution of only inner :any:`Associate`
    mappings is supported when a ``start_depth`` is specified.

    Parameters
    ----------
    start_depth : int, optional
        Starting depth for partial resolution of :any:`Associate`
    """
    # pylint: disable=unused-argument

    def __init__(self, start_depth=0, **kwargs):
        self.start_depth = start_depth
        super().__init__(**kwargs)

    def visit_Expression(self, o, **kwargs):
        return ResolveAssociateMapper(start_depth=self.start_depth)(o)

    def visit_Associate(self, o, **kwargs):
        """
        Replaces an :any:`Associate` node with its transformed body
        """

        # Establish traversal depth in kwargs
        depth = kwargs.get('depth', 1)

        # First head-recurse, so that all associate blocks beneath are resolved
        with dict_override(kwargs, {'depth': depth + 1}):
            body = self.visit(o.body, **kwargs)

        if depth <= self.start_depth:
            return o.clone(body=body)

        return body

    def visit_CallStatement(self, o, **kwargs):
        arguments = self.visit(o.arguments, **kwargs)
        kwarguments = tuple((k, self.visit(v, **kwargs)) for k, v in o.kwarguments)
        return o._rebuild(arguments=arguments, kwarguments=kwarguments)


def do_merge_associates(routine, max_parents=None):
    """
    Moves associate mappings in :any:`Associate` within a
    :any:`Subroutine` to the outermost parent scope.

    Please see :any:`MergeAssociatesTransformer` for mode details.

    Note
    ----
    This method can be combined with :any:`resolve_associates` to
    create a more unified look-and-feel for nested ASSOCIATE blocks.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to resolve all associate blocks.
    max_parents : int, optional
        Maximum number of parent symbols for valid selector to have.
    """
    transformer = MergeAssociatesTransformer(max_parents=max_parents)
    routine.body = transformer.visit(routine.body)


class MergeAssociatesTransformer(NestedTransformer):
    """
    :any:`NestedTransformer` that moves associate mappings in
    :any:`Associate` to parent nodes.

    If a selector expression depends on a symbol from a parent
    :any:`Associate` exists, it does not get moved.

    Additionally, a maximum parent-depth can be specified for the
    selector to prevent overly long symbols to be moved up.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to resolve all associate blocks.
    max_parents : int, optional
        Maximum number of parent symbols for valid selector to have.
    """

    def __init__(self, max_parents=None, **kwargs):
        self.max_parents = max_parents
        super().__init__(**kwargs)

    def visit_Associate(self, o, **kwargs):
        body = self.visit(o.body, **kwargs)

        if not o.parent or not isinstance(o.parent, ir.Associate):
            return o._rebuild(body=body, rescope_symbols=True)

        # Find all associate mapping that can be moved up
        to_move = tuple(
            (expr, name) for expr, name in o.associations
            if not expr.scope == o.parent
        )

        if self.max_parents:
            # Optionally filter by depth of symbol-parentage
            to_move = tuple(
                (expr, name) for expr, name in to_move
                if not len(expr.parents) > self.max_parents
            )

        # Move up to parent ...
        parent_assoc = tuple(
            (expr, name) for expr, name in to_move
            if (expr, name) not in o.parent.associations
        )
        o.parent._update(associations=o.parent.associations + parent_assoc)

        # ... and remove from this associate node
        new_assocs = tuple(
            (expr, name) for expr, name in o.associations
            if (expr, name) not in to_move
        )
        o = o._rebuild(
            body=body, associations=new_assocs, parent=o.parent,
            rescope_symbols=True, symbol_attrs=SymbolTable()
        )
        # We rebuild the local symbol-table from scratch to ensure
        # that moved associations get the correct defining scope
        o._derive_local_symbol_types(parent_scope=o.parent)
        return o
