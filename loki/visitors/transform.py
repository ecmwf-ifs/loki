"""
Visitor classes for transforming the IR
"""

from loki.ir import Node, Conditional
from loki.tools import flatten, is_iterable, as_tuple
from loki.visitors.visitor import Visitor


__all__ = ['Transformer', 'NestedTransformer', 'MaskedTransformer', 'NestedMaskedTransformer']


class Transformer(Visitor):
    r"""
    Visitor class to rebuild the tree and replace nodes according to a mapper.

    Given a control flow tree :math:`T` and a mapper from nodes in :math:`T`
    to a set of new nodes :math:`L, M : N \rightarrow L`, build a new control
    flow tree :math:`T'` where a node :math:`n \in N` is replaced with
    :math:`M(n)`.

    .. important::
       The mapping is applied before visiting any children of a node.

    *Removing nodes*: In the special case in which :math:`M(n)` is `None`,
    :math:`n` is dropped from :math:`T'`.

    *One to many mapping*: In the special case in which :math:`M(n)` is an
    iterable of nodes, all nodes in :math:`M(n)` are inserted into the tuple
    containing :math:`n`.

    .. warning::
       Applying a :class:`Transformer` to an IR tree rebuilds all nodes by
       default, which means individual nodes from the original IR are no longer
       found in the new tree. To update references to IR nodes, the attribute
       :any:`Transformer.rebuilt` provides a mapping from original to rebuilt
       nodes. Alternatively, with :data:`inplace` the mapping can be
       applied without rebuilding the tree, leaving existing references to
       individual IR nodes intact (as long as the mapping does not replace or
       remove them in the tree).

    Parameters
    ----------
    mapper : dict
        The mapping :math:`M : N \rightarrow L`.
    invalidate_source : bool, optional
        If set to `True`, this triggers invalidating the :data:`source`
        property of all parent nodes of a node :math:`n` if :math:`M(n)`
        has :data:`source=None`.
    inplace : bool, optional
        If set to `True`, all updates are performed on existing :any:`Node`
        objects, instead of rebuilding them, keeping the original tree intact.

    Attributes
    ----------
    rebuilt : dict
        After applying the :class:`Transformer` to an IR, this contains a
        mapping :math:`n \rightarrow n'` for every node of the original tree
        :math:`n \in T` to the rebuilt nodes in the new tree :math:`n' \in T'`.
    """

    def __init__(self, mapper=None, invalidate_source=True, inplace=False):
        super().__init__()
        self.mapper = mapper.copy() if mapper is not None else {}
        self.invalidate_source = invalidate_source
        self.rebuilt = {}
        self.inplace = inplace

    def _rebuild_without_source(self, o, children, **args):
        """
        Utility method to rebuild the given node without the source property.
        """
        args_frozen = o.args_frozen
        args_frozen.update(args)
        if 'source' in o.args_frozen:
            args_frozen['source'] = None

        if self.inplace:
            # Updated nodes in place, if requested
            o._update(*children, **args_frozen)
            return o

        # Rebuild updated nodes by default
        return o._rebuild(*children, **args_frozen)

    def _rebuild(self, o, children, **args):
        """
        Utility method to rebuild the given node with the provided children.

        If :data:`invalidate_source` is `True`, :data:`Node.source` is set to
        `None` whenever any of the children has :data:`source == None`.
        """
        args_frozen = o.args_frozen
        args_frozen.update(args)
        if self.invalidate_source and 'source' in args_frozen:
            child_has_no_source = [getattr(i, 'source', None) is None for i in flatten(children)]
            if any(child_has_no_source) or len(child_has_no_source) != len(flatten(o.children)):
                return self._rebuild_without_source(o, children, **args_frozen)

        if self.inplace:
            # Updated nodes in place, if requested
            o._update(*children, **args_frozen)
            return o

        # Rebuild updated nodes by default
        return o._rebuild(*children, **args_frozen)

    def visit_object(self, o, **kwargs):
        """Return the object unchanged."""
        return o

    def _inject_tuple_mapping(self, o):
        """
        Utility method for one-to-many mappings to insert iterables for
        the replaced node into a tuple.
        """
        for k, handle in self.mapper.items():
            if k in o and is_iterable(handle):
                i = o.index(k)
                o = o[:i] + tuple(handle) + o[i+1:]
        return o

    def visit_tuple(self, o, **kwargs):
        """
        Visit all elements in a tuple, injecting any one-to-many mappings.
        """
        o = self._inject_tuple_mapping(o)
        visited = tuple(self.visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        """
        Handler for :any:`Node` objects.

        It replaces :data:`o` by :data:`mapper[o]`, if it is in the mapper,
        otherwise visits all children before rebuilding the node.
        """
        if o in self.mapper:
            handle = self.mapper[o]
            if handle is None:
                # None -> drop /o/
                return None

            # For one-to-many mappings making sure this is not replaced again
            # as it has been inserted by visit_tuple already
            if not is_iterable(handle) or o not in handle:
                return handle._rebuild(**handle.args)

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)

    def visit(self, o, *args, **kwargs):
        """
        Apply this :class:`Transformer` to an IR tree.

        Parameters
        ----------
        o : :any:`Node`
            The node to visit.
        *args :
            Optional arguments to pass to the visit methods.
        **kwargs :
            Optional keyword arguments to pass to the visit methods.

        Returns
        -------
        :any:`Node` or tuple
            The rebuilt control flow tree.
        """
        obj = super().visit(o, *args, **kwargs)
        if isinstance(o, Node) and obj is not o:
            self.rebuilt[o] = obj
        return obj


class NestedTransformer(Transformer):
    """
    A :class:`Transformer` that applies replacements in a depth-first fashion.
    """

    def visit_Node(self, o, **kwargs):
        """
        Handler for :any:`Node` objects.

        It visits all children before applying the :data:`mapper`.
        """
        rebuilt = [self.visit(i, **kwargs) for i in o.children]
        handle = self.mapper.get(o, o)
        if handle is None:
            # None -> drop /o/
            return None
        if is_iterable(handle):
            if not o.children:
                raise ValueError
            extended = [tuple(handle) + rebuilt[0]] + rebuilt[1:]
            if self.invalidate_source:
                return self._rebuild_without_source(o, extended)
            return o._rebuild(*extended, **o.args_frozen)
        return self._rebuild(handle, rebuilt)


class MaskedTransformer(Transformer):
    """
    An enriched :class:`Transformer` that can selectively include or exclude
    parts of the tree.

    For that :class:`MaskedTransformer` is selectively switched on and
    off while traversing the tree. Nodes are only included in the new tree
    while it is "switched on".
    The transformer is switched on or off when it encounters nodes from
    :data:`start` or :data:`stop`, respectively. This can be used, e.g., to
    extract everything between two nodes, or to create a copy of the entire
    tree but without all nodes between two nodes.
    Multiple such ranges can be defined by providing more than one
    :data:`start` and :data:`stop` node, respectively.

    The sets :data:`start` and :data:`stop` are to be understood in a Pythonic
    way, i.e., :data:`start` nodes will be included in the result and
    :data:`stop` excluded.

    .. important::
       When recursing down a tree, any :any:`InternalNode` are only included
       in the tree if the :class:`MaskedTransformer` was switched on before
       visiting that :any:`InternalNode`. Importantly, this means the node is
       also not included if the transformer is switched on while traversing
       the internal node's body. In such a case, only the body nodes that are
       included are retained.

    Optionally as a variant, switching on can also be delayed until all nodes
    from :data:`start` have been encountered by setting
    :data:`require_all_start` to `True`.

    Optionally, traversal can be terminated early with :data:`greedy_stop`.
    If enabled, the :class:`MaskedTransformer` will stop completely to
    traverse the tree as soon as encountering a node from :data:`stop`.

    .. note::
       Enabling :data:`require_all_start` and :data:`greedy_stop` at the same
       time can be useful when you require the minimum number of nodes
       in-between multiple start and end nodes without knowing in which order
       they appear.

    Parameters
    ----------
    start : (iterable of) :any:`Node`, optional
        Encountering a node from :data:`start` during traversal switches the
        :class:`MaskedTransformer` on and includes that node and all
        subsequently traversed nodes in the produced tree.
    stop : (iterable of) :any:`Node`, optional
        Encountering a node from :data:`stop` during traversal switches the
        :class:`MaskedTransformer` off and excludes that node and all
        subsequently traversed nodes from the produced tree.
    active : bool, optional
        Switch the :class:`MaskedTransformer` on at the beginning of the
        traversal. By default, it is switched on only after encountering a node
        from :data:`start`.
    require_all_start : bool, optional
        Switch the :class:`MaskedTransformer` on only after encountering `all`
        nodes from :data:`start`. By default, it is switched on after
        encountering `any` node from :data:`start`.
    greedy_stop : bool, optional
        Stop traversing the tree as soon as any node from :data:`stop` is
        encountered. By default, traversal continues but nodes are excluded
        from the new tree until a node from :data:`start` is encountered.
    **kwargs : optional
        Keyword arguments that are passed to the parent class constructor.
    """

    def __init__(self, start=None, stop=None, active=False,
                 require_all_start=False, greedy_stop=False, **kwargs):
        super().__init__(**kwargs)

        self.start = set(as_tuple(start))
        self.stop = set(as_tuple(stop))
        self.active = active
        self.require_all_start = require_all_start
        self.greedy_stop = greedy_stop

    def visit(self, o, *args, **kwargs):
        # Vertical active status update
        if self.require_all_start:
            if o in self.start:
                # to record encountered nodes we remove them from the set of
                # start nodes and only if it is then empty we set active=True
                self.start.remove(o)
                self.active = self.active or not self.start
            else:
                self.active = self.active and o not in self.stop
        else:
            self.active = (self.active and o not in self.stop) or o in self.start
        if self.greedy_stop and o in self.stop:
            # to make sure that we don't include any following nodes we clear start
            self.start.clear()
            self.active = False
        return super().visit(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):
        if kwargs['parent_active']:
            # this is not an IR node but usually an expression tree or similar
            # we need to retain this only if the "parent" IR node is active
            return o
        return None

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            return super().visit_Node(o, **kwargs)

        # pass to children if this node is active
        kwargs['parent_active'] = self.active
        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        if kwargs['parent_active']:
            return self._rebuild(o, rebuilt)
        return tuple(i for i in rebuilt if i is not None) or None


class NestedMaskedTransformer(MaskedTransformer):
    """
    A :class:`MaskedTransformer` that retains parents for children that
    are included in the produced tree.

    In contrast to :class:`MaskedTransformer`, any encountered
    :any:`InternalNode` are included in the new tree as long as any of its
    children are included.
    """

    # Handler for leaf nodes


    def visit_object(self, o, **kwargs):
        """
        Return the object unchanged.

        Note that we need to keep them here regardless of the transformer
        being active because this handler takes care of properties for
        inactive parents that may still be retained if other children switch
        on the transformer.
        """
        return o

    def visit_LeafNode(self, o, **kwargs):
        """
        Handler for :any:`LeafNode` that are included in the tree if the
        :class:`NestedMaskedTransformer` is active.
        """
        if o in self.mapper:
            return super().visit_Node(o, **kwargs)
        if not self.active:
            # because any section/scope nodes are treated separately we can
            # simply drop inactive nodes
            return None

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)

    # Handler for block nodes

    def visit_InternalNode(self, o, **kwargs):
        """
        Handler for :any:`InternalNode` that are included in the tree as long
        as any :attr:`body` node is included.
        """
        if o in self.mapper:
            return super().visit_Node(o, **kwargs)

        rebuilt = [self.visit(i, **kwargs) for i in o.children]
        body_index = o._traversable.index('body')

        if rebuilt[body_index]:
            rebuilt[body_index] = flatten(rebuilt[body_index])

        # check if body still exists, otherwise delete this node
        if not rebuilt[body_index]:
            return None
        return self._rebuild(o, rebuilt)

    def visit_Conditional(self, o, **kwargs):
        """
        Handler for :any:`Conditional` to account for the :attr:`else_body`.

        .. note::
           This removes the :any:`Conditional` if :attr:`body` is empty. In
           that case, :attr:`else_body` is returned (which can be empty, too).
        """
        if o in self.mapper:
            return super().visit(o, **kwargs)

        condition = self.visit(o.condition, **kwargs)
        body = flatten(as_tuple(self.visit(o.body, **kwargs)))
        else_body = flatten(as_tuple(self.visit(o.else_body, **kwargs)))

        if not body:
            return else_body

        has_elseif = o.has_elseif and else_body and isinstance(else_body[0], Conditional)
        return self._rebuild(o, tuple((condition,) + (body,) + (else_body,)), has_elseif=has_elseif)

    def visit_MultiConditional(self, o, **kwargs):
        """
        Handler for :any:`MultiConditional` to account for all bodies.

        .. note::
           This removes the :any:`MultiConditional` if all of the
           :attr:`bodies` are empty. In that case, :attr:`else_body` is
           returned (which can be empty, too).
        """
        if o in self.mapper:
            return super().visit(o, **kwargs)

        # need to make (value, body) pairs to track vanishing bodies
        expr = self.visit(o.expr, **kwargs)
        branches = [(self.visit(c, **kwargs), self.visit(b, **kwargs))
                    for c, b in zip(o.values, o.bodies)]
        branches = [(c, b) for c, b in branches if flatten(as_tuple(b))]
        else_body = self.visit(o.else_body, **kwargs)

        # retain whatever is in the else body if all other branches are gone
        if not branches:
            return else_body

        # rebuild conditional with remaining branches
        values, bodies = zip(*branches)
        return self._rebuild(o, tuple((expr,) + (values,) + (bodies,) + (else_body,)))