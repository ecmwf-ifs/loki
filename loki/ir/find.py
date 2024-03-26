# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Visitor classes that allow searching the IR
"""
from itertools import groupby

from loki.ir.visitor import Visitor
from loki.tools import flatten

__all__ = [
    'FindNodes', 'SequenceFinder', 'PatternFinder', 'is_parent_of',
    'is_child_of', 'FindScopes'
]


class FindNodes(Visitor):
    """
    Find :any:`Node` instances that match a given criterion.

    Parameters
    ----------
    match :
        Node type(s) or node instance to look for.
    mode : optional
        Drive the search. Accepted values are:

        * ``'type'`` (default) : Collect all instances of type :data:`match`.
        * ``'scope'`` : Return the :any:`InternalNode` in which the object
          :data:`match` appears.
    greedy : bool, optional
        Do not recurse for children of a matched node.

    Returns
    -------
    list
        All nodes in the traversed IR that match the criteria.
    """

    @classmethod
    def default_retval(cls):
        """
        Default return value is an empty list.

        Returns
        -------
        list
        """
        return []

    rules = {
        'type': lambda match, o: isinstance(o, match),
        'scope': lambda match, o: match in flatten(o.children)
    }
    """
    Mapping of available :data:`mode` selectors to match rules.
    """

    def __init__(self, match, mode='type', greedy=False):
        super().__init__()
        self.match = match
        self.rule = self.rules[mode]
        self.greedy = greedy

    def visit_object(self, o, **kwargs):
        ret = kwargs.get('ret')
        return ret or self.default_retval()

    def visit_tuple(self, o, **kwargs):
        """
        Visit all elements in the iterable and return the combined result.
        """
        ret = kwargs.pop('ret', self.default_retval())
        for i in o:
            ret = self.visit(i, ret=ret, **kwargs)
        return ret or self.default_retval()

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        """
        Add the node to the returned list if it matches the criteria and visit
        all children.
        """
        ret = kwargs.pop('ret', self.default_retval())
        if self.rule(self.match, o):
            ret.append(o)
            if self.greedy:
                return ret
        for i in o.children:
            ret = self.visit(i, ret=ret, **kwargs)
        return ret or self.default_retval()

    def visit_TypeDef(self, o, **kwargs):
        """
        Custom handler for :any:`TypeDef` nodes that does not traverse the
        body (reason being that discovering nodes such as declarations from
        inside the type definition would be unexpected if called on a
        containing :any:`Subroutine` or :any:`Module`)
        """
        ret = kwargs.pop('ret', self.default_retval())
        if self.rule(self.match, o):
            ret.append(o)
            if self.greedy:
                return ret
        # Do not traverse children (i.e., TypeDef's body)
        return ret or self.default_retval()


def is_child_of(node, other):
    """
    Utility function to test relationship between nodes.

    Note that this can be expensive for large subtrees.

    Returns
    -------
    bool
        Return `True` if :data:`node` is contained in the IR below
        :data:`other`, otherwise return `False`.
    """
    return len(FindNodes(node, mode='scope', greedy=True).visit(other)) > 0


def is_parent_of(node, other):
    """
    Utility function to test relationship between nodes.

    Note that this can be expensive for large subtrees.

    Returns
    -------
    bool
        Return `True` if :data:`other` is contained in the IR below
        :data:`node`, otherwise return `False`.
    """
    return len(FindNodes(other, mode='scope', greedy=True).visit(node)) > 0


class FindScopes(FindNodes):
    """
    Find all parent nodes for node :data:`match`.

    Parameters
    ----------
    match : :any:`Node`
        The node for which the parent nodes are to be found.
    greedy : bool, optional
        Stop traversal when :data:`match` was found.
    """
    def __init__(self, match, greedy=True):
        super().__init__(match=match, greedy=greedy)
        self.rule = lambda match, o: match is o

    def visit_Node(self, o, **kwargs):
        """
        Add the node to the list of ancestors that is passed down to the
        children and, if :data:`o` is :data:`match`, return the list of
        ancestors.
        """
        ret = kwargs.pop('ret', self.default_retval())
        ancestors = kwargs.pop('ancestors', []) + [o]

        if self.rule(self.match, o):
            ret.append(ancestors)
            if self.greedy:
                return ret

        for i in o.children:
            ret = self.visit(i, ret=ret, ancestors=ancestors, **kwargs)
        return ret or self.default_retval()


class SequenceFinder(Visitor):
    """
    Find repeated nodes of the same type in lists/tuples within a given tree.

    Parameters
    ----------
    node_type :
        The node type to look for.
    """

    def __init__(self, node_type):
        super().__init__()
        self.node_type = node_type

    @classmethod
    def default_retval(cls):
        """
        Default return value is an empty list.

        Returns
        -------
        list
        """
        return []

    def visit_tuple(self, o, **kwargs):
        """
        Visit all children and look for sequences of matching type.
        """
        groups = []
        for c in o:
            # First recurse...
            subgroups = self.visit(c)
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        for t, group in groupby(o, type):
            # ... then add new groups
            g = tuple(group)
            if t is self.node_type and len(g) > 1:
                groups.append(g)
        return groups

    visit_list = visit_tuple


class PatternFinder(Visitor):
    """
    Find a pattern of nodes given as tuple/list of types within a given tree.

    Parameters
    ----------
    pattern : iterable of types
        The type pattern to look for.
    """

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    @classmethod
    def default_retval(cls):
        """
        Default return value is an empty list.

        Returns
        -------
        list
        """
        return []

    @staticmethod
    def match_indices(pattern, sequence):
        """ Return indices of matched patterns in sequence. """
        matches = []
        for i, elem in enumerate(sequence):
            if elem == pattern[0]:
                if tuple(sequence[i:i+len(pattern)]) == tuple(pattern):
                    matches.append(i)
        return matches

    def visit_tuple(self, o, **kwargs):
        """
        Visit all children and look for sequences of nodes with types matching
        the pattern.
        """
        matches = []
        for c in o:
            # First recurse...
            submatches = self.visit(c)
            if submatches is not None and len(submatches) > 0:
                matches += submatches
        types = list(map(type, o))
        idx = self.match_indices(self.pattern, types)
        for i in idx:
            matches.append(o[i:i+len(self.pattern)])
        return matches

    visit_list = visit_tuple
