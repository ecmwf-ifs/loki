# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
LoopTree data structure for analysing loop structure and utility methods for
loop analysis.
"""

from collections import deque
from loki.ir import nodes as ir, Visitor


__all__ = [
    'LoopTree', 'get_loop_tree'
]


class LoopTree:
    """
    A data structure that stores nested loops in a forest of trees.

    Parameters
    ----------
    roots : list of :any:`LoopTree.TreeNode`
        top level loops in the ir section of the LoopTree
    loop_map : dict of (:any:`ir.Loop`, :any:`TreeNode`)
        dictionary with key-value pairs of :any:`ir.Loop` nodes in the tree
        and corresponding :any:`TreeNode` nodes
    """

    class TreeNode:
        """
        Internal node class for nodes in the LoopTree

        Parameters
        ----------
        loop : :any:`ir.Loop`
            :any:`ir.Loop` of the tree node
        parent : :any:`LoopTree.Treenode` or None
            parent node in the LoopTree
        parent : list of :any:`LoopTree.Treenode`
            child nodes in the LoopTree
        depth : int
            nesting depth of the node (0 if root)
        """

        def __init__(self, loop: ir.Loop, parent=None):
            self.loop = loop
            self.parent = parent
            self.children = []
            self.depth = 0 if parent is None else parent.depth+1

            if parent:
                parent.children.append(self)

    def __init__(self):
        self.roots = []
        self.loop_map = {}

    def add_node(self, loop: ir.Loop, parent: TreeNode = None):
        """
        Helper function to add a loop to the LoopTree
        """
        tree_node = self.TreeNode(loop, parent)
        if parent is None:
            self.roots.append(tree_node)
        self.loop_map[loop] = tree_node
        return tree_node

    def get_tree_node(self, loop: ir.Loop):
        """
        Get LoopTree node corresponding to Loki IR Loop node.
        """
        return self.loop_map.get(loop)

    def walk_depth_first(self, pre_order=True):
        """
        Generator for depth first traversal of the loop tree.

        Parameters
        ----------
        pre_order : `bool`
            Order of depth first traversal, if `True` parent nodes are visited before children.
            default: True
        """
        def visit(tree_node):
            if pre_order:
                yield tree_node

            for child in tree_node.children:
                yield from visit(child)

            if not pre_order:
                yield tree_node

        for root in self.roots:
            yield from visit(root)

    def walk_breadth_first(self):
        """
        Generator for breadth first traversal of the loop tree.
        """
        queue = deque(self.roots)
        while queue:
            tree_node = queue.popleft()
            yield tree_node
            queue.extend(tree_node.children)


def get_loop_tree(region: ir.Node):
    """
    Function that construct a loop tree with all loops in an IR region
    """
    class LoopTreeBuilder(Visitor):
        def __init__(self):
            super().__init__()
            self.loop_tree = LoopTree()

        def visit_Loop(self, loop, **kwargs):
            parent = kwargs.pop('parent', None)
            tree_node = self.loop_tree.add_node(loop, parent)

            for o in loop.children:
                self.visit(o, parent=tree_node, **kwargs)

    tree_builder = LoopTreeBuilder()
    tree_builder.visit(region)
    return tree_builder.loop_tree
