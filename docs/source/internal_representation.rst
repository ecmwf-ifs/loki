.. _internal-representation:

===================================
Loki's internal representation (IR)
===================================

.. todo:: Talk about the container data structures :any:`Sourcefile`,
          :any:`Module` and :any:`Subroutine`.

The overall structure of a program is represented by a tree of control flow
nodes, which themselves can have control flow and expression nodes as children.
This separation is reflected in the internal representation, which is a tree
that is split into two levels:

1. Control flow (e.g., loops, conditionals, assignments, etc.);
   the corresponding classes are declared in :py:mod:`loki.ir` and described
   in this document.
2. Expressions (e.g., scalar/array variables, literals, operators, etc.);
   the structure of this layer is described in
   :py:mod:`loki.expression.symbols`.

All control flow nodes implement the common base class :any:`Node` and
can have an arbitrary number of children that are either control flow nodes
or expression nodes. Thus, any control flow node looks in principle like the
following:

.. code-block:: none

                      Node
                      / | \
              +------+  |  +---+
             /          |       \
            /           |        \
      Expression   Expression   Node   ...

As an example, consider a basic Fortran ``DO`` loop: it defines a loop
variable, a loop range and a loop body. The body can be one/multiple
statements or other control flow structures and therefore is a subtree of
control flow nodes. Loop variable and range, however, are expressions.

All control flow nodes fall into one of two categories:

* :any:`InternalNode`: nodes that have a :py:attr:`body` and therefore
  have other control flow nodes as children.
* :any:`LeafNode`: nodes that (generally) do not have any other
  control flow nodes as children.

Note that :any:`InternalNode` can have other properties than
:py:attr:`body` in which control flow nodes are contained as children
(for example, :py:attr:`else_body` in :any:`Conditional`).
All :any:`Node` may, however, have one or multiple expression trees
as children.

.. note:: All actual control flow nodes are implementations of one of the two
          base classes. Two notable exceptions to the above are the following:

          * :any:`MultiConditional` (for example, Fortran's ``SELECT CASE``):
            It has multiple bodies and thus does not fit the above framework.
            Conceptually, these could be converted into nested
            :any:`Conditional` but it would break string reproducibility.
            For that reason they are retained as a :any:`LeafNode` for the
            time being.
          * :any:`TypeDef`: This defines a new scope for symbols, which
            does not include symbols from the enclosing scope. Thus, it behaves
            like a leaf node although it has technically control flow nodes as
            children. It is therefore also implemented as a :any:`LeafNode`.

With this separation into two types of nodes, the schematics of the control flow
layer of the internal representation are as follows:

.. code-block:: none

                        InternalNode
                             |
                            body
                           /|||\
          +---------------+ /|\ +-------------+
         /          +------+ | +-----+         \
        /          /         |        \         \
    LeafNode InternalNode LeafNode LeafNode InternalNode ...
                  |                              |
                 body                           body
                /    \                         /    \
               /      \                         ....
         LeafNode  InternalNode
                        |
                       ...

.. todo:: Elaborate on how to traverse the internal representation.

What follows is a description of the base classes and all implementations of
those:

.. contents::
   :local:

Abstract base classes
=====================

.. autoclass:: loki.ir.Node

.. autoclass:: loki.ir.InternalNode

.. autoclass:: loki.ir.LeafNode

Internal node classes
=====================

.. autoclass:: loki.ir.Section

.. autoclass:: loki.ir.Associate

.. autoclass:: loki.ir.Loop

.. autoclass:: loki.ir.WhileLoop

.. autoclass:: loki.ir.Conditional

.. autoclass:: loki.ir.MaskedStatement

.. autoclass:: loki.ir.PragmaRegion

.. autoclass:: loki.ir.Interface

Leaf node classes
=================

.. autoclass:: loki.ir.Assignment

.. autoclass:: loki.ir.ConditionalAssignment

.. autoclass:: loki.ir.CallStatement

.. autoclass:: loki.ir.CallContext

.. autoclass:: loki.ir.Allocation

.. autoclass:: loki.ir.Deallocation

.. autoclass:: loki.ir.Nullify

.. autoclass:: loki.ir.Comment

.. autoclass:: loki.ir.CommentBlock

.. autoclass:: loki.ir.Pragma

.. autoclass:: loki.ir.PreprocessorDirective

.. autoclass:: loki.ir.Import

.. autoclass:: loki.ir.Declaration

.. autoclass:: loki.ir.DataDeclaration

.. autoclass:: loki.ir.TypeDef

.. autoclass:: loki.ir.MultiConditional

.. autoclass:: loki.ir.Intrinsic
