.. _internal-representation:

===================================
Loki's internal representation (IR)
===================================

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    `master` frequently by rebasing feature branches and upstreaming
    more general applicable work in the form of pull requests.

Loki's internal representation aims to achieve a balance between usability
and general applicability. This means that in places there may be shortcuts
taken to ease its use in the context of a source-to-source translation
utility but may break with established practices in compiler theory.
The IR was developed with Fortran source code in mind and that shows. Where
there exist similar concepts in other languages, things are transferable.
In other places, Fortran-specific annotations are included for the sole purpose
of enabling string reproducibility.

The internal representation is vertically divided into different layers,
roughly aligned with high level concepts found in Fortran and other
programming languages:

.. contents::
   :local:
   :depth: 1


Container data structures
=========================

Outermost are container data structures that conceptually translate to
Fortran's `program-units`, such as modules and subprograms.

Fortran modules are represented by :any:`Module` objects which comprise
a specification part (:py:attr:`Module.spec`) and a list of :any:`Subroutine`
objects contained in the module.

Subroutines and functions are represented by :any:`Subroutine` objects that
in turn have their own docstring (:py:attr:`Subroutine.docstring`),
specification part (:py:attr:`Subroutine.spec`), execution part
(:py:attr:`Subroutine.body`), and contained subprograms
(:py:attr:`Subroutine.members`).

To map these programming language concepts to source files and ease input or
output operations, any number of these container data structures can be
classes.

Available container classes
---------------------------

.. autosummary::

   loki.sourcefile.Sourcefile
   loki.module.Module
   loki.subroutine.Subroutine


.. _control-flow-ir:

Control flow tree
=================

Specification and execution parts of (sub)programs and modules are the central
components of container data structures. Each of them is represented by a tree
of control flow nodes, with a :any:`Section` as root node. This tree resembles
to some extend a hierarchical control flow graph where each node can have
control flow and expression nodes as children. Consequently, this separation on
node level is reflected in the internal representation, splitting the tree into
two levels:

1. `Control flow` (e.g., loops, conditionals, assignments, etc.);
   the corresponding classes are declared in :py:mod:`loki.ir` and described
   in this section.
2. `Expressions` (e.g., scalar/array variables, literals, operators, etc.);
   this is based on `Pymbolic <https://github.com/inducer/pymbolic>`__ with
   encapsulating classes declared in :py:mod:`loki.expression.symbols` and
   described below.

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

As an example, consider a basic Fortran ``DO i=1,n`` loop: it defines a loop
variable (``i``), a loop range (``1:n``) and a loop body. The body can be
one/multiple statements or other control flow structures and therefore is a
subtree of control flow nodes. Loop variable and range, however, are
expression nodes.

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


Available control flow nodes
----------------------------

Abstract base classes
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   loki.ir.Node
   loki.ir.InternalNode
   loki.ir.LeafNode

Internal node classes
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   loki.ir.Section
   loki.ir.Associate
   loki.ir.Loop
   loki.ir.WhileLoop
   loki.ir.Conditional
   loki.ir.MaskedStatement
   loki.ir.PragmaRegion
   loki.ir.Interface

Leaf node classes
^^^^^^^^^^^^^^^^^

.. autosummary::

   loki.ir.Assignment
   loki.ir.ConditionalAssignment
   loki.ir.CallStatement
   loki.ir.CallContext
   loki.ir.Allocation
   loki.ir.Deallocation
   loki.ir.Nullify
   loki.ir.Comment
   loki.ir.CommentBlock
   loki.ir.Pragma
   loki.ir.PreprocessorDirective
   loki.ir.Import
   loki.ir.Declaration
   loki.ir.DataDeclaration
   loki.ir.TypeDef
   loki.ir.MultiConditional
   loki.ir.Intrinsic


.. _expression-ir:

Expression tree
===============

Many control flow nodes contain one or multiple expressions, such as the
right-hand side of an assignment (:py:attr:`loki.ir.Assignment.rhs`) or the
condition of an ``IF`` statement (:py:attr:`loki.ir.Conditional.condition`).
Such expressions are represented by expression trees, comprising a single
node (e.g., the left-hand side of an assignment may be just a scalar variable)
or a large expression tree consisting of multiple nested sub-expressions.

Loki's expression representation is based on
`Pymbolic <https://github.com/inducer/pymbolic>`__ but encapsulates all
classes with bespoke own implementations. This allows to enrich expression
nodes by attaching custom metadata, implementing bespoke comparison operators,
or store type information.

The base class for all expression nodes is :any:`pymbolic.primitives.Expression`.

Available expression tree nodes
-------------------------------

Typed symbol nodes
^^^^^^^^^^^^^^^^^^

.. autosummary::

   loki.expression.symbols.TypedSymbol
   loki.expression.symbols.Variable
   loki.expression.symbols.Scalar
   loki.expression.symbols.Array
   loki.expression.symbols.ProcedureSymbol

Literals
^^^^^^^^

.. autosummary::

   loki.expression.symbols.Literal
   loki.expression.symbols.FloatLiteral
   loki.expression.symbols.IntLiteral
   loki.expression.symbols.LogicLiteral
   loki.expression.symbols.StringLiteral
   loki.expression.symbols.IntrinsicLiteral
   loki.expression.symbols.LiteralList

Mix-ins
^^^^^^^

.. autosummary::

   loki.expression.symbols.ExprMetadataMixin
   loki.expression.symbols.StrCompareMixin
