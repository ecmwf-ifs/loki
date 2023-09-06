===================
Working with the IR
===================

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    release frequently by rebasing feature branches and upstreaming
    more general applicable work in the form of pull requests.

.. contents:: Contents
   :local:

The most important tool for working with
:doc:`Loki's internal representation <internal_representation>` are utilities
that traverse the IR to find specific nodes or patterns, to modify or
replace subtrees, or to annotate the tree. In Loki there exist two types of
tree traversal tools, depending on which level of the IR they operate on:

* *Visitors* that traverse the tree of control flow nodes;
* *Mappers* (following Pymbolic's :py:mod:`pymbolic.mapper` naming
  convention) that traverse expression trees.

Visitors
========

Loki's visitors work by inspecting the type of each IR node they encounter and
then selecting the best matching handler method for that node type. This allows
implementing visitors that perform tasks either for very specific node types
or generally applicable for any node type, depending on the handler's name.

Loki includes a range of ready-to-use and configurable visitors for many common
use cases, such as discovering certain node types, modifying or replacing
nodes in the tree, or creating a string representation of the tree.
For some use cases it may be easier to implement new visitors tailored to the
task.


Searching the tree
------------------

The first category of visitors traverses the IR and collects a list of results
subject to certain criteria. In almost all cases :any:`FindNodes` is the tool
for that job with some bespoke variants for specific use cases.

.. autosummary::

   loki.visitors.FindNodes
   loki.visitors.FindScopes
   loki.visitors.SequenceFinder
   loki.visitors.PatternFinder

A common pattern for using :any:`FindNodes` is the following:

.. code-block:: python

   for loop in FindNodes((Loop, WhileLoop)).visit(routine.body):
       # ...do something with loop...

There are additional visitors that search all expression trees embedded in the
control flow IR, which are explained further down.


Transforming the tree
---------------------

A core feature of Loki is the ability to transform the IR, which is done using
the :any:`Transformer`. It is a visitor that rebuilds the tree and replaces
nodes according to a mapper.

.. autosummary::

   loki.visitors.Transformer
   loki.visitors.NestedTransformer
   loki.visitors.MaskedTransformer
   loki.visitors.NestedMaskedTransformer

:any:`Transformer` is commonly used in conjunction with :any:`FindNodes`, with
the latter being used to build the mapper for the first. The following example
removes all loops over the horizontal dimension and replaces them by
their body. This code snippet is a simplified version of a transformation used
in :any:`ExtractSCATransformation`:

.. code-block:: python

   routine = Subroutine(...)
   horizontal = Dimension(...)

   ...

   loop_map = {}
   for loop in FindNodes(Loop).visit(routine.body):
       if loop.variable == horizontal.variable:
           loop_map[loop] = loop.body
   routine.body = Transformer(loop_map).visit(routine.body)


Converting the tree to string
-----------------------------

The last step in a transformation pipeline is usually to write the transformed
IR to a file. This is a task for :doc:`Loki's backends <backends>` which
themselves are subclasses of :class:`loki.visitors.pprint.Stringifier`, yet
another visitor. :class:`loki.visitors.pprint.Stringifier` doubles as a
pretty-printer for the IR that is useful for debugging.

.. autosummary::

   loki.visitors.pprint.Stringifier
   loki.visitors.pprint

Implementing new visitors
-------------------------

Any new visitor should subclass :any:`Visitor` (or any of its subclasses).

The common base class for all visitors is :any:`GenericVisitor`, declared in
:py:mod:`loki.visitors` that provides the basic functionality for matching
objects to their handler methods. Derived from that is :any:`Visitor` which
adds a default handler :data:`visit_Node` (for :any:`Node`) and functionality
to recurse for all items in a list or tuple and return the combined result.

To define handlers in new visitors, they should define :data:`visit_Foo`
methods for each class :data:`Foo` they want to handle.
If a specific method for a class :data:`Foo` is not found, the MRO
of the class is walked in order until a matching method is found (all the
way until, for example, :any:`Visitor.visit_Node` applies).
The method signature is:

.. code-block:: python

   def visit_Foo(self, o, [*args, **kwargs]):
       pass

The handler is responsible for visiting the children (if any) of
the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
used to pass information up and down the call stack.  You can also
pass named keyword arguments, e.g.:

.. code-block:: python

    def visit_Foo(self, o, parent=None, *args, **kwargs):
        pass

Mappers
=======

Mappers are visitors that traverse
:ref:`expression trees <internal_representation:Expression tree>`.

They are built upon :py:mod:`pymbolic.mapper` classes and for that reason use
a slightly different way of determining the handler methods: each expression
tree node (:class:`pymbolic.primitives.Expression`) holds a class
attribute :attr:`mapper_method` with the name of the relevant method.

Loki provides, similarly to control flow tree visitors, ready-to-use mappers
for searching or transforming expression trees, all of which are implemented
in :mod:`loki.expression.mappers`. In addition,
:mod:`loki.expression.expr_visitors` provides visitors that apply the same mapper
to all expression trees in the IR.


Searching in expression trees
-----------------------------

The equivalent to :any:`FindNodes` for expression trees is
:any:`ExpressionRetriever`. Using a generic function handle, (almost) arbitrary
conditions can be used as a query that decides whether to include a given node
into the list of results.

.. autosummary::

   loki.expression.mappers.ExpressionRetriever

Note that mappers operate only on expression trees, i.e. using them directly
is only useful when working with a single property of a control flow node,
such as :attr:`loki.ir.Assignment.rhs`. If one wanted to search for expression
nodes in all expression trees in the IR, e.g. to find all variables, bespoke
visitors exist that apply :any:`ExpressionRetriever` to all expression trees.

.. autosummary::

   loki.expression.expr_visitors.ExpressionFinder
   loki.expression.expr_visitors.FindExpressions
   loki.expression.expr_visitors.FindTypedSymbols
   loki.expression.expr_visitors.FindVariables
   loki.expression.expr_visitors.FindInlineCalls
   loki.expression.expr_visitors.FindLiterals

For example, the following finds all function calls embedded in expressions
(:any:`InlineCall`, as opposed to subroutine calls in :any:`CallStatement`):

.. code-block:: python

   for call in FindInlineCalls().visit(routine.body):
       # ...do something with call...


Transforming expression trees
-----------------------------

Transformations of the expression tree are done very similar to
:any:`Transformer`, using the mapper :any:`SubstituteExpressionsMapper` that
is given a map to replace matching expression nodes.

.. autosummary::

   loki.expression.mappers.LokiIdentityMapper
   loki.expression.mappers.SubstituteExpressionsMapper

In the same way that searching can be done on all expression trees in the IR,
transformations can be applied to all expression trees at the same time using
:any:`SubstituteExpressions`:

.. autosummary::

   loki.expression.expr_visitors.SubstituteExpressions

The following example shows how expression node discovery and substitution can
be combined to replace all occurences of intrinsic function calls.
(The code snippet is taken from :any:`replace_intrinsics`, where two `dict`,
:data:`function_map` and :data:`symbol_map`, provide the mapping to rename
certain function calls that appear in :data:`routine`.)

.. code-block:: python

   from loki.expression import symbols as sym

   callmap = {}
   for c in FindInlineCalls(unique=False).visit(routine.body):
       cname = c.name.lower()

       if cname in symbol_map:
           callmap[c] = sym.Variable(name=symbol_map[cname], scope=routine.scope)

       if cname in function_map:
           fct_symbol = sym.ProcedureSymbol(function_map[cname], scope=routine.scope)
           callmap[c] = sym.InlineCall(fct_symbol, parameters=c.parameters,
                                       kw_parameters=c.kw_parameters)

   routine.body = SubstituteExpressions(callmap).visit(routine.body)


Converting expressions to string
--------------------------------

Every backend has their own mapper to convert expressions to a source
code string, according to the corresponding language specification.
All build on a common base class :any:`LokiStringifyMapper`, which is
also called automatically when converting any expression node to string.

.. autosummary::

   loki.expression.mappers.LokiStringifyMapper
