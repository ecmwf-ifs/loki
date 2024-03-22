.. _transformations:

========================
Transformation pipelines
========================

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    release frequently by rebasing feature branches and upstreaming
    more general applicable work in the form of pull requests.

.. contents:: Contents
   :local:


Transformations
===============

Transformations are the building blocks of a transformation pipeline in Loki.
They encode the workflow of converting a :any:`Sourcefile` or an individual
program unit (such as :any:`Module` or :any:`Subroutine`) to the desired
output format.

A transformation can encode a single modification, combine multiple steps,
or call other transformations to create complex changes. If a transformation
depends on another transformation, inheritance can be used to combine them.

Every transformation in a pipeline should implement the interface defined by
:any:`Transformation`. It provides generic entry points for transforming
different objects and thus allows for batch processing. To implement a new
transformation, only one or all of the relevant methods
:any:`Transformation.transform_subroutine`,
:any:`Transformation.transform_module`, or :any:`Transformation.transform_file`
need to be implemented.

*Example*: A transformation that inserts a comment at the beginning of every
module and subroutine:

.. code-block:: python

    class InsertCommentTransformation(Transformation):

        def _insert_comment(self, program_unit):
            program_unit.spec.prepend(Comment(text='! Processed by Loki'))

        def transform_subroutine(self, routine, **kwargs):
            self._insert_comment(routine)

        def transform_module(self, module, **kwargs):
            self._insert_comment(module)

The transformation can be applied by calling :meth:`apply` with
the relevant object.

.. code-block:: python

   source = Sourcefile(...)  # may contain modules and subroutines
   transformation = InsertCommentTransformation()
   for module in source.modules:
       transformation.apply(module)
   for routine in source.all_subroutines:
       transformation.apply(routine)

Note that we have to apply the transformation separately for every
relevant :any:`ProgramUnit`. The transformation can also be modified
such that it is automatically applied to all program units in a file,
despite only implementing logic for transforming modules and subroutines:

.. code-block:: python

    class InsertCommentTransformation(Transformation):

        # When called on a Sourcefile, automatically apply this to all modules
        # in the file
        recurse_to_modules = True

        # When called on a Sourcefile or Module, automatically apply this to all
        # Subroutines in the file or module
        recurse_to_procedures = True

        def _insert_comment(self, program_unit):
            program_unit.spec.prepend(Comment(text='! Processed by Loki'))

        def transform_subroutine(self, routine, **kwargs):
            self._insert_comment(routine)

        def transform_module(self, module, **kwargs):
            self._insert_comment(module)

With these two attributes added, we can now apply the transformation to all
modules and procedures in a single command:

.. code-block:: python

   source = Sourcefile(...)  # may contain modules and subroutines
   transformation = InsertCommentTransformation()
   transformation.apply(source)

Most transformations, however, will only require modifying those parts of a file
that are part of the call tree that is to be transformed to avoid unexpected
side-effects.

Typically, transformations should be implemented by users to encode the
transformation pipeline for their individual use-case. However, Loki comes
with a growing number of built-in transformations that are implemented in
the :mod:`loki.transform` namespace:

.. autosummary::

   loki.transform

This includes also a number of tools for common transformation tasks that
are provided as functions that can be readily used when implementing new
transformations.

Batch processing large source trees
===================================

Transformations can be applied over source trees using the :any:`Scheduler`.
It is a work queue manager that automatically discovers source files in a list
of paths and builds a dependency graph from a given starting point.
This dependency graph includes all called procedures and imported modules.

Calling :any:`Scheduler.process` on a source tree and providing it with a
:any:`Transformation` applies this transformation to all files, modules, or
routines that appear in the dependency graph. The exact traversal
behaviour can be parameterized in the implementation of the :any:`Transformation`.
The behaviour modifications include:

* limiting the processing only to specific node types in the dependency graph
* reversing the traversal direction, i.e., called routines or imported
  modules are processed before their caller, such that the starting point/root
  of the dependency is processed last
* traversing the file graph, i.e., processing full source files rather than
  individual routines or modules
* automatic recursion into contained program units, e.g., processing also all
  procedures in a module after the module has been processed

When applying the transformation to an item in the source tree, the scheduler
provides certain information about the item to the transformation:

* the transformation mode (provided in the scheduler's config),
* the item's role (e.g., ``'driver'`` or ``'kernel'``, configurable via the
  scheduler's config), and
* targets (dependencies that are depended on by the currently processed item,
  and are included in the scheduler's tree, i.e., are processed, too).

.. note::
   The scheduler's dependency graph will include all dependency types it discovers.
   This includes not only control-flow dependencies via procedure calls, but also
   dependencies on other modules via the import of global variables, or dependencies
   on derived type definitions.

   However, for backwards-compatibility with the original scheduler implementation,
   only control-flow dependencies are followed and processed by default, and reported
   as ``items`` in :any:`Scheduler.items`. To remove this limitation, which is required
   e.g., for the :any:`GlobalVarOffloadTransformation`, the ``enable_imports`` option
   can be set to ``True``. This can be done in the ``[default]`` block of the config,
   or as a constructor argument in the :any:`Scheduler`.

The Scheduler's dependency graph
--------------------------------

The :any:`Scheduler` builds a dependency graph consisting of :any:`Item`
instances as nodes. Every item corresponds to a specific node in Loki's
internal representation.

The name of an item refers to a symbol using a fully-qualified name in the
format: ``<scope_name>#<local_name>``. The ``<scope_name>`` corresponds to
a Fortran module, in which a subroutine, interface or derived type is
declared. That declaration's name (e.g., the name of the subroutine)
constitutes the ``<local_name>`` part. For subroutines that are not embedded
into a module, the ``<scope_name>`` is empty, i.e., the item's name starts with
a dash (``#``).

In most cases these IR nodes are scopes and the entry points for transformations:

* :any:`FileItem` corresponds to :any:`Sourcefile`
* :any:`ModuleItem` corresponds to :any:`Module`
* :any:`ProcedureItem` corresponds to :any:`Subroutine`

The remaining cases are items corresponding to IR nodes that constitute some
form of intermediate dependency, which are required to resolve the indirection
to the scope node:

* :any:`InterfaceItem` corresponding to :any:`Interface`, i.e., providing a
  callable target that resolves to one or multiple procedures that are defined
  in the interface.
* :any:`ProcedureBindingItem` corresponding to the :any:`ProcedureSymbol`
  that is declared in a :any:`Declaration` in a derived type. Similarly to
  interfaces, these resolve to one or multiple procedures that are defined in
  the procedure binding inside the derived type.
* :any:`TypeDefItem` corresponding to :any:`TypeDef`, which does not introduce
  a control flow dependency but is crucial to capture as a dependency to enable
  annotating type information for inter-procedural analysis.

Finally, :any:`ExternalItem` denotes items that the scheduler was unable to discover.
The expected item type of the missing item is stored in :any:`ExternalItem.origin_cls`.
When batch processing a transformation, the external items are ignored, unless the
config option ``strict=True`` is enabled. In that case, an error will be issued when
an external item is encountered that matches the ``item_filter`` that is provided by
the transformation's manifest (in :any:`Transformation.item_filter`).

To facilitate the creation of the dependency tree, every :any:`Item`
provides two key properties:

* :any:`Item.definitions`: A list of all IR nodes that constitute symbols/names
  that are made available by an item. For a :any:`FileItem`, this typically consists
  of all modules and procedures in that sourcefile, and for a :any:`ModuleItem` it
  comprises of procedures, interfaces, global variables and derived type definitions.
* :any:`Item.dependencies`: A list of all IR nodes that introduce a dependency
  on other items, e.g., :any:`CallStatement` or :any:`Import`.

This information is used to populate the scheduler's dependency graph, which is
constructed by the :any:`SGraph` class. Importantly, to improve processing speed
and limit parsing to the minimum of required files, this relies on incremental
parsing using the :any:`REGEX` frontend. Starting with only the top-level program
units in every discovered source file and a specified seed, the dependencies of each
item are used to determine the next set of items, which are generated on-demand
from the enclosing scope via partial re-parses. This may incur incremental parsing
with additional :any:`RegexParserClass` enabled to discover definitions or dependencies
as required. Only once the full dependency graph has been generated, a full parse
of the source files in the graph is performed, providing the complete internal
representation and automatically enriching type information with inter-procedural annotations.

Pruning the dependency graph
----------------------------

If the intention is not to process some items it is recommended to not
leave them dangling as :any:`ExternalItem`. Instead, they should be explicitly
excluded from the dependency graph and the ``strict`` mode enabled.
To exclude specific items, any of the following annotations can be used, resulting in
different behaviour:

* ``disable``: Dependency items matching an entry in this list are treated as if they
  don't exist, and their definitions are not searched for or parsed. This is useful, e.g.,
  to exclude frequently used utility routines or modules (such as the
  `yomhook module in IFS <https://github.com/ecmwf-ifs/fiat/blob/main/src/fiat/drhook/yomhook.F90>`_),
  which are not to be transformed.
* ``block``: Dependency items matching an entry in this list are not parsed or added to
  the dependency graph, and therefore excluded from transformations. They are, however,
  included for reference in the dependency graph visualization produced by
  :any:`Scheduler.callgraph`.
* ``ignore``: Dependency items matching an entry in this list are parsed and added to the
  dependency graph. This makes their definitions available for enrichment but they are
  not processed *by default*. Transformations can include them during batch processing
  by enabling the :any:`Transformation.process_ignored_items` option. A typical use case
  for this are dependencies that are part of a separate compilation target (and therefore
  transformed separately), but analysis passes may need to collect information across an
  entire call tree (e.g., use of temporary arrays).

These three lists can be supplied globally in the ``[default]`` section of the scheduler
config file, or per routine. The matching of items against entries in these lists is
supports basic patterns (via :any:`fnmatch`), and is also effective for entire scopes.
For example, a subroutine ``my_routine`` that is defined in a module ``my_mod`` would be
matched by any of the following:

* ``my_routine``
* ``my_mod``
* ``my_mod#my_routine``
* ``*_routine``

By default, all items are expanded during dependency discovery, i.e., for every item
all dependencies are added to the graph, and then dependencies of these dependencies are
added as well. This procedure continues until all dependencies have been included.
For individual items, this expansion can be disabled by setting ``expand=False`` for
them in the scheduler config.


Filtering graph traversals
--------------------------

Often, only specific item types are of interest when traversing the dependency graph.
For that purpose, the :any:`SFilter` class provides an iterator for an :any:`SGraph`,
which allows specifying an ``item_filter`` or reversing the direction of traversals.
Other traversal modes may be added in the future.



.. autosummary::

   loki.bulk.scheduler.Scheduler
   loki.bulk.scheduler.SGraph
   loki.bulk.scheduler.SFilter
   loki.bulk.configure.SchedulerConfig
   loki.bulk.configure.TransformationConfig
   loki.bulk.configure.ItemConfig
   loki.bulk.item.Item
   loki.bulk.item.FileItem
   loki.bulk.item.ModuleItem
   loki.bulk.item.ProcedureItem
   loki.bulk.item.TypeDefItem
   loki.bulk.item.ProcedureBindingItem
   loki.bulk.item.InterfaceItem
   loki.bulk.item.ItemFactory
