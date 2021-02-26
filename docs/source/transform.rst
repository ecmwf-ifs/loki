.. _transformations:

========================
Transformation pipelines
========================

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    `master` frequently by rebasing feature branches and upstreaming
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

*Example*: A transformation that renames every module and subroutine by
appending a given suffix:

.. code-block:: python

    class RenameTransformation(Transformation):

        def __init__(self, suffix):
            super().__init__()
            self._suffix = suffix

        def _rename(self, item):
            item.name += self._suffix

        def transform_subroutine(self, routine, **kwargs):
            self._rename(routine)

        def transform_module(self, module, **kwargs):
            self._rename(module)

The transformation can be applied by calling :meth:`apply` with
the relevant object.

.. code-block:: python

   source = Sourcefile(...)  # may contain modules and subroutines
   transformation = RenameTransformation(suffix='_example')
   transformation.apply(source)

Note that, despite only implementing logic for transforming modules and
subroutines, it works also for sourcefiles. While the :any:`Sourcefile`
object itself is not modified (because we did not implement
:meth:`transform_file`), the dispatch mechanism in the :any:`Transformation`
base class takes care of calling the relevant method for each member of the
given object.

Typically, transformations should be implemented by users to encode the
transformation pipeline for their individual use-case. However, Loki comes
with a few built-in transformations for common tasks and we expect this list
to grow in the future:

.. autosummary::

   loki.transform.transformation.Transformation
   loki.transform.dependency_transform.DependencyTransformation
   loki.transform.fortran_c_transform.FortranCTransformation
   loki.transform.fortran_max_transform.FortranMaxTransformation
   loki.transform.fortran_python_transform.FortranPythonTransformation

Further transformations are defined for specific use-cases but may prove
useful in a wider context. These are defined in :mod:`scripts.transformations`:

.. autosummary::

   scripts.transformations.argument_shape.InferArgShapeTransformation
   scripts.transformations.data_offload.DataOffloadTransformation
   scripts.transformations.derived_types.DerivedTypeArgumentsTransformation
   scripts.transformations.single_column_claw.ExtractSCATransformation
   scripts.transformations.single_column_claw.CLAWTransformation

Additionally, a number of tools for common transformation tasks are provided as
functions that can be readily used in a step of the transformation pipeline:

.. autosummary::

   loki.transform.transform_inline.inline_constant_parameters
   loki.transform.transform_inline.inline_elemental_functions
   loki.transform.transform_loop.loop_interchange
   loki.transform.transform_loop.loop_fusion
   loki.transform.transform_loop.loop_fission
   loki.transform.transform_region.region_hoist
   loki.transform.transform_region.region_to_call


Bulk processing large source trees
==================================

Transformations can be applied over source trees using the :any:`Scheduler`.
It is a work queue manager that automatically discovers source files in a list
of paths. Given the name of an entry routine, it allows to build a call graph
and thus derive the dependencies within this source tree.

Calling :any:`Scheduler.process` on a source tree and providing it with a
:any:`Transformation` applies this transformation to all modules and routines,
making sure that routines with the relevant :any:`CallStatement` are always
processed before their target :class:`Subroutine`.

When applying the transformation to an item in the source tree, the scheduler
provides certain information about the item to the transformation:

* the transformation mode (provided in the scheduler's config),
* the item's role (e.g., `'driver'` or `'kernel'`, configurable via the
  scheduler's config), and
* targets (routines that are called from the item and are included in the
  scheduler's tree, i.e., will be processed afterwards).

.. autosummary::

   loki.scheduler.Scheduler
   loki.scheduler.SchedulerConfig
   loki.scheduler.Item
