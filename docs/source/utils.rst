=========
Utilities
=========

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    release frequently by rebasing feature branches and upstreaming
    more general applicable work in the form of pull requests.

.. contents:: Contents
   :local:


To assist the development of custom transformations, a number of useful tools
for recurring tasks or house keeping are included with Loki.


Pragma utilities
================

An easy way of injecting information at specific locations in source files
is to insert pragmas. This allows to annotate declarations or loops, mark
source code regions or specify locations. Pragmas are represented by a unique
node type :any:`Pragma` and thus can be picked out easily during a
transformation.

A number of utility routines and context manager are available that allow
for easy parsing of pragmas, can attach pragmas to other nodes (such as
:any:`Loop`), or extract pragma regions and wrap them in a dedicated internal
node type :any:`PragmaRegion`:

.. autosummary::

   loki.pragma_utils.is_loki_pragma
   loki.pragma_utils.get_pragma_parameters
   loki.pragma_utils.pragmas_attached
   loki.pragma_utils.pragma_regions_attached


Dataflow analysis
=================

Rudimentary dataflow analysis utilities are included with Loki that determine
for each IR node what symbols it defines (i.e., assigns a value), reads
(i.e., uses before defining it), and which symbols are live (i.e., have been
defined before) entering the IR node in the control flow.

.. autosummary::

   loki.analyse.analyse_dataflow.dataflow_analysis_attached


Dimensions
==========

With the modification of data layouts and iteration spaces as one of the core
tasks in many transformation pipelines in mind, Loki has a :any:`Dimension`
class to define such a one-dimensional space.

.. autosummary::

   loki.dimension.Dimension


Python utilities
================

Some convenience utility routines, e.g., to simplify working with strings
or files are included in :mod:`loki.tools`:

.. autosummary::

   loki.tools.files
   loki.tools.strings
   loki.tools.util

A notable example is :any:`CaseInsensitiveDict`, a `dict` with strings as keys
for which the case is ignored. It is repeatedly used in other Loki data
structures when mapping symbol names that stem from Fortran source code. Since
Fortran is not case-sensitive, these names can potentially appear with mixed
case yet all refer to the same symbol and :any:`CaseInsensitiveDict` makes sure
no problems arise due to that.

Other frequently used utilities for working with lists and tuples are
:any:`as_tuple`, :any:`is_iterable` and :any:`flatten`.


Loki house keeping
==================

For internal purposes exist a global configuration
:class:`loki.config.Configuration` and logging functionality.

.. autosummary::

   loki.config
   loki.logging


Build subpackage
================

As part of Loki's test suite but also useful as a standalone package are the
build utilities :mod:`loki.build`:

.. autosummary::

   loki.build.binary.Binary
   loki.build.header.Header
   loki.build.lib.Lib
   loki.build.obj.Obj
   loki.build.builder.Builder
   loki.build.compiler
   loki.build.max_compiler
   loki.build.workqueue


Linting functionality
=====================

The source analysis capabilities of Loki can be used to build a static source
code analysis tool for Fortran. This is being developed as a standalone script
:doc:`loki-lint <loki_lint>` and includes a few data structures for the linter
mechanics in :mod:`loki.lint`:

.. autosummary::

   loki.lint.linter.Linter
   loki.lint.reporter
   loki.lint.rules.GenericRule
   loki.lint.utils.Fixer
