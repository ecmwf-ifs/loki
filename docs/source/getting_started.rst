===============
Getting started
===============

.. toctree::
   :hidden:

   INSTALL
   notebooks


Core concepts (the philosophical bit)
=====================================

On a fundamental level, converting between different programming
styles in a low-level compiled language like Fortran or C/C++ typically
requires assumptions to be made that are specific to the data and algorithm
and do not generalize to the entire language. This is why Loki provides a
programmable interface rather than a push-button solution, leaving it
up to developers to decide which assumptions about the original source
code can be used and how.

For example, converting large numbers of IFS physics code to a "single column"
format (see below) requires the explicit knowledge of which index variables
typically represent the parallel dependency-free horizontal dimension that
is to be lifted.

The aim of Loki is therefore to give developers all the tools to encode their
own code transformation in an elegant, pythonic fashion. The core concepts
provided for this are:

* :any:`Module` and :any:`Subroutine` classes (kernels) that each provide an
  :doc:`Intermediate Representation (IR) <internal_representation>`
  of their source code, as well as
  utilities to inspect and transform the underlying IR nodes.
* Expressions contained in IR nodes, such as :any:`Statement`, :any:`Loop`,
  and :any:`Conditional`, are represented as independent sub-trees, based on the
  `Pymbolic <https://github.com/inducer/pymbolic>`__ infrastructure.
* Three frontends are supported that are used to parse Fortran code
  either from source files or strings into the Loki IR trees. Multiple
  backends are provided to generate Fortran or (experimentally) C or (even more
  experimentally) Python code from the combined IR and expression trees.
* A :any:`Transformation` class is provided that allows users to encode
  individual code changes based on the abstract representation
  provided by Loki's IR and expression objects and can be applied
  to individual :any:`Subroutine` and :any:`Module` objects - much like simple
  compiler passes.
* A :any:`Scheduler` class that provides bulk processing
  and inter-procedural analysis (IPA) tools to apply individual changes
  over large numbers of files while honoring the call-tree that
  connects them.

Example transformations and current features
============================================

Loki is primarily an API and toolbox, requiring developers to create their
own head scripts to create and invoke source-to-source translation toolchains.
A small set of transformations considered generic enough are provided by the
package itself in :mod:`loki.transform`. The majority of more complex transformations
are collected in a separate Python package that lives under ``transformations``.

The ``loki_transform.py`` script is provided by the Loki install. The primary
transformation passes provided by these example transformations are:

* **Idempotence (Idem)** - A simple transformation that performs a
    neutral parse-unparse cycle on a kernel.
* **Single column abstraction (SCA)** - Transforms a set of kernels
  into Single column format by removing the specified horizontal
  iteration dimension. This transformation has a "driver" and a
  "kernel" mode, as it potentially changes the subroutine's call
  signature to remove derived types (structs do not expose
  dimensions).
* **Single column coalesced (SCC)** - Transforms a set of kernels
  from CPU-style (SIMD) vectorization format to a GPU-style (SIMT)
  loop layout. It removes the specified horizontal iteration
  dimension and re-inserts it outermost. Optionally, the horizontal
  loop can be stripped from kernels and re-inserted in the driver, to
  allow hoisting the allocation of temporary arrays to driver level (SCCH).
* **C transpilation** - A dedicated Fortran-to-C transpilation
  pipeline that converts Fortran source code into (column major,
  1-indexed) C kernel code. The transformation pipeline also creates
  the necessary header and `ISOC` wrappers to integrate this C kernel
  with a Fortran driver layer, as demonstrated with the
  `CLOUDSC ESCAPE dwarf <https://github.com/ecmwf-ifs/dwarf-p-cloudsc>`_.

First steps
===========

To start using Loki, follow the :doc:`installation instructions <INSTALL>`.
We recommend to study the :doc:`Jupyter notebooks <notebooks>` in the `example`
directory to get familiar with the basic API of Loki. The
:doc:`Using Loki <using_loki>` section provides more details on the inner
workings and underpinning concepts.

Contributions
=============

Contributions to Loki are welcome. In order to do so, please open an
issue in the `Github repository <https://github.com/ecmwf-ifs/loki/issues>`__
where a feature request or bug can be discussed.
Then create a pull request with your contribution. We require you to read and sign the
`contributors license agreement (CLA) <http://claassistant.ecmwf.int/ecmwf-ifs/loki>`__
before your contribution can be reviewed and merged.
