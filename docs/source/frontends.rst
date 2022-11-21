===========================
Reading Fortran source code
===========================

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    release frequently by rebasing feature branches and upstreaming
    more general applicable work in the form of pull requests.

.. contents:: Contents
   :local:


The first step in a transformation pipeline is reading Fortran source code
and converting it to :doc:`internal_representation`.


Parsing a file or string
========================

Typically, one has a source file that contains modules, functions and/or
subroutines. In Loki, this will be represented by a :any:`Sourcefile` that
stores the individual program units.
Reading source code from file is done via :any:`Sourcefile.from_file` and
requires only specifying the path to that file:

.. code-block:: python

   source = Sourcefile.from_file('/path/to/source/file.f90')

Optionally, a number of parameters can or sometimes should be supplied:

* If symbols from other modules are imported in the source file
  and type or procedure information is required (e.g., to inline constant
  parameters), a list of :any:`Module` objects can be provided via
  :data:`definitions`.
* When there are C-preprocessor macros or includes used in the source file,
  a C-preprocessor (`pcpp <https://github.com/ned14/pcpp>`_) can be applied
  to the file before reading it. For that, include paths and macro definitions
  can be specified.
* Choosing a different frontend (see below).

See the description of :any:`Sourcefile.from_file` for a description of all
available options.

As an alternative to reading source files one can also parse a Python string
directly using :any:`Sourcefile.from_source`. With that, it is also possible to
directly create modules or subroutines using the common parent routine
:any:`ProgramUnit.from_source`.  This is particularly useful for writing tests,
avoiding the detour via an external file.

.. code-block:: python

   fcode = """
   subroutine axpy(a, x, y)
     real, intent(in) :: a, x(:)
     real, intent(inout) :: y(:)

     y(:) = a * x(:) + y(:)
   end subroutine axpy
   """.strip()
   routine = Subroutine.from_source(fcode)

In rare cases (e.g., when parsing a pragma annotation), parsing a standalone
expression may be required. Experimental support for that is provided via
the utility function :any:`parse_fparser_expression`.

Frontends
=========

Three different externally developed frontends are currently supported, each
of them with individual advantages and shortcomings:

* `Fparser 2 <https://github.com/stfc/fparser>`_, developed by STFC as a
  rewrite of the original fparser that is included in
  `f2py <https://numpy.org/doc/stable/f2py/>`_, (now a part of numpy).
  It is written in pure Python, supports Fortran 2003 and some Fortran 2008,
  and is actively maintained. The default frontend in Loki.
* `Omni Compiler Frontend <https://omni-compiler.org/>`_, developed in the
  Omni Compiler Project. It is written in Java, supports Fortran 2008 and
  is also used in the `CLAW compiler <https://claw-project.github.io/>`_.
  Compared to the other frontends, OMNI performs a lot of transformations
  internally (unifies case, propagates constants, inlines statement
  functions, etc.), thus prevents string reproducibility. Biggest drawback
  is the very rigorous dependency chasing (with custom ``.xmod`` files), that
  disallows dangling symbol definitions via imports and therefore prevents
  partial source tree processing.
* `Open Fortran Parser <https://github.com/OpenFortranProject/open-fortran-parser>`_
  with a customized
  `Python wrapper <https://github.com/mlange05/open-fortran-parser-xml/tree/mlange05-dev>`_.
  It is written in Java, claims Fortran 2008 support, and is also part of the
  `ROSE Compiler framwork <http://rosecompiler.org/>`_. It is lacking support
  for some Fortran features, notably slower than the other frontends and not
  actively developed at the moment.

.. important::
   By default, Loki uses Fparser 2.

.. autosummary::

   loki.frontend.util.Frontend

When invoked, every frontend produces an abstract syntax tree that is then
transformed to Loki's own internal representation.

.. autosummary::

   loki.frontend.fparser
   loki.frontend.omni
   loki.frontend.ofp


Preprocessing
=============

When reading a source file, a C99-preprocessor can be applied to the file
before passing it to the frontend. This can be enabled by specifying
:data:`preprocess` when calling `Sourcefile.from_file`. The corresponding
routine carrying out the preprocessing is :any:`preprocess_cpp`.

Source sanitization
===================

Internally, Loki performs also another kind of preprocessing to work around
known shortcomings in frontends. This is done via a regex-based replacement
of known incompatibilities that are later-on reinserted into the Loki IR.
This preprocessing step is applied automatically and does not require any
user intervention.

.. autosummary::

   loki.frontend.preprocessing.sanitize_input
   loki.frontend.preprocessing.sanitize_registry
   loki.frontend.preprocessing.PPRule
