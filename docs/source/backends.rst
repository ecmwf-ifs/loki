======================
Generating source code
======================

.. important::
    Loki is still under active development and has not yet seen a stable
    release. Interfaces can change at any time, objects may be renamed, or
    concepts may be re-thought. Make sure to sync your work to the current
    `master` frequently by rebasing feature branches and upstreaming
    more general applicable work in the form of pull requests.


At the end of a source-to-source translation process the output source code
needs to be generated. Loki provides a number of different backends depending
on the target language, which, once again, are :doc:`Visitors <visitors>`.

All backends are subclasses of :class:`Stringifier` that convert the internal
representation to a string representation in the syntax of the target language.

.. autosummary::

   loki.visitors.pprint.Stringifier
   loki.visitors.pprint.pprint

Typically, this includes also a custom mapper for expression trees as a
subclass of :any:`LokiStringifyMapper`. For convenience, each of these
visitors is wrapped in a corresponding utility routine that allows to generate
code for any IR object via a simple function call, for example:

.. code-block:: python

   routine = Subroutine(...)
   ...
   fcode = fgen(routine)

Currently, Loki has backends to generate Fortran, C, Python, and Maxeler MaxJ
(a Java dialect that targets FPGAs).

.. autosummary::

   loki.backend.fgen.fgen
   loki.backend.cgen.cgen
   loki.backend.pygen.pygen
   loki.backend.dacegen.dacegen
   loki.backend.maxgen.maxjgen

.. warning::
   Backends do not make sure that the internal representation is
   compatible with the target language. Adapting the IR to the desired output
   format needs to be done before calling the relevant code generation routine.
   For language transpilation (e.g., Fortran to C), corresponding
   :doc:`transformations <transformations>` must be applied
   (e.g., :any:`FortranCTransformation`).
