=========
loki-lint
=========

.. contents:: Contents
   :local:

Loki's ability to parse Fortran source files into an internal representation
that is easy to work with has been used to create a protype implementation of
a static source code analysis tool for Fortran. The intention is to verify
compliance with the
`IFS coding standard <https://www.umr-cnrm.fr/gmapdoc/IMG/pdf/coding-rules.pdf>`_
but the tool itself is generic enough to be used for other use cases, too.


Installation
============

A generic Loki installation as described in :doc:`INSTALL.md` also installs the
linting script. However, the linting tool uses only the
:doc:`Fparser frontend <frontends>` and therefore it is a pure Python package.
If loki-lint is the only intended use case, a minimum installation that avoids
all Java dependencies is sufficient and can be done using the install script:

.. code-block:: bash

   ./install [--ecmwf] --without-ofp

(Note that ``--ecmwf`` is required on ECMWF workstations to load required
modules and apply proxy configuration.)

Basic usage
===========

To be able to use loki-lint, the virtual environment created as part of Loki's
installation needs to be loaded:

.. code-block:: bash

   cd /path/to/loki
   source loki_env/bin/activate

The basic command for loki-lint is

.. code-block:: bash

   loki-lint.py [--log <logfile>] check [--basedir <path>] [--include <pattern>] [--exclude <pattern>]

The most important option is ``--include`` that specifies the pattern of file
names that should be checked. For example: ``--include **/*.F90`` checks all
files with suffix ``.F90`` in all subdirectories of the current working
directory. This option can be specified multiple times to add multiple files
or more than one directory (sub-)tree.

Patterns can be absolute paths or relative to the current working directory.
Optionally, a different base directory relative to which patterns should be
interpreted can be specified with ``--basedir``.

An optional exclusion pattern can be given with ``--exclude`` that skips files
with names matching that pattern.

Progress and warnings are reported on the command line. Optionally, an
additional log file is written with the name given via ``--log``.

Examples
--------

.. dropdown:: Minimal example

   This checks only the ``cloudsc.F90`` file:

   .. code-block:: bash

      $~> loki-lint.py check --include ~/ifs-source/develop/ifs/phys_ec/cloudsc.F90
      Base directory: /var/tmp/tmpdir/nabr/git/loki/nabr-linter
      Include patterns:
        - /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90
      Exclude patterns:

      1 files selected for checking (0 files excluded).

      Using 4 worker.
      10 rules available.
      Checking against 10 rules.

      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2527-2531) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2610-2624) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2611-2617) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2619-2623) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2666-2674) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2676-2682) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2747-2763) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2779-2783) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2785-2817) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2787-2815) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 2821-2825) - Nesting of conditionals exceeds limit of 3.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2434) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2437) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2486) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2487) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2488) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2489) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2802) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2855) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 2984) - "0.8" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3117) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3297) - "1.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3297) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3298) - "273.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3298) - "1.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3298) - "393.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 3298) - "120.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.65" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4166) - "0.00001" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4171) - ".5" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4171) - ".001" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4172) - ".24" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4172) - ".11" without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4172) - "10." without explicit KIND declared.
      [4.7] ExplicitKindRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (l. 4172) - ".03" without explicit KIND declared.
      [2.2] LimitSubroutineStatementsRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 5-4215) in routine "CLOUDSC" - Subroutine has 1010 executable statements (should not have more than 300)
      [3.6] MaxDummyArgsRule: /home/rd/nabr/ifs-source/develop/ifs/phys_ec/cloudsc.F90 (ll. 5-4215) in routine "CLOUDSC" - Subroutine has 75 dummy arguments (should not have more than 50)

      1 files parsed successfully


.. dropdown:: Minimal example with a different ``--basedir``

   This checks only the ``cloudsc.F90`` file but specifies a different base
   directory. Note the difference in output:

   .. code-block:: bash

      $~> loki-lint.py check --basedir ~/ifs-source/develop/ifs --include phys_ec/cloudsc.F90
      Base directory: /home/rd/nabr/ifs-source/develop/ifs
      Include patterns:
        - phys_ec/cloudsc.F90
      Exclude patterns:

      1 files selected for checking (0 files excluded).

      Using 4 worker.
      10 rules available.
      Checking against 10 rules.

      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2527-2531) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2610-2624) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2611-2617) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2619-2623) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2666-2674) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2676-2682) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2747-2763) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2779-2783) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2785-2817) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2787-2815) - Nesting of conditionals exceeds limit of 3.
      [1.3] CodeBodyRule: phys_ec/cloudsc.F90 (ll. 2821-2825) - Nesting of conditionals exceeds limit of 3.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2434) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2437) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2486) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2487) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2488) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2489) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2802) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2855) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 2984) - "0.8" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3117) - "0.4" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3297) - "1.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3297) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3298) - "273.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3298) - "1.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3298) - "393.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 3298) - "120.0" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.65" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (ll. 3304-3305) - "0.5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4166) - "0.00001" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4171) - ".5" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4171) - ".001" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4172) - ".24" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4172) - ".11" without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4172) - "10." without explicit KIND declared.
      [4.7] ExplicitKindRule: phys_ec/cloudsc.F90 (l. 4172) - ".03" without explicit KIND declared.
      [2.2] LimitSubroutineStatementsRule: phys_ec/cloudsc.F90 (ll. 5-4215) in routine "CLOUDSC" - Subroutine has 1010 executable statements (should not have more than 300)
      [3.6] MaxDummyArgsRule: phys_ec/cloudsc.F90 (ll. 5-4215) in routine "CLOUDSC" - Subroutine has 75 dummy arguments (should not have more than 50)

      1 files parsed successfully


.. dropdown:: Example for a complete command line

   This specifies a custom path relative to which the patterns are to be
   interpreted and includes all F90-files in the ``phys_ec`` and ``module``
   directories. Note that single quotes may be necessary to ensure the shell
   does not expand the pattern. Output is written to a log file with current
   date and time in the file name.

   .. code-block:: bash

      loki-lint.py --log ifs_$(date +"%Y%m%d-%H%M").log check --basedir /path/to/ifs-source/branch/ifs --include 'phys_ec/*.F90' --include 'module/*.F90'


Help command
------------

loki-lint has a built-in help output detailing the use of the application. Run

.. code-block:: bash

   loki-lint.py --help

to display the generic help text, and

.. code-block:: bash

   loki-lint.py check --help

gives some advice about the usage of the source file checker and its options.
This includes some advanced options not mentioned here.

The list of available rules that source files are tested against can be
displayed by running (optionally with their ID and a short description for
each rule):

.. code-block:: bash

   loki-lint.py rules [--with-title]


Configuration
=============

The behaviour of Loki-lint and its rules can be configured using a YAML
configuration file. Currently, this allows to change settings for individual
rules as well as the list of rules to be checked.

For that, simply provide the config file in the command line like this:

.. code-block:: bash

   loki-lint.py check --config <configfile>

The default configuration can be displayed (and optionally written to file)
using:

.. code-block:: bash

   loki-lint.py default-config [--output-file <filename>]

This default configuration can then be used as a template for creating an
individual configuration file. Any options not specified explicitly in the
configuration file are chosen to be default values.


Implementing own rules
======================

All rules are implemented in :mod:`scripts.lint_rules`. Currently, this includes
only one module (:mod:`scripts.lint_rules.ifs_coding_standards_2011`) that
contains (a small subset of) the rules defined in the IFS coding standards
document. To be able to write own rules a rudimentary understanding of
:doc:`internal_representation` is helpful.

Each rule is represented by a subclass of :any:`GenericRule` with the
following structure:

.. code-block:: python

   class MyOwnRule(GenericRule):

       type = RuleType.WARN

       docs = {
           'id': '13.37',
           'title': 'Scientists should write {what_now}.',
       }

       config = {
           'some_option': 'some value',
           'what_now': 'sensible code',
           'another_option': ['a', 'list', 'of', 'values']
       }

       fixable = True

       @classmethod
       def check_module(cls, module, rule_report, config):
           # Implement checks on module level here
           rule_report.add("Problem in this module", module)

       @classmethod
       def check_subroutine(cls, subroutine, rule_report, config):
           # Implement checks on subroutine level here
           rule_report.add("Problem in this subroutine", subroutine)

       @classmethod
       def check_file(cls, sourcefile, rule_report, config):
           # Implement checks on source file level here
           rule_report.add("Problem in this file", sourcefile)

       @classmethod
       def fix_subroutine(cls, subroutine, rule_report, config):
           # Implements logic that attempts to fix the problems that
           # were flagged in rule_report


Properties of a rule
--------------------

* :attr:`type` : The type, category or severity of that rule. Available types
  are defined in :any:`RuleType` and comprise currently :attr:`INFO`,
  :attr:`WARN`, :attr:`SERIOUS`, :attr:`ERROR` (with increasing severity).

* :attr:`docs` : A short description of that rule. At the moment, this includes
  by default

   * :attr:`id` : The rule number according to the IFS Coding standards

   * :attr:`title` : A short description of that rule. It may contain placeholder
     values (such as ``{what_now}``) that are replaced by the corresponding
     value from the config when displaying the rules (see example above).

* :attr:`config` : A dictionary that allows to parametrize the rule, with given
  default values. These options are exposed via the config file mentioned
  above, where defaults can be overwritten.

* :attr:`fixable` : `True`/`False` to indicate if the rule has a method
  :meth:`fix_*` that can be used to make an attempt of automatically fixing
  the problems the corresponding :meth:`check_*` method reported. Defaults to
  `False`.

.. note::
   Automatic fixing of rules is currently in prototype stage and the API may
   change in the future.

Further **properties for future use** are already implemented but not currently
used:

* :attr:`deprecated` : `True`/`False` to indicate when a rule has been
  superseded by other rules (e.g., due to a new revision of the Coding
  Standards). Defaults to `False`.
* :attr:`replaced_by` : A tuple that can be used to specify the rule(s) that
  replaced this rule when it became deprecated.


Methods of a rule
-----------------

The core of a rule are its :meth:`check*` methods, which implement its behaviour.
Depending on the nature of a rule, it may require checks to be carried out on
different levels in the hierarchy of a source file (the :any:`Sourcefile` itself
or :any:`Module` or :any:`Subroutine` that are contained in it). For that reason,
there are multiple entry points that a rule can implement, depending on the
specific needs. Any function that is not required can simply be left out. The
driver of loki-lint calls each of the following routines for every entity in a
source file:

* :meth:`check_file` once for the file (:any:`Sourcefile`),
* :meth:`check_module` for every module (:any:`Module`) in that file, and
* :meth:`check_subroutine` for every subroutine (:any:`Subroutine`) in that
  file and for every subroutine contained in a module in that file, and for
  every subroutine contained in a subroutine in that file, etc.

**Arguments** given to each of those routines are

* A :any:`Sourcefile`, :any:`Module` or :any:`Subroutine` object;
* The reporter (:any:`RuleReport`) for this rule, to which detected problems
  can be reported (see below);
* A `dict` holding the configuration values (defaults or from the config file).


Reporting of problems
---------------------

Problems detected by a rule are reported by calling
``rule_report.add(message, location)``. Here, :data:`message` is an arbitrary
string describing the problem, and :data:`location` can be an arbitrary node of
the internal representation in which the problem occured. This parameter will
later be used to provide information about the location of the problem (e.g.,
line number).


Example of a rule
-----------------

To illustrate the use of :doc:`internal_representation` and how a rule is
implemented with that, consider the following example:

.. code-block:: python

   class MplCdstringRule(GenericRule):  # Coding standards 3.12

       type = RuleType.SERIOUS

       docs = {
           'id': '3.12',
           'title': 'Calls to MPL subroutines should provide a "CDSTRING" identifying the caller.',
       }

       @classmethod
       def check_subroutine(cls, subroutine, rule_report, config):
           '''Check all calls to MPL subroutines for a CDSTRING.'''
           for call in FindNodes(ir.CallStatement).visit(subroutine.ir):
               if call.name.upper().startswith('MPL_'):
                   for kw, _ in call.kwarguments:
                       if kw.upper() == 'CDSTRING':
                           break
                   else:
                       fmt_string = 'No "CDSTRING" provided in call to {}'
                       msg = fmt_string.format(call.name)
                       rule_report.add(msg, call)

This rule checks all calls to ``MPL_`` subroutines for the presence of a
keyword-argument ``CDSTRING`` that should provide identification of the
caller. Note the following implementation details of the class:

* The rule is categorized as :data:`SERIOUS`.
* Documentation contains its ID (3.12) and title (here, providing the full
  wording from the coding standards document).
* There is no config that modifies the behaviour of the rule.
* There is a single entry point to that rule: Only the method
  :meth:`check_subroutine` is implemented that is called for all subroutines
  in a source file (irrespective whether it is a free function in the file,
  or contained in a module or subroutine).

The implementation of :meth:`check_subroutine` features the following details:

* It uses the :doc:`visitor <visitors>` :any:`FindNodes` to find all
  :any:`CallStatement` nodes; this visitor is applied to the subroutine's IR,
  which is available via the attribute :any:`Subroutine.ir`.
* For every ``call`` node, it takes the name of the called routine
  (available as property :attr:`name` and converted to uppercase as Fortran is
  case-insensitive) and checks if it starts with ``MPL_``.
  For each such call node, it looks at all keyword arguments (available as list
  of ``(keyword, value)``-tuples in the property :attr:`kwarguments`).

  * If keyword ``CDSTRING`` is found, the search loop is stopped (with
    ``break``) and the outer visitor loop continues with the next call node;
  * if the loop terminates normally (i.e., break was not invoked) then no such
    keyword argument was found and the loop's ``else`` block is executed (this
    is a Python-specific feature allowing to execute a block of code only if a
    loop was not terminated "abnormally"). There, a message text is formed by
    inserting the name of the called routine into the ``fmt_string``. Then,
    this is reported to ``rule_report`` together with the problematic IR node
    ``call``. Later, the output handler will use this node to determine the
    exact position in the source file (e.g., to report the line number).

Note that this rule does not report anything if no problematic calls are present.

An example output of this rule looks as follows:

.. code-block:: text

  [3.12] MplCdstringRule: cma2odb/distio_mix.F90 (l. 821) - No "CDSTRING" provided in call to MPL_BROADCAST


Known issues
============

In general, bugs and open questions are collected in Loki's
`JIRA space <https://jira.ecmwf.int/projects/LOKI/issues>`_ and this is also
the best place to report any problems.

One important limitation is that loki-lint currently does not invoke a
C-preprocessor. Although Loki has now a built-in
:ref:`preprocessor <frontends:Preprocessing>`, this is not currently used in
loki-lint. Therefore, preprocessor directives are not interpreted but
essentially treated as comments. Thus, a code that does not reduce to
(syntactically) valid Fortran when ignoring PP directives, parsing that
file will fail (e.g., because each branch of an ``#ifdef ... #else ... #endif``
construct provides a different ``IF`` statement for a common ``ENDIF``).

For other limitations of Frontends or the IR, Loki has a built-in sanitizer for
input files to maneuver around some of the deficiencies. A relatively short
list of known files that cause problems is maintained in
`Confluence <https://confluence.ecmwf.int/display/~nabr/IFS+source+files+causing+problems>`_.
