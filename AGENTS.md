# AGENTS

## Loki Test Conventions

This repository contains Loki source code and tests.

## Package Overview

Loki is a Python toolkit for source-to-source analysis and transformation of
Fortran, aimed primarily at large scientific code bases where routines, modules,
loops, declarations, and expressions need to be inspected, rewritten, linted, or
regenerated programmatically. Its core workflow is to parse Fortran into Loki's
internal representation, operate on that representation with native IR objects,
visitor utilities, expression mappers, and transformation passes, then emit
Fortran again through a backend. Prefer working with the structured IR and
symbol semantics directly, preserving source intent while making explicit,
minimal transformations that can scale from single routines to whole source
trees via the scheduler.

## Repository Map

- `loki/`: main Python package containing Loki's IR, frontends, backends,
  transformations, scheduler, CLI, linter integration, and tests.
- `loki/frontend/`: Fortran source parsing frontends, preprocessing, and
  frontend-specific test utilities for FP, OMNI, and REGEX parsing modes.
- `loki/ir/`: intermediate representation nodes, visitors, finders,
  transformers, expression visitors, pragma utilities, and IR graph helpers.
- `loki/expression/`: symbolic expression parsing, symbols, mappers, and related
  expression utilities built around pymbolic.
- `loki/backend/`: code generators and stringifiers that render Loki IR back to
  Fortran or other textual representations.
- `loki/batch/`: scheduler, item graph, configuration, and bulk-processing
  infrastructure for source-tree transformations.
- `loki/transformations/`: reusable transformation passes and domain-specific
  transformation packages, with colocated tests under `tests/` subdirectories.
- `loki/analyse/`: static analysis helpers used by transformations and tests.
- `loki/build/` and `loki/jit_build/`: build and JIT compilation helpers used by
  tests and transformation workflows that need compiled Fortran artifacts.
- `loki/lint/`: core loki-lint framework and reporting infrastructure.
- `loki/cli/`: command-line entry points for `loki-transform.py` and
  `loki-lint.py`.
- `loki/tools/` and `loki/types/`: shared utility functions and Loki type-system
  objects.
- `lint_rules/`: separate installable package containing IFS/ARPEGE lint rules
  and their tests.
- `docs/`: Sphinx documentation sources and build helpers.
- `example/`: maintained notebooks and small Fortran examples demonstrating Loki
  usage.
- `cmake/`: CMake/ecbuild integration files for embedding Loki in CMake builds.

## Environment Setup

For normal development, prefer using the repository-provided Loki environment
instead of manually creating or managing a virtual environment:

```bash
source ./loki-activate
```

Reinstalling the environment is rarely needed. If dependencies or optional
frontends do need to be refreshed, use the repository `./install` script rather
than invoking `pip` manually. On ECMWF machines, use `--hpc2020` when running the
installer so the expected local modules and settings are applied.

## Local Validation Workflow

When validating Loki changes locally:

- always activate the local Loki environment first with `source ./loki-activate`
- use the repository-local pylint configuration with `pylint --rcfile .pylintrc ...`
- include both lint and test validation in normal verification
- during normal development, focus first on targeted tests for the touched local
  Loki sub-packages and make sure the touched code is linter compliant
- after targeted checks pass, broaden to full Loki pytest validation; use
  `-k "not cmake"` to avoid costly CMake tests unless changing the CMake layer
- avoid running CLOUDSC/ECWAM regression tests by default because they require
  local installs of the upstream packages; if regression validation is needed,
  ask the user for install directories and preferred architecture files

Typical validation commands:

```bash
source ./loki-activate
python -m pytest <relevant tests>
pylint --rcfile .pylintrc <relevant paths>
python -m pytest -v --pyargs loki -k "not cmake"
```

For broader validation, keep using the same activated environment and local `.pylintrc`.


## Frontend Test Conventions

When adding or updating parser-dependent tests:

- in general, test with all `available_frontends()` unless the feature is
  intentionally frontend-specific
- use explicit `skip` or `xfail` entries with a reason when a frontend has a
  known limitation or requires source context that the test does not provide
- expect OMNI to require special-casing more often than FP, especially for tests
  involving modules, derived types, or external definitions
- use `tmp_path` and pass `xmods=[tmp_path]` when tests need OMNI-generated
  boilerplate/module files to stay isolated between test invocations
- include the REGEX frontend only when the behavior is relevant to quick parsing,
  call-tree discovery, or bulk processing through `loki.batch.Scheduler`; it is
  not intended to validate full Fortran parsing semantics


## Development Practices

When adding or changing Loki code:

- always add docstrings for new methods and keep them compatible with the
  Sphinx-generated API documentation, using NumPy-style docstring conventions
- prefer fixing bugs or unsupported corner cases at their source over quick
  local patches or workarounds; when a proper fix is larger or changes
  established behavior, understand the cause and ask the user whether to
  implement it. For example, if you have to write a paragraph long comment to
  explain some code, then that's a strong hint that you should not be applying
  a local workaround.
- prefer using Loki's public API and native IR semantics where possible instead
  of adding local workarounds
- avoid overzealous type checking or defensive guards when the Loki API already
  provides the expected behavior
- for recurring usage patterns, consider extending the Loki API or an existing
  utility rather than duplicating bespoke logic in one local call site


## IR and API Usage

When editing Loki IR handling, transformation code, or tests:

- in core Loki implementation modules, import from the concrete subpackage that
  defines the API; in tests and user-facing transformation code, prefer imports
  from the first Loki sub-package layer
- prefer `import loki.ir as ir, FindNodes, Transformer` and refer to
  IR classes through `ir`, for example `ir.Loop` and `ir.Assignment`
- prefer `from loki.expression import symbols as sym` and refer to symbolic
  classes through `sym`, for example `sym.Variable`, `sym.Array`, and
  `sym.IntLiteral`
- use `FindNodes(...)` for control-flow IR node discovery and
  expression visitors such as `FindVariables`, `FindInlineCalls`, and
  `FindLiterals` for expression trees embedded in IR nodes
- use `Transformer` or `NestedTransformer` for IR node replacement,
  removal, or one-to-many rewrites; remember that transformers rebuild nodes by
  default, so stored node references may become stale unless you use
  `transformer.rebuilt` or intentionally pass `inplace=True`
- prefer in-place updates with `._update(...)` for performance when scoping and
  object identity side effects are understood; prefer `.clone(...)` when objects
  are re-scoped, intentionally duplicated, or otherwise need independent copies
- use `SubstituteExpressions` and expression mappers for expression
  replacement rather than manually walking every node field
- for logical expression matching, prefer direct comparisons against symbols,
  expressions, and strings; avoid eager `str(...)`, `.lower()`, or `.upper()`
  conversion because Loki symbol comparisons are usually case-insensitive
- do not add extra string-safety checks when a simple native, case-insensitive
  comparison is sufficient
- use `GenericStmt` only as a fallback for unsupported statements; if a useful
  standard statement type is missing, ask whether to implement a structured IR
  node instead of adding more generic text handling
- access `uses_symbols`, `defines_symbols`, and `live_symbols` only after
  attaching dataflow analysis, preferably via the `dataflow_analysis_attached(...)`
  context manager
- prefer existing utilities in `loki.transformations.utilities` and similar
  modules before adding new helpers; extend an existing utility when the usage
  pattern matches, but avoid simplistic helpers when direct API usage is clearer
- avoid broad post-hoc `rescope_symbols` calls; if explicit re-scoping is needed,
  prefer constructing replacements with `.clone(scope=new_scope)`
- when transformations add, remove, rename, or change dependencies between
  modules, routines, imports, or calls, consider `plan_*` methods and scheduler
  flags such as `creates_items` and `renames_items`; ask before adding this
  machinery if the need is not clear
- Do not create tests by copying external source code (for example the IFS).
  Instead create a minimal reproducer that doesn't allow to infer closed-source code.


## Commit Message Style

When creating commits in this repository:

- Use a sub-package or area marker in the subject, followed by a colon.
- Keep the subject concise and imperative, for example:
  `JIT: Add isolated JIT execution helper`
  `Transformations: Isolate parametrise JIT test execution`
- Wrap commit body lines like Python docstrings, preferably around 72 characters.
- Use the body to explain why the change is needed and what behavior it protects.


## Loki Test Assertions

When editing or adding tests, prefer assertions that match Loki's native IR and expression semantics rather than assertions that depend on rendered source formatting.

When editing test imports, keep `loki.*` import lines ordered alphabetically by submodule, but do not reorder names within a single import line unless there is a separate reason to do so.

- In structural IR tests, prefer native Loki node comparisons over `str(...)`.
- Loki symbols and expressions compare directly to strings, so prefer:
  - `node == 'a + b'` over `str(node) == 'a + b'`
  - `loop.variable == 'i'` over `str(loop.variable) == 'i'`
- Loki numeric literals compare directly to Python numbers, so prefer:
  - `loop.bounds.start == 1` over `str(loop.bounds.start) == '1'`
  - `literal == 5` over `str(literal) == '5'`
- When creating local test helpers, return native nodes whenever possible.
  - Good: `(assign.lhs, assign.rhs)`
  - Good: `(loop.variable, loop.bounds.start, loop.bounds.stop, loop.bounds.step)`
  - Avoid: `(str(assign.lhs), str(assign.rhs))`
  - Avoid: `(str(loop.variable), str(loop.bounds.start), str(loop.bounds.stop), ... )`
- Use stringification only when the test is explicitly about rendered output, pretty-printing, or a node type that does not compare reliably through Loki's native equality support.
- If stringification is still necessary in a structural test, keep it narrowly scoped and document why.
