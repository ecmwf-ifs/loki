# AGENTS

## Local Validation Workflow

When validating Loki changes locally:

- always activate the local environment first with `source ./loki-activate`
- use the repository-local pylint configuration with `pylint --rcfile .pylintrc ...`
- include both lint and test validation in normal verification
- prefer running targeted `pytest` suites for touched areas first, then broaden if needed

Typical validation commands:

```bash
source ./loki-activate
python -m pytest <relevant tests>
pylint --rcfile .pylintrc <relevant paths>
```

For broader validation, keep using the same activated environment and local `.pylintrc`.

## Loki Test Conventions

This repository contains Loki source code and tests.

## Local Validation Workflow

When validating Loki changes locally:

- always activate the local environment first with `source ./loki-activate`
- use the repository-local pylint configuration with `pylint --rcfile .pylintrc ...`
- include both lint and test validation in normal verification
- prefer running targeted `pytest` suites for touched areas first, then broaden if needed

Typical validation commands:

```bash
source ./loki-activate
python -m pytest <relevant tests>
pylint --rcfile .pylintrc <relevant paths>
```

For broader validation, keep using the same activated environment and local `.pylintrc`.


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

