# AGENTS

## Purpose

This repository contains Loki source code and tests. When editing or adding tests, prefer assertions that match Loki's native IR and expression semantics rather than assertions that depend on rendered source formatting.

## Loki Test Assertions

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
