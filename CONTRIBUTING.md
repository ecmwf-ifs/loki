<!--
SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
SPDX-License-Identifier: Apache-2.0
SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
-->

# Contributing to Loki

Thank you for your interest in contributing to Loki. Loki is an ECMWF research
tool for source-to-source analysis and transformation of Fortran code.

## Before You Start

For non-trivial changes, please open or comment on an issue first so the
approach, scope, and expected review effort can be discussed with the
maintainers.

By opening a pull request, you agree that your contribution is made under Loki's
licence, the Apache License Version 2.0, and under the terms of the ECMWF
Contributor Licence Agreement:

<https://github.com/ecmwf/codex/blob/main/Legal/Contributor-License-Agreement.md>

## Development Setup

Use the repository-provided environment:

```bash
source ./loki-activate
```

If dependencies need to be refreshed, prefer the repository installer:

```bash
./install --with-tests --with-examples
```

On ECMWF HPC systems, use the appropriate installer option for the platform,
for example `--hpc2020`.

## Making Changes

Keep changes focused and reviewable. Prefer small, direct changes that preserve
the existing APIs and coding style unless a broader change has been agreed with
the maintainers.

When working on Loki internals, use the structured IR and public Loki APIs where
possible. Avoid changing Fortran parser fixtures unless the fixture content is
itself the thing under test.

New or changed behaviour should include tests. Parser-dependent tests should use
the available Loki frontend test conventions unless the feature is explicitly
frontend-specific.

## Local Validation

Run the relevant targeted tests first:

```bash
source ./loki-activate
python -m pytest <relevant tests>
```

Run linting for changed Python code:

```bash
pylint --rcfile .pylintrc <relevant paths>
```

Check licensing metadata:

```bash
reuse lint
```

For broader validation, run:

```bash
python -m pytest -v --pyargs loki -k "not cmake"
```

Avoid running CLOUDSC or ECWAM regression tests by default unless the change
requires them and the required upstream packages are available locally.

## Pull Requests

Submit contributions through a GitHub pull request against `main`.

Before requesting review:

- Make sure the pull request has a clear description of the problem, solution,
  and any trade-offs.
- Link relevant issues or discussions.
- Include tests or explain why tests are not applicable.
- Update documentation when behaviour, public APIs, or user workflows change.
- Self-review the diff and remove accidental debug code or generated artefacts.
- Ensure CI checks pass.

ECMWF maintainers review pull requests for correctness, maintainability,
licensing, and project fit. Public contributions may require maintainer approval
before CI is run.

## Code of Conduct

All participation in this project is governed by the Loki
[Code of Conduct](CODE_OF_CONDUCT.md).
