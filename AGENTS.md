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
