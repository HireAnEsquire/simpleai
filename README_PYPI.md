# PyPI Publishing Guide

This guide explains how to publish `simpleai` to PyPI.

## Prerequisites

- PyPI account with project permissions.
- Clean git state.
- Updated version in `pyproject.toml`.
- Updated `CHANGELOG.md`.

## 1. Build artifacts

```bash
python -m pip install --upgrade build twine
python -m build
```

This creates:
- `dist/*.tar.gz`
- `dist/*.whl`

## 2. Validate distributions

```bash
twine check dist/*
```

## 3. Publish to TestPyPI (recommended)

```bash
twine upload --repository testpypi dist/*
```

Install test package:

```bash
pip install --index-url https://test.pypi.org/simple/ simpleai
```

## 4. Publish to PyPI

```bash
twine upload dist/*
```

## 5. Verify install

```bash
pip install simpleai
python -c "from simpleai import run_prompt; print(run_prompt)"
```

## Trusted Publishing (recommended)

For CI/CD, use PyPI Trusted Publishing (OIDC) in GitHub Actions instead of long-lived API tokens.

Docs:
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)

## Release checklist

1. Bump `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Run tests: `pytest`.
4. Build: `python -m build`.
5. Check: `twine check dist/*`.
6. Publish (TestPyPI then PyPI).
7. Tag release in git.
