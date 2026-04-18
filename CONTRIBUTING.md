# Contributing to Titan-GeoAlign

Thank you for considering a contribution to **Titan-GeoAlign**!
We welcome bug reports, feature requests, documentation improvements, and code contributions of all sizes.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Help](#getting-help)
3. [Reporting Bugs](#reporting-bugs)
4. [Requesting Features](#requesting-features)
5. [Development Setup](#development-setup)
6. [Branching & Workflow](#branching--workflow)
7. [Code Style](#code-style)
8. [Testing Requirements](#testing-requirements)
9. [Commit Message Format](#commit-message-format)
10. [Pull Request Process](#pull-request-process)
11. [Release Process](#release-process)

---

## Code of Conduct

This project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

---

## Getting Help

- Open a [GitHub Discussion](https://github.com/goad223/Titan-GeoAlign/discussions) for questions.
- Use [GitHub Issues](https://github.com/goad223/Titan-GeoAlign/issues) only for **bugs** and **feature requests**.

---

## Reporting Bugs

Before filing a bug, please:

1. Check existing [issues](https://github.com/goad223/Titan-GeoAlign/issues) to avoid duplicates.
2. Reproduce the bug on the **latest** `main` branch.

When filing, include:

- **Environment** – OS, Python version, package version (`titan-align --version`).
- **Minimal reproducible example** – the smallest possible code that demonstrates the issue.
- **Expected vs actual behaviour**.
- **Full traceback / logs**.

---

## Requesting Features

Open a GitHub Issue with the label `enhancement` and describe:

- The problem you are trying to solve.
- Your proposed solution or API design.
- Any alternatives you have considered.

---

## Development Setup

### Prerequisites

| Tool | Minimum version |
|------|-----------------|
| Python | 3.12 |
| Git | 2.40 |
| Git LFS | 3.x |
| GDAL system library | 3.8 |
| (Optional) CUDA toolkit | 12.4 |

### Fork & clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/Titan-GeoAlign.git
cd Titan-GeoAlign

# Add the upstream remote
git remote add upstream https://github.com/goad223/Titan-GeoAlign.git
```

### Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Install development dependencies

```bash
make install-dev
```

This installs all runtime + dev dependencies and registers the pre-commit hooks.

### Verify your setup

```bash
make test          # unit tests should pass
make lint          # linters should report no errors
```

---

## Branching & Workflow

We follow a **GitHub Flow** model:

```
main  ←  feature/your-feature-name
      ←  fix/issue-42-short-description
      ←  docs/update-readme
      ←  chore/bump-dependencies
```

| Branch prefix | Purpose |
|---------------|---------|
| `feature/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behaviour change |
| `perf/` | Performance improvements |
| `chore/` | Dependency bumps, CI tweaks, housekeeping |
| `test/` | Tests only |

### Step-by-step

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create your branch
git checkout -b feature/my-awesome-feature

# 3. Make changes, commit frequently (see commit format below)
git add .
git commit -m "feat(align): add sub-pixel registration via phase correlation"

# 4. Keep your branch up-to-date
git fetch upstream
git rebase upstream/main

# 5. Push and open a Pull Request
git push origin feature/my-awesome-feature
```

---

## Code Style

We enforce consistent style automatically using:

| Tool | Purpose | Config |
|------|---------|--------|
| **black** | Code formatting | `pyproject.toml [tool.black]` |
| **ruff** | Linting + import sorting | `pyproject.toml [tool.ruff]` |
| **mypy** | Static type checking | `pyproject.toml [tool.mypy]` |

Run all checks at once:

```bash
make lint       # check only
make format     # auto-fix formatting issues
```

### Key rules

- **Line length**: 100 characters.
- **Type annotations**: required on all public functions and methods.
- **Docstrings**: Google-style for all public modules, classes, and functions.
- **Imports**: stdlib → third-party → first-party (enforced by ruff/isort).
- No `print()` statements in library code — use `structlog` or `logging`.

### Example function signature

```python
def register_images(
    source: np.ndarray,
    target: np.ndarray,
    *,
    method: RegistrationMethod = RegistrationMethod.PHASE,
    max_shift_px: float = 50.0,
) -> RegistrationResult:
    """Align *source* to *target* and return the registration result.

    Args:
        source: Source image array, shape (H, W) or (H, W, C).
        target: Target/reference image array, same spatial shape as *source*.
        method: Registration algorithm to use.
        max_shift_px: Maximum allowed shift in pixels; raises if exceeded.

    Returns:
        A :class:`RegistrationResult` containing the transformation matrix
        and quality metrics.

    Raises:
        RegistrationError: If the shift exceeds *max_shift_px*.
    """
```

---

## Testing Requirements

All code changes **must** be accompanied by tests.

### Running tests

```bash
make test             # fast unit tests only
make test-all         # full suite including integration
make test-cov         # with HTML coverage report
make test-parallel    # parallelised with pytest-xdist
```

### Test structure

```
tests/
├── unit/                  # fast, no I/O, no network
│   ├── test_alignment.py
│   └── ...
├── integration/           # may hit filesystem, network
│   └── ...
└── conftest.py
```

### Coverage requirements

- Minimum **80 %** line coverage for new code.
- New public classes/functions **must** have at least one happy-path and one edge-case test.

### Markers

Use pytest markers to categorise tests:

```python
@pytest.mark.unit
def test_shift_computation():
    ...

@pytest.mark.integration
def test_full_pipeline_with_real_data():
    ...

@pytest.mark.gpu
def test_cuda_inference():
    ...

@pytest.mark.slow
def test_large_mosaic_registration():
    ...
```

---

## Commit Message Format

We follow the **Conventional Commits** specification (`<type>(<scope>): <description>`).

### Types

| Type | When to use |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation-only changes |
| `style` | Formatting, white-space (no logic change) |
| `refactor` | Code restructuring without behaviour change |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `build` | Build system, dependencies |
| `ci` | CI/CD configuration |
| `chore` | Housekeeping, no production code change |
| `revert` | Reverting a previous commit |

### Examples

```
feat(api): add /v1/align endpoint with batch support
fix(geo): correct CRS reprojection when EPSG codes differ
docs(readme): add installation instructions for GPU users
refactor(model): extract attention module into separate class
perf(loader): use dask lazy loading to reduce memory footprint
test(align): add edge cases for near-zero shift images
ci(github): add GPU smoke test job on self-hosted runner
```

### Breaking changes

Add `BREAKING CHANGE:` in the footer or append `!` to the type:

```
feat(api)!: rename /align to /v1/register

BREAKING CHANGE: The /align endpoint has been removed.
Clients must update to /v1/register.
```

---

## Pull Request Process

1. **Title** must follow the Conventional Commits format.
2. **Description** must include:
   - What problem this PR solves.
   - How it was implemented (key design decisions).
   - How to test it manually.
   - Link to the related issue (`Closes #123`).
3. Ensure **all CI checks pass** (lint, type-check, tests).
4. Add / update **documentation** if you changed public APIs.
5. Request review from at least **one maintainer**.
6. Keep PRs **focused**: one logical change per PR.
7. Squash commits only if the maintainer requests it; we generally prefer a clean merge commit.

### PR Checklist

```markdown
- [ ] Tests added / updated
- [ ] Documentation updated
- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] CHANGELOG.md entry added (under [Unreleased])
- [ ] No secrets or large binary files committed
```

---

## Release Process

Releases are managed by maintainers:

1. Update `CHANGELOG.md` — move items from `[Unreleased]` to the new version section.
2. Bump version in `pyproject.toml`.
3. Tag the release: `git tag -s v0.x.y -m "Release v0.x.y"`.
4. Push the tag; the CI pipeline publishes to PyPI automatically.

---

*Thank you for making Titan-GeoAlign better! 🛰️*
