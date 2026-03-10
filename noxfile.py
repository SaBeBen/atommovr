# noxfile.py
# ==============================
# This file defines the CI/test pipeline for atommovr.
# It is the single source of truth for:
#   - formatting checks (Black)
#   - linting / bug checks (Ruff)
#   - running tests + coverage (pytest)
# It can be run locally (`nox`) or remotely via GitHub Actions.
# ==============================

import nox

# ------------------------------
# FORMAT SESSION
# ------------------------------
@nox.session
def format(session):
    """Check that code is properly formatted with Black."""
    # Install Black
    session.install("black")
    # Run Black in "check" mode (will fail if code is not formatted)
    # Adjust paths as needed (src contains package, tests contain test code)
    session.run("black", "--check", "src", "tests")

# ------------------------------
# LINT SESSION
# ------------------------------
@nox.session
def lint(session):
    """Run Ruff linter to catch potential bugs or bad patterns."""
    session.install("ruff")
    # Run Ruff on package and tests directories
    session.run("ruff", "check", "src", "tests")

# ------------------------------
# TESTS SESSION
# ------------------------------
@nox.session
def tests(session):
    """Run pytest with coverage reporting."""
    # Install the package itself, pytest, and pytest-cov for coverage
    session.install("-e", ".", "pytest", "pytest-cov")
    # Run pytest; coverage options are defined in pyproject.toml
    session.run("pytest")

# ------------------------------
# USAGE
# ------------------------------
# Local:
#   nox -s format    # check formatting
#   nox -s lint      # run linter
#   nox -s tests     # run tests with coverage
#   nox              # run all sessions sequentially
#
# GitHub Actions:
#   Calls `nox` directly; ensures PRs cannot merge unless all sessions pass.