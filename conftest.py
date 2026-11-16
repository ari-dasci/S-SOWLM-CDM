"""Run the tests from the repository root: they read results by relative path."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _run_from_repo_root():
    previous = Path.cwd()
    os.chdir(Path(__file__).parent)
    yield
    os.chdir(previous)
