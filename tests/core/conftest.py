"""Small shared fixtures for the cached core family."""
from pathlib import Path

import pytest


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def core_fixtures() -> Path:
    return FIXTURES
