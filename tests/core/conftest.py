"""Small shared fixtures for the cached core family."""
from pathlib import Path

import pytest


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def core_fixtures() -> Path:
    return FIXTURES


def pytest_configure(config):
    """Keep each pytest's destructive basetemp initialization rank-private."""
    from core.rank_session import _resolve_proc_count, _resolve_proc_id
    if _resolve_proc_count() > 1 and config.option.basetemp:
        config.option.basetemp = f"{config.option.basetemp}-rank{_resolve_proc_id()}"
