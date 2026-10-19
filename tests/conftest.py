"""Seal the first-party source closure before any test imports a service.

A test body that imports ``gw.*`` first would resolve ``distrib_la`` (and its
peers) from whatever the interpreter installs, the sealed release under
``lx test``; a later ``file_io`` import in the same pytest worker then refuses
the mix (``SourceClosureError``).  The drivers seal at start-up; this does the
same once per test process (``runtime.source_closure``, idempotent).
"""
from runtime.source_closure import ensure_source_closure

ensure_source_closure()


import os

import pytest


@pytest.fixture(autouse=True)
def _thaw_env_freeze_after_each_test():
    """Each test starts with an unfrozen environment and leaves os.environ as it found it.

    ``runtime.env_guard`` freezes the process environment once the runtime starts its
    native helper threads (production arming is unchanged); in one pytest process that
    freeze would otherwise carry into every later test (GATE env-frozen on a later
    ``set_default_env``).  ``tests/test_env_guard.py`` arms and checks the freeze
    inside its own tests.
    """
    saved = dict(os.environ)
    yield
    from runtime import env_guard
    env_guard.thaw_for_tests()
    for key in [k for k in os.environ if k not in saved]:
        del os.environ[key]
    for key, value in saved.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _isolated_memory_model():
    """lxkit.memory's run figures and records are process state: no test leaks them."""
    from lxkit import memory
    yield
    memory.reset(run=True)
