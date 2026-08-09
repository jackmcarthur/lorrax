"""RED TWIN for the collection-time environment check (P19, 2026-08-09).

This module LEAKS ON PURPOSE.  It is the falsifying case for the conftest
hook pair ``pytest_collection`` / ``pytest_collection_finish``: without a
twin, "no test module mutates the environment at collection time" is a
check that has never been observed to fire, which is indistinguishable from
a check that cannot fire.

It is never collected by the default census -- the filename does not match
``python_files`` (``test_*.py``), so pytest walks past it -- and it is
reached only when ``tests/test_env_leak_gate.py`` names it explicitly on a
child pytest's command line.  The variable it sets is namespaced and read
by nothing, so even the deliberate leak is inert.

Do not "fix" this file.  Deleting the leak below turns the gate green for
the wrong reason; ``test_env_leak_gate.py`` asserts that this module makes
a session REFUSE.
"""
import os

#: The leak.  Module scope == collection time: pytest imports this before it
#: runs anything, and nothing unwinds it.
os.environ["LX_ENV_LEAK_TWIN"] = "1"


def test_twin_body_runs():
    """A body, so the twin is a real collected test and not an empty file.

    It never actually runs: the session refuses at the end of collection,
    before any test executes.  That ordering is itself part of the claim --
    the check catches the leak at COLLECTION, not after the damage.
    """
    assert os.environ.get("LX_ENV_LEAK_TWIN") == "1"
