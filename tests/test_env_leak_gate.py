"""THE GATE: no test module reconfigures the session at collection time.

P19 (2026-08-09).  ``tests/test_contract_bands.py`` used to carry a
module-scope ``os.environ.setdefault("LORRAX_BANDS_GEMM_FFI", "0")``.
Module scope runs at COLLECTION time -- pytest imports every selected
module into one process before running anything -- and a plain ``os.environ``
write never unwinds, so that one line reconfigured the entire session.

The measured cost landed on a different file.
``test_bse_coupling_routes_mesh_invariance.py`` -- the gate that closed the
K^d_B zeta-sharding CLASS -- calls ``build_bse_ring_matvec_full`` ->
``contract_bands_block_reshard`` -> ``gate.require``, which refuses on a box
with no mklblas host handler unless the dial announces the debug opt-out.
On an FFI-less box, same tree, same commit:

    pytest tests/test_bse_coupling_routes_mesh_invariance.py -> 13F / 2P
    pytest <that file> tests/test_contract_bands.py           -> 15P

A verdict that reverses with collection scope, on the one gate whose whole
argument is "a revert fails this on the NUMBER".  A reader who met 13 reds
on a laptop would have concluded the K^d_B fix regressed.

This file gates the CLASS rather than that instance:

  * :func:`test_the_current_suite_is_collection_inert` is the live gate --
    it collects the whole default census in a child process and asserts the
    conftest hook stays silent;
  * :func:`test_a_leaking_module_is_caught` is the RED TWIN -- it runs a
    module that leaks on purpose (``tests/_env_leak_twin.py``) and asserts
    the session REFUSES.  Without it, "the check never fired" and "the
    check cannot fire" are the same observation;
  * the rest pin the pure decision function's boundaries directly.

The check's allowlist (``harness.IMPORT_TIME_ENV_PREFIXES``) is deliberately
narrow and turns on WHEN A VALUE IS READ, not on who owns the name: jax/XLA
latch ``JAX_*``/``XLA_*`` at ``import jax``, so no fixture can be early
enough and module scope is the only option; ``LORRAX_*`` dials are read live
per call (``ffi.gate.Gate.enabled``), so a fixture serves them exactly.

Evidence: ~/lorrax_bse_perf_2026-08-08/FIX_p19_env_leak.md
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import harness

_TESTS = pathlib.Path(__file__).resolve().parent
_TWIN = _TESTS / "_env_leak_twin.py"


def _child_pytest(*args: str,
                  timeout: int = 900) -> subprocess.CompletedProcess:
    """Run pytest in a child process rooted at the repo, with this conftest.

    A child is the only honest way to test a collection hook: the hook has
    already fired for the session we are running inside.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    # Do not inherit an xdist worker id into the child.
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=str(_TESTS.parent), env=env, capture_output=True, text=True,
        timeout=timeout)


# ===========================================================================
#  THE RED TWIN -- the check is observed to fire
# ===========================================================================
def test_a_leaking_module_is_caught():
    """A module that writes os.environ at import must REFUSE the session."""
    assert _TWIN.exists(), (
        f"the red twin {_TWIN} is missing -- this gate cannot demonstrate "
        "that the check fires, which is the only thing it is for")
    proc = _child_pytest(str(_TWIN.relative_to(_TESTS.parent)), "-q")
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "the deliberately-leaking twin was NOT caught -- the collection-time "
        f"environment check is inert:\n{out[-3000:]}")
    assert "MUTATED os.environ AT COLLECTION TIME" in out, (
        f"the twin failed, but not for the reason under test:\n{out[-3000:]}")
    assert "LX_ENV_LEAK_TWIN" in out, (
        f"the refusal did not NAME the leaked variable:\n{out[-3000:]}")


def test_the_twin_fails_before_running_anything():
    """The refusal lands at COLLECTION, not after the damage is done.

    The twin's own body would PASS (it asserts the value it leaked), so a
    check that fired late would report a green test inside a red session.
    Catching it at collection is what makes the diagnosis unambiguous.
    """
    proc = _child_pytest(str(_TWIN.relative_to(_TESTS.parent)), "-q")
    out = proc.stdout + proc.stderr
    assert "test_twin_body_runs" not in out.replace(
        str(_TWIN), ""), (
        "the twin's test body was reported, so the check fired after "
        f"collection rather than at the end of it:\n{out[-3000:]}")


# ===========================================================================
#  THE LIVE GATE -- the tree as it stands is collection-inert
# ===========================================================================
def test_the_current_suite_is_collection_inert():
    """Collect the whole default census; the hook must stay silent.

    ``--collect-only`` imports every selected module (that IS collection)
    and runs no test, so this is the full leak surface at a few seconds'
    cost.  A new module-scope ``os.environ`` write anywhere under ``tests/``
    or ``services/`` turns this red at the commit that introduces it,
    instead of turning some unrelated gate red three arrangements later.

    ``--census`` because the leak SURFACE is the census: a bare ``pytest``
    is the fast default gate since 2026-08-09 and imports a few dozen
    modules, so collecting without it would shrink exactly the thing this
    cell is measuring.
    """
    proc = _child_pytest("--census", "--collect-only", "-q")
    out = proc.stdout + proc.stderr
    assert "MUTATED os.environ AT COLLECTION TIME" not in out, (
        "a test module now reconfigures the session at collection time:\n"
        + out[-4000:])
    assert proc.returncode == 0, f"collection itself failed:\n{out[-4000:]}"


# ===========================================================================
#  The pure decision function's boundaries
# ===========================================================================
def test_import_time_knobs_are_allowed():
    """jax/XLA knobs are latched at import, so module scope is their only
    home and the check must not police them."""
    for name in ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS",
                 "CUDA_VISIBLE_DEVICES"):
        assert harness.env_collection_offenders({}, {name: "1"}) == [], (
            f"{name} is import-time-latched; flagging it would force a "
            "fixture that cannot possibly be early enough")


def test_the_irreversible_sentinel_is_allowed_for_its_own_reason():
    """``_LORRAX_JAX_DISTRIBUTED_DONE`` is exempt on a DIFFERENT ground than
    the prefixes: not "read at import" but "guards something with no
    teardown".  ``jax.distributed.initialize`` runs once per process, so a
    fixture that unwound its sentinel would re-arm a double-init.  Pinned as
    an exact name, never a ``_LORRAX_`` prefix -- the other LORRAX_ dials
    are exactly what this check exists to catch."""
    assert harness.env_collection_offenders(
        {}, {"_LORRAX_JAX_DISTRIBUTED_DONE": "1"}) == []
    assert harness.env_collection_offenders(
        {}, {"_LORRAX_SOMETHING_ELSE": "1"}) == [
            ("_LORRAX_SOMETHING_ELSE", None, "1")]


def test_a_call_time_dial_is_not_allowed():
    """LORRAX_* dials are read live per call, so a fixture serves them."""
    assert harness.env_collection_offenders(
        {}, {"LORRAX_BANDS_GEMM_FFI": "0"}) == [
            ("LORRAX_BANDS_GEMM_FFI", None, "0")]


def test_changes_and_deletions_both_count():
    """A module that OVERWRITES or DELETES a value leaks just as hard as one
    that sets a new one -- all three reconfigure the rest of the session."""
    assert harness.env_collection_offenders({"A": "1"}, {"A": "2"}) == [
        ("A", "1", "2")]
    assert harness.env_collection_offenders({"A": "1"}, {}) == [
        ("A", "1", None)]
    assert harness.env_collection_offenders({"A": "1"}, {"A": "1"}) == []


def test_the_report_names_the_variable_and_the_fix():
    """A refusal that does not say what to do is a flake with extra steps."""
    text = harness.format_env_leak_report(
        [("LORRAX_BANDS_GEMM_FFI", None, "0")])
    assert "LORRAX_BANDS_GEMM_FFI" in text
    assert "monkeypatch.setenv" in text
    assert "IMPORT_TIME_ENV_PREFIXES" in text
