"""Shared pytest setup for the LORRAX_A test suite.

JAX must be configured for x64 BEFORE the first ``import jax`` in the
process, otherwise ``jnp.complex128`` silently degrades to complex64
(see jax-ml/jax#current-gotchas).  Pytest collects all test modules
into one process, so the first import wins — set the env here.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

# ---------------------------------------------------------------------------
# pytest-xdist: pin each worker to its own GPU (gw0 → GPU 0, gw1 → GPU 1, …)
# so the e2e regression gates — subprocess launchers that each need ONE
# GPU — run N-wide on an N-GPU node instead of serially on GPU 0.  Must
# run before the worker's first CUDA/JAX init, which is why it lives at
# conftest module scope.  This OVERRIDES any pre-set CUDA_VISIBLE_DEVICES
# (SLURM gres sets "0,1,2,3" for the task): without the override each
# worker — and every gate subprocess it launches — sees all N GPUs and
# runs the gate on an N-device mesh, which breaks the 1-GPU-frozen
# references.  Mapping goes through the existing list so SLURM's device
# selection is respected.  No-op without xdist.
# ---------------------------------------------------------------------------
_wid = os.environ.get("PYTEST_XDIST_WORKER", "")
if _wid.startswith("gw"):
    _preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _preset:
        _devs = [d for d in _preset.split(",") if d != ""]
    else:
        try:
            import subprocess as _sp
            _n = len(_sp.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True,
                timeout=10).stdout.strip().splitlines())
        except Exception:
            _n = 0
        _devs = [str(i) for i in range(_n)]
    if _devs:
        os.environ["CUDA_VISIBLE_DEVICES"] = _devs[int(_wid[2:]) % len(_devs)]


# ---------------------------------------------------------------------------
# Session-scoped e2e states (the Tier-1 gates double as prepared state for
# the Tier-2 invariance gates).
#
# ``gnppm_session`` runs the shrunk MoS2 3×3 GN-PPM fixture ONCE, fresh
# (restart = false), and keeps the run dir — including ``tmp/`` with the
# ISDF restart file (isdf_tensors_*.h5) and zeta_q.h5.  The Tier-1 frozen
# gate asserts on this run's outputs; every Tier-2 variant re-runs the
# driver with ``restart = true`` from a COPY of the state (the ζ-fit and
# V_q build — the dominant cost — are not redone).  Copies are mandatory:
# the driver writes W0_qmunu + head scalars back into the restart file
# (gw_output.persist_w0_and_head).
#
# ``gnppm_restart_baseline`` is the canonical restart variant (one-shot,
# freq-debug writers off — the config every other dynamic variant diffs
# against); its equality with the fresh session run IS the
# restart-roundtrip gate.
#
# ``bispinor_session`` is the fresh bispinor GN-PPM run (bispinor restart
# is not yet supported — gw_init.py marks the transverse bundle
# not-restartable — so its Tier-2 pad-flip gate reruns fresh).
#
# Under pytest-xdist each worker builds its own session state (session
# fixtures are per-process); tests stay order-independent and xdist-safe
# because no test mutates a session dir — every variant copies first.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
from types import SimpleNamespace as _NS

import pytest

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import harness  # noqa: E402


def _run_session_case(tmp_path_factory, case_name, input_name, output_name):
    import pytest as _pytest
    harness.skip_unless_gpu(_pytest)
    case_dir = harness.REG / case_name
    run_dir = harness.copy_fixture(
        case_dir, tmp_path_factory.mktemp(f"{case_name}_session") / case_name)
    res = harness.run_gw_jax(run_dir, input_name)
    if res.returncode != 0:
        _pytest.fail(
            f"{case_name} session run failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    out = run_dir / output_name
    assert out.exists(), f"session run wrote no {out}"
    return _NS(run_dir=run_dir, input_name=input_name,
               output_name=output_name, stdout=res.stdout)


@pytest.fixture(scope="session")
def gnppm_session(tmp_path_factory):
    """Fresh (restart=false) run of the gnppm fixture; Tier-1 state."""
    return _run_session_case(
        tmp_path_factory, "gnppm_debug", "gnppm_test.in",
        "sigma_diag_gnppm_test.dat")


@pytest.fixture(scope="session")
def gnppm_restart_baseline(gnppm_session, tmp_path_factory):
    """Canonical restart=true variant of the gnppm session state.

    One-shot, freq-debug writers off (the kij_stream accumulation mode
    crashes on the debug writers' None-Σ_c handling, so the baseline all
    dynamic variants diff against uses the same debug-off config).
    """
    run_dir = harness.copy_fixture(
        harness.REG / "gnppm_debug",
        tmp_path_factory.mktemp("gnppm_restart") / "baseline",
        tmp_from=gnppm_session.run_dir)
    harness.mutate_input(run_dir / "gnppm_test.in", {
        "restart = false": "restart = true",
        "sigma_freq_debug_output = true": "sigma_freq_debug_output = false",
        "sigma_debug_split_contrib = true": "sigma_debug_split_contrib = false",
    })
    res = harness.run_gw_jax(run_dir, "gnppm_test.in")
    if res.returncode != 0:
        pytest.fail(
            f"gnppm restart baseline failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    return _NS(run_dir=run_dir, input_name="gnppm_test.in",
               output_name=gnppm_session.output_name, stdout=res.stdout,
               session=gnppm_session)


@pytest.fixture(scope="session")
def bispinor_session(tmp_path_factory):
    """Fresh run of the bispinor GN-PPM fixture; Tier-1 state."""
    return _run_session_case(
        tmp_path_factory, "bispinor_debug", "bispinor_test.in",
        "sigma_diag_bispinor_test.dat")
