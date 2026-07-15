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


# The canonical restart-variant input mutations: restart=true, one-shot,
# freq-debug writers off (the kij_stream accumulation mode crashes on the
# debug writers' None-Σ_c handling, so the baseline all dynamic variants
# diff against uses the same debug-off config).
GNPPM_RESTART_MUTATIONS = {
    "restart = false": "restart = true",
    "sigma_freq_debug_output = true": "sigma_freq_debug_output = false",
    "sigma_debug_split_contrib = true": "sigma_debug_split_contrib = false",
}


@pytest.fixture(scope="session")
def gnppm_variant_bundle(gnppm_session, tmp_path_factory):
    """Every Tier-2 gnppm variant, run in ONE subprocess.

    Each variant used to be its own ``run_gw_jax`` launch (8-14 s of which
    only 2-9 s is compute — the rest is import + retrace, grid-independent).
    ``tests/run_variant_bundle.py`` amortizes that across the bundle.  Run
    dirs are prepared here exactly as the per-test launches prepared them;
    per-variant env knobs are applied (and restored) around each in-process
    run.  A variant failure does NOT fail this fixture — each test fails on
    its own variant via its status entry.
    """
    root = tmp_path_factory.mktemp("gnppm_variants")

    def prep(name, *, input_name="gnppm_test.in", restart_state=True,
             mutations=None, env=None):
        run_dir = harness.copy_fixture(
            harness.REG / "gnppm_debug", root / name,
            tmp_from=gnppm_session.run_dir if restart_state else None)
        if mutations:
            harness.mutate_input(run_dir / input_name, mutations)
        return {"name": name, "run_dir": run_dir,
                "input_name": input_name, "env": env or {}}

    base = GNPPM_RESTART_MUTATIONS
    variants = [
        prep("baseline", mutations=base),
        prep("pad12", mutations=base,
             env={"LORRAX_EXTRA_MU_PAD": "12"}),
        prep("kij_stream", mutations={
            **base,
            "sigma_omega_h5_file = sigma_mnk.h5":
                "sigma_omega_accumulation = kij_stream\n"
                "sigma_kij_h5_file = sigma_kij_stream.h5\n"
                "sigma_omega_h5_file = sigma_mnk.h5",
        }),
        prep("sc_iter1", mutations={
            **base,
            "qp_solver = one_shot_dft":
                "qp_solver = self_consistent\nsc_max_iter = 1",
        }),
        prep("fixed_point", mutations={
            **base,
            "qp_solver = one_shot_dft": "qp_solver = fixed_point"}),
        # IBZ-equivalence legs (static COHSEX input in the same fixture):
        # leg A restarts through the IBZ cascade; leg B is a FRESH full
        # pipeline forced full-BZ-direct (covers ζ writes + V_q unfold).
        prep("ibz_a", input_name="cohsex_ibz_test.in",
             mutations={"restart = false": "restart = true"},
             env={"LORRAX_FORCE_FULL_BZ": "0"}),
        prep("ibz_b", input_name="cohsex_ibz_test.in", restart_state=False,
             env={"LORRAX_FORCE_FULL_BZ": "1"}),
    ]
    results = harness.run_variant_bundle(variants, root)
    return {
        v["name"]: _NS(run_dir=_Path(v["run_dir"]),
                       input_name=v["input_name"],
                       status=results[v["name"]][0],
                       stdout=results[v["name"]][1],
                       session=gnppm_session)
        for v in variants
    }


@pytest.fixture(scope="session")
def gnppm_restart_baseline(gnppm_variant_bundle, gnppm_session):
    """Canonical restart=true variant (bundle member 'baseline')."""
    b = gnppm_variant_bundle["baseline"]
    if b.status != "ok":
        pytest.fail(f"gnppm restart baseline failed in bundle "
                    f"({b.status}).\nstdout:\n{b.stdout}")
    return _NS(run_dir=b.run_dir, input_name="gnppm_test.in",
               output_name=gnppm_session.output_name, stdout=b.stdout,
               session=gnppm_session)


@pytest.fixture(scope="session")
def bispinor_session(tmp_path_factory):
    """Fresh run of the bispinor GN-PPM fixture; Tier-1 state."""
    return _run_session_case(
        tmp_path_factory, "bispinor_debug", "bispinor_test.in",
        "sigma_diag_bispinor_test.dat")
