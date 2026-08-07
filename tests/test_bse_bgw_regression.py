"""Tier-1 frozen BSE gate — the one cross-code BSE anchor.

Runs the full BSE pipeline (ζ-fit → V_q → χ₀ → W → ISDF tensor persist →
BSE Hamiltonian → Lanczos) on the bulk Si fixture and checks the lowest
exciton eigenvalues two ways:

* against a frozen LORRAX reference, which detects any regression in the
  kernel assembly or the solver, and
* against an external BerkeleyGW reference, which keeps the gate honest
  as physics rather than a self-freeze.

The BGW check is stated as a BAND, not a point.  LORRAX does not use
BerkeleyGW's head-and-wing treatment for the BSE kernel, so exact
agreement is neither expected nor required; the band is set from the
measured agreement with headroom, and is documented in the fixture
README together with the number it was measured at.

The two checks have deliberately different tolerances.  The frozen
reference is a bit-reproducibility pin (two independent runs agree
exactly), so it is tight.  The BGW band is a physics bound and is loose
by construction.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (          # noqa: E402
    REG,
    REPO_ROOT,
    copy_fixture,
    requested_platform,
    skip_unless_gpu,
)

CASE_DIR = REG / "si_bse_debug"
INPUT_NAME = "bse_si_test.in"

# Solver settings are part of the pinned configuration: the Lanczos
# spectrum is only converged over a window of iteration counts, so the
# count is fixed here rather than left to a default.
N_VAL, N_COND, N_OCC = 4, 4, 8
N_ITER, N_EIG = 200, 20

# Frozen-reference tolerance: the reference is bit-reproducible across
# runs, so this only absorbs last-ULP GPU nondeterminism.
ATOL_FROZEN_EV = 1e-6

# External BerkeleyGW band.  See the fixture README for the measured
# values these are derived from.
BGW_MAE_BAND_EV = 10e-3
BGW_MAX_BAND_EV = 25e-3


def _run(module_args, run_dir, timeout=1800):
    import os
    env = os.environ.copy()
    platform = requested_platform()
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
    elif platform in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda,cpu"
        env["JAX_PLATFORM_NAME"] = "gpu"
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-u", "-m"] + module_args,
        cwd=run_dir, env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def _parse_eigenvalues(stdout):
    """Pull the printed lowest-N eigenvalue vector (eV) out of a BSE run."""
    match = re.search(
        r"Lowest \d+ eigenvalues \(eV\): \[(.*?)\]", stdout, re.S)
    assert match, "BSE run printed no eigenvalue vector"
    return np.array([float(tok) for tok in match.group(1).split()])


def _read_reference(path):
    return np.loadtxt(path, comments="#")[:, 1]


def _read_bgw(path):
    return np.loadtxt(path, comments="#")[:, 0]


@pytest.mark.regression
def test_bse_matches_frozen_and_bgw(tmp_path):
    """Si BSE: frozen LORRAX pin + external BerkeleyGW band."""
    skip_unless_gpu(pytest)
    run_dir = copy_fixture(CASE_DIR, tmp_path / CASE_DIR.name)

    gw = _run(["gw.gw_jax", "-i", INPUT_NAME], run_dir)
    if gw.returncode != 0:
        pytest.fail(f"GW stage failed.\nstdout:\n{gw.stdout}\n"
                    f"stderr:\n{gw.stderr}")
    # The BSE direct term reads the screened interaction back from the
    # persisted tensor file; a zero-filled placeholder there silently
    # removes the term instead of failing, so assert it was written.
    assert "Persisted W0_qmunu" in gw.stdout, (
        "GW stage did not persist W0_qmunu — the BSE direct term would "
        "be built from a placeholder")

    bse = _run([
        "bse.bse_jax", "-i", INPUT_NAME,
        "--bse", "--lanczos", "--tda", "--matvec-kind=ring",
        "--n-val", str(N_VAL), "--n-cond", str(N_COND), "--n-occ", str(N_OCC),
        "--n-reorth", "-1",
        "--max-lanczos-iter", str(N_ITER), "--n-eig", str(N_EIG),
        "--px", "2", "--py", "2",
    ], run_dir)
    if bse.returncode != 0:
        pytest.fail(f"BSE stage failed.\nstdout:\n{bse.stdout}\n"
                    f"stderr:\n{bse.stderr}")

    got = _parse_eigenvalues(bse.stdout)
    frozen = _read_reference(CASE_DIR / "bse_eigenvalues_ref.dat")
    bgw = _read_bgw(CASE_DIR / "bgw_eigenvalues_dft_ref.dat")[:len(frozen)]
    assert len(got) >= len(frozen)
    got = got[:len(frozen)]

    # 1. Regression pin against the frozen LORRAX reference.
    np.testing.assert_allclose(
        got, frozen, rtol=0.0, atol=ATOL_FROZEN_EV,
        err_msg="BSE eigenvalues drifted from the frozen LORRAX reference")

    # 2. External band against BerkeleyGW.  Reported on failure so a
    #    drift shows how far outside the band it landed.
    delta = np.abs(got - bgw)
    mae, worst = float(delta.mean()), float(delta.max())
    assert mae <= BGW_MAE_BAND_EV and worst <= BGW_MAX_BAND_EV, (
        f"BSE agreement with BerkeleyGW left its band: "
        f"MAE {mae * 1e3:.3f} meV (band {BGW_MAE_BAND_EV * 1e3:.1f}), "
        f"max {worst * 1e3:.3f} meV (band {BGW_MAX_BAND_EV * 1e3:.1f})")
