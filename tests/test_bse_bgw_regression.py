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
    # Column 0 is the excitation energy in eV.  BerkeleyGW writes this file in
    # BSE/absp_io.f90:120-123 -- under TDA, `write(14,'(4e16.8)') evals(ii),
    # cs(ii,ipol), dipoles_r(ii,ipol)` -- with evals converted to eV one step
    # earlier at BSE/diag.f90:766-767.  The four-column header this fixture
    # carries is the CPLX+TDA branch (absp_io.f90:106-108); the non-TDA branch
    # writes six.
    #
    # The rows are the algebraically LOWEST `neig` eigenvalues in ascending
    # order, on both solver paths.  Serial: BSE/diagonalize.f90:112-118 picks
    # range='I' whenever neig < nmat, and lines 261-262 set ilow=1, iup=neig.
    # ScaLAPACK, which is what actually produced this fixture (the run log
    # says "Using diagonalization routine: PZHEEVX"): BSE/diagonalize.f90:807
    # calls diagonalize_scalapack, and Common/scalapack_hl.f90:369-373 sets
    # range_char='I' for Neig/=N while the call at :445-451 passes il=1,
    # iu=Neig.  So this column is directly comparable to LORRAX's lowest-N
    # vector, with no offset and no reordering.
    return np.loadtxt(path, comments="#")[:, 0]


def _assert_aligned(got, ref):
    """Fail loudly when the two lists are not describing the same states.

    Both eigenvalue lists are compared BY INDEX: state i of LORRAX against
    state i of BerkeleyGW.  That is the right comparison only while both
    codes resolve the same states in the same order.  LORRAX takes its
    spectrum straight off the Lanczos tridiagonal -- `argsort(evals_T)[:n_eig]`
    in src/solvers/lanczos.py, with no ghost filtering -- so one spurious
    repeated Ritz value inserts an extra entry and shifts every state after
    it by one.  Index matching then compares state i to state i-1 and the
    result reads as a physics disagreement rather than as the bookkeeping
    accident it is.

    MEASURED on this fixture: injecting one duplicate Ritz value anywhere in
    the lowest 16 raises max |delta| from 8.9 meV to 69.6 meV, which breaches
    the band below and would otherwise be reported as "LORRAX disagrees with
    BerkeleyGW".  Dropping the offending entry recovers a normal match, and
    that asymmetry is the signature this check keys on.  It does not change
    which runs pass -- it changes what a failure is called.
    """
    n = len(ref)
    base = np.abs(got[:n] - ref)[:n - 1].mean()
    drops = [np.abs(np.delete(got[:n], k) - ref[:n - 1]).mean()
             for k in range(n)]
    best_k = int(np.argmin(drops))
    if drops[best_k] < 0.5 * base:
        raise AssertionError(
            "BSE eigenvalue lists are misaligned, not merely in disagreement: "
            f"index matching gives MAE {base * 1e3:.3f} meV, but dropping "
            f"LORRAX entry {best_k} (E = {got[best_k]:.6f} eV) gives "
            f"{drops[best_k] * 1e3:.3f} meV.  That is the signature of a "
            "spurious repeated Lanczos Ritz value shifting the list, not of "
            "a kernel or screening error.  Check the Lanczos iteration count "
            "before reading the delta below as physics.")


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
    #    drift shows how far outside the band it landed.  The alignment
    #    check runs first: an index-matching failure and a physics failure
    #    look identical in the numbers below, and only one of them is about
    #    the physics.
    _assert_aligned(got, bgw)
    delta = np.abs(got - bgw)
    mae, worst = float(delta.mean()), float(delta.max())
    assert mae <= BGW_MAE_BAND_EV and worst <= BGW_MAX_BAND_EV, (
        f"BSE agreement with BerkeleyGW left its band: "
        f"MAE {mae * 1e3:.3f} meV (band {BGW_MAE_BAND_EV * 1e3:.1f}), "
        f"max {worst * 1e3:.3f} meV (band {BGW_MAX_BAND_EV * 1e3:.1f})")
