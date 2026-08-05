"""Tier-2 invariance gates — two paths that must agree, run FROM PREPARED STATE.

Every major 2026 bug was a divergence between two code paths that must
produce the same physics (TRS-blind V_q unfold, stream-accumulator
dropped head, pad-extent LU, SC-seam rotation).  These gates check those
invariances directly — self-checking, no frozen references — and they
run cheaply: each is a ``restart = true`` variant launched from a COPY
of the Tier-1 gnppm session state (``conftest.gnppm_session``), so the
dominant ζ-fit + V_q build happens once per suite, not once per gate.

Copies are mandatory: the driver mutates the restart file in place
(``persist_w0_and_head``).  Ports of the pre-redesign standalone gates —
the parenthesised names are the FORMER files, none of which still exist;
each gate now lives in this one:

* restart ≡ fresh          (NEW — was implicit, never gated)
* μ-pad flip, gnppm        (test_mu_pad_invariance.py::gnppm_pad12)
* μ-pad flip, bispinor     (test_mu_pad_invariance.py::bispinor_pad4;
                            fresh runs — bispinor restart unsupported)
* SC-iter-1 ≡ one-shot     (test_sc_oneshot_equivalence.py)
* fixed-point rotations    (test_gw_jax_regression.py::
                            test_gnppm_fixed_point_reproduces_frozen_qp_rotations)
* IBZ ≡ full-BZ            (test_gw_jax_regression.py::
                            test_ibz_full_bz_equivalence — the IBZ leg now
                            runs from restart; the full-BZ leg stays a FULL
                            fresh pipeline so a wrong V_q build/unfold on
                            either side still shows up in the comparison)
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (          # noqa: E402
    REG,
    census_lines,
    copy_fixture,
    eqp_column,
    mutate_input,
    normalize_dat,
    numeric_tokens,
    parse_eqp_rows,
    run_gw_jax,
)

_GNPPM = REG / "gnppm_debug"
_LABELS = ("sigX", "sigC", "sigXC")
_OUT = "sigma_diag_gnppm_test.dat"

# The canonical restart-variant input mutations (must match
# conftest.gnppm_restart_baseline so variants diff against it cleanly).
_RESTART_MUTATIONS = {
    "restart = false": "restart = true",
    "sigma_freq_debug_output = true": "sigma_freq_debug_output = false",
    "sigma_debug_split_contrib = true": "sigma_debug_split_contrib = false",
}


def _run_restart_variant(tmp_path, session, name, *, extra_mutations=None,
                         extra_env=None, input_name="gnppm_test.in"):
    """Copy fixture + session restart state, mutate the input, run."""
    run_dir = copy_fixture(_GNPPM, tmp_path / name,
                           tmp_from=session.run_dir)
    mutations = dict(_RESTART_MUTATIONS)
    if extra_mutations:
        mutations.update(extra_mutations)
    mutate_input(run_dir / input_name, mutations)
    res = run_gw_jax(run_dir, input_name, extra_env=extra_env)
    if res.returncode != 0:
        pytest.fail(f"{name} run failed.\n"
                    f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    return run_dir, res.stdout


def _assert_rows_close(dir_a, dir_b, out_name, labels, atol, msg):
    a = parse_eqp_rows(dir_a / out_name, labels)
    b = parse_eqp_rows(dir_b / out_name, labels)
    assert a.shape == b.shape, f"{msg}: row-count mismatch {a.shape} vs {b.shape}"
    np.testing.assert_allclose(b[:, :6], a[:, :6], rtol=0.0, atol=atol,
                               err_msg=msg)


# ===========================================================================
#  restart ≡ fresh — the restart-roundtrip gate
# ===========================================================================

@pytest.mark.regression
def test_restart_equals_fresh(gnppm_restart_baseline):
    """restart=true from the fresh run's tmp/ must reproduce the fresh run.

    The baseline additionally flips the freq-debug writers off, so this
    also pins that those writers don't perturb Σ.  Measured bit-identical;
    atol=1e-6 absorbs GPU-nondeterministic last-ULP drift only.
    """
    base = gnppm_restart_baseline
    assert "Loaded restart tensors from H5." in base.stdout
    _assert_rows_close(base.session.run_dir, base.run_dir, _OUT, _LABELS,
                       1e-6, "restart Σ differs from fresh")
    for eqp in ("eqp0.dat", "eqp1.dat"):
        ta = numeric_tokens(base.session.run_dir / eqp)
        tb = numeric_tokens(base.run_dir / eqp)
        assert ta.shape == tb.shape, f"{eqp}: token count changed on restart"
        np.testing.assert_allclose(tb, ta, rtol=0.0, atol=1e-6,
                                   err_msg=f"restart {eqp} differs from fresh")


# ===========================================================================
#  μ-pad-extent invariance at fixed device count
#  (root cause: reports/device_invariance_2026-07-08/ROOT_CAUSE.md — solves
#  must run at the LOGICAL μ extent, not the P-padded one)
# ===========================================================================

@pytest.mark.regression
def test_mu_pad_flip_invariance_gnppm(gnppm_restart_baseline, tmp_path):
    """EXTRA_MU_PAD=12 vs 0 at fixed P=1 on the dynamic GN-PPM chain.

    Census / window node counts / invalid count / unfulfilled fraction and
    the Σ_X column: exactly equal.  Σ_C / Σ_XC / eqp: ≤ 2e-4 eV (the
    near-singular PPM fit ratio historically amplified 1-2 ULP ζ/V drift
    to 6.3e-5 eV on the 642-centroid fixture; the shrunk fixture measures
    0.0, but the bound is kept — any REAL pad defect is O(1)).
    """
    base = gnppm_restart_baseline
    pad_dir, pad_stdout = _run_restart_variant(
        tmp_path, base.session, "pad12",
        extra_env={"LORRAX_EXTRA_MU_PAD": "12"})

    assert census_lines(pad_stdout) == census_lines(base.stdout), (
        "PPM census/window signature changed under a pad-extent flip at "
        "fixed P — a census/stat still reads the pad extent.")
    a = parse_eqp_rows(base.run_dir / _OUT, _LABELS)
    b = parse_eqp_rows(pad_dir / _OUT, _LABELS)
    assert a.shape == b.shape
    # columns: kpt, band, sigX, sigC, sigXC, VH_re, VH_im
    np.testing.assert_array_equal(
        a[:, 2], b[:, 2],
        err_msg="Σ_X changed under the pad flip — the ζ/V_q/exchange "
                "chain reads the pad extent.")
    d = float(np.max(np.abs(a[:, 3:6] - b[:, 3:6])))
    assert d <= 2.0e-4, (
        f"Σ_C/Σ_XC moved {d:.3e} eV under the pad flip (> 2e-4 eV bound; "
        f"measured 0.0 on this fixture — this is a new pad defect).")
    for eqp in ("eqp0.dat", "eqp1.dat"):
        ea = eqp_column(base.run_dir / eqp)
        eb = eqp_column(pad_dir / eqp)
        assert ea.shape == eb.shape, f"{eqp} row count changed"
        de = float(np.max(np.abs(ea - eb))) if ea.size else 0.0
        assert de <= 2.0e-4, f"{eqp} moved {de:.3e} eV under the pad flip."


@pytest.mark.regression
def test_mu_pad_flip_invariance_bispinor(bispinor_session, tmp_path):
    """EXTRA_MU_PAD=4 vs 0, bispinor GN-PPM: full BIT-IDENTITY.

    Pins the historically catastrophic transverse pad-extent class
    (MoS2 668→672 moved Σ^B tile(2,2) from −0.15 to −117.9 eV — the
    transverse ζ_T pivoted LU amplified pad-shape roundoff O(1) in the
    near-null indefinite modes).  Fresh runs: bispinor restart is not
    yet supported (gw_init.py).
    """
    ses = bispinor_session
    run_dir = copy_fixture(REG / "bispinor_debug", tmp_path / "bispinor_pad4")
    res = run_gw_jax(run_dir, ses.input_name,
                     extra_env={"LORRAX_EXTRA_MU_PAD": "4"})
    if res.returncode != 0:
        pytest.fail(f"bispinor pad4 run failed.\n"
                    f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    assert census_lines(res.stdout) == census_lines(ses.stdout), (
        "bispinor PPM census changed under the pad flip")
    for out in (ses.output_name, "eqp0.dat", "eqp1.dat"):
        a = normalize_dat((ses.run_dir / out).read_text())
        b = normalize_dat((run_dir / out).read_text())
        assert a == b, (
            f"bispinor pad-extent flip changed {out} at FIXED P — a "
            f"computation still runs on the padded μ extent.")


# ===========================================================================
#  SC iteration 1 ≡ one-shot G0W0 (Phase-C acceptance; caught two real SC
#  bugs: converged-U rotate-back and the eigh-of-diagonal ULP roundtrip)
# ===========================================================================

@pytest.mark.regression
def test_sc_iteration1_equals_one_shot(gnppm_restart_baseline, tmp_path):
    base = gnppm_restart_baseline
    sc_dir, _ = _run_restart_variant(
        tmp_path, base.session, "sc_iter1",
        extra_mutations={
            "qp_solver = one_shot_dft":
                "qp_solver = self_consistent\nsc_max_iter = 1",
        })
    _assert_rows_close(base.run_dir, sc_dir, _OUT, _LABELS, 1e-6,
                       "SC iteration 1 sigma_diag differs from one-shot G0W0")
    for fname in ("eqp0.dat", "eqp1.dat"):
        t_os = numeric_tokens(base.run_dir / fname)
        t_sc = numeric_tokens(sc_dir / fname)
        assert t_os.shape == t_sc.shape, f"{fname}: token-count mismatch"
        np.testing.assert_allclose(
            t_sc, t_os, rtol=0.0, atol=1e-6,
            err_msg=f"SC iteration 1 {fname} differs from one-shot G0W0")


# ===========================================================================
#  fixed_point reproduces the frozen QP rotations
#  (pins that the 2026-07-08 qp_solver default flip was a re-labeling, not
#  a physics change; ref re-frozen same-code with the 2026-07-09 fixture
#  shrink, from this exact from-restart procedure, produced twice)
# ===========================================================================

@pytest.mark.regression
def test_fixed_point_frozen_qp_rotations(gnppm_restart_baseline, tmp_path):
    import h5py

    ref_file = _GNPPM / "eqp_rotations_fixedpoint_ref.npy"
    assert ref_file.exists(), f"missing frozen reference: {ref_file}"
    base = gnppm_restart_baseline
    fp_dir, _ = _run_restart_variant(
        tmp_path, base.session, "fixed_point",
        extra_mutations={"qp_solver = one_shot_dft": "qp_solver = fixed_point"})
    rot_file = fp_dir / "qp_wfn_rotations.h5"
    assert rot_file.exists(), f"missing {rot_file}"
    with h5py.File(rot_file, "r") as f:
        e_qp_ry = np.asarray(f["E_qp_nk_rydberg"])
    e_ref_ry = np.load(ref_file)
    assert e_qp_ry.shape == e_ref_ry.shape, (
        f"shape mismatch: {e_qp_ry.shape} vs frozen {e_ref_ry.shape}")
    np.testing.assert_allclose(e_qp_ry, e_ref_ry, rtol=0.0, atol=1e-6)


# ===========================================================================
#  IBZ cascade ≡ full-BZ-direct (the gate that catches a WRONG symmetry
#  unfold — the TRS-blind-bug class; the frozen gates cannot, because their
#  references were produced with the same unfold)
# ===========================================================================

@pytest.mark.regression
def test_ibz_equals_full_bz(gnppm_session, tmp_path):
    """Static COHSEX, IBZ cascade vs LORRAX_FORCE_FULL_BZ=1.

    The IBZ leg runs from the session restart state (its V_q was built
    through the IBZ cascade + unfold); the full-BZ leg is a FULL fresh
    pipeline (ζ-fit → full-BZ V_q → full-BZ W).  Comparing across the two
    therefore covers the ζ IBZ writes, the V_q unfold, AND the W-solve
    unfold — not just the W stage.  Static COHSEX because its unfold is
    purely algebraic (measured exactly equal at printed precision); the
    dynamic path has a benign ~0.12 meV q-set path-dependence through the
    nonlinear PPM fit.
    """
    labels = ("sigSX", "sigCOH", "sigTOT")
    out_name = "sigma_diag_ibzgate.dat"

    # Leg A — IBZ cascade, from restart (cheap: no ζ/V_q rebuild).
    ibz_dir = copy_fixture(_GNPPM, tmp_path / "ibz",
                           tmp_from=gnppm_session.run_dir)
    mutate_input(ibz_dir / "cohsex_ibz_test.in",
                 {"restart = false": "restart = true"})
    res_a = run_gw_jax(ibz_dir, "cohsex_ibz_test.in",
                       extra_env={"LORRAX_FORCE_FULL_BZ": "0"})
    if res_a.returncode != 0:
        pytest.fail(f"IBZ leg failed.\nstdout:\n{res_a.stdout}\n"
                    f"stderr:\n{res_a.stderr}")
    # The restart path must still take the IBZ decision for the W solve
    # (timing sections only appear when the cascade slices/unfolds).
    assert "Loaded restart tensors from H5." in res_a.stdout
    assert "W.slice_to_ibz" in res_a.stdout, (
        "restart static run skipped the IBZ W-solve cascade")
    assert "W.unfold_to_full_bz" in res_a.stdout
    assert "orbit closure failed" not in res_a.stdout

    # Leg B — full-BZ-direct, FRESH full pipeline.
    full_dir = copy_fixture(_GNPPM, tmp_path / "full")
    res_b = run_gw_jax(full_dir, "cohsex_ibz_test.in",
                       extra_env={"LORRAX_FORCE_FULL_BZ": "1"})
    if res_b.returncode != 0:
        pytest.fail(f"full-BZ leg failed.\nstdout:\n{res_b.stdout}\n"
                    f"stderr:\n{res_b.stderr}")
    assert "unfold=full-BZ" in res_b.stdout, (
        "FORCE_FULL_BZ leg unexpectedly took the IBZ cascade")

    _assert_rows_close(ibz_dir, full_dir, out_name, labels, 1e-6,
                       "IBZ cascade Σ differs from full-BZ-direct")