"""Tier-1 frozen e2e gates — the four physics regression pins.

Each gate runs the full pipeline (ζ-fit → V_q → χ₀ → W → Σ → QP
extraction → writers) on a small fixture and compares against a frozen
reference.  What each pin uniquely covers:

* ``si_cohsex_3d`` — the ONE external anchor: bulk Si 4×4×4, sys_dim=3
  Coulomb + analytic head; frozen values pinned to BerkeleyGW at
  0.12 meV MAE (see tests/regression/si_cohsex_debug/README.md).
  IRREPLACEABLE — do not shrink or re-freeze casually.
* ``cohsex`` — 2D static COHSEX on WFNsmall: the only IBZ-STORED WFN
  fixture (kgrid 3×3, nrk=4, ntran=12), so the ψ k-unfold and the
  12-op symmetry group run e2e ONLY here; also nspinor=2 static
  SX/COH kernels and the K_POINTS band-path input.
* ``gnppm`` — MoS2 3×3 GN-PPM: the dynamic workhorse (minimax
  screening, PPM fit, 4-branch τ-integration, analytic q→0 head,
  eqp0/eqp1 writers) with the IBZ cascade ACTIVE (asserted on the run
  log — the frozen values alone cannot see a silently deactivated
  cascade because IBZ ≡ full-BZ).  Its session run doubles as the
  prepared state for every Tier-2 from-restart invariance gate.
* ``bispinor`` — MoS2 3×3 nspinor=2 bispinor GN-PPM: dynamic Σ_c on the
  screened charge W plus bare Breit Σ^B folded into sigX; 4 ζ channels,
  7 V_q tiles, transverse γ̃ machinery no scalar gate touches.

atol notes: 1e-6 gates are pure freezes of deterministic runs (the
tolerance only absorbs GPU-nondeterministic last-ULP drift); the Si gate
uses 1e-3 eV — a physical bound above its 0.12 meV BGW agreement, tight
enough to catch a real 3D regression.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (          # noqa: E402
    REG,
    copy_fixture,
    parse_eqp_rows,
    run_gw_jax,
    skip_unless_gpu,
)

# (case_id, subdir, input_name, output_name, reference_name, sigma_labels, atol)
_CASES = [
    ("cohsex", REG / "cohsex_debug", "cohsex_test.in", "eqp_test.dat",
     "eqp_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-6),
    ("si_cohsex_3d", REG / "si_cohsex_debug", "cohsex_si_test.in",
     "eqp_si_test.dat", "eqp_si_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-3),
]


def _assert_matches_reference(output_file, reference_file, labels, atol,
                              case_id):
    assert output_file.exists(), f"no output written: {output_file}"
    if output_file.read_text() == reference_file.read_text():
        return
    ref_rows = parse_eqp_rows(reference_file, labels)
    out_rows = parse_eqp_rows(output_file, labels)
    assert out_rows.shape == ref_rows.shape, (
        f"Row-count mismatch: output {out_rows.shape}, "
        f"reference {ref_rows.shape}")
    # Compare only real-valued physics columns: kpt, band, <3 Σ>, VH_re
    # (byte-identity above is the primary check; this atol path only
    # absorbs GPU-nondeterministic last-ULP drift).
    try:
        np.testing.assert_allclose(
            out_rows[:, :6], ref_rows[:, :6], rtol=0.0, atol=atol)
    except AssertionError as exc:
        pytest.fail(
            f"{case_id} output differs from reference beyond tolerance.\n{exc}")


@pytest.mark.regression
@pytest.mark.parametrize(
    "case_id,case_dir,input_name,output_name,ref_name,labels,atol",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_gw_jax_matches_reference(
    tmp_path, case_id, case_dir, input_name, output_name, ref_name, labels, atol
):
    skip_unless_gpu(pytest)
    reference_file = case_dir / ref_name
    assert (case_dir / input_name).exists(), f"missing input: {input_name}"
    assert reference_file.exists(), f"missing reference: {reference_file}"

    run_dir = copy_fixture(case_dir, tmp_path / case_dir.name)
    result = run_gw_jax(run_dir, input_name)
    if result.returncode != 0:
        pytest.fail(
            f"{case_id} regression run failed.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    _assert_matches_reference(
        run_dir / output_name, reference_file, labels, atol, case_id)


@pytest.mark.regression
def test_gnppm_matches_reference(gnppm_session):
    """MoS2 3×3 GN-PPM frozen gate, on the session run (Tier-2's state)."""
    _assert_matches_reference(
        gnppm_session.run_dir / gnppm_session.output_name,
        REG / "gnppm_debug" / "sigma_diag_gnppm_ref.dat",
        ("sigX", "sigC", "sigXC"), 1e-6, "gnppm")
    # The frozen values CANNOT detect a silently deactivated IBZ cascade
    # (IBZ ≡ full-BZ numerically) — pin the activation on the log.
    assert "unfold=IBZ→full" in gnppm_session.stdout, (
        "gnppm session run did not take the IBZ cascade (V_q g-flat log "
        "line missing 'unfold=IBZ→full') — orbit closure regressed?")
    assert "orbit closure failed" not in gnppm_session.stdout


@pytest.mark.regression
def test_bispinor_gnppm_matches_reference(bispinor_session):
    """Bispinor GN-PPM frozen gate (Σ^B folded into sigX)."""
    _assert_matches_reference(
        bispinor_session.run_dir / bispinor_session.output_name,
        REG / "bispinor_debug" / "sigma_diag_bispinor_ref.dat",
        ("sigX", "sigC", "sigXC"), 1e-6, "bispinor")
    # Fixture properties (see its README): charge tiles full-BZ-direct,
    # transverse tiles through the IBZ cascade.
    assert "charge-centroid orbit closure failed" in bispinor_session.stdout
    assert "V_qmunu_TT_11" in bispinor_session.stdout