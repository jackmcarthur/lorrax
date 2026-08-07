"""Tier-1 frozen e2e gates — the physics regression pins.

Each gate runs the full pipeline (ζ-fit → V_q → χ₀ → W → Σ → QP
extraction → writers) on a small fixture and compares against a frozen
reference.  What each pin uniquely covers:

* ``si_cohsex_3d`` — bulk Si 4×4×4, sys_dim=3 Coulomb + analytic head.
  Two gates run off ONE session run of this deck:
    - ``test_si_production_matches_frozen_reference`` — bit-identity
      against ``eqp_si_ref.dat`` (catches "the code changed");
    - ``test_si_production_matches_berkeleygw`` — the ONE EXTERNAL check
      in this suite, against literal BerkeleyGW columns in
      ``bgw_sigma_hp_noavg.dat`` (catches "the code drifted away from
      BerkeleyGW", which no frozen-reference gate can see).
  IRREPLACEABLE — do not shrink or re-freeze.  Use ``si_cohsex_fast``
  when you want a quick Si run.
* ``si_cohsex_fast`` — the same crystal at 20 bands / 144 centroids,
  ~12 s.  A PURE SELF-FREEZE: measured 2109 meV MAE from BerkeleyGW
  (the band cut, not the centroid count — see the deck header).  It
  gates code changes fast; it says nothing about BGW agreement.
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

atol notes: the 1e-6 gates are pure freezes of deterministic runs (the
tolerance only absorbs GPU-nondeterministic last-ULP drift).  The Si
freezes use 1e-5 eV, the floor the .dat's 6-decimal format allows; both
Si decks reproduce bit-identically across independent runs, so that is
headroom, not slack.  The BerkeleyGW gate is separate and is stated in
meV against measured values — see ``_BGW_TOL``.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (          # noqa: E402
    REG,
    compare_to_bgw,
    copy_fixture,
    parse_eqp_rows,
    run_gw_jax,
    skip_unless_gpu,
)

SI_DIR = REG / "si_cohsex_debug"

#: Cross-machine tolerance for the FRONTERA-FROZEN Tier-1 pins, in eV.
#:
#: OWNER RULING, 2026-08-07: *"the micro-eV level is fine for comparisons
#: between machines."*
#:
#: The two pins below (`gnppm`, `bispinor`) are frozen from Frontera CLX
#: output (`f485b5a`, 2026-08-01, job 7885154) and re-measured on
#: Perlmutter at `svc/distrib_la-2026-08-07`.  Both drift by **exactly one
#: unit in the 6th printed decimal** — 20/2484 rows for gnppm, 24/1620 for
#: bispinor, max |Δ| exactly 1.000e-6 eV — in one direction on Frontera and
#: the other on Perlmutter.  A 6-decimal `.dat` at `atol=1e-6` has no room
#: for a cross-platform ULP, so "re-freeze on whichever machine ran last"
#: is a permanent ping-pong that silently turns the other machine red.
#: This is that ping-pong ended: 1e-5 eV is 10x the observed drift and
#: five orders below anything physical.
#:
#: WHAT STILL ANCHORS THESE TIGHTLY.  Loosening a cross-machine pin does
#: not loosen the tree.  Same-machine drift is caught by the Si COHSEX
#: byte-identity gate (`test_si_production_matches_frozen_reference`,
#: which returns early on an exact text match) and by the BerkeleyGW
#: anchor (`test_si_production_matches_berkeleygw`, `_BGW_TOL`, sub-meV
#: MAE against an external code).  Those are the gates that would catch a
#: physics change; these two answer "does the frozen MoS2 output still
#: reproduce on a different machine", and the answer should not depend on
#: the 6th decimal of a text file.
#:
#: DO NOT reach for this constant to make an unrelated red go green.  It
#: is scoped to the two cross-machine-frozen pins by name.
_XMACHINE_ATOL_EV = 1e-5

# (case_id, subdir, input_name, output_name, reference_name, sigma_labels, atol)
_CASES = [
    ("cohsex", REG / "cohsex_debug", "cohsex_test.in", "eqp_test.dat",
     "eqp_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-6),
]


def _report_headroom(case_id, msg):
    """Emit a comparison verdict that survives pytest's capture AND xdist.

    ``print`` is not enough: under ``-n>0`` a worker's stdout is only
    replayed for FAILING cells, so a PASSING cell's margin — the number
    that says how close the gate is to the edge — vanishes exactly when
    everything looks fine.  That is the number an auditor of the
    2026-08-07 cross-machine tolerance ruling needs, so it goes to stderr
    (replayed for passes under ``-rA``/``-s``) and to a warning, which
    xdist forwards to the controller unconditionally.
    """
    line = f"[xmachine] {case_id}: {msg}"
    print(line, flush=True)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    warnings.warn(line, stacklevel=2)


def _assert_matches_reference(output_file, reference_file, labels, atol,
                              case_id):
    assert output_file.exists(), f"no output written: {output_file}"
    # REPORT THE HEADROOM, ALWAYS — byte-identical, within tolerance, or
    # over it.  A tolerance nobody can see the margin on is a tolerance
    # nobody can audit: it absorbs a growing drift silently until the day
    # it does not, and then the first number anyone has is the one that
    # broke it.  Reporting only on the non-identical path is not enough
    # either — "it passed" would then be ambiguous between "bit-for-bit"
    # and "used 99% of the budget", which are completely different facts
    # about the same green cell.
    if output_file.read_text() == reference_file.read_text():
        _report_headroom(case_id, "BYTE-IDENTICAL to the reference "
                                  f"(atol {atol:.0e} not exercised)")
        return
    ref_rows = parse_eqp_rows(reference_file, labels)
    out_rows = parse_eqp_rows(output_file, labels)
    assert out_rows.shape == ref_rows.shape, (
        f"Row-count mismatch: output {out_rows.shape}, "
        f"reference {ref_rows.shape}")
    # Compare only real-valued physics columns: kpt, band, 3 Σ, VH_re
    # (byte-identity above is the primary check; this atol path only
    # absorbs GPU-nondeterministic last-ULP drift).
    _d = np.abs(out_rows[:, :6] - ref_rows[:, :6])
    _mx = float(_d.max()) if _d.size else 0.0
    _report_headroom(
        case_id,
        f"max |Δ| = {_mx:.3e} vs atol {atol:.0e} "
        f"({_mx / atol:.1%} of budget, {int((_d > atol).sum())} cells over, "
        f"{int((_d > 0).sum())} of {_d.size} cells differ at all)")
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


# ---------------------------------------------------------------------------
# Si 4×4×4 — production deck.  Two gates, ONE run (``si_session``).
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_si_production_matches_frozen_reference(si_session):
    """Si production deck vs its own frozen output — 'did the code change?'.

    This is a SELF-freeze and cannot see drift away from BerkeleyGW; that is
    what ``test_si_production_matches_berkeleygw`` below is for.
    """
    _assert_matches_reference(
        si_session.run_dir / si_session.output_name,
        SI_DIR / "eqp_si_ref.dat",
        ("sigSX", "sigCOH", "sigTOT"), 1e-5, "si_cohsex_3d")


# Tolerances for the BerkeleyGW anchor, in meV, as (MAE, max|Δ|).
#
# MEASURED 2026-08-07 at 04b8bba, production deck, full BZ (64 k × 16 bands)
# against tests/regression/si_cohsex_debug/bgw_sigma_hp_noavg.dat:
#
#     column     MAE      max|Δ|
#     sigSX     0.1416    0.3972
#     sigCOH    0.5871    3.1041
#     sigTOT    0.6439    2.9447
#
# The gate sits at roughly 2× the measured MAE and 1.6× the measured max.
# WHAT IT WOULD TAKE TO FAIL: the defect this file exists to catch —
# ``mc_average_vcoul_body`` reverting to its LORRAX default (true), i.e.
# MC-averaging v(q,G=0) at every q≠0 where BGW's ``cell_average_cutoff 1d-12``
# averages only the literal q+G=0 element — moves sigTOT by 141.65 meV MAE.
# That is ~95× this MAE budget, so the gate fires hard.  It would likewise
# fire on a broken v(q+G), a collapsed ISDF rank, or a wrong q→0 head.
# WHAT IT WOULD *NOT* CATCH: anything under ~1.5 meV MAE; and nothing about
# kin_ion or V_H, which are not BGW-compared columns (kin_ion is an input
# artifact this fixture does not regenerate).
_BGW_TOL = {
    "sigSX":  (1.5, 5.0),
    "sigCOH": (1.5, 5.0),
    "sigTOT": (1.5, 5.0),
}

# Symmetry-equivalent k MUST carry identical Σ.  MEASURED 2.611 meV on the
# production deck — NOT zero, because centroids_frac_960.txt is a literal
# (non-orbit-closed) point set, so the ISDF quadrature itself breaks the
# 48-op point group.  The fast deck's orbit-closed 144-point set measures
# exactly 0.000.  Gated loosely here to pin the known value without
# pretending it is clean; see README.md "Known defects".
_BGW_STAR_SPREAD_MAX_MEV = 5.0


@pytest.mark.regression
def test_si_production_matches_berkeleygw(si_session):
    """Si production deck vs literal BerkeleyGW columns — THE external anchor.

    Every other gate in this suite compares LORRAX to LORRAX.  This one is the
    only place BerkeleyGW enters the loop, so it is the only gate that can see
    the code drifting away from the reference implementation rather than merely
    away from its own past output.
    """
    fixture = SI_DIR / "bgw_sigma_hp_noavg.dat"
    assert fixture.exists(), (
        f"missing BerkeleyGW anchor fixture: {fixture}.  Regenerate with "
        f"tools/bgw_sigma_hp_to_fixture.py from a BGW sigma_hp.log — never "
        f"from LORRAX output.")
    stats = compare_to_bgw(
        si_session.run_dir / si_session.output_name, fixture)

    # Guard the comparison itself: if the k-matching silently found only a
    # handful of k-points, the MAEs below would be computed over almost
    # nothing and would pass vacuously.
    assert stats["_nstar"] == 64, (
        f"BGW anchor compared only {stats['_nstar']} LORRAX k-points; expected "
        f"the full 64-point BZ.  The IBZ→full-BZ k assignment is broken, so "
        f"the tolerances below would be measuring almost nothing.")

    failures = []
    for col, (mae_tol, max_tol) in _BGW_TOL.items():
        mae, mx = stats[col]
        if mae > mae_tol or mx > max_tol:
            failures.append(
                f"  {col}: MAE {mae:.4f} meV (limit {mae_tol}), "
                f"max|Δ| {mx:.4f} meV (limit {max_tol})")
    if stats["_star_spread"] > _BGW_STAR_SPREAD_MAX_MEV:
        failures.append(
            f"  star spread: {stats['_star_spread']:.4f} meV "
            f"(limit {_BGW_STAR_SPREAD_MAX_MEV}) — symmetry-equivalent "
            f"k-points disagree with each other")
    if failures:
        pytest.fail(
            "Si production deck no longer agrees with BerkeleyGW "
            "(bgw_sigma_hp_noavg.dat, full BZ, 64 k × 16 bands):\n"
            + "\n".join(failures)
            + "\n\nThis is the suite's ONLY external check.  Do not widen "
              "these tolerances to make it green — find what moved.")


# ---------------------------------------------------------------------------
# Si 4×4×4 — fast deck.  Self-freeze only.
# ---------------------------------------------------------------------------

_SI_FAST_REF = SI_DIR / "eqp_si_fast_ref.dat"


@pytest.mark.regression
@pytest.mark.skipif(
    not _SI_FAST_REF.exists(),
    reason=(
        "eqp_si_fast_ref.dat is not frozen yet — freezing a reference is the "
        "owner's call.  A candidate generated 2026-08-07 at 04b8bba lives in "
        "/pscratch/sd/j/jackm/si_consolidation_2026-08-07/run_fast_final/ "
        "(eqp_si_fast.dat); copy it in to enable this gate."))
def test_si_fast_matches_frozen_reference(si_fast_session):
    """Si fast deck vs its own frozen output.  NOT a BerkeleyGW anchor.

    MEASURED 2109 meV MAE from BGW — see cohsex_si_fast.in's header.  This
    gate answers "did the code change?" in ~12 s and nothing else.
    """
    _assert_matches_reference(
        si_fast_session.run_dir / si_fast_session.output_name,
        _SI_FAST_REF, ("sigSX", "sigCOH", "sigTOT"), 1e-5, "si_cohsex_fast")


@pytest.mark.regression
def test_gnppm_matches_reference(gnppm_session):
    """MoS2 3×3 GN-PPM frozen gate, on the session run (Tier-2's state).

    Frontera-frozen; runs on Perlmutter too at ``_XMACHINE_ATOL_EV``
    (owner ruling 2026-08-07 — read that constant before touching it).
    """
    _assert_matches_reference(
        gnppm_session.run_dir / gnppm_session.output_name,
        REG / "gnppm_debug" / "sigma_diag_gnppm_ref.dat",
        ("sigX", "sigC", "sigXC"), _XMACHINE_ATOL_EV, "gnppm")
    # The frozen values CANNOT detect a silently deactivated IBZ cascade
    # (IBZ ≡ full-BZ numerically) — pin the activation on the log.
    assert "unfold=IBZ→full" in gnppm_session.stdout, (
        "gnppm session run did not take the IBZ cascade (V_q g-flat log "
        "line missing 'unfold=IBZ→full') — orbit closure regressed?")
    assert "orbit closure failed" not in gnppm_session.stdout


@pytest.mark.regression
def test_bispinor_gnppm_matches_reference(bispinor_session):
    """Bispinor GN-PPM frozen gate (Σ^B folded into sigX).

    Frontera-frozen; runs on Perlmutter too at ``_XMACHINE_ATOL_EV``
    (owner ruling 2026-08-07 — read that constant before touching it).
    """
    _assert_matches_reference(
        bispinor_session.run_dir / bispinor_session.output_name,
        REG / "bispinor_debug" / "sigma_diag_bispinor_ref.dat",
        ("sigX", "sigC", "sigXC"), _XMACHINE_ATOL_EV, "bispinor")
    # Fixture properties (see its README): charge tiles full-BZ-direct,
    # transverse tiles through the IBZ cascade.
    assert "charge-centroid orbit closure failed" in bispinor_session.stdout
    assert "V_qmunu_TT_11" in bispinor_session.stdout
