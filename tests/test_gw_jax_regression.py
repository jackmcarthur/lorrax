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
* ``hbn_cohsex_3d`` — bulk hexagonal BN 3×3×2, the tree's only NON-CUBIC
  3D deck.  Two gates, two fresh runs:
    - ``test_hbn_matches_frozen_reference`` — the self-freeze;
    - ``test_hbn_mc_average_vcoul_body_moves_sigma`` — the NEGATIVE
      CONTROL, and the reason the fixture exists.  Si FCC satisfies
      ``bvec.T = P·bvec`` for a cyclic signed permutation, so the
      mini-BZ draw-convention bug class (358bb0b) is a pure RESEED
      there and no Si gate can ever see it.  hBN's hexagonal ``bvec``
      admits no such P: the same defect is a bias at z = 293.7.  This
      deck also pins no ``vhead``, so it is the only pinned deck that
      runs the native q→0 head ladder end to end.
  NOT a BerkeleyGW anchor — no BGW run exists on this WFN.
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
# production deck over the 16 bands this fixture compares — NOT zero.
#
# THE CAUSE IS NOT KNOWN, and specifically it is NOT the centroid set's
# non-closure, which is what this comment used to assert.  MEASURED
# 2026-08-15: an orbit-closed 960-point set for the same deck
# (centroids_frac_960_orbitclosed.txt, closed=True at 1.000e-06 on 48 ops)
# moves this number only 2.611 -> 1.964 meV, while making agreement with
# BerkeleyGW ~35x WORSE (sigTOT MAE 0.4329 -> 14.9426 meV).  The fast
# deck's 144-point set measures exactly 0.000, so the spread tracks
# centroid COUNT rather than closure; rank truncation in the zeta fit at
# 960 is the untested suspect.  Full table in tests/KNOWN_FAILURES.md.
#
# Gated loosely here to pin the known value without pretending it is
# clean; see README.md "Known defects".
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
    # Report the band-cut's degeneracy status and the subspace-invariant
    # twin alongside the gated number, so the margin AND its meaning are
    # both auditable.  MEASURED 2026-08-15: over these 16 bands the per-band
    # spread reads 2.611 meV and the multiplet-trace spread 0.593 meV — the
    # per-band figure is inflated ~4.4x by the arbitrariness of the band
    # label inside a degenerate multiplet, and on this deck EVERY band is
    # inside one.  Neither is gated on here; the gate stays on the
    # historical quantity so its threshold keeps its meaning.
    _report_headroom(
        "si_bgw_star_spread",
        f"per-band {stats['_star_spread']:.4f} meV; multiplet-trace "
        f"{stats['_star_spread_multiplet']} meV; band cut at {16} is "
        f"{'CLEAN' if stats['_cut_clean'] else 'NOT clean'} "
        f"(cut_clean={stats['_cut_clean']})")
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


# ---------------------------------------------------------------------------
# hBN 3×3×2 — the NON-CUBIC deck.  Two gates, TWO runs (the second one is the
# point).
# ---------------------------------------------------------------------------

HBN_DIR = REG / "hbn_cohsex_debug"
_HBN_WFN = HBN_DIR / "WFN.h5"
_HBN_REF = HBN_DIR / "eqp_hbn_ref.dat"

#: Freeze tolerance for the hBN deck, in eV.  Same constant as the two Si
#: freezes, chosen for the same documented reason.
#:
#: WHAT JUSTIFIES A TIGHT PIN HERE.  THREE independent runs of this deck agree
#: BYTE FOR BYTE on the data lines (md5 `d4a7e4502a277e4aa203303042e792ec`):
#: the two 2×2-mesh / 4-process runs that produced the reference, and a
#: 1×1-mesh / 1-process reproduction made at the freeze — which is the mesh
#: THIS gate runs on, because tests/conftest.py pins one GPU per process.
#: `delta_run2_vs_run1.txt` records every column MAE, max|Δ| and rms at
#: exactly 0.000000 meV over all 1440 rows.
#:
#: WHY NOT 1e-6.  Because that is the exact magnitude of the cross-machine ULP
#: that forced the `_XMACHINE_ATOL_EV` ruling above: a 6-decimal `.dat` at
#: atol=1e-6 has no room for one unit in the last printed digit.  Pinning
#: there would re-open a ping-pong this tree closed on 2026-08-07.
#:
#: WHY IT IS STILL TIGHT WHERE IT MATTERS.  1e-5 eV = 0.01 meV is **40×
#: below this deck's own Monte-Carlo seed width** (head-draw seed 42→43 moves
#: sigTOT by 0.396 meV MAE / 1.127 meV max) and **1400× below the knob this
#: fixture exists to watch** (13.995 meV MAE).  Nothing the fixture was built
#: to see can hide under it — and byte-identity, not this number, is the
#: primary check: `_assert_matches_reference` returns early on an exact text
#: match and says so.
_HBN_ATOL_EV = 1e-5

_HBN_SKIP_REASON = (
    f"hBN fixture inputs are missing ({_HBN_WFN} not found).  They are "
    f"TRACKED in git like every other regression fixture's binaries, so on a "
    f"full checkout this never fires — a skip here means a partial or "
    f"sparse checkout, not an optional gate.")


@pytest.mark.regression
@pytest.mark.skipif(not _HBN_WFN.exists(), reason=_HBN_SKIP_REASON)
def test_hbn_matches_frozen_reference(hbn_session):
    """hBN non-cubic deck vs its own frozen output.  NOT a BerkeleyGW anchor.

    The tree's only end-to-end deck on a cell where the mini-BZ head-draw
    convention is BIAS-sensitive.  Si FCC satisfies ``bvec.T = P·bvec`` for a
    cyclic signed permutation, so on Si the pre-358bb0b transposed draw is a
    pure reseed (measured z = 3.0, consistent with noise); hBN's hexagonal
    ``bvec`` admits no such P and the same defect is a bias at z = 293.7,
    55.8% of the whole mc-average correction.  This deck also pins no
    ``vhead``/``whead_0freq``, so it is the only pinned deck that runs the
    native q→0 head ladder end to end.

    A SELF-FREEZE.  No BerkeleyGW run exists on this WFN; see README.md.
    """
    _assert_matches_reference(
        hbn_session.run_dir / hbn_session.output_name,
        _HBN_REF, ("sigSX", "sigCOH", "sigTOT"), _HBN_ATOL_EV, "hbn_cohsex_3d")


# Liveness floor for the mc_average_vcoul_body control, in meV of sigTOT MAE.
#
# MEASURED at the freeze, single-variable A/B on this exact deck
# (delta_armA_mcavg_false.txt):
#
#     column     MAE       max|Δ|     signed mean      rms
#     sigSX      7.836     41.797       -5.616       15.294
#     sigCOH    17.217     50.333        9.932       20.304
#     sigTOT    13.995     49.732        4.316       17.422
#     VH         0.000      0.000        0.000        0.000
#     Eo         0.000      0.000        0.000        0.000
#
# and the MC seed width of the same column (head-draw seed 42→43) is 0.396 meV
# MAE / 1.127 meV max.  5.0 sits 12.6× above the seed noise, so seed or
# default drift can never flake this cell, and 2.8× below the measured effect,
# so the knob going dead cannot sneak past.  It is a LIVENESS pin, not a value
# pin — the value is what the frozen reference is for.  DO NOT tighten it
# toward 13.995: that would make this cell a second freeze and it would start
# failing for reasons that have nothing to do with the head table being live.
_HBN_MCAVG_MIN_MAE_MEV = 5.0


@pytest.mark.regression
@pytest.mark.skipif(not _HBN_WFN.exists(), reason=_HBN_SKIP_REASON)
def test_hbn_mc_average_vcoul_body_moves_sigma(hbn_session,
                                               hbn_mcavg_false_session):
    """THE NEGATIVE CONTROL — the cell this whole fixture exists for.

    A frozen-reference gate says "the numbers did not change".  It does NOT
    say "the mini-BZ head average is still LIVE and still reachable on a cell
    whose lattice can see it".  If ``build_v_head_miniBZ_fn_3d`` were
    silently disconnected — the transpose-bug class, or any future edit that
    routes around it — a re-freeze would pin the wrong numbers and stay green
    forever, exactly as Si has been structurally unable to notice for the
    whole life of this suite.

    So this cell constructs the case where the check comes out FALSE: run the
    same deck with ``mc_average_vcoul_body`` flipped and require Σ to MOVE.
    Both arms are live runs of the same code on the same machine at the same
    moment, differing in one deck key — the single-variable A/B, not a
    comparison against a file frozen at some other time.
    """
    ref_rows = parse_eqp_rows(
        hbn_session.run_dir / hbn_session.output_name,
        ("sigSX", "sigCOH", "sigTOT"))
    arm_rows = parse_eqp_rows(
        hbn_mcavg_false_session.run_dir / hbn_mcavg_false_session.output_name,
        ("sigSX", "sigCOH", "sigTOT"))
    assert arm_rows.shape == ref_rows.shape, (
        f"control arm row count {arm_rows.shape} != reference run "
        f"{ref_rows.shape} — the two arms are not the same calculation")
    # Guard the comparison itself: rows must be the same (k, band) in the same
    # order, or every MAE below is measuring a permutation instead of a knob.
    np.testing.assert_array_equal(
        arm_rows[:, :2], ref_rows[:, :2],
        err_msg="control arm k/band grid differs from the reference run")
    assert ref_rows.shape[0] == 1440, (
        f"expected 18 k × 80 bands = 1440 rows, parsed {ref_rows.shape[0]} — "
        f"a truncated parse would make the MAE below vacuous")

    d_mev = np.abs(arm_rows[:, 2:5] - ref_rows[:, 2:5]) * 1e3
    mae = d_mev.mean(axis=0)
    mx = d_mev.max(axis=0)
    _report_headroom(
        "hbn_mcavg_control",
        "mc_average_vcoul_body true->false moved "
        f"sigSX {mae[0]:.3f}/{mx[0]:.3f}, sigCOH {mae[1]:.3f}/{mx[1]:.3f}, "
        f"sigTOT {mae[2]:.3f}/{mx[2]:.3f} meV (MAE/max); freeze measured "
        f"13.995/49.732 on sigTOT, floor {_HBN_MCAVG_MIN_MAE_MEV}")

    assert mae[2] > _HBN_MCAVG_MIN_MAE_MEV, (
        f"mc_average_vcoul_body = false moved sigTOT by only {mae[2]:.4f} meV "
        f"MAE (max {mx[2]:.4f}), under the {_HBN_MCAVG_MIN_MAE_MEV} meV "
        f"floor.  The mini-BZ head average is NOT reaching Sigma on a cell "
        f"that can see it.  Measured 13.995 meV MAE / 49.732 meV max when "
        f"this fixture was frozen (2026-08-07), against a Monte-Carlo seed "
        f"width of 0.396 meV MAE.  Do NOT lower this floor to make the cell "
        f"green — find what disconnected the head table.")

    # The knob acts through the Coulomb head table and NOWHERE ELSE.  VH is a
    # mean-field column; it measured EXACTLY 0.000000 across the arm at the
    # freeze, so a moving VH means the flip perturbed something it has no
    # business touching and the MAE above is not measuring what it claims.
    vh = float(np.abs(arm_rows[:, 5] - ref_rows[:, 5]).max())
    assert vh <= _HBN_ATOL_EV, (
        f"mc_average_vcoul_body moved VH by {vh:.3e} eV (measured exactly "
        f"0.000000 at the freeze).  This knob touches the Coulomb head table, "
        f"not the mean field — the Sigma delta above is not a clean "
        f"single-variable measurement.")


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
    #
    # THE STRING MOVED, THE PROPERTY DID NOT.  This used to look for
    # "charge-centroid orbit closure failed", which was the wording of the
    # per-call-site fallback notice that `53908088` ("q-grid closure: one
    # resolution point, and the fallback stops being silent") replaced with
    # a single announcement composed by the service.  The property being
    # asserted is unchanged and still true — the bispinor deck's 256-centroid
    # CHARGE set is not orbit-closed (1 of 2 ops violating, worst residual
    # 1.436e-01, carried in
    # `test_symmetry_maps_qgrid_resolution.py::test_a_non_closed_set_...`),
    # so its tiles fall back to the full BZ and say so, while the closed
    # 209-centroid transverse set goes through the IBZ cascade silently.
    #
    # Asserted on the parts that carry meaning rather than on one long
    # literal: that a fallback was announced, and that the set it fell back
    # on was the CHARGE one.
    #
    # THE CALL SITE IS DELIBERATELY NOT ASSERTED, and that is a finding
    # rather than a shortcut.  ``announce_once`` dedupes on the centroid SET
    # (``res.announce_key``), so the ``where`` in the announcement names
    # whichever site reached the charge set FIRST — the bispinor g-flat
    # tile, the V_q/W reduction, or the W Dyson solve, depending on the
    # deck's path through the run.  An assertion on ``bispinor g-flat,
    # charge centroids`` was tried here and failed for exactly that reason
    # while the fallback itself was announced correctly.  Pinning the
    # winner of a dedup race is pinning an implementation detail.
    #
    # The WORST RESIDUAL is the honest discriminant instead: 1.436e-01
    # belongs to the 256-centroid charge set and to nothing else in this
    # deck (the 209-centroid transverse set is closed at 1.1e-16 and
    # announces nothing at all), so finding it in the announcement proves
    # both halves of the fixture property — charge fell back, transverse did
    # not.  It is the same number
    # `test_symmetry_maps_qgrid_resolution.py::test_a_non_closed_set_...`
    # carries for `bispinor_debug-256`, so the two cells move together.
    out = bispinor_session.stdout
    assert "q-grid symmetry: FALLBACK" in out
    assert "not orbit-closed" in out
    assert "1.436e-01" in out, "the charge set's residual should name it"
    assert "V_qmunu_TT_11" in out
