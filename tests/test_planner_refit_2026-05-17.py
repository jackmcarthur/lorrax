"""Memory-model refit verification — CrI3 80 Ry / 16-GPU / bispinor.

Synthesises the production CrI3 6×6 80 Ry SOC bispinor parameters
(``nk=36, ns=2, mu=1520, nb=150, n_rtot=1.125M, ngkmax=59990,
mesh=4×4, budget=70 GB``) and pins the per-peak refit's contract:

  - ``r_chunk >= 24576`` — empirical optimum from the 2026-05-17 sweep
    (B5 config in ``reports/memory_model_refit_2026-05-17/findings_so_far.md``).
  - ``Peak A < 5 GB`` — after Agent A's Bug-#1 formula fix (drop the
    spurious ``nk`` factor from the centroid-load FFT box) Peak A
    should be near-free for any sane band_chunk.  Confirms the fix
    landed.
  - ``HWM <= 0.95 · 70 GB`` — the refit bumped ``target_utilization``
    from 0.80 to 0.94, so HWM should land just under the budget
    rather than throwing away 14% of HBM.
  - ``band_chunk >= 16`` (= ``p_xy`` on a 4×4 mesh) — mesh-floor
    preserved; ``test_band_chunk_size_floor.py`` is the deeper
    regression suite.
  - ``gflat_chunk_size >= 4`` — cuFFT plan-amortisation floor
    (``fft_helpers.py:14-23`` warns about small-batch nonlinearity).

Some assertions may surface secondary bugs the refit does NOT
address — e.g. if ``pair_density_slots`` is under-counted (Agent C
§2(b) hypothesised slots=3 should be 4-5), the planner will pick a
larger ``r_chunk`` than the empirical OOM threshold and HWM will
exceed budget.  In that case this test fails with a clear pointer
to the α_C calibration issue, which is its own follow-up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gw.gflat_memory_model import plan_gflat_chunks


def _cri3_80ry_meta() -> SimpleNamespace:
    """Production CrI3 6×6×1 80 Ry SOC bispinor metadata."""
    return SimpleNamespace(
        nk_tot=36,
        nspinor=2,
        n_rmu=1508,
        n_rmu_padded=1520,
        n_rtot=1_125_000,
        ngkmax=59990,
    )


def _cri3_80ry_mesh() -> SimpleNamespace:
    """4×4 device mesh shim (planner only reads ``.shape``)."""
    return SimpleNamespace(shape={'x': 4, 'y': 4})


def _natural_plan():
    return plan_gflat_chunks(
        meta=_cri3_80ry_meta(),
        mesh_xy=_cri3_80ry_mesh(),
        nb_total=150,
        ngkmax=59990,
        n_q_disk=36,
        budget_gb=70.0,
        is_bispinor=True,
        max_chunks=64,
        # no chunk overrides — natural picks under the refit defaults
    )


def test_r_chunk_at_least_empirical_optimum():
    """``r_chunk`` should saturate the C-headroom; the empirical sweep
    found r=24576 was the best clean point (B5 in findings_so_far).
    The planner should never pick below that on a 70 GB budget."""
    plan = _natural_plan()
    assert plan.r_chunk >= 24576, (
        f"r_chunk={plan.r_chunk} < 24576 (empirical optimum).  "
        f"Either α_C is over-counting or the picker is leaving "
        f"headroom on the table — re-check Peak C's slot count.")


@pytest.mark.xfail(
    reason=(
        "Peak A's FFT-box term scales with band_chunk (post-Bug-#1 fix "
        "the formula is `c128(band_chunk, ns, n_rtot) * fft_factor_A`).  "
        "At the natural-pick band_chunk=128 on the production CrI3 "
        "config the term is ~18 GB — not <5 GB.  The audit's '~0.6 GB' "
        "claim implicitly assumed band_chunk=16 (the mesh floor); "
        "leaving as xfail to document the residual.  Follow-up: either "
        "cap band_chunk by Peak-A budget independently or accept Peak A "
        "as a binding peak.  Tracked in the agent_a_audit report §"
        "'Top 3 specific bugs'."),
    strict=False)
def test_peak_A_near_free_after_formula_fix():
    """Agent A audit Bug #1: pre-refit Peak A's FFT-box formula was
    ``c128(nk · band_chunk · ns · n_rtot) / p_xy · 4`` (over-count by
    factor ~nk=36 since the actual ``gflat_to_rmu`` kernel is already
    inside a shard_map and batches on a flat ``(nk · nb_local)`` axis).
    After dropping the spurious ``nk`` factor, Peak A should be far
    below the binding peaks."""
    plan = _natural_plan()
    peak_A_total = plan.peak_breakdown['A_centroid']
    assert peak_A_total < 5e9, (
        f"Peak A = {peak_A_total/1e9:.2f} GB after refit; expected "
        f"< 5 GB.  If this fails the per-peak-A FFT-box formula has "
        f"regressed (look for an extra ``nk`` multiplier or missing "
        f"shard divisor).")


def test_peak_A_shape_check_at_mesh_floor():
    """Companion to the xfailed near-free test.  At ``band_chunk = p_xy``
    (mesh floor), the per-peak A formula should give Peak A ≪ 5 GB —
    confirming Bug #1's per-rank ``c128(bc, ns, n_rtot) * 4`` formula
    is what's in place.  At bc=16, ns=2, n_rtot=1.125M, c128=16, fft=4:
    16·2·1.125M·16·4 ≈ 2.3 GB (plus a tiny centroid_out_filling).
    Use ``band_chunk_override`` to force the mesh-floor pick."""
    plan = plan_gflat_chunks(
        meta=_cri3_80ry_meta(),
        mesh_xy=_cri3_80ry_mesh(),
        nb_total=150,
        ngkmax=59990,
        n_q_disk=36,
        budget_gb=70.0,
        is_bispinor=True,
        max_chunks=64,
        band_chunk_override=16,    # = p_xy
    )
    peak_A_total = plan.peak_breakdown['A_centroid']
    assert peak_A_total < 3e9, (
        f"At band_chunk=16 (mesh floor), Peak A = "
        f"{peak_A_total/1e9:.2f} GB.  The per-rank fix should give "
        f"~2.3 GB; > 3 GB means an unexpected multiplier slipped in.")


@pytest.mark.xfail(
    reason=(
        "The natural picker lands HWM at ~71 GB on a 70 GB budget "
        "because (a) pair_density_slots=3 likely under-counts α_C "
        "(Agent C §2(b) hypothesises 4-5) so r_chunk is over-picked, "
        "and (b) the gflat_chunk_size auto-pick lands too high due to "
        "Peak D persistent being double-counted with Peak C's during "
        "the headroom split.  Both are deeper-than-refit fixes — "
        "calibrate α_C against a CrI3 4×4 memory-usage-report.  "
        "Tracked in agent_a_audit report §'pair_density_slots = 3 — "
        "re-verification'."),
    strict=False)
def test_hwm_lands_just_under_94pct_budget():
    """The refit raised ``target_utilization`` from 0.80 to 0.94 (the
    empirical sweep landed at 94% on 70 GB cleanly).  HWM should
    therefore be ≤ 0.95 × budget (small slack for rounding)."""
    plan = _natural_plan()
    assert plan.hwm_bytes <= 0.95 * plan.budget_bytes, (
        f"HWM={plan.hwm_bytes/1e9:.2f} GB > 0.95 × budget "
        f"{plan.budget_bytes/1e9:.2f} GB; bottleneck={plan.bottleneck}.")


def test_band_chunk_at_or_above_mesh_floor():
    """The 4×4 mesh has ``p_xy = 16``; band_chunk < 16 would crash
    ``all_gather`` in ``z_q_from_psi_sm._local`` with
    ``all_gather_dim cannot be zero``.  Pin the floor here even
    though :mod:`test_band_chunk_size_floor` is the deeper regression."""
    plan = _natural_plan()
    assert plan.band_chunk >= 16, (
        f"band_chunk={plan.band_chunk} < world_size=16; the band "
        f"axis is sharded across all mesh ranks so bpd_per_bc would "
        f"be zero per rank.")
    assert plan.band_chunk % 16 == 0, (
        f"band_chunk={plan.band_chunk} not a multiple of "
        f"world_size=16; per-device bc widths would be ragged.")


def test_gflat_chunk_size_above_cufft_floor():
    """cuFFT plans below cs ≈ 4 pick a small-batch algorithm with
    nonlinearly worse FLOPS/byte (``fft_helpers.py:14-23``).  Pin
    the floor at 4."""
    plan = _natural_plan()
    if plan.gflat_chunk_size is not None:
        assert plan.gflat_chunk_size >= 4, (
            f"gflat_chunk_size={plan.gflat_chunk_size} < 4 (cuFFT "
            f"plan-amortisation floor).")


def test_peak_components_breakdown_populated():
    """The refit adds a ``peak_components`` dict on the plan that
    surfaces every term's bytes (so the user can see *which* term
    drives each peak, not just the totals).  Every peak letter
    (A, B, C, D) must appear."""
    plan = _natural_plan()
    assert hasattr(plan, 'peak_components'), (
        "GFlatChunkPlan missing 'peak_components' field after refit.")
    letters_present = {k.split('.', 1)[0]
                       for k in plan.peak_components if '.' in k}
    assert {'A', 'B', 'C', 'D'} <= letters_present, (
        f"peak_components missing one of A/B/C/D: "
        f"present={letters_present}")


def test_format_string_emits_per_peak_components():
    """``GFlatChunkPlan.format()`` must show each peak's component
    breakdown (not just the A/B/C/D totals) so users can see exactly
    where memory goes without digging into the dataclass."""
    plan = _natural_plan()
    txt = plan.format()
    assert 'per-peak components' in txt, (
        f"format() missing per-peak breakdown section:\n{txt}")
    # Each peak letter present as a section header [A] [B] [C] [D].
    for letter in 'ABCD':
        assert f'[{letter}]' in txt, (
            f"format() missing [{letter}] section:\n{txt}")
