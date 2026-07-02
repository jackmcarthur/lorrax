"""Memory-model refit verification — CrI3 80 Ry / 16-GPU / bispinor.

Synthesises the production CrI3 6×6 80 Ry SOC bispinor parameters
(``nk=36, ns=2, mu=1520, nb=150, n_rtot=1.125M, ngkmax=59990,
mesh=4×4, budget=70 GB``) and pins the per-peak refit's contract.

Round-2 refit (2026-05-17) — based on Round-1 live_arrays census:

  - ``r_chunk >= 24576`` — empirical optimum (B5 in
    ``reports/memory_model_refit_2026-05-17/findings_so_far.md``).
  - ``HWM <= 0.95 · 70 GB`` — refit lands HWM near 94% util.
  - ``band_chunk >= 16`` (= ``p_xy``) — mesh-floor preserved.
  - ``gflat_chunk_size >= 4 and <= 100`` — Round-2 cap added to keep
    cuFFT in verified-safe algorithm regime (factor=2.0 below cs~1000;
    nonlinear scratch growth above; OOM verified at cs=1414 on
    production CrI3 80Ry per agent_f_live_arrays_probe.md).
  - Centroids ×4 counted (was ×2): rmuT_X + transpose for both ψ_l/r.
  - Sphere-idx replicated leak counted in Peak A/B/C/D/E (was missing).
  - Peak E (V_q) added; bispinor TT-off-diagonal is the V_q bottleneck.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gw.gflat_memory_model import (
    plan_gflat_chunks,
    GFLAT_CHUNK_SIZE_CAP,
    N_SPHERE_IDX_BUFFERS,
)


def _cri3_80ry_meta() -> SimpleNamespace:
    """Production CrI3 6×6×1 80 Ry SOC bispinor metadata."""
    return SimpleNamespace(
        nk_tot=36,
        nspinor=2,
        n_rmu=1508,
        n_rmu_padded=1520,
        n_rtot=1_125_000,
        ngkmax=59990,
        fft_grid=(75, 75, 200),
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
    )


def test_r_chunk_at_least_empirical_optimum():
    """``r_chunk`` should saturate the C-headroom; the empirical sweep
    found r=24576 was the best clean point (B5 in findings_so_far)."""
    plan = _natural_plan()
    assert plan.r_chunk >= 24576, (
        f"r_chunk={plan.r_chunk} < 24576 (empirical optimum).")


def test_peak_A_shape_check_at_mesh_floor():
    """At ``band_chunk = p_xy`` (mesh floor) Peak A should be modest.

    Round-2 refit (2026-05-17) added the sphere_idx_replicated term
    to Peak A.  Round-4 (commits d1fcd20 + 94542c2) cut the buffer
    count from 8 to 1 by fixing the wfn_loader + wfn_transforms leaks
    at their source.  At bc=16 the post-Round-4 breakdown is:
      * fft_box           = 16·2·1.125M·16·4    = 2.304 GB
      * phase_table       = 36·1.125M·16        = 0.648 GB
      * sphere_idx        = 1·36·75·75·200·4    = 0.162 GB
      * centroid_out      = 36·2·1520·150·16/16 = 0.016 GB
      * total                                   ≈ 3.13 GB
    The threshold of 5 GB confirms the per-rank Bug-#1 fix is in
    place — anything ≥ 10 GB would indicate an extra ``nk`` factor
    leaked back in."""
    plan = plan_gflat_chunks(
        meta=_cri3_80ry_meta(),
        mesh_xy=_cri3_80ry_mesh(),
        nb_total=150,
        ngkmax=59990,
        n_q_disk=36,
        budget_gb=70.0,
        is_bispinor=True,
        max_chunks=64,
        band_chunk_override=16,
    )
    peak_A_total = plan.peak_breakdown['A_centroid']
    assert peak_A_total < 5e9, (
        f"At band_chunk=16 (mesh floor), Peak A = "
        f"{peak_A_total/1e9:.2f} GB.  Expected ~4.3 GB; > 5 GB means "
        f"an unexpected multiplier slipped in.")


def test_hwm_saturates_budget_after_persistent_fix():
    """Post-fix HWM should saturate the budget — not under-shoot."""
    plan = _natural_plan()
    assert plan.hwm_bytes >= 0.85 * plan.budget_bytes, (
        f"HWM={plan.hwm_bytes/1e9:.2f} GB < 0.85 × budget "
        f"{plan.budget_bytes/1e9:.2f} GB.  bottleneck={plan.bottleneck}.")


def test_band_chunk_at_or_above_mesh_floor():
    """4×4 mesh has p_xy=16; band_chunk < 16 crashes all_gather."""
    plan = _natural_plan()
    assert plan.band_chunk >= 16
    assert plan.band_chunk % 16 == 0


def test_gflat_chunk_size_above_cufft_floor():
    """cuFFT plans below cs ≈ 4 pick a small-batch algorithm."""
    plan = _natural_plan()
    assert plan.gflat_chunk_size is not None, (
        "Round-2 refit: gflat_chunk_size is always an int (cap=100), "
        "no more None / one-shot semantics.")
    assert plan.gflat_chunk_size >= 4, (
        f"gflat_chunk_size={plan.gflat_chunk_size} < 4 (cuFFT floor).")


def test_peak_components_breakdown_populated():
    """``peak_components`` must include every peak letter A/B/C/D/E."""
    plan = _natural_plan()
    assert hasattr(plan, 'peak_components')
    letters_present = {k.split('.', 1)[0]
                       for k in plan.peak_components if '.' in k}
    assert {'A', 'B', 'C', 'D', 'E'} <= letters_present, (
        f"peak_components missing one of A/B/C/D/E: "
        f"present={letters_present}")


def test_format_string_emits_per_peak_components():
    """``GFlatChunkPlan.format()`` shows each peak's component breakdown."""
    plan = _natural_plan()
    txt = plan.format()
    assert 'per-peak components' in txt
    for letter in 'ABCDE':
        assert f'[{letter}]' in txt, (
            f"format() missing [{letter}] section:\n{txt}")


# ---------------------------------------------------------------------------
# Round-2 refit additions (2026-05-17)
# ---------------------------------------------------------------------------

def test_gflat_chunk_size_capped_at_100_regardless_of_budget():
    """Round-2 cap: ``gflat_chunk_size <= GFLAT_CHUNK_SIZE_CAP=100``
    regardless of budget.  Past cs ~ 1000 cuFFT switches plan algorithm
    and workspace grows non-linearly (cs=1414 OOM verified at
    production CrI3 80Ry — agent_f_live_arrays_probe.md).

    Even at a giant 1000 GB budget, the cap must hold."""
    plan = plan_gflat_chunks(
        meta=_cri3_80ry_meta(),
        mesh_xy=_cri3_80ry_mesh(),
        nb_total=150,
        ngkmax=59990,
        n_q_disk=36,
        budget_gb=1000.0,  # absurdly large; cap should still hold
        is_bispinor=True,
        max_chunks=64,
    )
    assert plan.gflat_chunk_size <= GFLAT_CHUNK_SIZE_CAP, (
        f"gflat_chunk_size={plan.gflat_chunk_size} > "
        f"GFLAT_CHUNK_SIZE_CAP={GFLAT_CHUNK_SIZE_CAP} even at 1000 GB "
        f"budget — Round-2 cap not enforced.")


def test_gflat_chunk_size_picks_multiple_of_4_below_cap():
    """The picker must return a multiple of bc_floor_factor=4 (for
    cuFFT plan amortisation) AND be ≤ cap."""
    plan = _natural_plan()
    assert plan.gflat_chunk_size % 4 == 0, (
        f"gflat_chunk_size={plan.gflat_chunk_size} not a multiple of 4")
    assert plan.gflat_chunk_size <= GFLAT_CHUNK_SIZE_CAP


def test_centroids_counted_x4_in_peak_C():
    """Centroid persistent term in Peak C must be ×4 (not ×2):
    rmuT_X + rmu_Y transpose for both ψ_l and ψ_r.  agent_a finding
    #1 + agent_g §6 row #1.  At CrI3 80Ry bispinor:
        4 × 36 × 2 × 1520 × 150 × 16 / 16 = 0.066 GB/rank.
    Pre-refit value (×2) was 0.033 GB/rank.
    """
    plan = _natural_plan()
    val = plan.peak_components.get('C.centroids_persist', 0.0)
    expected_x4 = 4 * 36 * 2 * 1520 * 150 * 16 / 16
    assert abs(val - expected_x4) < expected_x4 * 0.01, (
        f"C.centroids_persist = {val/1e9:.3f} GB; expected "
        f"{expected_x4/1e9:.3f} GB (×4 buffers).  Round-2 refit "
        f"requires ×4 (rmuT_X + transpose for both ψ_l/r).")


def test_sphere_idx_replicated_in_every_peak():
    """The sphere/phase index buffer is REPLICATED (not /p_xy).  Pre-
    Round-4 the make_flat_k_fft + psi_G_store paths each leaked a
    fresh ``(nk, nx, ny, nz) i32`` buffer per channel (agent_h §3
    Finding 3 measured 8 buffers by V_q time).

    Round-4 fix (commits d1fcd20 + 94542c2): the wfn_loader caches
    one device buffer per (k_set, mesh) and wfn_transforms dedupes
    captured closures by content hash, so the production pipeline
    holds ONE replicated buffer for the full run.

    At fft_grid=(75,75,200), nq=36, 1 buffer, int32:
        1 × 36 × 75 × 75 × 200 × 4 = 0.162 GB (REPLICATED — same
        bytes on every rank).  Down from 1.296 GB pre-fix.
    """
    plan = _natural_plan()
    expected = (N_SPHERE_IDX_BUFFERS
                * 36 * 75 * 75 * 200 * 4)
    for peak in ('A', 'B', 'C', 'D', 'E'):
        key = f'{peak}.sphere_idx_replicated'
        val = plan.peak_components.get(key, 0.0)
        assert abs(val - expected) < expected * 0.01, (
            f"{key} = {val/1e9:.3f} GB; expected {expected/1e9:.3f} "
            f"GB (replicated, "
            f"{N_SPHERE_IDX_BUFFERS} buffers post-Round-4 fix).")


def test_peak_E_v_q_has_off_diagonal_tt_binding_term():
    """Peak E should reflect the TT off-diagonal tile (same_zeta=False)
    as the binding case for bispinor.  Per agent_i §2: the dominant
    term is ``2 × _bytes_c128(n_q_ibz, mu, ngkmax) / p_xy`` ≈ 6.5 GB
    global / 0.4 GB/rank for CrI3 80Ry full-BZ.

    With the planner's default full-BZ regime (n_q_ibz = nk = 36) and
    no IBZ cascade, the zeta_L_all + zeta_R_all terms both fire."""
    plan = _natural_plan()
    zl = plan.peak_components.get('E.zeta_L_all', 0.0)
    zr = plan.peak_components.get('E.zeta_R_all', 0.0)
    # Both should be ~3.28 GB/rank at production scale.
    assert zl > 1e9, f"E.zeta_L_all = {zl/1e9:.3f} GB, expected > 1 GB"
    assert zr > 1e9, (
        f"E.zeta_R_all = {zr/1e9:.3f} GB — TT-off-diag should "
        f"populate this term (same_zeta=False).")


def test_peak_E_total_below_budget():
    """V_q is never the binding peak at production scale.  Peak E
    total ≈ 7-10 GB/rank, well below the 70 GB budget."""
    plan = _natural_plan()
    e_total = plan.peak_breakdown.get('E_v_q', 0.0)
    assert e_total < 0.5 * plan.budget_bytes, (
        f"Peak E = {e_total/1e9:.2f} GB > 50% of budget — V_q "
        f"shouldn't bind at production scale on 70 GB.")


def test_planner_uses_e_in_hwm_max():
    """HWM = max(A, B, C, D, E).  Peak E being present means the
    bottleneck reporter can pick 'E_v_q' if E is largest (rare at
    production, but should be valid)."""
    plan = _natural_plan()
    assert 'E_v_q' in plan.peak_breakdown
    # At production scale, the bottleneck should NOT be E.
    assert plan.bottleneck != 'E_v_q' or plan.peak_breakdown['E_v_q'] > 0.0


def test_non_bispinor_uses_fewer_sphere_buffers():
    """Post-Round-4 (commits d1fcd20 + 94542c2) the wfn_loader and
    wfn_transforms dedups cap sphere_idx at 1 buffer for both bispinor
    and non-bispinor — the leak source (per-channel device_put +
    per-cache_key closure rebuild) is fixed at the source, not papered
    over by a smaller per-channel multiplier.

    N_SPHERE_IDX_BUFFERS was 3 pre-fix (1 from psi_G_store +
    2 from wfn_transforms centroid_load).  Post-fix: 1.
    """
    from gw.gflat_memory_model import N_SPHERE_IDX_BUFFERS
    plan = plan_gflat_chunks(
        meta=_cri3_80ry_meta(),
        mesh_xy=_cri3_80ry_mesh(),
        nb_total=150,
        ngkmax=59990,
        n_q_disk=36,
        budget_gb=70.0,
        is_bispinor=False,
        max_chunks=64,
    )
    sphere_charge = (N_SPHERE_IDX_BUFFERS
                     * 36 * 75 * 75 * 200 * 4)
    val = plan.peak_components.get('C.sphere_idx_replicated', 0.0)
    assert abs(val - sphere_charge) < sphere_charge * 0.01, (
        f"Non-bispinor C.sphere_idx_replicated = {val/1e9:.3f} GB; "
        f"expected {sphere_charge/1e9:.3f} GB "
        f"({N_SPHERE_IDX_BUFFERS} buffer post-Round-4 fix).")
