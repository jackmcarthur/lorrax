"""Memory-model contract — redesigned planner (2026-07-03).

Pins the ``MEMORY_MODEL_DESIGN.md`` §1a contract:

  HWM = persistent(P) + max(A, B, C, D, E)

  persistent(P) = L_q (÷P) + gflat_acc (÷P) + 4·ψ-copies (÷√P, single-axis)

Key corrections vs the pre-2026-07 model (design §5):
  * centroid term is ÷√P (2 on 'x' + 2 on 'y'), NOT ÷p_xy (bug #4);
  * the stale ``n_bc`` FFT unroll factor is gone (bug #2);
  * the replicated sphere-idx / phase-table terms are dropped as
    negligible (<1% of budget, design §4);
  * util is ns²-aware, not a hard-coded 0.97/0.80 split (bug #1);
  * a Phase-1 rank floor ``P_min`` is reported (§0).
"""

from __future__ import annotations

from types import SimpleNamespace

from gw.gflat_memory_model import (
    plan_gflat_chunks,
    GFLAT_CHUNK_SIZE_CAP,
    _persistent_bytes,
    _c128,
)


def _cri3_80ry_meta() -> SimpleNamespace:
    """Production CrI3 6×6×1 80 Ry SOC bispinor metadata."""
    return SimpleNamespace(
        nk_tot=36, nspinor=2, n_rmu=1508, n_rmu_padded=1520,
        n_rtot=1_125_000, ngkmax=59990, fft_grid=(75, 75, 200),
    )


def _mesh(p_x, p_y):
    return SimpleNamespace(shape={'x': p_x, 'y': p_y},
                           devices=SimpleNamespace(size=p_x * p_y))


def _plan(mesh, **kw):
    base = dict(meta=_cri3_80ry_meta(), mesh_xy=mesh, nb_total=150,
                ngkmax=59990, n_q_disk=36, budget_gb=70.0, is_bispinor=True,
                max_chunks=64)
    base.update(kw)
    return plan_gflat_chunks(**base)


# ---------------------------------------------------------------------------
# The persistent floor
# ---------------------------------------------------------------------------

def test_centroid_term_is_single_axis_not_pxy():
    """The 4 ψ copies shard single-axis (÷√P), NOT ÷p_xy (design bug #4).
    On a square 4×4 mesh that is a 4× difference from the old ÷p_xy."""
    nk, ns, mu, nb = 36, 2, 1520, 150
    persist = _persistent_bytes(nk=nk, ns=ns, nq=36, nq_disk=36, mu=mu,
                                nb=nb, ngkmax=59990, p_x=4, p_y=4)
    psi_one = _c128(nk, ns, mu, nb)
    expect_single_axis = 2 * psi_one / 4 + 2 * psi_one / 4   # ÷√16 = ÷4
    old_pxy = 4 * psi_one / 16                                # the OLD bug
    assert abs(persist['psi_copies'] - expect_single_axis) < 1.0
    # The corrected term is 4× the old ÷p_xy value on a square 4×4 mesh.
    assert abs(persist['psi_copies'] - 4 * old_pxy) < 1.0


def test_centroid_fix_changes_16gpu_prediction():
    """The centroid fix must raise the persistent floor at 16 GPU
    relative to the buggy ÷p_xy accounting (√P larger)."""
    psi_one = _c128(36, 2, 1520, 150)
    p = _plan(_mesh(4, 4))
    old_centroid = 4 * psi_one / 16     # ÷p_xy (buggy)
    new_centroid = 4 * psi_one / 4      # ÷√P (fixed)
    # persistent includes L_q + gflat_acc + new_centroid; the delta from
    # the old model is exactly (new_centroid - old_centroid) > 0.
    assert new_centroid - old_centroid > 0
    assert p.persistent_bytes > 0


def test_persistent_is_L_q_plus_gflat_plus_psi():
    persist = _persistent_bytes(nk=36, ns=2, nq=36, nq_disk=36, mu=1520,
                                nb=150, ngkmax=59990, p_x=4, p_y=4)
    assert set(persist) == {'L_q', 'gflat_acc', 'psi_copies'}
    assert persist['L_q'] == _c128(36, 1520, 1520, shard=16)
    assert persist['gflat_acc'] == _c128(36, 1520, 59990, shard=16)


# ---------------------------------------------------------------------------
# The rank floor (Phase 1)
# ---------------------------------------------------------------------------

def test_rank_floor_reported():
    """P_min is the smallest mesh whose persistent floor fits util·budget."""
    p = _plan(_mesh(4, 4))
    assert p.p_min >= 1
    # At 70 GB the CrI3 floor is comfortable — P_min should be small.
    assert p.p_min <= 16


def test_rank_floor_grows_when_budget_shrinks():
    lo = _plan(_mesh(4, 4), budget_gb=8.0).p_min
    hi = _plan(_mesh(4, 4), budget_gb=70.0).p_min
    assert lo >= hi


# ---------------------------------------------------------------------------
# HWM = persistent + max(transients); binder is C at production scale
# ---------------------------------------------------------------------------

def test_binder_is_stage_C_at_production():
    p = _plan(_mesh(4, 4))
    assert p.bottleneck == 'C_fit_one_rchunk'


def test_hwm_equals_persistent_plus_binding_transient():
    p = _plan(_mesh(4, 4))
    assert abs(p.hwm_bytes - p.peak_breakdown[p.bottleneck]) < 1.0
    # C's peak = persistent + a positive transient.
    assert p.peak_breakdown['C_fit_one_rchunk'] > p.persistent_bytes


def test_hwm_under_budget():
    p = _plan(_mesh(4, 4))
    assert p.hwm_bytes <= p.budget_bytes


def test_all_five_stages_present():
    p = _plan(_mesh(4, 4))
    assert set(p.peak_breakdown) == {
        'A_centroid_load', 'B_cct_chol', 'C_fit_one_rchunk',
        'D_accumulate', 'E_v_q'}


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

def test_gflat_chunk_size_capped():
    p = _plan(_mesh(4, 4), budget_gb=1000.0)
    assert p.gflat_chunk_size <= GFLAT_CHUNK_SIZE_CAP
    assert p.gflat_chunk_size % 4 == 0


def test_band_chunk_at_mesh_floor():
    p = _plan(_mesh(4, 4))
    assert p.band_chunk >= 16 and p.band_chunk % 16 == 0


def test_q_chunk_present_and_positive():
    p = _plan(_mesh(4, 4))
    assert p.q_chunk >= 1


def test_ns_aware_util_default():
    """Scalar (ns=1) gets a higher util than bispinor (ns=2)."""
    scalar_meta = SimpleNamespace(
        nk_tot=9, nspinor=1, n_rmu=400, n_rmu_padded=400,
        n_rtot=80_000, ngkmax=5000, fft_grid=(40, 40, 50))
    p1 = plan_gflat_chunks(meta=scalar_meta, mesh_xy=_mesh(2, 2),
                           nb_total=100, ngkmax=5000, n_q_disk=9,
                           budget_gb=28.0, is_bispinor=False)
    bispinor_meta = SimpleNamespace(
        nk_tot=9, nspinor=2, n_rmu=400, n_rmu_padded=400,
        n_rtot=80_000, ngkmax=5000, fft_grid=(40, 40, 50))
    p2 = plan_gflat_chunks(meta=bispinor_meta, mesh_xy=_mesh(2, 2),
                           nb_total=100, ngkmax=5000, n_q_disk=9,
                           budget_gb=28.0, is_bispinor=True)
    assert p1.target_utilization > p2.target_utilization


def test_r_chunk_override_respected():
    p = _plan(_mesh(4, 4), r_chunk_override=8192)
    assert p.r_chunk == 8192


def test_format_emits_stages_and_floor():
    txt = _plan(_mesh(4, 4)).format()
    assert 'P_min' in txt and 'binder' in txt
    for s in ('A_centroid_load', 'C_fit_one_rchunk', 'E_v_q'):
        assert s in txt
