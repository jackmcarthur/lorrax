"""The streamed Stage-C ζ transform is priced over k tiles.

Refusal this gates (JID 58786388 step .0, CrI3 16x16 bispinor charge fit at
P64): the planner priced the cache-free ψ(G)->ψ(r) box at every k row at
once, a P-independent 105 GB (analytic) / 52 GB (measured) term, so
``P_min`` ran to the 2**20 sentinel and the only "fitting" r chunk was 8.
The plan prices the ψ(G) store's k rows (the 30 raw parents there) in
``gw.gflat_memory_model.zeta_fft_k_tile`` tiles.  (The r-tile executor that
transformed in those tiles is retired; the ζ fit runs on route G.)
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_zeta_k_tile_rule_is_the_centroid_bound_snapped_to_a_divisor():
    from gw.gflat_memory_model import zeta_fft_k_tile
    rule = zeta_fft_k_tile
    assert rule(n_k_rows=256, band_chunk=64, p_band=64) == 64
    assert rule(n_k_rows=256, band_chunk=512, p_band=64) == 64
    assert rule(n_k_rows=30, band_chunk=64, p_band=64) == 30
    assert rule(n_k_rows=30, band_chunk=16, p_band=16) == 15
    assert rule(n_k_rows=36, band_chunk=16, p_band=16) == 12
    assert rule(n_k_rows=37, band_chunk=16, p_band=16) == 1


def _cri3_bispinor_p64_plan():
    """Production inputs of JID 58786388 step .0 (the charge fit).

    Calibrated to its refusal receipt: persistent 6.00, B 40.34, E 8.42,
    F 3.85 GB/dev (mu_pad 4032 = 3998 on the 8x8 mesh, ngkmax 96000,
    face carrier 488, fit window 481, 30 raw parents / selected q).
    """
    from gw.gw_init import _plan_gflat_chunks_for_channel
    from gw.wavefunction_bundle import BandSlices

    meta = SimpleNamespace(
        nk_tot=256, nspinor=4, n_rmu=3998, n_rmu_padded=4032,
        n_rtot=80 * 80 * 250, fft_grid=(80, 80, 250))
    cfg = SimpleNamespace(
        zeta_nband=481,
        memory=SimpleNamespace(
            per_device_gb=70.32, chunk_target_utilization=0.0,
            band_chunk_size=16, r_chunk_override=0, gflat_chunk_size=0,
            low_mem_bands=False),
        backend=SimpleNamespace(distributed_zeta_solve="auto"))
    mesh = SimpleNamespace(
        shape={'x': 8, 'y': 8}, devices=np.empty(64, dtype=object))
    _, plan = _plan_gflat_chunks_for_channel(
        meta=meta, cfg=cfg,
        band_slices=BandSlices.from_band_edges(0, 0, 130, 183, 488),
        mesh_xy=mesh, is_bispinor=True, n_q_selected=30,
        parent_route=dict(n_parent=30, parents_only=True),
        print_fn=lambda *_: None)
    return plan


def test_cri3_bispinor_charge_fit_plans_at_p64():
    """Analytic (4x, conservative) FFT pricing; no refusal is raised."""
    plan = _cri3_bispinor_p64_plan()
    assert not plan.cache_psi_r
    assert abs(plan.persistent_bytes - 6.00e9) < 0.005e9  # the receipt's
    assert plan.zeta_k_chunk == 30
    # 30 k x 1 band/rank x 4 spinor x 1.6e6 r x 16 B x 4.0 analytic.
    assert plan.zeta_transform_fft_bytes == 12_288_000_000
    assert plan.p_min <= 64
    assert plan.hwm_bytes <= plan.budget_bytes * plan.target_utilization
    # A usable r chunk: at least the mu-wide performance floor.
    assert plan.r_chunk >= 4032 and plan.n_r_chunks <= 400, plan.format()
    assert "zeta FFT      = k_tile 30 (streamed)" in plan.format()


def test_planner_prices_the_one_tile_rule():
    repo = Path(__file__).resolve().parents[1]
    model = ast.parse((repo / "src/gw/gflat_memory_model.py").read_text())
    planner = next(n for n in model.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "plan_gflat_chunks")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "zeta_fft_k_tile" for n in ast.walk(planner))
