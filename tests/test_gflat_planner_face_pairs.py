"""Focused shape-contract tests for the scalar-pair face Stage-C plan."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gw.gflat_memory_model import (
    _face_pair_density_slots,
    _fft_box_bytes,
    plan_gflat_chunks,
)
from gw.gw_init import _plan_gflat_chunks_for_channel
from gw.wavefunction_bundle import BandSlices
from runtime.padding import bounded_partition_tile


def _meta(*, ns=4, mu=2400, nr=75 * 75 * 250, ngkmax=None):
    values = dict(
        nk_tot=36, nspinor=ns, n_rmu=mu, n_rmu_padded=mu,
        n_rtot=nr, fft_grid=(75, 75, 250))
    if ngkmax is not None:
        values["ngkmax"] = int(ngkmax)
    return SimpleNamespace(**values)


def _synthetic_plan(*, ns=4, face_nb=256, fit_nb=250, budget=80.0,
                    r_override=None, mesh=(4, 4), current_vertex=False):
    return plan_gflat_chunks(
        meta=_meta(ns=ns, ngkmax=30_000),
        mesh_xy=SimpleNamespace(shape={'x': mesh[0], 'y': mesh[1]}),
        nb_total=380, face_nb_total=face_nb, fit_nb_total=fit_nb,
        ngkmax=30_000, n_q_disk=36, budget_gb=budget,
        target_utilization=0.80, band_chunk_override=16,
        r_chunk_override=r_override,
        distributed_zeta_solve="distributed", low_mem_bands=True,
        face_current_vertex=current_vertex)


def _run50_plan():
    """Exact run50 deck/caller shape facts; no measured-FFT claim."""
    cfg = SimpleNamespace(
        zeta_nband=250,
        memory=SimpleNamespace(
            per_device_gb=60.0, chunk_target_utilization=0.0,
            band_chunk_size=16, r_chunk_override=0,
            gflat_chunk_size=0, low_mem_bands=True),
        backend=SimpleNamespace(distributed_zeta_solve="distributed"))
    bands = BandSlices.from_band_edges(0, 0, 130, 190, 256)
    mesh = SimpleNamespace(
        shape={'x': 4, 'y': 4}, devices=np.empty(16, dtype=object))
    _, plan = _plan_gflat_chunks_for_channel(
        meta=_meta(), cfg=cfg, band_slices=bands, mesh_xy=mesh,
        is_bispinor=True, print_fn=lambda *_: None)
    return plan


def _profile_cliff_plan(r_chunk):
    """Exact run158 current-channel geometry, with analytic FFT fallback."""
    return plan_gflat_chunks(
        meta=_meta(ns=4, mu=800, ngkmax=76_551),
        mesh_xy=SimpleNamespace(shape={'x': 4, 'y': 4}),
        nb_total=380, face_nb_total=256, fit_nb_total=250,
        ngkmax=76_551, n_q_disk=36, budget_gb=70.0,
        target_utilization=0.78, band_chunk_override=16,
        r_chunk_override=r_chunk,
        distributed_zeta_solve="distributed", low_mem_bands=True,
        face_current_vertex=True)


def test_run50_matched_deck_selects_bounded_y_cache_without_full_grid_cache():
    plan = _run50_plan()
    assert not plan.cache_psi_r
    assert plan.cache_face_y_blocks
    assert plan.face_y_cache_bytes > 0
    assert plan.r_chunk % 4 == 0
    assert plan.budget_bytes == 60.0e9
    assert plan.target_utilization == 0.78
    assert plan.centroid_fft_bytes == 5_760_000_000
    assert plan.zeta_transform_fft_bytes == 12_960_000_000
    assert plan.psi_layout_bytes == 176_947_200
    assert plan.persistent_bytes == 7_925_407_200
    assert plan.p_min == 8
    assert plan.r_chunk == 46_220
    assert plan.n_r_chunks == 31
    assert plan.face_y_cache_bytes == 6_815_416_320
    assert plan.peak_breakdown["A_centroid_load"] == 13_685_407_200
    assert plan.peak_breakdown["C_fit_one_rchunk"] == 30_825_047_520
    assert plan.peak_breakdown["C_face_y_cache_build"] == 28_769_833_440
    assert "C_face_y_cache_build" in plan.peak_breakdown


def test_cri3_prices_bounded_centroid_and_full_k_zeta_ffts_separately():
    mesh = SimpleNamespace(shape={'x': 4, 'y': 4})
    fft_args = dict(
        bc=16, ns=4, fft_grid=(75, 75, 250), mesh_xy=mesh, p_xy=16)
    assert _fft_box_bytes(nk=16, **fft_args) == 5_760_000_000
    assert _fft_box_bytes(nk=36, **fft_args) == 12_960_000_000

    plan = _run50_plan()
    persistent = plan.persistent_bytes
    assert plan.peak_breakdown["A_centroid_load"] - persistent == 5_760_000_000
    assert plan.centroid_fft_bytes == 5_760_000_000
    assert plan.zeta_transform_fft_bytes == 12_960_000_000
    # Cache-free run50 executes the full-nk transform inside the separate
    # current-r Y-cache build peak, never inside Stage A's centroid tile.
    assert plan.peak_breakdown["C_face_y_cache_build"] > (
        persistent + 12_960_000_000)

    hoisted = _synthetic_plan(budget=80.0)
    assert hoisted.cache_psi_r
    assert (hoisted.peak_breakdown["A_centroid_load"]
            - hoisted.persistent_bytes) == 5_760_000_000
    assert (hoisted.peak_breakdown["A_psi_r_cache_build"]
            - hoisted.persistent_bytes) == 12_960_000_000


def test_face_pair_arena_distinguishes_charge_and_current_executables():
    assert _face_pair_density_slots(ns=4, current_vertex=False) == 4
    assert _face_pair_density_slots(ns=4, current_vertex=True) == 16
    charge = _synthetic_plan(
        current_vertex=False, r_override=16_000, budget=200.0)
    current = _synthetic_plan(
        current_vertex=True, r_override=16_000, budget=200.0)
    assert charge.face_x_owner_cache_bytes == 0
    assert current.face_x_owner_cache_bytes > 0
    assert (current.peak_breakdown["C_fit_one_rchunk"]
            > charge.peak_breakdown["C_fit_one_rchunk"])
    tree = ast.parse(
        (Path(__file__).resolve().parents[1]
         / "src/gw/gflat_memory_model.py").read_text())
    owners = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_face_pair_density_slots"
    ]
    assert len(owners) == 1
    face_terms = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_stage_C_face_terms"
    )
    source = ast.unparse(face_terms)
    assert 'pair_peak = float(slots) * pair_rank3' in source
    assert 'z_slope' not in source
    assert 'kfft_scratch_slope' not in source


def test_x_owner_cache_is_current_cached_route_only():
    current = _profile_cliff_plan(41_472)
    # The helper above is the accepted current geometry; independently price
    # an otherwise matched charge executable to prove identity stays at zero.
    charge = plan_gflat_chunks(
        meta=_meta(ns=4, mu=800, ngkmax=76_551),
        mesh_xy=SimpleNamespace(shape={'x': 4, 'y': 4}),
        nb_total=380, face_nb_total=256, fit_nb_total=250,
        ngkmax=76_551, n_q_disk=36, budget_gb=70.0,
        target_utilization=0.78, band_chunk_override=16,
        r_chunk_override=41_472,
        distributed_zeta_solve="distributed", low_mem_bands=True,
        face_current_vertex=False)
    assert charge.cache_face_y_blocks
    assert current.cache_face_y_blocks
    assert charge.face_x_owner_cache_bytes == 0
    assert current.face_x_owner_cache_bytes == 117_964_800

    core_source = (
        Path(__file__).resolve().parents[1] / "src/isdf/core.py").read_text()
    assert (
        "cache_x_owner_blocks = bool(cache_y_blocks and not lhs_id)"
        in core_source)
    assert "def load_x_owner_block(bc_idx):" in core_source


def test_ns1_falls_back_and_large_band_selects_cache_feasible_outer_chunk():
    assert not _synthetic_plan(ns=1).cache_face_y_blocks
    large = _synthetic_plan(face_nb=4096, fit_nb=4096, budget=80.0)
    assert large.cache_face_y_blocks
    assert large.face_y_cache_r_tile == large.r_chunk
    assert large.r_chunk % large.face_y_cache_r_tile == 0
    # Auto planning prefers the largest cache-feasible outer chunk to the
    # larger repeated-transform route.  Explicit larger chunks can instead
    # keep their solve width through the tiled-cache path tested below.
    assert large.face_y_cache_bytes > 0


def test_face_uses_exact_carrier_inventory_and_only_py_r_alignment():
    plan = _synthetic_plan(
        face_nb=128, r_override=1028, mesh=(2, 4), budget=200.0)
    expected = 2 * 36 * 4 * 2400 * 128 * 16 / 8
    assert plan.psi_layout_bytes == expected
    assert plan.r_chunk == 1028  # divisible by Py=4, deliberately not P=8


def test_bounded_partition_tile_preserves_outer_extent_and_alignment():
    assert bounded_partition_tile(82_944, 41_472, 4) == 41_472
    assert bounded_partition_tile(60, 16, 4) == 12
    assert bounded_partition_tile(44, 7, 4) == 4
    assert bounded_partition_tile(44, 3, 4) == 0
    assert bounded_partition_tile(45, 16, 4) == 0


def test_run158_cliff_uses_two_41472_tiles_and_prices_compact_redistribution():
    full = _profile_cliff_plan(41_472)
    tiled = _profile_cliff_plan(82_944)
    assert full.face_y_cache_r_tile == 41_472
    assert tiled.r_chunk == 82_944
    assert tiled.face_y_cache_r_tile == 41_472
    assert tiled.cache_face_y_blocks
    assert tiled.face_y_cache_bytes == full.face_y_cache_bytes == 6_115_295_232
    completed_z_tile = 1_194_393_600
    # Pair/cache work is tile-sized.  The only incremental live set while
    # tile 1 runs is tile 0's completed all-P compact Z_q output.
    assert (tiled.peak_breakdown["C_fit_one_rchunk"]
            - full.peak_breakdown["C_fit_one_rchunk"]
            == completed_z_tile)
    # The canonical Y source/scatter/cache transaction is also tile-sized:
    # relative to the one-tile arm, the build peak grows only by completed
    # compact Z—not by another outer-width Y block.
    assert (tiled.peak_breakdown["C_face_y_cache_build"]
            - full.peak_breakdown["C_face_y_cache_build"]
            == completed_z_tile)
    assert tiled.peak_breakdown["C_face_y_cache_build"] == 23_817_765_600
    assert tiled.peak_breakdown["C_face_tile_concat"] == 7_310_858_976
    assert tiled.hwm_bytes == 29_108_099_808
