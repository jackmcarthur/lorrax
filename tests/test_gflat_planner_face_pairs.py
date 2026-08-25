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


def _meta(*, ns=4, mu=2400, nr=75 * 75 * 250, ngkmax=None):
    values = dict(
        nk_tot=36, nspinor=ns, n_rmu=mu, n_rmu_padded=mu,
        n_rtot=nr, fft_grid=(75, 75, 250))
    if ngkmax is not None:
        values["ngkmax"] = int(ngkmax)
    return SimpleNamespace(**values)


def _synthetic_plan(*, ns=4, face_nb=256, fit_nb=250, budget=80.0,
                    r_override=None, mesh=(4, 4)):
    return plan_gflat_chunks(
        meta=_meta(ns=ns, ngkmax=30_000),
        mesh_xy=SimpleNamespace(shape={'x': mesh[0], 'y': mesh[1]}),
        nb_total=380, face_nb_total=face_nb, fit_nb_total=fit_nb,
        ngkmax=30_000, n_q_disk=36, budget_gb=budget,
        target_utilization=0.80, band_chunk_override=16,
        r_chunk_override=r_override,
        distributed_zeta_solve="distributed", low_mem_bands=True)


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


def test_four_total_rank3_slots_have_one_owner_and_no_double_charge():
    assert _face_pair_density_slots() == 4
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


def test_ns1_and_large_band_envelope_keep_repeated_fallback():
    assert not _synthetic_plan(ns=1).cache_face_y_blocks
    large = _synthetic_plan(face_nb=4096, fit_nb=4096, budget=80.0)
    assert not large.cache_face_y_blocks
    assert large.face_y_cache_bytes == 0


def test_face_uses_exact_carrier_inventory_and_only_py_r_alignment():
    plan = _synthetic_plan(
        face_nb=128, r_override=1028, mesh=(2, 4), budget=200.0)
    expected = 2 * 36 * 4 * 2400 * 128 * 16 / 8
    assert plan.psi_layout_bytes == expected
    assert plan.r_chunk == 1028  # divisible by Py=4, deliberately not P=8
