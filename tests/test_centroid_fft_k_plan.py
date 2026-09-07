"""Focused gate for the GW planner -> centroid k-stream seam.

The run55 failure was one unbounded local FFT-row batch, not a new transform
algorithm.  These cells pin only the two facts that falsify that defect: the
planner's P-scaling and ownership of a non-divisible terminal k tile.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from gw.gflat_memory_model import (
    centroid_fft_tile_geometry,
    plan_gflat_chunks,
)


def _plan(p_x: int, p_y: int):
    meta = SimpleNamespace(
        nk_tot=36,
        nspinor=2,
        n_rmu=64,
        n_rmu_padded=64,
        n_rtot=8 * 8 * 8,
        fft_grid=(8, 8, 8),
    )
    mesh = SimpleNamespace(shape={"x": p_x, "y": p_y})
    return plan_gflat_chunks(
        meta=meta,
        mesh_xy=mesh,
        nb_total=64,
        fit_nb_total=64,
        ngkmax=64,
        n_q_disk=36,
        budget_gb=1000.0,
        band_chunk_override=16,
        pair_density_slots=3,
        low_mem_bands=True,
    )


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_stage_a_bounds_local_rows_at_p1_and_p16():
    """The shipping band tile means 16 local FFT rows on either mesh."""
    p1 = _plan(1, 1)
    p16 = _plan(4, 4)

    assert (p1.band_chunk, p1.centroid_k_chunk,
            p1.centroid_fft_rows_local) == (16, 1, 16)
    assert (p16.band_chunk, p16.centroid_k_chunk,
            p16.centroid_fft_rows_local) == (16, 16, 16)

    # CrI3 nk=36 therefore has two complete 16-k tiles and a four-k tail.
    # The physical carrier stays fixed at 16; the loader must own the pad.
    assert divmod(36, p16.centroid_k_chunk) == (2, 4)
    # Reuse contracts can start from an unrounded deck hint; the same owner
    # resolves its physical band tile through runtime.padding.round_up.
    assert centroid_fft_tile_geometry(
        nk=36, band_chunk=7, p_band=4) == (4, 8)


def test_planned_k_tile_reaches_the_one_fixed_shape_padding_owner():
    """Charge/current callers pass the plan; the common loader pads the tail."""
    repo = Path(__file__).resolve().parents[1]
    gw_tree = ast.parse((repo / "src/gw/gw_init.py").read_text())
    wfn_tree = ast.parse((repo / "src/common/wfn_transforms.py").read_text())
    pivot_tree = ast.parse(
        (repo / "src/centroid/pivoted_cholesky.py").read_text())

    planner = _function(gw_tree, "_plan_gflat_chunks_for_channel")
    planner_dicts = [
        node for node in ast.walk(planner)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "centroid_k_chunk"
            for key in node.keys
        )
    ]
    assert len(planner_dicts) == 1
    planned_value = next(
        value for key, value in zip(planner_dicts[0].keys, planner_dicts[0].values)
        if isinstance(key, ast.Constant) and key.value == "centroid_k_chunk"
    )
    assert ast.unparse(planned_value) == "int(gflat_plan.centroid_k_chunk)"

    prepare = _function(gw_tree, "_prepare_fresh_parent_faces")
    charge_calls = _calls(prepare, "load_centroids_band_chunked")
    assert len(charge_calls) == 1
    charge_k = next(
        kw.value for kw in charge_calls[0].keywords if kw.arg == "k_chunk_size"
    )
    assert "chunks['centroid_k_chunk']" in ast.unparse(charge_k)
    assert "zeta_contract.loader_k_chunk" in ast.unparse(charge_k)

    transverse = _function(gw_tree, "_transverse_wfn_data")
    transverse_calls = _calls(transverse, "load_centroids_band_chunked")
    assert len(transverse_calls) == 1
    transverse_k = next(
        kw.value for kw in transverse_calls[0].keywords
        if kw.arg == "k_chunk_size"
    )
    assert ast.unparse(transverse_k) == "k_chunk_size"

    contract = _function(gw_tree, "_resolve_zeta_fit_contract")
    geometry_calls = _calls(contract, "centroid_fft_tile_geometry")
    assert len(geometry_calls) == 1
    divisor_calls = _calls(contract, "spec_divisor")
    assert len(divisor_calls) == 1
    assert ast.unparse(divisor_calls[0]) == (
        "spec_divisor(mesh_xy, band_sphere_spec(), axis=1)"
    )
    contract_k = next(
        kw.value for call in _calls(contract, "_ZetaFitContract")
        for kw in call.keywords if kw.arg == "loader_k_chunk"
    )
    assert ast.unparse(contract_k) == "loader_k_chunk"

    loader_entry = _function(wfn_tree, "load_centroids_band_chunked")
    for stage in ("_centroid_stream_geometry", "_centroid_resident_bytes",
                  "_load_streamed_centroid_faces"):
        assert len(_calls(loader_entry, stage)) == 1
    owners = {"_centroid_stream_geometry", "_centroid_resident_bytes",
              "_sample_centroid_domain_tiles", "_centroid_fft_scan_chunk"}
    loader = ast.Module(body=[n for n in wfn_tree.body
                             if isinstance(n, ast.FunctionDef)
                             and n.name in owners], type_ignores=[])
    gram = _function(pivot_tree, "build_gram_q0_via_loadwfns")
    assert len(_calls(loader, "worst_process_resident_bytes")) == 1
    assert len(_calls(gram, "worst_process_resident_bytes")) == 1
    pad_inputs = {
        ast.unparse(call.args[0])
        for call in _calls(loader, "pad_axis")
        if any(kw.arg == "axis" and ast.unparse(kw.value) == "0"
               for kw in call.keywords)
    }
    assert {
        "psi_G_tile",
        "g_index_full[k0:k1]",
        "jnp.asarray(kvecs_frac_full[k0:k1])",
    } <= pad_inputs

    assignments = {
        target.id: ast.unparse(node.value)
        for node in ast.walk(loader)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["existing_live_bytes"] == (
        "worst_process_resident_bytes(existing_live_local_bytes)"
    )
    assert assignments["nk_accum"] == (
        "padded_axis(nk_tot, k_tile, name='centroid stream k "
        "accumulator').carrier"
    )
