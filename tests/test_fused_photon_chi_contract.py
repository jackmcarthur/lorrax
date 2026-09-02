"""Focused local contracts for fused Lorentz-family response assembly."""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw import w_isdf
from gw.photon_layout import (
    PhotonBasisLayout,
    accumulate_photon_block,
    photon_block_view,
    replace_photon_block,
)


ROOT = Path(__file__).resolve().parents[1]


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / "src/gw/w_isdf.py").read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_packed_block_add_and_replace_are_local_layout_operations():
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    mesh = Mesh(devices, ("x", "y"))
    layout = PhotonBasisLayout.from_centroid_extents(2, 1, mesh)
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    packed = jax.device_put(
        jnp.zeros((1, layout.packed_extent, layout.packed_extent),
                  dtype=jnp.complex128),
        sharding)
    block = jax.device_put(
        jnp.asarray([[[1.0 + 2.0j], [3.0 + 4.0j]]]), sharding)

    packed = accumulate_photon_block(packed, block, layout, 0, 1, mesh)
    packed = accumulate_photon_block(packed, block, layout, 0, 1, mesh)
    np.testing.assert_array_equal(
        np.asarray(photon_block_view(packed, layout, 0, 1, mesh)),
        2.0 * np.asarray(block))

    replacement = jax.device_put(jnp.full(block.shape, 7.0 + 0.0j), sharding)
    packed = replace_photon_block(packed, replacement, layout, 0, 1, mesh)
    np.testing.assert_array_equal(
        np.asarray(photon_block_view(packed, layout, 0, 1, mesh)),
        np.asarray(replacement))
    np.testing.assert_array_equal(
        np.asarray(photon_block_view(packed, layout, 1, 0, mesh)),
        np.zeros((1, 1, 2), dtype=np.complex128))


def test_run270_family_call_count_contract():
    # Only CC exceeds the half-device full-spin workspace budget at the
    # run270 P16 dimensions.  Old per-block execution was 62 calls/tau;
    # family fusion is 38 calls/tau, hence 806 -> 494 at 13 minimax nodes.
    streams = (True, False, False, False)
    assert w_isdf._fused_photon_gemm_calls_per_tau(streams) == 38
    assert 13 * w_isdf._fused_photon_gemm_calls_per_tau(streams) == 494
    old_per_tau = 2 * 16 + 2 * (3 + 3 + 9)
    assert old_per_tau == 62
    assert 13 * old_per_tau == 806


def test_fused_value_parity_with_per_block_oracle(monkeypatch):
    """Tiny P1 proof of all 16 blocks, including streamed CC and TT contact."""
    from types import SimpleNamespace

    import common.fft_helpers as fft_helpers
    import common.gpu_utils as gpu_utils
    import distrib_la
    from gw.photon_layout import pack_photon_response_tiles
    from gw.wavefunction_bundle import (
        BandSlices, PSI_MUN_SPEC, PSI_NMU_SPEC, Wavefunctions)

    def local_fft_factory(mesh, kgrid, spec, *, norm="ortho", out_spec=None):
        # nk=1 in this unit cell, so the normalized k FFT is exactly identity.
        assert tuple(kgrid) == (1, 1, 1)
        return lambda value: value

    def local_gemm_plan(mesh, *, m, k, n, nq, dtype,
                        batched_route="auto", **unused):
        return lambda left, right: jnp.matmul(left, right)

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", local_fft_factory)
    monkeypatch.setattr(distrib_la, "gemm_plan", local_gemm_plan)
    # At nC=3,nT=1 this budget streams CC only: CC=4752 B, CT=1584 B,
    # and the selection compares each against a 3000 B half-budget.
    monkeypatch.setattr(gpu_utils, "get_device_memory_gb", lambda: 6.0e-6)
    w_isdf._chi_minimax_kernel_cache.clear()
    w_isdf._fused_photon_chi_kernel_cache.clear()

    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    mesh = Mesh(devices, ("x", "y"))
    sharding_mun = NamedSharding(mesh, PSI_MUN_SPEC)
    sharding_nmu = NamedSharding(mesh, PSI_NMU_SPEC)
    sharding_rep2 = NamedSharding(mesh, P(None, None))
    rng = np.random.default_rng(20260902)
    slices = BandSlices.from_band_edges(0, 0, 1, 2, 2)
    enk = np.asarray([[-0.5, 0.5]], dtype=np.float64)
    occ = np.asarray([[1.0, 0.0]], dtype=np.float64)

    def bundle(nmu):
        psi = (rng.normal(size=(1, 2, 4, nmu))
               + 1j * rng.normal(size=(1, 2, 4, nmu))).astype(np.complex128)
        return Wavefunctions(
            psi_mun=jax.device_put(psi.transpose(0, 2, 3, 1), sharding_mun),
            psi_nmu=jax.device_put(psi, sharding_nmu),
            enk=jax.device_put(enk, sharding_rep2),
            occ=jax.device_put(occ, sharding_rep2),
            slices=slices, layout="face")

    charge = bundle(3)
    transverse = bundle(1)
    families = (charge, transverse, transverse, transverse)
    quad = SimpleNamespace(
        tau=np.asarray([0.0]), alpha=np.asarray([1.0]))
    meta = SimpleNamespace(nkx=1, nky=1, nkz=1, nk_tot=1)
    layout = PhotonBasisLayout.from_centroid_extents(3, 1, mesh)

    tiles = {}
    for A in range(4):
        for B in range(4):
            block = w_isdf.compute_no_pair_dirac_current_block(
                families[A], families[B], quad, meta, mesh,
                vertex_left=A, vertex_right=B,
                spin_pair_stream=(A == 0 and B == 0))
            tiles[A, B] = (w_isdf._subtract_static_tt_contact(block)
                           if A and B else block)
    oracle = pack_photon_response_tiles(tiles, 1, layout, mesh)
    fused = w_isdf.compute_experimental_no_pair_photon_chi0(
        charge, transverse, quad, meta, mesh, layout)
    np.testing.assert_allclose(
        np.asarray(fused), np.asarray(oracle), rtol=2.0e-13, atol=2.0e-13)


def test_production_uses_one_fused_path_and_keeps_block_oracle():
    production = _function("compute_experimental_no_pair_photon_chi0")
    calls = [_call_name(node) for node in ast.walk(production)
             if isinstance(node, ast.Call)]
    assert calls.count("_get_fused_photon_chi_kernel") == 1
    assert "compute_no_pair_dirac_current_block" not in calls
    assert "pack_photon_operator" not in calls

    fused = _function("_get_fused_photon_chi_kernel")
    fused_calls = [_call_name(node) for node in ast.walk(fused)
                   if isinstance(node, ast.Call)]
    assert fused_calls.count("gemm_plan") == 1  # executed for four families
    assert "accumulate_photon_block" in fused_calls
    assert "photon_block_view" in fused_calls
    assert "replace_photon_block" in fused_calls

    scans = [node for node in ast.walk(fused)
             if isinstance(node, ast.Call) and _call_name(node) == "scan"]
    assert scans
    assert all(any(kw.arg == "unroll"
                   and isinstance(kw.value, ast.Constant)
                   and kw.value.value == 1 for kw in call.keywords)
               for call in scans)

    # The retained public oracle still dispatches the incumbent per-block
    # face kernel, so fused value parity can be checked independently.
    oracle = _function("compute_no_pair_dirac_current_block")
    oracle_calls = [_call_name(node) for node in ast.walk(oracle)
                    if isinstance(node, ast.Call)]
    assert "_get_chi_minimax_kernel" in oracle_calls
