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


def _mesh():
    side = 2 if len(jax.devices()) >= 4 else 1
    devices = np.asarray(
        jax.devices()[:side * side], dtype=object).reshape(side, side)
    return Mesh(devices, ("x", "y"))


def test_packed_block_add_and_replace_are_local_layout_operations():
    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(3, 2, mesh)
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    packed = jax.device_put(
        jnp.zeros((1, layout.packed_extent, layout.packed_extent),
                  dtype=jnp.complex128),
        sharding)
    block_shape = layout.block_shape(1, 0, 1)
    block = jax.device_put(
        jnp.arange(np.prod(block_shape), dtype=jnp.float64).reshape(block_shape)
        + (1.0 + 2.0j), sharding)

    packed = accumulate_photon_block(packed, block, layout, 0, 1, mesh)
    packed = accumulate_photon_block(packed, block, layout, 0, 1, mesh)
    expected = 2.0 * np.asarray(block)
    expected[:, layout.logical_extent(0):, :] = 0
    expected[:, :, layout.logical_extent(1):] = 0
    np.testing.assert_array_equal(
        np.asarray(photon_block_view(packed, layout, 0, 1, mesh)), expected)

    replacement = jax.device_put(jnp.full(block.shape, 7.0 + 0.0j), sharding)
    packed = replace_photon_block(packed, replacement, layout, 0, 1, mesh)
    expected_replacement = np.asarray(replacement).copy()
    expected_replacement[:, layout.logical_extent(0):, :] = 0
    expected_replacement[:, :, layout.logical_extent(1):] = 0
    np.testing.assert_array_equal(
        np.asarray(photon_block_view(packed, layout, 0, 1, mesh)),
        expected_replacement)
    np.testing.assert_array_equal(
        np.asarray(photon_block_view(packed, layout, 1, 0, mesh)),
        np.zeros(layout.block_shape(1, 1, 0), dtype=np.complex128))


def test_streamed_family_call_and_flop_contracts():
    # P16/run270 streamed CC only: 806 -> 494 plan calls over 13 nodes.
    assert w_isdf._fused_photon_gemm_calls_per_tau(
        (True, False, False, False)) == 38

    # P64/run272 live evidence selects streamed CC/CT/TC and full TT.
    streams = (True, True, True, False)
    assert w_isdf._fused_photon_gemm_calls_per_tau(streams) == 162
    assert 13 * w_isdf._fused_photon_gemm_calls_per_tau(streams) == 2106
    incumbent_per_tau = 32 + 3 * 32 + 3 * 32 + 9 * 2
    assert incumbent_per_tau == 242
    assert 13 * incumbent_per_tau == 3146

    # Exact padded run272 extents.  Coefficients count m*n per plan call;
    # nq, k and the complex-MAC convention cancel in the ratio.
    n_charge, n_transverse = 6016, 2048
    incumbent_work = (
        32 * n_charge**2
        + 192 * n_charge * n_transverse
        + 288 * n_transverse**2)
    fused_work = (
        32 * n_charge**2
        + 128 * n_charge * n_transverse
        + 32 * n_transverse**2)
    assert incumbent_work == 4_731_699_200
    assert fused_work == 2_869_428_224
    assert np.isclose(1.0 - fused_work / incumbent_work, 0.3935734072022161)


def test_fused_factory_cache_separates_ffi_backend_dials(monkeypatch):
    import common.fft_helpers as fft_helpers
    import distrib_la
    import ffi

    monkeypatch.setattr(
        fft_helpers, "make_flat_k_fftn",
        lambda mesh, kgrid, spec, **kwargs: (lambda value: value))
    monkeypatch.setattr(
        distrib_la, "gemm_plan",
        lambda *args, **kwargs: (lambda left, right: jnp.matmul(left, right)))
    dial = ["backend-a"]
    monkeypatch.setattr(ffi, "ffi_dial_key", lambda: dial[0])
    w_isdf._fused_photon_chi_kernel_cache.clear()

    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(3, 1, mesh)
    charge_shape = (1, 2, layout.padded_extent(0), 4)
    transverse_shape = (1, 2, layout.padded_extent(1), 4)

    def factory():
        return w_isdf._get_fused_photon_chi_kernel(
            mesh, (1, 1, 1), layout, charge_shape, transverse_shape,
            stream_by_family=(False, False, False, False),
            distrib_la_batched_route="auto")

    kernel_a = factory()
    assert factory() is kernel_a
    dial[0] = "backend-b"
    kernel_b = factory()
    assert kernel_b is not kernel_a
    assert len(w_isdf._fused_photon_chi_kernel_cache) == 2


def test_fused_value_against_independent_dense_band_sum(monkeypatch):
    """Three-q/two-tau all-16 tolerance proof independent of gamma gathers."""
    from types import SimpleNamespace

    import common.fft_helpers as fft_helpers
    import common.gpu_utils as gpu_utils
    import distrib_la
    from gw.wavefunction_bundle import (
        BandSlices, PSI_MUN_SPEC, PSI_NMU_SPEC, Wavefunctions)

    def local_fft_factory(mesh, kgrid, spec, *, norm="ortho", out_spec=None):
        def transform(value):
            grid = jnp.reshape(value, tuple(kgrid) + value.shape[1:])
            grid = jnp.fft.fftn(grid, axes=(0, 1, 2), norm=norm)
            return jnp.reshape(grid, value.shape)
        return transform

    def local_gemm_plan(mesh, *, m, k, n, nq, dtype,
                        batched_route="auto", **unused):
        return lambda left, right: jnp.matmul(left, right)

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", local_fft_factory)
    monkeypatch.setattr(distrib_la, "gemm_plan", local_gemm_plan)
    mesh = _mesh()
    logical_c, logical_t = 3, 1
    layout = PhotonBasisLayout.from_centroid_extents(
        logical_c, logical_t, mesh)
    n_c = layout.padded_extent(0)
    n_t = layout.padded_extent(1)
    p = int(mesh.devices.size)
    full_cc = 33 * 3 * n_c * n_c * 16 // p
    full_ct = 33 * 3 * n_c * n_t * 16 // p
    if full_cc == full_ct:
        # Emulated P4 padding makes both tiny families equal; stream all four
        # to exercise the fused CT/TC/TT one-pair path and distributed layout.
        half_budget = 1
    else:
        # Ordinary local P1: streamed CC, full CT/TC/TT.
        half_budget = (full_cc + full_ct) // 2
    monkeypatch.setattr(
        gpu_utils, "get_device_memory_gb", lambda: 2 * half_budget / 1e9)
    w_isdf._chi_minimax_kernel_cache.clear()
    w_isdf._fused_photon_chi_kernel_cache.clear()

    sharding_mun = NamedSharding(mesh, PSI_MUN_SPEC)
    sharding_nmu = NamedSharding(mesh, PSI_NMU_SPEC)
    sharding_rep2 = NamedSharding(mesh, P(None, None))
    rng = np.random.default_rng(20260902)
    slices = BandSlices.from_band_edges(0, 0, 1, 2, 2)
    nk = 3
    enk = np.tile(np.asarray([[-0.5, 0.5]], dtype=np.float64), (nk, 1))
    occ = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (nk, 1))

    def bundle(nmu, logical_nmu):
        psi = (rng.normal(size=(nk, 2, 4, nmu))
               + 1j * rng.normal(size=(nk, 2, 4, nmu))).astype(np.complex128)
        psi[..., logical_nmu:] = 0
        wfns = Wavefunctions(
            psi_mun=jax.device_put(psi.transpose(0, 2, 3, 1), sharding_mun),
            psi_nmu=jax.device_put(psi, sharding_nmu),
            enk=jax.device_put(enk, sharding_rep2),
            occ=jax.device_put(occ, sharding_rep2),
            slices=slices, layout="face")
        return wfns, psi

    charge, psi_c = bundle(n_c, logical_c)
    transverse, psi_t = bundle(n_t, logical_t)
    families = (charge, transverse, transverse, transverse)
    psi_families = (psi_c, psi_t, psi_t, psi_t)
    tau = np.asarray([0.0, 0.7])
    # Gap is exactly one.  The production prefactor contributes exp(-tau),
    # so these two positive weights integrate the unit-gap denominator to 1.
    quad = SimpleNamespace(
        tau=tau, alpha=np.asarray([0.6, 0.4 * np.exp(0.7)]))
    meta = SimpleNamespace(nkx=nk, nky=1, nkz=1, nk_tot=nk)
    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
    gammas = tuple(np.asarray(jax.device_get(gamma))
                   for gamma in (gamma0, gamma1, gamma2, gamma3))
    extents = (n_c, n_t, n_t, n_t)
    expected = {}
    for A in range(4):
        psi_a = psi_families[A]
        for B in range(4):
            psi_b = psi_families[B]
            block = np.zeros((nk, extents[A], extents[B]), np.complex128)
            for q in range(nk):
                for k in range(nk):
                    kmq = (k - q) % nk
                    left_vc = np.einsum(
                        "am,aA,Am->m", psi_a[k, 0], gammas[A],
                        np.conj(psi_a[kmq, 1]), optimize=True)
                    right_vc = np.einsum(
                        "bn,Bb,Bn->n", np.conj(psi_b[k, 0]), gammas[B],
                        psi_b[kmq, 1], optimize=True)
                    left_cv = np.einsum(
                        "am,aA,Am->m", psi_a[k, 1], gammas[A],
                        np.conj(psi_a[kmq, 0]), optimize=True)
                    right_cv = np.einsum(
                        "bn,Bb,Bn->n", np.conj(psi_b[k, 1]), gammas[B],
                        psi_b[kmq, 0], optimize=True)
                    block[q] -= (
                        left_vc[:, None] * right_vc[None, :]
                        + left_cv[:, None] * right_cv[None, :]
                    ) / np.sqrt(float(nk))
            if A and B:
                block = block - block[0:1]
                block[0] = 0
            expected[A, B] = block

    fused = w_isdf.compute_experimental_no_pair_photon_chi0(
        charge, transverse, quad, meta, mesh, layout)
    wanted_sharding = NamedSharding(mesh, P(None, "x", "y"))
    assert fused.sharding.is_equivalent_to(wanted_sharding, fused.ndim)
    got = {}
    for A in range(4):
        for B in range(4):
            got[A, B] = np.asarray(
                photon_block_view(fused, layout, A, B, mesh))
            np.testing.assert_allclose(
                got[A, B], expected[A, B], rtol=2.0e-12, atol=2.0e-12)
    assert np.max(np.abs(got[1, 1][1:, :logical_t, :logical_t])) > 1.0e-6

    # Fused production CC streams in this fixture.  Its regrouped pairwise
    # packed accumulation has tolerance parity with the dense oracle above,
    # not a bit-identity contract.  Only the forced-nonstream incumbent route
    # below retains its established bit-equality claim.
    cc_vertex = w_isdf.compute_no_pair_dirac_current_block(
        charge, charge, quad, meta, mesh,
        vertex_left=0, vertex_right=0, spin_pair_stream=False)
    cc_charge = w_isdf.compute_chi0(charge, quad, meta, mesh)
    assert np.array_equal(np.asarray(cc_vertex), np.asarray(cc_charge))

    # Charge/mixed blocks retain the full-grid q <-> -q relation.  TT q=0
    # subtraction is an explicitly separate approximation and is checked by
    # the dense expected tensor above instead of being assumed reciprocal.
    neg = np.asarray([0, 2, 1])
    for A, B in ((0, 0), (0, 1), (0, 2), (0, 3),
                 (1, 0), (2, 0), (3, 0)):
        np.testing.assert_allclose(
            got[A, B], np.conj(got[A, B][neg]),
            rtol=2.0e-12, atol=2.0e-12)


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
    assert fused_calls.count("ffi_dial_key") == 1
    assert "accumulate_photon_block" in fused_calls
    assert "photon_block_view" in fused_calls
    assert "replace_photon_block" in fused_calls

    shared = _function("_streamed_face_vertex_orientations")
    shared_calls = [_call_name(node) for node in ast.walk(shared)
                    if isinstance(node, ast.Call)]
    assert shared_calls.count("build_gc_pair") == 1
    incumbent = _function("_get_chi_minimax_kernel_face")
    incumbent_calls = [_call_name(node) for node in ast.walk(incumbent)
                       if isinstance(node, ast.Call)]
    assert incumbent_calls.count("_streamed_face_vertex_orientations") == 1
    assert fused_calls.count("_streamed_face_vertex_orientations") == 1

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


def test_real_p4_gate_registers_fused_full_and_batch_reshard_cells():
    gate = (ROOT / "tests/multi_device/"
            "bispinor_transverse_vertex_face_gate.py").read_text()
    assert '("fused_photon_full_all16",' in gate
    assert '("fused_photon_batch_reshard_all16",' in gate
    assert gate.count("n_c=8, n_t=12, fused=True") == 2
    fused_batch = gate[gate.index('(\"fused_photon_batch_reshard_all16\",'):]
    assert 'distrib_la_batched_route="batch_reshard"' in fused_batch
    assert "packed.sharding.is_equivalent_to(" in gate
    assert "block.sharding.is_equivalent_to(" in gate
