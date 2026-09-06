"""GW adapter contracts for raw-parent centroid operators."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.grouped_layout import build_square_grouped_shard_layout
from common.centroid_basis import PackedCentroidBasis
from gw.gw_init import _prepare_parent_wavefunction_plan
from gw.centroid_k_unfold import (
    CentroidKUnfoldPlan,
    build_centroid_k_unfold_plan,
)
from gw.wavefunction_bundle import (
    BandSlices,
    attach_parent_green_carrier,
    build_wavefunctions_face,
    bundle_bytes_per_rank,
    green_face_kernel_kwargs,
)


def _mesh_2x2():
    if len(jax.devices()) < 4:
        pytest.skip("needs four emulated CPU devices")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))


def _symmetry_fixture():
    identity = np.eye(3, dtype=np.int32)
    swap_xy = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                         dtype=np.int32)

    def spinor_action(rows, *, nspinor):
        rows = np.asarray(rows)
        return np.broadcast_to(
            np.eye(nspinor, dtype=np.complex128),
            rows.shape + (nspinor, nspinor)).copy()

    return SimpleNamespace(
        sym_matrices=np.stack([identity, swap_xy]),
        translations=np.zeros((2, 3), dtype=np.float64),
        irr_idx_k=np.asarray([0, 0, 1], dtype=np.int32),
        sym_idx_k=np.asarray([0, 1, 0], dtype=np.int32),
        unfolded_kpts=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        kirr_fullids=np.asarray([0, 2], dtype=np.int32),
        spinor_action=spinor_action,
    )


def test_plan_packs_raw_parent_faces_and_unfolds_their_operator():
    mesh = _mesh_2x2()
    centroids = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.int32)
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(), centroids, (2, 2, 1), mesh,
        nspinor=1,
        parent_k_frac=np.asarray([[0.0, 0.0, 0.0],
                                  [0.5, 0.0, 0.0]]),
    )
    assert plan.n_parent == 2
    assert plan.n_full == 3
    assert plan.n_centroid_packed % 4 == 0

    rng = np.random.default_rng(22)
    psi_nmu_np = (
        rng.normal(size=(2, 4, 1, 4))
        + 1j * rng.normal(size=(2, 4, 1, 4)))
    psi_mun_np = psi_nmu_np.transpose(0, 2, 3, 1)
    # The loader samples the packed table directly; here the host packer
    # stands in for it (zero pad slots).
    psi_nmu = jnp.asarray(plan.layout.axis.pack_host(psi_nmu_np, axis=3))
    psi_mun = jnp.asarray(plan.layout.axis.pack_host(psi_mun_np, axis=2))
    with mesh:
        parent_op = jnp.einsum(
            'ksmn,kntv->ksmtv', psi_mun, jnp.conj(psi_nmu),
            optimize=True)
        full_op = plan.unfold_operator(parent_op)

    parent = np.asarray(parent_op)
    expected = np.empty_like(np.asarray(full_op))
    for child, (parent_row, sym_row) in enumerate(
            zip(plan.irr_idx, plan.sym_idx)):
        perm = plan.sym_perm[int(sym_row)]
        transported = np.take(parent[int(parent_row)], perm, axis=1)
        expected[child] = np.take(transported, perm, axis=3)
    np.testing.assert_allclose(np.asarray(full_op), expected, rtol=2e-13,
                               atol=2e-13)


def test_parent_scalar_rows_do_not_masquerade_as_parent_wavefunctions():
    mesh = _mesh_2x2()
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(),
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        (2, 2, 1), mesh, nspinor=1)
    full_energy = jnp.asarray([[2.0], [2.0], [7.0]])
    np.testing.assert_array_equal(
        np.asarray(plan.parent_rows(full_energy)), [[2.0], [7.0]])


def _full_children_from_parent(parent, plan):
    """Explicit spatial-only child ψ for the fixture's zero-phase actions,
    in the run's PACKED centroid order (pad slots zero)."""
    parent = np.asarray(parent)   # canonical order, n_logical slots
    out = np.zeros(
        (plan.n_full,) + parent.shape[1:-1]
        + (plan.n_centroid_packed,), dtype=parent.dtype)
    to_packed = plan.layout.axis.canonical_to_packed
    to_canonical = plan.layout.axis.packed_to_canonical
    for child, (parent_row, sym_row) in enumerate(
            zip(plan.irr_idx, plan.sym_idx)):
        for target in range(plan.n_centroid_logical):
            source_packed = plan.sym_perm[int(sym_row), to_packed[target]]
            source = to_canonical[source_packed]
            out[child, ..., to_packed[target]] = parent[
                int(parent_row), ..., source]
    return out


def _emulated_flat_k_fftn(mesh, kgrid, spec, *, norm="ortho",
                          out_spec=None):
    from common.fft_helpers import make_sharded_fftn_3d

    assert out_spec is None or out_spec == spec
    fft3 = make_sharded_fftn_3d(
        mesh, spec, spec, axes=(0, 1, 2), norm=norm)

    def flat(x):
        return fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(
            x.shape)

    return flat


def _local_gemm_plan(_mesh, **_kwargs):
    return lambda a, b: jnp.einsum("qmk,qkn->qmn", a, b, optimize=True)


def test_parent_carrier_matches_full_k_minimax_response(monkeypatch):
    """Parent contraction + transport equals the established full-k path."""
    import common.fft_helpers as fft_helpers
    import distrib_la
    from gw import w_isdf

    monkeypatch.setattr(
        fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    monkeypatch.setattr(distrib_la, "gemm_plan", _local_gemm_plan)
    w_isdf._chi_minimax_kernel_cache.clear()

    mesh = _mesh_2x2()
    centroids = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.int32)
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(), centroids, (2, 2, 1), mesh,
        nspinor=1,
        parent_k_frac=np.asarray([[0.0, 0.0, 0.0],
                                  [0.5, 0.0, 0.0]]),
    )
    rng = np.random.default_rng(20260901)
    nk_parent, nb, ns = plan.n_parent, 4, 1
    parent = (
        rng.normal(size=(nk_parent, nb, ns, len(centroids)))
        + 1j * rng.normal(size=(nk_parent, nb, ns, len(centroids))))
    # Both routes run in the run's packed centroid order (the loader samples
    # the packed table); the full-k children are formed in that order too.
    full = _full_children_from_parent(parent, plan)
    parent = plan.layout.axis.pack_host(parent, axis=3)
    full_energy = np.asarray([
        [-1.4, -0.5, 0.7, 1.6],
        [-1.4, -0.5, 0.7, 1.6],
        [-1.1, -0.3, 0.9, 1.8],
    ])
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)

    full_y = jax.device_put(
        jnp.asarray(full), NamedSharding(mesh, P(None, None, None, 'y')))
    full_x = jax.device_put(
        jnp.asarray(np.conj(full).transpose(0, 3, 1, 2)),
        NamedSharding(mesh, P(None, 'x', None, None)))
    enk = jax.device_put(
        jnp.asarray(full_energy), NamedSharding(mesh, P(None, None)))
    wfns_full = build_wavefunctions_face(
        full_y, full_x, enk_full=enk, slices=slices, mesh_xy=mesh)

    parent_y = jax.device_put(
        jnp.asarray(parent),
        NamedSharding(mesh, P(None, None, None, 'y')))
    parent_x = jax.device_put(
        jnp.asarray(np.conj(parent).transpose(0, 3, 1, 2)),
        NamedSharding(mesh, P(None, 'x', None, None)))
    wfns_parent = attach_parent_green_carrier(
        wfns_full, parent_y, parent_x, plan=plan, mesh_xy=mesh)

    assert green_face_kernel_kwargs(wfns_parent)["face_shape"][0] == 2
    from gw.w_isdf import (
        MinimaxNodes,
        _chi_face_kwargs,
        _chi_layout_operands,
        _chi_parent_face_kwargs,
        _get_chi_minimax_kernel,
    )
    assert _chi_face_kwargs(wfns_parent)["face_shape"][0] == 3
    assert _chi_parent_face_kwargs(wfns_parent)["face_shape"][0] == 2
    resident = bundle_bytes_per_rank(wfns_parent)
    assert resident["green_parent.psi_nmu"] > 0
    assert resident["green_parent.psi_mun"] > 0

    meta = SimpleNamespace(nkx=3, nky=1, nkz=1, nk_tot=3)
    quad = SimpleNamespace(tau=np.asarray([0.0, 0.37]),
                           alpha=np.asarray([0.6, 0.4]))
    got_full = w_isdf.compute_chi0(wfns_full, quad, meta, mesh)
    got_parent = w_isdf.compute_chi0(wfns_parent, quad, meta, mesh)
    np.testing.assert_allclose(
        np.asarray(got_parent), np.asarray(got_full),
        rtol=3e-12, atol=3e-12)

    # Parent->full transport stays local inside every scan iteration and
    # chi leaves in the run's packed order: no basis move, no collective
    # beyond the k FFTs.
    vmax, cmin = -0.3, 0.7
    tau = np.asarray(quad.tau, dtype=np.float64)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(
            -np.asarray(quad.alpha) * np.exp(-tau * (cmin - vmax)),
            dtype=jnp.complex128))
    kernel = _get_chi_minimax_kernel(
        mesh, (3, 1, 1), **_chi_parent_face_kwargs(wfns_parent))
    hlo = kernel.lower(
        nodes, *_chi_layout_operands(wfns_parent, 0.0),
        jnp.asarray(vmax), jnp.asarray(cmin)).compiler_ir(
            dialect="hlo").as_hlo_text().lower()
    assert "all-gather(" not in hlo
    assert "all-to-all(" not in hlo

    w_isdf._chi_minimax_kernel_cache.clear()


def test_symmetry_kernel_caches_do_not_capture_outer_jit_tracers():
    """A cold nested trace may be reused by a different outer executable."""
    import symmetry_maps.maps as maps_impl

    maps_impl._UNFOLD_ISDF_OPERATOR_JIT_CACHE.clear()
    mesh = _mesh_2x2()
    centroids = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.int32)
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(), centroids, (2, 2, 1), mesh,
        nspinor=1,
        parent_k_frac=np.asarray([[0.0, 0.0, 0.0],
                                  [0.5, 0.0, 0.0]]))
    n_pk = plan.n_centroid_packed
    operand = jax.device_put(
        jnp.ones((2, 1, n_pk, 1, n_pk), dtype=jnp.complex128),
        NamedSharding(mesh, P(None, None, 'x', None, 'y')))

    # The first outer jit creates the service cache.  The second consumes the
    # cached inner executable from a distinct trace; a captured tracer used to
    # escape here and fail before execution.
    packed = jax.jit(plan.unfold_operator)(operand)
    packed2 = jax.jit(lambda o: plan.unfold_operator(o) * 2.0)(operand)
    jax.block_until_ready((packed, packed2))
    assert packed.shape == (3, 1, n_pk, 1, n_pk)
    assert packed2.shape == (3, 1, n_pk, 1, n_pk)


def test_sigma_spatial_cache_owns_plan_and_selects_each_plans_parent_rows(monkeypatch):
    """Stub numerical backends to isolate cache lifetime and row selection.

    The independent projection parity test covers the physical action. Here
    two equal-shaped production plans must reuse only their own kernel, and
    the cache must keep its identity owner alive after the bundle is gone.
    """
    import gc
    import weakref
    from gw import ppm_tau_kernel as tau
    from gw.wavefunction_bundle import sigma_face_kernel_kwargs

    mesh = _mesh_2x2()
    centroids = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

    def bundle(rows):
        sym = _symmetry_fixture()
        sym.kirr_fullids = np.asarray(rows, dtype=np.int32)
        plan = build_centroid_k_unfold_plan(
            sym, centroids, (2, 2, 1), mesh, nspinor=1)
        return SimpleNamespace(
            layout="face", psi_mun=np.zeros((3, 1, 4, 1)),
            slices=SimpleNamespace(nb_full=1),
            green_parent=SimpleNamespace(
                plan=plan, psi_mun=np.zeros((2, 1, 4, 1)),
                psi_nmu=np.zeros((2, 1, 1, 4))))

    monkeypatch.setattr(tau, "_sigma_spatial_kernel_cache", {})
    monkeypatch.setattr(tau, "_fft_ffi_fused_enabled", lambda: True)
    monkeypatch.setattr(tau, "ensure_jax_compile_cache", lambda: None)
    monkeypatch.setattr("common.fft_helpers.make_flat_k_gw_conv",
                        lambda *a, **k: lambda g, w: g)
    monkeypatch.setattr("common.contract_bands.contract_bands_block_reshard",
                        lambda *a, **k: lambda left, operator, right: operator)
    monkeypatch.setattr("symmetry_maps.unfold_file_wedge_band_operator",
                        lambda sym, value, **k: value)

    def factory(wfns):
        return tau.get_sigma_spatial_kernel(
            mesh_xy=mesh, kgrid=(3, 1, 1), merged_x=True,
            **sigma_face_kernel_kwargs(wfns))

    first = bundle([0, 2])
    owner = weakref.ref(first.green_parent.plan)
    kernel = factory(first)
    assert factory(first) is kernel
    del first
    gc.collect()
    assert owner() is not None

    second = factory(bundle([1, 2]))
    assert second is not kernel
    np.testing.assert_array_equal(np.asarray(kernel.conv_project(
        None, None, jnp.asarray([10., 20., 30.]), None)), [10., 30.])
    np.testing.assert_array_equal(np.asarray(second.conv_project(
        None, None, jnp.asarray([10., 20., 30.]), None)), [20., 30.])


def test_four_spinor_face_vertex_follows_unfold_without_collectives():
    """Inversion anticommutes with α, so applying a vertex before unfold must fail."""
    from dataclasses import replace
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from common.shard_map import shard_map

    mesh = _mesh_2x2()
    centroids = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(), centroids, (2, 2, 1), mesh, nspinor=4)
    spin = np.broadcast_to(np.eye(4), (3, 4, 4)).copy().astype(complex)
    spin[1] = np.diag([1, 1, -1, -1])
    plan = replace(plan, spin_action_full=spin)
    rng = np.random.default_rng(92)
    face = jnp.asarray(plan.layout.axis.pack_host(
        rng.normal(size=(2, 4, 4, 4)) + 1j * rng.normal(size=(2, 4, 4, 4)), axis=2))
    spec = P(None, None, 'x', 'y')
    def body(value):
        return plan.unfold_face(value, vertex=2, spin_axis=1, mu_axis=2, mesh_axis='x')
    fn = jax.jit(shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec,
                          check_vma=False))
    bare = jax.jit(shard_map(
        lambda value: plan.unfold_face(value, spin_axis=1, mu_axis=2, mesh_axis='x'),
        mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
    perm, phase = gamma_perm_phase(2)
    expected = gamma_apply(bare(face), perm, phase, axis=1)
    np.testing.assert_allclose(fn(face), expected, atol=1e-14)
    wrong = bare(gamma_apply(face, perm, phase, axis=1))
    assert np.max(np.abs(np.asarray(wrong - expected))) > 1
    hlo = fn.lower(face).compile().as_text().lower()
    assert 'all-to-all(' not in hlo
    assert 'all-gather(' not in hlo


def test_parent_plan_keeps_the_unreduced_one_band_case():
    """An unreduced k table uses typed parents even with only one band."""
    mesh = _mesh_2x2()
    sym = _symmetry_fixture()
    sym.sym_matrices = np.eye(3, dtype=np.int32)[None]
    sym.translations = np.zeros((1, 3))
    sym.irr_idx_k = np.arange(3, dtype=np.int32)
    sym.sym_idx_k = np.zeros(3, dtype=np.int32)
    sym.kirr_fullids = np.arange(3, dtype=np.int32)
    centroids = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    basis = PackedCentroidBasis.build(centroids, sym, (2, 2, 1), mesh)
    meta = SimpleNamespace(nspinor=4, fft_grid=(2, 2, 1), mu_basis=basis)
    cfg = SimpleNamespace(compute_mode=SimpleNamespace(needs_screening=True),
                          screening=SimpleNamespace(diagrams="w_rpa"))
    wfn = SimpleNamespace(kvecs=lambda *, k: sym.unfolded_kpts)
    plan, green, storage = _prepare_parent_wavefunction_plan(
        cfg, meta, wfn, SimpleNamespace(nb_full=1), sym=sym,
        centroid_indices=centroids, mesh_xy=mesh)
    assert plan.n_parent == plan.n_full == 3
    assert green and storage
    np.testing.assert_array_equal(plan.irr_idx, [0, 1, 2])
    np.testing.assert_array_equal(plan.spin_action_full,
                                  np.broadcast_to(np.eye(4), (3, 4, 4)))


def test_non_rpa_consumer_refuses_before_parent_loading():
    """Unported screening cannot retain a hidden full-k wavefunction carrier."""
    cfg = SimpleNamespace(compute_mode=SimpleNamespace(needs_screening=True),
                          screening=SimpleNamespace(diagrams="w_bse"))
    with pytest.raises(ValueError, match="parent_screening_diagrams.*w_bse"):
        _prepare_parent_wavefunction_plan(
            cfg, None, None, None, sym=None, centroid_indices=None, mesh_xy=None)


def test_parent_plan_requires_only_consumed_canonical_actions():
    """An unused mirror stays unavailable while canonical TRS row2 remains row2."""
    mesh = _mesh_2x2()
    sym = _symmetry_fixture()
    sym.sym_matrices = np.array([np.eye(3, dtype=int), np.diag([1, 1, -1])])
    sym.sym_idx_k = np.array([0, 2, 0], dtype=np.int32)
    centroids = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int32)
    basis = PackedCentroidBasis.build(centroids, sym, (4, 4, 4), mesh)
    plan = build_centroid_k_unfold_plan(
        sym, centroids, (4, 4, 4), mesh, nspinor=4, layout=basis.layout)
    assert plan.n_sym_spatial == 2
    np.testing.assert_array_equal(plan.sym_idx, [0, 2, 0])
    assert np.all(plan.sym_perm[[1, 3]] == -1)
    assert np.all(plan.centroid_local_perm[[1, 3]] == -1)
    sym.sym_idx_k = np.array([0, 3, 0], dtype=np.int32)
    with pytest.raises(RuntimeError, match="closure"):
        build_centroid_k_unfold_plan(
            sym, centroids, (4, 4, 4), mesh, nspinor=4, layout=basis.layout)
