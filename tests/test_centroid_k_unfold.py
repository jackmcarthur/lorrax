"""GW adapter contracts for raw-parent centroid operators."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.grouped_layout import build_square_grouped_shard_layout
from gw.centroid_k_unfold import (
    CentroidKUnfoldPlan,
    build_centroid_k_unfold_plan,
    parent_k_contraction_profitable,
)
from gw.gw_init import _resolve_parent_green_admission
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


def test_parent_k_profitability_uses_measured_small_band_envelope():
    assert parent_k_contraction_profitable(
        n_full=64, n_parent=8, n_bands=4)
    assert not parent_k_contraction_profitable(
        n_full=64, n_parent=8, n_bands=3)
    assert not parent_k_contraction_profitable(
        n_full=64, n_parent=40, n_bands=256)
    assert not parent_k_contraction_profitable(
        n_full=64, n_parent=64, n_bands=1024)


def test_bispinor_parent_k_candidate_falls_back_by_name():
    """The production admission predicate cannot send current data to charge unfold."""
    cfg = SimpleNamespace(
        bispinor=True,
        memory=SimpleNamespace(low_mem_bands=True),
        compute_mode=SimpleNamespace(needs_screening=True),
        screening=SimpleNamespace(diagrams="w_rpa"),
        qp_solver="one_shot_dft",
    )
    meta = SimpleNamespace(nk_tot=64, nspinor=4)
    wfn = SimpleNamespace(nkpts=8)
    bands = SimpleNamespace(nb_full=4)
    records = []

    admitted, work_ok = _resolve_parent_green_admission(
        cfg, meta, wfn, bands, backend="gpu", print_fn=records.append)

    assert work_ok
    assert not admitted
    assert len(records) == 1
    record = records[0]
    for required in (
        "GATE parent_k_green_bispinor_vector_unfold_unimplemented",
        "bispinor = true",
        "psi_nk_irr-only",
        "full-k wavefunction-storage fallback",
        "scalar charge operators",
        "current vertices are Cartesian vectors",
        "k and k-q current endpoints",
        "q_stencil_orbit_table",
        "apply_band_matrix_symmetry",
        "set bispinor = false",
    ):
        assert required in record

    cfg.bispinor = False
    meta.nspinor = 2
    records.clear()
    admitted, work_ok = _resolve_parent_green_admission(
        cfg, meta, wfn, bands, backend="gpu", print_fn=records.append)
    assert work_ok
    assert admitted
    assert records == []


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
