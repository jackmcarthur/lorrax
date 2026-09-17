"""The photon Coulomb route hands the canonical carrier to the one packing owner.

``gw_init._finalize_vq_views`` is the only site that packs V_q into the run's
orbit-packed centroid order, and ``PackedCentroidBasis.pack_operator`` accepts
only the canonical file carrier.  The scalar route already delivers that
carrier.  The photon (bispinor) route used to pad its canonical V_q and G0 up
to ``meta.n_rmu_padded``, which is the PACKED extent, so on any orbit-packed
basis whose packed carrier exceeds the canonical one the pack refused:
``pack_axis expects the canonical carrier 392 on axis 1; got (9, 396, 396)``
(symmetric CrI3 3x3, JID 58454563.25).  A nosym or identity basis has equal
extents, which is why no earlier bispinor deck reached it.

The plant below is three C4 orbits of four centroids on a 2x2 emulated CPU
mesh: canonical carrier 12, packed carrier 16.  The file-side producers are
replaced by canonical arrays; everything from the route's return to the
packed operator is production code.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh_2x2():
    if len(jax.devices()) < 4:
        pytest.skip("needs four emulated CPU devices")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _c4_basis(mesh):
    from common.centroid_basis import PackedCentroidBasis
    c4 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    ops = [np.eye(3, dtype=np.int32)]
    for _ in range(3):
        ops.append(c4 @ ops[-1])
    sym = SimpleNamespace(sym_matrices=np.stack(ops),
                          translations=np.zeros((4, 3)))
    cents = np.asarray(
        [[1, 0, 0], [0, 1, 0], [7, 0, 0], [0, 7, 0],
         [2, 0, 0], [0, 2, 0], [6, 0, 0], [0, 6, 0],
         [1, 1, 0], [7, 1, 0], [7, 7, 0], [1, 7, 0]], dtype=np.int32)
    return PackedCentroidBasis.build(cents, sym, (8, 8, 1), mesh)


class _Zeta:
    """Stands in for the four ζ files the route opens."""

    n_rmu_disk = 12

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_photon_route_is_packed_once_on_an_orbit_packed_basis(
        monkeypatch, tmp_path):
    import file_io.centroids as centroids
    import file_io.restart_bundle as restart_bundle
    import gw.gw_init as gw_init
    import gw.v_q_bispinor as v_q_bispinor

    mesh = _mesh_2x2()
    basis = _c4_basis(mesh)
    n, n_can = basis.n_logical, basis.n_canonical
    assert basis.n_packed > n_can, "the plant must separate the two carriers"

    # Canonical, Hermitian, positive q=Gamma operator with exact-zero pads.
    rng = np.random.default_rng(20260917)
    a = rng.standard_normal((n, n))
    v = np.zeros((1, n_can, n_can), dtype=np.complex128)
    v[0, :n, :n] = a @ a.T + n * np.eye(n)
    V_canonical = jax.device_put(
        jnp.asarray(v), NamedSharding(mesh, P(None, "x", "y")))
    g0 = np.zeros((1, n_can), dtype=np.complex128)
    g0[0, :n] = rng.standard_normal(n)
    G0_canonical = jax.device_put(
        jnp.asarray(g0), NamedSharding(mesh, P(None, "x")))

    monkeypatch.setattr(gw_init, "ZetaLoader", _Zeta)
    monkeypatch.setattr(gw_init, "uses_coupled_photon_head", lambda cfg: False)
    monkeypatch.setattr(
        centroids, "load_centroids",
        lambda path, fft_grid: (None, np.zeros((n, 3), np.int32), None))
    monkeypatch.setattr(
        v_q_bispinor, "compute_V_q_bispinor_g_flat_to_h5",
        lambda **kw: (kw["output_h5_path"], (G0_canonical,) * 4))
    monkeypatch.setattr(
        restart_bundle, "read_photon_charge", lambda path, m: V_canonical)

    meta = SimpleNamespace(
        mu_basis=basis, n_rmu=n, n_rmu_padded=basis.n_packed,
        kgrid=(1, 1, 1), fft_grid=(8, 8, 1), cell_volume=1.0, sys_dim=3)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(centroids_file_current="current.txt"),
        memory=SimpleNamespace(vq_g_chunk_size=0),
        head=SimpleNamespace(bispinor_tt_head_correction=False,
                             mc_average_placement="off",
                             uses_bgw_metal_q0shift=False))
    zeta = str(tmp_path / "zeta_q.h5")
    V_raw, G0_all, head, photon = gw_init._compute_photon_vq(
        None, None, cfg, mesh, meta, lambda *a, **k: None, None, 1.0,
        SimpleNamespace(bdot=None),
        [str(tmp_path / f"zeta_q_mu{i}.h5") for i in (1, 2, 3)],
        str(tmp_path), zeta)
    V_packed, G0, _, _ = gw_init._finalize_vq_views(
        G0_all, V_raw, head, meta, photon, lambda *a, **k: None)

    assert V_packed.shape == (1, basis.n_packed, basis.n_packed)
    np.testing.assert_array_equal(
        np.asarray(V_packed), np.asarray(basis.pack_operator(V_canonical)))
    # G0 stays canonical: the restart writer persists it in file order.
    np.testing.assert_array_equal(np.asarray(G0), g0[0])
