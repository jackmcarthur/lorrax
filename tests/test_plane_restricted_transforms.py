"""Plane-restricted r-chunk transforms equal the full-box ones (2026-09-23).

``to_rpoints_inner(planes=...)`` (ψ(G) → ψ at a tile's points) and
``accumulate_rchunk_to_gflat(planes=...)`` (ζ(tile) → ζ(G)) run 2D FFTs over
the tile's planes and a partial DFT along the plane axis instead of one
full-box FFT per row.  This gate compares them with the full-box forms on
random data, for every plane axis, on a box whose three extents differ
(so a wrong axis bookkeeping cannot cancel), with per-k spheres that differ
(the cylinder is a union), Bloch phases, pad slots, and a tile that touches
non-adjacent planes.  Parity class: value-level, 1e-12 relative.

Red twin (TASTE 21): dropping one of the tile's planes from the plane list
must break the comparison by far more than the tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.mesh(4)

_FFT = (10, 12, 14)
_TOL = 1.0e-12


def _mesh(px, py):
    devs = jax.devices()
    if len(devs) < px * py:
        pytest.skip(f"needs {px * py} devices")
    return Mesh(np.array(devs[:px * py]).reshape(px, py), ("x", "y"))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _sphere(rng, nk, cut):
    nx, ny, nz = _FFT
    g = np.stack(np.meshgrid(*[np.fft.fftfreq(n) * n for n in _FFT],
                             indexing="ij"), -1).reshape(-1, 3)
    flat_all = []
    for k in range(nk):
        shift = rng.uniform(-0.5, 0.5, 3)
        inside = np.flatnonzero(np.sum((g + shift) ** 2, 1) < cut)
        flat_all.append(inside)
    ngkmax = max(f.size for f in flat_all)
    g_index = np.full((nk, nx * ny * nz), ngkmax, np.int32)
    sphere = np.zeros((nk, ngkmax), np.int32)
    for k, f in enumerate(flat_all):
        g_index[k, f] = np.arange(f.size)
        sphere[k, :f.size] = f
        sphere[k, f.size:] = f[0]            # pad: a valid box cell, masked later
    return g_index.reshape(nk, *_FFT), sphere, ngkmax, [f.size for f in flat_all]


def _tile(rng, axis, planes, width):
    n = _FFT
    flat = np.arange(np.prod(n))
    coords = (flat // (n[1] * n[2]), (flat // n[2]) % n[1], flat % n[2])
    pool = flat[np.isin(coords[axis], planes)]
    pick = rng.choice(pool, size=width - 3, replace=False)
    r_idx = np.concatenate([pick, np.prod(n) + np.arange(3)])   # 3 pad sentinels
    rng.shuffle(r_idx)
    return r_idx.astype(np.int32)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_to_rpoints_planes_matches_full_box(axis):
    from common.wfn_transforms import (psi_cylinder_tables, to_rpoints_inner)

    rng = np.random.default_rng(10 + axis)
    nk, nb, ns = 3, 2, 2
    g_index, _, ngkmax, _ = _sphere(rng, nk, cut=9.0)
    psi = (rng.standard_normal((nk, nb, ns, ngkmax))
           + 1j * rng.standard_normal((nk, nb, ns, ngkmax)))
    kv = rng.uniform(-0.5, 0.5, (nk, 3))
    planes = np.array([1, _FFT[axis] - 2, -1], np.int32)       # non-adjacent + pad
    r_idx = _tile(rng, axis, planes[:2], 40)
    real = r_idx < np.prod(_FFT)
    # The full-box form returns clipped-index values on pad slots (its
    # caller zeroes them); the plane form returns zeros there.  Compare the
    # real slots and check the plane form's pads directly.
    ref = np.asarray(to_rpoints_inner(
        jnp.asarray(psi), jnp.asarray(g_index), _FFT, jnp.asarray(r_idx),
        norm="ortho", kvecs_frac=jnp.asarray(kv)))[..., real]
    cyl = psi_cylinder_tables(jnp.asarray(g_index), _FFT, axis, ngkmax=ngkmax)
    got = np.asarray(to_rpoints_inner(
        jnp.asarray(psi), jnp.asarray(g_index), _FFT, jnp.asarray(r_idx),
        norm="ortho", kvecs_frac=jnp.asarray(kv),
        planes=jnp.asarray(planes), plane_axis=axis, cylinder=cyl))
    assert not np.any(got[..., ~real])
    got = got[..., real]
    print(f"[plane parity] psi axis={axis} rel={_rel(got, ref):.3e}")
    assert _rel(got, ref) <= _TOL
    # red twin: a plane missing from the list loses its points
    bad = np.asarray(to_rpoints_inner(
        jnp.asarray(psi), jnp.asarray(g_index), _FFT, jnp.asarray(r_idx),
        norm="ortho", kvecs_frac=jnp.asarray(kv),
        planes=jnp.asarray(np.array([planes[0], -1, -1], np.int32)),
        plane_axis=axis, cylinder=cyl))[..., real]
    assert _rel(bad, ref) > 1e-2


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_accumulate_planes_matches_full_box(axis):
    from common.wfn_transforms import accumulate_rchunk_to_gflat

    mesh = _mesh(2, 2)
    rng = np.random.default_rng(20 + axis)
    nq, n_mu = 3, 8
    _, sphere, ngkmax, _ = _sphere(rng, nq, cut=9.0)
    planes = np.array([2, _FFT[axis] - 1, -1], np.int32)
    r_idx = _tile(rng, axis, planes[:2], 36)
    r_len = r_idx.size
    z = (rng.standard_normal((nq, n_mu, r_len))
         + 1j * rng.standard_normal((nq, n_mu, r_len)))
    z[..., r_idx >= np.prod(_FFT)] = 0
    qv = rng.uniform(-0.5, 0.5, (nq, 3))
    sh = NamedSharding(mesh, P(None, ("x", "y"), None))
    acc0 = rng.standard_normal((nq, n_mu, ngkmax)) + 0j

    def run(**kw):
        return np.asarray(accumulate_rchunk_to_gflat(
            rchunk=jax.device_put(jnp.asarray(z), sh),
            gflat_acc=jax.device_put(jnp.asarray(acc0), sh),
            mesh=mesh, fft_grid=_FFT, r_indices=jnp.asarray(r_idx),
            sphere_idx=sphere, qvec_frac=qv, norm="backward",
            chunk_size=5, **kw))

    ref = run()
    got = run(planes=jnp.asarray(planes), plane_axis=axis)
    # V_q-shaped consumer: V_q(mu, nu) = sum_G zeta_q(mu, G) v_q(G) zeta_q(nu, G)*
    # with a fixed positive v (the Coulomb contraction's shape).
    v = np.random.default_rng(7).uniform(0.1, 2.0, (nq, ngkmax))
    vq = lambda zg: np.einsum("qmg,qg,qng->qmn", zg - acc0, v, np.conj(zg - acc0))
    print(f"[plane parity] zeta axis={axis} rel={_rel(got - acc0, ref - acc0):.3e} "
          f"V_q rel={_rel(vq(got), vq(ref)):.3e}")
    assert _rel(got - acc0, ref - acc0) <= _TOL
    assert _rel(vq(got), vq(ref)) <= _TOL
    bad = run(planes=jnp.asarray(np.array([planes[0], -1, -1], np.int32)),
              plane_axis=axis)
    assert _rel(bad - acc0, ref - acc0) > 1e-2


def test_tiles_are_orbit_closed_and_plane_local():
    """The plane-ordered tile builder: every point once, whole orbits per Y
    owner, fixed width, and each tile on few planes of its axis."""
    from gw.centroid_k_unfold import build_real_grid_orbit_tiles

    # C4 about z (mixes x and y planes) times sigma_h (pairs z with -z): the
    # layered-crystal case, where only the stacking axis keeps orbits on
    # few planes (two), so the incidence rule must pick it.
    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    sh = np.diag([1, 1, -1]).astype(np.int64)
    rots = [np.linalg.matrix_power(c4, i) for i in range(4)]
    ops = np.array(rots + [sh @ r for r in rots])
    tau = np.zeros((8, 3))
    fg = (8, 8, 10)
    tiles = build_real_grid_orbit_tiles(ops, tau, fg, n_y=2, target_width=32)
    assert tiles.plane_axis == 2
    act = tiles.r_index[tiles.r_index >= 0]
    assert np.array_equal(np.sort(act), np.arange(np.prod(fg)))
    for t in range(tiles.n_tiles):
        tiles.source_tables(t)            # refuses if an orbit crosses an owner
        n_planes = int(np.sum(tiles.tile_planes[t] >= 0))
        assert n_planes <= 4, (t, tiles.tile_planes[t])


def test_fmode_matches_resident_accumulator():
    """F-mode (accumulator-free) equals the resident zeta(G) accumulator.

    The grid is tiled by the production plane-ordered builder; every tile's
    zeta(r) is (resident) accumulated into zeta(G) through the plane path, and
    (F-mode) turned into per-plane 2D-FFT columns that are summed per plane
    and only then transformed along the axis onto the sphere.  zeta and a
    V_q-shaped contraction must agree to 1e-12; so must the full-box path.
    """
    from common.wfn_transforms import (
        accumulate_rchunk_to_gflat, plane_columns_to_sphere,
        rchunk_to_plane_columns, sphere_plane_columns)
    from gw.centroid_k_unfold import build_real_grid_orbit_tiles

    mesh = _mesh(2, 2)
    rng = np.random.default_rng(31)
    nq, n_mu = 3, 8
    n_rtot = int(np.prod(_FFT))
    _, sphere, ngkmax, _ = _sphere(rng, nq, cut=9.0)
    tiles = build_real_grid_orbit_tiles(
        np.eye(3, dtype=np.int64)[None], np.zeros((1, 3)), _FFT,
        n_y=2, target_width=48)
    axis = int(tiles.plane_axis)
    n_a = _FFT[axis]
    zr = (rng.standard_normal((nq, n_mu, n_rtot))
          + 1j * rng.standard_normal((nq, n_mu, n_rtot)))
    qv = rng.uniform(-0.5, 0.5, (nq, 3))
    sh = NamedSharding(mesh, P(None, ("x", "y"), None))
    columns, s_coords, cyl = sphere_plane_columns(sphere, _FFT, axis)

    acc_planes = jax.device_put(jnp.zeros((nq, n_mu, ngkmax), jnp.complex128), sh)
    acc_box = jax.device_put(jnp.zeros((nq, n_mu, ngkmax), jnp.complex128), sh)
    F_all = np.zeros((n_a, nq, n_mu, columns.size), np.complex128)
    for t in range(tiles.n_tiles):
        row = tiles.r_index[t]
        r_idx = np.where(row >= 0, row, n_rtot + np.arange(row.size)).astype(np.int32)
        chunk = np.where(row >= 0, zr[..., np.clip(row, 0, n_rtot - 1)], 0)
        chunk_d = jax.device_put(jnp.asarray(chunk), sh)
        planes = jnp.asarray(tiles.tile_planes[t], dtype=jnp.int32)
        kw = dict(mesh=mesh, fft_grid=_FFT, r_indices=jnp.asarray(r_idx),
                  sphere_idx=sphere, qvec_frac=qv, norm="backward", chunk_size=5)
        acc_planes = accumulate_rchunk_to_gflat(
            rchunk=chunk_d, gflat_acc=acc_planes, planes=planes,
            plane_axis=axis, **kw)
        acc_box = accumulate_rchunk_to_gflat(
            rchunk=chunk_d, gflat_acc=acc_box, **kw)
        Ft = np.asarray(rchunk_to_plane_columns(
            chunk_d, mesh=mesh, fft_grid=_FFT, r_indices=jnp.asarray(r_idx),
            planes=planes, plane_axis=axis, columns=columns, qvec_frac=qv,
            norm="backward", chunk_size=5))
        for p, a in enumerate(tiles.tile_planes[t]):
            if a >= 0:
                F_all[a] += Ft[p]
    zeta_F = np.asarray(plane_columns_to_sphere(
        jax.device_put(jnp.asarray(F_all),
                       NamedSharding(mesh, P(None, None, ("x", "y"), None))),
        mesh=mesh, fft_grid=_FFT, plane_axis=axis, s_coords=s_coords,
        cyl_of_sphere=cyl, norm="backward"))
    ref = np.asarray(acc_planes)
    v = rng.uniform(0.1, 2.0, (nq, ngkmax))
    vq = lambda z: np.einsum("qmg,qg,qng->qmn", z, v, np.conj(z))
    print(f"[plane parity] F-mode vs resident zeta rel={_rel(zeta_F, ref):.3e} "
          f"V_q rel={_rel(vq(zeta_F), vq(ref)):.3e}; resident planes vs box "
          f"{_rel(ref, np.asarray(acc_box)):.3e}")
    assert _rel(zeta_F, ref) <= _TOL
    assert _rel(vq(zeta_F), vq(ref)) <= _TOL
    assert _rel(ref, np.asarray(acc_box)) <= _TOL
