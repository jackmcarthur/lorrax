"""The plane-restricted r-chunk transform equals the full-box one (2026-09-23).

``accumulate_rchunk_to_gflat(planes=...)`` (ζ(tile) → ζ(G)) runs 2D FFTs over
the tile's planes and a partial DFT along the plane axis instead of one
full-box FFT per row.  This gate compares it with the full-box form on
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
    # g_index: the per-k sphere index (common.gvec_fft_box.
    # build_sphere_box_index), n_rtot + g on a pad slot.
    n_rtot = nx * ny * nz
    g_index = np.tile(n_rtot + np.arange(ngkmax, dtype=np.int32), (nk, 1))
    sphere = np.zeros((nk, ngkmax), np.int32)
    for k, f in enumerate(flat_all):
        g_index[k, :f.size] = f
        sphere[k, :f.size] = f
        sphere[k, f.size:] = f[0]            # pad: a valid box cell, masked later
    return g_index, sphere, ngkmax, [f.size for f in flat_all]


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
