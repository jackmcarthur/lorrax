"""distrib_la subspace XLA route: gated against the host LAPACK route on the same device.

The XLA provider (ROCm's row of SUBSPACE_ROUTE) must reproduce the host
provider's numbers, ignore poisoned (NaN) inactive storage as the native
provider does, compile once for every runtime window, and stage no host
callback (so a solver jit holding it stays cacheable).
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from distrib_la.subspace import (SUBSPACE_ROUTE, CpuSubspacePlan, XlaSubspacePlan,  # noqa: E402
                                 subspace_route)

CAP, D, B = 64, 128, 12


def _problem(seed=78):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(D, CAP)) + 1j * rng.normal(size=(D, CAP))
    v = np.linalg.qr(z)[0].T.copy()
    hv = v * np.arange(D)[None, :]
    c = rng.normal(size=(CAP, B)) + 1j * rng.normal(size=(CAP, B))
    p = rng.normal(size=(B, D)) + 1j * rng.normal(size=(B, D))
    return v, hv, c, p


def _poison(a, start, stop):
    a = a.copy()
    a[:start], a[stop:] = np.nan, np.nan
    return jnp.asarray(a)


def test_leg_rows():
    assert SUBSPACE_ROUTE == {"cpu": "host", "CUDA": "native", "ROCM": "xla"}
    assert subspace_route("ROCM") == "xla" and subspace_route("CUDA") == "native"
    assert subspace_route("cpu") == "host" and subspace_route("TPU") == "xla"


@pytest.mark.parametrize("active", [1, 12, 17, 35, 64])
def test_prefix_ops_match_host_and_ignore_nan_tail(active):
    v, hv, c, p = _problem()
    xla, host = XlaSubspacePlan(CAP, B), CpuSubspacePlan(CAP, B)
    vv, hh, cc = (_poison(a, 0, active) for a in (v, hv, c))
    h = jnp.full((CAP, CAP), jnp.nan, jnp.complex128)
    ref = v[:active].conj() @ hv[:active].T
    proj = jax.jit(xla.project)(vv, hh, active, h, 0, active)
    np.testing.assert_allclose(np.asarray(proj)[:active, :active], ref, atol=2e-12)
    np.testing.assert_allclose(np.asarray(proj)[:active, :active],
                               np.asarray(jax.jit(host.project)(vv, hh, active, h, 0, active))[:active, :active],
                               atol=2e-12)
    assert np.isnan(np.asarray(proj)[active:, active:]).all()   # untouched, as the host route
    e, coeff = jax.jit(xla.eigh)(proj, active)
    eh, _ = jax.jit(host.eigh)(proj, active)
    np.testing.assert_allclose(e, eh, atol=2e-12)
    np.testing.assert_allclose(e[:min(active, B)], np.linalg.eigvalsh(ref)[:B], atol=2e-12)
    coeff = np.asarray(coeff)
    np.testing.assert_allclose(ref @ coeff[:active], coeff[:active] * np.where(np.isfinite(e), e, 0), atol=2e-11)
    np.testing.assert_array_equal(coeff[active:], 0)
    assert np.isinf(np.asarray(e)[active:]).all()
    x, hx = jax.jit(xla.reconstruct)(vv, hh, cc, active, jnp.asarray(p))
    np.testing.assert_allclose(x, c[:active].T @ v[:active], atol=2e-12)
    np.testing.assert_allclose(hx, c[:active].T @ hv[:active], atol=2e-11)
    pp = jax.jit(xla.orthogonalize)(vv, jnp.asarray(p), active)
    np.testing.assert_allclose(pp, jax.jit(host.orthogonalize)(vv, jnp.asarray(p), active), rtol=2e-13, atol=3e-12)
    g = jax.jit(xla.gram)(vv, jnp.asarray(p), active)
    np.testing.assert_allclose(g, jax.jit(host.gram)(vv, jnp.asarray(p), active), atol=2e-12)


@pytest.mark.parametrize("start,count", [(0, 0), (5, 0), (3, 5), (7, 33), (40, 24), (0, 64)])
def test_windowed_ops_match_host_with_one_compilation(start, count):
    v, hv, c, p = _problem(419)
    xla, host = XlaSubspacePlan(CAP, B), CpuSubspacePlan(CAP, B)
    vv, cc = _poison(v, start, start + count), _poison(c, start, start + count)
    pj = jnp.asarray(p)
    i32 = jnp.int32
    ortho = jax.jit(xla.orthogonalize).lower(vv, pj, i32(6), start=i32(2)).compile()
    np.testing.assert_allclose(ortho(vv, pj, i32(count), start=i32(start)),
                               host.orthogonalize(vv, pj, count, start=start), rtol=2e-13, atol=3e-12)
    gram = jax.jit(xla.gram).lower(vv, pj, i32(6), start=i32(2)).compile()
    np.testing.assert_allclose(gram(vv, pj, i32(count), start=i32(start)),
                               host.gram(vv, pj, count, start=start), atol=2e-12)
    rec = jax.jit(xla.reconstruct, static_argnames=("compute_image",)).lower(
        vv, vv, cc, i32(6), pj, columns=i32(B), start=i32(2), compute_image=False).compile()
    got = rec(vv, vv, cc, i32(count), pj, columns=i32(5), start=i32(start))[0]
    want = host.reconstruct(vv, vv, cc, count, pj, columns=5, start=start, compute_image=False)[0]
    np.testing.assert_allclose(got, want, atol=2e-12)
    # Incremental projection: the new panel [start, start+count) against the prefix.
    stop = start + count
    if stop:
        hh = _poison(hv, 0, stop)
        vp = _poison(v, 0, stop)
        h0 = jnp.asarray(np.random.default_rng(1).normal(size=(CAP, CAP)) + 0j)
        proj = jax.jit(xla.project).lower(vp, hh, i32(9), h0, i32(2), i32(3)).compile()
        np.testing.assert_allclose(proj(vp, hh, i32(stop), h0, i32(start), i32(count)),
                                   host.project(vp, hh, stop, h0, start, count), atol=2e-12)


@pytest.mark.parametrize("start,active", [(-1, 1), (0, -1), (8, 60), (65, 0), (2**32, 0), (0, 2**32)])
def test_invalid_windows_return_nan(start, active):
    v, hv, c, p = _problem()
    xla = XlaSubspacePlan(CAP, B)
    s, a = jnp.asarray(start, jnp.int64), jnp.asarray(active, jnp.int64)
    vj, pj = jnp.asarray(v), jnp.asarray(p)
    assert np.isnan(np.asarray(jax.jit(xla.orthogonalize)(vj, pj, a, start=s))).all()
    assert np.isnan(np.asarray(jax.jit(xla.gram)(vj, pj, a, start=s))).all()
    assert np.isnan(np.asarray(jax.jit(xla.reconstruct)(vj, vj, jnp.asarray(c), a, pj, start=s)[0])).all()


def test_eigh_buckets_above_64_and_real_valued_block():
    """Capacity 300: extents 64/128/256/300; a real-valued (exactly zero imaginary) block."""
    cap, n_eig = 300, 7
    rng = np.random.default_rng(3)
    a = rng.normal(size=(cap, cap))
    a = (a + a.T) / 2 + 0j
    xla, host = XlaSubspacePlan(cap, n_eig), CpuSubspacePlan(cap, n_eig)
    run = jax.jit(xla.eigh).lower(jnp.asarray(a), jnp.int32(1)).compile()
    for m in (1, 5, 64, 65, 129, 200, 256, 257, 300):
        h = a.copy()
        h[m:, :] = np.nan
        h[:, m:] = np.nan
        e, vec = run(jnp.asarray(h), jnp.int32(m))
        eh, _ = host.eigh(jnp.asarray(h), m)
        np.testing.assert_allclose(e, eh, atol=2e-12 * max(1, np.abs(a).max() * np.sqrt(m)))
        k = min(m, n_eig)
        vec = np.asarray(vec)
        np.testing.assert_allclose(a[:m, :m] @ vec[:m, :k], vec[:m, :k] * np.asarray(e)[:k], atol=2e-11 * np.sqrt(m))
        np.testing.assert_allclose(vec[:m, :k].conj().T @ vec[:m, :k], np.eye(k), atol=1e-12)


def test_normalization_ignores_nan_tail():
    plan = XlaSubspacePlan(12, 12)
    p = jnp.eye(32, dtype=jnp.complex128)[:12]
    p = p.at[1].multiply(2.).at[2:].set(jnp.nan)
    result = np.asarray(jax.jit(plan.normalize)(p, jnp.int32(2)))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:2].conj() @ result[:2].T, np.eye(2), atol=1e-14)
    np.testing.assert_array_equal(result[2:], 0)


def test_no_host_callback_is_staged():
    v, hv, c, p = (jnp.asarray(a) for a in _problem())
    xla = XlaSubspacePlan(CAP, B)
    h = jnp.zeros((CAP, CAP), jnp.complex128)
    texts = [jax.jit(xla.eigh).lower(h, 5).as_text(),
             jax.jit(xla.project).lower(v, hv, 5, h, 2, 3).as_text(),
             jax.jit(xla.orthogonalize).lower(v, p, 5).as_text(),
             jax.jit(xla.reconstruct).lower(v, hv, c, 5, p).as_text(),
             jax.jit(xla.gram).lower(v, p, 5).as_text()]
    assert not any("callback" in t for t in texts)
    assert "callback" in jax.jit(CpuSubspacePlan(CAP, B).gram).lower(v, p, 5).as_text()


def test_distributed_wrapper_matches_host_route():
    """shard_map wrapper over four devices: only coefficients cross the mesh."""
    if jax.device_count() < 4 or jax.process_count() != 1:
        pytest.skip("needs four local devices (e.g. --xla_force_host_platform_device_count=4)")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from distrib_la.subspace import DistributedSubspacePlan
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    v, hv, c, p = _problem(5)
    shape = (D // 8, 8)
    put = lambda a, n: jax.device_put(jnp.asarray(a.reshape(n, *shape)), sh)
    vv, hh, pp = put(_poison(v, 0, 30).__array__(), CAP), put(_poison(hv, 0, 30).__array__(), CAP), put(p, B)
    xla = DistributedSubspacePlan(XlaSubspacePlan(CAP, B), sh)
    host = DistributedSubspacePlan(CpuSubspacePlan(CAP, B), sh)
    h = jnp.zeros((CAP, CAP), jnp.complex128)
    for name, args, kw in (("project", (vv, hh, 30, h, 20, 10), {}),
                           ("gram", (vv, pp, 12), {"start": 4}),
                           ("orthogonalize", (vv, pp, 12), {"start": 4})):
        got = jax.jit(getattr(xla, name))(*args, **kw)
        want = jax.jit(getattr(host, name))(*args, **kw)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=3e-12, atol=4e-12, err_msg=name)
    proj = jax.jit(xla.project)(vv, hh, 30, h, 0, 30)
    e, cf = jax.jit(xla.eigh)(proj, 30)
    eh, _ = jax.jit(host.eigh)(proj, 30)
    np.testing.assert_allclose(e, eh, atol=2e-12)
    x, _ = jax.jit(xla.reconstruct)(vv, hh, cf, 30, pp)
    xh, _ = jax.jit(host.reconstruct)(vv, hh, cf, 30, pp)
    np.testing.assert_allclose(np.asarray(x), np.asarray(xh), atol=2e-12)
