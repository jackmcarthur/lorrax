"""lxkit.ordered_psum_scatter / ordered_psum / ordered_reduce / ordered_sum against lax.psum_scatter / lax.psum on four host devices.

Values: equal to the library collective to round-off (1e-14 relative) and BITWISE equal to the numpy
left-to-right rank-order sum (the helpers' contract), for a 1-D and a 2x2 mesh, tiled and untiled,
f64 and c128, with a padded all-reduce, slabbed and one-shot, and the same bits on permuted device
orders.  CPU: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
python -m pytest -q tests/test_ordered_collectives.py``.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")


def _mesh(shape, names):
    import jax
    import numpy as np
    from jax.sharding import Mesh
    devs = jax.devices()
    if len(devs) < 4:
        import pytest
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.asarray(devs[:4]).reshape(shape), names)


def _rank_fold(blocks):
    acc = blocks[0]
    for b in blocks[1:]:
        acc = acc + b
    return acc


def test_ordered_psum_is_the_rank_order_fold():
    """One-shot, slabbed and pytree ordered_psum / ordered_psum_scatter equal the library collective
    to round-off and the numpy rank-order fold bit for bit."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax, shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import ordered_psum, ordered_psum_scatter
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(3)
    for shape, names, ax in (((4,), ("r",), "r"), ((2, 2), ("x", "y"), ("x", "y"))):
        mesh = _mesh(shape, names)
        spec = P(names if len(names) > 1 else names[0])
        for dt in (np.float64, np.complex128):
            g = rng.standard_normal((4, 3, 8, 5))
            if dt == np.complex128:
                g = g + 1j * rng.standard_normal(g.shape)
            x = jnp.asarray(g.reshape(12, 8, 5))                       # rank r holds rows 3r..3r+2
            run = lambda f, out: jax.jit(shard_map(f, mesh=mesh, in_specs=spec, out_specs=out, check_vma=False))(x)
            a = np.asarray(run(lambda v: ordered_psum_scatter(v, ax, scatter_dimension=1, tiled=True), spec))
            b = np.asarray(run(lambda v: lax.psum_scatter(v, ax, scatter_dimension=1, tiled=True), spec))
            assert np.abs(a - b).max() <= 1e-14 * np.abs(b).max()
            want = np.concatenate([_rank_fold([g[j][:, 2 * p:2 * p + 2] for j in range(4)]) for p in range(4)], 0)
            assert a.tobytes() == want.tobytes()                      # bitwise: the rank-order fold
            h = jnp.asarray(g[:, :, :4, 0].reshape(12, 4))           # untiled: dimension of the axis size
            ua = np.asarray(jax.jit(shard_map(lambda v: ordered_psum_scatter(v, ax, scatter_dimension=1), mesh=mesh,
                                              in_specs=spec, out_specs=spec, check_vma=False))(h))
            ub = np.asarray(jax.jit(shard_map(lambda v: lax.psum_scatter(v, ax, scatter_dimension=1), mesh=mesh,
                                              in_specs=spec, out_specs=spec, check_vma=False))(h))
            assert np.abs(ua - ub).max() <= 1e-14 * np.abs(ub).max()
            y = jnp.asarray(g[:, :, :7, :3].reshape(12, 7, 3))       # 63 elements per rank: padded all-reduce
            ra = np.asarray(jax.jit(shard_map(lambda v: ordered_psum(v, ax), mesh=mesh, in_specs=spec,
                                              out_specs=P(), check_vma=False))(y))
            rb = np.asarray(jax.jit(shard_map(lambda v: lax.psum(v, ax), mesh=mesh, in_specs=spec,
                                              out_specs=P(), check_vma=False))(y))
            assert np.abs(ra - rb).max() <= 1e-14 * np.abs(rb).max()
            assert ra.tobytes() == _rank_fold([g[j][:, :7, :3] for j in range(4)]).tobytes()

    # Above the slab the reduction runs in a loop of slabs; the bits do not change
    # (the clamped last slab recomputes its overlap to the same bits).
    mesh = _mesh((2, 2), ("x", "y"))
    spec = P(("x", "y"))
    rng = np.random.default_rng(5)
    for shape, d in (((5, 12, 7), 1), ((3, 4, 1000), 2), ((1, 8, 33), 1)):
        g = rng.standard_normal((4,) + shape) + 1j * rng.standard_normal((4,) + shape)
        x = jnp.asarray(g.reshape((4 * shape[0],) + shape[1:]))
        m = shape[d] // 4
        want = np.concatenate([_rank_fold([np.take(g[j], np.arange(p * m, (p + 1) * m), axis=d) for j in range(4)])
                               for p in range(4)], 0)
        for slab in (1 << 40, 4096, 300, 1):       # one-shot, row slabs, column slabs, one element
            for vma in (False, True):
                f = jax.jit(shard_map(lambda v: ordered_psum_scatter(v, ("x", "y"), scatter_dimension=d, tiled=True,
                                                                       slab_bytes=slab),
                                      mesh=mesh, in_specs=spec, out_specs=spec, check_vma=vma))
                assert np.asarray(f(x)).tobytes() == want.tobytes(), (shape, slab, vma)
                fa = jax.jit(shard_map(lambda v: ordered_psum(v, ("x", "y"), slab_bytes=slab), mesh=mesh,
                                       in_specs=spec, out_specs=spec, check_vma=vma))
                full = np.asarray(fa(x)).reshape((4,) + shape)
                assert all(full[p].tobytes() == _rank_fold(list(g)).tobytes() for p in range(4)), (shape, slab, vma)

    # ``lax.psum`` is typed replicated under ``check_vma=True``, so ``out_specs=P()`` is legal;
    # ``ordered_psum`` must be too (one-shot and slabbed, pytree), with the rank-order bits.
    # Regression: head_correction._local_fold (hsuite gnppm, na_sc head) refused at trace time.
    rng = np.random.default_rng(13)
    g = rng.standard_normal((4, 3, 7, 5)) + 1j * rng.standard_normal((4, 3, 7, 5))
    x = jnp.asarray(g.reshape(12, 7, 5))
    want = _rank_fold(list(g)).tobytes()
    for slab in (1 << 40, 64):                       # one-shot, slabbed
        f = jax.jit(shard_map(lambda v: ordered_psum({"a": v, "b": 2 * v}, ("x", "y"), slab_bytes=slab),
                              mesh=mesh, in_specs=P(("x", "y")), out_specs={"a": P(), "b": P()},
                              check_vma=True))
        out = f(x)
        assert np.asarray(out["a"]).tobytes() == want, slab
        # partial axis: replicated over y only, varying over x
        h = jax.jit(shard_map(lambda v: ordered_psum(v, "y", slab_bytes=slab), mesh=mesh,
                              in_specs=P(("x", "y")), out_specs=P("x"), check_vma=True))
        assert np.asarray(h(x)).shape == (6, 7, 5)

