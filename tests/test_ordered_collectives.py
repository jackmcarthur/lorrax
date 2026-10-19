"""lxkit.ordered_psum_scatter / ordered_psum against lax.psum_scatter / lax.psum on four host devices.

Values: equal to the library collective to round-off (1e-14 relative) and BITWISE equal to the numpy
left-to-right rank-order sum (the helpers' contract), for a 1-D and a 2x2 mesh, tiled and untiled,
f64 and c128, with a padded all-reduce.  CPU: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
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


def test_ordered_psum_scatter_and_psum():
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


def test_slabbed_equals_one_shot_and_fold():
    """Above the slab the reduction runs in a loop of slabs; the bits do not change
    (the clamped last slab recomputes its overlap to the same bits)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import ordered_psum, ordered_psum_scatter
    jax.config.update("jax_enable_x64", True)
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


def test_exact_cases_use_the_library_collective():
    """Two ranks, one rank and integer operands: the library collective, no all_to_all."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import ordered_psum, ordered_psum_scatter
    mesh = _mesh((2, 2), ("x", "y"))
    spec = P(("x", "y"))
    cases = (lambda v: ordered_psum(v, "y"),
             lambda v: ordered_psum_scatter(v, "x", scatter_dimension=1, tiled=True),
             lambda v: ordered_psum({"i": v.astype(jnp.int32)}, ("x", "y"))["i"].astype(v.dtype))
    for body in cases:
        f = jax.jit(shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
        hlo = f.lower(jnp.ones((8, 4))).compile().as_text()
        assert "all-to-all" not in hlo
    f = jax.jit(shard_map(lambda v: ordered_psum(v, ("x", "y")), mesh=mesh, in_specs=spec, out_specs=spec,
                          check_vma=False))
    assert "all-to-all" in f.lower(jnp.ones((8, 4))).compile().as_text()
    x = jnp.arange(32.0).reshape(8, 4)
    a = jax.jit(shard_map(lambda v: ordered_psum(v, "y"), mesh=mesh, in_specs=spec, out_specs=spec,
                          check_vma=False))(x)
    assert np.asarray(a).shape == (8, 4)


def test_reduction_guard_refuses_unmarked_raw_sum():
    import jax
    import jax.numpy as jnp
    import pytest
    from jax import lax, shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import collectives as C
    mesh = _mesh((2, 2), ("x", "y"))
    spec = P(("x", "y"))
    raw = {n: getattr(lax, n) for n in ("psum", "psum_scatter", "pmean")}
    C.install_reduction_guard("raise")
    try:
        run = lambda body, x: jax.jit(shard_map(body, mesh=mesh, in_specs=spec, out_specs=P(), check_vma=False))(x)
        with pytest.raises(RuntimeError, match="exact-reduce"):
            run(lambda v: lax.psum(v, ("x", "y")), jnp.ones((8, 2)))
        run(lambda v: lax.psum(v, ("x", "y")), jnp.ones((8, 2), jnp.int32))           # integer: exact
        run(lambda v: lax.psum(v, ("x", "y")), jnp.ones((8, 3)))  # exact-reduce: test of the marker
        run(lambda v: C.ordered_psum(v, ("x", "y")), jnp.ones((8, 4)))
        jax.jit(shard_map(lambda v: lax.psum(v, "y"), mesh=mesh, in_specs=spec, out_specs=spec,
                          check_vma=False))(jnp.ones((8, 5)))                             # two ranks: exact
        sites = [r["path"] for r in C.reduction_trace()]
        assert "exact-marked" in sites and "UNORDERED" in sites and "one-shot" in sites
    finally:
        for n, f in raw.items():
            setattr(lax, n, f)
        C._GUARD["mode"] = None
        C._TRACE.clear()


def _rows_gram(a):
    import jax.numpy as jnp
    ax = tuple(range(1, a.ndim))
    return jnp.tensordot(jnp.conj(a), a, axes=(ax, ax))


def test_ordered_reduce_on_sharded_arrays():
    """Eager reduction of a mesh-sharded array: the rank-order fold of the per-shard partials,
    replicated, and no library all-reduce over the four ranks."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import ordered_reduce
    jax.config.update("jax_enable_x64", True)
    mesh = _mesh((2, 2), ("x", "y"))
    rng = np.random.default_rng(9)
    g = rng.standard_normal((5, 8, 6)) + 1j * rng.standard_normal((5, 8, 6))
    a = jax.device_put(jnp.asarray(g), NamedSharding(mesh, P(None, "x", "y")))
    got = np.asarray(ordered_reduce(_rows_gram, a))
    parts = []
    for ix in range(2):                       # rank order = mesh order: (x, y) row-major
        for iy in range(2):
            blk = g[:, 4 * ix:4 * ix + 4, 3 * iy:3 * iy + 3]
            parts.append(np.asarray(_rows_gram(jnp.asarray(blk))))
    assert got.tobytes() == _rank_fold(parts).tobytes()
    from lxkit import collectives as C
    prog = next(v for k, v in C._REDUCE_PROGRAMS.items() if k[0] is _rows_gram)
    hlo = prog.lower(a).compile().as_text()
    assert "all-to-all" in hlo and not any("all-reduce" in l and "{0,1,2,3}" in l for l in hlo.splitlines())
    r = np.asarray(ordered_reduce(_rows_gram, jnp.asarray(g)))          # unsharded: plain call
    assert np.allclose(r, np.asarray(_rows_gram(jnp.asarray(g))))
    import pytest
    with pytest.raises(TypeError, match="shard_map"):                 # traced: refused, never unordered
        jax.jit(lambda v: ordered_reduce(_rows_gram, v))(a)


def test_ordered_sum_is_placement_independent():
    """``lxkit.ordered_sum`` / ``ordered_norm``: inside a jit on the caller's layout (``mesh=``,
    ``spec=``), and eagerly on a concrete array's own sharding -- no library all-reduce over
    the four ranks, the right value, the same bits on permuted device orders; a tracer
    without a layout is refused; a plain sum inside ``shard_map``."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from jax import shard_map
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit import ordered_norm, ordered_psum, ordered_sum
    jax.config.update("jax_enable_x64", True)
    _mesh((2, 2), ("x", "y"))
    devs = np.asarray(jax.devices()[:4])
    rng = np.random.default_rng(11)
    g = rng.standard_normal((5, 8, 12)) + 1j * rng.standard_normal((5, 8, 12))
    for spec, axis in ((P(None, "x", "y"), (-2, -1)), (P(None, "x", "y"), -1), (P(None, "x", None), (1, 2)),
                       (P(None, ("x", "y"), None), 1), (P(), (1, 2))):
        traced, eager = [], []
        for perm in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]):
            mesh = Mesh(devs[perm].reshape(2, 2), ("x", "y"))
            a = jax.device_put(jnp.asarray(g), NamedSharding(mesh, spec))
            f = jax.jit(lambda v: ordered_sum(v, axis, mesh=mesh, spec=spec))
            hlo = f.lower(a).compile().as_text()
            assert not any("all-reduce" in l and "{0,1,2,3}" in l for l in hlo.splitlines()), (spec, axis)
            r = np.asarray(f(a))
            assert np.allclose(r, g.sum(axis=axis)), (spec, axis)
            traced.append(r.tobytes())
            eager.append(np.asarray(ordered_sum(a, axis)).tobytes())
        assert traced[0] == traced[1] == traced[2] and eager[0] == eager[1] == eager[2], (spec, axis)
    mesh = Mesh(devs.reshape(2, 2), ("x", "y"))
    a = jax.device_put(jnp.asarray(g), NamedSharding(mesh, P(None, "x", "y")))
    assert np.allclose(np.asarray(ordered_norm(a, (-2, -1))), np.linalg.norm(g, axis=(1, 2)))
    assert np.allclose(np.asarray(ordered_sum(a, (1, 2), keepdims=True)), g.sum(axis=(1, 2), keepdims=True))
    with pytest.raises(TypeError, match="mesh= and spec="):
        jax.jit(lambda v: ordered_sum(v, -1))(a)
    sm = jax.jit(shard_map(lambda v: ordered_psum(ordered_sum(v, (-2, -1)), ("x", "y")), mesh=mesh,
                           in_specs=P(None, "x", "y"), out_specs=P(), check_vma=False))
    assert np.allclose(np.asarray(sm(a)), g.sum(axis=(1, 2)))


def test_planner_budget_reserves_the_ordered_transient():
    import runtime
    from lxkit import collectives as C
    assert C.ordered_transient_bytes() == C.ORDERED_TRANSIENT_SLABS * C.slab_bytes() == 2 * (8 << 20)
    total = 64e9
    plain = ((total - runtime.MEMORY_MARGIN * runtime.outside_pool_allowance_bytes("local"))
             / (1.0 + runtime.MEMORY_MARGIN * runtime.POOL_OVERSHOOT))
    want = max(min(plain - C.ordered_transient_bytes(), runtime.pool_reservation_bytes(total)), 0.25 * total)
    assert runtime.planner_budget_bytes(total, "local") == want


def test_ordered_psum_replicated_out_specs_under_check_vma():
    """``lax.psum`` is typed replicated under ``check_vma=True``, so ``out_specs=P()`` is legal;
    ``ordered_psum`` must be too (one-shot and slabbed, pytree), with the rank-order bits.
    Regression: head_correction._local_fold (hsuite gnppm, na_sc head) refused at trace time."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import ordered_psum
    jax.config.update("jax_enable_x64", True)
    mesh = _mesh((2, 2), ("x", "y"))
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
