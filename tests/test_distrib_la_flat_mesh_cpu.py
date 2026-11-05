"""distrib_la's flattened-mesh products against NumPy: project_out, column_norms, replicated_gram.

(a) ``project_out`` on a 2x2 mesh, both flattened orders, q = 1 and 2, a
    partial last chunk (``n_valid``) and a clamped last panel (``w`` does not
    divide ``n/P``) equals NumPy's sequential chunk projections.  ``Q`` is not
    orthonormal, so a column projected twice would show.  The eager call
    consumes ``Y``, the call inside a donating ``jit`` gives the same bits,
    and that program holds no second copy of the rank's block.
    ``column_norms`` returns the norms ``project_out`` reports, ``Y`` intact.
(b) ``replicated_gram`` inside a ``shard_map`` and in global view equals
    ``a^H b``, with the same bits on a mesh whose devices are permuted (the
    partials are added in rank order, not device order).

CPU only; needs four virtual devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
import numpy as np
import pytest


def _mesh(order=(0, 1, 2, 3)):
    import jax
    from jax.sharding import Mesh
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 virtual CPU devices")
    return Mesh(np.asarray(jax.devices()[:4])[list(order)].reshape(2, 2), ("x", "y"))


def _complex(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _numpy_project(Y, Q, n_valid, chunk):
    """``Y <- (1 - Q_c Q_c^H) Y`` chunk by chunk, columns of ``Q`` past ``n_valid`` dropped."""
    Y = Y.copy()
    for b in range(Y.shape[0]):
        for c0 in range(0, n_valid, chunk):
            Qc = Q[b, :, c0:min(c0 + chunk, n_valid)]
            Y[b] = Y[b] - Qc @ (Qc.conj().T @ Y[b])
    return Y


def test_project_out_and_column_norms_match_numpy():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    jax.config.update("jax_enable_x64", True)
    import distrib_la
    mesh = _mesh()
    rng = np.random.default_rng(2)
    R, n_loc, t, chunk = 24, 40, 12, 4
    for flat, q, n_valid in ((("y", "x"), 1, 10), (("x", "y"), 2, 12), (("y", "x"), 1, 0)):
        spec = P(None, None, flat)
        put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))  # noqa: E731
        Yh = _complex(rng, q, R, 4 * n_loc)
        Qh = _complex(rng, q, R, t) / np.sqrt(2 * R)
        ref = _numpy_project(Yh, Qh, n_valid, chunk)
        norms = (np.abs(ref) ** 2).sum(axis=1)
        panel = 16 * q * R * 3            # w = 3: 14 panels over 40 columns, the last clamped
        kw = dict(mesh=mesh, panel_bytes=panel, spec=spec, chunk=chunk)

        Y = put(Yh, spec)
        Ye, de = distrib_la.project_out(Y, put(Qh, P()), n_valid, **kw)
        assert Y.is_deleted()
        np.testing.assert_allclose(np.asarray(Ye), ref, rtol=0, atol=1e-12 * np.abs(ref).max())
        np.testing.assert_allclose(np.asarray(de), norms, rtol=1e-12)

        run = jax.jit(lambda Y, Q, k: distrib_la.project_out(Y, Q, k, **kw), donate_argnums=(0,))
        args = (put(Yh, spec), put(Qh, P()), jnp.int32(n_valid))
        temp = run.lower(*args).compile().memory_analysis().temp_size_in_bytes
        assert temp < Yh.nbytes / 4, (temp, Yh.nbytes / 4)   # less than one rank's block
        Yj, dj = run(*args)
        np.testing.assert_array_equal(np.asarray(Yj), np.asarray(Ye))
        np.testing.assert_array_equal(np.asarray(dj), np.asarray(de))

        Yn = put(np.asarray(Ye), spec)
        dn = distrib_la.column_norms(Yn, mesh=mesh, panel_bytes=panel, spec=spec)
        np.testing.assert_array_equal(np.asarray(dn), np.asarray(de))
        np.testing.assert_array_equal(np.asarray(Yn), np.asarray(Ye))

    with pytest.raises(ValueError, match="spec must be"):
        distrib_la.project_out(put(Yh, spec), put(Qh, P()), mesh=mesh, panel_bytes=panel,
                               spec=P(None, "x", "y"))


def test_replicated_gram_is_rank_ordered_and_matches_numpy():
    import jax
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    jax.config.update("jax_enable_x64", True)
    import distrib_la
    rng = np.random.default_rng(4)
    a, b = _complex(rng, 4 * 6, 5), _complex(rng, 4 * 6, 3)
    flat = ("y", "x")
    rows = P(flat, None)
    results = []
    for mesh in (_mesh(), _mesh(order=(3, 1, 0, 2))):
        A, B = (jax.device_put(x, NamedSharding(mesh, rows)) for x in (a, b))
        inside = jax.jit(shard_map(lambda a, b: distrib_la.replicated_gram(a, b, axes=flat),
                                   mesh=mesh, in_specs=(rows, rows), out_specs=P(),
                                   check_vma=False))
        H_ab = np.asarray(distrib_la.replicated_gram(A, B, axes=flat, mesh=mesh))
        H_aa = np.asarray(distrib_la.replicated_gram(A, axes=flat, mesh=mesh))
        np.testing.assert_array_equal(np.asarray(inside(A, B)), H_ab)
        np.testing.assert_allclose(H_ab, a.conj().T @ b, rtol=0, atol=1e-13 * np.abs(H_ab).max())
        np.testing.assert_allclose(H_aa, a.conj().T @ a, rtol=0, atol=1e-13 * np.abs(H_aa).max())
        results.append((H_ab, H_aa))
    for x, y in zip(*results):
        np.testing.assert_array_equal(x, y)
