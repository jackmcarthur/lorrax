"""ROCm single-matrix dispatch/triangle/failure contracts without a GPU."""
import runtime  # noqa: F401
import numpy as np
import pytest


def _owner():
    import jax
    jax.config.update('jax_enable_x64', True)
    from distrib_la import local_eigh as le
    return jax, le


def test_single_matrix_native_promotion_preserves_triangles_and_reflector(monkeypatch):
    jax, le = _owner()
    import jax.numpy as jnp
    monkeypatch.setattr(le, '_process_leg_key', lambda: 'ROCM')
    monkeypatch.setattr(le, 'eigh_local_jacobi', jnp.linalg.eigh)
    row = le.BATCHED_EIGH_ROUTE['ROCM']
    # A small CPU oracle exercises the same production driver; shape-only
    # admission below separately pins the actual measured n=128..3120 window.
    win = row.windows['complex128']._replace(min_n=2)
    monkeypatch.setitem(le.BATCHED_EIGH_ROUTE, 'ROCM', row._replace(
        windows={**row.windows, 'complex128': win}))
    monkeypatch.setattr(le, '_require_target', lambda _target: None)
    received = []

    def native(_target, a, driver, tile):
        received.append((a.shape, driver, tile))
        w, v = jnp.linalg.eigh(a)
        return w, v, jnp.zeros(a.shape[:-2], dtype=jnp.int32)

    monkeypatch.setattr(le, '_batched_call', native)
    rng = np.random.default_rng(318)
    raw = rng.normal(size=(10, 10))+1j*rng.normal(size=(10, 10))
    for uplo, sym in ((None, True), ('L', False), ('U', False)):
        full = ((raw+raw.conj().T)/2 if sym else
                np.tril(raw)+np.tril(raw, -1).conj().T if uplo == 'L' else
                np.triu(raw)+np.triu(raw, 1).conj().T)
        # Hermitian diagonal is real, as required by both LAPACK drivers.
        full[np.diag_indices(10)] = full.diagonal().real
        raw[np.diag_indices(10)] = raw.diagonal().real
        w, v = jax.jit(lambda a: le.eigh_local(a, UPLO=uplo,
                                              symmetrize_input=sym))(jnp.asarray(raw))
        w, v = np.asarray(w), np.asarray(v)
        np.testing.assert_allclose(full @ v, v*w[None], atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(v.conj().T @ v, np.eye(10), atol=2e-14)
        np.testing.assert_allclose(w, np.linalg.eigvalsh(full), atol=2e-14)
    assert received and all(shape == (1, 10, 10) and driver == 'heevd'
                            for shape, driver, _tile in received)


def test_single_matrix_window_and_workspace_price_are_the_same_owner(monkeypatch):
    jax, le = _owner()
    n = 3072
    row = le.batched_eigh_row((n, n), np.complex128, 'ROCM')
    assert row is not None and row.windows['complex128'].min_batch == 1
    assert le.batched_eigh_row((1, n, n), np.complex128, 'ROCM') == row
    win = row.windows['complex128']
    expected = win.workspace_tiles*n*n*16+n*8  # retained scalar Jacobi allowance
    arena = 29_483_264  # provider reply, deliberately not the former 4-tile guess
    monkeypatch.setattr(le, '_workspace_bytes', lambda *_args: arena)
    assert le.batched_workspace_bytes((n, n), np.complex128, 'ROCM') == arena
    assert le.batched_workspace_bytes((1, n, n), np.complex128, 'ROCM') == arena
    from types import SimpleNamespace
    from distrib_la import workspace
    from jax.sharding import Mesh
    monkeypatch.setattr(workspace, 'mesh_platform', lambda _mesh: 'ROCM')
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    receipt = workspace._off_cuda(SimpleNamespace(mesh=mesh), 'eigh',
                                  ((n, n),), np.dtype(np.complex128))
    assert receipt['device_bytes'] == expected
    assert 'local_eigh_workspace' in receipt['formula']
    for shape, dtype, platform in (((127, 127), np.complex128, 'ROCM'),
                                    ((3121, 3121), np.complex128, 'ROCM'),
                                    ((n, n), np.float64, 'ROCM'),
                                    ((n, n), np.complex128, 'CUDA'),
                                    ((n, n), np.complex128, 'CPU')):
        assert le.batched_eigh_row(shape, dtype, platform) is None
        assert le.batched_workspace_bytes(shape, dtype, platform) == 0


def test_single_matrix_keeps_native_refusal_and_failure_marking(monkeypatch):
    _jax, le = _owner()
    import jax.numpy as jnp
    import distrib_la.loader as loader
    monkeypatch.setattr(le, '_process_leg_key', lambda: 'ROCM')
    monkeypatch.setattr(le, 'eigh_local_jacobi', lambda a: (
        jnp.full(a.shape[:-1], jnp.nan), jnp.full_like(a, jnp.nan)))
    row = le.BATCHED_EIGH_ROUTE['ROCM']
    win = row.windows['complex128']._replace(min_n=2)
    monkeypatch.setitem(le.BATCHED_EIGH_ROUTE, 'ROCM', row._replace(
        windows={**row.windows, 'complex128': win}))
    a = jnp.asarray([[2., .3j], [-.3j, 4.]], dtype=jnp.complex128)
    monkeypatch.setattr(loader, 'probe_target', lambda *_args: (False, 'deliberately absent'))
    with pytest.raises(RuntimeError, match='GATE batched_eigh_target'):
        le.eigh_local(a)
    monkeypatch.setattr(le, '_require_target', lambda _target: None)
    for silent in (False, True):
        def failed(_target, b, _driver, _tile):
            return (jnp.zeros(b.shape[:-1]), jnp.full_like(b, jnp.nan) if silent
                    else jnp.broadcast_to(jnp.eye(2, dtype=b.dtype), b.shape),
                    jnp.zeros(b.shape[:-2], dtype=jnp.int32) if silent else
                    jnp.ones(b.shape[:-2], dtype=jnp.int32))
        monkeypatch.setattr(le, '_batched_call', failed)
        w, v = le.eigh_local(a)
        assert np.isnan(w).all() and np.isnan(v).all()


def test_single_matrix_retry_uses_same_input_only_after_a_finite_input_failure(monkeypatch):
    jax, le = _owner()
    import jax.numpy as jnp
    monkeypatch.setattr(le, '_process_leg_key', lambda: 'ROCM')
    row = le.BATCHED_EIGH_ROUTE['ROCM']
    monkeypatch.setitem(le.BATCHED_EIGH_ROUTE, 'ROCM', row._replace(windows={
        **row.windows, 'complex128': row.windows['complex128']._replace(min_n=2)}))
    monkeypatch.setattr(le, '_require_target', lambda _target: None)
    retries = []

    def native(_target, a, _driver, _tile):
        w, v = jnp.linalg.eigh(a)
        # The vendor's observed silent mode: finite values, NaN vectors, info=0.
        bad = jnp.real(a[..., 0, 0]) < 0
        return w, jnp.where(bad[..., None, None], jnp.nan, v), jnp.zeros(
            a.shape[:-2], dtype=jnp.int32)

    def jacobi(a):
        jax.debug.callback(lambda x: retries.append(np.asarray(x).copy()), a)
        return jnp.linalg.eigh(a)

    monkeypatch.setattr(le, '_batched_call', native)
    monkeypatch.setattr(le, 'eigh_local_jacobi', jacobi)
    solve = jax.jit(le._solve)
    good = jnp.asarray([[2., .3j], [-.3j, 4.]], dtype=jnp.complex128)
    failed = good - 5 * jnp.eye(2)
    for a in (good, failed):
        w, v = solve(a)
        jax.block_until_ready((w, v))
        np.testing.assert_allclose(a @ v, v * w[None], atol=2e-14)
        np.testing.assert_allclose(v.conj().T @ v, np.eye(2), atol=2e-14)
    invalid = good.at[0, 0].set(jnp.nan)
    w, v = solve(invalid)
    jax.block_until_ready((w, v))
    jax.effects_barrier()
    assert len(retries) == 1
    np.testing.assert_array_equal(retries[0], failed)
    assert not (np.isfinite(w).all() and np.isfinite(v).all())


def test_cpu_and_cuda_dispatch_stays_identical_to_jaxlib(monkeypatch):
    _jax, le = _owner()
    import jax.numpy as jnp
    a = jnp.asarray([[2., .3j], [-.3j, 4.]], dtype=jnp.complex128)
    def forbidden(*_args):
        raise AssertionError('No native ROCm dispatch is allowed on another leg')
    monkeypatch.setattr(le, '_native_batched_eigh', forbidden)
    expected = jnp.linalg.eigh(a)
    for leg in ('CPU', 'CUDA'):
        monkeypatch.setattr(le, '_process_leg_key', lambda: leg)
        actual = le.eigh_local(a)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)


def test_rocm_complex64_keeps_the_previous_solver(monkeypatch):
    _jax, le = _owner()
    import jax.numpy as jnp
    monkeypatch.setattr(le, '_process_leg_key', lambda: 'ROCM')
    def forbidden(*_args):
        raise AssertionError('The measured complex128 retry must not change complex64')
    monkeypatch.setattr(le, 'eigh_local_jacobi', forbidden)
    a = jnp.asarray([[2., .3j], [-.3j, 4.]], dtype=jnp.complex64)
    for got, expected in zip(le._solve(a), jnp.linalg.eigh(a)):
        np.testing.assert_array_equal(got, expected)
