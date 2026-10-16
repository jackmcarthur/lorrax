"""distrib_la.polar._host_spectrum: device-side agreement returns the replicated spectrum's exact bits."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


def _mesh():
    n = 4 if len(jax.devices()) >= 4 else 1
    return jax.sharding.Mesh(np.array(jax.devices()[:n]).reshape(n // 2 or 1, 2 if n == 4 else 1), ("x", "y"))


@pytest.mark.parametrize("shape", [(312,), (4, 312), (2, 3, 17)])
def test_agreed_spectrum_is_bitwise_the_replica(shape):
    jax.config.update("jax_enable_x64", True)
    from distrib_la import polar
    x = np.random.default_rng(1).standard_normal(shape)
    flat = x.reshape(-1)
    flat[0], flat[1], flat[2] = -0.0, np.nan, np.inf
    s = jax.device_put(x, NamedSharding(_mesh(), P()))
    got = polar._host_spectrum(s)
    want = x[..., ::-1].copy()
    assert got.dtype == np.float64 and got.shape == want.shape
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def test_retained_spectra_are_host_arrays():
    jax.config.update("jax_enable_x64", True)
    from distrib_la import polar
    mesh = _mesh()
    py = int(mesh.shape["y"])
    q = jax.device_put(np.eye(8 * py)[None].repeat(2, 0), NamedSharding(mesh, P(None, "x", "y")))
    values = (np.arange(8.0)[::-1].copy(), np.arange(8.0)[::-1].copy())
    Q, spectra = polar._retained_columns(q, values, (3, 5), mesh=mesh,
                                         column_extent=lambda n: -(-n // py) * py, descending=True)
    assert all(isinstance(r, np.ndarray) and r.dtype == np.float64 for r in spectra)
    assert [r.size for r in spectra] == [3, 5]
    np.testing.assert_array_equal(spectra[1], values[1][:5])


def test_device_cut_equals_host_cut():
    jax.config.update("jax_enable_x64", True)
    from distrib_la import polar
    mesh = _mesh()
    rep = NamedSharding(mesh, P())
    rng = np.random.default_rng(5)
    for trial in range(60):
        b, n = 4, int(rng.integers(2, 30))
        rows = np.sort(rng.random((b, n)) ** int(rng.integers(1, 6)))[:, ::-1].copy()
        for r in range(b):
            k = int(rng.integers(1, n)); rows[r, k] = rows[r, k - 1] * (1 - [1e-9, 1e-7, 1e-3][trial % 3])
        rows = np.sort(rows, axis=1)[:, ::-1].copy()
        agreed = jax.device_put(rows[:, ::-1].copy(), rep)
        tol, tau, real, mr = [1e-6, 1e-3, 0.0][trial % 3], [1e-3, 0.1, 0.5][trial % 3], 1 + trial % b, [None, 3][trial % 2]

        def cut(row):
            c = int(np.count_nonzero(row > tau * row[0]))
            return polar._close_spectral_cut(row, c if mr is None else min(c, mr), tol)
        host = polar._real_row_counts(tuple(rows), cut, real)
        dev = polar._device_cut_program(mesh, "singular", rows.shape, tau, mr, tol, real, None)(agreed)
        assert np.array_equal(np.asarray(dev), host)
        r, rc = int(rng.integers(1, n + 1)), [None, 1e-2][trial % 2]
        hostL = polar._leading_counts(tuple(rows), r, multiplet_tol=tol, real_rows=real, batched=True, rcond=rc)
        devL = polar._device_cut_program(mesh, "leading", rows.shape, r, None, tol, real, rc)(agreed)
        assert np.array_equal(np.asarray(devL), hostL)
