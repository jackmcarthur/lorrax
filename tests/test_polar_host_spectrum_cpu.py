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
