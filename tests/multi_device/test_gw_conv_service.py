"""GPU service routing and convolution values on a distributed band mesh."""

import numpy as np
import pytest

from runtime import initialize_communicator_stack

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common.fft_helpers import make_flat_k_gw_conv


@pytest.fixture(scope="module")
def runtime_mesh():
    return initialize_communicator_stack().mesh


@pytest.mark.parametrize("grid", [(3, 3, 1), (4, 4, 4), (3, 5, 2), (8, 8, 8)])
@pytest.mark.parametrize("mode,target", [
    ("off", "lorrax_mklfft_gw_conv"),
    ("on", "lorrax_cufft_conv_klead"),
    ("auto", "lorrax_cufft_conv_klead"),
])
def test_convolution_service_matches_numpy(grid, mode, target, monkeypatch, runtime_mesh):
    if jax.default_backend() != "gpu":
        pytest.skip("CUDA convolution comparison")
    monkeypatch.setenv("LORRAX_CONV_KLEAD_FFI", mode)
    mesh = runtime_mesh
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    nk, mu, nu = int(np.prod(grid)), 4 * px, 5 * py
    rng = np.random.default_rng(815)
    shape = (nk, 2, mu, 2, nu)
    g = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    w = rng.normal(size=(nk, mu, nu)) + 1j * rng.normal(size=(nk, mu, nu))
    g_spec, w_spec = P(None, None, "x", None, "y"), P(None, "x", "y")
    g_sharding, w_sharding = NamedSharding(mesh, g_spec), NamedSharding(mesh, w_spec)
    gd = jax.make_array_from_callback(g.shape, g_sharding, lambda idx: g[idx])
    wd = jax.make_array_from_callback(w.shape, w_sharding, lambda idx: w[idx])
    mult = -1 / np.sqrt(nk)
    run = jax.jit(make_flat_k_gw_conv(
        mesh, grid, P(None, None, None, None, "x", None, "y"),
        P(None, None, None, "x", "y"), mult=mult),
                  donate_argnums=(0,), out_shardings=g_sharding)
    compiled = run.lower(gd, wd).compile()
    hlo = compiled.as_text()
    assert f'custom_call_target="{target}"' in hlo
    assert "all-gather(" not in hlo
    assert "all-to-all(" not in hlo
    gr = np.fft.ifftn(g.reshape((*grid, *shape[1:])), axes=(0, 1, 2), norm="ortho")
    wr = np.fft.ifftn(w.reshape((*grid, mu, nu)), axes=(0, 1, 2), norm="ortho")
    reference = np.fft.fftn(gr * wr[:, :, :, None, :, None, :] * mult,
                             axes=(0, 1, 2), norm="ortho").reshape(shape)
    expected = jax.make_array_from_callback(
        reference.shape, g_sharding, lambda idx: reference[idx])
    actual = compiled(gd, wd)
    error = float(jax.device_get(jnp.max(jnp.abs(actual - expected))))
    scale = float(jax.device_get(jnp.max(jnp.abs(expected))))
    assert np.isfinite(error) and error <= 3e-12 * max(1.0, scale)
    assert actual.sharding.is_equivalent_to(g_sharding, actual.ndim)


def test_explicit_direct_refuses_unsupported_grid(monkeypatch, runtime_mesh):
    if jax.default_backend() != "gpu":
        pytest.skip("CUDA convolution policy")
    monkeypatch.setenv("LORRAX_CONV_KLEAD_FFI", "on")
    with pytest.raises(RuntimeError, match="runtime k-grid axis"):
        make_flat_k_gw_conv(runtime_mesh, (25, 1, 1),
                           P(None, None, None, None, "x", None, "y"),
                           P(None, None, None, "x", "y"))
