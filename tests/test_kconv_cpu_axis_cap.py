"""The k-conv router's 40-per-axis cap is the fp64 cuFFTDx limit, so only the CUDA leg applies it.

Red twin: before the cap moved onto the mathdx leg, the cpu doors (and
``common.fft_helpers.make_flat_k_fft`` through them) refused a 48-point k axis
with ``GATE mathdx-kconv-axis`` although FFTW has no such limit.
"""
import numpy as np
import pytest


def test_cpu_kfft_door_serves_an_axis_above_the_cuda_cap_and_mathdx_still_refuses():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P
    from ffi import fft as F

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    kg = (F.KCONV_AXIS_MAX + 8, 1, 1)
    nk = int(np.prod(kg))
    rng = np.random.default_rng(48)
    x = rng.standard_normal((nk, 3)) + 1j * rng.standard_normal((nk, 3))
    fn = F.make_kfft_klead(mesh, kg, P(None, None, None, None), kind="ifftn", norm="ortho")
    got = np.asarray(jax.jit(fn)(jnp.asarray(x, dtype=jnp.complex128)))
    ref = np.fft.ifftn(x.reshape(kg + (3,)), axes=(0, 1, 2), norm="ortho").reshape(nk, 3)
    assert np.max(np.abs(got - ref)) < 1e-12
    with pytest.raises(RuntimeError, match="GATE mathdx-kconv-axis"):
        F._check_kgrid(kg, "mathdx")
