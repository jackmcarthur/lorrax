from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import (
    make_jittable_local_ifftn_3d,
    make_sharded_fftn_3d,
    make_sharded_ifftn_3d,
    query_fft_peak_bytes,
)
from runtime.aot_memory import aot_kernel_peak_bytes


def _single_device_mesh() -> Mesh:
    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devs, ("x", "y"))


def test_sharded_ifft_matches_plain_jnp_ifft():
    mesh = _single_device_mesh()
    spec = P(None, ("x", "y"), None, None, None, None)
    sharding = NamedSharding(mesh, spec)

    x_host = (
        np.arange(2 * 10 * 2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 10, 2, 3, 4, 5)
        + 1j
    ).astype(np.complex128)
    x = jax.device_put(jnp.asarray(x_host), sharding)

    default_ifft = jax.jit(
        make_sharded_ifftn_3d(mesh, spec, spec),
        in_shardings=sharding,
        out_shardings=sharding,
    )

    y_default = default_ifft(x)
    np.testing.assert_allclose(
        np.asarray(y_default),
        np.asarray(jnp.fft.ifftn(jnp.asarray(x_host), axes=(-3, -2, -1))),
        rtol=1e-12,
        atol=1e-12,
    )


def test_sharded_ifft_chunked_matches_plain_jnp_ifft():
    mesh = _single_device_mesh()
    spec = P(None, ("x", "y"), None, None, None, None)
    sharding = NamedSharding(mesh, spec)

    rng = np.random.default_rng(3)
    x_host = (
        rng.standard_normal((2, 7, 2, 3, 4, 5))
        + 1j * rng.standard_normal((2, 7, 2, 3, 4, 5))
    ).astype(np.complex128)
    x = jax.device_put(jnp.asarray(x_host), sharding)

    chunked_ifft = jax.jit(
        make_sharded_ifftn_3d(mesh, spec, spec, fft_batch_chunks=3),
        in_shardings=sharding,
        out_shardings=sharding,
    )

    np.testing.assert_allclose(
        np.asarray(chunked_ifft(x)),
        np.asarray(jnp.fft.ifftn(jnp.asarray(x_host), axes=(-3, -2, -1))),
        rtol=1e-12,
        atol=1e-12,
    )


def test_make_jittable_alias_matches_sharded_helper():
    mesh = _single_device_mesh()
    spec = P(None, ("x", "y"), None, None, None, None)
    sharding = NamedSharding(mesh, spec)

    rng = np.random.default_rng(0)
    x = jax.device_put(
        jnp.asarray(
            rng.standard_normal((1, 4, 2, 2, 3, 4))
            + 1j * rng.standard_normal((1, 4, 2, 2, 3, 4))
        ),
        sharding,
    )

    legacy = jax.jit(
        make_jittable_local_ifftn_3d(mesh, spec, spec),
        in_shardings=sharding,
        out_shardings=sharding,
    )
    current = jax.jit(
        make_sharded_ifftn_3d(mesh, spec, spec),
        in_shardings=sharding,
        out_shardings=sharding,
    )

    np.testing.assert_allclose(np.asarray(legacy(x)), np.asarray(current(x)))


def test_query_fft_peak_tracks_sharded_plan():
    mesh = _single_device_mesh()
    spec = P(None, ("x", "y"), None, None, None, None)
    sharding = NamedSharding(mesh, spec)
    shape = (2, 8, 2, 3, 4, 5)

    queried_peak = query_fft_peak_bytes(
        input_shape=shape,
        fft_axes=(-3, -2, -1),
        sharding=sharding,
        dtype=jnp.complex128,
    )

    fftn = jax.jit(
        make_sharded_fftn_3d(mesh, spec, spec),
        out_shardings=sharding,
    )
    compiled = fftn.lower(
        jax.ShapeDtypeStruct(shape, jnp.complex128, sharding=sharding)
    ).compile(compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    breakdown = aot_kernel_peak_bytes(compiled)

    assert queried_peak == breakdown.total


def test_query_fft_peak_tracks_chunked_sharded_plan():
    mesh = _single_device_mesh()
    spec = P(None, ("x", "y"), None, None, None, None)
    sharding = NamedSharding(mesh, spec)
    shape = (2, 8, 2, 3, 4, 5)

    queried_peak = query_fft_peak_bytes(
        input_shape=shape,
        fft_axes=(-3, -2, -1),
        sharding=sharding,
        dtype=jnp.complex128,
        fft_batch_chunks=4,
    )

    fftn = jax.jit(
        make_sharded_fftn_3d(mesh, spec, spec, fft_batch_chunks=4),
        out_shardings=sharding,
    )
    compiled = fftn.lower(
        jax.ShapeDtypeStruct(shape, jnp.complex128, sharding=sharding)
    ).compile(compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    breakdown = aot_kernel_peak_bytes(compiled)

    assert queried_peak == breakdown.total
