"""CUDA value parity for both ISDF conv_kpair arms."""

import os

import numpy as np
import pytest


def _reference(jnp, a, b, perm_l, phase_l, perm_r, phase_r):
    ar = jnp.fft.ifftn(a, axes=(0, 1, 2), norm="forward")
    br = jnp.fft.ifftn(b, axes=(0, 1, 2), norm="forward")
    out = jnp.zeros(a.shape[:3] + a.shape[4:6], dtype=a.dtype)
    ns = a.shape[3]
    for alpha in range(ns):
        for beta in range(ns):
            out = out + (phase_l[alpha] * phase_r[beta]
                         * jnp.conj(ar[:, :, :, alpha, :, :, beta])
                         * br[:, :, :, perm_l[alpha], :, :, perm_r[beta]])
    return jnp.fft.fftn(out, axes=(0, 1, 2), norm="forward")


@pytest.mark.parametrize(
    "kgrid,arm",
    [
        ((1, 1, 1), "resident"),
        ((3, 3, 3), "resident"),
        ((4, 4, 4), "resident"),
        ((2, 3, 5), "resident"),
        ((5, 5, 5), "resident"),
        ((16, 16, 16), "two_stage"),
    ],
)
@pytest.mark.parametrize("ns", [1, 2, 4])
def test_conv_kpair_matches_xla(kgrid, arm, ns):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    if not jax.devices() or jax.devices()[0].platform != "gpu":
        pytest.skip("conv_kpair is CUDA-only")
    from ffi.fft import make_fused_conv_kpair

    os.environ["LORRAX_CONV_KPAIR_FFI"] = "on"
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("x",))
    if ns == 1:
        perm_l = perm_r = np.asarray([0], dtype=np.int64)
        phase_l = phase_r = np.asarray([1], dtype=np.complex128)
    elif ns == 2:
        # Two distinct non-identity monomials exercise both permutation and
        # all four exact phase quadrants across the double contraction.
        perm_l = np.asarray([1, 0], dtype=np.int64)
        phase_l = np.asarray([1j, -1j], dtype=np.complex128)
        perm_r = np.asarray([1, 0], dtype=np.int64)
        phase_r = np.asarray([-1, 1], dtype=np.complex128)
    else:
        # Production bispinor Meta lifts the WFN's two components to four
        # channels.  Exercise a nontrivial permutation and every exact phase
        # quadrant through the same generic device contract.
        perm_l = np.asarray([1, 0, 3, 2], dtype=np.int64)
        phase_l = np.asarray([1, 1j, -1, -1j], dtype=np.complex128)
        perm_r = np.asarray([2, 3, 0, 1], dtype=np.int64)
        phase_r = np.asarray([-1j, -1, 1j, 1], dtype=np.complex128)
    rng = np.random.default_rng(7300 + int(np.prod(kgrid)) + ns)
    shape = tuple(kgrid) + (ns, 2, 3, ns)
    a_np = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    b_np = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    a = jnp.asarray(a_np, dtype=jnp.complex128)
    b = jnp.asarray(b_np, dtype=jnp.complex128)
    native = jax.jit(make_fused_conv_kpair(
        mesh, kgrid, perm_l=perm_l, phase_l=phase_l,
        perm_r=perm_r, phase_r=phase_r, arm=arm, norm="forward"))
    got = native(a, b)
    ref = jax.jit(lambda x, y: _reference(
        jnp, x, y, perm_l, phase_l, perm_r, phase_r))(a, b)
    got_np, ref_np = np.asarray(got), np.asarray(ref)
    rel = np.max(np.abs(got_np - ref_np)) / max(np.max(np.abs(ref_np)), 1e-300)
    print(f"conv_kpair parity kgrid={kgrid} ns={ns} arm={arm} rel={rel:.16e}")
    assert rel <= 1e-13, (kgrid, ns, arm, rel)
