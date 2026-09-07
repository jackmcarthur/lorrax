"""Validate static kernel admission and canonical face shapes before backend setup."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.cohsex_sigma import (  # noqa: E402
    G_FFT7D_SPEC, V_FFT5D_SPEC, _face_kwargs, _make_cohsex_kernels,
    _make_static_convolution)
from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices, build_wavefunctions_face)


def _mesh_xy():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    elif devices.size >= 2:
        devices = devices[:2].reshape(1, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _bundle(mesh, *, nk=2, nb=6, ns=2, nmu=8):
    rng = np.random.default_rng(3)
    psi = (rng.standard_normal((nk, nb, ns, nmu))
           + 1j * rng.standard_normal((nk, nb, ns, nmu)))
    psi_rmuT_X = np.conj(psi).transpose(0, 3, 1, 2)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    y_in = _put(psi, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))
    return build_wavefunctions_face(y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)


# ---------------------------------------------------------------------------
# Fast refusals — before any FFT/FFI setup
# ---------------------------------------------------------------------------

def test_make_cohsex_kernels_rejects_bad_layout_before_fft_setup():
    mesh = _mesh_xy()
    with pytest.raises(ValueError, match="layout"):
        _make_cohsex_kernels(mesh, (1, 1, 2), 2, layout="bogus")


def test_make_cohsex_kernels_face_requires_face_shape_before_fft_setup():
    mesh = _mesh_xy()
    with pytest.raises(ValueError, match="face_shape"):
        _make_cohsex_kernels(mesh, (1, 1, 2), 2, layout="face")


# ---------------------------------------------------------------------------
# _face_kwargs
# ---------------------------------------------------------------------------

def test_face_kwargs_reads_shape_off_the_face_bundle():
    mesh = _mesh_xy()
    nk, nb, ns, nmu = 2, 6, 2, 8
    wfns = _bundle(mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    kw = _face_kwargs(wfns)
    assert kw["layout"] == "face"
    assert kw["face_shape"] == (nk, nb, nmu, ns)


def test_static_convolution_uses_certified_fused_owner(monkeypatch):
    """The face carrier must not materialize three full R-space G buffers."""
    mesh = _mesh_xy()
    called = {}

    def fake_factory(mesh_arg, kgrid, g_spec, v_spec, *, norm, mult):
        called.update(mesh=mesh_arg, kgrid=kgrid, g_spec=g_spec,
                      v_spec=v_spec, norm=norm, mult=float(mult))

        def fake_fused(G_k, V_q):
            return mult * (G_k + V_q[:, None, :, None, :])

        return fake_fused

    monkeypatch.setattr("ffi.mklfft.fused_fft_ffi_enabled", lambda: True)
    monkeypatch.setattr(
        "common.fft_helpers.make_flat_k_gw_conv", fake_factory)
    conv = _make_static_convolution(mesh, (1, 1, 2), 2)

    G = _put(np.ones((2, 1, 2, 1, 2), np.complex128),
             mesh, P(None, None, "x", None, "y"))
    V = _put(2.0 * np.ones((2, 2, 2), np.complex128),
             mesh, P(None, "x", "y"))
    got = np.asarray(conv(G, V, 2.5))

    assert called["mesh"] is mesh
    assert called["kgrid"] == (1, 1, 2)
    assert called["g_spec"] == G_FFT7D_SPEC
    assert called["v_spec"] == V_FFT5D_SPEC
    assert called["norm"] == "ortho"
    assert called["mult"] == pytest.approx(-1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(
        got, 2.5 * (-1.0 / np.sqrt(2.0)) * 3.0,
        rtol=0.0, atol=0.0)
