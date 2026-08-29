"""``cohsex_sigma`` face-layout dispatch: fast refusals and the
``_face_kwargs`` helper.  Emulated CPU mesh, unit-scope (QUALITY_PATTERNS'
FOUR-GPU RULE exempts unit/CPU cells).

The face NUMERIC kernels (``sigma_sx``/``sigma_coh``/``hartree`` under
``layout='face'``) go through ``distrib_la.gemm_plan`` (cuBLASMp,
CUDA-only today) AND the flat-k FFT FFI (unavailable in this sandbox's CPU
build — ``liblorrax_ffi_host.so`` missing/unloadable, a pre-existing,
already-registered environment gap, not something this file works around).
Their algebra parity against ``layout='legacy'`` is certified on the real
4-rank CUDA gate,
``tests/multi_device/low_mem_bands_g_projection_hartree_gate.py``.

What IS certified here, with no FFT and no gemm plan: ``layout`` and
``face_shape`` are validated BEFORE any FFT/FFI setup runs (a bad layout
string or a missing ``face_shape`` fails fast rather than surfacing as an
unrelated FFI probe error), and ``_face_kwargs`` reads the right shape off
a real face bundle.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.cohsex_sigma import _face_kwargs, _make_cohsex_kernels  # noqa: E402
from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices, build_wavefunctions, build_wavefunctions_face)


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


def _bundle(mesh, *, face: bool, nk=2, nb=6, ns=2, nmu=8):
    rng = np.random.default_rng(3)
    psi = (rng.standard_normal((nk, nb, ns, nmu))
           + 1j * rng.standard_normal((nk, nb, ns, nmu)))
    psi_rmuT_X = np.conj(psi).transpose(0, 3, 1, 2)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    y_in = _put(psi, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))
    builder = build_wavefunctions_face if face else build_wavefunctions
    return builder(y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)


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

def test_face_kwargs_empty_for_legacy_bundle():
    mesh = _mesh_xy()
    wfns = _bundle(mesh, face=False)
    assert _face_kwargs(wfns) == {}


def test_face_kwargs_reads_shape_off_the_face_bundle():
    mesh = _mesh_xy()
    nk, nb, ns, nmu = 2, 6, 2, 8
    wfns = _bundle(mesh, face=True, nk=nk, nb=nb, ns=ns, nmu=nmu)
    kw = _face_kwargs(wfns)
    assert kw["layout"] == "face"
    assert kw["face_shape"] == (nk, nb, nmu, ns)
