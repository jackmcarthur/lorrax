"""Validate canonical face construction, masks, residency and projection admission."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    PSI_MUN_SPEC,
    PSI_NMU_SPEC,
    Wavefunctions,
    build_wavefunctions_face,
    project,
)

_C128 = 16


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


def _host_inputs(rng, nk, nb, ns, nmu):
    """(psi_rmu_Y, psi_rmuT_X) host numpy, matching
    the loader conventions: psi_rmu_Y is
    un-conjugated ψ (nk,nb,ns,nmu); psi_rmuT_X is CONJUGATED ψ*
    (nk,nmu,nb,ns) — the same un-conjugated ψ, conjugated and permuted.
    """
    psi = (rng.standard_normal((nk, nb, ns, nmu))
           + 1j * rng.standard_normal((nk, nb, ns, nmu)))
    psi_rmuT_X = np.conj(psi).transpose(0, 3, 1, 2)   # (nk, nmu, nb, ns)
    return psi, psi_rmuT_X


def _gather(arr):
    return np.asarray(jax.device_get(arr))


def test_layout_defaults_and_bad_tag_refuses():
    mesh = _mesh_xy()
    rep = _put(np.zeros((1, 1)), mesh, P(None, None))
    slices = BandSlices.from_band_edges(0, 0, 1, 1, 1)
    face = Wavefunctions(enk=rep, occ=rep, slices=slices)
    assert face.layout == "face"
    assert face.psi_nmu is None and face.psi_mun is None
    for layout in ("legacy", "bogus"):
        with pytest.raises(ValueError):
            Wavefunctions(enk=rep, occ=rep, slices=slices, layout=layout)


def test_face_matches_input_transposes():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260822)
    nk, nb, ns, nmu = 3, 10, 2, 12
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 3, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    np.testing.assert_array_equal(_gather(face.psi_nmu), psi_rmu_Y)
    np.testing.assert_array_equal(
        _gather(face.psi_mun), psi_rmu_Y.transpose(0, 2, 3, 1))

    # shardings are the declared face specs, not whatever propagation
    # happened to produce
    got_nmu_spec = tuple(face.psi_nmu.sharding.spec)
    got_mun_spec = tuple(face.psi_mun.sharding.spec)
    want_nmu = tuple(PSI_NMU_SPEC)
    want_mun = tuple(PSI_MUN_SPEC)
    # PartitionSpec may report trailing Nones trimmed; pad for comparison.
    def _padded(t, n):
        return tuple(t) + (None,) * (n - len(t))
    assert _padded(got_nmu_spec, 4) == _padded(want_nmu, 4)
    assert _padded(got_mun_spec, 4) == _padded(want_mun, 4)
    assert face.layout == "face"


def test_memory_model_prices_resolved_layout():
    from gw.gflat_memory_model import _persistent_bytes

    nk, ns, mu, nb = 4, 2, 512, 64
    p_x, p_y = 2, 2
    s = _C128 * nk * ns * mu * nb  # one global complex128 psi image

    face = _persistent_bytes(
        nk=nk, ns=ns, nq=1, nq_disk=1, mu=mu, nb=nb, ngkmax=1, n_rtot=1,
        p_x=p_x, p_y=p_y)
    assert face["psi_copies"] == pytest.approx(2 * s / (p_x * p_y))


def test_face_carrier_addressable_bytes_match_2s_over_p():
    mesh = _mesh_xy()
    px, py = mesh.devices.shape
    p = px * py
    rng = np.random.default_rng(20260822)
    nk, nb, ns, nmu = 2, 8, 2, 8  # nmu divisible by 2 for a clean p_y=2 shard
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    s = _C128 * nk * ns * nmu * nb
    want_per_rank = 2 * s / p  # both faces, full mesh sharded

    for arr in (face.psi_nmu, face.psi_mun):
        for sh in arr.addressable_shards:
            got = int(np.asarray(sh.data).nbytes)
            # one shard's worth of ONE face: s/p -- the per-DEVICE figure
            # the guide's "~2S/P per rank" claim is about (one rank == one
            # device on the real multi-rank launch this emulates).
            assert got == pytest.approx(s / p), (
                f"face shard bytes {got} != expected s/p={s / p}")


def test_band_mask_matches_manual_boolean_slice():
    """Mask exactly the requested logical band interval on canonical faces."""
    mesh = _mesh_xy()
    rng = np.random.default_rng(7)
    nk, nb, ns, nmu = 3, 10, 2, 12
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 3, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    for bands, label in ((slices.val, "val"), (slices.cond, "cond"),
                        (slice(2, 7), "arbitrary")):
        lo = bands.start or 0
        hi = bands.stop
        want = np.zeros((nk, nb), dtype=bool)
        want[:, lo:hi] = True
        got = np.asarray(face.band_mask(bands))
        assert got.dtype == np.bool_
        assert np.array_equal(got, want), f"band_mask({label}) mismatch"


def test_band_mask_covers_the_full_extent_with_no_stop():
    mesh = _mesh_xy()
    rng = np.random.default_rng(8)
    nk, nb, ns, nmu = 2, 6, 1, 8
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))
    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    got = np.asarray(face.band_mask(slice(3, None)))
    want = np.zeros((nk, nb), dtype=bool)
    want[:, 3:] = True
    assert np.array_equal(got, want)


def test_project_rejects_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        project(jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)),
               jnp.zeros((1, 1, 1, 1, 1)), layout="bogus")


def test_project_face_requires_mesh_or_prebuilt_projector():
    with pytest.raises(ValueError, match="face_project_fn|mesh_xy"):
        project(jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)),
               jnp.zeros((1, 1, 1, 1, 1)), layout="face")
