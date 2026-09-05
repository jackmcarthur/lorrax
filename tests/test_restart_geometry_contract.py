"""Restart logical/storage geometry: hostile unit twins, no FFI required.

The production transport is SlabIO.  These cells replace only that transport
with a tiny h5py implementation so they can inspect the exact request made at
the boundary: all geometry decisions remain the production
``file_io.tagged_arrays`` code.
"""
from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from jax.sharding import Mesh

from common import collectives
from file_io import slab_io, tagged_arrays


class _HostSlabIO:
    """The subset of SlabIO needed by this boundary test."""

    opens = []

    def __init__(self, path, *, mode="w", mesh=None):
        del mesh
        self.opens.append((str(path), mode))
        self._f = h5py.File(path, mode)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._f.close()

    def write_attr(self, name, value):
        if name in self._f:
            del self._f[name]
        self._f.create_dataset(name, data=value)

    def create_dataset(self, name, *, shape, dtype, attrs=None):
        if name in self._f:
            ds = self._f[name]
            assert ds.shape == tuple(shape)
            assert ds.dtype == np.dtype(dtype)
        else:
            ds = self._f.create_dataset(name, shape=tuple(shape), dtype=dtype)
        for key, value in (attrs or {}).items():
            ds.attrs[key] = value

    def write_slab(self, name, arr, offset=None, **_kw):
        src = np.asarray(arr)
        ds = self._f[name]
        off = tuple(int(v) for v in (offset or (0,) * src.ndim))
        src_sel = []
        dst_sel = []
        for axis, size in enumerate(src.shape):
            n = min(int(size), int(ds.shape[axis]) - off[axis])
            src_sel.append(slice(0, n))
            dst_sel.append(slice(off[axis], off[axis] + n))
        ds[tuple(dst_sel)] = src[tuple(src_sel)]

    def read_slab(self, name, *, shape, dtype, offset=None, **_kw):
        src = np.asarray(self._f[name])
        out = np.zeros(tuple(int(v) for v in shape), dtype=dtype)
        off = tuple(int(v) for v in (offset or (0,) * len(shape)))
        src_sel = []
        dst_sel = []
        for axis, size in enumerate(out.shape):
            n = min(int(size), int(src.shape[axis]) - off[axis])
            src_sel.append(slice(off[axis], off[axis] + n))
            dst_sel.append(slice(0, n))
        out[tuple(dst_sel)] = src[tuple(src_sel)]
        return out


@pytest.fixture
def host_transport(monkeypatch):
    _HostSlabIO.opens = []
    monkeypatch.setattr(slab_io, "SlabIO", _HostSlabIO)
    monkeypatch.setattr(tagged_arrays.jax, "process_index", lambda: 0)
    monkeypatch.setattr(tagged_arrays.jax, "device_count", lambda: 1)
    monkeypatch.setattr(
        collectives, "device_put_process_local", lambda value, _sharding: value)
    return _HostSlabIO


def _bands(carrier):
    return SimpleNamespace(
        b0=0, b1=0, b2=46, b3=72, b4=int(carrier),
        b4_chi=int(carrier), b4_sigma=int(carrier), b4_logical=184)


@pytest.mark.parametrize("kind", ["mu", "band", "w0"])
def test_short_source_refuses_before_slabio_opens(
        tmp_path, monkeypatch, kind):
    """RED twins: neither a short μ nor a short band source mutates a file."""
    opens = []

    class _MustNotOpen:
        def __init__(self, *_a, **_k):
            opens.append(True)
            raise AssertionError("SlabIO opened before geometry preflight")

    monkeypatch.setattr(slab_io, "SlabIO", _MustNotOpen)
    path = str(tmp_path / "must_not_exist.h5")
    with pytest.raises(ValueError, match="SHORT"):
        if kind == "mu":
            tagged_arrays.write_restart_state_to_h5(
                path, n_rmu_logical=5,
                G0_mu_nu=np.zeros(4, dtype=np.complex128),
                mesh=object(), mode="w")
        elif kind == "band":
            tagged_arrays.write_restart_state_to_h5(
                path, n_rmu_logical=2,
                enk_full=np.zeros((1, 183), dtype=np.float64),
                band_slices=_bands(216), mesh=object(), mode="w")
        else:
            tagged_arrays.write_w0_qmunu_to_h5(
                path, np.zeros((1, 4, 4), dtype=np.complex128),
                n_rmu_logical=5, mesh=object())
    assert opens == []
    assert not (tmp_path / "must_not_exist.h5").exists()


def test_poisoned_mu_pad_is_clipped_and_both_shapes_are_receipted(
        tmp_path, host_transport):
    """A nonzero/NaN carrier tail is not promoted into logical storage."""
    path = str(tmp_path / "poison.h5")
    source = np.arange(8, dtype=np.complex128)
    source[5:] = np.nan + 9j
    tagged_arrays.write_restart_state_to_h5(
        path, n_rmu_logical=5, G0_mu_nu=source,
        mesh=object(), mode="w")

    with h5py.File(path, "r") as f:
        ds = f["G0_mu_nu"]
        np.testing.assert_array_equal(ds[:], source[:5])
        assert tuple(ds.attrs[tagged_arrays.RESTART_LOGICAL_SHAPE_ATTR]) == (5,)
        assert tuple(ds.attrs[tagged_arrays.RESTART_CARRIER_SHAPE_ATTR]) == (8,)


def test_charge_zeta_receipt_fresh_stamp_and_read(tmp_path, host_transport):
    path = str(tmp_path / "zeta_receipt.h5")
    receipt = {"scheme": "charge-zeta-v1", "digest": "abc123"}
    tagged_arrays.write_restart_state_to_h5(
        path, n_rmu_logical=2,
        V_qmunu=np.eye(2, dtype=np.complex128)[None],
        psi_full_y=np.ones((1, 1, 1, 2), dtype=np.complex128),
        charge_zeta_identity=receipt, mesh=object(), mode="w")
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))
    state = tagged_arrays.load_restart_state_from_h5(path, mesh)
    assert state.charge_zeta_identity == receipt


@pytest.mark.parametrize(
    "stored",
    [np.asarray([b"scheme"], dtype="S"),
     np.asarray([b"scheme", b""], dtype="S"),
     np.asarray([1, 2], dtype=np.int64)])
def test_malformed_charge_zeta_receipt_refuses_before_tensor_read(
        tmp_path, monkeypatch, stored):
    path = tmp_path / "bad_zeta_receipt.h5"
    with h5py.File(path, "w") as f:
        f["V_qmunu"] = np.eye(2, dtype=np.complex128)[None]
        f["psi_full_y"] = np.ones((1, 1, 1, 2), dtype=np.complex128)
        f[tagged_arrays.CHARGE_ZETA_IDENTITY_DATASET] = stored
    opens = []
    monkeypatch.setattr(
        slab_io, "SlabIO",
        lambda *_a, **_k: opens.append(True))
    with pytest.raises(ValueError, match="charge-zeta receipt"):
        tagged_arrays.read_restart_state_from_h5(str(path), object())
    assert opens == []


def _write_then_repad(tmp_path, host_transport, source_carrier, target_carrier):
    path = str(tmp_path / f"p{source_carrier}_to_p{target_carrier}.h5")
    logical = 184
    rng = np.random.default_rng(source_carrier * 1000 + target_carrier)
    psi = np.full((1, source_carrier, 1, 2),
                  complex(7.0e6, -3.0e6), dtype=np.complex128)
    psi[:, :logical] = (
        rng.standard_normal((1, logical, 1, 2))
        + 1j * rng.standard_normal((1, logical, 1, 2)))
    enk = np.full((1, source_carrier), 9.0e6, dtype=np.float64)
    enk[:, :logical] = np.linspace(-3.0, 5.0, logical)[None, :]
    V = np.eye(2, dtype=np.complex128)[None, :, :]

    tagged_arrays.write_restart_state_to_h5(
        path, n_rmu_logical=2, V_qmunu=V, psi_full_y=psi,
        enk_full=enk, band_slices=_bands(source_carrier),
        mesh=object(), mode="w")

    # The identity is the 184-band physical prefix; the source carrier is a
    # receipt, not part of the comparison.  A changed physical top still
    # refuses in the separate red twin below.
    tagged_arrays.assert_restart_window_matches(
        path, band_slices=_bands(target_carrier), n_rmu_logical=2)

    with h5py.File(path, "r") as f:
        assert tuple(f["band_window"][:]) == (0, 0, 46, 72, logical)
        assert tuple(f[tagged_arrays.BAND_WINDOW_CARRIER_DATASET][:]) == (
            0, 0, 46, 72, source_carrier)
        assert f["psi_full_y"].shape == (1, logical, 1, 2)
        assert f["enk_full"].shape == (1, logical)
        np.testing.assert_array_equal(f["psi_full_y"][:], psi[:, :logical])
        np.testing.assert_array_equal(f["enk_full"][:], enk[:, :logical])

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))
    state = tagged_arrays.read_restart_state_from_h5(
        path, mesh, n_band_carrier=target_carrier)
    psi_got, enk_got = state[2], state[3]
    assert psi_got.shape == (1, target_carrier, 1, 2)
    assert enk_got.shape == (1, target_carrier)
    np.testing.assert_array_equal(psi_got[:, :logical], psi[:, :logical])
    np.testing.assert_array_equal(enk_got[:, :logical], enk[:, :logical])
    assert not np.any(psi_got[:, logical:])
    np.testing.assert_array_equal(
        enk_got[:, logical:],
        np.full((1, target_carrier - logical), 6.0, dtype=np.float64))


def test_p36_to_p16_preserves_the_logical_prefix(tmp_path, host_transport):
    _write_then_repad(tmp_path, host_transport, 216, 192)


def test_p16_to_p36_preserves_the_logical_prefix(tmp_path, host_transport):
    _write_then_repad(tmp_path, host_transport, 192, 216)


def test_same_carrier_different_logical_band_identity_still_refuses(tmp_path):
    """RED twin: carrier agreement is never used to forgive changed physics."""
    path = tmp_path / "changed_physical_top.h5"
    with h5py.File(path, "w") as f:
        f["band_window_schema"] = np.int64(2)
        f["band_window"] = np.asarray((0, 0, 46, 72, 180), dtype=np.int64)
        f["band_window_carrier"] = np.asarray(
            (0, 0, 46, 72, 192), dtype=np.int64)
        f["band_window_split"] = np.asarray((180, 180), dtype=np.int64)
    with pytest.raises(ValueError, match="number_bands_chi"):
        tagged_arrays.assert_restart_window_matches(
            str(path), band_slices=_bands(192))


def test_torn_dataset_shape_receipt_refuses_before_slab_read(
        tmp_path, monkeypatch):
    path = tmp_path / "torn.h5"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("V_qmunu", shape=(1, 2, 2),
                              dtype=np.complex128)
        ds.attrs[tagged_arrays.RESTART_LOGICAL_SHAPE_ATTR] = (1, 3, 3)
        ds.attrs[tagged_arrays.RESTART_CARRIER_SHAPE_ATTR] = (1, 4, 4)
        f.create_dataset("psi_full_y", shape=(1, 1, 1, 2),
                         dtype=np.complex128)

    opens = []

    class _MustNotOpen:
        def __init__(self, *_a, **_k):
            opens.append(True)
            raise AssertionError("SlabIO opened before receipt validation")

    monkeypatch.setattr(slab_io, "SlabIO", _MustNotOpen)
    with pytest.raises(ValueError, match="stamped logical storage shape"):
        tagged_arrays.read_restart_state_from_h5(
            str(path), object(), n_band_carrier=1)
    assert opens == []
