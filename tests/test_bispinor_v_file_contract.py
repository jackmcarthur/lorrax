"""Hostile file-contract tests for the streamed bispinor-V artifact."""

import json

import h5py
import numpy as np
import pytest

from gw.v_q_bispinor import (
    BispinorVqReader,
    UNIQUE_TILES,
    V_QMUNU_DATA_READY_DATASET,
    V_QMUNU_INVENTORY_DATASET,
    _require_tile_carrier,
    _inventory_json,
    _publish_unique_tile_inventory,
    tile_dataset_name,
)


KGRID = (2, 1, 1)
N_Q = 2
N_C = 3
N_T = 2


def _logical_shape(mu_L, nu_L):
    n_L = N_C if mu_L == 0 else N_T
    n_R = N_C if nu_L == 0 else N_T
    return (N_Q, n_L, n_R)


def _write_tile_stage(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("kgrid", data=np.asarray(KGRID, dtype=np.int64))
        f.create_dataset("n_rmu_C", data=np.int64(N_C))
        f.create_dataset("n_rmu_T", data=np.int64(N_T))
        f.create_dataset("n_q_total", data=np.int64(N_Q))
        for mu_L, nu_L in UNIQUE_TILES:
            f.create_dataset(
                tile_dataset_name(mu_L, nu_L),
                shape=_logical_shape(mu_L, nu_L),
                dtype=np.complex128,
            )

    from symmetry_maps import QirrTables, stamp_qirr_tensor, verify_centroid_orbit_closure
    for a, b in UNIQUE_TILES:
        nmu = N_C if a == 0 else N_T
        points = np.zeros((nmu, 3))
        points[:, 0] = np.arange(nmu) / nmu
        verdict = verify_centroid_orbit_closure(
            points, np.eye(3, dtype=int)[None], tau=np.zeros((1, 3)))
        tables = QirrTables(np.arange(2), np.zeros(2, dtype=int),
            np.asarray([[0., 0., 0.], [.5, 0., 0.]]),
            np.tile(np.arange(nmu), (2, 1)), np.zeros((2, nmu, 3), dtype=int), 1)
        stamp_qirr_tensor(path, tile_dataset_name(a, b), tables=tables,
                           closure_verdict=verdict, n_rmu_logical=nmu)


def _publish_tile_stage(path):
    with h5py.File(path, "a") as f:
        _publish_unique_tile_inventory(
            f, filename=path, n_q_total=N_Q,
            n_rmu_C=N_C, n_rmu_T=N_T)


def _write_complete_file(path):
    _write_tile_stage(path)
    _publish_tile_stage(path)


def _refusing_collective(monkeypatch):
    opened = []

    class RefuseCollectiveOpen:
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))
            raise AssertionError("collective open preceded file validation")

    monkeypatch.setattr("file_io.slab_io.SlabIO", RefuseCollectiveOpen)
    return opened


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        ((N_Q - 1, N_C, N_C), np.complex128, "cannot supply"),
        ((N_Q, N_C - 1, N_C), np.complex128, "cannot supply"),
        ((N_Q, N_C, N_C), np.complex64, "silent precision cast"),
    ],
)
def test_writer_refuses_invalid_tile_carrier(shape, dtype, message):
    carrier = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match=message):
        _require_tile_carrier(
            carrier, name="V_qmunu_CC",
            logical_shape=(N_Q, N_C, N_C))


def test_writer_accepts_exact_q_and_padded_centroid_carrier():
    carrier = np.zeros((N_Q, N_C + 3, N_C + 5), dtype=np.complex128)
    _require_tile_carrier(
        carrier, name="V_qmunu_CC", logical_shape=(N_Q, N_C, N_C))


def test_reader_refuses_missing_unique_tile_before_collective_open(
        tmp_path, monkeypatch):
    path = tmp_path / "missing_tile.h5"
    _write_tile_stage(path)
    with h5py.File(path, "a") as f:
        del f[tile_dataset_name(2, 3)]
    with pytest.raises(ValueError, match="missing=.*V_qmunu_TT_23"):
        _publish_tile_stage(path)
    with h5py.File(path, "r") as f:
        assert not bool(f[V_QMUNU_DATA_READY_DATASET][()])

    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        del f[tile_dataset_name(2, 3)]
    opened = _refusing_collective(monkeypatch)

    with pytest.raises(ValueError, match="missing=.*V_qmunu_TT_23"):
        BispinorVqReader(path, object())
    assert not opened


def test_reader_refuses_short_tile_before_collective_open(tmp_path,
                                                          monkeypatch):
    path = tmp_path / "short_tile.h5"
    _write_tile_stage(path)
    name = tile_dataset_name(1, 2)
    with h5py.File(path, "a") as f:
        del f[name]
        f.create_dataset(name, shape=(N_Q, N_T - 1, N_T),
                         dtype=np.complex128)
    with pytest.raises(ValueError, match="expected exact logical shape"):
        _publish_tile_stage(path)
    with h5py.File(path, "r") as f:
        assert not bool(f[V_QMUNU_DATA_READY_DATASET][()])

    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        del f[name]
        f.create_dataset(name, shape=(N_Q, N_T - 1, N_T),
                         dtype=np.complex128)
    opened = _refusing_collective(monkeypatch)

    with pytest.raises(ValueError, match="expected exact logical shape"):
        BispinorVqReader(path, object())
    assert not opened


def test_reader_refuses_wrong_tile_dtype_before_collective_open(
        tmp_path, monkeypatch):
    path = tmp_path / "wrong_dtype.h5"
    _write_complete_file(path)
    name = tile_dataset_name(3, 3)
    with h5py.File(path, "a") as f:
        shape = f[name].shape
        del f[name]
        f.create_dataset(name, shape=shape, dtype=np.complex64)
    opened = _refusing_collective(monkeypatch)

    with pytest.raises(ValueError, match="complex64, expected complex128"):
        BispinorVqReader(path, object())
    assert not opened


def test_reader_refuses_unready_or_noncanonical_inventory(
        tmp_path, monkeypatch):
    path = tmp_path / "unready.h5"
    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        f[V_QMUNU_DATA_READY_DATASET][()] = np.bool_(False)
    opened = _refusing_collective(monkeypatch)
    with pytest.raises(ValueError, match="data_ready"):
        BispinorVqReader(path, object())
    assert not opened

    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        inventory = json.loads(f[V_QMUNU_INVENTORY_DATASET][()].decode())
        inventory["tiles"] = inventory["tiles"][:-1]
        del f[V_QMUNU_INVENTORY_DATASET]
        f.create_dataset(
            V_QMUNU_INVENTORY_DATASET,
            data=np.bytes_(json.dumps(inventory)),
        )
    with pytest.raises(ValueError, match="canonical inventory"):
        BispinorVqReader(path, object())
    assert not opened


def test_reader_accepts_complete_certified_file(tmp_path, monkeypatch):
    path = tmp_path / "complete.h5"
    _write_complete_file(path)
    with h5py.File(path, "r") as f:
        assert bool(f[V_QMUNU_DATA_READY_DATASET][()])
        assert f[V_QMUNU_INVENTORY_DATASET][()].decode() == _inventory_json(
            n_q_total=N_Q, n_rmu_C=N_C, n_rmu_T=N_T)
    opened = []

    class AcceptedCollectiveOpen:
        def __init__(self, filename, *, mode, mesh):
            opened.append((filename, mode, mesh))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mesh = object()
    monkeypatch.setattr("file_io.slab_io.SlabIO", AcceptedCollectiveOpen)
    with BispinorVqReader(path, mesh) as reader:
        assert reader.n_q_total == N_Q
        assert reader.n_rmu_C == N_C
        assert reader.n_rmu_T == N_T

    assert opened == [(path, "r", mesh)]


def test_reader_refuses_unstamped_full_q_file(tmp_path, monkeypatch):
    path = tmp_path / "legacy_full_q.h5"
    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        f[tile_dataset_name(0, 0)].attrs.clear()
    opened = _refusing_collective(monkeypatch)
    with pytest.raises(ValueError, match="rerun with restart=false"):
        BispinorVqReader(path, object())
    assert not opened


def test_reader_refuses_disagreeing_current_family_tables(tmp_path, monkeypatch):
    from symmetry_maps import QIRR_TABLE_SUFFIX
    path = tmp_path / "torn_current_family.h5"
    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        f[tile_dataset_name(2, 3) + QIRR_TABLE_SUFFIX]["L_table"][0, 0, 0] = 1
    opened = _refusing_collective(monkeypatch)
    with pytest.raises(ValueError, match="torn V tiles"):
        BispinorVqReader(path, object())
    assert not opened


@pytest.mark.parametrize("attribute,value", [
    ("qirr_format_version", 999), ("qirr_table_hash", "corrupted"),
    ("qirr_data_ready", False),
])
def test_reader_authenticates_each_tile_receipt(tmp_path, monkeypatch, attribute, value):
    path = tmp_path / "corrupt_receipt.h5"
    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        f[tile_dataset_name(1, 2)].attrs[attribute] = value
    opened = _refusing_collective(monkeypatch)
    with pytest.raises(ValueError, match="torn V tiles"):
        BispinorVqReader(path, object())
    assert not opened


def test_reader_refuses_sibling_centroid_stamp_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "mixed_centroid_sets.h5"
    _write_complete_file(path)
    with h5py.File(path, "a") as f:
        f[tile_dataset_name(1, 2)].attrs["qirr_centroid_hash"] = "f:" + "0" * 64
    opened = _refusing_collective(monkeypatch)
    with pytest.raises(ValueError, match="torn V tiles"):
        BispinorVqReader(path, object())
    assert not opened
