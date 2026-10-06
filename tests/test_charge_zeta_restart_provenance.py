"""Bounded charge-fit records authenticate before any restart tensor read."""
import json

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from file_io.commit_state import COMMIT_STATE
from file_io.restart_bundle import read_charge_zeta_provenance
from file_io.tagged_arrays import (
    CHARGE_ZETA_IDENTITY_DATASET, CHARGE_ZETA_PROVENANCE_DATASET,
    CHARGE_ZETA_PROVENANCE_MAX_BYTES, write_restart_state_to_h5,
)


IDENTITY = {"scheme": "charge-zeta-v1:canonical-provenance+bound-wfn",
            "digest": "a" * 64}
PROVENANCE = json.dumps({"schema": 2, "path": "atomic/α", "charge_fit_endpoint_weights": {
    "schema": "occupied_band_endpoints_v1", "occupied_stop": 36,
    "occupied_weight": 4., "empty_weight": 1.}}, ensure_ascii=False)


def _mesh():
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


def _records(path, *, identity=True, provenance=PROVENANCE):
    with h5py.File(path, "w") as stream:
        if identity:
            stream.create_dataset(CHARGE_ZETA_IDENTITY_DATASET,
                                  data=np.asarray(tuple(IDENTITY.values()), dtype="S"))
        if provenance is not None:
            stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET,
                                  data=np.bytes_(provenance.encode("utf-8")))


def test_actual_writer_roundtrip_keeps_exact_json_and_never_reads_large_payload(tmp_path, monkeypatch):
    path = tmp_path / "restart.h5"
    write_restart_state_to_h5(
        str(path), n_rmu_logical=2, charge_zeta_identity=IDENTITY,
        charge_zeta_provenance=PROVENANCE, mesh=_mesh())
    with h5py.File(path, "a") as stream:
        # Huge unallocated logical datasets make a mistaken tensor read costly.
        for name in ("V_qmunu", "psi_parent_y", "psi_parent_y_mun"):
            stream.create_dataset(name, shape=(16, 100000, 100000),
                                  chunks=(1, 10, 10), dtype="complex128")
    reads = []
    original = h5py.Dataset.__getitem__
    def observe(dataset, item):
        reads.append(dataset.name)
        assert dataset.name not in ("/V_qmunu", "/psi_parent_y", "/psi_parent_y_mun")
        return original(dataset, item)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe)
    assert read_charge_zeta_provenance(path) == {
        "charge_zeta_provenance": PROVENANCE, "charge_zeta_identity": IDENTITY}
    assert set(reads) <= {"/" + name for name in
        (COMMIT_STATE, CHARGE_ZETA_PROVENANCE_DATASET, CHARGE_ZETA_IDENTITY_DATASET)}


def test_default_writer_keeps_new_record_absent_and_legacy_states_explicit(tmp_path):
    path = tmp_path / "default.h5"
    write_restart_state_to_h5(str(path), n_rmu_logical=2, mesh=_mesh())
    assert read_charge_zeta_provenance(path) == {
        "charge_zeta_provenance": None, "charge_zeta_identity": None}
    with h5py.File(path, "r") as stream:
        assert CHARGE_ZETA_PROVENANCE_DATASET not in stream
        assert CHARGE_ZETA_IDENTITY_DATASET not in stream
    _records(path, provenance=None)
    assert read_charge_zeta_provenance(path) == {
        "charge_zeta_provenance": None, "charge_zeta_identity": IDENTITY}


@pytest.mark.parametrize("provenance", ["", "[]", "{", '{"x":NaN}',
    " " * (CHARGE_ZETA_PROVENANCE_MAX_BYTES + 1), {"schema": 2}])
def test_invalid_json_refuses_before_replacing_the_existing_file(tmp_path, provenance):
    path = tmp_path / "keep.h5"
    before = b"existing artifact must not be truncated"
    path.write_bytes(before)
    with pytest.raises(ValueError, match="charge_zeta_provenance"):
        write_restart_state_to_h5(
            str(path), n_rmu_logical=2, charge_zeta_identity=IDENTITY,
            charge_zeta_provenance=provenance)
    assert path.read_bytes() == before


def test_pairing_and_append_are_immutable_before_any_write(tmp_path):
    path = tmp_path / "keep.h5"
    path.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="requires charge_zeta_identity"):
        write_restart_state_to_h5(str(path), n_rmu_logical=2,
                                 charge_zeta_provenance=PROVENANCE)
    with pytest.raises(ValueError, match="immutable restart provenance"):
        write_restart_state_to_h5(
            str(path), mode="a", n_rmu_logical=2,
            charge_zeta_identity=IDENTITY, charge_zeta_provenance=PROVENANCE)
    assert path.read_bytes() == b"unchanged"
    _records(path, identity=False)
    with pytest.raises(ValueError, match="requires charge_zeta_identity"):
        read_charge_zeta_provenance(path)


@pytest.mark.parametrize("kind", ["shape", "vlen", "oversized", "utf8", "json", "identity", "uncommitted"])
def test_malformed_real_hdf5_metadata_refuses_before_any_payload(tmp_path, kind):
    path = tmp_path / "corrupt.h5"
    _records(path)
    with h5py.File(path, "a") as stream:
        if kind == "uncommitted":
            stream.create_dataset(COMMIT_STATE, data=[0])
        elif kind == "identity":
            del stream[CHARGE_ZETA_IDENTITY_DATASET]
            stream.create_dataset(CHARGE_ZETA_IDENTITY_DATASET, data=[1., 2.])
        else:
            del stream[CHARGE_ZETA_PROVENANCE_DATASET]
            if kind == "shape":
                stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET, data=np.asarray([b"{}"] ))
            elif kind == "vlen":
                stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET, data="{}",
                                      dtype=h5py.string_dtype())
            elif kind == "oversized":
                stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET, shape=(),
                    dtype=f"S{CHARGE_ZETA_PROVENANCE_MAX_BYTES + 1}")
            elif kind == "utf8":
                stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET, data=np.bytes_(b"\xff"))
            else:
                stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET, data=np.bytes_(b"[]"))
    with pytest.raises(ValueError, match="charge_zeta|bounded|committed"):
        read_charge_zeta_provenance(path)
