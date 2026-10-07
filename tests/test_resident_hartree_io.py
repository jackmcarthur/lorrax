"""Resident native J has its own band domain and immutable source provenance."""
if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    RUNTIME = initialize_communicator_stack()

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import barrier, device_put_process_local
from file_io.commit_state import COMMIT_STATE
from file_io.restart_bundle import (
    read_resident_hartree_metadata, read_resident_hartree_from_h5,
    load_restart_state_from_h5,
)
from file_io.tagged_arrays import (
    HARTREE_PARENT_DATASET, HARTREE_PROVENANCE_DATASET,
    normalize_resident_hartree_provenance, write_restart_state_to_h5,
    _hartree_band_axis, _loaded_band_axis,
)


def _identity(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _record():
    source = {"schema": "actual-occupied-source-test", "wfn": "a" * 64,
              "frame": "b" * 64, "occupations": [1., 1., 0.]}
    operator = {"schema": "represented-periodic-source-test",
                "source_identity": _identity(source), "fft_grid": [4, 4, 4],
                "G0": "zero", "neutral_mean": "both"}
    return dict(schema="lorrax.resident_charge_hartree.v1",
        source_binding=source, source_identity=_identity(source),
        operator_binding=operator, operator_identity=_identity(operator),
        band_range=[2, 5], parent_full_rows=[0, 3],
        parent_k_frac=[[0., 0., 0.], [0., 0., .5]],
        units="Ry", k_domain="file_wedge", trs_rule="conj")


def _mesh():
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


def _matrix():
    a = (np.arange(18).reshape(2, 3, 3)
         + 1j * np.arange(18, 36).reshape(2, 3, 3)).astype(np.complex128)
    return a + a.conj().swapaxes(-2, -1)


def _write(path, mesh=None):
    mesh = _mesh() if mesh is None else mesh
    axis = _hartree_band_axis(3, mesh)
    data = np.full((2, axis.carrier, axis.carrier), np.nan + 1j * np.nan)
    data[:, :3, :3] = _matrix()
    array = device_put_process_local(data, NamedSharding(mesh, P(None, "x", "y")))
    write_restart_state_to_h5(str(path), n_rmu_logical=2, mesh=mesh,
        hartree_parent_kij_ry=array, hartree_provenance=_record())
    return data


def test_real_roundtrip_has_independent_bands_and_fixed_provenance(tmp_path):
    path = tmp_path / "J.h5"
    _write(path)
    record = read_resident_hartree_metadata(path, expected={"source_identity": _identity(_record()["source_binding"])})
    assert set(record) == set(_record()) | {"payload_sha256"}
    result = read_resident_hartree_from_h5(path, _mesh(), required=True)
    np.testing.assert_array_equal(result["parent_kij_ry"], _matrix())
    assert result["provenance"] == record
    with h5py.File(path) as stream:
        assert stream[HARTREE_PARENT_DATASET].shape == (2, 3, 3)
        assert stream[HARTREE_PARENT_DATASET].dtype == np.dtype("complex128")


def test_none_default_keeps_member_absent(tmp_path):
    path = tmp_path / "default.h5"
    write_restart_state_to_h5(str(path), n_rmu_logical=2, mesh=_mesh())
    assert read_resident_hartree_metadata(path) is None
    assert read_resident_hartree_from_h5(path, _mesh()) is None
    with pytest.raises(ValueError, match="missing"):
        read_resident_hartree_metadata(path, required=True)
    with h5py.File(path) as stream:
        assert HARTREE_PARENT_DATASET not in stream
        assert HARTREE_PROVENANCE_DATASET not in stream


@pytest.mark.parametrize("case", ["source", "operator", "units", "domain", "trs", "range",
    "duplicate_rows", "coordinates", "unknown", "missing", "nonfinite"])
def test_bad_provenance_refuses_before_destructive_write(tmp_path, case):
    record = _record()
    if case in ("source", "operator"):
        record[case + "_identity"] = "0" * 64
    elif case == "units":
        record["units"] = "Ha"
    elif case == "domain":
        record["k_domain"] = "star_wedge"
    elif case == "trs":
        record["trs_rule"] = "none"
    elif case == "range":
        record["band_range"] = [2, 2]
    elif case == "duplicate_rows":
        record["parent_full_rows"] = [0, 0]
    elif case == "coordinates":
        record["parent_k_frac"] = [[0., 0., 0.]]
    elif case == "unknown":
        record["extra"] = 1
    elif case == "missing":
        del record["operator_binding"]
    else:
        record["source_binding"]["x"] = float("nan")
    path = tmp_path / "keep.h5"
    path.write_bytes(b"previous immutable artifact")
    with pytest.raises(ValueError, match="resident Hartree"):
        write_restart_state_to_h5(str(path), n_rmu_logical=2, mesh=_mesh(),
            hartree_parent_kij_ry=_matrix(), hartree_provenance=record)
    assert path.read_bytes() == b"previous immutable artifact"


@pytest.mark.parametrize("case", ["missing_payload", "missing_metadata", "append", "shape", "dtype", "nonfinite", "rows"])
def test_bad_producer_pair_refuses_before_write(tmp_path, case):
    path = tmp_path / "keep.h5"
    path.write_bytes(b"previous artifact")
    data, record, kwargs = _matrix(), _record(), {}
    if case == "missing_payload":
        data = None
    elif case == "missing_metadata":
        record = None
    elif case == "append":
        kwargs["mode"] = "a"
    elif case == "shape":
        data = data[:, :2, :2]
    elif case == "dtype":
        data = data.astype(np.complex64)
    elif case == "nonfinite":
        data[0, 0, 0] = np.nan
    else:
        kwargs["parent_k_rows"] = [0, 4]
    with pytest.raises(ValueError):
        write_restart_state_to_h5(str(path), n_rmu_logical=2, mesh=_mesh(),
            hartree_parent_kij_ry=data, hartree_provenance=record, **kwargs)
    assert path.read_bytes() == b"previous artifact"


@pytest.mark.parametrize("case", ["torn", "bytes", "metadata_shape", "metadata_vlen", "schema", "rows", "uncommitted"])
def test_actual_hdf5_corruption_refuses_before_unrelated_payload(tmp_path, monkeypatch, case):
    path = tmp_path / "corrupt.h5"
    _write(path)
    with h5py.File(path, "a") as stream:
        stream.create_dataset("unrelated_huge", shape=(100000, 100000),
                              chunks=(4, 4), dtype="complex128")
        if case == "torn":
            del stream[HARTREE_PROVENANCE_DATASET]
        elif case == "bytes":
            stream[HARTREE_PARENT_DATASET][0, 0, 0] += 1
        elif case == "metadata_shape":
            data = stream[HARTREE_PROVENANCE_DATASET][()]
            del stream[HARTREE_PROVENANCE_DATASET]
            stream.create_dataset(HARTREE_PROVENANCE_DATASET, data=[data])
        elif case == "metadata_vlen":
            data = bytes(stream[HARTREE_PROVENANCE_DATASET][()]).decode()
            del stream[HARTREE_PROVENANCE_DATASET]
            stream.create_dataset(HARTREE_PROVENANCE_DATASET, data=data, dtype=h5py.string_dtype())
        elif case == "schema":
            data = json.loads(bytes(stream[HARTREE_PROVENANCE_DATASET][()]))
            data["schema"] = "unknown"
            del stream[HARTREE_PROVENANCE_DATASET]
            stream.create_dataset(HARTREE_PROVENANCE_DATASET, data=np.bytes_(json.dumps(data).encode()))
        elif case == "rows":
            stream.create_dataset("psi_parent_k_rows", data=[0, 4])
        else:
            stream[COMMIT_STATE][0] = 0
    original = h5py.Dataset.__getitem__
    def bounded(dataset, index):
        assert dataset.name != "/unrelated_huge"
        return original(dataset, index)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded)
    with pytest.raises(ValueError):
        read_resident_hartree_metadata(path)


def test_changed_expected_binding_refuses_before_j_payload(tmp_path, monkeypatch):
    path = tmp_path / "J.h5"
    _write(path)
    original = h5py.Dataset.__getitem__
    def metadata_only(dataset, index):
        assert dataset.name != "/" + HARTREE_PARENT_DATASET
        return original(dataset, index)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", metadata_only)
    for expected in ({"source_identity": "wrong"}, {"operator_identity": "wrong"},
                     {"band_range": [2, 6]}, {"parent_full_rows": [0, 4]},
                     {"parent_k_frac": [[0, 0, 0], [.5, 0, 0]]}, {"unknown": 1}):
        with pytest.raises(ValueError):
            read_resident_hartree_metadata(path, expected=expected)


def test_later_parent_append_cross_checks_file_rows_before_mutation(tmp_path):
    path = tmp_path / "J.h5"
    _write(path)
    before = path.read_bytes()
    psi = np.zeros((2, 7, 4, 2), np.complex128)
    mun = np.zeros((2, 4, 2, 7), np.complex128)
    with pytest.raises(ValueError, match="FILE rows"):
        write_restart_state_to_h5(str(path), mode="a", n_rmu_logical=2,
            mesh=_mesh(), psi_parent_y=psi, psi_parent_y_mun=mun,
            parent_k_rows=[0, 4])
    assert path.read_bytes() == before
    write_restart_state_to_h5(str(path), mode="a", n_rmu_logical=2,
        mesh=_mesh(), psi_parent_y=psi, psi_parent_y_mun=mun,
        parent_k_rows=[0, 3])
    result = read_resident_hartree_from_h5(path, _mesh())
    np.testing.assert_array_equal(result["parent_kij_ry"], _matrix())


def run_p4_roundtrip(path):
    """Real MPI/FFI write/read: J3 versus psi7, source bands start at2."""
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    if jax.process_count() != 4 or mesh.size != 4:
        raise ValueError("resident Hartree real-P4 proof requires four MPI/GPU ranks")
    path = Path(path)
    if jax.process_index() == 0:
        if path.exists():
            raise ValueError("proof output already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
    barrier("resident_hartree_proof_output")
    _write(path, mesh)
    actual = read_resident_hartree_from_h5(path, mesh, required=True)
    # Compare owned shards to the literal logical payload, including read zeros.
    axis = _hartree_band_axis(3, mesh)
    expected = np.zeros((2, axis.carrier, axis.carrier), np.complex128)
    expected[:, :3, :3] = _matrix()
    for shard in actual["parent_kij_ry"].addressable_shards:
        np.testing.assert_array_equal(shard.data, expected[shard.index])
    bands = SimpleNamespace(b0=2, b1=2, b2=4, b3=5, b4=10,
                            b4_logical=9, b3_logical=5, b4_chi=10,
                            b4_chi_logical=9, b4_sigma=5, b4_sigma_logical=5)
    loaded_axis = _loaded_band_axis(7, mesh)
    from runtime.padding import padded_mu_axis, mesh_divisor
    mu_carrier = padded_mu_axis(2, mesh_divisor(mesh)).carrier
    # Persist the unrelated loaded-band carrier in a fresh cohesive bundle.
    psi = np.zeros((2, loaded_axis.carrier, 4, mu_carrier), np.complex128)
    mun = np.zeros((2, 4, mu_carrier, loaded_axis.carrier), np.complex128)
    enk = np.zeros((4, loaded_axis.carrier))
    jaxis = _hartree_band_axis(3, mesh)
    carrier = np.full((2, jaxis.carrier, jaxis.carrier), np.nan + 1j*np.nan)
    carrier[:, :3, :3] = _matrix()
    complete_path = str(path) + ".complete.h5"
    write_restart_state_to_h5(
        complete_path, n_rmu_logical=2, mesh=mesh, band_slices=bands,
        V_qmunu=device_put_process_local(np.zeros((4,mu_carrier,mu_carrier), np.complex128),
            NamedSharding(mesh, P(None,"x","y"))),
        psi_parent_y=device_put_process_local(psi, NamedSharding(mesh,P(None,"x",None,"y"))),
        psi_parent_y_mun=device_put_process_local(mun, NamedSharding(mesh,P(None,None,"y","x"))),
        parent_k_rows=[0,3], enk_full=enk, kgrid=(1,2,2),
        hartree_parent_kij_ry=device_put_process_local(carrier, NamedSharding(mesh,P(None,"x","y"))),
        hartree_provenance=_record())
    restored = load_restart_state_from_h5(complete_path, mesh, band_slices=bands,
                                           n_rmu_logical=2)
    for shard in restored.resident_hartree["parent_kij_ry"].addressable_shards:
        np.testing.assert_array_equal(shard.data, expected[shard.index])
    for expected_binding in ({"source_identity":"wrong"}, {"trs_rule":"none"},
                             {"band_range":[2,6]}, {"parent_full_rows":[0,4]}):
        try:
            read_resident_hartree_metadata(complete_path, expected=expected_binding)
        except ValueError:
            pass
        else:
            raise AssertionError("changed resident binding was admitted")
    if jax.process_index() == 0:
        with h5py.File(complete_path,"a") as stream:
            stream[HARTREE_PARENT_DATASET][0,0,0] += 1
    barrier("resident_hartree_corruption_written")
    try:
        read_resident_hartree_metadata(complete_path)
    except ValueError:
        pass
    else:
        raise AssertionError("modified payload was admitted")
    if jax.process_index() == 0:
        print(json.dumps(dict(status="PASS", ranks=4, physical_J_shape=[2,3,3],
            producer_J_carrier=axis.carrier, loaded_psi_logical=7,
            loaded_psi_carrier=loaded_axis.carrier, band_origin=2,
            pad_poison_inert=True, checksum_refusal=True)), flush=True)


if __name__ == "__main__":
    import sys
    from runtime import run_main_and_finalize
    run_main_and_finalize(lambda: run_p4_roundtrip(sys.argv[1]))
