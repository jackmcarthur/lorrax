"""Metadata seam tests; model validation/collectives have separate P4 coverage.

These cells exercise actual HDF5 restart metadata but substitute the model
validator and rank-0 transaction executor. No dense model payload is present.
"""
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from file_io import tagged_arrays
from file_io.commit_state import set_commit_state


@pytest.fixture
def member_case(tmp_path, monkeypatch):
    from file_io import shared_pole_store

    restart = tmp_path / "restart.h5"
    model = tmp_path / "shared_pole_SC0.h5"
    model.write_bytes(b"opaque model: membership must never write here")
    with h5py.File(restart, "w") as h5:
        set_commit_state(h5, True)
    identity = {"iteration_id": "SC0", "energies_sha256": "1" * 64}
    header = {"schema": "lorrax.shared-real-pole.v1", "digest": "2" * 64,
              "identity": deepcopy(identity)}
    calls = []
    mesh = object()

    def validate(path, *, expected_identity, mesh_xy):
        calls.append(Path(path))
        assert mesh_xy is mesh
        if not Path(path).exists():
            raise ValueError("model missing")
        if expected_identity != header["identity"]:
            raise ValueError("stale identity")
        return deepcopy(header)

    def transaction(path, *, stage, write, validate=None):
        if validate is not None:
            validate()
        write()

    monkeypatch.setattr(shared_pole_store, "validate_shared_pole_model", validate)
    monkeypatch.setattr(tagged_arrays, "rank0_transaction", transaction)
    return restart, model, identity, header, mesh, calls


def _register(case):
    restart, model, identity, _, mesh, _ = case
    return tagged_arrays.register_shared_pole_restart_member(
        restart, model, expected_identity=identity, mesh_xy=mesh)


def _read(case, identity=None):
    restart, _, current, _, mesh, _ = case
    return tagged_arrays.read_shared_pole_restart_member(
        restart, expected_identity=current if identity is None else identity,
        mesh_xy=mesh)


def test_member_roundtrip_is_relative_and_does_not_mutate_model(member_case):
    _, model, _, header, _, calls = member_case
    before = model.read_bytes()
    receipt = _register(member_case)
    assert receipt == {"path": model.name, "schema": header["schema"],
                       "digest": header["digest"], "iteration_id": "SC0"}
    assert _read(member_case) == receipt
    assert _register(member_case) == receipt
    assert calls == [model, model, model]
    assert model.read_bytes() == before


def test_member_missing_refuses(member_case):
    with pytest.raises(ValueError, match="no model member"):
        _read(member_case)


def test_member_changed_current_sc_identity_refuses(member_case):
    _register(member_case)
    changed = dict(member_case[2], energies_sha256="3" * 64)
    with pytest.raises(ValueError, match="stale identity"):
        _read(member_case, changed)


@pytest.mark.parametrize("key,value", [
    ("digest", "3" * 64), ("schema", "lorrax.shared-real-pole.v2"),
    ("iteration_id", "SC1"),
])
def test_member_linked_identity_change_refuses(member_case, key, value):
    _register(member_case)
    header = member_case[3]
    if key == "iteration_id":
        header["identity"][key] = value
        member_case[2][key] = value
    else:
        header[key] = value
    with pytest.raises(ValueError, match=f"linked {key} changed"):
        _read(member_case)


def test_member_replacement_refuses_without_changing_original(member_case):
    original = _register(member_case)
    member_case[3]["digest"] = "4" * 64
    with pytest.raises(ValueError, match="immutable member replacement"):
        _register(member_case)
    member_case[3]["digest"] = original["digest"]
    assert _read(member_case) == original


def test_member_partial_restart_refuses(member_case):
    _register(member_case)
    with h5py.File(member_case[0], "r+") as h5:
        set_commit_state(h5, False)
    with pytest.raises(ValueError, match="not globally committed"):
        _read(member_case)
    with pytest.raises(ValueError, match="not globally committed"):
        _register(member_case)


def test_member_missing_model_refuses(member_case):
    _register(member_case)
    member_case[1].unlink()
    with pytest.raises(ValueError, match="model missing"):
        _read(member_case)


@pytest.mark.parametrize("values,match", [
    (["model.h5", "schema"], "four UTF-8"),
    (["/model.h5", "schema", "a" * 64, "SC0"], "relative"),
    (["model.h5", "schema", "not-a-digest", "SC0"], "SHA256"),
    (["model.h5", "schema", "a" * 64, ""], "empty or invalid"),
])
def test_malformed_member_refuses_before_model_validation(member_case, values, match):
    with h5py.File(member_case[0], "r+") as h5:
        h5.create_dataset(tagged_arrays.SHARED_POLE_MEMBER_DATASET,
                          data=np.asarray(values, dtype="S"))
    with pytest.raises(ValueError, match=match):
        _read(member_case)
    assert not member_case[5]
