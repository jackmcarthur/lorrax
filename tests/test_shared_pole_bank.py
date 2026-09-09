"""Construction scratch resume contract on a real 2x2 P4 mesh.

The lane runner invokes these functions collectively with one shared tmp_path.
All scientific arrays are tiny fixtures; payload comparisons reduce on devices.
"""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io.shared_pole_store import (
    initialize_shared_pole_bank, read_shared_pole_bank,
    validate_shared_pole_bank, write_shared_pole_bank,
)
from file_io.slab_io import SlabIO
from common.collectives import rank0_transaction
from test_shared_pole_store import _fixture


def _bank_fixture():
    if len(jax.devices()) != 4:
        pytest.skip("scratch store contract requires exactly P4")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    meta, tables, recipe, identity = _fixture(mesh)
    recipe = deepcopy(recipe)
    # Fixture-local named codes test transport, not a claimed resolver ABI.
    recipe.update(
        role_codes={"line": 1, "imaginary": 2, "infinity": 3,
                    "held_line": 4, "held_imaginary": 5},
        z_ry=np.asarray([0.1j, 0.1j, 0.2+0.1j], dtype=np.complex128),
        role=np.asarray([1, 2, 4], dtype=np.int8),
        distinct_id=np.asarray([0, 0, 1], dtype=np.int64),
        held=np.asarray([False, False, True], dtype=np.bool_))
    return mesh, meta, tables, recipe, identity


def _matrix(meta, mesh, *, samples, value):
    basis = meta.mu_basis
    shape = ((1, 1) if samples else (1,)) + (basis.n_canonical,) * 2
    # Non-Hermitian complex data catches an accidental off-axis Hermitization.
    fixture = (value + np.arange(np.prod(shape)).reshape(shape)
               + 1j * np.arange(np.prod(shape)).reshape(shape)[..., ::-1]).astype(np.complex128)
    fixture[..., basis.n_logical:, :] = 0
    fixture[..., :, basis.n_logical:] = 0
    spec = P(None, None, 'x', 'y') if samples else P(None, 'x', 'y')
    array = jax.make_array_from_callback(
        shape, NamedSharding(mesh, spec), lambda index: fixture[index])
    return basis.pack_operator(array, spec=spec)


def test_bank_partial_resume_and_immutable_completion(tmp_path):
    mesh, meta, tables, recipe, identity = _bank_fixture()
    path = tmp_path / "scratch_partial.h5"
    header = initialize_shared_pole_bank(
        path, meta=meta, tables=tables, recipe=recipe, identity=identity, mesh_xy=mesh)
    assert not header["complete"]
    assert header["bank_shape"]["nsample"] == 2
    assert len(header["bank_sample_plan"]["role"]) == 3
    W = _matrix(meta, mesh, samples=True, value=3)
    D = _matrix(meta, mesh, samples=True, value=-5)
    M = _matrix(meta, mesh, samples=False, value=11)
    header = write_shared_pole_bank(
        path, q_span=(0, 1), sample_span=(0, 1), Wc=W,
        meta=meta, expected_identity=identity, mesh_xy=mesh)
    with pytest.raises(ValueError, match="incomplete"):
        validate_shared_pole_bank(
            path, expected_identity=identity, mesh_xy=mesh, require_complete=True)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        with pytest.raises(ValueError, match="incomplete"):
            read_shared_pole_bank(io, (0, 1), meta=meta, header=header,
                                  sample_span=(0, 1))
        actual = read_shared_pole_bank(
            io, (0, 1), meta=meta, header=header, sample_span=(0, 1), fields=("Wc",))
        assert bool(jnp.all(actual["Wc"] == W))
    with pytest.raises(ValueError, match="already committed"):
        write_shared_pole_bank(
            path, q_span=(0, 1), sample_span=(0, 1), Wc=W,
            meta=meta, expected_identity=identity, mesh_xy=mesh)
    # Resume only missing fields; the committed first value stays unchanged.
    nq = header["bank_shape"]["nq"]
    for q in range(nq):
        for sample in range(2):
            write_shared_pole_bank(
                path, q_span=(q, q+1), sample_span=(sample, sample+1),
                Wc=None if (q, sample) == (0, 0) else W, dWc_ds=D,
                meta=meta, expected_identity=identity, mesh_xy=mesh)
        header = write_shared_pole_bank(
            path, q_span=(q, q+1), M1=M, M3=2*M,
            meta=meta, expected_identity=identity, mesh_xy=mesh)
    header = validate_shared_pole_bank(
        path, expected_identity=identity, mesh_xy=mesh, require_complete=True)
    assert header["complete"] and header["final_commit"]
    with SlabIO(path, mode="r", mesh=mesh) as io:
        actual = read_shared_pole_bank(
            io, (0, 1), meta=meta, header=header, sample_span=(0, 1),
            fields=("Wc", "dWc_ds", "M1", "M3"))
        for key, expected in (("Wc", W), ("dWc_ds", D), ("M1", M), ("M3", 2*M)):
            assert bool(jnp.all(actual[key] == expected))
            assert actual[key].sharding == expected.sharding
    with pytest.raises(ValueError, match="immutable"):
        write_shared_pole_bank(
            path, q_span=(0, 1), M1=M, meta=meta,
            expected_identity=identity, mesh_xy=mesh)


def test_bank_stale_identity_and_invalid_spans(tmp_path):
    mesh, meta, tables, recipe, identity = _bank_fixture()
    path = tmp_path / "scratch_stale.h5"
    header = initialize_shared_pole_bank(
        path, meta=meta, tables=tables, recipe=recipe, identity=identity, mesh_xy=mesh)
    stale = dict(identity, iteration_id="another-iteration")
    with pytest.raises(ValueError, match="stale"):
        validate_shared_pole_bank(path, expected_identity=stale, mesh_xy=mesh)
    W = _matrix(meta, mesh, samples=True, value=2)
    with pytest.raises(ValueError, match="explicit sample_span"):
        write_shared_pole_bank(
            path, q_span=(0, 1), Wc=W, meta=meta,
            expected_identity=identity, mesh_xy=mesh)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        with pytest.raises(ValueError, match="contiguous"):
            read_shared_pole_bank(io, (0, 1), meta=meta, header=header,
                                  sample_span=slice(0, 2, 2))
    assert not np.asarray(validate_shared_pole_bank(
        path, expected_identity=identity, mesh_xy=mesh)["sample_written"]).any()
    # Typed on-disk roles must authenticate against the recipe, not merely
    # carry the right shape. This corrupts only a disposable partial fixture.
    def corrupt_role():
        import h5py
        with h5py.File(path, "a") as file:
            file["role"][0] = np.int8(4)
    rank0_transaction(path, stage="test.bank_role_corruption", write=corrupt_role)
    with pytest.raises(ValueError, match="typed plan role"):
        validate_shared_pole_bank(path, expected_identity=identity, mesh_xy=mesh)


def check_bank_roundtrip(mesh, path):
    """Combined P4 runner entry; path is the shared evidence directory."""
    assert tuple(mesh.axis_names) == ('x', 'y')
    assert int(mesh.shape['x']) > 1 and int(mesh.shape['y']) > 1
    Path(path).mkdir(parents=True, exist_ok=True)
    test_bank_partial_resume_and_immutable_completion(Path(path))
    test_bank_stale_identity_and_invalid_spans(Path(path))
