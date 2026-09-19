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
from file_io import shared_pole_store as store
from file_io.slab_io import SlabIO
from common.collectives import rank0_transaction
from test_shared_pole_store import _fixture


def _bank_fixture():
    if len(jax.devices()) != 4:
        pytest.skip("scratch store contract requires exactly P4")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    meta, tables, recipe, identity = _fixture(mesh)
    recipe = deepcopy(recipe)
    # Canonical code table published by IINPUTS under coordinator ruling 4.
    recipe.update(
        role_codes={"line": 0, "imaginary": 1, "infinity": 2,
                    "held_line": 3, "held_imaginary": 4},
        z_ry=np.asarray([0.1j, 0.1j, 0.2+0.1j], dtype=np.complex128),
        role=np.asarray([0, 1, 3], dtype=np.int8),
        distinct_id=np.asarray([0, 0, 1], dtype=np.int64),
        held=np.asarray([False, False, True], dtype=np.bool_),
        support_pair=np.asarray([[-1,-1],[-1,-1],[0,1]], dtype=np.int64),
        fit_ids=np.asarray([0], dtype=np.int64),
        held_ids=np.asarray([1], dtype=np.int64))
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


def test_ordered_scalar_bank_roundtrips_literal_mirrors(tmp_path):
    """A scalar broken-TRS bank stores exact -conj(z) mirror samples."""
    mesh, meta, tables, recipe, identity = _bank_fixture()
    tables["sym"].trs_allowed = False
    path = tmp_path / "ordered_scalar_bank.h5"
    header = initialize_shared_pole_bank(
        path, meta=meta, tables=tables, recipe=recipe,
        identity=identity, mesh_xy=mesh)
    assert header["ordered"] is True
    assert header["mirror_mode"] == "literal_same_operator_v1"
    fields = store._bank_sample_fields(header)
    assert "Wc_mirror" in fields and "dWc_mirror_ds" in fields
    W = _matrix(meta, mesh, samples=True, value=3)
    D = _matrix(meta, mesh, samples=True, value=-5)
    write_shared_pole_bank(
        path, q_span=(0, 1), sample_span=(0, 1), Wc=W, dWc_ds=D,
        Wc_mirror=W, dWc_mirror_ds=D,
        meta=meta, expected_identity=identity, mesh_xy=mesh)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        got = read_shared_pole_bank(
            io, (0, 1), meta=meta, header=header, sample_span=(0, 1),
            fields=("Wc_mirror", "dWc_mirror_ds"))
    assert bool(jnp.all(got["Wc_mirror"] == W))
    assert bool(jnp.all(got["dWc_mirror_ds"] == D))
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


def test_ordered_bank_carries_odd_moments(tmp_path):
    """An ordered bank stores M0/M2 beside M1/M3 (one source for the infinity
    block); a time-reversal-symmetric bank keeps its two moment fields only."""
    from types import SimpleNamespace
    mesh, meta, tables, recipe, identity = _bank_fixture()
    even_path = tmp_path / "scratch_even_fields.h5"
    even = initialize_shared_pole_bank(
        even_path, meta=meta, tables=tables, recipe=recipe, identity=identity, mesh_xy=mesh)
    assert "odd_moments" not in even and "ordered" not in even
    assert np.asarray(even["moment_written"]).shape == (even["bank_shape"]["nq"], 2)
    M = _matrix(meta, mesh, samples=False, value=7)
    with pytest.raises(ValueError, match="no M0 field"):
        write_shared_pole_bank(even_path, q_span=(0, 1), M0=M, meta=meta,
                               expected_identity=identity, mesh_xy=mesh)
    with SlabIO(even_path, mode="r", mesh=mesh) as io:
        with pytest.raises(ValueError, match="distinct"):
            read_shared_pole_bank(io, (0, 1), meta=meta, header=even, fields=("M0",))
    sym = SimpleNamespace(**vars(tables["sym"]))
    sym.trs_allowed = False
    path = tmp_path / "scratch_ordered.h5"
    header = initialize_shared_pole_bank(
        path, meta=meta, tables=dict(tables, sym=sym), recipe=recipe,
        identity=identity, mesh_xy=mesh)
    assert header["ordered"] and header["odd_moments"]
    assert header["representation"] == "charge-ordered-z"
    nq = header["bank_shape"]["nq"]
    assert np.asarray(header["moment_written"]).shape == (nq, 4)
    W = _matrix(meta, mesh, samples=True, value=3)
    D = _matrix(meta, mesh, samples=True, value=-5)
    moments = {name: _matrix(meta, mesh, samples=False, value=value)
               for name, value in (("M0", 1), ("M1", 11), ("M2", -2), ("M3", 13))}
    for q in range(nq):
        for sample in range(2):
            write_shared_pole_bank(
                path, q_span=(q, q+1), sample_span=(sample, sample+1), Wc=W, dWc_ds=D,
                meta=meta, expected_identity=identity, mesh_xy=mesh)
        header = write_shared_pole_bank(
            path, q_span=(q, q+1), M1=moments["M1"], M3=moments["M3"],
            meta=meta, expected_identity=identity, mesh_xy=mesh)
        assert not header["complete"]
        header = write_shared_pole_bank(
            path, q_span=(q, q+1), M0=moments["M0"], M2=moments["M2"],
            meta=meta, expected_identity=identity, mesh_xy=mesh)
    header = validate_shared_pole_bank(
        path, expected_identity=identity, mesh_xy=mesh, require_complete=True)
    assert header["complete"] and header["final_commit"]
    with SlabIO(path, mode="r", mesh=mesh) as io:
        actual = read_shared_pole_bank(io, (nq-1, nq), meta=meta, header=header,
                                       fields=("M0", "M1", "M2", "M3"))
    for name, expected in moments.items():
        assert bool(jnp.all(actual[name] == expected))


def test_bank_reads_parent_lists_in_face_and_batch_layout(tmp_path):
    """Parents by id list, in either layout, carry each parent's own committed values.

    Four parents with distinct values. A contiguous run in batch layout is one read in which
    each rank reads its own whole rows; a non-contiguous list with repeats (a round's synthetic
    slots) is one face read per parent stacked in order and moved by the staged exchange.
    Every leading row must equal that parent's single-parent face read bit for bit, in
    P(('x','y'), None, ...) or P(None, ..., 'x', 'y') as asked. RED TWIN: the list read in
    sorted order differs from the requested order.
    """
    from symmetry_maps import QirrTables
    mesh, meta, tables, recipe, identity = _bank_fixture()
    qt = tables["qirr"]
    tables = dict(tables, q_irr_full_idx=np.arange(4, dtype=np.int64),
                  qirr=QirrTables(irr_idx_q=np.arange(27, dtype=np.int32) % 4, sym_idx_q=qt.sym_idx_q,
                                  q_irr_frac=np.asarray([[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0], [0, 1/3, 0]]),
                                  sym_perm=qt.sym_perm, L_table=qt.L_table, n_sym_spatial=qt.n_sym_spatial))
    path = tmp_path / "scratch_lists.h5"
    header = initialize_shared_pole_bank(
        path, meta=meta, tables=tables, recipe=recipe, identity=identity, mesh_xy=mesh)
    nq = header["bank_shape"]["nq"]
    assert nq == 4
    for q in range(nq):
        for sample in range(2):
            write_shared_pole_bank(
                path, q_span=(q, q+1), sample_span=(sample, sample+1),
                Wc=_matrix(meta, mesh, samples=True, value=10*q + sample),
                dWc_ds=_matrix(meta, mesh, samples=True, value=-10*q - sample),
                meta=meta, expected_identity=identity, mesh_xy=mesh)
        header = write_shared_pole_bank(
            path, q_span=(q, q+1), M1=_matrix(meta, mesh, samples=False, value=100+q),
            M3=_matrix(meta, mesh, samples=False, value=200+q),
            meta=meta, expected_identity=identity, mesh_xy=mesh)
    header = validate_shared_pole_bank(path, expected_identity=identity, mesh_xy=mesh, require_complete=True)
    fields = ("Wc", "dWc_ds", "M1", "M3")
    batch = P(('x', 'y'))
    with SlabIO(path, mode="r", mesh=mesh) as io:
        single = [read_shared_pole_bank(io, (q, q+1), meta=meta, header=header, sample_span=(0, 2),
                                        fields=fields) for q in range(nq)]
        cases = {"span_face": dict(q_span=(0, 4)), "ids_face": dict(q_ids=[3, 1, 1, 0]),
                 "run_batch": dict(q_ids=[0, 1, 2, 3], partition_spec=batch),
                 "ids_batch": dict(q_ids=[2, 0, 3, 3], partition_spec=batch)}
        for label, request in cases.items():
            got = read_shared_pole_bank(io, meta=meta, header=header, sample_span=(0, 2), fields=fields, **request)
            ids = request.get("q_ids", list(range(4)))
            for name in fields:
                spec = got[name].sharding.spec
                if "partition_spec" in request:
                    assert tuple(spec[0]) == ('x', 'y') and all(e is None for e in tuple(spec)[1:]), (label, spec)
                else:
                    assert tuple(spec)[-2:] == ('x', 'y'), (label, spec)
                value = np.asarray(got[name])
                for row, q in enumerate(ids):
                    assert np.array_equal(value[row], np.asarray(single[q][name])[0]), (label, name, row)
            if label == "ids_batch":
                ordered = read_shared_pole_bank(io, meta=meta, header=header, sample_span=(0, 2), fields=("M1",),
                                                q_ids=sorted(ids), partition_spec=batch)
                assert not np.array_equal(np.asarray(ordered["M1"]), np.asarray(got["M1"]))
        with pytest.raises(ValueError, match="multiple of 4"):
            read_shared_pole_bank(io, meta=meta, header=header, fields=("M1",), q_ids=[0, 1, 2],
                                  partition_spec=batch)
        with pytest.raises(ValueError, match="exactly one of q_span or q_ids"):
            read_shared_pole_bank(io, (0, 1), meta=meta, header=header, fields=("M1",), q_ids=[0])
