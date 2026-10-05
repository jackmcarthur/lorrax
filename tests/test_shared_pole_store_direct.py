"""The shared-pole store's direct final write against the staged route (toy stores, one device).

From SC map 1 on, the held K extent is known before round 1, so each batch
lands in the final datasets (``_direct_extent``).  The file must hold the
same bytes, census, digest and commit as the staged route; a batch past the
held extent falls back to staging through one link move; a partial model
refuses finalization and carries no commit.
"""
import json
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

NQ, NMU, KP = 5, 6, 4
K = np.array([3, 2, 0, 4, 1], np.int64)
IDENTITY = {"iteration_id": 1, "label": "toy"}


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _payload():
    rng = np.random.default_rng(7)
    b = rng.normal(size=(NQ, NMU, 1, KP)) + 1j * rng.normal(size=(NQ, NMU, 1, KP))
    poles = np.sort(rng.uniform(0.5, 3.0, size=(NQ, KP)), axis=1)
    active = np.arange(KP)[None, :] < K[:, None]
    b = np.where(active[:, None, None, :], b, 0.0).astype(np.complex128)
    poles = np.where(active, poles, 1.0).astype(np.float64)
    return b, poles


def _toy(monkeypatch, held):
    import file_io  # noqa: F401  (service path bootstrap)
    import file_io.shared_pole_store as store
    mesh = _mesh()
    basis = SimpleNamespace(mesh_xy=mesh, n_packed=NMU, n_canonical=NMU, n_logical=NMU,
                            canonical_indices=np.arange(NMU), unpack_axis=lambda a, axis: a)
    meta = SimpleNamespace(mu_basis=basis, shared_pole_k_capacity=held, nspinor=2)
    header = {"schema": store.SCHEMA, "identity": IDENTITY, "n_q_irr": NQ, "n_mu_logical": NMU,
              "nspinor": 2, "finalized": False, "recipe": {}}
    monkeypatch.setattr(store, "_metadata", lambda *a, **k: dict(header))
    monkeypatch.setattr(store, "_write_metadata", lambda io, header: None)
    monkeypatch.setattr(store, "_admit", lambda *a, **k: None)
    monkeypatch.setattr(store, "_check_io_capacity", lambda *a, **k: None)
    monkeypatch.setattr(store, "_capacity", lambda meta: None)
    monkeypatch.setattr(store, "_conversion_bytes", lambda *a, **k: (0, 0, 0))
    monkeypatch.setattr(store, "_check_basis", lambda meta, header, basis=None: basis)
    return store, mesh, basis, meta


def _write(store, mesh, basis, meta, path, batches):
    b, poles = _payload()
    header = None
    for lo, hi in batches:
        factor = jax.device_put(b[lo:hi], NamedSharding(mesh, P(None, "x", None, "y")))
        p2 = jax.device_put(poles[lo:hi], NamedSharding(mesh, P()))
        header = store.write_shared_pole_model(
            str(path), factor, p2, K[lo:hi], q_span=(lo, hi), meta=meta, tables=None,
            recipe={}, receipts={"identity": IDENTITY}, ordered=False, basis=basis)
    return header


def _file(path):
    with h5py.File(path, "r") as f:
        header = json.loads(f["header_json"][()].decode())
        return dict(factor=f["factor"][:], poles=f["poles2_ry2"][:], K=f["K"][:],
                    written=f["written_q"][:], staging="staging" in f,
                    commit=f["final_commit"][()].decode(), header=header)


def _same_model(direct, staged):
    assert not direct["staging"] and not staged["staging"]
    assert direct["header"]["Kmax"] == staged["header"]["Kmax"]
    assert np.array_equal(direct["factor"], staged["factor"])
    assert np.array_equal(direct["poles"], staged["poles"])
    assert np.array_equal(direct["K"], K) and np.array_equal(staged["K"], K)
    assert direct["written"].all() and staged["written"].all()
    assert direct["header"]["digest"] == staged["header"]["digest"] == direct["commit"] == staged["commit"]
    assert direct["header"]["finalized"] and staged["header"]["finalized"]


@pytest.mark.parametrize("batches", [((0, 3), (3, 5)), ((0, 5),)])
def test_direct_write_matches_the_staged_route(tmp_path, monkeypatch, batches):
    held = {"None": 4}
    store, mesh, basis, meta = _toy(monkeypatch, dict(held))
    direct_header = _write(store, mesh, basis, meta, tmp_path / "direct.h5", batches)
    assert direct_header["direct_extent"] == direct_header["Kmax"] == 4
    with monkeypatch.context() as m:
        m.setattr(store, "_direct_extent", lambda *a, **k: None)
        meta.shared_pole_k_capacity = dict(held)
        staged_header = _write(store, mesh, basis, meta, tmp_path / "staged.h5", batches)
    assert staged_header["direct_extent"] is None
    _same_model(_file(tmp_path / "direct.h5"), _file(tmp_path / "staged.h5"))


def test_one_batch_without_a_held_extent_writes_direct(tmp_path, monkeypatch):
    store, mesh, basis, meta = _toy(monkeypatch, None)
    header = _write(store, mesh, basis, meta, tmp_path / "direct.h5", ((0, 5),))
    assert header["direct_extent"] == header["Kmax"] == int(K.max())
    with monkeypatch.context() as m:
        m.setattr(store, "_direct_extent", lambda *a, **k: None)
        _write(store, mesh, basis, meta, tmp_path / "staged.h5", ((0, 5),))
    _same_model(_file(tmp_path / "direct.h5"), _file(tmp_path / "staged.h5"))


def test_a_batch_past_the_held_extent_falls_back_to_staging(tmp_path, monkeypatch):
    # Batch 1 (K <= 3) lands direct at the held extent 3; batch 2 (K = 4) moves
    # the direct datasets into staging, and finalization grows the extent.
    store, mesh, basis, meta = _toy(monkeypatch, {"None": 3})
    header = _write(store, mesh, basis, meta, tmp_path / "direct.h5", ((0, 3), (3, 5)))
    assert header["direct_extent"] is None
    assert header["Kmax"] == int(np.ceil(4 * (1.0 + store._K_HEADROOM)))
    assert [b["name"] for b in header["batches"]] == ["staging/direct", "staging/q3_5"]
    with monkeypatch.context() as m:
        m.setattr(store, "_direct_extent", lambda *a, **k: None)
        meta.shared_pole_k_capacity = {"None": 3}
        _write(store, mesh, basis, meta, tmp_path / "staged.h5", ((0, 3), (3, 5)))
    _same_model(_file(tmp_path / "direct.h5"), _file(tmp_path / "staged.h5"))


def test_a_partial_direct_model_has_no_commit_and_refuses_finalization(tmp_path, monkeypatch):
    store, mesh, basis, meta = _toy(monkeypatch, {"None": 4})
    path = tmp_path / "partial.h5"
    header = _write(store, mesh, basis, meta, path, ((0, 3),))
    assert header["written_q"] == [True] * 3 + [False] * 2 and not header["finalized"]
    with h5py.File(path, "r") as f:
        assert "final_commit" not in f and "factor" in f
        assert not json.loads(f["header_json"][()].decode())["finalized"]
    with pytest.raises(Exception, match="every staged parent"):
        store.finalize_shared_pole_model(str(path), meta=meta, expected_identity=IDENTITY, basis=basis)
