"""The shared-pole store's direct final write against the staged route (toy stores, one device).

With a held K extent each batch lands in the final datasets; the file must hold the same
bytes, census, digest and commit as the staged route, a batch past the extent falls back to
staging through one link move, and a partial model carries no commit.
"""
import json
from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

NQ, NMU, KP = 5, 6, 4
K = np.array([3, 2, 0, 4, 1], np.int64)
IDENTITY = {"iteration_id": "toy-1", "label": "toy"}


def _toy(monkeypatch, held):
    import file_io  # noqa: F401  (service path bootstrap)
    import file_io.shared_pole_store as store
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    basis = SimpleNamespace(mesh_xy=mesh, n_packed=NMU, n_canonical=NMU, n_logical=NMU,
                            canonical_indices=np.arange(NMU), unpack_axis=lambda a, axis: a)
    meta = SimpleNamespace(mu_basis=basis, shared_pole_k_capacity=held, nspinor=2)
    header = {"schema": store.SCHEMA, "identity": IDENTITY, "n_q_irr": NQ, "n_mu_logical": NMU,
              "nspinor": 2, "finalized": False, "recipe": {}}
    for name, value in (("_metadata", lambda *a, **k: dict(header)), ("_write_metadata", lambda io, h: None),
                        ("_admit", lambda *a, **k: None), ("_check_io_capacity", lambda *a, **k: None),
                        ("_capacity", lambda meta: None), ("_conversion_bytes", lambda *a, **k: (0, 0, 0)),
                        ("_check_basis", lambda meta, header, basis=None: basis)):
        monkeypatch.setattr(store, name, value)
    rng = np.random.default_rng(7)
    active = np.arange(KP)[None, :] < K[:, None]
    b = np.where(active[:, None, None, :], rng.normal(size=(NQ, NMU, 1, KP)) + 1j * rng.normal(size=(NQ, NMU, 1, KP)), 0.0)
    poles = np.where(active, np.sort(rng.uniform(0.5, 3.0, size=(NQ, KP)), axis=1), 1.0)

    def write(path, batches, direct=True):
        meta.shared_pole_k_capacity = None if held is None else dict(held)
        with monkeypatch.context() as m:
            if not direct:
                m.setattr(store, "_direct_extent", lambda *a, **k: None)
            for lo, hi in batches:
                header = store.write_shared_pole_model(
                    str(path), jax.device_put(b[lo:hi], NamedSharding(mesh, P(None, "x", None, "y"))),
                    jax.device_put(poles[lo:hi], NamedSharding(mesh, P())), K[lo:hi], q_span=(lo, hi),
                    meta=meta, tables=None, recipe={}, receipts={"identity": IDENTITY}, ordered=False, basis=basis)
        return header
    return store, write


def _file(path):
    with h5py.File(path, "r") as f:
        header = json.loads(f["header_json"][()].decode())
        return (f["factor"][:], f["poles2_ry2"][:], f["K"][:], f["written_q"][:], "staging" in f,
                f["final_commit"][()].decode() if "final_commit" in f else None, header)


def _same_model(direct, staged):
    for d, s in zip(direct[:4], staged[:4]):
        assert np.array_equal(d, s)
    assert direct[4] is False and staged[4] is False and direct[3].all()
    assert direct[5] == staged[5] == direct[6]["digest"] == staged[6]["digest"]
    assert direct[6]["finalized"] and direct[6]["Kmax"] == staged[6]["Kmax"]


@pytest.mark.parametrize("held, batches, extent", [
    ({"None": 4}, ((0, 3), (3, 5)), 4),   # held extent, two SC rounds
    ({"None": 4}, ((0, 5),), 4),          # held extent, one round
    (None, ((0, 5),), 4),                 # no SC hold: the whole model in one batch (scalar constructor)
])
def test_direct_write_matches_the_staged_route(tmp_path, monkeypatch, held, batches, extent):
    store, write = _toy(monkeypatch, held)
    header = write(tmp_path / "direct.h5", batches)
    assert header["direct_extent"] == header["Kmax"] == extent
    assert write(tmp_path / "staged.h5", batches, direct=False)["direct_extent"] is None
    _same_model(_file(tmp_path / "direct.h5"), _file(tmp_path / "staged.h5"))


def test_a_batch_past_the_held_extent_falls_back_to_staging(tmp_path, monkeypatch):
    # Batch 1 (K <= 3) lands direct at the held extent 3; batch 2 (K = 4) moves
    # the direct datasets into staging and finalization grows the extent.
    store, write = _toy(monkeypatch, {"None": 3})
    header = write(tmp_path / "direct.h5", ((0, 3), (3, 5)))
    assert header["direct_extent"] is None and header["Kmax"] == int(np.ceil(4 * (1.0 + store._K_HEADROOM)))
    assert [b["name"] for b in header["batches"]] == ["staging/direct", "staging/q3_5"]
    write(tmp_path / "staged.h5", ((0, 3), (3, 5)), direct=False)
    _same_model(_file(tmp_path / "direct.h5"), _file(tmp_path / "staged.h5"))


def test_a_partial_direct_model_has_no_commit_and_is_refused(tmp_path, monkeypatch):
    store, write = _toy(monkeypatch, {"None": 4})
    header = write(tmp_path / "partial.h5", ((0, 3),))
    assert header["written_q"] == [True] * 3 + [False] * 2 and not header["finalized"]
    with h5py.File(tmp_path / "partial.h5", "r") as f:
        assert "final_commit" not in f and "factor" in f
        assert not json.loads(f["header_json"][()].decode())["finalized"]
    with pytest.raises(Exception, match="incomplete q census|shared_pole_store"):
        store.validate_shared_pole_model(str(tmp_path / "partial.h5"), expected_identity=IDENTITY, mesh_xy=None)
