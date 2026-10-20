"""Host-only execution of persistent-state guards; no JAX/GPU initialization."""
import ast
import json
from pathlib import Path
import sys
from types import ModuleType

import h5py
import numpy as np
import pytest


def owner(monkeypatch):
    path = Path(__file__).parents[1]/"src/gw/shared_pole_screening.py"
    tree = ast.parse(path.read_text())
    names = {"_require_persistent_state_identity", "_authenticated_constructor_resume",
             "_published_sector_handle"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(json=json, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    prepare = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "prepare_output")
    scope["_prepare_node"] = prepare
    store = ModuleType("file_io.shared_pole_store")
    store.calls = []
    def validate(*args, **kwargs):
        store.calls.append((args, kwargs))
        return {}
    store.validate_shared_pole_bank = validate
    slab = ModuleType("file_io.slab_io")
    slab.remove_stale_streamed_banks = lambda paths: []
    package = ModuleType("file_io")
    package.__path__ = []
    for module in (package, store, slab):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    scope["_store"] = store
    return scope


def sc_identity():
    return dict(iteration_id="sc_0007", hamiltonian="sc_map_7:same_f",
                wavefunctions="qp_rotation_unreceipted", energies="same_E",
                occupations="same_f", wfn="same_original_MF", recipe_hash="same_recipe")


def prepare(scope, root, identity):
    scope.update(root=root, identity=identity, recipe={}, photon=False, sc_scratch=True,
        run_dir=root.parent, _SWEPT=[], print_fn=lambda *args: None,
        coulomb=dict(sha256="current_V"),
        assert_committed=lambda *args, **kwargs: None)
    exec(compile(ast.Module(body=[scope["_prepare_node"]], type_ignores=[]),
                 "shared_pole_screening.prepare_output", "exec"), scope)
    return scope["prepare_output"]


@pytest.mark.parametrize("marker", [dict(iteration_id="sc_0001"),
    dict(hamiltonian="sc_map_1:f"), dict(wavefunctions="qp_rotation_unreceipted"),
    dict(authentication="NON-AUTHENTICATING")])
def test_all_restart_owner_non_authenticating_markers_refuse(monkeypatch, marker):
    with pytest.raises(ValueError, match="shared_pole_sc_reuse.*fresh run directory"):
        owner(monkeypatch)["_require_persistent_state_identity"](marker)


def test_same_energy_occupation_different_rotation_cannot_reuse_committed_model(tmp_path, monkeypatch):
    scope = owner(monkeypatch)
    energy, occupation = np.array([0., 1.]), np.array([1., 0.])
    first_u, second_u = np.eye(2), np.array([[0., 1.], [1., 0.]])
    assert not np.array_equal(first_u @ np.diag(energy) @ first_u.T,
                              second_u @ np.diag(energy) @ second_u.T)
    # The legacy producer did not bind either U: both identities collide.
    first, second = sc_identity(), sc_identity()
    assert first == second and np.array_equal(occupation, [1., 0.])
    model = tmp_path/"model.h5"
    with h5py.File(model, "w") as h5:
        h5.create_dataset("final_commit", data=1)
        h5.create_dataset("header_json", data=json.dumps(dict(identity=first)).encode())
    before = model.read_bytes()
    with pytest.raises(ValueError, match="shared_pole_sc_reuse"):
        prepare(scope, tmp_path, second)()
    assert model.read_bytes() == before
    assert list(tmp_path.iterdir()) == [model]


def test_unreceipted_complete_bank_refuses_before_validator_and_preserves_files(tmp_path, monkeypatch):
    scope, identity = owner(monkeypatch), sc_identity()
    for name in ("bank_receipt.json", "moments_receipt.json", "bank.h5"):
        (tmp_path/name).write_bytes(b"preserved old scratch")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="shared_pole_sc_reuse"):
        scope["_authenticated_constructor_resume"](tmp_path, identity, {})
    assert not scope["_store"].calls
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_unreceipted_sector_and_partial_scratch_are_not_overwritten(tmp_path, monkeypatch):
    scope = owner(monkeypatch)
    manifest = tmp_path/"sectors.json"
    manifest.write_text(json.dumps(dict(identity=sc_identity())))
    with pytest.raises(ValueError, match="shared_pole_sc_reuse"):
        scope["_published_sector_handle"](tmp_path, sc_identity())
    manifest.unlink()
    saved = tmp_path/"partial_receipt.json"
    saved.write_text("keep")
    with pytest.raises(ValueError, match="shared_pole_sc_reuse"):
        prepare(scope, tmp_path, sc_identity())()
    assert saved.read_text() == "keep"


def test_fresh_sc_construction_allowed_but_no_identity_upgrade(tmp_path, monkeypatch):
    scope, identity = owner(monkeypatch), sc_identity()
    root = tmp_path/"fresh_map"
    assert prepare(scope, root, identity)() is False
    assert root.is_dir() and not list(root.iterdir())
    assert identity == sc_identity()
    assert scope["_authenticated_constructor_resume"](root, identity, {}) is False


def test_authentic_bank_resume_and_committed_model_routes_unchanged(tmp_path, monkeypatch):
    scope = owner(monkeypatch)
    identity = dict(iteration_id="oneshot", hamiltonian="authenticated_H",
                    wavefunctions="authenticated_U", energies="E", occupations="f")
    coulomb = dict(sha256="current_V")
    for name in ("bank", "moments"):
        (tmp_path/f"{name}_receipt.json").write_text(json.dumps(dict(identity=identity,
            completion=True, bank_complete=True, coulomb_identity=coulomb)))
    (tmp_path/"bank.h5").write_bytes(b"")
    assert scope["_authenticated_constructor_resume"](
        tmp_path, identity, {}, coulomb_sha256="current_V") is True
    assert len(scope["_store"].calls) == 1
    assert scope["_authenticated_constructor_resume"](
        tmp_path, identity, {}, coulomb_sha256="other_V") is False
    with h5py.File(tmp_path/"model.h5", "w") as h5:
        h5.create_dataset("final_commit", data=1)
        h5.create_dataset("header_json", data=json.dumps(dict(identity=identity)).encode())
    # Preserve the existing committed-scratch branch for an authenticated identity.
    assert prepare(scope, tmp_path, identity)() == "committed"
