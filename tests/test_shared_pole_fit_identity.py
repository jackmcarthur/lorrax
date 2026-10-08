"""Physical-fit changes must invalidate pole reuse on an unchanged DFT WFN."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def _identities(monkeypatch):
    from common import parallel_transport
    from gw import response_bank
    from gw.gw_init import charge_zeta_identity, _ZETA_PROVENANCE_SCHEMA
    from gw.shared_pole_screening import shared_pole_identity

    monkeypatch.setattr(parallel_transport, "wfn_fingerprint", lambda wfn: "same-WFN")
    monkeypatch.setattr(response_bank, "response_weights", lambda wfns, meta: (
        None, dict(energy_sha256="same-energies", occupation_sha256="same-occupations")))
    points = np.array([[.03, .1, .2], [.53, .1, .2]])
    meta = SimpleNamespace(mu_basis=SimpleNamespace(
        canonical_indices=points, coordinate_kind="fractional"),
        shared_pole_recipe=dict(recipe_hash="same-recipe", gate_hash="same-gates"))

    def fit(target, path):
        provenance = dict(schema=_ZETA_PROVENANCE_SCHEMA, wfn_file=path,
            wfn_bytes=123, atomic_augmentation=target,
            charge_pair_training_domain="ordered_lr_plus_rl", band_range_left=[0, 64],
            band_range_right=[0, 120])
        receipt = charge_zeta_identity(json.dumps(provenance), wfn=None)
        identity = shared_pole_identity(None, meta, label="oneshot", wfn=None,
            binding=None, centroid_indices=points, charge_zeta_identity=receipt)
        return receipt, identity

    return meta, fit


def test_same_wfn_different_reconstructed_fit_refuses_pole_member(monkeypatch, tmp_path):
    from file_io import tagged_arrays, shared_pole_store
    from gw.shared_pole_recipe import CapacityLedger, shared_pole_restart_handle

    meta, fit = _identities(monkeypatch)
    _, original = fit("normalized-four-component", "/original/WFN.h5")
    _, relocated = fit("normalized-four-component", "/relocated/WFN.h5")
    _, changed = fit("implicit-Pauli-true-chi", "/original/WFN.h5")
    assert original == relocated
    for key in original.keys() - {"charge_zeta_identity"}:
        assert original[key] == changed[key]
    assert original["charge_zeta_identity"] != changed["charge_zeta_identity"]
    # Exercise the production early-member facade. Its payload reader is a
    # bounded fixture; the actual store identity gate remains the real owner.
    recipe = dict(recipe_version="v", recipe_hash="same-recipe", gate_version="v",
        gate_hash="same-gates", accuracy="production", eta_ev=.25, n=8)
    meta.shared_pole_recipe = recipe
    ledger = object.__new__(CapacityLedger)
    ledger._live_stages = ()
    meta.shared_pole_capacity = ledger
    calls = []

    def stored_member(path, *, expected_identity, **kwargs):
        calls.append(expected_identity)
        shared_pole_store._check_identity(original, expected_identity)
        return dict(path="own/model.h5", digest="own-digest"), dict(
            identity=original, recipe=recipe, K=[4])

    monkeypatch.setattr(tagged_arrays, "read_shared_pole_restart_member", stored_member)
    handle = shared_pole_restart_handle(tmp_path / "fit.h5",
        expected_identity=relocated, meta=meta, mesh_xy=None, print_fn=lambda *a: None)
    assert handle["identity"] == original and handle["digest"] == "own-digest"
    with pytest.raises(ValueError, match="charge_zeta_identity"):
        shared_pole_restart_handle(tmp_path / "fit.h5", expected_identity=changed,
            meta=meta, mesh_xy=None, print_fn=lambda *a: None)
    assert len(calls) == 2
    # Older identities lacking the fit cannot authenticate any current model.
    legacy = {k: v for k, v in original.items() if not k.startswith("charge_zeta_identity")}
    with pytest.raises(ValueError, match="charge_zeta_identity"):
        shared_pole_store._check_identity(legacy, original)


def test_existing_catalogue_receipt_reaches_shared_pole_route(monkeypatch):
    from gw.gw_config import ComputeMode, ScreeningDiagrams
    from gw.screening import compute_screening_model
    from gw import shared_pole_screening

    meta, fit = _identities(monkeypatch)
    receipt, _ = fit("normalized-four-component", "/original/WFN.h5")
    received = []

    def screen(*args, **kwargs):
        received.append(kwargs["charge_zeta_identity"])
        return {"shared_pole": "authenticated-model"}

    monkeypatch.setattr(shared_pole_screening, "screen_shared_poles", screen)
    config = SimpleNamespace(sigma=SimpleNamespace(w_model="shared_pole"),
        screening=SimpleNamespace(diagrams=ScreeningDiagrams.W_RPA))
    result = compute_screening_model(ComputeMode.MPA, None, None, quad=None,
        e_ref=None, sym=None, centroid_indices=meta.mu_basis.canonical_indices,
        config=config, meta=meta, mesh_xy=None, run_dir="unused", label="oneshot",
        material_class="insulator", charge_zeta_identity=receipt)
    assert result == {"shared_pole": "authenticated-model"}
    assert received == [receipt] and received[0] is receipt


def test_missing_fit_receipt_refuses_before_screening_or_model_access():
    from gw.shared_pole_screening import screen_shared_poles

    with pytest.raises(ValueError, match="shared_pole_charge_fit"):
        screen_shared_poles(None, None, None, None, mesh_xy=None, sym=None,
            centroid_indices=None, run_dir="unused", label="oneshot", wfn=None,
            wfn_fingerprint_binding=None, tensors_filename=None,
            occupation_state=None, print_fn=lambda *a: None)


@pytest.mark.parametrize("receipt", [
    {}, {"scheme": "s"}, {"scheme": "s", "digest": ""},
    {"scheme": "s", "digest": "d", "alias": "other-fit"},
])
def test_fit_identity_uses_existing_opaque_receipt_format(monkeypatch, receipt):
    from gw.shared_pole_screening import shared_pole_identity

    meta, _ = _identities(monkeypatch)
    with pytest.raises(ValueError, match="charge_zeta_identity"):
        shared_pole_identity(None, meta, label="oneshot", wfn=None, binding=None,
            centroid_indices=meta.mu_basis.canonical_indices,
            charge_zeta_identity=receipt)
