"""Hostile provenance and finalized-payload gates for MPA artifacts."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from file_io import mpa_store
from gw.mpa import fit_driver, model, sigma


_CERT = {
    "condition_max_allowed": 10.0,
    "backward_error_max_allowed": 1.0,
}
_SCHEME = "mean-field-content-v1:full-mf-header+bounded-wfns"


def test_every_driver_screening_call_carries_charge_zeta_identity():
    source = Path(__file__).parents[1] / "src" / "gw" / "gw_jax.py"
    tree = ast.parse(source.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_screening_model"
    ]

    assert calls
    assert all(
        "charge_zeta_identity" in {keyword.arg for keyword in call.keywords}
        for call in calls
    )


def _finalized_fit(path, *, provenance=None):
    mpa_store.allocate_fit_store(
        path, n_q=1, n_mu=2, n_p=1, energy_unit="Ry",
        grid_hash="grid", table_hash="table", centroid_hash="centroids",
        provenance=provenance)
    omega = np.full((1, 2, 2), 0.8 - 0.1j, dtype=np.complex128)
    residue = np.ones_like(omega)
    diagnostics = {
        "condition": np.ones((2, 2)),
        "backward_error": np.zeros((2, 2)),
    }
    mpa_store.write_fit_block(
        path, 0, [0, 1], omega, residue, diagnostics)
    mpa_store.finalize_fit_store(path, certification=_CERT)


def test_sample_writer_stamps_the_canonical_wfn_owner(
        tmp_path, monkeypatch):
    source_wfn = object()
    monkeypatch.setattr(
        "common.parallel_transport.wfn_fingerprint",
        lambda source: "current-wfn" if source is source_wfn else None)
    monkeypatch.setattr("common.collectives.process_rank", lambda: 0)
    monkeypatch.setattr(
        "common.collectives.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model.sample_plan, "plan_z",
        lambda _plan: np.asarray([0.0 + 0.1j, 1.0 + 0.1j]))
    monkeypatch.setattr(
        model.sample_plan, "refuse_unsupported", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model, "_q_wedge",
        lambda *args, **kwargs: (
            np.asarray([0]), object(), SimpleNamespace()))
    monkeypatch.setattr(
        model.mpa_store, "refuse_completed_artifact_replacement",
        lambda *args, **kwargs: ())
    captured = {}

    class ReachedSampleAllocation(RuntimeError):
        pass

    def prepare(_path, components, **kwargs):
        captured.update(components[0])
        raise ReachedSampleAllocation

    monkeypatch.setattr(
        model.mpa_store, "prepare_w_sample_store_collective", prepare)
    config = SimpleNamespace(
        mpa=SimpleNamespace(
            n_poles=1, sampling_alpha=1, sampling_schedule="nested",
            metal_origin_shift_ry=None, overwrite_completed_artifacts=False),
        screening=SimpleNamespace(diagrams="w_rpa"),
    )
    with pytest.raises(ReachedSampleAllocation):
        model.build_mpa_fit(
            tmp_path, "oneshot", wfns=None, wfn=source_wfn, V_q=None,
            quad=SimpleNamespace(x_max=1.0),
            sym=SimpleNamespace(trs_allowed=True), centroid_indices=None,
            head_resolver=None, config=config,
            meta=SimpleNamespace(n_rmu=2), mesh_xy=None, plan=object(),
            material_class="insulator",
            charge_zeta_identity={"scheme": "zeta-v1", "digest": "zeta-a"})

    assert captured["provenance"] == {
        "screening_diagrams": "w_rpa",
        "wfn_fingerprint_scheme": _SCHEME,
        "wfn_fingerprint": "current-wfn",
        "charge_zeta_identity_scheme": "zeta-v1",
        "charge_zeta_identity": "zeta-a",
    }


def test_fit_identity_is_copied_from_samples_not_injected_by_caller():
    identity = {
        "wfn_fingerprint_scheme": _SCHEME,
        "wfn_fingerprint": "sample-wfn",
        "charge_zeta_identity_scheme": "zeta-v1",
        "charge_zeta_identity": "zeta-a",
    }
    header = {"provenance": identity}
    got = fit_driver._bind_sample_artifact_identities(
        header, None, {"screening_diagrams": "w_rpa"})
    assert {key: got[key] for key in identity} == identity

    with pytest.raises(ValueError, match="cannot inject"):
        fit_driver._bind_sample_artifact_identities(
            {"provenance": {}}, None, identity)
    with pytest.raises(ValueError, match="partial"):
        fit_driver._bind_sample_artifact_identities(
            {"provenance": {"wfn_fingerprint": "orphan"}}, None, {})
    with pytest.raises(ValueError, match="different WFN identities"):
        fit_driver._bind_sample_artifact_identities(
            header,
            {"provenance": dict(identity, wfn_fingerprint="other")}, {})


def test_canonical_wfn_identity_uses_binding_and_refuses_wrong_object():
    from common.parallel_transport import bind_wfn_fingerprint

    source = SimpleNamespace(
        energies=np.asarray([[0.0]]), kpoints=np.asarray([[0.0, 0.0, 0.0]]),
        nelec=1, nspinor=1, nbands=1, path=None)
    other = SimpleNamespace(
        energies=source.energies, kpoints=source.kpoints,
        nelec=1, nspinor=1, nbands=1, path=None)
    binding = bind_wfn_fingerprint(source)

    identity = model._canonical_wfn_identity(source, binding)
    assert identity["wfn_fingerprint_scheme"] == _SCHEME
    with pytest.raises(ValueError, match="different loaded WFN object"):
        model._canonical_wfn_identity(other, binding)


def test_charge_zeta_identity_ignores_only_wfn_locator_fields(monkeypatch):
    from gw import gw_init

    source = object()
    monkeypatch.setattr(
        "common.parallel_transport.wfn_fingerprint",
        lambda candidate: "same-wfn" if candidate is source else "other")
    base = {
        "schema": gw_init._ZETA_PROVENANCE_SCHEMA,
        "wfn_file": "/stage/a/WFN.h5", "wfn_bytes": 111,
        "n_rmu": 2070, "zeta_cutoff_ry": 8.0,
    }
    moved = dict(base, wfn_file="/stage/b/copied-WFN.h5", wfn_bytes=999)
    changed = dict(base, zeta_cutoff_ry=8.5)
    first = gw_init.charge_zeta_identity(json.dumps(base), wfn=source)
    second = gw_init.charge_zeta_identity(
        json.dumps(moved), wfn=source)
    third = gw_init.charge_zeta_identity(
        json.dumps(changed), wfn=source)
    assert first == second
    assert first["digest"] != third["digest"]


def test_mpa_refuses_legacy_restart_without_charge_zeta_receipt():
    with pytest.raises(ValueError, match="restart = false"):
        model._canonical_charge_zeta_identity(None)
    with pytest.raises(ValueError, match="exactly scheme and digest"):
        model._canonical_charge_zeta_identity({"scheme": "only"})


def test_oneshot_driver_hands_charge_zeta_identity_to_every_screening_call():
    driver = Path(__file__).parents[1] / "src/gw/gw_jax.py"
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_screening_model"
    ]
    assert calls
    for call in calls:
        values = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "charge_zeta_identity" in values
        assert ast.unparse(values["charge_zeta_identity"]) == (
            "isdf.charge_zeta_identity")


def test_explicit_identity_refuses_missing_and_wrong_wfn(tmp_path):
    missing = tmp_path / "missing_identity.h5"
    _finalized_fit(missing)
    with pytest.raises(ValueError, match="wfn_fingerprint_scheme"):
        mpa_store.validate_fit_store(
            missing, expected_identity={
                "wfn_fingerprint_scheme": _SCHEME,
                "wfn_fingerprint": "current-wfn",
            })

    stamped = tmp_path / "wrong_identity.h5"
    _finalized_fit(stamped, provenance={
        "wfn_fingerprint_scheme": _SCHEME,
        "wfn_fingerprint": "source-a",
    })
    with pytest.raises(ValueError, match="wfn_fingerprint"):
        mpa_store.validate_fit_store(
            stamped, expected_identity={
                "wfn_fingerprint_scheme": _SCHEME,
                "wfn_fingerprint": "source-b",
            })
    ledger = mpa_store.validate_fit_store(
        stamped, expected_identity={
            "wfn_fingerprint_scheme": _SCHEME,
            "wfn_fingerprint": "source-a",
        })
    assert ledger["provenance"]["wfn_fingerprint"] == "source-a"


@pytest.mark.parametrize(
    ("dataset_name", "shape", "dtype", "message"),
    [
        ("Omega_p", (1, 1, 2, 1), np.complex128,
         "expected exact logical shape"),
        ("B_p", (1, 1, 2, 2), np.complex64,
         "complex64, expected complex128"),
    ],
)
def test_finalized_fit_refuses_short_or_wrong_dtype_pole_payload(
        tmp_path, dataset_name, shape, dtype, message):
    path = tmp_path / f"bad_{dataset_name}.h5"
    _finalized_fit(path)
    with h5py.File(path, "a") as h5:
        del h5[dataset_name]
        h5.create_dataset(dataset_name, shape=shape, dtype=dtype)

    with pytest.raises(ValueError, match=message):
        mpa_store.validate_fit_store(path)


def test_streamed_nonfinite_pole_slab_refuses_before_planning():
    omega = jnp.ones((2, 1, 2, 2), dtype=jnp.complex128)
    residue = jnp.asarray(np.ones((2, 1, 2, 2), np.complex128))
    residue = residue.at[1, 0, 0, 1].set(jnp.nan + 0.0j)

    with pytest.raises(ValueError, match=r"pole_range=\[4,6\).*B_p"):
        sigma._refuse_nonfinite_pole_slab(4, omega, residue)


def test_screening_hands_current_wfn_to_explicit_reuse(monkeypatch):
    from gw import screening
    from gw.gw_config import ComputeMode, ScreeningDiagrams

    source_wfn = object()
    captured = {}

    def validate(path, **kwargs):
        captured.update(kwargs)
        return path

    monkeypatch.setattr(model, "validate_reused_mpa_fit", validate)
    config = SimpleNamespace(
        screening=SimpleNamespace(diagrams=ScreeningDiagrams.W_RPA),
        mpa=SimpleNamespace(fit_reuse_file="certified-fit.h5"),
    )
    result = screening.compute_screening_model(
        ComputeMode.MPA, None, None, quad=None, e_ref=0.0,
        sym=SimpleNamespace(trs_allowed=True), centroid_indices=None,
        config=config, meta=None, mesh_xy=None, run_dir="unused",
        label="oneshot", wfn=source_wfn, mpa_plan=object(),
        wfn_fingerprint_binding="bound-current-wfn",
        charge_zeta_identity={"scheme": "zeta-v1", "digest": "zeta-a"},
        occupation_state=None, material_class="insulator",
        print_fn=lambda *args, **kwargs: None)

    assert result == {"mpa_fit": "certified-fit.h5", "mpa_fit_reused": True}
    assert captured["wfn"] is source_wfn
    assert captured["wfn_fingerprint_binding"] == "bound-current-wfn"
    assert captured["charge_zeta_identity"] == {
        "scheme": "zeta-v1", "digest": "zeta-a"}
