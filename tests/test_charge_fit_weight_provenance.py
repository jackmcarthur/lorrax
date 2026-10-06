"""A positive charge-fit loss is opt-in and invalidates incompatible restarts."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gw import gw_init
from gw.gw_config import LorraxConfig, _input_backend


def _config(tmp_path, extra=""):
    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnval=1\nncond=2\nnumber_bands=7\n"
                    "qp_solver=one_shot_dft\nlinalg=local\ncompute_mode=x_only\n"
                    "sys_dim=3\nbispinor=true\nbispinor_gw=coulomb_only\n" + extra)
    return LorraxConfig.from_input_file(
        str(deck), resolve_hardware=False, print_fn=lambda *_: None)


def _provenance(cfg, *, policy=None, vertex=0):
    return gw_init._zeta_fit_provenance(
        wfn=SimpleNamespace(_filename="", ecutwfc=80., ecutrho=320.),
        meta=SimpleNamespace(n_rmu=4, nspinor_wfnfile=2,
                             fft_grid=np.array([8, 8, 8]), current_basis_rows=None),
        cfg=cfg, band_range_left=(0, 4), band_range_right=(0, 8),
        logical_band_stop=7, zeta_cutoff=80., zeta_vcoul_cutoff=80.,
        write_ibz_only=True, band_norms=None, vertex_mu_L=vertex,
        charge_fit_weights=policy)


def test_unit_weight_retains_exact_legacy_provenance(tmp_path):
    default = _config(tmp_path)
    explicit = _config(tmp_path, "zeta_occupied_weight=1\n")
    assert default.backend.zeta_occupied_weight == 1.
    assert _provenance(default) == _provenance(explicit)
    assert _provenance(default) == _provenance(
        explicit, policy={"occupied_stop": 2, "occupied_weight": 1.})
    assert "charge_fit_endpoint_weights" not in json.loads(_provenance(default))


@pytest.mark.parametrize("value", ["nan", "inf", "0.99", "-4", "true", "garbage"])
def test_bad_input_weight_refuses_before_any_fit(tmp_path, value):
    with pytest.raises(ValueError, match="zeta_occupied_weight|could not convert string to float"):
        _config(tmp_path, f"zeta_occupied_weight={value}\n")


@pytest.mark.parametrize("value", [True, None, "garbage"])
def test_programmatic_backend_weight_refuses_a_nonnumber(value):
    linalg = SimpleNamespace(distributed_lu="off", distributed_cholesky="off")
    with pytest.raises(ValueError, match="zeta_occupied_weight"):
        _input_backend(linalg, {"zeta_occupied_weight": value}, "cpu")


def test_nonunit_weight_authenticates_boundary_and_empty_coverage(tmp_path):
    cfg = _config(tmp_path, "zeta_occupied_weight=4\n")
    policy = {"occupied_stop": 2, "occupied_weight": 4.}
    assert json.loads(_provenance(cfg, policy=policy))["charge_fit_endpoint_weights"] == {
        "schema": "occupied_band_endpoints_v1", "occupied_stop": 2,
        "occupied_weight": 4., "empty_weight": 1.}
    for bad in (None, {"occupied_stop": 2, "occupied_weight": 2.}):
        with pytest.raises(ValueError, match="resolved endpoint policy"):
            _provenance(cfg, policy=bad)
    with pytest.raises(ValueError, match="cannot stamp a current"):
        _provenance(cfg, policy=policy, vertex=1)
    # Charge priorities leave current-channel loss and provenance unchanged.
    assert _provenance(cfg, vertex=1) == _provenance(_config(tmp_path), vertex=1)


def test_weight_or_boundary_change_refuses_reuse_in_both_directions(tmp_path, monkeypatch):
    import file_io.restart_bundle as restart
    import common.parallel_transport as transport

    cents = np.arange(12, dtype=np.int32).reshape(4, 3)
    path = tmp_path / "zeta_q.h5"
    path.touch()
    holder = {}
    monkeypatch.setattr(restart, "read_isdf_header", lambda _: SimpleNamespace(
        zeta_is_done=True, fit_provenance=holder["stamp"], r_mu_fft_idx=cents,
        coordinate_kind="fft_indices", centroid_coordinates=cents))
    monkeypatch.delenv("LORRAX_FORCE_REFIT", raising=False)
    # Isolate fit semantics while holding the authoritative WFN identity fixed.
    monkeypatch.setattr(transport, "wfn_fingerprint", lambda _: "fixed-wfn-content")
    default = _provenance(_config(tmp_path))
    cfg4 = _config(tmp_path, "zeta_occupied_weight=4\n")
    four = _provenance(cfg4, policy={"occupied_stop": 2, "occupied_weight": 4.})
    boundary = _provenance(cfg4, policy={"occupied_stop": 3, "occupied_weight": 4.})
    two = _provenance(_config(tmp_path, "zeta_occupied_weight=2\n"),
                      policy={"occupied_stop": 2, "occupied_weight": 2.})
    for old, new in ((default, four), (four, default), (four, boundary), (four, two)):
        holder["stamp"] = old
        messages = []
        assert not gw_init._zeta_reuse_ok(str(path), new, cents, print_fn=messages.append)
        assert any("charge_fit_endpoint_weights" in message for message in messages)
        assert gw_init.charge_zeta_identity(old, wfn=object()) != gw_init.charge_zeta_identity(new, wfn=object())
    holder["stamp"] = four
    assert gw_init._zeta_reuse_ok(str(path), four, cents, print_fn=lambda *_: None)


@pytest.mark.parametrize("case", ["fractional", "hole", "k_boundary", "negative",
                                  "smearing", "broadening"])
def test_nonunit_loss_requires_fixed_unsmeared_occupied_boundary(tmp_path, case):
    cfg = _config(tmp_path, "zeta_occupied_weight=4\n")
    occ = np.zeros((1, 2, 7))
    occ[..., :2] = 1.
    if case == "fractional":
        occ[0, 0, 1] = .5
    elif case == "hole":
        occ[0, :, 1], occ[0, :, 2] = 0., 1.
    elif case == "k_boundary":
        occ[0, 1, 2] = 1.
    elif case == "negative":
        occ[0, 0, 3] = -1.
    elif case == "smearing":
        cfg = replace(cfg, occ_smearing_width_ry=.01)
    elif case == "broadening":
        # Exercise the pre-fit refusal on supplied programmatic metadata;
        # constructing an invalid one-shot LorraxConfig already refuses.
        cfg = SimpleNamespace(**cfg.__dict__)
        cfg.screening = SimpleNamespace(occ_broadening_ev=.01)
    bands = SimpleNamespace(b0=0, b1=0, b2=2, b3=2, b4=7, full_range=(0, 8))
    with pytest.raises(ValueError, match="zeta_occupied_weight"):
        gw_init._resolve_zeta_fit_contract(
            SimpleNamespace(occs=occ), None, SimpleNamespace(), None, None,
            cfg, bands, str(tmp_path), print_fn=lambda *_: None)
