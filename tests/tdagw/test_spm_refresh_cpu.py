"""CPU checks of the shared-pole collision refresh's host pieces (docs/theory/td_agw_dynamical.md §7.1).

NumPy references for the residue-weighted pole census (``tdagw.dynsig.pole_census`` on
a synthetic model file), the refit receipts' walls and gate rows, the recipe-site
comparison, the occupation duck, the support-site override's optional line height
(``gw.shared_pole_recipe.parse_support_sites``), the deck keys of the shared-pole
member (``tdagw.td_config``) and the refresh-line / log-summary formatting.
"""
import copy
import json
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from common.units import RYD_TO_EV  # noqa: E402
from gw.shared_pole_recipe import parse_support_sites  # noqa: E402
from tdagw import dynsig as D  # noqa: E402
from tdagw.td_config import inactive_keys, parse_td_deck  # noqa: E402


def _model_file(path, rng, K=(5, 3), n=(4, 2)):
    """A shared-pole store's payload: header_json K, poles2_ry2 [q, K], factor [q, a, b, K]."""
    kmax = max(K)
    lam = np.zeros((len(K), kmax))
    fac = np.zeros((len(K),) + tuple(n) + (kmax,), complex)
    for q, k in enumerate(K):
        lam[q, :k] = np.sort(rng.uniform(0.0005, 2.0, k))            # Ry^2: Omega 0.3-19 eV
        fac[q, ..., :k] = rng.normal(size=tuple(n) + (k,)) + 1j * rng.normal(size=tuple(n) + (k,))
    with h5py.File(path, "w") as f:
        f["header_json"] = json.dumps(dict(K=list(K)))
        f["poles2_ry2"] = lam
        f["factor"] = fac
    return lam, fac


def test_pole_census_matches_numpy(tmp_path):
    rng = np.random.default_rng(7)
    p = os.path.join(tmp_path, "model.h5")
    K = (5, 3)
    lam, fac = _model_file(p, rng, K=K)
    c = D.pole_census(p, cuts_ev=(0.3, 1.0, 5.0))
    om, wr, ws = [], [], []
    for q, k in enumerate(K):
        o = np.sqrt(lam[q, :k])
        nb = np.sum(np.abs(fac[q, ..., :k]) ** 2, axis=(0, 1))
        om.append(o * RYD_TO_EV), wr.append(nb / (2 * o)), ws.append(nb / (2 * o ** 2))
    o, r, s = (np.concatenate(x) for x in (om, wr, ws))
    assert c["K"] == list(K)
    assert c["omega_min_ev"] == pytest.approx(o.min(), rel=1e-14)
    for cut in (0.3, 1.0, 5.0):
        key = f"{cut:g}"
        assert c["n_below"][key] == int(np.sum(o < cut))
        assert c["frac_residue_below"][key] == pytest.approx(r[o < cut].sum() / r.sum(), abs=1e-15)
        assert c["frac_static_below"][key] == pytest.approx(s[o < cut].sum() / s.sum(), abs=1e-15)
    assert sum(c["hist_residue"]) == pytest.approx(r.sum(), rel=1e-12)
    assert sum(c["hist_static"]) == pytest.approx(s.sum(), rel=1e-12)
    assert len(c["hist_residue_q"]) == len(K)
    assert sum(c["hist_residue_q"][1]) == pytest.approx(wr[1].sum(), rel=1e-12)


def test_occupation_duck():
    f = np.array([[1.0, 0.95, 0.05, 0.0]])
    d = D._occupation_duck(f, -0.3)
    assert d.mu_ry == -0.3 and d.n_electrons == pytest.approx(2.0)
    assert np.array_equal(d.f_kn, f) and d.smearing_width_ry == 0.0


def _recipe():
    return dict(z_ry=np.array([0.1 + 0.2j, 0.3j]), role=np.array([0, 1], np.int8),
                distinct_id=np.array([0, 1]), held=np.array([False, False]),
                support_pair=-np.ones((2, 2), int), imaginary_width=4, infinity_width=2,
                line_direction_cap=8, pole_budget=16, height_ev=2.6)


def test_recipe_sites_equal():
    a = _recipe()
    assert D._recipe_sites_equal(a, copy.deepcopy(a))
    b = copy.deepcopy(a)
    b["z_ry"][0] += 1e-12
    assert not D._recipe_sites_equal(a, b)
    b = copy.deepcopy(a)
    b["height_ev"] = 1.0
    assert not D._recipe_sites_equal(a, b)


def test_support_sites_height_field():
    two = parse_support_sites("2.6, 3.5 | 1.0, 2.0, 4.0")
    assert two["height_ev"] is None
    # the two-field canonical text is the incumbent's (it enters recipe_hash)
    assert two["text"] == "2.6,3.5|1.0,2.0,4.0"
    three = parse_support_sites("2.6,3.5 | 1.0,2.0,4.0 | 1.0")
    assert three["height_ev"] == 1.0 and three["text"] == "2.6,3.5|1.0,2.0,4.0|1.0"
    assert parse_support_sites(three["text"]) == three
    assert parse_support_sites("") is None
    for bad in ("2.6,3.5|1,2|0", "2.6,3.5|1,2|-1", "2.6,3.5|1,2|nan", "2.6,3.5|1,2|x",
                "2.6,3.5|1,2|1|2", "2.6|1,2"):
        with pytest.raises(ValueError, match="GATE shared_pole_support_sites"):
            parse_support_sites(bad)


def _raw(**lorrax):
    raw = {"control": {"calc_type": "COHSEX",
                       "timesteps": {"t0": 0.0, "tf": 100.0, "dt_measure": 50.0},
                       "files": {"outdir": "/tmp/spm_cpu"}},
           "system": {"ibndmin": 129, "ibndmax": 132, "ivbm": 130, "geometry": "slab",
                      "esmear": 0.05, "Efunc": {"E_x": 0.0, "E_y": 0.0, "E_z": 0.0}},
           "measurements": {"types": ["Efield"]},
           "lorrax": {"method": "cfm4", "dt": 25.0, "n_corr": 2, "chunk": 5, "system": "bundle",
                      "bundle_input": "/a/07/oneshot.in", "dipole_file": "/a/07/dipole.h5",
                      "model": "isdf", "collision": "subtracted", "collision_refresh_fs": 0.5,
                      "collision_hermitian": "dynamic"}}
    raw["lorrax"].update(lorrax)
    return raw


def test_deck_keys_shared_pole():
    with pytest.raises(ValueError, match="GATE td_collision_keldysh_screening"):
        parse_td_deck(_raw(collision_screening="shared_pole",
                           collision_source="/a/04/oneshot.in",
                           collision_refresh_sweep="active",
                           collision_spm_support_sites_ev="2.6, 3.5 | 1.0,2.0, 4.0 | 1.0"),
                      source="cpu")
    # The frozen equilibrium W remains admitted.
    v0 = parse_td_deck(_raw(), source="cpu").values
    assert v0["lorrax.collision_source"] == "/a/07/oneshot.in"
    assert v0["lorrax.collision_spm_support_sites_ev"] == ""


@pytest.mark.parametrize("kw, gate", [
    (dict(collision_source="/a/04/oneshot.in"), "td_collision_source"),
    (dict(collision_screening="frozen", collision_rebase="on"), "td_collision"),
    (dict(collision_rebase="maybe"), "td_collision"),
    (dict(collision_screening="shared_pole", collision_spm_support_sites_ev="1|2"),
     "td_collision_keldysh_screening"),
])
def test_deck_keys_refused(kw, gate):
    with pytest.raises(Exception, match=f"GATE {gate}"):
        parse_td_deck(_raw(**kw), source="cpu")


def test_deck_inactive_sites():
    d = parse_td_deck(_raw(collision="off", collision_screening="refreshed",
                           collision_refresh_sweep="active",
                           collision_spm_support_sites_ev="2.6,3.5|1,2"), source="cpu")
    assert "lorrax.collision_spm_support_sites_ev" in inactive_keys(d)


def _receipts(root, gates_q):
    os.makedirs(root, exist_ok=True)
    json.dump(dict(seconds=dict(total=9.0, rule_construction=1.5, direct_dispatch=1.9,
                                sample_dyson_dispatch=3.4, line_selection=2.1),
                   rule=dict(interval_ry=[-0.1, 6.1], reused=False, decay_rate_ry_inv=0.2), nodes=64),
              open(os.path.join(root, "bank_receipt.json"), "w"))
    json.dump(dict(seconds=dict(total=0.64)), open(os.path.join(root, "moments_receipt.json"), "w"))
    json.dump(dict(seconds=dict(gram_reduction=8.0, direction_selection=2.0, writer=2.8,
                                passivity=0.5, rounds=[1, 2]),
                   q_receipts=[dict(gates=g) for g in gates_q], status="CONSTRUCTED"),
              open(os.path.join(root, "constructor_receipt.json"), "w"))


def test_refit_receipts_and_lines(tmp_path):
    g = lambda name, status, value=None: dict(name=name, status=status, value=value)
    gates_q = [[g("held_w", "PASS", 1e-3), g("capacity", "WARN"), g("passivity", "PASS", 0.0)],
               [g("held_w", "PASS", 4e-3), g("capacity", "WARN"), g("rule_validity", "NOT_MEASURED")]]
    root = os.path.join(tmp_path, "s0t0_shared_pole")
    _receipts(root, gates_q)
    r = D._refit_receipts(root)
    assert r["gates_not_pass"] == [dict(gate="capacity:WARN", n_q=2)]
    assert r["gates_counted"] == 6
    assert r["gate_values_max"]["held_w"] == pytest.approx(4e-3)
    assert r["bank_rule"]["nodes"] == 64 and r["bank_rule"]["interval_ry"] == [-0.1, 6.1]
    w = D._refit_walls(r)
    assert w["bank"] == 9.0 and w["moments"] == 0.64
    assert w["constructor"] == pytest.approx(8.0 + 2.0 + 2.8 + 0.5)       # rounds excluded
    om = [np.array([0.2, 1.5, 3.0]), np.array([0.5, 2.0])]
    wr = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    cen = D.census_summary(om, wr, [x / o for x, o in zip(wr, om)])
    sr = dict(r, recipe_s=0.1, screen_s=23.0, K=[3, 2], sites_moved=False, label="s0t0",
              census=cen, census_s=0.5)
    eq = dict(sr, K=[3, 3])
    rec = D.spm_record([sr, dict(sr)], eq)
    assert rec["walls"]["bank"] == 18.0 and rec["dK_eq"] == 1
    assert rec["gates_not_pass"] == ["capacity:WARN"]
    line = D.spm_line(dict(screen=rec, frozen_s=1.1, frozen_nodes=dict(distinct_tau=48),
                           active_s=0.9, rebase_s=2.0))
    assert "full refit" in line and "capacity:WARN" in line and "poles below 1 eV 2" in line
    t0 = D.spm_t0_line(dict(sr, variant="production", support_sites="",
                            model_vs_run=dict(bitwise=True)))
    assert "bitwise the run's" in t0 and "supports fixed" in t0
    from tdagw.td_log import TdReport
    summary = TdReport._screening_summary([dict(screen=rec, frozen_s=1.1, active_s=0.9, rebase_s=2.0)])
    assert "shared-pole refits" in summary and "capacity:WARN" in summary


def test_fresh_capacity_per_phase(monkeypatch):
    """Each window Sigma evaluation (on-shell pair, then the omega grid) prices on its own ledger.

    Regression (2026-10-19): window_sigma_c(omega_ry=...) reserved 'sigma.panel_budget' twice on
    one ledger (the pair, then the grid) and the ledger refused the repeated stage name.
    """
    from types import SimpleNamespace
    import common.gpu_utils as G
    from gw import ppm_sigma as PS
    monkeypatch.setattr(G, "device_budget_bytes", lambda: 4.0e9)
    meta = SimpleNamespace(nk_tot=4, nspinor=1, n_rmu=8)
    mesh = SimpleNamespace(shape={"x": 1, "y": 1})
    a = PS.fresh_shared_pole_capacity(meta, mesh)
    b = PS.fresh_shared_pole_capacity(a, mesh)
    kw = dict(resident_bytes_per_rank=8, workspace_bytes_per_rank=8)
    a.shared_pole_capacity.reserve("sigma.panel_budget", **kw)
    with pytest.raises(ValueError, match="new nonempty name"):
        a.shared_pole_capacity.reserve("sigma.panel_budget", **kw)
    b.shared_pole_capacity.reserve("sigma.panel_budget", **kw)
    assert a.shared_pole_capacity is not b.shared_pole_capacity
    assert b.shared_pole_capacity.live_stages == ()
    assert not hasattr(meta, "shared_pole_capacity")
