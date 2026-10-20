"""The TD-aGW deck parser: TDGW's ``tdgw.toml`` plus ``[lorrax]`` (no jax, ms).

G1 the fixture's TDGW deck, with load_header_from and a [lorrax] table,
   parses; times convert as -> a.u. and energies eV -> Ha; a TOML file of
   the same deck parses to the same values
G2 every refusal names its GATE id (one case per gate of parse_td_deck);
   magnus2 refuses a dt that does not divide dt_measure (td_dense_output)
   while cfm4 accepts it, and a restart into a new outdir parses
R1 the unmutated deck is refused by none of G2's checks, so each refusal
   is caused by its mutation and not by the base deck
G3 [lorrax] collision (docs/theory/td_agw_dynamical.md §1.5): its forms, the
   source gauge (system = bundle, collision_source = bundle_input), the refresh
   policy (fs -> a.u.), the restart refusal; the collision keys are flagged
   inactive with collision = off and enter the restart fingerprint only when on
"""
import copy
import json
import os

import pytest

from tdagw.td_config import TD_DEFAULTS, parse_td_deck, read_td_deck
from tdagw.tdgw_io import HA_IN_EV, T_IN_ATTOSEC

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "syn2v2c")


def _base():
    raw = json.load(open(os.path.join(_FIX, "deck_tdgw.json")))["deck"]
    raw["control"]["files"]["load_header_from"] = os.path.join(_FIX, "tdgw_ref")
    raw["lorrax"] = {"method": "cfm4", "dt": 2.5, "n_corr": 1}
    return raw


def _set(raw, dotted, value):
    node = raw
    *path, last = dotted.split(".")
    for p in path:
        node = node.setdefault(p, {})
    if value is _DROP:
        node.pop(last, None)
    elif value is _SAME_AS_OUTDIR:
        node[last] = raw["control"]["files"]["outdir"]
    else:
        node[last] = value


_DROP = object()
_SAME_AS_OUTDIR = object()
_CASES = [
    ("lorrax.metod", "cfm4", "td_unknown_key"),
    ("measurements.3.brange", [1, 2], "td_unknown_key"),
    ("lorrax.trials", [{"E_x": "0", "E_w": "1"}], "td_unknown_key"),
    ("control.calc_type", _DROP, "td_missing_key"),
    ("control.calc_type", "GW", "td_calc_type"),
    ("control.files.load_header_from", "", "td_header_source"),
    ("control.files.use_vmtxel", True, "td_header_source"),
    ("control.files.load_checkpoint_from", _SAME_AS_OUTDIR, "td_checkpoint_outdir"),
    ("system.patched_sampling", True, "td_patched_sampling"),
    ("system.geometry", "box", "td_geometry"),
    ("lorrax.method", "rk45", "td_method"),
    ("lorrax.n_corr", 0, "td_n_corr"),
    ("lorrax.model", "gw", "td_model"),
    ("lorrax.model", "isdf", "td_isdf_input"),
    ("lorrax.isdf_terms", "exchange", "td_isdf_terms"),
    ("lorrax.dt", _DROP, "td_dt"),
    ("lorrax.dt", 0.0, "td_dt"),
    ("measurements.types", ["Efield", "velocity_operator"], "td_measurement_unsupported"),
    ("measurements.types", ["Efield", "Efield"], "td_measurement_names"),
    ("lorrax.collision", "on", "td_collision"),
    ("lorrax.collision_hermitian", "half", "td_collision"),
    ("lorrax.collision_refresh_fs", -1.0, "td_collision_refresh"),
    ("lorrax.collision_refresh_df", "often", "td_collision_refresh"),
    ("lorrax.collision_screening", "dynamic", "td_collision"),
    ("lorrax.collision_frozen_bands", [4], "td_collision"),
    ("lorrax.collision_screening_gamma_ev", -1.0, "td_collision"),
    ("lorrax.collision", "subtracted", "td_collision_source"),
]


def _refuses(raw, gid):
    try:
        parse_td_deck(raw)
    except ValueError as exc:
        return str(exc).startswith(f"GATE {gid}:")
    return False


def test_fixture_deck_parses_with_units():
    deck = parse_td_deck(_base())
    v = deck.values
    assert set(TD_DEFAULTS) <= set(v)
    assert v["lorrax.method"] == "cfm4"
    assert v["lorrax.dt"] == pytest.approx(2.5 / T_IN_ATTOSEC, rel=1e-15)
    assert v["control.timesteps.dt_measure"] == pytest.approx(10.0 / T_IN_ATTOSEC, rel=1e-15)
    assert v["system.dephasing"] == pytest.approx(0.05 / HA_IN_EV, rel=1e-15)
    assert v["lorrax.model"] == "tdgw_kernel"
    assert [m[0] for m in deck.measurements] == ["Efield", "polarization", "current", "rho_cv"]
    assert len(deck.trials) == 1 and "E_x" in deck.trials[0]


def test_toml_file_matches_dict(tmp_path):
    from tdagw.tdgw_io import dump_toml
    raw = _base()
    path = tmp_path / "tdgw.toml"
    dump_toml(str(path), raw)
    assert read_td_deck(str(path)).values == parse_td_deck(raw).values


def test_magnus2_refuses_samples_inside_steps():
    raw = _base()
    raw["lorrax"] = {"method": "magnus2", "dt": 3.0}
    assert _refuses(raw, "td_dense_output")


def test_red_twin_cfm4_samples_inside_steps_parse():
    raw = _base()
    raw["lorrax"] = {"method": "cfm4", "dt": 3.0}
    assert not _refuses(raw, "td_dense_output")
    assert parse_td_deck(raw).values["lorrax.dt"] == pytest.approx(3.0 / T_IN_ATTOSEC)


def test_restart_with_a_new_outdir_parses():
    raw = _base()
    raw["control"]["files"]["load_checkpoint_from"] = "an_earlier_run"
    assert parse_td_deck(raw).values["control.files.load_checkpoint_from"] == "an_earlier_run"


def test_dp54_refuses_a_step():
    raw = _base()
    raw["lorrax"] = {"method": "dp54", "dt": 1.0}
    assert _refuses(raw, "td_dt")


@pytest.mark.parametrize("key,value,gid", _CASES, ids=[f"{c[2]}:{c[0]}" for c in _CASES])
def test_refusal_names_its_gate(key, value, gid):
    raw = copy.deepcopy(_base())
    _set(raw, key, value)
    assert _refuses(raw, gid)


def test_red_twin_base_deck_trips_no_gate():
    base = _base()
    assert not any(_refuses(copy.deepcopy(base), gid) for _, _, gid in _CASES)


def _bundle(**lorrax):
    raw = _base()
    raw["control"]["files"].pop("load_header_from")
    raw["lorrax"].update(system="bundle", bundle_input="gw/oneshot.in", dipole_file="gw/dipole.h5",
                         **lorrax)
    return raw


def test_collision_on_a_bundle_parses():
    deck = parse_td_deck(_bundle(collision="subtracted", collision_refresh_fs=0.5,
                                 collision_refresh_df=1e-3))
    v = deck.values
    assert v["lorrax.collision"] == "subtracted"
    assert v["lorrax.collision_source"] == "gw/oneshot.in"            # defaults to bundle_input
    assert v["lorrax.collision_refresh_fs"] == pytest.approx(500.0 / T_IN_ATTOSEC)
    assert v["lorrax.collision_refresh_df"] == 1e-3
    assert v["lorrax.collision_hermitian"] == "dynamic"
    assert v["lorrax.collision_refresh_sweep"] == "active"
    assert v["lorrax.collision_in_scattering"] == "full"
    assert v["lorrax.collision_rate_intermediates"] == "active"
    assert v["lorrax.collision_basis"] == "band"
    assert v["lorrax.collision_internal_pairing"] == "vertex"


def test_joint_snapshot_controls_parse_and_refuse_invalid_closures():
    deck = parse_td_deck(_bundle(collision="raw", collision_screening="joint",
                                 collision_basis="instantaneous", collision_density_rank=8,
                                 collision_snapshot_steps=32, collision_adiabatic_ev=0.05))
    assert deck.values["lorrax.collision_screening"] == "joint"
    assert deck.values["lorrax.collision_snapshot_steps"] == 32
    assert deck.values["lorrax.collision_hamiltonian"] == "hf"
    assert deck.values["lorrax.collision_projection_order"] == 400
    assert deck.values["lorrax.collision_pencil_mb"] == 128
    capacity = parse_td_deck(_bundle(collision="raw", collision_screening="joint",
        collision_basis="instantaneous", collision_projection_order=800, collision_pencil_mb=256))
    assert capacity.values["lorrax.collision_projection_order"] == 800
    assert capacity.values["lorrax.collision_pencil_mb"] == 256
    reference = parse_td_deck(_bundle(collision="raw", collision_screening="joint",
        collision_basis="instantaneous", collision_hamiltonian="reference"))
    assert reference.values["lorrax.collision_hamiltonian"] == "reference"
    assert _refuses(_bundle(collision="raw", collision_screening="joint",
        collision_basis="instantaneous", collision_hermitian="none"), "td_collision_snapshot")
    assert _refuses(_bundle(collision="raw", collision_screening="joint",
        collision_basis="instantaneous", stiff_modes=1), "td_collision_snapshot")
    assert _refuses(_bundle(collision="raw", collision_screening="joint"),
                    "td_collision_snapshot")
    for key, value in (("collision_snapshot_steps", 0), ("collision_density_rank", 0),
                       ("collision_fit_samples", 3), ("collision_snapshot_mixing", 1.1),
                       ("collision_projection_order", 0), ("collision_projection_order", True),
                       ("collision_pencil_mb", 0), ("collision_pencil_mb", 128.5),
                       ("collision_adiabatic_ev", 0), ("collision_fit_height_ev", float("inf")),
                       ("collision_hamiltonian", "screened")):
        assert _refuses(_bundle(collision="raw", **{key: value}), "td_collision_snapshot")


def test_population_and_symmetric_comparisons_are_not_propagation_modes():
    assert _refuses(_bundle(collision="raw", collision_in_scattering="populations"),
                    "td_collision_mgkba_closure")
    assert _refuses(_bundle(collision="raw", collision_in_scattering="populations",
                            collision_basis="band", collision_internal_pairing="symmetric"),
                    "td_collision_mgkba_closure")
    assert _refuses(_bundle(collision="raw", collision_basis="band",
                            collision_internal_pairing="symmetric"),
                    "td_collision_mgkba_closure")


def test_collision_refuses_another_source_and_a_restart():
    assert _refuses(_bundle(collision="raw", collision_source="other/gw.in"), "td_collision_source")
    raw = _bundle(collision="raw")
    raw["control"]["files"]["load_checkpoint_from"] = "an_earlier_run"
    assert _refuses(raw, "td_collision_restart")
    assert not _refuses(_bundle(collision="raw", collision_source="gw/oneshot.in"),
                        "td_collision_source")


def test_collision_screening_and_frozen_bands():
    on = dict(collision="subtracted", collision_screening="refreshed",
              collision_refresh_sweep="active")
    assert _refuses(_bundle(**on, collision_frozen_bands=[4, 8]),
                    "td_collision_keldysh_screening")
    assert parse_td_deck(_bundle(collision="subtracted")).values["lorrax.collision_screening"] == "frozen"
    # The column-sampled operator admits only the equilibrium W branches until the
    # independent nonequilibrium W< and joint response/noise bank exist.
    assert _refuses(_bundle(**on), "td_collision_keldysh_screening")
    assert _refuses(_bundle(collision="subtracted", collision_frozen_bands=[4, 8]), "td_collision")
    from tdagw.td_config import inactive_keys
    frozen = parse_td_deck(_bundle(collision="subtracted", collision_screening_gamma_ev=0.01))
    assert "lorrax.collision_screening_gamma_ev" in inactive_keys(frozen)


def test_collision_keys_inactive_when_off_and_fingerprint():
    from types import SimpleNamespace
    from tdagw.stages import restart_fingerprint
    from tdagw.td_config import inactive_keys
    off = parse_td_deck(_bundle(collision_hermitian="full", collision_refresh_fs=1.0))
    assert set(inactive_keys(off)) >= {"lorrax.collision_hermitian", "lorrax.collision_refresh_fs"}
    on = parse_td_deck(_bundle(collision="subtracted"))
    assert "lorrax.collision_hermitian" not in inactive_keys(on)
    sysm = SimpleNamespace(nk=9, nb=4, nspinor=2)
    model = SimpleNamespace(name="m")
    fp_off = json.loads(restart_fingerprint(off, sysm, model, 0.1))
    fp_on = json.loads(restart_fingerprint(on, sysm, model, 0.1))
    assert "collision" not in fp_off and fp_on["collision"]["lorrax.collision"] == "subtracted"
    assert fp_on["collision"]["lorrax.collision_projection_order"] == 400
    assert fp_on["collision"]["lorrax.collision_pencil_mb"] == 128


def test_joint_capacity_controls_forward_from_deck_to_runtime(monkeypatch):
    from types import SimpleNamespace as NS
    from tdagw import stages, snapshot_runtime

    deck = parse_td_deck(_bundle(collision="raw", collision_screening="joint",
        collision_basis="instantaneous", collision_projection_order=800, collision_pencil_mb=256))
    captured = {}

    def snapshot(*args, options, **kwargs):
        captured.update(options)
        raise RuntimeError("capacity forwarding checked")

    monkeypatch.setattr(snapshot_runtime, "snapshot_collision", snapshot)
    system = NS(ops=None, hdr={"kpts": None}, bands=[1])
    source = dict(config=NS(sigma=NS(regularization_ev=0.1)))
    with pytest.raises(RuntimeError, match="capacity forwarding checked"):
        stages._attach_collision(deck, system, None, None, source, None,
                                 say=lambda _: None, log=None)
    assert captured["max_projection_order"] == 800
    assert captured["max_pencil_bytes"] == 256 * 1024 ** 2
