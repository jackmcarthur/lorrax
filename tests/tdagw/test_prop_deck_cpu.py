"""The TD-aGW deck parser: TDGW's ``tdgw.toml`` plus ``[lorrax]`` (no jax, ms).

G1 the fixture's TDGW deck, with load_header_from and a [lorrax] table,
   parses; times convert as -> a.u. and energies eV -> Ha; a TOML file of
   the same deck parses to the same values
G2 every refusal names its GATE id (one case per gate of parse_td_deck);
   magnus2 refuses a dt that does not divide dt_measure (td_dense_output)
   while cfm4 accepts it, and a restart into a new outdir parses
R1 the unmutated deck is refused by none of G2's checks, so each refusal
   is caused by its mutation and not by the base deck
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
