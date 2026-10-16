"""tdagw/td_config.py — the TD-aGW deck: TDGW's ``tdgw.toml`` plus a ``[lorrax]`` table.

A TDGW deck (TDGW README "Variables of TDGW config file") is accepted as is;
the ``[lorrax]`` table selects the integrator, the step, the self-energy
model and batched drives.  ``TD_DEFAULTS`` is the accepted key set (dotted
TOML paths; ``measurements.<i>.<keyword>`` rows are checked against
``MEASUREMENT_KEYWORDS``); ``docs/input_reference.md`` is generated from it
(``tools/gen_input_reference.py``).  Every unknown key refuses in one
error naming all of them; a ``None`` default marks a required key.

Units are TDGW's: times in attoseconds, energies in eV, fields in V/m; the
parsed :class:`TdDeck` holds Hartree atomic units.
"""
from __future__ import annotations

import math
import os
from typing import NamedTuple

__all__ = ["TD_DEFAULTS", "TD_REQUIRED", "MEASUREMENT_KEYWORDS",
           "MEASUREMENT_TYPES", "TdDeck", "read_td_deck", "parse_td_deck"]

#: The accepted keys and their defaults.  None = required or unset.
TD_DEFAULTS = {
    "control.loglevel": "info",
    "control.calc_type": None,
    "control.timesteps.t0": None,
    "control.timesteps.dt_measure": None,
    "control.timesteps.tf": None,
    "control.timesteps.ntcheck": 5000,
    "control.timesteps.max_step": None,
    "control.ivp.rtol": 1e-6,
    "control.ivp.atol": 1e-12,
    "control.files.indir": None,
    "control.files.outdir": None,
    "control.files.use_vmtxel": False,
    "control.files.load_header_from": "",
    "control.files.load_checkpoint_from": "",
    "system.ibndmin": None,
    "system.ibndmax": None,
    "system.ivbm": None,
    "system.geometry": None,
    "system.center_of_vacuum": None,
    "system.esmear": 0.05,
    "system.depopulation": 0.0,
    "system.dephasing": 0.0,
    "system.patched_sampling": False,
    "system.Efunc.E_x": 0.0,
    "system.Efunc.E_y": 0.0,
    "system.Efunc.E_z": 0.0,
    "measurements.types": None,
    "lorrax.method": "dp54",
    "lorrax.dt": None,
    "lorrax.n_corr": 2,
    "lorrax.model": "",
    "lorrax.kernel_file": "",
    "lorrax.isdf_input": "",
    "lorrax.isdf_zeta": "",
    "lorrax.isdf_terms": "both",
    "lorrax.intraband": True,
    "lorrax.chunk": 0,
    "lorrax.trials": None,
}
#: Keys without a usable default.
TD_REQUIRED = ("control.calc_type", "control.timesteps.t0",
               "control.timesteps.dt_measure", "control.timesteps.tf",
               "control.files.indir", "control.files.outdir", "system.ibndmin",
               "system.ibndmax", "system.geometry", "measurements.types")
MEASUREMENT_TYPES = ("Efield", "polarization", "current", "density_matrix",
                     "lessG")
MEASUREMENT_KEYWORDS = ("dataset_name", "itrange", "ntsave", "brange1",
                        "brange2", "krange", "use_v0", "inter_only")
METHODS = ("dp54", "rk4", "etdrk4", "magnus2", "cfm4")
MODELS = ("none", "tdgw_kernel", "isdf")


class TdDeck(NamedTuple):
    """A parsed deck, Hartree atomic units.

    tdgw : the TDGW sections as read (for ``tdgw.toml`` beside the output).
    values : {dotted key: value} for every key in TD_DEFAULTS, converted.
    measurements : tuple of (dset_name, type, kwargs dict).
    trials : tuple of Efunc dicts (one per batched drive).
    source : the deck path.
    """
    tdgw: dict
    values: dict
    measurements: tuple
    trials: tuple
    source: str


def _gate(gid, got, want, fix):
    return ValueError(f"GATE {gid}: got: {got}; want: {want}; fix: {fix}")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key not in ("lorrax.trials",):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def read_td_deck(path: str) -> TdDeck:
    """Read and validate a TD-aGW deck (TOML)."""
    import tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse_td_deck(raw, source=os.path.abspath(path))


def parse_td_deck(raw: dict, *, source: str = "<dict>") -> TdDeck:
    """Validate ``raw`` (a parsed TOML table) into a :class:`TdDeck`."""
    from tdagw.tdgw_io import HA_IN_EV, T_IN_ATTOSEC

    meas_raw = dict(raw.get("measurements", {}))
    flat = _flatten({k: v for k, v in raw.items() if k != "measurements"})
    if "types" in meas_raw:
        flat["measurements.types"] = meas_raw.pop("types")
    unknown = sorted(k for k in flat if k not in TD_DEFAULTS)
    per_meas = {}
    for k, v in meas_raw.items():
        if not (str(k).isdigit() and isinstance(v, dict)):
            unknown.append(f"measurements.{k}")
            continue
        bad = sorted(set(v) - set(MEASUREMENT_KEYWORDS))
        unknown += [f"measurements.{k}.{b}" for b in bad]
        per_meas[int(k)] = v
    if unknown:
        raise _gate("td_unknown_key", ", ".join(unknown),
                    "keys of TD_DEFAULTS / MEASUREMENT_KEYWORDS",
                    "remove or correct them (docs/input_reference.md, TD-aGW)")
    missing = [k for k in TD_REQUIRED if flat.get(k) is None]
    if missing:
        raise _gate("td_missing_key", ", ".join(missing), "every required key",
                    "set them in the deck")
    vals = {k: flat.get(k, d) for k, d in TD_DEFAULTS.items()}

    if vals["control.calc_type"] not in ("IP", "COHSEX"):
        raise _gate("td_calc_type", repr(vals["control.calc_type"]),
                    "IP | COHSEX", "set [control] calc_type")
    if vals["control.files.use_vmtxel"]:
        raise _gate("td_header_source", "use_vmtxel = true",
                    "the header (structure, WFN header, dipole) of a TDGW run",
                    "set [control.files] load_header_from to a TDGW output")
    if not vals["control.files.load_header_from"]:
        raise _gate("td_header_source", "load_header_from empty",
                    "a TDGW output directory holding tdgw.h5",
                    "point [control.files] load_header_from at a TDGW run "
                    "(tdgw-run writes the header before it propagates)")
    if vals["control.files.load_checkpoint_from"]:
        raise _gate("td_checkpoint_restart", "load_checkpoint_from set",
                    "a run from t0", "restart is not implemented in LORRAX's "
                    "propagator; rerun from t0")
    if vals["system.patched_sampling"]:
        raise _gate("td_patched_sampling", "patched_sampling = true",
                    "a regular k grid", "use an unpatched header")
    if vals["system.geometry"] not in ("bulk", "slab", "wire"):
        raise _gate("td_geometry", repr(vals["system.geometry"]),
                    "bulk | slab | wire", "box geometry is not supported")
    if vals["lorrax.method"] not in METHODS:
        raise _gate("td_method", repr(vals["lorrax.method"]), " | ".join(METHODS),
                    "set [lorrax] method")
    if int(vals["lorrax.n_corr"]) < 1:
        raise _gate("td_n_corr", vals["lorrax.n_corr"], ">= 1",
                    "a Magnus predictor without a correction is unstable")
    model = vals["lorrax.model"] or (
        "none" if vals["control.calc_type"] == "IP" else "tdgw_kernel")
    if model not in MODELS:
        raise _gate("td_model", repr(model), " | ".join(MODELS),
                    "set [lorrax] model")
    if model == "isdf" and not vals["lorrax.isdf_input"]:
        raise _gate("td_isdf_input", "model = isdf without [lorrax] isdf_input",
                    "the GW deck of the restart bundle", "set isdf_input")
    if vals["lorrax.isdf_terms"] not in ("both", "sex", "hartree"):
        raise _gate("td_isdf_terms", repr(vals["lorrax.isdf_terms"]),
                    "both | sex | hartree", "set [lorrax] isdf_terms")
    vals["lorrax.model"] = model

    # units: as -> a.u., eV -> Ha
    for k in ("control.timesteps.t0", "control.timesteps.dt_measure",
              "control.timesteps.tf"):
        vals[k] = float(vals[k]) / T_IN_ATTOSEC
    ms = vals["control.timesteps.max_step"]
    vals["control.timesteps.max_step"] = (math.inf if ms is None
                                          else float(ms) / T_IN_ATTOSEC)
    for k in ("system.esmear", "system.depopulation", "system.dephasing"):
        vals[k] = float(vals[k]) / HA_IN_EV
    if vals["lorrax.method"] != "dp54":
        if vals["lorrax.dt"] is None:
            raise _gate("td_dt", "no [lorrax] dt for a fixed-step method",
                        "dt in attoseconds", "set [lorrax] dt")
        dt = float(vals["lorrax.dt"]) / T_IN_ATTOSEC
        ratio = vals["control.timesteps.dt_measure"] / dt
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio) or round(ratio) < 1:
            raise _gate("td_dt_measure_multiple",
                        f"dt_measure/dt = {ratio:.12g}",
                        "a positive integer (samples land on steps)",
                        "choose dt = dt_measure / n")
        vals["lorrax.dt"] = dt
        vals["lorrax.n_sub"] = int(round(ratio))
    else:
        vals["lorrax.n_sub"] = 1
        if vals["lorrax.dt"] is not None:
            raise _gate("td_dt", "[lorrax] dt with method = dp54",
                        "no dt (the controller chooses the steps)",
                        "remove dt or pick a fixed-step method")

    types = list(vals["measurements.types"])
    meas = []
    for i, typ in enumerate(types):
        if typ not in MEASUREMENT_TYPES:
            raise _gate("td_measurement_unsupported", repr(typ),
                        " | ".join(MEASUREMENT_TYPES),
                        "drop it (velocity_operator is TDGW-only)")
        kw = dict(per_meas.get(i, {}))
        name = kw.pop("dataset_name", typ)
        meas.append((name, typ, kw))
    names = [m[0] for m in meas]
    if len(set(names)) != len(names):
        raise _gate("td_measurement_names", names, "unique dataset names",
                    "set measurements.<i>.dataset_name")
    efunc = raw.get("system", {}).get("Efunc", {})
    trials = tuple(dict(t) for t in (vals["lorrax.trials"] or [efunc]))
    for t in trials:
        bad = sorted(set(t) - {"E_x", "E_y", "E_z"})
        if bad:
            raise _gate("td_unknown_key", ", ".join(f"lorrax.trials.{b}" for b in bad),
                        "E_x, E_y, E_z", "fix the trial table")
    tdgw = {k: v for k, v in raw.items() if k != "lorrax"}
    return TdDeck(tdgw=tdgw, values=vals, measurements=tuple(meas),
                  trials=trials, source=source)
