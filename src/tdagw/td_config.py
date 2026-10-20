"""tdagw/td_config.py — the TD-aGW deck: TDGW's ``tdgw.toml`` plus a ``[lorrax]`` table.

A TDGW deck (TDGW README "Variables of TDGW config file") is accepted as is;
the ``[lorrax]`` table selects the integrator, the step, the self-energy
model and batched drives.  ``TD_DEFAULTS`` is the accepted key set (dotted
TOML paths; ``measurements.<i>.<keyword>`` rows are checked against
``MEASUREMENT_KEYWORDS``) and ``TD_KEY_DOCS`` says what each key means, its
default's reading and the refusals that name it; the TD-aGW table of
``docs/input_reference.md`` is generated from the two
(``tools/gen_input_reference.py``; edit them, never the table).  Every
unknown key refuses in one error naming all of them; a ``None`` default
marks a required key.  :func:`resolved_deck_rows` is the run log's record of
the deck: every key, its value and whether the deck set it.

Units are TDGW's: times in attoseconds, energies in eV, fields in V/m; the
parsed :class:`TdDeck` holds Hartree atomic units.
"""
from __future__ import annotations

import math
import os
from typing import NamedTuple

__all__ = ["TD_DEFAULTS", "TD_REQUIRED", "TD_KEY_DOCS", "MEASUREMENT_KEYWORDS",
           "MEASUREMENT_TYPES", "SYSTEMS", "RESERVED_SAMPLE_PREFIX", "TdDeck", "read_td_deck",
           "parse_td_deck", "UNUSED_TDGW_KEYS", "unused_keys", "inactive_keys",
           "resolved_deck_rows"]

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
    "lorrax.predictor": "cubic",
    "lorrax.magnus_frame": "",
    "lorrax.model": "",
    "lorrax.kernel_file": "",
    "lorrax.isdf_input": "",
    "lorrax.isdf_zeta": "",
    "lorrax.isdf_terms": "both",
    "lorrax.intraband": True,
    "lorrax.chunk": 0,
    "lorrax.trials": None,
    "lorrax.system": "tdgw_header",
    "lorrax.bundle_input": "",
    "lorrax.eqp_file": "",
    "lorrax.dipole_file": "",
    "lorrax.pair_rule": "mask",
    "lorrax.pair_gap_ev": None,
    "lorrax.check_eigenvectors": "",
    "lorrax.degeneracy_mode": "",
    "lorrax.tda": False,
    "lorrax.stiff_modes": 0,
    "lorrax.stiff_krylov": 0,
    "lorrax.stiff_tol": 1e-9,
    "lorrax.stiff_iter": 6,
    "lorrax.collision": "off",
    "lorrax.collision_source": "",
    "lorrax.collision_hermitian": "dynamic",
    "lorrax.collision_refresh_fs": 0.0,
    "lorrax.collision_refresh_df": 0.0,
    "lorrax.collision_refresh_sweep": "active",
    "lorrax.collision_screening": "frozen",
    "lorrax.collision_statistics": "unspecified",
    "lorrax.collision_hamiltonian": "hf",
    "lorrax.collision_frozen_bands": None,
    "lorrax.collision_screening_gamma_ev": 0.001,
    "lorrax.collision_rebase": "auto",
    "lorrax.collision_spm_support_sites_ev": "",
    "lorrax.collision_spm_chi0_active": None,
    "lorrax.collision_in_scattering": "full",
    "lorrax.collision_rate_intermediates": "active",
    "lorrax.collision_basis": "band",
    "lorrax.collision_internal_pairing": "vertex",
    "lorrax.collision_snapshot_steps": 32,
    "lorrax.collision_snapshot_tol_ev": 1e-4,
    "lorrax.collision_snapshot_mixing": 0.7,
    "lorrax.collision_snapshot_acceleration": "anderson",
    "lorrax.collision_density_rank": 8,
    "lorrax.collision_pole_order": 48,
    "lorrax.collision_projection_order": 400,
    "lorrax.collision_pencil_mb": 128,
    "lorrax.collision_fit_samples": 192,
    "lorrax.collision_fit_real_axis": False,
    "lorrax.collision_fit_tolerance": 0.02,
    "lorrax.collision_adiabatic_ev": 0.05,
    "lorrax.collision_fit_height_ev": 0.15,
}
#: Keys without a usable default.
TD_REQUIRED = ("control.calc_type", "control.timesteps.t0",
               "control.timesteps.dt_measure", "control.timesteps.tf",
               "control.files.outdir", "system.ibndmin",
               "system.ibndmax", "system.geometry", "measurements.types")
#: TDGW's measurement workers, then two of LORRAX's own (not in TDGW):
#: ``carrier_density`` (conduction electrons and valence holes per cell) and
#: ``exciton_projection`` (the amplitudes ``Σ conj(A^S) ρ_cv`` on the states
#: of a BSE eigenvectors file).
MEASUREMENT_TYPES = ("Efield", "polarization", "current", "density_matrix",
                     "lessG", "carrier_density", "exciton_projection")
MEASUREMENT_KEYWORDS = ("dataset_name", "itrange", "ntsave", "brange1",
                        "brange2", "krange", "use_v0", "inter_only",
                        "eigenvectors", "n_states")
SYSTEMS = ("tdgw_header", "bundle")
#: ``lorrax.collision``: off, or the form of the on-shell collision term (§1.5).
COLLISION_FORMS = ("off", "subtracted", "raw")
#: ``lorrax.collision_hermitian``: the held Hermitian part (``tdagw.collision``).
COLLISION_HERMITIAN = ("dynamic", "full", "none")
#: ``lorrax.collision_refresh_sweep``: what a refresh sweeps (``tdagw.dynsig.REFRESH_SWEEPS``).
COLLISION_REFRESH_SWEEPS = ("full", "active")
#: ``lorrax.collision_screening``: the W of the rates (``tdagw.dynsig.COLLISION_SCREENINGS``).
COLLISION_SCREENINGS = ("frozen", "refreshed", "shared_pole", "joint", "joint_frozen")
#: ``lorrax.collision_rebase``: C[rho_eq] on the refreshed W (``auto``: on for shared_pole).
COLLISION_REBASE = ("auto", "on", "off")
#: ``lorrax.collision_in_scattering``: the rates' internal line (``tdagw.dynsig.IN_SCATTERINGS``).
COLLISION_IN_SCATTERINGS = ("populations", "full")
#: ``lorrax.collision_rate_intermediates``: the rates' intermediate bands
#: (``tdagw.dynsig.RATE_INTERMEDIATES``).
COLLISION_RATE_INTERMEDIATES = ("all", "active")
#: ``lorrax.collision_basis``: the lines' basis and energies (``tdagw.dynsig.COLLISION_BASES``).
COLLISION_BASES = ("band", "instantaneous")
#: ``lorrax.collision_internal_pairing``: the internal line's energy pairing
#: (``tdagw.dynsig.INTERNAL_PAIRINGS``).
COLLISION_INTERNAL_PAIRINGS = ("symmetric", "vertex")
METHODS = ("dp54", "rk4", "etdrk4", "magnus2", "cfm4")
MODELS = ("none", "tdgw_kernel", "isdf")
#: Sample names the driver keeps for its own per-chunk record (``tdagw.propagate.PROGRESS``);
#: a measurement ``dataset_name`` may not start with it (``td_measurement_names``).
RESERVED_SAMPLE_PREFIX = "lorrax."


class TdDeck(NamedTuple):
    """A parsed deck, Hartree atomic units.

    tdgw : the TDGW sections as read (for ``tdgw.toml`` beside the output).
    values : {dotted key: value} for every key in TD_DEFAULTS, converted.
    measurements : tuple of (dset_name, type, kwargs dict).
    trials : tuple of Efunc dicts (one per batched drive).
    source : the deck path.
    given : {dotted key: value as written} for the TD_DEFAULTS keys the deck sets
        (TDGW units, before conversion; the run log's resolved deck).
    """
    tdgw: dict
    values: dict
    measurements: tuple
    trials: tuple
    source: str
    given: dict = None


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


def _check_collision(vals: dict) -> None:
    """The ``lorrax.collision*`` keys (§1.5): forms, the source's gauge, the refresh policy."""
    form = vals["lorrax.collision"]
    if form not in COLLISION_FORMS:
        raise _gate("td_collision", repr(form), " | ".join(COLLISION_FORMS),
                    "set [lorrax] collision")
    if vals["lorrax.collision_hermitian"] not in COLLISION_HERMITIAN:
        raise _gate("td_collision", f"collision_hermitian = "
                    f"{vals['lorrax.collision_hermitian']!r}", " | ".join(COLLISION_HERMITIAN),
                    "set [lorrax] collision_hermitian")
    if vals["lorrax.collision_refresh_sweep"] not in COLLISION_REFRESH_SWEEPS:
        raise _gate("td_collision", f"collision_refresh_sweep = "
                    f"{vals['lorrax.collision_refresh_sweep']!r}",
                    " | ".join(COLLISION_REFRESH_SWEEPS), "set [lorrax] collision_refresh_sweep")
    if vals["lorrax.collision_screening"] not in COLLISION_SCREENINGS:
        raise _gate("td_collision", f"collision_screening = "
                    f"{vals['lorrax.collision_screening']!r}", " | ".join(COLLISION_SCREENINGS),
                    "set [lorrax] collision_screening")
    if vals["lorrax.collision_hamiltonian"] not in ("hf", "reference"):
        raise _gate("td_collision_snapshot", "unknown collision Hamiltonian",
                    "hf | reference", "set collision_hamiltonian")
    statistics = vals["lorrax.collision_statistics"]
    if statistics not in ("unspecified", "finite_lag", "lossless_bath", "retarded_reference"):
        raise _gate("td_collision_statistics", f"collision_statistics = {statistics!r}",
                    "unspecified | finite_lag | lossless_bath | retarded_reference",
                    "set [lorrax] collision_statistics")
    if (form != "off" and vals["lorrax.collision_screening"] in ("joint", "joint_frozen")
            and statistics == "unspecified"):
        raise _gate("td_collision_statistics", "joint screening statistics not explicitly selected",
                    "collision_statistics = 'retarded_reference', 'finite_lag' or 'lossless_bath'",
                    "select the statistical model explicitly; finite_lag is not generally thermal; "
                    "lossless_bath pinches screening coherences and uses a weak colored-bath prescription")
    if (form != "off" and statistics in ("lossless_bath", "retarded_reference")
            and vals["lorrax.collision_screening"] not in ("joint", "joint_frozen")):
        raise _gate("td_collision_statistics", f"{statistics} without joint screening",
                    "collision_screening = 'joint' or 'joint_frozen'",
                    "this statistical selection is a joint snapshot screening prescription")
    if (form != "off" and vals["lorrax.collision_screening"] in ("joint", "joint_frozen")
            and isinstance(vals["lorrax.stiff_modes"], int) and vals["lorrax.stiff_modes"] > 0):
        raise _gate("td_collision_snapshot", "stiff split with a joint snapshot",
                    "stiff_modes = 0",
                    "disable the unvalidated stiff-snapshot combination")
    if (form != "off" and vals["lorrax.collision_screening"] in ("joint", "joint_frozen")
            and vals["lorrax.collision_hermitian"] != "dynamic"):
        raise _gate("td_collision_snapshot", "joint screening without the full coherent correction",
                    "collision_hermitian = 'dynamic'",
                    "include M_H once in the material snapshot")
    if form != "off" and vals["lorrax.collision_screening"] in ("refreshed", "shared_pole"):
        raise _gate(
            "td_collision_keldysh_screening",
            f"collision_screening = {vals['lorrax.collision_screening']!r}",
            "collision_screening = 'frozen', 'joint' or 'joint_frozen'",
            "use the equilibrium W branches; the refreshed and shared-pole paths rebuild only "
            "retarded W from populations and do not supply the nonequilibrium W< required by "
            "the column-sampled MGKBA collision operator",
        )
    for key in ("collision_snapshot_steps", "collision_density_rank", "collision_pole_order",
                "collision_projection_order", "collision_pencil_mb", "collision_fit_samples"):
        value = vals[f"lorrax.{key}"]
        if (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise _gate("td_collision_snapshot", f"{key} = {value!r}",
                        "a positive integer", f"set [lorrax] {key}")
    if vals["lorrax.collision_fit_samples"] < 4:
        raise _gate("td_collision_snapshot", "collision_fit_samples < 4",
                    "at least four distinct supports", "increase collision_fit_samples")
    if not isinstance(vals["lorrax.collision_fit_real_axis"], bool):
        raise _gate("td_collision_snapshot", "collision_fit_real_axis must be boolean",
                    "true | false", "set [lorrax] collision_fit_real_axis")
    for key in ("collision_snapshot_tol_ev", "collision_fit_tolerance", "collision_adiabatic_ev",
                "collision_fit_height_ev", "collision_snapshot_mixing"):
        try:
            value = float(vals[f"lorrax.{key}"])
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0:
            raise _gate("td_collision_snapshot", f"{key} = {value!r}",
                        "a positive finite number", f"set [lorrax] {key}")
        vals[f"lorrax.{key}"] = value
    if vals["lorrax.collision_snapshot_mixing"] > 1.0:
        raise _gate("td_collision_snapshot", "collision_snapshot_mixing > 1",
                    "0 < mixing <= 1", "reduce collision_snapshot_mixing")
    if vals["lorrax.collision_snapshot_acceleration"] not in ("linear", "anderson"):
        raise _gate("td_collision_snapshot", "unknown snapshot acceleration",
                    "linear | anderson", "set collision_snapshot_acceleration")
    if (form != "off" and vals["lorrax.collision_screening"] in ("joint", "joint_frozen")
            and vals["lorrax.collision_rebase"] == "on"):
        raise _gate("td_collision_snapshot", "collision_rebase = 'on' with joint screening",
                    "one fixed equilibrium counterterm or raw collision",
                    "use collision_rebase = 'off' or 'auto'; rebasing removes physical noise")
    fb = vals["lorrax.collision_frozen_bands"]
    if fb is not None:
        ok = (isinstance(fb, (list, tuple)) and len(fb) == 2
              and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in fb))
        if not ok:
            raise _gate("td_collision", f"collision_frozen_bands = {fb!r}",
                        "[n_below, n_above], two integers >= 0",
                        "set [lorrax] collision_frozen_bands or leave it unset")
        if (vals["lorrax.collision_screening"] not in ("refreshed", "shared_pole")
                or vals["lorrax.collision_refresh_sweep"] != "active"):
            raise _gate("td_collision", "collision_frozen_bands with collision_screening = "
                        f"{vals['lorrax.collision_screening']!r}, collision_refresh_sweep = "
                        f"{vals['lorrax.collision_refresh_sweep']!r}",
                        "collision_screening = 'refreshed' or 'shared_pole' and "
                        "collision_refresh_sweep = 'active'",
                        "the truncation is of the frozen part's change under W(t); unset it")
        vals["lorrax.collision_frozen_bands"] = [int(n) for n in fb]
    try:
        ok = float(vals["lorrax.collision_screening_gamma_ev"]) >= 0.0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise _gate("td_collision", f"collision_screening_gamma_ev = "
                    f"{vals['lorrax.collision_screening_gamma_ev']!r}", ">= 0 (eV; 0 = exact, "
                    "degenerate pairs refused)", "set [lorrax] collision_screening_gamma_ev")
    vals["lorrax.collision_screening_gamma_ev"] = float(vals["lorrax.collision_screening_gamma_ev"])
    if vals["lorrax.collision_rebase"] not in COLLISION_REBASE:
        raise _gate("td_collision", f"collision_rebase = {vals['lorrax.collision_rebase']!r}",
                    " | ".join(COLLISION_REBASE), "set [lorrax] collision_rebase")
    if (vals["lorrax.collision_rebase"] == "on"
            and vals["lorrax.collision_screening"] == "frozen"):
        raise _gate("td_collision", "collision_rebase = 'on' with collision_screening = 'frozen'",
                    "a refreshed W (refreshed or shared_pole)", "the t0 rates are rho_eq's at W0; "
                    "unset collision_rebase")
    sites = str(vals["lorrax.collision_spm_support_sites_ev"])
    if sites.strip():
        from gw.shared_pole_recipe import parse_support_sites
        try:
            vals["lorrax.collision_spm_support_sites_ev"] = parse_support_sites(sites)["text"]
        except ValueError as exc:
            raise _gate("td_collision", f"collision_spm_support_sites_ev = {sites!r}",
                        "'<line eV list> | <imaginary eV list> [| <line height eV>]'",
                        str(exc)) from None
    ca = vals["lorrax.collision_spm_chi0_active"]
    if ca is not None and ca != "auto":
        ok = (isinstance(ca, (list, tuple)) and len(ca) == 2
              and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in ca)
              and ca[0] < ca[1])
        if not ok:
            raise _gate("td_collision", f"collision_spm_chi0_active = {ca!r}",
                        "'auto' or [a0, a1], two integers 0 <= a0 < a1 (0-based W-carrier bands)",
                        "set [lorrax] collision_spm_chi0_active or leave it unset")
        vals["lorrax.collision_spm_chi0_active"] = [int(n) for n in ca]
    for key, allowed in (("lorrax.collision_in_scattering", COLLISION_IN_SCATTERINGS),
                         ("lorrax.collision_rate_intermediates", COLLISION_RATE_INTERMEDIATES),
                         ("lorrax.collision_basis", COLLISION_BASES),
                         ("lorrax.collision_internal_pairing", COLLISION_INTERNAL_PAIRINGS)):
        if vals[key] not in allowed:
            raise _gate("td_collision", f"{key.split('.')[1]} = {vals[key]!r}", " | ".join(allowed),
                        f"set [lorrax] {key.split('.')[1]}")
        if (vals[key] != allowed[0]
                and vals["lorrax.collision_refresh_sweep"] != "active"):
            raise _gate("td_collision", f"{key.split('.')[1]} = {vals[key]!r} with "
                        f"collision_refresh_sweep = {vals['lorrax.collision_refresh_sweep']!r}",
                        "collision_refresh_sweep = 'active'",
                        "it acts on the active part of the split sweep")
    if form != "off":
        if (vals["lorrax.collision_screening"] in ("joint", "joint_frozen")
                and vals["lorrax.collision_basis"] != "instantaneous"):
            raise _gate("td_collision_snapshot", "joint screening with band basis",
                        "collision_basis = 'instantaneous'",
                        "use one converged material snapshot for screening and collision")
        required = {
            "lorrax.collision_refresh_sweep": "active",
            "lorrax.collision_in_scattering": "full",
            "lorrax.collision_rate_intermediates": "active",
            "lorrax.collision_internal_pairing": "vertex",
        }
        wrong = [f"{key.split('.')[1]} = {vals[key]!r}" for key, want in required.items()
                 if vals[key] != want]
        if wrong:
            raise _gate(
                "td_collision_mgkba_closure", ", ".join(wrong),
                ", ".join(f"{key.split('.')[1]} = {want!r}" for key, want in required.items()),
                "the column-sampled full-matrix MGKBA operator conserves the closed active "
                "window only with the full-rho vertex line; the populations, all-band-rate, "
                "and symmetric-pairing paths are comparison kernels, not propagation modes",
            )
    if (form != "off" and vals["lorrax.collision_basis"] == "instantaneous"
            and vals["lorrax.collision_screening"] not in ("frozen", "joint", "joint_frozen")):
        raise _gate("td_collision", "collision_basis = 'instantaneous' with collision_screening "
                    f"= {vals['lorrax.collision_screening']!r}", "collision_screening = 'frozen'",
                    "the slow Hamiltonian's basis is defined at fixed W (td_agw_dynamical.md §1.5)")
    for key in ("lorrax.collision_refresh_fs", "lorrax.collision_refresh_df"):
        try:
            ok = float(vals[key]) >= 0.0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise _gate("td_collision_refresh", f"{key} = {vals[key]!r}",
                        ">= 0 (0 = that trigger off)", f"set [lorrax] {key.split('.')[1]}")
        vals[key] = float(vals[key])
    if form == "off":
        return
    if vals["lorrax.system"] != "bundle":
        raise _gate("td_collision_source", "collision with system = 'tdgw_header'",
                    "system = 'bundle' (the rates are measured on the bundle's own "
                    "wavefunctions, the header's gauge)", "build the system from the bundle")
    if (vals["lorrax.collision_source"] not in ("", vals["lorrax.bundle_input"])
            and vals["lorrax.collision_screening"] not in ("shared_pole", "joint", "joint_frozen")):
        raise _gate("td_collision_source", "collision_source differs from bundle_input",
                    "the GN-PPM GW deck of the bundle (one wavefunction gauge); a separate "
                    "source only with collision_screening = 'shared_pole'",
                    "leave collision_source unset")
    if not vals["lorrax.collision_source"]:
        vals["lorrax.collision_source"] = vals["lorrax.bundle_input"]
    if vals["control.files.load_checkpoint_from"]:
        raise _gate("td_collision_restart", "load_checkpoint_from with collision",
                    "a fresh run (the held rates and Delta Sigma^h are not in the carry)",
                    "rerun from t0, or set collision = 'off'")


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
                    "the header (structure, WFN header, dipole) of a TDGW run "
                    "(a use_vmtxel run's header carries BerkeleyGW's dipoles) "
                    "or [lorrax] system = 'bundle'",
                    "set [control.files] load_header_from to a TDGW output")
    if vals["lorrax.system"] not in SYSTEMS:
        raise _gate("td_system", repr(vals["lorrax.system"]), " | ".join(SYSTEMS),
                    "set [lorrax] system")
    if vals["lorrax.system"] == "bundle":
        if vals["control.files.load_header_from"]:
            raise _gate("td_header_source", "load_header_from with system = 'bundle'",
                        "one header source", "drop load_header_from (the bundle "
                        "builds the header) or use system = 'tdgw_header'")
        for key in ("lorrax.bundle_input", "lorrax.dipole_file"):
            if not vals[key]:
                raise _gate("td_bundle_input", f"system = 'bundle' without {key}",
                            "the GW deck of the restart and the coarse WFN's dipole.h5",
                            f"set {key}")
        if vals["system.ivbm"] is None:
            raise _gate("td_ivbm", "system = 'bundle' without [system] ivbm",
                        "the number of occupied bands (1-based VBM index)", "set ivbm")
    else:
        for key in ("lorrax.bundle_input", "lorrax.eqp_file", "lorrax.dipole_file",
                    "lorrax.check_eigenvectors", "lorrax.degeneracy_mode"):
            if vals[key]:
                raise _gate("td_bundle_key", f"{key} with system = 'tdgw_header'",
                            "bundle keys only with system = 'bundle'",
                            "drop it or set [lorrax] system = 'bundle'")
        if not vals["control.files.load_header_from"]:
            raise _gate("td_header_source", "load_header_from empty",
                        "a TDGW output directory holding tdgw.h5",
                        "point [control.files] load_header_from at a TDGW run "
                        "(tdgw-run writes the header before it propagates), or "
                        "build the system from a GW bundle: [lorrax] system = 'bundle'")
    from tdagw.td_system import PAIR_RULES
    if vals["lorrax.degeneracy_mode"] not in ("", "strict", "snap", "off"):
        raise _gate("td_degeneracy_mode", repr(vals["lorrax.degeneracy_mode"]),
                    "strict | snap | off (unset = the band-window default)",
                    "set [lorrax] degeneracy_mode")
    if vals["lorrax.pair_rule"] not in PAIR_RULES:
        raise _gate("td_pair_rule", repr(vals["lorrax.pair_rule"]),
                    " | ".join(PAIR_RULES), "set [lorrax] pair_rule")
    chk = vals["control.files.load_checkpoint_from"]
    if chk and os.path.abspath(chk) == os.path.abspath(str(vals["control.files.outdir"])):
        raise _gate("td_checkpoint_outdir", f"load_checkpoint_from = outdir = {chk!r}",
                    "a new outdir (TDGW's rule: the restart writes a new tdgw.h5)",
                    "set [control.files] outdir to another directory")
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
        raise _gate("td_n_corr", vals["lorrax.n_corr"], ">= 1 (with no correction the node "
                    "history holds only predicted values and the step does not converge: "
                    "order 0)", "set [lorrax] n_corr >= 1 (2 is the default)")
    n_stiff = vals["lorrax.stiff_modes"]
    if not isinstance(n_stiff, int) or isinstance(n_stiff, bool) or n_stiff < 0:
        raise _gate("td_stiff_modes", repr(n_stiff), "an integer >= 0 (0: no split)",
                    "set [lorrax] stiff_modes")
    if n_stiff and vals["lorrax.method"] != "cfm4":
        raise _gate("td_stiff_method", f"stiff_modes = {n_stiff} with method "
                    f"{vals['lorrax.method']!r}", "cfm4 (the split solves dSigma's stiff part "
                    "implicitly at the two Gauss nodes of its predictor-corrector)",
                    "drop stiff_modes or use method = 'cfm4'")
    kry = vals["lorrax.stiff_krylov"]
    if not isinstance(kry, int) or isinstance(kry, bool) or kry < 0 or (
            n_stiff and kry and kry <= n_stiff):
        raise _gate("td_stiff_krylov", repr(kry), "0 (automatic) or an integer above "
                    "stiff_modes", "set [lorrax] stiff_krylov")
    if not float(vals["lorrax.stiff_tol"]) > 0.0 or int(vals["lorrax.stiff_iter"]) < 1:
        raise _gate("td_stiff_solve", f"stiff_tol = {vals['lorrax.stiff_tol']!r}, stiff_iter = "
                    f"{vals['lorrax.stiff_iter']!r}", "stiff_tol > 0 and stiff_iter >= 1",
                    "set [lorrax] stiff_tol / stiff_iter")
    if vals["lorrax.predictor"] not in ("cubic", "linear") or (
            vals["lorrax.predictor"] == "linear" and vals["lorrax.method"] != "cfm4"):
        raise _gate("td_predictor", f"predictor = {vals['lorrax.predictor']!r} with "
                    f"method {vals['lorrax.method']!r}",
                    "cubic | linear (linear: cfm4 only)", "set [lorrax] predictor")
    frame = vals["lorrax.magnus_frame"]
    if frame not in ("", "whole_rho", "interaction") or (
            frame and vals["lorrax.method"] not in ("magnus2", "cfm4")) or (
            frame == "whole_rho" and vals["lorrax.method"] != "magnus2"):
        raise _gate("td_magnus_frame", f"magnus_frame = {frame!r} with method "
                    f"{vals['lorrax.method']!r}", "'' | whole_rho | interaction, for magnus2; "
                    "cfm4 runs on Delta rho in the interaction picture only (its order 4 with "
                    "Gamma and the intraband term, and no rounding floor)",
                    "drop [lorrax] magnus_frame or use it with magnus2")
    model = vals["lorrax.model"] or (
        "none" if vals["control.calc_type"] == "IP" else "tdgw_kernel")
    if model not in MODELS:
        raise _gate("td_model", repr(model), " | ".join(MODELS),
                    "set [lorrax] model")
    bundle = vals["lorrax.system"] == "bundle"
    if model == "isdf" and bundle:
        if vals["lorrax.isdf_input"] not in ("", vals["lorrax.bundle_input"]):
            raise _gate("td_isdf_input", "isdf_input differs from bundle_input",
                        "one bundle for the system and the model",
                        "drop isdf_input (the bundle's model is used)")
        vals["lorrax.isdf_input"] = vals["lorrax.bundle_input"]
    if model == "isdf" and not vals["lorrax.isdf_input"]:
        raise _gate("td_isdf_input", "model = isdf without [lorrax] isdf_input",
                    "the GW deck of the restart bundle", "set isdf_input")
    if model == "isdf" and not bundle and not vals["control.files.indir"]:
        raise _gate("td_indir", "model = isdf on a TDGW header without [control.files] indir",
                    "TDGW's indir (its WFN.h5 fixes the ψ gauge of the header)",
                    "set indir, or build the system from the bundle: system = 'bundle'")
    if model == "tdgw_kernel" and not (vals["lorrax.kernel_file"]
                                       or vals["control.files.indir"]):
        raise _gate("td_kernel_file", "model = tdgw_kernel without kernel_file or indir",
                    "a TDGW kernel.h5", "set [lorrax] kernel_file or [control.files] indir")
    if vals["lorrax.isdf_terms"] not in ("both", "sex", "hartree"):
        raise _gate("td_isdf_terms", repr(vals["lorrax.isdf_terms"]),
                    "both | sex | hartree", "set [lorrax] isdf_terms")
    if vals["lorrax.tda"] and model != "isdf":
        raise _gate("td_tda_model", f"tda = true with model = {model!r}",
                    "model = 'isdf' (a dense kernel.h5 carries its own TDA flag)",
                    "drop tda, or use the ISDF model")
    vals["lorrax.model"] = model
    _check_collision(vals)

    # units: as -> a.u., eV -> Ha
    for k in ("control.timesteps.t0", "control.timesteps.dt_measure",
              "control.timesteps.tf"):
        vals[k] = float(vals[k]) / T_IN_ATTOSEC
    ms = vals["control.timesteps.max_step"]
    vals["control.timesteps.max_step"] = (math.inf if ms is None
                                          else float(ms) / T_IN_ATTOSEC)
    if vals["lorrax.pair_gap_ev"] is None:
        vals["lorrax.pair_gap_ev"] = float(vals["system.esmear"])
    for k in ("system.esmear", "system.depopulation", "system.dephasing"):
        vals[k] = float(vals[k]) / HA_IN_EV
    vals["lorrax.collision_refresh_fs"] *= 1000.0 / T_IN_ATTOSEC
    if vals["lorrax.dt"] == "auto":
        if vals["lorrax.method"] not in ("magnus2", "cfm4"):
            raise _gate("td_dt", f"dt = 'auto' with method {vals['lorrax.method']!r}",
                        "'auto' only for magnus2 / cfm4 (their step limit is measured "
                        "on the model at the start of the run)", "set [lorrax] dt in as")
    elif vals["lorrax.method"] != "dp54":
        if vals["lorrax.dt"] is None or float(vals["lorrax.dt"]) <= 0.0:
            raise _gate("td_dt", f"[lorrax] dt = {vals['lorrax.dt']!r} for a fixed-step "
                        "method", "dt > 0 in attoseconds", "set [lorrax] dt")
        dt = float(vals["lorrax.dt"]) / T_IN_ATTOSEC
        ratio = vals["control.timesteps.dt_measure"] / dt
        if (vals["lorrax.method"] == "magnus2"
                and abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio)):
            raise _gate("td_dense_output", f"magnus2 with dt_measure/dt = {ratio:.12g}",
                        "samples on steps (an integer ratio): magnus2 freezes dSigma at "
                        "one node, so a sample inside a step would be first order",
                        "choose dt = dt_measure / n, or cfm4 / etdrk4 / rk4 (dense output)")
        vals["lorrax.dt"] = dt
    else:
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
        if typ == "exciton_projection" and not kw.get("eigenvectors"):
            raise _gate("td_exciton_projection", f"measurement {i} without eigenvectors",
                        "a bse_jax --write-eigs eigenvectors.h5 on the same bundle and window",
                        f"set measurements.{i}.eigenvectors")
        meas.append((name, typ, kw))
    if sum(m[1] == "exciton_projection" for m in meas) > 1:
        raise _gate("td_exciton_projection", "two exciton_projection measurements",
                    "one", "project onto one eigenvector file (raise n_states)")
    names = [m[0] for m in meas]
    if len(set(names)) != len(names) or any(
            str(n).startswith(RESERVED_SAMPLE_PREFIX) for n in names):
        raise _gate("td_measurement_names", names, "unique dataset names, none starting "
                    f"with {RESERVED_SAMPLE_PREFIX!r} (the driver's own per-chunk record)",
                    "set measurements.<i>.dataset_name")
    efunc = raw.get("system", {}).get("Efunc", {})
    trials = tuple(dict(t) for t in (vals["lorrax.trials"] or [efunc]))
    for t in trials:
        bad = sorted(set(t) - {"E_x", "E_y", "E_z"})
        if bad:
            raise _gate("td_unknown_key", ", ".join(f"lorrax.trials.{b}" for b in bad),
                        "E_x, E_y, E_z", "fix the trial table")
    tdgw = {k: v for k, v in raw.items() if k != "lorrax"}
    given = {k: flat[k] for k in flat if k in TD_DEFAULTS}
    return TdDeck(tdgw=tdgw, values=vals, measurements=tuple(meas),
                  trials=trials, source=source, given=given)


#: TDGW keys a deck may carry that this driver reads and does not use: TDGW's
#: logging level, its vacuum centring for the dipole origin (the bundle and
#: TDGW-header routes take r from the header), and the snapshot stride of a
#: measurement (every sample on the dt_measure grid is written).
UNUSED_TDGW_KEYS = ("control.loglevel", "system.center_of_vacuum")


def unused_keys(deck) -> list:
    """The UNUSED_TDGW_KEYS and ``measurements.<i>.ntsave`` the deck sets."""
    def has(d, dotted):
        for part in dotted.split("."):
            if not isinstance(d, dict) or part not in d:
                return False
            d = d[part]
        return True
    out = [k for k in UNUSED_TDGW_KEYS if has(deck.tdgw, k)]
    out += [f"measurements.{i}.ntsave" for i, (_, _, kw) in enumerate(deck.measurements)
            if "ntsave" in kw]
    return out


def inactive_keys(deck) -> dict:
    """{key: why} for the keys the deck sets that this run does not read.

    The parser refuses a key that contradicts the configuration (a ``dt``
    with ``dp54``, ``stiff_modes`` with another method); these are the
    accepted-but-ignored ones: the SciPy controller of ``dp54`` under a
    fixed-step method, the Magnus corrections under a non-Magnus one, the
    closing solve without a split, a model's or a route's files under
    another, ``ntcheck`` when ``lorrax.chunk`` sets the chunk.  The run log
    flags each; nothing refuses.
    """
    v, given = deck.values, deck.given or {}
    method, model, bundle = v["lorrax.method"], v["lorrax.model"], v["lorrax.system"] == "bundle"
    rules = [
        (("control.ivp.rtol", "control.ivp.atol", "control.timesteps.max_step"),
         method != "dp54", f"dp54's step controller; method is {method}"),
        (("lorrax.n_corr", "lorrax.predictor"), method not in ("magnus2", "cfm4"),
         f"magnus2 / cfm4 only; method is {method}"),
        (("lorrax.stiff_krylov", "lorrax.stiff_tol", "lorrax.stiff_iter"),
         not v["lorrax.stiff_modes"], "the stiff split's; stiff_modes = 0"),
        (("lorrax.kernel_file",), model != "tdgw_kernel", f"model tdgw_kernel only; model is {model}"),
        (("lorrax.isdf_zeta", "lorrax.isdf_terms"), model != "isdf",
         f"model isdf only; model is {model}"),
        (("lorrax.isdf_input",), model != "isdf", f"model isdf only; model is {model}"),
        (("lorrax.pair_rule", "lorrax.pair_gap_ev"), not bundle,
         "system = 'bundle' only (the r of a TDGW header is TDGW's)"),
        (("control.timesteps.ntcheck",), bool(v["lorrax.chunk"]),
         "lorrax.chunk sets the samples per chunk and checkpoint"),
        (("control.files.indir",), bundle or model == "none" or (
            model == "tdgw_kernel" and bool(v["lorrax.kernel_file"])),
         "no kernel.h5 or WFN.h5 is read from it on this route"),
        (("system.ivbm",), model == "none" and not bundle,
         "model none on a TDGW header (occupations come from the header)"),
        (("lorrax.collision_source", "lorrax.collision_hermitian",
          "lorrax.collision_refresh_fs", "lorrax.collision_refresh_df",
          "lorrax.collision_refresh_sweep", "lorrax.collision_screening", "lorrax.collision_statistics",
          "lorrax.collision_frozen_bands", "lorrax.collision_screening_gamma_ev",
          "lorrax.collision_rebase", "lorrax.collision_spm_support_sites_ev",
          "lorrax.collision_spm_chi0_active", "lorrax.collision_in_scattering",
          "lorrax.collision_rate_intermediates", "lorrax.collision_basis",
          "lorrax.collision_internal_pairing"),
         v["lorrax.collision"] == "off", "the collision term's; collision = 'off'"),
        (("lorrax.collision_spm_support_sites_ev", "lorrax.collision_spm_chi0_active"),
         v["lorrax.collision"] != "off" and v["lorrax.collision_screening"] != "shared_pole",
         "collision_screening = 'shared_pole' only"),
        (("lorrax.collision_screening_gamma_ev",),
         v["lorrax.collision"] != "off" and v["lorrax.collision_screening"] != "refreshed",
         "collision_screening = 'refreshed' only"),
    ]
    out = {}
    for keys, inactive, why in rules:
        for key in keys:
            if inactive and key in given:
                out[key] = why
    return out


def _deck_text(value) -> str:
    """A deck value as TOML writes it (strings quoted, lists and tables inline)."""
    import json
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_deck_text(x)}" for k, x in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_deck_text(x) for x in value) + "]"
    return repr(value)


def resolved_deck_rows(deck) -> list:
    """The run log's resolved deck: (key, value, status, notes) rows.

    One row per ``TD_DEFAULTS`` key in its order: the value as the deck
    writes it (TDGW units: as, eV, V/m) or the default; ``status`` is
    ``set``, ``default`` or ``unset``; ``notes`` say that a set value equals
    the default, what the parser resolved an empty default to
    (``lorrax.model``, ``pair_gap_ev``, ``isdf_input``), and flag the keys
    of :func:`inactive_keys` and ``UNUSED_TDGW_KEYS``.  Then one row per
    measurement (its options) and per drive.  The step of ``dt = 'auto'``
    is the stability gate's and is reported there.
    """
    v, given = deck.values, deck.given or {}
    inactive = inactive_keys(deck)
    resolved = {"lorrax.model": v["lorrax.model"],
                "lorrax.pair_gap_ev": f"{v['lorrax.pair_gap_ev'] * 1.0:g} eV",
                "lorrax.isdf_input": v["lorrax.isdf_input"] or None}
    rows = []
    for key, default in TD_DEFAULTS.items():
        if key == "lorrax.trials":
            continue
        notes = []
        if key in given:
            value, status = given[key], "set"
            if value == default:
                notes.append("equals the default")
        else:
            value, status = default, ("unset" if default is None else "default")
        if key in resolved and resolved[key] is not None and value in (None, ""):
            notes.append(f"resolved to {resolved[key]}")
        if key in inactive:
            notes.append(f"not used: {inactive[key]}")
        if key in UNUSED_TDGW_KEYS and key in given:
            notes.append("read, not used by this driver")
        rows.append((key, _deck_text(value), status, notes))
    for i, (name, typ, kw) in enumerate(deck.measurements):
        opts = ", ".join(f"{k} = {_deck_text(x)}" for k, x in sorted(kw.items()))
        rows.append((f"measurements.{i}", f"{typ} as {name!r}" + (f" {{{opts}}}" if opts else ""),
                     "set", ["ntsave read, not used (every sample is written)"]
                     if "ntsave" in kw else []))
    trials_given = "lorrax.trials" in given
    for j, trial in enumerate(deck.trials):
        rows.append((f"lorrax.trials[{j}]" if trials_given else "system.Efunc",
                     _deck_text(dict(trial)),
                     "set" if (trials_given or trial) else "default", []))
    return rows


#: The deck reference, one entry per TD_DEFAULTS key:
#: (default note, meaning, refusals).  ``tools/gen_input_reference.py`` renders the
#: TD-aGW table of docs/input_reference.md from it: the default cell is the
#: TD_DEFAULTS value plus the note in parentheses, the meaning cell the text (Markdown;
#: ``|`` is escaped by the generator), the last cell the GATE ids that refuse a bad
#: value or combination.  A key without an entry, or an entry without a key, refuses
#: the generator (and ``--check``, so gate0).
TD_KEY_DOCS = {
    "control.loglevel": (
        "", "TDGW's log level. Accepted for compatibility and not used: the driver writes "
        "its own log, `tdagw.out` in `outdir`.", ()),
    "control.calc_type": (
        "required", "`IP` (no self-energy update) or `COHSEX` (ΔΣ[ρ] from `lorrax.model`).",
        ("td_missing_key", "td_calc_type")),
    "control.timesteps.t0": (
        "required", "Start time, as.", ("td_missing_key",)),
    "control.timesteps.dt_measure": (
        "required", "Sampling interval, as. Samples are TDGW's grid t0 + i·dt_measure. The "
        "fixed-step methods sample it by dense output, so `lorrax.dt` need not divide it, "
        "except `magnus2`, which needs an integer ratio.",
        ("td_missing_key", "td_dense_output")),
    "control.timesteps.tf": (
        "required", "Final time, as.", ("td_missing_key",)),
    "control.timesteps.ntcheck": (
        "", "Samples per checkpoint, as in TDGW: every ntcheck samples the driver appends "
        "the measurements and rewrites `tdgw_data/checkpoint`. It is also the jitted chunk "
        "length (one row of the log's progress table each) unless `lorrax.chunk` is set.",
        ()),
    "control.timesteps.max_step": (
        "= no limit", "`dp54` only: SciPy's `max_step`, as.", ()),
    "control.ivp.rtol": (
        "", "`dp54` relative tolerance (SciPy RK45 semantics: RMS error norm over the "
        "logical, unpadded entries).", ("td_dp54_step_too_small",)),
    "control.ivp.atol": (
        "", "`dp54` absolute tolerance, in units of ρ.", ("td_dp54_step_too_small",)),
    "control.files.indir": (
        "required by `tdgw_kernel` without `kernel_file`, and by `isdf` on a TDGW header",
        "TDGW's input directory: `kernel.h5` for `lorrax.model = tdgw_kernel`, `WFN.h5` for "
        "`isdf` on a TDGW header. Unused by `lorrax.system = bundle`.",
        ("td_kernel_file", "td_indir")),
    "control.files.outdir": (
        "required", "Output directory: `tdgw.h5` and `tdgw.toml`, or one `trial-<i>/` per "
        "drive with `lorrax.trials`, and the run log `tdagw.out` (one per run).",
        ("td_missing_key", "td_checkpoint_outdir")),
    "control.files.use_vmtxel": (
        "", "`true` refuses: the dipole comes from the TDGW header (a `use_vmtxel` run's "
        "header carries BerkeleyGW's dipoles) or from the bundle.", ("td_header_source",)),
    "control.files.load_header_from": (
        "required with `system = tdgw_header`", "A TDGW output directory; its `tdgw.h5` "
        "header supplies energies, the k grid and the dipole, and is copied into the "
        "output. Empty with `system = tdgw_header`, or set with `system = bundle`, refuses; "
        "its band window must be the deck's.", ("td_header_source", "td_band_window")),
    "control.files.load_checkpoint_from": (
        "", "An earlier run's output directory: the run continues from its last checkpoint "
        "exactly (the integrator's carry in `tdgw_data/checkpoint/lorrax`; the same samples, "
        "bitwise, as the uninterrupted run) and copies the rows before it. `outdir` must "
        "differ; a TDGW checkpoint, which holds no carry, refuses, and so do settings other "
        "than `outdir`, `load_checkpoint_from`, `lorrax.chunk` and, for fixed steps, `tf`.",
        ("td_checkpoint_outdir", "td_restart_carry", "td_restart_settings")),
    "system.ibndmin": (
        "required", "First band of the window, 1-based (TDGW). It must equal the header's "
        "window.", ("td_missing_key", "td_band_window")),
    "system.ibndmax": (
        "required", "Last band of the window, 1-based, inclusive.",
        ("td_missing_key", "td_band_window")),
    "system.ivbm": (
        "", "Valence-band maximum, 1-based. Required by `tdgw_kernel` and by "
        "`system = bundle`; `isdf` takes the occupied count from it.",
        ("td_ivbm", "td_window_occupation")),
    "system.geometry": (
        "required", "`bulk`, `slab` or `wire`: the periodic directions of the intraband "
        "k-derivative. `box` refuses.", ("td_missing_key", "td_geometry")),
    "system.center_of_vacuum": (
        "", "TDGW's position origin for its own dipole build; accepted and unused, since "
        "the header carries the dipole.", ()),
    "system.esmear": (
        "", "Gaussian width, eV, of the energy mask in TDGW's covariant overlap S̃ (the "
        "intraband connection), reproduced from the header's overlaps; also the default "
        "of `lorrax.pair_gap_ev`.", ()),
    "system.depopulation": (
        "", "Diagonal relaxation rate Γ_nn toward ρ_eq, eV. Nonzero makes Tr ρ² decay, so "
        "the log's purity drift is then not an invariant check.", ()),
    "system.dephasing": (
        "", "Off-diagonal relaxation rate Γ_nm, eV (as `depopulation` for the log's "
        "purity drift).", ()),
    "system.patched_sampling": (
        "", "`true` refuses: a regular k grid only.", ("td_patched_sampling",)),
    "system.Efunc.E_x": (
        "", "Field component along x, V/m: a number or a TDGW expression in `t` (as) with "
        "numpy as `np` and `HBAR_IN_EV_AS`, `EFIELD_IN_VPERM` in scope. The how-to's "
        "[example decks](how-to/td-agw-run.md#decks) give a δ kick and Gaussian pulses.",
        ()),
    "system.Efunc.E_y": (
        "", "As `E_x`, along y.", ()),
    "system.Efunc.E_z": (
        "", "As `E_x`, along z.", ()),
    "measurements.types": (
        "required", "The measurement list: `Efield`, `polarization`, `current`, "
        "`density_matrix`, `lessG` (TDGW's), and LORRAX's own `carrier_density` (conduction "
        "electrons and valence holes per cell, (2,) per sample) and `exciton_projection` "
        "(amplitudes c_S = Σ conj(A^S_kcv) ρ_cv(k) on the states of a BSE "
        "`eigenvectors.h5`; |c_S|²/N_k excitons per cell). `velocity_operator` refuses. "
        "Per-measurement options are `[measurements.<i>]` tables (below); dataset names "
        "are unique and do not start with `lorrax.`.",
        ("td_missing_key", "td_measurement_unsupported", "td_measurement_names",
         "td_exciton_projection", "td_brange", "td_krange", "td_projection_window",
         "td_projection_kpoints")),
    "lorrax.method": (
        "", "`dp54` (SciPy RK45, adaptive; takes TDGW's steps), `rk4`, `etdrk4` (Krogstad), "
        "`magnus2` (exponential midpoint) or `cfm4` (fourth-order commutator-free Magnus; "
        "fourth order also with Γ and the intraband term). The Magnus schemes integrate "
        "Δρ = ρ − ρ_eq in the interaction picture of the band energies and check their "
        "step at the start of the run. Which to use: "
        "[choosing an integrator](how-to/td-agw-run.md#integrators).", ("td_method",)),
    "lorrax.dt": (
        "required by every method but `dp54`", "The fixed step, as, for every method but "
        "`dp54` (refused there). `auto` (`magnus2`, `cfm4`): the largest step the start-of-run "
        "stability gate accepts, from the static limit down by 0.85 per try, floored to "
        "0.01 as (`magnus2`: dt_measure / n). A `magnus2` / `cfm4` step above the static "
        "limit, or one whose step probe at 1.25 dt grows, refuses, naming the limit or "
        "the measured growth; the log's gate block records both.",
        ("td_dt", "td_dense_output", "td_magnus_stability")),
    "lorrax.n_corr": (
        "", "Corrections per step of `magnus2` (midpoint Hamiltonian) and `cfm4` "
        "(Gauss-node Hamiltonians); each costs one ΔΣ evaluation per node, so `cfm4` makes "
        "2·n_corr ΔΣ calls per step. At least 1. The stable step is not monotone in "
        "`n_corr`: `cfm4` with the cubic predictor is stable for h·σ ≤ 0.117 / 0.730 / "
        "1.144 with 1 / 2 / 3 corrections, σ the spectral radius of the linearized ΔΣ "
        "commutator map, estimated at the start of the run.",
        ("td_n_corr", "td_magnus_stability")),
    "lorrax.predictor": (
        "", "The ΔΣ predictor of `magnus2` / `cfm4` before the corrections: `cubic` "
        "(`cfm4`: through the last two steps' four Gauss-node values; `magnus2`: quadratic "
        "through three midpoints) or `linear` (`cfm4` only, through the last step's two: "
        "the same order, about twice the error, a 1.6× larger stable step, h·σ ≤ 1.203 "
        "with two corrections).", ("td_predictor",)),
    "lorrax.magnus_frame": (
        "= `whole_rho` for `magnus2`, `interaction` for `cfm4`", "The frame of `magnus2`: "
        "`whole_rho`, the exponential midpoint on the whole ρ with ΔΣ at the average of "
        "the end states (the post-pulse energy is conserved up to rounding; a rounding "
        "floor ∝ 1/E₀ in weak fields), or `interaction`, Δρ in the interaction picture of "
        "the band energies (no floor, exact band phases, an O(h²) energy drift). `cfm4` "
        "runs in the interaction picture only.", ("td_magnus_frame",)),
    "lorrax.model": (
        "= `none` for IP, `tdgw_kernel` for COHSEX", "The self-energy update: `none`, "
        "`tdgw_kernel` (TDGW's dense `kernel.h5`) or `isdf` (the LORRAX ISDF model of a GW "
        "restart bundle). The log's self-energy block names the terms, N_μ and the wall "
        "of one ΔΣ call.", ("td_model",)),
    "lorrax.kernel_file": (
        "= `indir/kernel.h5`", "The BerkeleyGW-layout kernel for `tdgw_kernel`: complex "
        "flavor, TDA (one block) or full (four), on the header's k list and inside the "
        "window.", ("td_kernel_file", "td_kernel_kpoints", "td_kernel_flavor",
                    "td_kernel_blocks", "td_kernel_bands")),
    "lorrax.isdf_input": (
        "= `bundle_input` with `system = bundle`", "The GW deck of the restart bundle, "
        "required by `isdf`; with `system = bundle` it must be (or default to) "
        "`bundle_input`.", ("td_isdf_input", "td_isdf_window", "td_isdf_kpoints")),
    "lorrax.isdf_zeta": (
        "= the deck's", "The bundle's ζ file, when not at the deck's location.", ()),
    "lorrax.isdf_terms": (
        "", "Terms of the ISDF model: `both`, `sex` (screened exchange) or `hartree`.",
        ("td_isdf_terms",)),
    "lorrax.intraband": (
        "", "Keep the intraband term E·[r^intra, ρ] (TDGW keeps it); `false` drops it. The "
        "bundle route has no intraband term.", ()),
    "lorrax.chunk": (
        "= `ntcheck`", "Samples per jitted chunk: one host write, one checkpoint and one row "
        "of the log's progress table each; capped so a density-matrix buffer fits the "
        "streaming tile.", ()),
    "lorrax.trials": (
        "", "A list of `{E_x, E_y, E_z}` tables propagated together on a leading trial axis "
        "(for example two circular polarizations); one output directory per trial, one "
        "log.", ("td_unknown_key",)),
    "lorrax.system": (
        "", "Where the header comes from: `tdgw_header` (a TDGW run's `tdgw.h5`, "
        "`load_header_from`) or `bundle` (built from the GW restart bundle `bundle_input` by "
        "`tdagw.td_system`, no TDGW WFN). `bundle` refuses `load_header_from` and needs "
        "`bundle_input`, `dipole_file` and `system.ivbm`; the bundle-only keys refuse with "
        "`tdgw_header`.", ("td_system", "td_header_source", "td_bundle_input", "td_ivbm",
                           "td_bundle_key")),
    "lorrax.bundle_input": (
        "", "The GW deck of the restart bundle (`system = bundle`); a deck with "
        "`bse_k_grid` builds the densified system. With `model = isdf` it is also the "
        "model's deck.", ("td_bundle_input", "td_bundle_key", "td_densified_velocity",
                          "td_densified_k_order")),
    "lorrax.eqp_file": (
        "= DFT energies", "The BSE run's `eqp1.dat` (`system = bundle`): QP energies by DFT "
        "index on a coarse deck, the htransform's QP f(H) states and energies on a "
        "`bse_k_grid` deck.", ("td_bundle_key",)),
    "lorrax.dipole_file": (
        "", "The coarse WFN's `dipole.h5` (`psp.get_dipole_mtxels`) for r = v/(iΔE) with the "
        "velocity's own (DFT) energies (`system = bundle`); it must cover the window and "
        "carry the restart's DFT energies.", ("td_bundle_input", "td_bundle_key",
                                               "td_dipole_bands", "td_dipole_energies")),
    "lorrax.pair_rule": (
        "", "Same-occupation pairs (vv', cc') of the bundle's r: `mask` multiplies r by "
        "1 − exp(−ΔE²/(2 gap²)), `cut` zeroes pairs with |ΔE| < gap, `cv` keeps the c↔v "
        "blocks only (TDGW's `use_vmtxel` layout). n = m is always zero (no intraband "
        "commutator on this route).", ("td_pair_rule", "td_pair_gap_route_b")),
    "lorrax.pair_gap_ev": (
        "= `system.esmear`", "The energy scale of `pair_rule`, eV; also the smallest "
        "accepted c↔v gap.", ("td_cv_gap", "td_pair_gap_route_b")),
    "lorrax.check_eigenvectors": (
        "", "A `bse_jax --write-eigs --dipole` file on the same bundle and window: the "
        "builder reproduces its ⟨0|r|S⟩ from the header (the gauge receipt, logged and "
        "stored in the header). It costs a load of the eigenvectors; leave it unset in "
        "production once the receipt has been read.", ("td_bundle_key", "td_eigenvector_kpts")),
    "lorrax.degeneracy_mode": (
        "= `strict`", "The band-window degeneracy gate of the bundle load (`strict` refuses a "
        "window edge inside a QP multiplet, `snap` widens the window, `off` keeps it; TDGW "
        "applies no such check, so reproducing a TDGW window may need `off`).",
        ("td_degeneracy_mode", "td_bundle_key")),
    "lorrax.tda": (
        "", "TDGW's TDA semantics for the ISDF ΔΣ: only δρ_cv drives, only ΔΣ_cv (and its "
        "Hermitian conjugate) is kept, the resonant–antiresonant coupling is dropped (the "
        "projection of `tdagw_bridge --tda`). ISDF model only.", ("td_tda_model",)),
    "lorrax.stiff_modes": (
        "= no split", "`cfm4` only: split ΔΣ exactly on the invariant subspace of the n "
        "largest eigenvalues of the coupling map L (a real Arnoldi on L at the start of the "
        "run, `stiff_krylov` steps; n raised to the end of a ± / conjugate group): "
        "ΔΣ[x] = (ΔΣ[x] − Σ_j ⟨u_j, x⟩ b_j) + Σ_j ⟨u_j, x⟩ b_j, b_j = ΔΣ[u_j]. The first "
        "part keeps the predictor and `n_corr` corrections; the second is solved "
        "implicitly at the Gauss nodes (simplified Newton on exponentials and "
        "projections, no ΔΣ call), so the static limit and the probe bind the slow map "
        "L(1 − Π) only. Needs an affine model. On the 30×30 CrI3 ISDF model 10 modes "
        "double the stable step (50 → 100 as); it pays at P(t) errors near 1e-4, not at "
        "1e-5 ([choosing an integrator](how-to/td-agw-run.md#integrators)).",
        ("td_stiff_modes", "td_stiff_method", "td_stiff_model", "td_stiff_basis")),
    "lorrax.stiff_krylov": (
        "= max(80, 5·stiff_modes)", "Arnoldi steps on L for the stiff basis (one ΔΣ call "
        "each, once per run); above `stiff_modes`. The log records the selected moduli, "
        "the next one, the cut and the invariance residual; 120 resolved the 10 outliers "
        "of the 30×30 CrI3 model to a zero residual.", ("td_stiff_krylov",)),
    "lorrax.stiff_tol": (
        "", "Relative residual ‖⟨u, x(a)⟩ − a‖ / ‖a‖ of the stiff coefficients a at the "
        "nodes that ends a step's closing solve. A step that misses it in `stiff_iter` "
        "node-map evaluations refuses the run at its chunk end, before the chunk's samples "
        "are written; so does a probe of the stability gate.", ("td_stiff_solve",)),
    "lorrax.stiff_iter": (
        "", "Node-map evaluations (each: the exponentials to the two nodes, no ΔΣ call) of "
        "the closing solve per step, at least 1.", ("td_stiff_solve",)),
    "lorrax.collision": (
        "", "The column-sampled, one-sided Markov GW collision term, with W frozen at the "
        "source run's equilibrium poles "
        "([td_agw_dynamical.md §1.5](theory/td_agw_dynamical.md)): `off`; `subtracted`, "
        "C[ρ] − C[ρ_eq], an explicit reference counterterm; `raw`, the literal C[ρ]. "
        "The complex particle column `cond[1]` and the adjoint hole row `val[0]^†` "
        "define the total Hermitian width Γ, injection D and coherent M_H without an "
        "arithmetic average of external energies. The dissipative flow "
        "x' = −½{Γ, x} + S is solved exactly in the eigenbasis of Γ "
        "in the remainder slot (Magnus) or added to the right-hand side (dp54, rk4, "
        "etdrk4); M_H enters the held Hamiltonian once. Needs `system = bundle` on a "
        "GN-PPM bundle; the GW driver reruns on a "
        "copy of `collision_source` (default `bundle_input`) in `outdir/collision_source` "
        "at the start (no restart written) to recover the poles or, with "
        "`collision_screening = shared_pole`, the shared-pole model and its W stage's "
        "inputs (a run of the wrong member, or one whose window occupations differ from "
        "the TD system's, refuses there: `dynsig_gnppm_mode`, `dynsig_collision_member`, "
        "`dynsig_collision_occupations`). The log's `Collision refreshes` section has one "
        "line per refresh.",
        ("td_collision", "td_collision_source", "td_collision_restart")),
    "lorrax.collision_source": (
        "= `bundle_input`", "The GW deck of the rates: the GN-PPM deck of the bundle, which "
        "it must be (or default to) for `frozen` and `refreshed` screening (one "
        "wavefunction gauge); with `collision_screening = shared_pole` a one-shot "
        "`sigma_w_model = shared_pole` deck on the bundle's wavefunctions and window "
        "(`dynsig_collision_occupations` refuses another window).", ("td_collision_source",)),
    "lorrax.collision_hermitian": (
        "", "The held Hermitian part ΔΣ^h added to K δρ at each refresh: `dynamic`, "
        "M_H[ρ] − M_H[ρ_eq] less its static-W limit (the response of W^c(0), which "
        "K already holds); `full`, the whole M_H[ρ] − M_H[ρ_eq]; `none`. This term is "
        "not repeated in the dissipative remainder.",
        ("td_collision",)),
    "lorrax.collision_refresh_fs": (
        "= never", "Refresh the rates, the dissipator and ΔΣ^h every this many fs (at "
        "the first chunk end past it); 0 holds the rates of ρ_eq.",
        ("td_collision_refresh",)),
    "lorrax.collision_refresh_df": (
        "= never", "Refresh at the first chunk end where max |f − f_last| over the "
        "window's populations exceeds this; 0 = no population trigger.",
        ("td_collision_refresh",)),
    "lorrax.collision_refresh_sweep": (
        "= active", "What a refresh sweeps: `full`, the window Σ over every intermediate band, "
        "planned on the GW run's on-shell list; `active`, the window's own intermediate "
        "bands only, added to the frozen bands' part evaluated once at t0 (exact with W "
        "frozen: each Σ branch is linear in the intermediate populations and the frozen "
        "bands keep theirs), both parts planned on the window's own on-shell energies, each "
        "on its own held quadrature session ([td_agw_dynamical.md §1.6]"
        "(theory/td_agw_dynamical.md)). The refresh lines name the sweep's τ nodes.",
        ("td_collision",)),
    "lorrax.collision_screening": (
        "", "The W of the collision coefficients and of ΔΣ^h. The column-sampled MGKBA "
        "operator admits `frozen`, the GW run's equilibrium W0 and its explicit "
        "lesser/greater branches; `joint`, snapshot rational "
        "screening at every converged snapshot; and `joint_frozen`, the identical "
        "reduced equilibrium screening held fixed as a control. Joint modes require "
        "`collision_basis = instantaneous`. Density rank, rational supports/order, "
        "held-out errors and finite lag decay have explicit controls below. Joint modes "
        "also require explicit `collision_statistics`: `retarded_reference` fits only "
        "full-rho retarded response and uses equilibrium-style reference branches; "
        "`finite_lag` is the historical "
        "nonthermal statistical model; `lossless_bath` uses an explicitly secular "
        "screening bath with lossless RPA modes and weak colored-transition statistics. "
        "`refreshed` and `shared_pole` are refused by "
        "`td_collision_keldysh_screening`: those experimental paths reconstruct retarded W from "
        "instantaneous populations but lack the full instantaneous-energy/density snapshot "
        "adapter and declared branch prescription of the joint modes. Their earlier implementation "
        "was: "
        "`refreshed`, at every refresh W is re-screened at both GN-PPM supports (0 and "
        "iω_p) from the window populations — δχ0 of the RPA bubble (cross-gap pairs on the "
        "run's own Laplace rules, Pauli blocking, same-side pairs exact), the Dyson solve "
        "with W0 in the place of v and the run's own pole fit "
        "(`gw.ppm_sigma.screening_update`) — and the window Σ is swept on the new poles; "
        "the q → 0 head stays the run's. The frozen bands' part changes with W, so it is "
        "swept at every refresh (with `collision_refresh_sweep = active`). The static-W "
        "limit subtracted from ΔΣ^h stays the run's W0 (K δρ holds −W^c_0(0) δf), so both "
        "the static and the dynamical part of δW are kept "
        "([td_agw_dynamical.md §7.1](theory/td_agw_dynamical.md)). The refresh lines add "
        "the walls (δχ0, Dyson + fit, frozen and active sweeps), the invalid-mode count "
        "and flips, max|ΔΩ| and max|δW|/|W0| at both supports. `shared_pole` (a "
        "shared-pole `collision_source`): at every refresh the source's whole W stage is "
        "rerun at the window populations — the response bank of χ0(f(t)) at the support "
        "set, the moments, the recipe and the constructor, a full refit "
        "(`gw.shared_pole_screening.screen_shared_poles`; bitwise the run's model at "
        "ρ_eq) — and the window Σ is swept on the new model; the head stays the run's, "
        "the frozen part is swept per refresh as with `refreshed`, and the static limit "
        "is the equilibrium model's own W^c(0). The refresh lines add the refit walls "
        "(bank, moments, constructor), the pole count, whether the support sites moved, "
        "and the pole weight below 0.3 and 1 eV.",
        ("td_collision",)),
    "lorrax.collision_statistics": (
        "= not selected", "Statistical closure for joint and joint_frozen screening. "
        "Active joint propagation refuses the default `unspecified` before GW setup. "
        "`retarded_reference` fits only the retarded polarization from the full "
        "instantaneous density. Equilibrium-style zero-temperature pole branches "
        "then define the reference collision, without fitting independent lesser "
        "fluctuations. Signed/complex residues are retained; this is not a complete "
        "nonequilibrium Keldysh model or a positivity guarantee. "
        "`finite_lag` explicitly accepts a lag-decay "
        "statistical model whose screened equilibrium FDT is known not to hold in "
        "general, even as the decay tends to zero. `lossless_bath` instead declares "
        "a stationary screening bath: instantaneous-energy screening coherences are "
        "pinched, while full rho remains in the material Hamiltonian and electronic "
        "collision lines. A bounded signed Hermitian response reduction supplies real "
        "RPA poles; weak colored-transition baths assign fluctuation residues, including "
        "independent response-free noise. The common-temperature limit obeys the "
        "reduced model's equilibrium FDT. Stability, covariance and held-out response "
        "checks must pass. This is an explicit approximation, not exact coherent "
        "screening or an electronic Pauli-bound guarantee. This key is not a fit or "
        "convergence tolerance. Legacy frozen screening and collision-off defaults "
        "are unchanged.", ("td_collision_statistics",)),
    "lorrax.collision_rebase": (
        "= auto", "With refreshed or shared-pole screening and `collision = subtracted`: "
        "`on`, the subtracted C[ρ_eq] is re-evaluated on each refresh's W (one more "
        "window sweep at the equilibrium populations), so ρ_eq stays a fixed point of "
        "every refresh's dissipator and the regularization tails cancel at every W; "
        "`off`, the t0 rates of ρ_eq throughout; `auto`, on for `shared_pole`, off "
        "otherwise. The refresh lines add the rebase wall.",
        ("td_collision",)),
    "lorrax.collision_spm_support_sites_ev": (
        "= the production ladder", "With `collision_screening = shared_pole`: the refits' "
        "support set, `'<line eV list> | <imaginary eV list> [| <line height eV>]'` "
        "(`sigma_w_support_sites_ev` of the GW deck, with an optional line height), "
        "replacing the production ladder re-resolved at every state; the equilibrium "
        "model is then refitted on this set at t0 and is the run's reference (its "
        "difference from the source's model is in the t0 line).",
        ("td_collision",)),
    "lorrax.collision_in_scattering": (
        "= full", "With `collision_refresh_sweep = active`: the internal "
        "line of the active part's rates. `populations` is a diagnostic kernel using the "
        "band populations f = diag ρ "
        "(hole f_m, particle 1 − f_m), the coherences of ρ are not in the rates. `full`: the "
        "window's whole ρ, hole ρ_mn and particle (1 − ρ)_mn, contracted in the Σ sweep as "
        "G = L diag(w) R^H on the orbital panels ψ and ψC with the energy pairing of "
        "`collision_internal_pairing` ([td_agw_dynamical.md §1.5, Eqs. 8a and "
        "8c](theory/td_agw_dynamical.md)); at a diagonal ρ it is `populations` to rounding. "
        "The frozen part is unchanged (its populations never change). Collision-enabled "
        "propagation requires `full`; `populations` is refused because it does not close the "
        "full-matrix number balance.",
        ("td_collision",)),
    "lorrax.collision_rate_intermediates": (
        "= active", "With `collision_refresh_sweep = active`: the intermediate "
        "bands used to build the dissipative coefficients Gamma and D. `all` is a diagnostic "
        "using every "
        "band of the Σ sum. `active`: the window's own; the frozen bands' part then enters "
        "the coherent M_H correction (ΔΣ^h, and under refreshed W its change), only. Frozen "
        "populations never update, so their transitions into the window (energy-forbidden "
        "at T = 0, alive only in the η tails) are a particle source or sink; with "
        "`active` and a closed window, instantaneous trace conservation is checked from the "
        "reported loss/injection balance. Collision-enabled propagation requires `active`.",
        ("td_collision",)),
    "lorrax.collision_basis": (
        "= band", "With `collision_in_scattering = full`: the basis and "
        "energies of the rates' lines. `band`: the band basis at the collision source's "
        "bundle energies (the GW run's ε^G). `instantaneous`: solve the material "
        "Hamiltonian using the full current ρ. Joint screening uses bare Hartree/Fock "
        "increments and the convention selected by `collision_hamiltonian`; the frozen "
        "legacy path uses the reference screened-exchange kernel. "
        "The coherent held correction and its eigenpairs converge together, controlled "
        "by `collision_snapshot_*`; the same final U,E enter the electronic lines and "
        "column shells, and joint screening when selected. Explicit driving is excluded "
        "from the internal snapshot. Frozen bands keep their step occupations and edge "
        "scissors. The source contains authenticated kinetic/ionic, Hartree and Fock "
        "matrices for absolute HF reconstruction without needing V_xc "
        "([td_agw_dynamical.md §1.5, Eq. 8b](theory/td_agw_dynamical.md)).",
        ("td_collision",)),
    "lorrax.collision_hamiltonian": (
        "", "With joint screening: `hf` solves H_s = h_HF[ρ] + M_H using the source's "
        "authenticated kinetic/ionic, Hartree and bare Fock baseline. `reference` uses "
        "diag(e_ref) + Δh_HF + (M_H − M_H_eq), an explicitly reference-anchored model "
        "for comparison to the original TD-aGW. Both use bare HF increments and count "
        "dispersion once; no static reduced-W subtraction is made. A DFT-prepared "
        "density is not assumed stationary for the absolute `hf` Hamiltonian. Joint "
        "snapshots require `stiff_modes = 0`; the coupled stiff basis is not validated.", ("td_collision",)),
    "lorrax.collision_snapshot_steps": (
        "", "Maximum coherent self-energy sweeps per instantaneous snapshot. "
        "An unconverged snapshot refuses; no last iterate is installed.", ("td_collision",)),
    "lorrax.collision_snapshot_tol_ev": (
        "eV", "Maximum per-trial/k Frobenius norm of the coherent fixed-point residual. "
        "Accepted Hamiltonian and collision tables share the same final sweep eigenpairs.",
        ("td_collision",)),
    "lorrax.collision_snapshot_mixing": (
        "", "Fraction of the new coherent fixed-point map mixed into the predictor, "
        "0 < mixing <= 1. It controls convergence, not the collision regularization.",
        ("td_collision",)),
    "lorrax.collision_snapshot_acceleration": (
        "", "Instantaneous snapshot solver: `anderson` uses the existing single-evaluation "
        "Anderson accelerator with bounded history; `linear` uses bounded mixing alone. "
        "Both obey the same sweep budget and physical residual acceptance test.", ("td_collision",)),
    "lorrax.collision_density_rank": (
        "", "With `joint` or `joint_frozen` screening: rank of the fixed Coulomb "
        "density-space projection. All supplied electronic bands remain in the "
        "polarization; screening coherences are retained by retarded_reference and "
        "finite_lag but explicitly "
        "pinched by lossless_bath. Converge this independent spatial "
        "approximation against collision coefficients and dynamics.", ("td_collision",)),
    "lorrax.collision_pole_order": (
        "", "Maximum rational response order for retarded_reference, common "
        "lesser/greater order for finite_lag, or reduced response order for lossless_bath. Response "
        "and fluctuation statistics use the same reduced model; independent "
        "response-free noise must also be retained.", ("td_collision",)),
    "lorrax.collision_projection_order": (
        "", "With finite_lag: maximum reduced state count in the causal Keldysh projection workspace. "
        "This explicit capacity guard defaults to 400; increase it when converging "
        "rational order only if the required dense host memory and cubic solve cost "
        "are available. Unused by lossless_bath. It does not relax response or noise acceptance.", ("td_collision",)),
    "lorrax.collision_pencil_mb": (
        "MiB", "Maximum size of each complex full-block Loewner pencil, in MiB "
        "(2^20 bytes). Factorization workspace and copies require approximately eight "
        "such arrays. For lossless_bath it instead bounds the retained orthonormal "
        "transition basis plus conservative dense pole-bank, reciprocal-bank and "
        "downstream common-union workspace estimates. This explicit capacity guard "
        "does not choose rational order or "
        "relax response or noise acceptance.", ("td_collision",)),
    "lorrax.collision_fit_samples": (
        "", "Number of finite complex frequency supports for joint rational fitting "
        "or lossless rational Krylov shifts. Independent held-out response checks "
        "must pass; finite_lag additionally checks dressed noise on held-out frequencies, "
        "while lossless_bath certifies collective normalization and statistical residues. "
        "Retarded_reference checks response only, without an independent noise fit.",
        ("td_collision",)),
    "lorrax.collision_fit_real_axis": (
        "", "Also train the joint rational fit on exact real parts of its complex "
        "fit supports. The declared positive finite lag decay makes these boundary "
        "samples finite. Real training supports are disjoint from the independent "
        "real dressed-noise checks. This adds sampling and pencil cost, not frequency "
        "integration, a regulator change or relaxed acceptance. Unused by lossless_bath "
        "and retarded_reference.", ("td_collision",)),
    "lorrax.collision_fit_tolerance": (
        "", "Relative held-out tolerance in reduced Coulomb coordinates: finite_lag "
        "checks both fluctuation bubbles, retarded screening and dressed fluctuation noise; "
        "lossless_bath checks retarded screening, with separate normalization and "
        "lesser/greater residue-positivity gates. Retarded_reference checks the "
        "retarded polarization and screened response only.",
        ("td_collision",)),
    "lorrax.collision_adiabatic_ev": (
        "eV", "Declared positive exponential decay of frozen polarization lag kernels "
        "in finite_lag or retarded_reference response sampling. The historical key name does not imply "
        "a validated physical adiabatic limit: screened equilibrium FDT can fail even "
        "as this decay tends to zero. Separate from collision eta and fit offsets. "
        "Unused by lossless_bath: statistical weights are defined before collision "
        "regularization rather than selected by artificial damping.", ("td_collision",)),
    "lorrax.collision_fit_height_ev": (
        "eV", "Imaginary offset of the rational fit supports. It is numerical sampling "
        "geometry and is separate from finite lag decay and collision broadening.",
        ("td_collision",)),
    "lorrax.collision_internal_pairing": (
        "= vertex", "With `collision_in_scattering = full`: the energy pairing of the "
        "internal line in the two on-shell tables Σ_ij(ε_i), Σ_ij(ε_j). `symmetric` is "
        "the earlier diagnostic ½ {g(ε'), C} in both (one sweep per branch on 2 n_A internal "
        "columns); it is refused for propagation because it is not number conserving after "
        "column sampling. `vertex`: "
        "each external index with the internal index on its own vertex, Σ_ij(ε_i) from "
        "G_L = ψ' e^{−iε'τ} C ψ'^† and Σ_ij(ε_j) from G_R = ψ' C e^{−iε'τ} ψ'^† (four "
        "sweeps); it conserves Σ_k Tr C[ρ] on reciprocal poles "
        "([td_agw_dynamical.md §1.5, Eq. 8c](theory/td_agw_dynamical.md)).",
        ("td_collision",)),
    "lorrax.collision_spm_chi0_active": (
        "= one rule on every pair", "With `collision_screening = shared_pole`: `auto` or `[a0, a1]`, "
        "the active bands A of the refits' chi0 (0-based on the W carrier's band axis, "
        "half-open), extended to the smallest isolated band group (positive gaps below and "
        "above) containing it and the window; `auto` starts from the window. chi0 is then "
        "four (hole × electron) blocks FF, FA, AF, AA, F the other bands at their 0/1 step: "
        "only AA crosses zero, each block's rule comes from band energies and is built at "
        "the first refit and held for the run. Refused only if no isolated group exists.",
        ("td_collision",)),
    "lorrax.collision_frozen_bands": (
        "= every frozen band", "With `collision_screening = refreshed` or `shared_pole` and "
        "`collision_refresh_sweep = active`: `[n_below, n_above]`, the frozen bands whose "
        "part is re-swept under W(t) — [b_lo − n_below, b_lo) and [b_hi, b_hi + n_above); "
        "the others keep their part at W0 (Σ^F[W(t)] ≈ Σ^F[W0] + Σ^F_kept[W(t)] − "
        "Σ^F_kept[W0], exact at ρ_eq). A truncation: its error is measured, not bounded.",
        ("td_collision",)),
    "lorrax.collision_screening_gamma_ev": (
        "", "With `collision_screening = refreshed`: the width (eV) of the principal value "
        "taken for the static support's same-side pairs, (f_a − f_b)(E_a − E_b)/((E_a − "
        "E_b)² + γ²): exact for |E_a − E_b| ≫ γ, zero for a degenerate pair. A driven "
        "carrier distribution makes symmetry-equivalent k unequal, and the exact static "
        "term of such a pair is a Drude pole; 0 keeps the exact term and refuses those "
        "pairs by name (`dynsig_degenerate_population`). The refresh lines count the "
        "degenerate pairs taken at their principal value.",
        ("td_collision",)),
}
