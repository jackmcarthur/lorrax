"""tdagw/stages.py — the stages of the TD-aGW driver, one function each.

``tdagw.tdgw_jax.main`` calls these in order: the TDGW header, the k-sharded
operators of Eq. (1)–(2) of :mod:`tdagw.eom`, the self-energy model, the
outputs, the propagation with its per-chunk writer.  Files are written by
process 0 from gathered host arrays; every process runs every collective.
"""
from __future__ import annotations

import os
import time
from typing import NamedTuple

import numpy as np

from common import timing
from common.collectives import process_rank
from tdagw import propagate as tp
from tdagw import td_config, tdgw_io
from tdagw.self_energy import (SelfEnergyModel, dense_kernel_from_tdgw_layout,
                               dense_kernel_model, isdf_model, zero_model)
from tdagw.td_config import TdDeck

__all__ = ["TdSystem", "load_system", "build_self_energy", "open_outputs",
           "sample_spec", "restart_fingerprint", "load_checkpoint", "propagate",
           "describe", "describe_rows", "output_files", "summarize"]


class TdSystem(NamedTuple):
    """The propagated system: header, k-sharded operators, sizes.

    hdr : raw TDGW header (host).   ops : tdagw.eom.TdOperators on the mesh.
    nk, nk_pad, nb : logical k, padded k, bands.  cell_volume (bohr^3),
    nspinor.  bands : (start, stop, step) 1-based absolute window.
    model : the bundle's ``load_td_model`` output (system = 'bundle'), reused
    by the ISDF ΔΣ, else None.  receipts : ``tdagw.td_system`` receipts or None.
    """
    hdr: dict
    ops: object
    nk: int
    nk_pad: int
    nb: int
    cell_volume: float
    nspinor: int
    bands: tuple
    model: object = None
    receipts: dict = None


@timing.timed("tdagw.system")
def load_system(deck: TdDeck, mesh) -> TdSystem:
    """The header (``load_header_from``, or built from a GW bundle) and Eq. (1)-(2)."""
    v = deck.values
    bundle_model = receipts = None
    if v["lorrax.system"] == "bundle":
        from tdagw.td_system import build_td_system
        built = build_td_system(
            v["lorrax.bundle_input"], mesh,
            bands=(int(v["system.ibndmin"]) - 1, int(v["system.ibndmax"])),
            dipole_file=v["lorrax.dipole_file"], eqp_file=v["lorrax.eqp_file"] or None,
            n_occ=int(v["system.ivbm"]), pair_rule=v["lorrax.pair_rule"],
            gap_ev=float(v["lorrax.pair_gap_ev"]), zeta_file=v["lorrax.isdf_zeta"] or None,
            eigenvectors_file=v["lorrax.check_eigenvectors"] or None,
            degeneracy_mode=v["lorrax.degeneracy_mode"] or None,
            log=lambda s: print(s) if process_rank() == 0 else None)
        hdr, bundle_model, receipts = built.hdr, built.model, built.receipts
    else:
        hdr = tdgw_io.read_header(v["control.files.load_header_from"])
    start, stop, step = hdr["bands"]
    want = (int(v["system.ibndmin"]), int(v["system.ibndmax"]) + 1, 1)
    if (start, stop, step) != want:
        raise ValueError(
            f"GATE td_band_window: got: header bands {(start, stop, step)}; "
            f"want: [system] ibndmin..ibndmax = {want}; fix: point "
            f"load_header_from at a run over the same window")
    host = tdgw_io.header_operators(hdr, geometry=v["system.geometry"],
                                    esmear_ha=v["system.esmear"],
                                    intraband=bool(v["lorrax.intraband"]))
    nb = host["e_qp"].shape[1]
    eye = np.eye(nb)
    host["gamma"] = (v["system.depopulation"] * eye
                     + v["system.dephasing"] * (1.0 - eye))
    ops, nk_pad = tp.place_operators(host, mesh)
    proj = [kw for _, typ, kw in deck.measurements if typ == "exciton_projection"]
    if proj:
        ops = ops._replace(proj=tp.place_projector(
            _projector(proj[0], hdr), mesh, nk_pad))
    return TdSystem(hdr=hdr, ops=ops, nk=host["e_qp"].shape[0], nk_pad=nk_pad,
                    nb=nb, cell_volume=host["cell_volume"],
                    nspinor=host["nspinor"], bands=hdr["bands"],
                    model=bundle_model, receipts=receipts)


def _n_val(hdr: dict) -> int:
    """Occupied window rows (TDGW occupations: the leading full bands)."""
    f = np.asarray(hdr["f_kn"])
    n_val = int(np.rint(f[0].sum()))
    if not np.array_equal(f, np.broadcast_to((np.arange(f.shape[1]) < n_val) * 1.0, f.shape)):
        raise ValueError("GATE td_occupations: got: occupations that are not the leading "
                         "rows at every k; want: valence rows first; fix: a gapped window")
    return n_val


def _projector(kw: dict, hdr: dict) -> np.ndarray:
    """The ``exciton_projection`` eigenvectors, checked against the header."""
    from bse.absorption_common import load_eigenvectors_h5
    _, A, params = load_eigenvectors_h5(kw["eigenvectors"])
    n = A.shape[0] if kw.get("n_states") is None else min(int(kw["n_states"]), A.shape[0])
    nk, nb = np.asarray(hdr["e_kn"]).shape
    n_val = _n_val(hdr)
    if A.shape[1:] != (nk, nb - n_val, n_val):
        raise ValueError(f"GATE td_projection_window: got: eigenvectors (nk, nc, nv) = "
                         f"{A.shape[1:]}; want: {(nk, nb - n_val, n_val)}; fix: solve the "
                         f"BSE on the TD window and k grid")
    from tdagw.td_system import eigenvector_kpts
    hk = np.asarray(hdr["kpts"], float)
    dk = eigenvector_kpts(params, hk.shape[0]) - hk
    if np.abs(dk - np.rint(dk)).max() > 1e-6:
        raise ValueError("GATE td_projection_kpoints: got: an eigenvector k list that "
                         "differs from the header's; want: the same k points in the same "
                         "order; fix: solve the BSE on the bundle the header comes from")
    return A[:n]


@timing.timed("tdagw.self_energy")
def build_self_energy(deck: TdDeck, system: TdSystem, mesh) -> SelfEnergyModel:
    """The dSigma[rho] model the deck names (``[lorrax] model``)."""
    v = deck.values
    if v["lorrax.model"] == "none":
        return zero_model()
    if v["lorrax.model"] == "isdf" and system.model is not None:
        terms = v["lorrax.isdf_terms"]
        return isdf_model(system.model, mesh, k_frac=system.hdr["kpts"],
                          rho_eq=system.ops.rho_eq, sex=terms in ("both", "sex"),
                          hartree=terms in ("both", "hartree"),
                          tda_n_val=_n_val(system.hdr) if v["lorrax.tda"] else None)
    if v["lorrax.model"] == "isdf":
        from tdagw.model import load_td_model
        terms = v["lorrax.isdf_terms"]
        model = load_td_model(
            v["lorrax.isdf_input"], mesh,
            bands=(int(v["system.ibndmin"]) - 1, int(v["system.ibndmax"])),
            wfn_file=os.path.join(v["control.files.indir"], "WFN.h5"),
            zeta_file=v["lorrax.isdf_zeta"] or None,
            n_occ=None if v["system.ivbm"] is None else int(v["system.ivbm"]))
        return isdf_model(model, mesh, k_frac=system.hdr["kpts"],
                          rho_eq=system.ops.rho_eq, sex=terms in ("both", "sex"),
                          hartree=terms in ("both", "hartree"),
                          tda_n_val=_n_val(system.hdr) if v["lorrax.tda"] else None)
    path = v["lorrax.kernel_file"] or os.path.join(v["control.files.indir"],
                                                   "kernel.h5")
    if v["system.ivbm"] is None:
        raise ValueError("GATE td_ivbm: got: no [system] ivbm with a kernel "
                         "model; want: the 1-based VBM index; fix: set ivbm")
    kern, tda, cs, vs, kpts = tdgw_io.read_bgw_kernel(
        path, ivbm=int(v["system.ivbm"]), bands=system.bands)
    if kern.shape[0] != system.nk or not np.allclose(
            np.remainder(kpts - system.hdr["kpts"] + 0.5, 1.0) - 0.5, 0.0,
            atol=1e-8):
        raise ValueError(
            f"GATE td_kernel_kpoints: got: {kern.shape[0]} kernel k points "
            f"(or a different k list) for {system.nk} header k; want: the "
            f"header's k list; fix: interpolate the kernel on the WFN k grid")
    kmat = tp.place_kernel(dense_kernel_from_tdgw_layout(kern, system.nk_pad), mesh)
    if tda:
        return dense_kernel_model(kmat, tda=True, cslice=cs, vslice=vs)
    return dense_kernel_model(kmat, tda=False, rho_eq=system.ops.rho_eq)


class TdOutputs(NamedTuple):
    """Per-trial output files and the measurement row bookkeeping."""
    paths: tuple
    meas: tuple          # (name, type, kwargs, itrange)
    mtsteps: np.ndarray


def _brange_slice(rng, bands):
    """TDGW brange (1-based, [start, stop)) -> window positions."""
    start, stop, _ = bands
    lo = start if (rng is None or rng[0] is None) else int(rng[0])
    hi = stop if (rng is None or len(rng) < 2 or rng[1] is None) else int(rng[1])
    if not start <= lo < hi <= stop:
        raise ValueError(f"GATE td_brange: got: brange [{lo}, {hi}); want: "
                         f"inside the window [{start}, {stop}); fix: correct it")
    return (lo - start, hi - start)


def sample_spec(deck: TdDeck, system: TdSystem) -> tp.SampleSpec:
    """The per-sample measurement list for the propagator."""
    kinds = []
    for name, typ, kw in deck.measurements:
        opts = {}
        if typ in ("density_matrix", "lessG"):
            if kw.get("krange") not in (None, [None]):
                raise ValueError("GATE td_krange: got: a krange; want: all k; "
                                 "fix: drop krange (slice the output instead)")
            opts["bslice1"] = _brange_slice(kw.get("brange1"), system.bands)
            opts["bslice2"] = _brange_slice(kw.get("brange2"), system.bands)
        if typ == "current":
            opts["use_v0"] = bool(kw.get("use_v0", False))
            opts["inter_only"] = bool(kw.get("inter_only", False))
        if typ == "exciton_projection":
            opts["n_val"] = _n_val(system.hdr)
        kinds.append((name, typ, tuple(sorted(opts.items()))))
    return tp.SampleSpec(kinds=tuple(kinds))


@timing.timed("tdagw.outputs")
def open_outputs(deck: TdDeck, system: TdSystem) -> TdOutputs:
    """``tdgw.h5`` + ``tdgw.toml`` per trial (outdir, or outdir/trial-<i>)."""
    v = deck.values
    t0, tf, mdt = (v["control.timesteps.t0"], v["control.timesteps.tf"],
                   v["control.timesteps.dt_measure"])
    mtsteps = np.arange(t0, tf - 1e-8 * mdt, mdt)       # TDGW BaseDriver
    outdir = v["control.files.outdir"]
    ntr = len(deck.trials)
    dirs = [outdir] if ntr == 1 else [os.path.join(outdir, f"trial-{i}")
                                      for i in range(ntr)]
    meas = []
    shapes = []
    for name, typ, kw in deck.measurements:
        itr = range(*kw["itrange"]) if "itrange" in kw else range(mtsteps.size)
        meas.append((name, typ, kw, itr))
        if typ == "Efield":
            shapes.append((name, mtsteps[itr.start:itr.stop:itr.step], (3,), np.float64))
        elif typ in ("polarization", "current"):
            shapes.append((name, mtsteps[itr.start:itr.stop:itr.step], (3,), np.complex128))
        elif typ == "carrier_density":
            shapes.append((name, mtsteps[itr.start:itr.stop:itr.step], (2,), np.float64))
        elif typ == "exciton_projection":
            n_states = int(system.ops.proj.shape[0])
            shapes.append((name, mtsteps[itr.start:itr.stop:itr.step], (n_states,),
                           np.complex128))
        else:
            b1 = _brange_slice(kw.get("brange1"), system.bands)
            b2 = _brange_slice(kw.get("brange2"), system.bands)
            shapes.append((name, mtsteps[itr.start:itr.stop:itr.step],
                           (system.nk, b1[1] - b1[0], b2[1] - b2[0]), np.complex128))
    if process_rank() == 0:
        for i, d in enumerate(dirs):
            os.makedirs(d, exist_ok=True)
            tdgw_io.init_output(os.path.join(d, "tdgw.h5"), system.hdr, t0=t0,
                                tf=tf, mdt=mdt, mtsteps=mtsteps, measurements=shapes)
            cfg = {k: (dict(val) if isinstance(val, dict) else val)
                   for k, val in deck.tdgw.items()}
            sys_tab = dict(cfg.get("system", {}))
            sys_tab["Efunc"] = dict(deck.trials[i])
            cfg["system"] = sys_tab
            tdgw_io.dump_toml(os.path.join(d, "tdgw.toml"), cfg)
    return TdOutputs(paths=tuple(os.path.join(d, "tdgw.h5") for d in dirs),
                     meas=tuple(meas), mtsteps=mtsteps)


def _write_chunk(outs: TdOutputs, system: TdSystem, i0, i1, samples, state_fn,
                 t_state, fingerprint: str):
    """Write measurement rows [i0, i1) and the checkpoint, process 0 only.

    The checkpoint holds TDGW's fields in every trial's file and the
    integrator's whole carry (logical k) in the first trial's file.
    """
    rho, leaves, step = state_fn()                       # every rank gathers
    if process_rank() != 0:
        return
    nk = system.nk
    logical = [np.asarray(a) if np.ndim(a) < 3 else np.take(np.asarray(a), np.arange(nk),
                                                            axis=np.ndim(a) - 3)
               for a in leaves]
    for tr, path in enumerate(outs.paths):
        rows = {}
        for name, typ, kw, itr in outs.meas:
            sel = [i for i in range(i0, i1) if i in itr]
            if not sel:
                continue
            data = np.asarray(samples[name])[np.asarray(sel) - i0, tr]
            if typ in ("density_matrix", "lessG"):
                data = data[:, :system.nk]
            rows[name] = ((sel[0] - itr.start) // itr.step, data)
        tdgw_io.write_samples(path, rows)
        carry = (dict(leaves=logical, step=step, fingerprint=fingerprint)
                 if tr == 0 else None)
        tdgw_io.write_checkpoint(path, it_next=i1, tchkp=t_state,
                                 lessG=1j * np.asarray(rho)[tr, :system.nk], carry=carry)


def restart_fingerprint(deck: TdDeck, system: TdSystem, model: SelfEnergyModel,
                        dt=None) -> str:
    """The settings a checkpoint's carry is valid for, as one JSON string.

    Everything that changes the arithmetic of a step or the sample grid:
    integrator, step and controller, run window (tf only for dp54, whose
    last step is clipped at tf), drives, dissipation, intraband term,
    window, k list size, header source, self-energy model, measurements.
    The chunk length is not in it (chunking does not change a value).
    ``dt`` is the step the run uses (``lorrax.dt = 'auto'`` resolved).
    """
    import json
    v = dict(deck.values)
    if dt is not None:
        v["lorrax.dt"] = float(dt)
    keys = ["lorrax.method", "lorrax.dt", "lorrax.n_corr", "lorrax.predictor", "control.ivp.atol",
            "control.ivp.rtol", "control.timesteps.max_step", "control.timesteps.t0",
            "control.timesteps.dt_measure", "system.depopulation", "system.dephasing",
            "system.esmear", "system.geometry", "system.ibndmin", "system.ibndmax",
            "lorrax.intraband", "lorrax.system", "lorrax.bundle_input",
            "control.files.load_header_from", "lorrax.stiff_modes", "lorrax.stiff_krylov",
            "lorrax.stiff_tol", "lorrax.stiff_iter"]
    if v["lorrax.method"] == "dp54":
        keys.append("control.timesteps.tf")
    fp = {k: (repr(v[k]) if isinstance(v[k], float) else v[k]) for k in keys}
    fp["lorrax.magnus_frame"] = magnus_frame(v)
    if v.get("lorrax.collision", "off") != "off":     # absent when off: those carries unchanged
        fp["collision"] = {k: (repr(v[k]) if isinstance(v[k], float) else v[k]) for k in (
            "lorrax.collision", "lorrax.collision_source", "lorrax.collision_hermitian",
            "lorrax.collision_refresh_fs", "lorrax.collision_refresh_df",
            "lorrax.collision_refresh_sweep", "lorrax.collision_screening",
            "lorrax.collision_frozen_bands", "lorrax.collision_screening_gamma_ev",
            "lorrax.collision_rebase", "lorrax.collision_spm_support_sites_ev")}
        if v.get("lorrax.collision_spm_chi0_active") is not None:   # absent when unset
            fp["collision"]["lorrax.collision_spm_chi0_active"] = v["lorrax.collision_spm_chi0_active"]
        for key, default in (("lorrax.collision_in_scattering", "populations"),
                             ("lorrax.collision_rate_intermediates", "all"),
                             ("lorrax.collision_basis", "band"),
                             ("lorrax.collision_internal_pairing", "symmetric")):
            if v.get(key, default) != default:                       # absent at the default
                fp["collision"][key] = v[key]
        for key in ("collision_snapshot_steps", "collision_snapshot_tol_ev",
                    "collision_snapshot_mixing", "collision_snapshot_acceleration", "collision_hamiltonian", "collision_statistics", "collision_density_rank",
                    "collision_pole_order", "collision_projection_order", "collision_pencil_mb",
                    "collision_fit_samples", "collision_fit_real_axis", "collision_fit_tolerance",
                    "collision_adiabatic_ev", "collision_fit_height_ev"):
            fp["collision"][f"lorrax.{key}"] = v[f"lorrax.{key}"]
    fp.update(trials=[dict(t) for t in deck.trials], model=model.name,
              nk=int(system.nk), nb=int(system.nb), nspinor=int(system.nspinor),
              measurements=[[n, t, {k: kw[k] for k in sorted(kw)}]
                            for n, t, kw in deck.measurements])
    return json.dumps(fp, sort_keys=True, default=str)


def load_checkpoint(deck: TdDeck, system: TdSystem, model: SelfEnergyModel,
                    outs: TdOutputs, dt=None):
    """The resume record of ``[control.files] load_checkpoint_from``, or None.

    Reads the checkpointed run's ``tdgw.h5`` (the first trial's file holds
    the carry), refuses a carry made under other settings
    (:func:`restart_fingerprint`), copies the measurement rows already
    taken into this run's files (process 0) and returns dict(carry, step,
    i_next) for :func:`tdagw.propagate.run_propagation`.
    """
    import json
    src = deck.values["control.files.load_checkpoint_from"]
    if not src:
        return None
    ntr = len(deck.trials)
    paths = ([os.path.join(src, "tdgw.h5")] if ntr == 1 else
             [os.path.join(src, f"trial-{i}", "tdgw.h5") for i in range(ntr)])
    rec = tdgw_io.read_checkpoint(paths[0])
    if rec["leaves"] is None or rec["it_next"] < 0:
        raise ValueError(
            f"GATE td_restart_carry: got: {paths[0]} without a LORRAX integrator carry "
            f"(it_next {rec['it_next']}); want: a checkpoint written by "
            f"tdagw.tdgw_jax; why: a TDGW checkpoint holds the state, not the integrator's "
            f"carry (dSigma history, step controller), so the run cannot continue exactly; "
            f"fix: rerun from t0")
    want = json.loads(restart_fingerprint(deck, system, model, dt))
    got = json.loads(rec["fingerprint"])
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    if diff:
        raise ValueError(
            f"GATE td_restart_settings: got: checkpoint settings differing in "
            f"{', '.join(diff)}; want: the checkpointed run's settings; why: its carry "
            f"would continue a different calculation; fix: restart with the deck of "
            f"{src} (only outdir, load_checkpoint_from, chunk and, for fixed steps, tf "
            f"may change)")
    if process_rank() == 0:
        for path_src, path_dst in zip(paths, outs.paths):
            rows = {name: len(range(itr.start, min(itr.stop, rec["it_next"]), itr.step))
                    for name, _, _, itr in outs.meas}
            tdgw_io.copy_samples(path_src, path_dst, rows)
    return dict(carry=rec["leaves"], step=rec["step"], i_next=rec["it_next"],
                tchkp=rec["tchkp"], source=src)


def magnus_frame(v) -> str:
    """The Magnus frame of a deck: [lorrax] magnus_frame, else whole_rho for magnus2."""
    return v["lorrax.magnus_frame"] or ("whole_rho" if v["lorrax.method"] == "magnus2"
                                        else "interaction")


def _magnus_step_check(cfg, system: TdSystem, model: SelfEnergyModel, drive, mesh, *,
                       auto: bool, say, tracking_reference: bool = False, n_trials: int = 1):
    """The corrected Magnus step limit on this model: refuse, or pick dt for 'auto'.

    Two tests on the model itself.  (1) σ = :func:`tdagw.propagate.coupling_radius`
    (power iteration) and the scalar recurrence's limit dt_σ =
    :func:`tdagw.propagate.magnus_dt_limit`: a step above it refuses.  (2) The
    scheme's own step, field off, from a random coherence
    (:func:`tdagw.propagate.step_growth`) at PROBE_STEP_FACTOR × dt must not grow
    (g <= 1 + PROBE_TOLERANCE): an accepted step lies a measured 25 % below the
    instability.  The recurrence treats the coupling as static; in the
    interaction picture it is not, and on the 30x30 df714 CrI3 model the real
    limit lies below dt_σ (above it on the 6x6 2v2c one).  'auto' starts at
    dt_σ and backtracks by PROBE_BACKTRACK until the probe passes (floored to
    0.01 as; magnus2: dt_measure / n).  The whole_rho frame of magnus2 has no
    recurrence (its predictor extrapolates the lab-frame dSigma): the probe
    alone decides, and 'auto' starts at dt_measure.  Returns (cfg with dt, the
    receipt): sigma, the static limit, every probe (step, growth, wall), the
    power iteration's measured walls (its program compiled ahead of time, so
    exec / calls is the wall of one dSigma call), the gate's exact dSigma
    count and wall, and the settings it judged.
    """
    T = tdgw_io.T_IN_ATTOSEC
    t_gate = time.perf_counter()
    sigma, est, power = tp.coupling_radius(None, system.ops, model, drive, mesh, t0=cfg.t0,
                                           timed=True, n_trials=n_trials)
    if not np.isfinite(sigma):
        raise ValueError(
            f"GATE td_magnus_stability: got: coupling radius sigma = {sigma} (power iteration on "
            f"the linearized dSigma commutator map, {len(est)} estimates, finite: "
            f"{int(np.sum(np.isfinite(est)))}); want: a finite radius (0 for a model without "
            f"interaction); why: the model's dSigma returns non-finite values, and every step "
            f"would propagate them; fix: check the self-energy model (kernel export, operands, "
            f"units) and its tests")
    dt_sig, y_lim = tp.magnus_dt_limit(cfg, sigma)
    if cfg.frame == "whole_rho":                       # no static limit: the probe decides
        dt_sig, y_lim = cfg.dt_measure, float("nan")
    probes = []

    def snap(dt):
        dt_as = np.floor(dt * T * 100.0) / 100.0
        if cfg.method == "magnus2":
            dt_as = cfg.dt_measure * T / np.ceil(cfg.dt_measure / (dt_as / T) - 1e-9)
        return float(dt_as) / T

    def probe(dt):
        h = tp.PROBE_STEP_FACTOR * dt
        t_probe = time.perf_counter()
        if cfg.method == "cfm4" and cfg.stiff:
            g, _, fc = tp.step_growth(cfg._replace(dt=h), system.ops, model, mesh, h=h, t0=cfg.t0,
                                      counters=True, tracking_reference=tracking_reference,
                                      n_trials=n_trials)
        else:
            g, fc = tp.step_growth(cfg._replace(dt=h), system.ops, model, mesh, h=h, t0=cfg.t0,
                                  tracking_reference=tracking_reference, n_trials=n_trials)[0], None
        row = dict(dt_as=dt * T, probe_dt_as=h * T, growth=g,
                   wall_s=time.perf_counter() - t_probe)
        if fc is not None:              # the split: a probe whose closing solve failed did not pass
            row.update(fast_unconverged=fc["unconverged"], fast_max_evaluations=fc["max_evaluations"],
                       fast_worst_rel=fc["worst_rel"])
        probes.append(row)
        return g <= 1.0 + tp.PROBE_TOLERANCE and not (fc and fc["unconverged"])

    if sigma <= 0.0:                                  # no coupling: any step is stable
        if auto:
            cfg = cfg._replace(dt=cfg.dt_measure)
        stable = True
    elif auto:
        dt = snap(dt_sig)
        for _ in range(tp.PROBE_BACKTRACKS):
            if probe(dt):
                break
            dt = snap(tp.PROBE_BACKTRACK * dt)
        else:
            raise ValueError(
                f"GATE td_magnus_stability: got: no step down to {probes[-1]['dt_as']:.4g} as "
                f"whose probe at {tp.PROBE_STEP_FACTOR} dt stays bounded (growth "
                f"{probes[-1]['growth']:.4g} per step); want: a step the scheme is stable at "
                f"with a 25 % margin; why: a mode of the coupling grows from rounding, "
                f"independent of the drive; fix: set [lorrax] dt explicitly, more corrections, "
                f"or method = 'etdrk4' / 'dp54'")
        cfg = cfg._replace(dt=dt)
        stable = True
    else:
        stable = (cfg.frame == "whole_rho" or cfg.dt <= dt_sig) and probe(cfg.dt)
    n_hist = len(tp._magnus_parts(cfg)[3])
    per_probe = n_hist + tp.PROBE_STEPS * tp.dsigma_per_step(cfg.method, cfg.n_corr)
    per_probe *= 2 if tracking_reference else 1
    receipt = dict(sigma_ev=sigma * tdgw_io.HA_IN_EV, y=cfg.dt * sigma, y_lim=y_lim,
                   dt_max_as=dt_sig * T,
                   change=float(abs(est[-1] - est[-2]) / est[-1]) if sigma > 0.0 else 0.0,
                   probes=probes, auto=bool(auto), dt_as=cfg.dt * T, method=cfg.method,
                   n_corr=int(cfg.n_corr), predictor=cfg.predictor, frame=cfg.frame,
                   stiff=int(cfg.stiff), margin=tp.STABILITY_MARGIN,
                   probe_factor=tp.PROBE_STEP_FACTOR, probe_steps=tp.PROBE_STEPS,
                   probe_tolerance=tp.PROBE_TOLERANCE, power_iterations=len(est),
                   probe_tracking_reference=bool(tracking_reference), probe_trials=int(n_trials),
                   power=power, dsigma_wall_s=(power["exec_s"] / power["dsigma"]
                                               if power["dsigma"] else float("nan")),
                   dsigma=int(power["dsigma"]) + len(probes) * per_probe,
                   wall_s=time.perf_counter() - t_gate)
    if sigma <= 0.0:
        say(f"  stability       : no coupling (sigma = 0: the dSigma commutator map vanishes on "
            f"this model, below {tp.COUPLING_ZERO_FLOOR:.0e} Ha), any step is stable"
            + (f"; auto dt = {cfg.dt * T:.4g} as" if auto else ""))
        return cfg, receipt
    pr = "; ".join(f"{q['growth']:.4g} per step at {q['probe_dt_as']:.4g} as"
                   + (f" (closing solve unconverged on {q['fast_unconverged']} steps)"
                      if q.get("fast_unconverged") else "") for q in probes)
    slow = (f" of the slow map L(1 - Pi), {cfg.stiff} stiff modes split off"
            if cfg.method == "cfm4" and cfg.stiff else "")
    say(f"  stability       : sigma = {receipt['sigma_ev']:.4g} eV (spectral radius{slow} of the "
        f"dSigma commutator map, {len(est)} power iterations, last change "
        f"{100 * receipt['change']:.2g} %), h*sigma = {receipt['y']:.3g}, "
        + (f"limit {y_lim:.3g} ({cfg.method}, {cfg.predictor} predictor, n_corr {cfg.n_corr}) "
           f"-> dt <= {receipt['dt_max_as']:.4g} as" if cfg.frame != "whole_rho" else
           f"whole_rho frame (no static limit)") + (f"; probe {pr}" if probes else "")
        + (f"; auto dt = {cfg.dt * T:.4g} as" if auto else ""))
    if not stable and cfg.frame != "whole_rho" and cfg.dt > dt_sig:
        alts = []
        for nc, pre in ((cfg.n_corr + 1, cfg.predictor), (cfg.n_corr, "linear")):
            if pre == "linear" and cfg.method != "cfm4":
                continue
            d, _ = tp.magnus_dt_limit(cfg._replace(n_corr=nc, predictor=pre), sigma)
            alts.append(f"n_corr = {nc}, predictor = {pre!r}: dt <= {d * T:.4g} as")
        raise ValueError(
            f"GATE td_magnus_stability: got: dt = {cfg.dt * T:.4g} as, h*sigma = "
            f"{receipt['y']:.3g} with sigma = {receipt['sigma_ev']:.4g} eV (the spectral radius "
            f"of the linearized dSigma commutator map on this model); want: h*sigma*"
            f"{tp.STABILITY_MARGIN} <= {y_lim:.3g}, the imaginary-axis stability limit of "
            f"{cfg.method} with the {cfg.predictor} predictor and n_corr = {cfg.n_corr}; why: "
            f"a mode of the coupling grows from rounding, independent of the drive; fix: "
            f"[lorrax] dt = 'auto', or dt <= {receipt['dt_max_as']:.4g} as subject to the probe, or "
            + "; ".join(alts) + ", or method = 'etdrk4' / 'dp54'")
    if not stable and not probes:
        raise ValueError(
            f"GATE td_magnus_stability: got: dt = {cfg.dt * T:.4g} as with sigma = "
            f"{receipt['sigma_ev']:.4g} eV and a sigma limit of {receipt['dt_max_as']:.4g} as, "
            f"so no probe ran; want: a finite sigma limit to compare dt with, then the probe; "
            f"why: without either test the step is not known to be stable; fix: dt = 'auto', "
            f"another n_corr / predictor, or method = 'etdrk4' / 'dp54'")
    if not stable and probes and probes[-1].get("fast_unconverged"):
        q = probes[-1]
        raise ValueError(
            f"GATE td_stiff_solve: got: the probe of dt = {cfg.dt * T:.4g} as (the split step at "
            f"{q['probe_dt_as']:.4g} as, field off, {tp.PROBE_STEPS} steps) missed the closing-solve "
            f"tolerance {cfg.stiff_tol:.1e} on {q['fast_unconverged']} steps (worst rel. "
            f"{q['fast_worst_rel']:.3e}, growth {q['growth']:.4g} per step); want: a converged fast "
            f"part at every probe step; why: the probe would measure another scheme; fix: a smaller "
            f"dt, more stiff_iter, or stiff_modes = 0")
    if not stable:
        q = probes[-1]
        raise ValueError(
            f"GATE td_magnus_stability: got: dt = {cfg.dt * T:.4g} as passes the sigma limit "
            f"({receipt['dt_max_as']:.4g} as{'; whole_rho: none' if cfg.frame == 'whole_rho' else ''}) "
            f"but the scheme's own step at "
            f"{tp.PROBE_STEP_FACTOR} dt = {q['probe_dt_as']:.4g} as grows by {q['growth']:.4g} "
            f"per step on this model (field off, {tp.PROBE_STEPS} steps from a random "
            f"coherence); want: growth <= {1.0 + tp.PROBE_TOLERANCE:.4g} at {tp.PROBE_STEP_FACTOR} "
            f"dt, a 25 % margin; why: in the interaction picture the coupling is time dependent, "
            f"so the static limit can overestimate the stable step, and a mode grows from "
            f"rounding independent of the drive; fix: [lorrax] dt = 'auto' (the largest step "
            f"that passes), a smaller dt, or method = 'etdrk4' / 'dp54'")
    return cfg, receipt


def collision_source(deck: TdDeck, mesh, *, say=None):
    """The GW source of ``[lorrax] collision`` (None when off): a GN-PPM or shared-pole run.

    Runs the GW driver on a copy of ``collision_source`` in
    ``outdir/collision_source`` (``tdagw.dynsig.gw_window_source``: the run's own
    carrier, screening model and head, and with refreshed or shared-pole screening
    the inputs of its W stage; no restart written).  Collective; call it before
    the TD driver's ``RunSession`` (the GW driver owns the timing table and
    stdout while it runs).  ``say`` gets its one line (None: kept in the
    returned dict's ``line`` for the run log).
    """
    v = deck.values
    if v["lorrax.collision"] == "off":
        return None
    from tdagw.dynsig import gw_window_source
    lines = []
    work = os.path.join(os.path.abspath(str(v["control.files.outdir"])), "collision_source")
    src = gw_window_source(v["lorrax.collision_source"], mesh, workdir=work,
                           screening=v["lorrax.collision_screening"] in (
                               "refreshed", "shared_pole", "joint", "joint_frozen"),
                           say=say or lines.append)
    src["line"] = lines[0] if lines else ""
    return src


def _attach_collision(deck: TdDeck, system: TdSystem, model: SelfEnergyModel, cfg, source,
                      mesh, *, say, log):
    """The held collision term on the operators (``ops.collision``) and the model (§1.5).

    ``tdagw.dynsig.window_collision`` at ρ_eq: the dissipator of the equilibrium
    rates (``ops.collision``; the Magnus steps' remainder, ``cfg.remainder``),
    the held ΔΣ^h (zero at t0; ``tdagw.self_energy.with_held_term``) and the
    refresh hook.  Its stdout lines go to ``log.refresh`` (the log's
    ``Collision refreshes`` section) or ``say``.  Returns (built, system,
    model, cfg).
    """
    from tdagw import collision as C
    from tdagw.dynsig import window_collision
    from tdagw.self_energy import with_held_term
    v = deck.values
    emit = say if log is None else log.refresh
    if source.get("line"):
        emit(source["line"])
    policy = C.RefreshPolicy(every=float(v["lorrax.collision_refresh_fs"]),
                             df=float(v["lorrax.collision_refresh_df"]))
    rebase = {"auto": None, "on": True, "off": False}[v["lorrax.collision_rebase"]]
    if v["lorrax.collision_screening"] in ("joint", "joint_frozen"):
        from tdagw.snapshot_runtime import snapshot_collision
        options = dict(
            screening=v["lorrax.collision_screening"], form=v["lorrax.collision"],
            statistics=v["lorrax.collision_statistics"],
            hermitian=v["lorrax.collision_hermitian"], policy=policy,
            basis=v["lorrax.collision_basis"],
            hamiltonian=v["lorrax.collision_hamiltonian"],
            spm_support_sites=v["lorrax.collision_spm_support_sites_ev"],
            spm_scratch=os.path.join(os.path.abspath(str(
                v["control.files.outdir"])), "collision_models"),
            fit_tolerance=v["lorrax.collision_fit_tolerance"],
            adiabatic_ev=v["lorrax.collision_adiabatic_ev"],
            snapshot_steps=v["lorrax.collision_snapshot_steps"],
            snapshot_tol_ev=v["lorrax.collision_snapshot_tol_ev"],
            snapshot_mixing=v["lorrax.collision_snapshot_mixing"],
            snapshot_acceleration=v["lorrax.collision_snapshot_acceleration"],
            eta_ev=float(source["config"].sigma.regularization_ev))
        built = snapshot_collision(source, system.ops, mesh=mesh,
                                   k_frac=system.hdr["kpts"], b_lo=int(system.bands[0]) - 1,
                                   ntr=len(deck.trials), model=model, options=options, say=emit)
    else:
        built = window_collision(source, system.ops, mesh=mesh, k_frac=system.hdr["kpts"],
                             b_lo=int(system.bands[0]) - 1, ntr=len(deck.trials),
                             form=v["lorrax.collision"], hermitian=v["lorrax.collision_hermitian"],
                             policy=policy, sweep=v["lorrax.collision_refresh_sweep"],
                             screening=v["lorrax.collision_screening"],
                             frozen_bands=v["lorrax.collision_frozen_bands"],
                             screening_gamma_ev=v["lorrax.collision_screening_gamma_ev"],
                             rebase=rebase,
                             spm_support_sites=v["lorrax.collision_spm_support_sites_ev"],
                             spm_chi0_active=v["lorrax.collision_spm_chi0_active"],
                             in_scattering=v["lorrax.collision_in_scattering"],
                             rate_intermediates=v["lorrax.collision_rate_intermediates"],
                             basis=v["lorrax.collision_basis"],
                             pairing=v["lorrax.collision_internal_pairing"], model=model,
                             snapshot_steps=v["lorrax.collision_snapshot_steps"],
                             snapshot_tol_ha=v["lorrax.collision_snapshot_tol_ev"] / tdgw_io.HA_IN_EV,
                             snapshot_mixing=v["lorrax.collision_snapshot_mixing"],
                             snapshot_acceleration=v["lorrax.collision_snapshot_acceleration"],
                             spm_scratch=os.path.join(os.path.abspath(str(
                                 v["control.files.outdir"])), "collision_refits"), say=emit)
    every = policy.every * tdgw_io.T_IN_ATTOSEC / 1000.0
    fb = v["lorrax.collision_frozen_bands"]
    say(f"  collision       : {v['lorrax.collision']}, hermitian "
        f"{v['lorrax.collision_hermitian']}, {v['lorrax.collision_refresh_sweep']} sweep, "
        f"{v['lorrax.collision_screening']} screening, in-scattering "
        f"{v['lorrax.collision_in_scattering']}, rate intermediates "
        f"{v['lorrax.collision_rate_intermediates']}"
        + ("" if v["lorrax.collision_basis"] == "band" else
           f", {v['lorrax.collision_basis']} basis")
        + ("" if v["lorrax.collision_internal_pairing"] == "symmetric" else
           f", {v['lorrax.collision_internal_pairing']} internal pairing")
        + ("" if v["lorrax.collision_screening"] != "refreshed" else
           f" (same-side principal value {v['lorrax.collision_screening_gamma_ev']:g} eV)")
        + ("" if v["lorrax.collision_screening"] != "shared_pole" else
           f" (full refit every refresh; supports "
           f"{v['lorrax.collision_spm_support_sites_ev'] or 'the production ladder'}"
           + ("" if v["lorrax.collision_spm_chi0_active"] is None else
              f"; chi0 in four blocks, active bands {v['lorrax.collision_spm_chi0_active']}")
           + ")")
        + ("" if v["lorrax.collision"] == "raw" else
           f"; equilibrium counterterm {'rebased on W(t)' if built.get('rebase') else 'fixed at t0'}")
        + ("" if fb is None else f" (frozen change over {fb[0]} + {fb[1]} bands)") + "; refresh "
        + (" or ".join(([f"every {every:g} fs"] if policy.every else [])
                       + ([f"max|df| >= {policy.df:g}"] if policy.df else []))
           or "never (rates of rho_eq)")
        + ("; the exact held flow in the remainder" if cfg.method in ("magnus2", "cfm4")
           else "; its generator in the right-hand side"))
    system = system._replace(ops=system.ops._replace(collision=built["diss"]))
    model = built.get("model", model)
    if built["held"] is not None:
        model = with_held_term(model, built["held"])
    return built, system, model, cfg._replace(remainder=True)


def propagate(deck: TdDeck, system: TdSystem, model: SelfEnergyModel, drive,
              outs: TdOutputs, mesh, *, say=print, stiff=None, log=None,
              collision=None) -> dict:
    """Run the propagation from t0, or from the checkpoint the deck names.

    ``collision`` (:func:`collision_source`, or None) adds the held on-shell
    collision term of ``[lorrax] collision`` after the stability gate (the gate
    measures the coherent problem, unchanged; :func:`_attach_collision`) and
    refreshes it at chunk ends by the deck's policy.

    ``[control.files] load_checkpoint_from`` continues that run exactly
    (:func:`load_checkpoint`: its rows are copied, its carry resumed).
    ``[lorrax] stiff_modes`` > 0 builds the stiff split's basis
    (:func:`tdagw.propagate.stiff_basis`) before the stability gate, or
    takes ``stiff`` (its dict, already built for this system and model, of
    the deck's stiff_modes) so several runs of one system pay it once.
    ``say`` gets the stdout lines (the stiff split, the gate's verdict, one
    line per chunk).  ``log`` (the run's ``tdagw.td_log.TdReport``, or None)
    gets the structured record: ``hold()`` before the split and the gate,
    ``integrator(...)`` after them (the self-energy and gate blocks, the
    progress table's header), ``chunk(line, row)`` per chunk (the stdout line
    and the table row), each row computed from the chunk's host samples and
    counters (:data:`tdagw.propagate.PROGRESS`, ``COUNTERS``) with no device
    work of its own.  Returns counters, walls (seconds), the gate's receipt,
    the basis receipt and the progress rows.
    """
    v = deck.values
    ntr = len(deck.trials)
    rho0 = np.zeros((ntr, system.nk_pad, system.nb, system.nb), complex)
    idx = np.arange(system.nb)
    rho0[:, :system.nk, idx, idx] = np.asarray(system.hdr["f_kn"], float)
    nt = outs.mtsteps.size
    chunk = int(v["lorrax.chunk"]) or int(v["control.timesteps.ntcheck"])
    from runtime.tiles import TILE_BYTES
    dm_bytes = sum(ntr * system.nk_pad * system.nb * system.nb * 16
                   for _, typ, _, _ in outs.meas if typ in ("density_matrix", "lessG"))
    if dm_bytes:
        chunk = max(1, min(chunk, int(TILE_BYTES // dm_bytes)))
    auto = v["lorrax.dt"] == "auto"
    cfg = tp.PropagatorConfig(
        method=v["lorrax.method"], t0=v["control.timesteps.t0"],
        tf=v["control.timesteps.tf"], dt_measure=v["control.timesteps.dt_measure"],
        dt=float(v["control.timesteps.dt_measure"] if auto or not v["lorrax.dt"]
                 else v["lorrax.dt"]),
        n_corr=int(v["lorrax.n_corr"]), predictor=v["lorrax.predictor"], frame=magnus_frame(v),
        atol=float(v["control.ivp.atol"]), rtol=float(v["control.ivp.rtol"]),
        max_step=float(v["control.timesteps.max_step"]), chunk=min(chunk, nt),
        remainder=bool(v["system.depopulation"] != 0.0
                       or v["system.dephasing"] != 0.0
                       or system.ops.intra is not None),
        stiff=int(v["lorrax.stiff_modes"]), stiff_tol=float(v["lorrax.stiff_tol"]),
        stiff_iter=int(v["lorrax.stiff_iter"]))
    basis = None
    basis_wall = 0.0
    if log is not None:
        log.hold()                 # the split and gate lines: stdout now, the log after the gate
    coll = None
    joint_collision = (collision is not None
                       and v["lorrax.collision_screening"] in ("joint", "joint_frozen"))
    if joint_collision:
        # This adapter replaces screened exchange by bare HF and can install
        # an absolute material Hamiltonian. Probe that actual coherent model,
        # not the original TD-aGW kernel; exclude dissipative gain from the probe.
        with timing.section("tdagw.collision"):
            coll, system, model, cfg = _attach_collision(
                deck, system, model, cfg, collision, mesh, say=say, log=log)
    if cfg.stiff:
        if not model.linear:
            raise ValueError(f"GATE td_stiff_model: got: stiff_modes = {cfg.stiff} with the "
                             f"model {model.name!r} (not affine in rho - rho_eq); want: an affine "
                             f"dSigma; why: the split dSigma[x] = dSigma[(1 - Pi) x] + sum_j <u_j, x> "
                             f"b_j is exact only for a linear model; fix: stiff_modes = 0")
        t_basis = time.perf_counter()
        with timing.section("tdagw.stiff_basis"):
            basis = stiff if stiff is not None else tp.stiff_basis(
                system.ops, model, mesh, t0=cfg.t0, n_s=cfg.stiff,
                krylov=int(v["lorrax.stiff_krylov"]))
        basis_wall = time.perf_counter() - t_basis
        system = system._replace(ops=system.ops._replace(stiff=(basis["u"], basis["b"])))
        cfg = cfg._replace(stiff=int(basis["n_s"]))
        top = np.abs(basis["ritz_ev"])
        say(f"  stiff split     : {basis['n_s']} modes of the coupling map's invariant subspace "
            f"(|lambda| {top[basis['n_s'] - 1]:.4g} .. {top[0]:.4g} eV above the cut "
            f"{basis['cut_ev']:.4g} eV; next {top[basis['n_s']]:.4g} eV), {basis['krylov']} real "
            f"Arnoldi steps, invariance residual {basis['residual_ev']:.2e} eV; "
            f"{basis['dsigma'] if stiff is None else 0} dSigma calls to build"
            + ("" if stiff is None else " (reused)"))
    check = None
    if cfg.method in ("magnus2", "cfm4"):
        with timing.section("tdagw.stability_gate"):
            gate_system = (system._replace(ops=system.ops._replace(collision=None))
                           if joint_collision else system)
            cfg, check = _magnus_step_check(cfg, gate_system, model, drive, mesh,
                                           auto=auto, say=say, tracking_reference=joint_collision,
                                           n_trials=ntr if joint_collision else 1)
    if collision is not None and coll is None:
        with timing.section("tdagw.collision"):
            coll, system, model, cfg = _attach_collision(deck, system, model, cfg, collision,
                                                         mesh, say=say, log=log)
    resume = load_checkpoint(deck, system, model, outs, cfg.dt)
    count = ntr * system.nk * system.nb * system.nb
    fingerprint = restart_fingerprint(deck, system, model, cfg.dt)
    basis_info = None if basis is None else dict(
        n_s=int(basis["n_s"]), krylov=int(basis["krylov"]), cut_ev=float(basis["cut_ev"]),
        residual_ev=float(basis["residual_ev"]), moduli_ev=np.abs(basis["ritz_ev"]).tolist(),
        dsigma=0 if stiff is not None else int(basis["dsigma"]), reused=stiff is not None,
        wall_s=basis_wall)
    if log is not None:
        log.integrator(deck=deck, system=system, model=model, cfg=cfg, gate=check,
                       basis=basis_info, resume=resume)
    walls = []
    rows = []
    t_start = time.perf_counter()
    progress = _progress_reference(deck, system, outs, cfg, resume)

    def on_chunk(i0, i1, samples, state_fn, t_state):
        _write_chunk(outs, system, i0, i1, samples, state_fn, t_state, fingerprint)
        walls.append(time.perf_counter())
        line = (f"  samples {i0:7d}-{i1 - 1:7d}  t = {t_state * tdgw_io.T_IN_ATTOSEC:12.3f} as"
                f"  wall {walls[-1] - t_start:9.2f} s")
        rows.append(_progress_row(progress, rows, i0, i1, samples, t_state,
                                  walls[-1] - t_start))
        if log is None:
            say(line)
        else:
            log.chunk(line, rows[-1])

    with timing.section("tdagw.propagate"):
        stats, _ = tp.run_propagation(
            cfg, sample_spec(deck, system), tp.place_state(rho0, mesh),
            system.ops, model, drive, mesh, cell_volume=system.cell_volume,
            nspinor=system.nspinor, count=count, mtsteps=outs.mtsteps,
            on_chunk=on_chunk, resume=resume,
            refresh=None if coll is None else coll["refresh"])
    stats["check_dsigma"] = 0 if check is None else int(check["dsigma"])
    stats["check_probes"] = [] if check is None else check["probes"]
    stats["gate"] = check
    stats["basis"] = basis_info
    stats["rows"] = rows
    stats["stiff_dsigma"] = 0 if (basis is None or stiff is not None) else int(basis["dsigma"])
    stats["stiff_modes"] = int(cfg.stiff) if cfg.method == "cfm4" else 0
    stats["collision"] = None if coll is None else dict(
        t0=coll["t0"], history=coll["state"]["history"], source_wall_s=collision["wall_s"],
        form=v["lorrax.collision"], hermitian=v["lorrax.collision_hermitian"])
    stats["dt"] = cfg.dt
    stats["wall"] = time.perf_counter() - t_start
    stats["wall_first_chunk"] = (walls[0] - t_start) if walls else 0.0
    stats["chunk"] = cfg.chunk
    t_from = cfg.t0 if resume is None else float(resume["tchkp"])
    stats["simulated_fs"] = (stats["t_final"] - t_from) * tdgw_io.T_IN_ATTOSEC / 1000.0
    return stats


def _progress_reference(deck: TdDeck, system: TdSystem, outs: TdOutputs, cfg, resume) -> dict:
    """What the progress rows are measured against: the t0 invariants, the run's span.

    tr0 = Σ_k w_k Σ_n f_kn and pur0 = Σ_k w_k Σ_n f_kn² are Tr ρ and Tr ρ² of
    the equilibrium state every trial starts from (host, exact); the rows
    report |X(t) - X0| / X0 for both.  ``hamiltonian``: no Γ, so both are
    invariants of the exact flow (the intraband term moves charge between k
    but keeps the sums).
    """
    f = np.asarray(system.hdr["f_kn"], float)
    w = np.asarray(system.hdr["w_k"], float)
    v = deck.values
    names = {typ: name for name, typ, _ in reversed(deck.measurements)}
    t_from = cfg.t0 if resume is None else float(resume["tchkp"])
    return dict(tr0=float(np.sum(w[:, None] * f)), pur0=float(np.sum(w[:, None] * f * f)),
                hamiltonian=v["system.depopulation"] == 0.0 and v["system.dephasing"] == 0.0,
                polarization=names.get("polarization"), carriers=names.get("carrier_density"),
                t_from_fs=t_from * tdgw_io.T_IN_ATTOSEC / 1000.0,
                t_end_fs=cfg.tf * tdgw_io.T_IN_ATTOSEC / 1000.0)


def _progress_row(ref: dict, rows: list, i0, i1, samples, t_state, wall) -> dict:
    """One progress row from a chunk's host samples (no device work).

    Rate s/fs of this chunk; ETA from the mean rate of the chunks that ran an
    already-built program (``new_program``: a chunk of a new step or sample
    count, the first always, compiles in its own wall), or this one's; the ΔΣ
    count of the run so far (steps and samples); |P|, n_c and |E| at the
    chunk's last sample (max over drives; P and n_c only when measured); the
    largest relative drift of Tr ρ and Tr ρ² over the chunk's samples and
    drives.
    """
    t_fs = t_state * tdgw_io.T_IN_ATTOSEC / 1000.0
    prev_t, prev_w = (rows[-1]["t_fs"], rows[-1]["wall_s"]) if rows else (ref["t_from_fs"], 0.0)
    span = max(t_fs - prev_t, 1e-30)
    rate = (wall - prev_w) / span
    cnt = samples[tp.COUNTERS]
    new = bool(cnt.get("new_program", not rows))
    steady = [r["s_per_fs"] for r in rows if not r["new_program"]] + ([] if new else [rate])
    eta_rate = float(np.mean(steady)) if steady else rate
    prog = np.asarray(samples[tp.PROGRESS], float)                   # (n, ntr, 3)
    row = dict(i0=int(i0), i1=int(i1), t_fs=t_fs, wall_s=float(wall), s_per_fs=rate,
               eta_s=max(ref["t_end_fs"] - t_fs, 0.0) * eta_rate,
               dsigma=int(cnt["dsigma"]) + int(cnt.get("sample_dsigma", 0)),
               steps=int(cnt["accepted"]), rejected=int(cnt["rejected"]), new_program=new,
               p_abs=None, n_c=None, e_abs=None, d_trace=None, d_purity=None)
    if prog.shape[0]:
        row.update(e_abs=float(np.max(prog[-1, :, 2])) * tdgw_io.EFIELD_IN_VPERM,
                   d_trace=float(np.max(np.abs(prog[..., 0] - ref["tr0"])) / abs(ref["tr0"])),
                   d_purity=float(np.max(np.abs(prog[..., 1] - ref["pur0"])) / abs(ref["pur0"])))
        if ref["polarization"] is not None:
            p = np.asarray(samples[ref["polarization"]])[-1]          # (ntr, 3)
            row["p_abs"] = float(np.max(np.sqrt(np.sum(np.abs(p) ** 2, axis=-1))))
        if ref["carriers"] is not None:
            row["n_c"] = float(np.max(np.asarray(samples[ref["carriers"]])[-1, :, 0]))
    return row


def describe(deck: TdDeck, system: TdSystem, model: SelfEnergyModel, runtime) -> str:
    """The run header: what will be integrated, how, where (stdout)."""
    return "\n".join(describe_rows(deck, system, model, runtime).values())


def describe_rows(deck: TdDeck, system: TdSystem, model: SelfEnergyModel, runtime) -> dict:
    """The run header's lines by name, in stdout order (``describe`` joins them).

    Names: title, deck, header, window, integrator, self_energy, intraband,
    drives, time, mesh, then td_system and gauge (bundle route), restart and
    unused when they apply.  The run log (``tdagw.td_log``) files each line
    under its section.
    """
    v = deck.values
    T = tdgw_io.T_IN_ATTOSEC
    step = ("adaptive (SciPy RK45 controller), atol %.3g rtol %.3g, first step "
            "dt_measure" % (v["control.ivp.atol"], v["control.ivp.rtol"])
            if v["lorrax.method"] == "dp54" else
            "fixed dt = auto (the stability gate picks it below)" if v["lorrax.dt"] == "auto" else
            "fixed dt = %.6g as (dt_measure / dt = %.6g)" % (
                v["lorrax.dt"] * T, v["control.timesteps.dt_measure"] / v["lorrax.dt"]))
    rows = dict(
        title="TD-aGW propagation (TDGW conventions)",
        deck=f"  deck            : {deck.source}",
        header=f"  header          : {system.hdr['path'] or 'built from the GW bundle ' + str(v['lorrax.bundle_input'])}",
        window=f"  window          : bands {system.bands[0]}..{system.bands[1] - 1}, "
        f"nb = {system.nb}, nk = {system.nk} (padded {system.nk_pad}), "
        f"nspinor = {system.nspinor}",
        integrator=f"  integrator      : {v['lorrax.method']}, {step}"
        + (f", n_corr = {v['lorrax.n_corr']}" if v["lorrax.method"] in ("magnus2", "cfm4") else ""),
        self_energy=f"  self-energy     : {model.name}",
        intraband=f"  intraband [r,.] : {'on' if system.ops.intra is not None else 'off'}",
        drives=f"  drives          : {len(deck.trials)}",
        time=f"  window in time  : {v['control.timesteps.t0'] * T:.3f} .. "
        f"{v['control.timesteps.tf'] * T:.3f} as, samples every "
        f"{v['control.timesteps.dt_measure'] * T:.6g} as",
        mesh=f"  mesh            : {runtime.mesh_shape} on {runtime.n_devices} "
        f"{runtime.platform} device(s)",
    )
    rec = system.receipts
    if rec:                                  # the bundle route's receipts (also in the header)
        pr, vr = rec["pairs"], rec["velocity"]
        rows["td_system"] = (
            f"  TD system       : {'bse_k_grid' if rec['densified'] else 'native'} bundle, "
            f"energies {rec['energy_source']}, r = v/(i dE_DFT) from {vr['route']}; "
            f"pair rule {pr['rule']} at {pr['gap_ev']:g} eV (vv' below: {pr['vv']['below_gap']} "
            f"of {pr['vv']['pairs']}, cc' below: {pr['cc']['below_gap']} of {pr['cc']['pairs']}), "
            f"min c-v gap {pr['min_cv_gap_ev']:.4f} eV, max|r| {pr['max_abs_r_bohr']:.3g} bohr")
        if "bse_dipoles" in rec:
            bd = rec["bse_dipoles"]
            rows["gauge"] = (
                f"  gauge receipt   : <0|r|S> of {bd['n_states']} BSE states from the header r "
                f"and the eigenvectors vs bse_jax's: rel dev {bd['dipole_rel_dev']:.2e}, "
                f"I+ {bd['i_plus_rel_dev']:.2e}, I- {bd['i_minus_rel_dev']:.2e}")
    if v["control.files.load_checkpoint_from"]:
        rows["restart"] = (f"  restart from    : {v['control.files.load_checkpoint_from']} "
                           f"(its integrator carry; values continue bitwise)")
    unused = td_config.unused_keys(deck)
    if unused:
        rows["unused"] = ("  read, not used  : " + ", ".join(unused)
                          + " (TDGW logging / vacuum centring / snapshot stride)")
    return rows


def output_files(outs: TdOutputs) -> list:
    """The file table of a finished run: each trial's ``tdgw.h5`` and ``tdgw.toml``."""
    return ([("tdgw.h5", "written", p) for p in outs.paths]
            + [("tdgw.toml", "written", os.path.join(os.path.dirname(p), "tdgw.toml"))
               for p in outs.paths])


def summarize(deck: TdDeck, stats: dict) -> str:
    """Counters and walls of the finished propagation."""
    fs = max(stats["simulated_fs"], 1e-30)
    lines = [
        "propagation done",
        f"  steps           : {stats['accepted']} accepted, {stats['rejected']} rejected",
        f"  dSigma calls    : {stats['dsigma']} in steps + {stats['sample_dsigma']} at samples"
        f" ({stats['dsigma'] / fs:.1f} per fs)",
        f"  wall            : {stats['wall']:.2f} s ({stats['wall'] / fs:.3f} s per fs; "
        f"first chunk incl. compile {stats['wall_first_chunk']:.2f} s)",
        f"  final time      : {stats['t_final'] * tdgw_io.T_IN_ATTOSEC:.3f} as",
    ]
    if stats.get("fast"):
        fc, n = stats["fast"], max(int(stats["accepted"]), 1)
        lines.append(f"  stiff split     : {stats['stiff_modes']} modes; closing solve "
                     f"{fc['evaluations'] / n:.2f} node-map evaluations per step (max "
                     f"{fc['max_evaluations']}), worst rel. residual {fc['worst_rel']:.2e}; "
                     f"{stats.get('stiff_dsigma', 0)} dSigma calls to build the basis")
    return "\n".join(lines)
