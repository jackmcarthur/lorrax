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
           "describe", "summarize"]


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
            "control.files.load_header_from"]
    if v["lorrax.method"] == "dp54":
        keys.append("control.timesteps.tf")
    if v["lorrax.node_solver"] != "truncated":      # converged nodes: the solve's settings
        keys += ["lorrax.node_solver", "lorrax.node_tol", "lorrax.node_memory"]
    fp = {k: (repr(v[k]) if isinstance(v[k], float) else v[k]) for k in keys}
    fp["lorrax.magnus_frame"] = magnus_frame(v)
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
                       auto: bool, say):
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
    alone decides, and 'auto' starts at dt_measure.  Converged nodes
    (``node_solver = 'anderson'``) have no static limit either (the converged
    step is A-stable on the imaginary axis): the probe decides, and it also
    solves each probe step's node equations, so a solve that does not converge
    at 1.25 dt refuses.  Returns (cfg with dt, the receipt).
    """
    T = tdgw_io.T_IN_ATTOSEC
    implicit = tp.converged_nodes(cfg)
    if implicit and auto:
        raise ValueError("GATE td_dt: got: dt = 'auto' with converged nodes; want: an explicit "
                         "dt; fix: set [lorrax] dt in as")
    sigma, est = tp.coupling_radius(None, system.ops, model, drive, mesh, t0=cfg.t0)
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
        g, _, cnt = tp.step_growth(cfg._replace(dt=h), system.ops, model, mesh, h=h,
                                   t0=cfg.t0, with_counts=True)
        q = dict(dt_as=dt * T, probe_dt_as=h * T, growth=g, dsigma=cnt["dsigma"])
        if implicit:
            q.update(n_eval=cnt["n_eval"], n_eval_max=cnt["n_eval_max"], n_fail=cnt["n_fail"])
        probes.append(q)
        return g <= 1.0 + tp.PROBE_TOLERANCE

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
    receipt = dict(sigma_ev=sigma * tdgw_io.HA_IN_EV, y=cfg.dt * sigma, y_lim=y_lim,
                   dt_max_as=dt_sig * T,
                   change=float(abs(est[-1] - est[-2]) / est[-1]) if sigma > 0.0 else 0.0,
                   probes=probes)
    if sigma <= 0.0:
        say(f"  stability       : no coupling (sigma = 0: the dSigma commutator map vanishes on "
            f"this model, below {tp.COUPLING_ZERO_FLOOR:.0e} Ha), any step is stable"
            + (f"; auto dt = {cfg.dt * T:.4g} as" if auto else ""))
        return cfg, receipt
    pr = "; ".join(f"{q['growth']:.4g} per step at {q['probe_dt_as']:.4g} as"
                   + (f" (node map {q['n_eval'] / tp.PROBE_STEPS:.2f} mean / {q['n_eval_max']} max "
                      f"evaluations per step, {q['n_fail']} unconverged)" if implicit else "")
                   for q in probes)
    if implicit:
        limit = (f"converged nodes ({cfg.node_solver}, node_tol {cfg.node_tol:.3g}, memory "
                 f"{cfg.node_memory}): no static limit; the plain node iteration would contract "
                 f"by h*sigma/(2 sqrt 3) = {receipt['y'] / (2.0 * np.sqrt(3.0)):.3g} per sweep")
    elif cfg.frame != "whole_rho":
        limit = (f"limit {y_lim:.3g} ({cfg.method}, {cfg.predictor} predictor, n_corr "
                 f"{cfg.n_corr}) -> dt <= {receipt['dt_max_as']:.4g} as")
    else:
        limit = "whole_rho frame (no static limit)"
    say(f"  stability       : sigma = {receipt['sigma_ev']:.4g} eV (spectral radius of the "
        f"dSigma commutator map, {len(est)} power iterations, last change "
        f"{100 * receipt['change']:.2g} %), h*sigma = {receipt['y']:.3g}, " + limit
        + (f"; probe {pr}" if probes else "")
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
    if not stable and implicit:
        q = probes[-1]
        why = (f"{q['n_fail']} of its {tp.PROBE_STEPS} steps' node equations did not converge "
               f"in node_max_iter = {cfg.node_max_iter} evaluations" if q["n_fail"] else
               f"it grows by {q['growth']:.4g} per step")
        raise ValueError(
            f"GATE td_magnus_stability: got: dt = {cfg.dt * T:.4g} as with converged nodes, but "
            f"the scheme's own step at {tp.PROBE_STEP_FACTOR} dt = {q['probe_dt_as']:.4g} as is "
            f"not usable on this model: {why} (field off, {tp.PROBE_STEPS} steps from a random "
            f"coherence); want: growth <= {1.0 + tp.PROBE_TOLERANCE:.4g} per step and every node "
            f"solve converged at {tp.PROBE_STEP_FACTOR} dt, a 25 % margin; why: a step whose "
            f"node solve fails or grows there is not known to be usable at dt; fix: a smaller "
            f"dt, a larger node_max_iter / node_memory, or node_solver = 'truncated'")
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


def propagate(deck: TdDeck, system: TdSystem, model: SelfEnergyModel, drive,
              outs: TdOutputs, mesh, *, say=print) -> dict:
    """Run the propagation from t0, or from the checkpoint the deck names.

    ``[control.files] load_checkpoint_from`` continues that run exactly
    (:func:`load_checkpoint`: its rows are copied, its carry resumed).
    Returns counters and walls (seconds).
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
        node_solver=v["lorrax.node_solver"], node_tol=float(v["lorrax.node_tol"]),
        node_max_iter=int(v["lorrax.node_max_iter"]), node_memory=int(v["lorrax.node_memory"]),
        atol=float(v["control.ivp.atol"]), rtol=float(v["control.ivp.rtol"]),
        max_step=float(v["control.timesteps.max_step"]), chunk=min(chunk, nt),
        remainder=bool(v["system.depopulation"] != 0.0
                       or v["system.dephasing"] != 0.0
                       or system.ops.intra is not None))
    check = None
    if cfg.method in ("magnus2", "cfm4"):
        cfg, check = _magnus_step_check(cfg, system, model, drive, mesh, auto=auto, say=say)
    resume = load_checkpoint(deck, system, model, outs, cfg.dt)
    count = ntr * system.nk * system.nb * system.nb
    fingerprint = restart_fingerprint(deck, system, model, cfg.dt)
    walls = []
    t_start = time.perf_counter()

    def on_chunk(i0, i1, samples, state_fn, t_state):
        _write_chunk(outs, system, i0, i1, samples, state_fn, t_state, fingerprint)
        walls.append(time.perf_counter())
        say(f"  samples {i0:7d}-{i1 - 1:7d}  t = {t_state * tdgw_io.T_IN_ATTOSEC:12.3f} as"
            f"  wall {walls[-1] - t_start:9.2f} s")

    with timing.section("tdagw.propagate"):
        stats, _ = tp.run_propagation(
            cfg, sample_spec(deck, system), tp.place_state(rho0, mesh),
            system.ops, model, drive, mesh, cell_volume=system.cell_volume,
            nspinor=system.nspinor, count=count, mtsteps=outs.mtsteps,
            on_chunk=on_chunk, resume=resume)
    stats["check_dsigma"] = 0 if check is None else (     # power iteration + every probe call
        1 + 2 * tp.COUPLING_POWER_ITERATIONS + sum(q["dsigma"] for q in check["probes"]))
    stats["check_probes"] = [] if check is None else check["probes"]
    stats["dt"] = cfg.dt
    stats["wall"] = time.perf_counter() - t_start
    stats["wall_first_chunk"] = (walls[0] - t_start) if walls else 0.0
    stats["chunk"] = cfg.chunk
    t_from = cfg.t0 if resume is None else float(resume["tchkp"])
    stats["simulated_fs"] = (stats["t_final"] - t_from) * tdgw_io.T_IN_ATTOSEC / 1000.0
    return stats


def describe(deck: TdDeck, system: TdSystem, model: SelfEnergyModel, runtime) -> str:
    """The run header: what will be integrated, how, where."""
    v = deck.values
    T = tdgw_io.T_IN_ATTOSEC
    step = ("adaptive (SciPy RK45 controller), atol %.3g rtol %.3g, first step "
            "dt_measure" % (v["control.ivp.atol"], v["control.ivp.rtol"])
            if v["lorrax.method"] == "dp54" else
            "fixed dt = %.6g as (dt_measure / dt = %.6g)" % (
                v["lorrax.dt"] * T, v["control.timesteps.dt_measure"] / v["lorrax.dt"]))
    lines = [
        "TD-aGW propagation (TDGW conventions)",
        f"  deck            : {deck.source}",
        f"  header          : {system.hdr['path'] or 'built from the GW bundle ' + str(v['lorrax.bundle_input'])}",
        f"  window          : bands {system.bands[0]}..{system.bands[1] - 1}, "
        f"nb = {system.nb}, nk = {system.nk} (padded {system.nk_pad}), "
        f"nspinor = {system.nspinor}",
        f"  integrator      : {v['lorrax.method']}, {step}"
        + ((f", Gauss nodes converged ({v['lorrax.node_solver']}, node_tol "
            f"{v['lorrax.node_tol']:.3g}, memory {v['lorrax.node_memory']}, at most "
            f"{v['lorrax.node_max_iter']} node-map evaluations per step)")
           if v["lorrax.node_solver"] != "truncated" else
           f", n_corr = {v['lorrax.n_corr']}" if v["lorrax.method"] in ("magnus2", "cfm4") else ""),
        f"  self-energy     : {model.name}",
        f"  intraband [r,.] : {'on' if system.ops.intra is not None else 'off'}",
        f"  drives          : {len(deck.trials)}",
        f"  window in time  : {v['control.timesteps.t0'] * T:.3f} .. "
        f"{v['control.timesteps.tf'] * T:.3f} as, samples every "
        f"{v['control.timesteps.dt_measure'] * T:.6g} as",
        f"  mesh            : {runtime.mesh_shape} on {runtime.n_devices} "
        f"{runtime.platform} device(s)",
    ]
    rec = system.receipts
    if rec:                                  # the bundle route's receipts (also in the header)
        pr, vr = rec["pairs"], rec["velocity"]
        lines.append(
            f"  TD system       : {'bse_k_grid' if rec['densified'] else 'native'} bundle, "
            f"energies {rec['energy_source']}, r = v/(i dE_DFT) from {vr['route']}; "
            f"pair rule {pr['rule']} at {pr['gap_ev']:g} eV (vv' below: {pr['vv']['below_gap']} "
            f"of {pr['vv']['pairs']}, cc' below: {pr['cc']['below_gap']} of {pr['cc']['pairs']}), "
            f"min c-v gap {pr['min_cv_gap_ev']:.4f} eV, max|r| {pr['max_abs_r_bohr']:.3g} bohr")
        if "bse_dipoles" in rec:
            bd = rec["bse_dipoles"]
            lines.append(
                f"  gauge receipt   : <0|r|S> of {bd['n_states']} BSE states from the header r "
                f"and the eigenvectors vs bse_jax's: rel dev {bd['dipole_rel_dev']:.2e}, "
                f"I+ {bd['i_plus_rel_dev']:.2e}, I- {bd['i_minus_rel_dev']:.2e}")
    if v["control.files.load_checkpoint_from"]:
        lines.append(f"  restart from    : {v['control.files.load_checkpoint_from']} "
                     f"(its integrator carry; values continue bitwise)")
    unused = td_config.unused_keys(deck)
    if unused:
        lines.append("  read, not used  : " + ", ".join(unused)
                     + " (TDGW logging / vacuum centring / snapshot stride)")
    return "\n".join(lines)


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
    nd = stats.get("nodes")
    if nd:
        lines.insert(3, f"  node solves     : {nd['n_eval']} node-map evaluations ({nd['mean']:.3f} "
                        f"per step, at most {nd['max_run']}), largest final residual "
                        f"{nd['res_max']:.3g} (node_tol {nd['tol']:.3g}, {nd['solver']}, "
                        f"memory {nd['memory']})")
    return "\n".join(lines)
