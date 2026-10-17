"""The held collision term in the TD-aGW propagator, on the TDGW fixture (child of test_prop_collision_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_collision_case.py OUTDIR '[{"tag": "x", "kind": "fixed", "method": "cfm4", "dt": 2.5}]'

The fixture (``fixtures/syn2v2c``: 3x3 k, 2v2c, COHSEX TDA kernel, Gamma 0.05
eV, intraband on, a 5e8 V/m pulse, 100 samples, chunks of 20) runs through
``tdagw.stages.propagate`` with ``collision=`` a synthetic source: the
production attach (``stages._attach_collision``) is replaced by one that holds
random positive semidefinite rates G~>, G~< per k (``rate_ha``) and builds the
dissipator with the production ``tdagw.propagate.collision_program``; the
refresh hook (``refresh``: every chunk) scales the rates by 1 + 0.2 n and sets
the held Delta Sigma^h to n c (c Hermitian, ``held_ha``) at the n-th refresh.  Per
case one ``RESULT`` JSON line: the final state rho (from the checkpoint's
lessG), max |rho - rho_eq| over the run, the chunk programs and their traced
specializations, the compiles per chunk (``jax_log_compiles``), and for
``kind = order`` the exact solution's error.

Kinds: ``fixed`` (no field: rho_eq stays put), ``order`` (IP, no field, Gamma 0,
no intraband term, raw form: x' = -i[H0, x] - 1/2 {G~, x} + C[rho_eq] has the
exact solution of an augmented matrix exponential per k), ``driven`` (the
fixture's drive and model), ``hlo`` (the compiled chunk program's collectives,
collision on vs off).  Options: ``form`` (subtracted | raw | off), ``refresh``
(bool), ``no_rebase`` (the hook drops delta: the Magnus history keeps the old
held term), ``stale`` (RED TWIN: the refresh leaves the carry alone, so rk4's
first-same-as-last derivative is the old operators'), ``frame`` (magnus2),
``rtol``/``atol`` (dp54).
"""
import copy
import json
import logging
import os
import re
import sys

import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from runtime.source_closure import ensure_source_closure   # noqa: E402

ensure_source_closure()

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "syn2v2c")


class CompileLog(logging.Handler):
    """Function names of the XLA compiles (``jax_log_compiles``), in order."""

    PAT = re.compile(r"Compiling (\S+)")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.names = []

    def emit(self, record):
        m = self.PAT.search(record.getMessage())
        if m:
            self.names.append(m.group(1))


def _deck(case, outdir):
    meta = json.load(open(os.path.join(FIXTURE, "deck_tdgw.json")))
    raw = copy.deepcopy(meta["deck"])
    raw["control"]["files"]["indir"] = os.path.join(FIXTURE, "indata")
    raw["control"]["files"]["load_header_from"] = os.path.join(FIXTURE, "tdgw_ref")
    raw["control"]["files"]["outdir"] = outdir
    raw["lorrax"] = {"method": case["method"], "n_corr": int(case.get("n_corr", 2)), "chunk": 20}
    # the whole window at every sample (the checkpoint's lessG of magnus2 / whole_rho
    # is rho_eq + rho, an incumbent quirk this case does not read through)
    types = raw["measurements"]["types"]
    raw["measurements"][str(len(types))] = {"dataset_name": "rho_full", "brange1": [5, 9],
                                            "brange2": [5, 9]}
    raw["measurements"]["types"] = types + ["density_matrix"]
    if case.get("dt") is not None:
        raw["lorrax"]["dt"] = case["dt"]
    if case.get("frame"):
        raw["lorrax"]["magnus_frame"] = case["frame"]
    if case["method"] == "dp54":
        raw["control"]["ivp"].update(rtol=case.get("rtol", 1e-10), atol=case.get("atol", 1e-14))
    if case["kind"] in ("fixed", "order"):
        raw["system"]["Efunc"] = {"E_x": 0.0, "E_y": 0.0, "E_z": 0.0}
    if case["kind"] == "order":
        raw["control"]["calc_type"] = "IP"
        raw["system"]["depopulation"] = 0.0
        raw["system"]["dephasing"] = 0.0
        raw["lorrax"]["intraband"] = False
    return raw


def _rates(system, rate_ha, seed=5):
    rng = np.random.default_rng(seed)
    nk, nb = system.nk, system.nb

    def psd():
        a = rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(size=(nk, nb, nb))
        return rate_ha * (a @ np.conj(np.swapaxes(a, -1, -2))) / nb
    g_out, g_in = psd(), psd()
    c = rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(size=(nk, nb, nb))
    return g_out, g_in, 0.5 * (c + np.conj(np.swapaxes(c, -1, -2)))


def _synthetic_attach(case, records):
    """A stand-in for stages._attach_collision: synthetic rates, the production program."""
    def attach(deck, system, model, cfg, source, mesh, *, say, log):
        from common.collectives import gather_to_host
        from tdagw import propagate as tp
        from tdagw.self_energy import held_term, with_held_term
        ntr = len(deck.trials)
        nk, nb, nk_pad = system.nk, system.nb, system.nk_pad
        g_out, g_in, c = _rates(system, float(case.get("rate_ha", 5e-3)))
        tile = lambda a: np.broadcast_to(a, (ntr,) + a.shape).copy()
        put = lambda a: tp.place_window_blocks(tile(a), mesh, nk_pad)
        prog = tp.collision_program(mesh, raw=case.get("form") == "raw")
        rho_eq = np.asarray(gather_to_host(system.ops.rho_eq))[:nk]
        geq = (put(g_out), put(g_in))
        res = prog(geq[0], geq[1], put(rho_eq), system.ops.rho_eq, geq[0], geq[1],
                   system.ops.w_k)
        held0 = put(np.zeros((nk, nb, nb), complex))
        model = with_held_term(model, held0)
        state = dict(n=0, history=[])
        records["host"] = dict(g_out=g_out, g_in=g_in, rho_eq=rho_eq, c=c,
                               e_qp=np.asarray(gather_to_host(system.ops.e_qp))[:nk],
                               c_eq_diag=float(res["c_eq_diag"]))
        held_ha = float(case.get("held_ha", 1e-3))

        def refresh(t, rho, ops, params):
            if not case.get("refresh"):
                return None
            state["n"] += 1
            s = 1.0 + float(case.get("rate_step", 0.2)) * state["n"]
            out = prog(put(g_out * s), put(g_in * s), rho, ops.rho_eq, geq[0], geq[1], ops.w_k)
            term, set_term = held_term(params)
            new = put(state["n"] * held_ha * c)
            records.setdefault("refresh_t", []).append(float(t))
            records.setdefault("compiles_at_refresh", []).append(len(records["log"].names))
            return dict(ops=ops._replace(collision=out["diss"]), params=set_term(new),
                        delta=None if case.get("no_rebase") else new - term)

        built = dict(diss=res["diss"], held=held0, refresh=refresh, state=state,
                     t0=dict(wall_s=0.0, why="t0"))
        system = system._replace(ops=system.ops._replace(collision=res["diss"]))
        return built, system, model, cfg._replace(remainder=True)
    return attach


def _hlo_collectives(prog_text):
    ops = ("all-reduce", "all-gather", "collective-permute", "all-to-all", "reduce-scatter")
    return {op: len(re.findall(rf"\b{op}(?:-start)?\(", prog_text)) for op in ops}


def _exact_order_reference(host, times, switches=(), held_ha=0.0, rate_step=0.2):
    """Exact rho(t) of rho' = -i[H0 + c_n, rho] + C_n[rho] (raw form) from rho_eq, per k.

    Piecewise constant in time: on [switches[n-1], switches[n]) the rates are
    (1 + rate_step n) G~ and the held term n held_ha c (the synthetic hook's schedule,
    n = 0 before the first refresh), the raw form's source C_n[rho_eq] at those
    rates.  ``times`` ascending; returns (nt, nk, nb, nb).
    """
    from scipy.linalg import expm
    g_out, g_in, rho_eq, e, c = (host[k] for k in ("g_out", "g_in", "rho_eq", "e_qp", "c"))
    nk, nb = e.shape
    one = np.eye(nb)
    edges = [0.0] + [float(t) for t in switches] + [np.inf]

    def gen(k, n):
        s = 1.0 + rate_step * n
        gam = s * (g_out[k] + g_in[k])
        h = np.diag(e[k]) + n * held_ha * c[k]
        src = (-1j * (h @ rho_eq[k] - rho_eq[k] @ h)                 # [H0, rho_eq] = 0, [c, rho_eq] not
               - 0.5 * (gam @ rho_eq[k] + rho_eq[k] @ gam) + s * g_in[k])
        M = np.zeros((nb * nb + 1, nb * nb + 1), complex)
        M[:nb * nb, :nb * nb] = (-1j * (np.kron(h, one) - np.kron(one, h.T))
                                 - 0.5 * (np.kron(gam, one) + np.kron(one, gam.T)))
        M[:nb * nb, nb * nb] = src.reshape(-1)
        return M

    out = np.empty((len(times), nk, nb, nb), complex)
    for k in range(nk):
        v = np.zeros(nb * nb + 1, complex)
        v[-1] = 1.0
        t_at, n = 0.0, 0
        for i, t in enumerate(times):
            while t >= edges[n + 1]:                   # cross the switches before t
                v = expm((edges[n + 1] - t_at) * gen(k, n)) @ v
                t_at, n = edges[n + 1], n + 1
            out[i, k] = (expm((t - t_at) * gen(k, n)) @ v)[:nb * nb].reshape(nb, nb)
    return rho_eq[None] + out


def run_case(case, outdir, mesh):
    from tdagw import propagate as tp
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function
    raw = _deck(case, outdir)
    deck = parse_td_deck(raw, source="collision_case")
    system = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, system, mesh)
    outs = stages.open_outputs(deck, system)
    records = dict(log=CompileLog())
    logging.getLogger("jax").addHandler(records["log"])
    jax.config.update("jax_log_compiles", True)
    form = case.get("form", "subtracted")
    orig_attach, orig_refresh, orig_chunk = (stages._attach_collision, tp._refresh_program,
                                             tp._chunk_program)
    hlo = {}
    if form != "off":
        stages._attach_collision = _synthetic_attach(case, records)
    if case.get("stale"):
        tp._refresh_program = lambda cfg, model, drive, mesh: (
            lambda carry, ops, params, delta, t: carry)
    if case["kind"] == "hlo":
        def spy(*a, **kw):
            fn = orig_chunk(*a, **kw)

            def wrapped(*args):
                if "text" not in hlo:
                    hlo["text"] = fn.lower(*args).compile().as_text()
                return fn(*args)
            wrapped._cache_size = fn._cache_size
            return wrapped
        tp._chunk_program = spy
    chunk_compiles = []

    def say(line):
        if line.lstrip().startswith("samples"):
            chunk_compiles.append(len(records["log"].names))
    try:
        stats = stages.propagate(deck, system, model, efield_function(deck.trials), outs, mesh,
                                 say=say, collision=None if form == "off" else dict(wall_s=0.0))
    finally:
        stages._attach_collision, tp._refresh_program, tp._chunk_program = (
            orig_attach, orig_refresh, orig_chunk)
        jax.config.update("jax_log_compiles", False)
        logging.getLogger("jax").removeHandler(records["log"])
    with h5py.File(outs.paths[0], "r") as f:
        cp = f["tdgw_data/checkpoint"]
        tchkp = float(cp["tchkp"][()])
        rho_t = f["tdgw_data/time_domain/rho_full/data"][()]           # (nt, nk, nb, nb)
        rho_cv = f["tdgw_data/time_domain/rho_cv/data"][()]
        pol = f["tdgw_data/time_domain/polarization/data"][()]
    rho = rho_t[-1]
    times = np.asarray(outs.mtsteps, float)
    rho_eq = np.asarray(stages.tdgw_io.read_header(raw["control"]["files"]["load_header_from"])
                        ["f_kn"], float)
    eq = np.stack([np.diag(r) for r in rho_eq]).astype(complex)
    out = dict(tag=case["tag"], kind=case["kind"], method=case["method"], dt=case.get("dt"),
               form=form, mesh=list(mesh.devices.shape), t_final_au=tchkp,
               max_dev_final=float(np.max(np.abs(rho - eq))),
               max_dev=float(np.max(np.abs(rho_t - eq[None]))),
               refresh_t=records.get("refresh_t", []),
               max_rho_cv=float(np.max(np.abs(rho_cv))), max_pol=float(np.max(np.abs(pol))),
               programs=int(stats["programs"]), jit_cache=int(stats["jit_cache"]),
               refreshes=int(stats["refreshes"]), steps=int(stats["accepted"]),
               chunk_compiles=chunk_compiles, compiles=records["log"].names,
               refresh_compiles=records.get("compiles_at_refresh", []))
    np.save(os.path.join(outdir, "rho_t.npy"), rho_t)
    if case["kind"] == "order":
        ref = _exact_order_reference(records["host"], times, records.get("refresh_t", ()),
                                     float(case.get("held_ha", 1e-3)) if case.get("refresh")
                                     else 0.0, float(case.get("rate_step", 0.2)))
        out["err_exact"] = float(np.max(np.abs(rho_t - ref)) / np.max(np.abs(ref - eq[None])))
        out["dev_exact"] = float(np.max(np.abs(ref - eq[None])))
    if case["kind"] == "hlo":
        out["collectives"] = _hlo_collectives(hlo.get("text", ""))
    if "host" in records:
        out["c_eq_diag"] = records["host"]["c_eq_diag"]
    return out


def main():
    root, cases = sys.argv[1], json.loads(sys.argv[2])
    from common.collectives import resolve_mesh
    mesh = resolve_mesh()
    for case in cases:
        outdir = os.path.join(root, case["tag"])
        os.makedirs(outdir, exist_ok=True)
        try:
            row = run_case(case, outdir, mesh)
        except Exception as exc:                       # one failed case does not hide the rest
            import traceback
            row = dict(tag=case["tag"], error=f"{type(exc).__name__}: {exc}",
                       trace=traceback.format_exc()[-3000:])
        print("RESULT " + json.dumps(row, default=float), flush=True)


if __name__ == "__main__":
    main()
