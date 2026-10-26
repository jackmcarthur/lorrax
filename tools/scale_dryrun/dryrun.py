"""Scale dry-run: price every large stage of a synthetic deck with LORRAX's own planners.

Not a production path. Nothing here allocates device memory or compiles: it calls
the pure pricing functions the run itself uses (lxkit.memory Price/Term/sizing,
gw.tangential_capacity.tangential_byte_terms, gw.tangential_execution.
constructor_side_upper_bound, gw.tangential_recipe line/imaginary counts,
gw.gflat_memory_model.plan_zeta_route_g, gw.v_q_g_flat.vq_tile_bytes) and the
closed forms written in gw.w_isdf._dyson_route, gw.response_bank.parent_span and
docs/architecture/memory-model.md (stage inventory). Terms whose production
price is a compiled executable are modelled from their documented leading terms
and are tagged ``model``; the rest are tagged ``code``.

    PYTHONPATH=src:services/lxkit/src python tools/scale_dryrun/dryrun.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import common.gpu_utils as _gu

# Perlmutter GPU node: 256 GB host, 4 ranks. The planner reads the live host otherwise.
_gu.host_bytes_per_process = lambda fraction=0.9: fraction * 256e9 / 4

from lxkit import memory  # noqa: E402
from lxkit.memory import Price, Term  # noqa: E402
from gw.tangential_capacity import tangential_byte_terms  # noqa: E402
from gw.tangential_execution import constructor_side_upper_bound  # noqa: E402
from gw.tangential_recipe import (ROLE_CODES, imaginary_sample_count,  # noqa: E402
                                  support_line_count, shared_real_pole_v1_r3b as RECIPE)
from gw.gflat_memory_model import plan_zeta_route_g  # noqa: E402
from gw.v_q_g_flat import vq_tile_bytes  # noqa: E402

memory.set_sink(lambda *a, **k: None)   # the planners' banners; this harness prints its own table

GB = 1e9
C = 16.0
# A100-80GB, memory-model.md "The rule": B at linalg local / distributed; L = M - O_leg.
B_LOCAL, B_DIST, L_PHYS = 66.4 * GB, 63.4 * GB, (80.0 - 3.6) * GB
CTX = 3.6 * GB                      # lxkit.legs.DISTRIBUTED_CONTEXT_BYTES
TILE = float(1 << 30)               # runtime.tiles.TILE_BYTES
HOST_PER_RANK = 0.8 * 256e9 / 4     # gflat_memory_model._ZSTORE_HOST_FRAC of a node, per rank


def system(atoms, kgrid, spinor, *, nmu_per_atom=300, nb_per_atom=30, eta_ev=0.25,
           reach_ev=10.0, vol_bohr3_per_atom=200.0, ecut_ry=60.0, trs=True, side_frac=1.0):
    """Synthetic deck: N_mu = 10 N_b, N_b = 30/atom (x2 spinor), occupied 4/atom (x2), P1.

    ``trs``: time reversal kept (sym.trs_allowed, measured from the WFN); only a broken TRS
    (magnetic) takes the ordered bank, a nonmagnetic SOC deck stays on the even route.
    ``side_frac``: the reduction's admitted side as a fraction of the recipe bound
    (constructor_side_upper_bound); the actual side is min(signed bound, bound), 0.3-0.7 of it
    in the CrI3/Fe compile notes of tangential_capacity."""
    ns = 2 if spinor else 1
    nk = kgrid ** 3
    nq = {1: 1, 2: 8, 4: 36}[kgrid]
    vol = vol_bohr3_per_atom * atoms
    ng = int(vol * ecut_ry ** 1.5 / (6 * math.pi ** 2))      # psi sphere per k
    n1 = int(round((0.258 * vol * ecut_ry ** 1.5) ** (1 / 3)))  # density FFT box side
    nb = nb_per_atom * atoms * ns
    nv = 4 * atoms * ns
    return dict(atoms=atoms, kgrid=kgrid, spinor=spinor, ns=ns, nk=nk, nq=nq, ng=ng,
                fft=(n1, n1, n1), nmu=nmu_per_atom * atoms, nb=nb, nv=nv, nc=nb - nv,
                eta=eta_ev, reach=reach_ev, bse_v=atoms * ns, bse_c=atoms * ns,
                nk_fine=(2 * kgrid) ** 3, trs=bool(trs), side_frac=float(side_frac))


def pencil_side(s, ordered):
    """constructor_side_upper_bound on a production recipe built from the deck's counts."""
    n = s["nmu"]
    imag = imaginary_sample_count(20.0, "production")   # kappa = L/u_min ~ 20 eV / 1 eV
    line, _ = support_line_count([s["reach"]], 4 * s["eta"], imag, 2)
    pol = RECIPE["production"]
    recipe = dict(held=[False] * (line + imag) + [True] * 2,
                  role=[ROLE_CODES["line"]] * line + [ROLE_CODES["imaginary"]] * imag + [ROLE_CODES["line"]] * 2,
                  line_direction_cap=math.ceil(n * pol["line_direction_cap_fraction"]),
                  imaginary_width=math.ceil(n * pol["imaginary_width_fraction"]),
                  infinity_width=math.ceil(n * pol["infinity_width_fraction"]))
    side = constructor_side_upper_bound(recipe, ordered=ordered, odd_moments=True, logical_n=n)
    return side, line, imag, math.ceil(n * pol["pole_budget_fraction"])


def tangential_prices(s, P, ordered):
    """Local route (one parent whole per rank) and face route (one parent tiled over P).

    Selection is priced at the recipe bound, as admitted before the bank is read; the
    reduction and its eigh at ``side_frac`` of it."""
    n = s["nmu"]
    bound, line, imag, budget = pencil_side(s, ordered)
    side = max(1, math.ceil(s.get("side_frac", 1.0) * bound))
    meta = SimpleNamespace(n_rmu_padded=n)
    px = int(math.isqrt(P))
    mesh = SimpleNamespace(shape={"x": px, "y": P // px})
    out = {}
    for route, layout in (("local", "local"), ("face", "distributed")):
        res = SimpleNamespace(layout=layout)
        rows = []
        for phase, r, kw in (("selection", bound, dict(sample_batch=imag + 2)),
                             ("reduction", side, dict(sample_batch=1, ritz_budget=budget if ordered else None))):
            rows.append(tangential_byte_terms(meta, mesh_xy=mesh, resolution=res, pencil_side=r,
                                              parent_batch=1, phase=phase, **kw)["resident_bytes_per_rank"])
        # native eigh workspace [model]: syevd ~2 R^2 whole (local); cuSOLVERMp ~3 R^2/P + contexts (face)
        ws = 2 * C * side ** 2 if route == "local" else 3 * C * side ** 2 / P + CTX
        out[route] = (max(rows) + ws) * (1.41 if (s.get("calib") and route == "face") else 1.0)
    return out, dict(bound=bound, bound_over_nmu=round(bound / n, 3), side=side,
                     side_over_nmu=round(side / n, 3), ordered=ordered, line=line, imag=imag)


def stages(s, P, mode):
    """{stage: (src, ((route, bytes/rank), ...), host bytes/rank)} at P ranks; fastest route first."""
    n, Q, nk, ns, nb, ng = s["nmu"], s["nq"], s["nk"], s["ns"], s["nb"], s["ng"]
    n2 = C * n * n
    px = int(math.isqrt(P))
    st = {"psi(G) faces": ("code", (("only", C * nk * nb * ns * ng / P),), 0.0)}
    meta = SimpleNamespace(nk_tot=nk, nspinor=ns, n_rmu_padded=n, n_rmu=n, fft_grid=s["fft"])
    memory.reset()   # each plan is priced alone, not beside earlier probes in this process
    try:
        z = plan_zeta_route_g(meta=meta, mesh_xy=P, n_q_selected=Q, ngkmax=ng, psi_ngkmax=ng,
                              fit_nb=nb, n_col=nb, n_s=ns, budget_gb=B_LOCAL / GB)
        zeta = z.min_config_bytes
        if s.get("zeta_c_face"):
            # The planner prices the charge factor whole per q-owner, ceil(Q/P) mu^2; the charge
            # channel's B runs on the face (isdf.zeta_mubatch._FACE_KINDS, ZetaG._contract_v_face).
            zeta += C * Q * n * n / P - z.breakdown["C factor"]
        st["zeta fit (route G)"] = ("code" if not s.get("zeta_c_face") else "code, C on face",
                                    (("only", zeta),), z.store_bytes)
        s["_zstore"] = dict(placement=z.placement, store_gb_per_rank=round(z.store_bytes / GB, 1), b=z.b, n_batch=z.n_batch)
    except Exception as e:  # reported, never hidden
        st["zeta fit (route G)"] = (f"error {type(e).__name__}: {e}", (("only", float("nan")),), 0.0)
    v = vq_tile_bytes(n_q=Q, n_rmu_L=n, n_rmu_R=n, ngkmax=ng, same_zeta=True, n_sub=64,
                      p_x=px, p_y=P // px, g_chunk=4096)
    st["V_q accumulate"] = ("code", (("only", v["resident"] + v["per_q"] + v["work"]),), v["host_per_q"])
    st["V_q unfold"] = ("code", (("only", C * nk * n * n / P),), 0.0)
    # Coulomb roots V^1/2, V^-1/2 of the q stack (compiled in production) [model]
    st["Coulomb roots"] = ("model", (("local", n2 * (3 * Q / P + 4 * -(-Q // P))),
                                     ("distributed", n2 * (5 * Q + 2) / P + CTX)), 0.0)
    # W Dyson: exactly gw.w_isdf._dyson_route
    st["W Dyson"] = ("code", (("local", 3 * Q * n2 / P + n2 * (3 * -(-Q // P) + 2)),
                              ("distributed", (3 * Q + 2) * n2 / P + CTX)), 0.0)
    if mode == "tangential":
        ordered = not s.get("trs", True)   # tangential_constructor: ordered = not sym.trs_allowed
        t, info = tangential_prices(s, P, ordered)
        s["_pencil"] = info
        # gw.response_bank.parent_span: ~6 faces of 16 n^2/P per parent; under linalg=local a
        # round holds P parents (whole matrices per rank), on the full mesh one.
        st["bank round (value+slope)"] = ("code", (("local", min(Q, P) * 6 * n2 / P),
                                                   ("distributed", 6 * n2 / P + CTX)), 0.0)
        st["tangential constructor"] = ("code+model", (("local", t["local"]), ("face", t["face"])), 0.0)
        # Sigma tau: resident pole fields 32 N_q n Kbar / P, Kbar = 1.0 n, plus one row tile
        if s.get("calib"):
            # Measured, Al2O3 4^3, P = 4, N_mu 2400 / 6000 (job 60310320): 6.04 / 35.07 GB = 322 / 300 B Q mu^2/P
            st["Sigma tau sweep"] = ("measured fit", (("only", 300.0 * Q * n * n / P),), 0.0)
            # Unreceipted "ISDF setup + I/O" peak, same runs: 6.26 / 38.98 GB = 335 / 333 B Q mu^2/P
            st["ISDF setup + I/O (unpriced)"] = ("measured fit", (("only", 333.0 * Q * n * n / P),), 0.0)
        else:
            st["Sigma tau sweep"] = ("code", (("only", 32 * Q * n * n / P + TILE),), 0.0)
    else:  # gn_ppm [model]: chi0 carry at two nodes, the Wc pair, one Green row tile
        st["static/GN-PPM screening"] = ("model", (("only", 2 * nk * n2 / P + 2 * Q * n2 / P + TILE),), 0.0)
        st["Sigma tau sweep"] = ("model", (("only", 2 * Q * n2 / P + TILE),), 0.0)
    # BSE with kgrid_fine: coarse W + fine W_q (mu, nu, fine k) on the centroid face + one tile
    nkf, nvb, ncb = s["nk_fine"], s["bse_v"], s["bse_c"]
    st["BSE coarse->fine W"] = ("code", (("only", Q * n2 / P + nkf * n2 / P + TILE),), 0.0)
    st["BSE matvec + Haydock"] = ("model", (("only", 2 * C * nkf * (nvb + ncb) * n / P
                                             + 3 * C * nkf * nvb * ncb / P),), 0.0)
    return st


def evaluate(s, P, mode):
    rows = []
    for name, (src, routes, host) in stages(s, P, mode).items():
        best = min(routes, key=lambda r: r[1])
        bud = B_LOCAL if best[0] in ("local", "only") else B_DIST
        verdict = "ok" if best[1] <= bud else ("warn" if best[1] <= L_PHYS else "OOM")
        rows.append(dict(stage=name, src=src, route=best[0], fast_route=routes[0][0],
                         fast_fits=routes[0][1] <= B_LOCAL, gb=best[1] / GB, budget_gb=bud / GB,
                         verdict=verdict, host_gb=host / GB,
                         host_verdict="ok" if host <= HOST_PER_RANK else "disk"))
    return rows


def min_p(s, mode):
    """lxkit.memory.sizing over these prices: each stage's cheapest route as a Term.grid(P)."""
    names = list(stages(s, 4, mode))

    def grid(name, route):
        def f(p):
            return dict(stages(s, p, mode)[name][1]).get(route, float("inf"))
        return f
    plans = {name: tuple((r, Price(operands=Term(grid=grid(name, r), source="dryrun"), ranks=1))
                         for r, _ in stages(s, 4, mode)[name][1]) for name in names}
    return memory.sizing(plans, limit_bytes=L_PHYS, per_node=4, platform="Perlmutter A100-80GB")


DECK_KEYS = dict(atoms=int, k=str, spinor=int, nmu=int, nb=int, nv=int, nq=int, ng=int, fft=int,
                 vol_bohr3=float, ecut_ry=float, kfine=str, bse_v=int, bse_c=int, reach=float, eta=float,
                 trs=int, side_frac=float, zeta_c_face=int, calib=int)


def deck(text):
    """One deck from ``key=val,...`` (DECK_KEYS); unset keys take :func:`system`'s per-atom defaults.

    ``k`` and ``kfine`` are ``n`` or ``n1xn2xn3``; N_q defaults to the TRS count (N_k + N_TRIM)/2."""
    kv = dict(item.split("=", 1) for item in text.split(",") if item)
    bad = set(kv) - set(DECK_KEYS)
    if bad:
        raise SystemExit(f"unknown deck keys {sorted(bad)}; known: {sorted(DECK_KEYS)}")
    kv = {key: DECK_KEYS[key](val) for key, val in kv.items()}
    grid = lambda t: tuple(int(x) for x in t.split("x")) if "x" in t else (int(t),) * 3
    kg = grid(kv.get("k", "2"))
    atoms, spinor = kv.get("atoms", 10), bool(kv.get("spinor", 0))
    vol = kv.get("vol_bohr3", 200.0 * atoms)
    s = system(atoms, 2, spinor, vol_bohr3_per_atom=vol / atoms, ecut_ry=kv.get("ecut_ry", 60.0),
               trs=kv.get("trs", 1), side_frac=kv.get("side_frac", 1.0))
    nk = math.prod(kg)
    s.update(kgrid="x".join(map(str, kg)), nk=nk, nq=(nk + math.prod(2 if g % 2 == 0 else 1 for g in kg)) // 2,
             nk_fine=math.prod(grid(kv["kfine"])) if "kfine" in kv else math.prod(2 * g if g > 1 else 1 for g in kg))
    s["zeta_c_face"] = bool(kv.get("zeta_c_face", 0))
    # calib=1: scale by the one-node Al2O3 tangential measurements (face constructor x1.41 at the
    # real side, Sigma tau and the unreceipted ISDF setup peak as fitted Q mu^2/P coefficients)
    s["calib"] = bool(kv.get("calib", 0))
    for key, field in (("nmu", "nmu"), ("nb", "nb"), ("nv", "nv"), ("nq", "nq"), ("ng", "ng"),
                       ("bse_v", "bse_v"), ("bse_c", "bse_c"), ("reach", "reach"), ("eta", "eta")):
        if key in kv:
            s[field] = kv[key]
    if "fft" in kv:
        s["fft"] = (kv["fft"],) * 3
    s["nc"] = s["nb"] - s["nv"]
    return s


def sweep_decks():
    """The generic sweep: 10-100 atoms at k 2^3/4^3 (spinor decks with TRS and magnetic), plus the
    nonmagnetic 148-atom MoS2 4x4 / 2 hBN 5x5 slab (3x3x1, N_mu 8k and 16k, 23350 bohr^3); each
    at the recipe bound and at 0.6 of it."""
    for frac in (1.0, 0.6):
        for atoms in (10, 20, 40, 100):
            for kgrid in (2, 4):
                for spinor, trs in ((False, True), (True, True), (True, False)):
                    yield system(atoms, kgrid, spinor, trs=trs, side_frac=frac)
        for nmu in (8000, 16000):
            for spinor in (0, 1):
                yield deck(f"atoms=148,k=3x3x1,spinor={spinor},nmu={nmu},nb={nmu // 10 * (1 + spinor)},"
                           f"nv={300 * (1 + spinor)},vol_bohr3=23350,trs=1,side_frac={frac}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("--deck", action="append",
                    help="price this deck instead of the sweep: key=val,... with keys " + ",".join(DECK_KEYS)
                    + " (e.g. atoms=148,k=3x3x1,spinor=1,nmu=12000,nb=2400,nv=600,vol_bohr3=23350)")
    ap.add_argument("--modes", default="tangential,gn_ppm")
    ap.add_argument("--P", default="4,16,64,256", help="mesh sizes (square sizes are what LORRAX runs)")
    ap.add_argument("--stages", action="store_true", help="also print every stage's GB/rank and route")
    a = ap.parse_args()
    decks = [deck(t) for t in a.deck] if a.deck else list(sweep_decks())
    out = []
    for s0 in decks:
        atoms, kgrid, spinor = s0["atoms"], s0["kgrid"], s0["spinor"]
        for mode in a.modes.split(","):
            if mode != "tangential" and s0.get("side_frac", 1.0) != 1.0 and not a.deck:
                continue   # the side fraction prices only the tangential constructor
            s = dict(s0)
            for P in (int(p) for p in a.P.split(",")):
                rows = evaluate(s, P, mode)
                top = max(rows, key=lambda r: r["gb"] / r["budget_gb"])
                out.append(dict(atoms=atoms, kgrid=kgrid, spinor=spinor, trs=s0.get("trs", True),
                                side_frac=s0.get("side_frac", 1.0), mode=mode, P=P,
                                deck={k: v for k, v in s0.items() if not k.startswith("_")},
                                nmu=s["nmu"], nb=s["nb"], nq=s["nq"], ng=s["ng"],
                                pencil=s.get("_pencil"), zstore=s.get("_zstore"), rows=rows, binding=top["stage"],
                                binding_gb=top["gb"], binding_route=top["route"],
                                verdict=top["verdict"],
                                not_ok=[(r["stage"], r["route"], round(r["gb"], 1)) for r in rows if r["verdict"] != "ok"],
                                host_fail=[(r["stage"], round(r["host_gb"], 1)) for r in rows if r["host_verdict"] != "ok"]))
            z = min_p(s, mode)
            out[-1]["sizing"] = dict(p=z["p"], binding=z["binding"], line=z["line"])
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    for r in out:
        if a.stages:
            print(f"-- A={r['atoms']} k={r['kgrid']} ns={2 if r['spinor'] else 1} N_mu={r['nmu']} N_b={r['nb']} "
                  f"N_q={r['nq']} N_G={r['ng']} {r['mode']} P={r['P']} pencil={r['pencil']} zeta={r['zstore']}")
            for x in r["rows"]:
                print(f"     {x['stage']:26s} {x['route']:11s} {x['gb']:9.2f} GB/rank {x['verdict']:4s} [{x['src']}]"
                      + (f" host/disk {x['host_gb']:.1f} GB/rank" if x["host_gb"] else ""))
        print(f"A={r['atoms']:3d} k={r['kgrid']} ns={2 if r['spinor'] else 1} trs={int(r['trs'])} "
              f"side={r['side_frac']:.1f}xbound {r['mode']:10s} P={r['P']:3d} "
              f"mu={r['nmu']:6d} | {r['binding']:26s} {r['binding_route']:11s} {r['binding_gb']:9.1f} GB {r['verdict']:4s}"
              + (f" host:{r['host_fail']}" if r["host_fail"] else "")
              + (f" | sizing P={r['sizing']['p']} ({r['sizing']['binding']})" if "sizing" in r else ""))


if __name__ == "__main__":
    main()
