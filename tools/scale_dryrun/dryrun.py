"""Scale dry-run: price every large stage of a synthetic deck with LORRAX's own planners.

Not a production path. Nothing here allocates device memory or compiles. The GN-PPM
stages, and the ISDF, ζ, V(q) and Σ stages tangential shares, are
:mod:`gw.stage_inventory` (the runtime's own price functions, calibrated against
measured stage peaks); the tangential pencil calls gw.tangential_capacity.
tangential_byte_terms at gw.tangential_execution.constructor_side_upper_bound, and the
bank round and BSE rows keep the closed forms of gw.response_bank.parent_span and
docs/architecture/memory-model.md, tagged ``model``.

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
from gw.stage_inventory import Deck, stage_prices, zeta_plan  # noqa: E402

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


def inventory_deck(s):
    """The :class:`gw.stage_inventory.Deck` of a harness deck."""
    kg = tuple(int(x) for x in str(s["kgrid"]).split("x")) if "x" in str(s["kgrid"]) else (int(s["kgrid"]),) * 3
    return Deck(nk=s["nk"], n_parent=s.get("nparent") or s["nq"], nq=s["nq"], ns=s["ns"], nb=s["nb"],
                nb_qp=s.get("nb_qp") or min(s["nb"], s["nv"] + 16 * s["ns"]), ng=s["ng"],
                fft=tuple(s["fft"]), nmu=s["nmu"], kgrid=kg, trs=bool(s.get("trs", True)),
                sys_dim=s.get("sys_dim", 3))


def stages(s, P, mode):
    """{stage: (src, ((route, bytes/rank), ...), host bytes/rank)} at P ranks; fastest route first."""
    n, Q, nk = s["nmu"], s["nq"], s["nk"]
    n2 = C * n * n
    d = inventory_deck(s)
    inv = stage_prices(d, P)
    z = zeta_plan(d, P)
    s["_zstore"] = dict(placement=z.placement, store_gb_per_rank=round(z.store_bytes / GB, 1), b=z.b,
                        n_batch=z.n_batch)
    shared = ["ISDF setup + I/O", "zeta C = (psi psi)(psi psi)^H", "zeta factor B", "zeta mu-batch", "V(q)",
              "Sigma exchange", "Sigma Hartree"]
    if mode != "tangential":
        shared += ["chi0", "W", "Sigma tau sweep", "Sigma other"]
    st = {name: ("code" if name in ("zeta mu-batch", "V(q)", "W", "Sigma exchange", "Sigma tau sweep")
                 else "code+trace", tuple((r, p.bytes()) for r, p in inv[name]),
                 z.store_bytes if name == "zeta mu-batch" else 0.0) for name in shared}
    if mode == "tangential":
        ordered = not s.get("trs", True)   # tangential_constructor: ordered = not sym.trs_allowed
        t, info = tangential_prices(s, P, ordered)
        s["_pencil"] = info
        # Coulomb roots V^1/2, V^-1/2 of the q stack (compiled in production) [model]
        st["Coulomb roots"] = ("model", (("local", n2 * (3 * Q / P + 4 * -(-Q // P))),
                                         ("distributed", n2 * (5 * Q + 2) / P + CTX)), 0.0)
        # gw.response_bank.parent_span: ~6 faces of 16 n^2/P per parent; under linalg=local a
        # round holds P parents (whole matrices per rank), on the full mesh one.
        st["bank round (value+slope)"] = ("code", (("local", min(Q, P) * 6 * n2 / P),
                                                   ("distributed", 6 * n2 / P + CTX)), 0.0)
        st["tangential constructor"] = ("code+model", (("local", t["local"]), ("face", t["face"])), 0.0)
        if s.get("calib"):
            # Measured, Al2O3 4^3, P = 4, N_mu 2400 / 6000: the pole-synthesis Sigma tau, 6.04 / 35.07 GB
            # = 322 / 300 B Q mu^2/P (its own compiled price 42.78 GB, gamma 0.82)
            st["Sigma tau sweep"] = ("measured fit", (("only", 300.0 * Q * n * n / P),), 0.0)
        else:
            st["Sigma tau sweep"] = ("code", (("only", 32 * Q * n * n / P + TILE),), 0.0)
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


DECK_KEYS = dict(atoms=int, k=str, spinor=int, nmu=int, nb=int, nv=int, nq=int, ng=int, fft=str,
                 vol_bohr3=float, ecut_ry=float, kfine=str, bse_v=int, bse_c=int, reach=float, eta=float,
                 trs=int, side_frac=float, zeta_c_face=int, calib=int, nparent=int, nb_qp=int, sys_dim=int)


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
    # zeta_c_face: accepted for old command lines; the charge factor is always priced on the face
    # calib=1: scale the tangential rows by the one-node Al2O3 measurements (face constructor x1.41
    # at the real side; the pole-synthesis Sigma tau as a fitted Q mu^2/P coefficient).  The old
    # "ISDF setup + I/O" fit (333 B Q mu^2/P) was the Coulomb all-q validation at its widest
    # admitted batch (64 q), which fills the room; it is priced at its floor now.
    s["calib"] = bool(kv.get("calib", 0))
    for key, field in (("nmu", "nmu"), ("nb", "nb"), ("nv", "nv"), ("nq", "nq"), ("ng", "ng"),
                       ("nparent", "nparent"), ("nb_qp", "nb_qp"), ("sys_dim", "sys_dim"),
                       ("bse_v", "bse_v"), ("bse_c", "bse_c"), ("reach", "reach"), ("eta", "eta")):
        if key in kv:
            s[field] = kv[key]
    if "fft" in kv:
        s["fft"] = grid(kv["fft"])
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
