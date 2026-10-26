"""Summarise dryrun.py output as the markdown tables of SCALE_READINESS.md.

    PYTHONPATH=src:services/lxkit/src python tools/scale_dryrun/analyze.py /tmp/scale_dryrun.json
"""
from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, "tools/scale_dryrun")
import dryrun as D  # noqa: E402

BSE = ("BSE coarse->fine W", "BSE matvec + Haydock")


def cell(rows, keep):
    rows = [r for r in rows if keep(r["stage"])]
    bad = [r for r in rows if r["verdict"] != "ok"]
    if not bad:
        top = max(rows, key=lambda r: r["gb"] / r["budget_gb"])
        return f"ok ({top['gb']:.0f})"
    w = max(bad, key=lambda r: r["gb"] / r["budget_gb"])
    tag = "OOM" if w["verdict"] == "OOM" else "warn"
    short = {"tangential constructor": "pencil", "BSE coarse->fine W": "W_fine", "W Dyson": "Dyson",
             "Coulomb roots": "roots", "bank round (value+slope)": "bank", "zeta fit (route G)": "zeta",
             "static/GN-PPM screening": "chi0", "Sigma tau sweep": "Sigma", "V_q unfold": "Vq-unfold"}
    more = f"+{len(bad) - 1}" if len(bad) > 1 else ""
    return f"**{tag} {short.get(w['stage'], w['stage'])} {w['gb']:.0f}**{more}"


def local_limits():
    """Largest N_mu at which each stage's fastest (whole-matrix) route still fits B_LOCAL."""
    out = {}
    for spinor in (False, True):
        for name in ("tangential constructor", "bank round (value+slope)", "W Dyson", "Coulomb roots"):
            lo, hi = 100, 200000
            while hi - lo > 50:
                mid = (lo + hi) // 2
                s = D.system(10, 2, spinor, nmu_per_atom=mid / 10)
                s["nmu"] = mid
                b = dict(D.stages(s, 8, "tangential")[name][1])   # one q-parent per rank: P = N_q = 8
                fast = list(b.values())[0]
                lo, hi = (mid, hi) if fast <= D.B_LOCAL else (lo, mid)
            out[(name, spinor)] = lo
    return out


def times(s, P):
    """Ideal-scaling time walls from measured anchors (QSGW_LADDER_ASSESSMENT.md section 2.1)."""
    n, Q = s["nmu"], s["nq"]
    side = D.pencil_side(s, bool(s["spinor"]))[0]
    # CrI3 ML 24^2 charge QSGW, 16 A100: W 152 s, Sigma_tau 120 s at N_mu 3300, N_q 61
    scale = (Q / 61) * (n / 3300) ** 3 * (16 / P)
    # Ladder W, Si 4^3 SOC, 4 GPUs: ~300 s per z; probes N_q N_mu, each N_it C_HX (same N_it)
    def chx(nk, ns, mu, nc, nv):
        return 2 * (16 * nk * ns ** 2 * mu ** 2 * min(nc, nv) + 32 * nk * nc * nv * mu)
    anchor = 8 * 480 * chx(64, 2, 480, 20, 8)
    nkf = s["nk"]
    target = Q * n * chx(nkf, s["ns"], n, s["bse_c"], s["bse_v"])
    return dict(W_h=152 * scale / 3600, Sigma_h=120 * scale / 3600,
                pencil_eigh_flop=side ** 3, ladder_gpu_h_per_z=4 * 300 / 3600 * target / anchor)


def main(path):
    d = json.load(open(path))
    print("## First wall per GPU count (GB/rank; GW stages | BSE stages)\n")
    print("| atoms | N_mu | k | n_s | mode | P=4 | P=16 | P=64 | P=256 | min P (GW) | BSE @P=4/16/64/256 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    key = lambda r: (r["atoms"], r["kgrid"], r["spinor"], r["mode"])
    groups = {}
    for r in d:
        groups.setdefault(key(r), []).append(r)
    for (a, k, sp, mode), rs in groups.items():
        rs.sort(key=lambda r: r["P"])
        gw = [cell(r["rows"], lambda st: st not in BSE) for r in rs]
        bse = [cell(r["rows"], lambda st: st in BSE) for r in rs]
        s = D.system(a, k, sp)
        plans = {}
        for name in D.stages(s, 4, mode):
            if name in BSE:
                continue
            def grid(name=name, route=None):
                return lambda p: dict(D.stages(s, p, mode)[name][1]).get(route, float("inf"))
            plans[name] = tuple((rt, D.Price(operands=D.Term(grid=grid(name, rt), source="dryrun"), ranks=1))
                                for rt, _ in D.stages(s, 4, mode)[name][1])
        z = D.memory.sizing(plans, limit_bytes=D.L_PHYS, per_node=4)
        print(f"| {a} | {rs[0]['nmu']} | {k}³ | {2 if sp else 1} | {mode} | " + " | ".join(gw)
              + f" | {z['p']} ({z['binding']}) | " + " / ".join(bse) + " |")
    print("\n## Largest N_mu at which the whole-matrix (local) route fits B = 66.4 GB\n")
    print("| stage | non-spinor (TRS) | spinor (ordered) |\n|---|---|---|")
    lim = local_limits()
    for name in ("tangential constructor", "bank round (value+slope)", "W Dyson", "Coulomb roots"):
        print(f"| {name} | {lim[(name, False)]} | {lim[(name, True)]} |")
    print("\n## 100-atom objects (N_mu = 30000, N_b = 3000 / 6000)\n")
    for k in (2, 4):
        for sp in (False, True):
            s = D.system(100, k, sp)
            side, line, imag, _ = D.pencil_side(s, sp)
            n2 = 16 * s["nmu"] ** 2 / 1e9
            t = times(s, 256)
            z = next(r for r in d if r["atoms"] == 100 and r["kgrid"] == k and r["spinor"] == sp and r["P"] == 256)
            print(f"- k={k}³ n_s={s['ns']} N_q={s['nq']}: one N_mu² matrix {n2:.1f} GB; pencil side R = {side} "
                  f"({side / s['nmu']:.2f} N_mu, {line} line + {imag} imaginary supports); "
                  f"V_q unfold 16 N_k N_mu² = {n2 * s['nk'] / 1e3:.2f} TB; W_fine (k_f={2 * k}³) "
                  f"{n2 * s['nk_fine'] / 1e3:.1f} TB; Z store {z['zstore']['store_gb_per_rank'] * 256 / 1e3:.1f} TB "
                  f"({z['zstore']['placement']}, {z['zstore']['n_batch']} μ-batches of {z['zstore']['b']} at P=256); "
                  f"ideal time at P=256: W {t['W_h']:.1f} h, Σ_τ {t['Sigma_h']:.1f} h per map; "
                  f"ladder W {t['ladder_gpu_h_per_z']:.2e} GPU-h per z")


if __name__ == "__main__":
    main(sys.argv[1])
