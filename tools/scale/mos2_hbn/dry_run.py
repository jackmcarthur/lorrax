"""Zero-GPU price of GW (gn_ppm, tangential) and BSE (kgrid_fine) for the hBN/MoS2/hBN target.

The tangential constructor is priced by LORRAX's own ``gw.tangential_capacity.
tangential_byte_terms`` (local and face layouts) at the recipe-only side bound of
``gw.tangential_execution.constructor_side_upper_bound`` (restated below: that module
imports jax). The other stages use the leading terms of docs/architecture/memory-model.md
(stage inventory) with G_tile = 16 N_k n_s^2 mu^2 / P. Native eigh workspace, which
tangential_byte_terms excludes, is added as 2 R^2 complex (local) or 2 R^2 / P (face).
Times are flop counts over an assumed sustained rate per GPU (RATE_*), to be replaced
by the probe's measured rates.

Usage: python dry_run.py [--mu 4900 8160 16320] [--P 4 9 16 25 36] [--rate-gemm 4e12]
"""
import argparse
import importlib.util
import math
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src"))
spec = importlib.util.spec_from_file_location("tc", os.path.join(SRC, "gw", "tangential_capacity.py"))
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

GB = 1e9
DEVICE = 80 * GB
ROOM = 0.85 * DEVICE - 1.5 * GB     # BFC@0.85 less CUDA/XLA contexts

# Target (build_inputs.py): 148 atoms, 816 electrons, SOC spinors, 60 Ry, 3x3x1.
NS = 2                 # spinor components
NOCC = 816             # occupied spinor bands = 408 Kramers pairs
NK = 9                 # full 3x3x1 grid
NQ = 3                 # q parents under C3 + TRS (Gamma, K, one Gamma-M line orbit)
NG_PSI = 183_000       # PWs per spinor component at 60 Ry, V = 23188 bohr^3
NR = 120 * 120 * 240   # density FFT box
NTAU = 40              # minimax tau nodes, occupied + empty legs


def side_bound(mu, *, lines=16, imaginary=2, ordered=True, odd=True):
    """constructor_side_upper_bound for the production recipe (18 fitted supports)."""
    cap, im, inf = math.ceil(mu / 16), math.ceil(mu / 4), math.ceil(mu / 8)
    if ordered:
        return 4 * (lines * cap + imaginary * im) + (2 * inf if odd else 0)
    return 2 * lines * cap + imaginary * im + inf


def tangential(mu, P, layout):
    R = side_bound(mu)
    meta = SimpleNamespace(n_rmu_padded=mu)
    mesh = SimpleNamespace(shape={"x": int(math.isqrt(P)), "y": P // int(math.isqrt(P))})
    res = SimpleNamespace(layout="local" if layout == "local" else "distributed")
    b = P if layout == "local" else 1
    red = tc.tangential_byte_terms(meta, mesh_xy=mesh, resolution=res, pencil_side=R,
                                   parent_batch=b, sample_batch=1, phase="reduction",
                                   ritz_budget=mu)["resident_bytes_per_rank"]
    sel = tc.tangential_byte_terms(meta, mesh_xy=mesh, resolution=res, pencil_side=R,
                                   parent_batch=b, sample_batch=1, phase="selection",
                                   selection_faces=2 * 18)["resident_bytes_per_rank"]
    ws = 16 * 2 * R * R / (1 if layout == "local" else P)
    return max(red, sel) + ws, R


def stages(mu, P, nb_spinor, nkf):
    gt = 16 * NK * NS**2 * mu**2 / P                         # G_tile
    out = {}
    out["zeta fit"] = (16 * NK * nb_spinor * NG_PSI * NS / P        # conj psi(G), both spinor rows
                       + 16 * NQ * mu**2 / P + gt)
    out["V_q"] = 16 * NK * mu**2 / P * 2
    out["gn_ppm (local Dyson)"] = 4 * 16 * mu**2 * math.ceil(NQ / P) + 2 * gt + 4 * 16 * NQ * mu**2 / P
    out["gn_ppm (distributed)"] = 4 * 16 * mu**2 * NQ / P + 2 * gt + 4 * 16 * NQ * mu**2 / P
    for lay in ("local", "face"):
        t, R = tangential(mu, P, lay)
        out[f"tangential {lay} (R={R})"] = t + 6 * 16 * mu**2 * (1 if lay == "local" else 1 / P)
    out["Sigma(tau)"] = 2 * gt + 32 * NQ * mu**2 / P
    out[f"BSE kgrid_fine {nkf}"] = 16 * NK * mu**2 / P + 16 * nkf * mu**2 / P + 2**30
    return out


def flops(mu, nb_spinor):
    R = side_bound(mu)
    nbl = NOCC + 64
    return {
        "zeta fit": 8 * NK * NR * (nbl + nb_spinor) * mu + 8 * NQ * mu**3,
        "chi0+W (gn_ppm)": NTAU * 8 * NK * nb_spinor * NS**2 * mu**2 + 2 * NQ * 8 * mu**3,
        "tangential pencil eigh": NQ * 36 * R**3,
        "Sigma": NTAU * (8 * NQ * mu**3 + 8 * NK * nb_spinor * NS**2 * mu**2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=int, nargs="+", default=[4900, 8160, 16320])
    ap.add_argument("--P", type=int, nargs="+", default=[4, 9, 16, 25, 36])
    ap.add_argument("--nb", type=int, default=1632, help="number_bands, spinor bands")
    ap.add_argument("--nkf", type=int, default=81)
    ap.add_argument("--rate-gemm", type=float, default=4e12)
    ap.add_argument("--rate-eigh", type=float, default=1e12)
    a = ap.parse_args()
    print(f"room per A100-80GB: {ROOM / GB:.1f} GB (BFC@0.85 less 1.5 GB contexts)")
    for mu in a.mu:
        print(f"\n## N_mu = {mu}  (number_bands = {a.nb} spinor, N_k = {NK}, N_q = {NQ})")
        rows = {P: stages(mu, P, a.nb, a.nkf) for P in a.P}
        print("| stage | " + " | ".join(f"P={P}" for P in a.P) + " |")
        print("|---|" + "---|" * len(a.P))
        for name in rows[a.P[0]]:
            cells = []
            for P in a.P:
                v = rows[P][name]
                cells.append(f"{v / GB:.1f}" + ("" if v <= ROOM else " X"))
            print(f"| {name} | " + " | ".join(cells) + " |")
        f = flops(mu, a.nb)
        for P in a.P:
            t = {k: v / (P * (a.rate_eigh if "eigh" in k else a.rate_gemm)) for k, v in f.items()}
            print(f"P={P}: " + ", ".join(f"{k} {v / 60:.0f} min" for k, v in t.items()))


if __name__ == "__main__":
    main()
