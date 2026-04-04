#!/usr/bin/env python3
from __future__ import annotations

"""
Test epsilon-v products at q=0 to determine which assembly yields a Hermitian W
and compare vcoul from eps0mat.h5 to a locally reconstructed 2D-truncated v(G).

Usage:
  python -m gw.test_eps_v_products -i cohsex_prod.in --eps0 eps0mat.h5

What it prints:
  - Basic shapes and cell volume
  - vcoul(G) from eps0mat vs locally reconstructed v(G) stats (excluding G=0)
  - Hermiticity diagnostics for
      W_left  = diag(v) @ epsinv
      W_right = epsinv @ diag(v)
      W_sym   = diag(sqrt(v)) @ epsinv @ diag(sqrt(v))
    (all with head/wings zeroed at Γ)
  - Top-left 3x3 tables (Re) for each variant
"""

import os
import argparse
import numpy as np

from common.wfnreader import WFNReader
from common.epsreader import EPSReader


def wrap_points_to_voronoi(randcart: np.ndarray, bvec: np.ndarray, nmax: int = 1) -> np.ndarray:
    grid = np.arange(-nmax, nmax + 1)
    shifts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    candidate_shifts = shifts @ bvec
    diff = randcart[:, None, :] - candidate_shifts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    best_idx = np.argmin(dists, axis=1)
    return randcart - candidate_shifts[best_idx]


def compute_v_trunc_2d_for_eps(G_comps_eps: np.ndarray, bvec: np.ndarray) -> np.ndarray:
    """Return 2D truncated Coulomb (8π/|G|^2)*f2d in Cartesian, no 1/Ω, for eps order.
    Head (G=0) is left as 0 to avoid infs; caller can compare excluding it.
    """
    G_cart = (G_comps_eps @ bvec)  # (ng,3)
    denom = np.einsum('ij,ij->i', G_cart, G_cart)
    kxy = np.linalg.norm(G_cart[:, :2], axis=1)
    kz = G_cart[:, 2]
    zc = np.pi / bvec[2, 2]
    f2d = (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
    v = np.zeros_like(denom, dtype=np.complex128)
    mask = denom > 1e-30
    v[mask] = (8.0 * np.pi / denom[mask]) * f2d[mask]
    return v


def top_left_3x3(A: np.ndarray) -> str:
    A3 = np.real(A[:3, :3])
    rows = [" ".join(f"{x:10.2f}" for x in A3[i]) for i in range(A3.shape[0])]
    return "\n".join(rows)


def herm_stats(A: np.ndarray) -> tuple[float, float]:
    res = float(np.max(np.abs(A - A.conj().T)))
    di = float(np.max(np.abs(np.imag(np.diag(A)))))
    return res, di


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Probe epsilon-v products at q=0 and vcoul consistency")
    ap.add_argument("-i", "--input", required=True, help="cohsex input file to locate WFN.h5")
    ap.add_argument("--eps0", default="eps0mat.h5", help="Path to eps0mat.h5 (epsilon^{-1} at q=0)")
    args = ap.parse_args(argv)

    # Resolve paths
    from psp.get_DFT_mtxels import read_cohsex_input
    inp_dir = os.path.dirname(os.path.abspath(args.input))
    params = read_cohsex_input(args.input)
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(inp_dir, wfn_path)
    eps0_path = args.eps0 if os.path.isabs(args.eps0) else os.path.join(inp_dir, args.eps0)

    # Load data
    wfn = WFNReader(wfn_path)
    eps = EPSReader(eps0_path)
    try:
        print(f"eps0mat.h5: icutv={int(eps.icutv)} (truncation flag)")
    except Exception:
        pass
    bvec = np.asarray(wfn.blat * wfn.bvec, dtype=float)
    print(f"cell volume Ω = {float(wfn.cell_volume):.6f}")

    # q=0 only
    iq = 0
    epsinv = eps.get_eps_matrix(iq)
    n_eps = int(eps.nmtx[iq])
    # eps-space G components in RHO basis (unfold) — cohsex_isdf convention
    G_eps = np.asarray(eps.unfold_eps_comps(iq, np.eye(3, dtype=int), np.array([0.0, 0.0, 0.0])), dtype=int)[:n_eps]
    # vcoul from eps file
    v_eps = np.asarray(eps.vcoul[iq, :n_eps], dtype=np.complex128)
    # locally reconstructed v (no 1/Ω)
    v_loc = compute_v_trunc_2d_for_eps(G_eps, bvec)
    # exclude head
    gind = np.asarray(eps.gind_eps2rho[iq, :n_eps])
    G0 = int(np.where(gind == 0)[0][0]) if np.any(gind == 0) else None
    mask = np.ones(n_eps, dtype=bool)
    if G0 is not None:
        mask[G0] = False

    # Compare vcoul
    dv = np.abs(v_eps[mask] - v_loc[mask])
    print(f"vcoul comparison (exclude G=0): mean|Δ|={float(np.mean(dv)):.3e} max|Δ|={float(np.max(dv)):.3e}")
    # Show a few entries
    for idx in [0, 1, 2, 3, 4]:
        if idx >= n_eps:
            break
        if idx == G0:
            continue
        print(f"  G[{idx:3d}] comps={tuple(G_eps[idx])}  v_eps={v_eps[idx].real:.6f}  v_loc={v_loc[idx].real:.6f}")

    # Drill into kxy==0 (Gx=Gy=0) behavior to highlight 2D truncation parity effect
    kxy0 = np.where((G_eps[:, 0] == 0) & (G_eps[:, 1] == 0))[0]
    if kxy0.size > 0:
        print(f"kxy==0 entries: {kxy0.size} (excluding G0={G0})")
        for idx in kxy0[:6]:
            if idx == G0:
                continue
            print(f"  Gz={int(G_eps[idx,2]):3d}  v_eps={v_eps[idx].real:.6f}  v_loc={v_loc[idx].real:.6f}")

    # Build W variants in eps-space
    def zero_head_wings(W: np.ndarray) -> np.ndarray:
        if G0 is not None:
            W[G0, :] = 0.0
            W[:, G0] = 0.0
        return W

    Dv = np.diag(v_eps)
    W_left = zero_head_wings(Dv @ epsinv.copy())
    W_right = zero_head_wings(epsinv.copy() @ Dv)
    # symmetric metric (diagnostic only)
    sqrtv = np.sqrt(np.clip(v_eps.real, 0.0, None)).astype(np.complex128)
    W_sym = zero_head_wings((sqrtv[:, None] * epsinv) * sqrtv[None, :])

    # Hermiticity stats
    for name, W in ("W_left=diag(v)@epsinv", W_left), ("W_right=epsinv@diag(v)", W_right), ("W_sym=√v epsinv √v", W_sym):
        res, di = herm_stats(W)
        print(f"[{name}] herm_resid={res:.3e} max|Im diag|={di:.3e}")
        print("  top-left 3x3 (Re):\n" + top_left_3x3(W))

    # Print first 10 entries (excluding head) with v and sample W elements
    print("\nFirst 10 eps-order entries (excluding G=0):")
    header = (
        f"{'idx':>4}  {'Gx':>3} {'Gy':>3} {'Gz':>3}  "
        f"{'v_eps':>12} {'v_loc':>12}  "
        f"{'Wl[ii]':>12} {'Wr[ii]':>12} {'Ws[ii]':>12}  "
        f"{'Wl[i,i+1]':>12} {'Wr[i,i+1]':>12} {'Ws[i,i+1]':>12}"
    )
    print(header)
    count = 0
    i = 0
    while count < 10 and i < n_eps - 1:
        if i == G0:
            i += 1
            continue
        gx, gy, gz = int(G_eps[i, 0]), int(G_eps[i, 1]), int(G_eps[i, 2])
        ve = float(np.real(v_eps[i]))
        vl = float(np.real(v_loc[i]))
        wl_ii = float(np.real(W_left[i, i]))
        wr_ii = float(np.real(W_right[i, i]))
        ws_ii = float(np.real(W_sym[i, i]))
        wl_iip = float(np.real(W_left[i, i + 1]))
        wr_iip = float(np.real(W_right[i, i + 1]))
        ws_iip = float(np.real(W_sym[i, i + 1]))
        print(
            f"{i:4d}  {gx:3d} {gy:3d} {gz:3d}  "
            f"{ve:12.6f} {vl:12.6f}  "
            f"{wl_ii:12.2f} {wr_ii:12.2f} {ws_ii:12.2f}  "
            f"{wl_iip:12.2f} {wr_iip:12.2f} {ws_iip:12.2f}"
        )
        count += 1
        i += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
