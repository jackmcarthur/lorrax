#!/usr/bin/env python3
from __future__ import annotations

"""
Compute wcoul0 at q=0 from eps0mat.h5's epsilon^{-1} head, following cohsex_isdf.

Formula (Ismail‑Beigi PRB 2006; same as cohsex_isdf.get_V_qG):

  1/eps^{-1}(q=0) = 1 + v(q0) f(q0),  f(q) = gamma |q|^2 (alpha=0 here)
  => gamma = (1/epshead.real - 1) / (|q0|^2 * v_trunc(q0))

Then Monte Carlo (Voronoi cell) average at q=0 with 2D truncation:

  v(q) = 8π/|q|^2 * (1 - e^{-zc kxy} cos(kz zc))
  vc(q) = (1 - e^{-zc kxy}) / kxy^2
  w(q)  = vc(q) / (1 + vc(q) * kxy^2 * gamma)
  wcoul0 = 8π * < w(q) >

The final result used in μν space is scaled by 1/Ω.
"""

import os
import argparse
import numpy as np

from isdf.common.wfnreader import WFNReader
from isdf.common import symmetry_maps
from isdf.common.epsreader import EPSReader


def wrap_points_to_voronoi(randcart: np.ndarray, bvec: np.ndarray, nmax: int = 1) -> np.ndarray:
    grid = np.arange(-nmax, nmax + 1)
    shifts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    candidate_shifts = shifts @ bvec
    diff = randcart[:, None, :] - candidate_shifts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    best_idx = np.argmin(dists, axis=1)
    return randcart - candidate_shifts[best_idx]


def compute_wcoul0_from_epshead(
    wfn: WFNReader,
    sym,
    epshead: complex,
    q0_crys=(0.001, 0.0, 0.0),
    nsamples: int = 131072,
    method: str = "sobol",
    reps: int = 1,
) -> tuple[float, float, float, float]:
    bvec = np.asarray(wfn.blat * wfn.bvec, dtype=float)
    kgrid = np.asarray(wfn.kgrid, dtype=float)
    zc = np.pi / bvec[2, 2]

    q0 = np.asarray(q0_crys, dtype=float)
    q0_cart = q0 @ bvec
    q0len = float(np.linalg.norm(q0_cart))
    vc_q0 = (1.0 - np.exp(-q0len * zc)) / max(q0len * q0len, 1e-30)
    epsr = float(np.real(epshead))
    gamma = (1.0 / max(epsr, 1e-30) - 1.0) / max(vc_q0 * q0len * q0len, 1e-30)

    # Monte Carlo in Voronoi cell of mini-BZ tile around Γ
    randlims = bvec.T @ (np.diag(1.0 / kgrid) @ np.linalg.inv(bvec.T))
    # Generate points: prefer Sobol QMC with Owen scrambling (power of two)
    use_sobol = method.lower() == "sobol"
    if use_sobol:
        try:
            from scipy.stats import qmc as _qmc  # type: ignore
            m = max(1, int(np.floor(np.log2(max(2, int(nsamples))))))
            npts = 1 << m
            means = []
            for rep in range(max(1, int(reps))):
                sob = _qmc.Sobol(d=3, scramble=True, seed=rep)
                U = sob.random_base2(m)
                randcart = (bvec.T @ U.T).T
                wrapped_cart = wrap_points_to_voronoi(randcart, bvec, nmax=1)
                rq = (randlims @ wrapped_cart.T).T
                rq[:, 2] = 0.0
                kxy = np.linalg.norm(rq[:, :2], axis=1)
                vc_q = (1.0 - np.exp(-kxy * zc)) / np.clip(kxy * kxy, 1e-30, None)
                wq = vc_q / (1.0 + vc_q * (kxy * kxy) * gamma)
                means.append(float(np.mean(wq)))
            wmean = float(np.mean(np.asarray(means)))
            wcoul0 = 8.0 * np.pi * wmean
            # Also compute vcoul0 consistently on the same points
            # We can reuse the last batch's kxy and vc_q to estimate vcoul0
            vc0 = 8.0 * np.pi * float(np.mean(vc_q))
            vol = float(wfn.cell_volume)
            wcoul0_scaled = wcoul0 / vol
            vc0_scaled = vc0 / vol
            return wcoul0, wcoul0_scaled, vc0, vc0_scaled
        except Exception as _e:
            print(f"Warning: Sobol sampling unavailable ({_e}); falling back to uniform.")
            use_sobol = False

    # Uniform fallback
    U = np.random.RandomState(0).rand(nsamples, 3)
    randcart = (bvec.T @ U.T).T
    wrapped_cart = wrap_points_to_voronoi(randcart, bvec, nmax=1)
    randqcart = (randlims @ wrapped_cart.T).T
    # 2D truncation: evaluate at kz=0 shell (same as cohsex_isdf)
    randqcart[:, 2] = 0.0
    kxy = np.linalg.norm(randqcart[:, :2], axis=1)

    vc_q = (1.0 - np.exp(-kxy * zc)) / np.clip(kxy * kxy, 1e-30, None)
    wq = vc_q / (1.0 + vc_q * (kxy * kxy) * gamma)
    wcoul0 = 8.0 * np.pi * float(np.mean(wq))
    vc0 = 8.0 * np.pi * float(np.mean(vc_q))
    vol = float(wfn.cell_volume)
    wcoul0_scaled = wcoul0 / vol
    vc0_scaled = vc0 / vol
    return wcoul0, wcoul0_scaled, vc0, vc0_scaled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compute wcoul0 from eps0mat.h5 epshead via Ismail–Beigi model")
    ap.add_argument("-i", "--input", default="cohsex_test.in", help="cohsex input file (to locate WFN.h5)")
    ap.add_argument("--eps0", default="eps0mat.h5", help="Path to eps0mat.h5 (epsilon^{-1} at q=0)")
    ap.add_argument("--q0", default="0.001,0,0", help="Crystal q0 for gamma calibration (comma-separated)")
    ap.add_argument("--nsamples", type=int, default=131072, help="MC samples (131072 recommended)")
    ap.add_argument("--method", default="sobol", choices=["sobol", "uniform"], help="Sampling method")
    ap.add_argument("--reps", type=int, default=1, help="Sobol replicate batches (scrambled)")
    args = ap.parse_args(argv)

    # Locate WFN.h5 from input
    from isdf.psp.get_DFT_mtxels import read_cohsex_input  # reuse robust parser
    params = read_cohsex_input(args.input)
    inp_dir = os.path.dirname(os.path.abspath(args.input))
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(inp_dir, wfn_path)

    # Load WFN and sym
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)

    # Load epshead from eps0mat.h5
    eps0_path = args.eps0 if os.path.isabs(args.eps0) else os.path.join(inp_dir, args.eps0)
    eps = EPSReader(eps0_path)
    epshead = eps.epshead

    q0 = tuple(float(x) for x in args.q0.split(","))
    wc0, wc0_scaled, vc0, vc0_scaled = compute_wcoul0_from_epshead(
        wfn,
        sym,
        epshead,
        q0_crys=q0,
        nsamples=int(args.nsamples),
        method=str(args.method),
        reps=int(args.reps),
    )

    print(f"epshead (epsilon^(-1) head, q=0): {complex(epshead)}")
    print(f"wcoul0 (unscaled): {wc0:.6f}")
    print(f"wcoul0 / Ω (for μν): {wc0_scaled:.6f}")
    print(f"vcoul0 (unscaled): {vc0:.6f}")
    print(f"vcoul0 / Ω (for μν): {vc0_scaled:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
