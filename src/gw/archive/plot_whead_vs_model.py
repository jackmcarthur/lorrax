#!/usr/bin/env python3
from __future__ import annotations

"""
Plot ε^{-1}_{00}(q, ω) head from χ via ε = I − χ v, then compare to small‑q models
adapted to ε^{-1} (so that multiplying by v reproduces W).

Series plotted
- ε^{-1}_{00}(q) (full): green diamonds (not connected)
- ε^{-1}_{00}(q) (γ model): connected line calibrated at the smallest nonzero |q|
- ε^{-1}_{00}(q) (S(0) model): connected line using q^T S(0) q
- ε^{-1}_{00}(q) (Schur): connected line using a block Schur on ε to get the head

Inputs
- chi0mat.h5 or chimat.h5 (BGW chi files; EPSReader layout)
- Optional eps0mat.h5 to calibrate γ from epshead as in cohsex_isdf
- WFN.h5 to read reciprocal lattice bvec and cell volume

Usage
  uv run python -m gw.plot_whead_vs_model -i cohsex_prod.in \
      --chi chi0mat.h5 --eps0 eps0mat.h5 --omega-index 0 --out W_head_vs_model.png

Notes
- Uses EPSReader to access chi matrices and vcoul per q in EPS ordering.
- Head index G=0 is taken from gind_rho2eps[iq,0].
- W is built in eps-space as diag(v) @ inv(I − χ @ diag(v)), then head is taken directly.
  (No wing/head zeroing; this is the exact W head for each q.)
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import os as _os
_os.environ.setdefault("JAX_ENABLE_X64", "1")
from common.epsreader import EPSReader
from common.wfnreader import WFNReader
from common.chi_from_dipole import read_dipole_h5, compute_S_omega


def v2d_trunc_head_from_q(q_crys: np.ndarray, bvec: np.ndarray) -> float:
    """Return 2D truncated head v(q) (8π/|q|^2)*f2d using q in crystal coords.
    (Used only for gamma calibration if eps0 is absent.)
    """
    q_cart = q_crys @ bvec
    denom = np.dot(q_cart, q_cart)
    if denom <= 1e-30:
        return 0.0
    kxy = np.linalg.norm(q_cart[:2])
    kz = q_cart[2]
    zc = np.pi / bvec[2, 2]
    f2d = 2.0 * (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
    return float(8.0 * np.pi / denom * f2d)


def calibrate_gamma_from_epshead(epshead: complex, bvec: np.ndarray, q0_crys=(0.001, 0.0, 0.0)) -> float:
    q0 = np.asarray(q0_crys, dtype=float)
    q0_cart = q0 @ bvec
    q0len = float(np.linalg.norm(q0_cart))
    zc = np.pi / bvec[2, 2]
    vc_q0 = (1.0 - np.exp(-q0len * zc)) / (q0len * q0len if q0len > 0 else 1.0)
    epsr = float(np.real(epshead))
    gamma = (1.0 / max(epsr, 1e-30) - 1.0) / max(vc_q0 * q0len * q0len, 1e-30)
    return float(gamma)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot W_head from full inversion vs small‑q model")
    ap.add_argument("-i", "--input", required=True, help="cohsex input file for locating WFN.h5")
    ap.add_argument("--chi", default="chi0mat.h5", help="Path to chi0mat.h5 or chimat.h5")
    ap.add_argument("--eps0", default=None, help="Optional eps0mat.h5 to calibrate gamma from epshead (unused by default)")
    ap.add_argument("--omega-index", type=int, default=0, help="Frequency index to use from chi file")
    ap.add_argument("--out", default="W_head_vs_model.png", help="Output plot path")
    ap.add_argument("--dipole", default="dipole.h5", help="Optional dipole.h5 (to compute S(0) model)")
    ap.add_argument("--fit-gamma", action="store_true", help="Fit gamma to first few q points (overrides eps0)")
    ap.add_argument("--nfit", type=int, default=4, help="Number of smallest-|q| points to use for gamma fit")
    args = ap.parse_args(argv)

    # Resolve paths
    from psp.get_DFT_mtxels import read_cohsex_input
    inp_dir = os.path.dirname(os.path.abspath(args.input))
    params = read_cohsex_input(args.input)
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(inp_dir, wfn_path)
    chi_path = args.chi if os.path.isabs(args.chi) else os.path.join(inp_dir, args.chi)
    dip_path = args.dipole if os.path.isabs(args.dipole) else os.path.join(inp_dir, args.dipole)
    eps0_path = None
    if args.eps0:
        eps0_path = args.eps0 if os.path.isabs(args.eps0) else os.path.join(inp_dir, args.eps0)

    # Load readers
    wfn = WFNReader(wfn_path)
    eps_chi = EPSReader(chi_path)
    bvec = np.asarray(wfn.blat * wfn.bvec, dtype=float)

    # Prepare q magnitudes and head index per q
    nq = int(eps_chi.nq)
    q_crys = np.asarray(eps_chi.qpts)
    if q_crys.ndim == 1:
        q_crys = q_crys.reshape(nq, -1)
    q_mag = np.linalg.norm(q_crys.astype(float), axis=1)
    g0_idx = []
    for iq in range(nq):
        try:
            g0_idx.append(int(eps_chi.gind_rho2eps[iq, 0]))
        except Exception:
            g0_idx.append(0)

    # Compute epsinv head and V_head per q
    omega_idx = int(args.omega_index)
    epsinv_full_list = []
    V_head = []
    kxy_list = []
    vc_head_list = []
    for iq in range(nq):
        nmtx_q = int(eps_chi.nmtx[iq])
        gi = g0_idx[iq]
        if gi < 0 or gi >= nmtx_q:
            epsinv_full_list.append(np.nan + 0j)
            V_head.append(np.nan)
            continue
        mat = np.asarray(eps_chi.get_eps_matrix(iq, ifreq=omega_idx, imatrix=0), dtype=np.complex128)[:nmtx_q, :nmtx_q]
        v = np.asarray(eps_chi.vcoul[iq, :nmtx_q], dtype=np.complex128)
        mtype = int(getattr(eps_chi, 'matrix_type', 2))
        # Build epsilon matrix E consistently
        if mtype == 2:  # chi0
            E = np.eye(nmtx_q, dtype=np.complex128) - mat @ np.diag(v)
        elif mtype == 1:  # epsilon
            E = mat
        else:  # 0: epsilon^{-1}
            try:
                E = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                E = np.linalg.pinv(mat)
        # epsinv head from full inversion
        try:
            Einv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            Einv = np.linalg.pinv(E)
        epsinv_full_list.append(complex(Einv[gi, gi]))
        V_head.append(float(np.real(v[gi])))
        # kxy for model
        q_cart = q_crys[iq] @ bvec
        kxy_list.append(float(np.linalg.norm(q_cart[:2])))
        vc_head_list.append(float(np.real(v[gi])))

    q_mag = np.asarray(q_mag, dtype=float)
    epsinv_full = np.asarray(epsinv_full_list, dtype=complex)
    V_head = np.asarray(V_head, dtype=float)
    kxy_arr = np.asarray(kxy_list, dtype=float)
    vc_head_arr = np.asarray(vc_head_list, dtype=float)

    # Calibrate gamma (default: single-point calibration at smallest nonzero |q|)
    gamma = None
    # Optional: fit over first few points
    if args.fit_gamma:
        idx_sorted = np.argsort(q_mag)
        idx_use = [i for i in idx_sorted if i < len(q_mag) and q_mag[i] > 1e-12][: max(1, int(args.nfit))]
        if idx_use:
            def sse(g):
                pred = vc_head_arr[idx_use] / (1.0 + vc_head_arr[idx_use] * (kxy_arr[idx_use] ** 2) * g)
                return float(np.sum((np.real(V_head[idx_use]) - pred) ** 2))
            grid = np.logspace(-8, 4, 64)
            vals = np.array([sse(g) for g in grid])
            gamma = float(grid[np.argmin(vals)])
    # Single-point calibration at first nonzero |q|
    if gamma is None:
        idx_sorted = np.argsort(q_mag)
        i1 = next((i for i in idx_sorted if q_mag[i] > 1e-12), None)
        if i1 is None:
            print("No nonzero q entries found; cannot calibrate model.")
        else:
            v1 = float(vc_head_arr[i1])
            e1 = complex(epsinv_full[i1]) if np.isfinite(epsinv_full[i1]) else None
            kxy1 = float(kxy_arr[i1])
            if e1 is None or float(np.real(e1)) == 0.0 or v1 == 0.0 or kxy1 == 0.0:
                print("Insufficient data at first nonzero q to calibrate gamma.")
            else:
                # epsinv1 = 1/(1 + v1 kxy1^2 gamma) => gamma = (1/epsinv1 - 1)/(v1 kxy1^2)
                epsinv1 = float(np.real(e1))
                gamma = (1.0 / epsinv1 - 1.0) / (v1 * (kxy1 ** 2))
                print(f"Calibrated gamma from first nonzero q idx={i1}: v={v1:.6e}, epsinv={epsinv1:.6e}, kxy={kxy1:.6e} => gamma={gamma:.6e}")

    # Build gamma-model (from single-point or fit)
    W_model = None
    if gamma is not None:
        W_model = vc_head_arr / (1.0 + vc_head_arr * (kxy_arr ** 2) * gamma)

    # Build S(0)-based model if dipole.h5 is available; fall back to fit S from chi heads
    W_sdip = None
    epsinv_sdip = None
    if os.path.exists(dip_path):
        try:
            dipole_cart, deltaE = read_dipole_h5(dip_path)
            # Use only the first k-point, to mirror plot_vmtxel_chi0q convention
            dipole_cart = np.asarray(dipole_cart)
            deltaE = np.asarray(deltaE)
            nk_dip, nb = int(dipole_cart.shape[1]), int(dipole_cart.shape[2])
            dipole_cart_k0 = dipole_cart[:, :1, :, :]
            deltaE_k0 = deltaE[:1, :, :]
            nelec = int(wfn.nelec)
            f_nk = np.zeros((1, nb), dtype=float)
            f_nk[:, :max(0, min(nelec, nb))] = 1.0
            S_all = compute_S_omega(
                dipole_cart_k0,
                deltaE_k0,
                np.asarray(f_nk, dtype=np.float64),
                float(wfn.cell_volume),
                1,
                int(wfn.nspin),
                int(wfn.nspinor),
                np.asarray([0.0], dtype=np.float64),
                eta=0.0,
            )
            S_cart = np.asarray(S_all[0], dtype=np.complex128)
            # Evaluate W_sdip(q) = v(q) / (1 - v(q) q^T S q)
            W_sdip = np.zeros(nq, dtype=np.complex128)
            epsinv_sdip = np.zeros(nq, dtype=np.complex128)
            zc = np.pi / bvec[2, 2]
            for i in range(nq):
                q_cart = q_crys[i] @ bvec
                denom = float(np.dot(q_cart, q_cart))
                if denom <= 1e-30:
                    W_sdip[i] = np.nan + 0j
                    epsinv_sdip[i] = 1.0 + 0j
                    continue
                kxy = float(np.linalg.norm(q_cart[:2]))
                kz = float(q_cart[2])
                # Use the same 2D truncation convention as in vcoul.py: v = (8π/|q|^2) * (1 - e^{-zc kxy} cos(kz zc))
                f2d = (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
                vq = (8.0 * np.pi / denom) * f2d
                qSq = (q_cart.T @ S_cart @ q_cart).astype(np.complex128)
                W_sdip[i] = vq / (1.0 - vq * qSq)
                epsinv_sdip[i] = 1.0 / (1.0 - vq * qSq)
        except Exception as e:
            print(f"Warning: failed to compute S(0) model from dipole.h5: {e}")
            W_sdip = None
            epsinv_sdip = None

    # If S(0) path failed, fit S from chi head over first few q
    if epsinv_sdip is None:
        try:
            idx_sorted = np.argsort(q_mag)
            fit_idx = [i for i in idx_sorted if q_mag[i] > 1e-12][:4]
            feats = []
            targets = []
            for i in fit_idx:
                nmtx_q = int(eps_chi.nmtx[i])
                gi = g0_idx[i]
                mat = np.asarray(eps_chi.get_eps_matrix(i, ifreq=omega_idx, imatrix=0), dtype=np.complex128)[:nmtx_q, :nmtx_q]
                if getattr(eps_chi, 'matrix_type', 2) != 2:
                    continue  # only fit from chi files
                chi00 = complex(mat[gi, gi])
                qc = (q_crys[i] @ bvec).astype(float)
                feats.append([qc[0]**2, qc[1]**2, qc[2]**2, 2*qc[0]*qc[1], 2*qc[0]*qc[2], 2*qc[1]*qc[2]])
                targets.append(float(np.real(chi00)))
            if feats:
                A = np.asarray(feats, dtype=float)
                b = np.asarray(targets, dtype=float)
                params, *_ = np.linalg.lstsq(A, b, rcond=None)
                Fxx, Fyy, Fzz, Fxy2, Fxz2, Fyz2 = params
                Fxy, Fxz, Fyz = 0.5*Fxy2, 0.5*Fxz2, 0.5*Fyz2
                S_cart_fit = np.array([[Fxx, Fxy, Fxz], [Fxy, Fyy, Fyz], [Fxz, Fyz, Fzz]], dtype=np.complex128)
                zc = np.pi / bvec[2, 2]
                epsinv_sdip = np.zeros(nq, dtype=np.complex128)
                for i in range(nq):
                    qc = (q_crys[i] @ bvec).astype(float)
                    denom = float(np.dot(qc, qc))
                    if denom <= 1e-30:
                        epsinv_sdip[i] = 1.0 + 0j
                        continue
                    kxy = float(np.linalg.norm(qc[:2])); kz = float(qc[2])
                    f2d = (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
                    vq = (8.0 * np.pi / denom) * f2d
                    qSq = (qc.T @ S_cart_fit @ qc).astype(np.complex128)
                    epsinv_sdip[i] = 1.0 / (1.0 - vq * qSq)
        except Exception as e:
            print(f"Warning: failed to fit S from chi heads: {e}")

    # Plot
    # Build epsinv head arrays from prior W arrays by dividing by V_head
    # (but recompute directly for clarity below)
    plt.figure(figsize=(8, 5))
    # Placeholders; actual epsinv series computed below
    # Labels and styles per request
    # Will plot: epsinv_full (green diamonds, no line), epsinv_gamma, epsinv_sdip, epsinv_schur (lines)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = Path(args.out)
    # Will save after computing epsinv series

    # Print all points
    # Now compute epsinv series from χ and models (epsinv_full computed above)

    # Schur head: epsinv_00 = 1 / (ε00 − ε0w εww^{-1} εw0)
    epsinv_schur = np.full_like(epsinv_full, np.nan + 0j)
    for iq in range(nq):
        nmtx_q = int(eps_chi.nmtx[iq])
        gi = g0_idx[iq]
        if gi < 0 or gi >= nmtx_q:
            continue
        chi = eps_chi.get_eps_matrix(iq, ifreq=omega_idx, imatrix=0)
        chi = np.asarray(chi, dtype=np.complex128)[:nmtx_q, :nmtx_q]
        v = np.asarray(eps_chi.vcoul[iq, :nmtx_q], dtype=np.complex128)
        E = np.eye(nmtx_q, dtype=np.complex128) - chi @ np.diag(v)
        # Partition
        mask = np.ones(nmtx_q, dtype=bool)
        mask[gi] = False
        E00 = complex(E[gi, gi])
        E0w = E[gi, mask]
        Ew0 = E[mask, gi]
        Eww = E[mask, :][:, mask]
        try:
            y = np.linalg.solve(Eww, Ew0)
        except np.linalg.LinAlgError:
            y = np.linalg.pinv(Eww) @ Ew0
        Eff = E00 - (E0w @ y)
        if Eff != 0:
            epsinv_schur[iq] = 1.0 / Eff
        else:
            epsinv_schur[iq] = np.nan + 0j

    # Gamma-model epsinv: epsinv_gamma = 1/(1 + v_head kxy^2 gamma)
    epsinv_gamma = None
    if gamma is not None:
        epsinv_gamma = 1.0 / (1.0 + vc_head_arr * (kxy_arr ** 2) * gamma)

    # S(0)-based epsinv: epsinv_sdip = 1/(1 − v_head · q^T S q)
    # epsinv_sdip already computed above (either from dipole S(0) or S-fit); ensure shape

    # Wing-tensor linear fit in epsinv domain:
    # For small q, 1/epsinv(q) − 1 = v(q) · q^T (S(0)+F) q. Define
    # t(q) = 1/epsinv_full(q) − 1 − v(q) q^T S(0) q = v(q) · q^T F q.
    # Fit F from first few q using features v(q)·[ux^2, uy^2, uz^2, 2uxuy, 2uxuz, 2uyuz].
    epsinv_sdip_Ffit = None
    try:
        # Need epsinv_full and (optionally) S_cart
        idx_sorted = np.argsort(q_mag)
        fit_idx = [i for i in idx_sorted if q_mag[i] > 1e-12][:4]
        feats = []
        targets = []
        zc = np.pi / bvec[2, 2]
        for i in fit_idx:
            qc = (q_crys[i] @ bvec).astype(float)
            qlen = float(np.linalg.norm(qc))
            if qlen <= 0 or not np.isfinite(epsinv_full[i]):
                continue
            kxy = float(np.linalg.norm(qc[:2])); kz = float(qc[2])
            f2d = (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
            vq = (8.0 * np.pi) / float(np.dot(qc, qc)) * f2d
            # t = 1/epsinv - 1 - v q^T S(0) q
            t_i = float((1.0 / np.real(epsinv_full[i])) - 1.0)
            if 'S_cart' in locals() and S_cart is not None:
                t_i -= float(vq * np.real(qc.T @ S_cart @ qc))
            u = qc / qlen
            feats.append(vq * np.array([u[0]*u[0], u[1]*u[1], u[2]*u[2], 2*u[0]*u[1], 2*u[0]*u[2], 2*u[1]*u[2]], dtype=float))
            targets.append(t_i)
        if feats:
            A = np.asarray(feats, dtype=float)
            b = np.asarray(targets, dtype=float)
            params, *_ = np.linalg.lstsq(A, b, rcond=None)
            Fxx, Fyy, Fzz, Fxy2, Fxz2, Fyz2 = params
            Fxy, Fxz, Fyz = 0.5*Fxy2, 0.5*Fxz2, 0.5*Fyz2
            F_fit = np.array([[Fxx, Fxy, Fxz], [Fxy, Fyy, Fyz], [Fxz, Fyz, Fzz]], dtype=np.complex128)
            epsinv_sdip_Ffit = np.zeros(nq, dtype=np.complex128)
            for i in range(nq):
                qc = (q_crys[i] @ bvec).astype(float)
                denom = float(np.dot(qc, qc))
                if denom <= 1e-30:
                    epsinv_sdip_Ffit[i] = 1.0 + 0j
                    continue
                kxy = float(np.linalg.norm(qc[:2])); kz = float(qc[2])
                f2d = (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
                vq = (8.0 * np.pi / denom) * f2d
                Seff = (S_cart + F_fit) if 'S_cart' in locals() and S_cart is not None else F_fit
                qSq = (qc.T @ Seff @ qc).astype(np.complex128)
                epsinv = 1.0 / (1.0 - vq * qSq)
                epsinv_sdip_Ffit[i] = epsinv
    except Exception as e:
        print(f"Warning: epsinv-domain wing fit failed: {e}")
    # Wing-corrected model using a frozen Schur block from the first nonzero |q|
    # Build sigma_wing_ref = (ε0w εww^{-1} εw0) at q_ref, then for each q set
    #   ε00_model(q) = 1 − v_head(q) · q^T S(0) q, and
    #   epsinv_sdip_schur(q) = 1 / (ε00_model(q) − sigma_wing_ref)
    epsinv_sdip_schur = None
    if epsinv_sdip is not None:
        # Find reference q index (first nonzero |q|)
        idx_sorted = np.argsort(q_mag)
        iref = next((i for i in idx_sorted if q_mag[i] > 1e-12), None)
        if iref is not None:
            # Build epsilon at q_ref
            nmtx_q = int(eps_chi.nmtx[iref])
            gi = g0_idx[iref]
            if 0 <= gi < nmtx_q:
                chi_ref = np.asarray(eps_chi.get_eps_matrix(iref, ifreq=omega_idx, imatrix=0), dtype=np.complex128)[:nmtx_q, :nmtx_q]
                v_ref = np.asarray(eps_chi.vcoul[iref, :nmtx_q], dtype=np.complex128)
                E_ref = np.eye(nmtx_q, dtype=np.complex128) - chi_ref @ np.diag(v_ref)
                mask = np.ones(nmtx_q, dtype=bool); mask[gi] = False
                E00_ref = complex(E_ref[gi, gi])
                E0w_ref = E_ref[gi, mask]
                Ew0_ref = E_ref[mask, gi]
                Eww_ref = E_ref[mask, :][:, mask]
                try:
                    y = np.linalg.solve(Eww_ref, Ew0_ref)
                except np.linalg.LinAlgError:
                    y = np.linalg.pinv(Eww_ref) @ Ew0_ref
                sigma_wing_ref = E0w_ref @ y  # scalar
                # Recompute epsilon00_model(q) from S(0)
                epsinv_sdip_schur = np.full_like(epsinv_sdip, np.nan + 0j)
                if W_sdip is not None:
                    # recover q^T S q from W_sdip formula: W = v / (1 - v qSq) => qSq = (1 - v/W)/v
                    # but numerical noise may be large; instead recompute qSq directly if S_cart available
                    try:
                        # reuse S_cart built earlier
                        S_cart  # type: ignore[name-defined]
                    except NameError:
                        S_cart = None
                for i in range(nq):
                    q_cart = q_crys[i] @ bvec
                    denom = float(np.dot(q_cart, q_cart))
                    if denom <= 1e-30:
                        epsinv_sdip_schur[i] = 1.0 + 0j
                        continue
                    zc = np.pi / bvec[2, 2]
                    kxy = float(np.linalg.norm(q_cart[:2])); kz = float(q_cart[2])
                    f2d = (1.0 - np.exp(-zc * kxy) * np.cos(kz * zc))
                    vq = (8.0 * np.pi / denom) * f2d
                    if 'S_cart' in locals() and S_cart is not None:
                        qSq = (q_cart.T @ S_cart @ q_cart).astype(np.complex128)
                    else:
                        # Fall back: derive qSq from epsinv_sdip if available
                        if V_head[i] != 0 and np.isfinite(epsinv_sdip[i]):
                            # epsinv = 1/(1 - v qSq) => qSq = (1 - 1/epsinv)/v
                            qSq = (1.0 - 1.0/epsinv_sdip[i]) / vq
                        else:
                            qSq = 0.0 + 0.0j
                    E00_model = 1.0 - vq * qSq
                    Eff_model = E00_model - sigma_wing_ref
                    epsinv_sdip_schur[i] = (1.0 / Eff_model) if Eff_model != 0 else (np.nan + 0j)

    # Add explicit q=0 point with epsinv(0)=1.0 for all series
    q_mag0 = np.concatenate(([0.0], q_mag))
    V_head0 = np.concatenate(([np.nan], V_head))
    kxy0 = np.concatenate(([0.0], kxy_arr))
    epsinv_full0 = np.concatenate(([1.0 + 0j], epsinv_full))
    epsinv_gamma0 = np.concatenate(([1.0], epsinv_gamma)) if epsinv_gamma is not None else None
    epsinv_sdip0 = np.concatenate(([1.0 + 0j], epsinv_sdip)) if epsinv_sdip is not None else None
    epsinv_schur0 = np.concatenate(([1.0 + 0j], epsinv_schur))

    # Plot with requested styling
    # 1) ε^{-1}_{00} (full): green diamonds, no lines
    plt.plot(q_mag0, np.real(epsinv_full0), marker='D', linestyle='None', color='green', label=r'$\varepsilon^{-1}_{00}\,(\mathrm{full})$')
    # 2) ε^{-1}_{00} (γ model), connected line
    if epsinv_gamma0 is not None:
        plt.plot(q_mag0, np.real(epsinv_gamma0), 'b-', linewidth=1.5, label=r'$\varepsilon^{-1}_{00}\,(\gamma\ \mathrm{model})$')
    # 3) ε^{-1}_{00} (S(0) model), connected line
    if epsinv_sdip0 is not None:
        plt.plot(q_mag0, np.real(epsinv_sdip0), 'm:', linewidth=1.8, label=r'$\varepsilon^{-1}_{00}\,(S(0)\ \mathrm{model})$')
    # Only display the three requested series for now: full, gamma model, S(0) model

    plt.xlabel(r'$|\mathbf{q}|$ (crystal)')
    plt.ylabel(r'$\varepsilon^{-1}_{00}(\mathbf{q},\omega)$')
    plt.title(r'$\varepsilon^{-1}_{00}(q)$: full vs models')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Wrote plot to {out.resolve()}")

    # Print a full table with epsinv series
    if epsinv_gamma0 is None and epsinv_sdip0 is None:
        print("\nAll points (|q|, Re epsinv_full, Im epsinv_full, V_head, kxy):")
        for i in range(len(q_mag0)):
            print(f"  {q_mag0[i]:.6e}  {np.real(epsinv_full0[i]): .6e}  {np.imag(epsinv_full0[i]): .6e}  {V_head0[i]: .6e}  {kxy0[i]: .6e}")
    else:
        print(f"\nGamma used for model: {gamma:.6e}" if gamma is not None else "\nGamma not used")
        header = "All points (|q|, Re epsinv_full, V_head, kxy, epsinv_gamma, epsinv_S(0), epsinv_Schur, epsinv_S(0)+Schur(q1), epsinv_S(0)+F_fit):"
        print(header)
        for i in range(len(q_mag0)):
            eg = float(np.real(epsinv_gamma0[i])) if epsinv_gamma0 is not None else float('nan')
            es = float(np.real(epsinv_sdip0[i])) if epsinv_sdip0 is not None else float('nan')
            esc = float(np.real(epsinv_schur0[i]))
            print(f"  {q_mag0[i]:.6e}  {np.real(epsinv_full0[i]): .6e}  {V_head0[i]: .6e}  {kxy0[i]: .6e}  {eg: .6e}  {es: .6e}  {esc: .6e}")
    # Re-plot using two stacked subplots and overwrite output with requested layout
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        # Top: eps^{-1}_{00}
        ax1.plot(q_mag0, np.real(epsinv_full0), marker='D', linestyle='None', color='green',
                 label=r'$\varepsilon^{-1}_{00}\,(\mathrm{full})$')
        if epsinv_gamma0 is not None:
            ax1.plot(q_mag0, np.real(epsinv_gamma0), 'b-', linewidth=1.5,
                     label=r'$\varepsilon^{-1}_{00}\,(\gamma\ \mathrm{model})$')
        if epsinv_sdip0 is not None:
            ax1.plot(q_mag0, np.real(epsinv_sdip0), 'm:', linewidth=1.8,
                     label=r'$\varepsilon^{-1}_{00}\,(S(0)\ \mathrm{model})$')
        ax1.set_ylabel(r'$\varepsilon^{-1}_{00}(\mathbf{q},\omega)$')
        ax1.set_title(r'$\varepsilon^{-1}_{00}(q)$ and $W_{00}(q)$: full vs models')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Bottom: W_{00}(q) = v_head * epsinv, with V_{00} overlay
        v_head0_plot = V_head0.copy()
        if len(v_head0_plot) > 0:
            v_head0_plot[0] = 1.0e8
        W_full0 = np.real(epsinv_full0) * v_head0_plot
        ax2.plot(q_mag0, W_full0, marker='D', linestyle='None', color='green',
                 label=r'$W_{00}\,(\mathrm{full})$')
        if epsinv_gamma0 is not None:
            W_gamma0 = np.real(epsinv_gamma0) * v_head0_plot
            ax2.plot(q_mag0, W_gamma0, 'b-', linewidth=1.5,
                     label=r'$W_{00}\,(\gamma\ \mathrm{model})$')
        if epsinv_sdip0 is not None:
            W_sdip0 = np.real(epsinv_sdip0) * v_head0_plot
            ax2.plot(q_mag0, W_sdip0, 'm:', linewidth=1.8,
                     label=r'$W_{00}\,(S(0)\ \mathrm{model})$')
        ax2.plot(q_mag0, v_head0_plot, 'k--', linewidth=1.0, label=r'$V_{00}$')
        ax2.set_xlabel(r'$|\mathbf{q}|$ (crystal)')
        ax2.set_ylabel(r'$W_{00}(\mathbf{q},\omega)$')
        ax2.set_ylim(0, 1.1*W_full0[1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        fig.tight_layout()
        fig.savefig(out, dpi=160)
        print(f"Overwrote plot with two-subplot figure at {out.resolve()}")
    except Exception as e:
        print(f"Warning: failed to write two-subplot figure: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
