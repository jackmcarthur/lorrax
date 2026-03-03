#!/usr/bin/env python3
from __future__ import annotations

"""
Compare χ_{00}(q, ω) from BGW chi*.h5 (EPSReader) and from the dipole-based
k·p small-q model on the same axes for the first two frequencies.

Usage
  uv run python -m gw_isdf.plot_chi0q_compare \
      -i cohsex_test.in --chi chi0mat.h5 --dipole dipole.h5 \
      --out chi_compare.png

Notes
- Frequencies for comparison are taken from the chi file header (first two entries).
- The dipole S(ω) path uses compute_S_omega with occupations from WFN.h5.
- χ(q, ω) from S is evaluated as q^T (B^T S B) q using the same q-points as in chi0mat.h5.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os as _os
_os.environ.setdefault("JAX_ENABLE_X64", "1")

from isdf.common.epsreader import EPSReader
from isdf.common.wfnreader import WFNReader
from isdf.common import symmetry_maps
from isdf.common.chi_from_dipole import read_dipole_h5, compute_S_omega
from isdf.psp.get_DFT_mtxels import read_cohsex_input  # type: ignore


def build_occupations(wfn: WFNReader) -> np.ndarray:
    nb = int(wfn.nbands)
    nk = int(wfn.nkpts) if hasattr(wfn, 'nkpts') else int(wfn.kpoints.shape[0])
    nelec = int(wfn.nelec)
    nelec = max(0, min(nelec, nb))
    f = np.zeros((nk, nb), dtype=float)
    f[:, :nelec] = 1.0
    return f


def read_bgw_chi00_series(chi_path: str, pick_ifreq: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    eps = EPSReader(chi_path)
    nq = int(eps.nq)
    q_crys = np.asarray(eps.qpts)
    if q_crys.ndim == 1:
        q_crys = q_crys.reshape(nq, -1)
    q_abs = np.linalg.norm(q_crys.astype(float), axis=1)

    out: dict[int, np.ndarray] = {}
    # Head G=0 index for each q
    g0_idx = []
    for iq in range(nq):
        try:
            g0 = int(eps.gind_rho2eps[iq, 0])
        except Exception:
            g0 = 0
        g0_idx.append(g0)
    for ifreq in pick_ifreq:
        vals = np.zeros(nq, dtype=complex)
        for iq in range(nq):
            nmtx_q = int(eps.nmtx[iq])
            gi = g0_idx[iq]
            if gi < 0 or gi >= nmtx_q:
                vals[iq] = np.nan + 0j
                continue
            mat = np.asarray(eps.get_eps_matrix(iq, ifreq=ifreq, imatrix=0), dtype=complex)[:nmtx_q, :nmtx_q]
            vals[iq] = mat[gi, gi]
        out[ifreq] = vals
    return q_abs, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Overlay χ_{00}(q,ω) from BGW chi and dipole S(ω)")
    ap.add_argument('-i', '--input', default='cohsex_test.in', help='cohsex input to find WFN.h5')
    ap.add_argument('--chi', default='chi0mat.h5', help='Path to chi0mat.h5')
    ap.add_argument('--dipole', default='dipole.h5', help='Path to dipole.h5')
    ap.add_argument('--out', default='chi_compare.png', help='Output figure')
    ap.add_argument('--eta', type=float, default=0.0, help='Broadening eta (eV) for S(ω) if header lacks it')
    args = ap.parse_args(argv)

    # Resolve paths relative to input
    inp = Path(args.input).resolve()
    params = read_cohsex_input(str(inp))
    wfn_path = Path(params.get('wfn_file', 'WFN.h5'))
    if not wfn_path.is_absolute():
        wfn_path = (inp.parent / wfn_path).resolve()
    chi_path = Path(args.chi)
    if not chi_path.is_absolute():
        chi_path = (inp.parent / chi_path).resolve()
    dip_path = Path(args.dipole)
    if not dip_path.is_absolute():
        dip_path = (inp.parent / dip_path).resolve()

    # Readers
    wfn = WFNReader(str(wfn_path))
    eps = EPSReader(str(chi_path))
    sym = symmetry_maps.SymMaps(wfn)

    # Frequencies: take first two rows from chi header; use eps-provided broadening
    nfreq = int(eps.nfreq)
    pick_ifreq = [i for i in range(min(2, max(1, nfreq)))]
    try:
        F = np.asarray(eps.freqs)
        if F.ndim == 2 and F.shape[1] >= 2:
            # Interpret header as eV; convert to Ry for computation
            RYD2EV = 13.605693009
            omega_eV = F[:, 0]
            eta_eV = F[:, 1]
            omegas_eV = [float(omega_eV[i]) for i in pick_ifreq]
            etas_eV = [float(eta_eV[i]) for i in pick_ifreq]
            omegas_Ry = [w / RYD2EV for w in omegas_eV]
            etas_Ry = [g / RYD2EV for g in etas_eV]
        else:
            RYD2EV = 13.605693009
            omega_eV = F.reshape(-1)
            omegas_eV = [float(omega_eV[i]) for i in pick_ifreq]
            etas_eV = [float(args.eta) for _ in pick_ifreq]
            omegas_Ry = [w / RYD2EV for w in omegas_eV]
            etas_Ry = [g / RYD2EV for g in etas_eV]
    except Exception:
        RYD2EV = 13.605693009
        omegas_eV = [0.0 for _ in pick_ifreq]
        etas_eV = [float(args.eta) for _ in pick_ifreq]
        omegas_Ry = [0.0 for _ in pick_ifreq]
        etas_Ry = [g / RYD2EV for g in etas_eV]

    # BGW χ00(q,ω)
    q_abs, chi_bgw = read_bgw_chi00_series(str(chi_path), pick_ifreq)

    # Dipole S(ω)
    dipole_cart, deltaE = read_dipole_h5(str(dip_path))
    # Use all k-points as in plot_vmtxel_chi0q to match its frequency dependence
    nk_tot = int(dipole_cart.shape[1])
    nb = int(wfn.nbands)
    nelec = int(wfn.nelec)
    nelec = max(0, min(nelec, nb))
    f_nk = np.zeros((nk_tot, nb), dtype=float)
    f_nk[:, :nelec] = 1.0
    # compute_S_omega expects JAX arrays; build f_nk as jnp now
    import jax.numpy as jnp
    # Evaluate S(ω) for each requested ω with its own broadening η from chi header
    S_list = []
    for w_val, eta_val in zip(omegas_Ry, etas_Ry):
        S_j = compute_S_omega(
            jnp.asarray(dipole_cart),
            jnp.asarray(deltaE),
            jnp.asarray(f_nk, dtype=jnp.float64),
            float(wfn.cell_volume),
            nk_tot,
            int(wfn.nspin),
            int(wfn.nspinor),
            jnp.asarray([w_val], dtype=jnp.float64),
            eta=float(eta_val),
        )[0]
        S_list.append(np.asarray(S_j))
    S_all = np.stack(S_list, axis=0)

    # Transform S_cart -> S_crys using B so that χ(q)=q^T S_crys q with q in crystal coords
    B = np.asarray(wfn.bvec, dtype=float).T * float(wfn.blat)
    q_crys = np.asarray(eps.qpts)
    if q_crys.ndim == 1:
        q_crys = q_crys.reshape(-1, 3)

    chi_sdip: dict[int, np.ndarray] = {}
    for idx, ifreq in enumerate(pick_ifreq):
        S_cart = np.asarray(S_all[idx])  # (3,3)
        S_crys = B.T @ S_cart @ B
        vals = np.einsum('qi,ij,qj->q', q_crys, S_crys, q_crys, optimize=True)
        chi_sdip[ifreq] = vals.astype(complex)

    # Plot overlay
    plt.figure(figsize=(8, 5))
    colors = ['tab:blue', 'tab:orange']
    main_ax = plt.gca()
    for j, ifreq in enumerate(pick_ifreq):
        w_eV = omegas_eV[j] if 'omegas_eV' in locals() else (omegas_Ry[j] * 13.605693009)
        eta_eV = etas_eV[j] if 'etas_eV' in locals() else (etas_Ry[j] * 13.605693009)
        label_b = rf"BGW $\omega={w_eV:.4g}+{eta_eV:.3g}i$"
        label_s = rf"S-dip $\omega={w_eV:.4g}+{eta_eV:.3g}i$"
        main_ax.plot(q_abs, np.real(chi_bgw[ifreq]), 'o-', ms=3, lw=1.2, color=colors[j], label=label_b)
        main_ax.plot(q_abs, np.real(chi_sdip[ifreq]), 'x--', ms=3, lw=1.2, color=colors[j], label=label_s)

    plt.xlabel(r"$|q|$ (crystal units)")
    plt.ylabel(r"$\chi_{00}(q,\omega)$ (real)")
    plt.title(r"$\chi_{00}(q,\omega)$: BGW vs dipole $S(\omega)$")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', ncols=2)

    # Add zoomed inset in bottom-left quadrant focusing on x∈[0,0.05], y∈[-0.0025,0]
    axins = inset_axes(main_ax, width="45%", height="55%", loc="lower left", borderpad=1.0)
    for j, ifreq in enumerate(pick_ifreq):
        axins.plot(q_abs, np.real(chi_bgw[ifreq]), 'o-', ms=2.5, lw=1.0, color=colors[j])
        axins.plot(q_abs, np.real(chi_sdip[ifreq]), 'x--', ms=2.5, lw=1.0, color=colors[j])
    axins.set_xlim(0.0, 0.05)
    axins.set_ylim(-0.0015, 0.0003)
    # Hide inset axes: no ticks, labels, or grid; keep the box frame
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axins.grid(False)

    plt.tight_layout()
    out = Path(args.out)
    plt.savefig(out, dpi=160)
    print(f"Wrote plot to {out.resolve()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
