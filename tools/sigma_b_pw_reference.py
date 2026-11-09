#!/usr/bin/env python3
"""Exact plane-wave (ISDF-free) bare transverse exchange Σ^B of a bispinor run.

    python3 tools/sigma_b_pw_reference.py --wfn WFN.h5 --nsigma 38 --out pw.npz \
        [--ecut RY] [--sys-dim 2] [--alpha-scale 1] [--kleft 0,3] [--xcheck] [--threads N]

The transverse twin of the exact Σ_x of ``docs/theory/isdf-exchange-accuracy.md``:
the reference the ISDF ``sigTT`` of a ``bispinor = true``, ``bispinor_gw =
bare_transverse``, ``compute_mode = x_only`` run converges to, with no centroid,
no ζ and no V tile anywhere:

    Σ^B_nm(k) = (1/N_k) Σ_k' Σ_{v occ} Σ_{K = k−k'+G, 0 < |K|² ≤ E_c}
                v(K) Σ_ij P^T_ij(K̂) M^i_nv(K) R^j_vm(K)                       (Ry)
    Σ^x_nm(k) = −(1/N_k) Σ_k' Σ_{v occ} Σ_K v(K) M^0_nv(K) R^0_vm(K)

    M^I_nv(K) = ∫_Ω Ψ_nk† γ̃^I e^{+iK·r} Ψ_vk',   R^J_vm(K) = ∫_Ω Ψ_vk'† γ̃^J e^{−iK·r} Ψ_mk,
    γ̃^0 = 1_4, γ̃^i = α^i,  P^T = δ − K̂K̂,
    v(K) = (8π/(Ω K²)) [1 − e^{−z_c K_∥} cos(K_z z_c)]   (sys_dim 2, z_c = L_z/2; 3D: no bracket).

Σ^B = −Σ_ij α^i G α^j t_ij with t = −v P^T (``docs/theory/bispinor-gw.md`` §2, §4)
on the occupied G: the Coulomb-gauge minus and the exchange minus cancel, so
Σ^B is positive semidefinite.  Ψ is the normalized RKB lift
r(k+G)[ψ_L; (α/2)σ·(k+G)ψ_L], r = (1 + (α/2)²|k+G|²)^(−1/2) (§1).  Every pair
function is formed on the WFN's FFT box (a pair spectrum fits it without
aliasing) with the Bloch phases carried explicitly: the stored k and k' enter
only through K = k − k' + G.  R is a second, independent FFT of the reversed
pair (R^J_vm(K) = conj(M^J_mv(K)) exactly), so Σ_nm = conj(Σ_mn) and
R = conj(M) are tests of the bookkeeping, not identities of the code.  The
q = 0, G = 0 slot is excluded (``head_correction = off``: no TT head, v(0) = 0)
and no mini-BZ body average is taken (``mc_average_vcoul_body`` is inert in 2D).
The WFN must hold the whole k grid; nothing is unfolded here.

Output (npz): ``sigma_b_ry`` and ``sigma_x_ry`` ``(nk, nsigma, nsigma)`` complex
at the k rows named by ``k_rows`` (WFN order), ``kpts`` crystal, ``enk_ev``,
and a JSON ``receipt`` of every check.  ``--xcheck`` (LORRAX ``src`` on
PYTHONPATH, CPU JAX) compares this file's lift with
``common.bispinor_init.lift_to_4spinor`` and its kernel with the production TT
tile builder ``gw.v_q_bispinor._make_per_q_v_builder_for_tile``.

Cost: 8·nsigma·nocc box FFTs per (k, k') pair and every lifted band on the box
in memory.  MoS2 3×3 (24×24×80 box, 38 Σ bands, 26 occupied) takes ~100 s per
k row on 7 cores, so one process per ``--kleft`` row on one CPU node; a cell
with a 10⁶-point box needs a device port.  First use: the current-centroid
ladder of the shipped selector on MoS2 3×3 SOC (2026-11-09).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

RY_EV = 13.605693122994
#: α_FS/2 in Hartree units; equal to ``common.bispinor_init.HALFALPHA``
#: (asserted under ``--xcheck``), spelled here so the reference stands alone.
HALFALPHA = 0.00364867628215
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                 dtype=np.complex128)


def _fft_backend(threads):
    try:
        import scipy.fft as sf
        return (lambda a: sf.fftn(a, axes=(-3, -2, -1), workers=threads),
                lambda a: sf.ifftn(a, axes=(-3, -2, -1), workers=threads))
    except ImportError:   # pragma: no cover
        return (lambda a: np.fft.fftn(a, axes=(-3, -2, -1)),
                lambda a: np.fft.ifftn(a, axes=(-3, -2, -1)))


def read_wfn(path, nb):
    """ψ_L(G) of bands [0, nb) at every stored k, and the cell (bohr, Ry)."""
    with h5py.File(path, "r") as f:
        h = f["mf_header"]
        kp, cr = h["kpoints"], h["crystal"]
        if int(kp["nspin"][()]) != 1 or int(kp["nspinor"][()]) != 2:
            raise ValueError("need a noncollinear spinor WFN (nspin 1, nspinor 2)")
        out = dict(
            rk=np.asarray(kp["rk"][()], np.float64),
            ngk=np.asarray(kp["ngk"][()], np.int64),
            kgrid=tuple(int(x) for x in kp["kgrid"][()]),
            occ=np.asarray(kp["occ"][()], np.float64)[0],
            el=np.asarray(kp["el"][()], np.float64)[0],
            ecutwfc=float(kp["ecutwfc"][()]),
            avec=np.asarray(cr["avec"][()], np.float64) * float(cr["alat"][()]),
            bvec=np.asarray(cr["bvec"][()], np.float64) * float(cr["blat"][()]),
            celvol=float(cr["celvol"][()]),
            fft=tuple(int(x) for x in h["gspace/FFTgrid"][()]),
        )
        gv = np.asarray(f["wfns/gvecs"][()], np.int64)
        raw = f["wfns/coeffs"][:nb]                       # (nb, 2, ngtot, re/im)
    c = raw[..., 0] + 1j * raw[..., 1]
    off = np.concatenate([[0], np.cumsum(out["ngk"])])
    out["gk"] = [gv[off[i]:off[i + 1]] for i in range(len(out["ngk"]))]
    out["psi"] = [c[:, :, off[i]:off[i + 1]] for i in range(len(out["ngk"]))]
    return out


def rkb_lift(psi_L, K, h):
    """Ψ = r(K)[ψ_L; h σ·K ψ_L], r = (1 + h²K²)^(−1/2); psi_L (nb, 2, G), K (G, 3)."""
    sK = np.einsum("cab,gc->gab", PAULI, K)                      # σ·K per G
    psi_S = h * np.einsum("gab,nbg->nag", sK, psi_L)
    r = 1.0 / np.sqrt(1.0 + h * h * np.sum(K * K, axis=1))
    return np.concatenate([psi_L, psi_S], axis=1) * r[None, None, :]


def coulomb(K, *, celvol, sys_dim, zc):
    """v(K) in Ry including 1/Ω (the production tiles' convention); 0 at K = 0."""
    K2 = np.sum(K * K, axis=1)
    v = np.zeros_like(K2)
    nz = K2 > 1e-12
    v[nz] = 8.0 * np.pi / K2[nz]
    if sys_dim == 2:
        kxy = np.sqrt(K[:, 0] ** 2 + K[:, 1] ** 2)
        v *= 1.0 - np.exp(-zc * kxy) * np.cos(K[:, 2] * zc)
    elif sys_dim != 3:
        raise ValueError("sys_dim 2 or 3")
    return v / celvol


def transverse_projector(K):
    """P^T_ij(K̂) = δ_ij − K̂_i K̂_j, (G, 3, 3); K = 0 never reaches it (v = 0 there)."""
    K2 = np.maximum(np.sum(K * K, axis=1), 1e-300)
    return np.eye(3)[None] - K[:, :, None] * K[:, None, :] / K2[:, None, None]


def vertex_apply(u):
    """γ̃^I u for I = 0..3 on 4-spinor fields u (..., 4, *grid) → (4, ..., 4, *grid)."""
    up, lo = u[..., 0:2, :, :, :], u[..., 2:4, :, :, :]
    out = [u]
    for s in PAULI:
        a_up = np.einsum("ab,...bxyz->...axyz", s, lo)
        a_lo = np.einsum("ab,...bxyz->...axyz", s, up)
        out.append(np.concatenate([a_up, a_lo], axis=-4))
    return np.stack(out)


def miller_box(fft):
    ax = [np.arange(-(n // 2), n - n // 2) for n in fft]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--nsigma", type=int, required=True,
                    help="Σ bands [0, nsigma) (the run's nelec + ncond)")
    ap.add_argument("--ecut", type=float, default=None,
                    help="bare-Coulomb sphere |K|² ≤ E_c in Ry (default: WFN ecutwfc, "
                         "the production bare_coulomb_cutoff default)")
    ap.add_argument("--sys-dim", type=int, default=2)
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="multiply α_FS (Σ^B must scale as its square)")
    ap.add_argument("--kleft", default=None,
                    help="comma list of k rows to evaluate (default all)")
    ap.add_argument("--kright", default=None,
                    help="debug only: restrict the k' sum to these rows (the result is then partial)")
    ap.add_argument("--nchunk", type=int, default=4, help="left bands per FFT batch")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--xcheck", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    fftn, ifftn = _fft_backend(a.threads)
    w = read_wfn(a.wfn, a.nsigma)
    rk, kgrid, fft = w["rk"], w["kgrid"], w["fft"]
    nk, nr = len(rk), int(np.prod(fft))
    if nk != int(np.prod(kgrid)):
        raise ValueError(f"WFN holds {nk} k of the {kgrid} grid; this reference needs the full grid")
    occ = w["occ"]
    nocc_k = np.sum(occ > 0.5, axis=1)
    if np.any(np.abs(occ - np.round(occ)) > 1e-8) or len(set(nocc_k.tolist())) != 1:
        raise ValueError("insulating 0/1 occupations with one occupied count are required")
    nocc = int(nocc_k[0])
    if a.nsigma < nocc:
        raise ValueError("nsigma must include every occupied band")
    ecut = w["ecutwfc"] if a.ecut is None else float(a.ecut)
    bvec, avec, om = w["bvec"], w["avec"], w["celvol"]
    if abs(abs(np.linalg.det(avec)) - om) > 1e-6 * om:
        raise ValueError("celvol differs from |det avec|")
    if np.max(np.abs(avec @ bvec.T - 2 * np.pi * np.eye(3))) > 1e-8:
        raise ValueError("avec·bvec ≠ 2π")
    zc = None
    if a.sys_dim == 2:
        if np.max(np.abs(avec[2, :2])) > 1e-10 or np.max(np.abs(avec[:2, 2])) > 1e-10:
            raise ValueError("sys_dim 2 needs a3 ∥ z ⟂ a1, a2")
        zc = 0.5 * abs(avec[2, 2])                                    # L_z / 2
    h = HALFALPHA * a.alpha_scale
    rows = (list(range(nk)) if a.kleft is None else [int(x) for x in a.kleft.split(",")])
    kright = (list(range(nk)) if a.kright is None else [int(x) for x in a.kright.split(",")])
    receipt = dict(wfn=os.path.realpath(a.wfn), nk=nk, kgrid=kgrid, fft=fft, nocc=nocc,
                   nsigma=a.nsigma, ecut_ry=ecut, sys_dim=a.sys_dim, zc_bohr=zc,
                   alpha_fs=2 * h, celvol=om, k_rows=rows,
                   partial_kright=(None if a.kright is None else kright))

    # ---- normalization, then every band's lifted 4-spinor on the FFT box (Σ_G c e^{iG·r})
    norm_dev = max(float(np.max(np.abs(np.sum(np.abs(p) ** 2, axis=(1, 2)) - 1.0))) for p in w["psi"])
    receipt["max_norm_dev_psiL"] = norm_dev
    u = np.zeros((nk, a.nsigma, 4) + fft, np.complex128)
    lift_dev = 0.0
    for ik in range(nk):
        K = (rk[ik][None] + w["gk"][ik]) @ bvec
        Psi = rkb_lift(w["psi"][ik], K, h)
        lift_dev = max(lift_dev, float(np.max(np.abs(
            np.sum(np.abs(Psi) ** 2, axis=(1, 2)) - np.sum(np.abs(w["psi"][ik]) ** 2, axis=(1, 2))))))
        g = w["gk"][ik]
        box = np.zeros((a.nsigma, 4) + fft, np.complex128)
        box[:, :, g[:, 0] % fft[0], g[:, 1] % fft[1], g[:, 2] % fft[2]] = Psi
        u[ik] = ifftn(box) * nr
    receipt["lift_isometry_dev"] = lift_dev        # ⟨Ψ|Ψ⟩ = ⟨ψ_L|ψ_L⟩ band by band
    if a.xcheck:
        receipt.update(_xcheck(w, bvec, om, a.sys_dim, ecut, kgrid, h, zc))

    box_g = miller_box(fft)
    half = np.array(fft) // 2
    sig_b = np.zeros((len(rows), a.nsigma, a.nsigma), np.complex128)
    sig_x = np.zeros_like(sig_b)
    rm_dev, rm_scale, gmax_frac = 0.0, 0.0, 0.0
    nK_tot = 0
    for ir, ik in enumerate(rows):
        uk = u[ik]
        for jk in kright:
            Q = rk[ik] - rk[jk]                                 # K = Q + G
            K_all = (Q[None] + box_g) @ bvec
            K2 = np.sum(K_all * K_all, axis=1)
            sel = (K2 <= ecut) & (K2 > 1e-12)
            G = box_g[sel]
            K = K_all[sel]
            gmax_frac = max(gmax_frac, float(np.max(np.abs(G) / half[None])))
            nK_tot += len(G)
            idx = (G[:, 0] % fft[0], G[:, 1] % fft[1], G[:, 2] % fft[2])
            wK = coulomb(K, celvol=om, sys_dim=a.sys_dim, zc=zc)
            PT = transverse_projector(K)
            uv = u[jk, :nocc]                                   # (nocc, 4, grid) at k'
            # right vertex: R^J_vm(K) = (1/N_r) Σ_r [Ψ_v† γ̃^J Ψ_m](r) e^{−iG·r}
            gam_um = vertex_apply(uk)                           # (4, nsigma, 4, grid)
            gR = np.einsum("vaxyz,Jmaxyz->Jvmxyz", np.conj(uv), gam_um, optimize=True)
            R = fftn(gR)[..., idx[0], idx[1], idx[2]] / nr      # (4, nocc, nsigma, nK)
            del gR, gam_um
            gam_uv = vertex_apply(uv)                           # (4, nocc, 4, grid)
            for n0 in range(0, a.nsigma, a.nchunk):
                n1 = min(a.nsigma, n0 + a.nchunk)
                # left vertex: M^I_nv(K) = (1/N_r) Σ_r [Ψ_n† γ̃^I Ψ_v](r) e^{+iG·r}
                fL = np.einsum("naxyz,Ivaxyz->Invxyz", np.conj(uk[n0:n1]), gam_uv, optimize=True)
                M = ifftn(fL)[..., idx[0], idx[1], idx[2]]      # (4, nb, nocc, nK)
                del fL
                # R^J_vm = conj(M^J_mv) must hold to round-off: Bloch phases and FFT signs
                Rb = R[:, :, n0:n1, :]
                rm_dev = max(rm_dev, float(np.max(np.abs(Rb - np.conj(np.swapaxes(M, 1, 2))))))
                rm_scale = max(rm_scale, float(np.max(np.abs(M))))
                sig_x[ir, n0:n1] -= np.einsum("nvK,K,vmK->nm", M[0], wK, R[0], optimize=True) / nk
                T = np.einsum("invK,K,Kij->jnvK", M[1:], wK, PT, optimize=True)
                sig_b[ir, n0:n1] += np.einsum("jnvK,jvmK->nm", T, R[1:], optimize=True) / nk
            del R, gam_uv
        print(f"[sigma_b_pw] k row {ik} done {time.time() - t0:.1f}s  "
              f"diag Σ^B (meV) {np.round(np.real(np.diag(sig_b[ir]))[:8] * RY_EV * 1e3, 4).tolist()}",
              flush=True)
    receipt["R_vs_conjM_maxdev_rel"] = rm_dev / max(rm_scale, 1e-300)
    receipt["sphere_max_miller_over_half_box"] = gmax_frac
    receipt["n_K_per_kpair_mean"] = nK_tot / (len(rows) * len(kright))
    for name, S in (("sigma_b", sig_b), ("sigma_x", sig_x)):
        herm = float(np.max(np.abs(S - np.conj(np.swapaxes(S, 1, 2)))))
        receipt[f"{name}_hermiticity_maxdev_ry"] = herm
        receipt[f"{name}_max_abs_ry"] = float(np.max(np.abs(S)))
    receipt["sigma_b_min_eig_ry"] = float(min(np.linalg.eigvalsh(
        0.5 * (S + S.conj().T)).min() for S in sig_b))
    receipt["wall_s"] = time.time() - t0
    np.savez(a.out, sigma_b_ry=sig_b, sigma_x_ry=sig_x, k_rows=np.array(rows),
             kpts=rk[rows], enk_ev=w["el"][rows, :a.nsigma] * RY_EV,
             receipt=json.dumps(receipt))
    print("[sigma_b_pw] receipt " + json.dumps(receipt), flush=True)
    return 0


def _xcheck(w, bvec, om, sys_dim, ecut, kgrid, h, zc):
    """This file's lift and kernel against the production owners (CPU JAX)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    from common.bispinor_init import HALFALPHA as H_PROD, lift_to_4spinor
    from gw.v_q_bispinor import _make_per_q_v_builder_for_tile
    out = {"halfalpha_prod": float(H_PROD), "halfalpha_here": HALFALPHA}
    dev_lift = 0.0
    for ik in range(len(w["rk"])):
        g = w["gk"][ik]
        mine = rkb_lift(w["psi"][ik], (w["rk"][ik][None] + g) @ bvec, HALFALPHA)
        prod = np.asarray(lift_to_4spinor(
            w["psi"][ik][None], g[None].astype(np.float64), w["rk"][ik][None], bvec,
            representation="normalized_rkb"))[0]
        dev_lift = max(dev_lift, float(np.max(np.abs(mine - prod))))
    out["lift_vs_prod_maxdev"] = dev_lift
    # kernel on three q (Γ, the first two other rows), every Miller of the box, ij all 9
    box_g = miller_box(w["fft"])
    dev_t, dev_cut, scale = 0.0, 0.0, 0.0
    for jk in range(min(3, len(w["rk"]))):
        Q = w["rk"][0] - w["rk"][jk]
        K = (Q[None] + box_g) @ bvec
        K2 = np.sum(K * K, axis=1)
        keep = K2 <= 2.0 * ecut                     # include a shell past the cutoff
        Gk, Kk, K2k = box_g[keep], K[keep], K2[keep]
        mine = (coulomb(Kk, celvol=om, sys_dim=sys_dim, zc=zc)[:, None, None]
                * transverse_projector(Kk))
        mine[K2k > ecut] = 0.0
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                b = _make_per_q_v_builder_for_tile(
                    mu_L=i, nu_L=j, bvec=bvec, cell_volume=om, sys_dim=sys_dim,
                    vcoul_cutoff_ry=ecut, kgrid=tuple(kgrid), tt_head_correction=False)
                prod = np.asarray(b(Q[None], Gk.T[None].astype(np.int32)))[0]   # = t_ij = −v P^T
                inside = K2k <= ecut
                dev_t = max(dev_t, float(np.max(np.abs(-prod[inside] - mine[inside, i - 1, j - 1]))))
                dev_cut = max(dev_cut, float(np.max(np.abs(prod[~inside])))) if np.any(~inside) else dev_cut
                scale = max(scale, float(np.max(np.abs(prod))))
    out["tt_kernel_vs_prod_maxdev_rel"] = dev_t / max(scale, 1e-300)
    out["tt_kernel_prod_outside_cutoff_max"] = dev_cut
    return out


if __name__ == "__main__":
    sys.exit(main())
