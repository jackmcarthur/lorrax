#!/usr/bin/env python3
"""Orbital magnetization (modern theory) from a LORRAX spinor WFN.

Computes the per-cell orbital magnetic moment of a spin-orbit-coupled
(2-component spinor) crystal directly from a BerkeleyGW-format ``WFN.h5``,
using the gauge-invariant modern-theory formula evaluated in the
*sum-over-states* (k.p) representation.  The k-space derivative of the
Hamiltonian is taken **analytically** via ``dH/dk = 2(k+G) + dV_NL/dk`` —
no finite differences anywhere in the velocity operator (the only finite
differences in this script are an *optional* Hellmann-Feynman validation
of the group velocity).

Physics (Rydberg atomic units: hbar=1, 2 m_e = 1, energies in Ry, lengths
in Bohr).  Per-cell orbital moment, component gamma, in Bohr magnetons:

    m_gamma / mu_B = (-1/2) * sum_k w_k * Im sum_{n occ} sum_{m != n}
                       eps_{gamma a b} v^a_nm v^b_mn (eps_m + eps_n - 2 mu)
                                                     / (eps_n - eps_m)^2

with v^a_nm = <u_nk| dH_k/dk_a |u_mk> the velocity matrix element (Ry*Bohr,
exactly what ``dft_operators.velocity_matrix_k`` returns), w_k the k-point
weights (sum to 1), and the leading -1/2 the electron-charge gyromagnetic
prefactor m_e/hbar^2 = 1/(2 Ry a0^2) carrying the orbital moment = -mu_B L/hbar
sign.  See ``orbital_magnetization_THEORY.md`` for the full derivation,
sources, and the absolute-sign discussion.

The script also computes the spin moment <sigma_z> from the same WFN as an
internal calibration: it must be ~ +/-6 mu_B for CrI3, which both validates
the wavefunction/occupations and fixes the physical axis so the orbital
moment can be reported *relative to the spin moment* (parallel / antiparallel)
in a convention-robust way.

Orbital magnetization is identically zero without spin-orbit coupling for a
collinear ferromagnet, so the script requires ``nspinor == 2``.
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")  # MUST precede jax import (f64)
if "--cpu" in sys.argv:  # force CPU backend (avoid GPU contention); must precede jax import
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "32")  # courteous cap on shared nodes

import numpy as np
import jax  # noqa: F401  (sets up x64; devices queried lazily)

# Allow `python orbital_magnetization.py` as well as `-m psp.orbital_magnetization`
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import generate_gvectors_k, gather_psi_G_from_crys
from psp.get_dipole_mtxels import compute_p_operator_k, compute_vnl_velocity_cart
from psp.pseudos import load_pseudopotentials, print_atomic_structure
import psp.vnl_ops as vnl_ops

RY2EV = 13.605693122994
MU_B_PREFACTOR = 0.5  # |m_e/hbar^2| in Ry-a0^2 units (magnitude; sign handled below)


# ----------------------------------------------------------------------
#  Per-k velocity assembly  (dH/dk = kinetic 2(k+G) + nonlocal dV_NL/dk)
# ----------------------------------------------------------------------
def velocity_at_k(wfn, sym, meta, vnl_setup, ik, nb):
    """Return (v_kin, v_nl, eps, sz) for full-BZ k-index ``ik``.

    v_kin, v_nl : (3, nb, nb) complex128, the kinetic 2(k+G) and nonlocal
                  dV_NL/dk velocity matrices, v[a, m, n] = <u_m|v^a|u_n>
                  (Ry*Bohr).  Kept separate so the kinetic/nonlocal relative
                  sign can be validated by Hellmann-Feynman before summing.
    eps         : (nb,) DFT eigenvalues at this k (Ry).
    sz          : (nb,) <sigma_z> = sum_G(|c_up|^2 - |c_dn|^2) per band.
    """
    wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)            # (nb, ns, nx,ny,nz)
    Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
    kpoint = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)  # crystal coords

    v_kin = np.asarray(compute_p_operator_k(
        wfn_k, Gk_crys, kpoint,
        getattr(wfn, "bdot", None), wfn.bvec, wfn.blat))         # (3, nb, nb)
    if vnl_setup is None:                                         # --skip-vnl diagnostic
        v_nl = np.zeros_like(v_kin)
    else:
        v_nl = np.asarray(compute_vnl_velocity_cart(
            wfn_k, Gk_crys, kpoint, vnl_setup))                   # (3, nb, nb)

    k_red = int(sym.irr_idx_k[ik])
    eps = np.asarray(wfn.energies[0, k_red, :nb], dtype=np.float64)

    psi_G = np.asarray(gather_psi_G_from_crys(wfn_k, Gk_crys))    # (nb, ns, nG)
    sz = (np.abs(psi_G[:, 0]) ** 2 - np.abs(psi_G[:, 1]) ** 2).sum(axis=1).real
    del wfn_k, psi_G
    return v_kin, v_nl, eps, sz


# ----------------------------------------------------------------------
#  Modern-theory sum-over-states summand at one k
# ----------------------------------------------------------------------
def orbital_sum_at_k(v, eps, nocc, mu, deps_tol, mceil=None):
    """Complex (Cx, Cy, Cz) = sum_{n occ} sum_{m!=n, m<mceil} cross * weight.

    cross_gamma[n,m] = eps_{gamma a b} v^a_nm v^b_mn, with the index map
    v^a_nm = v[a, n, m] (bra n, ket m) and v^b_mn = v[b, m, n].  Therefore
    cross_z = v[0]*v[1].T - v[1]*v[0].T  (element-wise (nb,nb) products), etc.
    weight[n,m] = (eps_m + eps_n - 2 mu) / (eps_n - eps_m)^2, masked to 0 for
    |eps_n-eps_m| <= deps_tol (removes m=n and degenerate denominators).
    The physical prefactor (-1/2) and Im[.] are applied by the caller.
    """
    nb = v.shape[1]
    vt = np.swapaxes(v, 1, 2)                       # vt[a, n, m] = v[a, m, n]
    cz = v[0] * vt[1] - v[1] * vt[0]                # (nb, nb)
    cx = v[1] * vt[2] - v[2] * vt[1]
    cy = v[2] * vt[0] - v[0] * vt[2]

    deps = eps[:, None] - eps[None, :]              # eps_n - eps_m
    esum = eps[:, None] + eps[None, :] - 2.0 * mu
    mask = np.abs(deps) > deps_tol
    W = np.where(mask, esum / np.where(mask, deps, 1.0) ** 2, 0.0)

    occ = np.zeros((nb, 1)); occ[:nocc, 0] = 1.0    # outer sum over occupied n
    if mceil is not None and mceil < nb:
        col = np.ones((1, nb)); col[0, mceil:] = 0.0  # inner m-sum ceiling
        W = W * col
    Wn = occ * W
    return np.array([np.sum(Wn * cx), np.sum(Wn * cy), np.sum(Wn * cz)],
                    dtype=np.complex128)


# ----------------------------------------------------------------------
#  Hellmann-Feynman group-velocity check (fixes kinetic/nonlocal sign)
# ----------------------------------------------------------------------
def hf_group_velocity_check(Vp, Vnl, eps_grid, kcrys_grid, B, kgrid, nbands_show=8):
    """Compare diagonal Re<n|dH/dk|n> to FD band slopes d eps_n / dk.

    Returns a dict with, for each candidate nonlocal sign s in {+1,-1}, the
    RMS mismatch (Ry*Bohr) between Re diag(Vp + s*Vnl) and the central-FD
    Cartesian band gradient, over a set of dispersive, non-degenerate bands
    at interior k-points.  The sign with the smaller mismatch is the physical
    velocity convention.
    """
    nkx, nky, nkz = (int(x) for x in kgrid)
    Binv = np.linalg.inv(np.asarray(B, dtype=np.float64))
    # map rounded crystal coord -> full-BZ index
    key = lambda kc: (int(round(kc[0] * nkx)) % nkx,
                      int(round(kc[1] * nky)) % nky,
                      int(round(kc[2] * nkz)) % nkz)
    idx_of = {key(kc): i for i, kc in enumerate(kcrys_grid)}

    def grad_cart(ik_full, n):
        kc = kcrys_grid[ik_full]
        g_crys = np.zeros(3)
        steps = [(0, nkx), (1, nky)]                # in-plane only (kz single layer)
        for axis, N in steps:
            if N < 3:
                continue
            kp = kc.copy(); kp[axis] += 1.0 / N
            km = kc.copy(); km[axis] -= 1.0 / N
            ip, im = idx_of.get(key(kp)), idx_of.get(key(km))
            if ip is None or im is None:
                return None
            g_crys[axis] = (eps_grid[ip, n] - eps_grid[im, n]) / (2.0 / N)
        return Binv @ g_crys                         # Cartesian gradient

    results = {1: [], -1: []}
    detail = []
    for ik in range(len(kcrys_grid)):
        eps = eps_grid[ik]
        # pick non-degenerate, dispersive bands (diag velocity = dε/dk needs a
        # non-degenerate band; 2e-3 Ry ≈ 27 meV separation from neighbors)
        for n in range(min(eps.shape[0], 200)):
            if n + 1 < eps.shape[0] and abs(eps[n + 1] - eps[n]) < 2e-3:
                continue
            if n - 1 >= 0 and abs(eps[n] - eps[n - 1]) < 2e-3:
                continue
            gc = grad_cart(ik, n)
            if gc is None or np.linalg.norm(gc[:2]) < 0.02:
                continue
            for s in (1, -1):
                vdiag = (Vp[ik] + s * Vnl[ik])[:, n, n].real
                results[s].append(np.abs(vdiag[:2] - gc[:2]))
            if len(detail) < nbands_show:
                vp = Vp[ik][:, n, n].real
                detail.append((ik, n,
                               (Vp[ik] + Vnl[ik])[:, n, n].real[:2].copy(),
                               (Vp[ik] - Vnl[ik])[:, n, n].real[:2].copy(),
                               gc[:2].copy(), vp[:2].copy()))
    out = {}
    for s in (1, -1):
        arr = np.array(results[s]) if results[s] else np.zeros((1, 2))
        out[s] = float(np.sqrt(np.mean(arr ** 2)))
    out["detail"] = detail
    out["nsamples"] = len(results[1])
    return out


# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Per-cell orbital magnetic moment (modern theory, dH/dk).")
    p.add_argument("--wfn", required=True, help="WFN.h5 (BGW format, nspinor=2)")
    p.add_argument("--nbnd", type=int, default=None,
                   help="Inner-sum band ceiling (default: all bands in file)")
    p.add_argument("--nocc", type=int, default=None,
                   help="Occupied-band count (default: wfn.nelec)")
    p.add_argument("--mu", type=float, default=None,
                   help="Chemical potential in eV (default: midgap)")
    p.add_argument("--mu-scan", action="store_true",
                   help="Also report m_z at mu = VBM, midgap, CBM (Chern/dM/dmu check)")
    p.add_argument("--deps-tol", type=float, default=1.4e-3,
                   help="Degenerate-denominator skip tolerance in eV (default 1.4e-3)")
    p.add_argument("--pseudo-dir", default=None,
                   help="Directory of *.upf (default: auto-discover near WFN)")
    p.add_argument("--vnl-sign", choices=["auto", "plus", "minus"], default="auto",
                   help="Kinetic/nonlocal relative sign: 'auto' uses the "
                        "Hellmann-Feynman group-velocity check (recommended)")
    p.add_argument("--skip-vnl", action="store_true",
                   help="DIAGNOSTIC: kinetic-only velocity (physically incomplete)")
    p.add_argument("--cpu", action="store_true",
                   help="Force JAX CPU backend (handled before jax import; "
                        "use when GPUs are occupied)")
    p.add_argument("--convergence", action="store_true",
                   help="Report m_z vs inner-m band ceiling")
    p.add_argument("--per-band", action="store_true",
                   help="Report m_z contribution per occupied band")
    p.add_argument("--out", default=None, help="Optional .npz dump of v, eps, sz")
    args = p.parse_args(argv)

    wfn_path = Path(args.wfn).resolve()
    print(f"\n[orbmag] WFN: {wfn_path}")
    wfn = WFNReader(str(wfn_path))
    sym = symmetry_maps.SymMaps(wfn)

    nspinor = int(wfn.nspinor)
    if nspinor != 2:
        sys.exit(f"[orbmag] ERROR: nspinor={nspinor}. Orbital magnetization is "
                 "identically zero without spin-orbit coupling for a collinear "
                 "ferromagnet; this script requires a 2-component spinor WFN.")

    nbnd = int(args.nbnd) if args.nbnd else int(wfn.nbands)
    nbnd = min(nbnd, int(wfn.nbands))
    nocc = int(args.nocc) if args.nocc else int(wfn.nelec)
    deps_tol = args.deps_tol / RY2EV                 # eV -> Ry

    # Meta: load all `nbnd` bands; nval/ncond just set band-window markers.
    nval = int(wfn.nelec)
    ncond = max(0, nbnd - int(wfn.nelec))
    meta = Meta.from_system(wfn, sym, nval, ncond, nbnd, 0, False)  # bispinor=False

    print(f"[orbmag] nspinor={nspinor}  nbnd={nbnd}  nocc={nocc}  "
          f"nk_ibz={int(wfn.nrk) if hasattr(wfn,'nrk') else len(wfn.kweights)}  "
          f"nk_full={int(sym.nk_tot)}")

    # Pseudopotentials for the nonlocal velocity (dV_NL/dk).
    pdirs = [args.pseudo_dir] if args.pseudo_dir else []
    pdirs += [str(wfn_path.parent),
              str(wfn_path.parent / ".." / "qe" / "scf"),
              str(wfn_path.parent / ".." / "qe" / "nscf")]
    pseudos = {}
    for d in pdirs:
        if d and Path(d).exists():
            pseudos = load_pseudopotentials(d)
            if pseudos:
                print(f"[orbmag] pseudopotentials from: {d}  -> {list(pseudos)}")
                break
    if not args.skip_vnl and not pseudos:
        sys.exit("[orbmag] ERROR: no *.upf found (need them for dV_NL/dk). "
                 "Pass --pseudo-dir, or --skip-vnl for a kinetic-only diagnostic.")
    if pseudos:
        try:
            print_atomic_structure(wfn, pseudos)
        except Exception:
            pass

    vnl_setup = None
    if not args.skip_vnl:
        vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos, nspinor=nspinor)

    # ---- per-k velocity matrices + spin density --------------------------
    nk = int(sym.nk_tot)
    w_k = 1.0 / nk                                   # uniform full-BZ weight
    Vp = np.zeros((nk, 3, nbnd, nbnd), dtype=np.complex128)
    Vnl = np.zeros((nk, 3, nbnd, nbnd), dtype=np.complex128)
    E = np.zeros((nk, nbnd), dtype=np.float64)
    SZ = np.zeros((nk, nbnd), dtype=np.float64)
    Kc = np.zeros((nk, 3), dtype=np.float64)
    print(f"[orbmag] assembling velocity matrices over {nk} full-BZ k-points...")
    for ik in range(nk):
        vk, vnlk, eps, sz = velocity_at_k(wfn, sym, meta, vnl_setup, ik, nbnd)
        Vp[ik], Vnl[ik], E[ik], SZ[ik] = vk, vnlk, eps, sz
        Kc[ik] = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
        if (ik + 1) % 6 == 0 or ik == nk - 1:
            print(f"         k {ik+1}/{nk}")

    B = np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)

    # ---- decide kinetic/nonlocal relative sign --------------------------
    sign = +1
    if args.skip_vnl:
        sign = 0
    else:
        hf = hf_group_velocity_check(Vp, Vnl, E, Kc, B, wfn.kgrid)
        print("\n[orbmag] Hellmann-Feynman group-velocity check "
              f"({hf['nsamples']} band/k samples):")
        print(f"         RMS |Re diag(v) - d eps/dk|:  "
              f"p+vNL = {hf[1]:.4f}   p-vNL = {hf[-1]:.4f}  (Ry*Bohr)")
        # PHYSICAL SIGN = p + vNL.  velocity_matrix_k = p + vnl_velocity_from_dZ,
        # and vnl_velocity_from_dZ is bit-identical to vnl_velocity_matrix
        # (= compute_vnl_velocity_cart).  Verified by an off-diagonal finite
        # difference of <m|V_NL(k)|n>: compute_vnl_velocity_cart == +dV_NL/dk
        # to ratio +1.000.  The nonlocal velocity is ~900x larger off-diagonal
        # than on-diagonal, so the diagonal HF slope test is INSENSITIVE to the
        # sign (it ties) — HF validates the kinetic part/units only.  Default
        # to the proven canonical p+vNL; only flip on a large HF margin.
        if args.vnl_sign != "auto":
            chosen = args.vnl_sign
        elif hf["nsamples"] > 0 and hf[-1] < 0.8 * hf[1]:
            chosen = "minus"
            print("         (HF strongly prefers p-vNL — unexpected; check sign)")
        else:
            chosen = "plus"
            print("         (HF diagonal test insensitive to nonlocal sign; "
                  "using canonical p+vNL — verified +dV_NL/dk off-diagonally)")
        sign = +1 if chosen == "plus" else -1
        print(f"         -> using v = p {'+' if sign>0 else '-'} vNL "
              f"({'auto' if args.vnl_sign=='auto' else 'forced'})")
        for (ik, n, vpls, vmin, gc, vp) in hf["detail"][:6]:
            print(f"           k{ik:2d} n{n:3d}: FD={np.array2string(gc,precision=3)}  "
                  f"p+vNL={np.array2string(vpls,precision=3)}  "
                  f"p-vNL={np.array2string(vmin,precision=3)}")

    V = Vp + sign * Vnl                              # (nk,3,nb,nb) physical velocity

    # ---- chemical potential ---------------------------------------------
    VBM = float(E[:, nocc - 1].max())
    CBM = float(E[:, nocc].min()) if nocc < nbnd else VBM
    if args.mu is not None:
        mu = args.mu / RY2EV
    else:
        mu = 0.5 * (VBM + CBM)
    gap_eV = (CBM - VBM) * RY2EV
    print(f"\n[orbmag] VBM={VBM*RY2EV:.4f} eV  CBM={CBM*RY2EV:.4f} eV  "
          f"indirect gap={gap_eV:.4f} eV   mu={mu*RY2EV:.4f} eV ({mu:.5f} Ry)")
    if gap_eV < 0:
        print("         NOTE: negative indirect gap at this k-sampling -> the "
              "moment is mu-dependent (run --mu-scan).")

    # ---- spin moment (calibration / axis) -------------------------------
    S_sum = float((w_k * SZ[:, :nocc].sum(axis=1)).sum())  # sum_k w_k sum_occ <sz>
    m_spin_z = -1.0 * S_sum                                 # mu_B, file frame
    print(f"\n[orbmag] spin moment  sum_occ <sigma_z> = {S_sum:+.4f}  -> "
          f"|m_spin| = {abs(m_spin_z):.3f} mu_B  (expect ~6 for CrI3)")

    # ---- orbital moment --------------------------------------------------
    C = np.zeros(3, dtype=np.complex128)
    perband = np.zeros(nocc) if args.per_band else None
    for ik in range(nk):
        Ck = orbital_sum_at_k(V[ik], E[ik], nocc, mu, deps_tol)
        C += w_k * Ck
        if args.per_band:
            for n in range(nocc):
                cn = orbital_sum_at_k(V[ik], E[ik], n + 1, mu, deps_tol)[2] \
                     - orbital_sum_at_k(V[ik], E[ik], n, mu, deps_tol)[2]
                perband[n] += w_k * cn.imag
    m_orb = -MU_B_PREFACTOR * C.imag                        # (3,) mu_B, file frame

    frame = 1.0 if m_spin_z >= 0 else -1.0
    m_orb_par = float(frame * m_orb[2])                     # along spin-moment axis

    print("\n" + "=" * 64)
    print("ORBITAL MAGNETIC MOMENT  (per unit cell, mu_B)")
    print("=" * 64)
    print(f"  m_x = {m_orb[0]:+.5f}   m_y = {m_orb[1]:+.5f}   "
          f"(should be ~0 by symmetry)")
    print(f"  m_z = {m_orb[2]:+.5f}   (file z-axis = crystal c, out of plane)")
    print(f"  orbital moment along spin axis: {m_orb_par:+.5f} mu_B  "
          f"({'PARALLEL' if m_orb_par>0 else 'ANTIPARALLEL'} to spin)")
    print(f"  spin moment |m_spin| = {abs(m_spin_z):.3f} mu_B")
    print("=" * 64)

    if args.mu_scan and nocc < nbnd:
        print("\n[orbmag] mu-scan (m_z, mu_B):")
        for label, m in [("VBM", VBM), ("midgap", 0.5 * (VBM + CBM)), ("CBM", CBM)]:
            Cs = sum(w_k * orbital_sum_at_k(V[ik], E[ik], nocc, m, deps_tol)
                     for ik in range(nk))
            print(f"   mu={m*RY2EV:8.4f} eV ({label:6s}):  "
                  f"m_z = {-MU_B_PREFACTOR*float(Cs[2].imag):+.5f}")

    if args.convergence and not args.skip_vnl:
        print("\n[orbmag] convergence vs inner-m band ceiling (m_z, mu_B):")
        ceilings = sorted(set([int(0.5*nbnd), int(0.7*nbnd), int(0.85*nbnd), nbnd]))
        for mc in ceilings:
            Cc = sum(w_k * orbital_sum_at_k(V[ik], E[ik], nocc, mu, deps_tol, mceil=mc)
                     for ik in range(nk))
            print(f"   mceil={mc:4d}:  m_z = {-MU_B_PREFACTOR*float(Cc[2].imag):+.5f}")

    if args.per_band:
        m_par_band = frame * (-MU_B_PREFACTOR) * perband
        print("\n[orbmag] per-occupied-band m_z (along spin axis, mu_B):")
        order = np.argsort(np.abs(m_par_band))[::-1]
        for n in order[:12]:
            print(f"   band {n:3d}: {m_par_band[n]:+.5f}")

    if args.out:
        np.savez_compressed(args.out, Vp=Vp, Vnl=Vnl, E=E, SZ=SZ, Kc=Kc,
                            sign=sign, mu=mu, nocc=nocc, w_k=w_k,
                            m_orb=m_orb, m_spin_z=m_spin_z)
        print(f"\n[orbmag] wrote {args.out}")


if __name__ == "__main__":
    main()
