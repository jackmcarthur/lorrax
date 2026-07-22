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
from psp.dft_operators import (generate_gvectors_k, gather_psi_G_from_crys,
                               momentum_matrix_k)
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
def orbital_pieces_at_k(v, eps, nocc, deps_tol):
    """mu-independent building blocks of the orbital-moment summand at one k.

    Returns (PA, PB), each (3, nb, nb) complex, with the per-(gamma, n, m) terms

        PA[g,n,m] = occ[n] * cross_g[n,m] * (eps_n + eps_m) / (eps_n-eps_m)^2
        PB[g,n,m] = occ[n] * cross_g[n,m] /               (eps_n-eps_m)^2

    where cross_g[n,m] = eps_{g a b} v^a_nm v^b_mn, index map v^a_nm = v[a,n,m]
    (bra n, ket m), so cross_z = v[0]*v[1].T - v[1]*v[0].T (element-wise).
    The full summand at chemical potential mu is then linear in mu:

        summand_g(mu)[n,m] = PA[g,n,m] - 2*mu*PB[g,n,m]

    so ANY mu, the per-band breakdown (sum over m), and the band-ceiling
    convergence (cumsum over m) all follow from one pass — no recomputation.
    The (-1/2) prefactor and Im[.] are applied by the caller.  Degenerate /
    diagonal denominators (|eps_n-eps_m| <= deps_tol) are masked to 0.
    """
    nb = v.shape[1]
    vt = np.swapaxes(v, 1, 2)                        # vt[a, n, m] = v[a, m, n]
    cross = np.stack([v[1] * vt[2] - v[2] * vt[1],   # x  (eps_xab, ab=yz)
                      v[2] * vt[0] - v[0] * vt[2],   # y  (ab=zx)
                      v[0] * vt[1] - v[1] * vt[0]])   # z  (ab=xy)   (3,nb,nb)
    deps = eps[:, None] - eps[None, :]               # eps_n - eps_m
    mask = np.abs(deps) > deps_tol
    inv2 = np.where(mask, 1.0 / np.where(mask, deps, 1.0) ** 2, 0.0)  # 1/Delta^2
    occ = np.zeros((nb, 1)); occ[:nocc, 0] = 1.0     # outer sum over occupied n
    Wa = occ * ((eps[:, None] + eps[None, :]) * inv2)
    Wb = occ * inv2
    return cross * Wa[None], cross * Wb[None]         # PA, PB : (3, nb, nb)


# ----------------------------------------------------------------------
#  Symmetry-reduced (IBZ) mode: magnetic point group + axial-vector unfold
# ----------------------------------------------------------------------
def magnetic_point_group(sym, m_axis_cart, tol=1e-6):
    """Cartesian rotations R_g and det(R_g) of the MAGNETIC point group.

    Returns (R_mpg (|G|,3,3), det_mpg (|G|,), idx (|G|,)).  We take the
    spatial-only half of sym.R_cart (rows [0,ntran); the [ntran:] half is the
    time-reversal-augmented -R, symmetry_maps.py:~1217) and keep operation s
    iff it preserves the magnetization AXIAL vector:  det(R_s) R_s @ m == m.
    This drops time reversal AND the field-reversing unitary ops (vertical
    mirrors, in-plane C2) that survive only as products with T — exactly the
    magnetic point group QE uses to reduce a noncollinear FM k-grid.
    """
    ntran = int(sym.sym_matrices.shape[0])               # spatial-only count
    Rc = np.asarray(sym.R_cart[:ntran], dtype=np.float64)  # (ntran,3,3) Cartesian
    detR = np.linalg.det(Rc)                              # +1 proper / -1 improper
    m = np.asarray(m_axis_cart, dtype=np.float64)
    m = m / np.linalg.norm(m)
    keep = np.array([np.allclose(detR[s] * (Rc[s] @ m), m, atol=tol)
                     for s in range(ntran)])
    idx = np.where(keep)[0]
    return Rc[idx], detR[idx], idx


def axial_projector(R_mpg, det_mpg):
    """Pmat = (1/|G|) sum_g det(R_g) R_g — the (3,3) trivial-rep projector for
    an AXIAL vector.  Real; idempotent on little-group-invariant input.  NB the
    per-op R_cart differs from the velocity's K-frame rotation by a transpose,
    but the group is closed under inverse so this projector is transpose-
    invariant (test Pmat at the group level, not per op)."""
    G = R_mpg.shape[0]
    return (det_mpg[:, None, None] * R_mpg).sum(axis=0) / G   # (3,3) real


def run_ibz(wfn, sym, meta, vnl_setup, nbnd, nocc, deps_tol, m_axis, sign):
    """Symmetry-reduced orbital-magnetization accumulation.

    Loops the nrk STORED irreducible k-points (raw G-flat coefficients, NO
    symmetry unfold of psi), builds v = 2(k+G) + sign*dV_NL/dk at each, forms
    the mu-independent pieces, accumulates the IBZ-weighted (3,) sums and the
    band-resolved (3,nb,nb) arrays, then symmetrizes with the magnetic-point-
    group axial projector.  Returns the same interface as the full-BZ branch:
    (cA, cB, PA_band_z, PB_band_z, m_spin_z, E, info).
    """
    import jax.numpy as jnp
    nrk = int(wfn.nkpts)                                  # stored IBZ count (BGW nrk)
    w_ibz = np.asarray(wfn.kweights, dtype=np.float64)    # IBZ weights, sum = 1
    if abs(float(w_ibz.sum()) - 1.0) > 1e-6:
        w_ibz = w_ibz / w_ibz.sum()
    B = np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)

    R_mpg, det_mpg, idx_mpg = magnetic_point_group(sym, m_axis)
    Pmat = axial_projector(R_mpg, det_mpg)
    ntran = int(sym.sym_matrices.shape[0])
    print(f"[orbmag-ibz] nk_ibz={nrk}  full-BZ={int(sym.nk_tot)}  "
          f"|G|={len(idx_mpg)} of {ntran} spatial ops (T + field-reversing "
          f"ops excluded)  mag-axis={tuple(float(x) for x in m_axis)}")
    print(f"[orbmag-ibz] kept op indices {idx_mpg.tolist()}; "
          f"Pmat@[0,0,1]={np.round(Pmat @ np.array([0,0,1.0]),4).tolist()} "
          f"Pmat@[1,0,0]={np.round(Pmat @ np.array([1.0,0,0]),4).tolist()}")

    psi_ibz = wfn.load(bands=(0, nbnd), k="ibz", sharding=None)  # (nrk,nb,ns,ngkmax)
    gvecs_ibz = np.asarray(wfn.gvecs(k="ibz"))            # (nrk,ngkmax,3) raw sphere
    ngk_v = np.asarray(wfn.ngk_valid(k="ibz"))           # (nrk,) valid G count

    cA = np.zeros(3, dtype=np.complex128); cB = np.zeros(3, dtype=np.complex128)
    PA_band = np.zeros((3, nbnd, nbnd), dtype=np.complex128)
    PB_band = np.zeros((3, nbnd, nbnd), dtype=np.complex128)
    S_sum = 0.0
    E = np.zeros((nrk, nbnd), dtype=np.float64)
    for i in range(nrk):
        ng = int(ngk_v[i])
        G_int = jnp.asarray(gvecs_ibz[i, :ng], dtype=jnp.int32)
        psi_G = jnp.asarray(np.asarray(psi_ibz[i])[:, :, :ng])   # (nb,ns,ng)
        k_crys = jnp.asarray(np.asarray(wfn.kpoints[i]), dtype=jnp.float64)
        eps = np.asarray(wfn.energies[0, i, :nbnd], dtype=np.float64)
        E[i] = eps
        v = np.asarray(momentum_matrix_k(psi_G, G_int, k_crys, jnp.asarray(B)))
        if sign != 0:
            kdata = vnl_ops.build_vnl_kdata_from_kvec(
                np.asarray(wfn.kpoints[i], dtype=float),
                np.asarray(gvecs_ibz[i, :ng], dtype=int), vnl_setup, compute_dZ=True)
            nsE = kdata.E_super.shape[0]
            psi_phys = psi_G[:, :nsE, :] if psi_G.shape[1] > nsE else psi_G
            v = v + sign * np.asarray(vnl_ops.vnl_velocity_matrix(
                psi_phys, kdata.Z, kdata.dZ, kdata.E_super))
        psi_np = np.asarray(psi_G)
        sz = (np.abs(psi_np[:, 0]) ** 2 - np.abs(psi_np[:, 1]) ** 2).sum(axis=1).real
        S_sum += w_ibz[i] * float(sz[:nocc].sum())
        pa, pb = orbital_pieces_at_k(v, eps, nocc, deps_tol)
        cA += w_ibz[i] * pa.sum(axis=(1, 2)); cB += w_ibz[i] * pb.sum(axis=(1, 2))
        PA_band += w_ibz[i] * pa; PB_band += w_ibz[i] * pb
        if (i + 1) % 6 == 0 or i == nrk - 1:
            print(f"         k_ibz {i+1}/{nrk}")

    cA = Pmat.astype(np.complex128) @ cA                 # symmetrize the (3,) vector
    cB = Pmat.astype(np.complex128) @ cB
    PA_band_z = np.einsum('a,anm->nm', Pmat[2], PA_band)  # z-row band-resolved
    PB_band_z = np.einsum('a,anm->nm', Pmat[2], PB_band)
    info = {"nk_ibz": nrk, "nG": len(idx_mpg), "idx": idx_mpg.tolist()}
    return cA, cB, PA_band_z, PB_band_z, -1.0 * S_sum, E, info


# ----------------------------------------------------------------------
#  Band-sum-free orbital magnetization (Sternheimer covariant derivative)
# ----------------------------------------------------------------------
def run_sternheimer_orbmag(wfn, sym, meta, vnl_setup, pseudos, nbnd, nocc,
                           truncation_2d):
    """Orbital magnetization WITHOUT an empty-band sum, via the covariant
    derivative |∂̃_a u_v⟩ = Q_k ∂_{k_a} u_v solved from H, dH/dk and the
    occupied projector (Sternheimer / DFPT linear response).

    Per full-BZ k, per occupied band v: solve the Sternheimer equation (reusing
    ``run_sternheimer.compute_kp_tangent_at_kvec``) for |∂̃_a u_v⟩ (a=x,y,z),
    then the per-k orbital-moment AXIAL VECTOR
        m_γ(k) = (−1/2) Im Σ_v ε_{γab} ⟨∂̃_a u_v|(H_k+ε_v−2μ)|∂̃_b u_v⟩.
    The conduction manifold is summed exactly inside the Sternheimer inverse, so
    the result is BAND-COUNT INDEPENDENT (no SOS tail).  μ-linear split:
        cA from the (H_k+ε_v)-sandwich, cB from the overlap ⟨∂̃_a|∂̃_b⟩
    ⇒ C_of_mu(μ) = cA − 2μ·cB, reusing the shared reporting verbatim.

    Returns the same 7-tuple as :func:`run_ibz`.  V_scf (the local KS potential)
    is reconstructed from the WFN's own density (`scf_potential`); no extra files.
    """
    import jax.numpy as jnp
    from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G,
                                   compute_ngkmax)
    from psp.run_sternheimer import (_psi_box_to_G_sphere,
                                     compute_kp_tangent_at_kvec)
    from psp.scf_potential import build_rho_val_from_wfn, build_dft_potentials
    from solvers.sternheimer_precond import (compute_per_band_kinetic,
                                             tpa_preconditioner_diag)

    nk = int(sym.nk_tot)
    w_k = 1.0 / nk
    bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    # --- V_scf = V_loc[UPF] + V_H[ρ_val] + V_xc[ρ_val] (rebuilt from the WFN) --
    print("[orbmag-sternheimer] reconstructing V_scf from the WFN density "
          "(full-BZ ρ_val; integral must equal nelec)...")
    rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
    V_scf, V_loc, _vnl2 = build_dft_potentials(
        wfn, pseudos, rho_val, truncation_2d=truncation_2d, verbose=True)
    ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, dtype=np.float64),
                                np.asarray(wfn.bdot), float(wfn.ecutwfc), fft_grid))
    # QE-DFPT level shift α_pv = 2(E_max − E_min) over occupied bands (all k)
    en_occ = np.asarray(wfn.energies[0, :, :nocc], dtype=np.float64)
    alpha_pv = jnp.asarray(2.0 * (float(en_occ.max()) - float(en_occ.min())),
                           dtype=jnp.float64)

    cA = np.zeros(3, dtype=np.complex128)
    cB = np.zeros(3, dtype=np.complex128)
    S_sum = 0.0
    E = np.zeros((nk, nbnd), dtype=np.float64)
    print(f"[orbmag-sternheimer] {nk} full-BZ k × {nocc} occ bands, "
          f"α_pv={float(alpha_pv):.2f} Ry; solving Sternheimer (no band sum)...")

    def _axial(M):                                       # M (nv,3,3) -> (3,) axial
        A = M - jnp.swapaxes(M, 1, 2)
        return jnp.stack([A[:, 1, 2], A[:, 2, 0], A[:, 0, 1]], axis=-1).sum(axis=0)

    for ik in range(nk):
        kv = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
        k_red = int(sym.irr_idx_k[ik])
        eps_full = np.asarray(wfn.energies[0, k_red, :nbnd], dtype=np.float64)
        E[ik] = eps_full
        eps_v = jnp.asarray(eps_full[:nocc], dtype=jnp.float64)

        box = load_kpoint_fftbox(wfn, sym, meta, ik, nbnd)     # unfolded ψ box
        H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta,
                                  V_loc_r=V_loc, ngkmax=ngkmax)
        Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
        maskf = H_k.mask[None, None, :]
        U_val_G = (_psi_box_to_G_sphere(box, Gk_int)[:nocc]
                   * maskf.astype(box.dtype))               # (nv, ns, nG)

        K_bar_sq = compute_per_band_kinetic(U_val_G, H_k.T_diag)
        precond = tpa_preconditioner_diag(H_k.T_diag, K_bar_sq)

        # |∂̃_a u_v⟩ via Sternheimer — (3, nv, ns, nG), occupied only, no band sum
        d = compute_kp_tangent_at_kvec(
            kv, np.asarray(Gk_int), vnl_setup, V_scf, H_k.mask,
            H_k.Gx, H_k.Gy, H_k.Gz, fft_grid, bdot, H_k.vnl_E,
            U_val_G, eps_v, alpha_pv, precond, tol=1e-10, max_iter=200)

        md = d * maskf.astype(d.dtype)

        def _Heps(da):                                   # (H_k + ε_v) |∂̃_a u_v⟩
            Hd = apply_H_k_from_G(da, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy,
                                  H_k.Gz, H_k.vnl_Z, H_k.vnl_E, H_k.mask)
            return Hd + eps_v[:, None, None] * (da * maskf.astype(da.dtype))
        opA = jax.vmap(_Heps)(d)                         # (3, nv, ns, nG)

        M0 = jnp.einsum('avsG,bvsG->vab', jnp.conj(md), opA, optimize=True)
        M1 = jnp.einsum('avsG,bvsG->vab', jnp.conj(md), md, optimize=True)
        cA += w_k * np.asarray(_axial(M0))               # (H+ε) sandwich
        cB += w_k * np.asarray(_axial(M1))               # overlap (the −2μ piece)

        psi_np = np.asarray(U_val_G)
        sz = (np.abs(psi_np[:, 0]) ** 2 - np.abs(psi_np[:, 1]) ** 2).sum(axis=1).real
        S_sum += w_k * float(sz.sum())
        if (ik + 1) % 6 == 0 or ik == nk - 1:
            print(f"         k {ik+1}/{nk}")

    z = np.zeros((nbnd, nbnd), dtype=np.complex128)
    info = {"nk_ibz": nk, "nG": 0, "idx": [], "method": "sternheimer"}
    return cA, cB, z, z, -1.0 * S_sum, E, info


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
    p.add_argument("--ibz", action="store_true",
                   help="Symmetry-reduced mode: loop the stored IBZ k-points "
                        "(no psi unfold) and symmetrize the axial-vector moment "
                        "density over the magnetic point group.")
    p.add_argument("--mag-axis", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                   metavar=("MX", "MY", "MZ"),
                   help="Cartesian magnetization direction selecting the magnetic "
                        "point group (default +z = crystal c). Op g kept iff "
                        "det(R_g) R_g @ axis == axis.")
    p.add_argument("--method", choices=["sos", "sternheimer"], default="sos",
                   help="Evaluation route: 'sos' = direct sum-over-states (band-"
                        "convergence pathological); 'sternheimer' = band-sum-free "
                        "covariant derivative (occupied states only, no empty-band "
                        "sum). Default sos.")
    p.add_argument("--truncation-2d", dest="truncation_2d",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="2D slab Coulomb truncation in V_H when rebuilding V_scf "
                        "(sternheimer mode). Default True (monolayer CrI3).")
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

    nrk = int(wfn.nkpts)
    if not args.ibz and nrk < int(sym.nk_tot):
        print(f"[orbmag] NOTE: WFN is symmetry-reduced (nrk={nrk} < nk_full="
              f"{int(sym.nk_tot)}); pass --ibz for the magnetic-symmetry "
              "axial-vector unfold (faster + exact).")

    out_extra = {}                                   # branch-specific --out payload
    if args.method == "sternheimer":
        # ---- band-sum-free Sternheimer covariant-derivative branch -------
        if vnl_setup is None:
            sys.exit("[orbmag] ERROR: --method sternheimer needs the full KS H "
                     "(no --skip-vnl).")
        cA, cB, PA_band_z, PB_band_z, m_spin_z, E, info = run_sternheimer_orbmag(
            wfn, sym, meta, vnl_setup, pseudos, nbnd, nocc, args.truncation_2d)

        def C_of_mu(m):
            return cA - 2.0 * m * cB
        out_extra = {"cA": cA, "cB": cB, "method": "sternheimer"}
    elif args.ibz:
        # ---- symmetry-reduced (IBZ) branch -------------------------------
        sign = 0 if args.skip_vnl else (-1 if args.vnl_sign == "minus" else +1)
        print(f"[orbmag] IBZ mode: v = p "
              f"{'(kinetic only)' if sign==0 else ('+' if sign>0 else '-')+' vNL'}"
              " (canonical p+vNL; HF FD-slope check is full-BZ-only)")
        cA, cB, PA_band_z, PB_band_z, m_spin_z, E, info = run_ibz(
            wfn, sym, meta, vnl_setup, nbnd, nocc, deps_tol,
            np.asarray(args.mag_axis, dtype=np.float64), sign)

        def C_of_mu(m):                              # (3,) complex; already symmetrized
            return cA - 2.0 * m * cB
        out_extra = {"cA": cA, "cB": cB, "ibz_ops": np.asarray(info["idx"]),
                     "sign": sign}
    else:
        # ---- full-BZ branch ----------------------------------------------
        # PERF: per-k FFT-box load + projector build is the wall-clock floor;
        # for dense grids prefer --ibz (loops nrk IBZ points, G-flat, no box).
        nk = int(sym.nk_tot)
        w_k = 1.0 / nk                               # uniform full-BZ weight
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

        # decide kinetic/nonlocal sign (HF diagonal test; canonical p+vNL)
        sign = 0
        if not args.skip_vnl:
            hf = hf_group_velocity_check(Vp, Vnl, E, Kc, B, wfn.kgrid)
            print("\n[orbmag] Hellmann-Feynman group-velocity check "
                  f"({hf['nsamples']} band/k samples):")
            print(f"         RMS |Re diag(v) - d eps/dk|:  "
                  f"p+vNL = {hf[1]:.4f}   p-vNL = {hf[-1]:.4f}  (Ry*Bohr)")
            # Physical sign = p+vNL (canonical velocity_matrix_k); verified
            # compute_vnl_velocity_cart == +dV_NL/dk off-diagonally (ratio 1.000).
            # dV_NL/dk is ~900x larger off-diagonal, so the diagonal HF test ties
            # — it validates kinetic part/units only.  Flip only on a large margin.
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

        V = Vp + sign * Vnl
        PA = np.zeros((3, nbnd, nbnd), dtype=np.complex128)
        PB = np.zeros((3, nbnd, nbnd), dtype=np.complex128)
        for ik in range(nk):
            pa, pb = orbital_pieces_at_k(V[ik], E[ik], nocc, deps_tol)
            PA += w_k * pa
            PB += w_k * pb
        PA_band_z, PB_band_z = PA[2], PB[2]
        m_spin_z = -1.0 * float((w_k * SZ[:, :nocc].sum(axis=1)).sum())

        def C_of_mu(m):
            return (PA - 2.0 * m * PB).sum(axis=(1, 2))
        out_extra = {"Vp": Vp, "Vnl": Vnl, "SZ": SZ, "Kc": Kc, "sign": sign}

    # ---- shared: chemical potential + reporting --------------------------
    VBM = float(E[:, nocc - 1].max())
    CBM = float(E[:, nocc].min()) if nocc < nbnd else VBM
    mu = args.mu / RY2EV if args.mu is not None else 0.5 * (VBM + CBM)
    gap_eV = (CBM - VBM) * RY2EV
    print(f"\n[orbmag] VBM={VBM*RY2EV:.4f} eV  CBM={CBM*RY2EV:.4f} eV  "
          f"indirect gap={gap_eV:.4f} eV   mu={mu*RY2EV:.4f} eV ({mu:.5f} Ry)")
    if gap_eV < 0:
        print("         NOTE: negative indirect gap at this k-sampling -> the "
              "moment is mu-dependent (run --mu-scan).")
    print(f"\n[orbmag] spin moment  sum_occ <sigma_z> = {-m_spin_z:+.4f}  -> "
          f"|m_spin| = {abs(m_spin_z):.3f} mu_B  (expect ~6 for CrI3)")

    m_orb = -MU_B_PREFACTOR * C_of_mu(mu).imag       # (3,) mu_B, file frame
    frame = 1.0 if m_spin_z >= 0 else -1.0
    m_orb_par = float(frame * m_orb[2])              # along spin-moment axis

    print("\n" + "=" * 64)
    print("ORBITAL MAGNETIC MOMENT  (per unit cell, mu_B)")
    print("=" * 64)
    print(f"  m_x = {m_orb[0]:+.5f}   m_y = {m_orb[1]:+.5f}   "
          f"(should be ~0 by symmetry)")
    print(f"  m_z = {m_orb[2]:+.5f}   (file z-axis = crystal c, out of plane)")
    print(f"  orbital moment along spin axis: {m_orb_par:+.5f} mu_B  "
          f"({'PARALLEL' if m_orb_par>0 else 'ANTIPARALLEL'} to spin)")
    print(f"  spin moment |m_spin| = {abs(m_spin_z):.3f} mu_B")
    if args.method == "sternheimer":
        print(f"  [Sternheimer covariant-derivative: BAND-SUM-FREE, "
              f"{info['nk_ibz']} full-BZ k, occupied-only]")
    elif args.ibz:
        print(f"  [IBZ-symmetrized: nk_ibz={info['nk_ibz']}, |G|={info['nG']}, "
              f"magnetic-group ops {info['idx']}]")
    print("=" * 64)

    if args.mu_scan and nocc < nbnd:
        print("\n[orbmag] mu-scan (m_z, mu_B):")
        for label, m in [("VBM", VBM), ("midgap", 0.5 * (VBM + CBM)), ("CBM", CBM)]:
            print(f"   mu={m*RY2EV:8.4f} eV ({label:6s}):  "
                  f"m_z = {-MU_B_PREFACTOR*float(C_of_mu(m)[2].imag):+.5f}")

    if args.method == "sternheimer" and (args.convergence or args.per_band):
        print("\n[orbmag] (--convergence/--per-band N/A for sternheimer: the "
              "result is BAND-COUNT INDEPENDENT by construction — the conduction "
              "manifold is summed exactly inside the Sternheimer inverse.)")

    if args.convergence and not args.skip_vnl and args.method != "sternheimer":
        print("\n[orbmag] convergence vs inner-m band ceiling (m_z, mu_B):")
        col_z = (PA_band_z - 2.0 * mu * PB_band_z).sum(axis=0)  # sum over occupied n
        cum = np.cumsum(col_z)                                  # partial sums over m
        for mc in sorted(set([int(0.5*nbnd), int(0.7*nbnd), int(0.85*nbnd), nbnd])):
            print(f"   mceil={mc:4d}:  m_z = {-MU_B_PREFACTOR*float(cum[mc-1].imag):+.5f}")

    if args.per_band and args.method != "sternheimer":
        band_z = (PA_band_z - 2.0 * mu * PB_band_z).sum(axis=1)  # per outer-n
        m_par_band = frame * (-MU_B_PREFACTOR) * band_z.imag
        print("\n[orbmag] per-occupied-band m_z (along spin axis, mu_B):")
        order = np.argsort(np.abs(m_par_band[:nocc]))[::-1]
        for n in order[:12]:
            print(f"   band {n:3d}: {m_par_band[n]:+.5f}")

    if args.out:
        # colA_z/colB_z: z-component band-resolved columns (summed over occ n,
        # BZ-weighted) — cumsum over the inner-m index gives m_z vs band ceiling
        # (the band-convergence curve) at any mu, in either mode.
        _mode = args.method if args.method == "sternheimer" else (
            "ibz" if args.ibz else "full")
        np.savez_compressed(args.out, E=E, mu=mu, nocc=nocc, m_orb=m_orb,
                            m_spin_z=m_spin_z, mode=_mode,
                            colA_z=PA_band_z.sum(axis=0), colB_z=PB_band_z.sum(axis=0),
                            **out_extra)
        print(f"\n[orbmag] wrote {args.out}")


if __name__ == "__main__":
    main()
