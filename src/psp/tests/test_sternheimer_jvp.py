#!/usr/bin/env python3
"""Sanity + validation tests for ``psp.run_sternheimer``'s JVP path.

Covers:
  1. Forward primal χ on all 9 reduced q of MoS2 3×3 nosym — physical reality
     check (pure-real, D₃-symmetric orbits).  Already validated in the
     ``sternheimer_signed_q.h5`` write-out; this test re-checks it lives.
  2. **q → 0 limit** of ``chi_col_total``: primal χ ≡ 0, and the *first
     derivative* ∂χ/∂q should also vanish at q=0 by inversion symmetry
     (χ is quadratic in q around Γ, so the gradient is null).  This also
     confirms the primal fast-path (‖b‖<1e-12 → δu=0) composes cleanly
     with the ``@jax.custom_jvp`` rule (i.e. the tangent solve still runs
     and returns something sensible).
  3. **Schur warm-start + JVP composition.**  Run the same first-derivative
     FD check at q=(1/3, 1/3) with ``use_schur=True`` and verify the
     tangent matches FD (same tolerance as ``use_schur=False``).  Both
     solves should go through the same operator and just differ in x₀;
     the custom_jvp rule threads ``use_schur`` through to the tangent.
  4. **BZ-boundary q guard.**  Prints a warning (does not fail) if any q
     in the reduced grid has a component |q_i − round(q_i)| = 1/2 (BZ
     boundary in one direction), since my "generic-q" derivation drops
     the ``2q ∈ reciprocal lattice`` term that would otherwise appear.
     On a 3×3 mesh no such q exists; on 4×4 or coarser grids this would
     fire on M-type points.

Run:
    cd <dir with WFN.h5 + UPFs>
    lxrun python3 -u -m psp.tests.test_sternheimer_jvp
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

from runtime import set_default_env
set_default_env()

import jax
import jax.numpy as jnp

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WfnLoader as WFNReader
from psp.dft_operators import setup_H_k_from_kvec
from psp.pseudos import load_pseudopotentials
from psp.run_sternheimer import (
    _psi_box_to_G_sphere,
    build_Gprime_list,
    build_sternheimer_source,
    chi_col_contrib_at_kvec_traced,
    make_density_perturbation,
    make_umklapp_phase,
)
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag


# ═══════════════════════════════════════════════════════════════════════
#  Test fixtures
# ═══════════════════════════════════════════════════════════════════════

def _build_fixture(wfn_path: str, pseudo_dir: str, truncation_2d: bool,
                   n_extra: int = 20):
    """Build test fixture.  ``n_extra`` = # extra conduction bands loaded for
    the Schur warm-start tests (0 = plain-CG-only)."""
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    n_occ = int(wfn.nelec)
    n_band = n_occ + max(0, int(n_extra))
    n_band = min(n_band, int(wfn.nbands))
    meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=n_band-n_occ,
                            nband=n_band, n_rmu=0, bispinor=False)
    pseudos = load_pseudopotentials(pseudo_dir)
    rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
    V_scf, V_loc, vnl_setup = build_dft_potentials(
        wfn, pseudos, rho_val, truncation_2d=truncation_2d, verbose=False)
    nk_full = int(sym.nk_tot)
    nspinor = int(wfn.nspinor)
    psi_box_full = jnp.stack(
        [load_kpoint_fftbox(wfn, sym, meta, ik, n_band) for ik in range(nk_full)],
        axis=0)
    en_full = jnp.asarray(
        wfn.energies[0, np.asarray(sym.irr_idx_k), :n_band])
    return dict(
        wfn=wfn, sym=sym, n_occ=n_occ, nk_full=nk_full, nspinor=nspinor,
        n_extra=n_band-n_occ, n_band=n_band,
        meta=meta, vnl_setup=vnl_setup, psi_box_full=psi_box_full,
        en_full=en_full, V_scf=V_scf, V_loc=V_loc,
    )


def _per_k_data(fx, ik_full, iq_red, q_signed):
    """Per-(k_full, q) numeric + JAX scaffolding for chi_col_contrib."""
    sym = fx['sym']
    wfn = fx['wfn']
    meta = fx['meta']
    n_occ = fx['n_occ']
    V_scf = fx['V_scf']; V_loc = fx['V_loc']; vnl_setup = fx['vnl_setup']
    psi_box_full = fx['psi_box_full']
    en_full = fx['en_full']

    kvec_k = np.asarray(sym.unfolded_kpts[ik_full], dtype=np.float64)
    ik_kminq = int(sym.kq_map[ik_full, iq_red])
    kvec_kminq_wrap = np.asarray(sym.unfolded_kpts[ik_kminq], dtype=np.float64)

    H_k     = setup_H_k_from_kvec(kvec_k,          V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    H_kminq = setup_H_k_from_kvec(kvec_kminq_wrap, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    Gk_int_j     = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    Gkminq_int_j = jnp.stack([H_kminq.Gx, H_kminq.Gy, H_kminq.Gz], axis=-1).astype(jnp.int32)
    Gkminq_np = np.asarray(Gkminq_int_j)

    U_val_k_box = psi_box_full[ik_full, :n_occ]
    U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, Gk_int_j)
    U_val_kminq_G = _psi_box_to_G_sphere(psi_box_full[ik_kminq, :n_occ], Gkminq_int_j)

    G_wrap_np = np.rint((kvec_k - q_signed) - kvec_kminq_wrap).astype(np.int32)
    phase_wrap = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=+1)
    phase_unwrap = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=-1)
    V_pert = make_density_perturbation(wfn.fft_grid) * phase_wrap

    Q = make_Q_kminq(U_val_kminq_G)
    b = build_sternheimer_source(U_val_k_box, Gkminq_int_j, V_pert, Q)
    eps_vk = en_full[ik_full, :n_occ]
    K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
    precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)

    # Schur extras at k-q (for warm-start tests):
    n_extra = fx.get('n_extra', 0)
    if n_extra > 0:
        U_extra_kminq_G = _psi_box_to_G_sphere(
            psi_box_full[ik_kminq, n_occ:n_occ + n_extra], Gkminq_int_j)
        eps_extra_kminq = en_full[ik_kminq, n_occ:n_occ + n_extra]
    else:
        U_extra_kminq_G = None
        eps_extra_kminq = None

    return dict(
        Gkminq_int_np=Gkminq_np,
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        V_scf=H_kminq.V_scf, mask=H_kminq.mask,
        vnl_E_super=H_kminq.vnl_E,
        U_val_kminq_G=U_val_kminq_G, eps_v=eps_vk,
        precond_diag=precond_diag,
        U_val_k_box=U_val_k_box, b=b,
        phase_unwrap=phase_unwrap,
        kvec_kminq_wrap_j=jnp.asarray(kvec_kminq_wrap, dtype=jnp.float64),
        U_extra_kminq_G=U_extra_kminq_G,
        eps_extra_kminq=eps_extra_kminq,
    )


def _make_chi_col_total(fx, iq_red, q_signed, ng_out, tol, max_iter, use_schur):
    """Build the end-to-end  qvec → χ_col[G']  function for a given iq."""
    wfn = fx['wfn']
    nk_full = fx['nk_full']
    nspinor = fx['nspinor']
    n_occ = fx['n_occ']
    N_grid = int(np.prod(wfn.fft_grid))
    V_cell = float(wfn.cell_volume)
    spin_factor = 2 if nspinor == 1 else 1
    prefactor = jnp.asarray(
        (2.0 * spin_factor * np.sqrt(N_grid)) / (V_cell * nk_full),
        dtype=jnp.float64)
    bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    Gprime_int = build_Gprime_list(q_signed, wfn, ng_out)
    Gprime_j = jnp.asarray(Gprime_int)
    per_k = [_per_k_data(fx, ik, iq_red, q_signed) for ik in range(nk_full)]
    en_occ = np.asarray(fx['en_full'])
    alpha_pv_j = jnp.asarray(
        2.0 * (en_occ.max() - en_occ.min()), dtype=jnp.float64)
    q_base_j = jnp.asarray(q_signed, dtype=jnp.float64)

    def chi_col_total(qvec):
        chi = jnp.zeros((ng_out,), dtype=jnp.complex128)
        for pk in per_k:
            kvec_p = pk['kvec_kminq_wrap_j'] - (qvec - q_base_j)
            chi += chi_col_contrib_at_kvec_traced(
                kvec_p_traced=kvec_p,
                Gkminq_int_np=pk['Gkminq_int_np'],
                vnl_setup=fx['vnl_setup'],
                V_scf=pk['V_scf'], mask=pk['mask'],
                Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
                fft_grid=wfn.fft_grid,
                bdot=bdot, vnl_E_super=pk['vnl_E_super'],
                U_val_kminq_G=pk['U_val_kminq_G'], eps_v=pk['eps_v'],
                alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
                U_val_k_box=pk['U_val_k_box'], b=pk['b'],
                Gprime_int=Gprime_j, phase_unwrap=pk['phase_unwrap'],
                prefactor=prefactor,
                tol=tol, max_iter=max_iter, use_schur=use_schur,
                U_extra_G=pk['U_extra_kminq_G'] if use_schur else None,
                eps_extra=pk['eps_extra_kminq'] if use_schur else None,
            )
        return chi

    return chi_col_total, Gprime_int, q_base_j


# ═══════════════════════════════════════════════════════════════════════
#  Individual tests
# ═══════════════════════════════════════════════════════════════════════

def _print_hdr(name):
    bar = '=' * max(40, len(name) + 4)
    print(f"\n{bar}\n  {name}\n{bar}")


def test_bz_boundary_guard(fx):
    _print_hdr("BZ-boundary-q guard")
    wfn = fx['wfn']
    kgrid = tuple(int(x) for x in wfn.kgrid)
    flagged = []
    for iq in range(int(wfn.nkpts)):
        q = np.asarray(wfn.kpoints[iq])
        q_signed = q - np.round(q)
        # A BZ-boundary component satisfies  |q_signed_i| == 1/2 exactly
        # (or more robustly, within float tolerance).  In those cases 2q is
        # a reciprocal lattice vector and the "2q+G'" contribution to χ
        # is non-zero — my generic-q derivation drops it.
        bdry = [abs(abs(qc) - 0.5) < 1e-6 for qc in q_signed]
        if any(bdry):
            flagged.append((iq, q_signed, bdry))
    if flagged:
        print(f"  WARNING: {len(flagged)} BZ-boundary q-point(s) in reduced grid:")
        for iq, qs, bd in flagged:
            print(f"    iq={iq}  q_signed={qs}   boundary axes={bd}")
        print(f"  → χ at these q is missing a ``2q·r`` contribution.")
        print(f"    (On {kgrid} mesh this happens for q=1/2 components.)")
    else:
        print(f"  OK — no BZ-boundary q's in the reduced-{kgrid} grid.")
    return len(flagged)


def test_forward_chi_symmetry(fx):
    _print_hdr("Forward χ_{00}(q, 0) — orbit consistency")
    from solvers.sternheimer_solve import sternheimer_solve
    # Use nk_full-many q's (the reduced grid); compute χ₀₀ at each and group
    # by |q_signed|² into orbits.  Within each orbit values should be equal.
    vals = {}
    for iq in range(int(fx['wfn'].nkpts)):
        q = np.asarray(fx['wfn'].kpoints[iq])
        q_signed = q - np.round(q)
        chi_fn, Gprime, _ = _make_chi_col_total(
            fx, iq, q_signed, ng_out=1, tol=1e-8, max_iter=120, use_schur=False)
        chi0 = chi_fn(jnp.asarray(q_signed, dtype=jnp.float64))[0]
        chi0.block_until_ready()
        bdot = np.asarray(fx['wfn'].bdot)
        qsq = float(q_signed @ bdot @ q_signed)
        key = round(qsq, 8)
        vals.setdefault(key, []).append((iq, q_signed, complex(chi0)))
    max_orbit_spread = 0.0
    for qsq, items in sorted(vals.items()):
        values = [v for (_, _, v) in items]
        centre = values[0]
        spread = max(abs(v - centre) for v in values) if values else 0
        max_orbit_spread = max(max_orbit_spread, spread)
        tag = "=" if spread < 1e-4 else "≠"
        print(f"  |q|²={qsq:.4f}  n={len(items)}: "
              f"{', '.join(f'{v.real:+.4f}{v.imag:+.1e}j' for v in values[:3])}"
              f"{'  ...' if len(values)>3 else ''}   spread={spread:.2e} {tag}")
    print(f"  max orbit spread: {max_orbit_spread:.2e}   "
          f"{'PASS' if max_orbit_spread < 1e-4 else 'FAIL'}")
    return max_orbit_spread


def test_q_zero_jvp(fx):
    _print_hdr("q → 0 JVP limit (primal=0, tangent=0 by inversion symmetry)")
    # Pick iq=0 for q=0 exactly (primal fast path).  Take a directional
    # derivative; by χ being quadratic around Γ, ∂χ/∂q|_{q=0} should be 0.
    iq_base = 0
    q_base = np.zeros(3)
    chi_fn, Gprime, q_base_j = _make_chi_col_total(
        fx, iq_base, q_base, ng_out=4, tol=1e-10, max_iter=200, use_schur=False)
    chi0 = chi_fn(q_base_j); chi0.block_until_ready()
    dq = jnp.asarray([1.0, 0.0, 0.0])
    _, chi_dot = jax.jvp(chi_fn, (q_base_j,), (dq,))
    chi_dot.block_until_ready()
    primal_max = float(jnp.max(jnp.abs(chi0)))
    tangent_max = float(jnp.max(jnp.abs(chi_dot)))
    print(f"  max|χ(0)|       = {primal_max:.3e}  (expect 0)")
    print(f"  max|∂χ/∂q_x|_0  = {tangent_max:.3e}  (expect 0 by inversion sym)")
    status = 'PASS' if (primal_max < 1e-12 and tangent_max < 1e-8) else 'FAIL'
    print(f"  {status}")
    return primal_max, tangent_max


def test_finite_q_jvp(fx, use_schur=False):
    _print_hdr(f"∂χ/∂q at q=(1/3,1/3):  JVP vs FD   (use_schur={use_schur})")
    iq_base = 4   # (1/3, 1/3) signed = (1/3, 1/3)  (non-wrap, K-orbit)
    q_base = np.asarray(fx['wfn'].kpoints[iq_base])
    q_base = q_base - np.round(q_base)
    chi_fn, Gprime, q_base_j = _make_chi_col_total(
        fx, iq_base, q_base, ng_out=4, tol=1e-10, max_iter=200, use_schur=use_schur)
    dq_x = jnp.asarray([1.0, 0.0, 0.0])
    _, chi_dot = jax.jvp(chi_fn, (q_base_j,), (dq_x,))
    chi_dot.block_until_ready()

    h = 1e-5
    chi_p = chi_fn(q_base_j + h*dq_x)
    chi_m = chi_fn(q_base_j - h*dq_x)
    chi_dot_fd = (chi_p - chi_m) / (2*h)
    rel = float(jnp.linalg.norm(chi_dot - chi_dot_fd)) / float(jnp.linalg.norm(chi_dot_fd) + 1e-30)
    print(f"  rel_err(JVP vs FD) = {rel:.3e}")
    status = 'PASS' if rel < 1e-4 else 'FAIL'
    print(f"  {status}")
    return rel


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", default="WFN.h5")
    ap.add_argument("--pseudo_dir", default=".")
    ap.add_argument("--truncation-2d", action="store_true", default=True)
    ap.add_argument("--skip-symmetry", action="store_true",
                    help="Skip the forward-χ orbit-consistency test (slow).")
    args = ap.parse_args()

    fx = _build_fixture(args.wfn, args.pseudo_dir, args.truncation_2d)

    fails = []
    n_bdry = test_bz_boundary_guard(fx)
    if not args.skip_symmetry:
        s = test_forward_chi_symmetry(fx)
        if s > 1e-4:
            fails.append(f"orbit-consistency spread={s:.2e}")
    pm, tm = test_q_zero_jvp(fx)
    if pm > 1e-12 or tm > 1e-8:
        fails.append(f"q=0 JVP: primal={pm:.2e} tangent={tm:.2e}")
    r0 = test_finite_q_jvp(fx, use_schur=False)
    if r0 > 1e-4:
        fails.append(f"finite-q JVP (plain CG): rel_err={r0:.2e}")
    r1 = test_finite_q_jvp(fx, use_schur=True)
    if r1 > 1e-4:
        fails.append(f"finite-q JVP (Schur): rel_err={r1:.2e}")

    _print_hdr("Summary")
    if fails:
        print("  FAILED:")
        for f in fails:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  All JVP tests PASSED.  {n_bdry} BZ-boundary q-pts flagged.")


if __name__ == "__main__":
    main()
