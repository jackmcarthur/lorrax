from __future__ import annotations

"""
Utilities to compute S_{alpha,beta}(omega) from dipole.h5.

The velocity convention is the FILE's, not this module's: read it from
the h5's ``prov_vnl_velocity_sign`` attribute (-1 is the legacy arm, +1
the default since 2026-08-09; an absent attribute means a file written
before the stamp existed, which is the legacy arm).

Public API
----------
- read_dipole_h5(path) -> (dipole_cart, deltaE)
- compute_S_omega(dipole_cart, deltaE, f_nk, cell_volume, nk_tot, nspin, nspinor, omegas, eta=0.0)
"""


# ============================================================================
# WARNING: this S(ω) is in the DFT basis only — must be re-derived for SC GW
# ============================================================================
#
# The three inputs ``dipole_cart``, ``deltaE``, ``f_nk`` are all evaluated
# from a single *fixed* basis: the DFT eigenstates that wrote ``dipole.h5``.
# Concretely:
#
#   dipole_cart[α, k, m, n] = ⟨ψ^DFT_m,k | v_α | ψ^DFT_n,k⟩      (3, nk, nb, nb)
#   deltaE[k, m, n]         = E^DFT_n,k − E^DFT_m,k               (nk, nb, nb)
#   f_nk[k, n]              = occupation at DFT eigenvalue        (nk, nb)
#
# For a one-shot G0W0 / single-cycle QSGW, this is fine — the bra/ket states
# v_α is sandwiched between are the DFT eigenstates the rest of the pipeline
# also uses.  But every quantity here is wrong as soon as the pipeline
# rotates ψ to the QP basis (i.e. SC GW with U ≠ 1):
#
#   1. Basis-change of the velocity matrix
#      With U[k, m, n] = ⟨DFT_m,k | QP_n,k⟩ from eigh(H_qp_dft):
#
#          V_α^QP(k)_{mn} = Σ_{pq} U^*(k)_{pm} V_α^DFT(k)_{pq} U(k)_{qn}
#                         = (U(k)^† V_α^DFT(k) U(k))_{mn}
#
#      einsum: 'kpm,akpq,kqn->akmn' with (np.conj(U), v_dft, U).  This is
#      the OPPOSITE direction of ``sc_iteration._rotate_to_dft_basis``,
#      which does U·O·U^† to bring Σ from QP back to DFT.  The dipole is
#      naturally a DFT-basis operator that we want to express in QP basis,
#      hence the dagger is on the left.
#
#   2. Energy denominators
#      ΔE in the dipole formula appears in the denominator
#      ΔE · (ω² − ΔE²); if ΔE keeps DFT values while V is QP-rotated, the
#      head spectrum mis-aligns with the actual QP transitions by the QP
#      shifts.  After SC convergence, ΔE_QP[k, m, n] = E_QP[k, n] − E_QP[k, m]
#      should be used.
#
#   3. Occupations
#      For an insulator that stays insulating across SC, f_nk does not
#      change (still 1 for n < nelec, 0 otherwise — by sorted band index).
#      For metals or near-gap-closure systems where SC can swap valence
#      and conduction near E_F, f_nk must be rebuilt from E_QP vs. the
#      converged E_F^QP.
#
#   4. Self-energy contribution to the velocity operator (TODO, separate)
#      v = p/m + (i/ℏ)[r, V_KS].  At the GW level the KS potential is
#      replaced by V_H + Σ(ω), so the velocity operator itself acquires a
#      Σ-derivative piece that dipole.h5 does NOT capture.  The full
#      "GW-level dipole" is therefore not just U^† v^DFT U — it has an
#      additional ⟨m|∂Σ/∂k|n⟩-like term that needs a separate evaluation
#      in the QP basis.  Rotating the existing DFT dipoles is the
#      necessary first step but isn't sufficient for full QSGW
#      consistency.
#
# Consequences for the SC head correction in compute_ppm_sigma_pipeline:
#
#   - HeadResolver.at(ω) caches S(ω) computed at iteration 1's basis
#     (which is DFT for the iteration-0 init we use today).  The SC loop
#     reuses these cached samples for all subsequent iterations, so the
#     analytic head_gn is fitted to DFT-basis transitions and never
#     refreshed.  For an insulator with QP shifts ≪ gap this is a small
#     error (~ few percent on each head value, propagating to ~10s of
#     meV per band); for systems near gap closure it is wrong.
#
#   - The proper SC fix is to (a) refresh dipole_cart per-iteration via
#     the rotation above, (b) recompute deltaE from the iter's E_QP, (c)
#     refresh f_nk from E_QP vs. E_F^QP, (d) invalidate HeadResolver's
#     cache and reseed at the requested ω's.  Cleanest factoring is to
#     hold the FULL "GW-level dipole" (point 4 above) on SCInputs and
#     update it in lockstep with the H_qp_dft carry.  Until this is done,
#     SC GW results carry an O(QP-shift / gap) systematic on the head.
#
# This file is intentionally NOT modified to do any of (1)-(4); the
# above is a roadmap for whoever does the full GW-level dipole update
# (it has to also include the Σ-derivative piece, which is a separate
# computation outside this module).
# ============================================================================


import functools

import numpy as np
import h5py
import jax
import jax.numpy as jnp


def read_dipole_h5(path: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    with h5py.File(path, "r") as h5:
        dipole_np = np.asarray(h5["dipole_cart"])  # (3, nk, nb, nb)
        deltaE_np = np.asarray(h5["deltaE"])       # (nk, nb, nb)
    return jnp.asarray(dipole_np, dtype=jnp.complex128), jnp.asarray(deltaE_np, dtype=jnp.float64)


@functools.partial(jax.jit, static_argnames=('nelec', 'nb', 'omegas_is_scalar'))
def _compute_S_omega_jit(
    dipole_cart, deltaE, f_nk, omegas, pref_c, eta_c,
    *, nelec: int, nb: int, omegas_is_scalar: bool,
):
    """JIT'd body — all eager arithmetic (arange/gather/subtract/where/
    einsum/vmap) compiled into one XLA module per (nelec, nb,
    omegas_is_scalar) combo.  Replaces ~30 eager-pjit cache misses per
    call (lines 40-50 of the original Python wrapper) with one cached
    compile.
    """
    c_idx = jnp.arange(nelec, nb)
    v_idx = jnp.arange(0, nelec)

    v_cvk = dipole_cart[:, :, c_idx[:, None], v_idx[None, :]]  # (3,nk,nc,nv)
    dE_cv = deltaE[:, c_idx[:, None], v_idx[None, :]]           # (nk,nc,nv)
    f_v = f_nk[:, v_idx]
    f_c = f_nk[:, c_idx]
    fv_minus_fc = f_v[:, None, :] - f_c[:, :, None]

    def S_one(omega_val):
        w_c = omega_val + eta_c
        denom = dE_cv * (w_c * w_c - dE_cv * dE_cv)
        W = jnp.where(jnp.abs(denom) > 1e-16, fv_minus_fc / denom, 0.0 + 0.0j)
        W = pref_c * W
        return jnp.einsum('ancv,ncv,bncv->ab', jnp.conj(v_cvk), W, v_cvk, optimize=True)

    if omegas_is_scalar:
        return S_one(omegas)[None, :, :]
    return jax.vmap(S_one, in_axes=0, out_axes=0)(omegas)


def compute_S_omega(
    dipole_cart: jnp.ndarray,
    deltaE: jnp.ndarray,
    f_nk: jnp.ndarray,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    omegas: jnp.ndarray,
    eta: float = 0.0,
) -> jnp.ndarray:
    nb = int(f_nk.shape[1])
    occ0 = jnp.asarray(f_nk[0], dtype=jnp.float64)
    nelec = int(jnp.clip(jnp.sum(occ0 > 0.5), 0, nb))

    pref = 4.0 / (float(cell_volume) * float(nk_tot) * float(max(nspin, 1)) * float(max(nspinor, 1)))
    pref_c = jnp.asarray(pref, dtype=jnp.complex128)
    eta_c = jnp.asarray(1j * float(eta), dtype=jnp.complex128)

    omegas_arr = jnp.asarray(omegas, dtype=jnp.complex128)
    omegas_is_scalar = omegas_arr.ndim == 0
    return _compute_S_omega_jit(
        dipole_cart, deltaE, f_nk, omegas_arr, pref_c, eta_c,
        nelec=nelec, nb=nb, omegas_is_scalar=omegas_is_scalar,
    )


__all__ = [
    "read_dipole_h5",
    "compute_S_omega",
]

