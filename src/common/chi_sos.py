"""SOS chi head/wing values + S/w tensors.

Sister to ``common.chi_from_dipole`` (which only does the q=0 quadratic
head, ``S_{αβ}``, in the Adler–Wiser dipole form).  This module:

  * consumes the finite-q matrix elements written by the augmented
    ``psp.get_dipole_mtxels --with-finite-q``  (``rho_cvkq``, ``v_cvkq``,
    ``kminq_idx`` in ``dipole.h5/finite_q``)
  * computes the head + wing values of  χ⁰_{0G'}(q, ω)  on the coarse
    q-grid,
  * computes the q=0 S-tensor and the linear wing-coefficient ``w_{α,μ}``
    (= ``∂χ⁰_wing,μ/∂q_α |_0``),
  * returns objects in the same conventions as the existing Sternheimer
    backend so head/wing-correction code is backend-agnostic.

Physics — pulling from the user math sketch (Adler–Wiser, two-pole at
ω=0 reduces to the static form, sign convention as in ``compute_S_omega``)::

    F_t(q, ω)  =  (f_v − f_c) · 2 Δ_t(q) / (ω² − Δ_t(q)²)

    Δ_t(q)     =  ε_{c, k-q}  −  ε_{v, k}              (electron-hole gap)

    h_t(q)     =  ⟨u_{c, k-q} | u_{v, k}⟩_cell       (= ``rho_cvkq``[c,v,k,q])

    χ_head(q, ω)        = (1 / N_k Ω) · Σ_t F_t(q, ω) · |h_t(q)|²
    χ_wing,μ(q, ω)      = (1 / N_k Ω) · Σ_t F_t(q, ω) · b*_{t,μ}(q) · h_t(q)

    S_{αβ}(ω)           = (1 / N_k Ω) · Σ_t F_t(0, ω) · d_t^{α*} d_t^β
                                                       (= ``compute_S_omega``)

    w_{α,μ}(ω)          = (1 / N_k Ω) · Σ_t F_t(0, ω) · b^{0*}_{t,μ} · d_t^α
                                                       (linear-in-q wing coef)

with  ``d_t^α  =  v^α_t / Δ_t``  (the Adler–Wiser dipole; from
``v_cvkq``[α, c, v, k, q=0] / (ε_c[k] - ε_v[k])).

The ``b_{t,μ}(q)`` centroid pair-density vertex is *not* yet plumbed
through — the head/wing values that depend on it require ISDF centroid
data (``ψ_{m,k-q}(r_μ) ψ*_{n,k}(r_μ)``).  ``compute_chi_head_at_q`` runs
without ``b_{t,μ}``;  ``compute_chi_wing_at_q`` and ``compute_w_tensor``
take ``b_t_mu`` arrays as input but their construction is the caller's
job (typically by gathering from the existing ISDF zeta pipeline).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Δ_t(q)  helper
# ---------------------------------------------------------------------------

def deltaE_cv_at_q(en_full: np.ndarray, kminq_idx_kq: np.ndarray,
                    v_lo: int, c_lo: int, c_hi: int) -> np.ndarray:
    """Build the electron-hole gap tensor  Δ_t(q) = ε_{c, k-q} − ε_{v, k}.

    Parameters
    ----------
    en_full       : (nk_full, nb) — band energies on the full BZ.
    kminq_idx_kq  : (nk_full, nq) int — output of ``get_dipole_mtxels --with-finite-q``.
    v_lo, c_lo, c_hi : int — band slice [v_lo : c_lo) (valence), [c_lo : c_hi) (cond).

    Returns
    -------
    Δ_cvkq : (nc, nv, nk_full, nq) float64
    """
    en = np.asarray(en_full)
    en_c_kmq = en[kminq_idx_kq, c_lo:c_hi]                    # (nk, nq, nc)
    en_v_k   = en[:, v_lo:c_lo]                                # (nk, nv)
    # broadcast: ε_c[k-q, c] - ε_v[k, v]
    delta = en_c_kmq[:, :, :, None] - en_v_k[:, None, None, :]    # (nk, nq, nc, nv)
    return np.transpose(delta, (2, 3, 0, 1)).astype(np.float64)   # (nc, nv, nk, nq)


# ---------------------------------------------------------------------------
# χ head / wing values at arbitrary q
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('omegas_is_scalar',))
def _compute_chi_head_jit(
    rho_cvkq, deltaE_cvkq, occ_v, occ_c, omegas, prefactor,
    *, omegas_is_scalar: bool,
):
    """χ_head(q, ω) = pref · Σ_{c,v,k} F_t(q, ω) · |rho_cvkq|².

    Inputs (all jax arrays):
      rho_cvkq    : (nc, nv, nk, nq) complex
      deltaE_cvkq : (nc, nv, nk, nq) float64
      occ_v       : (nv,) float — f_v (1.0 for valence)
      occ_c       : (nc,) float — f_c (0.0 for cond)
      omegas      : scalar or (n_w,) complex
      prefactor   : complex scalar (= 2 / (N_k · Ω) for the standard
                     Adler–Wiser convention with no spin doubling, OR
                     4 / (N_k Ω) when including spin sum factor of 2;
                     match whatever ``chi_from_dipole.compute_S_omega``
                     uses to keep convention)
    """
    fvfc = (occ_v[None, :, None, None] - occ_c[:, None, None, None])   # (nc, nv, 1, 1)
    rho_sq = jnp.abs(rho_cvkq) ** 2                                     # (nc, nv, nk, nq)
    weight = fvfc * rho_sq                                               # (nc, nv, nk, nq)

    def _chi_one(omega_val):
        denom = omega_val * omega_val - deltaE_cvkq * deltaE_cvkq
        denom_safe = jnp.where(jnp.abs(denom) > 1e-16, denom, 1.0)
        F = jnp.where(jnp.abs(denom) > 1e-16,
                       2.0 * deltaE_cvkq / denom_safe, 0.0)
        # Sum over c, v, k → leaves only q axis
        return prefactor * jnp.einsum('cvkq,cvkq->q', weight, F, optimize=True)

    if omegas_is_scalar:
        return _chi_one(omegas)[None, :]                                 # (1, nq)
    return jax.vmap(_chi_one)(omegas)                                    # (n_w, nq)


def compute_chi_head_at_q(
    rho_cvkq: np.ndarray,
    deltaE_cvkq: np.ndarray,
    occ_v: np.ndarray,
    occ_c: np.ndarray,
    omegas,
    cell_volume: float,
    nk_tot: int,
    nspin: int = 1,
    nspinor: int = 1,
):
    """Public driver: χ_head(q, ω) on the q-list embedded in rho_cvkq.

    Returns ``(n_w, nq)`` complex (singleton ω axis when scalar).
    """
    pref = 4.0 / (float(cell_volume) * float(nk_tot) *
                   float(max(nspin, 1)) * float(max(nspinor, 1)))
    pref_c = jnp.asarray(pref, dtype=jnp.complex128)
    omegas_arr = jnp.asarray(omegas, dtype=jnp.complex128)
    omegas_is_scalar = omegas_arr.ndim == 0

    return _compute_chi_head_jit(
        jnp.asarray(rho_cvkq, dtype=jnp.complex128),
        jnp.asarray(deltaE_cvkq, dtype=jnp.float64),
        jnp.asarray(occ_v, dtype=jnp.float64),
        jnp.asarray(occ_c, dtype=jnp.float64),
        omegas_arr, pref_c,
        omegas_is_scalar=omegas_is_scalar,
    )


@functools.partial(jax.jit, static_argnames=('omegas_is_scalar',))
def _compute_chi_wing_jit(
    b_cvkq_mu, rho_cvkq, deltaE_cvkq, occ_v, occ_c, omegas, prefactor,
    *, omegas_is_scalar: bool,
):
    """χ_wing,μ(q, ω) = pref · Σ_{c,v,k} F_t · b*_{t,μ}(q) · h_t(q).

    Inputs:
      b_cvkq_mu : (nc, nv, nk, nq, n_mu) complex — centroid vertex.
      rho_cvkq  : (nc, nv, nk, nq) complex — h_t(q).
    Returns (n_w, nq, n_mu) complex.
    """
    fvfc = (occ_v[None, :, None, None] - occ_c[:, None, None, None])
    weight_h = fvfc * rho_cvkq                                            # (nc, nv, nk, nq)

    def _chi_one(omega_val):
        denom = omega_val * omega_val - deltaE_cvkq * deltaE_cvkq
        denom_safe = jnp.where(jnp.abs(denom) > 1e-16, denom, 1.0)
        F = jnp.where(jnp.abs(denom) > 1e-16,
                       2.0 * deltaE_cvkq / denom_safe, 0.0).astype(jnp.complex128)
        # b* h: take conj(b) and multiply h, then sum.
        # einsum: 'cvkqm, cvkq, cvkq -> qm'  with conj(b) absorbed.
        return prefactor * jnp.einsum(
            'cvkqm,cvkq,cvkq->qm',
            jnp.conj(b_cvkq_mu), weight_h, F, optimize=True)

    if omegas_is_scalar:
        return _chi_one(omegas)[None, :, :]
    return jax.vmap(_chi_one)(omegas)


def compute_chi_wing_at_q(
    b_cvkq_mu: np.ndarray,
    rho_cvkq: np.ndarray,
    deltaE_cvkq: np.ndarray,
    occ_v: np.ndarray,
    occ_c: np.ndarray,
    omegas,
    cell_volume: float,
    nk_tot: int,
    nspin: int = 1,
    nspinor: int = 1,
):
    """Public driver: χ_wing,μ(q, ω) on the q-list embedded in rho_cvkq."""
    pref = 4.0 / (float(cell_volume) * float(nk_tot) *
                   float(max(nspin, 1)) * float(max(nspinor, 1)))
    pref_c = jnp.asarray(pref, dtype=jnp.complex128)
    omegas_arr = jnp.asarray(omegas, dtype=jnp.complex128)
    omegas_is_scalar = omegas_arr.ndim == 0

    return _compute_chi_wing_jit(
        jnp.asarray(b_cvkq_mu, dtype=jnp.complex128),
        jnp.asarray(rho_cvkq,  dtype=jnp.complex128),
        jnp.asarray(deltaE_cvkq, dtype=jnp.float64),
        jnp.asarray(occ_v, dtype=jnp.float64),
        jnp.asarray(occ_c, dtype=jnp.float64),
        omegas_arr, pref_c,
        omegas_is_scalar=omegas_is_scalar,
    )


# ---------------------------------------------------------------------------
# q=0 S-tensor and linear wing coefficient w
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('omegas_is_scalar',))
def _compute_S_tensor_jit(
    d_alpha_cvk, deltaE_cv_q0, occ_v, occ_c, omegas, prefactor,
    *, omegas_is_scalar: bool,
):
    """S_{αβ}(ω) = pref · Σ_t F_t(0,ω) · d_t^{α*} d_t^β  (full Hessian).

    Inputs:
      d_alpha_cvk  : (3, nc, nv, nk) complex — d_t^α = v^α_{cvk}/Δ_{cvk}.
      deltaE_cv_q0 : (nc, nv, nk) float — Δ_t(0).
    Returns (n_w, 3, 3) complex.

    Convention matches existing ``chi_from_dipole.compute_S_omega``:
    full Hessian of χ_head w.r.t. q_i, q_j (NOT quadratic-coef).
    """
    fvfc = (occ_v[None, :, None] - occ_c[:, None, None])                  # (nc, nv, nk)
    weight = fvfc[None, ...] * d_alpha_cvk                                 # (3, nc, nv, nk)

    def _S_one(omega_val):
        denom = omega_val * omega_val - deltaE_cv_q0 * deltaE_cv_q0
        denom_safe = jnp.where(jnp.abs(denom) > 1e-16, denom, 1.0)
        F = jnp.where(jnp.abs(denom) > 1e-16,
                       2.0 * deltaE_cv_q0 / denom_safe, 0.0).astype(jnp.complex128)
        # S_{αβ} = pref · Σ_{c,v,k}  conj(d^α) · d^β · F_t
        return prefactor * jnp.einsum(
            'acvk,bcvk,cvk->ab',
            jnp.conj(d_alpha_cvk), weight, F, optimize=True)

    if omegas_is_scalar:
        return _S_one(omegas)[None, :, :]
    return jax.vmap(_S_one)(omegas)


def compute_S_tensor_sos(
    v_alpha_cvk_q0: np.ndarray,
    deltaE_cv_q0: np.ndarray,
    occ_v: np.ndarray,
    occ_c: np.ndarray,
    omegas,
    cell_volume: float,
    nk_tot: int,
    nspin: int = 1,
    nspinor: int = 1,
):
    """SOS S-tensor at q=0.

    ``v_alpha_cvk_q0`` is the *velocity* matrix element at q=0 (= the
    ``dipole_cart`` slice in ``dipole.h5``).  We form  d^α_t  =
    v^α_t / Δ_t  here, with the same denominator regularisation pattern
    as ``compute_S_omega`` (which sets the term to 0 when |Δ| < 1e-12).
    """
    pref = 4.0 / (float(cell_volume) * float(nk_tot) *
                   float(max(nspin, 1)) * float(max(nspinor, 1)))
    pref_c = jnp.asarray(pref, dtype=jnp.complex128)

    delta = jnp.asarray(deltaE_cv_q0, dtype=jnp.float64)
    delta_safe = jnp.where(jnp.abs(delta) > 1e-12, delta, 1.0)
    d_alpha = jnp.where(jnp.abs(delta) > 1e-12,
                        jnp.asarray(v_alpha_cvk_q0, dtype=jnp.complex128)
                          / delta_safe.astype(jnp.complex128),
                        jnp.asarray(0.0 + 0.0j))

    omegas_arr = jnp.asarray(omegas, dtype=jnp.complex128)
    omegas_is_scalar = omegas_arr.ndim == 0
    return _compute_S_tensor_jit(
        d_alpha, delta,
        jnp.asarray(occ_v, dtype=jnp.float64),
        jnp.asarray(occ_c, dtype=jnp.float64),
        omegas_arr, pref_c,
        omegas_is_scalar=omegas_is_scalar,
    )


@functools.partial(jax.jit, static_argnames=('omegas_is_scalar',))
def _compute_w_tensor_jit(
    d_alpha_cvk, b0_cvk_mu, deltaE_cv_q0, occ_v, occ_c, omegas, prefactor,
    *, omegas_is_scalar: bool,
):
    """w_{α,μ}(ω) = pref · Σ_t F_t(0,ω) · b^{0*}_{t,μ} · d_t^α.

    Returns (n_w, 3, n_mu) complex.
    """
    fvfc = (occ_v[None, :, None] - occ_c[:, None, None])                  # (nc, nv, nk)

    def _w_one(omega_val):
        denom = omega_val * omega_val - deltaE_cv_q0 * deltaE_cv_q0
        denom_safe = jnp.where(jnp.abs(denom) > 1e-16, denom, 1.0)
        F = jnp.where(jnp.abs(denom) > 1e-16,
                       2.0 * deltaE_cv_q0 / denom_safe, 0.0).astype(jnp.complex128)
        # Σ_{c,v,k}  conj(b^0_μ) · d^α · (f_v - f_c) · F_t
        return prefactor * jnp.einsum(
            'cvkm,acvk,cvk,cvk->am',
            jnp.conj(b0_cvk_mu), d_alpha_cvk,
            fvfc.astype(jnp.complex128), F, optimize=True)

    if omegas_is_scalar:
        return _w_one(omegas)[None, :, :]
    return jax.vmap(_w_one)(omegas)


def compute_w_tensor_sos(
    v_alpha_cvk_q0: np.ndarray,
    b0_cvk_mu: np.ndarray,
    deltaE_cv_q0: np.ndarray,
    occ_v: np.ndarray,
    occ_c: np.ndarray,
    omegas,
    cell_volume: float,
    nk_tot: int,
    nspin: int = 1,
    nspinor: int = 1,
):
    """SOS linear-in-q wing coefficient w_{α,μ} at q=0.

    Same input convention as :func:`compute_S_tensor_sos` for the
    velocity / Δ_t factors.  ``b0_cvk_mu`` is the q=0 centroid pair-
    density vertex  ``b^0_{t,μ} = Σ_s ψ*_{c,k,s}(r_μ) ψ_{v,k,s}(r_μ)``.
    """
    pref = 4.0 / (float(cell_volume) * float(nk_tot) *
                   float(max(nspin, 1)) * float(max(nspinor, 1)))
    pref_c = jnp.asarray(pref, dtype=jnp.complex128)

    delta = jnp.asarray(deltaE_cv_q0, dtype=jnp.float64)
    delta_safe = jnp.where(jnp.abs(delta) > 1e-12, delta, 1.0)
    d_alpha = jnp.where(jnp.abs(delta) > 1e-12,
                        jnp.asarray(v_alpha_cvk_q0, dtype=jnp.complex128)
                          / delta_safe.astype(jnp.complex128),
                        jnp.asarray(0.0 + 0.0j))

    omegas_arr = jnp.asarray(omegas, dtype=jnp.complex128)
    omegas_is_scalar = omegas_arr.ndim == 0
    return _compute_w_tensor_jit(
        d_alpha,
        jnp.asarray(b0_cvk_mu, dtype=jnp.complex128),
        delta,
        jnp.asarray(occ_v, dtype=jnp.float64),
        jnp.asarray(occ_c, dtype=jnp.float64),
        omegas_arr, pref_c,
        omegas_is_scalar=omegas_is_scalar,
    )


__all__ = [
    "deltaE_cv_at_q",
    "compute_chi_head_at_q",
    "compute_chi_wing_at_q",
    "compute_S_tensor_sos",
    "compute_w_tensor_sos",
]
