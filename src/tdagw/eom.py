"""tdagw/eom.py — the TD-aGW equation of motion in TDGW's conventions.

The state is the one-particle density matrix in the quasiparticle band
basis, rho_k,nm = -i G^<_k,nm.  TDGW propagates G^< = i rho; the two differ
by the exact factor i, so SciPy's RK45 error norm, and the steps it chooses,
are the same for either.  With E(t) the field in Hartree atomic units and
the electron charge -e (dipole coupling +E·r, length gauge):

    d rho_k/dt = -i ( [H_k(t), rho_k] + E(t)·[r^intra, rho]_k )
                 - Gamma ∘ (rho_k - rho^eq_k)                          (1)
    H_k(t)     = diag(E^QP_k) + E(t)·r^inter_k + dSigma_k[rho(t)]      (2)

(TDGW ``td/driver/agw.py`` ``aGWDriver.calc_rhs``).  Gamma is the
depopulation rate on the diagonal and the dephasing rate off it.
[r^intra, M] is the covariant k-derivative commutator of PRB 92, 235307
(2015) Eq. (6) in Yang-Hao's unitary-overlap form,

    [r^intra, M]_k = i sum_b w_b b (S~_{k,b} M_{k+b} S~^H_{k,b} - M_k)
                     + (r^intra_{k,n} - r^intra_{k,m}) M_{k,nm},       (3)

the second term only along a vacuum direction (``mf/dipole/dipwfn.py``).

Array layout everywhere: a leading TRIAL axis (several drives propagated
together, length >= 1) then the flat k axis, ``rho`` is
``(ntr, nk, nb, nb)``; k rides the whole mesh (``P(None, ('x','y'), ...)``)
and is padded to a multiple of the device count with inert zeros (zero
weight, zero operators), so no reduction or commutator sees a pad.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

__all__ = ["IntraBand", "TdOperators", "field_coupling", "field_hamiltonian", "commutator",
           "intraband_commutator", "field_intraband", "eom_rhs",
           "polarization", "current_operator", "current",
           "dissipation_rates"]


class IntraBand(NamedTuple):
    """The intraband position operator of Eq. (3).

    stilde : (nbvec, nk, nb, nb) complex — the unitary intraband overlap
        S~_{k,b} = U V^H of the energy-masked <u_k|u_{k+b}> (TDGW
        ``covariant_olp``); zero rows on k pads.
    kmap : (nbvec, nk) int32 — flat index of k+b; a pad maps to itself.
    wb : (3, nbvec) float — w_b times b in Cartesian bohr^-1 (TDGW ``wb_ib``).
    rintra : (3, nk, nb) float or None — <u_kn| r |u_kn> along the vacuum
        direction(s), bohr; None for bulk.
    """
    stilde: jax.Array
    kmap: jax.Array
    wb: jax.Array
    rintra: Optional[jax.Array]


class TdOperators(NamedTuple):
    """The static one-body operators of Eqs. (1)-(2), k-sharded.

    e_qp : (nk, nb) float, quasiparticle energies, Ha.
    rho_eq : (nk, nb, nb) complex, diag(f_kn) (valence 1, conduction 0).
    r_inter : (3, nk, nb, nb) complex, <u_kn| r |u_km>(1 - delta_nm), bohr.
    gamma : (nb, nb) float, depopulation (diagonal) / dephasing (off), Ha.
    w_k : (nk,) float, k weights summing to 1 over the logical k; 0 on pads.
    v0_inter : (3, nk, nb, nb) complex, i (E_n - E_m) r_nm, the
        equilibrium interband velocity.
    v0_intra : (3, nk, nb) complex or None, i dE_kn/dk (TDGW ``v0intra``).
    intra : IntraBand or None (no intraband commutator).
    """
    e_qp: jax.Array
    rho_eq: jax.Array
    r_inter: jax.Array
    gamma: jax.Array
    w_k: jax.Array
    v0_inter: jax.Array
    v0_intra: Optional[jax.Array]
    intra: Optional[IntraBand]


def dissipation_rates(depopulation_ha: float, dephasing_ha: float, nb: int):
    """Gamma_nm = depopulation (n = m) or dephasing (n != m), Ha (TDGW)."""
    eye = jnp.eye(nb)
    return depopulation_ha * eye + dephasing_ha * (1.0 - eye)


def commutator(a, b):
    """[a, b] = a b - b a over the trailing two axes."""
    return a @ b - b @ a


def field_coupling(ops: TdOperators, efield):
    """E(t)·r^inter_k, one per trial: (ntr, 3) -> (ntr, nk, nb, nb), Ha."""
    return jnp.einsum("ti,iknm->tknm", efield.astype(ops.r_inter.dtype),
                      ops.r_inter)


def field_hamiltonian(ops: TdOperators, efield):
    """diag(E^QP_k) + E(t)·r^inter_k of Eq. (2), one per trial.

    efield : (ntr, 3) float, a.u.  Returns (ntr, nk, nb, nb) complex.
    """
    h0 = jnp.zeros(ops.r_inter.shape[1:], ops.r_inter.dtype)
    nb = ops.e_qp.shape[-1]
    h0 = h0.at[:, jnp.arange(nb), jnp.arange(nb)].set(ops.e_qp.astype(h0.dtype))
    return h0[None] + field_coupling(ops, efield)


def intraband_commutator(intra: IntraBand, m):
    """[r^intra, m] of Eq. (3), all three Cartesian components.

    m : (ntr, nk, nb, nb).  Returns (3, ntr, nk, nb, nb).
    The k+b gather is a global take on the k axis (the one cross-k access
    of the propagator outside dSigma).
    """
    m_kb = jnp.take(m, intra.kmap, axis=1)                  # (ntr, nbvec, nk, nb, nb)
    s = intra.stilde[None]
    sgs = s @ m_kb @ jnp.conj(jnp.swapaxes(s, -1, -2)) - m[:, None]
    out = 1j * jnp.einsum("ib,tbknm->itknm", intra.wb.astype(m.dtype), sgs)
    if intra.rintra is not None:
        r = intra.rintra.astype(m.dtype)                    # (3, nk, nb)
        out = out + r[:, None, :, :, None] * m[None] - m[None] * r[:, None, :, None, :]
    return out


def field_intraband(intra: IntraBand, efield, m):
    """E(t)·[r^intra, m] of Eq. (1), contracted over the field components.

    efield : (ntr, 3).  Returns (ntr, nk, nb, nb); never materializes the
    three components.
    """
    m_kb = jnp.take(m, intra.kmap, axis=1)
    s = intra.stilde[None]
    sgs = s @ m_kb @ jnp.conj(jnp.swapaxes(s, -1, -2)) - m[:, None]
    coef = efield.astype(m.dtype) @ intra.wb.astype(m.dtype)    # (ntr, nbvec)
    out = 1j * jnp.einsum("tb,tbknm->tknm", coef, sgs)
    if intra.rintra is not None:
        er = jnp.einsum("ti,ikn->tkn", efield.astype(m.dtype),
                        intra.rintra.astype(m.dtype))
        out = out + er[..., :, None] * m - m * er[..., None, :]
    return out


def eom_rhs(ops: TdOperators, efield, dsigma, rho):
    """d rho/dt of Eq. (1).

    efield : (ntr, 3) a.u.; dsigma : (ntr, nk, nb, nb) Ha (dSigma[rho(t)]);
    rho : (ntr, nk, nb, nb).  Same layout out.
    """
    h = field_hamiltonian(ops, efield) + dsigma
    drive = commutator(h, rho)
    if ops.intra is not None:
        drive = drive + field_intraband(ops.intra, efield, rho)
    return -1j * drive - ops.gamma.astype(rho.dtype) * (rho - ops.rho_eq[None])


def _expectation(w_k, cell_volume, nspinor, rho, op):
    """sum_k w_k Tr(rho_k op_i,k) / Omega, x2 for nspinor = 1 (TDGW).

    rho : (ntr, nk, nb, nb); op : (3, [ntr,] nk, nb, nb).  Returns (ntr, 3).
    """
    if op.ndim == 4:
        tr = jnp.einsum("tkmn,iknm->tik", rho, op)
    else:
        tr = jnp.einsum("tkmn,itknm->tik", rho, op)
    out = (tr @ w_k.astype(tr.dtype)) / cell_volume
    return 2.0 * out if int(nspinor) == 1 else out


def polarization(ops: TdOperators, rho, *, cell_volume: float, nspinor: int):
    """P_i = -sum_k w_k Tr(rho_k r^inter_i,k)/Omega (x2 if nspinor = 1).

    Interband polarization, a.u. (e/bohr^2); complex, the imaginary part is
    kept as TDGW does for diagnosis.  Returns (ntr, 3).
    """
    return -_expectation(ops.w_k, cell_volume, nspinor, rho, ops.r_inter)


def current_operator(ops: TdOperators, dsigma, *, use_v0: bool,
                     inter_only: bool):
    """v = i[H, r] in the band basis (TDGW ``MVelocityOperator``).

    v0 = v0_inter (+ diag(v0_intra) unless ``inter_only``); unless
    ``use_v0`` the self-energy term i[dSigma, r^inter] is added and, unless
    ``inter_only``, -i[r^intra, dSigma].  Returns (3, ntr, nk, nb, nb).
    """
    nb = ops.e_qp.shape[-1]
    v = ops.v0_inter
    if not inter_only and ops.v0_intra is not None:
        v = v.at[:, :, jnp.arange(nb), jnp.arange(nb)].set(ops.v0_intra)
    ntr = dsigma.shape[0]
    v = jnp.broadcast_to(v[:, None], (3, ntr) + v.shape[1:])
    if not use_v0:
        r = ops.r_inter[:, None]
        v = v + 1j * (dsigma[None] @ r - r @ dsigma[None])
        if not inter_only and ops.intra is not None:
            v = v - 1j * intraband_commutator(ops.intra, dsigma)
    return v


def current(ops: TdOperators, rho, dsigma, *, cell_volume: float,
            nspinor: int, use_v0: bool, inter_only: bool):
    """j_i = -sum_k w_k Tr(rho_k v_i,k)/Omega (x2 if nspinor = 1), a.u.

    Returns (ntr, 3) complex (TDGW ``MCurrent``).
    """
    v = current_operator(ops, dsigma, use_v0=use_v0, inter_only=inter_only)
    return -_expectation(ops.w_k, cell_volume, nspinor, rho, v)
