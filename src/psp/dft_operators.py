"""
psp/dft_operators.py — Unified PW DFT operator kernels.

Provides fused, JIT-compiled operator kernels for the plane-wave DFT
Hamiltonian: T + V_loc + V_NL [+ V_H].  All other modules that need
DFT matrix elements or matvecs should source core functionality here.

Representations
---------------
  sparse-G : (nvec, nspinor, nG) — coefficients at valid G-vectors only
  FFT box  : (nvec, nspinor, nx, ny, nz) — dense 3-D grid (used
             internally by V_loc for the real-space multiply)

Normalization
-------------
All operators use the convention:

    <m|O|n> = sum_{s,G} conj(psi_m[s,G]) * (O psi)_n[s,G]

with no volume prefactors.  See hamiltonian_matvec.py docstring for
the derivation showing scale * deltaV * fft_norm * sqrt(1/Omega) = 1.
"""
from __future__ import annotations

import os
import argparse
import functools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import common.timing as timing


# ---------------------------------------------------------------------------
# JAX-traceable real spherical harmonics (replaces qe_real_sph_harmonics)
# ---------------------------------------------------------------------------

def ylmr2_jax(lmax: int, K_cart: jax.Array) -> jax.Array:
    """QE-convention real spherical harmonics, vectorised in JAX.

    Replicates QE's ``ylmr2`` subroutine: associated Legendre
    recurrence in cos(theta) with cos(m*phi)/sin(m*phi) azimuthal
    part and 4-pi normalisation.

    Autodiff-safe: uses soft regularisation (eps = 1e-30) to avoid
    NaN gradients from sqrt(0), 1/0, and arctan2(0,0) at K=0.

    Parameters
    ----------
    lmax : int — maximum angular momentum (static)
    K_cart : (nG, 3) — Cartesian K = k+G vectors (traced)

    Returns
    -------
    ylm : (nG, (lmax+1)**2) — real harmonics ordered as QE:
          l=0: [Y_00]
          l=1: [Y_10, Y_11c, Y_11s]
          l=2: [Y_20, Y_21c, Y_21s, Y_22c, Y_22s]
          ...
    """
    _EPS2 = 1e-60  # epsilon^2 — negligible vs values, prevents grad NaN
    fpi = 4.0 * jnp.pi
    gx = K_cart[:, 0]
    gy = K_cart[:, 1]
    gz = K_cart[:, 2]
    gmod_sq = gx**2 + gy**2 + gz**2
    # Regularise |K|: add eps^2 to avoid grad(sqrt(0)) = inf
    gmod = jnp.sqrt(gmod_sq + _EPS2)
    cost = gz / gmod
    # sent must vanish at |K|=0 (all l>0 harmonics are zero there).
    # Physical: sin(theta) = sqrt(gx²+gy²)/|g|. Use this form directly
    # instead of sqrt(1-cos²θ) which gives 1 at the origin.
    gperp = jnp.sqrt(gx**2 + gy**2 + _EPS2)
    sent = gperp / gmod
    # phi: safe arctan2 — mask inputs at |K|=0 to give phi=0
    phi = jnp.arctan2(gy * gmod, gx * gmod + _EPS2)

    nG = K_cart.shape[0]
    lmax2 = (lmax + 1) ** 2
    # Build the un-normalised ylm array: (nG, lmax2)
    # Stored flat; arr[l*l + 2*m] for even/odd indexing per QE convention.
    arr = jnp.zeros((nG, lmax2))
    arr = arr.at[:, 0].set(1.0)
    if lmax >= 1:
        arr = arr.at[:, 1].set(cost)            # l=1, m=0
        arr = arr.at[:, 3].set(-sent / jnp.sqrt(2.0))  # l=1, m=1 (sin slot)

    for l in range(2, lmax + 1):
        for m in range(0, l - 1):
            lm = l * l + 2 * m
            lm1 = (l - 1) * (l - 1) + 2 * m
            lm2 = (l - 2) * (l - 2) + 2 * m
            denom = jnp.sqrt(float(l * l - m * m))
            arr = arr.at[:, lm].set(
                cost * (2 * l - 1) * arr[:, lm1] / denom
                - jnp.sqrt(float((l - 1)**2 - m * m)) * arr[:, lm2] / denom
            )
        # m = l-1
        lm1 = l * l + 2 * (l - 1)
        lm2 = (l - 1) * (l - 1) + 2 * (l - 1)
        arr = arr.at[:, lm1].set(
            cost * jnp.sqrt(float(2 * l - 1)) * arr[:, lm2]
        )
        # m = l
        lm = l * l + 2 * l
        arr = arr.at[:, lm].set(
            -(jnp.sqrt(float(2 * l - 1)) / jnp.sqrt(float(2 * l)))
            * sent * arr[:, lm2]
        )

    # Normalise and apply azimuthal cos/sin
    arr = arr.at[:, 0].divide(jnp.sqrt(fpi))
    for l in range(1, lmax + 1):
        c = jnp.sqrt((2 * l + 1) / fpi)
        lm0 = l * l
        arr = arr.at[:, lm0].multiply(c)
        for m in range(1, l + 1):
            cos_idx = l * l + 2 * m - 1
            sin_idx = l * l + 2 * m
            val = c * jnp.sqrt(2.0) * arr[:, sin_idx]
            arr = arr.at[:, cos_idx].set(val * jnp.cos(m * phi))
            arr = arr.at[:, sin_idx].set(val * jnp.sin(m * phi))

    return arr


def real_sph_harmonics_jax(l: int, K_cart: jax.Array) -> jax.Array:
    """JAX-traceable QE real spherical harmonics for a single l.

    Parameters
    ----------
    l : int — angular momentum (static)
    K_cart : (nG, 3) — Cartesian K vectors (traced)

    Returns
    -------
    Y : (2l+1, nG) — real harmonics for this l only
    """
    ylm_all = ylmr2_jax(l, K_cart)
    start = l * l
    end = (l + 1) * (l + 1)
    return ylm_all[:, start:end].T


# ---------------------------------------------------------------------------
# JAX-traceable B-spline evaluation (replaces scipy splev for autodiff)
# ---------------------------------------------------------------------------

def extract_spline_data(spl) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract (knots, coeffs, degree) from a scipy spline object.

    Works with InterpolatedUnivariateSpline and any FITPACK-backed
    scipy spline.  The returned arrays can be passed to ``splev_jax``.
    """
    t, c, k = spl._eval_args
    return np.asarray(t, dtype=np.float64), np.asarray(c, dtype=np.float64), int(k)


def _fpbspl_jax(t, x, l, k):
    """FITPACK fpbspl: k+1 nonzero B-spline basis values at scalar x.

    Implements the de Boor-Cox recurrence bottom-up from degree 0 to k.
    All loops are over the static degree k (≤5) and unrolled by JIT.
    """
    h = jnp.zeros(k + 1)
    h = h.at[0].set(1.0)
    for j in range(1, k + 1):
        hh = h
        h = h.at[0].set(0.0)
        for i in range(j):
            li = l + i + 1
            lj = l + i + 1 - j
            f = hh[i] / (t[li] - t[lj])
            h = h.at[i].add(f * (t[li] - x))
            h = h.at[i + 1].set(f * (x - t[lj]))
    return h


def _splev_scalar(x, t, c, k):
    """Evaluate B-spline at a single scalar x."""
    n = t.shape[0]
    l = jnp.searchsorted(t, x, side='right') - 1
    l = jnp.clip(l, k, n - k - 2)
    h = _fpbspl_jax(t, x, l, k)
    c_window = jax.lax.dynamic_slice(c, (l - k,), (k + 1,))
    return jnp.dot(c_window, h)


def splev_jax(x, t, c, k=3):
    """JAX-traceable B-spline evaluation, vectorised over x.

    Drop-in replacement for ``scipy_spline(x)`` that is compatible
    with ``jax.grad``, ``jax.jacfwd``, and ``jax.jit``.

    Parameters
    ----------
    x : (nG,) evaluation points — may be JAX-traced
    t : (n,) knot vector — concrete (from ``extract_spline_data``)
    c : (n,) coefficients — concrete
    k : int, spline degree (default 3, must be static)

    Returns
    -------
    (nG,) spline values, same dtype as x
    """
    return jax.vmap(lambda xi: _splev_scalar(xi, t, c, k))(x)
from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox

from psp.get_DFT_mtxels import (
    read_cohsex_input,
    generate_gvectors_k,
    load_pseudopotentials,
    build_atom_pp_assignments,
    compute_valence_density,
    compute_hartree_potential_real,
)
from psp.build_projectors_qe import (
    build_local_ionic_potential_on_G_total,
    qe_real_sph_harmonics,
)
from psp.projector_pipeline import build_vnl_plan


# ---------------------------------------------------------------------------
# Autodiff-compatible V_NL: differentiate through k
# ---------------------------------------------------------------------------

@dataclass
class VNLChannelData:
    """Pre-extracted k-independent data for one (species, l) VNL channel.

    All fields are plain arrays suitable for passing into JIT/autodiff.
    """
    tau: jax.Array              # (natoms, 3) atomic positions (crystal)
    prefactor: float            # 4 pi / sqrt(Omega)
    l: int                      # angular momentum
    nbeta: int                  # number of beta projectors for this l
    spline_t: list[jax.Array]   # [knots_array per beta]  — F_l(q) spline
    spline_c: list[jax.Array]   # [coeffs_array per beta]
    spline_k: int               # spline degree
    # G_l(q) = F_l(q)/q^l splines — smooth at q=0, used by solid-harmonic
    # autodiff path to avoid catastrophic cancellation in F_l(q)/q^l.
    reduced_spline_t: list[jax.Array]   # [knots per beta]
    reduced_spline_c: list[jax.Array]   # [coeffs per beta]
    E: jax.Array                # (nspinor, nspinor, R, R) D-matrix


def _build_reduced_spline(spl, l: int):
    """Build G_l(q) = F_l(q)/q^l spline from an F_l(q) spline.

    G_l is smooth at q=0 (since F_l ~ q^l for small q).
    The q=0 value is computed via L'Hopital / Taylor of the
    underlying radial integral.
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    t_F, c_F, k_F = spl._eval_args
    q_max = float(t_F[-1])
    # Evaluate F_l on a fine grid.  Use extra density near q=0 where
    # G_l = F_l/q^l needs good derivative accuracy for Gamma-point
    # autodiff.  A uniform grid with many points works; the spline
    # construction is a one-time setup cost.
    n_pts = max(50000, len(t_F) * 10)
    q_grid = np.linspace(0, q_max, n_pts)
    F_vals = np.asarray(spl(q_grid), dtype=np.float64)
    if l == 0:
        G_vals = F_vals
    else:
        G_vals = np.empty_like(F_vals)
        G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
        # q=0: use limit.  F_l(q) ~ a*q^l for small q, so G_l(0) = a.
        # Approximate: G_l(0) ≈ F_l(dq)/dq^l for smallest nonzero dq.
        G_vals[0] = F_vals[1] / q_grid[1] ** l
    spl_G = InterpolatedUnivariateSpline(q_grid, G_vals, k=min(k_F, 3))
    return extract_spline_data(spl_G)


def extract_vnl_channel_data(
    plan: dict,
    nspinor: int = 2,
) -> list[VNLChannelData]:
    """Extract all VNL channel data from a plan into autodiff-ready form.

    For each (l, beta), also precomputes the *reduced* radial spline
    G_l(q) = F_l(q)/q^l which is smooth at q=0 and avoids catastrophic
    cancellation when autodiff-ed through the solid-harmonic path.
    """
    channels = []
    for _key, sp in plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=np.float64)
        if tau.size == 0:
            continue
        if tau.ndim == 1:
            tau = tau.reshape(1, 3)
        pref = float(sp['prefactor'])
        splines = sp['splines']

        for l_key, info in sp['l_channels'].items():
            l = int(l_key)
            E_np = info['E']
            if E_np is None:
                continue
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue

            spl_t_list, spl_c_list = [], []
            red_t_list, red_c_list = [], []
            for bid in beta_ids:
                spl = splines[(l, int(bid))]
                t, c, k = extract_spline_data(spl)
                spl_t_list.append(jnp.asarray(t))
                spl_c_list.append(jnp.asarray(c))
                # Reduced spline G_l = F_l / q^l
                t_r, c_r, _ = _build_reduced_spline(spl, l)
                red_t_list.append(jnp.asarray(t_r))
                red_c_list.append(jnp.asarray(c_r))

            E_j = jnp.asarray(E_np, dtype=jnp.complex128)[:nspinor, :nspinor]
            channels.append(VNLChannelData(
                tau=jnp.asarray(tau, dtype=jnp.float64),
                prefactor=pref,
                l=l,
                nbeta=len(beta_ids),
                spline_t=spl_t_list,
                spline_c=spl_c_list,
                spline_k=int(k),
                reduced_spline_t=red_t_list,
                reduced_spline_c=red_c_list,
                E=E_j,
            ))
    return channels


def _solid_harmonics_jax(l: int, K_cart: jax.Array) -> jax.Array:
    """Solid harmonics S_lm(x,y,z) = r^l Y_lm(r-hat) in QE convention.

    Pure polynomials in K_cart — no trig, no sqrt, no singularities.
    Perfectly smooth everywhere including K=0.  Autodiff-friendly.

    Returns (2l+1, nG) matching the QE ordering [m=0, 1c, 1s, 2c, 2s, ...].
    Supports l = 0, 1, 2, 3.
    """
    x = K_cart[:, 0]
    y = K_cart[:, 1]
    z = K_cart[:, 2]
    fpi = 4.0 * jnp.pi

    if l == 0:
        return jnp.stack([jnp.ones_like(x) / jnp.sqrt(fpi)], axis=0)

    elif l == 1:
        c = jnp.sqrt(3.0 / fpi)
        return jnp.stack([c * z, -c * x, -c * y], axis=0)

    elif l == 2:
        c2 = jnp.sqrt(5.0 / fpi)
        c2s3 = c2 * jnp.sqrt(3.0)
        return jnp.stack([
            c2 / 2.0 * (2 * z**2 - x**2 - y**2),   # m=0
            -c2s3 * x * z,                            # m=1 cos
            -c2s3 * y * z,                            # m=1 sin
            c2s3 / 2.0 * (x**2 - y**2),              # m=2 cos
            c2s3 * x * y,                             # m=2 sin
        ], axis=0)

    elif l == 3:
        c3 = jnp.sqrt(7.0 / fpi)
        s3 = jnp.sqrt(3.0)
        s5 = jnp.sqrt(5.0)
        s6 = jnp.sqrt(6.0)
        s10 = jnp.sqrt(10.0)
        s15 = jnp.sqrt(15.0)
        r2 = x**2 + y**2 + z**2
        return jnp.stack([
            c3 / 2.0 * z * (2*z**2 - 3*x**2 - 3*y**2),       # m=0
            -c3*s6/4.0 * x * (4*z**2 - x**2 - y**2),          # m=1c
            -c3*s6/4.0 * y * (4*z**2 - x**2 - y**2),          # m=1s
            c3*s15/2.0 * z * (x**2 - y**2),                    # m=2c
            c3*s15 * x * y * z,                                 # m=2s
            -c3*s10/4.0 * x * (x**2 - 3*y**2),                # m=3c
            -c3*s10/4.0 * y * (3*x**2 - y**2),                # m=3s
        ], axis=0)

    else:
        raise NotImplementedError(f"solid_harmonics_jax: l={l} > 3 not implemented")


def _build_Z_channel_jax(
    K_crys: jax.Array,
    K_cart: jax.Array,
    ch: VNLChannelData,
) -> jax.Array:
    """Build projector matrix Z for one channel.  Pure JAX, k-traceable.

    Uses the solid-harmonic factorisation:

        Z = pref (i)^l  [F_l(q)/q^l]  S_lm(K_x,K_y,K_z)  exp(-2pi i K.tau)

    where S_lm are solid harmonics (Cartesian polynomials) and
    F_l(q)/q^l is smooth at q=0 (since F_l(q) ~ q^l for small q).
    This avoids all angle-based singularities at K=0.

    Parameters
    ----------
    K_crys : (nG, 3) — K = k + G in crystal coords (traced through k)
    K_cart : (nG, 3) — K in Cartesian coords (traced through k)
    ch : VNLChannelData

    Returns
    -------
    Z : (natoms, R, nG) complex128, where R = nbeta * (2l+1)
    """
    nG = K_crys.shape[0]
    l = ch.l
    pref = ch.prefactor

    # ── radial × angular via custom JVP ───────────────────────────
    # G_l(|K|) * S_lm(K) is smooth at K=0, but naive autodiff through
    # sqrt(K��K) produces catastrophic cancellation.  We provide the
    # analytically stable JVP via @custom_jvp.
    radial_times_S = _radial_times_solid_harm(K_cart, ch, pref)
    # radial_times_S: (nbeta, msize, nG) complex128

    # Atomic structure factors
    phase = jnp.exp(
        -2j * jnp.pi * (K_crys @ ch.tau.T)
    ).T                                           # (natoms, nG)

    # Z = phase * radial_times_S
    Z_atoms = phase[:, None, None, :] * radial_times_S[None, ...]
    R = ch.nbeta * (2 * l + 1)
    return Z_atoms.reshape(ch.tau.shape[0], R, nG)


def _radial_times_solid_harm(K_cart, ch, pref):
    """Compute pref * (i)^l * G_l(|K|) * S_lm(K_cart).

    Delegates to ``_radial_times_solid_harm_impl`` which has a
    ``@custom_jvp`` providing an analytically stable tangent at K=0.
    """
    # Pack spline data as a flat tuple of arrays for the custom_jvp
    # boundary (custom_jvp requires array-typed primals).
    l = ch.l
    nbeta = ch.nbeta
    return _radial_times_solid_harm_impl(
        K_cart,
        tuple(ch.reduced_spline_t),
        tuple(ch.reduced_spline_c),
        pref, l, nbeta, ch.spline_k,
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def _radial_times_solid_harm_impl(K_cart, spl_t, spl_c, pref, l, nbeta, spl_k):
    _EPS2 = 1e-60
    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)
    G_list = [splev_jax(q, spl_t[ib], spl_c[ib], spl_k) for ib in range(nbeta)]
    G_bG = jnp.stack(G_list, axis=0)
    S = _solid_harmonics_jax(l, K_cart)
    return pref * (1j) ** l * G_bG[:, None, :] * S[None, :, :]


@_radial_times_solid_harm_impl.defjvp
def _radial_times_solid_harm_jvp(
    spl_t, spl_c, pref, l, nbeta, spl_k,
    primals, tangents,
):
    """Stable JVP for G_l(|K|) * S_lm(K).

    d/dK [G_l(q) S_lm(K)] = G'_l(q) (K.dK)/q S_lm + G_l(q) dS_lm

    At K=0: (K.dK)/q = 0 (stable: numerator is zero, denominator is
    eps).  dS_lm/dK_i is a polynomial derivative (constant for l=1,
    linear for l=2).  No inf*0, no cancellation.
    """
    (K_cart,) = primals
    (dK_cart,) = tangents
    _EPS2 = 1e-60

    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)

    # G_l(q) and G'_l(q) via spline
    G_list, Gp_list = [], []
    for ib in range(nbeta):
        G_list.append(splev_jax(q, spl_t[ib], spl_c[ib], spl_k))
        Gp_list.append(jax.vmap(
            jax.grad(lambda qi, t=spl_t[ib], c=spl_c[ib]: _splev_scalar(qi, t, c, spl_k))
        )(q))
    G_bG = jnp.stack(G_list, axis=0)
    Gp_bG = jnp.stack(Gp_list, axis=0)
    S = _solid_harmonics_jax(l, K_cart)

    primal_out = pref * (1j) ** l * G_bG[:, None, :] * S[None, :, :]

    # Radial tangent: G'(q) * (K.dK)/q — numerically stable at K=0
    K_dot_dK = jnp.sum(K_cart * dK_cart, axis=1)
    dq = K_dot_dK / q            # = (K.dK)/sqrt(K²+eps²) → 0 at K=0
    dG = Gp_bG * dq[None, :]

    # Angular tangent: dS via JVP of the polynomial
    _, dS = jax.jvp(
        lambda K: _solid_harmonics_jax(l, K), (K_cart,), (dK_cart,)
    )

    tangent_out = pref * (1j) ** l * (
        dG[:, None, :] * S[None, :, :] + G_bG[:, None, :] * dS[None, :, :]
    )
    return primal_out, tangent_out


def vnl_matrix_at_k(
    k_crys: jax.Array,
    psi_G: jax.Array,
    G_int: jax.Array,
    B: jax.Array,
    channels: list[VNLChannelData],
) -> jax.Array:
    """V_NL matrix elements as a pure function of k.

    Fully JAX-traceable — ``jax.jacfwd`` w.r.t. ``k_crys`` gives the
    nonlocal velocity matrix elements (3, nb, nb).

    Parameters
    ----------
    k_crys : (3,) — k-point in crystal coordinates (**traced**)
    psi_G : (nb, nspinor, nG) — wavefunction coefficients at valid G
    G_int : (nG, 3) — integer G-vectors (crystal)
    B : (3, 3) — crystal-to-Cartesian matrix (blat * bvec^T)
    channels : list[VNLChannelData] from ``extract_vnl_channel_data``

    Returns
    -------
    V_NL_mn : (nb, nb) complex128
    """
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]  # (nG, 3)
    K_cart = K_crys @ B                                     # (nG, 3)

    nb = psi_G.shape[0]
    V_NL = jnp.zeros((nb, nb), dtype=jnp.complex128)

    for ch in channels:
        Z = _build_Z_channel_jax(K_crys, K_cart, ch)  # (natoms, R, nG)
        # KB: project, apply D, contract
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', ch.E, proj, optimize=True)
        vnl_G = jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)
        V_NL = V_NL + jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G), vnl_G, optimize=True,
        )

    return V_NL


def vnl_velocity_autodiff(
    k_crys: jax.Array,
    psi_G: jax.Array,
    G_int: jax.Array,
    B: jax.Array,
    channels: list[VNLChannelData],
) -> jax.Array:
    """Nonlocal velocity matrix: d V_NL / d K_cart_i via forward-mode autodiff.

    Returns (3, nb, nb) — Cartesian (d/dK_cart_i) <m|V_NL|n>.

    This is the nonlocal contribution to the velocity operator
    v_i = dH/dk_i in Ry atomic units.  The caller adds the local
    momentum part 2*(k+G)_i to get the full velocity / dipole.

    Note: ``compute_V_NL_velocity_k`` returns the *negative* of this
    quantity and the dipole code negates it again.  This function
    returns +dV_NL/dK_cart directly.
    """
    # Jacobian in crystal k-coordinates: shape (nb, nb, 3)
    jac_crys = jax.jacfwd(vnl_matrix_at_k)(
        k_crys, psi_G, G_int, B, channels,
    )
    # Transform to Cartesian:  dV/dK_cart_j = (B^-1)_{ja} dV/dk_crys_a
    Binv = jnp.linalg.inv(B)
    jac_cart = jnp.einsum('mna,ja->mnj', jac_crys, Binv)
    return jnp.transpose(jac_cart, (2, 0, 1))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OperatorSetup:
    """K-point-independent operator data.  Built once, shared across k."""
    V_r: jax.Array                  # (nx, ny, nz) V_loc [+V_H] [Ry]
    vnl_plan: dict
    assignments: list
    species_payload: list
    bdot: np.ndarray                # (3,3)
    B: np.ndarray                   # crystal-to-Cartesian K = k+G
    cell_volume: float
    fft_grid: tuple[int, int, int]


@dataclass
class KPointOperators:
    """Per-k precomputed operator data."""
    T_diag: jax.Array               # (nG,) kinetic diagonal [Ry]
    Gx: jax.Array                   # (nG,) int32
    Gy: jax.Array                   # (nG,) int32
    Gz: jax.Array                   # (nG,) int32
    V_r: jax.Array                  # shared ref to OperatorSetup.V_r
    vnl_projectors: list[tuple[jax.Array, jax.Array]]  # [(Z, E), ...]
    nG: int
    fft_grid: tuple[int, int, int]


# ---------------------------------------------------------------------------
# Setup builders  (thin wrappers — the heavy lifting is in existing code)
# ---------------------------------------------------------------------------

def build_operator_setup(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    pseudos: dict,
    *,
    include_hartree: bool = False,
    global_psi_G: jax.Array | None = None,
    truncation_2d: bool = True,
) -> OperatorSetup:
    """Precompute k-independent data: V_loc, VNL plan, optional V_H."""
    atom_pos = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)
    atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
    assignments = build_atom_pp_assignments(atom_pos, atom_types, pseudos)

    tmp: dict[int, dict] = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        entry = tmp.setdefault(key, {"pseudo": ap.pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    species_payload = [
        (e["pseudo"],
         np.asarray(e["positions"], dtype=float)
         if e["positions"] else np.zeros((0, 3), dtype=float))
        for e in tmp.values()
    ]

    V_loc_r = build_local_ionic_potential_on_G_total(
        assignments=[
            {"pseudo": ap.pseudo,
             "position": np.asarray(ap.position, dtype=float)}
            for ap in assignments
        ],
        species_groups=[
            (sp[0],
             (np.asarray(sp[1], dtype=float)
              if np.asarray(sp[1]).size > 0
              else np.zeros((0, 3), dtype=float)))
            for sp in species_payload
        ],
        fft_grid=tuple(int(x) for x in meta.fft_grid),
        bdot=np.asarray(wfn.bdot, dtype=float),
        cell_volume=float(wfn.cell_volume),
        bvec=np.asarray(wfn.bvec, dtype=float),
        blat=float(wfn.blat),
        truncation_2d=truncation_2d,
    )
    V_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    if include_hartree:
        if global_psi_G is None:
            raise ValueError("global_psi_G required for include_hartree")
        rho_val = compute_valence_density(global_psi_G, sym, wfn)
        V_H_r = compute_hartree_potential_real(
            rho_val,
            jnp.asarray(wfn.bdot, dtype=jnp.float64),
            bvec=jnp.asarray(wfn.bvec, dtype=jnp.float64),
            blat=float(wfn.blat),
            truncation_2d=False,
        )
        V_r = V_r + jnp.asarray(V_H_r, dtype=jnp.float64)

    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    q_max = 0.0
    for ik in range(sym.nk_tot):
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
        K_cart = (np.asarray(Gk_crys, dtype=float) + kvec[None, :]) @ B
        qk = np.sqrt(np.sum(K_cart**2, axis=1))
        if qk.size:
            q_max = max(q_max, float(np.max(qk)))

    vnl_plan = build_vnl_plan(
        pseudos, assignments, float(wfn.cell_volume), float(q_max)
    )

    return OperatorSetup(
        V_r=V_r,
        vnl_plan=vnl_plan,
        assignments=assignments,
        species_payload=species_payload,
        bdot=np.asarray(wfn.bdot, dtype=float),
        B=B,
        cell_volume=float(wfn.cell_volume),
        fft_grid=tuple(int(x) for x in meta.fft_grid),
    )


def build_kpoint_operators(
    k_idx: int,
    setup: OperatorSetup,
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    nspinor: int | None = None,
) -> KPointOperators:
    """Build per-k data: kinetic diagonal and VNL projectors Z."""
    if nspinor is None:
        nspinor = int(meta.nspinor)

    Gk_crys, _ = generate_gvectors_k(k_idx, sym, wfn, meta)
    Gk_np = np.asarray(Gk_crys, dtype=int)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    nG = Gk_np.shape[0]

    Gx = jnp.asarray(Gk_np[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_np[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_np[:, 2], dtype=jnp.int32)

    G_float = np.asarray(Gk_np, dtype=float)
    K_crys = G_float + kvec[None, :]
    T_diag = jnp.asarray(
        np.einsum('gi,ij,gj->g', K_crys, setup.bdot, K_crys),
        dtype=jnp.float64,
    )

    K_cart = K_crys @ setup.B
    K_norm = np.sqrt(np.sum(K_cart**2, axis=1))
    K_crys_j = jnp.asarray(K_crys, dtype=jnp.float64)

    vnl_projectors: list[tuple[jax.Array, jax.Array]] = []
    Y_cache: dict[int, jax.Array] = {}

    for _key, sp in setup.vnl_plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=float)
        if tau.size == 0:
            continue
        if tau.ndim == 1:
            tau = tau.reshape(1, 3)
        natoms = tau.shape[0]
        pref = float(sp['prefactor'])
        splines = sp['splines']

        tau_j = jnp.asarray(tau, dtype=jnp.float64)
        phase = jnp.exp(
            -2j * jnp.pi * (K_crys_j @ tau_j.T)
        ).T

        for l_key, info in sp['l_channels'].items():
            l = int(l_key)
            E_np = info['E']
            if E_np is None:
                continue
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue
            nbeta = len(beta_ids)
            msize = 2 * l + 1

            F_bG = np.stack(
                [splines[(l, int(bid))](K_norm) for bid in beta_ids],
                axis=0,
            )
            radial = jnp.asarray(
                pref * (1j) ** l * F_bG, dtype=jnp.complex128,
            )

            if l not in Y_cache:
                Y_cache[l] = jnp.asarray(
                    qe_real_sph_harmonics(l, K_cart), dtype=jnp.complex128,
                )
            Y = Y_cache[l]

            Z_bmg = radial[:, None, :] * Y[None, :, :]
            Z_atoms = phase[:, None, None, :] * Z_bmg[None, ...]
            R = nbeta * msize
            Z_flat = Z_atoms.reshape(natoms, R, nG)

            E_j = info.get('E_j')
            if E_j is None:
                E_j = jnp.asarray(E_np, dtype=jnp.complex128)
            E_j = E_j[:nspinor, :nspinor]

            vnl_projectors.append((Z_flat, E_j))

    return KPointOperators(
        T_diag=T_diag,
        Gx=Gx, Gy=Gy, Gz=Gz,
        V_r=setup.V_r,
        vnl_projectors=vnl_projectors,
        nG=nG,
        fft_grid=setup.fft_grid,
    )


# ---------------------------------------------------------------------------
# Core fused kernels
# ---------------------------------------------------------------------------

@jax.jit
def apply_H_k(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_ZE):
    """Fused H|psi>: FFT-box in, sparse-G out.  Single JIT dispatch.

    Parameters
    ----------
    psi_box : (nvec, nspinor, nx, ny, nz) — trial vectors in FFT box
    T_diag  : (nG,) — kinetic diagonal
    V_r     : (nx, ny, nz) — real-space local potential
    Gx,Gy,Gz : (nG,) int32 — G-vector FFT-box indices
    vnl_ZE  : tuple of (Z, E) per VNL channel

    Returns
    -------
    H_psi_G : (nvec, nspinor, nG) — sparse-G
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]
    H_G = T_diag[None, None, :] * psi_G
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    H_G = H_G + jnp.fft.fftn(
        psi_r * V_r, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz]
    for Z, E in vnl_ZE:
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', E, proj, optimize=True)
        H_G = H_G + jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)
    return H_G


@jax.jit
def build_matrix_k(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_ZE):
    """Fused H matrix elements: <m|H|n> for all bands at one k-point.

    Same physics as apply_H_k, but contracts to (nb, nb) instead of
    returning sparse-G.  Single JIT dispatch.

    Parameters
    ----------
    psi_box : (nb, nspinor, nx, ny, nz) — wavefunctions
    (other args: same as apply_H_k)

    Returns
    -------
    H_mn : (nb, nb) complex128
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]            # (nb, ns, nG)

    # T
    H_mn = jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G),
        T_diag[None, None, :] * psi_G, optimize=True,
    )

    # V_loc
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    Vpsi_G = jnp.fft.fftn(
        psi_r * V_r, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz]
    H_mn = H_mn + jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G), Vpsi_G, optimize=True,
    )

    # V_NL
    for Z, E in vnl_ZE:
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', E, proj, optimize=True)
        vnl_G = jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)
        H_mn = H_mn + jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G), vnl_G, optimize=True,
        )

    return H_mn


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def apply(psi_box: jax.Array, kops: KPointOperators) -> jax.Array:
    """Apply H|psi> at one k-point.  FFT-box in, sparse-G out."""
    return apply_H_k(
        psi_box, kops.T_diag, kops.V_r,
        kops.Gx, kops.Gy, kops.Gz,
        tuple(kops.vnl_projectors),
    )


def matrix(psi_box: jax.Array, kops: KPointOperators) -> jax.Array:
    """Build H_mn at one k-point.  Returns (nb, nb)."""
    return build_matrix_k(
        psi_box, kops.T_diag, kops.V_r,
        kops.Gx, kops.Gy, kops.Gz,
        tuple(kops.vnl_projectors),
    )


# ---------------------------------------------------------------------------
# Batched kin_ion writer (replaces gw/kin_ion_io_chunked logic)
# ---------------------------------------------------------------------------

def compute_kin_ion_all(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    setup: OperatorSetup,
    nb: int | None = None,
) -> np.ndarray:
    """Compute kin+ion matrix for all k-points.  Returns (nk, nb, nb).

    Distributes k-points across available devices (1 k per device),
    overlapping host-side prep with device execution.
    """
    if nb is None:
        nb = int(meta.b_id_4)
    nk = sym.nk_tot
    nspinor = int(meta.nspinor)
    devices = jax.devices()
    n_dev = len(devices)
    kin_ion = np.zeros((nk, nb, nb), dtype=np.complex128)

    if n_dev == 1:
        # Single device: simple sequential
        for ik in range(nk):
            with timing.section(f"dft_operators.kin_ion_k{ik}"):
                kops = build_kpoint_operators(ik, setup, wfn, sym, meta,
                                              nspinor=nspinor)
                wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
                H_k = matrix(wfn_k, kops)
                kin_ion[ik] = np.asarray(H_k)
                del wfn_k
        return kin_ion

    # Multi-device: pre-place all data, then fire all kernels.
    # Phase 1: build operator data + load wavefunctions + transfer
    #           to target devices (round-robin).
    # Phase 2: dispatch all kernels (async, overlapping across devices).
    # Phase 3: collect results.

    with timing.section("dft_operators.kin_ion.prep"):
        placed_args = []
        for ik in range(nk):
            dev = devices[ik % n_dev]
            kops = build_kpoint_operators(ik, setup, wfn, sym, meta,
                                          nspinor=nspinor)
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
            placed_args.append((
                jax.device_put(wfn_k, dev),
                jax.device_put(kops.T_diag, dev),
                jax.device_put(kops.V_r, dev),
                jax.device_put(kops.Gx, dev),
                jax.device_put(kops.Gy, dev),
                jax.device_put(kops.Gz, dev),
                tuple(
                    (jax.device_put(Z, dev), jax.device_put(E, dev))
                    for Z, E in kops.vnl_projectors
                ),
            ))
            del wfn_k
        jax.block_until_ready([a[0] for a in placed_args])

    with timing.section("dft_operators.kin_ion.compute"):
        futures = [build_matrix_k(*args) for args in placed_args]
        # Wait for all kernels to finish before any D2H transfer
        jax.block_until_ready(futures)
    with timing.section("dft_operators.kin_ion.collect"):
        for ik, H_k in enumerate(futures):
            kin_ion[ik] = np.asarray(H_k)

    return kin_ion


# ---------------------------------------------------------------------------
# CLI: validate + benchmark
# ---------------------------------------------------------------------------

def main(argv=None):
    argp = argparse.ArgumentParser(
        description="dft_operators — validate and benchmark",
    )
    argp.add_argument("-i", "--input", required=True, help="cohsex.in path")
    argp.add_argument("-n", "--nb", type=int, default=None)
    args = argp.parse_args(argv)

    timing.reset()
    input_dir = os.path.dirname(os.path.abspath(args.input))
    params = read_cohsex_input(args.input)
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(input_dir, wfn_path)

    nband = int(params.get("nband", 80))
    nval = int(params.get("nval", 26))
    ncond = int(params.get("ncond", 54))
    bispinor = bool(params.get("bispinor", False))
    nb = int(args.nb) if args.nb else nband

    print("== dft_operators: validate & benchmark ==")
    with timing.section("load"):
        wfn = WFNReader(wfn_path)
        sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, nval, ncond, nb, 0, bispinor)
    print(f"  k={sym.nk_tot}, bands={nb}, nspinor={meta.nspinor}, "
          f"grid={meta.fft_grid}, devices={jax.device_count()}")

    pseudos = load_pseudopotentials(input_dir)

    with timing.section("build_setup"):
        setup = build_operator_setup(wfn, sym, meta, pseudos)

    # -- validate build_matrix_k against old code ---------------------------
    from psp.get_DFT_mtxels import compute_kinetic_k, compute_local_V_k
    from psp.projector_pipeline import compute_V_NL_k_minimal

    print("\nValidating build_matrix_k against old code...")
    all_pass = True
    for ik in range(sym.nk_tot):
        kops = build_kpoint_operators(ik, setup, wfn, sym, meta)
        wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)

        # New fused path
        H_new = matrix(wfn_k, kops)

        # Old path
        Gk_crys, kpoint = generate_gvectors_k(ik, sym, wfn, meta)
        bdot = np.asarray(wfn.bdot, dtype=float)
        kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
        T_old = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot)
        V_old = compute_local_V_k(wfn_k, Gk_crys, setup.V_r, wfn.cell_volume)
        K_crys = np.asarray(Gk_crys, dtype=float) + kvec[None, :]
        K_cart = K_crys @ setup.B
        VNL_old = compute_V_NL_k_minimal(
            wfn_k, Gk_crys, K_crys, K_cart,
            setup.vnl_plan, float(wfn.cell_volume),
        )
        H_old = jnp.asarray(T_old + V_old + VNL_old)

        err = float(jnp.max(jnp.abs(H_new - H_old)))
        ok = err < 1e-8
        all_pass = all_pass and ok
        print(f"  k={ik}: {'PASS' if ok else 'FAIL'}  max|err|={err:.2e}")

    # -- benchmark: fused build_matrix_k ------------------------------------
    print("\nBenchmark: fused build_matrix_k...")
    kops = build_kpoint_operators(0, setup, wfn, sym, meta)
    wfn_k = load_kpoint_fftbox(wfn, sym, meta, 0, nb)
    _ = matrix(wfn_k, kops); jax.block_until_ready(_)

    import time
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        H = matrix(wfn_k, kops); jax.block_until_ready(H)
    dt_fused = (time.perf_counter() - t0) / N

    # Old separate-dispatch path
    Gk_crys, kpoint = generate_gvectors_k(0, sym, wfn, meta)
    bdot = np.asarray(wfn.bdot, dtype=float)
    kvec = np.asarray(sym.unfolded_kpts[0], dtype=float)
    _ = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot); jax.block_until_ready(_)
    t0 = time.perf_counter()
    for _ in range(N):
        T = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot)
        V = compute_local_V_k(wfn_k, Gk_crys, setup.V_r, wfn.cell_volume)
        K_crys = np.asarray(Gk_crys, dtype=float) + kvec[None, :]
        K_cart = K_crys @ setup.B
        VNL = compute_V_NL_k_minimal(
            wfn_k, Gk_crys, K_crys, K_cart,
            setup.vnl_plan, float(wfn.cell_volume),
        )
        H = T + V + VNL
        jax.block_until_ready(H)
    dt_old = (time.perf_counter() - t0) / N

    print(f"  Fused build_matrix_k: {dt_fused*1e3:.2f} ms/k")
    print(f"  Old separate calls:   {dt_old*1e3:.2f} ms/k")
    print(f"  Speedup: {dt_old/dt_fused:.1f}x")

    # -- benchmark: full kin_ion computation --------------------------------
    print(f"\nBenchmark: compute_kin_ion_all ({sym.nk_tot} k-points)...")
    with timing.section("kin_ion_all"):
        kin_ion = compute_kin_ion_all(wfn, sym, meta, setup, nb)
    print(f"  Shape: {kin_ion.shape}")

    timing.report(title="\n--- Timing (seconds) ---")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
