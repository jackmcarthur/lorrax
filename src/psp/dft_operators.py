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
    """JAX-traceable B-spline evaluation, batched over x.

    Drop-in replacement for ``scipy_spline(x)`` that is compatible
    with ``jax.grad``, ``jax.jacfwd``, and ``jax.jit``.

    Uses a fully vectorised implementation (no vmap) that processes
    all evaluation points simultaneously.  Much faster than the
    per-scalar vmap under ``jacfwd``.

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
    n = t.shape[0]
    # Find knot intervals: l in [k, n-k-2]
    l = jnp.searchsorted(t, x, side='right') - 1      # (nG,)
    l = jnp.clip(l, k, n - k - 2)

    # Batched de Boor-Cox recurrence: h has shape (nG, k+1)
    nG = x.shape[0]
    h = jnp.zeros((nG, k + 1))
    h = h.at[:, 0].set(1.0)
    for j in range(1, k + 1):
        hh = h
        h = h.at[:, 0].set(0.0)
        for i in range(j):
            li = l + i + 1              # (nG,)
            lj = l + i + 1 - j          # (nG,)
            t_li = t[li]                 # (nG,) — fancy indexing
            t_lj = t[lj]                 # (nG,)
            f = hh[:, i] / (t_li - t_lj)
            h = h.at[:, i].add(f * (t_li - x))
            h = h.at[:, i + 1].set(f * (x - t_lj))

    # Dot with coefficient window: S(x) = sum_{i=0}^{k} c[l-k+i] * h[i]
    # Build coefficient windows for all points at once
    offsets = l - k                                    # (nG,)
    idx = offsets[:, None] + jnp.arange(k + 1)[None, :]  # (nG, k+1)
    c_win = c[idx]                                     # (nG, k+1)
    return jnp.sum(c_win * h, axis=1)
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
import psp.vnl_ops as vnl_ops


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
    # G_l(q) = F_l(q)/q^l splines — smooth at q=0, for solid-harmonic path
    reduced_spline_t: list[jax.Array]   # [knots per beta]
    reduced_spline_c: list[jax.Array]   # [coeffs per beta]
    # G'_l(q) derivative splines — precomputed for fast custom_jvp
    reduced_deriv_t: list[jax.Array]    # [knots per beta]
    reduced_deriv_c: list[jax.Array]    # [coeffs per beta]
    reduced_deriv_k: int                # degree of derivative spline (k-1)
    E: jax.Array                # (nspinor, nspinor, R, R) D-matrix


def _build_reduced_spline(spl, l: int):
    """Build G_l(q) = F_l(q)/q^l spline AND its derivative G'_l spline.

    G_l is smooth at q=0 (since F_l ~ q^l for small q).
    Returns ((t_G, c_G, k_G), (t_Gp, c_Gp, k_Gp)).
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    t_F, c_F, k_F = spl._eval_args
    q_max = float(t_F[-1])
    n_pts = max(50000, len(t_F) * 10)
    q_grid = np.linspace(0, q_max, n_pts)
    F_vals = np.asarray(spl(q_grid), dtype=np.float64)
    if l == 0:
        G_vals = F_vals
    else:
        G_vals = np.empty_like(F_vals)
        G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
        G_vals[0] = F_vals[1] / q_grid[1] ** l
    deg = min(k_F, 3)
    spl_G = InterpolatedUnivariateSpline(q_grid, G_vals, k=deg)
    spl_Gp = spl_G.derivative()
    return extract_spline_data(spl_G), extract_spline_data(spl_Gp)


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
            rdd_t_list, rdd_c_list = [], []
            deriv_k = None
            for bid in beta_ids:
                spl = splines[(l, int(bid))]
                t, c, k = extract_spline_data(spl)
                spl_t_list.append(jnp.asarray(t))
                spl_c_list.append(jnp.asarray(c))
                # Reduced spline G_l = F_l / q^l + derivative G'_l
                (t_r, c_r, _), (t_d, c_d, k_d) = _build_reduced_spline(spl, l)
                red_t_list.append(jnp.asarray(t_r))
                red_c_list.append(jnp.asarray(c_r))
                rdd_t_list.append(jnp.asarray(t_d))
                rdd_c_list.append(jnp.asarray(c_d))
                deriv_k = int(k_d)

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
                reduced_deriv_t=rdd_t_list,
                reduced_deriv_c=rdd_c_list,
                reduced_deriv_k=deriv_k if deriv_k is not None else max(0, int(k) - 1),
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
        tuple(ch.reduced_deriv_t),
        tuple(ch.reduced_deriv_c),
        pref, l, nbeta, ch.spline_k,
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def _radial_times_solid_harm_impl(
    K_cart, spl_t, spl_c, spl_dt, spl_dc, pref, l, nbeta, spl_k,
):
    _EPS2 = 1e-60
    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)
    G_list = [splev_jax(q, spl_t[ib], spl_c[ib], spl_k) for ib in range(nbeta)]
    G_bG = jnp.stack(G_list, axis=0)
    S = _solid_harmonics_jax(l, K_cart)
    return pref * (1j) ** l * G_bG[:, None, :] * S[None, :, :]


@_radial_times_solid_harm_impl.defjvp
def _radial_times_solid_harm_jvp(
    spl_t, spl_c, spl_dt, spl_dc, pref, l, nbeta, spl_k,
    primals, tangents,
):
    """Stable JVP for G_l(|K|) * S_lm(K).

    d/dK [G_l(q) S_lm(K)] = G'_l(q) (K.dK)/q S_lm + G_l(q) dS_lm

    At K=0: (K.dK)/q = 0 (stable).  dS_lm/dK = polynomial gradient.
    G'_l(q) is evaluated via a **precomputed derivative spline** — no
    autodiff through the B-spline evaluation.
    """
    (K_cart,) = primals
    (dK_cart,) = tangents
    _EPS2 = 1e-60
    deriv_k = max(0, spl_k - 1)

    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)

    # G_l(q) from reduced spline, G'_l(q) from precomputed derivative spline
    G_list, Gp_list = [], []
    for ib in range(nbeta):
        G_list.append(splev_jax(q, spl_t[ib], spl_c[ib], spl_k))
        Gp_list.append(splev_jax(q, spl_dt[ib], spl_dc[ib], deriv_k))
    G_bG = jnp.stack(G_list, axis=0)
    Gp_bG = jnp.stack(Gp_list, axis=0)
    S = _solid_harmonics_jax(l, K_cart)

    primal_out = pref * (1j) ** l * G_bG[:, None, :] * S[None, :, :]

    # Radial tangent: G'(q) * (K.dK)/q — stable at K=0
    K_dot_dK = jnp.sum(K_cart * dK_cart, axis=1)
    dq = K_dot_dK / q
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


def build_Z_and_dZ(
    k_crys: jax.Array,
    G_int: jax.Array,
    B: jax.Array,
    channels: list[VNLChannelData],
) -> list[tuple[jax.Array, jax.Array, jax.Array]]:
    """Precompute VNL projectors Z and their Cartesian k-derivatives dZ.

    Returns one (Z, dZ, E) per channel where:
      Z  : (natoms, R, nG) complex128
      dZ : (3, natoms, R, nG) complex128 — d/dK_cart_j
      E  : (nspinor, nspinor, R, R) complex128

    Uses solid harmonics + reduced radial splines for correct derivatives
    at K=0 (Gamma point).
    """
    _EPS2 = 1e-60
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]
    K_cart = K_crys @ B
    Binv = jnp.linalg.inv(B)
    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)

    result = []
    for ch in channels:
        l = ch.l
        pref = ch.prefactor
        natoms = ch.tau.shape[0]
        msize = 2 * l + 1

        # G_l(q), G'_l(q) from precomputed splines
        G_list, Gp_list = [], []
        for ib in range(ch.nbeta):
            G_list.append(splev_jax(q, ch.reduced_spline_t[ib],
                                    ch.reduced_spline_c[ib], ch.spline_k))
            Gp_list.append(splev_jax(q, ch.reduced_deriv_t[ib],
                                     ch.reduced_deriv_c[ib], ch.reduced_deriv_k))
        G_bG = jnp.stack(G_list, axis=0)       # (nbeta, nG)
        Gp_bG = jnp.stack(Gp_list, axis=0)     # (nbeta, nG)

        # Solid harmonics and their Cartesian gradients
        S = _solid_harmonics_jax(l, K_cart)     # (msize, nG)
        # dS/dK_cart_j: evaluate for each Cartesian direction
        dS_list = []
        for j in range(3):
            ej = jnp.zeros((K_cart.shape[0], 3)).at[:, j].set(1.0)
            _, dS_j = jax.jvp(
                lambda K: _solid_harmonics_jax(l, K), (K_cart,), (ej,)
            )
            dS_list.append(dS_j)
        dS = jnp.stack(dS_list, axis=0)         # (3, msize, nG)

        # Phases and phase derivatives
        tau_j = ch.tau
        phase = jnp.exp(-2j * jnp.pi * (K_crys @ tau_j.T)).T  # (natoms, nG)
        # dphase/dK_cart_j = -2πi Σ_α (B^{-1})_{αj} τ_α * phase
        # τ is in crystal coords; B^{-1} converts K_cart derivative to K_crys
        tau_cart_eff = tau_j @ Binv.T            # (natoms, 3) effective for Cartesian deriv
        # dphase/dK_cart_j = -2πi τ_cart_eff_j * phase
        dphase = -2j * jnp.pi * tau_cart_eff[:, :, None] * phase[:, None, :]  # (natoms, 3, nG)

        # Z = pref (i)^l G_l S phase  → (natoms, nbeta, msize, nG)
        c_il = pref * (1j) ** l
        radS = G_bG[:, None, :] * S[None, :, :]           # (nbeta, msize, nG)
        Z_atoms = c_il * phase[:, None, None, :] * radS[None, ...]  # (na, nb, ms, nG)

        # dZ/dK_cart_j = pref (i)^l [G' K_j/q S phase + G dS_j phase + G S dphase_j]
        K_over_q = K_cart / q[:, None]                     # (nG, 3) — stable at K=0
        # Term 1: radial derivative
        drad = Gp_bG[:, None, :] * K_over_q.T[None, :, :]  # (nbeta, 3, nG) — G' K_j/q per beta
        term1 = drad[:, :, None, :] * S[None, None, :, :]   # (nbeta, 3, msize, nG)
        # Term 2: angular derivative
        term2 = G_bG[:, None, None, :] * dS[None, :, :, :]  # (nbeta, 3, msize, nG)
        # Term 3: phase derivative
        dZ_core = c_il * (term1 + term2)                     # (nbeta, 3, msize, nG)
        dZ_from_core = phase[:, None, None, None, :] * dZ_core[None, ...]  # (na, nb, 3, ms, nG)
        dZ_from_phase = c_il * radS[None, :, None, :, :] * dphase[:, None, :, None, :]
        dZ_full = dZ_from_core + dZ_from_phase               # (na, nb, 3, ms, nG)

        R = ch.nbeta * msize
        Z_flat = Z_atoms.reshape(natoms, R, K_cart.shape[0])
        # dZ: (3, natoms, R, nG) — rearrange axes
        dZ_flat = dZ_full.transpose(2, 0, 1, 3, 4).reshape(3, natoms, R, K_cart.shape[0])

        result.append((Z_flat, dZ_flat, ch.E))

    return result


def vnl_velocity_from_dZ(
    psi_G: jax.Array,
    Z_dZ_E: list[tuple[jax.Array, jax.Array, jax.Array]],
) -> jax.Array:
    """Nonlocal velocity from precomputed Z and dZ.

    Returns (3, nb, nb) = -(dZ† E Z + Z† E dZ) contracted with psi.

    Parameters
    ----------
    psi_G : (nb, nspinor, nG)
    Z_dZ_E : list of (Z, dZ, E) from ``build_Z_and_dZ``
    """
    nb = psi_G.shape[0]
    v = jnp.zeros((3, nb, nb), dtype=jnp.complex128)
    for Z, dZ, E in Z_dZ_E:
        # P = Z†ψ, dP = dZ†ψ
        P = jnp.einsum('arG,vtG->artv', jnp.conj(Z), psi_G, optimize=True)
        D = jnp.einsum('strq,aqtv->arsv', E, P, optimize=True)
        # term1: <m| dZ E Z† |n>
        for j in range(3):
            dP_j = jnp.einsum('arG,vtG->artv', jnp.conj(dZ[j]), psi_G, optimize=True)
            dD_j = jnp.einsum('strq,aqtv->arsv', E, dP_j, optimize=True)
            # dZ† E Z† psi contracted with psi*
            vnl_G = jnp.einsum('arG,arsv->vsG', Z, dD_j, optimize=True)
            t1 = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), vnl_G, optimize=True)
            # Z† E dZ† psi contracted with psi*
            vnl_G2 = jnp.einsum('arG,arsv->vsG', dZ[j], D, optimize=True)
            t2 = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), vnl_G2, optimize=True)
            v = v.at[j].add(t1 + t2)
    return v


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
# Momentum matrix elements (local part of velocity)
# ---------------------------------------------------------------------------

@jax.jit
def momentum_matrix_k(psi_G, G_int, k_crys, B):
    """Local momentum matrix: 2*(k+G)_cart_i * <m|G><G|n>.

    Returns (3, nb, nb) — the kinetic contribution to the velocity
    operator dT/dk_i = 2*(k+G)_i in Ry units.

    Parameters
    ----------
    psi_G : (nb, nspinor, nG) — sparse-G coefficients
    G_int : (nG, 3) int — G-vectors (crystal)
    k_crys : (3,) — k-point (crystal)
    B : (3, 3) — crystal-to-Cartesian transformation
    """
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]
    K_cart = K_crys @ B                          # (nG, 3)
    # p_i[m,n] = 2 * Σ_{s,G} conj(ψ_m[s,G]) K_cart[G,i] ψ_n[s,G]
    p = 2.0 * jnp.einsum(
        'msG,Gi,nsG->imn', jnp.conj(psi_G), K_cart, psi_G, optimize=True,
    )
    return p


# ---------------------------------------------------------------------------
# Full velocity / dipole matrix (momentum + VNL)
# ---------------------------------------------------------------------------

def velocity_matrix_k(
    psi_G: jax.Array,
    G_int: jax.Array,
    k_crys: jax.Array,
    B: jax.Array,
    channels: list[VNLChannelData],
    *,
    Z_dZ_E: list | None = None,
) -> jax.Array:
    """Full velocity matrix: v_i = dH/dk_i = 2(k+G)_i + dV_NL/dk_i.

    Returns (3, nb, nb).

    Parameters
    ----------
    Z_dZ_E : optional precomputed (Z, dZ, E) from ``build_Z_and_dZ``.
        If not provided, built on the fly.
    """
    p = momentum_matrix_k(psi_G, G_int, k_crys, B)
    if Z_dZ_E is None:
        Z_dZ_E = build_Z_and_dZ(k_crys, G_int, B, channels)
    vNL = vnl_velocity_from_dZ(psi_G, Z_dZ_E)
    return p + vNL


def compute_dipole_all(
    wfn: WFNReader,
    sym,
    meta: Meta,
    setup: OperatorSetup,
    nb: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity/dipole matrix elements for all k-points.

    Returns
    -------
    dipole : (3, nk, nb, nb) complex128 — velocity matrix elements
    deltaE : (nk, nb, nb) float64 — energy differences E_m - E_n
    """
    if nb is None:
        nb = int(meta.b_id_4)
    nk = sym.nk_tot
    nspinor = int(meta.nspinor)

    channels = extract_vnl_channel_data(setup.vnl_plan, nspinor=nspinor)
    B_j = jnp.asarray(setup.B, dtype=jnp.float64)

    dipole = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    deltaE = np.zeros((nk, nb, nb), dtype=np.float64)
    energies = np.asarray(wfn.energies)

    for ik in range(nk):
        with timing.section(f"dft_operators.dipole_k{ik}"):
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
            Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
            kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
            Gx = np.asarray(Gk_crys)[:, 0]
            Gy = np.asarray(Gk_crys)[:, 1]
            Gz = np.asarray(Gk_crys)[:, 2]
            psi_G = wfn_k[:, :, Gx, Gy, Gz]
            G_int = jnp.asarray(np.asarray(Gk_crys, dtype=int), dtype=jnp.int32)
            k_j = jnp.asarray(kvec, dtype=jnp.float64)

            Z_dZ_E = build_Z_and_dZ(k_j, G_int, B_j, channels)
            v = velocity_matrix_k(psi_G, G_int, k_j, B_j, channels,
                                  Z_dZ_E=Z_dZ_E)
            dipole[:, ik] = np.asarray(v)

            # Energy differences
            try:
                k_red = int(sym.irk_to_k_map[ik])
            except Exception:
                k_red = int(ik)
            if energies.ndim >= 3:
                e_b = np.asarray(energies[0, k_red, :nb], dtype=float)
            else:
                e_b = np.asarray(energies[:nb], dtype=float)
            deltaE[ik] = e_b[:, None] - e_b[None, :]
            del wfn_k

    return dipole, deltaE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OperatorSetup:
    """K-point-independent operator data.  Built once, shared across k."""
    V_r: jax.Array                  # (nx, ny, nz) V_loc [+V_H] [Ry]
    vnl_plan: dict
    vnl_setup: object               # vnl_ops.VNLSetup (dense VNL backend)
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
    # Dense VNL from vnl_ops
    vnl_Z: jax.Array                # (total_R, nG)
    vnl_E: jax.Array                # (nspinor, nspinor, total_R, total_R)
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

    # Dense VNL backend (vnl_ops)
    vsetup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos,
                                     nspinor=int(meta.nspinor))

    return OperatorSetup(
        V_r=V_r,
        vnl_plan=vnl_plan,
        vnl_setup=vsetup,
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
    """Build per-k data: kinetic diagonal and dense VNL projectors."""
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

    # Dense VNL via vnl_ops
    kdata = vnl_ops.build_vnl_kdata(k_idx, setup.vnl_setup, wfn, sym, meta)

    return KPointOperators(
        T_diag=T_diag,
        Gx=Gx, Gy=Gy, Gz=Gz,
        V_r=setup.V_r,
        vnl_Z=kdata.Z,
        vnl_E=kdata.E_super,
        nG=nG,
        fft_grid=setup.fft_grid,
    )


# ---------------------------------------------------------------------------
# Core fused kernels
# ---------------------------------------------------------------------------

@jax.jit
def apply_H_k(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_Z, vnl_E):
    """Fused H|psi>: FFT-box in, sparse-G out.  Single JIT dispatch.

    Uses dense VNL from vnl_ops (single einsum, no channel loop).

    Parameters
    ----------
    psi_box : (nvec, nspinor, nx, ny, nz) — trial vectors in FFT box
    T_diag  : (nG,) — kinetic diagonal
    V_r     : (nx, ny, nz) — real-space local potential
    Gx,Gy,Gz : (nG,) int32 — G-vector FFT-box indices
    vnl_Z   : (total_R, nG) — dense VNL projector from vnl_ops
    vnl_E   : (nspinor, nspinor, total_R, total_R) — block-diag D matrix

    Returns
    -------
    H_psi_G : (nvec, nspinor, nG) — sparse-G
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]
    # T
    H_G = T_diag[None, None, :] * psi_G
    # V_loc
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    H_G = H_G + jnp.fft.fftn(
        psi_r * V_r, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz]
    # V_NL (dense, no loop)
    P = jnp.einsum('RG,vsG->Rsv', jnp.conj(vnl_Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtv->Rsv', vnl_E, P, optimize=True)
    H_G = H_G + jnp.einsum('RG,Rsv->vsG', vnl_Z, D, optimize=True)
    return H_G


@jax.jit
def build_matrix_k(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_Z, vnl_E):
    """Fused H matrix elements: <m|H|n> for all bands at one k-point.

    Uses dense VNL from vnl_ops (single einsum, no channel loop).

    Returns (nb, nb) complex128.
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]

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

    # V_NL (dense, no loop)
    H_mn = H_mn + vnl_ops.vnl_matrix(psi_G, vnl_Z, vnl_E)

    return H_mn


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def apply(psi_box: jax.Array, kops: KPointOperators) -> jax.Array:
    """Apply H|psi> at one k-point.  FFT-box in, sparse-G out."""
    return apply_H_k(
        psi_box, kops.T_diag, kops.V_r,
        kops.Gx, kops.Gy, kops.Gz,
        kops.vnl_Z, kops.vnl_E,
    )


def matrix(psi_box: jax.Array, kops: KPointOperators) -> jax.Array:
    """Build H_mn at one k-point.  Returns (nb, nb)."""
    return build_matrix_k(
        psi_box, kops.T_diag, kops.V_r,
        kops.Gx, kops.Gy, kops.Gz,
        kops.vnl_Z, kops.vnl_E,
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
                jax.device_put(kops.vnl_Z, dev),
                jax.device_put(kops.vnl_E, dev),
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
