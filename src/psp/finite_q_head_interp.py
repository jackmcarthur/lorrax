"""psp/finite_q_head_interp.py — Finite-q head/wing interpolation for screened W (PoC).

Build smooth, interpolable representations of  V_{μν}(Q)  and  W_{μν}(Q, ω)
in the ISDF centroid basis on a coarse q-grid, with the singular Coulomb
channel split off analytically.  Designed to be combined with htransform's
fine-grid wavefunctions (``bandstructure.bse_setup.compute_wfns_fi``) so
the BSE Hamiltonian can be evaluated at any target  Q = q_red + q'  on a
finer k-grid than the GW solve was run on.

Math summary
============

We use the user's notation throughout. Let ``z_{q,μ}(r) = e^{-iq·r} ζ_{q,μ}(r)``
and ``z_{q,μ}(G)`` its plane-wave coefficient. Define the absolute-channel
projection

    g_μ(Q) = z_{q_red, μ}(G_Q),

where  Q = q_red + G_Q  is the reciprocal-vector decomposition.  The bare
Coulomb in the centroid basis splits as

    V_{μν}(Q) = V^body_{μν}(Q) + g*_μ(Q) · v_head(Q) · g_ν(Q),                 (V)

with  v_head(Q) = v(Q)  the singular Coulomb head (4π/Q² in 3D; 2D-truncated
slab variant in 2D), and  V^body  a smooth *G≠G_Q* sum.  The screened W,
sampled only on the coarse q-grid, similarly admits a Schur-complement
reconstruction once we work with the head-removed body solve

    W^0_body(Q, ω) = [1 − V^body(Q) · χ_body(Q, ω)]^{-1} · V^body(Q),         (W0)

and the smooth, head-removed susceptibility pieces  χ_head, χ_wing, χ_wing'
that come from sum-over-states evaluations of pair densities at  G = 0
(the head vertex  h_t = ⟨c,k-Q|e^{iQ·r}|v,k⟩) and at the centroids
(the body vertex  b_{t,μ} = Σ_s ψ*_{c,k-Q,s}(r_μ) ψ_{v,k,s}(r_μ)).

The **interpolation targets** — all smooth as functions of Q — are

    g_μ(Q),   V^body_{μν}(Q),   W^0_body_{μν}(Q,ω),
    χ_head_eff(Q,ω),   A_wing_μ(Q,ω),   A_wing'_ν(Q,ω),

where the local-field-corrected head/wing combinations are

    χ_head_eff = χ_head + χ_wing' · W^0_body · χ_wing,                       (chi_eff)
    A_wing      = W^0_body · χ_wing,                                          (A)
    A_wing'     = χ_wing' · W^0_body.

Inside the mini-BZ around q=0 we replace the values by the analytic leading
forms, parametrised by the dispersion tensors

    h_t(q')   = q'_a · d_{a,t} + O(q'²),
    S_ab      = (1/N_k Ω) Σ_t F_t(0, ω) · d*_{a,t} · d_{b,t},
    w_{a,μ}   = (1/N_k Ω) Σ_t F_t(0, ω) · b*_{t,μ}(0) · d_{a,t},
    w'_{a,ν}  = (1/N_k Ω) Σ_t F_t(0, ω) · d*_{a,t} · b_{t,ν}(0),
    S_eff_ab  = S_ab + w'_a^T · W^0_body(0) · w_b,
    B_{a,μ}   = W^0_body(0) · w_{a,·},
    B'_{a,ν}  = w'_{a,·} · W^0_body(0).

Then

    W_head(q') = v_head(q') / (1 − v_head(q') · q'^a S_eff_ab q'^b),
    A_wing  → q'_a · B_{a,μ}      (and analogously A_wing'),

so the screened reconstruction

    W_{μν}(Q,ω) = W^0_body_{μν}(Q,ω)
                + g*_μ(Q) · W_head(Q,ω) · g_ν(Q)
                + A_wing_μ(Q,ω) · W_head(Q,ω) · g_ν(Q)
                + g*_μ(Q) · W_head(Q,ω) · A_wing'_ν(Q,ω)
                + A_wing_μ(Q,ω) · W_head(Q,ω) · A_wing'_ν(Q,ω)

remains valid through the q'→0 limit, with all singular behaviour analytic.

PoC scope (this file)
=====================

Serial, single-device.  The intent is correctness and pedagogical clarity, not
production performance.  Public surface:

  - :func:`v_head_3d`, :func:`v_head_2d_slab`           singular head v(Q)
  - :func:`compute_g_mu_at_q`                            absolute-channel projector
  - :func:`compute_V_body_from_zeta`                     V_body as in compute_V_q (G_q-zeroed)
  - :func:`compute_pair_density_head`                    h_t(q) = ⟨c,k-q|e^{iq·r}|v,k⟩
  - :func:`compute_pair_density_centroid`                b_{t,μ}(q) = Σ_s ψ*ψ at r_μ
  - :func:`compute_chi_head_wing_body_q`                 χ_head, χ_wing, χ_wing', χ_body (SOS)
  - :func:`compute_dipole_dt`                            d_{a,t} = ⟨c,k|∇_k|v,k⟩/(ε_c−ε_v)
  - :func:`compute_q0_dispersion_tensors`                S_ab, w_aμ, w'_aν
  - :func:`solve_W_body0`                                (1-V_body χ_body)^{-1} V_body
  - :func:`build_chi_head_eff_and_A_wings`               local-field combinations
  - :func:`reconstruct_V_munu`, :func:`reconstruct_W_munu`   per-Q reconstruction
  - :func:`fourier_interpolate_coarse_to_fine`           crude Fourier upscaler
  - :func:`reconstruct_W_at_target_Q`                    end-to-end reconstruction at Q

A synthetic smoke test (``__main__``) builds toy data, takes the q'→0 limit,
and checks against the rank-1 q=0 head injection used today in
``gw.head_correction.apply_q0_head_rank1``.

Hooking up a real run
=====================

The intended driver flow is:

  1. Read coarse-grid restart (V_qmunu, W0_qmunu, vhead, whead, G0_mu_nu, zeta_q).
  2. For every coarse q ≠ 0: build h_t(q), b_{t,μ}(q) from full-zone ψ_n,k (via
     symmetry_maps + load_wfns), then assemble χ_head/wing/wing'/body.
  3. At q = 0: build d_{a,t} via psp.get_dipole_mtxels (or velocity matrix
     elements), build S_ab, w_aμ, w'_aν.
  4. Form W^0_body(Q) by reusing :func:`gw.w_isdf.solve_w` per-q with the
     head-removed V_body.
  5. Persist the smooth interpolation tensors next to the existing restart;
     downstream BSE-on-fine-grid (``bandstructure.bse_setup.compute_wfns_fi``)
     calls :func:`reconstruct_W_at_target_Q` for each fine Q.

Step 5 closes the loop with C's q=0 head rank-1 work
(``gw.head_correction.apply_q0_head_rank1``); the q'→0 limit of this
reconstruction is exactly that rank-1 update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# JAX is optional in this PoC — pure numpy keeps the math readable.  The
# heavy lifting (FFTs, big einsums) can later be promoted to jax.numpy by
# swapping `xp` below; ``compute_pair_density_*`` already accept jnp arrays
# transparently because numpy/jnp agree on the einsum contract.
import jax.numpy as jnp


# ════════════════════════════════════════════════════════════════════════
#  Coulomb head  v_head(Q)
# ════════════════════════════════════════════════════════════════════════

def v_head_3d(Qcart: np.ndarray) -> np.ndarray:
    """Singular Coulomb head in 3D bulk:  v(Q) = 8π/|Q|² (Ry-Bohr units).

    The factor of 8π (not 4π) matches the LORRAX/BGW convention used in
    ``gw.compute_vcoul``: see the ``8.0 * π`` prefactor in the ``sys_dim=3``
    branch of :func:`compute_sqrt_vcoul_0d`.

    Q must NOT be the absolute-zone origin — the caller is responsible for
    routing  Q→0  through :func:`reconstruct_W_at_target_Q` with the q=0
    dispersion tensors.
    """
    Qcart = np.asarray(Qcart, dtype=np.float64)
    q2 = float(np.dot(Qcart, Qcart))
    if q2 < 1e-30:
        raise ValueError("v_head_3d called at Q=0; use the q=0 dispersion path.")
    return 8.0 * math.pi / q2


def v_head_2d_slab(Qcart: np.ndarray, zc: float) -> np.ndarray:
    """Singular Coulomb head in a slab (2D-truncated) geometry.

    For an in-plane Q with magnitude  q∥ = |Q∥|  and out-of-plane component
    Q_z, the Ismail-Beigi truncated head is

        v(Q) = (8π / |Q|²) · [1 − e^{−q∥ z_c} · cos(Q_z z_c)]

    with z_c the truncation half-length. The caller supplies the cartesian
    Q and z_c (Bohr).  The (q∥, Q_z) split assumes the slab normal is the
    cartesian-z axis (LORRAX/BGW convention).

    For purely in-plane Q (Q_z = 0) and small q∥ this reduces to the 1/q∥
    "2D Coulomb" leading form — ie head singularity is weaker than 3D.
    """
    Qcart = np.asarray(Qcart, dtype=np.float64)
    q_par = float(np.hypot(Qcart[0], Qcart[1]))
    q_z = float(Qcart[2])
    q2 = q_par * q_par + q_z * q_z
    if q2 < 1e-30:
        raise ValueError("v_head_2d_slab called at Q=0; use the q=0 dispersion path.")
    return (8.0 * math.pi / q2) * (1.0 - math.exp(-q_par * zc) * math.cos(q_z * zc))


# ════════════════════════════════════════════════════════════════════════
#  Absolute-channel projection  g_μ(Q)
# ════════════════════════════════════════════════════════════════════════

def compute_g_mu_at_q(
    zeta_q: np.ndarray,            # (n_μ, n_rtot) ζ_{q,μ}(r) on the FFT box (real-space)
    fft_grid: Tuple[int, int, int],
    Q_int: np.ndarray,             # (3,) integer reciprocal vector (the absolute-channel
                                    #       label inside ζ's wrap convention).  For Q at the
                                    #       reduced-zone origin this is (0,0,0).
) -> np.ndarray:
    """Return  g_μ(Q) = z_{q_red, μ}(G_Q) = (FFT of e^{-iq_red·r} ζ_μ(r)) at G_Q.

    Convention matches :func:`gw.compute_vcoul.compute_V_q_from_zeta_array`:
    ``g0_mu = ζ_μ(G=0)`` is exactly this routine called with ``Q_int = (0,0,0)``.

    For finite-Q targets that don't wrap (G_Q = 0), this is just the (0,0,0)
    plane-wave coefficient of  e^{-iq_red·r} ζ_μ.  When the target Q crosses
    a BZ boundary, ``Q_int`` is the integer G that brings Q back into the
    reduced zone.
    """
    nx, ny, nz = fft_grid
    n_mu = zeta_q.shape[0]
    z_box = zeta_q.reshape(n_mu, nx, ny, nz)
    # FFT here — full grid because we may need an arbitrary G_Q.  This PoC
    # path doesn't try to be clever for the G_Q=0 case (which compute_V_q
    # already optimises).
    # NB: the e^{-iq_red·r} phase is handled outside in compute_V_q; we mirror
    # that and assume zeta_q is the *cell-periodic* ζ_μ.  For correctness in
    # the wrap path the caller must compose ζ_μ with the same phase.
    Z_G = np.fft.fftn(z_box, axes=(-3, -2, -1))
    ix = int(Q_int[0]) % nx
    iy = int(Q_int[1]) % ny
    iz = int(Q_int[2]) % nz
    return Z_G[:, ix, iy, iz]


# ════════════════════════════════════════════════════════════════════════
#  Body Coulomb  V^body_{μν}(q)  — thin wrapper for clarity
# ════════════════════════════════════════════════════════════════════════
#
# In production we already have ``gw.compute_vcoul.compute_V_q_from_zeta_array``
# which returns (V_body, g0_mu) for one q. The "body" Coulomb is the same
# routine *with the absolute channel zeroed* — currently only at q_red = 0
# is G=0 explicitly excluded (``compute_vcoul`` zeros the G=G'=0 element).
# The interpolation needs the same exclusion at every coarse q (with G_Q
# the relevant absolute channel).  The user's note suggests storing
# ``V_qmunu`` with G_Q always zeroed; that is a one-line change in
# compute_V_q_from_zeta_array (subtract the rank-1  g* g  contribution
# at G_Q after the full sum, which is what apply_q0_head_rank1 reverses
# at q=0 today).
#
# This PoC takes the *already-G_Q-zeroed* V_body arrays as input — exactly
# what GW persists once the small compute_vcoul change is in place.


# ════════════════════════════════════════════════════════════════════════
#  Sum-over-states evaluators
# ════════════════════════════════════════════════════════════════════════
#
# Conventions (matching gw.w_isdf.compute_chi0 and BGW):
#
#   χ_GG'(q,ω) = (4 / N_k Ω) · Σ_{v,c,k} M*_t(G) M_t(G')
#                                · [ 1/(ω − Δε_t + iη) − 1/(ω + Δε_t − iη) ]
#
#   Δε_t = ε_{c,k-q} − ε_{v,k},     M_t(G) = ⟨c,k-q | e^{i(q+G)·r} | v,k⟩
#
# In the centroid basis the body matrix element is  Σ_s ψ*_{c,k-q,s}(r_μ) ψ_{v,k,s}(r_μ),
# and the head element (G=0) is  M_t(0) = h_t(q) = ⟨c,k-q|e^{iq·r}|v,k⟩.
#
# The PoC denominator factor is parametrised: pass any ``denom_fn(eps_v,
# eps_c, omega)`` returning F_t(q, ω). For static screening (COHSEX) use
# the convenience :func:`F_static`.

def F_static(eps_v: np.ndarray, eps_c: np.ndarray, omega: complex = 0.0+0j,
             eta: float = 1e-6) -> np.ndarray:
    """Static-limit Adler–Wiser denominator factor (without the leading 4/N_k Ω).

        F_t = 1/(ω − Δε_t + iη) − 1/(ω + Δε_t − iη)

    For ω → 0 this reduces to  −2 Δε_t / (Δε_t² + η²)  — pure real, negative.

    Shapes:  eps_v[v,k] and eps_c[c,k] broadcast against the transition axis.
    Returns an array shaped like the broadcast outer  (v,c,k)  product.
    """
    deps = eps_c[:, None, :] - eps_v[None, :, :]                    # (c, v, k)
    denom1 = (omega - deps + 1j * eta)
    denom2 = (omega + deps - 1j * eta)
    return 1.0 / denom1 - 1.0 / denom2                              # (c, v, k)


def compute_pair_density_head(
    psi_v_k_box: np.ndarray,        # (n_v, n_spinor, nx, ny, nz)  ψ_{v,k} on FFT box (full ψ, not cell-periodic)
    psi_c_kmq_box: np.ndarray,      # (n_c, n_spinor, nx, ny, nz)  ψ_{c, k-q}, similarly
) -> np.ndarray:
    """Return  h_t(q) = ⟨c, k-q | e^{iq·r} | v, k⟩  for one (k, q).

    Cell-periodic identity:  ⟨c,k-q|e^{iq·r}|v,k⟩
                          = (1/Ω_cell) ∫_cell  u*_{c,k-q}(r) u_{v,k}(r) dr
                          = sum over the FFT box of  conj(ψ_c) ψ_v / N_FFT
                          (the e^{iq·r} cancels the Bloch-phase mismatch).

    Inputs are full ψ (with Bloch phase) on the FFT box — exactly what the
    LORRAX wfn-loader returns.  The cancellation of  e^{−i k·r} e^{i (k-q)·r}
    against  e^{iq·r}  is automatic.

    Shapes:
      out: (n_c, n_v)  complex
    """
    n_v = psi_v_k_box.shape[0]
    n_c = psi_c_kmq_box.shape[0]
    n_FFT = float(np.prod(psi_v_k_box.shape[2:]))
    # Sum over spinor + spatial axes; broadcast over c (rows) and v (cols).
    return jnp.einsum('csxyz,vsxyz->cv',
                      jnp.conj(psi_c_kmq_box), psi_v_k_box,
                      optimize=True) / n_FFT


def compute_pair_density_centroid(
    psi_v_k_rmu: np.ndarray,        # (n_v, n_spinor, n_μ)  ψ_{v,k}(r_μ)
    psi_c_kmq_rmu: np.ndarray,      # (n_c, n_spinor, n_μ)  ψ_{c, k-q}(r_μ)
) -> np.ndarray:
    """Return  b_{t,μ}(q) = Σ_s ψ*_{c,k-q,s}(r_μ) ψ_{v,k,s}(r_μ)  for one (k, q).

    Output shape:  (n_c, n_v, n_μ)  complex.  Centroid pair density before
    the ζ-overlap weighting — this is the canonical "B_at_mu" used by
    htransform / bse_setup, for the (c,k-q ; v,k) pair.
    """
    return jnp.einsum('csm,vsm->cvm',
                      jnp.conj(psi_c_kmq_rmu), psi_v_k_rmu,
                      optimize=True)


def compute_chi_head_wing_body_q(
    h: np.ndarray,                  # (n_c, n_v, n_k)             h_t(q)
    b: np.ndarray,                  # (n_c, n_v, n_k, n_μ)        b_{t,μ}(q)
    eps_v: np.ndarray,              # (n_v, n_k)                  ε_{v,k}
    eps_c_kmq: np.ndarray,          # (n_c, n_k)                  ε_{c, k-q}  (already mapped)
    cell_volume: float,
    n_k_total: int,                 # full-BZ N_k for the prefactor
    spin_factor: float = 4.0,       # BGW χ leading 4 (= 2·spin·2 from Adler–Wiser)
    denom_fn: Callable = F_static,
    omega: complex = 0.0+0j,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the four (head, wing, wing', body) susceptibility pieces at one q.

    Returns (all in the natural BGW units, *before* the head/wing scaling
    convention used in epsilon.x — same convention as ``compute_chi0``):

      χ_head   : ()                     scalar
      χ_wing   : (n_μ,)                 (b*_{t,μ} h_t  contracted)
      χ_wing'  : (n_ν,)                 (h*_t b_{t,ν}  contracted)
      χ_body   : (n_μ, n_ν)             (b*_{t,μ} b_{t,ν} contracted)

    Numerical formula:

      F_t  = denom_fn(eps_v, eps_c_kmq, omega)             (n_c, n_v, n_k)
      pref = spin_factor / (cell_volume · n_k_total)

      χ_head     = pref · Σ_t F_t · |h_t|²
      χ_wing[μ]  = pref · Σ_t F_t · b*_{t,μ} · h_t
      χ_wing'[ν] = pref · Σ_t F_t · h*_t   · b_{t,ν}
      χ_body[μν] = pref · Σ_t F_t · b*_{t,μ} · b_{t,ν}

    All transition contractions are done with one ``einsum`` per piece to
    keep memory bounded (no  (c,v,k,μ,ν)  intermediates).
    """
    F = denom_fn(eps_v, eps_c_kmq, omega)                  # (c, v, k)
    pref = spin_factor / (cell_volume * float(n_k_total))

    # Head — pure SOS sum.
    chi_head = pref * jnp.einsum('cvk,cvk,cvk->',
                                 F, jnp.conj(h), h, optimize=True)

    # Wings — these are (n_μ,) and (n_ν,).
    # Memory: (c,v,k,μ) only inside einsum; jax/numpy keeps the loop fused.
    chi_wing = pref * jnp.einsum('cvk,cvkm,cvk->m',
                                 F, jnp.conj(b), h, optimize=True)
    chi_wingp = pref * jnp.einsum('cvk,cvk,cvkn->n',
                                  F, jnp.conj(h), b, optimize=True)

    # Body — output (n_μ, n_ν), the heaviest contraction.  This recomputes
    # what gw.w_isdf.compute_chi0 already does; useful as a sanity reference
    # for small systems, but in production the caller should reuse compute_chi0.
    chi_body = pref * jnp.einsum('cvk,cvkm,cvkn->mn',
                                 F, jnp.conj(b), b, optimize=True)

    return chi_head, chi_wing, chi_wingp, chi_body


# ════════════════════════════════════════════════════════════════════════
#  q → 0 dispersion tensors  (S_ab, w_aμ, w'_aν)
# ════════════════════════════════════════════════════════════════════════
#
# At q = 0 the head vertex h_t(q) vanishes:  ⟨c,k|v,k⟩ = δ_{cv} = 0 in our
# transition convention. Its leading derivative is the k·p / dipole element
#
#     d_{a,t}  =  q·d/dq h_t(q)|_{q=0}        (a labels Cartesian directions)
#               =  i ⟨c,k| ∂/∂k_a |v,k⟩
#                = i · ⟨c,k| p_a / m | v,k⟩ / (ε_{c,k} − ε_{v,k})        (BGW dipole)
#
# In LORRAX this is computed by ``psp.get_dipole_mtxels`` (matrix elements
# of the velocity operator i[H,r] including non-local commutators).  The
# PoC accepts d_{a,t} as input so users can plug in either path.

def compute_q0_dispersion_tensors(
    d: np.ndarray,                  # (n_a=3, n_c, n_v, n_k)  d_{a,t}
    b0: np.ndarray,                 # (n_c, n_v, n_k, n_μ)    b_{t,μ}(q=0)
    eps_v: np.ndarray,              # (n_v, n_k)
    eps_c: np.ndarray,              # (n_c, n_k)              ε_{c,k}  (no shift)
    cell_volume: float,
    n_k_total: int,
    spin_factor: float = 4.0,
    denom_fn: Callable = F_static,
    omega: complex = 0.0+0j,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build  S_{ab}, w_{a,μ}, w'_{a,ν}, χ_body(q=0)  as the q'→0 expansion
    coefficients of the susceptibility pieces.

    Returns
    -------
    S_ab       : (3, 3)             head Hessian  ∂²χ_head/∂q'_a ∂q'_b|_0
    w_a_mu     : (3, n_μ)           leading wing slope along q'_a, μ-component
    w_a_nu     : (3, n_ν)           leading wing' slope along q'_a, ν-component
    chi_body_0 : (n_μ, n_ν)         χ_body(q=0)
    """
    F = denom_fn(eps_v, eps_c, omega)                       # (c, v, k)
    pref = spin_factor / (cell_volume * float(n_k_total))

    # S_ab = pref · Σ_t F_t · d*_{a,t} d_{b,t}    (small (3,3))
    S = pref * jnp.einsum('cvk,acvk,bcvk->ab',
                          F, jnp.conj(d), d, optimize=True)

    # w_{a,μ}  = pref · Σ_t F_t · b*_{t,μ}(0) · d_{a,t}
    w_mu = pref * jnp.einsum('cvk,cvkm,acvk->am',
                             F, jnp.conj(b0), d, optimize=True)

    # w'_{a,ν} = pref · Σ_t F_t · d*_{a,t} · b_{t,ν}(0)
    w_nu = pref * jnp.einsum('cvk,acvk,cvkn->an',
                             F, jnp.conj(d), b0, optimize=True)

    chi_body_0 = pref * jnp.einsum('cvk,cvkm,cvkn->mn',
                                   F, jnp.conj(b0), b0, optimize=True)
    return S, w_mu, w_nu, chi_body_0


# ════════════════════════════════════════════════════════════════════════
#  Body W solve — wraps gw.w_isdf.solve_w semantics for the PoC
# ════════════════════════════════════════════════════════════════════════

def solve_W_body0(V_body: np.ndarray, chi_body: np.ndarray) -> np.ndarray:
    """Return  W^0_body = (1 − V_body · χ_body)^{-1} V_body  for one q, ω.

    PoC version: a single dense solve.  Production reuses
    ``gw.w_isdf.solve_w`` (sharded, low/high-mem variants).
    """
    n_mu = V_body.shape[0]
    eye = np.eye(n_mu, dtype=np.complex128)
    A = eye - V_body @ chi_body
    return np.linalg.solve(A, V_body)


# ════════════════════════════════════════════════════════════════════════
#  Local-field combinations (smooth interpolation targets)
# ════════════════════════════════════════════════════════════════════════

def build_chi_head_eff_and_A_wings(
    chi_head: complex,
    chi_wing: np.ndarray,           # (n_μ,)
    chi_wingp: np.ndarray,          # (n_ν,)
    W_body0: np.ndarray,            # (n_μ, n_ν)
) -> Tuple[complex, np.ndarray, np.ndarray]:
    """Form the smooth, head-removed combinations the user identified as
    the right targets to interpolate (chi_eff above):

        A_wing_μ      = Σ_ν W^0_body_{μν} · χ_wing_ν                       (n_μ,)
        A_wing'_ν     = Σ_μ χ_wing'_μ     · W^0_body_{μν}                  (n_ν,)
        χ_head_eff    = χ_head + χ_wing'_μ · W^0_body_{μν} · χ_wing_ν

    Returns (χ_head_eff, A_wing, A_wing').
    """
    A_wing  = W_body0 @ chi_wing                      # (n_μ,)
    A_wingp = chi_wingp @ W_body0                     # (n_ν,)  row-times-mat
    chi_head_eff = chi_head + chi_wingp @ A_wing
    return complex(chi_head_eff), A_wing, A_wingp


def build_q0_dispersion_eff(
    S_ab: np.ndarray,               # (3, 3)
    w_mu: np.ndarray,                # (3, n_μ)
    w_nu: np.ndarray,                # (3, n_ν)
    W_body0_q0: np.ndarray,         # (n_μ, n_ν)        at q=0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form  S_eff_ab,  B_{a,μ},  B'_{a,ν}  — the analytic q'→0 expansion
    coefficients of  (χ_head_eff, A_wing, A_wing')  through O(q').

    Returns (S_eff, B, Bp) with shapes ((3,3), (3, n_μ), (3, n_ν)).

    Index conventions:
        A_wing_i      = Σ_j W_b0[i,j] χ_wing[j]    →  with χ_wing[j] ≈ q'_a w_mu[a,j]
                        ⇒ B[a,i]  = Σ_j W_b0[i,j] w_mu[a,j]
        A_wing'_i     = Σ_j χ_wing'[j] W_b0[j,i]   →  with χ_wing'[j] ≈ q'_a w_nu[a,j]
                        ⇒ Bp[a,i] = Σ_j w_nu[a,j] W_b0[j,i]
        χ_head_eff    = χ_head + χ_wing' · A_wing
                        ⇒ S_eff[a,b] = S[a,b] + Σ_ij w_nu[a,i] W_b0[i,j] w_mu[b,j]
    """
    B  = jnp.einsum('ij,aj->ai', W_body0_q0, w_mu, optimize=True)   # (3, n_μ)
    Bp = jnp.einsum('aj,ji->ai', w_nu, W_body0_q0, optimize=True)   # (3, n_ν=n_μ)
    S_eff = S_ab + jnp.einsum('ai,ij,bj->ab',
                              w_nu, W_body0_q0, w_mu, optimize=True)
    return S_eff, B, Bp


# ════════════════════════════════════════════════════════════════════════
#  Reconstruction at a target Q
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CoarseQDataNonzero:
    """Smooth interpolation tensors stored at one nonzero coarse q."""
    g_mu:        np.ndarray         # (n_μ,)
    V_body:      np.ndarray         # (n_μ, n_ν)
    W_body0:     np.ndarray         # (n_μ, n_ν)         at chosen ω
    chi_head_eff: complex            # scalar
    A_wing:      np.ndarray         # (n_μ,)
    A_wingp:     np.ndarray         # (n_ν,)


@dataclass
class CoarseQDataZero:
    """Analytic dispersion tensors at the q=0 mini-cell."""
    g_mu_at_q0:  np.ndarray         # (n_μ,)              ζ_{q=0,μ}(G=0)
    V_body_q0:   np.ndarray         # (n_μ, n_ν)          (G=G'=0 zeroed)
    W_body0_q0:  np.ndarray         # (n_μ, n_ν)
    S_eff:       np.ndarray         # (3, 3)
    B:           np.ndarray         # (3, n_μ)
    Bp:          np.ndarray         # (3, n_ν)


def reconstruct_V_munu(
    g_mu: np.ndarray,               # (n_μ,)
    V_body: np.ndarray,              # (n_μ, n_ν)
    v_head_value: float,            # v(Q) — analytic
) -> np.ndarray:
    """V(Q) = V_body(Q) + g*_μ(Q) · v_head(Q) · g_ν(Q)."""
    return V_body + v_head_value * jnp.einsum('m,n->mn', jnp.conj(g_mu), g_mu)


def reconstruct_W_munu(
    g_mu: np.ndarray,
    W_body0: np.ndarray,
    W_head_scalar: complex,
    A_wing: np.ndarray,
    A_wingp: np.ndarray,
) -> np.ndarray:
    """W(Q,ω) = W_body0 + (g* + A_wing) · W_head(Q,ω) · (g + A_wing').

    Equivalent to the five-term decomposition  W_body + g*W_h g + A_wing W_h g
    + g* W_h A_wing' + A_wing W_h A_wing'  in the user's spec — collected to
    one outer product for clarity.
    """
    left  = jnp.conj(g_mu) + A_wing
    right = g_mu          + A_wingp
    return W_body0 + W_head_scalar * jnp.einsum('m,n->mn', left, right)


def W_head_scalar_from_chi_eff(
    v_head_value: float,
    chi_head_eff: complex,
) -> complex:
    """W_head(Q,ω) = v_head / (1 − v_head · χ_head_eff)."""
    return v_head_value / (1.0 - v_head_value * chi_head_eff)


def reconstruct_W_at_target_Q_nonzero(
    coarse: CoarseQDataNonzero,
    v_head_value: float,
) -> Tuple[np.ndarray, np.ndarray, complex]:
    """End-to-end reconstruction at a nonzero target Q given the coarse
    interpolation tensors at that Q (already interpolated, in the caller).

    Returns (V(Q), W(Q,ω), W_head_scalar(Q,ω)).
    """
    V = reconstruct_V_munu(coarse.g_mu, coarse.V_body, v_head_value)
    Wh = W_head_scalar_from_chi_eff(v_head_value, coarse.chi_head_eff)
    W = reconstruct_W_munu(coarse.g_mu, coarse.W_body0, Wh,
                           coarse.A_wing, coarse.A_wingp)
    return V, W, Wh


def reconstruct_W_at_target_Q_zero_cell(
    coarse: CoarseQDataZero,
    qprime_cart: np.ndarray,        # (3,) cartesian small displacement from Γ
    v_head_value: float,            # v_head(qprime) — caller picks 3D / 2D
) -> Tuple[np.ndarray, np.ndarray, complex]:
    """Reconstruction inside the q=0 mini-BZ cell.  ``qprime_cart`` is the
    displacement from Γ in cartesian Bohr⁻¹.

    Uses the analytic q'→0 expansion of (χ_head_eff, A_wing, A_wing'):

        χ_head_eff(q')  ≈  q'_a S_eff_ab q'_b
        A_wing(q')      ≈  q'_a B_{a,·}
        A_wing'(q')     ≈  q'_a B'_{a,·}

    g_μ(q') is taken to first order as g_μ(0) — for the head/wing
    structure this is the right leading behaviour because g_μ(q) = g_μ(0)
    + O(q) and the head amplification by v_head ~ 1/q² makes the leading
    g(0)·g(0) the dominant non-analytic piece.

    Returns (V(q'), W(q', ω), W_head(q', ω)).
    """
    qp = np.asarray(qprime_cart, dtype=np.float64)
    chi_eff = complex(qp @ coarse.S_eff @ qp)
    A_w  = qp @ coarse.B            # (n_μ,)
    A_wp = qp @ coarse.Bp           # (n_ν,)

    V = reconstruct_V_munu(coarse.g_mu_at_q0, coarse.V_body_q0, v_head_value)
    Wh = W_head_scalar_from_chi_eff(v_head_value, chi_eff)
    W = reconstruct_W_munu(coarse.g_mu_at_q0, coarse.W_body0_q0, Wh,
                           A_w, A_wp)
    return V, W, Wh


# ════════════════════════════════════════════════════════════════════════
#  Crude Fourier interpolation between coarse q's
# ════════════════════════════════════════════════════════════════════════
#
# This is a deliberately simple stand-in for the production interpolator —
# the bse_setup htransform path interpolates ψ via the same pattern. For
# smooth quantities like g_μ(Q), V_body(Q), W^0_body(Q), and the smooth
# wing combinations, a real-space Fourier kernel is a reasonable first
# choice; the singular Coulomb head is *not* in this list — it is
# reconstructed analytically per Q.

def fourier_interpolate_coarse_to_fine(
    values_coarse: np.ndarray,      # shape (nkx_c, nky_c, nkz_c, *trailing)
    fine_kgrid: Tuple[int, int, int],
) -> np.ndarray:
    """Interpolate a smooth coarse-q tensor onto a finer uniform Q-grid.

    Uses an FFT to real-space (R-space) representation of the coarse data,
    then evaluates the inverse FT at the fine Q-points.  For a small coarse
    grid this is exact spectral interpolation; for non-uniform target Q
    points the caller can use the same R-coefficients with a direct sum.

    Inputs
    ------
    values_coarse : (nkx_c, nky_c, nkz_c, *T)
        Coarse-grid samples of any complex-valued tensor smooth in Q.
    fine_kgrid : (Nx, Ny, Nz)
        Target uniform fine grid (must be a multiple of the coarse grid in
        each direction for trivial padding; else use the explicit-Q path).

    Returns
    -------
    values_fine : (Nx, Ny, Nz, *T)
    """
    nkx_c, nky_c, nkz_c = values_coarse.shape[:3]
    Nx, Ny, Nz = fine_kgrid
    if (Nx % nkx_c, Ny % nky_c, Nz % nkz_c) != (0, 0, 0):
        raise NotImplementedError(
            "Fine grid must be an integer multiple of the coarse grid in this "
            "PoC; for arbitrary Q targets use the direct-sum interpolation "
            "with the same R-space kernel.")
    # FT to R-space, zero-pad in R, IFT back: this is the standard "spectral
    # zero-padding" upscale.  Effectively assumes the function is smooth and
    # bandlimited by the coarse grid — fine for the smooth interpolation
    # targets here, NOT fine for the singular Coulomb head (which is why we
    # split it off analytically).
    # Use ifftn on the q-axes (BGW convention: q-space is forward FFT).
    R = np.fft.ifftn(values_coarse, axes=(0, 1, 2))
    # Pad to (Nx, Ny, Nz) preserving the FFT-frequency layout.
    pad_shape = (Nx, Ny, Nz) + values_coarse.shape[3:]
    R_padded = np.zeros(pad_shape, dtype=values_coarse.dtype)
    # Copy the four corners (positive + negative frequencies) into the pad.
    half_x = nkx_c // 2; half_y = nky_c // 2; half_z = nkz_c // 2
    # (Simple corner copy — works for even coarse grids.  For odd coarse
    # grids the Nyquist split is one-sided; this PoC asserts even.)
    if any(c % 2 for c in (nkx_c, nky_c, nkz_c)):
        raise NotImplementedError("Coarse grid axes must be even in this PoC.")
    sx = slice; sy = slice; sz = slice  # readability
    R_padded[:half_x, :half_y, :half_z]       = R[:half_x, :half_y, :half_z]
    R_padded[-half_x:, :half_y, :half_z]      = R[half_x:, :half_y, :half_z]
    R_padded[:half_x, -half_y:, :half_z]      = R[:half_x, half_y:, :half_z]
    R_padded[:half_x, :half_y, -half_z:]      = R[:half_x, :half_y, half_z:]
    R_padded[-half_x:, -half_y:, :half_z]     = R[half_x:, half_y:, :half_z]
    R_padded[-half_x:, :half_y, -half_z:]     = R[half_x:, :half_y, half_z:]
    R_padded[:half_x, -half_y:, -half_z:]     = R[:half_x, half_y:, half_z:]
    R_padded[-half_x:, -half_y:, -half_z:]    = R[half_x:, half_y:, half_z:]
    # Renormalise so that ifft/fft round-trips at the same magnitudes.
    scale = (Nx * Ny * Nz) / (nkx_c * nky_c * nkz_c)
    return np.fft.fftn(R_padded, axes=(0, 1, 2)) * scale


# ════════════════════════════════════════════════════════════════════════
#  Synthetic smoke test — validates the q'→0 limit against rank-1 head
# ════════════════════════════════════════════════════════════════════════

def _smoke_test_q0_limit():
    """Check that the q=0 reconstruction reduces to the rank-1 head update
    used today in ``gw.head_correction.apply_q0_head_rank1`` when:

        χ_head_eff = 0     (no induced screening of the head)
        A_wing = 0         (no body coupling)
        A_wing' = 0
        W_body0(q=0) = V_body(q=0)
        v_head_value = vh = some scalar
        W_head_scalar = vh / (1 − vh·0) = vh

    Then  W(0) = V_body + g* vh g  =  V(0) — exactly what the rank-1 update
    achieves on the body-only V/W stored in the restart.
    """
    rng = np.random.default_rng(0)
    n_mu = 8
    g0 = (rng.standard_normal(n_mu) + 1j * rng.standard_normal(n_mu)) / np.sqrt(n_mu)
    V_body0 = (rng.standard_normal((n_mu, n_mu)) + 1j * rng.standard_normal((n_mu, n_mu)))
    V_body0 = (V_body0 + V_body0.conj().T) / 2          # Hermitian
    vh = 1.7

    coarse = CoarseQDataZero(
        g_mu_at_q0=g0,
        V_body_q0=V_body0,
        W_body0_q0=V_body0,                               # zero screening regime
        S_eff=np.zeros((3, 3), dtype=np.complex128),
        B=np.zeros((3, n_mu), dtype=np.complex128),
        Bp=np.zeros((3, n_mu), dtype=np.complex128),
    )

    qprime = np.array([1e-3, 0.0, 0.0])                   # tiny — only direction matters
    V_recon, W_recon, Wh = reconstruct_W_at_target_Q_zero_cell(
        coarse, qprime, v_head_value=vh)

    # Reference rank-1 update (mirroring apply_q0_head_rank1):
    g0g0 = jnp.einsum('m,n->mn', np.conj(g0), g0)
    V_ref = V_body0 + vh * np.array(g0g0)
    W_ref = V_body0 + Wh * np.array(g0g0)                # Wh ≡ vh in this regime

    err_V = float(np.max(np.abs(np.array(V_recon) - V_ref)))
    err_W = float(np.max(np.abs(np.array(W_recon) - W_ref)))
    err_Wh = float(abs(complex(Wh) - vh))
    print(f"[smoke] |W_recon-W_ref|_∞  = {err_W:.3e}")
    print(f"[smoke] |V_recon-V_ref|_∞  = {err_V:.3e}")
    print(f"[smoke] |Wh − vh|          = {err_Wh:.3e}")
    assert err_V < 1e-12, err_V
    assert err_W < 1e-12, err_W
    assert err_Wh < 1e-14, err_Wh

    # Now turn on a nontrivial S_eff: reconstruction should reproduce the
    # closed-form W_head(q') = vh / (1 − vh · q'^a S_eff_ab q'^b).
    S_eff = np.array([[2.0+0.1j, 0.3, 0.0],
                      [0.3,      1.5, 0.0],
                      [0.0,      0.0, 0.8]], dtype=np.complex128)
    S_eff = (S_eff + S_eff.conj().T) / 2
    coarse2 = CoarseQDataZero(
        g_mu_at_q0=g0,
        V_body_q0=V_body0,
        W_body0_q0=V_body0,
        S_eff=S_eff,
        B=np.zeros((3, n_mu), dtype=np.complex128),
        Bp=np.zeros((3, n_mu), dtype=np.complex128),
    )
    qp = np.array([0.07, -0.04, 0.02])
    _, _, Wh2 = reconstruct_W_at_target_Q_zero_cell(coarse2, qp, v_head_value=vh)
    chi_eff_explicit = complex(qp @ S_eff @ qp)
    Wh_ref = vh / (1.0 - vh * chi_eff_explicit)
    err_Wh2 = abs(complex(Wh2) - Wh_ref)
    print(f"[smoke] |Wh(q') − vh/(1−vh q'S q')| = {err_Wh2:.3e}")
    assert err_Wh2 < 1e-12, err_Wh2

    # Local-field combinations: feeding nonzero χ_wing/wing' through
    # build_chi_head_eff_and_A_wings  and feeding the result back through
    # reconstruct_W_at_target_Q_nonzero must match a direct screened-W
    # solve  (1 − V χ)^-1 V  in the bordered (n_μ + 1) basis.
    chi_head = -0.4 + 0.2j
    chi_wing  = (rng.standard_normal(n_mu) + 1j * rng.standard_normal(n_mu)) * 0.05
    chi_wingp = (rng.standard_normal(n_mu) + 1j * rng.standard_normal(n_mu)) * 0.05
    chi_body  = (rng.standard_normal((n_mu, n_mu)) +
                 1j * rng.standard_normal((n_mu, n_mu))) * 0.05
    chi_body  = (chi_body + chi_body.conj().T) / 2

    W_body0 = solve_W_body0(V_body0, chi_body)            # (n_μ, n_μ)
    chi_eff, A_w, A_wp = build_chi_head_eff_and_A_wings(
        chi_head, chi_wing, chi_wingp, W_body0)
    Wh = W_head_scalar_from_chi_eff(vh, chi_eff)
    W_recon = reconstruct_W_munu(g0, W_body0, Wh, A_w, A_wp)

    # Bordered reference: solve the (1+n_μ)-dim Dyson with V block-diagonal
    # (head and body decoupled in the BARE Coulomb), χ fully coupled.  The
    # centroid-basis screened W_{μν} is the projection through the embedding
    # z_μ ≅ (g_μ, e_μ): each centroid has amplitude g_μ in the head channel
    # and amplitude δ in its own body coordinate.  The reconstruction in
    # this PoC must match this projection.
    n = n_mu + 1
    V_full = np.zeros((n, n), dtype=np.complex128)
    chi_full = np.zeros((n, n), dtype=np.complex128)
    V_full[0, 0]   = vh                              # bare head Coulomb
    V_full[1:, 1:] = V_body0                          # bare body Coulomb (G_Q-excluded)

    chi_full[0, 0] = chi_head
    chi_full[0, 1:] = chi_wingp
    chi_full[1:, 0] = chi_wing
    chi_full[1:, 1:] = chi_body

    W_full = np.linalg.solve(np.eye(n) - V_full @ chi_full, V_full)
    # Project the bordered W onto the centroid basis through (g_μ, e_μ):
    #   W_centroid[μ,ν] = g*_μ W_full[0,0] g_ν
    #                   + g*_μ W_full[0, ν+1]
    #                   + W_full[μ+1, 0] g_ν
    #                   + W_full[μ+1, ν+1]
    W_centroid = (
        np.einsum('m,n->mn', np.conj(g0), g0) * W_full[0, 0]
        + np.einsum('m,n->mn', np.conj(g0), W_full[0, 1:])
        + np.einsum('m,n->mn', W_full[1:, 0], g0)
        + W_full[1:, 1:]
    )
    err_full = float(np.max(np.abs(np.array(W_recon) - W_centroid)))
    print(f"[smoke] |W_recon - W_centroid(bordered Dyson)|_∞ = {err_full:.3e}")
    assert err_full < 1e-9, err_full

    print("[smoke] ALL OK")


def _smoke_test_q0_dispersion_path():
    """Validate that the q=0 dispersion-tensor reconstruction reproduces the
    explicit Schur path to O(q'²): build random S_ab, w_mu, w_nu, χ_body,
    g0, V_body0, then check that

        reconstruct_W_at_target_Q_zero_cell(coarse_zero, q')
            ≈ reconstruct_W_at_target_Q_nonzero(coarse_nonzero(q'), v_head(q'))

    when χ_wing(q') = q'_a w_mu[a,·] and χ_wing'(q') = q'_a w_nu[a,·] exactly
    (linearised model with no quadratic correction in q').
    """
    rng = np.random.default_rng(2026)
    n_mu = 6
    g0 = (rng.standard_normal(n_mu) + 1j * rng.standard_normal(n_mu)) / np.sqrt(n_mu)
    V_body0 = (rng.standard_normal((n_mu, n_mu)) + 1j * rng.standard_normal((n_mu, n_mu))) * 0.4
    V_body0 = (V_body0 + V_body0.conj().T) / 2

    chi_body0 = (rng.standard_normal((n_mu, n_mu)) +
                 1j * rng.standard_normal((n_mu, n_mu))) * 0.05
    chi_body0 = (chi_body0 + chi_body0.conj().T) / 2

    # Random S, w_mu, w_nu (3, n_μ).
    S = rng.standard_normal((3, 3)).astype(np.complex128) * 0.3
    S = (S + S.T) / 2                          # real-symmetric (typical for static)
    w_mu = (rng.standard_normal((3, n_mu)) + 1j * rng.standard_normal((3, n_mu))) * 0.1
    w_nu = (rng.standard_normal((3, n_mu)) + 1j * rng.standard_normal((3, n_mu))) * 0.1

    W_b0_q0 = solve_W_body0(V_body0, chi_body0)
    S_eff, B, Bp = build_q0_dispersion_eff(S, w_mu, w_nu, W_b0_q0)

    # Pick a small q' (cartesian).  Take vhead(q') = v3D(q') for concreteness.
    qp = np.array([0.02, -0.015, 0.01])
    # Reciprocal-space "Q" cartesian magnitude proxy: use |q'| in cartesian
    # for the v_head — since this is synthetic we don't need a real metric.
    vhead_qp = 8.0 * math.pi / float(qp @ qp)

    coarse_zero = CoarseQDataZero(
        g_mu_at_q0=g0, V_body_q0=V_body0, W_body0_q0=W_b0_q0,
        S_eff=S_eff, B=B, Bp=Bp,
    )
    V_disp, W_disp, Wh_disp = reconstruct_W_at_target_Q_zero_cell(
        coarse_zero, qp, v_head_value=vhead_qp)

    # Explicit Schur path with linearised χ at q'.
    chi_head_q  = complex(qp @ S @ qp)
    chi_wing_q  = qp @ w_mu                                     # (n_μ,)
    chi_wingp_q = qp @ w_nu
    chi_eff, A_w, A_wp = build_chi_head_eff_and_A_wings(
        chi_head_q, chi_wing_q, chi_wingp_q, W_b0_q0)
    Wh_explicit = W_head_scalar_from_chi_eff(vhead_qp, chi_eff)
    W_explicit = reconstruct_W_munu(g0, W_b0_q0, Wh_explicit, A_w, A_wp)

    err_W = float(np.max(np.abs(np.array(W_disp) - np.array(W_explicit))))
    err_Wh = abs(complex(Wh_disp) - complex(Wh_explicit))
    print(f"[smoke-disp] |Wh(q') paths| = {err_Wh:.3e}")
    print(f"[smoke-disp] |W(q') paths|  = {err_W:.3e}")
    # Differences should be at machine precision because χ is linear in q' here.
    assert err_W < 1e-11, err_W
    assert err_Wh < 1e-11, err_Wh
    print("[smoke-disp] ALL OK")


if __name__ == "__main__":
    _smoke_test_q0_limit()
    _smoke_test_q0_dispersion_path()
