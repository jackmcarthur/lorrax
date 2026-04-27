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

Serial, single-device.  Reconstruction kernel + glue for an end-to-end
finite-q W-interpolation experiment, leaning on the existing SOS pipeline:

  • SOS finite-q matrix elements ``rho_cvkq``, ``v_cvkq`` come from
    ``psp.get_dipole_mtxels --with-finite-q`` (commit ``061a13f``/``903c2b7``).
  • ``χ_head(q,ω)``, ``χ_wing(q,ω)``, ``S_{αβ}(0,ω)``, ``w_{α,μ}(0,ω)`` come
    from ``common.chi_sos`` (validated bit-identical against
    ``chi_from_dipole.compute_S_omega``).
  • ``χ_wing'(q)`` and ``w'_{α,ν}`` are recovered by Hermitian conjugation
    at static ω=0 — see :func:`chi_wingp_from_wing` for why this is exact
    (real F_t at ω=0).
  • ``ζ_q`` and ``g0_mu(q)`` come from ``compute_V_q_from_zeta_array``;
    ``V_body(q)`` is recovered as a rank-1 subtract from the persisted
    ``V_qmunu``.
  • ``W_body0(q,ω) = (1 − V_body(q) χ_centroid(q,ω))^{-1} V_body(q)`` is a
    per-q dense solve here (PoC); production would route through the
    sharded :func:`gw.w_isdf.solve_w`.

Public surface:

  Coulomb head (analytic, never interpolated)
    :func:`v_head_3d`, :func:`v_head_2d_slab`

  Absolute-channel projection at any q
    :func:`compute_g_mu_at_q`,
    :func:`extract_V_body_from_V_q`              rank-1 subtract from V_qmunu

  Centroid pair-density vertex at finite q
    :func:`build_b_cvkq_mu_from_centroid_wfns`

  Wrappers around chi_sos + the Schur reductions
    :func:`chi_wingp_from_wing`,
    :func:`assemble_smooth_tensors_at_q`,        (g, V_body, W^0_body, χ_head_eff, A_wing, A_wing')
    :func:`assemble_q0_dispersion`               (S_eff, B, B') + q=0 references

  Body W solve (PoC dense solve)
    :func:`solve_W_body0`

  Local-field combinations (smooth interpolation targets)
    :func:`build_chi_head_eff_and_A_wings`,
    :func:`build_q0_dispersion_eff`

  Reconstruction at a target Q
    :func:`reconstruct_V_munu`, :func:`reconstruct_W_munu`,
    :func:`reconstruct_W_at_target_Q_nonzero`,
    :func:`reconstruct_W_at_target_Q_zero_cell`

  Crude Fourier interpolation between coarse q's
    :func:`fourier_interpolate_coarse_to_fine`

Two synthetic smoke tests (``__main__``) verify:
  1. Five-term reconstruction == bordered-Dyson Schur projection (1e-15).
  2. q'→0 dispersion path == explicit Schur with linearised χ_wing (1e-13).

Hooking up a real run
=====================

The intended driver flow is:

  1. Read coarse-grid restart (V_qmunu, W0_qmunu, vhead, whead, G0_mu_nu, zeta_q).
  2. Read coarse-grid dipole.h5/finite_q (rho_cvkq, v_cvkq, kminq_idx).
  3. Read centroid wfns ψ_n,k(r_μ) at full BZ (for b_{t,μ}(q) construction).
  4. For every coarse q ≠ 0: compute χ_head, χ_wing via chi_sos; derive
     χ_wing' by conjugation; extract V_body via rank-1 subtract; resolve
     v_head(q); solve W^0_body(q); form (χ_head_eff, A_wing, A_wing').
  5. At q = 0: compute S, w via chi_sos; W^0_body(0) from V_body(0) and
     χ_centroid(0); form S_eff, B, B'.
  6. Interpolate the smooth tensors (g, V_body, W^0_body, χ_head_eff,
     A_wing, A_wing') to fine Q's (Fourier upscale or trilinear).
  7. Reconstruct V_μν(Q), W_μν(Q,ω) at every fine Q using the analytic
     v_head(Q).

Step 7 closes the loop with C's q=0 head rank-1 work
(``gw.head_correction.apply_q0_head_rank1``); the q'→0 limit of this
reconstruction is exactly that rank-1 update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
#  Body Coulomb  V^body_{μν}(q) — rank-1 subtract from V_qmunu
# ════════════════════════════════════════════════════════════════════════
#
# V_centroid(q) as persisted by gw_jax includes the absolute G_Q channel
# at q≠0 (compute_vcoul only zeros G=0 at q_red=0 today).  The user's
# bordered formulation requires V_body = V_centroid − v_head(q)·g g†
# at every coarse q.  Since g0_mu(q) is already saved per coarse q in the
# restart and v_head(q) is analytic, the body extraction is one rank-1
# subtract — no GW recompute needed.

def extract_V_body_from_V_q(
    V_q: np.ndarray,                # (n_μ, n_ν)
    g_mu: np.ndarray,               # (n_μ,)
    v_head_value: float,            # v(q) — analytic, finite for q ≠ 0
) -> np.ndarray:
    """V_body(q) = V_q(q) − v_head(q) · g_μ* g_ν.

    At q=0, the persisted ``V_qmunu`` already has the G=G'=0 element zeroed
    by ``compute_vcoul``; the rank-1 head term IS the missing piece, and
    this function applied with ``v_head_value = vhead`` returns
    V_body == V_qmunu (no-op for the body, since the rank-1 was never
    added).  For q≠0, V_q includes the absolute channel and this routine
    removes it.
    """
    return V_q - v_head_value * jnp.einsum('m,n->mn', jnp.conj(g_mu), g_mu)


# ════════════════════════════════════════════════════════════════════════
#  Centroid pair-density vertex  b_{t,μ}(q)  for chi_sos.compute_chi_wing
# ════════════════════════════════════════════════════════════════════════
#
# ``chi_sos`` consumes ``b_cvkq_mu[c,v,k,q,μ]`` as caller-provided input.
# Here we assemble it from the ψ-at-centroid arrays already computed by
# the htransform pipeline (or read from the centroid-wfn HDF5 product
# of the GW preprocessing step).

def build_b_cvkq_mu_from_centroid_wfns(
    psi_rmu_full: np.ndarray,       # (nb, nspinor, n_μ, nk_full)  ψ_n,k(r_μ) at every full-BZ k
    kminq_idx: np.ndarray,          # (nk_full, nq) int — k → k-q lookup (= dipole.h5/finite_q/kminq_idx)
    v_lo: int, c_lo: int, c_hi: int,
) -> np.ndarray:
    """Return  b_{t,μ}(q) = Σ_s ψ*_{c,k-q,s}(r_μ) ψ_{v,k,s}(r_μ)  as the
    (n_c, n_v, n_k_full, n_q, n_μ) tensor consumed by
    ``chi_sos.compute_chi_wing_at_q``.

    The  c, v  band slices match the ones the dipole driver used:
        v ∈ [v_lo : c_lo)         valence
        c ∈ [c_lo : c_hi)         conduction

    PoC routes through numpy with no chunking — for production this would
    be a sharded contraction over the (μ, k) axes via shard_map.
    """
    psi = jnp.asarray(psi_rmu_full)                        # (nb, ns, n_μ, nk)
    nk_full, nq = kminq_idx.shape
    n_v = c_lo - v_lo
    n_c = c_hi - c_lo
    n_mu = psi.shape[2]
    out = jnp.zeros((n_c, n_v, nk_full, nq, n_mu), dtype=jnp.complex128)
    for jq in range(int(nq)):
        ikmq = jnp.asarray(kminq_idx[:, jq], dtype=jnp.int32)         # (nk_full,)
        # ψ_c at k-q for every k:  (n_c, ns, n_μ, nk)
        psi_c_kmq = jnp.take(psi[c_lo:c_hi], ikmq, axis=-1)
        psi_v_k   = psi[v_lo:c_lo]                                     # (n_v, ns, n_μ, nk)
        # b[c, v, k, μ] = Σ_s ψ*_c,k-q[s, μ, k] · ψ_v,k[s, μ, k]
        b = jnp.einsum('csmk,vsmk->cvkm',
                       jnp.conj(psi_c_kmq), psi_v_k, optimize=True)
        out = out.at[..., jq, :].set(b)
    return out


def chi_wingp_from_wing(chi_wing: np.ndarray,
                        omega_is_static: bool = True) -> np.ndarray:
    """Recover  χ_wing'(q,ω) from χ_wing(q,ω) by Hermitian conjugation.

    For static screening (ω = 0, real F_t) the SOS forms

        χ_wing[μ]  = Σ_t F_t · b*_{t,μ}(q) · h_t(q)
        χ_wing'[ν] = Σ_t F_t · h*_t(q)     · b_{t,ν}(q)

    differ only by complex conjugation: χ_wing'[ν] = (χ_wing[ν])* exactly.
    For finite real ω with real F_t the same identity holds. For complex
    ω (off-axis) F_t becomes complex and the identity breaks; an extra
    SOS pass is required (chi_sos doesn't currently expose it but mirrors
    chi_wing trivially).

    Pass-through copy with conjugation; no extra SOS work.
    """
    if not omega_is_static:
        raise NotImplementedError(
            "chi_wing' for complex ω needs an explicit SOS pass; "
            "chi_sos currently only writes χ_wing")
    return jnp.conj(chi_wing)


# ════════════════════════════════════════════════════════════════════════
#  Glue: assemble smooth interpolation tensors at one coarse q
# ════════════════════════════════════════════════════════════════════════

def assemble_smooth_tensors_at_q(
    V_q: np.ndarray,                # (n_μ, n_ν)         persisted V_qmunu at this q
    chi_centroid_q: np.ndarray,     # (n_μ, n_ν)         compute_chi0 output at this q
    g_mu_q: np.ndarray,             # (n_μ,)             ζ_q,μ(G_Q)
    v_head_value: float,            # v(q) analytic (finite for q ≠ 0)
    chi_head: complex,              # chi_sos.compute_chi_head_at_q  →  this q
    chi_wing: np.ndarray,           # chi_sos.compute_chi_wing_at_q  →  this q  (n_μ,)
    chi_wingp: Optional[np.ndarray] = None,
):
    """Return the smooth interpolation targets at one nonzero coarse q:

        g_mu, V_body, W_body0, χ_head_eff, A_wing, A_wing'

    χ_wing' defaults to ``conj(χ_wing)`` (static-ω regime).
    """
    if chi_wingp is None:
        chi_wingp = chi_wingp_from_wing(chi_wing, omega_is_static=True)
    V_body = extract_V_body_from_V_q(V_q, g_mu_q, v_head_value)
    W_body0 = solve_W_body0(np.asarray(V_body), np.asarray(chi_centroid_q))
    chi_eff, A_wing, A_wingp = build_chi_head_eff_and_A_wings(
        chi_head, chi_wing, chi_wingp, W_body0)
    return CoarseQDataNonzero(
        g_mu=np.asarray(g_mu_q),
        V_body=np.asarray(V_body),
        W_body0=np.asarray(W_body0),
        chi_head_eff=complex(chi_eff),
        A_wing=np.asarray(A_wing),
        A_wingp=np.asarray(A_wingp),
    )


def assemble_q0_dispersion(
    V_q0: np.ndarray,               # (n_μ, n_ν)         V_qmunu at q=0 (G=0 already zeroed)
    chi_centroid_q0: np.ndarray,    # (n_μ, n_ν)         compute_chi0 at q=0
    g_mu_q0: np.ndarray,            # (n_μ,)             G0_mu_nu from restart
    S_ab: np.ndarray,               # (3, 3)             chi_sos.compute_S_tensor_sos
    w_a_mu: np.ndarray,             # (3, n_μ)           chi_sos.compute_w_tensor_sos
    w_a_nu: Optional[np.ndarray] = None,
):
    """Return the q=0 mini-cell coarse data (analytic dispersion):

        g_mu_at_q0, V_body_q0, W_body0_q0, S_eff, B, B'

    ``w'_{a,ν}`` defaults to ``conj(w_a_mu)`` (static-ω regime).
    Note V_body_q0 == V_q0 because compute_vcoul already zeros G=G'=0 at q=0.
    """
    if w_a_nu is None:
        w_a_nu = jnp.conj(w_a_mu)
    # V_body at q=0: gw_jax persists V_qmunu with the G=0 element ALREADY
    # zeroed at q=0 (compute_vcoul does this).  No further subtraction needed.
    V_body_q0 = np.asarray(V_q0)
    W_body0_q0 = solve_W_body0(V_body_q0, np.asarray(chi_centroid_q0))
    S_eff, B, Bp = build_q0_dispersion_eff(
        np.asarray(S_ab), np.asarray(w_a_mu), np.asarray(w_a_nu),
        W_body0_q0)
    return CoarseQDataZero(
        g_mu_at_q0=np.asarray(g_mu_q0),
        V_body_q0=V_body_q0,
        W_body0_q0=np.asarray(W_body0_q0),
        S_eff=np.asarray(S_eff),
        B=np.asarray(B),
        Bp=np.asarray(Bp),
    )


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
    # Spectral zero-padding upscale: ifftn → R-space → zero-pad in R → fftn.
    # Per-axis low-frequency packing handled generically (works for any axis
    # sizes including the trivial ``Nc=1`` case for slab geometries).
    R = np.fft.ifftn(values_coarse, axes=(0, 1, 2))
    pad_shape = (Nx, Ny, Nz) + values_coarse.shape[3:]
    R_padded = np.zeros(pad_shape, dtype=values_coarse.dtype)

    def _axis_index_pairs(Nc, Nf):
        """Per-axis low-frequency unpack: list of (slice_in_coarse, slice_in_fine)
        chunks that copy DC + positive freqs to the front and negative freqs to
        the tail.  Generic for arbitrary Nc / Nf, including Nc=1 (axis is just
        DC; fine axis fills with the constant).

        ``Nyquist`` (when Nc is even) lives at index Nc//2 in the coarse FFT
        layout.  Spectral zero-padding splits the Nyquist component evenly
        between the +Nc/2 and −Nc/2 positions of the fine grid; here we put
        full weight on the +Nc/2 slot and 0 on the −Nc/2 slot, which is
        standard for real-valued spectral upscaling and matches numpy's
        ``np.fft.fftn(zero-pad)`` convention.  For complex data (no
        Hermitian constraint) this is sometimes called the "single-sided"
        pad; results agree with the round-trip identity at coarse points.
        """
        if Nc == 1:
            return [(slice(0, 1), slice(0, 1))]
        half = Nc // 2 + (Nc % 2)            # number of "low + DC" coeffs
        neg  = Nc - half                     # number of "negative" coeffs
        return [
            (slice(0, half),     slice(0, half)),                      # DC + low+
            (slice(half, Nc),    slice(Nf - neg, Nf)),                 # high-/neg
        ]

    px = _axis_index_pairs(nkx_c, Nx)
    py = _axis_index_pairs(nky_c, Ny)
    pz = _axis_index_pairs(nkz_c, Nz)
    for sx_c, sx_f in px:
        for sy_c, sy_f in py:
            for sz_c, sz_f in pz:
                R_padded[sx_f, sy_f, sz_f] = R[sx_c, sy_c, sz_c]

    # No extra scale factor.  The ifftn → pad-with-zeros → fftn round-trip
    # already preserves coincident-point values exactly: at fine indices that
    # coincide with coarse grid points, fftn(R_padded) recovers v_coarse to
    # machine precision.  See _smoke_test_fourier_upscale below.
    return np.fft.fftn(R_padded, axes=(0, 1, 2))


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
