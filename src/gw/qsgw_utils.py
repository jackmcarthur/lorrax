"""Diagonal-Σ(E) fixed point, QSGW Σ_xc build, and SC-COHSEX diagnostics.

The post-self-energy plumbing in ``gw_jax`` operates on **replicated**
``(nk, nb, nb)`` arrays uniformly; the only object that must remain
sharded is the dynamic correlation ``Σ_c(ω, k, m_X, n_Y)`` produced by
``ppm_sigma`` because its ω-axis fan-out makes the full tensor too
large to replicate.  Everything in this module is structured around
that seam:

- :func:`solve_diagonal_sigma_fixed_point` runs on host NumPy with
  vectorised linear interpolation over the (nk, nb) energy grid.  Its
  input ``Σ_diag(ω, k, n)`` is small enough to live replicated.
- :func:`build_qsgw_sigma_xc` is a JIT'd JAX kernel that takes the
  on-device sharded ``Σ_c(ω)`` and the QP energies ``E_kn`` (replicated)
  and returns the Hermitised QSGW Σ_xc replicated on the mesh.  No disk
  round-trip; restart-friendly because the same kernel can be re-called
  with a refreshed ``Σ_c(ω)`` and ``E_kn`` at every QSGW iteration.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# SC-COHSEX diagnostics
# ---------------------------------------------------------------------------

def print_scf_diagnostics(Gij_final, U_full, nelec, nb_sigma, print_fn=print):
    """Print Gij trace, U unitarity, and mixing diagnostics for SC-COHSEX."""
    Gij_trace = float(jnp.real(jnp.trace(Gij_final[0])))
    print_fn(f"[SC] Gij trace at k=0: {Gij_trace:.4f} (should be {nelec})")
    unitarity_err = float(jnp.max(jnp.abs(
        jnp.einsum('kim,kin->kmn', jnp.conj(U_full[0:1]), U_full[0:1])[0]
        - jnp.eye(nb_sigma))))
    print_fn(f"[SC] U unitarity error at k=0: {unitarity_err:.2e}")
    U_diag = jnp.abs(jnp.diagonal(U_full[0]))
    print_fn(f"[SC] |U| diagonal[:5] at k=0: {np.array(U_diag[:5])}")


# ---------------------------------------------------------------------------
# Diagonal-Σ(E) fixed point  (host NumPy, vectorised)
# ---------------------------------------------------------------------------

def solve_diagonal_sigma_fixed_point(
    h0_diag_ev: np.ndarray,
    sigma_omega_diag_ev: np.ndarray,
    omega_ev: np.ndarray,
    *,
    max_iter: int = 80,
    tol_ev: float = 1.0e-6,
    mixing: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve E = h0 + Re Σ(E) per (k, n) by linear mixing.

    Vectorised over (k, n): each iteration performs one ``np.searchsorted``
    + two fancy-index gathers over the ω-axis, no Python (k, n) loop.

    Parameters
    ----------
    h0_diag_ev : (nk, nb)
        Static one-body diagonal (typically ``diag(kin_ion + V_H)``) in eV.
    sigma_omega_diag_ev : (nω, nk, nb), complex
        Diagonal Σ_xc(ω) in eV.  Caller is responsible for adding the
        static Σ_x diagonal to the dynamic Σ_c diagonal before invocation.
    omega_ev : (nω,)
        ω-grid in eV, monotonically increasing.

    Returns
    -------
    E : (nk, nb)
        Converged QP eigenvalues in eV.  Bands whose final fixed-point
        argument lies outside ``[ω_min, ω_max]`` are clipped at the grid
        edge for the Σ evaluation; callers should patch out-of-grid bands
        via the scissor (see :mod:`gw.scissor`) if a flatter extrapolation
        is desired.
    converged : (nk, nb), bool
        Per-band convergence flag from the final iteration.
    n_iter : int
        Iterations performed (≤ ``max_iter``).
    """
    h0 = np.asarray(h0_diag_ev, dtype=np.float64)
    sigma_w = np.asarray(sigma_omega_diag_ev, dtype=np.complex128)
    omega = np.asarray(omega_ev, dtype=np.float64)
    if sigma_w.ndim != 3:
        raise ValueError("sigma_omega_diag_ev must have shape (nω, nk, nb).")
    n_omega, nk, nb = sigma_w.shape
    if h0.shape != (nk, nb):
        raise ValueError(
            f"shape mismatch: h0={h0.shape}, sigma_w={sigma_w.shape}")

    omega_lo = float(omega[0])
    omega_hi = float(omega[-1])
    k_idx = np.arange(nk)[:, None]
    n_idx = np.arange(nb)[None, :]

    def _sigma_at(E_kn: np.ndarray) -> np.ndarray:
        Ec = np.clip(E_kn, omega_lo, omega_hi)
        idx_hi = np.clip(np.searchsorted(omega, Ec, side="left"), 1, n_omega - 1)
        idx_lo = idx_hi - 1
        ω_lo = omega[idx_lo]
        ω_hi = omega[idx_hi]
        denom = np.where(ω_hi > ω_lo, ω_hi - ω_lo, 1.0)
        w_hi = (Ec - ω_lo) / denom
        w_lo = 1.0 - w_hi
        # Σ_diag(ω) is (nω, nk, nb); fancy-index along ω with per-(k, n) idx.
        sig_lo = sigma_w[idx_lo, k_idx, n_idx]
        sig_hi = sigma_w[idx_hi, k_idx, n_idx]
        return w_lo * sig_lo + w_hi * sig_hi

    i0 = int(np.argmin(np.abs(omega)))
    E = h0 + np.real(sigma_w[i0])
    mix = float(np.clip(mixing, 0.0, 1.0))

    for it in range(max_iter):
        E_new = h0 + np.real(_sigma_at(E))
        E_next = (1.0 - mix) * E + mix * E_new
        diff = np.abs(E_next - E)
        E = E_next
        if bool(np.all(diff < tol_ev)):
            return E, diff < tol_ev, it + 1
    return E, np.abs(E_new - E) < tol_ev, max_iter


# ---------------------------------------------------------------------------
# Σ diagonal extraction from sharded Σ_c(ω, k, m_X, n_Y)
# ---------------------------------------------------------------------------

def extract_sigma_diag_replicated(
    sigma_w_kij: jax.Array,
    mesh_xy: Mesh,
) -> jax.Array:
    """Pull ``Σ[..., n, n]`` from a sharded matrix-valued ω-tensor.

    Input ``sigma_w_kij`` is sharded ``P(None, None, 'x', 'y')`` for
    ``(nω, nk, nb, nb)``.  The (m, n) axes are on **different** mesh
    axes, so a naive ``einsum('...ii->...i')`` computes a per-shard
    block-diagonal which is the global diagonal only on the
    ``ix == iy`` shards — off-diagonal mesh shards silently produce
    garbage off-diagonal values.  We sidestep that by forcing an
    allgather of the full ω-tensor onto each device before the trace.

    Memory note: the materialised tensor is ``nω · nk · nb² · 16 B``
    per device (≈ 270 MB for MoS2 4×4×1, 80 bands, 41 ω-points).  Fits
    comfortably in the 28 GB device budget; a shard_map specialisation
    for the very-large-system case is left for follow-up.
    """
    rep_3d = NamedSharding(mesh_xy, P(None, None, None))
    rep_4d = NamedSharding(mesh_xy, P(None, None, None, None))

    @jax.jit
    def _extract(M):
        M_full = jax.lax.with_sharding_constraint(M, rep_4d)
        diag = jnp.einsum("...ii->...i", M_full)
        return jax.lax.with_sharding_constraint(diag, rep_3d)

    return _extract(sigma_w_kij)


# ---------------------------------------------------------------------------
# QSGW Σ_xc build — sharded ω-tensor + replicated E_kn → replicated (k, m, n)
# ---------------------------------------------------------------------------

def build_qsgw_sigma_xc(
    sigma_c_omega_ry: jax.Array,
    sigma_x_kij_ry: jax.Array,
    omega_ev: np.ndarray,
    e_qp_kn_ev: np.ndarray,
    mesh_xy: Mesh,
) -> tuple[jax.Array, dict[str, float]]:
    """Build the static Hermitian QSGW Σ_xc[k, m, n].

    Implements the standard QSGW ansatz

        Σ_xc^QSGW_ij(k) = ½[ Σ_xc_ij(k, E_i(k)) + Σ_xc_ij(k, E_j(k)) ]ʰ

    where ``[·]ʰ`` denotes the Hermitian part (real-symmetrisation against
    interpolation noise).  Σ_xc = Σ_c + Σ_x; Σ_x is ω-independent so it
    is added once after the Σ_c(ω) interpolation.

    Energy domain
    -------------
    ``omega_ev`` and ``e_qp_kn_ev`` must share a common reference (typically
    Fermi-relative — ω-grid is centered on E_F by construction in
    ``ppm_pipeline``, and ``E_qp - E_F`` is what the diagonal fixed point
    produces after applying the scissor).

    Sharding
    --------
    - ``sigma_c_omega_ry`` is consumed in place at its native sharding
      ``P(None, None, 'x', 'y')``; the per-shard ``take_along_axis``
      gather is local because the ω-axis (over which we interp) is fully
      replicated and ``e_qp_kn_ev`` is broadcast to all shards.
    - The intermediate ``A``, ``B`` arrays inherit ``P(None, 'x', 'y')``
      sharding for ``(k, m, n)``.
    - The final result is forced replicated via
      ``with_sharding_constraint`` before Hermitisation, so the
      ``½(M + M†)`` step doesn't generate cross-shard transpose comms.

    Returns
    -------
    sigma_xc_qsgw_kij_ry : jax.Array, (nk, nb, nb), complex128, replicated.
    diagnostics : dict with ``n_clipped`` (count of ``E_kn`` outside
        ``[ω_min, ω_max]`` clamped to the grid) and ``omega_min/max_ev``.
    """
    omega = np.asarray(omega_ev, dtype=np.float64)
    E = np.asarray(e_qp_kn_ev, dtype=np.float64)
    if sigma_c_omega_ry.ndim != 4:
        raise ValueError(
            f"sigma_c_omega_ry must have shape (nω, nk, nb, nb); "
            f"got {sigma_c_omega_ry.shape}.")
    if sigma_x_kij_ry.ndim != 3:
        raise ValueError(
            f"sigma_x_kij_ry must have shape (nk, nb, nb); "
            f"got {sigma_x_kij_ry.shape}.")
    n_omega, nk, nb, nb2 = sigma_c_omega_ry.shape
    if nb != nb2 or sigma_x_kij_ry.shape != (nk, nb, nb):
        raise ValueError(
            f"shape mismatch: sigma_c={sigma_c_omega_ry.shape}, "
            f"sigma_x={sigma_x_kij_ry.shape}")
    if E.shape != (nk, nb):
        raise ValueError(
            f"e_qp_kn_ev must have shape ({nk}, {nb}); got {E.shape}.")

    # Linear-interp index/weight arrays, host-side then pushed replicated.
    omega_lo = float(omega[0])
    omega_hi = float(omega[-1])
    E_clamped = np.clip(E, omega_lo, omega_hi)
    n_clipped = int(np.count_nonzero(E_clamped != E))
    idx_hi = np.clip(np.searchsorted(omega, E_clamped, side="left"),
                     1, n_omega - 1)
    idx_lo = idx_hi - 1
    ω_lo = omega[idx_lo]
    ω_hi = omega[idx_hi]
    denom = np.where(ω_hi > ω_lo, ω_hi - ω_lo, 1.0)
    w_hi = (E_clamped - ω_lo) / denom
    w_lo = 1.0 - w_hi

    rep_2d = NamedSharding(mesh_xy, P(None, None))
    rep_3d = NamedSharding(mesh_xy, P(None, None, None))
    idx_lo_j = jax.device_put(jnp.asarray(idx_lo, dtype=jnp.int32), rep_2d)
    idx_hi_j = jax.device_put(jnp.asarray(idx_hi, dtype=jnp.int32), rep_2d)
    w_lo_j   = jax.device_put(jnp.asarray(w_lo, dtype=jnp.complex128), rep_2d)
    w_hi_j   = jax.device_put(jnp.asarray(w_hi, dtype=jnp.complex128), rep_2d)

    @jax.jit
    def _kernel(sig_w, sig_x, ilo, ihi, wlo, whi):
        # ilo/ihi/wlo/whi: (nk, nb) replicated; sig_w: (nω, nk, nb_m_X, nb_n_Y).
        # A[k, m, n] = Σ_c[idx[k, m], k, m, n] (interp at E_m(k))
        # B[k, m, n] = Σ_c[idx[k, n], k, m, n] (interp at E_n(k))
        full = sig_w.shape  # (nω, nk, nb, nb)

        ilo_m = jnp.broadcast_to(ilo[None, :, :, None], full)
        ihi_m = jnp.broadcast_to(ihi[None, :, :, None], full)
        A_lo = jnp.take_along_axis(sig_w, ilo_m, axis=0)[0]
        A_hi = jnp.take_along_axis(sig_w, ihi_m, axis=0)[0]
        A = wlo[:, :, None] * A_lo + whi[:, :, None] * A_hi

        ilo_n = jnp.broadcast_to(ilo[None, :, None, :], full)
        ihi_n = jnp.broadcast_to(ihi[None, :, None, :], full)
        B_lo = jnp.take_along_axis(sig_w, ilo_n, axis=0)[0]
        B_hi = jnp.take_along_axis(sig_w, ihi_n, axis=0)[0]
        B = wlo[:, None, :] * B_lo + whi[:, None, :] * B_hi

        # Half-sum, then add static Σ_x and force replicated before
        # Hermitisation (avoids a sharded transpose).
        M = 0.5 * (A + B) + sig_x
        M = jax.lax.with_sharding_constraint(M, rep_3d)
        return 0.5 * (M + jnp.conj(jnp.swapaxes(M, -1, -2)))

    sigma_xc_qsgw = _kernel(
        sigma_c_omega_ry, sigma_x_kij_ry,
        idx_lo_j, idx_hi_j, w_lo_j, w_hi_j,
    )
    sigma_xc_qsgw.block_until_ready()

    diagnostics = {
        "n_clipped": float(n_clipped),
        "frac_clipped": float(n_clipped) / float(nk * nb) if nk * nb else 0.0,
        "omega_min_ev": omega_lo,
        "omega_max_ev": omega_hi,
    }
    return sigma_xc_qsgw, diagnostics


# ---------------------------------------------------------------------------
# QP-energy comparison plot
# ---------------------------------------------------------------------------

def plot_qp_energy_comparison(
    output_png: str,
    e_ref_kn_ev: np.ndarray,
    e_static_kn_ev: np.ndarray,
    e_dyn0_kn_ev: np.ndarray,
    e_diag_sc_kn_ev: np.ndarray,
) -> str:
    """Scatter + k=0 trend of QP energies for the three approximations."""
    import matplotlib.pyplot as plt

    x = np.asarray(e_ref_kn_ev, dtype=np.float64).reshape(-1)
    y_static = np.asarray(e_static_kn_ev, dtype=np.float64).reshape(-1)
    y_dyn0 = np.asarray(e_dyn0_kn_ev, dtype=np.float64).reshape(-1)
    y_diag = np.asarray(e_diag_sc_kn_ev, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(x, y_static, s=10, alpha=0.6, label="Static COHSEX")
    axes[0].scatter(x, y_dyn0, s=10, alpha=0.6, label="Bare X + Σ_c(0)")
    axes[0].scatter(x, y_diag, s=10, alpha=0.6, label="Diagonal SC Σ(E)")
    mn = float(min(np.min(x), np.min(y_static), np.min(y_dyn0), np.min(y_diag)))
    mx = float(max(np.max(x), np.max(y_static), np.max(y_dyn0), np.max(y_diag)))
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1)
    axes[0].set_xlabel("Reference energy (eV)")
    axes[0].set_ylabel("QP energy (eV)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("All (k, n)")

    b = np.arange(e_ref_kn_ev.shape[1])
    axes[1].plot(b, e_static_kn_ev[0], "-o", ms=3, label="Static COHSEX")
    axes[1].plot(b, e_dyn0_kn_ev[0], "-o", ms=3, label="Bare X + Σ_c(0)")
    axes[1].plot(b, e_diag_sc_kn_ev[0], "-o", ms=3, label="Diagonal SC Σ(E)")
    axes[1].set_xlabel("Band index")
    axes[1].set_ylabel("Energy at k = 0 (eV)")
    axes[1].set_title("k = 0")
    axes[1].legend(fontsize=8)

    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png


__all__ = [
    "build_qsgw_sigma_xc",
    "extract_sigma_diag_replicated",
    "plot_qp_energy_comparison",
    "print_scf_diagnostics",
    "solve_diagonal_sigma_fixed_point",
]
