"""Scissor-shift extrapolation for self-consistent GW.

The dynamic self-energy Σ_c(ω) in LORRAX is computed on a small frequency grid
(typically ±10 eV around E_F) while the full band range used in the QP
Hamiltonian spans tens of eV below E_F to hundreds or thousands of eV above.
Inside the grid we evaluate Σ_xc by interpolating the stored (ω, k, m, n)
tensor; outside the grid we extrapolate the QP correction

    ΔE_nk := E_QP_nk − E_DFT_nk  =  ⟨n| Σ_xc − V_xc^DFT |n⟩_k

by a pair of affine laws, refit at each SC iteration — one for valence bands
(occupied at DFT occupation) and one for conduction:

    ΔE_v(E) = α_v · E + β_v
    ΔE_c(E) = α_c · E + β_c

The scissor enters the Hamiltonian as a diagonal addition — no off-diagonal
coupling is introduced for the out-of-grid bands, matching the first-order
scissor approximation.

Units and layout
----------------
- Energies: eV.  The caller decides whether to work in absolute or
  Fermi-referenced coordinates; the fit and ``predict`` are domain-agnostic
  as long as inputs at fit time and apply time share the same reference.
- Per-band arrays: ``(nk, nb)``.  Fit / extrapolate are pure NumPy since the
  fit consumes O(nk·nb_σ) scalars.
- Hamiltonian update accepts a JAX array ``H[k, m, n]`` sharded as
  ``P(None, 'x', 'y')`` on a 2-D ``('x', 'y')`` device mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScissorFit:
    """Affine scissor-shift law fit to ΔE vs E for valence and conduction.

    Attributes
    ----------
    slope_v, intercept_v : float
        ΔE_v(E) = slope_v · E + intercept_v, in eV.
    slope_c, intercept_c : float
        Analogous for conduction.
    n_fit_v, n_fit_c : int
        Number of (k, n) points used in each fit.
    rmse_v_ev, rmse_c_ev : float
        Fit residual RMSE in eV (0 when the fit has fewer than 2 points).
    """

    slope_v: float
    intercept_v: float
    slope_c: float
    intercept_c: float
    n_fit_v: int
    n_fit_c: int
    rmse_v_ev: float
    rmse_c_ev: float

    def predict(self, E_ev: np.ndarray, valence_mask: np.ndarray) -> np.ndarray:
        """Evaluate the scissor law at each (k, n).

        Parameters
        ----------
        E_ev : np.ndarray, (nk, nb)
            Input energies, same reference as was used at fit time.
        valence_mask : np.ndarray, (nk, nb) of bool
            True where the valence law applies; False → conduction.
        """
        E = np.asarray(E_ev, dtype=np.float64)
        vm = np.asarray(valence_mask, dtype=bool)
        v_pred = self.slope_v * E + self.intercept_v
        c_pred = self.slope_c * E + self.intercept_c
        return np.where(vm, v_pred, c_pred)

    def summary(self) -> str:
        return (
            f"ScissorFit(val: α={self.slope_v:+.4f}, β={self.intercept_v:+.4f} eV, "
            f"n={self.n_fit_v}, rmse={self.rmse_v_ev:.3f} eV; "
            f"cond: α={self.slope_c:+.4f}, β={self.intercept_c:+.4f} eV, "
            f"n={self.n_fit_c}, rmse={self.rmse_c_ev:.3f} eV)"
        )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form OLS line fit; returns (slope, intercept, RMSE)."""
    n = int(x.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, float(y[0]), 0.0
    xm = float(x.mean())
    ym = float(y.mean())
    dx = x - xm
    denom = float(np.dot(dx, dx))
    if denom < 1.0e-30:
        return 0.0, ym, float(np.sqrt(np.mean((y - ym) ** 2)))
    slope = float(np.dot(dx, y - ym) / denom)
    intercept = float(ym - slope * xm)
    resid = y - (slope * x + intercept)
    return slope, intercept, float(np.sqrt(np.mean(resid * resid)))


def fit_scissor(
    E_kn_ev: np.ndarray,
    delta_e_kn_ev: np.ndarray,
    valence_mask_kn: np.ndarray,
    fit_mask_kn: np.ndarray,
) -> ScissorFit:
    """Fit valence / conduction scissor lines to (E, ΔE) samples.

    Parameters
    ----------
    E_kn_ev : np.ndarray, (nk, nb)
        Input energy per (k, n), usually ``E_DFT − E_F`` in eV.
    delta_e_kn_ev : np.ndarray, (nk, nb), real or complex
        QP correction (E_GW − E_DFT) in eV.  Complex inputs are reduced to
        the real part, matching ``solve_diagonal_sigma_fixed_point``.
    valence_mask_kn : np.ndarray, (nk, nb) of bool
        True for occupied bands (at DFT occupation), False for conduction.
    fit_mask_kn : np.ndarray, (nk, nb) of bool
        True where the point is trusted enough to enter the fit — typically
        "E_kn lies inside the Σ(ω) grid AND the diagonal fixed point
        converged".  Applied independently within valence and conduction.
    """
    E = np.asarray(E_kn_ev, dtype=np.float64)
    dE = np.real(np.asarray(delta_e_kn_ev, dtype=np.complex128))
    vm = np.asarray(valence_mask_kn, dtype=bool)
    fm = np.asarray(fit_mask_kn, dtype=bool)
    if not (E.shape == dE.shape == vm.shape == fm.shape):
        raise ValueError(
            f"shape mismatch: E={E.shape} dE={dE.shape} "
            f"valence={vm.shape} fit={fm.shape}"
        )

    mask_v = vm & fm
    mask_c = (~vm) & fm
    slope_v, int_v, rmse_v = _ols_line(E[mask_v], dE[mask_v])
    slope_c, int_c, rmse_c = _ols_line(E[mask_c], dE[mask_c])

    return ScissorFit(
        slope_v=slope_v, intercept_v=int_v,
        slope_c=slope_c, intercept_c=int_c,
        n_fit_v=int(mask_v.sum()), n_fit_c=int(mask_c.sum()),
        rmse_v_ev=rmse_v, rmse_c_ev=rmse_c,
    )


# ---------------------------------------------------------------------------
# Extrapolate to full bandrange
# ---------------------------------------------------------------------------

def extrapolate_delta_e(
    E_kn_ev: np.ndarray,
    delta_e_kn_ev: np.ndarray,
    valence_mask_kn: np.ndarray,
    in_grid_mask_kn: np.ndarray,
    fit: ScissorFit,
) -> np.ndarray:
    """Return ΔE over the full bandrange.

    In-grid bands keep their measured ΔE; out-of-grid bands get
    ``fit.predict(E, valence_mask)``.
    """
    dE = np.real(np.asarray(delta_e_kn_ev, dtype=np.complex128))
    ig = np.asarray(in_grid_mask_kn, dtype=bool)
    return np.where(ig, dE, fit.predict(E_kn_ev, valence_mask_kn))


# ---------------------------------------------------------------------------
# Sharded Hamiltonian-diagonal add
# ---------------------------------------------------------------------------

def add_diag_to_H_kmn(
    H_kmn: jax.Array,
    diag_kn: Union[np.ndarray, jax.Array],
    mesh_xy: Mesh,
) -> jax.Array:
    """Add a per-(k, n) diagonal ``d[k, n]`` into ``H[k, m, n]`` at m == n.

    Parameters
    ----------
    H_kmn : jax.Array, shape (nk, nb, nb)
        Complex Hamiltonian sharded as ``P(None, 'x', 'y')`` on ``mesh_xy``.
    diag_kn : array, shape (nk, nb)
        Per-band diagonal to add.  Replicated across devices (small — nk·nb
        scalars, typically ≲ 100 kB).  Real or complex; cast to H's dtype.
    mesh_xy : Mesh
        2-D device mesh with named axes ('x', 'y').

    Returns
    -------
    H_kmn with the diagonal added, same sharding.

    Notes
    -----
    Implemented with ``shard_map`` so each device touches only the diagonal
    elements it owns inside its local (bm × bn) block.  Off-diagonal process
    blocks apply a masked no-op.  Requires ``nb % x_size == 0`` and
    ``nb % y_size == 0``; pad ``H`` and ``diag_kn`` first if ``nb_sigma`` is
    not divisible.
    """
    from jax.experimental.shard_map import shard_map

    x_size, y_size = mesh_xy.devices.shape
    if H_kmn.ndim != 3:
        raise ValueError(f"H_kmn must be 3-D (nk, nb, nb); got {H_kmn.shape}.")
    nk, nb, nb2 = H_kmn.shape
    if nb != nb2:
        raise ValueError(f"H_kmn must be square in (m, n); got {H_kmn.shape}.")
    if nb % x_size != 0 or nb % y_size != 0:
        raise ValueError(
            f"add_diag_to_H_kmn requires nb ({nb}) divisible by both x_size "
            f"({x_size}) and y_size ({y_size}); pad H first."
        )
    bm = nb // x_size
    bn = nb // y_size

    d = jnp.asarray(diag_kn, dtype=H_kmn.dtype)
    if d.shape != (nk, nb):
        raise ValueError(f"diag_kn must have shape ({nk}, {nb}); got {tuple(d.shape)}.")
    d_rep = jax.device_put(d, NamedSharding(mesh_xy, P(None, None)))

    def _local_add(H_local, d_full):
        # H_local: (nk, bm, bn); d_full: (nk, nb) fully replicated.
        # Scatter the nb diagonal contributions onto the shard-local block;
        # entries whose global diagonal index doesn't land on this shard are
        # zeroed via `valid` and thus contribute nothing to the .at[].add.
        ix = lax.axis_index('x')
        iy = lax.axis_index('y')
        j = jnp.arange(nb)
        local_row = j - ix * bm
        local_col = j - iy * bn
        valid = (
            (local_row >= 0) & (local_row < bm) &
            (local_col >= 0) & (local_col < bn)
        )
        r_safe = jnp.clip(local_row, 0, bm - 1)
        c_safe = jnp.clip(local_col, 0, bn - 1)
        contrib = jnp.where(valid[None, :], d_full, jnp.zeros_like(d_full))
        return H_local.at[:, r_safe, c_safe].add(contrib)

    return shard_map(
        _local_add,
        mesh=mesh_xy,
        in_specs=(P(None, 'x', 'y'), P(None, None)),
        out_specs=P(None, 'x', 'y'),
        check_rep=False,
    )(H_kmn, d_rep)


__all__ = [
    "ScissorFit",
    "fit_scissor",
    "extrapolate_delta_e",
    "add_diag_to_H_kmn",
]
