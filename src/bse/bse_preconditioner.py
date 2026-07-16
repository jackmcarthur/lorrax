"""Band-energy-difference helper for BSE diagonal / preconditioner terms.

Provides ``energy_diff_cv_k`` -- the single-particle energy differences
E_c(k) - E_v(k) that form the D-term diagonal of H_BSE. Consumed by
bse_serial (apply_D) and bse_feast (diagonal preconditioner assembly).
"""

from __future__ import annotations

import jax


def energy_diff_cv_k(eps_c: jax.Array, eps_v: jax.Array) -> jax.Array:
    """Compute energy differences delta_E(c,v,k) = eps_c(k) - eps_v(k).

    Args:
        eps_c: (nk, nc) conduction energies
        eps_v: (nk, nv) valence energies

    Returns:
        delta_E: (nc, nv, nk) array
    """
    # eps_c: (nk, nc) -> (nc, 1, nk)
    # eps_v: (nk, nv) -> (1, nv, nk)
    return eps_c.T[:, None, :] - eps_v.T[None, :, :]
