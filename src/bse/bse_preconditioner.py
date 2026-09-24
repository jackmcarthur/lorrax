"""The two elementwise BSE-basis helpers every solver shares.

``energy_diff_cv_k`` -- the single-particle energy differences E_c(k) - E_v(k)
that form the D-term diagonal of H_BSE (bse_feast's diagonal preconditioner);
``compute_pair_amplitude`` -- the exchange vertex M(k,c,v,μ) the loaders hoist
into every matvec.  Both lived in the deleted single-device ``bse_serial``
module until 2026-09-24 (C10).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_pair_amplitude(psi_c: jax.Array, psi_v: jax.Array) -> jax.Array:
    """M(k,c,v,μ) = Σ_s conj(ψ_c[k,c,s,μ]) * ψ_v[k,v,s,μ]."""
    return jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c), psi_v)


def exchange_spin_weight(nspinor: int) -> float:
    """Spin weight of the bare-Coulomb exchange channel: 2 scalar, 1 spinor.

    This is the single owner of the textbook singlet split, ``D + 2V - W`` for a
    spin-restricted scalar run (``nspinor == 1``) against ``D + V - W`` for spinors
    (bse/context/README.md, "Note on spin factors"). A scalar run stores one of
    the two degenerate spin channels per transition. A spinor run's
    :func:`compute_pair_amplitude` already sums both components through ``Σ_s``.

    Every exchange ENCODE applies it: the stack TDA/pair V and head terms, both
    ring encodes, and the exact diagonal. Decodes do not apply it, since they sum
    over no transitions. Until 2026-09-24 only the ring encodes applied it, so on
    a scalar run the stack matvec built ``D + V - W`` while the ring built
    ``D + 2V - W``: that is ``test_cross_solver_agreement[ring]``'s relerr 1.02,
    measured against a dense reference that also lacked the weight. Spinor runs
    were never affected.
    """
    return 2.0 if int(nspinor) == 1 else 1.0


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
