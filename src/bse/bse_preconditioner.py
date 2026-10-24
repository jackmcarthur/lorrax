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


def bgw_spin_kernel(nspinor: int, include_w: bool = True, *, channel="singlet") -> int:
    """BerkeleyGW's ``spin_kernel`` (``Common/nrtype.f90`` ``BSE_KERNEL_*``) of this
    run: 3 spinor (whatever the kernel, as ``BSE/input_fi.f90`` forces it),
    2 local fields, 0 triplet, 1 singlet."""
    if int(nspinor) == 2:
        return 3
    if not include_w:
        return 2
    return 0 if exchange_spin_weight(nspinor, channel) == 0 else 1


def exchange_spin_weight(nspinor: int, channel: str = "singlet") -> float:
    """Spin weight of the bare-Coulomb exchange channel: 2 scalar singlet, 0 scalar
    triplet, 1 spinor.

    The triplet of a spin-restricted scalar run has no exchange (BerkeleyGW's
    ``spin_triplet``). The channel is explicit structural input to every
    optical factory and diagonal cache. Charge screening always requests
    singlet; a spinor run has one kernel, weight 1. No process-global state
    may select a Hamiltonian at JIT trace time.

    This is the single owner of the textbook singlet split, ``D + 2V - W`` for a
    spin-restricted scalar run (``nspinor == 1``) against ``D + V - W`` for spinors
    (``docs/architecture/bse.md#hamiltonian``). A scalar run stores one of
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
    if channel not in ("singlet", "triplet"):
        raise ValueError(f"exchange channel {channel!r}: want singlet or triplet")
    if int(nspinor) != 1:
        return 1.0
    return 0.0 if channel == "triplet" else 2.0


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
