"""bse/bse_davidson_helpers.py — initial-subspace + diagonal preconditioner
helpers tailored to the (block, nc_pad, nv_pad, nk) BSE vector layout.

These plug into ``solvers.davidson.davidson`` without that solver knowing
anything about excitons. The trial-vector layout matches the matvec from
``bse_simple.build_bse_simple_matvec``:

    X.shape   = (m, nc_pad, nv_pad, nk)         (m = batch axis)
    X.sharding = P(None, "x", "y", None)        (c on x, v on y)

The leading-energy initial subspace + (E_c − E_v) diagonal preconditioner
are the BSE analogues of plane-wave selection / QE's ``g_psi`` for DFT.

Multi-process notes
-------------------
JAX 0.5+ refuses to *fetch* (``np.asarray``) or *close over* sharded arrays
that span devices belonging to other processes. The helpers below therefore:

- ``init_bse_subspace`` calls ``multihost_utils.process_allgather`` on the
  eps tensors before reading them on host. Each process replicates the
  full eps; this is cheap (eps is tiny, ~kilobytes).
- ``bse_diagonal_precond`` closes only over a python-side ``delta_E_host``
  (numpy) and rebuilds the device tensor inside the jit'd function from
  ``eps_c`` and ``eps_v`` passed at *call time* — the closure does not
  capture any global jax.Array.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ═══════════════════════════════════════════════════════════════════════
#  Initial subspace: lowest (c, v, k) transitions + random tail
# ═══════════════════════════════════════════════════════════════════════

def _gather_to_host(arr) -> np.ndarray:
    """Bring a (possibly multi-process) jax.Array to host as numpy.

    Mirrors ``file_io._slab_io_allgather._to_host`` — for sharded JAX
    arrays uses ``process_allgather(tiled=False)`` (returns
    ``(world, *shape)``; collapse to ``[0]`` for replicated data).
    """
    if isinstance(arr, np.ndarray):
        return arr
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(arr))
    from jax.experimental import multihost_utils
    gathered = multihost_utils.process_allgather(arr, tiled=False)
    host = np.asarray(jax.device_get(gathered))
    expected = tuple(int(d) for d in arr.shape)
    if host.ndim == len(expected) + 1 and host.shape[1:] == expected:
        host = host[0]
    return host


def init_bse_subspace(
    eps_c,
    eps_v,
    n_eig: int,
    *,
    n_random: int = 5,
    mesh: Optional[Mesh] = None,
    sharding: Optional[NamedSharding] = None,
    seed: int = 0,
    dtype=jnp.complex128,
) -> jax.Array:
    """Build a starting subspace of ``n_eig`` BSE trial vectors.

    Returns the trial subspace ``V`` of shape ``(n_eig, nc, nv, nk)``,
    composed of two parts:

    - The first ``n_eig − n_random`` vectors are unit vectors at the
      lowest ``(c, v, k)`` transitions sorted by ``E_c − E_v``. Cheap
      and diverse; tends to span the low excitons because H_BSE's
      dominant diagonal is the energy difference.
    - The last ``n_random`` are deterministic complex Gaussian vectors
      (normalised), seeded by ``seed``. They cover spectral regions
      missed by the energy-sorted picks.

    Parameters
    ----------
    eps_c : (nk, nc) jax array (any sharding)
    eps_v : (nk, nv) jax array (any sharding)
    n_eig : total subspace size returned
    n_random : how many of those slots are random (default 5)
    mesh, sharding : if given, the result is shard-constrained to
        ``sharding`` (canonically P(None, "x", "y", None) for ``bse_simple``).
    seed : RNG seed (numpy keyed). Default 0 → reproducible.

    Returns
    -------
    V : (n_eig, nc, nv, nk) — unit-norm complex jax array, sharded if
        ``sharding`` was provided.
    """
    eps_c_np = _gather_to_host(eps_c)
    eps_v_np = _gather_to_host(eps_v)
    nk, nc = eps_c_np.shape
    _, nv = eps_v_np.shape

    n_random = max(0, min(n_random, n_eig))
    n_pick = n_eig - n_random

    # ── Part 1: lowest (c, v, k) by energy ────────────────────────────
    delta_E = eps_c_np[:, :, None] - eps_v_np[:, None, :]   # (nk, nc, nv)
    flat = delta_E.reshape(-1)
    # Skip padded / non-physical bands (zero or negative ΔE).
    finite = np.isfinite(flat) & (flat > 1e-12)
    order = np.argsort(np.where(finite, flat, np.inf))
    picks = order[:n_pick]
    k_idx, c_idx, v_idx = np.unravel_index(picks, (nk, nc, nv))

    V_np = np.zeros((n_eig, nc, nv, nk), dtype=np.complex128)
    V_np[np.arange(n_pick), c_idx, v_idx, k_idx] = 1.0

    # ── Part 2: random tail ───────────────────────────────────────────
    if n_random > 0:
        rng = np.random.default_rng(seed)
        rand = (rng.standard_normal((n_random, nc, nv, nk))
                + 1j * rng.standard_normal((n_random, nc, nv, nk)))
        norms = np.sqrt(
            np.sum(np.abs(rand) ** 2, axis=(1, 2, 3), keepdims=True))
        rand = rand / np.maximum(norms, 1e-30)
        V_np[n_pick:] = rand

    if sharding is not None:
        V = jax.make_array_from_callback(
            V_np.shape,
            sharding,
            lambda idx: jnp.asarray(V_np[idx], dtype=dtype),
        )
    else:
        V = jnp.asarray(V_np, dtype=dtype)
    return V


# ═══════════════════════════════════════════════════════════════════════
#  Diagonal preconditioner: 1 / (ΔE − λ + ε)
# ═══════════════════════════════════════════════════════════════════════

def bse_diagonal_precond(
    eps_c,
    eps_v,
    *,
    epsilon_shift: float = 1e-3,
    sharding: Optional[NamedSharding] = None,
):
    """Build a diagonal preconditioner ``precond_fn(R, Lambda) → P``.

    The BSE Hamiltonian's leading diagonal element is
    ``H_diag[c, v, k] ≈ (E_c[k] − E_v[k])`` (the V & W diagonals contribute
    a smaller correction). Davidson convergence improves dramatically
    when the residual is filtered by the inverse of ``(H_diag − λ)``.

    Multi-process safety
    --------------------
    The jit'd ``_impl`` does NOT close over ``eps_c`` / ``eps_v``; instead
    the outer plain-Python ``precond_fn`` captures them and forwards as
    arguments at call time. This avoids the "Closing over jax.Array that
    spans non-addressable devices" runtime error multi-process JAX raises
    when sharded arrays are baked into a jit closure.

    Parameters
    ----------
    eps_c : (nk, nc) jax array
    eps_v : (nk, nv) jax array
    epsilon_shift : Ry, regularises near band edges. Default 1e-3 ≈ 13.6 meV.
    sharding : optional NamedSharding for the (nc, nv, nk) ΔE tensor;
        canonically the X sharding with batch axis dropped (e.g.
        ``P("x", "y", None)``).

    Returns
    -------
    precond_fn : (R, Lambda) → P
        R, P shape (m, nc, nv, nk); Lambda shape (m,).
    """
    @jax.jit
    def _impl(R, Lambda, eps_c_in, eps_v_in):
        # ΔE[c, v, k] = E_c[k] − E_v[k]; same convention as bse_simple's D term.
        delta_E = eps_c_in.T[:, None, :] - eps_v_in.T[None, :, :]
        if sharding is not None:
            delta_E = jax.lax.with_sharding_constraint(delta_E, sharding)
        # Broadcast: R is (m, c, v, k), Lambda is (m,), delta_E is (c, v, k).
        denom = (delta_E[None, :, :, :]
                 - Lambda[:, None, None, None]
                 + jnp.asarray(epsilon_shift, dtype=delta_E.dtype))
        denom_safe = jnp.where(jnp.abs(denom) < 1e-12,
                               jnp.asarray(1e-12, dtype=denom.dtype),
                               denom)
        P_out = R / denom_safe
        norms = jnp.sqrt(jnp.sum(jnp.abs(P_out) ** 2, axis=(1, 2, 3),
                                 keepdims=True))
        return P_out / jnp.maximum(norms, 1e-30)

    def precond_fn(R, Lambda):
        return _impl(R, Lambda, eps_c, eps_v)

    return precond_fn


__all__ = ["init_bse_subspace", "bse_diagonal_precond"]
