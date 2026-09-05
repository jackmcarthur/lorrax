"""Shared modern-theory orbital-response algebra.

This module owns the small, representation-independent contractions used by
both the explicit-WFN and interpolated-Hamiltonian routes.  It does not load a
WFN, construct a velocity, or choose a k mesh.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


MU_B_PREFACTOR = 0.5


def orbital_pieces_at_k(v, eps, nocc: int, deps_tol: float):
    """Return the incumbent mu-independent ``(cA,cB)`` pair matrices."""
    v = np.asarray(v)
    eps = np.asarray(eps, dtype=np.float64)
    nb = int(v.shape[1])
    vt = np.swapaxes(v, 1, 2)
    cross = np.stack((
        v[1] * vt[2] - v[2] * vt[1],
        v[2] * vt[0] - v[0] * vt[2],
        v[0] * vt[1] - v[1] * vt[0],
    ))
    deps = eps[:, None] - eps[None, :]
    mask = np.abs(deps) > float(deps_tol)
    safe = np.where(mask, deps, 1.0)
    inv2 = np.where(mask, 1.0 / np.square(safe), 0.0)
    occ = (np.arange(nb) < int(nocc))[:, None]
    return (
        cross * (occ * (eps[:, None] + eps[None, :]) * inv2)[None],
        cross * (occ * inv2)[None],
    )


def orbital_cA_cB_jax(velocity_bamn, energies_bn, *, nocc: int,
                       deps_tol_ry: float):
    """Batch the total modern-theory contraction without retaining pairs.

    Parameters are ``velocity[b,a,m,n]`` and ``energies[b,n]``.  The return
    arrays have shape ``(b,3)`` and retain the chemical-potential split
    ``C(mu) = cA - 2*mu*cB`` used by the explicit-WFN driver.
    """
    velocity = jnp.asarray(velocity_bamn, dtype=jnp.complex128)
    energies = jnp.asarray(energies_bn, dtype=jnp.float64)
    nb = int(velocity.shape[-1])
    vt = jnp.swapaxes(velocity, -1, -2)
    cross = jnp.stack((
        velocity[:, 1] * vt[:, 2] - velocity[:, 2] * vt[:, 1],
        velocity[:, 2] * vt[:, 0] - velocity[:, 0] * vt[:, 2],
        velocity[:, 0] * vt[:, 1] - velocity[:, 1] * vt[:, 0],
    ), axis=1)
    deps = energies[:, :, None] - energies[:, None, :]
    mask = jnp.abs(deps) > float(deps_tol_ry)
    safe = jnp.where(mask, deps, 1.0)
    inv2 = jnp.where(mask, 1.0 / jnp.square(safe), 0.0)
    occ = (jnp.arange(nb) < int(nocc))[None, :, None]
    cA = jnp.sum(
        cross * (occ * (energies[:, :, None] + energies[:, None, :])
                 * inv2)[:, None],
        axis=(-2, -1),
    )
    cB = jnp.sum(cross * (occ * inv2)[:, None], axis=(-2, -1))
    return cA, cB


def magnetic_axial_projector(sym, magnetization_axis=(0.0, 0.0, 1.0)):
    """Return the authenticated axial/time-odd magnetic-group projector."""
    rows = np.asarray(sym.active_symmetry_rows, dtype=np.int32)
    action = np.asarray(
        sym.cartesian_action(rows, axial=True, time_odd=True),
        dtype=np.float64,
    )
    axis = np.asarray(magnetization_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or norm <= 1.0e-12:
        raise ValueError(
            "orbital magnetization axis must be a finite nonzero 3-vector; "
            f"got {magnetization_axis!r}.")
    axis = axis / norm
    if not bool(sym.trs_allowed):
        if str(sym.operation_typing_source) != "qe-schema":
            raise RuntimeError(
                "orbital magnetization on a time-reversal-broken reference "
                "requires authenticated QE operation typing.")
        mapped = np.einsum("sij,j->si", action, axis)
        bad = np.flatnonzero(~np.all(np.isclose(
            mapped, axis[None], rtol=0.0, atol=1.0e-6), axis=1))
        if bad.size:
            raise ValueError(
                "orbital magnetization axis is inconsistent with active "
                f"typed operation row {int(rows[bad[0]])}.")
    return action.mean(axis=0), rows


def moment_mu_b(cA, cB, mu_ry: float):
    """Convert the shared ``C(mu)`` split to a per-cell moment in mu_B."""
    return -MU_B_PREFACTOR * np.imag(
        np.asarray(cA) - 2.0 * float(mu_ry) * np.asarray(cB))
