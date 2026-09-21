"""Orbital moments from a specified physical velocity, in Ry/Bohr units.

The caller owns the velocity operator and its Hilbert-space completeness.
A bare Dirac current in a positive-energy subspace is not automatically the
derivative of a nonlocal quasiparticle Hamiltonian.
"""
from __future__ import annotations

import jax.numpy as jnp


def orbital_velocity_products(velocity, energies, deps_tol):
    """Return cross products, inverse gaps and degenerate-band flags.

    Inputs are ``(...,3,n,n)`` velocities (Ry Bohr) and ``(...,n)`` energies
    (Ry). Matrix entries are bra n, ket m. Degenerate internal transitions
    are omitted; individual moments inside such a multiplet are gauge
    dependent, while its trace against external states remains meaningful.
    """
    v, e = jnp.asarray(velocity), jnp.asarray(energies)
    if v.shape[-3:] != (3, e.shape[-1], e.shape[-1]):
        raise ValueError('orbital velocity must have shape (...,3,n,n)')
    vt = jnp.swapaxes(v, -1, -2)
    cross = jnp.stack((v[..., 1, :, :] * vt[..., 2, :, :]
                       - v[..., 2, :, :] * vt[..., 1, :, :],
                       v[..., 2, :, :] * vt[..., 0, :, :]
                       - v[..., 0, :, :] * vt[..., 2, :, :],
                       v[..., 0, :, :] * vt[..., 1, :, :]
                       - v[..., 1, :, :] * vt[..., 0, :, :]), axis=-3)
    gap = e[..., :, None] - e[..., None, :]
    resolved = jnp.abs(gap) > deps_tol
    inverse = jnp.where(resolved, 1 / jnp.where(resolved, gap, 1), 0)
    degenerate = jnp.any(~resolved & ~jnp.eye(e.shape[-1], dtype=bool), axis=-1)
    return cross, inverse, degenerate


def orbital_moments(velocity, energies, *, deps_tol_ry=1e-8):
    """Return wavepacket moments (mu_B), Berry curvature (Bohr²), flags.

    Outputs have shapes ``(...,3,n)``, ``(...,3,n)``, ``(...,n)``.
    With electron charge -e and v_nm=<n|dH/dk|m>,
    m_n/mu_B = -1/2 Im sum_m (v_nm cross v_mn)/(E_n-E_m).
    Omega_n = -Im sum_m (v_nm cross v_mn)/(E_n-E_m)^2.
    These are per-band quantities, without occupations or a cell volume.
    """
    cross, inv, degenerate = orbital_velocity_products(
        velocity, energies, deps_tol_ry)
    moment = -0.5 * jnp.sum(jnp.imag(cross) * inv[..., None, :, :], axis=-1)
    berry = -jnp.sum(jnp.imag(cross) * inv[..., None, :, :] ** 2, axis=-1)
    return moment, berry, degenerate


def orbital_magnetization(velocity, energies, *, mu_ry, width_ry,
                          deps_tol_ry=1e-8):
    """Return per-k thermodynamic moment in mu_B at Fermi-Dirac temperature.

    Sum with normalized BZ weights for the per-cell moment. ``width_ry`` is
    k_B T, not an extra regulator. At zero T the Berry term is
    (mu-E)*Theta(mu-E); at finite T it is T*log(1+exp((mu-E)/T)).
    The result is a trace over all supplied states, including complete
    degenerate multiplets. A truncated intermediate-state sum stays truncated.
    """
    moment, berry, _ = orbital_moments(
        velocity, energies, deps_tol_ry=deps_tol_ry)
    e = jnp.asarray(energies)
    if width_ry < 0:
        raise ValueError('Fermi-Dirac width must be nonnegative')
    if width_ry == 0:
        f = (e < mu_ry).astype(e.dtype)
        grand = jnp.maximum(mu_ry - e, 0)
    else:
        from gw.efermi import fd_occupations
        x = (mu_ry - e) / width_ry
        grand = width_ry * jnp.logaddexp(0, x)
        f = fd_occupations(e, mu_ry, width_ry)
    return jnp.sum(moment * f[..., None, :] + berry * grand[..., None, :], axis=-1)
