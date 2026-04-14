"""
psp/solid_harmonics.py — QE-convention solid harmonics as Cartesian polynomials.

Separated into its own module to avoid circular imports between
dft_operators and vnl_ops.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def solid_harmonics_jax(l: int, K_cart: jax.Array) -> jax.Array:
    """Solid harmonics S_lm(x,y,z) = r^l Y_lm(r-hat) in QE convention.

    Pure polynomials in K_cart — no trig, no sqrt, no singularities.
    Perfectly smooth everywhere including K=0.  Autodiff-friendly.

    Returns (2l+1, nG) matching the QE ordering [m=0, 1c, 1s, 2c, 2s, ...].
    Supports l = 0, 1, 2, 3.
    """
    x = K_cart[:, 0]
    y = K_cart[:, 1]
    z = K_cart[:, 2]
    fpi = 4.0 * jnp.pi

    if l == 0:
        return jnp.stack([jnp.ones_like(x) / jnp.sqrt(fpi)], axis=0)

    elif l == 1:
        c = jnp.sqrt(3.0 / fpi)
        return jnp.stack([c * z, -c * x, -c * y], axis=0)

    elif l == 2:
        c2 = jnp.sqrt(5.0 / fpi)
        c2s3 = c2 * jnp.sqrt(3.0)
        return jnp.stack([
            c2 / 2.0 * (2 * z**2 - x**2 - y**2),
            -c2s3 * x * z,
            -c2s3 * y * z,
            c2s3 / 2.0 * (x**2 - y**2),
            c2s3 * x * y,
        ], axis=0)

    elif l == 3:
        c3 = jnp.sqrt(7.0 / fpi)
        s6 = jnp.sqrt(6.0)
        s10 = jnp.sqrt(10.0)
        s15 = jnp.sqrt(15.0)
        return jnp.stack([
            c3 / 2.0 * z * (2*z**2 - 3*x**2 - 3*y**2),
            -c3*s6/4.0 * x * (4*z**2 - x**2 - y**2),
            -c3*s6/4.0 * y * (4*z**2 - x**2 - y**2),
            c3*s15/2.0 * z * (x**2 - y**2),
            c3*s15 * x * y * z,
            -c3*s10/4.0 * x * (x**2 - 3*y**2),
            -c3*s10/4.0 * y * (3*x**2 - y**2),
        ], axis=0)

    else:
        raise NotImplementedError(f"solid_harmonics_jax: l={l} > 3 not implemented")


def all_solid_harmonics(K_cart: jax.Array, l_max: int = 3) -> jax.Array:
    """All solid harmonics l=0..l_max in one pass.  No Python branching.

    Returns (l_max+1, 2*l_max+1, nG) padded with zeros for m > 2l.
    Use S_all[l, :2*l+1, :] to get the harmonics for a given l.

    Pure JAX — no if/else on l, safe inside lax.scan/JIT.
    """
    x = K_cart[:, 0]
    y = K_cart[:, 1]
    z = K_cart[:, 2]
    nG = K_cart.shape[0]
    fpi = 4.0 * jnp.pi
    mmax = 2 * l_max + 1

    S = jnp.zeros((l_max + 1, mmax, nG))

    # l=0
    S = S.at[0, 0].set(jnp.ones(nG) / jnp.sqrt(fpi))

    if l_max >= 1:
        c1 = jnp.sqrt(3.0 / fpi)
        S = S.at[1, 0].set(c1 * z)
        S = S.at[1, 1].set(-c1 * x)
        S = S.at[1, 2].set(-c1 * y)

    if l_max >= 2:
        c2 = jnp.sqrt(5.0 / fpi)
        c2s3 = c2 * jnp.sqrt(3.0)
        S = S.at[2, 0].set(c2 / 2.0 * (2 * z**2 - x**2 - y**2))
        S = S.at[2, 1].set(-c2s3 * x * z)
        S = S.at[2, 2].set(-c2s3 * y * z)
        S = S.at[2, 3].set(c2s3 / 2.0 * (x**2 - y**2))
        S = S.at[2, 4].set(c2s3 * x * y)

    if l_max >= 3:
        c3 = jnp.sqrt(7.0 / fpi)
        s6 = jnp.sqrt(6.0); s10 = jnp.sqrt(10.0); s15 = jnp.sqrt(15.0)
        S = S.at[3, 0].set(c3 / 2.0 * z * (2*z**2 - 3*x**2 - 3*y**2))
        S = S.at[3, 1].set(-c3*s6/4.0 * x * (4*z**2 - x**2 - y**2))
        S = S.at[3, 2].set(-c3*s6/4.0 * y * (4*z**2 - x**2 - y**2))
        S = S.at[3, 3].set(c3*s15/2.0 * z * (x**2 - y**2))
        S = S.at[3, 4].set(c3*s15 * x * y * z)
        S = S.at[3, 5].set(-c3*s10/4.0 * x * (x**2 - 3*y**2))
        S = S.at[3, 6].set(-c3*s10/4.0 * y * (3*x**2 - y**2))

    return S
