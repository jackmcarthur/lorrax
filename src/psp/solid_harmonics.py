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
