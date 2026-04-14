"""psp/radial_jax.py — Shared JAX radial transforms and uniform tables.

This module is the single backend for UPF radial data used to build:

- nonlocal projector form factors F_l(q)
- local ionic short-range transforms V_loc^SR(q)
- NLCC/core-charge transforms rho_core(G)

The representation is intentionally simple:

- radial functions are tabulated on a uniform q-grid
- host-side consumers use ``RadialTable.__call__`` / ``derivative()``
- JAX consumers use ``interp_uniform_jax`` directly

This avoids parallel SciPy spline and JAX table codepaths.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf


@dataclass(frozen=True)
class RadialTable:
    """Uniformly tabulated 1-D radial function with zero exterior."""

    q0: float
    dq: float
    values: np.ndarray

    def __call__(self, q) -> np.ndarray:
        return interp_uniform_np(q, self.q0, self.dq, self.values)

    def derivative(self, order: int = 1) -> "RadialTable":
        if order != 1:
            raise NotImplementedError("Only first derivatives are supported")
        return RadialTable(
            q0=self.q0,
            dq=self.dq,
            values=differentiate_uniform_table(self.values, self.dq),
        )


def make_uniform_q_grid(q_max: float, n_q: int) -> np.ndarray:
    """Uniform q-grid including both endpoints."""
    return np.linspace(0.0, max(float(q_max), 1e-8), int(n_q))


def radial_weights(
    r: np.ndarray,
    rab: np.ndarray | None,
    *,
    scheme: str,
) -> np.ndarray:
    """Integration weights for radial transforms.

    Parameters
    ----------
    r : (n_r,)
    rab : (n_r,) or None
        QE radial integration measure when available.
    scheme : ``"rab"`` or ``"simpson_rab"`` or ``"trapz"``
    """
    r = np.asarray(r, dtype=float)
    if scheme == "rab":
        if rab is not None:
            return np.asarray(rab, dtype=float)
        scheme = "trapz"

    if scheme == "simpson_rab":
        if rab is None:
            raise ValueError("simpson_rab requires rab")
        n_r = len(r)
        w = np.ones(n_r, dtype=np.float64)
        w[0] = 1.0 / 3.0
        for i in range(1, n_r - 1, 2):
            w[i] = 4.0 / 3.0
        for i in range(2, n_r - 1, 2):
            w[i] = 2.0 / 3.0
        w[n_r - 1] = 1.0 / 3.0
        return w * np.asarray(rab, dtype=np.float64)

    if scheme == "trapz":
        dr = float(r[1] - r[0]) if len(r) > 1 else 1.0
        w = np.ones_like(r, dtype=np.float64)
        if len(r) > 0:
            w[0] = 0.5
            w[-1] = 0.5
        return w * dr

    raise ValueError(f"Unknown radial integration scheme: {scheme}")


def differentiate_uniform_table(values: np.ndarray, dq: float) -> np.ndarray:
    """Second-order finite-difference derivative on a uniform grid."""
    values = np.asarray(values, dtype=np.float64)
    deriv = np.empty_like(values)
    deriv[1:-1] = (values[2:] - values[:-2]) / (2.0 * dq)
    deriv[0] = (values[1] - values[0]) / dq
    deriv[-1] = (values[-1] - values[-2]) / dq
    return deriv


def interp_uniform_np(
    q,
    q0: float,
    dq: float,
    values: np.ndarray,
) -> np.ndarray:
    """Linear interpolation with zero outside the tabulated interval."""
    q_np = np.asarray(q, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    n_q = vals.shape[-1]
    q_max = q0 + dq * (n_q - 1)

    idx_f = (q_np - q0) / dq
    idx = np.floor(idx_f).astype(np.int64)
    idx = np.clip(idx, 0, n_q - 2)
    t = np.clip(idx_f - idx, 0.0, 1.0)

    left = np.take(vals, idx, axis=-1)
    right = np.take(vals, idx + 1, axis=-1)
    out = (1.0 - t) * left + t * right
    mask = (q_np >= q0) & (q_np <= q_max)
    return np.where(mask, out, 0.0)


@jax.jit
def interp_uniform_jax(
    q: jax.Array,
    q0: float,
    dq: float,
    table: jax.Array,
) -> jax.Array:
    """Linear interpolation with zero outside the tabulated interval.

    Parameters
    ----------
    q : (...,)
    table : (...table_axes, n_q)
    """
    n_q = table.shape[-1]
    q_max = q0 + dq * (n_q - 1)
    idx_f = (q - q0) / dq
    idx = jnp.floor(idx_f).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_q - 2)
    t = jnp.clip(idx_f - idx, 0.0, 1.0)

    left = jnp.take(table, idx, axis=-1)
    right = jnp.take(table, idx + 1, axis=-1)
    out = (1.0 - t) * left + t * right
    mask = (q >= q0) & (q <= q_max)
    return jnp.where(mask, out, 0.0)


def spherical_jn_jax(l: int, x: jax.Array) -> jax.Array:
    """Spherical Bessel j_l(x) via Miller's backward recurrence.

    Forward recurrence is unstable for l >= 2 at large x (mixes in
    neumann functions).  Miller's algorithm recurs downward from a
    high l_start, producing unnormalized j_l, then rescales using
    the known j_0 = sin(x)/x.
    """
    small = jnp.abs(x) < 0.1
    sx = jnp.where(small, 1.0, x)
    sinx = jnp.sin(sx)
    cosx = jnp.cos(sx)

    j0_exact = sinx / sx

    if l == 0:
        x2 = x * x
        j0_small = 1.0 - x2 / 6.0 + x2**2 / 120.0
        return jnp.where(small, j0_small, j0_exact)

    if l == 1:
        j1_large = sinx / sx**2 - cosx / sx
        x2 = x * x
        j1_small = x / 3.0 - x * x2 / 30.0 + x * x2**2 / 840.0
        return jnp.where(small, j1_small, j1_large)

    # Miller's backward recurrence for l >= 2, using lax.fori_loop
    # to avoid unrolling 80 iterations into the XLA graph.
    # Recurrence: j_{n-1} = (2n+1)/x j_n - j_{n+1}, downward from l_start to 0.
    # Capture j_l during the sweep, normalize at the end via j_0 = sin(x)/x.
    l_start = max(l + 30, 80)

    def _miller_body(step, carry):
        # step counts 0..l_start-1; actual n = l_start - step
        jn_lo, jn_hi, jl_unnorm = carry
        n = l_start - step
        jn_prev = (2 * n + 1) / sx * jn_lo - jn_hi
        # Capture j_l when we reach n-1 == l
        jl_unnorm = jnp.where(n - 1 == l, jn_prev, jl_unnorm)
        return (jn_prev, jn_lo, jl_unnorm)

    init = (jnp.ones_like(sx), jnp.zeros_like(sx), jnp.zeros_like(sx))
    jn_lo, _, jl_unnorm = jax.lax.fori_loop(0, l_start, _miller_body, init)

    scale = j0_exact / jn_lo
    jl_large = jl_unnorm * scale

    # Small-x Taylor: j_l(x) ≈ x^l / (2l+1)!! [1 - x²/(2(2l+3)) + ...]
    x2 = x * x
    double_fact = 1.0
    for k in range(1, 2 * l + 2, 2):
        double_fact *= k
    jl_small = (
        jnp.abs(x) ** l / double_fact
        * (1.0 - x2 / (2.0 * (2 * l + 3))
           + x2**2 / (8.0 * (2 * l + 3) * (2 * l + 5)))
    )
    return jnp.where(small, jl_small, jl_large)


def spherical_jn_deriv_jax(l: int, x: jax.Array) -> jax.Array:
    """Derivative j'_l(x), stable near x=0."""
    if l == 0:
        return -spherical_jn_jax(1, x)
    jl = spherical_jn_jax(l, x)
    jlm1 = spherical_jn_jax(l - 1, x)
    safe_x = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
    return jlm1 - (l + 1) / safe_x * jl


@functools.partial(jax.jit, static_argnames=("l",))
def spherical_hankel_table_jax(
    l: int,
    r: jax.Array,
    f_r: jax.Array,
    q: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Tabulate ∫ dr r² f(r) j_l(qr) for one radial function."""
    qr = q[:, None] * r[None, :]
    j_l = spherical_jn_jax(l, qr)
    return jnp.sum(j_l * (r * r * f_r)[None, :] * weights[None, :], axis=1)


@functools.partial(jax.jit, static_argnames=("l",))
def spherical_hankel_table_batch_jax(
    l: int,
    r: jax.Array,
    f_br: jax.Array,
    q: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Batch version for multiple radial functions with the same l.

    Parameters
    ----------
    f_br : (n_f, n_r)
    Returns
    -------
    (n_f, n_q)
    """
    qr = q[:, None] * r[None, :]
    j_l = spherical_jn_jax(l, qr)
    base = j_l * weights[None, :]
    return jnp.einsum("qr,fr,r->fq", base, f_br, r * r, optimize=True)


def _spherical_hankel_table_np(
    l: int,
    r: np.ndarray,
    f_r: np.ndarray,
    q: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """CPU-only Hankel transform: ∫ dr r² f(r) j_l(qr) w(r).

    Uses scipy.special.spherical_jn — no JIT compilation overhead.
    Preferred for one-time table construction at setup.
    """
    from scipy.special import spherical_jn as _scipy_jn

    r = np.asarray(r, dtype=np.float64)
    f_r = np.asarray(f_r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    qr = q[:, None] * r[None, :]                         # (n_q, n_r)
    j_l = _scipy_jn(l, qr)                               # (n_q, n_r)
    integrand = j_l * (r * r * f_r)[None, :] * weights[None, :]
    return np.sum(integrand, axis=1)                      # (n_q,)


def make_projector_table(
    l: int,
    r: np.ndarray,
    beta_r: np.ndarray,
    q_grid: np.ndarray,
    weights: np.ndarray,
) -> RadialTable:
    """F_l(q) table for one beta projector."""
    values = _spherical_hankel_table_np(l, r, beta_r, q_grid, weights)
    return RadialTable(q0=float(q_grid[0]), dq=float(q_grid[1] - q_grid[0]), values=values)


def make_local_sr_table(
    r: np.ndarray,
    vloc_r: np.ndarray,
    z_valence: float,
    q_grid: np.ndarray,
    weights: np.ndarray,
) -> RadialTable:
    """QE short-range local-potential transform table."""
    r_np = np.asarray(r, dtype=np.float64)
    v_np = np.asarray(vloc_r, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = v_np + z_valence * 2.0 * np.where(
            r_np > 0.0,
            np.asarray(erf(jnp.asarray(r_np))) / r_np,
            2.0 / np.sqrt(np.pi),
        )
    values = _spherical_hankel_table_np(0, r_np, sr, np.asarray(q_grid), weights)
    return RadialTable(q0=float(q_grid[0]), dq=float(q_grid[1] - q_grid[0]), values=values)


def make_core_charge_table(
    r: np.ndarray,
    rho_core_r: np.ndarray,
    q_grid: np.ndarray,
    weights: np.ndarray,
) -> RadialTable:
    """NLCC/core-charge l=0 transform table."""
    values = _spherical_hankel_table_np(0, r, rho_core_r, np.asarray(q_grid), weights)
    return RadialTable(q0=float(q_grid[0]), dq=float(q_grid[1] - q_grid[0]), values=values)


__all__ = [
    "RadialTable",
    "_spherical_hankel_table_np",
    "differentiate_uniform_table",
    "interp_uniform_jax",
    "interp_uniform_np",
    "make_core_charge_table",
    "make_local_sr_table",
    "make_projector_table",
    "make_uniform_q_grid",
    "radial_weights",
    "spherical_hankel_table_batch_jax",
    "spherical_hankel_table_jax",
    "spherical_jn_deriv_jax",
    "spherical_jn_jax",
]
