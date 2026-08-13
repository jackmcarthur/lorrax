"""Bounded in-memory MPA pole fitting and its missing-producer boundary.

The production tree does not yet have a shared-kernel strip-frequency chi0
executor.  This module therefore accepts a ``read_w_columns``-shaped callback:
one ``(q, nu-column)`` tile across the sampling plan at a time.  Each sample
tile is fitted and released before the next; no full frequency-resolved W
tensor exists here and no file I/O is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from gw.mpa import pade_fit, sample_plan, tiling


@dataclass(frozen=True)
class MPAModel:
    """Body and scalar-head poles carried in memory after fitting.

    Body arrays use Sigma's pole-first ``(n_p,n_q,n_mu,n_mu)`` layout; head
    arrays use ``(n_p,)``.  The much larger ``Wc(z)`` samples are not retained.
    """

    z_samples: np.ndarray
    Omega_p: np.ndarray
    B_p: np.ndarray
    head_Omega_p: np.ndarray
    head_B_p: np.ndarray
    energy_unit: str


def _fit_tile(block, z, n_p, *, guards, rcond):
    """Fit one ``(z,mu,nu_columns)`` block into pole-first arrays."""
    n_z, n_mu, n_cols = block.shape
    rows = jnp.moveaxis(block, 0, -1).reshape(n_mu * n_cols, n_z)
    Omega, B, _ = pade_fit.fit_mpa_poles_batched(
        rows, z, n_p, guards=guards, rcond=rcond)

    def pole_first(values):
        a = np.asarray(jax.device_get(values))
        return np.moveaxis(a.reshape(n_mu, n_cols, n_p), -1, 0)

    return pole_first(Omega), pole_first(B)


def produce_model(
    plan,
    n_q,
    n_mu,
    body_tile_producer: Callable,
    Wc_head_samples,
    *,
    tile_bytes=None,
    energy_unit="Ry",
    guards=None,
    rcond=1.0e-13,
) -> MPAModel:
    """Fit an MPA model while holding only one W body sample tile.

    ``body_tile_producer(plan, q, mu_columns)`` must return complex
    ``Wc = W - V`` samples with shape ``(2*n_p, n_mu, len(mu_columns))`` in
    sampling-plan order.  The callback is invoked in
    :func:`gw.mpa.tiling.fit_schedule` order and its result is released after
    that tile is fitted.  It is the future connection point for the shared
    complex-frequency chi0 executor and per-sample distributed ``solve_w``.

    ``Wc_head_samples`` is the tiny ``(2*n_p,)`` scalar head correction.  It
    stays resident and is fitted once with the same Padé kernel.  This function
    neither reads nor writes HDF5 and does not provide a transition-resolvent
    fallback for the missing production chi0 producer.
    """
    if str(energy_unit) not in ("Ry", "Ha"):
        raise ValueError("energy_unit must be 'Ry' or 'Ha'")
    n_q, n_mu = int(n_q), int(n_mu)
    if n_q < 1 or n_mu < 1:
        raise ValueError("n_q and n_mu must be positive")
    z = np.asarray(sample_plan.plan_z(plan), dtype=np.complex128)
    if z.ndim != 1 or z.size == 0 or z.size % 2:
        raise ValueError(
            "MPA fit plan must contain a nonempty even number of samples")
    n_p = z.size // 2

    head = jnp.asarray(Wc_head_samples, dtype=jnp.complex128)
    if head.shape != (z.size,):
        raise ValueError(
            f"Wc_head_samples must have shape {(z.size,)}, got {head.shape}")
    if not bool(jax.device_get(jnp.all(jnp.isfinite(head)))):
        raise ValueError("Wc_head_samples contains non-finite values")

    Omega_p = np.empty((n_p, n_q, n_mu, n_mu), dtype=np.complex128)
    B_p = np.empty_like(Omega_p)
    for q, lo, hi in tiling.fit_schedule(n_q, n_mu, z.size, tile_bytes):
        columns = np.arange(lo, hi, dtype=np.int64)
        block = jnp.asarray(
            body_tile_producer(plan, q, columns), dtype=jnp.complex128)
        expected = (z.size, n_mu, hi - lo)
        if block.shape != expected:
            raise ValueError(
                "body_tile_producer returned shape "
                f"{block.shape}; expected {expected} for q={q}, "
                f"columns=[{lo},{hi})")
        if not bool(jax.device_get(jnp.all(jnp.isfinite(block)))):
            raise ValueError(
                f"body_tile_producer returned non-finite values for q={q}, "
                f"columns=[{lo},{hi})")
        Omega_block, B_block = _fit_tile(
            block, z, n_p, guards=guards, rcond=rcond)
        Omega_p[:, q, :, lo:hi] = Omega_block
        B_p[:, q, :, lo:hi] = B_block
        del block, Omega_block, B_block

    head_Omega, head_B, _ = pade_fit.fit_mpa_poles(
        head, z, n_p, guards=guards, rcond=rcond)
    return MPAModel(
        z_samples=z,
        Omega_p=Omega_p,
        B_p=B_p,
        head_Omega_p=np.asarray(jax.device_get(head_Omega)),
        head_B_p=np.asarray(jax.device_get(head_B)),
        energy_unit=str(energy_unit),
    )


__all__ = ["MPAModel", "produce_model"]
