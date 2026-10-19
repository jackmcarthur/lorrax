"""Shared k → k-q plumbing for Sternheimer + SOS finite-q matrix elements.

Two distinct pipelines need to map a Bloch state at ``k`` to its partner at
``k − q`` (with the appropriate umklapp phase) and gather wavefunctions
accordingly:

  * the Sternheimer χ-column / S-tensor solver in ``psp/run_sternheimer.py``
  * the sum-over-states finite-q matrix elements written by the augmented
    ``get_dipole_mtxels`` / consumed by ``common.chi_sos``.

Both used to inline their own ``round((k − q) − k_kmq)`` umklapp shift and
``exp(±2πi G_wrap · r)`` phase factor.  This module is the single source
of truth.

API
---
``kminq_idx_for_iq(sym, iq_red) -> (nk_full,) int32``
    Per-source-k index of ``k − q`` in the full-BZ k-list.  Wraps
    ``SymMaps.kq_map[:, iq_red]``.

``umklapp_G_wrap(kvec_full, kvec_kmq_full, qvec) -> (nk_full, 3) int32``
    Integer G-shift  ``G_wrap = round((k − q) − k_kmq)``  per source k.
    Both ``kvec_full`` and ``qvec`` are in crystal coords; ``qvec`` is the
    BGW *signed* representative in (−½, ½]³ (positive half-grid tie).

``umklapp_phase_box_batched(G_wrap, fft_grid, sign=+1) -> (nk, nx, ny, nz)``
    Cell-periodic phase  ``e^{i sign 2π G_wrap · r}``  on the FFT box,
    batched over the leading G_wrap axis.  ``sign=+1`` is the wrap
    direction (naive→wrapped gauge); ``-1`` unwraps.

``gather_kminq_box(psi_full, kminq_idx) -> psi_full[kminq_idx]``
    Trivial fancy-indexing helper, kept here for the symmetry of the API.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


def kminq_idx_for_iq(sym, iq_red: int) -> np.ndarray:
    """Return ``(nk_full,) int32`` array  s.t.  ``kvec_full[idx[ik]] = k_ik − q``.

    Reads ``sym.kq_map[:, iq_red]`` — the canonical SymMaps lookup.  Pulled
    out as a thin helper so callers don't reach into ``sym`` internals
    directly and the convention (full-BZ ``ik`` × reduced-BZ ``iq``) is
    stated once.
    """
    return np.asarray(sym.kq_map[:, int(iq_red)], dtype=np.int32)


def umklapp_G_wrap(
    kvec_full: jax.Array,
    kvec_kmq_full: jax.Array,
    qvec: jax.Array,
) -> jax.Array:
    """Compute the per-k umklapp shift ``G_wrap = round((k − q) − k_kmq)``.

    All inputs in crystal coords.  ``kvec_full[ik]`` is the source-k for
    the ik-th full-BZ point; ``kvec_kmq_full[ik]`` is the canonical
    representative of ``k − q`` (= ``kvec_full[kminq_idx[ik]]``); ``qvec``
    is the BGW signed representative in (−½, ½]³.  Output dtype is i32.

    Used inside JIT'd stack builders; safe under ``vmap`` over batched q.
    """
    return jnp.round(
        (kvec_full - qvec[None, :]) - kvec_kmq_full
    ).astype(jnp.int32)


@functools.partial(jax.jit, static_argnames=('fft_grid',))
def umklapp_phase_box_batched(
    G_wrap: jax.Array,
    fft_grid: tuple[int, int, int],
    sign: float = +1.0,
) -> jax.Array:
    """Cell-periodic phase ``e^{i·sign·2π·G_wrap·r}`` on the FFT box,
    batched over the leading axis of ``G_wrap``.

    Parameters
    ----------
    G_wrap : (nk, 3) int32
    fft_grid : (nx, ny, nz)  static
    sign : ±1.0 — ``+1`` to push naive→wrapped, ``-1`` to unwrap.

    Returns
    -------
    (nk, nx, ny, nz) complex128
    """
    nx, ny, nz = fft_grid
    fx = jnp.arange(nx, dtype=jnp.float64) / nx
    fy = jnp.arange(ny, dtype=jnp.float64) / ny
    fz = jnp.arange(nz, dtype=jnp.float64) / nz

    def _phase_for_k(G):
        arg = (2.0 * jnp.pi * sign) * (
            G[0].astype(jnp.float64) * fx[:, None, None]
            + G[1].astype(jnp.float64) * fy[None, :, None]
            + G[2].astype(jnp.float64) * fz[None, None, :])
        return jnp.exp(1j * arg).astype(jnp.complex128)

    return jax.vmap(_phase_for_k)(G_wrap)


def gather_kminq_box(psi_full: jax.Array, kminq_idx: jax.Array) -> jax.Array:
    """Index ``psi_full`` along axis 0 by ``kminq_idx`` (= ``psi_full[kminq_idx]``).

    Thin alias kept here for API symmetry with the other helpers — call
    sites read ``gather_kminq_box(psi, idx)`` rather than the bare fancy
    index, making the kq pipeline self-documenting.
    """
    return psi_full[kminq_idx]


__all__ = [
    "kminq_idx_for_iq",
    "umklapp_G_wrap",
    "umklapp_phase_box_batched",
    "gather_kminq_box",
]
