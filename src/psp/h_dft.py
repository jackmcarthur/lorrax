"""psp/h_dft.py — DFT Hamiltonian H@ψ black box for Davidson.

Provides make_apply_H(H_k) → callable that maps
  (batch, n_channels, dim) → (batch, n_channels, dim)
in sparse-G representation.  Uses a single JIT'd kernel shared across
all k-points (no per-k recompilation).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from psp.dft_operators import HamiltonianK, apply_H_k, setup_H_k_from_kvec, build_h_diag


# The sparse-G kernel: JIT'd ONCE with static FFT grid dims.
# H_k data passed as explicit arguments → same compiled code for all k.
@functools.partial(jax.jit, static_argnames=("_nx", "_ny", "_nz"))
def _apply_H_sparse(psi_G, T, V, Gx, Gy, Gz, Z, E, mask, _nx, _ny, _nz):
    """Sparse-G → sparse-G H|ψ⟩.  Single JIT trace for all k-points."""
    mask_f = mask[None, None, :].astype(psi_G.dtype)
    psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
    psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_G * mask_f)
    return apply_H_k(psi_box, T, V, Gx, Gy, Gz, Z, E, mask)


def make_apply_H(H_k: HamiltonianK):
    """Return a callable (batch, n_channels, dim) → (batch, n_channels, dim).

    The returned function passes H_k data as explicit JAX arrays to a
    single shared JIT kernel — no per-k recompilation.
    """
    nx, ny, nz = H_k.fft_grid
    # Bind the H_k data but NOT via closure-capture of a JIT'd function.
    # _apply_H_sparse is JIT'd once; H_k arrays are just different inputs.
    def apply_H(psi_G):
        return _apply_H_sparse(psi_G, H_k.T_diag, H_k.V_scf,
                               H_k.Gx, H_k.Gy, H_k.Gz,
                               H_k.vnl_Z, H_k.vnl_E, H_k.mask,
                               nx, ny, nz)
    return apply_H


__all__ = [
    "make_apply_H",
    "_apply_H_sparse",
    "setup_H_k_from_kvec",
    "build_h_diag",
    "HamiltonianK",
]
