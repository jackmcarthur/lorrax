"""psp/h_dft.py — DFT Hamiltonian H@ψ black box for Davidson.

Provides make_apply_H(H_k, fft_grid) → callable that maps
  (batch, n_channels, dim) → (batch, n_channels, dim)
in sparse-G representation.  This is the only function Davidson needs.

Also provides setup_H_k_from_kvec to build the HamiltonianK dataclass,
and build_h_diag for the preconditioner diagonal.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from psp.dft_operators import HamiltonianK, apply_H_k, setup_H_k_from_kvec, build_h_diag


# ═══════════════════════════════════════════════════════════════════════
#  H @ ψ callable factory
# ═══════════════════════════════════════════════════════════════════════

def make_apply_H(H_k: HamiltonianK):
    """Return a callable (batch, n_channels, dim) → (batch, n_channels, dim).

    The returned function is the sparse-G DFT Hamiltonian matvec,
    suitable as ``apply_H`` for solvers.davidson.
    """
    nx, ny, nz = H_k.fft_grid

    @functools.partial(jax.jit, static_argnames=("_nx", "_ny", "_nz"))
    def _apply(psi_G, T, V, Gx, Gy, Gz, Z, E, mask, _nx, _ny, _nz):
        mask_f = mask[None, None, :].astype(psi_G.dtype)
        psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
        psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_G * mask_f)
        return apply_H_k(psi_box, T, V, Gx, Gy, Gz, Z, E, mask)

    def apply_H(psi_G):
        return _apply(psi_G, H_k.T_diag, H_k.V_scf,
                      H_k.Gx, H_k.Gy, H_k.Gz,
                      H_k.vnl_Z, H_k.vnl_E, H_k.mask,
                      nx, ny, nz)

    return apply_H


# Re-export for convenience — callers import everything from h_dft
__all__ = [
    "make_apply_H",
    "setup_H_k_from_kvec",
    "build_h_diag",
    "HamiltonianK",
]
