from __future__ import annotations

"""
Utilities to compute S_{alpha,beta}(omega) from dipole.h5 (p + i[r,V_NL]).

Public API
----------
- read_dipole_h5(path) -> (dipole_cart, deltaE)
- compute_S_omega(dipole_cart, deltaE, f_nk, cell_volume, nk_tot, nspin, nspinor, omegas, eta=0.0)
"""

import numpy as np
import h5py
import jax
import jax.numpy as jnp


def read_dipole_h5(path: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    with h5py.File(path, "r") as h5:
        dipole_np = np.asarray(h5["dipole_cart"])  # (3, nk, nb, nb)
        deltaE_np = np.asarray(h5["deltaE"])       # (nk, nb, nb)
    return jnp.asarray(dipole_np, dtype=jnp.complex128), jnp.asarray(deltaE_np, dtype=jnp.float64)


def compute_S_omega(
    dipole_cart: jnp.ndarray,
    deltaE: jnp.ndarray,
    f_nk: jnp.ndarray,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    omegas: jnp.ndarray,
    eta: float = 0.0,
) -> jnp.ndarray:
    nk = int(f_nk.shape[0]); nb = int(f_nk.shape[1])
    occ0 = jnp.asarray(f_nk[0], dtype=jnp.float64)
    nelec = int(jnp.clip(jnp.sum(occ0 > 0.5), 0, nb))

    c_idx = jnp.arange(nelec, nb)
    v_idx = jnp.arange(0, nelec)

    v_cvk = dipole_cart[:, :, c_idx[:, None], v_idx[None, :]]  # (3,nk,nc,nv)
    dE_cv = deltaE[:, c_idx[:, None], v_idx[None, :]]           # (nk,nc,nv)
    f_v = f_nk[:, v_idx]
    f_c = f_nk[:, c_idx]
    fv_minus_fc = f_v[:, None, :] - f_c[:, :, None]

    pref = 4.0 / (float(cell_volume) * float(nk_tot) * float(max(nspin, 1)) * float(max(nspinor, 1)))
    pref_c = jnp.asarray(pref, dtype=jnp.complex128)

    def S_one(omega_val: jnp.ndarray) -> jnp.ndarray:
        w_c = jnp.asarray(omega_val, dtype=jnp.complex128) + 1j * jnp.asarray(float(eta), dtype=jnp.float64)
        denom = dE_cv * (w_c * w_c - dE_cv * dE_cv)
        W = jnp.where(jnp.abs(denom) > 1e-16, fv_minus_fc / denom, 0.0 + 0.0j)
        W = pref_c * W
        return jnp.einsum('ancv,ncv,bncv->ab', jnp.conj(v_cvk), W, v_cvk, optimize=True)

    if jnp.ndim(omegas) == 0:
        return S_one(omegas)[None, :, :]
    return jax.vmap(S_one, in_axes=0, out_axes=0)(jnp.asarray(omegas, dtype=jnp.complex128))


__all__ = [
    "read_dipole_h5",
    "compute_S_omega",
]

