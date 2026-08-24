"""Direct-Dirac current weight for transverse-channel ISDF centroids.

The only current-specific operation here is
``sum_{n,k,i} |Psi_nk^dagger alpha_i Psi_nk / alpha_fs|^2``.  WFN symmetry
unfolding, the kinetic-balance lift, G-sphere placement, and the FFT stay in
their existing owners.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from common.bispinor_init import ALPHA_FS
from common.gamma_matrices import gamma_apply, gamma_perm_phase


@jax.jit
def _dirac_current_weight(psi_r, band_mask=None):
    """Band/k-summed physical current square from a real-space 4-spinor.

    ``psi_r`` has shape ``(nk, nb, 4, nx, ny, nz)``.  The three matrices in
    ``common.gamma_matrices`` are LORRAX's stored ``γ̃^i = γ⁰γ^i = α_i``;
    :func:`gamma_apply` is their canonical monomial action.  Dividing the
    bilinear by ``α_fs`` matches the existing Gordon-current convention.
    """
    if psi_r.ndim != 6 or psi_r.shape[2] != 4:
        raise ValueError(
            "_dirac_current_weight expects (nk, nb, 4, nx, ny, nz), got "
            f"{psi_r.shape}")
    if band_mask is None:
        band_mask = jnp.ones((psi_r.shape[1],), dtype=jnp.float64)
    mask = jnp.asarray(band_mask, dtype=jnp.float64).reshape(
        1, -1, 1, 1, 1)
    out = jnp.zeros(psi_r.shape[-3:], dtype=jnp.float64)
    psi_dag = jnp.conj(psi_r)
    for mu in (1, 2, 3):
        # Resolve these canonical tables while tracing this one kernel, not
        # as eager JAX slices at module import.
        perm, phase = gamma_perm_phase(mu)
        alpha_psi = gamma_apply(psi_r, perm, phase, axis=2)
        current = (jnp.sum(psi_dag * alpha_psi, axis=2).real / ALPHA_FS)
        out = out + jnp.sum(current * current * mask, axis=(0, 1))
    return out


def build_current_density(wfn, sym, n_occ: int, *, verbose: bool = True):
    """Build the occupied-state current weight on the WFN FFT grid."""
    from common.collectives import single_device_mesh
    from common.wfn_transforms import to_rbox
    from wfn_loader import WfnLoader
    from .charge_density import _uniform_band_windows

    if not isinstance(wfn, WfnLoader):
        raise TypeError(
            "build_current_density requires the driver's open WfnLoader; "
            f"got {type(wfn).__name__}")
    if int(wfn.nspinor) != 2:
        raise ValueError(
            "density-mode=current requires a two-component Pauli WFN so the "
            f"canonical kinetic-balance lift is defined; got nspinor="
            f"{int(wfn.nspinor)}")

    n_occ = int(n_occ)
    if n_occ < 1 or n_occ > int(wfn.nbands):
        raise ValueError(
            f"n_occ must lie in [1, {int(wfn.nbands)}], got {n_occ}")
    fft_grid = tuple(int(v) for v in wfn.fft_grid)
    n_grid = int(np.prod(fft_grid))
    # Same fixed-shape window rule as the charge-density stream.  The 3x
    # factor prices the r-box plus the current kernel's two live equivalents.
    bytes_per_band = 3 * 4 * n_grid * np.dtype(np.complex128).itemsize
    chunk = max(1, min(n_occ, (4 * 1024 ** 3) // bytes_per_band))
    windows = [
        (lo, jnp.asarray(mask))
        for lo, mask in _uniform_band_windows(0, n_occ, chunk)
    ]

    t0 = time.perf_counter()
    last_log = t0
    mesh = single_device_mesh()

    def field_for_k(k_spec):
        box_index = wfn.box_index(k=k_spec)
        field = jnp.zeros(fft_grid, dtype=jnp.float64)
        for b_lo, band_mask in windows:
            psi_g = wfn.load_process_local(
                bands=(b_lo, b_lo + chunk), k=k_spec, bispinor=True)
            psi_r = to_rbox(
                psi_g,
                box_index,
                fft_grid,
                mesh=mesh,
                norm="ortho",
            )
            field = field + _dirac_current_weight(psi_r, band_mask)
            field.block_until_ready()
            del psi_g, psi_r
        return field

    rho_curr = jnp.zeros(fft_grid, dtype=jnp.float64)
    nk_full = int(sym.nk_tot)
    # Exact selected-row symmetry action stays in WfnLoader.  A global scalar
    # projector is not equivalent for sum_n |j_nn|^2 under degenerate-band
    # mixing, even though the resulting field is time-reversal even.
    for ik in range(nk_full):
        rho_curr = rho_curr + field_for_k([ik])
        rho_curr.block_until_ready()
        if verbose and time.perf_counter() - last_log > 5.0:
            last_log = time.perf_counter()
            print(f"    [Dirac current] {ik + 1}/{nk_full} full-BZ k "
                  f"points after {last_log - t0:.1f}s", flush=True)

    rho_curr = rho_curr / float(nk_full)
    rho_curr.block_until_ready()
    rho_np = np.asarray(rho_curr)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"  ρ_current (direct Dirac, WfnLoader full-BZ, chunk={chunk}) built "
              f"in {dt:.2f}s; max={float(rho_np.max()):.3e}, "
              f"mean={float(rho_np.mean()):.3e}", flush=True)
    return rho_np


__all__ = ["build_current_density"]
