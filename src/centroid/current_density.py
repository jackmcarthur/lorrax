"""Direct-Dirac current weight for transverse-channel ISDF centroids.

The only current-specific operation here is
``sum_{n,k,i} |Psi_nk^dagger alpha_i Psi_nk / alpha_fs|^2``.  The symmetry
mapping, kinetic-balance lift, G-sphere placement, and FFT stay in their
existing owners.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from common.bispinor_init import ALPHA_FS, NO_PAIR_DIRAC_CURRENT_MODEL
from common.gamma_matrices import gamma_apply, gamma_perm_phase


@jax.jit
def _dirac_current_weight(psi_r, band_mask, wavefunction_scale):
    """Band/k-summed physical current square from a real-space 4-spinor.

    ``psi_r`` has shape ``(nk, nb, 4, nx, ny, nz)`` in the unitary-FFT
    convention.  ``wavefunction_scale = sqrt(N_grid / Omega)`` converts it to
    the physical unit-cell normalization before the bilinear is formed.  The
    three matrices in ``common.gamma_matrices`` are LORRAX's stored
    ``γ̃^i = γ⁰γ^i = α_i``; :func:`gamma_apply` is their canonical
    monomial action.  Dividing the bilinear by ``α_fs`` matches the existing
    Gordon-current convention.
    """
    if psi_r.ndim != 6 or psi_r.shape[2] != 4:
        raise ValueError(
            "_dirac_current_weight expects (nk, nb, 4, nx, ny, nz), got "
            f"{psi_r.shape}")
    mask = jnp.asarray(band_mask, dtype=jnp.float64).reshape(
        1, -1, 1, 1, 1)
    out = jnp.zeros(psi_r.shape[-3:], dtype=jnp.float64)
    psi_dag = jnp.conj(psi_r)
    current_scale = (jnp.asarray(wavefunction_scale, dtype=jnp.float64) ** 2
                     / ALPHA_FS)
    for mu in (1, 2, 3):
        # Resolve these canonical tables while tracing this one kernel, not
        # as eager JAX slices at module import.
        perm, phase = gamma_perm_phase(mu)
        alpha_psi = gamma_apply(psi_r, perm, phase, axis=2)
        current = (jnp.sum(psi_dag * alpha_psi, axis=2).real
                   * current_scale)
        out = out + jnp.sum(current * current * mask, axis=(0, 1))
    return out


def build_current_density(wfn, sym, n_occ: int, *, verbose: bool = True):
    """Build the occupied-state current weight on the WFN FFT grid.

    The expensive lift and FFT are evaluated once per referenced raw WFN IBZ
    row.  For a full-zone member ``k`` with parent ``p = irr_idx_k[k]`` and
    selected symmetry row ``s = sym_idx_k[k]``, the direct Dirac current is a
    polar vector (and is odd under time reversal), hence its Cartesian norm is
    a scalar:

        sum_i j_i(k, r_new)^2 = sum_i j_i(p, r_old)^2.

    ``symmetry_maps.fft_grid_pullback_perm`` supplies the exact nonsymmorphic
    ``r_new -> r_old`` gather.  A TRS-augmented row uses the same spatial
    pullback as its first-half partner; its extra current sign drops out of
    the square.  Thus the star accumulation is exactly the full-BZ
    WfnLoader-unfold result at fixed band index, without rebuilding any
    wavefunction, G-vector, FFT-box, or symmetry action here.
    """
    from common.collectives import single_device_mesh
    from common.wfn_transforms import to_rbox
    from symmetry_maps import fft_grid_pullback_perm, star_tables_of
    from wfn_loader import IBZRows, WfnLoader
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
    cell_volume = float(wfn.cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError(
            f"WFN cell volume must be finite and positive, got {cell_volume}")
    wavefunction_scale = float(np.sqrt(n_grid / cell_volume))
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
            field = field + _dirac_current_weight(
                psi_r, band_mask, wavefunction_scale)
            field.block_until_ready()
            del psi_g, psi_r
        return field

    nk_full = int(sym.nk_tot)
    # Keep the full-k parent row, selected symmetry row, and spatial/TRS
    # split inseparable.  ``star_tables_of`` derives the split from the same
    # TRS-augmented table that the WFN unfold consumes.
    parent_for_k, sym_row_for_k, n_spatial = star_tables_of(sym)
    if n_spatial < 1:
        raise ValueError("SymMaps contains no spatial symmetry row")
    sym_mats_k_shape = np.asarray(sym.sym_mats_k).shape
    if sym_mats_k_shape != (2 * n_spatial, 3, 3):
        raise ValueError(
            "SymMaps.sym_mats_k must contain spatial rows followed by their "
            "TRS partners; got "
            f"{sym_mats_k_shape}, expected {(2 * n_spatial, 3, 3)}")
    if parent_for_k.shape != (nk_full,) or sym_row_for_k.shape != (nk_full,):
        raise ValueError(
            "SymMaps full-k tables disagree with nk_tot: "
            f"irr_idx_k={parent_for_k.shape}, sym_idx_k={sym_row_for_k.shape}, "
            f"nk_tot={nk_full}")
    if (np.any(parent_for_k < 0)
            or np.any(parent_for_k >= int(wfn.nkpts))):
        raise ValueError(
            "SymMaps.irr_idx_k contains a row outside the raw WFN k axis "
            f"[0,{int(wfn.nkpts)})")
    if (np.any(sym_row_for_k < 0)
            or np.any(sym_row_for_k >= 2 * n_spatial)):
        raise ValueError(
            "SymMaps.sym_idx_k contains a row outside its spatial/TRS-"
            f"augmented table [0,{2 * n_spatial})")

    grid_pullback = fft_grid_pullback_perm(
        np.asarray(sym.sym_matrices)[:n_spatial],
        np.asarray(sym.translations)[:n_spatial],
        fft_grid,
        validate=True,
    )
    if grid_pullback.shape != (n_spatial, n_grid):
        raise ValueError(
            "symmetry-service FFT-grid pullback has the wrong shape: "
            f"{grid_pullback.shape} != {(n_spatial, n_grid)}")

    # One physical grid field at a time on both device and host.  Enumerating
    # the selected full-k rows preserves the exact star multiplicities and
    # avoids assuming that WFN kweights or stored-row ordering encode a
    # particular reduction convention.  No (n_star, n_grid) temporary is
    # formed: each service-owned pullback is gathered and accumulated alone.
    rho_flat = np.zeros(n_grid, dtype=np.float64)
    parents_used = np.unique(parent_for_k)
    full_rows_done = 0
    for parent in parents_used:
        parent = int(parent)
        field_flat = np.asarray(
            field_for_k(IBZRows((parent,))), dtype=np.float64).reshape(-1)
        members = np.flatnonzero(parent_for_k == parent)
        for ik in members:
            spatial_row = int(sym_row_for_k[int(ik)]) % n_spatial
            rho_flat += field_flat[grid_pullback[spatial_row]]
        full_rows_done += int(members.size)
        if verbose and time.perf_counter() - last_log > 5.0:
            last_log = time.perf_counter()
            print(f"    [Dirac current] {full_rows_done}/{nk_full} full-BZ k "
                  f"points from {parent + 1}/{int(wfn.nkpts)} raw WFN rows "
                  f"after {last_log - t0:.1f}s", flush=True)

    rho_np = (rho_flat / float(nk_full)).reshape(fft_grid)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"  ρ_current ({NO_PAIR_DIRAC_CURRENT_MODEL}, "
              "WFN-IBZ + exact star pullback, "
              f"chunk={chunk}) built "
              f"in {dt:.2f}s; max={float(rho_np.max()):.3e}, "
              f"mean={float(rho_np.mean()):.3e}", flush=True)
    return rho_np


__all__ = ["build_current_density"]
