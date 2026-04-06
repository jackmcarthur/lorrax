"""
psp/charge_density.py — Build charge density from IBZ wavefunctions.

Computes rho(r) from wavefunctions at irreducible k-points using
k-weights for multiplicity, then symmetrizes rho(G) by averaging
over the star of each G-vector (same strategy as QE's sum_band +
sym_rho).

Usage
-----
    from psp.charge_density import build_density_from_ibz

    rho_r = build_density_from_ibz(wfn, sym, meta, n_occ)
    # rho_r: (nx, ny, nz) real-space density in e/bohr^3
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox


def build_density_from_ibz(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    n_occ: int,
) -> jax.Array:
    """Build symmetrised charge density from IBZ wavefunctions.

    Steps:
    1. Accumulate rho(r) = sum_{ik in IBZ} wk(ik) sum_{n,s} |psi_nks(r)|^2
    2. FFT to G-space
    3. Symmetrise rho(G) by averaging over symmetry star of each G
    4. FFT back to real space

    Parameters
    ----------
    wfn : WFNReader with wavefunctions at IBZ k-points
    sym : SymMaps with symmetry operations
    meta : system metadata
    n_occ : number of occupied bands (per spin channel)

    Returns
    -------
    rho_r : (nx, ny, nz) float64 — electron density in e/bohr^3
    """
    nx, ny, nz = meta.fft_grid
    nk_ibz = int(wfn.nkpts)
    vol = float(wfn.cell_volume)
    N_grid = nx * ny * nz
    kweights = np.asarray(wfn.kweights, dtype=np.float64)

    # -- Step 1: accumulate rho(r) from IBZ --------------------------------
    #
    # rho(r) = sum_{ik} wk(ik) sum_{n=1}^{n_occ} sum_{s} |psi_{n,ik,s}(r)|^2
    #
    # The FFT convention in load_kpoint_fftbox gives psi in the ortho-norm
    # FFT box.  IFFT with norm='ortho' gives psi(r) * sqrt(N_grid/vol).
    # So |psi(r)|^2 = |IFFT(psi_box)|^2 * (N_grid/vol).
    # The density in e/bohr^3 is: rho = sum |psi(r)|^2 * (vol/N_grid)
    #   = sum |IFFT|^2 * (N_grid/vol) * (vol/N_grid) = sum |IFFT|^2
    # Wait — let me be careful.
    #
    # load_kpoint_fftbox gives psi_box(G) such that the orthonormal IFFT
    # gives psi(r) with <psi|psi> = sum_G |psi_box(G)|^2 = 1 (normalised).
    # The real-space representation is:
    #   psi(r_j) = (1/sqrt(N)) sum_G psi_box(G) exp(iGr_j)  [ortho IFFT]
    #
    # Then |psi(r_j)|^2 has sum_j |psi(r_j)|^2 = N (from Parseval).
    # The continuous density is rho(r) = (N_grid/vol) |psi(r)|^2, and
    # integral rho d^3r = sum_j |psi(r_j)|^2 * (vol/N_grid) = 1 per band.
    #
    # With nspinor=2 and n_occ occupied bands, total charge = n_occ * nspinor
    # (but bands are already counting spinor degrees of freedom, so actually
    # total charge = n_occ since each band has norm 1 across both spinors).
    #
    # For the weights: sum_ik wk(ik) = 1, and each weight accounts for
    # the multiplicity of that IBZ k-point.  Total electron count:
    #   N_el = 2 * sum_ik wk(ik) * n_occ  [factor 2 for spin if nspinor=1]
    # For nspinor=2 (FR): N_el = sum_ik wk(ik) * n_occ
    #   (each band already includes both spinor components)

    rho_r = jnp.zeros((nx, ny, nz), dtype=jnp.float64)

    nspinor = int(meta.nspinor)
    # Spin degeneracy factor: 2 if scalar (nspinor=1), 1 if FR (nspinor=2)
    spin_factor = 2 if nspinor == 1 else 1

    for ik in range(nk_ibz):
        wk = float(kweights[ik])
        psi_box = load_kpoint_fftbox(wfn, sym, meta, ik, n_occ)
        # psi_box: (n_occ, nspinor, nx, ny, nz)

        # IFFT to real space (ortho normalisation)
        psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')

        # |psi|^2 summed over bands and spinor components
        rho_k = jnp.sum(jnp.abs(psi_r) ** 2, axis=(0, 1))  # (nx, ny, nz)

        rho_r = rho_r + spin_factor * wk * rho_k

    # Convert to e/bohr^3:  rho(r) * N_grid / vol
    rho_r = rho_r * (N_grid / vol)

    # Check: integral should give nelec
    integral = float(jnp.sum(rho_r)) * vol / N_grid
    print(f"  Density integral: {integral:.4f} e (expected {wfn.nelec})")

    # -- Step 2-4: symmetrise in G-space -----------------------------------
    rho_r = _symmetrise_density(rho_r, sym, meta)

    return rho_r


def _symmetrise_density(
    rho_r: jax.Array,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
) -> jax.Array:
    """Symmetrise density in G-space: rho(G) → (1/N_sym) Σ_S rho(S·G).

    Uses the reciprocal-space rotation matrices from SymMaps.
    """
    nx, ny, nz = meta.fft_grid
    nsym = sym.sym_matrices.shape[0]

    # FFT to G-space
    rho_G = jnp.fft.fftn(rho_r)  # (nx, ny, nz) complex

    # Build G-vector grid indices
    gx = np.fft.fftfreq(nx, d=1.0/nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0/ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0/nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_grid = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1)  # (N, 3)

    # For each symmetry operation S, rotate the G-vectors and look up rho(SG)
    # R_grid are the reciprocal-space rotation matrices: G' = R_grid @ G
    R_all = np.asarray(sym.R_grid, dtype=int)  # (nsym, 3, 3)

    rho_G_flat = rho_G.ravel()
    rho_sym = jnp.zeros_like(rho_G_flat)

    for isym in range(nsym):
        R = R_all[isym]
        # Rotated G-vectors
        G_rot = G_grid @ R.T  # (N, 3)
        # Wrap into FFT grid
        ix = G_rot[:, 0] % nx
        iy = G_rot[:, 1] % ny
        iz = G_rot[:, 2] % nz
        # Linear index into the FFT array
        idx = ix * ny * nz + iy * nz + iz
        rho_sym = rho_sym + rho_G_flat[idx]

    rho_sym = rho_sym / nsym
    rho_G_sym = rho_sym.reshape(nx, ny, nz)

    # FFT back to real space
    rho_r_sym = jnp.real(jnp.fft.ifftn(rho_G_sym))

    return rho_r_sym
