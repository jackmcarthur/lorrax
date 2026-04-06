"""
psp/charge_density.py — Build charge density from IBZ wavefunctions.

Computes rho(r) from wavefunctions at irreducible k-points using
k-weights for multiplicity, then symmetrizes rho(G) by averaging
over the star of each G-vector (same strategy as QE's sum_band +
sym_rho).

Also provides grad_rho_sq(r) = |∇ρ|² for GGA functionals and
build_V_xc for constructing the exchange-correlation potential
on the real-space grid.

Usage
-----
    from psp.charge_density import build_density_from_ibz, compute_grad_rho_sq, build_V_xc

    rho_r = build_density_from_ibz(wfn, sym, meta, n_occ)
    grad_rho_sq = compute_grad_rho_sq(rho_r, wfn)
    V_xc_r = build_V_xc(rho_r, grad_rho_sq)
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


# ---------------------------------------------------------------------------
# Density gradient for GGA
# ---------------------------------------------------------------------------

def build_core_density(
    wfn,
    pseudos: dict,
    meta,
) -> jax.Array:
    """Build NLCC core charge density ρ_core(r) on the FFT grid.

    Fourier transforms the radial core charge from the UPF file onto
    the PW FFT grid with atomic structure factors, just like V_loc.

    Returns zeros if no pseudopotential has core_correction=True.

    Parameters
    ----------
    wfn : WFNReader
    pseudos : dict from load_pseudopotentials
    meta : Meta

    Returns
    -------
    rho_core_r : (nx, ny, nz) float64 — core density on FFT grid
                 same units/normalization as valence density
    """
    nx, ny, nz = meta.fft_grid
    vol = float(wfn.cell_volume)

    # G-vectors on the FFT grid
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_crys = np.stack([Gx, Gy, Gz], axis=-1).astype(float)
    bvec = np.asarray(wfn.bvec, dtype=float)
    B = float(wfn.blat) * bvec.T
    G_cart = np.einsum('...i,ij->...j', G_crys, B)
    G_norm = np.sqrt(np.sum(G_cart ** 2, axis=-1))  # (nx, ny, nz)

    atom_pos = np.asarray(wfn.atom_crys, dtype=float)   # (nat, 3) crystal
    atom_types = np.asarray(wfn.atom_types, dtype=int)

    # Build element→pseudo mapping by matching Z
    from psp.get_DFT_mtxels import _symbol_to_Z
    z_to_pseudo = {}
    for elem, pseudo in pseudos.items():
        z = _symbol_to_Z(elem)
        if z is not None:
            z_to_pseudo[z] = pseudo

    rho_core_G = np.zeros((nx, ny, nz), dtype=complex)
    N = nx * ny * nz

    for iat in range(len(atom_pos)):
        z_atom = int(atom_types[iat])
        pseudo = z_to_pseudo.get(z_atom)
        if pseudo is None:
            continue
        hdr = getattr(pseudo, 'pp_header', None)
        cc = getattr(hdr, 'core_correction', None) if hdr else None
        if cc is None or str(cc) != 'UpfLogical.T':
            continue
        nlcc = getattr(pseudo, 'pp_nlcc', None)
        if nlcc is None:
            continue

        # Radial core charge: ρ_core(r) on radial grid
        rho_c_r = np.asarray(nlcc.value, dtype=float)
        mesh = pseudo.pp_mesh
        r = np.asarray(mesh.pp_r.value, dtype=float)
        rab = np.asarray(mesh.pp_rab.value, dtype=float)

        # Fourier transform: ρ_core(G) = (4π/Ω) ∫ r² ρ_core(r) j₀(Gr) dr
        # Vectorized over all G-values at once: shape (n_G, n_r)
        integrand = r ** 2 * rho_c_r * rab  # r² ρ_core(r) dr

        G_flat = G_norm.ravel()              # (n_G,)
        Gr = np.outer(G_flat, r)             # (n_G, n_r)

        # j₀(x) = sin(x)/x, with j₀(0) = 1
        j0 = np.sinc(Gr / np.pi)            # np.sinc(x) = sin(πx)/(πx)

        # ρ_core(G) = (4π/Ω) Σ_i integrand_i j₀(G r_i)
        rho_c_G_flat = (4 * np.pi / vol) * (j0 @ integrand)  # (n_G,)
        rho_c_G_atom = rho_c_G_flat.reshape(nx, ny, nz)

        # Structure factor: e^{-2πi G_crys · τ}
        tau = atom_pos[iat]
        phase = np.exp(-2j * np.pi * (
            Gx * tau[0] + Gy * tau[1] + Gz * tau[2]
        ))

        rho_core_G += rho_c_G_atom * phase

    # IFFT to real space — match the normalization of compute_valence_density
    # rho_core_G is the continuous FT: ρ(G) = (1/Ω) ∫ ρ(r) e^{-iGr} d³r
    # To get rho on the FFT grid: rho_grid = N * IFFT(rho_G_continuous)
    # With ortho IFFT: rho_grid = sqrt(N) * IFFT_ortho(rho_G_continuous)
    # But we need rho_grid in the same convention as compute_valence_density:
    #   rho_cvd has sum(rho) * vol/N = nelec
    # Our ρ(G=0) = Q_core/Ω, and IFFT should give rho_grid = ρ(G=0) * N (unnorm)
    #   → rho_grid(uniform) = Q_core/Ω * N = Q_core * N/Ω
    #   → sum(rho_grid) * Ω/N = Q_core ✓ (matches CVD convention)

    # Use standard (unnormalized) IFFT: rho_grid = N * IFFT(rho_G)
    rho_core_r = np.real(np.fft.ifftn(rho_core_G)) * N

    integral = float(np.sum(rho_core_r)) * vol / N
    print(f"  Core density integral: {integral:.4f} e "
          f"(from NLCC, {len(atom_pos)} atoms)")

    return jnp.asarray(rho_core_r, dtype=jnp.float64)


def compute_grad_rho_sq(
    rho_r: jax.Array,
    wfn,
) -> jax.Array:
    """Compute |∇ρ|² on the real-space grid via G-space derivatives.

    ∂ρ/∂x_i in G-space is i G_cart_i ρ(G).  Transform each component
    back to real space, then |∇ρ|² = Σ_i (∂ρ/∂x_i)².

    Parameters
    ----------
    rho_r : (nx, ny, nz) real-space density
    wfn : WFNReader (for bvec, blat to get Cartesian G-vectors)

    Returns
    -------
    grad_rho_sq : (nx, ny, nz) = |∇ρ(r)|² in (e/bohr^4)²
    """
    nx, ny, nz = rho_r.shape

    # G-vectors in crystal coordinates (integers)
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_crys = np.stack([Gx, Gy, Gz], axis=-1).astype(float)  # (nx,ny,nz,3)

    # Crystal → Cartesian: G_cart = G_crys @ (2π b / a_lat)
    # where b = bvec (columns are reciprocal vectors), blat = 2π/alat
    # In QE convention: G_cart = G_crys @ (blat * bvec^T)
    bvec = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    B = blat * bvec.T  # (3,3) — same as in dft_operators
    G_cart = np.einsum('...i,ij->...j', G_crys, B)  # (nx,ny,nz,3)
    G_cart_j = jnp.asarray(G_cart, dtype=jnp.float64)

    # ρ(G)
    rho_G = jnp.fft.fftn(rho_r)

    # ∂ρ/∂x_i(r) = IFFT(i G_cart_i ρ(G))
    grad_rho_sq = jnp.zeros_like(rho_r)
    for i in range(3):
        drho_G = 1j * G_cart_j[..., i] * rho_G
        drho_r = jnp.real(jnp.fft.ifftn(drho_G))
        grad_rho_sq = grad_rho_sq + drho_r ** 2

    return grad_rho_sq


# ---------------------------------------------------------------------------
# V_xc builder (PBE by default)
# ---------------------------------------------------------------------------

def build_V_xc(
    rho_r: jax.Array,
    wfn,
    *,
    grad_rho_sq: jax.Array | None = None,
    xc_func: str = 'pbe',
) -> jax.Array:
    """Build V_xc(r) on the real-space grid.

    For LDA: V_xc = d(ρ ε_xc)/dρ
    For GGA: V_xc = d(ρ ε_xc)/dρ - 2∇·[d(ρ ε_xc)/d(σ) ∇ρ]

    Parameters
    ----------
    rho_r : (nx, ny, nz) electron density [e/bohr³]
    wfn : WFNReader (for bvec, blat — needed for G-space gradient)
    grad_rho_sq : |∇ρ|² — computed automatically if None
    xc_func : 'pbe' (default) or 'lda'

    Returns
    -------
    V_xc_r : (nx, ny, nz) exchange-correlation potential [Ry]
    """
    if xc_func == 'lda':
        return _build_V_xc_lda(rho_r)
    elif xc_func == 'pbe':
        if grad_rho_sq is None:
            grad_rho_sq = compute_grad_rho_sq(rho_r, wfn)
        return _build_V_xc_gga(rho_r, grad_rho_sq, wfn)
    else:
        raise ValueError(f"Unknown xc_func: {xc_func}")


def _build_V_xc_lda(rho_r: jax.Array) -> jax.Array:
    """LDA V_xc = d(ρ ε_xc)/dρ, no gradient terms."""
    from jax_xc_local.pbe import pbe_xc
    rho_safe = jnp.maximum(rho_r, 1e-30)

    def f_xc(rho_val):
        return rho_val * pbe_xc(rho_val, 0.0)

    return jax.vmap(jax.grad(f_xc))(rho_safe.ravel()).reshape(rho_r.shape)


def _build_V_xc_gga(
    rho_r: jax.Array,
    grad_rho_sq: jax.Array,
    wfn,
) -> jax.Array:
    """GGA V_xc via White-Bird functional derivative.

    V_xc(r) = ∂(ρ ε_xc)/∂ρ - 2 ∇·[∂(ρ ε_xc)/∂σ ∇ρ]

    where σ = |∇ρ|².  The divergence term is computed in G-space.
    """
    from jax_xc_local.pbe import pbe_xc

    rho_safe = jnp.maximum(rho_r, 1e-30)
    sigma = jnp.maximum(grad_rho_sq, 0.0)

    # ∂(ρ ε_xc)/∂ρ and ∂(ρ ε_xc)/∂σ at each grid point
    def f_xc(rho_val, sigma_val):
        return rho_val * pbe_xc(rho_val, sigma_val)

    df_drho = jax.vmap(jax.grad(f_xc, argnums=0))(
        rho_safe.ravel(), sigma.ravel()
    ).reshape(rho_r.shape)

    df_dsigma = jax.vmap(jax.grad(f_xc, argnums=1))(
        rho_safe.ravel(), sigma.ravel()
    ).reshape(rho_r.shape)

    # GGA correction: -2 ∇·[df_dsigma ∇ρ]
    nx, ny, nz = rho_r.shape
    rho_G = jnp.fft.fftn(rho_r)

    # G-vectors in Cartesian
    G_cart = _build_G_cart_grid(nx, ny, nz, wfn)

    # ∇ρ(r) components and divergence
    gga_correction = jnp.zeros_like(rho_r)
    for i in range(3):
        drho_ri = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G))
        h_i = df_dsigma * drho_ri
        h_i_G = jnp.fft.fftn(h_i)
        gga_correction = gga_correction + jnp.real(
            jnp.fft.ifftn(1j * G_cart[..., i] * h_i_G)
        )

    return df_drho - 2.0 * gga_correction


def _build_G_cart_grid(nx, ny, nz, wfn):
    """Build Cartesian G-vector grid for FFT-based gradients."""
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_crys = np.stack([Gx, Gy, Gz], axis=-1).astype(float)
    bvec = np.asarray(wfn.bvec, dtype=float)
    B = float(wfn.blat) * bvec.T
    G_cart = np.einsum('...i,ij->...j', G_crys, B)
    return jnp.asarray(G_cart, dtype=jnp.float64)
