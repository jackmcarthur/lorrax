"""Gordon-decomposed Pauli current weight for ISDF centroid selection.

W_curr(r) = Σ_{n∈occ, k, i} | j^Gordon_{n,k,i}(r) |²

with

  j^Gordon_{n,k}(r) = Im[ψ_L^† ∇ ψ_L]  +  (1/2) ∇ × (ψ_L^† σ ψ_L).
                     └────────┬────────┘   └──────────┬──────────┘
                       paramagnetic            spin-curl

Vectorised across bands at fixed k (batched FFTs) to amortise JIT
dispatch.  The 5-D ``∂_j s^k`` intermediate that blew up the naïve
implementation is avoided by computing each curl component separately,
each of which only touches two of the three s components.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from common.gamma_matrices import sigma_x, sigma_y, sigma_z


def _build_K_cart(gvecs_k, kvec_frac, bvec_dimless, alat):
    gk_frac = (gvecs_k.astype(np.float64) + kvec_frac[None, :])
    return gk_frac @ bvec_dimless * (2.0 * math.pi / float(alat))


def build_current_density(wfn, sym, n_occ: int, *,
                          verbose: bool = True) -> np.ndarray:
    """Gordon-current weight on the FFT grid.

    Vectorised over all occupied bands at each k — one JIT compile
    per (ngk_k) shape.  Peak GPU memory ≈ O(n_occ × FFT-grid) scalars
    plus a few transient buffers; for CrI3 6×6×1 (n_occ=70, FFT
    75×75×200) that's ~7 GB, fits comfortably in 1× A100.
    """
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    nx, ny, nz = fft_grid
    nk_full = int(sym.nk_tot)
    nspinor_wfn = int(wfn.nspinor)

    bvec_dimless = np.asarray(wfn.bvec, dtype=np.float64)
    alat = float(wfn.alat)

    # FFT-grid Cartesian momentum for the spatial Fourier dual of unit-cell
    # periodic fields like s(r).  Bohr⁻¹ via 2π/alat conversion.
    gx_int = jnp.fft.fftfreq(nx, d=1.0/nx).astype(jnp.float64)
    gy_int = jnp.fft.fftfreq(ny, d=1.0/ny).astype(jnp.float64)
    gz_int = jnp.fft.fftfreq(nz, d=1.0/nz).astype(jnp.float64)
    bvec_cart = jnp.asarray(bvec_dimless * (2.0 * math.pi / alat),
                            dtype=jnp.float64)

    def _Kc_axis(j):
        return (gx_int[:, None, None] * bvec_cart[0, j]
                + gy_int[None, :, None] * bvec_cart[1, j]
                + gz_int[None, None, :] * bvec_cart[2, j]).astype(jnp.complex128)

    Kc_x, Kc_y, Kc_z = _Kc_axis(0), _Kc_axis(1), _Kc_axis(2)
    sigmas = jnp.stack([sigma_x, sigma_y, sigma_z], axis=0)  # (3, 2, 2)

    @jax.jit
    def _kpt_contrib(psi_G_k, K_cart_g, ng_x, ng_y, ng_z):
        """All-bands-at-once Σ_n,i (j^Gordon_{n,i})² for a single k.

        psi_G_k: (n_occ, 2, ngk)
        K_cart_g: (ngk, 3) Cartesian (k+G) in Bohr⁻¹
        ng_{x,y,z}: (ngk,) int FFT-bin indices

        Returns: (nx, ny, nz) float64 — band-summed j² + (1/2 curl_s)² + cross term.
        """
        n_occ_local = psi_G_k.shape[0]

        # Place ψ_G into FFT box: (n_occ, 2, nx, ny, nz)
        zero_box = jnp.zeros((n_occ_local, 2, nx, ny, nz), dtype=jnp.complex128)
        psi_box = zero_box.at[:, :, ng_x, ng_y, ng_z].set(psi_G_k)
        psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
        # (n_occ, 2, nx, ny, nz)

        def _grad_i(i):
            """∂_i ψ_L(r): IFFT of i (k+G)_i ψ_L(G).  Output (n_occ, 2, nx, ny, nz)."""
            iK_g = (1j * K_cart_g[:, i]).astype(jnp.complex128)
            grad_G = psi_G_k * iK_g[None, None, :]
            grad_box = zero_box.at[:, :, ng_x, ng_y, ng_z].set(grad_G)
            return jnp.fft.ifftn(grad_box, axes=(-3, -2, -1), norm='ortho')

        # Paramagnetic j^para_i(r) per band (use grad_i, then free)
        grad_x = _grad_i(0)
        j_x_n = jnp.einsum('naxyz,naxyz->nxyz', jnp.conj(psi_r), grad_x).imag
        del grad_x
        grad_y = _grad_i(1)
        j_y_n = jnp.einsum('naxyz,naxyz->nxyz', jnp.conj(psi_r), grad_y).imag
        del grad_y
        grad_z = _grad_i(2)
        j_z_n = jnp.einsum('naxyz,naxyz->nxyz', jnp.conj(psi_r), grad_z).imag
        del grad_z

        # Spin density per band per direction: (n_occ, nx, ny, nz) per i
        s_x_n = jnp.einsum('naxyz,ab,nbxyz->nxyz',
                            jnp.conj(psi_r), sigmas[0], psi_r).real
        s_y_n = jnp.einsum('naxyz,ab,nbxyz->nxyz',
                            jnp.conj(psi_r), sigmas[1], psi_r).real
        s_z_n = jnp.einsum('naxyz,ab,nbxyz->nxyz',
                            jnp.conj(psi_r), sigmas[2], psi_r).real
        del psi_r

        # FFT each component (per band); produce (n_occ, nx, ny, nz) cplx
        sx_G = jnp.fft.fftn(s_x_n.astype(jnp.complex128),
                             axes=(-3, -2, -1), norm='ortho')
        sy_G = jnp.fft.fftn(s_y_n.astype(jnp.complex128),
                             axes=(-3, -2, -1), norm='ortho')
        sz_G = jnp.fft.fftn(s_z_n.astype(jnp.complex128),
                             axes=(-3, -2, -1), norm='ortho')
        del s_x_n, s_y_n, s_z_n

        # Curl per direction (only two s components per call → no 5-D blowup)
        curl_x = jnp.fft.ifftn(1j * (Kc_y[None, ...] * sz_G - Kc_z[None, ...] * sy_G),
                                axes=(-3, -2, -1), norm='ortho').real
        curl_y = jnp.fft.ifftn(1j * (Kc_z[None, ...] * sx_G - Kc_x[None, ...] * sz_G),
                                axes=(-3, -2, -1), norm='ortho').real
        curl_z = jnp.fft.ifftn(1j * (Kc_x[None, ...] * sy_G - Kc_y[None, ...] * sx_G),
                                axes=(-3, -2, -1), norm='ortho').real
        del sx_G, sy_G, sz_G

        # j^Gordon_n,i = j^para_n,i + (1/2) curl_n,i; sum over (n, i) of squares
        out = ((j_x_n + 0.5 * curl_x) ** 2
               + (j_y_n + 0.5 * curl_y) ** 2
               + (j_z_n + 0.5 * curl_z) ** 2)
        return jnp.sum(out, axis=0)

    t0 = time.perf_counter()
    rho_curr = jnp.zeros(fft_grid, dtype=jnp.float64)
    last_log = t0

    # WfnLoader covers per-k unfold (U_spinor + τ-phase + TRS-conj) +
    # G-vector rotation in one place; the per-ik loop stays so the
    # full ψ_all × full-BZ-k slab never has to fit in host RAM.
    from file_io.wfn_loader import WfnLoader
    with WfnLoader(wfn._filename) as loader:
        gvecs_full = loader.gvecs(k="full_bz")        # (nk_full, ngkmax, 3)
        ngk_valid = loader.ngk_valid(k="full_bz")     # (nk_full,)
        for ik in range(nk_full):
            n = int(ngk_valid[ik])
            gvecs_k = gvecs_full[ik, :n]
            kvec_frac = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
            K_cart = _build_K_cart(gvecs_k, kvec_frac, bvec_dimless, alat)

            psi_k = loader.load(
                bands=(0, n_occ), k=[ik], sharding=None)   # (1, n_occ, ns, ngkmax)
            psi_G_k_np = np.zeros((n_occ, 2, n), dtype=np.complex128)
            psi_G_k_np[:, :nspinor_wfn, :] = np.asarray(
                psi_k[0, :, :, :n])

            rho_curr = rho_curr + _kpt_contrib(
                jnp.asarray(psi_G_k_np),
                jnp.asarray(K_cart, dtype=jnp.complex128),
                jnp.asarray(gvecs_k[:, 0], dtype=jnp.int32),
                jnp.asarray(gvecs_k[:, 1], dtype=jnp.int32),
                jnp.asarray(gvecs_k[:, 2], dtype=jnp.int32),
            )

            if verbose and (time.perf_counter() - last_log > 5.0):
                rho_curr.block_until_ready()
                last_log = time.perf_counter()
                print(f"    [j^Gordon] {ik+1}/{nk_full} k-points "
                      f"after {last_log - t0:.1f}s", flush=True)

    rho_curr.block_until_ready()
    rho_curr = rho_curr / nk_full
    rho_np = np.asarray(rho_curr)

    if verbose:
        dt = time.perf_counter() - t0
        print(f"  ρ_current (Gordon Pauli) built in {dt:.2f}s; "
              f"max={float(rho_np.max()):.3e}, mean={float(rho_np.mean()):.3e}",
              flush=True)
    return rho_np


__all__ = ["build_current_density"]
