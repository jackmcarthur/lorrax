#!/usr/bin/env python3
"""
Dipole/velocity matrix elements calculation: <mk|v|nk> and related pieces.

This module mirrors the setup of get_DFT_mtxels.py and provides a scaffold to
compute:
- Momentum operator matrix elements p_i: sum_G (k+G)_i c*_mk(G) c_nk(G)
- Nonlocal pseudopotential contribution to velocity (initialized projectors)

The nonlocal velocity term is stubbed pending detailed implementation; the code
initializes QE-style projectors so downstream development can fill it in.
"""

import os
import argparse
import functools
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io import WFNReader
from common import symmetry_maps
from common.load_wfns import read_Gvecs_to_devices
from common import Meta
from psp.get_DFT_mtxels import read_cohsex_input
from psp.pseudos import load_pseudopotentials, print_atomic_structure
from psp.dft_operators import generate_gvectors_k, gather_psi_G_from_crys, momentum_matrix_k
import psp.vnl_ops as vnl_ops
import h5py
# --------------------------
# K+G helpers
# --------------------------


def compute_p_operator_k(wfn_k: jax.Array, Gk_crys: np.ndarray, kpoint_crys: np.ndarray, bdot: np.ndarray, bvec: np.ndarray, blat: float) -> jax.Array:
	"""Compute p-operator matrix elements per Cartesian component.

	Returns array of shape (3, nb, nb) for components x,y,z.
	p_i = sum_G (k+G)_cart[i] c*_mk(G) c_nk(G)
	"""
	C_bsg = gather_psi_G_from_crys(wfn_k, Gk_crys)
	k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
	G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
	B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
	return momentum_matrix_k(C_bsg, G_int, k_crys, B)


def compute_vnl_matrix_from_setup(
	wfn_k: jax.Array,
	Gk_crys: np.ndarray,
	kpoint_crys: np.ndarray,
	vnl_setup,
) -> jax.Array:
	"""Return <m|V_NL(k)|n> using the unified JAX VNL setup."""
	kdata = vnl_ops.build_vnl_kdata_from_kvec(
		np.asarray(kpoint_crys, dtype=float),
		np.asarray(Gk_crys, dtype=int),
		vnl_setup,
		compute_dZ=False,
	)
	psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys)
	# Slice to physical spinor components for bispinor wavefunctions
	nspinor_E = kdata.E_super.shape[0]
	if psi_G.shape[1] > nspinor_E:
		psi_G = psi_G[:, :nspinor_E, :]
	return vnl_ops.vnl_matrix(psi_G, kdata.Z, kdata.E_super)


def compute_vnl_velocity_cart(
	wfn_k: jax.Array,
	Gk_crys: np.ndarray,
	kpoint_crys: np.ndarray,
	vnl_setup,
) -> jax.Array:
	"""Return dV_NL/dK_cart using the unified JAX VNL path."""
	kdata = vnl_ops.build_vnl_kdata_from_kvec(
		np.asarray(kpoint_crys, dtype=float),
		np.asarray(Gk_crys, dtype=int),
		vnl_setup,
		compute_dZ=True,
	)
	psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys)
	# Slice to physical spinor components for bispinor wavefunctions
	nspinor_E = kdata.E_super.shape[0]
	if psi_G.shape[1] > nspinor_E:
		psi_G = psi_G[:, :nspinor_E, :]
	return vnl_ops.vnl_velocity_matrix(psi_G, kdata.Z, kdata.dZ, kdata.E_super)


def compute_projected_momentum_bgw_like(*args, **kwargs):
    raise NotImplementedError("Debug-only momentum projection removed")


def compute_block_direct_cnk(*args, **kwargs):
    raise NotImplementedError("Debug-only direct cnk path removed")


# --------------------------
# Finite-q matrix elements for SOS chi head/wing/S/w pipeline
# --------------------------

def _build_g_lookup(Gk_int_kmq: np.ndarray, Gk_int_k: np.ndarray,
                     G_wrap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-(k, q) integer lookup that translates between the G-spheres of
    ``k`` and ``canonical(k − q)`` under umklapp.

    For each μ_k in the ket's G-sphere, we want the μ_kmq such that
    ``Gk_int_kmq[μ_kmq] == Gk_int_k[μ_k] + G_wrap``.  When that G-vector
    lies outside the canonical-kmq sphere we mark it −1 (the bra
    coefficient there is zero anyway, so the contribution to the
    overlap is zero).

    Returns
    -------
    map_arr : (nG_k,) int32  — μ_kmq index per μ_k (0 placeholder where
              not found; use ``mask`` to gate).
    mask    : (nG_k,) bool   — True where the lookup succeeded.
    """
    target = Gk_int_k + G_wrap[None, :]                                 # (nG_k, 3)
    g_dict = {tuple(int(x) for x in g): i for i, g in enumerate(Gk_int_kmq)}
    nG_k = target.shape[0]
    map_arr = np.empty(nG_k, dtype=np.int32)
    mask = np.empty(nG_k, dtype=bool)
    for i, t in enumerate(target):
        idx = g_dict.get((int(t[0]), int(t[1]), int(t[2])), -1)
        mask[i] = idx >= 0
        map_arr[i] = idx if idx >= 0 else 0
    return map_arr, mask


@jax.jit
def _cell_overlap_with_lookup(c_can_m, c_n_k, vket_alpha, vbra_alpha,
                                map_arr, mask):
    """Symmetric-velocity cell overlap on G-sphere with umklapp lookup.

    All inputs in the canonical-kmq / k G-sphere layouts:
      c_can_m    : (nc, ns, nG_can)  — bra coefs on canonical k-q sphere
      c_n_k      : (nv, ns, nG_k)    — ket coefs on k sphere
      vket_alpha : (3, nv, ns, nG_k) — v applied to ket at k (kin + VNL(k))
      vbra_alpha : (3, nc, ns, nG_can) — v applied to bra at k_can_kmq
                                         (kin + VNL(k_can_kmq))
      map_arr    : (nG_k,) int32 — μ_can index for each μ_k
      mask       : (nG_k,) bool  — gate for out-of-sphere G's

    The bra is gathered along the canonical-kmq G-axis at the indices
    ``map_arr[μ_k]`` so the contracted-G-axis is the *ket's* G-sphere.

    Returns
    -------
    rho_mn   : (nc, nv) complex128
    v_mn_alp : (3, nc, nv) complex128 — symmetrized:
                 v_sym = ½(v_R + v_L)
                 v_R = ⟨bra | (kin + VNL(k))|ket⟩       — bra unchanged
                 v_L = ⟨(kin + VNL(k_can_kmq))|bra⟩† |ket⟩
    """
    # Bra aligned to ket's G-axis: (nc, ns, nG_k).
    bra_aligned = jnp.take(c_can_m, map_arr, axis=-1)
    bra_aligned = jnp.where(mask[None, None, :], bra_aligned,
                              jnp.zeros((), dtype=bra_aligned.dtype))
    vbra_aligned = jnp.take(vbra_alpha, map_arr, axis=-1)               # (3, nc, ns, nG_k)
    vbra_aligned = jnp.where(mask[None, None, None, :], vbra_aligned,
                               jnp.zeros((), dtype=vbra_aligned.dtype))

    rho_mn = jnp.einsum('msG,nsG->mn', jnp.conj(bra_aligned), c_n_k,
                          optimize=True)
    # v_R: original bra dotted with v-applied ket  (bra at k-q, ket apply v(k))
    v_R = jnp.einsum('msG,ansG->amn', jnp.conj(bra_aligned), vket_alpha,
                       optimize=True)
    # v_L: v-applied bra (now aligned and conjugated) dotted with original ket
    v_L = jnp.einsum('amsG,nsG->amn', jnp.conj(vbra_aligned), c_n_k,
                       optimize=True)
    v_sym = 0.5 * (v_R + v_L)
    return rho_mn, v_sym


@functools.partial(jax.jit, static_argnames=('fft_grid',))
def _apply_kinetic_velocity_Gbox(psi_Gbox_k, kvec, bvec_blat, fft_grid):
    """Compute v^α_kin |ψ_n,k⟩ in the G-space FFT-box layout.

    ``load_kpoint_fftbox`` stores ``c_nk(G)`` scattered into an FFT-box
    array (zero outside the G-sphere).  The kinetic velocity in G-space
    is just an elementwise multiplication by ``(k + G)_cart^α`` — no
    FFT.  Note: this matches the convention of
    ``dft_operators.momentum_matrix_k``  (atomic-unit cartesian
    velocity, no V_cell factor).

    Parameters
    ----------
    psi_Gbox_k : (nb, nspinor, nx, ny, nz) complex128 — c_n,k(G) in box.
    kvec       : (3,) float64 crystal coords.
    bvec_blat  : (3, 3) float64 — blat·bvec (cartesian rec-lat).
    fft_grid   : static (nx, ny, nz).

    Returns
    -------
    (3, nb, nspinor, nx, ny, nz) — c_n,k(G) · (k+G)_cart^α per α.
    """
    nx, ny, nz = fft_grid
    gx = jnp.fft.fftfreq(nx, d=1.0 / nx).astype(jnp.float64)
    gy = jnp.fft.fftfreq(ny, d=1.0 / ny).astype(jnp.float64)
    gz = jnp.fft.fftfreq(nz, d=1.0 / nz).astype(jnp.float64)
    kGc_x = kvec[0] + gx[:, None, None]
    kGc_y = kvec[1] + gy[None, :, None]
    kGc_z = kvec[2] + gz[None, None, :]
    kG_cart_x = (kGc_x * bvec_blat[0, 0] + kGc_y * bvec_blat[1, 0]
                  + kGc_z * bvec_blat[2, 0])
    kG_cart_y = (kGc_x * bvec_blat[0, 1] + kGc_y * bvec_blat[1, 1]
                  + kGc_z * bvec_blat[2, 1])
    kG_cart_z = (kGc_x * bvec_blat[0, 2] + kGc_y * bvec_blat[1, 2]
                  + kGc_z * bvec_blat[2, 2])

    return jnp.stack((
        psi_Gbox_k * kG_cart_x[None, None, :, :, :].astype(psi_Gbox_k.dtype),
        psi_Gbox_k * kG_cart_y[None, None, :, :, :].astype(psi_Gbox_k.dtype),
        psi_Gbox_k * kG_cart_z[None, None, :, :, :].astype(psi_Gbox_k.dtype),
    ), axis=0)


@functools.partial(jax.jit, static_argnames=('fft_grid',))
def _cell_overlaps_at_q_Gbox(
    psi_Gbox_k_n, vpsi_Gbox_k_n, psi_Gbox_kmq_m_canonical, G_wrap, fft_grid,
):
    """G-space cell overlaps for one (k, q) pair — kinetic-only velocity.

    Compute
        rho_mn(k, q) = ⟨u_{m, k-q} | u_{n, k}⟩_cell
        v_mn_α(k, q) = ⟨u_{m, k-q} | v^α | u_{n, k}⟩_cell  (kinetic part)

    in G-space.  Umklapp from canonical-(k-q) to actual k-q is handled
    via a 3-axis ``jnp.roll`` of the bra:
        c_{m, k-q}(G) = c_{m, canonical}(G + G_wrap)
                       = roll(c_{m, canonical}, shift=−G_wrap)(G)
    so the bra in G-box is roll(c_can, −G_wrap_int) along the (gx, gy, gz)
    axes.  No 1/N_grid factor — convention matches
    ``momentum_matrix_k`` (sum over G of c* (k+G) c gives the AU velocity
    matrix element directly).

    Parameters
    ----------
    psi_Gbox_k_n            : (nb_n, nspinor, nx, ny, nz)
    vpsi_Gbox_k_n           : (3, nb_n, nspinor, nx, ny, nz)
    psi_Gbox_kmq_m_canonical : (nb_m, nspinor, nx, ny, nz)
    G_wrap                  : (3,) int32 — umklapp shift for THIS (k, q).
    fft_grid                : static.

    Returns
    -------
    rho_mn   : (nb_m, nb_n) complex128
    v_mn_alp : (3, nb_m, nb_n) complex128
    """
    # Roll the bra by −G_wrap along the 3 G-axes (last 3).
    # ``shift`` is the number of places to shift TOWARDS HIGHER indices;
    # roll(x, +s)[i] = x[i − s].  We want bra[G] = c_can[G + G_wrap], so
    # shift = −G_wrap.
    # G_wrap is a 3-vector traced under vmap; jnp.roll accepts traced
    # shifts in modern JAX, but we re-roll one axis at a time so the
    # codepath is robust across versions.
    bra_can = jnp.conj(psi_Gbox_kmq_m_canonical)
    bra = jnp.roll(bra_can, shift=-G_wrap[0], axis=-3)
    bra = jnp.roll(bra,     shift=-G_wrap[1], axis=-2)
    bra = jnp.roll(bra,     shift=-G_wrap[2], axis=-1)

    rho_mn = jnp.einsum('msxyz,nsxyz->mn', bra, psi_Gbox_k_n,
                          optimize=True)
    v_mn_alp = jnp.einsum('msxyz,ansxyz->amn', bra, vpsi_Gbox_k_n,
                          optimize=True)
    return rho_mn, v_mn_alp


def compute_finite_q_mtxels(
    wfn, sym, meta, vnl_setup, wfn_k_sharded, Gk_crys_all,
    *,
    iq_list: list[int],
    nv_block: int,
    nc_block: int,
    verbose: bool = True,
):
    """Driver: produce symmetric finite-q matrix elements on G-sphere.

    Returns numpy arrays:
      rho_cvkq[nc, nv, nk, nq] complex128  — ⟨u_{c, k-q} | u_{v, k}⟩_cell
      v_cvkq[3, nc, nv, nk, nq] complex128 — symmetric (v_R + v_L)/2 of
                                              ⟨u_{c, k-q} | v^α | u_{v, k}⟩_cell
                                              including kinetic + VNL.
      kminq_idx[nk, nq] int32 — canonical k-q lookup.

    Plumbing:
      • G-sphere throughout (no FFT box).  Wfns are taken from
        ``wfn_k_sharded`` (already loaded in the q=0 dipole pass).
      • Kinetic apply via ``apply_kinetic_velocity_to_ket``.
      • VNL apply via ``vnl_ops.apply_vnl_velocity_to_ket`` with k-side
        Z(k) projectors for v_R and bra-side Z(k_can_kmq) projectors
        for v_L.  Both projector tables come from
        ``vnl_ops.build_vnl_kdata_from_kvec`` per k.
      • Umklapp via per-(k, q) integer G-lookup table that maps
        ket's μ_k → bra's μ_can such that
        ``Gk_int_can[μ_can] == Gk_int_k[μ_k] + G_wrap``.  The
        unmatched G's contribute 0 (their canonical coefficients are
        outside the cutoff sphere anyway).

    Stored c-v block:
      m ∈ [n_occ, n_occ + nc_block)  (conduction)
      n ∈ [n_occ - nv_block, n_occ)  (valence)
    """
    from common.kq_mapping import kminq_idx_for_iq, umklapp_G_wrap
    from psp.dft_operators import apply_kinetic_velocity_to_ket
    import psp.vnl_ops as vnl_ops

    nk_full = int(sym.nk_tot)
    bvec_blat = jnp.asarray(np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat),
                             dtype=jnp.float64)
    n_occ = int(wfn.nelec)
    v_lo = max(0, n_occ - int(nv_block))
    c_lo = n_occ
    c_hi = min(n_occ + int(nc_block), int(wfn_k_sharded.shape[1]))
    nv_eff = n_occ - v_lo
    nc_eff = c_hi - c_lo

    kpts_full = np.asarray(sym.unfolded_kpts, dtype=np.float64)

    # ── Per-k apply: kinetic + VNL on the ket side, plus same on bra side ──
    # Note: the apply'd vectors live on each k's own G-sphere.  We need
    # the apply at every full-BZ k since both ket-side (k) and bra-side
    # (canonical k-q) draw from the same set of full-BZ k-vectors.  Build
    # once.
    if verbose:
        print(f"  finite-q: applying v_kin + V_NL  to {nv_eff} valence + "
              f"{nc_eff} conduction × {nk_full} k-points (G-sphere)")

    # Per-k (kdata, vket_v, vket_c) — different shapes per k (G-sphere
    # length varies) so we keep a Python list rather than vmap'ing.
    vket_v_per_k = []     # (3, nv, ns, nG_k)  each
    vket_c_per_k = []     # (3, nc, ns, nG_k)  each
    psi_v_per_k  = []     # (nv, ns, nG_k)
    psi_c_per_k  = []     # (nc, ns, nG_k)
    for ik in range(nk_full):
        kvec = jnp.asarray(kpts_full[ik], dtype=jnp.float64)
        Gk_int = Gk_crys_all[ik]                                       # (nG_k, 3) int
        # Slice psi → G-sphere shape (nb, ns, nG_k).  ``wfn_k_sharded`` is
        # (nk, nb, ns, nG_k); the gather already happened upstream via
        # ``read_Gvecs_to_devices``.
        psi_k = wfn_k_sharded[ik]                                      # (nb, ns, nG_k)
        # Gather FFT-box layout → G-sphere coeffs at this k's integer G-list.
        psi_k_G = gather_psi_G_from_crys(psi_k, Gk_int)               # (nb, ns, nG_k)
        psi_v = psi_k_G[v_lo:n_occ]
        psi_c = psi_k_G[c_lo:c_hi]
        psi_v_per_k.append(psi_v)
        psi_c_per_k.append(psi_c)

        # Kinetic apply (Rydberg p = 2(k+G)).
        v_kin_v = apply_kinetic_velocity_to_ket(psi_v, Gk_int, kvec, bvec_blat)
        v_kin_c = apply_kinetic_velocity_to_ket(psi_c, Gk_int, kvec, bvec_blat)
        # VNL apply: build Z, dZ at this k, then apply with compute_dZ=True.
        kdata = vnl_ops.build_vnl_kdata_from_kvec(
            np.asarray(kpts_full[ik], dtype=float),
            np.asarray(Gk_int, dtype=int),
            vnl_setup, compute_dZ=True,
        )
        # vnl_ops applies a global sign flip relative to the BGW convention
        # (see comment in main(): vNL_cart = -vNL_cart). Match here.
        v_NL_v = -vnl_ops.apply_vnl_velocity_to_ket(
            psi_v[:, :int(kdata.E_super.shape[0])],
            kdata.Z, kdata.dZ, kdata.E_super)
        v_NL_c = -vnl_ops.apply_vnl_velocity_to_ket(
            psi_c[:, :int(kdata.E_super.shape[0])],
            kdata.Z, kdata.dZ, kdata.E_super)
        # The VNL apply may return only nspinor_E spinors; pad to full nspinor.
        if v_NL_v.shape[2] < v_kin_v.shape[2]:
            pad = v_kin_v.shape[2] - v_NL_v.shape[2]
            v_NL_v = jnp.concatenate([v_NL_v, jnp.zeros(
                v_NL_v.shape[:2] + (pad,) + v_NL_v.shape[3:], dtype=v_NL_v.dtype)], axis=2)
            v_NL_c = jnp.concatenate([v_NL_c, jnp.zeros(
                v_NL_c.shape[:2] + (pad,) + v_NL_c.shape[3:], dtype=v_NL_c.dtype)], axis=2)
        vket_v_per_k.append(v_kin_v + v_NL_v)
        vket_c_per_k.append(v_kin_c + v_NL_c)

    # ── Per (k, q) loop ──
    nq = len(iq_list)
    rho_cvkq = np.zeros((nc_eff, nv_eff, nk_full, nq), dtype=np.complex128)
    v_cvkq   = np.zeros((3, nc_eff, nv_eff, nk_full, nq), dtype=np.complex128)
    kminq_idx_kq = np.zeros((nk_full, nq), dtype=np.int32)

    for jq, iq_red in enumerate(iq_list):
        kminq_idx = kminq_idx_for_iq(sym, iq_red)
        kminq_idx_kq[:, jq] = kminq_idx

        qvec_pos = np.asarray(wfn.kpoints[iq_red], dtype=np.float64)
        qvec = qvec_pos - np.round(qvec_pos)

        max_rho = max_v = 0.0
        for ik in range(nk_full):
            ikmq = int(kminq_idx[ik])
            kvec_k_np   = kpts_full[ik]
            kvec_kmq_np = kpts_full[ikmq]
            G_wrap_np = np.round((kvec_k_np - qvec) - kvec_kmq_np).astype(np.int32)

            Gk_int_k   = np.asarray(Gk_crys_all[ik],   dtype=np.int32)
            Gk_int_can = np.asarray(Gk_crys_all[ikmq], dtype=np.int32)
            map_arr, mask = _build_g_lookup(Gk_int_can, Gk_int_k, G_wrap_np)
            map_arr_j = jnp.asarray(map_arr, dtype=jnp.int32)
            mask_j    = jnp.asarray(mask)

            rho_mn, v_sym = _cell_overlap_with_lookup(
                psi_c_per_k[ikmq], psi_v_per_k[ik],
                vket_v_per_k[ik],  vket_c_per_k[ikmq],
                map_arr_j, mask_j,
            )
            rho_cvkq[:, :, ik, jq] = np.asarray(rho_mn)
            v_cvkq[:, :, :, ik, jq] = np.asarray(v_sym)
            max_rho = max(max_rho, float(jnp.max(jnp.abs(rho_mn))))
            max_v   = max(max_v,   float(jnp.max(jnp.abs(v_sym))))
        if verbose:
            print(f"    iq={iq_red:>3d}  q_signed={tuple(float(v) for v in qvec)}  "
                  f"|rho|_∞={max_rho:.3e}  |v|_∞={max_v:.3e}")

    return rho_cvkq, v_cvkq, kminq_idx_kq, n_occ, v_lo, c_hi

# --------------------------
# Main driver
# --------------------------

def main(argv=None):
	parser = argparse.ArgumentParser(description="Dipole/velocity matrix elements <mk|v|nk>")
	parser.add_argument(
		"-i",
		"--input",
		default="cohsex.in",
		help="Input file (INI-like) with [cohsex] block",
	)
	parser.add_argument(
		"--divide-energy",
		action="store_true",
		help="Divide selected debug submatrices by ΔE (E_c - E_v). Default: off",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Print 4x6 x-direction debug table (momentum and momentum+vNL) for first k",
	)
	parser.add_argument(
		"--vnl-mode",
		choices=["analytic", "numeric"],
		default="analytic",
		help="Choose nonlocal velocity evaluation: analytic (dZ) or numeric FD(Z)",
	)
	parser.add_argument(
		"--vnl-h",
		type=float,
		default=1e-5,
		help="Finite-difference step h for --vnl-mode=numeric (in |K_cart| units)",
	)
	parser.add_argument(
		"--vnl-h-rel",
		type=float,
		default=0.0,
		help="Relative step: fraction of median |K|; used if larger than --vnl-h",
	)
	parser.add_argument(
		"--vnl-num-scheme",
		choices=["naive", "richardson"],
		default="naive",
		help="Numeric FD on V_NL: naive central or Richardson-extrapolated",
	)
	parser.add_argument(
		"--debug-kindex",
		type=int,
		default=1,
		help="k-point index to use for --debug table (0-based). Default: 1",
	)
	parser.add_argument(
		"--with-finite-q",
		action="store_true",
		help="Also compute finite-q SOS matrix elements rho_mnkq + v_mnkq "
			 "for the c-v block on a list of reduced-BZ q-points.  Output "
			 "datasets are added to dipole.h5 alongside the existing q=0 "
			 "(dipole_cart, deltaE) blocks.",
	)
	parser.add_argument(
		"--iq-list",
		type=int,
		nargs='+',
		default=None,
		help="Reduced-BZ q-indices for --with-finite-q (default: all 0..nk-1).",
	)
	args = parser.parse_args(argv)


	input_path = Path(args.input).resolve()
	params = read_cohsex_input(str(input_path))
	# Resolve WFN relative to input file directory as preferred
	wfn_path = Path(params.get("wfn_file", "WFN.h5"))
	if not wfn_path.is_absolute():
		wfn_path = (input_path.parent / wfn_path).resolve()

	# Open WFN and symmetry
	wfn = WFNReader(str(wfn_path))
	sym = symmetry_maps.SymMaps(wfn)

	nval = int(params.get("nval", 5))
	ncond = int(params.get("ncond", 5))
	# Choose target band count: at least nelec+ncond, clipped to file bands; honor user nband if larger
	try:
		nband_param = params.get("nband", None)
		if nband_param is None:
			nband = max(int(wfn.nbands), int(wfn.nelec) + int(ncond))
		else:
			nband = int(nband_param)
	except Exception:
		nband = max(int(wfn.nbands), int(wfn.nelec) + int(ncond))
	bispinor = bool(params.get("bispinor", False))

	print("\nCreating system metadata...")
	meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, bispinor)

	# JAX mesh (simple 1D default; minimal sharding for demo)
	devices = np.array(jax.devices()).reshape(1, -1)
	mesh_xy = Mesh(devices, ['x', 'y'])

	print("\nLoading wavefunction coefficients to devices...")
	# Ensure we load enough conduction bands for debug/output comparisons
	nband_eff = min(int(wfn.nbands), max(int(wfn.nelec) + int(ncond), int(nband)))
	brange = (0, nband_eff)
	global_psi_G, nb_actual = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
	print(f"  Loaded {nb_actual} bands in G-space, shape: {global_psi_G.shape}")

	print("\nScanning for pseudopotential files...")
	pseudos = load_pseudopotentials(str(input_path.parent))
	if not pseudos:
		# Also try the QE subdirectory (common sandbox layout)
		for fallback in [str(input_path.parent / '..' / 'qe' / 'scf'),
						 str(input_path.parent / '..' / 'qe' / 'nscf')]:
			pseudos = load_pseudopotentials(fallback)
			if pseudos:
				print(f"Found pseudopotentials in {fallback}")
				break

	# Structure summary (reuse DFT helper)
	print_atomic_structure(wfn, pseudos)

	# Precompute G scaffolding for all k-points
	Gk_crys_all: list[jnp.ndarray] = []
	for i in range(sym.nk_tot):
		Gk_crys_i, _ = generate_gvectors_k(i, sym, wfn, meta)
		Gk_crys_all.append(jnp.asarray(Gk_crys_i, dtype=jnp.int32))

	# Build unified VNL setup once; radial tables and custom JAX JVPs stay centralized here.
	vnl_setup = vnl_ops.build_vnl_setup(
		wfn,
		sym,
		meta,
		pseudos,
		nspinor=int(wfn.nspinor),
	)

	# Reshard wavefunctions over mesh
	k_xy_shard = NamedSharding(mesh_xy, P(('x','y'), None, None, None, None, None))
	wfn_k_sharded = jax.lax.with_sharding_constraint(global_psi_G, k_xy_shard)

	nk = sym.nk_tot
	nb = int(wfn_k_sharded.shape[1])
	dipole = np.zeros((3, nk, nb, nb), dtype=np.complex128)
	deltaE = np.zeros((nk, nb, nb), dtype=np.float64)

	for i in range(nk):
		wfn_k = wfn_k_sharded[i]
		kpoint = jnp.asarray(sym.unfolded_kpts[i], dtype=jnp.float64)
		Gk_crys = Gk_crys_all[i]
		# Momentum per component
		p_cart = compute_p_operator_k(
			wfn_k,
			Gk_crys,
			kpoint,
			jnp.asarray(wfn.bdot, dtype=jnp.float64),
			jnp.asarray(wfn.bvec, dtype=jnp.float64),
			float(wfn.blat),
		)  # (3, nb, nb)
		# Nonlocal velocity components via commutator i[r_i, V_NL]
		if args.vnl_mode == "numeric":
			# Numeric derivative on V_NL with optional Richardson and adaptive h
			B = (np.asarray(wfn.bvec, dtype=float)) * float(wfn.blat)
			Binv = np.linalg.inv(B)
			vNL_cart = np.zeros((3, nb, nb), dtype=np.complex128)
			K_cart_this = (np.asarray(Gk_crys, dtype=float) + np.asarray(kpoint, dtype=float)[None, :]) @ B
			K_med = float(np.median(np.linalg.norm(K_cart_this, axis=1))) if K_cart_this.size else 1.0
			h_base = max(float(args.vnl_h), float(args.vnl_h_rel) * max(K_med, 1.0))
			h1 = h_base
			h2 = 0.5 * h_base
			for ic in range(3):
				# D1 at h1
				d1 = np.zeros((3,), dtype=float); d1[ic] = h1
				d1c = d1 @ Binv
				kp1 = np.asarray(kpoint, dtype=float) + d1c
				km1 = np.asarray(kpoint, dtype=float) - d1c
				Vp1 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, kp1, vnl_setup)
				Vm1 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, km1, vnl_setup)
				D1 = - (Vp1 - Vm1) / (2.0 * h1)
				if args.vnl_num_scheme == "richardson":
					# D2 at h2
					d2 = np.zeros((3,), dtype=float); d2[ic] = h2
					d2c = d2 @ Binv
					kp2 = np.asarray(kpoint, dtype=float) + d2c
					km2 = np.asarray(kpoint, dtype=float) - d2c
					Vp2 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, kp2, vnl_setup)
					Vm2 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, km2, vnl_setup)
					D2 = - (Vp2 - Vm2) / (2.0 * h2)
					vNL_cart[ic] = (4.0 * D2 - D1) / 3.0
				else:
					vNL_cart[ic] = D1
		else:
			vNL_cart = compute_vnl_velocity_cart(wfn_k, Gk_crys, kpoint, vnl_setup)

		# Sign convention note (Liu-2024 Eq. 17 / BGW k·p):
		# Our internal assembly returns v^NL = -(∂_q + ∂_{q'}) V_NL, while BGW’s
		# reported ⟨v⟩ uses the opposite sign convention. Flip here so users don’t
		# need --vnl-scale=-1.0 when comparing to BGW outputs.
		vNL_cart = -vNL_cart
		dipole[:, i] = np.asarray(p_cart + vNL_cart)

		# ΔE matrix for this k from band energies
		try:
			k_red = int(sym.irk_to_k_map[i])
		except Exception:
			k_red = int(i)
		energies = np.asarray(wfn.energies)
		if energies.ndim >= 3:
			e_b = np.asarray(energies[0, k_red, :nb], dtype=float)
		else:
			e_b = np.asarray(energies[:nb], dtype=float)
		deltaE[i] = e_b[:, None] - e_b[None, :]

		# Optional debug: print 4x6 x-direction blocks for selected k index
		if args.debug and int(i) == int(args.debug_kindex):
			# Choose up to 6 valence (highest) and up to 4 conduction (lowest) bands
			nelec = int(wfn.nelec)
			v_count = min(6, max(0, nelec))
			c_count = min(4, max(0, nb - nelec))
			if v_count == 0 or c_count == 0:
				print("[DEBUG] Skipping 4x6 debug blocks: insufficient v/c bands (v_count=", v_count, ", c_count=", c_count, ")")
				continue
			v_idx = np.arange(nelec - 1, nelec - v_count - 1, -1, dtype=int)  # descending
			c_idx = np.arange(nelec, nelec + c_count, dtype=int)              # ascending
			p_x = np.asarray(p_cart[0])
			full_x = np.asarray(p_cart[0] + vNL_cart[0])
			if args.divide_energy:
				with np.errstate(divide='ignore', invalid='ignore'):
					dE = deltaE[i]
					p_x = np.where(np.abs(dE) < 1e-12, 0.0, p_x / dE)
					full_x = np.where(np.abs(dE) < 1e-12, 0.0, full_x / dE)
			mom_block = p_x[np.ix_(c_idx, v_idx)]
			full_block = full_x[np.ix_(c_idx, v_idx)]
			print("\n[DEBUG] 4x6 x-direction momentum block (real):")
			for r in range(mom_block.shape[0]):
				print(' '.join(f"{np.real(mom_block[r, c]):.5f}" for c in range(mom_block.shape[1])))
			print("[DEBUG] 4x6 x-direction momentum block (imag):")
			for r in range(mom_block.shape[0]):
				print(' '.join(f"{np.imag(mom_block[r, c]):.5f}" for c in range(mom_block.shape[1])))
			print("[DEBUG] 4x6 x-direction (p + vNL) block (real):")
			for r in range(full_block.shape[0]):
				print(' '.join(f"{np.real(full_block[r, c]):.5f}" for c in range(full_block.shape[1])))
			print("[DEBUG] 4x6 x-direction (p + vNL) block (imag):")
			for r in range(full_block.shape[0]):
				print(' '.join(f"{np.imag(full_block[r, c]):.5f}" for c in range(full_block.shape[1])))

			# 2x3 grid of 2x2 Frobenius norms from the 4x6 (p+vNL) block, matching parse_vmtxel.py
			if full_block.shape[0] >= 4 and full_block.shape[1] >= 6:
				B00 = full_block[0:2, 0:2]
				B01 = full_block[0:2, 2:4]
				B02 = full_block[0:2, 4:6]
				B10 = full_block[2:4, 0:2]
				B11 = full_block[2:4, 2:4]
				B12 = full_block[2:4, 4:6]
				fn00 = float(np.linalg.norm(B00, ord='fro'))
				fn01 = float(np.linalg.norm(B01, ord='fro'))
				fn02 = float(np.linalg.norm(B02, ord='fro'))
				fn10 = float(np.linalg.norm(B10, ord='fro'))
				fn11 = float(np.linalg.norm(B11, ord='fro'))
				fn12 = float(np.linalg.norm(B12, ord='fro'))
				print("[DEBUG] 2x3 grid of 2x2 Frobenius norms (|p+vNL|, x-direction):")
				print(f"  {fn00:.6f} {fn01:.6f} {fn02:.6f}")
				print(f"  {fn10:.6f} {fn11:.6f} {fn12:.6f}")


	# Optional: finite-q matrix elements for the SOS chi head/wing/S/w pipeline.
	rho_cvkq = v_cvkq = kminq_idx_kq = None
	cv_meta = None
	if args.with_finite_q:
		print("\nComputing finite-q matrix elements (SOS pipeline)...")
		iq_list = args.iq_list if args.iq_list is not None else list(range(int(sym.nk_tot)))
		rho_cvkq, v_cvkq, kminq_idx_kq, n_occ_eff, v_lo, c_hi = compute_finite_q_mtxels(
			wfn, sym, meta, vnl_setup, wfn_k_sharded, Gk_crys_all,
			iq_list=iq_list,
			nv_block=int(nval),
			nc_block=int(ncond),
			verbose=True,
		)
		cv_meta = {
			'iq_list': np.asarray(iq_list, dtype=np.int32),
			'n_occ': int(n_occ_eff),
			'v_lo': int(v_lo),
			'c_hi': int(c_hi),
		}

	# Save to dipole.h5 with deltaE
	out_path = Path('dipole.h5').resolve()
	with h5py.File(str(out_path), 'w') as h5:
		h5.create_dataset('dipole_cart', data=dipole)
		h5.create_dataset('deltaE', data=deltaE)
		h5.attrs['nbands'] = int(wfn.nbands)
		h5.attrs['nk'] = int(sym.nk_tot)
		h5.attrs['note'] = 'dipole_cart[3,x,y] = p_i + i[r_i, V_NL]; deltaE[k,:,:] = E_b - E_b\''
		if rho_cvkq is not None:
			fq = h5.create_group('finite_q')
			fq.create_dataset('rho_cvkq', data=rho_cvkq)         # (nc, nv, nk, nq)
			fq.create_dataset('v_cvkq',   data=v_cvkq)           # (3, nc, nv, nk, nq)
			fq.create_dataset('kminq_idx', data=kminq_idx_kq)    # (nk, nq)
			fq.create_dataset('iq_list',   data=cv_meta['iq_list'])
			fq.attrs['n_occ']   = cv_meta['n_occ']
			fq.attrs['v_lo']    = cv_meta['v_lo']
			fq.attrs['c_hi']    = cv_meta['c_hi']
			fq.attrs['note'] = (
				"rho_cvkq[c, v, k, q] = <u_{c, k-q}|u_{v, k}>_cell; "
				"v_cvkq[a, c, v, k, q] = symmetric (v_R + v_L)/2 of "
				"<u_{c, k-q}|v^a|u_{v, k}>_cell  (kinetic + VNL); "
				"kminq_idx[k, q] = canonical k-q index in unfolded_kpts.")
	print(f"\nWrote dipole data to {out_path}")


if __name__ == '__main__':
    main()
