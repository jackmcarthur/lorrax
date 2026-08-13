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
import time
from pathlib import Path

# THE startup call (runtime module docstring), before any device is
# touched.  Two of its steps used to be called here by name and the rest
# were skipped.  Without ``jax.distributed`` every rank of a multi-process
# launch is its own single-process job: ``jax.process_count()`` reads 1
# everywhere, so ``collectives.local_share`` hands EVERY rank the whole
# k list and the closing ``gather_indexed_blocks`` never runs — P ranks
# each do the whole sweep and write the same file.  Without the mesh
# clique warm-up that gather dies at P>1 under impl=mpi when its
# communicator is first created from an XLA pool thread (job 7884867
# class).  The startup call does both, in the right order, above jax.
from runtime import initialize_communicator_stack
RUNTIME = initialize_communicator_stack()

import numpy as np
import jax
import jax.numpy as jnp

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import timing
from common.collectives import barrier, gather_k_blocks
from common.mtxel_sweep import (VNL_VELOCITY_SIGN_FLIPPED,
                                VNL_VELOCITY_SIGN_SHIPPED, SweepGeometry,
                                band_sphere_spec, blocks_to_host,
                                dipole_operator, sweep_matrix_elements)
from common.parallel_transport import wfn_fingerprint
from common.wfn_transforms import load_kpoint_fftbox_local
from common import Meta
from gw.gw_config import read_lorrax_input as read_cohsex_input
from psp.pseudos import load_pseudopotentials, print_atomic_structure
from psp.dft_operators import (padded_gvectors, gather_psi_G_from_crys,
                               momentum_matrix_k)
import psp.vnl_ops as vnl_ops
import h5py
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import symmetry_maps                                            # noqa: E402
# --------------------------
# K+G helpers
# --------------------------


def compute_p_operator_k(wfn_k: jax.Array, Gk_crys: np.ndarray, kpoint_crys: np.ndarray, bdot: np.ndarray, bvec: np.ndarray, blat: float, *, g_mask=None) -> jax.Array:
	"""Compute p-operator matrix elements per Cartesian component.

	Returns array of shape (3, nb, nb) for components x,y,z.
	p_i = sum_G (k+G)_cart[i] c*_mk(G) c_nk(G)

	``g_mask`` is 1 on the k's physical G and 0 on the ngkmax pad.  It
	is applied to ψ_G, which is enough: the sum above closes against
	conj(ψ) so a zeroed column contributes nothing.  Omitting it on a
	padded G-list does NOT crash — pad rows hold (0,0,0), which aliases
	Γ — it silently adds (ngkmax − ngk) extra copies of the Γ term.
	"""
	C_bsg = gather_psi_G_from_crys(wfn_k, Gk_crys, g_mask)
	k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
	G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
	B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
	return momentum_matrix_k(C_bsg, G_int, k_crys, B)


def compute_vnl_matrix_from_setup(
	wfn_k: jax.Array,
	Gk_crys: np.ndarray,
	kpoint_crys: np.ndarray,
	vnl_setup,
	*,
	g_mask=None,
) -> jax.Array:
	"""Return <m|V_NL(k)|n> using the unified JAX VNL setup."""
	kdata = vnl_ops.build_vnl_kdata_from_kvec(
		np.asarray(kpoint_crys, dtype=float),
		np.asarray(Gk_crys, dtype=int),
		vnl_setup,
		compute_dZ=False,
	)
	# Z is finite on the pad columns (it is evaluated at K = kvec there),
	# so the mask has to reach ψ — every contraction in ``vnl_matrix``
	# runs through ψ_G at least once.
	psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys, g_mask)
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
	*,
	g_mask=None,
) -> jax.Array:
	"""Return dV_NL/dK_cart using the unified JAX VNL path."""
	kdata = vnl_ops.build_vnl_kdata_from_kvec(
		np.asarray(kpoint_crys, dtype=float),
		np.asarray(Gk_crys, dtype=int),
		vnl_setup,
		compute_dZ=True,
	)
	psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys, g_mask)
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
                     G_wrap: np.ndarray, *,
                     ngk_kmq: int, ngk_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-(k, q) integer lookup that translates between the G-spheres of
    ``k`` and ``canonical(k − q)`` under umklapp.

    For each μ_k in the ket's G-sphere, we want the μ_kmq such that
    ``Gk_int_kmq[μ_kmq] == Gk_int_k[μ_k] + G_wrap``.  When that G-vector
    lies outside the canonical-kmq sphere we mark it −1 (the bra
    coefficient there is zero anyway, so the contribution to the
    overlap is zero).

    Both G-lists are the loader's fixed ``(ngkmax, 3)`` tables, so the
    PHYSICAL extents ``ngk_kmq`` / ``ngk_k`` are passed explicitly and the
    pad rows take no part:

    * the bra-side dictionary is built over the first ``ngk_kmq`` rows —
      the pad rows are all ``(0,0,0)``, so including them would rebind
      the Γ key to the LAST pad index and every ket G that maps to Γ
      would then read a zero coefficient;
    * the ket-side loop runs over the first ``ngk_k`` rows and the rest
      of the mask is False — a pad row would otherwise look up
      ``(0,0,0) + G_wrap``, which is frequently a real member of the bra
      sphere, and contribute a spurious ψ(Γ) term.

    The Python dictionary work is therefore unchanged from the ragged
    route; only the returned arrays are widened to ``ngkmax``.

    Returns
    -------
    map_arr : (ngkmax,) int32  — μ_kmq index per μ_k (0 placeholder where
              not found or padded; use ``mask`` to gate).
    mask    : (ngkmax,) bool   — True where the lookup succeeded.
    """
    ngkmax = int(Gk_int_k.shape[0])
    ngk_k = int(ngk_k)
    target = Gk_int_k[:ngk_k] + G_wrap[None, :]                         # (ngk_k, 3)
    g_dict = {tuple(int(x) for x in g): i
              for i, g in enumerate(Gk_int_kmq[:int(ngk_kmq)])}
    map_arr = np.zeros(ngkmax, dtype=np.int32)
    mask = np.zeros(ngkmax, dtype=bool)
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
    wfn, sym, meta, vnl_setup, gtab,
    *,
    nb: int,
    bispinor: bool,
    iq_list: list[int],
    nv_block: int,
    nc_block: int,
    vnl_velocity_sign: float = VNL_VELOCITY_SIGN_SHIPPED,
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
      • G-sphere throughout.  Each k's FFT box is loaded, gathered to
        that k's G-sphere and DROPPED before the next k
        (:func:`common.wfn_transforms.load_kpoint_fftbox_local`), so the
        box residency is one k's worth, not ``n_k``'s.  What survives the
        loop is the four G-sphere lists below, which are ~4 % of the box
        per k (``ngkmax`` vs ``nx·ny·nz``: 1964 vs 46080 on MoS₂ 4×4).
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
    c_hi = min(n_occ + int(nc_block), int(nb))
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

    # Per-k (kdata, vket_v, vket_c).  Every k now presents the SAME
    # (ngkmax) G-axis, so these lists hold uniformly-shaped arrays and
    # each kernel below lowers once for the whole sweep.
    #
    # THESE FOUR LISTS ARE THE REMAINING n_k-SCALING RESIDENCY of the
    # finite-q path, and they are NOT removable by streaming: the (k, q)
    # loop below pairs ket k with bra canonical(k−q), so an arbitrary
    # pair of k must be live at once.  Cost is
    # ``n_k · 4 · (nv+nc) · ns · ngkmax · 16`` B — 514 MB on MoS₂ 4×4 at
    # nval=26/ncond=102, against the 3.0 GB of FFT box the old route held
    # for the same sweep.  Sharding them needs a k-partition of the (k, q)
    # loop plus one gather of the G-sphere blocks; it is not done here.
    vket_v_per_k = []     # (3, nv, ns, ngkmax)  each
    vket_c_per_k = []     # (3, nc, ns, ngkmax)  each
    psi_v_per_k  = []     # (nv, ns, ngkmax)
    psi_c_per_k  = []     # (nc, ns, ngkmax)
    for ik in range(nk_full):
        kvec = jnp.asarray(kpts_full[ik], dtype=jnp.float64)
        G_pad, g_mask = gtab.at(ik)
        Gk_int = jnp.asarray(G_pad, dtype=jnp.int32)                   # (ngkmax, 3)
        # ONE k's FFT box, process-local: (nb, ns, nx, ny, nz).  The
        # previous route indexed a resident (nk, nb, ns, nx, ny, nz)
        # array, i.e. it held EVERY k's box for the whole call.
        psi_k = load_kpoint_fftbox_local(wfn, meta, ik, int(nb),
                                         bispinor=bool(bispinor))
        # Gather FFT-box layout → G-sphere coeffs at this k's integer G-list.
        # The mask zeroes the pad columns, which is what makes the finite,
        # K=kvec values of Z/dZ on those columns inert downstream.
        psi_k_G = gather_psi_G_from_crys(psi_k, Gk_int, g_mask)        # (nb, ns, ngkmax)
        del psi_k                       # the box dies here, not at loop end
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
        # THE SAME KNOB THE q = 0 ROUTES READ.  These tables feed the SOS
        # chi head/wing pipeline, which is a consumer of the assembled
        # velocity exactly like ``dipole_cart`` is, so a sign that moved
        # at q = 0 and not here would leave one run's head and its wings
        # built from two different operators.  Written as a branch rather
        # than a multiply so the shipped arm executes the SAME negation
        # it always did.
        _vel_v = vnl_ops.apply_vnl_velocity_to_ket(
            psi_v[:, :int(kdata.E_super.shape[0])],
            kdata.Z, kdata.dZ, kdata.E_super)
        _vel_c = vnl_ops.apply_vnl_velocity_to_ket(
            psi_c[:, :int(kdata.E_super.shape[0])],
            kdata.Z, kdata.dZ, kdata.E_super)
        if vnl_velocity_sign < 0.0:
            v_NL_v, v_NL_c = -_vel_v, -_vel_c
        else:
            v_NL_v, v_NL_c = _vel_v, _vel_c
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

            Gk_int_k   = np.asarray(gtab.gvecs[ik],   dtype=np.int32)
            Gk_int_can = np.asarray(gtab.gvecs[ikmq], dtype=np.int32)
            map_arr, mask = _build_g_lookup(
                Gk_int_can, Gk_int_k, G_wrap_np,
                ngk_kmq=int(gtab.ngk[ikmq]), ngk_k=int(gtab.ngk[ik]))
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
# Provenance: which WFN and which band window produced this dipole.h5
# --------------------------
#
# ``dipole.h5`` is generated ONCE, out of band, and then read by every
# later GW run in the directory.  Its contents depend on (a) the WFN's
# eigenstates and (b) the deck's band window — and nothing on disk
# recorded either, so regenerating WFN.h5 (new nbnd, new NSCF, new
# k-grid) and leaving the old dipole.h5 in place produces a file of the
# right SHAPE with the wrong CONTENTS.  ``gw.head_correction`` already
# checks the band COUNT and ``nk``; neither catches that case.
#
# The fingerprint is over the eigenvalue table and the k-list, i.e. the
# two things that change whenever the underlying DFT solution changes,
# and it is cheap because ``WfnLoader`` has both in memory already.

_PROV_ATTRS = ("prov_wfn_sha256", "prov_nval", "prov_ncond", "prov_nband",
               "prov_nb_written", "prov_bispinor", "prov_skip_vnl",
               "prov_vnl_mode", "prov_wfn_file", "prov_vnl_velocity_sign")

#: Word spellings of the two arms, so a deck can say which one it means
#: rather than carrying a bare ``-1`` whose meaning is a source comment.
_VNL_SIGN_WORDS = {"shipped": VNL_VELOCITY_SIGN_SHIPPED,
                   "minus": VNL_VELOCITY_SIGN_SHIPPED,
                   "flipped": VNL_VELOCITY_SIGN_FLIPPED,
                   "plus": VNL_VELOCITY_SIGN_FLIPPED}


def resolve_vnl_velocity_sign(cli_value, deck_value):
    """Which sign the i[r, V_NL] term enters this run's velocity with.

    Three tiers, in the order a reader would guess: an explicit
    ``--vnl-velocity-sign`` beats the deck key ``vnl_velocity_sign``,
    which beats the default.  ``None`` and the empty string both mean
    NOT DECLARED, so a deck that has never heard of the key and a deck
    that omits it are the same run -- the same reading
    ``resolve_soc_mode`` gives ``soc``.

    The default is
    :data:`common.mtxel_sweep.VNL_VELOCITY_SIGN_FLIPPED`, and it is read
    from there rather than written here so that the producer and the
    operator cannot disagree about what "as shipped" means.  The value
    is stamped into ``dipole.h5`` by :func:`stamp_dipole_provenance`,
    because the two arms differ by 31 % in eps00(0) on silicon and a
    file that does not say which one built it is a file nobody can
    attribute.
    """
    raw = cli_value if cli_value is not None else deck_value
    if isinstance(raw, str):
        raw = raw.strip().lower()
        raw = _VNL_SIGN_WORDS.get(raw, raw)
    if raw is None or raw == "":
        return VNL_VELOCITY_SIGN_FLIPPED
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = None
    if val not in (VNL_VELOCITY_SIGN_SHIPPED, VNL_VELOCITY_SIGN_FLIPPED):
        raise ValueError(
            f"GATE vnl_velocity_sign: {raw!r} resolves to no arm.  The only "
            f"values are {VNL_VELOCITY_SIGN_SHIPPED} (shipped, the default) "
            f"and {VNL_VELOCITY_SIGN_FLIPPED} (flipped); the words "
            f"{sorted(_VNL_SIGN_WORDS)} spell the same two.  This is a "
            f"SIGN, not a scale: an arbitrary multiplier would produce a "
            f"velocity operator that is neither arm of the open question "
            f"and that no comparison with BerkeleyGW characterises.")
    return val


def stamp_dipole_provenance(h5, *, wfn, wfn_path, nval, ncond, nband,
                             nb_written, bispinor, skip_vnl, vnl_mode,
                             vnl_velocity_sign=None) -> None:
    """Record what this ``dipole.h5`` was built from.

    ``vnl_velocity_sign`` is the RESOLVED relative sign of the nonlocal
    commutator term, on the same reading: ``None`` is a file written
    before the knob existed, which is a file built with the shipped
    ``-1`` but which cannot say so on its own authority.  It is stamped
    unconditionally once resolved because the two arms differ by 31 % in
    eps00(0) on silicon with the SHAPE of the frequency dependence still
    right -- one global scale of 1.377 on (eps - 1) leaves a 0.3 %
    median residual ON THE IMAGINARY AXIS -- so nothing downstream will
    notice which arm it was handed.
    """
    h5.attrs["prov_wfn_sha256"] = wfn_fingerprint(wfn)
    h5.attrs["prov_wfn_file"] = str(wfn_path)
    h5.attrs["prov_nval"] = int(nval)
    h5.attrs["prov_ncond"] = int(ncond)
    h5.attrs["prov_nband"] = int(nband)
    h5.attrs["prov_nb_written"] = int(nb_written)
    h5.attrs["prov_bispinor"] = bool(bispinor)
    h5.attrs["prov_skip_vnl"] = bool(skip_vnl)
    h5.attrs["prov_vnl_mode"] = str(vnl_mode)
    if vnl_velocity_sign is not None:
        h5.attrs["prov_vnl_velocity_sign"] = float(vnl_velocity_sign)


def check_dipole_provenance(path, *, wfn, nval, ncond, nband,
                             print_fn=print) -> bool:
    """Does ``path`` match the WFN and band window of the CURRENT run?

    Returns True only when a stamp exists AND agrees.  Disagreement goes
    through ``common.sanity.warn`` (the same channel
    ``gw.head_correction`` uses for its coverage check) so a strict run
    turns it into a refusal and a permissive one still prints loudly.
    A MISSING stamp is reported as such and returns False — an
    unstamped file predates this guard and cannot be vouched for.
    """
    from common import sanity

    try:
        with h5py.File(str(path), "r") as h5:
            attrs = {k: h5.attrs[k] for k in _PROV_ATTRS if k in h5.attrs}
    except OSError as exc:
        print_fn(f"  [dipole provenance] cannot open {path} "
                 f"({type(exc).__name__}: {exc})")
        return False

    if "prov_wfn_sha256" not in attrs:
        print_fn(f"  [dipole provenance] {path} carries no provenance stamp "
                 f"(written before the guard existed).  Regenerate with "
                 f"`python -m psp.get_dipole_mtxels` to make it checkable.")
        return False

    want = {"prov_wfn_sha256": wfn_fingerprint(wfn),
            "prov_nval": int(nval), "prov_ncond": int(ncond),
            "prov_nband": int(nband)}
    bad = [(k, attrs[k], v) for k, v in want.items()
           if k in attrs and _prov_ne(attrs[k], v)]
    if bad:
        detail = "; ".join(f"{k}: file={_prov_show(got)} run={_prov_show(exp)}"
                           for k, got, exp in bad)
        sanity.warn(
            f"{path} was generated from a DIFFERENT DFT solution or band "
            f"window than this run ({detail}).  dipole.h5 has the right "
            f"shape either way, so nothing downstream will notice: the q→0 "
            f"head S(ω), and every Σ_SX/Σ_COH correction built from it, "
            f"would be assembled from stale velocity matrix elements.  "
            f"Regenerate it with `python -m psp.get_dipole_mtxels -i <deck>`.",
            print_fn=print_fn)
        return False

    print_fn(f"  dipole.h5 provenance OK (WFN {want['prov_wfn_sha256'][:12]}…, "
             f"window nval={int(nval)} ncond={int(ncond)} nband={int(nband)})")
    return True


def _prov_ne(got, expected) -> bool:
    if isinstance(expected, str):
        got = got.decode() if isinstance(got, bytes) else str(got)
        return got != expected
    return int(np.asarray(got)) != int(expected)


def _prov_show(v) -> str:
    if isinstance(v, bytes):
        v = v.decode()
    return (v[:12] + "…") if isinstance(v, str) and len(v) > 13 else str(v)


# --------------------------
# Main driver
# --------------------------

def main(argv=None):
	_t_main = time.perf_counter()
	parser = argparse.ArgumentParser(allow_abbrev=False, description="Dipole/velocity matrix elements <mk|v|nk>")
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
		help="Nonlocal velocity evaluation.  `analytic` (dZ) is the "
		     "production arm and the default -- every dipole.h5 in the "
		     "tree is built with it.  `numeric` is the FINITE-DIFFERENCE "
		     "VALIDATION ARM for the analytic path: it exists to check "
		     "that the analytic derivative is implemented right, it is "
		     "far slower, and it is NOT FOR PRODUCTION RUNS.  The two "
		     "are gated against each other by "
		     "tests/test_vnl_velocity_fd_agreement.py.",
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
		"--skip-vnl",
		action="store_true",
		help="Skip the i[r,V_NL] commutator term — write p̂ only. Used to "
		     "match BGW's `use_momentum` keyword for apples-to-apples "
		     "absorption comparison.",
	)
	parser.add_argument(
		"--pseudo-dir", "--pseudo_dir",
		dest="pseudo_dir",
		default=None,
		help="Directory holding the deck's *.upf files.  Default: the input "
		     "file's own directory, then ../qe/scf and ../qe/nscf.  THREE OF "
		     "THE FOUR tests/regression decks do not carry their UPFs (only "
		     "cohsex_debug does), so a fixture re-cut from a clean checkout "
		     "needs this flag — see gw.kin_ion_io, which has had it all along.",
	)
	parser.add_argument(
		"--vnl-velocity-sign",
		type=float,
		choices=[VNL_VELOCITY_SIGN_SHIPPED, VNL_VELOCITY_SIGN_FLIPPED],
		default=None,
		help="Relative sign of the i[r,V_NL] term in the assembled "
		     "velocity: -1 is the shipped assembly and the default, +1 is "
		     "the flipped arm.  Overrides the deck key `vnl_velocity_sign`; "
		     "with neither given the shipped sign is used and every "
		     "dipole.h5 in the tree is reproduced bit for bit.  The "
		     "resolved value is stamped as `prov_vnl_velocity_sign`.",
	)
	parser.add_argument(
		"--out",
		type=str,
		default="dipole.h5",
		help="Output filename (default: dipole.h5)",
	)
	parser.add_argument(
		"--parallel-transport-out",
		type=str,
		default=None,
		help="Opt in to the SlabIO parallel-transport preprocessing artifact. "
		     "The default dipole path and dipole.h5 schema are unchanged.",
	)
	parser.add_argument(
		"--parallel-transport-rcond",
		type=float,
		default=1.0e-10,
		help="Relative singular-value cutoff passed to distributed "
		     "polar_factor (default: 1e-10).",
	)
	parser.add_argument(
		"--parallel-transport-validation-atol",
		type=float,
		default=5.0e-4,
		help="Absolute tolerance for the mandatory reconstructed-vs-exact "
		     "DFT velocity gate (default: 5e-4).",
	)
	parser.add_argument(
		"--parallel-transport-validation-rtol",
		type=float,
		default=5.0e-3,
		help="Relative tolerance for the mandatory reconstructed-vs-exact "
		     "DFT velocity gate (default: 5e-3).",
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

	if args.parallel_transport_out is not None:
		if Path(args.parallel_transport_out).resolve() == Path(args.out).resolve():
			parser.error(
				"--parallel-transport-out and --out must name different files; "
				"the dipole writer opens --out with truncation")
		if args.vnl_mode != "analytic":
			parser.error(
				"--parallel-transport-out requires --vnl-mode=analytic: "
				"the artifact stores the exact sharded DFT velocity from "
				"the production sweep, not the gathered numeric debug arm")
		if args.skip_vnl:
			parser.error(
				"--parallel-transport-out cannot be combined with --skip-vnl: "
				"velocity_dft_cart must contain p + i[r,V_NL]")
		if not np.isfinite(args.parallel_transport_rcond) \
				or float(args.parallel_transport_rcond) <= 0.0:
			parser.error("--parallel-transport-rcond must be finite and positive")
		for name, value in (
				("--parallel-transport-validation-atol",
				 args.parallel_transport_validation_atol),
				("--parallel-transport-validation-rtol",
				 args.parallel_transport_validation_rtol),
		):
			if not np.isfinite(value) or float(value) < 0.0:
				parser.error(f"{name} must be finite and non-negative")

	input_path = Path(args.input).resolve()
	params = read_cohsex_input(str(input_path))
	# The relative sign of i[r, V_NL] in the assembled velocity, resolved
	# ONCE here so that the two producer routes below -- the analytic
	# ``dipole_operator`` sweep and the numeric finite-difference block --
	# cannot take different arms in one run, and so that the value the
	# file is stamped with is the value that was used rather than the
	# value that was asked for.  See ``mtxel_sweep.dipole_operator``'s
	# SIGN section for the measurement that makes this an open question.
	vnl_velocity_sign = resolve_vnl_velocity_sign(
		args.vnl_velocity_sign, params.get("vnl_velocity_sign", ""))
	# The four-arm table names the arms by the SIGN OF i[r, V_NL] in the
	# stored convention, which is the opposite of the sign this knob
	# carries -- the internal assembly returns -(∂_q + ∂_q') V_NL and the
	# knob multiplies that.  Both spellings are printed together so a log
	# can be read against the table without doing the flip in one's head.
	_arm = ("p + i[r, V_NL]  (LEGACY -1 arm)" if vnl_velocity_sign < 0.0
	        else "p - i[r, V_NL]  (default since 2026-08-09)")
	if not args.skip_vnl:
		print(f"  velocity assembly: {_arm}, "
		      f"vnl_velocity_sign = {vnl_velocity_sign:+.1f}")
	# Resolve WFN relative to input file directory as preferred
	wfn_path = Path(params.get("wfn_file", "WFN.h5"))
	if not wfn_path.is_absolute():
		wfn_path = (input_path.parent / wfn_path).resolve()

	# Open WFN and symmetry.  The mesh comes from the module-top
	# ``initialize_communicator_stack()``; with it, ``backend=auto`` can
	# pick the collective phdf5 read at P>1 instead of the per-rank eager
	# h5py read (scorecard BD.2).  The per-k sweep below stays on the
	# process-local read path either way.
	wfn = WfnLoader(str(wfn_path))
	wfn.adopt_mesh(RUNTIME.mesh)
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

	# Every communicator the sweep and the closing gather will use was
	# warmed by the module-top ``initialize_communicator_stack()``
	# (mandatory under impl=mpi, from the main thread).  ``RUNTIME.mesh``
	# is the run's own ('x','y') mesh and IS used below: the k-scan holds
	# ψ and ⟨mk|v|nk⟩ as globally-sharded arrays over it.

	# Ensure we load enough conduction bands for debug/output comparisons.
	# ψ is NOT loaded here — see the k sweep below.
	nband_eff = min(int(wfn.nbands), max(int(wfn.nelec) + int(ncond), int(nband)))

	print("\nScanning for pseudopotential files...")
	searched = [str(args.pseudo_dir)] if args.pseudo_dir else [str(input_path.parent)]
	pseudos = load_pseudopotentials(searched[0])
	if not pseudos and not args.pseudo_dir:
		# Also try the QE subdirectory (common sandbox layout)
		for fallback in [str(input_path.parent / '..' / 'qe' / 'scf'),
						 str(input_path.parent / '..' / 'qe' / 'nscf')]:
			searched.append(fallback)
			pseudos = load_pseudopotentials(fallback)
			if pseudos:
				print(f"Found pseudopotentials in {fallback}")
				break

	# ── PRE-FLIGHT.  THE ONE CHECK THIS DRIVER NEVER RAN. ────────────────
	# ``psp.operator_checks`` was written for exactly three callers and its
	# own module docstring names them: "before computing kin+ion, DIPOLE
	# matrix elements, or any other quantity that depends on
	# pseudopotentials".  ``gw.kin_ion_io`` and ``psp.get_DFT_mtxels`` call
	# it; this driver never did, and that omission is the whole defect
	# behind the 2026-08-09 ``kdata.dZ is None`` blocker.
	#
	# WITHOUT PSEUDOS THIS DRIVER DOES NOT REFUSE — IT PRODUCES THREE
	# DIFFERENT WRONG THINGS, one per arm, and only one of them is loud:
	#
	#   --vnl-mode analytic (DEFAULT)  ``build_vnl_setup`` returns a setup
	#       with ``channels == []``, so ``_build_vnl_kdata_core`` has no
	#       ``dZ`` block to concatenate and hands back ``dZ=None``.  Thirty
	#       seconds later ``apply_vnl_velocity_to_ket`` conjugates it:
	#       ``TypeError: conjugate requires ndarray or scalar arguments,
	#       got <class 'NoneType'>`` — a stack six frames inside a jitted
	#       einsum that names neither the deck nor the missing file.
	#   --vnl-mode numeric   finite-differences a projector set that is
	#       EMPTY, so V_NL ≡ 0.  rc=0, an h5 written, and
	#       ``prov_skip_vnl=False`` stamped on a file that has no V_NL in
	#       it.  MEASURED on si_cohsex_debug: that artifact agrees with the
	#       ``--skip-vnl`` run to 5.8e-15 — i.e. it IS the --skip-vnl run,
	#       wearing the other arm's provenance.
	#   --skip-vnl           correct, and the only arm entitled to run
	#       without pseudopotentials at all.
	#
	# So the refusal is gated on ``--skip-vnl``, not on the mode: the p̂-only
	# arm genuinely needs no projectors, and every other arm needs them or
	# it is lying in its provenance block.
	if not args.skip_vnl:
		# Imported here, not at module scope, for the same reason
		# ``get_DFT_mtxels`` does it: ``operator_checks`` runs
		# ``_services.ensure_on_path()`` at import time and this module's
		# own ``ffi`` bootstrap is further down the import block.
		from psp.operator_checks import validate_operator_inputs
		try:
			sys_dim = int(params.get("sys_dim", 3))
		except (TypeError, ValueError):
			sys_dim = 3
		try:
			validate_operator_inputs(pseudos=pseudos, wfn=wfn,
			                          sys_dim=sys_dim,
			                          caller="get_dipole_mtxels")
		except RuntimeError as exc:
			raise SystemExit(
				f"{exc}\n"
				f"  searched: {', '.join(searched)}\n"
				"  The dipole is p + i[r, V_NL]; without projectors the "
				"nonlocal half is silently zero (--vnl-mode numeric) or "
				"crashes inside the sweep with 'conjugate ... got NoneType' "
				"(--vnl-mode analytic).\n"
				"  Fix: stage the deck's *.upf next to the input file, pass "
				"--pseudo-dir DIR, or ask for p̂ only with --skip-vnl."
			) from exc

	# Structure summary (reuse DFT helper)
	print_atomic_structure(wfn, pseudos)

	# G scaffolding: the loader's own fixed-shape (nk, ngkmax, 3) table
	# plus its pad mask (owner decision D10).  Every per-k kernel below
	# therefore sees ONE operand shape for the whole sweep.
	gtab = padded_gvectors(wfn, k="full_bz")

	# Build unified VNL setup once; radial tables and custom JAX JVPs stay centralized here.
	vnl_setup = vnl_ops.build_vnl_setup(
		wfn,
		sym,
		meta,
		pseudos,
		nspinor=int(wfn.nspinor),
	)

	nk = int(sym.nk_tot)
	nb = int(nband_eff)

	# ── ΔE: pure host arithmetic on the band energy table ───────────────
	# No ψ, no device, nk·nb²·8 B (2 MB at MoS₂ 4×4 / 128 bands), so it is
	# built for every k on every rank instead of riding the k partition and
	# paying a second gather.  Arithmetic is verbatim what the fused loop
	# did, which is why the pinned ``deltaE`` parity is EXACTLY 0.
	#
	# ΔE IS PROVABLY REDUNDANT AND IS STILL NOT WORTH COMPRESSING — measured
	# 2026-08-08 on all four committed dipole.h5 fixtures, and written down
	# because the redundancy is obvious enough that it will keep being
	# proposed and the numbers settle it in either direction.
	#
	# THE REDUNDANCY IS TOTAL.  ``deltaE[k]`` is bit-identical — max|Δ|
	# exactly 0.000e+00, not "agrees to round-off" — to the outer difference
	# of a single WFN eigenvalue row, at every k of every fixture.  The whole
	# (nk, nb, nb) f64 array therefore carries at most (nrk, nb) numbers, and
	# those numbers are already in WFN.h5:
	#
	#     deck               deltaE      as (nrk, nb)   on the dataset
	#     cohsex_debug       0.87 MB        2 640 B         330x
	#     gnppm_debug        0.46 MB        3 200 B         144x
	#     hbn_cohsex_debug   0.92 MB       11 520 B          80x
	#     si_cohsex_debug    1.84 MB        3 840 B         480x
	#
	# THE FILE BARELY MOVES.  ``deltaE`` is 14.3 % of dipole.h5 on all four,
	# and that fraction is structural rather than incidental: ``dipole_cart``
	# is three complex128 planes against one f64 plane, exactly 6:1.  So
	# deleting ΔE outright takes dipole.h5 to 85.8 % of its size — 1.17x —
	# and the remaining 85.7 % is the half carrying a Cartesian index, which
	# needs the proper-rotation treatment and is exactly why the dipole was
	# REGISTERED rather than claimed.  The redundancy is total in the half
	# that was never the problem.
	#
	# NOT IMPLEMENTED, deliberately.  It would touch three sites — this
	# writer, ``bse.absorption_common.load_dipole_h5`` and
	# ``common.chi_from_dipole.read_dipole_h5``, none of which has a consumer
	# cell in the tree today — to buy 1.17x.  FOR THE OWNER: if
	# ``dipole_cart``'s rotation work is ever done, take the ΔE half in the
	# SAME change.  Its correctness is free — store the ``e_b`` vector this
	# loop already holds and rebuild with this same expression, bit-identical
	# by construction rather than by measurement — and it is 14.2 % of the
	# file on top of whatever ``dipole_cart`` buys.
	#
	# ONE FIXTURE ANOMALY, recorded rather than chased.  On cohsex_debug, 3
	# of 9 k reproduce ``el[0, 1]`` where today's ``SymMaps`` gives
	# ``irr_idx_k[k] = 2``.  Both rows reproduce the committed ΔE
	# bit-identically through the row that matches, and the two rows differ
	# from each other by 1.066e-14 Ry — so that fixture's k→IBZ map and this
	# tree's are physically equivalent and not the same map.  It says nothing
	# about the redundancy, which holds on that deck too, and everything
	# about the age of the fixture.
	energies = np.asarray(wfn.energies)
	deltaE = np.zeros((nk, nb, nb), dtype=np.float64)
	for i in range(nk):
		try:
			k_red = int(sym.irr_idx_k[i])
		except Exception:
			k_red = int(i)
		if energies.ndim >= 3:
			e_b = np.asarray(energies[0, k_red, :nb], dtype=float)
		else:
			e_b = np.asarray(energies[:nb], dtype=float)
		deltaE[i] = e_b[:, None] - e_b[None, :]

	def _print_debug_blocks(i, p_cart, vNL_cart):
		"""The --debug 4x6 tables for one k.  Forces a readback; debug only."""
		# Choose up to 6 valence (highest) and up to 4 conduction (lowest) bands
		nelec = int(wfn.nelec)
		v_count = min(6, max(0, nelec))
		c_count = min(4, max(0, nb - nelec))
		if v_count == 0 or c_count == 0:
			print("[DEBUG] Skipping 4x6 debug blocks: insufficient v/c bands (v_count=", v_count, ", c_count=", c_count, ")")
			return
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

	def _dipole_block(i):
		"""⟨mk|v|nk⟩ at this run's arm, for ONE k — ``(3, nb, nb)`` on device.

		THE LOCAL PLAN, kept for two callers only: ``--vnl-mode=numeric``
		(whose finite difference picks its step from THIS k's median |K|
		on the host, and costs 4–8 extra projector builds per component
		per k) and the ``--debug`` table, which needs p and p+v_NL
		SEPARATELY — the sweep sums them on the ket and no longer has
		them apart.  The default analytic path is
		``common.mtxel_sweep``; see the sweep below.

		THE MEMORY CONTRACT, and why the default no longer pays it.  ψ
		enters through ``load_kpoint_fftbox_local``, which reads and
		boxes exactly this k: ``nb·nspinor·nx·ny·nz·16`` B, 189 MB on
		MoS₂ 4×4 at 128 bands.  It is dropped when the block returns.
		The sweep forms no box at all — 2(k+G)ψ and ∂V_NL/∂K ψ are
		diagonal in G and a projector sum respectively, so both act on
		the stored G-sphere.
		"""
		wfn_k = load_kpoint_fftbox_local(wfn, meta, i, nb,
		                                 bispinor=bispinor)
		kpoint = jnp.asarray(sym.unfolded_kpts[i], dtype=jnp.float64)
		Gk_crys, g_mask = gtab.at(i)
		# Momentum per component
		p_cart = compute_p_operator_k(
			wfn_k,
			Gk_crys,
			kpoint,
			jnp.asarray(wfn.bdot, dtype=jnp.float64),
			jnp.asarray(wfn.bvec, dtype=jnp.float64),
			float(wfn.blat),
			g_mask=g_mask,
		)  # (3, nb, nb)
		# Nonlocal velocity components via commutator i[r_i, V_NL]
		if args.skip_vnl:
			vNL_cart = np.zeros((3, nb, nb), dtype=np.complex128)
		elif args.vnl_mode == "numeric":
			# Numeric derivative on V_NL with optional Richardson and adaptive h
			B = (np.asarray(wfn.bvec, dtype=float)) * float(wfn.blat)
			Binv = np.linalg.inv(B)
			vNL_cart = np.zeros((3, nb, nb), dtype=np.complex128)
			# Physical rows only: the pad rows are G=(0,0,0), so including
			# them would drag the median |K| toward |k| and shrink the FD step.
			G_phys = np.asarray(Gk_crys, dtype=float)[np.asarray(g_mask) > 0.0]
			K_cart_this = (G_phys + np.asarray(kpoint, dtype=float)[None, :]) @ B
			K_med = float(np.median(np.linalg.norm(K_cart_this, axis=1))) if K_cart_this.size else 1.0
			h_base = max(float(args.vnl_h), float(args.vnl_h_rel) * max(K_med, 1.0))
			h1 = h_base
			h2 = 0.5 * h_base
			# ONE INTERNAL CONVENTION FOR BOTH MODES: ``vNL_cart`` means
			# ``+dV_NL/dK_cart``, which is what the analytic branch below
			# returns (``compute_vnl_velocity_cart``'s docstring, and
			# ``orbital_magnetization.py:601`` records it verified
			# off-diagonally at ratio 1.000).  These differences used to
			# carry a leading MINUS, which made ``--vnl-mode numeric``
			# the arithmetic negative of ``--vnl-mode analytic``: both
			# then passed through the same knob-controlled flip and the
			# same ``p_cart + vNL_cart``, so the two modes came out on
			# OPPOSITE arms of the very sign question this file's knob
			# parameterises.  Two implementations of one derivative
			# cannot both be right, and nothing in the tree compared
			# them, which is why it survived.  The finite difference is
			# the unambiguous one -- it is a literal numerical
			# derivative of ``compute_vnl_matrix_from_setup``, which
			# returns <m|V_NL(k)|n> with no sign convention of its own --
			# so the analytic branch is the definition both now share
			# and the numeric branch stops negating.
			for ic in range(3):
				# D1 at h1
				d1 = np.zeros((3,), dtype=float); d1[ic] = h1
				d1c = d1 @ Binv
				kp1 = np.asarray(kpoint, dtype=float) + d1c
				km1 = np.asarray(kpoint, dtype=float) - d1c
				Vp1 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, kp1, vnl_setup, g_mask=g_mask)
				Vm1 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, km1, vnl_setup, g_mask=g_mask)
				D1 = (Vp1 - Vm1) / (2.0 * h1)
				if args.vnl_num_scheme == "richardson":
					# D2 at h2
					d2 = np.zeros((3,), dtype=float); d2[ic] = h2
					d2c = d2 @ Binv
					kp2 = np.asarray(kpoint, dtype=float) + d2c
					km2 = np.asarray(kpoint, dtype=float) - d2c
					Vp2 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, kp2, vnl_setup, g_mask=g_mask)
					Vm2 = compute_vnl_matrix_from_setup(wfn_k, Gk_crys, km2, vnl_setup, g_mask=g_mask)
					D2 = (Vp2 - Vm2) / (2.0 * h2)
					vNL_cart[ic] = (4.0 * D2 - D1) / 3.0
				else:
					vNL_cart[ic] = D1
		else:
			vNL_cart = compute_vnl_velocity_cart(wfn_k, Gk_crys, kpoint, vnl_setup, g_mask=g_mask)

		# Sign convention note (Liu-2024 Eq. 17 / BGW k·p):
		# Our internal assembly returns v^NL = -(∂_q + ∂_{q'}) V_NL, while BGW’s
		# reported ⟨v⟩ uses the opposite sign convention. Flip here so users don’t
		# need to patch a source file when comparing to BGW outputs.
		#
		# THE SAME KNOB THE SWEEP PATH READS, and it has to be read here
		# too: this branch is what ``--vnl-mode numeric`` runs, and a flag
		# that is parsed, stamped and honoured on only one of two routes
		# is a knob that lies about half the runs it labels.  Written as a
		# branch and not as a multiply by ±1 so the shipped arm executes
		# the SAME negation it always did: a complex array times a real
		# ``-1.0`` goes through numpy's full complex product and turns a
		# ``+0.0`` imaginary part into ``-0.0``, which is numerically
		# nothing and is not bit-identity.
		if vnl_velocity_sign < 0.0:
			vNL_cart = -vNL_cart

		# Optional debug: print 4x6 x-direction blocks for selected k index
		if args.debug and int(i) == int(args.debug_kindex):
			_print_debug_blocks(i, p_cart, vNL_cart)

		return p_cart + vNL_cart

	# ⟨mk|v|nk⟩: ONE k-scan with THIS k's bands sharded over every process
	# (``common.mtxel_sweep``), replacing the k-partitioned
	# ``gather_k_blocks`` route.  That route took a whole k per rank, so
	# its wall was one full-band k however large P was and it could not
	# use more than ``nk`` ranks at all; the scan makes ``nk`` a trip
	# count and parallel efficiency ``nb_logical/nb_padded``.  Measured
	# on the sibling V_H sweep, b600-class at P=64, worst rank: 4.975 s /
	# 10.83 GiB before, 2.162 s / 8.21 GiB after (jobs 7888877, 7888907);
	# at P = nk it is ~1.45x slower, which is the documented crossover.
	#
	# The three Cartesian components ride ONE sweep, so the hoisted
	# m-side reshard is paid once rather than three times; only the
	# per-k reshard payload is 3x.
	pt_path = None
	write_pt_remainder = None
	if args.vnl_mode == "numeric":
		with timing.section("dipole_sweep"):
			dip_k_major = gather_k_blocks(nk, _dipole_block,
			                              item_shape=(3, nb, nb),
			                              label="dipole", owner_only=True)
	else:
		if args.debug and jax.process_index() == 0:
			_dipole_block(int(args.debug_kindex))     # the table, nothing else
		psi_G = wfn.load(bands=(0, nb), k="full_bz",
		                 sharding=band_sphere_spec(), bispinor=bispinor)
		geom = SweepGeometry(mesh=RUNTIME.mesh, fft_grid=meta.fft_grid,
		                     ngkmax=int(psi_G.shape[3]), nb=nb,
		                     ns=int(psi_G.shape[2]), nk=nk,
		                     cell_volume=float(wfn.cell_volume))
		op = dipole_operator(
			geom, bvec=wfn.bvec, blat=wfn.blat,
			vnl_setup=None if args.skip_vnl else vnl_setup,
			vnl_velocity_sign=vnl_velocity_sign)
		with timing.section("dipole_sweep"):
			H_v = sweep_matrix_elements(
				psi_G, operator=op, geom=geom,
				gvecs=gtab.gvecs, gmask=gtab.mask,
				box_index=wfn.box_index(k="full_bz"),
				kvecs=np.asarray(sym.unfolded_kpts))
			if args.parallel_transport_out is not None:
				# The feature is deliberately opt-in: the default dipole
				# artifact and its bitwise production path remain untouched.
				# Keep H_v sharded and direction-major it only inside the
				# SlabIO writer; no host gather or second velocity evaluation.
				from file_io.parallel_transport import (
					initialize_parallel_transport_artifact,
					validate_parallel_transport_artifact,
					write_parallel_transport_artifact)
				pt_path = Path(args.parallel_transport_out).resolve()
				with timing.section("parallel_transport_velocity"):
					initialize_parallel_transport_artifact(
						pt_path, wfn=wfn, sym=sym, mesh=RUNTIME.mesh,
						nbands=nb,
						effective_nspinor=int(meta.nspinor),
						bispinor=bispinor,
						velocity_dft_kmajor=H_v,
						wfn_path=str(wfn_path),
						wfn_fingerprint=wfn_fingerprint(wfn),
						rcond=float(args.parallel_transport_rcond))
				write_pt_remainder = write_parallel_transport_artifact
				validate_pt_artifact = validate_parallel_transport_artifact
			# THE BOUNDARY, named rather than implied: the only consumer
			# of the (nk, 3, nb, nb) table is the serial h5py write on
			# rank 0 below, which cannot take a sharded operand.
			# ``owner_only`` keeps it off the peers (BD.4) and the gather
			# runs in chunks so a peer's transient is one chunk.
			dip_k_major = blocks_to_host(H_v, nb=nb, owner_only=True)
		del H_v, psi_G
		if pt_path is not None:
			# The SlabIO velocity transaction above is closed and durable,
			# and the all-k psi/H_v device arrays are now dead.  The link
			# stream therefore holds only one central and one neighbour WFN
			# plus one distributed band matrix, never both preprocessing
			# representations at once.
			with timing.section("parallel_transport_links"):
				write_pt_remainder(
					pt_path, wfn=wfn, sym=sym, mesh=RUNTIME.mesh,
					nbands=nb, bispinor=bispinor,
					rcond=float(args.parallel_transport_rcond))
			with timing.section("parallel_transport_validation"):
				metrics = validate_pt_artifact(
					pt_path, mesh=RUNTIME.mesh, kgrid=wfn.kgrid,
					nbands=nb,
					bvec_cart=np.asarray(wfn.bvec) * float(wfn.blat),
					atol=float(args.parallel_transport_validation_atol),
					rtol=float(args.parallel_transport_validation_rtol))
			print(
				"  DFT covariant-velocity validation: PASS "
				f"max_abs={metrics['max_abs']:.6e}, "
				f"max_rel={metrics['max_rel']:.6e}")
			print(f"\nWrote parallel-transport data to {pt_path}")
	if dip_k_major is not None:
		dipole = np.ascontiguousarray(np.moveaxis(dip_k_major, 0, 1))
	else:
		dipole = None                        # non-root: never consumed
	del dip_k_major

	# Optional: finite-q matrix elements for the SOS chi head/wing/S/w pipeline.
	rho_cvkq = v_cvkq = kminq_idx_kq = None
	cv_meta = None
	if args.with_finite_q:
		print("\nComputing finite-q matrix elements (SOS pipeline)...")
		iq_list = args.iq_list if args.iq_list is not None else list(range(int(sym.nk_tot)))
		with timing.section("finite_q"):
			rho_cvkq, v_cvkq, kminq_idx_kq, n_occ_eff, v_lo, c_hi = compute_finite_q_mtxels(
			wfn, sym, meta, vnl_setup, gtab,
			nb=nb,
			bispinor=bispinor,
			iq_list=iq_list,
			nv_block=int(nval),
			nc_block=int(ncond),
			vnl_velocity_sign=vnl_velocity_sign,
			verbose=True,
		)
		cv_meta = {
			'iq_list': np.asarray(iq_list, dtype=np.int32),
			'n_occ': int(n_occ_eff),
			'v_lo': int(v_lo),
			'c_hi': int(c_hi),
		}

	# Save to dipole.h5 with deltaE
	out_path = Path(args.out).resolve()
	note = ('dipole_cart[3,x,y] = p_i (V_NL skipped, --skip-vnl); '
	        if args.skip_vnl
	        else f'dipole_cart[3,x,y] = {_arm} '
	             f'[vnl_velocity_sign = {vnl_velocity_sign:+.1f}]; ')
	note += 'deltaE[k,:,:] = E_b - E_b\''
	# Rank-0 writes.  Every rank holds the same gathered host arrays, so a
	# multi-process launch previously had all of them open the SAME path with
	# mode 'w' concurrently -- serial h5py has no cross-process locking, so
	# that is a genuine corruption hazard (it merely happened not to bite at
	# 4 ranks).  Barrier afterwards so no rank races ahead of the file
	# existing on disk.
	if jax.process_index() != 0:
		barrier("dipole_write")
		print(f"\nWrote dipole data to {out_path} (by rank 0)")
		return 0
	with timing.section("write_h5"), h5py.File(str(out_path), 'w') as h5:
		h5.create_dataset('dipole_cart', data=dipole)
		h5.create_dataset('deltaE', data=deltaE)
		h5.attrs['nbands'] = int(wfn.nbands)
		h5.attrs['nk'] = int(sym.nk_tot)
		h5.attrs['skip_vnl'] = bool(args.skip_vnl)
		h5.attrs['note'] = note
		stamp_dipole_provenance(
			h5, wfn=wfn, wfn_path=str(wfn_path), nval=nval, ncond=ncond,
			nband=nband, nb_written=nb, bispinor=bispinor,
			skip_vnl=bool(args.skip_vnl), vnl_mode=str(args.vnl_mode),
			vnl_velocity_sign=vnl_velocity_sign)
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
	barrier("dipole_write")
	print(f"\nWrote dipole data to {out_path}")
	timing.report(title="--- Timing (seconds) ---",
	              wall=time.perf_counter() - _t_main)


if __name__ == '__main__':
    raise SystemExit(main())
