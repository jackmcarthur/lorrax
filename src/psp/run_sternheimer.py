"""psp/run_sternheimer.py — Insulating Sternheimer driver for the (G=0) source
column  χ_{G'0}(q, ω=0).

Goal:  compute one column of the density-density response (the head and wing
column at G=0) using a Sternheimer linear-solve approach, which avoids
the slow conduction-band-count convergence of explicit sum-over-states
expressions.

Physics (see ``psp/sternheimer_guidelines.md`` and Cancès et al.
arXiv:2210.04512):

    For every (v, k) and a fixed reduced q, solve

        Q_{k-q} · (H_{k-q} − ε_{v,k}) · Q_{k-q} · |δu_{v,k}^q⟩
            = −Q_{k-q} · V_pert(r) · |u_{v,k}⟩

    with V_pert(r) = e^{-iq·r} for the density-response case, and then

        χ_{G'0}(q, 0) = (cell FT at G′ of  Σ_{v,k} u_{v,k}(r) · conj(δu_{v,k}^q(r)) ).

The cell-periodic "u" convention is used throughout (coefficients sit on
the k-sphere; real-space multiplications build up the response density on
the FFT box).  Convention ``p = k − q`` matches ``SymMaps.kq_map[ik_full,
iq_red]``, so ``Q_{k-q}`` is written ``Q_kminq`` everywhere.

The **perturbation** enters through a single real-space array
``V_pert_real(r)`` of shape ``(nx, ny, nz)`` — the source builder is
perturbation-agnostic so this driver can later be adapted to phonon
DFPT (``∂V_scf/∂R_α``), E-field perturbations (``-i r·ê``), etc.

Single-GPU, single-k-loop, single-q-loop.  Multi-GPU sharding will land
in a later commit; for now the entire wavefunction buffer lives on
device 0.

Driver style mirrors ``psp/run_nscf.py``: CrystalData-free (duck-typed
over WFNReader for all structural data, using full-BZ sums rather than
QE-save reads), section dividers for pipeline stages, and a ``main()``
that is either input-file- or clarg-driven.

Usage
-----
    lxrun python3 -u -m psp.run_sternheimer -i sternheimer.in
    lxrun python3 -u -m psp.run_sternheimer \\
        --wfn WFN.h5 --pseudo_dir . --n-cond-bands 20 --iq-list 0 1
"""
from __future__ import annotations

from runtime import set_default_env
set_default_env()  # BEFORE `import jax`

import argparse
import os
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from runtime import init_jax_distributed
init_jax_distributed()

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.dft_operators import setup_H_k_from_kvec
from psp.h_dft import make_apply_H
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.vnl_ops import _build_vnl_kdata_core
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import (
    compute_per_band_kinetic,
    tpa_preconditioner_diag,
)
from solvers.sternheimer_solve import SternheimerOp, sternheimer_solve


# ═══════════════════════════════════════════════════════════════════════
#  Jitted real-space kernels (single-GPU; no sharding)
# ═══════════════════════════════════════════════════════════════════════

def _batched_real_norm_host(a: jax.Array) -> jax.Array:
    """``(batch,)`` real L2 norm over the trailing two axes — for the b≈0 fast path."""
    return jnp.sqrt(jnp.real(
        jnp.einsum('vsG,vsG->v', jnp.conj(a), a, optimize=True)))


def _gather_box_at_G(box: jax.Array, G_int: jax.Array) -> jax.Array:
    """Gather values at the signed integer G-indices (wrapped mod FFT-grid).

    Works for any leading batch dims; the FFT-box axes are the last three.
    """
    ix = jnp.mod(G_int[:, 0], box.shape[-3])
    iy = jnp.mod(G_int[:, 1], box.shape[-2])
    iz = jnp.mod(G_int[:, 2], box.shape[-1])
    return box[..., ix, iy, iz]


def build_sternheimer_source_preQ(
    U_box_k: jax.Array,
    Gkminq_int: jax.Array,
    V_pert_real: jax.Array,
) -> jax.Array:
    """V_pert(r)·u_{v,k_s}(r) gathered on the (k-q) G-sphere — BEFORE Q-projection.

    Split out of ``build_sternheimer_source`` so the caller can apply
    Q_{k-q} with a Q that varies with q (needed when unfreezing the
    projector for ∂χ/∂q — see ``chi_col_contrib_at_kvec_traced``).
    """
    u_r = jnp.fft.ifftn(U_box_k, axes=(-3, -2, -1), norm='ortho')
    Vu_r = V_pert_real[None, None, :, :, :] * u_r
    Vu_box = jnp.fft.fftn(Vu_r, axes=(-3, -2, -1), norm='ortho')
    return _gather_box_at_G(Vu_box, Gkminq_int)


def build_sternheimer_source(
    U_box_k: jax.Array,            # (nv, nspinor, nx, ny, nz)  G-space scatter
    Gkminq_int: jax.Array,         # (ngk_p, 3)
    V_pert_real: jax.Array,        # (nx, ny, nz) complex
    Q_kminq,
) -> jax.Array:
    """Build  b_v = Q_{k-q} · V_pert(r) · u_{v,k}  on the (k-q) G-sphere.

    Perturbation-agnostic: ``V_pert_real`` is any cell-sized real-space
    array.  For the density-response head/wing column use
    ``V_pert_real = exp(-i q · r)`` (see :func:`make_density_perturbation`).
    Phonon / E-field perturbations can plug in here without touching this
    pipeline.

    Parameters
    ----------
    U_box_k : (nv, nspinor, nx, ny, nz) complex
        Occupied orbitals at *source* k in **G-space FFT-box layout**: the
        direct output of ``common.load_wfns.load_kpoint_fftbox`` — coefficients
        scattered into a zero-padded box, NOT the real-space wavefunction.
    Gkminq_int : (ngk_p, 3) int32
        Integer G-vectors of the (k-q) G-sphere (from
        ``sym.get_gvecs_kfull(wfn, ik_kminq)``).
    V_pert_real : (nx, ny, nz) complex — e^{-iq·r} for density response.
    Q_kminq : callable, projector onto conduction subspace at k-q.

    Returns
    -------
    b : (nv, nspinor, ngk_p) complex — in range(Q_{k-q}).
    """
    # Ortho IFFT maps the G-box to the cell-periodic part u_{v,k}(r).
    u_r = jnp.fft.ifftn(U_box_k, axes=(-3, -2, -1), norm='ortho')
    # Multiply by V_pert(r) elementwise.
    Vu_r = V_pert_real[None, None, :, :, :] * u_r
    # Back to G-box, then gather on the (k-q) sphere.
    Vu_box = jnp.fft.fftn(Vu_r, axes=(-3, -2, -1), norm='ortho')
    Vu_G = _gather_box_at_G(Vu_box, Gkminq_int)
    return Q_kminq(Vu_G)


def accumulate_chi_density(
    U_box_k: jax.Array,            # (nv, nspinor, nx, ny, nz)   G-space scatter at k
    delta_u_G: jax.Array,          # (nv, nspinor, ngk_p)        G-sphere coeffs at k-q (wrapped gauge)
    Gkminq_int: jax.Array,         # (ngk_p, 3)
    fft_grid,
    phase_unwrap: jax.Array | None = None,  # (nx, ny, nz) e^{-i G_wrap·r}; default 1
) -> jax.Array:
    """Return the (nx,ny,nz) real-space contribution  Σ_v u_{v,k}(r) ·
    conj(δu^{q,\\mathrm{naive}}_{v,k}(r))  for the χ_{G'0}(q,0) induced density.

    The CG solves in the wrapped-(k-q) gauge where δu_wrap(r) = e^{+iG_wrap·r}·
    δu_naive(r).  Multiplying δu_wrap by phase_unwrap = e^{-iG_wrap·r} in real
    space recovers δu_naive, which is what enters the physical density at the
    naive Bloch momentum — putting its Fourier content into the correct G-slot
    of the chi column.
    """
    u_r = jnp.fft.ifftn(U_box_k, axes=(-3, -2, -1), norm='ortho')
    nv, nspinor, _ = delta_u_G.shape
    nx, ny, nz = fft_grid
    ix = jnp.mod(Gkminq_int[:, 0], nx)
    iy = jnp.mod(Gkminq_int[:, 1], ny)
    iz = jnp.mod(Gkminq_int[:, 2], nz)
    du_box = jnp.zeros((nv, nspinor, nx, ny, nz), dtype=delta_u_G.dtype)
    du_box = du_box.at[:, :, ix, iy, iz].set(delta_u_G)
    du_wrap_r = jnp.fft.ifftn(du_box, axes=(-3, -2, -1), norm='ortho')
    # Gauge conversion wrapped → naive.
    if phase_unwrap is not None:
        du_naive_r = phase_unwrap[None, None, :, :, :] * du_wrap_r
    else:
        du_naive_r = du_wrap_r
    return jnp.sum(u_r * jnp.conj(du_naive_r), axis=(0, 1))


def project_density_to_Gsphere(
    delta_n_r: jax.Array,          # (nx, ny, nz) complex
    G_out_int: jax.Array,          # (ng_out, 3) int32 — target G' list
) -> jax.Array:
    """FFT δn(r) and gather at target G' indices.  Ortho FFT."""
    box = jnp.fft.fftn(delta_n_r, axes=(-3, -2, -1), norm='ortho')
    return _gather_box_at_G(box[None], G_out_int)[0]


# ═══════════════════════════════════════════════════════════════════════
#  Perturbation factories
# ═══════════════════════════════════════════════════════════════════════

def make_density_perturbation(fft_grid) -> jax.Array:
    """Cell-periodic part of the density-response perturbation — identically 1.

    The q-dependence of the density-response source is carried by the change
    of Bloch momentum (k → k-q) under V_ext(r) = e^{iq·r}; no explicit
    real-space phase is needed when the (k-q)-sector does not wrap.  When the
    wrap does occur, an additional ``phase_wrap`` = e^{+iG_wrap·r} is
    multiplied into V_pert_cell at driver-level to convert the source from
    the naive to the wrapped cell-periodic gauge (see driver q-loop).

    For phonon DFPT the caller would pass V_pert_cell(r) = ∂V_scf/∂R_α
    (cell-periodic by construction); the e^{iqr} plane-wave factor still
    comes in for free via the k → k±q G-sphere shift.
    """
    nx, ny, nz = fft_grid
    return jnp.ones((nx, ny, nz), dtype=jnp.complex128)


def make_umklapp_phase(G_wrap: np.ndarray, fft_grid, sign: int = +1) -> jax.Array:
    """Return e^{i·sign·G_wrap·r} on the FFT grid, with r in crystal coords.

    Build exp(sign · 2π i · G_wrap · r_frac) at FFT grid points r_frac =
    (j_x/nx, j_y/ny, j_z/nz).  This is cell-periodic when G_wrap is an
    integer reciprocal vector; it converts a cell-periodic function from the
    naive-(k-q) gauge to the wrapped-(k-q) gauge (sign=+1) or back (sign=-1):

        u_{wrap}(r) = e^{-i G_wrap · r} · u_{naive}(r)      [user derivation §1]
        δu_{naive}(r) = e^{-i G_wrap · r} · δu_{wrap}(r)   [user derivation §2]

    So the source builder multiplies by phase_wrap  = exp(+2πi G_wrap · r_frac)
    to push u_{v,k_s}(r) into the wrapped gauge before projecting on Q_{k-q}_wrap,
    and the accumulator multiplies δu_wrap by phase_unwrap = conj(phase_wrap)
    to recover δu_naive before contracting with u_{v,k_s}(r).

    If ``G_wrap == 0``, returns constant 1 (no-op).
    """
    nx, ny, nz = fft_grid
    if int(G_wrap[0]) == 0 and int(G_wrap[1]) == 0 and int(G_wrap[2]) == 0:
        return jnp.ones((nx, ny, nz), dtype=jnp.complex128)
    fx = jnp.arange(nx, dtype=jnp.float64) / nx
    fy = jnp.arange(ny, dtype=jnp.float64) / ny
    fz = jnp.arange(nz, dtype=jnp.float64) / nz
    phase_arg = (2.0 * jnp.pi * sign) * (
        float(G_wrap[0]) * fx[:, None, None]
        + float(G_wrap[1]) * fy[None, :, None]
        + float(G_wrap[2]) * fz[None, None, :]
    )
    return jnp.exp(1j * phase_arg).astype(jnp.complex128)


# ═══════════════════════════════════════════════════════════════════════
#  k·p response  ∂u_{v, kvec_p}/∂kvec_p  — "unfreeze the projector"
# ═══════════════════════════════════════════════════════════════════════

def compute_kp_tangent_at_kvec(
    kvec_p_base_np: np.ndarray,        # (3,) — where to evaluate ∂u/∂kvec_p
    Gkminq_int_np: np.ndarray,         # (nG_p, 3) — G-sphere at kvec_p_base
    vnl_setup,
    V_scf: jax.Array, mask: jax.Array,
    Gx: jax.Array, Gy: jax.Array, Gz: jax.Array,
    fft_grid,
    bdot: jax.Array,
    vnl_E_super: jax.Array,
    U_val_G: jax.Array,                 # (nv, ns, nG_p) — occupied at kvec_p_base
    eps_v_at_kvec_p: jax.Array,         # (nv,) — eigenvalues AT kvec_p (not source k!)
    alpha_pv_sc: jax.Array,
    precond_diag: jax.Array,            # (nv, 1, nG_p) — TPA at kvec_p_base
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    use_schur: bool = False,
    U_extra_G: jax.Array | None = None,
    eps_extra_v_at_kvec_p: jax.Array | None = None,
) -> jax.Array:
    """Return  ∂u_{v, kvec_p}/∂kvec_p  as a (3, nv, nspinor, nG) array.

    k·p response at a specific kvec_p, obtained by solving the Sternheimer
    equation with source  b_i = Q_{kvec_p} · (∂H/∂kvec_p_i) · u_{v, kvec_p} ,
    for each cartesian crystal-coordinate direction ``i ∈ {0, 1, 2}``.

    The chain-rule link to q-derivatives:  with kvec_p(q) = kvec_p_base −
    (q − q_base) (the convention used in
    :func:`chi_col_contrib_at_kvec_traced`), we have

        ∂U_val(q) / ∂q_i   =  −  ∂U_val / ∂kvec_p_i    at q = q_base.

    So the driver passes this ``U_val_grad`` (returned here) into its
    end-to-end pipeline as a linearisation of ``U_val(kvec_p)``, and
    ``jax.jvp`` composes it automatically into the ∂χ/∂q tangent.  Result:
    the first derivative of χ is no longer the "frozen-projector"
    approximation (guide §5 "simplest first impl"); it includes the
    full ∂P_occ/∂q contribution.

    Cost: 3 batched CG solves per call (one per cartesian direction,
    batched over all ``nv`` valence bands).  Each solve reuses the same
    jitted ``sternheimer_solve`` kernel as the main χ computation.
    Cache the result per unique k_wrap if you plan multiple q-derivative
    evaluations — the k·p response is kvec_p-local and doesn't depend on q.

    Parameters
    ----------
    eps_v_at_kvec_p : (nv,)
        **Eigenvalues at kvec_p**, not at the source k.  ``U_val_G`` is an
        eigenstate of H_{kvec_p}; the k·p Sternheimer uses those
        eigenvalues for its shifts.  This is different from the main
        χ-column's ``eps_v`` (which comes from the source k).
    """
    from solvers.sternheimer_solve import (
        SternheimerOp, sternheimer_solve, _apply_A_inline)

    def _op_at(kvec_p_t):
        """Build the level-shifted Sternheimer op at kvec_p_t (traced)."""
        return build_sternheimer_op_at_kvec_traced(
            kvec_p_t, Gkminq_int_np, vnl_setup,
            V_scf=V_scf, mask=mask, Gx=Gx, Gy=Gy, Gz=Gz, fft_grid=fft_grid,
            bdot=bdot, vnl_E_super=vnl_E_super,
            U_val_kminq_G=U_val_G, eps_v=eps_v_at_kvec_p,
            alpha_pv_sc=alpha_pv_sc, precond_diag=precond_diag,
            U_extra_G=U_extra_G, eps_extra=eps_extra_v_at_kvec_p,
        )

    kvec_p_base = jnp.asarray(kvec_p_base_np, dtype=jnp.float64)
    op_base = _op_at(kvec_p_base)

    # Q_{kvec_p} projector for the source.
    def Q_of(x):
        coefs = jnp.einsum('msG,vsG->vm', jnp.conj(U_val_G), x, optimize=True)
        return x - jnp.einsum('vm,msG->vsG', coefs, U_val_G, optimize=True)

    # 3 cartesian directions stacked as a leading batch dim then solved as
    # one batched CG: build the 3 RHS vectors in parallel via vmap, then
    # collapse (3, nv) → (3·nv) for a single sternheimer_solve call.  Faster
    # than the prior 3-iteration Python loop because the inner CG kernel
    # is reused once (one compile, one launch sequence).
    def _rhs_one_dir(dk):
        _, A_dot_u = jax.jvp(
            lambda k: _apply_A_inline(_op_at(k), U_val_G),
            (kvec_p_base,), (dk,))
        return Q_of(A_dot_u)
    eye3 = jnp.eye(3, dtype=jnp.float64)
    rhs_3 = jax.vmap(_rhs_one_dir)(eye3)               # (3, nv, ns, nG)
    nv = U_val_G.shape[0]
    rhs_flat = rhs_3.reshape(-1, *rhs_3.shape[2:])     # (3·nv, ns, nG)
    # Replicate eps_v 3× so the batched solver's level-shift matches each
    # direction-band pair.  This requires reshaping op_base.eps_v.
    op_b3 = SternheimerOp(
        T_diag=op_base.T_diag, V_scf=op_base.V_scf,
        Gx=op_base.Gx, Gy=op_base.Gy, Gz=op_base.Gz,
        vnl_Z=op_base.vnl_Z, vnl_E=op_base.vnl_E, mask=op_base.mask,
        U_val=op_base.U_val,
        eps_v=jnp.tile(op_base.eps_v, 3),
        alpha_pv=op_base.alpha_pv,
        precond_diag=jnp.tile(op_base.precond_diag, (3, 1, 1)),
        fft_grid=op_base.fft_grid,
        U_extra=op_base.U_extra, eps_extra=op_base.eps_extra,
    )
    du_flat = sternheimer_solve(op_b3, rhs_flat,
                                  tol=tol, max_iter=max_iter, use_schur=use_schur)
    return du_flat.reshape(3, nv, *du_flat.shape[1:])  # (3, nv, ns, nG)


def compute_kp2_tangent_at_kvec(
    kvec_p_base_np: np.ndarray,        # (3,)
    Gkminq_int_np: np.ndarray,         # (nG_p, 3)
    vnl_setup,
    V_scf: jax.Array, mask: jax.Array,
    Gx: jax.Array, Gy: jax.Array, Gz: jax.Array,
    fft_grid,
    bdot: jax.Array,
    vnl_E_super: jax.Array,
    U_val_G: jax.Array,                 # (nv, ns, nG_p)
    U_val_grad_kp: jax.Array,           # (3, nv, ns, nG_p) from compute_kp_tangent_at_kvec
    eps_v_at_kvec_p: jax.Array,
    alpha_pv_sc: jax.Array,
    precond_diag: jax.Array,
    *,
    tol: float = 1e-10,
    max_iter: int = 300,
    use_schur: bool = False,
    U_extra_G: jax.Array | None = None,
    eps_extra_v_at_kvec_p: jax.Array | None = None,
) -> jax.Array:
    """Return ∂²u_v/∂kvec_p_i ∂kvec_p_j as a (3, 3, nv, nspinor, nG) array,
    symmetric over (i, j).

    Second-order k·p response.  The standard insulator derivation gives

        Q (H − ε) Q · ∂²u/∂k_i∂k_j  =  − Q · [ (∂²H/∂k_i∂k_j) u
                                              + (∂H/∂k_i) ∂u/∂k_j
                                              + (∂H/∂k_j) ∂u/∂k_i ]

    (the ε-derivatives don't survive the Q-projection since ∂u/∂k are already
    in range(Q)).  Implementation uses nested ``jax.jvp`` on ``A(k) · x``
    with ``x`` in turn equal to ``U_val_G`` or ``U_val_grad_kp[i]``; the
    operator tangent ``∂A/∂k_i`` reduces to ``∂H/∂k_i`` because the α·P_val
    and −ε_v terms are kvec_p-independent.

    This is the missing ingredient for the exact S-tensor at q=0: the
    linear U_val_eff ansatz in ``chi_col_contrib_at_kvec_traced`` captures
    only first-order projector drift; picking up the second-order projector
    correction requires this Hess of U_val to appear in the quadratic
    U_val_eff term.

    Cost: 6 batched CG solves per call (symmetric 3×3 matrix; only i≤j pairs
    are solved; diagonal entries reuse the single cross term).  Each solve
    is the same jitted ``sternheimer_solve`` kernel as the main χ column.
    """
    from solvers.sternheimer_solve import (
        SternheimerOp, sternheimer_solve, _apply_A_inline)

    def _op_at(kvec_p_t):
        return build_sternheimer_op_at_kvec_traced(
            kvec_p_t, Gkminq_int_np, vnl_setup,
            V_scf=V_scf, mask=mask, Gx=Gx, Gy=Gy, Gz=Gz, fft_grid=fft_grid,
            bdot=bdot, vnl_E_super=vnl_E_super,
            U_val_kminq_G=U_val_G, eps_v=eps_v_at_kvec_p,
            alpha_pv_sc=alpha_pv_sc, precond_diag=precond_diag,
            U_extra_G=U_extra_G, eps_extra=eps_extra_v_at_kvec_p,
        )

    kvec_p_base = jnp.asarray(kvec_p_base_np, dtype=jnp.float64)
    op_base = _op_at(kvec_p_base)

    def Q_of(x):
        coefs = jnp.einsum('msG,vsG->vm', jnp.conj(U_val_G), x, optimize=True)
        return x - jnp.einsum('vm,msG->vsG', coefs, U_val_G, optimize=True)

    def e_vec(k):
        return jnp.zeros(3, dtype=jnp.float64).at[k].set(1.0)

    hess = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            ei = e_vec(i)
            ej = e_vec(j)

            # (∂²A/∂k_i∂k_j) · U_val  via nested jax.jvp on k ↦ A(k)·U_val.
            def _A_u(k):
                return _apply_A_inline(_op_at(k), U_val_G)
            def _grad_j_A_u(k):
                _, out = jax.jvp(_A_u, (k,), (ej,))
                return out
            _, A_ij_u = jax.jvp(_grad_j_A_u, (kvec_p_base,), (ei,))

            # (∂A/∂k_i) · (∂u/∂k_j)
            _, A_i_duj = jax.jvp(
                lambda k: _apply_A_inline(_op_at(k), U_val_grad_kp[j]),
                (kvec_p_base,), (ei,))
            # (∂A/∂k_j) · (∂u/∂k_i)  — equals A_i_duj when i == j
            if i == j:
                A_j_dui = A_i_duj
            else:
                _, A_j_dui = jax.jvp(
                    lambda k: _apply_A_inline(_op_at(k), U_val_grad_kp[i]),
                    (kvec_p_base,), (ej,))

            rhs_ij = A_ij_u + A_i_duj + A_j_dui
            b_ij = Q_of(rhs_ij)
            d2u_ij = sternheimer_solve(op_base, b_ij,
                                        tol=tol, max_iter=max_iter,
                                        use_schur=use_schur)
            # Diagnostic: residual ‖A·d2u + b‖  (solver solves A·d2u = -b).
            res_ij = _apply_A_inline(op_base, d2u_ij) + b_ij
            res_norm_ij = jnp.max(jnp.sqrt(jnp.real(
                jnp.einsum('vsG,vsG->v', jnp.conj(res_ij), res_ij, optimize=True))))
            b_norm_ij = jnp.max(jnp.sqrt(jnp.real(
                jnp.einsum('vsG,vsG->v', jnp.conj(b_ij), b_ij, optimize=True))))
            import os as _os
            if _os.environ.get('KP2_DEBUG'):
                print(f"    kp2 (i={i}, j={j}) at kvec_p_base={kvec_p_base_np}: "
                      f"‖b_ij‖_max = {float(b_norm_ij):.3e}, "
                      f"‖residual‖_max = {float(res_norm_ij):.3e}, "
                      f"‖d2u_ij‖ = {float(jnp.sqrt(jnp.sum(jnp.abs(d2u_ij)**2))):.3e}")
            hess[i][j] = d2u_ij
            hess[j][i] = d2u_ij

    return jnp.stack([jnp.stack(row, axis=0) for row in hess], axis=0)


# ═══════════════════════════════════════════════════════════════════════
#  Explicit S-tensor at q = 0 — bypasses nested jvp entirely
# ═══════════════════════════════════════════════════════════════════════

def _q_project(U_val_G: jax.Array, x: jax.Array) -> jax.Array:
    """Q_{k-q}(x) = x - Σ_m U_val[m]·<U_val[m]|x>."""
    coefs = jnp.einsum('msG,vsG->vm', jnp.conj(U_val_G), x, optimize=True)
    return x - jnp.einsum('vm,msG->vsG', coefs, U_val_G, optimize=True)


def compute_s_tensor_contrib_at_q0(
    kvec_p_base_np: np.ndarray,        # (3,) = kvec_k at q=0 (non-wrapping k's)
    Gkminq_int_np: np.ndarray,         # (nG_p, 3)
    vnl_setup,
    V_scf: jax.Array,
    mask: jax.Array,
    Gx: jax.Array, Gy: jax.Array, Gz: jax.Array,
    fft_grid,
    bdot: jax.Array,
    vnl_E_super: jax.Array,
    U_val_G: jax.Array,                 # (nv, ns, nG_p) — U_val at kvec_p_base
    U_val_grad_kp: jax.Array,           # (3, nv, ns, nG_p) ∂U/∂kvec_p
    eps_v: jax.Array,                   # (nv,)
    alpha_pv_sc: jax.Array,
    precond_diag: jax.Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
    use_schur: bool = False,
    U_extra_G: jax.Array | None = None,
    eps_extra: jax.Array | None = None,
) -> jax.Array:
    """Return the per-k (3, 3) complex contribution to the density-response
    S-tensor  S_{ij} = ∂²χ_{00}/∂q_i∂q_j|_0  (in units of
    ``Σ_v <grad_v_i | δu̇_v_j>``, pre-``prefactor``).

    Physics — why we don't need Hess.  At q = 0 the primal Sternheimer solve
    has b = Q_k · u_{v,k} = 0 and δu(0) = 0.  The naive quadratic expansion
    δu(q) = q_i · δu̇_i + (1/2) q_i q_j · δü_ij   in a *fixed* k-basis
    must respect the constraint Q(q) · δu(q) = δu(q) where Q(q) = Q_{k-q} is
    q-dependent through the occupied subspace.  Expanding:

        P_val(0) · δü_ij  =  2 ·( Q̇_i · δu̇_j  +  Q̇_j · δu̇_i )

    The **Q-part** of δü_ij (containing the k·p² Hess) is orthogonal to
    u_v,k and therefore does NOT contribute to the density overlap
    <u_v | δü>_G .  Only the **P_val-part** contributes, and it's determined
    entirely by the first tangent δu̇_i:

        <u_v | δü_ij>  =  2 ·(<grad_v_i | δu̇_v_j> + <grad_v_j | δu̇_v_i>)

    So the S-tensor at q = 0 reduces to a **cross-correlation of k·p
    gradients through A⁻¹**, which is exactly the sum-over-states structure
    of the Adler–Wiser dipole formula (``common.chi_from_dipole``).

    Cost: 3 Sternheimer solves per k (for δu̇_i = −A⁻¹·grad_i).  The
    k·p² Hess machinery (6 more solves per k) is **not used** here — it was
    a dead-end for S-tensor at q = 0 though it may still be useful for
    ∂²χ/∂q² evaluated at q ≠ 0.

    Returns
    -------
    (3, 3) complex array of per-k G-sphere overlaps
    ``Σ_v [<grad_v_i | δu̇_v_j> + <grad_v_j | δu̇_v_i>]``.
    The caller multiplies by ``(2 · spin_factor / (V_cell · N_k))`` to
    match the standard Adler–Wiser normalisation — see main run_sternheimer.
    """
    from solvers.sternheimer_solve import sternheimer_solve

    # ── First tangents  δu̇_i  (3 solves) ─────────────────────────────────
    # solve(op, b) satisfies A·x = −b, so passing b = grad_i[v] yields
    # x = −A⁻¹·grad_i[v] = δu̇_i  (matches physical ẋ_i = −A⁻¹·∂b/∂q_i|_0
    # with ∂b/∂q_i|_0 = +grad_i[v]).
    op = build_sternheimer_op_at_kvec_traced(
        jnp.asarray(kvec_p_base_np, dtype=jnp.float64),
        Gkminq_int_np, vnl_setup,
        V_scf=V_scf, mask=mask, Gx=Gx, Gy=Gy, Gz=Gz, fft_grid=fft_grid,
        bdot=bdot, vnl_E_super=vnl_E_super,
        U_val_kminq_G=U_val_G, eps_v=eps_v,
        alpha_pv_sc=alpha_pv_sc, precond_diag=precond_diag,
        U_extra_G=U_extra_G, eps_extra=eps_extra,
    )
    # 3 first-tangent solves collapsed into one batched CG over (3·nv).
    # Match eps_v / precond_diag by tiling 3×.
    nv = U_val_G.shape[0]
    op_3 = SternheimerOp(
        T_diag=op.T_diag, V_scf=op.V_scf,
        Gx=op.Gx, Gy=op.Gy, Gz=op.Gz,
        vnl_Z=op.vnl_Z, vnl_E=op.vnl_E, mask=op.mask,
        U_val=op.U_val,
        eps_v=jnp.tile(op.eps_v, 3),
        alpha_pv=op.alpha_pv,
        precond_diag=jnp.tile(op.precond_diag, (3, 1, 1)),
        fft_grid=op.fft_grid,
        U_extra=op.U_extra, eps_extra=op.eps_extra,
    )
    rhs_flat = U_val_grad_kp.reshape(-1, *U_val_grad_kp.shape[2:])  # (3·nv, ns, nG)
    du_flat = sternheimer_solve(op_3, rhs_flat,
                                  tol=tol, max_iter=max_iter, use_schur=use_schur)
    delta_u_dot = du_flat.reshape(3, nv, *du_flat.shape[1:])

    # ── <grad_v_i | δu̇_v_j>  for all (i, j) pairs in one fused einsum ──
    # From the Q(q)-constraint Taylor expansion at q = 0 (symmetric δü):
    #     P_val(0)·δü_ab  =  Q̇_a·δu̇_b  +  Q̇_b·δu̇_a
    # so   <u_v | δü_ab>_P  =  <grad_a | δu̇_b>  +  <grad_b | δu̇_a>.
    ovlp = jnp.einsum('ivsG,jvsG->ij',
                      jnp.conj(U_val_grad_kp), delta_u_dot,
                      optimize=True)            # (3, 3)
    return ovlp + ovlp.T                         # symmetrize

def build_sternheimer_op_at_kvec_traced(
    kvec_p_traced: jax.Array,                 # (3,) — JAX tracer (the k-q at which to evaluate H)
    Gkminq_int_np: np.ndarray,                # (nG_p, 3) — INTEGER G-list, constant across jvp
    vnl_setup,                                # from psp.vnl_ops.build_vnl_setup
    V_scf: jax.Array,                         # (nx, ny, nz) — V_scf is k-independent
    mask: jax.Array,                          # (nG_p,) — sphere mask, constant
    Gx, Gy, Gz: jax.Array,                    # (nG_p,) int32, each — constant across jvp
    fft_grid,                                 # tuple, static
    bdot: jax.Array,                          # (3, 3)
    vnl_E_super: jax.Array,                   # (nspinor, nspinor, R, R) — k-indep KB energies
    U_val_kminq_G: jax.Array,                 # (nv, nspinor, nG_p) — frozen-projector approx
    eps_v: jax.Array,                         # (nv,)
    alpha_pv_sc: jax.Array,                   # () scalar
    precond_diag: jax.Array,                  # (nv, 1, nG_p)
    U_extra_G: jax.Array | None = None,       # (M, nspinor, nG_p) or None
    eps_extra: jax.Array | None = None,       # (M,) or None
):
    """Rebuild a ``SternheimerOp`` from a traced ``kvec_p`` — JAX-jvp-friendly.

    Use this when you want to take derivatives of a Sternheimer solve with
    respect to ``q`` (or any parameter that shifts ``kvec_p = k − q``):

        solve_of_q = lambda q: sternheimer_solve(
            build_sternheimer_op_at_kvec_traced(kvec_p_base - (q - q_base), ...),
            b, tol, max_iter)
        dx = jax.jvp(solve_of_q, (q,), (dq,))[1]

    The ∂/∂q tangent then propagates through:

      * T_diag = |k+G|² (via bdot metric)  — autodiff through ``jnp.einsum``.
      * vnl_Z via ``_build_vnl_kdata_core`` — which has its own internal custom
        JVPs for the solid-harmonic / radial-table kernels (see
        ``psp/vnl_ops.py``).  Validated against FD to 3e-10 rel err.

    The **simplest-first approximation** per the guide is frozen-Q_{k-q}
    (``∂U_val/∂q ≈ 0``); ``U_val_kminq_G`` is passed in as a static array.
    For full consistency one would also update U_val via a Davidson sub-solve
    at the perturbed kvec, but that's a much bigger integration.

    Identity used internally (per the user-provided physics note):

        ∂V_NL / ∂q  =  − ∂V_NL / ∂k     (k-q is the Bloch momentum fed to H)

    handled automatically because ``_build_vnl_kdata_core`` differentiates
    wrt its ``kvec`` argument (= ``kvec_p``), and the caller composes that
    with ``kvec_p_traced = kvec_p_base - (q - q_base)`` (sign from the
    chain rule flips ∂/∂k → −∂/∂q).
    """
    # T_diag:  |k+G|² on the bdot metric.
    Gk_float = jnp.asarray(Gkminq_int_np, dtype=jnp.float64)
    kG = Gk_float + kvec_p_traced[None, :]
    T_diag = jnp.einsum('gi,ij,gj->g', kG, bdot, kG)

    # Exact VNL-Z via the traceable kernel.
    kdata = _build_vnl_kdata_core(kvec_p_traced, Gkminq_int_np, vnl_setup,
                                   compute_dZ=False)

    from solvers.sternheimer_solve import SternheimerOp
    return SternheimerOp(
        T_diag=T_diag, V_scf=V_scf,
        Gx=Gx, Gy=Gy, Gz=Gz,
        vnl_Z=kdata.Z, vnl_E=vnl_E_super, mask=mask,
        U_val=U_val_kminq_G, eps_v=eps_v,
        alpha_pv=alpha_pv_sc,
        precond_diag=precond_diag,
        fft_grid=fft_grid,
        U_extra=U_extra_G, eps_extra=eps_extra,
    )


# ═══════════════════════════════════════════════════════════════════════
#  End-to-end χ_{G'0}(q, 0) as a pure JAX function  (qvec → chi_col)
# ═══════════════════════════════════════════════════════════════════════

def chi_col_contrib_at_kvec_traced(
    kvec_p_traced: jax.Array,         # traced — carries the q-tangent
    Gkminq_int_np: np.ndarray,
    vnl_setup,
    V_scf: jax.Array,
    mask: jax.Array,
    Gx, Gy, Gz: jax.Array,
    fft_grid,
    bdot: jax.Array,
    vnl_E_super: jax.Array,
    U_val_kminq_G: jax.Array,
    eps_v: jax.Array,
    alpha_pv_sc: jax.Array,
    precond_diag: jax.Array,
    U_val_k_box: jax.Array,           # (nv, nspinor, nx, ny, nz) G-space scatter at source k
    b: jax.Array,                     # (nv, nspinor, nG_p) — Q-projected source for the frozen-
                                      #                        projector path; ignored when Vu_G_preQ
                                      #                        is supplied (unfrozen path)
    Gprime_int: jax.Array,            # (ng_out, 3) int32 — output G'-list (constant)
    phase_unwrap: jax.Array,          # (nx, ny, nz) e^{-iG_wrap·r} (unity if no wrap)
    prefactor: jax.Array,             # () scalar χ-normalisation (spin_factor·2·√N/(V·N_k))
    tol: float = 1e-6,
    max_iter: int = 100,
    use_schur: bool = False,
    U_extra_G: jax.Array | None = None,
    eps_extra: jax.Array | None = None,
    U_val_grad_kp: jax.Array | None = None,   # (3, nv, ns, nG_p) ∂U_val/∂kvec_p  — unfreezes projector
    kvec_p_base_for_grad: jax.Array | None = None,
    Vu_G_preQ: jax.Array | None = None,   # (nv, nspinor, nG_p) — V_pert(r)·u_{v,k_s} gathered
                                           # on the p-sphere, pre-Q-projection.  Pass this
                                           # (together with U_val_grad_kp) to get the correct
                                           # q-dependence of the source via Q_{k-q}(q).
    U_val_hess_kp: jax.Array | None = None,   # (3, 3, nv, ns, nG_p) ∂²U_val/∂kvec_p_i∂kvec_p_j
                                               # — upgrades U_val_eff to quadratic-in-q ansatz;
                                               # required for the exact S-tensor (∂²χ/∂q²) at q=0.
) -> jax.Array:
    """Return the (ng_out,) χ_{G'0}(q, 0) contribution from a single source-k pair.

    Single JAX-jittable function:  qvec (via kvec_p_traced) → χ_col[G'].
    Wrapping ``jax.jvp`` / ``jax.jacfwd`` around this gives ∂χ/∂q for this
    (k_source, k-q) pair; summing over source k's gives the total column's
    q-derivative.  All non-q-dependent data is passed through unchanged.

    Why this helper exists:  the driver loops over source k's in Python and
    accumulates the density in real space before a single final FFT.  For
    jvp purposes we exploit linearity:

        χ_col(q) = Σ_k (chi_col_contrib_at_kvec_traced at k's kvec_p(q))
        ∂χ_col/∂q = Σ_k  jax.jvp(contrib_at_k, ...)

    Each contribution is a clean JAX function so the custom-jvp machinery
    for the inner Sternheimer solve composes through the density-project
    step.
    """
    from solvers.sternheimer_solve import sternheimer_solve

    # Projector linearisation (see compute_kp_tangent_at_kvec).  If the caller
    # pre-computed the k·p gradient, use the linear model
    #     U_val(kvec_p) = U_val_base + (kvec_p - kvec_p_base) · U_val_grad_kp,
    # so that  ∂U_val/∂q = −U_val_grad_kp · dq  propagates through jax.jvp
    # (via the chain rule  ∂/∂q = −∂/∂kvec_p).  At q = q_base the primal
    # U_val is unchanged; only the tangent picks up the projector correction.
    if U_val_grad_kp is not None and kvec_p_base_for_grad is not None:
        dk = kvec_p_traced - kvec_p_base_for_grad                  # (3,)
        U_val_eff = U_val_kminq_G + jnp.einsum(
            'i,ivsG->vsG', dk, U_val_grad_kp, optimize=True)
        if U_val_hess_kp is not None:
            U_val_eff = U_val_eff + 0.5 * jnp.einsum(
                'i,j,ijvsG->vsG', dk, dk, U_val_hess_kp, optimize=True)
    else:
        U_val_eff = U_val_kminq_G

    # Build the Sternheimer operator at this kvec_p, then solve for δu_wrap.
    op = build_sternheimer_op_at_kvec_traced(
        kvec_p_traced, Gkminq_int_np, vnl_setup,
        V_scf=V_scf, mask=mask,
        Gx=Gx, Gy=Gy, Gz=Gz, fft_grid=fft_grid,
        bdot=bdot, vnl_E_super=vnl_E_super,
        U_val_kminq_G=U_val_eff, eps_v=eps_v,
        alpha_pv_sc=alpha_pv_sc, precond_diag=precond_diag,
        U_extra_G=U_extra_G, eps_extra=eps_extra,
    )

    # Source b for this q.  The PHYSICAL source is Q_{k-q}(q) · V_pert · u_{v,k_s}.
    # When the projector is unfrozen, Q_{k-q} depends on q through U_val_eff,
    # and b must be recomputed here so its q-tangent enters the custom_jvp
    # rule.  If Vu_G_preQ (= V_pert · u_{v,k_s} on the p-sphere, BEFORE
    # Q-projection) is provided, apply Q(U_val_eff) here; otherwise fall
    # back to the frozen-source path for backward compatibility.
    if Vu_G_preQ is not None:
        # Q_{k-q}(x) = x − U_val_eff · (U_val_eff^† x)
        coefs = jnp.einsum('msG,vsG->vm', jnp.conj(U_val_eff), Vu_G_preQ, optimize=True)
        b_eff = Vu_G_preQ - jnp.einsum('vm,msG->vsG', coefs, U_val_eff, optimize=True)
    else:
        b_eff = b

    delta_u = sternheimer_solve(op, b_eff, tol=tol, max_iter=max_iter, use_schur=use_schur)

    # ── Density contribution (naive frame; phase_unwrap handles G_wrap) ──
    # accumulate_chi_density:  δn_contrib(r) = Σ_v u_{v,k_s}(r) · conj(δu_naive(r))
    # where δu_naive(r) = phase_unwrap(r) · δu_wrap(r).
    delta_n_contrib_r = accumulate_chi_density(
        U_val_k_box, delta_u, jnp.asarray(Gkminq_int_np), fft_grid,
        phase_unwrap=phase_unwrap)

    # ── FFT δn to G-space, gather at target G' list ──
    chi_col_raw = project_density_to_Gsphere(delta_n_contrib_r, Gprime_int)

    # ── Apply Adler-Wiser normalization prefactor ──
    return prefactor * chi_col_raw


# ═══════════════════════════════════════════════════════════════════════
#  Full-BZ WFN buffer
# ═══════════════════════════════════════════════════════════════════════

def _load_unfolded_wfns(wfn, sym, meta, nb_load: int, verbose: bool):
    """Load unfolded ψ for all full-BZ k-points and up to ``nb_load`` bands.

    Single-GPU: everything lives in one big on-device array of shape
    ``(nk_full, nb_load, nspinor, nx, ny, nz)`` (FFT-box layout so later
    real-space operations don't need to re-scatter).

    Returns
    -------
    psi_box_full : jax.Array
    en_full      : (nk_full, nb_load) float, expanded from IBZ energies.
    """
    t0 = time.perf_counter()
    nk_full = int(sym.nk_tot)
    irk_to_k = np.asarray(sym.irk_to_k_map)          # (nk_full,) IBZ index per full-BZ k

    # Pre-allocate the full-BZ FFT-box buffer.  For MoS2 3×3 with 9 k-points,
    # 26 occupied + 20 conduction = 46 bands, 72×72×108 grid, nspinor=2:
    #   9 × 46 × 2 × 72 × 72 × 108 × 16 bytes ≈ 7.4 GiB.  Fits on A100 (40 GiB).
    psi_list = []
    for ik in range(nk_full):
        psi_list.append(load_kpoint_fftbox(wfn, sym, meta, ik, nb_load))
    psi_box_full = jnp.stack(psi_list, axis=0)

    en_irk = np.asarray(wfn.energies[0, :, :nb_load], dtype=np.float64)
    en_full = jnp.asarray(en_irk[irk_to_k, :])

    if verbose:
        dt = time.perf_counter() - t0
        print(f"  Unfolded WFN buffer: shape {psi_box_full.shape}  "
              f"({psi_box_full.nbytes / 1e9:.2f} GB)  in {dt:.1f}s")
    return psi_box_full, en_full


def _psi_box_to_G_sphere(psi_box, G_int):
    """Gather G-space scatter (``load_kpoint_fftbox`` output) at the given integer
    G-indices to recover the G-sphere coefficient list."""
    ix = jnp.mod(G_int[:, 0], psi_box.shape[-3])
    iy = jnp.mod(G_int[:, 1], psi_box.shape[-2])
    iz = jnp.mod(G_int[:, 2], psi_box.shape[-1])
    return psi_box[..., ix, iy, iz]


# ═══════════════════════════════════════════════════════════════════════
#  G' output list
# ═══════════════════════════════════════════════════════════════════════

def build_Gprime_list(qvec_crys: np.ndarray, wfn: WFNReader, ng_out: int) -> np.ndarray:
    """Return the ``ng_out`` lowest-|q+G'| integer G'-vectors (cell-periodic FFT-box
    indices, signed).  This is the column we output — G'=0 is the head, the rest
    are the wings."""
    # Build a box of integer G up to ±⌊fft_grid/2⌋ and sort by |q+G'|_cart².
    nx, ny, nz = (int(v) for v in wfn.fft_grid)
    ix = np.concatenate([np.arange(0, nx // 2 + 1), np.arange(-(nx // 2), 0)])
    iy = np.concatenate([np.arange(0, ny // 2 + 1), np.arange(-(ny // 2), 0)])
    iz = np.concatenate([np.arange(0, nz // 2 + 1), np.arange(-(nz // 2), 0)])
    Gx, Gy, Gz = np.meshgrid(ix, iy, iz, indexing='ij')
    G_int = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.int32)

    B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)   # (3,3) cartesian
    qG_cart = (np.asarray(qvec_crys, dtype=np.float64)[None, :] + G_int) @ B
    qG_sq = np.sum(qG_cart ** 2, axis=1)
    order = np.argsort(qG_sq)
    return G_int[order[:ng_out]]


# ═══════════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════════

def run_sternheimer(
    wfn_path: str,
    pseudo_dir: str,
    *,
    n_cond_bands: int = 0,
    iq_list: list[int] | None = None,
    ng_out: int = 64,
    tol: float = 1e-6,
    max_iter: int = 200,
    truncation_2d: bool = False,
    output_path: str = "sternheimer.h5",
    verbose: bool = True,
):
    """Forward Sternheimer G=0 column for a list of reduced-BZ q-points.

    Parameters
    ----------
    wfn_path : path to WFN.h5 from an insulating NSCF calculation.
    pseudo_dir : directory containing matching UPF pseudopotentials.
    n_cond_bands : int
        Extra conduction bands loaded into the WFN buffer (used later for
        Schur preconditioning; set to 0 to keep memory minimal for Stage 1).
    iq_list : list of reduced-BZ q-indices to run.  Default ``[0, 1]``.
    ng_out : number of G' to emit per q (sorted by |q+G'|_cart²).
    tol, max_iter : MINRES knobs.
    truncation_2d : apply 2D Coulomb truncation in V_H (slab geometries).
    output_path : sternheimer.h5 output.
    """
    if iq_list is None:
        iq_list = [0, 1]

    verbose = verbose and jax.process_index() == 0
    if verbose:
        print(f"Sternheimer G=0 column driver")
        print(f"  wfn       = {wfn_path}")
        print(f"  pseudos   = {pseudo_dir}")
        print(f"  iq_list   = {iq_list}")
        print(f"  tol/iter  = {tol:.0e} / {max_iter}")
        print(f"  trunc_2d  = {truncation_2d}")

    # ── Load WFN + SymMaps + Meta ──────────────────────────────────────
    t_setup = time.perf_counter()
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    nspinor = int(wfn.nspinor)

    # Occupied count per k (insulator).  wfn.nelec is ifmax — total occupied
    # states, nspinor-aware: for nspinor=1 nelec counts doubly-occupied
    # orbitals (= N_el/2); for nspinor=2 it is N_el.
    n_occ = int(wfn.nelec)
    nb_load = n_occ + max(0, int(n_cond_bands))
    if nb_load > int(wfn.nbands):
        raise ValueError(
            f"Requested n_occ({n_occ}) + n_cond_bands({n_cond_bands}) = "
            f"{nb_load} bands, but WFN has only {int(wfn.nbands)}.")

    # Minimal Meta (we only consume fft_grid / nspinor / cell_volume).
    # bispinor=False → meta.nspinor = wfn.nspinor (native 2-component for FR,
    # 1 for scalar).  The 4-component (bispinor=True) path is a specialised
    # LORRAX mode not needed for the density-response Sternheimer.
    meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=max(0, n_cond_bands),
                            nband=nb_load, n_rmu=0, bispinor=False)

    # ── Pseudos + V_scf ────────────────────────────────────────────────
    pseudos = load_pseudopotentials(pseudo_dir)
    if verbose:
        print(f"\n── ρ_val from full-BZ sum ──")
    rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=verbose)
    if verbose:
        print(f"\n── V_scf ──")
    V_scf, V_loc, vnl_setup = build_dft_potentials(
        wfn, pseudos, rho_val,
        truncation_2d=truncation_2d, verbose=verbose)

    # ── Unfolded WFN buffer (all k, all bands we need) ─────────────────
    if verbose:
        print(f"\n── Loading unfolded ψ (nk_full={sym.nk_tot}, nb={nb_load}) ──")
    psi_box_full, en_full = _load_unfolded_wfns(wfn, sym, meta, nb_load, verbose=verbose)
    # FFT-box layout: (nk_full, nb, ns, nx, ny, nz)

    # Per-k G-vector lists — but the **canonical** G-order we use for coefficients
    # is ``H_k.{Gx, Gy, Gz}`` from ``setup_H_k_from_kvec``, NOT SymMaps' order.
    # Those two orderings are permutations of each other (both are sphere-sorts,
    # but differ in the tie-breaking rule).  Mixing them corrupts apply_H, which
    # scatters coefficients using H_k's order — the first MoS2 run revealed this
    # by giving <u|H|u> − ε = +3.77 Ry with huge off-diagonals; using H_k's order
    # recovers <u|H|u> = ε to 0.04 mRy (NSCF convergence residual).
    #
    # We pre-compute and cache per-k (Gk_int, H_k) pairs so the inner loop
    # doesn't re-setup H_k twice per (ik_full, iq).  H_k's Gx/Gy/Gz are
    # jax.Arrays; we store them alongside the H_k object.
    H_cache = []   # list of (H_k, Gk_int_canonical) per full-BZ k
    for ik in range(sym.nk_tot):
        kv = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
        H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
        Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
        H_cache.append((H_k, Gk_int))
    if verbose:
        print(f"  H_k cache: {len(H_cache)} Hamiltonians, max nG={max(int(H.nG) for H, _ in H_cache)}")

    if verbose:
        dt = time.perf_counter() - t_setup
        print(f"\n  Setup complete ({dt:.1f}s)")

    # ── α_pv: level-shift for QE-DFPT-style positive-definite solve ────
    # Following LR_Modules/setup_alpha_pv.f90:  α_pv = 2·(E_max − E_min) of the
    # loaded-band spectrum.  This is enough to make
    #   A_v = H_{k-q} − ε_{v,k} + α_pv · P_val^{k-q}
    # positive-definite on the ENTIRE Hilbert space: on range(P_val) the
    # eigenvalues are ≈ α_pv − (ε_{v'} − ε_{v,k}) > 0 by construction, and on
    # range(Q) they are ε_c − ε_{v,k} > 0 for any insulator.  With A_v PD we
    # run plain CG (no projection inside iterations) — the mathematical gap
    # between "projected MINRES" and "level-shifted CG" lands us at the same
    # solution (both: δu ⊥ occupied at k-q when b is) but CG avoids MINRES'
    # pseudo-convergence-NaN pitfall under our JIT'd fixed-iter loop.
    en_occ = np.asarray(en_full[:, :n_occ], dtype=np.float64)
    alpha_pv = float(2.0 * (en_occ.max() - en_occ.min()))
    if verbose:
        print(f"\n  α_pv = 2·(E_max − E_min) = {alpha_pv:.3f} Ry "
              f"(from {n_occ} occupied bands × {int(sym.nk_tot)} k-points)")

    # ── HDF5 output ────────────────────────────────────────────────────
    out_h5 = h5py.File(output_path, "w")
    out_h5.attrs['tol'] = tol
    out_h5.attrs['max_iter'] = max_iter
    out_h5.attrs['n_cond_bands'] = n_cond_bands
    out_h5.attrs['n_occ'] = n_occ
    out_h5.attrs['truncation_2d'] = truncation_2d
    out_h5.attrs['note'] = "chi_col[iq, ig] = χ_{G'0}(q, ω=0); G' sorted by |q+G'|_cart²"
    out_h5.create_dataset('q_crys', data=np.asarray(wfn.kpoints[iq_list]))
    out_h5.create_dataset('iq_reduced', data=np.asarray(iq_list, dtype=np.int32))

    # ══════════════════════════════════════════════════════════════════
    #  q-loop
    # ══════════════════════════════════════════════════════════════════
    nk_full = int(sym.nk_tot)
    nx, ny, nz = (int(v) for v in wfn.fft_grid)
    N_grid = nx * ny * nz
    vol = float(wfn.cell_volume)

    for q_idx, iq_red in enumerate(iq_list):
        qvec_pos = np.asarray(wfn.kpoints[iq_red], dtype=np.float64)
        # Signed representative q_signed ∈ [-1/2, 1/2)³.  Using signed-q makes
        # V_pert_base(r) = 1 self-consistent (the physical perturbation
        # e^{i q_signed · r} has a purely plane-wave character with no extra
        # reciprocal-lattice phase) AND minimises the number of k's for which
        # (k - q) wraps back into the BZ.  Non-wrap k contributions then give
        # pure-real χ₀₀ trivially; wrap k's pick up a small G_wrap phase.
        qvec = qvec_pos - np.round(qvec_pos)                        # → [-0.5, 0.5)
        # ``sym.kq_map[ik, iq_red]`` is indexed by the **positive** iq_red,
        # so we keep iq_red as-is for the index lookup; only the q-value
        # itself flips to its signed representative.
        if verbose:
            print(f"\n══ q[{q_idx}] = {qvec}   (signed; reduced idx {iq_red}) ══")

        Gprime_int = build_Gprime_list(qvec, wfn, ng_out)        # (ng_out, 3)
        Gprime_j = jnp.asarray(Gprime_int)

        # Cell-periodic part of the density-response perturbation is identically
        # 1; any wrap-gauge factor e^{+iG_wrap·r} is added per-k inside the
        # k-loop (see below).  For phonon DFPT the caller would pass a real
        # ∂V_scf/∂R_α here instead.
        V_pert_base = make_density_perturbation(wfn.fft_grid)

        # Accumulator for δn(r) over all k.
        delta_n_r = jnp.zeros((nx, ny, nz), dtype=jnp.complex128)

        # Per-q diagnostics.
        total_res = 0.0
        total_conv = True
        total_leak = 0.0

        t_q = time.perf_counter()
        for ik_full in range(nk_full):
            ik_kminq = int(sym.kq_map[ik_full, iq_red])

            # ── Pull cached H / G-order (canonical = H_k's Gx/Gy/Gz) ──
            H_k,     Gk_int     = H_cache[ik_full]
            H_kminq, Gkminq_int = H_cache[ik_kminq]
            apply_H_kminq = make_apply_H(H_kminq)

            # ── Umklapp G_wrap for the wrapped ↔ naive gauge conversion ──
            # Convention:  p_naive = k - q,  p_wrap ∈ [0,1)³ from kq_map,
            # so  G_wrap = p_naive − p_wrap ∈ ℤ³.  Then
            #     u_{v, k_wrap}(r)  =  e^{+i G_wrap · r} · u_{v, k_naive}(r)
            # (the Bloch phase identity for wrapping by a reciprocal vector).
            # The source is pushed naive → wrapped by multiplying by
            # phase_wrap = e^{+i G_wrap · r} (it's zero / unity when no wrap).
            # δu_wrap from the CG is pulled back wrapped → naive by
            # phase_unwrap = e^{-i G_wrap · r} inside accumulate_chi_density.
            kvec_k_np          = np.asarray(sym.unfolded_kpts[ik_full],  dtype=np.float64)
            kvec_kminq_wrap_np = np.asarray(sym.unfolded_kpts[ik_kminq], dtype=np.float64)
            G_wrap_np = np.rint(
                (kvec_k_np - qvec) - kvec_kminq_wrap_np).astype(np.int32)   # (3,)
            phase_wrap   = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=+1)
            phase_unwrap = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=-1)
            # Compose the perturbation used by the source builder:
            #   V_pert_real(r) = V_pert_base(r) · phase_wrap(r)
            # For density response: V_pert_base = 1, so V_pert_real = phase_wrap.
            V_pert_real = V_pert_base * phase_wrap

            # ── ψ coefficients: box view (for FFT path) + G-sphere gather (for ops) ──
            U_val_k_box     = psi_box_full[ik_full,  :n_occ]           # (nv, ns, nx, ny, nz) G-space scatter
            U_val_k_G       = _psi_box_to_G_sphere(U_val_k_box, Gk_int)  # (nv, ns, ngk_k)
            U_val_kminq_box = psi_box_full[ik_kminq, :n_occ]
            U_val_kminq_G   = _psi_box_to_G_sphere(U_val_kminq_box, Gkminq_int)

            # Eigenvalues at the SOURCE k (not k-q).
            eps_vk = en_full[ik_full, :n_occ]                      # (nv,)

            # ── Projector (for source construction) ──
            Q_kminq = make_Q_kminq(U_val_kminq_G)

            # ── Source b_{v,k} = Q_{k-q} · V_pert(r) · u_{v,k} ──
            b = build_sternheimer_source(
                U_val_k_box, Gkminq_int, V_pert_real, Q_kminq)

            # Projector orthogonality sanity (‖U_p^† b‖ should be ~0).
            U_dag_b = jnp.einsum('nsG,vsG->vn', jnp.conj(U_val_kminq_G), b)
            leak_b = float(jnp.max(jnp.abs(U_dag_b)))
            total_leak = max(total_leak, leak_b)

            # ── Level-shifted Sternheimer primitive via ``sternheimer_solve`` ──
            # The operator  A_v = H_{k-q} − ε_{v,k} + α_pv · P_val^{k-q}  is
            # bundled into a single ``SternheimerOp`` pytree; the primitive
            # itself is ``@jax.jit``'d once and reused across all (ik, iq)
            # with matching shapes.  ``jax.custom_jvp`` on the primitive
            # means that wrapping this whole block in ``jax.jvp`` would give
            # the q-derivative via an implicit-differentiation solve.
            K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
            precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)

            # ── Schur-block extras: M low-energy conduction Ritz vectors at p=k-q ──
            # When ``n_cond_bands > 0`` is passed, we load that many extra
            # bands from WFN.h5 past n_occ and use them as the T-block in the
            # Schur split.  Because these are H_{k-q} eigenstates, A_TT is
            # diagonal with entries (ε_{c,p} − ε_v), and the T-block initial
            # guess reduces to the standard k·p formula — see
            # ``solvers.sternheimer_solve._schur_initial_guess``.  Since
            # n_cond_bands is a runtime knob (may be 0), we thread U_extra as
            # None vs populated.
            if n_cond_bands > 0:
                U_extra_kminq_box = psi_box_full[ik_kminq, n_occ:n_occ + n_cond_bands]
                U_extra_kminq_G = _psi_box_to_G_sphere(U_extra_kminq_box, Gkminq_int)
                eps_extra_kminq = en_full[ik_kminq, n_occ:n_occ + n_cond_bands]
            else:
                U_extra_kminq_G = None
                eps_extra_kminq = None

            op = SternheimerOp(
                T_diag=H_kminq.T_diag, V_scf=H_kminq.V_scf,
                Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
                vnl_Z=H_kminq.vnl_Z, vnl_E=H_kminq.vnl_E, mask=H_kminq.mask,
                U_val=U_val_kminq_G, eps_v=eps_vk,
                alpha_pv=jnp.asarray(alpha_pv, dtype=jnp.float64),
                precond_diag=precond_diag,
                fft_grid=H_kminq.fft_grid,
                U_extra=U_extra_kminq_G, eps_extra=eps_extra_kminq,
            )

            # ── Primal fast path for ‖b‖ ≈ 0 (q=0, G=0 case) ──
            b_norm_max = float(jnp.max(_batched_real_norm_host(b)))
            if b_norm_max < 1e-12:
                delta_u = jnp.zeros_like(b)
                res_max = 0.0
                converged = True
            else:
                use_schur = (n_cond_bands > 0)
                delta_u = sternheimer_solve(op, b, tol=tol, max_iter=max_iter,
                                             use_schur=use_schur)
                # Residual check outside the primitive (don't bake it into
                # the JIT trace — the primitive is converged-by-design).
                from solvers.sternheimer_solve import _apply_A_inline
                residual = -b - _apply_A_inline(op, delta_u)
                res_max = float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(residual)**2, axis=(1,2)))))
                converged = res_max < tol * b_norm_max

            total_res = max(total_res, res_max)
            total_conv = total_conv and converged

            # δu orthogonality sanity: ‖U_p^† δu‖ should be ~0 (from CG + shift).
            U_dag_du = jnp.einsum('nsG,vsG->vn',
                                   jnp.conj(U_val_kminq_G), delta_u)
            leak_du = float(jnp.max(jnp.abs(U_dag_du)))
            total_leak = max(total_leak, leak_du)

            # ── Density contribution — convert δu_wrap → δu_naive in real space ──
            delta_n_r = delta_n_r + accumulate_chi_density(
                U_val_k_box, delta_u, Gkminq_int, wfn.fft_grid,
                phase_unwrap=phase_unwrap)

        # ── Project δn → χ column at G' ──  Adler–Wiser normalisation.
        #
        # Derivation (see CHANGELOG 2026-04-24):
        #   1. ortho-IFFT convention:  u_r[j] = (1/√N) Σ_G c_u(G) e^{iG·r_j} = u(r_j)/√N.
        #   2. accumulate_chi_density returns  δn_r[j] = Σ_{v,k} (1/N) u(r_j)·conj(δu(r_j)).
        #   3. ortho-FFT at G'=0:  (1/√N)·Σ_j δn_r[j]  =  my raw chi_col[0].
        #   4. Continuous cell Fourier coef:  (δn)_{G'=0} = (1/V)∫_cell δn(r) dr
        #      = (1/N)·standard_FFT(δn)[0] = (1/√N)·ortho_FFT(δn)[0].
        #
        # Substituting the exact Sternheimer solution δu = Σ_c (-M_cv/ΔE) u_c,k-q,
        #   (my raw chi_col[G'=0]) = -(1/(√N·N_k)) · Σ_{v,c,k} |M_vc|² / ΔE_cv
        # whereas the Rydberg-atomic-units Adler–Wiser convention (e.g. BGW
        # epsmat.h5, and Hybertsen–Louie PRB 34 5390) is
        #   χ_physical = -(2·spin_factor/(V_cell·N_k)) · Σ |M|² / ΔE
        # with the "2" from the +ω/−ω pole combination at ω=0.  Therefore
        #
        #   χ_physical = (my raw chi_col) · [2·spin_factor·√N_grid / V_cell]
        #
        # Cross-checked against on-file sum-over-states at several q-points
        # on MoS2 3×3 FR: ratio agrees to CG-tolerance + band-cutoff residual.
        spin_factor = 2 if nspinor == 1 else 1
        prefactor = (2.0 * spin_factor * np.sqrt(N_grid)) / (vol * nk_full)
        delta_n_r = delta_n_r * prefactor

        chi_col = project_density_to_Gsphere(delta_n_r, Gprime_j)
        chi_col_np = np.asarray(chi_col)

        # ── Sanity print ──
        if verbose:
            dt = time.perf_counter() - t_q
            print(f"  ── q[{q_idx}] summary ──")
            print(f"    max CG residual          = {total_res:.3e}")
            print(f"    all v converged?         = {total_conv}")
            print(f"    max projector-leak       = {total_leak:.3e}")
            print(f"    χ_{{00}}(q, 0)           = {chi_col_np[0]:.6e}")
            # q=0 sanity check: should be ~0 for all G' (source b is 0 by Q_k·u_v=0).
            if np.allclose(qvec, 0.0):
                max_abs = float(np.max(np.abs(chi_col_np)))
                print(f"    q=0 check: max|χ(G')|   = {max_abs:.3e}  (expect ~0)")
            print(f"    time                     = {dt:.1f}s")

        # ── Write to HDF5 ──
        grp = out_h5.create_group(f"q_{q_idx}")
        grp.create_dataset('chi_col', data=chi_col_np)
        grp.create_dataset('G_int', data=Gprime_int)
        grp.attrs['q_crys'] = qvec
        grp.attrs['iq_reduced'] = int(iq_red)
        grp.attrs['max_residual'] = total_res
        grp.attrs['max_projector_leak'] = total_leak
        grp.attrs['all_converged'] = total_conv

    out_h5.close()
    if verbose:
        print(f"\nWrote χ columns → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Insulating Sternheimer G=0 column driver "
                    "(χ_{G'0}(q, ω=0) via projected MINRES).")
    parser.add_argument("-i", "--input", default=None,
                        help="INI-style input file (cohsex.in / sternheimer.in)")
    parser.add_argument("--wfn", default=None, help="WFN.h5 path")
    parser.add_argument("--pseudo_dir", default=None,
                        help="Directory with matching UPF pseudopotentials")
    parser.add_argument("--n-cond-bands", type=int, default=0,
                        help="Extra conduction bands in the WFN buffer (Schur prep)")
    parser.add_argument("--iq-list", type=int, nargs='+', default=[0, 1],
                        help="Reduced-BZ q indices to run (default: 0 1)")
    parser.add_argument("--ng-out", type=int, default=64,
                        help="# of G' in output column (sorted by |q+G'|_cart²)")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--truncation-2d", action="store_true",
                        help="Apply 2D Coulomb truncation in V_H (slab systems)")
    parser.add_argument("-o", "--output", default="sternheimer.h5")
    args = parser.parse_args(argv)

    # Input-file path wins for defaults; clargs override.
    wfn_path = args.wfn
    pseudo_dir = args.pseudo_dir
    iq_list = args.iq_list
    truncation_2d = args.truncation_2d
    n_cond_bands = args.n_cond_bands
    if args.input:
        from psp.get_DFT_mtxels import read_cohsex_input
        input_path = Path(args.input).resolve()
        params = read_cohsex_input(str(input_path))
        if wfn_path is None:
            wp = Path(params.get("wfn_file", "WFN.h5"))
            wfn_path = str((input_path.parent / wp).resolve()
                           if not wp.is_absolute() else wp)
        if pseudo_dir is None:
            pseudo_dir = str(input_path.parent)

    if wfn_path is None:
        parser.error("--wfn (or -i with wfn_file set in the input) is required")
    if pseudo_dir is None:
        pseudo_dir = str(Path(wfn_path).parent)

    # Enable JAX persistent compile cache (mirrors run_nscf).
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as _e:
        if jax.process_index() == 0:
            print(f"  [jax compile cache] skipped: {_e}", flush=True)

    run_sternheimer(
        wfn_path=wfn_path,
        pseudo_dir=pseudo_dir,
        n_cond_bands=n_cond_bands,
        iq_list=list(iq_list),
        ng_out=args.ng_out,
        tol=args.tol,
        max_iter=args.max_iter,
        truncation_2d=truncation_2d,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
