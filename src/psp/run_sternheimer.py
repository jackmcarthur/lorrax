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
from solvers.cg_posdef import cg_posdef
from solvers.projectors import make_P_val, make_Q_kminq
from solvers.sternheimer_precond import (
    compute_per_band_kinetic,
    make_tpa_preconditioner,
)


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
    delta_u_G: jax.Array,          # (nv, nspinor, ngk_p)        G-sphere coeffs at k-q
    Gkminq_int: jax.Array,         # (ngk_p, 3)
    fft_grid,
) -> jax.Array:
    """Return the (nx,ny,nz) real-space contribution
    ``Σ_v  u_{v,k}(r) · conj(δu_{v,k}^q(r))``  of χ_{G'0}(q,0)'s induced density.

    The +q momentum component of  ψ^*·δψ + c.c.  that picks up ``e^{-i(q+G')·r}``
    in the χ integral is  u · conj(δu)  (see module docstring derivation).
    Summed over v here; caller sums over k and FFTs afterwards.
    """
    # u_{v,k}(r): IFFT of the already-scattered G-box.
    u_r = jnp.fft.ifftn(U_box_k, axes=(-3, -2, -1), norm='ortho')
    # δu_{v,k}^q(r): scatter G-sphere coeffs to a fresh box, then IFFT.
    nv, nspinor, _ = delta_u_G.shape
    nx, ny, nz = fft_grid
    ix = jnp.mod(Gkminq_int[:, 0], nx)
    iy = jnp.mod(Gkminq_int[:, 1], ny)
    iz = jnp.mod(Gkminq_int[:, 2], nz)
    du_box = jnp.zeros((nv, nspinor, nx, ny, nz), dtype=delta_u_G.dtype)
    du_box = du_box.at[:, :, ix, iy, iz].set(delta_u_G)
    du_r = jnp.fft.ifftn(du_box, axes=(-3, -2, -1), norm='ortho')
    return jnp.sum(u_r * jnp.conj(du_r), axis=(0, 1))


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

    The full external perturbation is ``V_ext(r) = e^{iq·r}``, which has no
    cell-periodic part.  Acting on ψ_{v,k} = e^{ik·r}u_{v,k}(r), the product
    ``e^{iq·r} · ψ_{v,k}(r) = e^{i(k+q)·r} · u_{v,k}(r)`` has Bloch momentum
    k+q and the **same** cell-periodic part u_{v,k}(r).  In our p = k-q
    convention the symmetric result is that the cell-periodic source at p
    is u_{v,k} itself — i.e. ``V_pert_cell(r) = 1``.  The q-dependence
    enters only through the change of G-sphere (k → k-q).

    This is why the forward χ_{G'0}(q, 0) pipeline does NOT apply any
    real-space phase for the density-response case — the ``e^{-iq·r}`` phase
    that appears in the analytic integral ``∫ dr e^{-i(q+G')r} δn(r)`` is
    accounted for by the p = k-q momentum-shift bookkeeping inside
    ``build_sternheimer_source`` + ``accumulate_chi_density``.

    For phonon DFPT the caller would instead pass
    ``V_pert_cell(r) = ∂V_scf/∂R_α`` (cell-periodic by construction); the
    e^{iqr} plane-wave factor still comes in for free via the k → k+q shift.
    """
    nx, ny, nz = fft_grid
    return jnp.ones((nx, ny, nz), dtype=jnp.complex128)


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
        qvec = np.asarray(wfn.kpoints[iq_red], dtype=np.float64)
        if verbose:
            print(f"\n══ q[{q_idx}] = {qvec}   (reduced idx {iq_red}) ══")

        Gprime_int = build_Gprime_list(qvec, wfn, ng_out)        # (ng_out, 3)
        Gprime_j = jnp.asarray(Gprime_int)

        # Cell-periodic part of the density-response perturbation: identically
        # 1.  (See ``make_density_perturbation`` docstring for why the e^{iq·r}
        # factor enters through the k→k-q G-sphere shift instead.)
        V_pert_real = make_density_perturbation(wfn.fft_grid)

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

            # ── ψ coefficients: box view (for FFT path) + G-sphere gather (for ops) ──
            U_val_k_box     = psi_box_full[ik_full,  :n_occ]           # (nv, ns, nx, ny, nz) G-space scatter
            U_val_k_G       = _psi_box_to_G_sphere(U_val_k_box, Gk_int)  # (nv, ns, ngk_k)
            U_val_kminq_box = psi_box_full[ik_kminq, :n_occ]
            U_val_kminq_G   = _psi_box_to_G_sphere(U_val_kminq_box, Gkminq_int)

            # Eigenvalues at the SOURCE k (not k-q).
            eps_vk = en_full[ik_full, :n_occ]                      # (nv,)

            # ── Projector + level-shift operator for CG ──
            Q_kminq = make_Q_kminq(U_val_kminq_G)
            P_val_kminq = make_P_val(U_val_kminq_G)

            # ── Source b_{v,k} = Q_{k-q} · V_pert(r) · u_{v,k} ──
            b = build_sternheimer_source(
                U_val_k_box, Gkminq_int, V_pert_real, Q_kminq)

            # Projector orthogonality sanity (‖U_p^† b‖ should be ~0).
            U_dag_b = jnp.einsum('nsG,vsG->vn', jnp.conj(U_val_kminq_G), b)
            leak_b = float(jnp.max(jnp.abs(U_dag_b)))
            total_leak = max(total_leak, leak_b)

            # ── TPA preconditioner ──
            # K̄²_v = ⟨ψ_{v,k}|T_k|ψ_{v,k}⟩ from the *source* k's kinetic diagonal.
            K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
            precond = make_tpa_preconditioner(H_kminq.T_diag, K_bar_sq)

            # ── QE-DFPT level-shifted Sternheimer operator ──
            #   A_v(x) = H_{k-q} x − ε_{v,k} x + α_pv · P_val^{k-q}(x)
            # Positive-definite on the whole Hilbert space ⇒ plain CG converges
            # without projector drift.  The solution is automatically orthogonal
            # to occupied because the RHS is Q_{k-q}-projected (anything in
            # range(P_val) on the LHS must come from a P_val component of b).
            def apply_A(x, apply_H_kminq=apply_H_kminq,
                         P_val_kminq=P_val_kminq,
                         eps_vk=eps_vk, alpha_pv=alpha_pv):
                Hx = apply_H_kminq(x)
                shifted = Hx - eps_vk[:, None, None].astype(x.dtype) * x
                return shifted + alpha_pv * P_val_kminq(x)

            # ── Primal fast path: ‖b‖ below numerical noise → δu = 0 ──
            # At q = 0, b = Q_k · u_{v,k} = 0 analytically; floating-point
            # gives ~1e-15 residue.  Running CG on numerical noise is not only
            # wasteful, it's harmful (see the dead-band saga in solvers/minres.py).
            # The correct primal answer is x = 0; the non-trivial content at
            # q = 0 sits in the *derivative*, which comes from a custom JVP
            # (see Stage 3) that solves A·ẋ = ḃ with the differentiated source —
            # not from running CG on zero.
            b_norms = _batched_real_norm_host(b)       # (nv,) float — cheap, on-device
            b_norm_max = float(jnp.max(b_norms))
            if b_norm_max < 1e-12:
                delta_u = jnp.zeros_like(b)
                res_max = 0.0
                converged = True
            else:
                delta_u, info = cg_posdef(
                    apply_A, -b, precond=precond,
                    tol=tol, max_iter=max_iter)
                res_max = float(jnp.max(info.res_norms))
                converged = bool(jnp.all(info.converged))

            total_res = max(total_res, res_max)
            total_conv = total_conv and converged

            # δu orthogonality sanity: ‖U_p^† δu‖ should be ~0 (from CG + shift).
            U_dag_du = jnp.einsum('nsG,vsG->vn',
                                   jnp.conj(U_val_kminq_G), delta_u)
            leak_du = float(jnp.max(jnp.abs(U_dag_du)))
            total_leak = max(total_leak, leak_du)

            # ── Density contribution ──
            delta_n_r = delta_n_r + accumulate_chi_density(
                U_val_k_box, delta_u, Gkminq_int, wfn.fft_grid)

        # ── Project δn → χ column at G' ──
        # Prefactor = spin_factor / N_k.  Spin factor is 2 for scalar (nspinor=1)
        # and 1 for FR (nspinor=2); the conventional Adler–Wiser "2" for
        # spin degeneracy.  The +q and -q components of δn are NOT both summed
        # into χ_{G'0}(q, 0) — only the +q component matches e^{-i(q+G')·r}
        # in the integral, so no extra factor of 2 from pole combination.
        #
        # Absolute normalisation (cell volume, FFT-box factors from ortho-FFT)
        # is deferred: downstream consumers compare trend-wise against BGW
        # band-sum output, which pins down the multiplicative constant.
        spin_factor = 2 if nspinor == 1 else 1
        prefactor = spin_factor / nk_full
        delta_n_r = delta_n_r * prefactor

        chi_col = project_density_to_Gsphere(delta_n_r, Gprime_j)
        chi_col_np = np.asarray(chi_col)

        # ── Sanity print ──
        if verbose:
            dt = time.perf_counter() - t_q
            print(f"  ── q[{q_idx}] summary ──")
            print(f"    max MINRES residual      = {total_res:.3e}")
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
