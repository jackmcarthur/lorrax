"""Pivoted-Cholesky pruning of over-sampled ISDF candidate points.

Implements the q=0 candidate-pruning stage described in
``pivoted_cholesky.md`` (sandbox root). The idea: k-means gives a set of M
candidate points ``{r̃_a}`` (M > N_μ); the pair-product rows
``z_{a,(vck)} = φ*_{v,k}(r̃_a) ψ_{c,k}(r̃_a)`` define a Hermitian PSD Gram
matrix ``G^{(0)} ∈ ℂ^{M×M}``. Greedy pivoted Cholesky picks the N_μ pivots
with the largest residual Schur-complement diagonal, and the corresponding
``r̃_a`` become the final ISDF points. This is strictly better than picking
on amplitude alone because it targets the coherence structure of the
valence-conduction pair-product space the ISDF fit will actually use.

Architectural map to ``common/isdf_fitting.py``:

    compute_pair_density_spin_traced  ←→  per-k P^{(v/c)}(a, b)
                                          (same einsum, just evaluated at
                                          candidates r̃_a not chosen r_μ)
    compute_CCT_from_left_right       ←→  k→q FFT of the cross-product
                                          → at q=0 this is just sum_k
    (nothing)                         ←→  pivoted_cholesky_select  (new)

This module is deliberately single-device for the first cut per the md's
guidance ("Do not start with distributed pivoting"). The ``build_gram_q0``
step is a k-serial ``lax.fori_loop``; the select step is a single
``lax.fori_loop`` with static ``k_keep``. Both jit cleanly.

Shapes (following the md):

    phi_val_cand   (nk, nv_eff, M)   complex  φ_{v,k}(r̃_a)
    psi_cond_cand  (nk, nc_eff, M)   complex  ψ_{c,k}(r̃_a)
    G              (M, M)            complex  Hermitian PSD
    L              (M, k_keep)       complex  Cholesky columns (padded)
    piv            (k_keep,)         int32    pivot indices (−1 past rank)
    d_final        (M,)              real     Schur-complement residuals

``nv_eff`` / ``nc_eff`` fold the spinor axis into the band axis
(nv_eff = nv_bands · nspinor), matching the md's "assume spin has already
been folded" convention.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental.shard_map import shard_map
from functools import partial

from file_io import WFNReader
from common import symmetry_maps


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — gather wavefunctions at candidate points
# ═══════════════════════════════════════════════════════════════════════


def gather_wfn_at_candidates(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    cand_idx: np.ndarray,
    band_start: int,
    band_end: int,
) -> jnp.ndarray:
    """Evaluate ψ_{n,k}(r̃_a) at M candidate FFT-grid points for bands in a slice.

    Uses IBZ wavefunctions (no full-BZ unfold). The implementation is:

      1. For each irreducible k-point:
         a. Read raw coefficients ``c_n,k(G)`` for the requested band slice.
         b. Scatter onto the QE FFT grid (n_x, n_y, n_z).
         c. iFFT to real space.
         d. Gather at the M candidate indices.
      2. Stack along k and return.

    Spinor components are preserved: output shape is
    ``(nkpts, nb, nspinor, M)``. The caller typically flattens the spinor
    axis into the band axis for Gram assembly.

    Args:
        wfn: open WFNReader.
        sym: matching SymMaps (currently unused — raw IBZ coefficients
            suffice because |ψ|² and cross products don't care about the
            fractional-translation phase that SymMaps applies — but kept in
            the signature for future use).
        cand_idx: (M, 3) int FFT-grid indices, already reduced mod fft_grid.
        band_start, band_end: half-open band range [band_start, band_end).

    Returns:
        psi_cand: (nkpts, band_end - band_start, nspinor, M) complex64
            ψ_{n,k}(r̃_a), one slab per IBZ k-point.
    """
    del sym  # reserved; see docstring
    nx, ny, nz = (int(x) for x in wfn.fft_grid)
    nspinor = int(wfn.nspinor)
    nb = int(band_end - band_start)
    M = int(cand_idx.shape[0])
    band_indices = np.arange(band_start, band_end)
    ix = np.asarray(cand_idx[:, 0], dtype=np.int64) % nx
    iy = np.asarray(cand_idx[:, 1], dtype=np.int64) % ny
    iz = np.asarray(cand_idx[:, 2], dtype=np.int64) % nz

    # IFFT scale: pick same convention as get_DFT_mtxels.compute_valence_density
    # (norm='ortho') so that |ψ|² integrates to 1 per orbital.
    psi_out = np.zeros((wfn.nkpts, nb, nspinor, M), dtype=np.complex64)
    for ik in range(wfn.nkpts):
        gvecs_k = np.asarray(wfn.get_gvec_nk(ik))  # (ngk, 3)
        cnk = wfn.get_cnk_batch(ik, band_indices)  # (nb, nspinor, ngk)
        # Scatter → FFT box, iFFT, gather at M candidate points.
        box = np.zeros((nb, nspinor, nx, ny, nz), dtype=np.complex128)
        box[:, :, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = cnk
        psi_r = np.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
        psi_out[ik] = psi_r[..., ix, iy, iz].astype(np.complex64)
    return jnp.asarray(psi_out)


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Gram matrix G^{(0)}_{ab}
# ═══════════════════════════════════════════════════════════════════════


def _fold_spin_into_band(psi: jnp.ndarray) -> jnp.ndarray:
    """(nk, nb, nspinor, M) → (nk, nb * nspinor, M). Pure reshape."""
    nk, nb, ns, M = psi.shape
    return psi.reshape(nk, nb * ns, M)


@partial(jax.jit, static_argnames=('enforce_hermitian',))
def build_candidate_gram_q0(
    phi_val_cand: jnp.ndarray,
    psi_cond_cand: jnp.ndarray,
    k_weights: jnp.ndarray | None = None,
    enforce_hermitian: bool = True,
) -> jnp.ndarray:
    """Build the (M, M) Hermitian PSD candidate Gram for q = 0.

        G_{ab} = Σ_k w_k · [Σ_v  φ_{v,k}(r̃_a)  φ*_{v,k}(r̃_b)]
                        · [Σ_c  ψ*_{c,k}(r̃_a) ψ_{c,k}(r̃_b)]

    (pivoted_cholesky.md §1, §4.2)

    The k-loop is a ``lax.fori_loop`` so the whole thing fits in a single
    jit; memory peak is (M, M) complex plus two (nv|nc, M) slabs per step.

    Args:
        phi_val_cand:  (nk, nv_eff, M) complex — spinor folded into band.
        psi_cond_cand: (nk, nc_eff, M) complex — same.
        k_weights: optional (nk,) real. If None, all k-points get weight 1.
        enforce_hermitian: if True, symmetrize ``(G + G^H) / 2`` at exit.

    Returns:
        G: (M, M) complex.
    """
    nk, nv, M = phi_val_cand.shape
    _, nc, M2 = psi_cond_cand.shape
    del nv, nc  # for pyflakes
    if k_weights is None:
        k_weights = jnp.ones((nk,), dtype=phi_val_cand.real.dtype)

    def body_fun(k, G):
        phi_k = phi_val_cand[k]   # (nv, M)
        psi_k = psi_cond_cand[k]  # (nc, M)

        # Valence projector: P_v[a,b] = Σ_v φ_{v}(a) φ*_{v}(b)
        # In matrix form: P_v = φ.T @ conj(φ)  with φ: (nv, M) → (M, M)
        P_v = phi_k.T @ jnp.conj(phi_k)

        # Conduction projector: P̃_c[a,b] = Σ_c ψ*_{c}(a) ψ_{c}(b)
        #                   = conj(ψ).T @ ψ                    (M, M)
        P_c_tilde = jnp.conj(psi_k).T @ psi_k

        return G + k_weights[k] * (P_v * P_c_tilde)

    G = jnp.zeros((M, M2), dtype=phi_val_cand.dtype)
    G = lax.fori_loop(0, nk, body_fun, G)

    if enforce_hermitian:
        G = 0.5 * (G + jnp.conj(G.T))
    return G


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — jitted exact greedy pivoted Cholesky
# ═══════════════════════════════════════════════════════════════════════


@partial(jax.jit, static_argnames=('k_keep',))
def pivoted_cholesky_select(
    G: jnp.ndarray,
    k_keep: int,
    tol_rel: float = 1e-10,
    tol_abs: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact greedy pivoted Cholesky on an explicit Hermitian PSD matrix.

    Standard LAPACK pstrf-style algorithm written so it jits cleanly:

      * ``k_keep`` is static, so ``L`` has a fixed (M, k_keep) shape.
      * ``lax.fori_loop`` drives the outer iteration (no Python break).
      * Stopping handled by a ``done`` flag in the carry; past-convergence
        iterations become masked no-ops.
      * ``L[:, :j]`` is replaced by a full-width masked matvec so the
        shape is static under jit.
      * G itself is never physically reordered — the permutation is just
        the ``piv`` output.

    The algorithm (textbook, see pivoted_cholesky.md §4.3):

      d_0 = diag(G)
      for j = 0 .. k_keep - 1:
        p = argmax d over active set
        if d[p] < tol: mark done and break (via mask)
        L[:, j] = (G[:, p] - L[:, :j] @ conj(L[p, :j])) / sqrt(d[p])
        L[p, j] = sqrt(d[p])                # cleanup pivot entry
        d := max(d - |L[:, j]|^2, 0)        # Schur complement
        d[p] := -inf                         # mark p inactive
        piv[j] = p

    Args:
        G: (M, M) complex Hermitian PSD matrix.
        k_keep: static number of pivots to take.
        tol_rel: stopping tolerance relative to max(diag(G)). Once the
            largest residual diagonal drops below this, later loop
            iterations do nothing.
        tol_abs: absolute stopping tolerance on the residual diagonal.

    Returns:
        piv: (k_keep,) int32. Entries past ``rank`` are −1.
        L: (M, k_keep) complex. Columns past ``rank`` are zero.
        rank: int32 scalar — how many pivots were actually taken.
        d_final: (M,) real — residual Schur-complement diagonal at end;
            +inf sentinels (inactive rows) are returned as 0.
    """
    M = G.shape[0]
    real_dtype = G.real.dtype
    eps = jnp.finfo(real_dtype).eps
    minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)

    diag0 = jnp.maximum(jnp.real(jnp.diag(G)), 0.0)
    d0max = jnp.max(diag0)

    L0 = jnp.zeros((M, k_keep), dtype=G.dtype)
    piv0 = -jnp.ones((k_keep,), dtype=jnp.int32)
    d0 = diag0
    active0 = jnp.ones((M,), dtype=bool)
    done0 = jnp.array(False)
    rank0 = jnp.array(0, dtype=jnp.int32)

    col_ids = jnp.arange(k_keep)

    def body_fun(j, carry):
        d, L, piv, active, done, rank = carry

        # Step 1: pick the pivot = row with the largest residual diagonal
        # among the still-active rows.
        masked_d = jnp.where(active, d, minus_inf)
        p = jnp.argmax(masked_d)
        pivot_val = masked_d[p]

        # Should we take this pivot this iteration? (False if already done,
        # or if residual is below tolerance.)
        take = (~done) & (pivot_val > tol_abs) & (pivot_val > tol_rel * d0max)

        # Step 2: compute the new Cholesky column. Need
        #   L[:, j] = (G[:, p] - Σ_{i<j} L[:, i] * conj(L[p, i])) / sqrt(d[p])
        # Implemented as a full-width matvec with ``col_ids < j`` mask, so
        # the shape is static under jit.
        prev_mask = (col_ids < j).astype(G.dtype)
        gcol = G[:, p]                            # (M,)
        corr = L @ (jnp.conj(L[p, :]) * prev_mask)
        denom = jnp.where(take, jnp.sqrt(jnp.maximum(pivot_val, eps)), 1.0)
        newcol = (gcol - corr) / denom
        newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))
        # Numerical cleanup: the pivot entry should be exactly sqrt(d[p]).
        newcol = newcol.at[p].set(
            jnp.where(take, denom.astype(G.dtype), newcol[p])
        )

        # Step 3: write the new column and pivot index (no-op when ~take).
        oldcol = L[:, j]
        L = L.at[:, j].set(jnp.where(take, newcol, oldcol))
        piv = piv.at[j].set(jnp.where(take, p.astype(jnp.int32), piv[j]))

        # Step 4: Schur-complement update of the residual diagonal.
        #   d_new[i] = d[i] - |L[i, j]|²
        # Clamp at 0 to kill fp32 noise, then disable the picked row.
        d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
        idx = jnp.arange(M)
        pivot_mask = (idx == p)
        active = active & ~(take & pivot_mask)
        d = jnp.where(take & pivot_mask, minus_inf,
                      jnp.where(take, d_new, d))

        # Step 5: bookkeeping.
        done = done | (~take)
        rank = rank + take.astype(jnp.int32)

        return d, L, piv, active, done, rank

    d, L, piv, active, done, rank = lax.fori_loop(
        0, k_keep, body_fun, (d0, L0, piv0, active0, done0, rank0)
    )
    del active, done

    d_final = jnp.where(jnp.isfinite(d), d, 0.0)
    return piv, L, rank, d_final


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — end-to-end wrapper
# ═══════════════════════════════════════════════════════════════════════


def prune_candidates_by_pivoted_cholesky(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    cand_idx: np.ndarray,
    n_keep: int,
    *,
    n_val: int | None = None,
    n_cond: int | None = None,
    k_weights: np.ndarray | None = None,
    tol_rel: float = 1e-10,
    verbose: bool = True,
) -> tuple[np.ndarray, int, jnp.ndarray, jnp.ndarray]:
    """Full q=0 pruning pipeline: gather → Gram → pivoted Cholesky → pivots.

    Suggested oversampling per the md: ``cand_idx.shape[0] ≈ 1.5 · n_keep``
    or ``2 · n_keep``. This function does not pick the oversampling ratio;
    the caller decides how many candidates to pass in.

    Args:
        wfn: open WFNReader.
        sym: matching SymMaps (passed through to the gather step).
        cand_idx: (M, 3) int32 FFT-grid indices of candidate points. M is
            the oversampled count.
        n_keep: target number of pruned points (N_μ).
        n_val: number of valence bands to include. Defaults to
            ``wfn.nelec`` (one band per occupied spinor state).
        n_cond: number of conduction bands above N_val to include.
            Defaults to ``min(n_val, wfn.nbands - wfn.nelec)`` — a sensible
            matched window; pass an explicit value for production runs.
        k_weights: optional (nkpts,) real weights. Defaults to
            ``wfn.kweights`` if present else uniform.
        tol_rel: relative tolerance for pivot acceptance (see
            ``pivoted_cholesky_select``).
        verbose: print diagnostic summary (rank, residual decay).

    Returns:
        keep_idx: (rank, 3) int32 FFT-grid indices of the N_μ pruned
            points. If the numerical rank came out < n_keep, only
            ``rank`` rows are returned.
        rank: int number of actually accepted pivots.
        G: (M, M) complex Gram matrix (returned for inspection / caching).
        d_final: (M,) real final Schur-complement residual diagonal.
    """
    M = int(cand_idx.shape[0])
    if n_val is None:
        n_val = int(wfn.nelec)
    n_tot = int(wfn.nbands)
    if n_cond is None:
        n_cond = min(n_val, n_tot - n_val)
    if n_val + n_cond > n_tot:
        raise ValueError(
            f"wfn.nbands={n_tot} < n_val + n_cond = {n_val} + {n_cond}"
        )

    if k_weights is None:
        kw_arr = (np.asarray(wfn.kweights, dtype=np.float64)
                  if hasattr(wfn, 'kweights') else
                  np.ones(wfn.nkpts, dtype=np.float64) / wfn.nkpts)
    else:
        kw_arr = np.asarray(k_weights, dtype=np.float64)
    kw_j = jnp.asarray(kw_arr, dtype=jnp.float32)

    if verbose:
        print(f"[pivoted_cholesky] M={M} candidates, n_keep={n_keep}, "
              f"n_val={n_val}, n_cond={n_cond}, nk_irr={wfn.nkpts}")

    # Gather φ (valence, [0, n_val)) and ψ (conduction, [n_val, n_val+n_cond)).
    phi = gather_wfn_at_candidates(wfn, sym, cand_idx, 0, n_val)
    psi = gather_wfn_at_candidates(wfn, sym, cand_idx, n_val, n_val + n_cond)

    # Fold spinor into band so the Gram einsum sees (nk, nv_eff, M).
    phi_flat = _fold_spin_into_band(phi)
    psi_flat = _fold_spin_into_band(psi)

    G = build_candidate_gram_q0(phi_flat, psi_flat, k_weights=kw_j)
    G.block_until_ready()

    if verbose:
        diag = np.asarray(jnp.real(jnp.diag(G)))
        print(f"[pivoted_cholesky] G built, shape={G.shape}, "
              f"diag range [{diag.min():.3e}, {diag.max():.3e}]")

    piv, L, rank, d_final = pivoted_cholesky_select(
        G, n_keep, tol_rel=tol_rel, tol_abs=0.0
    )
    del L  # only useful if the caller needs the Cholesky factor
    rank = int(rank)
    piv_np = np.asarray(piv)

    if verbose:
        d_np = np.asarray(d_final)
        print(f"[pivoted_cholesky] rank={rank}/{n_keep}, "
              f"residual diag at exit: max={d_np.max():.3e}, "
              f"mean_inactive={d_np[d_np > 0].mean() if np.any(d_np > 0) else 0.0:.3e}")
        if rank < n_keep:
            print(f"  ⚠ rank-deficient ({rank} < {n_keep}): either the "
                  f"candidate pool is too small, or tol_rel={tol_rel} cut "
                  f"off the residual diagonal.")

    # piv may contain -1 past rank; slice.
    keep_piv = piv_np[:rank]
    keep_idx = np.asarray(cand_idx)[keep_piv]
    return keep_idx, rank, G, d_final


# ═══════════════════════════════════════════════════════════════════════
# Multi-device — WIP: sharded Gram build
# ═══════════════════════════════════════════════════════════════════════
#
# For large candidate pools (M ≳ 10⁴) the (M, M) complex Gram matrix no
# longer fits on a single A100 (M = 16384 ⇒ 2.1 GiB in complex64). The
# natural first step toward distributed pruning is to row-shard G on a
# 1-D mesh 'x' so each device stores a (M / n_dev, M) slab. The per-k
# matmul structure makes this trivial: each row-slab is produced by a
# local matmul between the full (nk, nv, M) wavefunction replica and the
# M_slab slice of column vectors — no cross-device communication during
# the build.
#
# The sharded pivoted-Cholesky SELECT path is not in this commit — it
# needs a small `psum`/`pmax` pattern per iteration (broadcast the pivot
# row L[p, :] from its owning shard, and a global max-index reduction
# over the residual diagonal). Coming in a follow-up. For now, callers
# who want the sharded Gram can row-gather it back via
# ``jax.device_put`` before calling the single-device
# ``pivoted_cholesky_select`` — only useful when the Gram fits on one
# device but the build fits better distributed.


def make_sharded_gram_q0(
    mesh: Mesh,
    M: int,
    *,
    enforce_hermitian: bool = True,
):
    """Build a jitted Gram-assembly closure over ``mesh`` ('x',).

    The returned function signature matches ``build_candidate_gram_q0``:
    it takes replicated (nk, nv, M) and (nk, nc, M) wavefunction tensors
    and a replicated (nk,) weight vector, and returns a row-sharded
    (M, M) complex Gram matrix with ``NamedSharding(mesh, P('x', None))``.

    The row shard is the natural output of ``φ_local.T @ conj(φ)`` when
    the left factor is sliced to M_slab columns: each device computes
    its own (M_slab, M) slab of both projectors, element-wise-multiplies,
    and accumulates over k.

    Args:
        mesh: 1-axis device mesh named 'x'. ``M`` must be divisible by
            ``mesh.shape['x']``.
        M: candidate count. Static.
        enforce_hermitian: if True, do a ``(G + G^H)/2`` symmetrization at
            the end. This is an *all-pairs* operation over the sharded
            axis — it requires an all-gather of G across the mesh and
            therefore breaks the sharding; the caller can defer this
            step until after the select if they want to avoid the gather.

    Returns:
        A callable ``(phi, psi, kw) -> G_row_sharded``.
    """
    if 'x' not in mesh.axis_names:
        raise ValueError(f"mesh must have an 'x' axis, got {mesh.axis_names}")
    n_dev = mesh.shape['x']
    if M % n_dev != 0:
        raise ValueError(f"M={M} must be divisible by mesh 'x' size {n_dev}")
    M_slab = M // n_dev

    rep = PartitionSpec()
    row_shard = PartitionSpec('x', None)

    in_specs = (rep, rep, rep)           # phi, psi, kw — all replicated
    out_specs = row_shard                # G row-sharded

    @partial(jax.jit, static_argnames=())
    def step(phi, psi, kw):
        def body_local(phi_full, psi_full, kw_full):
            # Pick out this shard's columns of the left factors. The
            # right factor stays the full M so the output has shape
            # (M_slab, M).
            my_idx = lax.axis_index('x')
            a_start = my_idx * M_slab
            phi_a_block = lax.dynamic_slice_in_dim(
                phi_full, a_start, M_slab, axis=2
            )   # (nk, nv, M_slab)
            psi_a_block = lax.dynamic_slice_in_dim(
                psi_full, a_start, M_slab, axis=2
            )   # (nk, nc, M_slab)

            nk = phi_full.shape[0]

            def k_body(k, G_slab):
                phi_k = phi_full[k]        # (nv, M)
                psi_k = psi_full[k]        # (nc, M)
                phi_k_block = phi_a_block[k]   # (nv, M_slab)
                psi_k_block = psi_a_block[k]   # (nc, M_slab)

                # P_v_slab[a, b] = Σ_v φ(a) φ*(b), a in local slab, b full.
                P_v_slab = phi_k_block.T @ jnp.conj(phi_k)           # (M_slab, M)
                # P̃_c_slab[a, b] = Σ_c ψ*(a) ψ(b)
                P_c_slab = jnp.conj(psi_k_block).T @ psi_k           # (M_slab, M)
                return G_slab + kw_full[k] * (P_v_slab * P_c_slab)

            G_slab = jnp.zeros((M_slab, M), dtype=phi_full.dtype)
            G_slab = lax.fori_loop(0, nk, k_body, G_slab)
            return G_slab

        G_sharded = shard_map(
            body_local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
            check_rep=False,
        )(phi, psi, kw)

        if enforce_hermitian:
            # Hermitian symmetrization requires full G on one device. For
            # now we pay the all-gather — acceptable while G still fits
            # post-gather on one device. For truly out-of-core G this
            # step should be deferred to the caller (and the select step
            # can tolerate small non-Hermiticity via its own clamp).
            G_full = jax.lax.with_sharding_constraint(
                G_sharded, NamedSharding(mesh, PartitionSpec()),
            )
            G_full = 0.5 * (G_full + jnp.conj(G_full.T))
            # Re-shard back so the caller sees the same layout.
            G_sharded = jax.lax.with_sharding_constraint(
                G_full, NamedSharding(mesh, row_shard),
            )
        return G_sharded

    return step
