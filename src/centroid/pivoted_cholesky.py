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
from jax.experimental import multihost_utils as _mh
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
        psi_cand: (nkpts, band_end - band_start, nspinor, M) complex128
            ψ_{n,k}(r̃_a), one slab per IBZ k-point. Kept in complex128 so
            the downstream Gram + pivoted-Cholesky select operate in fp64
            throughout (matches the gw_jax data path's precision).
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

    # IFFT in complex128, host-stage in complex128 — no fp32 downcast on
    # the way to the device. Matches load_wfns' precision.
    psi_out = np.zeros((wfn.nkpts, nb, nspinor, M), dtype=np.complex128)
    for ik in range(wfn.nkpts):
        gvecs_k = np.asarray(wfn.get_gvec_nk(ik))  # (ngk, 3)
        cnk = wfn.get_cnk_batch(ik, band_indices)  # (nb, nspinor, ngk)
        # Scatter → FFT box, iFFT, gather at M candidate points.
        box = np.zeros((nb, nspinor, nx, ny, nz), dtype=np.complex128)
        box[:, :, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = cnk
        psi_r = np.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
        psi_out[ik] = psi_r[..., ix, iy, iz]
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
        d_taken: (k_keep,) real — pivot value (residual diagonal) at the
            moment each pivot was accepted, in pivot order. Entries past
            ``rank`` are 0. Useful for diagnosing whether the chosen
            ``k_keep`` is too small (last entry still ≫ tol_rel × d0max
            ⇒ stopping early would have been premature) or too large
            (last entries near zero ⇒ already extracting noise).
        trR_over_trG: (k_keep + 1,) real — running ``tr(R_k) / tr(G)``,
            i.e. the candidate-pool relative Frobenius residual after
            each pivot. ``trR_over_trG[0] = 1`` (no pivots), then
            ``trR_over_trG[k+1]`` reflects the state after step k.
            Monotone non-increasing under exact arithmetic. Comparable
            across band windows / N_c (it's already normalized by the
            per-config trace), so this is the right curve to plot
            ``accuracy vs # pivots`` for ISDF point-set design.
    """
    M = G.shape[0]
    real_dtype = G.real.dtype
    eps = jnp.finfo(real_dtype).eps
    minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)

    diag0 = jnp.maximum(jnp.real(jnp.diag(G)), 0.0)
    d0max = jnp.max(diag0)
    # Total trace of G — invariant of the pivoting and the natural
    # normalizer for the running residual ‖G − L_k L_k^H‖²_F (more
    # precisely tr R_k, which is the candidate-pool relative Frobenius
    # mass not yet captured by k pivots).
    trG = jnp.sum(diag0)

    L0 = jnp.zeros((M, k_keep), dtype=G.dtype)
    piv0 = -jnp.ones((k_keep,), dtype=jnp.int32)
    d0 = diag0
    active0 = jnp.ones((M,), dtype=bool)
    done0 = jnp.array(False)
    rank0 = jnp.array(0, dtype=jnp.int32)
    d_taken0 = jnp.zeros((k_keep,), dtype=real_dtype)
    # trR_over_trG[j+1] = sum(d_after_step_j) / trG. Index 0 stores the
    # initial value (= 1.0); size is k_keep+1 so we get the full
    # trajectory through the last accepted pivot.
    trR_over_trG0 = jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0)

    col_ids = jnp.arange(k_keep)

    def body_fun(j, carry):
        d, L, piv, active, done, rank, d_taken, trR_over_trG = carry

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
        # Record the pivot value as picked (or leave the slot at 0 if no-op).
        d_taken = d_taken.at[j].set(
            jnp.where(take, pivot_val.astype(real_dtype), d_taken[j])
        )

        # Step 4: Schur-complement update of the residual diagonal.
        #   d_new[i] = d[i] - |L[i, j]|²
        # Clamp at 0 to kill fp64 noise. d_new[p] is naturally ≈ 0
        # (because |newcol[p]|² == pivot_val by the cleanup above), so
        # summing d_new gives the correct trace of the new Schur
        # complement R_k. We must use d_new HERE (before substituting
        # the -inf sentinel into d) so the trace sum stays meaningful.
        d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
        # trR / trG after this step. If take is False this iteration is
        # a no-op, so just propagate the previous value.
        trR_step = jnp.sum(d_new)
        trR_over_trG = trR_over_trG.at[j + 1].set(
            jnp.where(take, trR_step / trG, trR_over_trG[j])
        )
        idx = jnp.arange(M)
        pivot_mask = (idx == p)
        active = active & ~(take & pivot_mask)
        d = jnp.where(take & pivot_mask, minus_inf,
                      jnp.where(take, d_new, d))

        # Step 5: bookkeeping.
        done = done | (~take)
        rank = rank + take.astype(jnp.int32)

        return d, L, piv, active, done, rank, d_taken, trR_over_trG

    d, L, piv, active, done, rank, d_taken, trR_over_trG = lax.fori_loop(
        0, k_keep, body_fun,
        (d0, L0, piv0, active0, done0, rank0, d_taken0, trR_over_trG0),
    )
    del active, done

    d_final = jnp.where(jnp.isfinite(d), d, 0.0)
    return piv, L, rank, d_final, d_taken, trR_over_trG


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
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
    tol_rel: float = 1e-10,
    verbose: bool = True,
    mesh: Mesh | None = None,
    bispinor: bool = False,
) -> tuple[np.ndarray, int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        n_val: number of valence bands for the legacy v × c Gram.
            Used only if ``band_range_left`` is not given.
        n_cond: number of conduction bands above n_val for the legacy
            v × c Gram.
        band_range_left: explicit left window ``(b_start, b_end)``. In
            the gw_jax / ISDF convention this is ``(b0, b3)`` =
            "all val + sigma cond". Overrides (n_val, n_cond).
        band_range_right: explicit right window ``(b_start, b_end)``.
            In the gw_jax convention this is ``(b1, b4)`` = "sigma val +
            all cond". Overrides (n_val, n_cond).
        band_norms: optional (nbands,) pseudoband norms (e.g.
            ``wfn.band_norms``). Applied as ``ψ /= max(norm, 1.0)`` on
            both left and right windows before the pair density — same
            clamp recipe as ``isdf_fitting.fit_zeta_chunked_to_h5``.
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
    n_tot = int(wfn.nbands)
    asymmetric = (band_range_left is not None and band_range_right is not None)

    if not asymmetric:
        # Legacy (n_val, n_cond) path — left = (0, n_val), right = (n_val, n_val + n_cond).
        if n_val is None:
            n_val = int(wfn.nelec)
        if n_cond is None:
            n_cond = min(n_val, n_tot - n_val)
        if n_val + n_cond > n_tot:
            raise ValueError(
                f"wfn.nbands={n_tot} < n_val + n_cond = {n_val} + {n_cond}"
            )
        max_band = int(n_val) + int(n_cond)
    else:
        if band_range_left[1] > n_tot or band_range_right[1] > n_tot:
            raise ValueError(
                f"wfn.nbands={n_tot} < max(left={band_range_left[1]}, "
                f"right={band_range_right[1]})"
            )
        max_band = max(int(band_range_left[1]), int(band_range_right[1]))

    # Plane-wave-basis sanity check. For centroid pruning to be
    # meaningful, the pair-product space must be significantly smaller
    # than the full plane-wave basis; once we include > 50 % of the
    # available PW degrees of freedom, the candidate-vs-grid distinction
    # blurs and the user should be pruning the real-space grid directly.
    ngk_max = int(np.max(wfn.ngk)) if hasattr(wfn, 'ngk') else None
    nspinor = int(wfn.nspinor)
    if ngk_max is not None:
        npw_basis = ngk_max * nspinor  # size of the plane-wave basis per k
        if max_band > 0.5 * npw_basis:
            raise ValueError(
                f"Requested band window touches band {max_band}, which "
                f"exceeds 50 % of the plane-wave basis size "
                f"({0.5 * npw_basis:.0f} = 0.5 · ngk_max · nspinor = "
                f"0.5 · {ngk_max} · {nspinor}). Centroid pruning is "
                f"ill-posed in this regime — prune on the full real-space "
                f"grid directly instead."
            )

    # Decide which pipeline to use based on the mesh shape:
    #   • 2-D mesh with both 'x' and 'y'  → load_wfns-based pipeline:
    #         read_Gvecs_to_devices → get_sharded_wfns_centroids →
    #         compute_pair_density_spin_traced → compute_gram_q0_from_left_right
    #         (G produced as P('x','y'); reshard to P(('x','y'), None) for select)
    #   • 1-D mesh (just 'x')             → single-axis make_sharded_gram_q0
    #   • No mesh                         → single-device build_candidate_gram_q0
    # The 2-D path skips the host-side gather entirely and matches gw_jax's
    # ISDF data loading; the 1-D path is kept as a fallback (and compatibility
    # with earlier CLI invocations).
    is_2d_mesh = (mesh is not None
                  and 'x' in mesh.axis_names
                  and 'y' in mesh.axis_names)

    if not is_2d_mesh:
        if k_weights is None:
            kw_arr = (np.asarray(wfn.kweights, dtype=np.float64)
                      if hasattr(wfn, 'kweights') else
                      np.ones(wfn.nkpts, dtype=np.float64) / wfn.nkpts)
        else:
            kw_arr = np.asarray(k_weights, dtype=np.float64)
        kw_j = jnp.asarray(kw_arr, dtype=jnp.float64)

    if verbose:
        if asymmetric:
            window_tag = (f"left={band_range_left}, right={band_range_right}, "
                          f"norms={'on' if band_norms is not None else 'off'}")
        else:
            window_tag = f"n_val={n_val}, n_cond={n_cond}"
        if is_2d_mesh:
            print(f"[pivoted_cholesky] M={M} candidates, n_keep={n_keep}, "
                  f"{window_tag}, nk_tot={sym.nk_tot} "
                  f"(full-BZ via load_wfns on mesh "
                  f"x={mesh.shape['x']}, y={mesh.shape['y']})")
        else:
            print(f"[pivoted_cholesky] M={M} candidates, n_keep={n_keep}, "
                  f"{window_tag}, nk_irr={wfn.nkpts} "
                  f"(IBZ via host-gather)")

    if asymmetric and not is_2d_mesh:
        raise NotImplementedError(
            "Explicit (band_range_left, band_range_right) requires the "
            "2-D load_wfns pipeline; pass a 2-D mesh."
        )

    if is_2d_mesh:
        # --- 2-D pipeline: full-BZ load_wfns + gw_jax pair-density helpers ---
        G = build_gram_q0_via_loadwfns(
            wfn, sym, jnp.asarray(cand_idx),
            n_val=n_val, n_cond=n_cond,
            mesh_xy=mesh, bispinor=bispinor, verbose=verbose,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            band_norms=band_norms,
        )
        # Reshard the Gram output ('x','y') → combined-axis row shard
        # (('x','y'), None) so the existing sharded select has its
        # column-access pattern stay collective-free.
        if verbose:
            print(f"[pivoted_cholesky] resharding G P('x','y') → "
                  f"P(('x','y'), None) for select")
        G = jax.lax.with_sharding_constraint(
            G, NamedSharding(mesh, PartitionSpec(('x', 'y'), None)),
        )
    else:
        # --- Legacy 1-D / single-device path: host-gather + IBZ sum ---
        # Gather φ (valence, [0, n_val)) and ψ (conduction, [n_val, n_val+n_cond)).
        phi = gather_wfn_at_candidates(wfn, sym, cand_idx, 0, n_val)
        psi = gather_wfn_at_candidates(wfn, sym, cand_idx, n_val, n_val + n_cond)
        phi_flat = _fold_spin_into_band(phi)
        psi_flat = _fold_spin_into_band(psi)

        if mesh is not None:
            n_dev = mesh.shape['x']
            if M % n_dev != 0:
                if verbose:
                    print(f"[pivoted_cholesky] M={M} not divisible by mesh 'x' "
                          f"size {n_dev}; falling back to single-device.")
                mesh = None
            elif verbose:
                print(f"[pivoted_cholesky] sharded build+select on mesh 'x' "
                      f"of size {n_dev}")

        if mesh is not None:
            gram_step = make_sharded_gram_q0(mesh, M, enforce_hermitian=True)
            G = gram_step(phi_flat, psi_flat, kw_j)
        else:
            G = build_candidate_gram_q0(phi_flat, psi_flat, k_weights=kw_j)
    G.block_until_ready()

    if verbose:
        diag = np.asarray(jnp.real(jnp.diag(G)))
        print(f"[pivoted_cholesky] G built, shape={G.shape}, "
              f"diag range [{diag.min():.3e}, {diag.max():.3e}]")

    if mesh is not None:
        # Pick the select's axis to match G's current sharding: for the
        # 2-D pipeline we already with_sharding_constraint'd G to
        # P(('x','y'), None), so the select is also combined-axis.
        select_axis = ('x', 'y') if is_2d_mesh else 'x'
        select_step = make_sharded_pivoted_cholesky_select(
            mesh, M, n_keep, tol_rel=tol_rel, tol_abs=0.0,
            mesh_axis=select_axis,
        )
        piv, L, rank, d_final, d_taken, trR_over_trG = select_step(G)
    else:
        piv, L, rank, d_final, d_taken, trR_over_trG = pivoted_cholesky_select(
            G, n_keep, tol_rel=tol_rel, tol_abs=0.0
        )
    del L  # only useful if the caller needs the Cholesky factor
    rank = int(rank)
    piv_np = np.asarray(piv)
    d_taken_np = np.asarray(d_taken)
    trR_over_trG_np = np.asarray(trR_over_trG)
    # d_final is sharded along the select's mesh axis (`row_shard_1d`);
    # in multi-process JAX a bare ``np.asarray`` raises because remote
    # shards are non-addressable. Gather it once here so the host-side
    # diagnostics + the wrapper's d_final return are safe to read.
    if mesh is not None:
        d_final_np = np.asarray(_mh.process_allgather(d_final, tiled=True))
    else:
        d_final_np = np.asarray(d_final)

    if verbose:
        d_np = d_final_np
        leftover_max = float(d_np.max()) if d_np.size else 0.0
        leftover_mean = float(d_np[d_np > 0].mean()) if np.any(d_np > 0) else 0.0
        # Pivot-decay summary: first/last picked, and a couple of percentile
        # checkpoints. The right diagnostic is whether the LAST picked pivot
        # is much bigger than the biggest leftover (leftover_max). If yes,
        # we cut at a meaningful drop. If no, k_keep was about right or too
        # large.
        first = float(d_taken_np[0]) if rank > 0 else 0.0
        last = float(d_taken_np[rank - 1]) if rank > 0 else 0.0
        mid = float(d_taken_np[rank // 2]) if rank > 0 else 0.0
        print(f"[pivoted_cholesky] rank={rank}/{n_keep}")
        print(f"[pivoted_cholesky] picked-pivot residuals: "
              f"first={first:.3e}, mid={mid:.3e}, last={last:.3e}")
        # Relative-Frobenius accuracy at the same checkpoints.
        if rank > 0:
            print(f"[pivoted_cholesky] tr(R_k)/tr(G): "
                  f"first={trR_over_trG_np[1]:.3e}, "
                  f"mid={trR_over_trG_np[rank // 2 + 1]:.3e}, "
                  f"last={trR_over_trG_np[rank]:.3e}")
        print(f"[pivoted_cholesky] leftover residuals: "
              f"max={leftover_max:.3e}, mean(>0)={leftover_mean:.3e}")
        if last > 0 and leftover_max > 0:
            ratio = last / leftover_max
            if ratio < 2:
                print(f"  ⚠ last picked / biggest leftover = {ratio:.2f}: "
                      f"k_keep cuts in a region where picked and leftover "
                      f"residuals are similar — try increasing k_keep or "
                      f"shrinking the candidate pool.")
            elif ratio > 100:
                print(f"  ℹ last picked / biggest leftover = {ratio:.1f}×: "
                      f"clear cutoff; could likely lower k_keep.")
        if rank < n_keep:
            print(f"  ⚠ rank-deficient ({rank} < {n_keep}): either the "
                  f"candidate pool is too small, or tol_rel={tol_rel} cut "
                  f"off the residual diagonal.")

    # piv may contain -1 past rank; slice.
    keep_piv = piv_np[:rank]
    keep_idx = np.asarray(cand_idx)[keep_piv]
    return keep_idx, rank, G, d_final_np, d_taken_np, trR_over_trG_np


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
            my_idx = lax.axis_index('x')   # legacy 1-D 'x'-only path
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


# ═══════════════════════════════════════════════════════════════════════
# Multi-device — sharded pivoted-Cholesky select
# ═══════════════════════════════════════════════════════════════════════
# Companion to ``make_sharded_gram_q0``. Consumes a row-sharded
# G ∈ ℂ^(M×M) on a 1-D mesh 'x' (sharded on axis 0) and runs the same
# greedy pivoted-Cholesky select as the single-device version. Sharded
# along M: each device owns (M_slab, M) of G and (M_slab, k_keep) of L.
#
# Collectives per iteration (one per Lloyd-like step):
#
#   pmax(local_pv, 'x')       — 1 scalar: finds the global pivot value
#   pmax(-winner_p, 'x')      — 1 int32:  breaks ties by lowest device idx
#   psum(local_Lp, 'x')       — (k_keep,) array: broadcasts L[p, :] from
#                                its owning shard to every device
#
# Total comm per iter: O(k_keep). Total over k_keep iters: O(k_keep²).
# Matmul/Schur update are local — each device does L_slab @ (scalar)
# + elementwise ops on (M_slab,)- and (M_slab, k_keep)-shaped arrays.
#
# Column access `G[:, global_p]`: because G is ROW-sharded, each device's
# local slab already contains its portion of column p — no collective.
# This is why row-sharding is preferred over column-sharding for this
# algorithm.


def make_sharded_pivoted_cholesky_select(
    mesh: Mesh,
    M: int,
    k_keep: int,
    *,
    tol_rel: float = 1e-10,
    tol_abs: float = 0.0,
    mesh_axis: str | tuple[str, ...] = 'x',
):
    """Build a jitted sharded pivoted-Cholesky select closure over ``mesh``.

    The returned function signature matches ``pivoted_cholesky_select``
    up to sharding: it takes a row-sharded ``G`` (sharded on
    ``mesh_axis`` along its first axis) and returns piv (replicated),
    L (row-sharded same as G), rank (replicated), d_final (row-sharded),
    d_taken (replicated).

    Args:
        mesh: device mesh containing every axis in ``mesh_axis``.
        M: total candidate count. Static.
        k_keep: target pivot count. Static.
        tol_rel, tol_abs: stopping tolerances; see
            ``pivoted_cholesky_select`` for semantics. Static.
        mesh_axis: axis (or tuple of axes for combined-axis sharding) on
            ``mesh`` along which the row dim of G is sharded. Default
            ``'x'`` matches the legacy 1-D path. Pass e.g. ``('x', 'y')``
            after a ``with_sharding_constraint`` from a 2-D-built G to a
            flat row-sharded layout (the natural step before pivoting).

    Returns:
        A callable ``(G_row_sharded,) -> (piv, L, rank, d_final, d_taken)``.
    """
    axis_names = (mesh_axis,) if isinstance(mesh_axis, str) else tuple(mesh_axis)
    for ax in axis_names:
        if ax not in mesh.axis_names:
            raise ValueError(
                f"mesh_axis {mesh_axis!r} references '{ax}' not in "
                f"mesh axes {mesh.axis_names}"
            )
    n_dev = 1
    for a in axis_names:
        n_dev *= mesh.shape[a]
    if M % n_dev != 0:
        raise ValueError(f"M={M} must be divisible by product of mesh axes "
                         f"{mesh_axis} (= {n_dev})")
    M_slab = M // n_dev

    row_shard = PartitionSpec(mesh_axis, None)
    row_shard_1d = PartitionSpec(mesh_axis)
    rep = PartitionSpec()

    in_specs = (row_shard,)
    # outputs: piv, L, rank, d_final, d_taken, trR_over_trG
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep)

    @partial(jax.jit, static_argnames=())
    def step(G):
        def body_local(G_slab):
            # G_slab: (M_slab, M) on each device.
            real_dtype = G_slab.real.dtype
            eps = jnp.finfo(real_dtype).eps
            minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
            my_idx = lax.axis_index(mesh_axis)

            # Initial local diagonal: each device owns rows
            # [my_idx*M_slab, (my_idx+1)*M_slab), and the diagonal of G
            # falls at column index = row index. So
            #   G_slab[i, my_idx*M_slab + i]  for  i = 0..M_slab-1
            # extracts the local diagonal.
            my_offset = my_idx * M_slab
            col_ids_local = my_offset + jnp.arange(M_slab)
            local_diag = jnp.real(
                G_slab[jnp.arange(M_slab), col_ids_local]
            )
            local_diag = jnp.maximum(local_diag, 0.0)

            # d0max is a global statistic — reduce across shards.
            d0max = lax.pmax(jnp.max(local_diag), axis_name=mesh_axis)
            # tr(G) is also global — one extra psum, replicated everywhere.
            trG = lax.psum(jnp.sum(local_diag), axis_name=mesh_axis)

            L_slab = jnp.zeros((M_slab, k_keep), dtype=G_slab.dtype)
            piv = -jnp.ones((k_keep,), dtype=jnp.int32)
            d_slab = local_diag
            active_slab = jnp.ones((M_slab,), dtype=bool)
            done = jnp.array(False)
            rank = jnp.array(0, dtype=jnp.int32)
            d_taken = jnp.zeros((k_keep,), dtype=real_dtype)
            # See single-device select for shape rationale (k_keep + 1).
            trR_over_trG = jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0)
            col_ids_k = jnp.arange(k_keep)

            def body(j, carry):
                d, L, piv, active, done, rank, d_taken, trR_over_trG = carry

                # Step 1: pick the global pivot.
                # Per-device argmax over active rows.
                masked_d = jnp.where(active, d, minus_inf)
                local_p_idx = jnp.argmax(masked_d)                   # [0, M_slab)
                local_pv = masked_d[local_p_idx]                     # scalar
                local_global_p = (my_idx * M_slab + local_p_idx).astype(jnp.int32)

                # Reduce to find the global pivot value.
                global_pv = lax.pmax(local_pv, mesh_axis)
                # Tie-break: among devices that match the global max, take
                # the one with the smallest global_p. Non-winners contribute
                # a sentinel (INT_MAX) so the min is over just the winners.
                i_am_winner = local_pv >= global_pv
                winner_p = jnp.where(
                    i_am_winner, local_global_p, jnp.int32(2**30)
                )
                global_p = -lax.pmax(-winner_p, mesh_axis)           # min over winners
                pivot_val = global_pv

                take = (~done) & (pivot_val > tol_abs) & (pivot_val > tol_rel * d0max)

                # Step 2: compute the new Cholesky column, L[:, j].
                # gcol: column p of G. Each device's local rows of that
                # column are already in G_slab — column access from a
                # row-sharded matrix is collective-free.
                gcol_slab = G_slab[:, global_p]                       # (M_slab,)

                # L[p, :]: broadcast from the owning shard via psum-with-mask.
                pivot_owner_dev = global_p // M_slab
                my_has_p = (pivot_owner_dev == my_idx)
                local_p_rel = global_p - my_idx * M_slab
                safe_idx = jnp.clip(local_p_rel, 0, M_slab - 1)
                local_Lp = L[safe_idx, :]                             # (k_keep,) — may be wrong on non-owners
                local_Lp = jnp.where(
                    my_has_p, local_Lp, jnp.zeros_like(local_Lp)
                )
                L_p = lax.psum(local_Lp, mesh_axis)                  # (k_keep,) replicated

                prev_mask = (col_ids_k < j).astype(G_slab.dtype)
                corr_slab = L @ (jnp.conj(L_p) * prev_mask)           # (M_slab,)

                denom = jnp.where(take, jnp.sqrt(jnp.maximum(pivot_val, eps)), 1.0)
                newcol = (gcol_slab - corr_slab) / denom              # (M_slab,)
                newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))
                # Numerical cleanup of the pivot entry (only on owner).
                fix_row_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                newcol = jnp.where(
                    fix_row_mask & take, denom.astype(G_slab.dtype), newcol
                )

                # Step 3: write the new column locally. Piv and d_taken
                # are replicated — every device writes the same value.
                oldcol = L[:, j]
                L = L.at[:, j].set(jnp.where(take, newcol, oldcol))
                piv = piv.at[j].set(
                    jnp.where(take, global_p, piv[j])
                )
                d_taken = d_taken.at[j].set(
                    jnp.where(take, pivot_val, d_taken[j])
                )

                # Step 4: Schur-complement update of local d.
                d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
                # Trace of the new Schur complement: sum over all M rows
                # of d_new — psum across shards. d_new[p] is naturally ≈ 0
                # by the pivot cleanup, so this is exactly tr(R_k).
                trR_step = lax.psum(jnp.sum(d_new), axis_name=mesh_axis)
                trR_over_trG = trR_over_trG.at[j + 1].set(
                    jnp.where(take, trR_step / trG, trR_over_trG[j])
                )
                pivot_row_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                active = active & ~(take & pivot_row_mask)
                d = jnp.where(
                    take & pivot_row_mask, minus_inf,
                    jnp.where(take, d_new, d)
                )

                done = done | (~take)
                rank = rank + take.astype(jnp.int32)

                return d, L, piv, active, done, rank, d_taken, trR_over_trG

            (d_final, L_out, piv_out, active_out, done_out, rank_out,
             d_taken_out, trR_over_trG_out) = lax.fori_loop(
                0, k_keep, body,
                (d_slab, L_slab, piv, active_slab, done, rank, d_taken,
                 trR_over_trG),
            )
            del active_out, done_out

            d_final = jnp.where(jnp.isfinite(d_final), d_final, 0.0)
            return (piv_out, L_out, rank_out, d_final, d_taken_out,
                    trR_over_trG_out)

        return shard_map(
            body_local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
            check_rep=False,
        )(G)

    return step


# ═══════════════════════════════════════════════════════════════════════
# Full 2-D Gram pipeline: load_wfns → pair density → q=0 Gram
# ═══════════════════════════════════════════════════════════════════════
#
# Uses the same data-loading path as the gw_jax ISDF fit:
#
#   read_Gvecs_to_devices(...)                       — full-BZ G-space wfns
#      ↓
#   get_sharded_wfns_centroids(...)                  — iFFT + gather at
#                                                      candidate points;
#                                                      returns psi_rmu_Y and
#                                                      psi_rmuT_X (the latter
#                                                      already conjugated)
#      ↓
#   compute_pair_density_spin_traced(psi_rmuT_X, psi_rmu_Y, mesh)
#      ↓    P_k[mu_X, nu_Y] = Σ_{n,s} ψ*(μ) ψ(ν)        (gw_jax convention)
#
# Called once for valence (→ P_v_k) and once for conduction (→ P_c_k). Then
# at q=0:
#
#   G[mu_X, nu_Y] = Σ_k w_k · conj(P_v_k) · P_c_k
#                = common.isdf_fitting.compute_gram_q0_from_left_right(
#                      P_v_k, P_c_k, k_weights, mesh
#                  )
#
# The conj() on P_v_k flips it from gw_jax's Σ_v φ*(μ)φ(ν) to the
# valence-projector form Σ_v φ(a)φ*(b) the Gram definition needs.
#
# Uses full-BZ unfold with uniform k-weights = 1/nk_tot (read_Gvecs_to_devices
# unfolds symmetry, so IBZ-weighted IBZ data are not the inputs). This is the
# correct convention to match gw_jax's pair-density pipeline exactly.


def build_gram_q0_via_loadwfns(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    cand_idx: jnp.ndarray,
    n_val: int | None = None,
    n_cond: int | None = None,
    mesh_xy: Mesh | None = None,
    *,
    bispinor: bool = False,
    verbose: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
) -> jnp.ndarray:
    """Build the q=0 candidate Gram on a 2-D mesh using gw_jax's data path.

    Two call modes:

    * Simple ``(n_val, n_cond)`` (legacy): left window = ``(0, n_val)``,
      right window = ``(n_val, n_val + n_cond)``. This is the literal
      valence × conduction pair-product Gram used in the original assay.

    * Explicit ``(band_range_left, band_range_right)`` (gw_jax / ISDF
      convention): left = ``(b0, b3)`` = "all val + sigma cond", right =
      ``(b1, b4)`` = "sigma val + all cond". Matches the windowing used
      by ``gw_init.fit_zeta`` → ``isdf_fitting.fit_zeta_chunked_to_h5``.
      Passing ``band_norms`` additionally applies the pseudoband
      normalization ``ψ /= max(norm, 1.0)`` on both left and right
      (same clamp recipe as ``isdf_fitting.py:838-847``).

    Full-BZ unfold: one ``load_wfns.read_Gvecs_to_devices`` per window,
    ``get_sharded_wfns_centroids`` at the candidate indices, sharded
    pair densities via ``compute_pair_density_spin_traced``, combined
    with the q=0 sum via ``compute_gram_q0_from_left_right``. k-weights
    are uniform ``1 / nk_tot`` because we've unfolded.

    Args:
        wfn: open WFNReader.
        sym: matching SymMaps.
        cand_idx: (M, 3) int32 FFT-grid indices of candidate points.
            Must be a ``jnp.ndarray`` (will be ``jnp.asarray``-ed if not).
        n_val: valence-window size for the legacy mode (see above).
            Required when ``band_range_left`` is not given.
        n_cond: conduction-window size for the legacy mode. Required
            when ``band_range_left`` is not given.
        mesh_xy: 2-D device mesh with axes ``'x'`` and ``'y'``. (Other
            axis names work too, as long as both are present; the pair
            density / Gram helpers hard-code the axis names ``'x'`` and
            ``'y'`` at present — we follow that convention.)
        bispinor: if True, upcast the spin structure to 4 components
            (matches gw_jax's bispinor mode). Default False.
        verbose: print progress lines.
        band_range_left: optional explicit left window (start, end).
            When given, takes precedence over (n_val, n_cond).
        band_range_right: optional explicit right window.
        band_norms: optional (nbands,) array of band norms
            (``wfn.band_norms``) for pseudoband reweighting. When given,
            applied to both left and right ψ via
            ``ψ /= max(norm_slice, 1.0)`` before the pair-density
            einsum.

    Returns:
        G: (M, M) complex, sharded ``P('x','y')`` on the mesh — ready to
           be reshard-constrained to a 1-D row-shard for the select
           stage.
    """
    # Lazy imports — these modules pull in the full gw_jax dep chain and
    # we don't want to charge the single-device prune path for it.
    from common.meta import Meta
    from common.load_wfns import (
        read_Gvecs_to_devices,
        get_sharded_wfns_centroids,
    )
    from common.isdf_fitting import (
        compute_pair_density_spin_traced,
        compute_gram_q0_from_left_right,
    )

    # Resolve windows.
    if band_range_left is None or band_range_right is None:
        if n_val is None or n_cond is None:
            raise ValueError(
                "Must supply either (n_val, n_cond) or "
                "(band_range_left, band_range_right)"
            )
        left_range = (0, int(n_val))
        right_range = (int(n_val), int(n_val) + int(n_cond))
    else:
        left_range = (int(band_range_left[0]), int(band_range_left[1]))
        right_range = (int(band_range_right[0]), int(band_range_right[1]))

    nb_left = left_range[1] - left_range[0]
    nb_right = right_range[1] - right_range[0]
    if nb_left <= 0 or nb_right <= 0:
        raise ValueError(
            f"Empty band window: left={left_range} right={right_range}"
        )

    # Meta's nband must cover whichever of left/right reaches higher.
    max_band = max(left_range[1], right_range[1])
    # Keep Meta.b0..b4 consistent with the *legacy* nval/ncond semantics
    # when the caller passed those; otherwise use (max_band, max_band) so
    # the metadata bounds don't constrain anything downstream.
    meta_nval = int(n_val) if n_val is not None else nb_left
    meta_ncond = int(n_cond) if n_cond is not None else max(1, max_band - meta_nval)

    M = int(cand_idx.shape[0])
    cand_idx = jnp.asarray(cand_idx, dtype=jnp.int64)

    meta = Meta.from_system(
        wfn, sym,
        nval=meta_nval, ncond=meta_ncond,
        nband=max_band,
        n_rmu=M,
        bispinor=bispinor,
    )

    kw = jnp.ones((sym.nk_tot,), dtype=jnp.float64) / float(sym.nk_tot)
    kgrid = np.asarray(wfn.kgrid, dtype=np.float64)
    kvecs_frac = np.asarray(sym.kvecs_asints) / kgrid[None, :]

    # Optional pseudoband norms — same clamp recipe as isdf_fitting.
    if band_norms is not None:
        band_norms_np = np.asarray(band_norms, dtype=np.float64)
        if band_norms_np.shape[0] < max_band:
            raise ValueError(
                f"band_norms has {band_norms_np.shape[0]} entries but "
                f"the left/right windows touch band {max_band}"
            )
        norms_l = np.maximum(
            band_norms_np[left_range[0]:left_range[1]], 1.0,
        )
        norms_r = np.maximum(
            band_norms_np[right_range[0]:right_range[1]], 1.0,
        )
        norms_l_j = jnp.asarray(norms_l, dtype=jnp.float64)
        norms_r_j = jnp.asarray(norms_r, dtype=jnp.float64)
    else:
        norms_l_j = None
        norms_r_j = None

    if verbose:
        print(f"[pivoted_cholesky] 2-D Gram build via load_wfns: "
              f"nk_tot={sym.nk_tot}, left={left_range} (nb={nb_left}), "
              f"right={right_range} (nb={nb_right}), M={M}, "
              f"norms={'on' if band_norms is not None else 'off'}")

    # ---- Left window ----
    psi_G_l, _ = read_Gvecs_to_devices(
        wfn, sym, left_range, meta, bispinor, mesh_xy,
    )
    psi_l_rmu_Y, psi_l_rmuT_X = get_sharded_wfns_centroids(
        psi_G_l, meta, cand_idx, kvecs_frac, mesh_xy, left_range,
    )
    if norms_l_j is not None:
        # Y shape (nk, nb, ns, n_rmu); X shape (nk, n_rmu, nb, ns)
        psi_l_rmu_Y = psi_l_rmu_Y / norms_l_j[None, :, None, None]
        psi_l_rmuT_X = psi_l_rmuT_X / norms_l_j[None, None, :, None]
    P_l_k = compute_pair_density_spin_traced(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
    del psi_G_l, psi_l_rmu_Y, psi_l_rmuT_X

    # ---- Right window ----
    psi_G_r, _ = read_Gvecs_to_devices(
        wfn, sym, right_range, meta, bispinor, mesh_xy,
    )
    psi_r_rmu_Y, psi_r_rmuT_X = get_sharded_wfns_centroids(
        psi_G_r, meta, cand_idx, kvecs_frac, mesh_xy, right_range,
    )
    if norms_r_j is not None:
        psi_r_rmu_Y = psi_r_rmu_Y / norms_r_j[None, :, None, None]
        psi_r_rmuT_X = psi_r_rmuT_X / norms_r_j[None, None, :, None]
    P_r_k = compute_pair_density_spin_traced(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
    del psi_G_r, psi_r_rmu_Y, psi_r_rmuT_X

    # ---- q=0 Gram: sum_k w_k · conj(P_l_k) · P_r_k ----
    G = compute_gram_q0_from_left_right(P_l_k, P_r_k, kw, mesh_xy)
    G.block_until_ready()
    return G
