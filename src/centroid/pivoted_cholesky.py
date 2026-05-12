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

    pair_density                      ←→  per-k open-spin P^{(v/c)}_{αβ}(a,b)
                                          (rank-5; same einsum at candidates
                                          r̃_a not chosen r_μ)
    c_q_from_pair                     ←→  k→q FFT of the cross-product
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

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, timing


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
    from file_io.wfn_loader import WfnLoader
    from common.wfn_transforms import to_rmu

    # Reduce ``cand_idx`` mod fft_grid to match the convention to_rmu
    # expects (FFT-grid indices in ``[0, fft_grid[a])``).
    nx, ny, nz = (int(x) for x in wfn.fft_grid)
    cand_mod = np.stack([
        np.asarray(cand_idx[:, 0], dtype=np.int64) % nx,
        np.asarray(cand_idx[:, 1], dtype=np.int64) % ny,
        np.asarray(cand_idx[:, 2], dtype=np.int64) % nz,
    ], axis=-1).astype(np.int32)

    # WfnLoader (eager) + to_rmu: IBZ raw → IFFT → gather at candidates.
    # ``norm='ortho'`` matches the legacy convention (1/√N on both
    # directions); pivoted-Cholesky selection is scale-invariant
    # but we preserve byte-for-byte numerics anyway.
    with WfnLoader(wfn._filename) as loader:
        psi = loader.load(bands=(band_start, band_end), k="ibz")
        return to_rmu(psi, loader.box_index(k="ibz"), loader.fft_grid,
                      cand_mod, norm="ortho")


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
    orbit_id: jnp.ndarray | None = None,
):
    """Greedy pivoted Cholesky on an Hermitian PSD ``G``. Always runs k_keep
    iterations. Returns ``(piv, L, rank, d_final, d_taken, trR_over_trG)``.

    When ``orbit_id`` is given (shape ``(M,)`` int), each pivot iteration
    marks the **whole orbit** of the picked point as inactive — i.e. one
    pivot per orbit. With a sym-invariant Gram (e.g. ρ-symmetric ISDF
    candidate Gram), all orbit members of the picked pivot have the same
    residual diagonal and the column update on any one of them is, by
    symmetry, the optimal full-orbit removal. The caller unfolds picked
    pivots through their orbits at output time to recover the full
    centroid set.
    """
    M = G.shape[0]
    real_dtype = G.real.dtype
    eps = jnp.finfo(real_dtype).eps
    minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
    if orbit_id is None:
        orbit_id = jnp.arange(M, dtype=jnp.int32)         # each point its own orbit

    diag0 = jnp.maximum(jnp.real(jnp.diag(G)), 0.0)
    trG = jnp.sum(diag0)

    init = (
        diag0,                                                       # d
        jnp.zeros((M, k_keep), dtype=G.dtype),                       # L
        -jnp.ones((k_keep,), dtype=jnp.int32),                       # piv
        jnp.ones((M,), dtype=bool),                                  # active
        jnp.zeros((k_keep,), dtype=real_dtype),                      # d_taken
        jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0),   # trR/trG
    )
    col_ids = jnp.arange(k_keep)

    def body(j, carry):
        d, L, piv, active, d_taken, trR_over_trG = carry

        masked_d = jnp.where(active, d, minus_inf)
        p = jnp.argmax(masked_d)
        pivot_val = jnp.maximum(masked_d[p], eps)         # eps for div-safety

        # L[:, j] = (G[:, p] - Σ_{i<j} L[:, i] · conj(L[p, i])) / sqrt(d[p])
        prev_mask = (col_ids < j).astype(G.dtype)
        corr = L @ (jnp.conj(L[p, :]) * prev_mask)
        denom = jnp.sqrt(pivot_val)
        newcol = (G[:, p] - corr) / denom
        # Pivot entry exactly sqrt(d[p]) — kills rounding drift.
        newcol = newcol.at[p].set(denom.astype(G.dtype))

        L = L.at[:, j].set(newcol)
        piv = piv.at[j].set(p.astype(jnp.int32))
        d_taken = d_taken.at[j].set(pivot_val.astype(real_dtype))

        # Schur-complement update; d_new[p] ≈ 0 by the cleanup above.
        d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
        trR_over_trG = trR_over_trG.at[j + 1].set(jnp.sum(d_new) / trG)
        # Mark p (or its whole orbit, if orbit_id was provided) inactive.
        kill_mask = orbit_id == orbit_id[p]
        d = jnp.where(kill_mask, minus_inf, d_new)
        active = active & ~kill_mask

        return d, L, piv, active, d_taken, trR_over_trG

    d, L, piv, _, d_taken, trR_over_trG = lax.fori_loop(0, k_keep, body, init)
    d_final = jnp.where(jnp.isfinite(d), d, 0.0)
    # Effective rank = #pivots above a fp-noise floor relative to ‖G‖.
    # The algorithm always runs k_keep iterations (residual decay is smooth
    # in the production use case); rank is only reported so callers can
    # detect rank-deficient synthetic inputs.
    floor = jnp.sqrt(eps) * jnp.max(jnp.real(jnp.diag(G)))
    rank = jnp.sum(d_taken > floor).astype(jnp.int32)
    # Zero out post-rank entries so callers can rely on d_taken[rank:] == 0.
    d_taken = jnp.where(jnp.arange(k_keep) < rank, d_taken, 0.0)
    return piv, L, rank, d_final, d_taken, trR_over_trG




# ═══════════════════════════════════════════════════════════════════════
# Step 4 — end-to-end wrapper
# ═══════════════════════════════════════════════════════════════════════


def prune_candidates_by_pivoted_cholesky(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    cand_idx: np.ndarray,
    n_keep: int,
    mesh: Mesh,
    *,
    n_val: int | None = None,
    n_cond: int | None = None,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
    verbose: bool = True,
    bispinor: bool = False,
    orbit_id: np.ndarray | None = None,
    use_phdf5: bool = False,
):
    """End-to-end pruning: gather wfns → Gram → pivoted Cholesky → keep.

    Requires a 2-D mesh ``('x', 'y')`` (single-device callers pass a 1×1
    mesh — same shape gw_jax uses). Wavefunction loading goes through
    ``load_centroids_band_chunked`` so the prune path is agnostic to which
    G-space backend (WFNReader / phdf5 / future jax-multihost) is in use.

    When ``orbit_id`` is provided (one int per candidate, equal for sym-
    equivalent candidates), PC picks one pivot per orbit and the returned
    ``keep_idx`` is the union of orbits of the picked pivots — guaranteed
    orbit-closed under the sym group used to assign ``orbit_id``. In that
    mode ``n_keep`` counts ORBITS (final unfolded centroid count is
    ``Σ orbit_size`` for picked orbits).

    Returns ``(keep_idx, rank, G, d_final, d_taken, trR_over_trG)``.
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

    if not ('x' in mesh.axis_names and 'y' in mesh.axis_names):
        raise ValueError(
            f"prune_candidates_by_pivoted_cholesky requires a 2-D mesh "
            f"with axes ('x', 'y'); got {mesh.axis_names}. Build the mesh "
            f"the same way gw_jax does (single-device → 1×1)."
        )

    # The sharded select kernel requires M to be divisible by the product
    # of the mesh axis sizes (each shard owns M/n_dev rows). Orbit-unfold
    # counts can land on awkward M (special-position orbits don't all have
    # size n_sym), so check up-front and give a hint instead of letting
    # ``make_sharded_pivoted_cholesky_select`` fail with a cryptic message.
    n_dev = int(mesh.shape['x']) * int(mesh.shape['y'])
    if M % n_dev != 0:
        raise ValueError(
            f"M={M} (number of candidates) must be divisible by the "
            f"product of mesh axes 'x' and 'y' (= {n_dev}). The sharded "
            f"pivoted-Cholesky select kernel splits M evenly across "
            f"shards. Either drop the last {M % n_dev} candidate(s) before "
            f"calling this function, run on a mesh size that divides M, "
            f"or pass ``--no-shard`` to use a single-device 1×1 mesh."
        )

    if verbose:
        window_tag = (f"left={band_range_left}, right={band_range_right}, "
                      f"norms={'on' if band_norms is not None else 'off'}"
                      if asymmetric else f"n_val={n_val}, n_cond={n_cond}")
        print(f"[pivoted_cholesky] M={M}, n_keep={n_keep}, {window_tag} "
              f"(load_wfns 2-D, mesh axes {mesh.axis_names})")

    with timing.section("prune.gram"):
        G = build_gram_q0_via_loadwfns(
            wfn, sym, jnp.asarray(cand_idx),
            n_val=n_val, n_cond=n_cond,
            mesh_xy=mesh, bispinor=bispinor, verbose=verbose,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            band_norms=band_norms,
            use_phdf5=use_phdf5,
        )
        # Reshard ('x','y') → row-sharded for the column-major pivot scan.
        G = jax.lax.with_sharding_constraint(
            G, NamedSharding(mesh, PartitionSpec(('x', 'y'), None)),
        )
        G.block_until_ready()
    select_axis = ('x', 'y')

    if verbose:
        diag = jnp.real(jnp.diag(G))
        print(f"[pivoted_cholesky] G built, shape={G.shape}, "
              f"diag range [{float(diag.min()):.3e}, {float(diag.max()):.3e}]")

    # Run select on the row-sharded Gram. Orbit-aware mode passes orbit_id
    # row-sharded the same way as G; the body marks the whole orbit
    # inactive after each pivot pick (orbit_id of the pivot is broadcast
    # via psum-with-mask, same idiom as the L[p, :] broadcast).
    with timing.section("prune.select"):
        select_step = make_sharded_pivoted_cholesky_select(
            mesh, M, n_keep, mesh_axis=select_axis,
        )
        if orbit_id is None:
            piv, L, rank, d_final, d_taken, trR_over_trG = select_step(G)
        else:
            orbit_id_jax = jax.device_put(
                jnp.asarray(orbit_id, dtype=jnp.int32),
                NamedSharding(mesh, PartitionSpec(select_axis)),
            )
            piv, L, rank, d_final, d_taken, trR_over_trG = select_step(G, orbit_id_jax)
        piv.block_until_ready()
    del L

    if verbose:
        print(f"[pivoted_cholesky] picked-pivot residuals: "
              f"first={float(d_taken[0]):.3e}, "
              f"mid={float(d_taken[n_keep // 2]):.3e}, "
              f"last={float(d_taken[-1]):.3e}")
        print(f"[pivoted_cholesky] tr(R_k)/tr(G): "
              f"first={float(trR_over_trG[1]):.3e}, "
              f"mid={float(trR_over_trG[n_keep // 2 + 1]):.3e}, "
              f"last={float(trR_over_trG[n_keep]):.3e}")

    piv_np = np.asarray(piv)
    if orbit_id is None:
        keep_idx = np.asarray(cand_idx)[piv_np]
    else:
        # Unfold: kept = union of orbits of picked pivots.
        orbit_id_np = np.asarray(orbit_id)
        picked_orbits = orbit_id_np[piv_np]
        in_kept = np.isin(orbit_id_np, picked_orbits)
        keep_idx = np.asarray(cand_idx)[in_kept]
        if verbose:
            print(f"[pivoted_cholesky] orbit-aware: {len(piv_np)} orbits picked "
                  f"→ {len(keep_idx)} unfolded centroids (orbit-closed)")
    d_final_np = np.asarray(_mh.process_allgather(d_final, tiled=True))
    return keep_idx, int(rank), G, d_final_np, np.asarray(d_taken), np.asarray(trR_over_trG)


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
    mesh_axis: str | tuple[str, ...] = 'x',
):
    """Sharded pivoted-Cholesky select on a row-sharded Gram. Always runs
    k_keep iterations (no tolerance early-stop). Returns the same tuple as
    ``pivoted_cholesky_select``: ``(piv, L, rank, d_final, d_taken,
    trR_over_trG)`` with shardings (replicated, row-sharded, replicated,
    row-sharded-1d, replicated, replicated)."""
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

    # Two input layouts: G alone (no orbit) or (G, orbit_id) with orbit_id
    # row-sharded the same way as G's row dim.
    in_specs_no_orbit = (row_shard,)
    in_specs_orbit    = (row_shard, row_shard_1d)
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep)

    @jax.jit
    def step(G, orbit_id=None):
        def body_local(G_slab, orbit_id_slab=None):
            real_dtype = G_slab.real.dtype
            eps = jnp.finfo(real_dtype).eps
            minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
            my_idx = lax.axis_index(mesh_axis)

            # Local diagonal of G: each device owns rows [my_idx*M_slab,
            # (my_idx+1)*M_slab); the diag entry sits at col == row.
            col_ids_local = my_idx * M_slab + jnp.arange(M_slab)
            local_diag = jnp.maximum(
                jnp.real(G_slab[jnp.arange(M_slab), col_ids_local]), 0.0,
            )
            trG = lax.psum(jnp.sum(local_diag), axis_name=mesh_axis)
            col_ids_k = jnp.arange(k_keep)

            init = (
                local_diag,                                              # d_slab
                jnp.zeros((M_slab, k_keep), dtype=G_slab.dtype),         # L_slab
                -jnp.ones((k_keep,), dtype=jnp.int32),                   # piv
                jnp.ones((M_slab,), dtype=bool),                         # active
                jnp.zeros((k_keep,), dtype=real_dtype),                  # d_taken
                jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0),
            )

            def body(j, carry):
                d, L, piv, active, d_taken, trR_over_trG = carry

                # Pick global pivot: per-device argmax then pmax + tie-break
                # to lowest global index.
                masked_d = jnp.where(active, d, minus_inf)
                local_p_idx = jnp.argmax(masked_d)
                local_pv = masked_d[local_p_idx]
                global_pv = lax.pmax(local_pv, mesh_axis)
                local_global_p = (my_idx * M_slab + local_p_idx).astype(jnp.int32)
                winner_p = jnp.where(
                    local_pv >= global_pv, local_global_p, jnp.int32(2**30),
                )
                global_p = -lax.pmax(-winner_p, mesh_axis)
                pivot_val = jnp.maximum(global_pv, eps)

                # Column p of G (no collective: G is row-sharded).
                gcol_slab = G_slab[:, global_p]

                # Row p of L: broadcast from owning shard via masked psum.
                my_has_p = (global_p // M_slab == my_idx)
                local_p_rel = global_p - my_idx * M_slab
                safe_idx = jnp.clip(local_p_rel, 0, M_slab - 1)
                local_Lp = jnp.where(
                    my_has_p, L[safe_idx, :], jnp.zeros_like(L[safe_idx, :]),
                )
                L_p = lax.psum(local_Lp, mesh_axis)

                # New column.
                prev_mask = (col_ids_k < j).astype(G_slab.dtype)
                corr = L @ (jnp.conj(L_p) * prev_mask)
                denom = jnp.sqrt(pivot_val)
                newcol = (gcol_slab - corr) / denom
                # Pivot-row entry exactly sqrt(d[p]), only on the owner.
                fix_row_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                newcol = jnp.where(fix_row_mask, denom.astype(G_slab.dtype), newcol)

                L = L.at[:, j].set(newcol)
                piv = piv.at[j].set(global_p)
                d_taken = d_taken.at[j].set(pivot_val)

                # Schur update; mark p (or its whole orbit) inactive.
                d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
                trR_over_trG = trR_over_trG.at[j + 1].set(
                    lax.psum(jnp.sum(d_new), axis_name=mesh_axis) / trG,
                )
                if orbit_id_slab is None:
                    kill_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                else:
                    # Broadcast orbit_id of the picked pivot via psum-with-mask
                    # (same idiom as the L[p, :] broadcast above), then mark
                    # all local orbit-mates inactive.
                    local_op_val = jnp.where(
                        my_has_p, orbit_id_slab[safe_idx], jnp.int32(0),
                    )
                    orbit_id_p = lax.psum(local_op_val, mesh_axis)
                    kill_mask = orbit_id_slab == orbit_id_p
                active = active & ~kill_mask
                d = jnp.where(kill_mask, minus_inf, d_new)

                return d, L, piv, active, d_taken, trR_over_trG

            d_final, L_out, piv_out, _, d_taken, trR_over_trG = lax.fori_loop(
                0, k_keep, body, init,
            )
            d_final = jnp.where(jnp.isfinite(d_final), d_final, 0.0)
            d0max_global = lax.pmax(jnp.max(local_diag), axis_name=mesh_axis)
            floor = jnp.sqrt(eps) * d0max_global
            rank = jnp.sum(d_taken > floor).astype(jnp.int32)
            d_taken = jnp.where(jnp.arange(k_keep) < rank, d_taken, 0.0)
            return piv_out, L_out, rank, d_final, d_taken, trR_over_trG

        if orbit_id is None:
            return shard_map(
                lambda g: body_local(g, None), mesh=mesh,
                in_specs=in_specs_no_orbit, out_specs=out_specs,
                check_rep=False,
            )(G)
        else:
            return shard_map(
                body_local, mesh=mesh,
                in_specs=in_specs_orbit, out_specs=out_specs,
                check_rep=False,
            )(G, orbit_id)

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
    band_chunk_size: int = 64,
    use_phdf5: bool = False,
    memory_per_device_gb: float | None = None,
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
    from common.load_wfns import load_centroids_band_chunked
    from common.isdf_fitting import (
        pair_density,
        gram_q0_from_pair,
    )

    # Resolve windows.
    if band_range_left is None or band_range_right is None:
        if n_val is None or n_cond is None:
            raise ValueError(
                "Must supply either (n_val, n_cond) or "
                "(band_range_left, band_range_right)"
            )
        # v×(v+c) default: left = (0, n_val), right = (0, n_val+n_cond).
        # The centroids that prune-Cholesky picks then span the val×val
        # diagonals that V_H and any G_RI band-diagonal projection
        # consume, on top of the val×cond pair densities χ₀/W/Σ_xc need.
        # On MoS2 4×4 this cut V_H |err| at the CBM ~3× vs the legacy
        # v×c window (right=(n_val, n_val+n_cond)) at the same centroid
        # count.  Σ_xc is unaffected since the (v+c) right pool is a
        # superset of the legacy cond range; conditioning of the Gram
        # only improves (more PSD contributions).  Callers needing the
        # strict legacy v×c Gram should pass ``band_range_left=(0,nval)``
        # and ``band_range_right=(nval, nval+ncond)`` explicitly.
        left_range = (0, int(n_val))
        right_range = (0, int(n_val) + int(n_cond))
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

    # Memory budget for the band-+k-chunker. ``load_centroids_band_chunked``
    # reads ``meta.memory_per_device_gb`` to size the FFT-box per chunk; if
    # the caller didn't pin a budget, auto-detect device HBM the same way
    # gw_config does so the prune path tracks whatever the rest of LORRAX
    # is using.
    if memory_per_device_gb is None or memory_per_device_gb <= 0:
        try:
            from common.gpu_utils import get_device_memory_gb
            memory_per_device_gb = float(get_device_memory_gb())
        except Exception:
            memory_per_device_gb = 0.0  # falls back to the 36 GB default
    setattr(meta, "memory_per_device_gb", float(memory_per_device_gb))

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
              f"norms={'on' if band_norms is not None else 'off'}, "
              f"backend={'phdf5' if use_phdf5 else 'WFNReader'}, "
              f"budget={meta.memory_per_device_gb:g} GB/device, "
              f"band_chunk_size={band_chunk_size}")

    # ---- Left window ----
    with timing.section("left.load"):
        psi_l_rmu_Y, psi_l_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta, cand_idx, bispinor, mesh_xy, left_range,
            band_chunk_size=band_chunk_size, use_phdf5=use_phdf5,
        )
        if norms_l_j is not None:
            # Y shape (nk, nb, ns, n_rmu); X shape (nk, n_rmu, nb, ns)
            psi_l_rmu_Y = psi_l_rmu_Y / norms_l_j[None, :, None, None]
            psi_l_rmuT_X = psi_l_rmuT_X / norms_l_j[None, None, :, None]
        psi_l_rmu_Y.block_until_ready()
    with timing.section("left.pair"):
        P_l_k = pair_density(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
        P_l_k.block_until_ready()
    del psi_l_rmu_Y, psi_l_rmuT_X

    # ---- Right window ----
    with timing.section("right.load"):
        psi_r_rmu_Y, psi_r_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta, cand_idx, bispinor, mesh_xy, right_range,
            band_chunk_size=band_chunk_size, use_phdf5=use_phdf5,
        )
        if norms_r_j is not None:
            psi_r_rmu_Y = psi_r_rmu_Y / norms_r_j[None, :, None, None]
            psi_r_rmuT_X = psi_r_rmuT_X / norms_r_j[None, None, :, None]
        psi_r_rmu_Y.block_until_ready()
    with timing.section("right.pair"):
        P_r_k = pair_density(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
        P_r_k.block_until_ready()
    del psi_r_rmu_Y, psi_r_rmuT_X

    # ---- q=0 Gram: sum_k w_k · Σ_{αβ} conj(P_l_k,αβ) · P_r_k,αβ ----
    # γ̃ identity (charge channel) — open-spin Frobenius reduction.
    with timing.section("q0_sum"):
        G = gram_q0_from_pair(P_l_k, P_r_k, kw, mesh_xy=mesh_xy)
        G.block_until_ready()
    return G
