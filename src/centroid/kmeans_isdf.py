"""Density-weighted k-means with PBC for ISDF sampling-point selection.

Distances use the metric tensor ``G = avec @ avec.T`` (rows-of-avec are
lattice vectors) and the min image of ``df + n`` over a precomputed
integer-offset table. Its bounds are derived from the metric, so a skew
or unreduced cell can require offsets beyond the adjacent images.

Public surface:

* ``weighted_kmeans_jax`` — driver. The entire Lloyd loop runs to
  convergence on device, with the real-space grid distributed.
* ``kmeans_pp_init`` — k-means++ seed via Gumbel-max on log-weights.
* ``build_min_image_offsets`` — startup-time table of relevant images.
* ``snap_centroids_to_grid``, ``ensure_unique_centroids`` — host-side
  helpers used by the CLI.

Device placement, mesh construction and the one collective this
algorithm needs live in :mod:`centroid.distribution`, not here.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from common import timing
from runtime.padding import pad_axis
from . import distribution as dist
from ffi import _services as _lx_services        # noqa: F401
_lx_services.ensure_on_path()
import symmetry_maps as _sym                     # noqa: E402


BOHR_TO_ANG = 0.529177210544


# ─────────────────────────────────────────────────────────────────────────
# PBC distance: lattice-aware min-image offset table
# ─────────────────────────────────────────────────────────────────────────

def build_min_image_offsets(metric_tensor, search_radius: int = 1) -> np.ndarray:
    """Complete offset table for the closest image of a centred displacement.

    Returns ``(M, 3)`` int32 with identity at ``[0]``. Typical M: 1 for
    cubic / orthorhombic, 5 for 2D-hex-with-orthogonal-c, 7 for primitive
    FCC. ``search_radius`` is a lower bound on the enumerated range, never
    a cap that can silently omit a winning image.

    A winner ``x = df+n`` has ``x.T@G@x <= df.T@G@df <= D``, where ``D`` is
    the largest value at the eight vertices of the centred unit cube.
    Cauchy--Schwarz gives ``|x_i| <= sqrt(D * (G^-1)[i,i])``. Hence every
    possible winner lies in a finite integer box. Within that box, discard
    an offset ``n`` only if another ``m`` is no farther *everywhere* in the
    cube: ``n.T@G@n - m.T@G@m >= ||G@(n-m)||_1``. This is an exact linear
    inequality over the cube, unlike sampling displacements on a mesh.
    """
    import itertools as _it
    G = np.asarray(metric_tensor, dtype=np.float64)
    if (G.shape != (3, 3) or not np.all(np.isfinite(G))
            or not np.allclose(G, G.T, rtol=1e-12, atol=0.0)):
        raise ValueError("min-image metric must be a finite symmetric 3x3 matrix")
    G = (G + G.T) / 2
    if np.linalg.eigvalsh(G)[0] <= 0:
        raise ValueError("min-image metric must be positive definite")
    R = int(search_radius)
    if R < 0:
        raise ValueError("search_radius must be nonnegative")

    # Long double reduces cancellation at a nearly coincident Voronoi face.
    # The 3x3 adjugate also keeps the bound in that precision; NumPy's
    # general inverse currently drops long double back to float64.
    G = G.astype(np.longdouble)
    vertices = np.asarray(list(_it.product((-0.5, 0.5), repeat=3)),
                          dtype=np.longdouble)
    D = max(v @ G @ v for v in vertices)
    cof_diag = np.array([
        G[1, 1] * G[2, 2] - G[1, 2] ** 2,
        G[0, 0] * G[2, 2] - G[0, 2] ** 2,
        G[0, 0] * G[1, 1] - G[0, 1] ** 2,
    ], dtype=np.longdouble)
    determinant = (G[0, 0] * cof_diag[0]
                   - G[0, 1] * (G[0, 1] * G[2, 2] - G[0, 2] * G[1, 2])
                   + G[0, 2] * (G[0, 1] * G[1, 2] - G[0, 2] * G[1, 1]))
    if determinant <= 0 or np.any(cof_diag <= 0):
        raise ValueError("min-image metric is too ill-conditioned to bound images")
    radius_bound = np.longdouble(0.5) + np.sqrt(D * cof_diag / determinant)
    if (np.any(~np.isfinite(radius_bound))
            or np.any(radius_bound >= np.iinfo(np.int32).max - 2)
            or R >= np.iinfo(np.int32).max - 2):
        raise ValueError("min-image metric requires offsets beyond int32 range")
    radius = np.maximum(R, np.ceil(np.nextafter(
        radius_bound, np.longdouble(np.inf))).astype(np.int64))
    candidate_count = int(np.prod(2 * radius + 1, dtype=object))
    if candidate_count > 1_000_000:
        raise ValueError(
            "min-image metric requires more than one million translation "
            "candidates; use a reduced lattice basis")

    candidates = np.asarray(list(_it.product(*(
        range(-int(r), int(r) + 1) for r in radius))), dtype=np.longdouble)
    zero = np.all(candidates == 0, axis=1)
    Gn = candidates @ G
    q = np.sum(Gn * candidates, axis=1)
    # If zero dominates n throughout the cube, n cannot change the minimum.
    possible = zero | (q < np.sum(np.abs(Gn), axis=1))
    candidates, q = candidates[possible], q[possible]
    if len(candidates) > 10_000:
        raise ValueError(
            "min-image metric has more than 10000 potentially relevant "
            "translations; use a reduced lattice basis")

    keep = np.ones(len(candidates), dtype=bool)
    for i, n in enumerate(candidates):
        if np.all(n == 0):
            continue
        # The minimum of d_n²-d_m² over df in [-1/2,1/2]^3 is the
        # quadratic difference minus the L1 norm of G(n-m).
        min_difference = q[i] - q - np.sum(
            np.abs((n - candidates) @ G), axis=1)
        min_difference[i] = -np.inf
        if np.any(min_difference >= 0):
            keep[i] = False
    kept = candidates[keep].astype(np.int32)
    return np.concatenate((np.zeros((1, 3), dtype=np.int32),
                           kept[np.any(kept != 0, axis=1)]), axis=0)


def _quadform_G(dx, dy, dz, g00, g01, g02, g11, g12, g22):
    """Fused d² = δ^T G δ (no K=3 GEMM)."""
    return (g00 * dx + g01 * dy + g02 * dz) * dx \
         + (g01 * dx + g11 * dy + g12 * dz) * dy \
         + (g02 * dx + g12 * dy + g22 * dz) * dz


@jax.jit
def pbc_distance_sq_single(
    positions_frac: jnp.ndarray,
    centroid_frac: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """(P,) min-image squared PBC distances from positions to one centroid."""
    dx0 = positions_frac[:, 0] - centroid_frac[0]
    dx0 = dx0 - jnp.round(dx0)
    dy0 = positions_frac[:, 1] - centroid_frac[1]
    dy0 = dy0 - jnp.round(dy0)
    dz0 = positions_frac[:, 2] - centroid_frac[2]
    dz0 = dz0 - jnp.round(dz0)
    g00, g01, g02 = metric_tensor[0, 0], metric_tensor[0, 1], metric_tensor[0, 2]
    g11, g12, g22 = metric_tensor[1, 1], metric_tensor[1, 2], metric_tensor[2, 2]

    def body(i, best):
        nx, ny, nz = offsets[i, 0], offsets[i, 1], offsets[i, 2]
        return jnp.minimum(best, _quadform_G(
            dx0 + nx, dy0 + ny, dz0 + nz, g00, g01, g02, g11, g12, g22))

    return lax.fori_loop(0, offsets.shape[0], body, jnp.full_like(dx0, jnp.inf))


_DEFAULT_C_BLOCK = 32
"""C-chunk for the assignment scan — ~30 MB peak per axis at P=1e6 in fp32."""


@partial(jax.jit, static_argnames=['n_c', 'c_block'])
def assign_labels_chunked(
    positions_frac: jnp.ndarray,
    centroids_frac: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    n_c: int,
    c_block: int = _DEFAULT_C_BLOCK,
    offsets: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """(P,) int32 nearest-centroid labels under min-image distance.

    Chunked along C so peak memory is ``(P, c_block)`` per axis. Centroids
    are NaN-padded to a multiple of ``c_block`` so any ``n_c`` works.
    """
    if offsets is None:
        offsets = jnp.zeros((1, 3), dtype=jnp.int32)
    n_chunks = (n_c + c_block - 1) // c_block
    n_c_padded = n_chunks * c_block
    if n_c_padded != n_c:
        pad = jnp.full((n_c_padded - n_c, 3), jnp.nan, dtype=centroids_frac.dtype)
        centroids_frac = jnp.concatenate([centroids_frac, pad], axis=0)
    centroids_blocked = centroids_frac.reshape(n_chunks, c_block, 3)

    g00, g01, g02 = metric_tensor[0, 0], metric_tensor[0, 1], metric_tensor[0, 2]
    g11, g12, g22 = metric_tensor[1, 1], metric_tensor[1, 2], metric_tensor[2, 2]

    def body(carry, cent_chunk):
        best_d, best_c, chunk_idx = carry
        dx0 = positions_frac[:, None, 0] - cent_chunk[None, :, 0]
        dx0 = dx0 - jnp.round(dx0)
        dy0 = positions_frac[:, None, 1] - cent_chunk[None, :, 1]
        dy0 = dy0 - jnp.round(dy0)
        dz0 = positions_frac[:, None, 2] - cent_chunk[None, :, 2]
        dz0 = dz0 - jnp.round(dz0)

        def offset_body(i, dchunk):
            nx, ny, nz = offsets[i, 0], offsets[i, 1], offsets[i, 2]
            return jnp.minimum(dchunk, _quadform_G(
                dx0 + nx, dy0 + ny, dz0 + nz, g00, g01, g02, g11, g12, g22))

        d_chunk = lax.fori_loop(
            0, offsets.shape[0], offset_body, jnp.full_like(dx0, jnp.inf)
        )
        # NaN propagates through the offset loop; mask once at the end so
        # padded centroids never win argmin.
        d_chunk = jnp.where(jnp.isnan(d_chunk), jnp.inf, d_chunk)
        local_c = jnp.argmin(d_chunk, axis=1).astype(jnp.int32)
        local_d = jnp.take_along_axis(d_chunk, local_c[:, None], axis=1)[:, 0]
        better = local_d < best_d
        return (jnp.where(better, local_d, best_d),
                jnp.where(better, chunk_idx * c_block + local_c, best_c),
                chunk_idx + 1), None

    P = positions_frac.shape[0]
    init = (jnp.full((P,), jnp.inf, dtype=metric_tensor.dtype),
            jnp.zeros((P,), dtype=jnp.int32),
            jnp.int32(0))
    (_, labels, _), _ = lax.scan(body, init, centroids_blocked, unroll=1)
    return labels


# ─────────────────────────────────────────────────────────────────────────
# Orbit-aware nearest-rep assignment (phase 2)
# ─────────────────────────────────────────────────────────────────────────

def _orbit_d2_chunk(positions, image_chunk, metric, offsets):
    """(P, c_block) min-image squared distance from positions to a block of
    centroid images (one image per (s, μ) pair). Reuses the same
    fori_loop-over-offsets pattern as assign_labels_chunked."""
    g00, g01, g02 = metric[0, 0], metric[0, 1], metric[0, 2]
    g11, g12, g22 = metric[1, 1], metric[1, 2], metric[2, 2]
    dx0 = positions[:, None, 0] - image_chunk[None, :, 0]
    dx0 = dx0 - jnp.round(dx0)
    dy0 = positions[:, None, 1] - image_chunk[None, :, 1]
    dy0 = dy0 - jnp.round(dy0)
    dz0 = positions[:, None, 2] - image_chunk[None, :, 2]
    dz0 = dz0 - jnp.round(dz0)

    def body(i, dchunk):
        nx, ny, nz = offsets[i, 0], offsets[i, 1], offsets[i, 2]
        return jnp.minimum(dchunk, _quadform_G(
            dx0 + nx, dy0 + ny, dz0 + nz, g00, g01, g02, g11, g12, g22))
    return lax.fori_loop(0, offsets.shape[0], body, jnp.full_like(dx0, jnp.inf))


def _orbit_d2_per_point(positions, image_per_point, metric, offsets):
    """(P,) min-image squared distance with one image per point."""
    g00, g01, g02 = metric[0, 0], metric[0, 1], metric[0, 2]
    g11, g12, g22 = metric[1, 1], metric[1, 2], metric[2, 2]
    dx0 = positions[:, 0] - image_per_point[:, 0]
    dx0 = dx0 - jnp.round(dx0)
    dy0 = positions[:, 1] - image_per_point[:, 1]
    dy0 = dy0 - jnp.round(dy0)
    dz0 = positions[:, 2] - image_per_point[:, 2]
    dz0 = dz0 - jnp.round(dz0)

    def body(i, best):
        nx, ny, nz = offsets[i, 0], offsets[i, 1], offsets[i, 2]
        return jnp.minimum(best, _quadform_G(
            dx0 + nx, dy0 + ny, dz0 + nz, g00, g01, g02, g11, g12, g22))
    return lax.fori_loop(0, offsets.shape[0], body, jnp.full_like(dx0, jnp.inf))


@partial(jax.jit, static_argnames=['n_rep', 'c_block'])
def assign_labels_orbit_chunked(
    positions: jnp.ndarray,
    reps: jnp.ndarray,
    metric: jnp.ndarray,
    n_rep: int,
    c_block: int = _DEFAULT_C_BLOCK,
    offsets: jnp.ndarray | None = None,
    Rinv: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    tie_tol: float = 1e-10,
):
    """Orbit-aware nearest-rep assignment.

    For each grid point, finds the **representative** μ whose orbit
    contains the closest image, plus a boolean tie mask of the symmetry
    operations that achieve that minimum (for the winning rep only).

    The scan computes orbit distances via fori_loop(sym,
    fori_loop(offset, ...)) with a peak buffer of (P, c_block). Once the
    globally winning rep is known, one final symmetry loop builds its
    (P, n_sym) tie mask. A losing chunk never needs a tie mask.

    Returns
    -------
    labels   : (P,) int32  — winning rep
    best_d2  : (P,) fp64   — winning orbit distance²
    tie_mask : (P, n_sym) bool — for the winning rep, the sym ops that tie
    """
    if offsets is None:
        offsets = jnp.zeros((1, 3), dtype=jnp.int32)
    if Rinv is None:
        Rinv = jnp.eye(3, dtype=jnp.int32)[None]
        tau = jnp.zeros((1, 3), dtype=positions.dtype)
    n_sym = Rinv.shape[0]

    # NaN-pad reps to a multiple of c_block — same trick as assign_labels_chunked.
    n_chunks = (n_rep + c_block - 1) // c_block
    n_rep_padded = n_chunks * c_block
    if n_rep_padded != n_rep:
        pad = jnp.full((n_rep_padded - n_rep, 3), jnp.nan, dtype=reps.dtype)
        reps = jnp.concatenate([reps, pad], axis=0)
    reps_blocked = reps.reshape(n_chunks, c_block, 3)

    P = positions.shape[0]

    def body(carry, rep_chunk):
        best_d, best_label, chunk_idx = carry

        # Pass 1: orbit distance per (point, rep-in-chunk).
        def sym_loop(s, orbit_d):
            # NO WRAP, deliberately: `_orbit_d2_chunk` is a minimum-image
            # metric carrying its own explicit offset table, so folding the
            # image into [0,1) first would put it on the wrong replica.
            image_chunk = _sym.r_action_forward_one(
                rep_chunk, Rinv[s], tau[s], wrap=False)
            return jnp.minimum(orbit_d, _orbit_d2_chunk(
                positions, image_chunk, metric, offsets))

        orbit_d = lax.fori_loop(
            0, n_sym, sym_loop, jnp.full((P, c_block), jnp.inf)
        )
        # NaN sentinel for padded reps: NaN propagates through subtract/quadform,
        # jnp.minimum(NaN, x) = NaN, so masking once at the end suffices.
        orbit_d = jnp.where(jnp.isnan(orbit_d), jnp.inf, orbit_d)

        local_c = orbit_d.argmin(axis=1).astype(jnp.int32)        # (P,)
        local_d = jnp.take_along_axis(orbit_d, local_c[:, None], axis=1)[:, 0]

        better = local_d < best_d
        return (
            jnp.where(better, local_d, best_d),
            jnp.where(better, chunk_idx * c_block + local_c, best_label),
            chunk_idx + 1,
        ), None

    init = (
        jnp.full((P,), jnp.inf, dtype=metric.dtype),
        jnp.zeros((P,), dtype=jnp.int32),
        jnp.int32(0),
    )
    (best_d2, labels, _), _ = lax.scan(body, init, reps_blocked, unroll=1)
    winning_rep = reps[labels]

    def tie_sym_loop(s, tie_mask):
        image_p = _sym.r_action_forward_one(
            winning_rep, Rinv[s], tau[s], wrap=False)
        d2_s = _orbit_d2_per_point(positions, image_p, metric, offsets)
        return tie_mask.at[:, s].set(d2_s <= best_d2 + tie_tol)

    tie_mask = lax.fori_loop(
        0, n_sym, tie_sym_loop, jnp.zeros((P, n_sym), dtype=bool)
    )
    return labels, best_d2, tie_mask


# ─────────────────────────────────────────────────────────────────────────
# Min-image displacement (vector) + Lloyd update internals
# ─────────────────────────────────────────────────────────────────────────

def _min_image_delta(delta, metric_tensor, offsets):
    """The (..., 3) lattice image of ``delta`` minimising δ^T G δ."""
    delta = delta - jnp.round(delta)
    g00, g01, g02 = metric_tensor[0, 0], metric_tensor[0, 1], metric_tensor[0, 2]
    g11, g12, g22 = metric_tensor[1, 1], metric_tensor[1, 2], metric_tensor[2, 2]
    dx0, dy0, dz0 = delta[..., 0], delta[..., 1], delta[..., 2]

    def body(i, carry):
        best_d2, best_delta = carry
        nx, ny, nz = offsets[i, 0], offsets[i, 1], offsets[i, 2]
        dxi, dyi, dzi = dx0 + nx, dy0 + ny, dz0 + nz
        d2_i = _quadform_G(dxi, dyi, dzi, g00, g01, g02, g11, g12, g22)
        better = d2_i < best_d2
        return (jnp.where(better, d2_i, best_d2),
                jnp.where(better[..., None],
                          jnp.stack([dxi, dyi, dzi], axis=-1), best_delta))

    _, best_delta = lax.fori_loop(
        0, offsets.shape[0], body, (jnp.full_like(dx0, jnp.inf), delta)
    )
    return best_delta


def _local_update_accumulators(positions_frac, centroids_frac, rho_flat,
                               labels, n_c, metric_tensor, offsets):
    """Per-centroid weighted-sum accumulators; O(P+C) memory."""
    delta = _min_image_delta(
        positions_frac - centroids_frac[labels], metric_tensor, offsets
    )
    return (
        jax.ops.segment_sum(rho_flat[:, None] * delta, labels, num_segments=n_c),
        jax.ops.segment_sum(rho_flat, labels, num_segments=n_c),
    )


def _orbit_local_update_accumulators(positions, reps, rho, labels, tie_mask,
                                     n_rep, metric, offsets, R, Rinv, tau):
    """Orbit-aware accumulator. For each point assigned to rep μ:

        - n_tied(p) = #(sym ops achieving the orbit minimum)         (pass-2 of assign)
        - tie_share(p, s) = 1/n_tied(p) on tied ops, 0 elsewhere
        - For each tied op s, fold the displacement back into the rep frame:
              δ_image(p, s) = min-image(x_p − image(r_μ, s))
              δ_rep(p, s)   = δ_image(p, s) @ R[s].T
        - Contribute (ρ_p · tie_share(p, s), ρ_p · tie_share(p, s) · δ_rep(p, s))
          to (sum_w[μ], sum_wd[μ]).

    Total per-point mass = ρ_p (independent of orbit / tie multiplicity);
    contribution to centroid update is ρ_p × ⟨δ_rep⟩_{tied s}, which projects
    onto the stabiliser-invariant subspace at special positions. Memory:
    O(P + n_rep), no (P, n_rep, n_sym) buffer.
    """
    n_sym = R.shape[0]
    n_tied = tie_mask.sum(axis=1).astype(rho.dtype)             # (P,)
    inv_n_tied = jnp.where(n_tied > 0, 1.0 / n_tied, 0.0)        # (P,)
    rep_per_point = reps[labels]                                # (P, 3)

    def sym_body(s, carry):
        sum_wd, sum_w = carry
        image_p = _sym.r_action_forward_one(                    # (P, 3)
            rep_per_point, Rinv[s], tau[s], wrap=False)     # min-image metric
        delta_image = _min_image_delta(
            positions - image_p, metric, offsets
        )                                                        # (P, 3)
        delta_rep = delta_image @ R[s].T                         # (P, 3) fold-back
        w_s = rho * inv_n_tied * tie_mask[:, s].astype(rho.dtype)  # (P,)
        sum_wd = sum_wd + jax.ops.segment_sum(
            w_s[:, None] * delta_rep, labels, num_segments=n_rep
        )
        sum_w = sum_w + jax.ops.segment_sum(
            w_s, labels, num_segments=n_rep
        )
        return sum_wd, sum_w

    return lax.fori_loop(
        0, n_sym, sym_body,
        (jnp.zeros((n_rep, 3), dtype=rho.dtype),
         jnp.zeros((n_rep,), dtype=rho.dtype)),
    )


def _finalize_update(centroids_frac, sum_weighted_delta, sum_weights,
                     metric_tensor, offsets):
    """Weighted-mean update, wrap to [0, 1), min-image movement²."""
    avg_delta = jnp.where(
        sum_weights[:, None] > 0,
        sum_weighted_delta / jnp.maximum(sum_weights[:, None], 1e-10),
        0.0,
    )
    new_centroids_frac = (centroids_frac + avg_delta) % 1.0
    move = _min_image_delta(new_centroids_frac - centroids_frac,
                            metric_tensor, offsets)
    movement_sq = jnp.einsum('ci,ij,cj->c', move, metric_tensor, move)
    return new_centroids_frac, movement_sq


def _orbit_finalize_update(reps, sum_wd, sum_w, metric, offsets, Rinv, tau):
    """Orbit-aware finalize: same weighted-mean step as ``_finalize_update``,
    plus canonicalisation of every new rep (so reps don't chatter between
    equivalent orbit members from one Lloyd iteration to the next).

    Movement² is measured between the OLD rep and the canonicalised NEW
    rep in the rep frame (min-image distance). This is the right thing
    for convergence: if the rep just hops to a different orbit member,
    canonicalisation puts it back, and movement² is tiny.
    """
    avg_delta = jnp.where(
        sum_w[:, None] > 0,
        sum_wd / jnp.maximum(sum_w[:, None], 1e-10),
        0.0,
    )
    new_reps = (reps + avg_delta) % 1.0

    # Canonicalise every new rep (vmap'd version of _canonicalize_rep).
    canon = jax.vmap(lambda r: _canonicalize_rep(r, Rinv, tau))(new_reps)

    move = _min_image_delta(canon - reps, metric, offsets)
    movement_sq = jnp.einsum('ci,ij,cj->c', move, metric, move)
    return canon, movement_sq


# ─────────────────────────────────────────────────────────────────────────
# One Lloyd iteration, on this rank's slice of the grid
# ─────────────────────────────────────────────────────────────────────────

def _lloyd_step(positions, centroids, rho, metric, offsets,
                n_c, c_block, grid_axis):
    """Assign → accumulate → move, with the grid distributed.

    Nearest-centroid assignment is embarrassingly parallel over grid
    points; the weighted-mean update needs each centroid's total over ALL
    points, which is the two sums over the grid below."""
    labels = assign_labels_chunked(
        positions, centroids, metric, n_c, c_block=c_block, offsets=offsets,
    )
    local_wd, local_w = _local_update_accumulators(
        positions, centroids, rho, labels, n_c, metric, offsets,
    )
    return _finalize_update(
        centroids,
        dist.sum_over_grid(local_wd, grid_axis),
        dist.sum_over_grid(local_w, grid_axis),
        metric, offsets,
    )


def _orbit_lloyd_step(positions, reps, rho, metric, offsets,
                      R, Rinv, tau, n_rep, c_block, grid_axis):
    """Sym-aware Lloyd step. Same two sums over the grid — the orbit
    fold-back is local to each rank's points, before the per-rep
    accumulators are reduced."""
    labels, _, tie_mask = assign_labels_orbit_chunked(
        positions, reps, metric, n_rep, c_block=c_block,
        offsets=offsets, Rinv=Rinv, tau=tau,
    )
    local_wd, local_w = _orbit_local_update_accumulators(
        positions, reps, rho, labels, tie_mask,
        n_rep, metric, offsets, R, Rinv, tau,
    )
    return _orbit_finalize_update(
        reps,
        dist.sum_over_grid(local_wd, grid_axis),
        dist.sum_over_grid(local_w, grid_axis),
        metric, offsets, Rinv, tau,
    )


# ─────────────────────────────────────────────────────────────────────────
# Full Lloyd loop — iterates to convergence on device
# ─────────────────────────────────────────────────────────────────────────

def make_lloyd_loop(
    mesh,
    n_c: int,
    max_steps: int,
    tolerance: float,
    offsets,
    *,
    R=None, Rinv=None, tau=None,
    c_block: int = _DEFAULT_C_BLOCK,
    mesh_axis: str | tuple[str, ...] = 'x',
):
    """Lloyd iteration to ``max(movement) < tolerance`` or ``max_steps``.

    ``(R, Rinv, tau)`` select the **orbit-aware** step, in which centroids
    are orbit representatives and every point's contribution is folded
    back to the rep frame through the minimising symmetry image; without
    them each centroid is a literal point.

    The whole loop is one ``lax.while_loop`` that never returns to the
    host, so convergence costs zero syncs per iteration.  Returns
    ``(final_centroids, steps_taken, max_movement_sq)``.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    tol_sq = jnp.float64(tolerance) ** 2
    orbit_aware = R is not None
    if orbit_aware:
        R = jnp.asarray(R, dtype=jnp.int32)
        Rinv = jnp.asarray(Rinv, dtype=jnp.int32)
        tau = jnp.asarray(tau, dtype=jnp.float64)

    def iterate(positions, centroids, rho, metric):
        def not_converged(state):
            _, step, max_mv_sq = state
            return (step < max_steps) & (max_mv_sq >= tol_sq)

        def step_once(state):
            c, step, _ = state
            if orbit_aware:
                new_c, movement_sq = _orbit_lloyd_step(
                    positions, c, rho, metric, offsets,
                    R, Rinv, tau, n_c, c_block, mesh_axis,
                )
            else:
                new_c, movement_sq = _lloyd_step(
                    positions, c, rho, metric, offsets,
                    n_c, c_block, mesh_axis,
                )
            return new_c, step + 1, jnp.max(movement_sq)

        return lax.while_loop(
            not_converged, step_once,
            (centroids, jnp.int32(0), jnp.float64(jnp.inf)),
        )

    return dist.grid_parallel_loop(
        mesh, iterate, grid_axis=mesh_axis, who="make_lloyd_loop")


# ─────────────────────────────────────────────────────────────────────────
# k-means++ init (Gumbel-max sampling, one fori_loop)
# ─────────────────────────────────────────────────────────────────────────

def _gumbel_sample_argmax(log_weights: jnp.ndarray, key) -> jnp.ndarray:
    """One categorical draw via Gumbel-max on log-weights. ``-inf`` weights
    are never selected; ``u`` clamped away from 0 so ``-log(-log u)`` is finite."""
    n = log_weights.shape[0]
    eps = jnp.finfo(log_weights.dtype).tiny
    u = jax.random.uniform(key, (n,), dtype=log_weights.dtype,
                           minval=eps, maxval=1.0 - eps)
    return jnp.argmax(log_weights + (-jnp.log(-jnp.log(u))))


def _orbit_distance_sq(positions, rep, metric, offsets, Rinv, tau):
    """(P,) min-image squared distance from positions to the ORBIT of rep.

    For non-orbit-aware callers, pass Rinv = identity[None] and tau = 0 —
    the n_sym=1 fori_loop is a no-op and this reduces to
    ``pbc_distance_sq_single``.
    """
    def body(s, best):
        image = _sym.r_action_forward_one(rep, Rinv[s], tau[s], wrap=True)
        return jnp.minimum(
            best, pbc_distance_sq_single(positions, image, metric, offsets)
        )
    return lax.fori_loop(
        0, Rinv.shape[0], body, jnp.full((positions.shape[0],), jnp.inf)
    )


def _canonicalize_rep(rep, Rinv, tau):
    """Lex-smallest member of rep's orbit. Thin wrapper around
    ``symmetry_maps.canonicalize_orbit`` for the single-rep case."""
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import canonicalize_orbit
    return canonicalize_orbit(rep[None, :], Rinv, tau)[0]


@partial(jax.jit, static_argnames=['n_c'])
def kmeans_pp_init(
    positions_frac: jnp.ndarray,
    rho_flat: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    n_c: int,
    key,
    offsets: jnp.ndarray | None = None,
    Rinv: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Density-weighted k-means++ init.

    First centroid drawn from Categorical(ρ); each subsequent from
    Categorical(D²·ρ) where D² is the **orbit** distance when ``Rinv`` /
    ``tau`` are given (default: identity-only ⇒ ordinary kmeans++).
    Centroids are canonicalised to a deterministic orbit member when
    sym data is provided.
    """
    if offsets is None:
        offsets = jnp.zeros((1, 3), dtype=jnp.int32)
    orbit_aware = Rinv is not None
    if not orbit_aware:
        Rinv = jnp.eye(3, dtype=jnp.int32)[None]
        tau = jnp.zeros((1, 3), dtype=positions_frac.dtype)

    dtype = positions_frac.dtype
    tiny = jnp.finfo(dtype).tiny
    log_rho = jnp.log(jnp.maximum(rho_flat.astype(dtype), tiny))

    key, sub = jax.random.split(key)
    first_cent = positions_frac[_gumbel_sample_argmax(log_rho, sub)]
    if orbit_aware:
        first_cent = _canonicalize_rep(first_cent, Rinv, tau)
    centroids = jnp.zeros((n_c, 3), dtype=dtype).at[0].set(first_cent)
    min_d2 = _orbit_distance_sq(
        positions_frac, first_cent, metric_tensor, offsets, Rinv, tau
    )

    def body(c_idx, state):
        centroids, min_d2, key = state
        key, sub = jax.random.split(key)
        log_w = jnp.log(jnp.maximum(min_d2, tiny)) + log_rho
        new_cent = positions_frac[_gumbel_sample_argmax(log_w, sub)]
        if orbit_aware:
            new_cent = _canonicalize_rep(new_cent, Rinv, tau)
        centroids = centroids.at[c_idx].set(new_cent)
        d2_new = _orbit_distance_sq(
            positions_frac, new_cent, metric_tensor, offsets, Rinv, tau
        )
        return centroids, jnp.minimum(min_d2, d2_new), key

    centroids, _, _ = lax.fori_loop(1, n_c, body, (centroids, min_d2, key))
    return centroids


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────

def _pick_c_block(n_c: int, preferred: int = _DEFAULT_C_BLOCK) -> int:
    """C-chunk for the assignment scan: ``min(preferred, n_c)``. NaN padding
    handles the non-divisor case so we keep the CUDA-friendly default."""
    return min(preferred, n_c)


def weighted_kmeans_jax(
    avec,
    rho,
    N_c: int = 10,
    max_steps: int = 200,
    tolerance: float = 5e-3,
    seed: int = 0,
    *,
    mesh,
    mesh_axis: str | tuple[str, ...] = 'x',
    init_method: str = 'kpp',
    R=None,
    Rinv=None,
    tau=None,
    print_fn=print,
):
    """Density-weighted PBC k-means over the real-space FFT grid.

    Seed the centroids (k-means++ on ρ, or an i.i.d. ρ-weighted draw),
    then Lloyd-iterate to convergence: assign every grid point to its
    nearest centroid under the min-image lattice metric, move each
    centroid to the ρ-weighted mean of its cell, repeat.

    ``avec`` is the (3, 3) real-space lattice (rows are lattice vectors)
    and ``rho`` the (Nx, Ny, Nz) weight; both may be numpy.

    When ``(R, Rinv, tau)`` are supplied, runs the **orbit-aware** path:
    centroids become orbit *representatives* and each point's
    contribution is folded back to the rep frame through the minimising
    symmetry image, so the final set is closed under the point group.
    Pass the table from ``orbit_syms.real_space_action_tables(wfn, sym)``.

    Returns:
        labels  : cluster assignment for each grid point.  Its distributed
            extent is padded to the mesh divisor when necessary; entries
            beyond ``rho.size`` are the sentinel ``-1`` and are not logical
            grid points.
        centroids: (N_c, 3) fractional centroids / canonical reps in [0, 1).
        steps_taken: int.
        max_movement_sq: final max movement² in avec units squared.
    """
    if init_method not in ('kpp', 'random'):
        raise ValueError(f"init_method must be 'kpp' or 'random', got {init_method!r}")
    orbit_aware = R is not None
    if orbit_aware and (Rinv is None or tau is None):
        raise ValueError("orbit mode requires all of (R, Rinv, tau)")

    avec = jnp.asarray(avec, dtype=jnp.float64)
    metric_tensor = avec @ avec.T
    offsets = jnp.asarray(build_min_image_offsets(metric_tensor))
    print_fn(f"PBC min-image offsets: {offsets.shape[0]} (1 ⇒ orthorhombic); "
             f"orbit mode: {'on (n_sym=' + str(int(R.shape[0])) + ')' if orbit_aware else 'off'}")

    Nx, Ny, Nz = rho.shape
    fx, fy, fz = (jnp.linspace(0, 1, n, endpoint=False, dtype=jnp.float64)
                  for n in (Nx, Ny, Nz))
    positions = jnp.stack(jnp.meshgrid(fx, fy, fz, indexing="ij"),
                          axis=-1).reshape(-1, 3)
    rho_flat = jnp.asarray(rho, dtype=jnp.float64).reshape(-1)
    print_fn(f"Grid: {Nx}×{Ny}×{Nz} = {positions.shape[0]} points; N_c = {N_c}")

    with timing.section("init"):
        if init_method == 'kpp':
            centroids = kmeans_pp_init(
                positions, rho_flat, metric_tensor, N_c,
                jax.random.PRNGKey(seed), offsets=offsets,
                Rinv=Rinv if orbit_aware else None,
                tau=tau if orbit_aware else None,
            )
        else:
            rho_p = rho_flat / jnp.sum(rho_flat)
            idx = jax.random.choice(
                jax.random.PRNGKey(seed), rho_flat.shape[0],
                shape=(N_c,), p=rho_p, replace=False,
            )
            centroids = positions[idx]
        centroids.block_until_ready()

    # The grid is the distributed axis: each rank owns a slice of the
    # points and the whole centroid list.  Seed on the LOGICAL grid above:
    # a zero-weight padding point must not enter either k-means++ or the
    # random weighted draw.  Only then round the distributed extent up to
    # the product-mesh divisor.  The shared padding helper returns the same
    # object on already-divisible paths, preserving their computation.
    grid_divisor = dist.n_shards(mesh, mesh_axis)
    positions_pad = pad_axis(
        positions, grid_divisor, axis=0, fill=0.0,
    )
    rho_pad = pad_axis(
        rho_flat, grid_divisor, axis=0, fill=0.0,
    )
    if (positions_pad.logical != rho_pad.logical
            or positions_pad.padded != rho_pad.padded):
        raise AssertionError(
            "k-means grid position/weight padding extents disagree: "
            f"positions=({positions_pad.logical}, {positions_pad.padded}), "
            f"weights=({rho_pad.logical}, {rho_pad.padded})"
        )
    n_grid_logical = positions_pad.logical
    n_grid_padded = positions_pad.padded
    if n_grid_padded != n_grid_logical:
        print(
            "K-means grid mesh padding: "
            f"{n_grid_logical} -> {n_grid_padded} points "
            f"(divisor {grid_divisor}); tail weights are zero"
        )
    positions_unplaced = positions_pad.array
    rho_unplaced = rho_pad.array
    metric_unplaced = metric_tensor
    positions = dist.place(positions_unplaced, mesh, mesh_axis, None)
    rho_flat = dist.place(rho_unplaced, mesh, mesh_axis)
    metric_tensor = dist.place(metric_unplaced, mesh)
    centroids = dist.place(centroids, mesh)
    # dist.place stages through a host array and creates independent placed
    # buffers.  Do not keep the original full-grid device arrays alive for
    # the duration of Lloyd.
    positions_unplaced.delete()
    rho_unplaced.delete()
    metric_unplaced.delete()
    del positions_pad, rho_pad, positions_unplaced, rho_unplaced, metric_unplaced

    lloyd = make_lloyd_loop(
        mesh, N_c, max_steps, tolerance, offsets,
        R=R, Rinv=Rinv, tau=tau,
        c_block=_pick_c_block(N_c), mesh_axis=mesh_axis,
    )
    with timing.section("lloyd"):
        centroids, steps, max_mv_sq = lloyd(
            positions, centroids, rho_flat, metric_tensor,
        )
        centroids.block_until_ready()
    steps = int(steps)
    print_fn(f"Lloyd: {steps} steps, max movement = "
             f"{float(jnp.sqrt(max_mv_sq)):.6f}")

    with timing.section("assign_labels"):
        if orbit_aware:
            labels, _, _ = assign_labels_orbit_chunked(
                positions, centroids, metric_tensor, N_c,
                c_block=_pick_c_block(N_c), offsets=offsets,
                Rinv=Rinv, tau=tau,
            )
        else:
            labels = assign_labels_chunked(
                positions, centroids, metric_tensor, N_c,
                c_block=_pick_c_block(N_c), offsets=offsets,
            )
        if n_grid_padded != n_grid_logical:
            # Preserve the sharding-compatible padded shape, but make the
            # non-grid tail impossible to mistake for a physical cluster
            # assignment in any downstream inspection or pruning path.
            labels = labels.at[n_grid_logical:].set(-1)
        labels.block_until_ready()

    max_mv_sq_host = float(max_mv_sq)
    # These placed arrays are the completed Lloyd working set.  They are
    # not returned and dist.place created independent device buffers, so
    # release them before the caller begins centroid snapping/pruning.
    for work_array in (positions, rho_flat, metric_tensor):
        work_array.delete()
    return labels, centroids, steps, max_mv_sq_host


# ─────────────────────────────────────────────────────────────────────────
# Host-side helpers for the CLI
# ─────────────────────────────────────────────────────────────────────────

def snap_centroids_to_grid(
    centroids_frac: np.ndarray,
    fft_grid: tuple[int, int, int],
    deduplicate: bool = True,
    *,
    print_fn=print,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Round fractional centroids to the nearest FFT-grid integer indices.

    Returns ``(indices, fractional, n_duplicates_removed)``.
    """
    indices = np.round(centroids_frac * np.array(fft_grid)).astype(int) % fft_grid
    n_original = indices.shape[0]
    if deduplicate:
        indices = np.unique(indices, axis=0)
        n_duplicates = n_original - indices.shape[0]
        if n_duplicates > 0:
            print_fn(f"snap_centroids_to_grid: {n_duplicates} duplicates "
                     f"({n_original} → {indices.shape[0]} unique)")
    else:
        n_duplicates = 0
    return indices, indices.astype(float) / fft_grid, n_duplicates


def ensure_unique_centroids(
    centroids_frac: np.ndarray,
    fft_grid: tuple[int, int, int],
    rho: np.ndarray | None = None,
    *,
    print_fn=print,
) -> np.ndarray:
    """Snap to the FFT grid; redistribute duplicates onto the highest-density
    unoccupied grid points (arbitrary order if ρ is None). O(N log N), fully
    vectorized."""
    Nx, Ny, Nz = fft_grid
    indices = (np.round(centroids_frac * np.array(fft_grid)).astype(int)
               % np.array(fft_grid))
    # Encode (ix, iy, iz) → flat int for set arithmetic.
    lin = indices[:, 0] * (Ny * Nz) + indices[:, 1] * Nz + indices[:, 2]
    occupied, first_idx = np.unique(lin, return_index=True)
    n_dups = lin.shape[0] - occupied.shape[0]
    if n_dups == 0:
        return indices.astype(float) / fft_grid

    free = np.setdiff1d(np.arange(Nx * Ny * Nz), occupied)
    if rho is not None:
        free = free[np.argsort(-rho.ravel()[free])]
    take = free[:min(n_dups, free.shape[0])]
    if take.shape[0] < n_dups:
        print_fn(f"ensure_unique_centroids: only {take.shape[0]} unoccupied "
                 f"grid points for {n_dups} duplicates; the rest are dropped.")

    extra = np.stack([take // (Ny * Nz), (take // Nz) % Ny, take % Nz], axis=1)
    out = np.concatenate([indices[first_idx], extra], axis=0)
    return out.astype(float) / fft_grid


# ─────────────────────────────────────────────────────────────────────────
# Init-method heuristics (host-side)
# ─────────────────────────────────────────────────────────────────────────
#
# When N_c is a sizeable fraction of the FFT grid, kmeans++ init costs
# O(N_c · P). The density-weighted random fallback avoids that seeding cost;
# it can reach a different Lloyd local minimum.

_KPP_SKIP_FRACTION = 0.10
_DENSE_WARN_FRACTION = 0.25


def _decide_init_method(n_c: int, n_rtot: int,
                        threshold: float = _KPP_SKIP_FRACTION,
                        ) -> tuple[str, str | None]:
    if n_c > int(n_rtot * threshold):
        return ('random',
                f"N_c = {n_c} > {int(n_rtot * threshold)} = "
                f"n_rtot · {threshold:g}; using density-weighted random init.")
    return 'kpp', None


def _warn_dense_grid_regime(m_cand: int, n_c: int, n_rtot: int,
                            threshold: float = _DENSE_WARN_FRACTION,
                            ) -> str | None:
    if n_c > int(n_rtot * threshold):
        return (f"WARNING: N_c = {n_c} > {int(n_rtot * threshold)} = "
                f"n_rtot · {threshold:g}; the requested selection is dense "
                f"on the FFT grid, so check candidate coverage and Gram "
                f"conditioning. Proceeding with k-means + prune.")
    return None
