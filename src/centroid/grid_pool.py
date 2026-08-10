"""The deterministic candidate pool: every point of the real-space grid.

WHY THIS EXISTS.  The ISDF interpolation points have always been chosen by
``weighted_kmeans_jax`` — a seeded Lloyd iteration — and the pivoted-Cholesky
stage in :mod:`centroid.pivoted_cholesky` only ever *pruned* what that draw
happened to produce.  So the whole selector was a lottery: the same deck, the
same grid, the same count and the next seed move Sigma_x by tens of meV, and
the standing recipe on the Si cross-code row is "draw five sets and keep the
one with the most independent directions".  Best-of-k is a selection procedure,
not a method — it cannot be quoted in a comparison table without also quoting
the distribution it was drawn from.

This module removes the draw.  The candidate pool becomes the **entire FFT
grid**, in canonical C-order, and the greedy pivoted Cholesky on the
pair-density Gram picks the interpolation points out of it directly.  That is
the standard rank-revealing ISDF selector (diagonal-pivoted Cholesky on the
pair-density Gram; the QRCP/DEIM route differs only in which factorization
reveals the rank).  There is no RNG on this path at all — not in the pool, not
in the ordering, not in the tie-break — so two runs of the same deck return the
same point set, bit for bit.

WHAT IS DETERMINISTIC, STATED.

1. **The pool.**  ``full_grid_candidates`` enumerates every grid point once, in
   C-order over ``(ix, iy, iz)``.  No weight, no threshold, no sampling.
2. **The orbits.**  ``grid_orbit_ids`` labels each point by the *lex-smallest
   flat index* in its orbit under the real-space action ``r' = Rinv·r + tau``
   — the same BGW convention ``unfold_orbit_unique_with_id`` and
   ``centroid_source_map_and_wrap`` use — computed in integer grid indices
   rather than at a floating tolerance, so no rounding decision enters.
3. **The pivot order.**  The greedy select takes ``argmax`` of the residual
   diagonal, which resolves a tie to the *lowest candidate index*; the sharded
   kernel breaks a cross-device tie to the *lowest device index*.  Both are
   already contracts of :mod:`centroid.pivoted_cholesky` and neither consults
   an RNG.
4. **Orbit closure** is applied exactly as the k-means path applies it: one
   pivot per orbit, and the delivered set is the union of the picked orbits.

The cost is the Gram, which is ``O(M^2)`` in the pool size, and the pool is now
the grid rather than ``1.5 x N_mu`` points.  On the Si 4x4x4 24^3 deck that is
M = 13824 instead of M = 1992 — about fifty times the Gram work, three or four
minutes instead of ten seconds, and still cheaper than the five k-means draws
plus five Sigma evaluations the lottery would otherwise cost.  The Gram build
is column-blocked (see ``build_gram_q0_via_loadwfns``), which is what keeps the
(nk, ns, ns, M, cols) pair tensors inside one device's budget.

Pure NumPy on purpose: nothing here needs a device, so it can be tested without
one.
"""
from __future__ import annotations

import numpy as np


def full_grid_candidates(fft_grid) -> np.ndarray:
    """Every point of the FFT grid, once, in C-order.

    Returns ``(M, 3) int64`` integer grid indices with
    ``M = nx * ny * nz``.  Row ``m`` is the point whose flat C-order index
    is ``m``, so the pool's order is a property of the grid and of nothing
    else.
    """
    nx, ny, nz = (int(v) for v in fft_grid)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"degenerate FFT grid {fft_grid!r}")
    ix, iy, iz = np.meshgrid(
        np.arange(nx, dtype=np.int64),
        np.arange(ny, dtype=np.int64),
        np.arange(nz, dtype=np.int64),
        indexing="ij",
    )
    return np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)


def _flat(idx: np.ndarray, fft_grid) -> np.ndarray:
    """C-order flat index of integer grid triples."""
    nx, ny, nz = (int(v) for v in fft_grid)
    return ((idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2])


def grid_images(cand_idx: np.ndarray, Rinv, tau, fft_grid,
                *, tol: float = 1e-8) -> np.ndarray:
    """All ``n_sym`` images of each grid point, as integer grid indices.

    Applies the BGW real-space action ``r' = Rinv · r + tau`` (row-vector
    form ``r @ Rinv.T + tau``) in fractional coordinates and converts back
    to grid indices.  REFUSES when an image does not land on a grid point,
    because that is the orbit-closure failure mode — a grid the symmetry
    group does not map to itself, or a fractional translation that is not
    commensurate with it — and it must not be rounded away silently.

    Returns ``(n_sym, M, 3) int64``.
    """
    grid = np.asarray([int(v) for v in fft_grid], dtype=np.float64)
    Rinv = np.asarray(Rinv)
    tau = np.asarray(tau, dtype=np.float64)
    frac = np.asarray(cand_idx, dtype=np.float64) / grid           # (M, 3)
    # r @ Rinv.T + tau, per op; then back into grid units.
    img = (np.einsum("mi,sji->smj", frac, Rinv) + tau[:, None, :]) * grid
    rounded = np.rint(img)
    resid = float(np.max(np.abs(img - rounded))) if img.size else 0.0
    if resid > tol:
        raise ValueError(
            f"symmetry images do not land on the FFT grid {tuple(fft_grid)}: "
            f"worst departure {resid:.3e} > {tol:g} grid units.  Either the "
            f"grid is not invariant under the group being used for orbit "
            f"closure, or a fractional translation tau is not commensurate "
            f"with it.  Rounding this away would produce a set that is NOT "
            f"orbit-closed while still looking like one."
        )
    return (rounded.astype(np.int64) % np.asarray(
        [int(v) for v in fft_grid], dtype=np.int64))


def grid_orbit_ids(cand_idx: np.ndarray, Rinv, tau, fft_grid,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Dense orbit labels for a set of grid points, computed on integers.

    Two points get the same label iff they lie in the same orbit of the
    real-space action ``r' = Rinv·r + tau``.  The label of an orbit is
    derived from its **lex-smallest flat C-order index**, so the labelling
    is a pure function of the grid and the symmetry table — no tolerance,
    no ordering choice, no RNG.

    ``cand_idx`` must be orbit-closed (the full grid is, by construction);
    this refuses otherwise, because a pool that is missing orbit members
    cannot deliver an orbit-closed centroid set no matter what the select
    picks.

    Returns ``(orbit_id, orbit_sizes)`` — ``(M,) int32`` dense labels in
    ``[0, n_orbits)`` assigned in increasing canonical flat index, and
    ``(n_orbits,) int64`` member counts.
    """
    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    images = grid_images(cand_idx, Rinv, tau, fft_grid)             # (S, M, 3)
    flat_img = _flat(images, fft_grid)                              # (S, M)
    pool = _flat(cand_idx, fft_grid)                                # (M,)

    missing = np.setdiff1d(np.unique(flat_img), pool, assume_unique=False)
    if missing.size:
        raise ValueError(
            f"candidate pool is not orbit-closed: {missing.size} symmetry "
            f"images of pool points are not themselves in the pool (first "
            f"missing flat index {int(missing[0])}).  Orbit closure has to "
            f"hold on the POOL for it to hold on the selection."
        )

    canonical = flat_img.min(axis=0)                                # (M,)
    uniq, orbit_id, counts = np.unique(
        canonical, return_inverse=True, return_counts=True)
    return orbit_id.astype(np.int32), counts.astype(np.int64)


def describe_pool(cand_idx: np.ndarray, orbit_id: np.ndarray | None,
                  fft_grid) -> str:
    """One log line naming the pool, so the provenance is in the artifact."""
    m = int(np.asarray(cand_idx).shape[0])
    gram_gb = (m * m * 16) / 1e9
    if orbit_id is None:
        return (f"[grid pool] DETERMINISTIC: all {m} points of the "
                f"{tuple(int(v) for v in fft_grid)} grid, C-order, no "
                f"symmetry reduction.  Gram is {gram_gb:.2f} GB.")
    sizes = np.bincount(np.asarray(orbit_id))
    return (f"[grid pool] DETERMINISTIC: all {m} points of the "
            f"{tuple(int(v) for v in fft_grid)} grid, C-order, in "
            f"{sizes.size} orbits (sizes {int(sizes.min())}..."
            f"{int(sizes.max())}, mean {sizes.mean():.1f}).  "
            f"Gram is {gram_gb:.2f} GB.  No RNG on this path.")
