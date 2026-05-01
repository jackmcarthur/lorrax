"""Real-space symmetry helpers for orbit-aware k-means.

Bridges between LORRAX's ``SymMaps`` (which stores rotations and the
BGW-convention fractional translations) and the row-vector fractional
convention used by the kmeans kernels.

Conventions
-----------
For a row-vector fractional position ``r``:

    image_row(r, s) = r @ Rinv[s].T  +  tau[s]                 (mod 1)
    fold_back(δ, s) = δ @ R[s].T

* ``R[s]``   = ``sym.R_grid[s]``      — acts on G-vectors (col convention)
* ``Rinv[s]`` = ``sym.Rinv_grid[s]``  — acts on real-space r (col convention)
* ``tau[s]`` = ``wfn.translations[s] / (2π)`` — BGW sign already baked in.

This is the same convention used by ``sym.validate_atomic_symmetries`` and
``charge_density._symmetrise_density``; see ``symmetry_maps.py:339-345``
for the canonical example.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────
# Build sym table (host)
# ─────────────────────────────────────────────────────────────────────────

def build_real_space_syms(wfn, sym, validate: bool = True):
    """Spatial-only sym data for r-space orbit construction.

    Parameters
    ----------
    wfn, sym : LORRAX WFNReader / SymMaps.
    validate : if True, run ``sym.validate_atomic_symmetries(wfn)`` and
        raise if it returns failures.

    Returns
    -------
    R, Rinv : (n_sym, 3, 3) int32 jax arrays.
    tau     : (n_sym, 3) float64 jax array, BGW sign convention.
    """
    if validate:
        failures = sym.validate_atomic_symmetries(wfn)
        if failures:
            raise RuntimeError(
                f"sym data fails atomic-orbit closure: {failures[:3]}"
            )
    n_sym = int(wfn.ntran)
    R = jnp.asarray(np.asarray(sym.R_grid[:n_sym], dtype=np.int32))
    Rinv = jnp.asarray(np.asarray(sym.Rinv_grid[:n_sym], dtype=np.int32))
    tau = jnp.asarray(
        np.asarray(wfn.translations[:n_sym], dtype=np.float64) / (2.0 * np.pi)
    )
    return R, Rinv, tau


def identity_syms():
    """Dummy 1-element sym table — no symmetry, for non-orbit-aware callers."""
    return (jnp.eye(3, dtype=jnp.int32)[None],
            jnp.eye(3, dtype=jnp.int32)[None],
            jnp.zeros((1, 3), dtype=jnp.float64))


# ─────────────────────────────────────────────────────────────────────────
# Orbit utilities
# ─────────────────────────────────────────────────────────────────────────

@jax.jit
def orbit_images(reps: jnp.ndarray,
                 Rinv: jnp.ndarray,
                 tau: jnp.ndarray) -> jnp.ndarray:
    """Apply every sym op to every rep. Returns (n_sym, n_rep, 3) mod 1."""
    return jax.vmap(lambda Ri, t: (reps @ Ri.T + t) % 1.0)(Rinv, tau)


_CANON_INV = jnp.int64(10**12)
"""Integer precision for canonicalisation lex keys: 12 digits → 1 ppm
resolution on fractional coords ([0, 1)). Coarser than fp64 noise (1e-15)
by ~3 orders of magnitude — enough to absorb single-op drift without
collapsing distinct orbit members."""


def _orbit_lex_winner(images: jnp.ndarray) -> jnp.ndarray:
    """For each rep, the index s* of the lex-smallest orbit image.

    images: (n_sym, n_rep, 3) fp64 in [0, 1).
    Returns: (n_rep,) int32 — index in [0, n_sym).

    Uses integer lex ordering (`jnp.lexsort` on int64 triples) instead of a
    floating composite key. Robust against fp noise; deterministic.
    """
    keys = jnp.round(images * _CANON_INV).astype(jnp.int64)  # (n_sym, n_rep, 3)
    # lexsort: rightmost key is primary. We want primary = x, secondary = y,
    # tertiary = z, so pass (z, y, x). Result is sorted indices along axis 0.
    n_rep = images.shape[1]
    def winner_for_rep(i):
        order = jnp.lexsort((keys[:, i, 2], keys[:, i, 1], keys[:, i, 0]))
        return order[0].astype(jnp.int32)
    return jax.vmap(winner_for_rep)(jnp.arange(n_rep))


@jax.jit
def canonicalize_orbit(reps: jnp.ndarray,
                       Rinv: jnp.ndarray,
                       tau: jnp.ndarray) -> jnp.ndarray:
    """Map each rep to the lex-smallest member of its orbit (integer-key
    lex). Idempotent. Static shape (n_rep, 3)."""
    images = orbit_images(reps, Rinv, tau)                 # (n_sym, n_rep, 3)
    best_s = _orbit_lex_winner(images)                      # (n_rep,)
    return jnp.take_along_axis(
        images, best_s[None, :, None], axis=0
    )[0]


def snap_orbits_to_grid(reps_frac: np.ndarray,
                        fft_grid: tuple[int, int, int],
                        Rinv: jnp.ndarray,
                        tau: jnp.ndarray,
                        ) -> tuple[np.ndarray, np.ndarray, int]:
    """Snap fractional reps to the FFT grid, canonicalise to the
    lex-smallest on-grid orbit member, then deduplicate **by orbit**.

    Two reps that snap to different points but share an orbit are
    counted as duplicates here (the older ``snap_centroids_to_grid``
    only catches literal-point duplicates).

    Requires the FFT grid to be commensurate with every τ in the sym
    table — i.e. ``(τ × fft_grid)`` must be integer to roundoff. This
    is checked by ``build_real_space_syms`` indirectly via
    ``validate_atomic_symmetries`` (which would fail loudly if the
    atom basis didn't close on the grid).

    Returns
    -------
    indices : (n_unique, 3) int — canonical orbit reps as FFT indices.
    frac    : (n_unique, 3) fp — same, as fractional coords.
    n_dups  : number of orbit-duplicates dropped.
    """
    indices = np.round(reps_frac * np.array(fft_grid)).astype(int) % fft_grid
    snapped = indices.astype(float) / fft_grid
    canon = np.asarray(canonicalize_orbit(jnp.asarray(snapped), Rinv, tau))
    canon_idx = np.round(canon * np.array(fft_grid)).astype(int) % fft_grid
    unique_idx = np.unique(canon_idx, axis=0)
    n_dups = canon_idx.shape[0] - unique_idx.shape[0]
    if n_dups > 0:
        print(f"snap_orbits_to_grid: {n_dups} orbit duplicates "
              f"({canon_idx.shape[0]} → {unique_idx.shape[0]} unique orbits)")
    return unique_idx, unique_idx.astype(float) / fft_grid, n_dups


def unfold_orbit_unique_with_id(reps_np: np.ndarray,
                                Rinv: np.ndarray,
                                tau: np.ndarray,
                                tol: float = 1e-6,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Unfold reps into all distinct orbit images; also return ``orbit_id``
    — a per-candidate integer that's the same for two candidates iff they
    lie in the same **physical** orbit under the WFN's sym group.

    orbit_id is the integer encoding of each candidate's canonical
    (lex-smallest) orbit member, then run through ``np.unique`` to make
    them dense [0, n_orbits). This is the right key for the orbit-aware
    pivoted Cholesky: two kmeans reps that drifted into the same physical
    orbit during Lloyd get identical orbit_ids, so PC sees them as one
    orbit (not two).
    """
    Rinv = np.asarray(Rinv); tau = np.asarray(tau)
    inv = np.int64(round(1.0 / tol))         # int! avoid fp64-precision loss at 1e18

    # 1. Unfold + dedupe at fp tolerance.
    images = (np.einsum('ri,sji->srj', reps_np, Rinv) + tau[:, None, :]) % 1.0
    flat = images.reshape(-1, 3)
    keys = np.round(flat * inv).astype(np.int64) % inv
    _, first_idx = np.unique(keys, axis=0, return_index=True)
    flat = flat[np.sort(first_idx)]

    # 2. For each unique candidate, compute its canonical (lex-min) orbit
    #    member, then dense-encode the canonical triples as orbit_id.
    cand_imgs = (np.einsum('ci,sji->scj', flat, Rinv) + tau[:, None, :]) % 1.0
    cand_keys = np.round(cand_imgs * inv).astype(np.int64) % inv     # (n_sym, n, 3)
    # Lex via np.lexsort on (z, y, x) — primary key first in argument order
    # is leftmost; lexsort treats the LAST key as primary. So pass (z, y, x)
    # so x is primary, y secondary, z tertiary. Returns sorted indices along axis 0.
    n = flat.shape[0]
    s_idx = np.array([
        np.lexsort((cand_keys[:, i, 2], cand_keys[:, i, 1], cand_keys[:, i, 0]))[0]
        for i in range(n)
    ])
    canonical_keys = cand_keys[s_idx, np.arange(n)]                  # (n, 3)
    _, orbit_id = np.unique(canonical_keys, axis=0, return_inverse=True)
    return flat, orbit_id.astype(np.int32)


def unfold_orbit_unique(reps_np, Rinv, tau, tol=1e-6) -> np.ndarray:
    """Backwards-compatible wrapper: drops the orbit_id second return."""
    flat, _ = unfold_orbit_unique_with_id(reps_np, Rinv, tau, tol=tol)
    return flat


