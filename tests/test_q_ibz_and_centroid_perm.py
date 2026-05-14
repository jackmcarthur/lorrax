"""Tests for the C2 sym helpers:

- ``SymMaps.find_irreducible_qpoints``
- ``orbit_syms.compute_centroid_sym_perm``

We don't need a real WFN.h5 — both helpers operate on integer / floating
arrays directly.  We stub ``SymMaps`` minimally (just the attributes the
q-IBZ helper reads).
"""
from __future__ import annotations

import numpy as np
import pytest

from centroid.orbit_syms import compute_centroid_sym_perm
from common.symmetry_maps import SymMaps, find_irreducible_bz_points


# ---------------------------------------------------------------------------
# SymMaps stub — populates the eager q-IBZ attrs that __init__ would set.
# ---------------------------------------------------------------------------

def _sym_stub(kgrid, sym_mats_k):
    """Build a SymMaps-shaped object with the eager q-IBZ tables populated."""
    obj = object.__new__(SymMaps)
    kg = np.asarray(kgrid, dtype=np.int64)
    kx, ky, kz = np.meshgrid(np.arange(kg[0]), np.arange(kg[1]),
                              np.arange(kg[2]), indexing='ij')
    obj.kvecs_asints = np.stack(
        [kx.flatten(), ky.flatten(), kz.flatten()], axis=1)
    obj.sym_mats_k = np.asarray(sym_mats_k, dtype=np.int64)
    obj.irr_idx_q, obj.sym_idx_q, obj.q_irr_kgrid_int = find_irreducible_bz_points(
        obj.kvecs_asints, obj.sym_mats_k, irr_kgrid_int=None,
    )
    _, first_occ = np.unique(obj.irr_idx_q, return_index=True)
    obj.q_irr_full_idx = np.sort(first_occ).astype(np.int32)
    return obj


# ---------------------------------------------------------------------------
# find_irreducible_qpoints
# ---------------------------------------------------------------------------

def test_identity_only_keeps_full_bz():
    """No symmetry → IBZ = full BZ."""
    sym = _sym_stub(kgrid=(2, 2, 2), sym_mats_k=[np.eye(3, dtype=int)])
    q_irr = sym.q_irr_kgrid_int; full_to_irr_idx = sym.irr_idx_q; full_to_irr_sym = sym.sym_idx_q; q_irr_full_idx = sym.q_irr_full_idx
    assert q_irr.shape == (8, 3)
    np.testing.assert_array_equal(full_to_irr_idx, np.arange(8))
    np.testing.assert_array_equal(full_to_irr_sym, np.zeros(8))


def test_inversion_pairs_q_with_neg_q():
    """Inversion + identity on a 2x2x2 grid: q ≡ -q reduces 8 → 5
    orbits (one self-symmetric at Γ, three self-symmetric on the
    boundary, four ±-pairs collapsing to two)."""
    sym = _sym_stub(
        kgrid=(2, 2, 2),
        sym_mats_k=[np.eye(3, dtype=int), -np.eye(3, dtype=int)],
    )
    q_irr = sym.q_irr_kgrid_int; full_to_irr_idx = sym.irr_idx_q; full_to_irr_sym = sym.sym_idx_q; q_irr_full_idx = sym.q_irr_full_idx
    # 2x2x2: q's are {0, 1}³.  Under q ≡ -q (mod 2), -1 = 1, so the
    # full grid is self-symmetric: every q maps to itself.  IBZ = 8.
    assert q_irr.shape == (8, 3)
    np.testing.assert_array_equal(full_to_irr_idx, np.arange(8))
    # Sym 0 (identity) suffices for every q.
    np.testing.assert_array_equal(full_to_irr_sym, np.zeros(8))


def test_inversion_3x3x1_reduces():
    """Inversion + identity on 3x3x1: (0,0,0) self; (1,0,0)↔(2,0,0);
    (0,1,0)↔(0,2,0); (1,1,0)↔(2,2,0); (1,2,0)↔(2,1,0).  9 q's → 5 orbits."""
    sym = _sym_stub(
        kgrid=(3, 3, 1),
        sym_mats_k=[np.eye(3, dtype=int), -np.eye(3, dtype=int)],
    )
    q_irr = sym.q_irr_kgrid_int; full_to_irr_idx = sym.irr_idx_q; full_to_irr_sym = sym.sym_idx_q; q_irr_full_idx = sym.q_irr_full_idx
    assert q_irr.shape == (5, 3)
    # Every full-BZ q must reach its IBZ partner under sym_mats_k[full_to_irr_sym].
    full = sym.kvecs_asints
    kg = np.array([3, 3, 1], dtype=np.int64)
    for iq in range(9):
        s = int(full_to_irr_sym[iq])
        irr_idx = int(full_to_irr_idx[iq])
        recon = (sym.sym_mats_k[s] @ q_irr[irr_idx]) % kg
        np.testing.assert_array_equal(recon, full[iq],
            err_msg=f"q_full[{iq}]={full[iq]} sym {s} q_irr[{irr_idx}]={q_irr[irr_idx]} → {recon}")


def test_q_irr_full_idx_matches_kvecs_asints():
    """q_irr_full_idx points back into kvecs_asints — kvecs_asints[i]
    should equal q_irr[k] for the k-th IBZ q at flat-q index i."""
    sym = _sym_stub(
        kgrid=(3, 3, 1),
        sym_mats_k=[np.eye(3, dtype=int), -np.eye(3, dtype=int)],
    )
    q_irr = sym.q_irr_kgrid_int; q_irr_full_idx = sym.q_irr_full_idx
    assert q_irr_full_idx.shape == (q_irr.shape[0],)
    np.testing.assert_array_equal(
        sym.kvecs_asints[q_irr_full_idx], q_irr)
    # Strictly increasing (np.unique + np.sort).
    assert np.all(np.diff(q_irr_full_idx) > 0)


def test_full_bz_lookup_is_bijective_per_orbit():
    """Every full-BZ q must point to a valid IBZ index AND the
    chosen sym op must actually reconstruct it."""
    sym = _sym_stub(
        kgrid=(4, 4, 1),
        sym_mats_k=[
            np.eye(3, dtype=int),
            np.diag([-1, -1, 1]).astype(int),    # inversion in 2D
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),   # C4
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),   # C4^-1
        ],
    )
    q_irr = sym.q_irr_kgrid_int; full_to_irr_idx = sym.irr_idx_q; full_to_irr_sym = sym.sym_idx_q; q_irr_full_idx = sym.q_irr_full_idx
    full = sym.kvecs_asints
    kg = np.array([4, 4, 1], dtype=np.int64)
    for iq in range(full.shape[0]):
        s = int(full_to_irr_sym[iq])
        irr_idx = int(full_to_irr_idx[iq])
        recon = (sym.sym_mats_k[s] @ q_irr[irr_idx]) % kg
        np.testing.assert_array_equal(recon, full[iq])
    # IBZ should be strictly smaller than full BZ for this sym set.
    assert q_irr.shape[0] < full.shape[0]


# ---------------------------------------------------------------------------
# compute_centroid_sym_perm
# ---------------------------------------------------------------------------

def _orbit_close_centroids(fft_grid, sym_matrices):
    """Build a centroid set closed under the given symmetry group.

    Starts from a seed FFT index, applies every sym op, collects unique
    images.  The result is guaranteed orbit-closed (since the sym group
    is a group).
    """
    sym = np.asarray(sym_matrices, dtype=np.int64)
    Rinv = np.rint(np.linalg.inv(sym)).astype(np.int64)
    fg = np.asarray(fft_grid, dtype=np.int64)
    seeds = np.array([[1, 2, 3], [5, 5, 0], [2, 2, 2]], dtype=np.int64)
    images = []
    for seed in seeds:
        for s_idx in range(sym.shape[0]):
            r_frac = seed.astype(np.float64) / fg
            r_img = r_frac @ Rinv[s_idx].T.astype(np.float64)
            r_img = r_img - np.floor(r_img)
            idx = np.rint(r_img * fg).astype(np.int64) % fg
            images.append(idx)
    arr = np.stack(images)
    # Deduplicate.
    _, first = np.unique(arr, axis=0, return_index=True)
    return arr[np.sort(first)]


def test_centroid_sym_perm_identity_only_is_identity():
    """One-element sym group → π_0 = identity permutation."""
    fft_grid = (8, 8, 8)
    r_mu = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], dtype=np.int32)
    perm = compute_centroid_sym_perm(
        r_mu, sym_matrices=np.eye(3, dtype=int)[None],
        translations=np.zeros((1, 3)),
        fft_grid=fft_grid,
    )
    np.testing.assert_array_equal(perm, np.arange(3)[None])


def test_centroid_sym_perm_under_inversion_closed_set():
    """A centroid set closed under inversion: π_inv must permute it."""
    fft_grid = (8, 8, 8)
    sym = np.stack([np.eye(3, dtype=int),
                    -np.eye(3, dtype=int)], axis=0)
    r_mu = _orbit_close_centroids(fft_grid, sym).astype(np.int32)
    perm = compute_centroid_sym_perm(
        r_mu, sym_matrices=sym,
        translations=np.zeros((2, 3)),
        fft_grid=fft_grid,
    )
    assert perm.shape == (2, r_mu.shape[0])
    # Identity row is just np.arange.
    np.testing.assert_array_equal(perm[0], np.arange(r_mu.shape[0]))
    # Each row is a permutation.
    np.testing.assert_array_equal(np.sort(perm[1]), np.arange(r_mu.shape[0]))
    # Composing twice gives identity (inversion is its own inverse).
    np.testing.assert_array_equal(perm[1][perm[1]], np.arange(r_mu.shape[0]))


def test_centroid_sym_perm_raises_on_open_orbit():
    """A centroid set that is NOT orbit-closed under the sym must
    raise a clear error (this is the main protection for callers
    that pass --no-orbit kmeans output)."""
    fft_grid = (8, 8, 8)
    sym = np.stack([np.eye(3, dtype=int),
                    -np.eye(3, dtype=int)], axis=0)
    # Pick a centroid whose inversion partner is missing.
    r_mu = np.array([[1, 2, 3]], dtype=np.int32)
    with pytest.raises(RuntimeError, match="orbit closure failed"):
        compute_centroid_sym_perm(
            r_mu, sym_matrices=sym,
            translations=np.zeros((2, 3)),
            fft_grid=fft_grid,
        )


def test_centroid_sym_perm_with_nonsymmorphic_tau():
    """Non-symmorphic op {S | τ}: image must include τ shift on the
    FFT grid.  τ × fft_grid must be integer for this to round cleanly."""
    fft_grid = (4, 4, 4)
    # 180° rotation about z + half-translation along x (commensurate
    # with fft_grid[0]=4: τ_x = 0.5 → 2 FFT-grid units).
    S = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=int)
    sym = np.stack([np.eye(3, dtype=int), S], axis=0)
    # BGW convention: translations stored as τ_BGW = 2π · τ_frac.
    tau = np.array([[0.0, 0.0, 0.0],
                    [2 * np.pi * 0.5, 0.0, 0.0]], dtype=np.float64)
    # Closed centroid set: include each seed AND its image under {S|τ}.
    seeds = np.array([[1, 2, 0], [3, 1, 0]], dtype=np.int64)
    images = [seeds]
    Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    fg = np.asarray(fft_grid, dtype=np.int64)
    for seed in seeds:
        r_frac = seed.astype(np.float64) / fg
        r_img = (r_frac @ Rinv.T.astype(np.float64)
                 + np.array([0.5, 0.0, 0.0]))
        r_img = r_img - np.floor(r_img)
        idx = np.rint(r_img * fg).astype(np.int64) % fg
        images.append(idx[None, :])
    r_mu = np.vstack(images).astype(np.int32)
    _, uidx = np.unique(r_mu, axis=0, return_index=True)
    r_mu = r_mu[np.sort(uidx)]
    perm = compute_centroid_sym_perm(
        r_mu, sym_matrices=sym, translations=tau, fft_grid=fft_grid,
    )
    np.testing.assert_array_equal(perm[0], np.arange(r_mu.shape[0]))
    # {S|τ} should be its own inverse here (S² = I, 2τ = 0 mod 1).
    np.testing.assert_array_equal(perm[1][perm[1]], np.arange(r_mu.shape[0]))
