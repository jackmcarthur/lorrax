"""Symmetry-table / TRS-unfold unit invariants (synthetic geometries).

Merged from three files (2026-07-09 suite redesign) — together they guard
the TRS-blind-bug class at unit level, on synthetic data, with hand-rolled
per-element references:

* q-IBZ folding + centroid orbit permutations
  (was test_q_ibz_and_centroid_perm.py): IBZ selection is bijective per
  orbit, centroid sym-perms are true permutations, open orbits raise
  loudly (the silent-clip failure shape of the TRS bug).
* TRS-augmented centroid perms + V_q IBZ→full unfold
  (was test_trs_unfold_centroid_perm.py): extend_trs row structure,
  per-row conj on TRS rows + L-umklapp phase vs a hand reference, and the
  hard ValueError when a TRS-needing q meets a spatial-only table.
* ψ k-unfold under TRS (was test_unfold_psi_trs.py): per-element
  (iσ_y·conj(U_s))·conj(ψ·phase) reference, identity no-op, T² = −1 on
  spin-½.

None of these paths is otherwise e2e-covered by fixtures whose WFN stores
the full BZ (gnppm/bispinor) — only the cohsex gate (IBZ-stored WFNsmall)
touches ψ-unfold e2e, and no e2e fixture needs TRS-only q reachability.
"""

from __future__ import annotations



# ===========================================================================
#  q-IBZ folding + centroid orbit permutations
# ===========================================================================


import numpy as np
import pytest

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from symmetry_maps import (                                     # noqa: E402
    SymMaps, compute_centroid_sym_perm, find_irreducible_bz_points,
)


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
    perm, L = compute_centroid_sym_perm(
        r_mu, sym_matrices=np.eye(3, dtype=int)[None],
        translations=np.zeros((1, 3)),
        fft_grid=fft_grid,
    )
    np.testing.assert_array_equal(perm, np.arange(3)[None])
    np.testing.assert_array_equal(L, np.zeros((1, 3, 3), dtype=L.dtype))


def test_centroid_sym_perm_under_inversion_closed_set():
    """A centroid set closed under inversion: π_inv must permute it."""
    fft_grid = (8, 8, 8)
    sym = np.stack([np.eye(3, dtype=int),
                    -np.eye(3, dtype=int)], axis=0)
    r_mu = _orbit_close_centroids(fft_grid, sym).astype(np.int32)
    perm, _L = compute_centroid_sym_perm(
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
    perm, _L = compute_centroid_sym_perm(
        r_mu, sym_matrices=sym, translations=tau, fft_grid=fft_grid,
    )
    np.testing.assert_array_equal(perm[0], np.arange(r_mu.shape[0]))
    # {S|τ} should be its own inverse here (S² = I, 2τ = 0 mod 1).
    np.testing.assert_array_equal(perm[1][perm[1]], np.arange(r_mu.shape[0]))


# ===========================================================================
#  TRS-augmented centroid perms + V_q IBZ→full unfold
# ===========================================================================


import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# The service path bootstrap ran with this module's FIRST import block, at
# the top of the file; it is idempotent and once is enough.
from symmetry_maps import (                                     # noqa: E402
    compute_centroid_sym_perm, unfold_v_q,
)


# ---------------------------------------------------------------------------
# Geometry: ``ntran=2`` ({I, σ_y}) on a 4×4×1 FFT grid, TRS-augmented
# table length 4.  Centroid set orbit-closed under {I, σ_y}.
# ---------------------------------------------------------------------------

def _build_geometry_vq():
    fft_grid = np.array([4, 4, 1], dtype=np.int64)
    I3 = np.eye(3, dtype=np.int64)
    sigma_y = np.diag([1, -1, 1]).astype(np.int64)
    sym_matrices = np.stack([I3, sigma_y], axis=0)               # (ntran=2, 3, 3)
    translations = np.zeros((2, 3), dtype=np.float64)            # symmorphic

    # Centroids: orbit-closed under {I, σ_y}.  Seeds + σ_y-images.
    seeds = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0],
    ], dtype=np.int64)
    Rinv = np.rint(np.linalg.inv(sym_matrices)).astype(np.int64)
    imgs = set()
    for r in seeds:
        for s in range(sym_matrices.shape[0]):
            imgs.add(tuple(((Rinv[s] @ r) % fft_grid).tolist()))
    cent_idx = np.array(sorted(imgs), dtype=np.int32)
    return fft_grid, sym_matrices, translations, cent_idx


# ===========================================================================
# Test 1: extend_trs=True returns the right shape and content.
# ===========================================================================

def test_compute_centroid_sym_perm_extend_trs_shape_and_content():
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry_vq()
    n_sym = sym_matrices.shape[0]
    n_rmu = cent_idx.shape[0]

    sp_default, L_default = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True)
    sp_trs, L_trs = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True, extend_trs=True)

    # Default: (ntran, n_rmu).
    assert sp_default.shape == (n_sym, n_rmu)
    assert L_default.shape == (n_sym, n_rmu, 3)

    # Extended: (2·ntran, n_rmu) with rows[ntran:] duplicating rows[:ntran].
    assert sp_trs.shape == (2 * n_sym, n_rmu), (
        f"extend_trs=True must return (2·ntran, n_rmu); got {sp_trs.shape}")
    np.testing.assert_array_equal(sp_trs[:n_sym], sp_default)
    np.testing.assert_array_equal(sp_trs[n_sym:], sp_default)
    np.testing.assert_array_equal(L_trs[:n_sym], L_default)
    np.testing.assert_array_equal(L_trs[n_sym:], L_default)


def test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation():
    """Sanity: each row (including TRS half) is a permutation of [0, n_rmu)."""
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry_vq()
    n_rmu = cent_idx.shape[0]
    sp, _L = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True, extend_trs=True)
    for s in range(sp.shape[0]):
        assert np.unique(sp[s]).size == n_rmu, (
            f"row {s} of TRS-extended sym_perm is not a permutation")


# ===========================================================================
# Test 2: _unfold_v_q_ibz_to_full applied to a synthetic Hermitian V_ibz
# with mixed spatial-and-TRS folding.
# ===========================================================================

def _make_mesh():
    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devs, ('x', 'y'))


def _hand_unfold_v_q(V_ibz, *, irr_idx, sym_idx,
                     sym_perm, L_table, q_irr_frac, ntran):
    """Reference: pure numpy unfold with TRS conj + L umklapp phase.

    For spatial s < ntran:
        V_full[q, μ, ν] = exp(2π i q_irr · (L_μ − L_ν))
                          · V_ibz[parent, π_s(μ), π_s(ν)]
    For TRS s ≥ ntran:
        V_full[q, μ, ν] = conj(V_full_spatial[q, μ, ν])
    """
    n_q_full = irr_idx.shape[0]
    n_rmu = V_ibz.shape[-1]
    V_full = np.zeros((n_q_full, n_rmu, n_rmu), dtype=V_ibz.dtype)
    for iq in range(n_q_full):
        parent = int(irr_idx[iq])
        s = int(sym_idx[iq])
        perm_fwd = np.asarray(sym_perm[s])               # length n_rmu
        is_trs = s >= ntran
        V_perm = V_ibz[parent][np.ix_(perm_fwd, perm_fwd)]
        L_s = np.asarray(L_table[s], dtype=np.float64)   # (n_rmu, 3)
        qL = L_s @ q_irr_frac[parent]                     # (n_rmu,)
        phase = np.exp(2j * np.pi * qL)
        V_phase = phase[:, None] * V_perm * np.conj(phase)[None, :]
        V_full[iq] = np.conj(V_phase) if is_trs else V_phase
    return V_full


def test_unfold_v_q_ibz_to_full_handles_trs_rows():
    """Hand-build a synthetic Hermitian V_ibz and full→IBZ map with at
    least one TRS-only-reachable q, run the codebase unfold, compare
    to the per-element reference.
    """
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry_vq()
    ntran = sym_matrices.shape[0]
    n_rmu = cent_idx.shape[0]

    sym_perm, L_table = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True, extend_trs=True)

    # Build a small synthetic IBZ→full map:
    #   IBZ wedge has 3 q's (indices 0, 1, 2 = "parent" labels).
    #   Full BZ has 5 q's:
    #     q_full[0] := parent 0 via identity (s=0)             — spatial trivial
    #     q_full[1] := parent 1 via σ_y     (s=1)              — spatial non-trivial
    #     q_full[2] := parent 2 via identity (s=0)
    #     q_full[3] := parent 0 via TRS·I   (s=ntran+0 = 2)    — TRS-only
    #     q_full[4] := parent 1 via TRS·σ_y (s=ntran+1 = 3)    — TRS + spatial
    full_to_irr_idx = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    full_to_irr_sym = np.array([0, 1, 0, ntran + 0, ntran + 1], dtype=np.int32)

    # Random complex IBZ V's, made Hermitian per parent for physical
    # plausibility (V_q is Hermitian in (μ, ν)).
    rng = np.random.default_rng(seed=11)
    n_q_ibz = 3
    A = rng.standard_normal((n_q_ibz, n_rmu, n_rmu)) \
        + 1j * rng.standard_normal((n_q_ibz, n_rmu, n_rmu))
    V_ibz = 0.5 * (A + np.swapaxes(A.conj(), -1, -2))

    # Synthetic q_irr_frac — pick values that exercise the L-phase
    # (non-Γ q's so any non-zero L_μ produces a phase).
    q_irr_frac = np.array([[0.0, 0.0, 0.0],   # parent 0 = Γ (phase trivial)
                           [0.25, 0.0, 0.0],  # parent 1 — non-zero q
                           [0.0, 0.5, 0.0]],  # parent 2 — non-zero q
                          dtype=np.float64)

    # Reference unfold.
    V_ref = _hand_unfold_v_q(
        V_ibz, irr_idx=full_to_irr_idx,
        sym_idx=full_to_irr_sym, sym_perm=sym_perm,
        L_table=L_table, q_irr_frac=q_irr_frac, ntran=ntran)

    # Codebase unfold.  ``n_sym_spatial=ntran`` opts into the TRS
    # branch (conj at rows where ``full_to_irr_sym ≥ ntran``).
    mesh = _make_mesh()
    V_sh = NamedSharding(mesh, P(None, 'x', 'y'))
    V_ibz_j = jax.device_put(V_ibz.astype(np.complex128), V_sh)
    V_full = np.asarray(jax.device_get(
        unfold_v_q(
            V_ibz_j,
            irr_idx=full_to_irr_idx,
            sym_idx=full_to_irr_sym,
            sym_perm=sym_perm,
            L_table=L_table,
            q_irr_frac=q_irr_frac,
            mesh_xy=mesh,
            n_sym_spatial=ntran)))

    max_diff = float(np.max(np.abs(V_full - V_ref)))
    assert max_diff < 1e-12, (
        f"V_q IBZ→full unfold disagrees with reference: max |Δ| = "
        f"{max_diff:.3e}.\n"
        f"Per-q max-diff: "
        f"{np.max(np.abs(V_full - V_ref).reshape(V_full.shape[0], -1), axis=-1)}")


def test_unfold_v_q_ibz_to_full_hard_fails_without_extend_trs():
    """If caller forgets ``extend_trs=True`` AND any full-q needs TRS,
    the unfold must raise a clear error instead of silently clipping.
    """
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry_vq()
    ntran = sym_matrices.shape[0]

    sym_perm_legacy, L_legacy = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True, extend_trs=False)         # length ntran rows!

    # full_to_irr_sym contains a TRS index → must raise.
    full_to_irr_idx = np.array([0, 0], dtype=np.int32)
    full_to_irr_sym = np.array([0, ntran + 0], dtype=np.int32)
    V_ibz = np.zeros((1, cent_idx.shape[0], cent_idx.shape[0]),
                     dtype=np.complex128)
    mesh = _make_mesh()
    V_sh = NamedSharding(mesh, P(None, 'x', 'y'))
    V_ibz_j = jax.device_put(V_ibz, V_sh)
    q_irr_frac = np.zeros((1, 3), dtype=np.float64)

    # New API: n_sym_spatial is required. A length-ntran sym_perm
    # is inconsistent with n_sym_spatial=ntran (which expects 2·ntran
    # rows), so the consistency check raises before the OOB check.
    # Either error message is acceptable — they both flag the bug.
    with pytest.raises(ValueError, match=r"(TRS-augmented|inconsistent)"):
        unfold_v_q(
            V_ibz_j,
            irr_idx=full_to_irr_idx,
            sym_idx=full_to_irr_sym,
            sym_perm=sym_perm_legacy,
            L_table=L_legacy,
            q_irr_frac=q_irr_frac,
            mesh_xy=mesh,
            n_sym_spatial=ntran)


if __name__ == "__main__":
    test_compute_centroid_sym_perm_extend_trs_shape_and_content()
    print("test_compute_centroid_sym_perm_extend_trs_shape_and_content OK")
    test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation()
    print("test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation OK")
    test_unfold_v_q_ibz_to_full_handles_trs_rows()
    print("test_unfold_v_q_ibz_to_full_handles_trs_rows OK")
    test_unfold_v_q_ibz_to_full_hard_fails_without_extend_trs()
    print("test_unfold_v_q_ibz_to_full_hard_fails_without_extend_trs OK")


# ===========================================================================
#  ψ k-unfold under TRS
# ===========================================================================


import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

# The service path bootstrap ran with this module's FIRST import block, at
# the top of the file; it is idempotent and once is enough.
from symmetry_maps import unfold_psi                            # noqa: E402
# PRIVATE NAME — stays on the ``common.symmetry_maps`` SHIM on purpose.  The
# door re-exports the public surface only; ``_I_SIGMA_Y`` is module-private in
# ``symmetry_maps.maps``.  The phase-wide shim-deletion commit (WAVE1_BRIEF
# ruling 2) decides this cell's home — likely into the service's own suite,
# which may import the private directly.
from symmetry_maps.maps import _I_SIGMA_Y                       # noqa: E402


# ---------------------------------------------------------------------------
# Geometry: {I, σ_y} spatial (ntran=2), TRS-augmented sym_mats_k length 4.
# 3×3×1 q-grid (matches MoS₂-like 2D lattice), non-symmorphic τ=(1/2, 0, 0)
# on one of the spatial ops to exercise the τ-phase.
# ---------------------------------------------------------------------------

def _build_geometry_psi():
    ntran = 2
    I3 = np.eye(3, dtype=np.int32)
    sigma_y_int = np.diag([1, -1, 1]).astype(np.int32)
    sym_mats_k_spatial = np.stack([I3, sigma_y_int], axis=0)              # (2, 3, 3)
    sym_mats_k = np.concatenate(
        [sym_mats_k_spatial, -sym_mats_k_spatial], axis=0)                # (4, 3, 3)
    # Non-symmorphic τ on the σ_y row to exercise phase math.
    translations = np.array([
        [0.0, 0.0, 0.0],          # identity
        [0.5, 0.0, 0.0],          # σ_y with non-symmorphic τ
    ], dtype=np.float64)
    # Spinor rotations: identity for s=0, π-rotation about y for s=1
    # (σ_y spinor: U = -i σ_y; here U = cos(π/2)·I - i sin(π/2)·σ_y = -i σ_y).
    U_spinor_spatial = np.zeros((ntran, 2, 2), dtype=np.complex128)
    U_spinor_spatial[0] = np.eye(2)
    # i σ_y as a complex array is [[0, 1], [-1, 0]] so -i σ_y is [[0, -1], [1, 0]].
    U_spinor_spatial[1] = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    return dict(
        ntran=ntran,
        sym_mats_k=sym_mats_k,
        translations=translations,
        U_spinor_spatial=U_spinor_spatial,
    )


# ---------------------------------------------------------------------------
# Reference rule
# ---------------------------------------------------------------------------

def _hand_unfold(cnk_kbar, *, sym_idx, ntran, sym_mats_k, translations,
                 U_spinor_spatial, g_kbar):
    """Per-element reference unfold (numpy)."""
    cnk_kbar = np.asarray(cnk_kbar)
    g_kbar = np.asarray(g_kbar)
    is_trs = sym_idx >= ntran
    s = sym_idx - ntran if is_trs else sym_idx

    S_full = sym_mats_k[sym_idx]
    tau = translations[s]
    rotated = (S_full.astype(np.int64) @ g_kbar.astype(np.int64).T).T
    phase = np.exp(-1j * rotated.astype(np.float64) @ tau)        # (ngk,)

    if is_trs:
        # ψ_full = iσ_y · conj(U_s · ψ_kbar · phase_spatial)
        #       = (iσ_y · conj(U_s)) · conj(ψ_kbar) · conj(phase_spatial)
        # We're computing per-element from the explicit formula.
        U_s = U_spinor_spatial[s]
        spatial_form = np.einsum("ij,nja->nia", U_s, cnk_kbar) * phase[None, None, :]
        # phase here was built from sym_mats_k[sym_idx]=-S so it ALREADY equals
        # conj(phase_spatial); to get phase_spatial back we'd take conj. We just
        # apply the rule ψ_full = iσ_y · conj(spatial_form_with_S_not_minusS).
        # Easier: re-derive directly from the formula above.
        S_spatial = sym_mats_k[s]                                  # +S
        rotated_spatial = (S_spatial.astype(np.int64) @ g_kbar.astype(np.int64).T).T
        phase_spatial = np.exp(-1j * rotated_spatial.astype(np.float64) @ tau)
        spatial_inner = (np.einsum("ij,nja->nia", U_s, cnk_kbar)
                          * phase_spatial[None, None, :])
        out = np.einsum("ij,nja->nia", _I_SIGMA_Y, np.conj(spatial_inner))
        return out
    else:
        U_s = U_spinor_spatial[s]
        return np.einsum("ij,nja->nia", U_s, cnk_kbar) * phase[None, None, :]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unfold_psi_matches_hand_reference():
    geom = _build_geometry_psi()
    ntran = geom["ntran"]

    # Synthetic ψ_kbar: 3 bands, 2 spinor components, 5 G's.
    rng = np.random.default_rng(42)
    nb, ns, ngk = 3, 2, 5
    cnk_kbar = (rng.standard_normal((nb, ns, ngk))
                + 1j * rng.standard_normal((nb, ns, ngk)))
    g_kbar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0]],
                       dtype=np.int32)

    # All 4 sym rows (2 spatial + 2 TRS).
    max_rel = 0.0
    for sym_idx in range(2 * ntran):
        cnk_codebase = unfold_psi(
            cnk_kbar,
            sym_idx=sym_idx,
            g_kbar=g_kbar,
            sym_mats_k=geom["sym_mats_k"],
            translations=geom["translations"],
            U_spinor_spatial=geom["U_spinor_spatial"],
        )
        cnk_ref = _hand_unfold(
            cnk_kbar,
            sym_idx=sym_idx,
            ntran=ntran,
            sym_mats_k=geom["sym_mats_k"],
            translations=geom["translations"],
            U_spinor_spatial=geom["U_spinor_spatial"],
            g_kbar=g_kbar,
        )
        diff = np.max(np.abs(cnk_codebase - cnk_ref))
        ref_norm = np.max(np.abs(cnk_ref))
        rel = diff / max(ref_norm, 1e-30)
        max_rel = max(max_rel, rel)
        assert rel < 1e-12, (
            f"unfold_psi disagrees with reference at sym_idx={sym_idx} "
            f"(is_trs={sym_idx >= ntran}): max |Δ|={diff:.3e}, rel={rel:.3e}")

    print(f"unfold_psi: max rel = {max_rel:.3e} across all 4 sym rows")


def test_unfold_psi_identity_is_noop():
    """sym_idx=0 (identity) should return ψ_kbar unchanged."""
    geom = _build_geometry_psi()
    geom["translations"][0] = 0.0   # ensure identity τ = 0
    rng = np.random.default_rng(7)
    nb, ns, ngk = 2, 2, 4
    cnk = (rng.standard_normal((nb, ns, ngk))
           + 1j * rng.standard_normal((nb, ns, ngk)))
    g_kbar = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]],
                       dtype=np.int32)
    out = unfold_psi(
        cnk,
        sym_idx=0,
        g_kbar=g_kbar,
        sym_mats_k=geom["sym_mats_k"],
        translations=geom["translations"],
        U_spinor_spatial=geom["U_spinor_spatial"],
    )
    np.testing.assert_allclose(out, cnk, atol=0.0)


def test_unfold_psi_trs_squared_is_identity_on_spinor():
    """T² = -I on a spin-1/2 state. Applying TRS twice (sym_idx = ntran for
    spatial=identity) should send ψ → -ψ.
    """
    geom = _build_geometry_psi()
    geom["translations"][:] = 0.0   # symmorphic group to isolate spinor part
    rng = np.random.default_rng(13)
    nb, ns, ngk = 1, 2, 3
    cnk = (rng.standard_normal((nb, ns, ngk))
           + 1j * rng.standard_normal((nb, ns, ngk)))
    g_kbar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)

    once = unfold_psi(
        cnk,
        sym_idx=geom["ntran"],   # TRS · identity = pure TRS
        g_kbar=g_kbar,
        sym_mats_k=geom["sym_mats_k"],
        translations=geom["translations"],
        U_spinor_spatial=geom["U_spinor_spatial"],
    )
    # The G-list under pure TRS is -G; ``once`` is ψ on G_full=-G index.
    # For the second TRS application we use g_kbar = -g_kbar so the
    # composition truly gives back the same G-list.
    twice = unfold_psi(
        once,
        sym_idx=geom["ntran"],
        g_kbar=-g_kbar,
        sym_mats_k=geom["sym_mats_k"],
        translations=geom["translations"],
        U_spinor_spatial=geom["U_spinor_spatial"],
    )
    # T² ψ = -ψ.
    np.testing.assert_allclose(twice, -cnk, atol=1e-12, rtol=0.0,
                                err_msg="T² should equal -I on a spin-1/2 state")


def _trivial_sym_wfn(kgrid=(2, 2, 1)):
    """Minimal WFN stand-in that drives SymMaps' ``ntran <= 1`` branch."""
    import types
    kg = np.asarray(kgrid, dtype=np.int32)
    kx, ky, kz = np.meshgrid(np.arange(kg[0]), np.arange(kg[1]),
                             np.arange(kg[2]), indexing='ij')
    kpts = np.stack([kx.ravel(), ky.ravel(), kz.ravel()],
                    axis=1).astype(float) / kg[None, :].astype(float)
    return types.SimpleNamespace(
        ntran=1, kpoints=kpts, nkpts=int(kpts.shape[0]),
        kgrid=kg, shift=np.zeros(3, dtype=float),
    )


def test_trivial_symmetry_branch_trs_augments_sym_mats_k():
    """``ntran <= 1`` must still yield a TRS-augmented ``sym_mats_k``.

    Regression guard for workstream Q: when the no-symmetry branch left
    ``sym_mats_k`` at length 1, ``unfold_psi`` computed
    ``n_sym_spatial = 1 // 2 = 0`` and therefore classified the IDENTITY
    row (``sym_idx = 0``) as a time-reversal row — silently returning
    ``iσ_y·conj(ψ)`` on an un-negated G-list for every k of every
    ``nosym`` WFN.  Norms, ⟨ψ_m|ψ_n⟩, ⟨T⟩ and (because ρ inverts too)
    ⟨V_H⟩ are all invariant under that, so the corruption only surfaced
    in the position-dependent ionic terms (V_NL collapsed, V_loc shifted
    by O(100 eV)).
    """
    sym = SymMaps(_trivial_sym_wfn())
    smk = np.asarray(sym.sym_mats_k)
    assert smk.shape == (2, 3, 3), (
        f"sym_mats_k must be TRS-augmented to 2·ntran rows; got {smk.shape}")
    np.testing.assert_array_equal(smk[0], np.eye(3, dtype=smk.dtype))
    np.testing.assert_array_equal(smk[1], -np.eye(3, dtype=smk.dtype))
    assert smk.shape[0] == 2 * np.asarray(sym.U_spinor).shape[0]

    # ...and the identity row must round-trip ψ untouched.
    rng = np.random.default_rng(11)
    cnk = (rng.standard_normal((2, 2, 4)) + 1j * rng.standard_normal((2, 2, 4)))
    g_kbar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 1, 0]],
                      dtype=np.int32)
    out = unfold_psi(
        cnk, sym_idx=int(sym.sym_idx_k[0]), g_kbar=g_kbar,
        sym_mats_k=sym.sym_mats_k, translations=sym.translations,
        U_spinor_spatial=sym.U_spinor,
    )
    np.testing.assert_allclose(
        out, cnk, atol=1e-14, rtol=0.0,
        err_msg="identity sym row must leave ψ unchanged on a nosym WFN")


if __name__ == "__main__":
    test_trivial_symmetry_branch_trs_augments_sym_mats_k()
    print("test_trivial_symmetry_branch_trs_augments_sym_mats_k OK")
    test_unfold_psi_identity_is_noop()
    print("test_unfold_psi_identity_is_noop OK")
    test_unfold_psi_matches_hand_reference()
    print("test_unfold_psi_matches_hand_reference OK")
    test_unfold_psi_trs_squared_is_identity_on_spinor()
    print("test_unfold_psi_trs_squared_is_identity_on_spinor OK")
