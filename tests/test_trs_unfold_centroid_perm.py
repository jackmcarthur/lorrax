"""Unit tests for the TRS-aware V_q IBZ-unfold path.

Two contracts under test:

1. ``compute_centroid_sym_perm(extend_trs=True)`` returns a
   ``(2·n_sym, n_rmu)`` table whose rows ``[n_sym:]`` duplicate rows
   ``[:n_sym]`` — under time-reversal symmetry the real-space centroid
   permutation is unchanged (TRS keeps r fixed).
2. ``_unfold_v_q_ibz_to_full`` correctly unfolds an IBZ V_q to the full
   BZ on a synthetic Hermitian V where some full-BZ q's reach their IBZ
   parent only via the TRS-augmented half of ``sym_mats_k``.  The TRS
   rule under the Hermitian-V_q derivation:

       V_{full}^{TRS-q, π_s(μ), π_s(ν)} = conj(V_{ibz}^{parent, μ, ν})

   (equivalently V_{ibz, ν, μ} by Hermiticity).  The unfold helper
   applies the per-row complex conjugation when ``full_to_irr_sym ≥
   ntran``.

This is a small synthetic test — no WFN, no centroid generator, no
real ζ.  Runs on a single CPU/GPU device in <1 s.

See ``reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md`` Site
#1 for the failure mode this test guards against.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from centroid.orbit_syms import compute_centroid_sym_perm
from gw.v_q_tile import _unfold_v_q_ibz_to_full


# ---------------------------------------------------------------------------
# Geometry: ``ntran=2`` ({I, σ_y}) on a 4×4×1 FFT grid, TRS-augmented
# table length 4.  Centroid set orbit-closed under {I, σ_y}.
# ---------------------------------------------------------------------------

def _build_geometry():
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
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry()
    n_sym = sym_matrices.shape[0]
    n_rmu = cent_idx.shape[0]

    sp_default = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True)
    sp_trs = compute_centroid_sym_perm(
        cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
        validate=True, extend_trs=True)

    # Default: (ntran, n_rmu).
    assert sp_default.shape == (n_sym, n_rmu)

    # Extended: (2·ntran, n_rmu) with rows[ntran:] duplicating rows[:ntran].
    assert sp_trs.shape == (2 * n_sym, n_rmu), (
        f"extend_trs=True must return (2·ntran, n_rmu); got {sp_trs.shape}")
    np.testing.assert_array_equal(sp_trs[:n_sym], sp_default)
    np.testing.assert_array_equal(sp_trs[n_sym:], sp_default)


def test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation():
    """Sanity: each row (including TRS half) is a permutation of [0, n_rmu)."""
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry()
    n_rmu = cent_idx.shape[0]
    sp = compute_centroid_sym_perm(
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


def _hand_unfold_v_q(V_ibz, *, full_to_irr_idx, full_to_irr_sym,
                     sym_perm, ntran):
    """Reference: pure numpy unfold with the per-q TRS conj rule.

    For spatial s < ntran:
        V_full[q, μ, ν] = V_ibz[parent, π_s^{-1}(μ), π_s^{-1}(ν)]
    For TRS s ≥ ntran:
        V_full[q, μ, ν] = conj(V_ibz[parent, π_s^{-1}(μ), π_s^{-1}(ν)])
    """
    inv_perm = np.argsort(sym_perm, axis=-1)
    n_q_full = full_to_irr_idx.shape[0]
    n_rmu = V_ibz.shape[-1]
    V_full = np.zeros((n_q_full, n_rmu, n_rmu), dtype=V_ibz.dtype)
    for iq in range(n_q_full):
        parent = int(full_to_irr_idx[iq])
        s = int(full_to_irr_sym[iq])
        perm_inv = inv_perm[s]               # length n_rmu
        is_trs = s >= ntran
        V_perm = V_ibz[parent][np.ix_(perm_inv, perm_inv)]
        V_full[iq] = np.conj(V_perm) if is_trs else V_perm
    return V_full


def test_unfold_v_q_ibz_to_full_handles_trs_rows():
    """Hand-build a synthetic Hermitian V_ibz and full→IBZ map with at
    least one TRS-only-reachable q, run the codebase unfold, compare
    to the per-element reference.
    """
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry()
    ntran = sym_matrices.shape[0]
    n_rmu = cent_idx.shape[0]

    sym_perm = compute_centroid_sym_perm(
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

    # Reference unfold.
    V_ref = _hand_unfold_v_q(
        V_ibz, full_to_irr_idx=full_to_irr_idx,
        full_to_irr_sym=full_to_irr_sym, sym_perm=sym_perm, ntran=ntran)

    # Codebase unfold.  ``n_sym_spatial=ntran`` opts into the TRS
    # branch (conj at rows where ``full_to_irr_sym ≥ ntran``).
    mesh = _make_mesh()
    V_sh = NamedSharding(mesh, P(None, 'x', 'y'))
    V_ibz_j = jax.device_put(V_ibz.astype(np.complex128), V_sh)
    V_full = np.asarray(jax.device_get(
        _unfold_v_q_ibz_to_full(
            V_ibz_j,
            full_to_irr_idx=full_to_irr_idx,
            full_to_irr_sym=full_to_irr_sym,
            sym_perm=sym_perm, mesh_xy=mesh,
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
    fft_grid, sym_matrices, translations, cent_idx = _build_geometry()
    ntran = sym_matrices.shape[0]

    sym_perm_legacy = compute_centroid_sym_perm(
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

    with pytest.raises(ValueError, match=r"TRS-augmented"):
        _unfold_v_q_ibz_to_full(
            V_ibz_j,
            full_to_irr_idx=full_to_irr_idx,
            full_to_irr_sym=full_to_irr_sym,
            sym_perm=sym_perm_legacy, mesh_xy=mesh)


if __name__ == "__main__":
    test_compute_centroid_sym_perm_extend_trs_shape_and_content()
    print("test_compute_centroid_sym_perm_extend_trs_shape_and_content OK")
    test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation()
    print("test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation OK")
    test_unfold_v_q_ibz_to_full_handles_trs_rows()
    print("test_unfold_v_q_ibz_to_full_handles_trs_rows OK")
    test_unfold_v_q_ibz_to_full_hard_fails_without_extend_trs()
    print("test_unfold_v_q_ibz_to_full_hard_fails_without_extend_trs OK")
