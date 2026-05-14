"""Synthetic regression test for the bispinor TRS handling in
``common.symmetry_maps.unfold_psi``.

Background
----------
PR3 (2026-05-14) consolidates the ψ k-unfold into a single free function
``unfold_psi`` and fixes three pre-PR3 bugs from Agent 1's scope (Sites
#5, #6, #7 in ``reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md``):

1. ``sym.U_spinor[ntran:]`` was wrong on the TRS-augmented half. The TRS
   spinor is ``iσ_y · conj(U_spinor[s])``, not ``get_spinor_rotations(-S)``
   (which the pre-PR3 ``det<0 → R=-R`` flip silently de-signed).
2. The τ-phase was skipped on TRS rows.
3. ``_get_umklapp_vector`` skipped the τ-phase on TRS rows (related — same
   missing-conj-phase bug propagated through).

This test:
* Synthesizes a non-inversion bispinor system ({I, σ_y} spatial × TRS).
* Hand-builds ψ_kbar with non-trivial spinor structure.
* Calls ``unfold_psi`` for each full-BZ k.
* Verifies against the per-element rule
  ``ψ_full = (iσ_y · conj(U_s)) · conj(ψ_kbar · phase_spatial)``  for TRS rows,
  ``ψ_full = U_s · ψ_kbar · phase_spatial``                        for spatial rows.

All tests must pass at max relative error < 1e-12.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from common.symmetry_maps import unfold_psi, _I_SIGMA_Y


# ---------------------------------------------------------------------------
# Geometry: {I, σ_y} spatial (ntran=2), TRS-augmented sym_mats_k length 4.
# 3×3×1 q-grid (matches MoS₂-like 2D lattice), non-symmorphic τ=(1/2, 0, 0)
# on one of the spatial ops to exercise the τ-phase.
# ---------------------------------------------------------------------------

def _build_geometry():
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
    geom = _build_geometry()
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
            n_sym_spatial=ntran,
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
    geom = _build_geometry()
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
        n_sym_spatial=geom["ntran"],
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
    geom = _build_geometry()
    geom["translations"][:] = 0.0   # symmorphic group to isolate spinor part
    rng = np.random.default_rng(13)
    nb, ns, ngk = 1, 2, 3
    cnk = (rng.standard_normal((nb, ns, ngk))
           + 1j * rng.standard_normal((nb, ns, ngk)))
    g_kbar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)

    once = unfold_psi(
        cnk,
        sym_idx=geom["ntran"],   # TRS · identity = pure TRS
        n_sym_spatial=geom["ntran"],
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
        n_sym_spatial=geom["ntran"],
        g_kbar=-g_kbar,
        sym_mats_k=geom["sym_mats_k"],
        translations=geom["translations"],
        U_spinor_spatial=geom["U_spinor_spatial"],
    )
    # T² ψ = -ψ.
    np.testing.assert_allclose(twice, -cnk, atol=1e-12, rtol=0.0,
                                err_msg="T² should equal -I on a spin-1/2 state")


if __name__ == "__main__":
    test_unfold_psi_identity_is_noop()
    print("test_unfold_psi_identity_is_noop OK")
    test_unfold_psi_matches_hand_reference()
    print("test_unfold_psi_matches_hand_reference OK")
    test_unfold_psi_trs_squared_is_identity_on_spinor()
    print("test_unfold_psi_trs_squared_is_identity_on_spinor OK")
