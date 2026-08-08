"""Σ_PPM WS0 keystone gate G2 — per-branch / per-window reference tiles.

G2 is the bit-identity pin the WS3 file split (moving _SigmaWindow /
_SigmaBranch / _iter_branches / _build_*_sigma_windows /
_build_windows_for_branch into ppm_windows.py) must stay identical
against.  Also guards against a split silently dropping a branch (all 4
branches × their windows asserted non-empty).

G1 (the kij ↔ kij_stream accumulator parity gate that detected Bug B)
was RETIRED 2026-07-31 with the removal of the kij_stream mode itself.
G3 (the head negative-branch regression) lives in
``tests/test_head_correction.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_REG = REPO_ROOT / "tests" / "regression"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import gpu_available, requested_platform  # noqa: E402


# ===========================================================================
#  G2 — per-branch / per-window reference tiles (GREEN, WS3 pin)
# ===========================================================================
#
# We drive the window builders directly (the "equivalent per-branch τ-node /
# mask arrays that _build_windows_for_branch produces" the spec explicitly
# permits) rather than scraping an e2e run: WS3 is a *pure move* of exactly
# these symbols, so a unit test that imports and exercises them is the precise
# pin.  Inputs are synthetic but structured to populate every window kind:
#   * a symmetric ω grid (±10 eV) → all 4 (ω-sign × cond/val) branches;
#   * E_A spanning the window threshold T → the core + a_stripe split;
#   * Ω_q spanning T → the b_slab (Ω>T) window as well.

G2_REF = _REG / "sigma_ppm_gates" / "g2_branch_window_ref.npz"


def _build_branch_windows():
    """Build the 4 branches × their windows from controlled synthetic inputs.

    Returns ``(branches, flat)`` where ``branches`` is a list of
    ``(tag, space, [_SigmaWindow, ...])`` and ``flat`` is the dict of
    plain numpy arrays / scalars captured for the reference freeze.
    """
    import jax.numpy as jnp
    from common.units import RYD_TO_EV
    from gw.ppm_windows import _build_windows_for_branch, _iter_branches

    nk, nb = 2, 6                       # 3 valence + 3 conduction bands
    # E_cond (conduction) and H_val (valence) energies-above-Fermi (Ry),
    # chosen to straddle T ≈ 0.76 Ry so both E_A≤T and E_A>T are populated.
    e_above = np.array([0.30, 0.90, 1.60], dtype=np.float64)      # Ry
    E_cond = np.zeros((nk, nb), dtype=np.float64)
    H_val = np.zeros((nk, nb), dtype=np.float64)
    cond_mask = np.zeros((nk, nb), dtype=bool)
    val_mask = np.zeros((nk, nb), dtype=bool)
    for k in range(nk):
        val_mask[k, 0:3] = True
        cond_mask[k, 3:6] = True
        H_val[k, 0:3] = e_above
        E_cond[k, 3:6] = e_above

    # Ω_q (nq, μ, μ) straddling T so mask_B le_t / gt_t are both non-empty.
    Omega_abs = np.array(
        [[[0.40, 0.50], [0.55, 0.45]],
         [[1.00, 1.20], [1.10, 1.30]]], dtype=np.float64)
    B_mask = Omega_abs > 1.0e-14

    # Symmetric ω grid in Ry — both halves present ⇒ all 4 branches.
    omega_ry = np.arange(-10.0, 10.0 + 1e-9, 0.5, dtype=np.float64) / RYD_TO_EV
    idx_pos = np.where(omega_ry >= 0.0)[0]
    idx_neg = np.where(omega_ry < 0.0)[0]
    omega_pos = omega_ry[idx_pos]
    omega_neg_abs = -omega_ry[idx_neg]

    branches = _iter_branches(
        omega_pos=omega_pos, idx_pos=idx_pos,
        omega_neg_abs=omega_neg_abs, idx_neg=idx_neg,
        E_cond=jnp.asarray(E_cond), H_val=jnp.asarray(H_val),
        cond_mask=jnp.asarray(cond_mask), val_mask=jnp.asarray(val_mask),
    )

    quad = dict(
        regularization_width_ry=0.25 / RYD_TO_EV,
        edge_factor=1.5,
        target_error=1e-6,
        max_nodes=64,
        crossing_eps_q=1e-3,
        crossing_max_nodes=500,
        use_shipped_minimax_tables=True,
    )

    out = []
    flat: dict[str, np.ndarray] = {}
    for br in branches:
        windows = _build_windows_for_branch(
            omega_nonneg_ry=br.omega_abs,
            E_A=br.E_A, base_mask_A=br.base_mask_A,
            Omega_q=jnp.asarray(Omega_abs), base_mask_B=jnp.asarray(B_mask),
            space=br.space, neg_omega_half=br.neg_omega_half,
            log_tag=br.tag, print_fn=lambda *a, **k: None,
            **quad,
        )
        out.append((br.tag, br.space, windows))
        for wi, w in enumerate(windows):
            key = f"{br.tag}|{wi}|{w.name}"
            flat[f"{key}|t"] = np.asarray(w.nodes.t, dtype=np.complex128)
            flat[f"{key}|alpha"] = np.asarray(w.nodes.alpha, dtype=np.complex128)
            flat[f"{key}|mask_A"] = np.asarray(w.mask_A, dtype=bool)
            flat[f"{key}|meta"] = np.array([
                float(w.E_ref_A), float(w.E_ref_B),
                float(w.omega_sign), float(w.prefactor),
                float(w.project_code), float(w.n_tau),
                float(w.mask_B_threshold) if w.mask_B_threshold is not None else np.nan,
            ], dtype=np.float64)
            flat[f"{key}|tags"] = np.array(
                [w.project, w.mask_B_mode, str(w.crossing_kind)], dtype="<U16")
    return out, flat


def _regenerate_g2_reference():
    """(Re)write the frozen G2 reference .npz.  Call manually when the window
    builders legitimately change; never inside the test."""
    _, flat = _build_branch_windows()
    G2_REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(G2_REF, **flat)
    return G2_REF


@pytest.mark.regression
def test_g2_branch_window_tiles_are_frozen():
    if requested_platform() in {"gpu", "cuda"} and not gpu_available():
        pytest.skip("CUDA GPU not available for requested platform=gpu.")

    branches, flat = _build_branch_windows()

    # --- structural guards: no branch or window may silently vanish -------
    tags = [t for t, _, _ in branches]
    assert tags == ["ω≥E_F cond", "ω≥E_F val", "ω<E_F cond", "ω<E_F val"], tags
    win_names = {t: [w.name for w in ws] for t, _, ws in branches}
    # The crossing branch (pole S can coincide with a grid ω) gets the 3-window
    # crossing/stripe/slab split: conduction on the +ω half, valence on the −ω
    # half.  The sign-definite branches get the single Laplace window.
    assert win_names["ω≥E_F cond"] == ["core", "a_stripe", "b_slab"], win_names
    assert win_names["ω<E_F val"] == ["core", "a_stripe", "b_slab"], win_names
    assert win_names["ω≥E_F val"] == ["single"], win_names
    assert win_names["ω<E_F cond"] == ["single"], win_names
    for _t, _s, ws in branches:
        assert ws, f"branch {_t!r} produced no windows"
        for w in ws:
            assert bool(np.any(w.mask_A)), f"{_t}:{w.name} empty mask_A"
            assert w.n_tau > 0, f"{_t}:{w.name} zero τ nodes"

    # --- bit-identity against the frozen reference ------------------------
    # NO CROSS-MACHINE TOLERANCE HERE, and the 2026-08-07 owner ruling
    # ("the micro-eV level is fine for comparisons between machines") does
    # NOT reach this gate.  The Perlmutter/Frontera disagreement in this
    # cell is the crossing-core node ladder, and that is an INTEGER count of
    # quadrature nodes riding in a float64 `meta` row, not a rounding
    # difference and not a quantity in eV — an atol would hide a real
    # change in how many τ points the window integrates over.  Whatever
    # this cell's answer is, it is not "loosen the comparison".
    #
    # THE REFERENCE FOLLOWS THE PERLMUTTER GRID.  Owner ruling P1b, taken at
    # the 2026-08-08 service-phase landing, and this file was re-frozen from
    # a Perlmutter run at the integration head to match it.  Perlmutter is
    # the blessed machine for this reference.
    #
    # THE FRONTERA DIFFERENCE IS STRUCTURAL, NOT NUMERICAL, and that is why
    # it cannot be absorbed by any tolerance.  Measured at the landing, 40
    # tile keys, Frontera-frozen array vs a Perlmutter build:
    #
    #   32 keys bit-identical
    #    4 keys SHAPE-mismatched — the crossing-core `t` and `alpha` ladders
    #      of BOTH crossing branches (ω≥E_F cond, ω<E_F val):
    #      Frontera (98,) vs Perlmutter (100,)
    #    2 `meta` rows differ by exactly 2.0 — the same node count, riding
    #      in the float64 meta row, which is what made this look like a
    #      micro-eV row and got it mis-filed under P1
    #    2 `meta` rows compare unequal with max|Δ| == 0.0: NaN-vs-NaN in the
    #      mask_B_threshold slot, identical in content
    #
    # The τ-node POSITIONS disagree from the first element (Perlmutter
    # [2.666561, 6.499882, ...] vs Frontera [5.442279e-09, 7.894766, ...]),
    # so these are two different quadratures, not one quadrature sampled
    # twice.  A machine running the Frontera minimax tables will therefore
    # fail this cell LOUDLY and by shape, which is the correct outcome: it
    # says "this build integrates over a different number of τ points",
    # which is exactly the fact the gate exists to surface.  Do not add a
    # tolerance and do not re-freeze on Frontera to make it green — bring
    # the question back to the owner, because only one grid is blessed.
    assert G2_REF.exists(), (
        f"missing G2 reference {G2_REF}; regenerate with "
        f"tests.test_sigma_ppm_gates._regenerate_g2_reference()")
    ref = np.load(G2_REF, allow_pickle=False)
    assert set(ref.files) == set(flat), (
        f"window-tile key set drifted:\n  new-only={set(flat) - set(ref.files)}"
        f"\n  ref-only={set(ref.files) - set(flat)}")
    for key in ref.files:
        got = flat[key]
        want = ref[key]
        if got.dtype.kind == "U":
            assert np.array_equal(got, want), f"{key} tag mismatch: {got} != {want}"
        else:
            np.testing.assert_array_equal(got, want, err_msg=f"{key} not bit-identical")
