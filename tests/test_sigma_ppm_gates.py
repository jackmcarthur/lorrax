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


def test_crossing_core_rescales_the_physical_error_contract(monkeypatch):
    """The HGL service request and certificate follow the same xi scaling.

    This uses a non-unit regularization width so an omitted conversion cannot
    pass by coincidence.  The explicit sine values also prove that the
    incumbent ``tau/xi, alpha/xi`` rule represents ``G(x/xi)/xi``; no solver
    or random fixture is involved.
    """
    from gw import ppm_windows
    from gw.minimax_screening import CrossingMinimaxQuadrature

    xi = 2.5
    target_error_phys = 4.0e-7
    tau_hat = np.array([0.5, 1.5], dtype=np.float64)
    alpha_hat = np.array([0.2, -0.1], dtype=np.float64)
    error_hat = 5.0e-7
    seen = {}

    def _served(A_dim, **kwargs):
        seen.update(A_dim=A_dim, **kwargs)
        return CrossingMinimaxQuadrature(
            A_dim=float(A_dim), tau=tau_hat, alpha=alpha_hat,
            max_error=error_hat, target_kind="hgl",
            provenance="deterministic scaling fixture")

    monkeypatch.setattr(ppm_windows, "solve_phase_minimax_bandwidth", _served)
    windows = ppm_windows._build_three_sigma_windows(
        E_A=np.array([0.2], dtype=np.float64),
        base_mask_A=np.array([True]),
        mask_B_all_count=1,
        mask_B_le_count=1,
        mask_B_le_min=0.4,
        mask_B_le_max=0.4,
        mask_B_gt_count=0,
        mask_B_gt_min=None,
        mask_B_gt_max=None,
        omega_nonneg_ry=np.array([0.3], dtype=np.float64),
        neg_omega_half=False,
        regularization_width_ry=xi,
        edge_factor=1.5,
        target_error=target_error_phys,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=500,
        use_shipped_tables=True,
    )

    assert [window.name for window in windows] == ["core"]
    core = windows[0]
    assert seen["target_error"] == target_error_phys * xi
    assert core.max_error == error_hat / xi
    np.testing.assert_allclose(np.asarray(core.nodes.t).real, tau_hat / xi)
    np.testing.assert_allclose(np.asarray(core.nodes.alpha).real, alpha_hat / xi)

    u = np.array([0.0, 0.25, 0.75, 1.5], dtype=np.float64)
    x = xi * u
    fitted_hat = np.sin(np.outer(u, tau_hat)) @ alpha_hat
    fitted_phys = (
        np.sin(np.outer(x, np.asarray(core.nodes.t).real))
        @ np.asarray(core.nodes.alpha).real
    )
    np.testing.assert_allclose(fitted_phys, fitted_hat / xi, rtol=1.0e-14)


def test_shared_omega_clusters_preserve_gap_only_owner_and_cap_spans():
    """The shared owner keeps MPA's gap-only result and adds a span cap.

    Deliberately scramble the branch order: returned index arrays must retain
    that order even though clusters themselves are ordered by energy.
    """
    from gw.ppm_windows import _omega_clusters

    omega = np.array([3.2, 0.1, 0.2, 3.1], dtype=np.float64)
    incumbent = _omega_clusters(omega, 1.0)
    assert [(i.tolist(), lo, hi) for i, lo, hi in incumbent] == [
        ([1, 2], 0.1, 0.2), ([0, 3], 3.1, 3.2)]

    capped = _omega_clusters(omega, 1.0, max_span_ry=0.05)
    assert [(i.tolist(), lo, hi) for i, lo, hi in capped] == [
        ([1], 0.1, 0.1), ([2], 0.2, 0.2),
        ([3], 3.1, 3.1), ([0], 3.2, 3.2)]


def test_hgl_capacity_owner_keeps_the_roundoff_band_on_incumbent_family():
    from gw.ppm_windows import _CROSSING_A_MAX, hgl_partition_required

    xi = 0.2
    edge = 1.5
    omega_at_capacity = (0.5 * _CROSSING_A_MAX - edge) * xi
    eps = np.finfo(np.float64).eps
    assert not hgl_partition_required(
        np.array([-omega_at_capacity, omega_at_capacity]), xi, edge)
    assert not hgl_partition_required(
        np.array([omega_at_capacity * (1.0 + 4.0 * eps)]), xi, edge)
    assert hgl_partition_required(
        np.array([omega_at_capacity * (1.0 + 32.0 * eps)]), xi, edge)


def test_hgl_cell_plan_tiles_direct_denominator_and_respects_capacity():
    """Exact omega x A x B ownership and first-principles sign bounds.

    No quadrature and no random arrays: the direct retarded denominator is
    evaluated on every deterministic cell boundary.  Its cell-selected sum
    must be the direct value, including the repository-wide ``(lo, hi]``
    downward assignment at both A and B boundaries.
    """
    from gw.ppm_windows import plan_hgl_crossing_cells

    omega = np.array([0.2, 0.4, 3.1, 3.3], dtype=np.float64)
    energies = np.array([[0.1, 0.3, 1.0, 9.0]], dtype=np.float64)
    base = np.array([[True, True, True, False]])
    xi = 0.2
    edge = 1.5
    A_max = 4.0
    plan = plan_hgl_crossing_cells(
        omega_abs=omega, E_A=energies, base_mask_A=base,
        regularization_width_ry=xi, edge_factor=edge,
        omega_cluster_gap_ry=1.0, omega_max_span_ry=0.25,
        crossing_A_max=A_max)

    assert plan.omega_cluster_count == 2
    assert plan.energy_pane_count == 4
    assert len(plan.cells) == 12
    assert plan.max_A_dim <= A_max

    z = edge * xi
    live_e = energies[base]
    for cell in plan.cells:
        assert cell.omega_hi - cell.omega_lo <= 0.25
        if cell.kind == "crossing":
            corners = [
                w - e - b
                for w in (cell.omega_lo, cell.omega_hi)
                for e in (cell.e_min, cell.e_max)
                for b in (cell.b_lo, cell.b_hi)
            ]
            assert max(abs(x) for x in corners) <= cell.A_dim * xi * (
                1.0 + 8.0 * np.finfo(np.float64).eps)
        elif cell.kind == "positive":
            # The least-positive corner sits at the closed upper B edge.
            x_min = cell.omega_lo - cell.e_max - cell.b_hi
            assert x_min >= z * (1.0 - 8.0 * np.finfo(np.float64).eps)
        else:
            assert cell.kind == "negative"
            # The open lower B edge is approached from above.
            x_sup = cell.omega_hi - cell.e_min - cell.b_lo
            assert x_sup <= -z * (1.0 - 8.0 * np.finfo(np.float64).eps)

    finite_edges = sorted({
        bound for cell in plan.cells for bound in (cell.b_lo, cell.b_hi)
        if np.isfinite(bound)})
    b_probe = np.array(
        [finite_edges[0] - 0.2, *finite_edges,
         *[(a + b) / 2.0 for a, b in zip(finite_edges, finite_edges[1:])],
         finite_edges[-1] + 0.2], dtype=np.float64)
    for iw, w in enumerate(omega):
        for e in live_e:
            for b in b_probe:
                owners = [
                    cell for cell in plan.cells
                    if iw in cell.omega_indices
                    and e > cell.e_lo and e <= cell.e_hi
                    and b > cell.b_lo and b <= cell.b_hi
                ]
                assert len(owners) == 1, (iw, e, b, owners)
                direct = 1.0 / (w - e - b + 1j * xi)
                decomposed = sum(
                    1.0 / (w - e - b + 1j * xi) for _cell in owners)
                np.testing.assert_array_equal(decomposed, direct)


def test_hgl_cell_rules_rephase_the_direct_scalar_kernel(monkeypatch):
    """The +/crossing/- rows reproduce ``exp(i*t*(omega-E-B))``.

    This is the complete scalar coefficient product used by the production
    G, W, and omega projector, including both reference phases.  One A value,
    one omega point, and three deterministic pole values populate exactly one
    cell of each kind; no random tensor or frontend fixture is involved.
    """
    from gw import ppm_windows
    from gw.minimax_screening import (
        CrossingMinimaxQuadrature,
        LaplaceMinimaxQuadrature,
    )

    tau_laplace = 0.7
    tau_cross_hat = 0.5

    def _laplace(x_min, x_max, **kwargs):
        return LaplaceMinimaxQuadrature(
            x_min=float(x_min), x_max=float(x_max),
            tau=np.array([tau_laplace]), alpha=np.array([1.0]),
            max_error=0.0, provenance="deterministic scalar rule")

    def _crossing(A_dim, **kwargs):
        return CrossingMinimaxQuadrature(
            A_dim=float(A_dim), tau=np.array([tau_cross_hat]),
            alpha=np.array([1.0]), max_error=0.0, target_kind="hgl",
            provenance="deterministic scalar rule")

    monkeypatch.setattr(
        ppm_windows, "solve_laplace_minimax_interval", _laplace)
    monkeypatch.setattr(
        ppm_windows, "solve_phase_minimax_bandwidth", _crossing)

    xi = 0.2
    omega = 1.0
    energy = 0.2
    poles = np.array([[[0.4, 0.8, 1.2]]], dtype=np.float64)
    windows, _plan = ppm_windows._build_partitioned_hgl_windows(
        E_A=np.array([[energy]], dtype=np.float64),
        base_mask_A=np.array([[True]]),
        Omega_q=ppm_windows.jnp.asarray(poles),
        base_mask_B=ppm_windows.jnp.ones(poles.shape, dtype=bool),
        omega_nonneg_ry=np.array([omega]),
        neg_omega_half=False,
        regularization_width_ry=xi,
        edge_factor=1.5,
        target_error=1.0e-6,
        max_nodes=64,
        crossing_eps_q=1.0e-3,
        crossing_max_nodes=64,
        use_shipped_tables=False,
    )
    assert [w.name for w in windows] == [
        "pane_positive", "pane_crossing", "pane_negative"]

    pole_for = {"pane_positive": 0.4, "pane_crossing": 0.8,
                "pane_negative": 1.2}
    for window in windows:
        pole = pole_for[window.name]
        t = complex(np.asarray(window.nodes.t)[0])
        alpha = complex(np.asarray(window.nodes.alpha)[0])
        factorized = (
            np.exp(-1j * (energy - window.E_ref_A) * t)
            * np.exp(-1j * (pole - window.E_ref_B) * t)
            * np.exp(-1j * (
                window.E_ref_A + window.E_ref_B - omega) * t)
        )
        direct_phase = np.exp(1j * t * (omega - energy - pole))
        np.testing.assert_allclose(
            factorized, direct_phase, rtol=2.0e-15, atol=2.0e-15)

        direct_weighted = window.prefactor * alpha * direct_phase
        factorized_weighted = window.prefactor * alpha * factorized
        if window.project == "imag":
            # Diagonal band-adjoint completion is Im(Z).
            np.testing.assert_allclose(
                np.imag(factorized_weighted),
                np.imag(direct_weighted), rtol=2.0e-15, atol=2.0e-15)
        else:
            np.testing.assert_allclose(
                factorized_weighted, direct_weighted,
                rtol=2.0e-15, atol=2.0e-15)

    assert complex(np.asarray(windows[0].nodes.t)[0]).imag > 0.0
    assert complex(np.asarray(windows[2].nodes.t)[0]).imag < 0.0
    assert [w.prefactor for w in windows] == [-1.0, -1.0, 1.0]


def test_memory_tile_sink_splices_disjoint_omega_clusters():
    """Cluster rows assemble on the existing bracket-then-omega layout."""
    from gw.ppm_accumulators import _MemoryTileSink

    sink = _MemoryTileSink(shape=(1, 5, 1, 1, 1), sharding=None)
    shard_index = [(slice(None), slice(None), slice(None), slice(None))]
    devices = [None]
    sink.consume_window(
        [np.array([2.0, 3.0], dtype=np.complex128).reshape(1, 2, 1, 1, 1)],
        shard_index, devices, omega_indices=np.array([0, 3]))
    sink.consume_window(
        [np.array([5.0, 7.0], dtype=np.complex128).reshape(1, 2, 1, 1, 1)],
        shard_index, devices, omega_indices=np.array([1, 4]))
    tiles, _, _ = sink.host_tiles()
    np.testing.assert_array_equal(
        tiles[0].reshape(-1), np.array([2.0, 5.0, 0.0, 3.0, 7.0]))
    with pytest.raises(RuntimeError, match="cannot mix clustered"):
        sink.consume_window(
            [np.zeros((1, 5, 1, 1, 1), dtype=np.complex128)],
            shard_index, devices)


def test_sign_definite_omega_panes_exhaust_extreme_tail_exactly():
    """CrI3-shaped pole tails are partitioned, never dropped/staticised."""
    import jax.numpy as jnp
    from types import SimpleNamespace
    from gw.ppm_windows import (
        _plan_sign_definite_omega_panes,
        window_mask_B_bounds,
    )

    # Two 0.2% tails around a compact body, with the frozen run33 scales.
    omega = np.concatenate([
        np.array([2.0e-4, 4.0e-4]),
        np.linspace(0.05, 4.0, 996),
        np.array([95.8565, 97.8518]),
    ]).astype(np.float64)
    mask = np.ones_like(omega, dtype=bool)
    E_min, E_max, omega_eval = 0.0343332986397257, 5.40437906406350, 1.46997235298981
    panes = _plan_sign_definite_omega_panes(
        Omega_q=jnp.asarray(omega), base_mask_B=jnp.asarray(mask),
        mask_B_count=omega.size,
        mask_B_min=float(omega.min()), mask_B_max=float(omega.max()),
        E_min=E_min, E_max=E_max, omega_max=omega_eval,
        target_error=1.0e-6,
        max_nodes=64,
        use_shipped_tables=False,
    )

    ownership = np.zeros(omega.size, dtype=np.int64)
    pane_sum = 0.0 + 0.0j
    residues = (np.linspace(0.2, 1.2, omega.size)
                + 1j * np.linspace(-0.3, 0.4, omega.size))
    for lo, hi, count, actual_min, actual_max in panes:
        # Explicit B_lo/B_hi wins over mask_B_mode="all" in the existing
        # runtime selector; no second mask convention is hidden in the test.
        got_lo, got_hi = window_mask_B_bounds(SimpleNamespace(
            B_lo=lo, B_hi=hi, mask_B_mode="all",
            mask_B_threshold=None))
        selected = (omega > got_lo) & (omega <= got_hi)
        ownership += selected
        assert int(np.sum(selected)) == count
        assert actual_min == float(np.min(omega[selected]))
        assert actual_max == float(np.max(omega[selected]))
        R = ((E_max + actual_max + omega_eval)
             / (E_min + actual_min))
        assert R <= 2.0 ** np.sqrt(64) or actual_min == actual_max
        pane_sum += np.sum(residues[selected] / (0.7 + E_max + omega[selected]))

    np.testing.assert_array_equal(ownership, np.ones_like(ownership))
    direct = np.sum(residues / (0.7 + E_max + omega))
    np.testing.assert_allclose(pane_sum, direct, rtol=2e-15, atol=2e-15)
    assert [panes[0][2], panes[-1][2]] == [2, 2]


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
