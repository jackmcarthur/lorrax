"""``sigma_out_of_grid`` (owner 2026-09-24): cover (default) grows the SC grid
over every non-frozen protected identity; clamp reads the nearest grid edge;
static reads omega = 0.  One classification (``qsgw_utils.omega_coverage``)
feeds the Sigma build, the growth and the tail mask.  Host NumPy + one CPU
device."""
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from gw.qsgw_utils import build_qsgw_sigma_xc, sigma_eval_omega
from gw.sc_iteration import BandPartition, _sc_sampled_support

GRID = np.arange(-12.0, 10.0 + 1e-9, 0.25)


def test_each_policy_reads_where_the_docs_say():
    e = np.array([[-13.0, 0.3, 9.9, 11.7]])
    for policy, want in (("cover", [0.0, 0.3, 9.9, 0.0]),
                         ("static", [0.0, 0.3, 9.9, 0.0]),
                         ("clamp", [-12.0, 0.3, 9.9, 10.0])):
        read, covered = sigma_eval_omega(GRID, e, policy)
        np.testing.assert_allclose(read, [want])
        np.testing.assert_array_equal(covered, [[False, True, True, False]])
    with pytest.raises(ValueError):
        sigma_eval_omega(GRID, e, "tail")


def test_clamp_keeps_the_diagonal_qp_equation_continuous_across_the_edge():
    # Sigma(w) = a + s w with the measured Fe H-point jump Sigma(0) - Sigma(10) = +1.87.
    a, s = 0.3, -0.187
    sig = (a + s * GRID)[:, None, None]
    w = np.linspace(9.0, 12.0, 301)[None, :]
    for policy, jumps in (("static", True), ("clamp", False)):
        read, _ = sigma_eval_omega(GRID, w, policy)
        sigma_at = np.interp(read[0], GRID, sig[:, 0, 0])
        step = np.abs(np.diff(sigma_at)).max()
        assert bool(step > 1.0) == jumps


def test_the_sigma_build_reads_the_edge_under_clamp():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    grid = np.array([-1.0, 0.0, 1.0])
    cube = np.zeros((3, 1, 1, 1), dtype=np.complex128)
    cube[:, 0, 0, 0] = [5.0, 7.0, 11.0]                   # Sigma(-1), Sigma(0), Sigma(+1)
    zero_x = jnp.zeros((1, 1, 1), dtype=jnp.complex128)
    for policy, want in (("static", 7.0), ("clamp", 11.0), ("cover", 7.0)):
        sig, diag = build_qsgw_sigma_xc(jnp.asarray(cube), zero_x, grid,
                                        np.array([[3.0]]), mesh, out_of_grid=policy)
        np.testing.assert_allclose(np.real(np.asarray(sig))[0, 0, 0], want)
        assert diag["n_clipped"] == 1.0


def _inputs(policy, n_frozen, session=None, grid=GRID):
    cfg = NS(compute_mode=NS(is_dynamic=True), omega_grid_ev=grid,
             sigma=NS(out_of_grid=policy, omega_min_ev=-12.0, omega_max_ev=8.0,
                      omega_step_ev=0.25,
                      classification_window_ev=lambda: (-12.0, 8.0)),
             sc=NS(frozen_core_bands=n_frozen))
    return NS(config=cfg, fixed_quadrature_session=session)


def test_cover_grows_over_every_non_frozen_identity_and_nothing_else():
    part = BandPartition(protected_mask=np.ones(4, bool), in_range_mask=np.ones(4, bool))
    e = np.array([[-100.0, -5.0, 9.8, 11.7]])            # rel mu; band 1 is semicore
    _, grown, *_ = _sc_sampled_support(_inputs("cover", 1), part, e, 0.0)
    assert grown[0] == GRID[0]                           # frozen core does not grow it
    assert 11.7 + 0.5 + 1.17 <= grown[-1] < 11.7 + 0.5 + 1.17 + 0.25
    # Bounded by the spectrum it covers: unfrozen semicore reaches E - pad(E).
    _, grown, *_ = _sc_sampled_support(_inputs("cover", 0), part, e, 0.0)
    assert -100.0 - 10.5 - 0.25 < grown[0] <= -100.0 - 10.5
    # ... unless the W model calls it inactive (shared_pole_recipe.active_band_mask):
    # then no fc key is needed and the semicore keeps Sigma(0) as under static.
    from gw.shared_pole_recipe import active_band_mask
    from common.units import RYD_TO_EV
    active = active_band_mask(e / RYD_TO_EV, 0.0)
    np.testing.assert_array_equal(active, [False, True, True, True])
    _, grown, *_ = _sc_sampled_support(_inputs("cover", 0), part, e, 0.0, active)
    assert grown[0] == GRID[0] and grown[-1] > 11.7
    for policy in ("static", "clamp"):                   # window rule: +9.8 and +11.7 lie
        _, grown, *_ = _sc_sampled_support(_inputs(policy, 1), part, e, 0.0)
        np.testing.assert_array_equal(grown, GRID)       # beyond the +9.44 padded top


@pytest.mark.parametrize("policy", ("clamp", "static"))
def test_frozen_core_cannot_extend_the_sampled_grid(policy):
    part = BandPartition(protected_mask=np.ones(2, bool),
                         in_range_mask=np.ones(2, bool))
    # The core is just outside the sampled grid, inside the SC pad.  Only
    # the valence identity can request a new Sigma sample.
    e = np.array([[-12.25, -1.0]])
    _, frozen_grid, _, required, _ = _sc_sampled_support(
        _inputs(policy, 1), part, e, 0.0)
    np.testing.assert_array_equal(frozen_grid, GRID)
    np.testing.assert_array_equal(required, [[False, True]])
    _, live_grid, *_ = _sc_sampled_support(
        _inputs(policy, 0), part, e, 0.0)
    assert live_grid[0] < GRID[0]


def test_the_deck_key_defaults_to_cover_and_refuses_anything_else():
    from gw.gw_config import DynamicSigmaConfig
    base = dict(omega_min_ev=-5.0, omega_max_ev=5.0, omega_step_ev=0.25,
                regularization_ev=0.25, window_edge_factor=1.0,
                fermi_reference="vbm", sigma_at_dft_extrapolate=False,
                sigma_at_dft_energies=False)
    assert DynamicSigmaConfig(**base).out_of_grid == "cover"
    with pytest.raises(ValueError, match="sigma_out_of_grid"):
        DynamicSigmaConfig(**base, out_of_grid="matched_tail")


def test_coverage_is_judged_in_the_frame_the_sigma_build_uses():
    # MoS2 3x3 QSGW (CLAIMS 2725): the PPM Sigma frame is the CURRENT
    # spectrum's midgap, 1.4 eV above the DFT midgap the partition uses.
    from gw.gw_config import ComputeMode
    from gw.efermi import sigma_frame_mu_ev
    from common.units import RYD_TO_EV
    e_ev = np.array([[-14.3, -5.1, -0.8, 3.0], [-14.2, -5.3, -0.9, 3.2]])
    e_ry, step_ry = e_ev / RYD_TO_EV, -3.0 / RYD_TO_EV       # 2 occupied per k
    wfn = NS(efermi=-4.332 / RYD_TO_EV, vbm=-5.4 / RYD_TO_EV)
    def frame(mode, ref):
        config = NS(compute_mode=mode, sigma=NS(fermi_reference=ref))
        return sigma_frame_mu_ev(config, wfn, e_ry, step_ry, None)
    np.testing.assert_allclose(frame(ComputeMode.GN_PPM, "midgap"), 0.5 * (-5.1 - 0.9), atol=1e-12)
    np.testing.assert_allclose(frame(ComputeMode.GN_PPM, "vbm"), -5.1, atol=1e-12)
    np.testing.assert_allclose(frame(ComputeMode.MPA, "midgap"), -4.332, atol=1e-12)


def _commit(session, support):
    """What the SC map writes back after logging (sc_iteration)."""
    sampled, grown, _, _, event = support
    session["omega_grid_ev"] = tuple(grown)
    session["window_plan"] = {"index": 0 if event == "plan" else 1, "event": event}
    return grown, event


def test_sc_window_plan_one_shot_then_plan_then_hold_then_extend():
    """Owner 2026-09-25: map 0 is the one-shot grid, map 1 plans once at 1 eV,
    later maps hold while every read support [E - 0.5, E + 0.5] is inside,
    and a crossing extends only its edge, to E + 1 eV."""
    from gw.scissor import SC_WINDOW_PAD_EV, sc_read_halfwidth_ev
    assert SC_WINDOW_PAD_EV == (2.0, 1.0) and sc_read_halfwidth_ev() == 0.5
    part = BandPartition(protected_mask=np.ones(3, bool), in_range_mask=np.ones(3, bool))
    session = {}
    inputs = _inputs("cover", 0, session, grid=np.arange(-12.0, 8.0 + 1e-9, 0.25))
    grid0, event = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 9.9]]), 0.0))
    assert event == "plan"                                  # one-shot growth: E + pad(E)
    assert 9.9 + 0.5 + 0.99 <= grid0[-1] < 9.9 + 0.5 + 0.99 + 0.25
    grid1, event = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 12.0]]), 0.0))
    assert event == "re-plan"
    assert grid1[0] == -12.0 and 13.0 <= grid1[-1] < 13.25   # from the requested grid, 1 eV
    held, event = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 12.5]]), 0.0))
    assert event == "hold"                                  # 12.5 + 0.5 <= 13.0
    np.testing.assert_array_equal(held, grid1)
    grown, event = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 12.6]]), 0.0))
    assert event == "extend"                                # 12.6 + 0.5 > 13.0
    assert grown[0] == grid1[0] and 13.6 <= grown[-1] < 13.85
    np.testing.assert_array_equal(grown[:grid1.size], grid1)   # old samples kept
    shrunk, event = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 10.0]]), 0.0))
    assert event == "hold" and shrunk.size == grown.size    # a hold never shrinks


def test_a_single_map_run_keeps_the_one_shot_rule():
    part = BandPartition(protected_mask=np.ones(2, bool), in_range_mask=np.ones(2, bool))
    *_, event = _sc_sampled_support(_inputs("cover", 0), part, np.array([[0.3, 9.9]]), 0.0)
    assert event == "one-shot"


def test_unset_grid_edges_derive_the_grid_from_the_bands():
    """Owner 2026-09-25: omega_min/max are optional. Unset, the requested grid
    is the sample next to E_F on each side and the bands set the rest; set,
    they are a minimum extent kept on every map."""
    from gw.gw_config import DynamicSigmaConfig
    base = dict(omega_step_ev=0.25, regularization_ev=0.25, window_edge_factor=1.0,
                fermi_reference="vbm", sigma_at_dft_extrapolate=False,
                sigma_at_dft_energies=False)
    unset = DynamicSigmaConfig(omega_min_ev=None, omega_max_ev=None, **base)
    assert unset.requested_edges_ev() == (-0.25, 0.25)
    assert unset.classification_window_ev() == (-np.inf, np.inf)
    half = DynamicSigmaConfig(omega_min_ev=-3.0, omega_max_ev=None, **base)
    assert half.requested_edges_ev() == (-3.0, 0.25)
    for policy in ("clamp", "static"):
        with pytest.raises(ValueError, match="needs sigma_out_of_grid = cover"):
            DynamicSigmaConfig(omega_min_ev=None, omega_max_ev=5.0,
                               out_of_grid=policy, **base)
    from gw.scissor import grow_sigma_support_ev
    requested = np.arange(-0.25, 0.25 + 1e-9, 0.25)
    grown, _ = grow_sigma_support_ev(unset, 0, requested, np.array([[-6.0, 2.0]]),
                                     np.ones((1, 2), bool))
    assert grown[0] <= -6.0 - 1.1 and grown[-1] >= 2.0 + 0.7
    assert np.isclose(grown, 0.0).any()


def test_a_patch_deck_sets_both_edges_before_the_cover_only_refusal(tmp_path):
    """A ``sigma_omega_patches_ev`` deck under static constructs: the patches
    set both edges before DynamicSigmaConfig is built (core fixture B's
    ``[static]`` deck).  Unset edges without patches still refuse."""
    from gw.gw_config import LorraxConfig
    base = ("[cohsex]\nsys_dim = 3\nnval = 2\nncond = 2\nnband = 10\n"
            "memory_per_device_gb = 4.0\nsigma_out_of_grid = static\n")
    deck = tmp_path / "patches.in"
    deck.write_text(base + "sigma_omega_patches_ev = -12:0, 0.1:9.4\n"
                    "sigma_omega_step_ev = 0.1\n")
    sigma = LorraxConfig.from_input_file(str(deck), print_fn=lambda *a, **k: None).sigma
    assert (sigma.omega_min_ev, sigma.omega_max_ev) == (-12.0, 9.4)
    assert sigma.out_of_grid == "static"
    deck.write_text(base)
    with pytest.raises(ValueError, match="needs sigma_out_of_grid = cover"):
        LorraxConfig.from_input_file(str(deck), print_fn=lambda *a, **k: None)
