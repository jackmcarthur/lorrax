"""``sigma_out_of_grid`` (owner 2026-09-24): cover (default) grows the SC grid
over every requested state (gw.qp_support, flat pads, owner 2026-09-27); clamp
reads the nearest grid edge; static reads omega = 0.  One classification (``qsgw_utils.omega_coverage``)
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
    assert 11.7 + 2.0 <= grown[-1] < 11.7 + 2.0 + 0.25   # flat 2 eV, not 10% of E
    # Bounded by the spectrum it covers: unfrozen semicore reaches E - 2 eV.
    _, grown, *_ = _sc_sampled_support(_inputs("cover", 0), part, e, 0.0)
    assert -100.0 - 2.0 - 0.25 < grown[0] <= -100.0 - 2.0
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
    support = _sc_sampled_support(_inputs(policy, 1), part, e, 0.0)
    np.testing.assert_array_equal(support.grown, GRID)
    np.testing.assert_array_equal(support.requested, [[False, True]])
    _, live_grid, *_ = _sc_sampled_support(
        _inputs(policy, 0), part, e, 0.0)
    assert live_grid[0] < GRID[0]


def test_the_deck_key_defaults_to_cover_and_refuses_anything_else():
    from gw.gw_config import DynamicSigmaConfig
    base = dict(omega_min_ev=-5.0, omega_max_ev=5.0, omega_step_ev=0.25,
                regularization_ev=0.25, window_edge_factor=1.0,
                fermi_reference="vbm",
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
    grown, event = support.grown, support.event
    session["omega_grid_ev"] = tuple(grown)
    session["support_envelope_ev"] = support.envelope
    session["window_plan"] = {"index": 0 if event == "plan" else 1, "event": event}
    return grown, event


def test_sc_window_plan_one_shot_then_plan_then_hold_then_extend():
    """Owner 2026-09-25/27: map 0 is the one-shot grid (requested states +/- 2
    eV), map 1 plans once at 1 eV, later maps hold while every read support
    [E - 0.5, E + 0.5] is inside, and a crossing extends only its edge, to
    E + 1 eV."""
    from gw.qp_support import SUPPORT_BUFFER_EV, SUPPORT_PAD_EV, read_halfwidth_ev
    assert SUPPORT_PAD_EV == (2.0, 1.0) and SUPPORT_BUFFER_EV == 1.0
    assert read_halfwidth_ev() == 0.5
    part = BandPartition(protected_mask=np.ones(3, bool), in_range_mask=np.ones(3, bool))
    session = {}
    inputs = _inputs("cover", 0, session, grid=np.arange(-12.0, 8.0 + 1e-9, 0.25))
    grid0, event = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 9.9]]), 0.0))
    assert event == "plan"                                  # the first plan: E + 2 eV
    assert 9.9 + 2.0 <= grid0[-1] < 9.9 + 2.0 + 0.25
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
    support = _sc_sampled_support(_inputs("cover", 0), part, np.array([[0.3, 9.9]]), 0.0)
    assert support.event == "one-shot"


def test_unset_grid_edges_derive_the_grid_from_the_bands():
    """Owner 2026-09-25: omega_min/max are optional. Unset, the requested grid
    is the sample next to E_F on each side and the bands set the rest; set,
    they are a minimum extent kept on every map."""
    from gw.gw_config import DynamicSigmaConfig
    base = dict(omega_step_ev=0.25, regularization_ev=0.25, window_edge_factor=1.0,
                fermi_reference="vbm",
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
    from gw.qp_support import plan_support_ev
    requested = np.arange(-0.25, 0.25 + 1e-9, 0.25)
    grown, envelope = plan_support_ev(unset, requested, np.array([[-6.0, 2.0]]),
                                      np.ones((1, 2), bool), 0)
    assert grown[0] == -8.0 and grown[-1] == 4.0 and envelope == (-8.0, 4.0)
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


def test_a_runaway_state_cannot_grow_the_grid():
    """Owner 2026-09-27: the support stays inside the deck request joined with
    the requested quasiparticles' E_in +/- pad.  A requested state whose Z at
    the previous map lies outside (0, 1] (Na 8^3 b63: Z = -382, eqp1 -1031 eV)
    does not move it, on the map-1 plan or on a held map; a quasiparticle
    just past the edge still extends it.  Roots are no input at all."""
    part = BandPartition(protected_mask=np.ones(3, bool), in_range_mask=np.ones(3, bool))
    session = {}
    inputs = _inputs("cover", 0, session, grid=np.arange(-12.0, 8.0 + 1e-9, 0.25))
    grid0, _ = _commit(session, _sc_sampled_support(
        inputs, part, np.array([[-5.0, 0.3, 9.9]]), 0.0))
    # Map 1: the third state ran away with Z = -0.003; the plan ignores it.
    runaway = np.array([[-5.0, 0.3, -1031.0]])
    z_bad = np.array([[0.8, 0.9, -0.003]])
    support = _sc_sampled_support(inputs, part, runaway, 0.0, None, z_bad)
    grid1, event = _commit(session, support)
    assert event == "re-plan"
    np.testing.assert_array_equal(grid1, np.arange(-12.0, 8.0 + 1e-9, 0.25))
    np.testing.assert_array_equal(support.no_qp, [[False, False, True]])
    assert support.envelope == (-6.0, 1.3)
    # Map 2: still running away (+500 eV, Z = 2.8); the grid holds.
    held = _sc_sampled_support(inputs, part, np.array([[-5.0, 0.3, 500.0]]), 0.0,
                               None, np.array([[0.8, 0.9, 2.8]]))
    assert held.event == "hold"
    np.testing.assert_array_equal(held.grown, grid1)
    # The same energy with Z = 1 (a quasiparticle, flat Sigma) extends it.
    moved = _sc_sampled_support(inputs, part, np.array([[-5.0, 0.3, 8.0]]), 0.0,
                                None, np.array([[0.8, 0.9, 1.0]]))
    assert moved.event == "extend" and 9.0 <= moved.grown[-1] < 9.25


def test_the_support_envelope_refuses_a_grid_grown_past_it():
    """The invariant check: a grid that reaches past D joined with the
    requested envelope (for instance one grown from a root or from every
    band) is a refusal; a grid inside it passes."""
    from gw.qp_support import assert_support_in_envelope, grow_support_ev
    deck = np.arange(-12.0, 8.0 + 1e-9, 0.25)
    envelope = (-7.0, 11.9)
    inside = grow_support_ev(deck, np.array([[9.9]]), np.ones((1, 1), bool), 0.25,
                             pad_ev=2.0, trigger_ev=2.0)
    assert_support_in_envelope(inside, deck, envelope, 0.25, context="test")
    grown_by_a_root = grow_support_ev(deck, np.array([[-1031.0]]), np.ones((1, 1), bool),
                                      0.25, pad_ev=2.0, trigger_ev=2.0)
    with pytest.raises(ValueError, match="GATE sigma_support_envelope"):
        assert_support_in_envelope(grown_by_a_root, deck, envelope, 0.25, context="test")
    with pytest.raises(ValueError, match="GATE sigma_support_envelope"):
        assert_support_in_envelope(np.arange(-12.0, 12.5, 0.25), deck, envelope, 0.25,
                                   context="test")


def test_the_requested_window_is_e_f_plus_minus_w_and_multiplet_closed():
    """Owner 2026-09-27: the requested Sigma states lie within E_F +/- W
    (sigma_window_ev); a state outside it cannot grow the grid, and the edge
    never splits a multiplet at one k."""
    from gw.qp_support import sigma_window_states
    e = np.array([[-12.0, -3.0, 9.9995, 10.0004, 23.8],
                  [-9.0, 0.0, 10.5, 10.5, 12.0]])
    np.testing.assert_array_equal(sigma_window_states(e, 10.0),
                                  [[False, True, True, True, False],
                                   [True, True, False, False, False]])
    np.testing.assert_array_equal(sigma_window_states(e, None), np.ones(e.shape, bool))
    with pytest.raises(ValueError, match="sigma_window_ev"):
        sigma_window_states(e, 0.0)
    part = BandPartition(protected_mask=np.ones(5, bool), in_range_mask=np.ones(5, bool))
    inputs = _inputs("cover", 0)
    inputs.config.sigma.window_ev = 10.0
    support = _sc_sampled_support(inputs, part, e, 0.0)
    assert support.grown[0] == -12.0                    # -12 is outside W: the deck edge holds
    assert 10.0004 + 2.0 <= support.grown[-1] < 10.0004 + 2.0 + 0.25   # +23.8 does not grow it
    np.testing.assert_array_equal(support.requested[0], [False, True, True, True, False])
