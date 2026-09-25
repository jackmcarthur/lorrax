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


def _inputs(policy, n_frozen):
    cfg = NS(compute_mode=NS(is_dynamic=True), omega_grid_ev=GRID,
             sigma=NS(out_of_grid=policy, omega_min_ev=-12.0, omega_max_ev=8.0,
                      omega_step_ev=0.25),
             sc=NS(frozen_core_bands=n_frozen))
    return NS(config=cfg, fixed_quadrature_session=None)


def test_cover_grows_over_every_non_frozen_identity_and_nothing_else():
    part = BandPartition(protected_mask=np.ones(4, bool), in_range_mask=np.ones(4, bool))
    e = np.array([[-100.0, -5.0, 9.8, 11.7]])            # rel mu; band 1 is semicore
    _, grown, _, _ = _sc_sampled_support(_inputs("cover", 1), part, e, 0.0)
    assert grown[0] == GRID[0]                           # frozen core does not grow it
    assert 11.7 + 0.5 + 1.17 <= grown[-1] < 11.7 + 0.5 + 1.17 + 0.25
    # Bounded by the spectrum it covers: unfrozen semicore reaches E - pad(E).
    _, grown, _, _ = _sc_sampled_support(_inputs("cover", 0), part, e, 0.0)
    assert -100.0 - 10.5 - 0.25 < grown[0] <= -100.0 - 10.5
    # ... unless the W model calls it inactive (shared_pole_recipe.active_band_mask):
    # then no fc key is needed and the semicore keeps Sigma(0) as under static.
    from gw.shared_pole_recipe import active_band_mask
    from common.units import RYD_TO_EV
    active = active_band_mask(e / RYD_TO_EV, 0.0)
    np.testing.assert_array_equal(active, [False, True, True, True])
    _, grown, _, _ = _sc_sampled_support(_inputs("cover", 0), part, e, 0.0, active)
    assert grown[0] == GRID[0] and grown[-1] > 11.7
    for policy in ("static", "clamp"):                   # window rule: +9.8 and +11.7 lie
        _, grown, _, _ = _sc_sampled_support(_inputs(policy, 1), part, e, 0.0)
        np.testing.assert_array_equal(grown, GRID)       # beyond the +9.44 padded top


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
    from gw.sc_iteration import _sigma_frame_mu_ev
    from common.units import RYD_TO_EV
    e_ev = np.array([[-14.3, -5.1, -0.8, 3.0], [-14.2, -5.3, -0.9, 3.2]])
    e_ry, step_ry = e_ev / RYD_TO_EV, -3.0 / RYD_TO_EV       # 2 occupied per k
    wfn = NS(efermi=-4.332 / RYD_TO_EV, vbm=-5.4 / RYD_TO_EV)
    def frame(mode, ref):
        inputs = NS(config=NS(compute_mode=mode, sigma=NS(fermi_reference=ref)), wfn=wfn)
        return _sigma_frame_mu_ev(inputs, e_ry, step_ry, None)
    np.testing.assert_allclose(frame(ComputeMode.GN_PPM, "midgap"), 0.5 * (-5.1 - 0.9), atol=1e-12)
    np.testing.assert_allclose(frame(ComputeMode.GN_PPM, "vbm"), -5.1, atol=1e-12)
    np.testing.assert_allclose(frame(ComputeMode.MPA, "midgap"), -4.332, atol=1e-12)
