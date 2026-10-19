"""The one-shot reads only its off-request coarse states on coarse windows (CPU only, seconds).

``qp_support.oneshot_support_ev`` on an AgI-like ladder (E - mu, eV): a deep level
at -91.3, two at -58.35/-45, a shallow semicore level at -13.6 inside the W model's
active depth, valence to -5.3 and conduction above +0.6.  The coarse floor sits
under the valence (the 4 eV gap rule), so every level below -5.3 is coarse.

1. The near grid is the plan without coarse windows: the requested set alone sets
   it, so the requested coarse state (-13.6) keeps its deck-eta read.
2. Every coarse state outside the requested set has its Z stencil inside a coarse
   window below the near grid; the requested coarse state is in no window.
3. Without a coarse state outside the requested set the support is the plain plan,
   and a deck lo:hi:eta window warns and is ignored (an SC map refuses it).
"""
from types import SimpleNamespace

import numpy as np

SIGMA = SimpleNamespace(omega_step_ev=0.25, coarse_windows_ev=lambda: ())
DECK = np.array([-0.25, 0.0, 0.25])


def test_only_off_request_coarse_states_read_coarse_windows():
    from gw.qp_support import oneshot_support_ev, plan_support_ev, read_halfwidth_ev
    levels = np.array([-91.3, -58.35, -45.0, -13.6, -5.3, -2.0, -0.6, 0.6, 3.0, 19.9])
    energy = np.vstack([levels, levels + 0.02])
    requested = np.broadcast_to(energy.max(axis=0) >= -15.0, energy.shape)
    coarse = energy < -5.35
    grid, windows = oneshot_support_ev(SIGMA, DECK, energy, requested, coarse)
    near, _ = plan_support_ev(SIGMA, DECK, energy, requested)
    assert windows, "the deep states must read coarse windows"
    np.testing.assert_array_equal(grid[-near.size:], near)
    h = read_halfwidth_ev()
    for e in energy[coarse & ~requested]:
        assert any(w[0] <= e - h and e + h <= w[1] for w in windows), e
    assert all(w[1] < near[0] for w in windows)
    assert not any(w[0] <= e <= w[1] for e in energy[:, 3] for w in windows)


def test_no_off_request_coarse_state_keeps_the_plain_plan():
    from gw.qp_support import oneshot_support_ev, plan_support_ev
    energy = np.array([[-13.6, -5.3, -2.0, -0.6, 0.6, 3.0]])
    requested = np.ones(energy.shape, dtype=bool)
    grid, windows = oneshot_support_ev(SIGMA, DECK, energy, requested, energy < -5.35)
    near, _ = plan_support_ev(SIGMA, DECK, energy, requested)
    assert windows == ()
    np.testing.assert_array_equal(grid, near)


def test_a_user_coarse_window_with_no_state_to_serve_warns_and_is_ignored():
    import pytest
    from gw.qp_support import coarse_windows_plan, oneshot_support_ev, plan_support_ev
    sigma = SimpleNamespace(omega_step_ev=0.25, coarse_windows_ev=lambda: ((-16.0, -11.0, 1.0),))
    energy = np.array([[-13.6, -5.3, -2.0, -0.6, 0.6, 3.0]])
    requested = np.ones(energy.shape, dtype=bool)
    near, _ = plan_support_ev(sigma, DECK, energy, requested)
    for coarse in (energy < -5.35, np.zeros(energy.shape, dtype=bool)):
        with pytest.warns(RuntimeWarning, match="no state to serve in this one-shot"):
            grid, windows = oneshot_support_ev(sigma, DECK, energy, requested, coarse)
        assert windows == ()
        np.testing.assert_array_equal(grid, near)
    # An SC map (coarse_windows_plan with an empty coarse class) still refuses.
    with pytest.raises(ValueError, match="GATE sigma_coarse_window"):
        coarse_windows_plan(sigma, energy, np.zeros(energy.shape, dtype=bool), float(near[0]))
