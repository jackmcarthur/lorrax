"""GN-PPM probe quadrature comes from the analytic response rule (CRIU 2026-09-22).

The i*omega_p chi0 sample on a measured-broken-TR deck needs the even kernel
x/(x^2+wp^2) and the odd kernel wp/(x^2+wp^2) on one set of positive times.
``build_imag_probe_response_rule`` takes both from ``minimax.response_laplace_rule``
(the shared-pole bank's analytic placement).  Checked here against the exact
kernels on the window, not against the runtime-minimax rule it replaces.
"""
import numpy as np
import pytest

from gw.minimax_config import MinimaxConfig
from gw.minimax_screening import (LaplaceMinimaxQuadrature,
                                  build_imag_probe_response_rule)


def _window(x_min, x_max):
    return LaplaceMinimaxQuadrature(x_min=x_min, x_max=x_max, tau=np.ones(1),
                                    alpha=np.ones(1), max_error=0.0)


@pytest.mark.parametrize("x_min,x_max,wp", [(0.068, 6.3, 2.0), (0.5, 4.0, 2.0), (0.068, 6.3, 0.5)])
@pytest.mark.parametrize("odd", [False, True])
def test_probe_rule_reconstructs_both_kernels(x_min, x_max, wp, odd):
    cfg = MinimaxConfig(target_error=1e-6)
    q = build_imag_probe_response_rule(_window(x_min, x_max), wp, cfg, with_odd_kernel=odd)
    assert np.all(q.tau > 0) and q.n_odd_extra == 0
    x = np.geomspace(x_min, x_max, 20001)
    B = np.exp(-np.outer(x, q.tau))
    even = x / (x * x + wp * wp)
    assert np.max(np.abs(B @ q.alpha - even) / even) < 5e-6       # relative, dense grid
    if odd:
        k = wp / (x * x + wp * wp)
        assert q.alpha_odd is not None and q.alpha_odd.shape == q.alpha.shape
        assert np.max(np.abs(B @ q.alpha_odd - k) / k) < 5e-6
    else:
        assert q.alpha_odd is None
    assert "response_laplace_rule" in q.provenance and "PASS" in q.provenance
