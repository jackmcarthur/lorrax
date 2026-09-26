"""Pole-census refusals of the box planner's summaries (``gw.mpa.sigma_windows``)."""
import jax.numpy as jnp
import numpy as np

from gw.mpa import sigma_windows as SW
from gw.ppm_windows import _SigmaBranch


def _branches():
    omega = np.asarray([0.0, 0.5, 1.0])
    idx = np.arange(omega.size)
    cond = jnp.asarray([[0.2, 1.15, 1.5]])
    # A slightly negative occupied/empty edge is allowed when the actual
    # denominator remains sign-definite; occupation policy is separate.
    val = jnp.asarray([[-0.02, 1.1, 1.45]])
    mask = jnp.ones_like(cond, dtype=bool)
    return [
        _SigmaBranch("pos_cond", cond, mask, "cond", False, omega, idx),
        _SigmaBranch("pos_val", val, mask, "val", False, omega, idx),
        _SigmaBranch("neg_cond", cond, mask, "cond", True, omega, idx),
        _SigmaBranch("neg_val", val, mask, "val", True, omega, idx),
    ]


def test_live_negative_real_pole_refuses():
    poles = jnp.asarray([[[[-0.2 - 0.1j]]]])
    try:
        SW.summarize_sigma_poles(
            poles, jnp.ones_like(poles), _branches(),
            regularization_width_ry=0.2,
            edge_factor=1.5)
    except ValueError as exc:
        assert "Re Omega <= 0" in str(exc)
    else:
        raise AssertionError("an unsupported negative-real pole was omitted")
