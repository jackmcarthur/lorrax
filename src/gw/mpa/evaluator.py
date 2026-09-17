"""MPA scalar oracle and physical door to the minimax quadrature service.

The four positive-time rule builders are owned by ``minimax.damped_rules``.
The names below remain for existing GW consumers and tests; no node selection
or error-bound calculation is duplicated here.
"""
from __future__ import annotations

import numpy as np
from ffi import _services
from gw.mpa import sample_plan

_services.ensure_on_path()
import minimax as _minimax  # noqa: E402

DEFAULT_REL_TOL = _minimax.DEFAULT_DAMPED_REL_TOL
DEFAULT_WAVELENGTHS_PER_PANEL = _minimax.DEFAULT_WAVELENGTHS_PER_PANEL

damped_line_rule = _minimax.damped_line_rule
damped_rectangle_rule = _minimax.damped_rectangle_rule
damped_rectangle_gauss_rule = _minimax.damped_rectangle_gauss_rule
damped_rectangle_positive_rule = _minimax.damped_rectangle_positive_rule


def damped_kernel(z, delta):
    """``K_z(Delta) = -2 Delta / (Delta**2 - z**2)``.  Host-side numpy.

    The exact value of the authoritative MPA chapter's damped-tau integral; see
    the MPA kernel derivation.  Broadcasts ``z`` against
    ``delta``: pass ``z`` with a trailing axis of ones to get the full
    ``(n_z, n_delta)`` table.

    Refuses a ``(z, Delta)`` pair that sits on the kernel's own pole,
    because the number there is infinite and a caller who meant to
    sample there has a geometry bug rather than a numerical one.
    """

    zc = np.asarray(z, dtype=np.complex128)
    d = np.asarray(delta, dtype=np.float64)
    denom = d ** 2 - zc ** 2
    if np.any(denom == 0.0):
        raise ValueError(
            "GATE off_kernel_pole: a sample z coincides with +/- a "
            "transition energy Delta, where K_z(Delta) is infinite. "
            "FALSE case: no sample point equals +/- any transition "
            "energy -- which every sample with varpi > 0 satisfies "
            "automatically.")
    return sample_plan.KERNEL_FACTOR * d / denom
