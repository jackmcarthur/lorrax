"""MPA scalar oracle and physical entry point to the minimax quadrature service.

The four positive-time rule builders are exposed through the public ``minimax``
API. The names below remain as import-compatible aliases for existing callers
and tests; no node selection or error-bound calculation is duplicated here.
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

