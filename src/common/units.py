"""Physical-unit conversion constants used across LORRAX.

Single source of truth for the Rydberg ↔ eV conversion factor; previously
inlined as ``ryd2ev = 13.6056980659`` at ~25 sites across the codebase.
"""
from __future__ import annotations

RYD_TO_EV: float = 13.6056980659
EV_TO_RYD: float = 1.0 / RYD_TO_EV
