"""exciton_bands Gamma gate: report the whole overlap spectrum.

RED TWIN recorded in FIX_driver_blockers.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# ===========================================================================
# 3. the Gamma gate reports the whole overlap spectrum
# ===========================================================================
def test_gamma_gate_reports_the_overlap_spectrum():
    """``min-sval`` alone cannot separate a lost subspace direction from a
    uniform normalisation error; the spectrum can, and the gate must print it.

    RED TWIN: revert to tracking only ``smin`` in
    ``exciton_bands.gate_htransform_vs_stored`` and the ``overlap svals`` line
    disappears — which is the state in which ``min-sval = 0.4626`` was
    undiagnosable.  With the line, ``[0.9985 0.4626]`` reads immediately as one
    boundary direction lost to a cut degenerate multiplet.
    """
    import bse.exciton_bands as eb

    src = Path(eb.__file__).read_text(encoding="utf-8")
    assert "overlap svals at the worst k" in src
    assert "sv_worst" in src and "k_worst" in src
