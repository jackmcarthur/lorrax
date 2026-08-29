"""One fail-closed selector for every frequency-dependent Sigma route."""

from __future__ import annotations

import os


def resolve_sigma_plan() -> str:
    """Resolve ``LORRAX_SIGMA_PLAN`` as ``panes`` or ``delivered``.

    Unset and blank preserve the incumbent panes planner. Unknown values
    refuse before either GN-PPM or MPA performs planning work.
    """
    raw = os.environ.get("LORRAX_SIGMA_PLAN", "panes").strip().lower()
    mode = raw or "panes"
    if mode not in ("panes", "delivered"):
        raise ValueError(
            "LORRAX_SIGMA_PLAN must be 'panes' or 'delivered'; "
            f"got {raw!r}")
    return mode


__all__ = ["resolve_sigma_plan"]
