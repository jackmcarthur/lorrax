"""Fail-closed dynamic-Sigma route selector for production and pane controls."""

from __future__ import annotations

import os


def resolve_sigma_plan() -> str:
    """Resolve ``LORRAX_SIGMA_PLAN`` as ``box`` or ``panes``.

    The box rule is the production default.  ``panes`` is retained solely as
    the independent comparison route; no third planner is reachable.
    """
    raw = os.environ.get("LORRAX_SIGMA_PLAN", "box").strip().lower()
    mode = raw or "box"
    if mode not in ("box", "panes"):
        raise ValueError(
            "LORRAX_SIGMA_PLAN must be 'box' (default) or 'panes' "
            "(control); "
            f"got {raw!r}")
    return mode


__all__ = ["resolve_sigma_plan"]
