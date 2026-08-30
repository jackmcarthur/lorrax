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


def resolve_delivered_tau_grid() -> str:
    """Resolve the delivered planner's per-branch tau-grid policy.

    ``free`` preserves independently fitted window grids. ``shared`` makes
    every quadrature window of a causal branch use one identical grid with
    independently re-solved weights. Unknown values refuse before planning.
    """
    raw = os.environ.get("LORRAX_DELIVERED_TAU_GRID", "free").strip().lower()
    mode = raw or "free"
    if mode not in ("free", "shared"):
        raise ValueError(
            "LORRAX_DELIVERED_TAU_GRID must be 'free' or 'shared'; "
            f"got {raw!r}")
    return mode


def resolve_delivered_plan_cache() -> str | None:
    """Resolve the optional measured-problem delivered-plan cache path.

    The cache stores only fitted quadrature rules.  Its fingerprint is made
    after the live pole-measure pass, so changing restart tensors or any
    planning input invalidates the receipt instead of reusing stale nodes.
    """
    raw = os.environ.get("LORRAX_DELIVERED_PLAN_CACHE", "").strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


__all__ = ["resolve_delivered_plan_cache", "resolve_delivered_tau_grid",
           "resolve_sigma_plan"]
