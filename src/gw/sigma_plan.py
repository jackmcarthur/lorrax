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


def resolve_delivered_max_direct_terms() -> int:
    """Resolve the per-branch ceiling on exact direct terms.

    Direct summation is an escape hatch, not a route: the owner ruling
    retires per-pair stages, and the planner must refuse rather than
    exceed this ceiling. The default keeps the hatch at a few dozen
    terms; zero is a legal request and forces pure quadrature.
    """
    raw = os.environ.get("LORRAX_DELIVERED_MAX_DIRECT_TERMS", "32").strip()
    value = raw or "32"
    try:
        ceiling = int(value)
    except ValueError as error:
        raise ValueError(
            "LORRAX_DELIVERED_MAX_DIRECT_TERMS must be a nonnegative "
            f"integer; got {raw!r}") from error
    if ceiling < 0:
        raise ValueError(
            "LORRAX_DELIVERED_MAX_DIRECT_TERMS must be a nonnegative "
            f"integer; got {raw!r}")
    return ceiling


__all__ = ["resolve_delivered_max_direct_terms", "resolve_delivered_tau_grid",
           "resolve_sigma_plan"]
