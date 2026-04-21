"""Design-of-experiments helpers for AOT memory sweeps.

The "smart DoE" pattern: from a baseline (sys, knobs, mesh), produce a
one-at-a-time axis sweep where each factor varies independently while the
others stay at baseline.  For N factors at K levels each, this is
``1 + N*(K-1)`` points instead of ``K^N``.

A separate ``build_product_check`` helper generates the pair of points
that verify a "product-only" dim: doubling one factor and halving another
should leave predicted peak unchanged if they enter the fit only as a
product.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Callable, Iterable

from .core import SysDims, Knobs, MeshSpec


def build_doe_axes(
    baseline_sys: SysDims,
    baseline_knobs: Knobs,
    baseline_mesh: MeshSpec,
    *,
    sys_axes: dict[str, list] | None = None,
    knob_axes: dict[str, list[int]] | None = None,
    mesh_axes: list[MeshSpec] | None = None,
) -> list[tuple[SysDims, Knobs, MeshSpec]]:
    """One-at-a-time sweep around the baseline.

    Each axis is a list of alternate values for one factor.  The baseline
    is emitted once; then each axis emits its non-baseline points.

    sys_axes: {"n_k": [32, 128], "n_rmu": [1200, 3200]}   — replace field by field
    knob_axes: {"chunk_r": [512, 2048]}                   — set one knob to a value
    mesh_axes: [MeshSpec(1,4), MeshSpec(4,1)]             — test px/py asymmetry
    """
    out: list[tuple[SysDims, Knobs, MeshSpec]] = [
        (baseline_sys, baseline_knobs, baseline_mesh)
    ]
    if sys_axes:
        for field_name, values in sys_axes.items():
            for v in values:
                new_sys = replace(baseline_sys, **{field_name: v})
                if new_sys == baseline_sys:
                    continue
                out.append((new_sys, baseline_knobs, baseline_mesh))
    if knob_axes:
        for knob_name, values in knob_axes.items():
            base_kv = dict(baseline_knobs.values)
            for v in values:
                kv = dict(base_kv)
                kv[knob_name] = v
                new_knobs = Knobs.of(**kv)
                if new_knobs == baseline_knobs:
                    continue
                out.append((baseline_sys, new_knobs, baseline_mesh))
    if mesh_axes:
        for m in mesh_axes:
            if m == baseline_mesh:
                continue
            out.append((baseline_sys, baseline_knobs, m))
    return out


def build_product_check(
    baseline_sys: SysDims, baseline_knobs: Knobs, baseline_mesh: MeshSpec,
    *, field_a: str, field_b: str, factor: int = 2,
) -> list[tuple[SysDims, Knobs, MeshSpec]]:
    """Two-point test that ``field_a * field_b`` enters as a product only.

    Emits (2·a, b/factor) and (a/factor, 2·b).  If the fit predicts the
    same peak for both, a·b is confirmed product-only in every primitive.
    """
    a0 = getattr(baseline_sys, field_a)
    b0 = getattr(baseline_sys, field_b)
    s1 = replace(baseline_sys, **{field_a: a0 * factor, field_b: b0 // factor})
    s2 = replace(baseline_sys, **{field_a: a0 // factor, field_b: b0 * factor})
    return [(s1, baseline_knobs, baseline_mesh),
            (s2, baseline_knobs, baseline_mesh)]
