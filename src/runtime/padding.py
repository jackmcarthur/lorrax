"""Shape-padding helpers for sharded arrays.

The driver-level building blocks for the n_rmu / n_band / etc. padding
contract: arrays in memory may be padded to mesh-divisibility (so JIT
boundaries don't trigger uneven-sharding all-gather), but files on
disk store the logical (unpadded) extent so they can be re-read on
any process count.

This module provides only the **shape arithmetic** — the array-level
``pad_to_mesh`` / ``unpad`` helpers that copy with zero-fill belong in
the next agent's padding-refactor branch.  Keeping this here as a
single source of truth for the rounding rule.
"""
from __future__ import annotations

from typing import Sequence


def pad_shape_to_mesh(shape: Sequence[int], partition_spec, mesh) -> tuple[int, ...]:
    """Round each dim up so the resulting shape divides cleanly into a
    sharded layout on ``mesh`` with the given ``partition_spec``.

    A dim sharded by mesh axis ``a`` (or by a tuple ``(a, b, ...)``)
    must be divisible by ``mesh.shape[a]`` (or the product of mesh
    sizes).  Unsharded dims pass through unchanged.

    Logical → physical example, mesh ``{'x': 4, 'y': 4}``,
    spec ``P(None, None, ('x','y'))``, shape ``(9, 60, 668)``:
    ``668 % 16 = 12`` → 668 → 672.  Returns ``(9, 60, 672)``.

    Used by drivers to know how much zero-padding they need before a
    JIT call whose interior demands product-divisible sharding.
    """
    physical = [int(s) for s in shape]
    for d, axis in enumerate(partition_spec):
        if axis is None:
            continue
        axes = (axis,) if isinstance(axis, str) else tuple(axis)
        prod = 1
        for ax in axes:
            prod *= int(mesh.shape[ax])
        if prod <= 1:
            continue
        rem = physical[d] % prod
        if rem:
            physical[d] += (prod - rem)
    return tuple(physical)


def logical_shape_from_padded(padded: Sequence[int], logical: Sequence[int]) -> tuple[int, ...]:
    """Trivial validator: logical[d] <= padded[d] for every axis.

    Returns ``logical`` unchanged after the assertion.  Mainly a
    documentation hook so callers tagging padding metadata write the
    same dataflow direction every time.
    """
    p = tuple(int(s) for s in padded)
    l = tuple(int(s) for s in logical)
    if len(p) != len(l):
        raise ValueError(
            f"rank mismatch: padded={p} (rank {len(p)}) vs logical={l} (rank {len(l)})"
        )
    for d, (pd, ld) in enumerate(zip(p, l)):
        if ld > pd:
            raise ValueError(
                f"axis {d}: logical {ld} > padded {pd}"
            )
    return l


__all__ = ["pad_shape_to_mesh", "logical_shape_from_padded"]
