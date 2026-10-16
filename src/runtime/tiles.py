"""The one fixed tile bound every streamed loop sizes against.

A loop that streams over k, q, bands, centroids, samples or rows takes the
most units whose per-rank scaling bytes fit :data:`TILE_BYTES`. The tile
comes from the loop's own shapes alone: it never reads free device memory or
the deck budget, so every rank computes the same tile, and a result never
depends on how much memory a run was given. 1 GiB per rank saturates the
kernels that stream (the mode-11 split-arm scratch is bounded by it).

Planners that still size from the deck budget (``memory_per_device_gb``) are
listed under "Not yet conforming" in docs/architecture/decisions.md
(the fixed-tile ruling); each converts to :func:`tile_units`.
"""

#: Per-rank bytes of one tile's scaling set.
TILE_BYTES = 1 << 30


def tile_units(per_unit_bytes: float, n_units: int, *, floor: int = 1) -> int:
    """Units per tile: the most whose ``units · per_unit_bytes`` fits
    :data:`TILE_BYTES`, at least ``floor`` and at most ``n_units``."""
    n_units = int(n_units)
    fit = int(TILE_BYTES // max(float(per_unit_bytes), 1.0))
    return max(min(int(floor), n_units), min(n_units, fit))


def tile_plan(per_unit_bytes: float, n_units: int) -> tuple[int, tuple[int, ...]]:
    """``(width, starts)``: equal-width tiles covering ``n_units`` units.

    ``width`` is :func:`tile_units` evened out, so the fewest tiles that fit
    :data:`TILE_BYTES` cover the axis with the least overlap:
    ``n = ceil(n_units / fit)``, ``width = ceil(n_units / n)``.  ``starts``
    are ``j·width`` for ``j < n``, with the last clamped to
    ``n_units − width``, so every tile has the same width and one compiled
    program serves all of them; no tile runs past the end and no pad unit is
    ever introduced.  A unit inside the overlap is computed twice by the same
    program from the same operands and written twice with the same value.
    """
    n_units = int(n_units)
    if n_units < 1:
        raise ValueError(
            f"GATE tile_plan_extent: got: n_units={n_units}; want: >= 1; "
            f"why: an empty axis has no tile")
    fit = tile_units(per_unit_bytes, n_units)
    n_tiles = -(-n_units // fit)
    width = -(-n_units // n_tiles)
    starts = tuple(min(j * width, n_units - width) for j in range(n_tiles))
    return width, starts
