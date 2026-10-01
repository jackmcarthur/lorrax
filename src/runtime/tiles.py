"""The one fixed tile bound every streamed loop sizes against (owner, 2026-09-30).

A loop that streams over k, q, bands, centroids, samples or rows takes the
most units whose per-rank scaling bytes fit :data:`TILE_BYTES`. The tile
comes from the loop's own shapes alone: it never reads free device memory or
the deck budget, so every rank computes the same tile, and a result never
depends on how much memory a run was given. 1 GiB per rank saturates the
kernels that stream (R153's mode-11 scratch bound; DEVROOMA measured no
site but the response sample group losing > 10 % per map at 256 MiB on
Fe 4³ and Na 8³).

The exceptions size from the deck budget (``memory_per_device_gb``) through
a ledger: the shared-pole response sample group
(``gw.response_bank.response_group_size``, the capacity ledger), where a
larger group buys > 10 % per map, and the Galerkin whole-state planner
(``isdf.galerkin``), whose capacity also bounds its resident rows.
"""

#: Per-rank bytes of one tile's scaling set.
TILE_BYTES = 1 << 30


def tile_units(per_unit_bytes: float, n_units: int, *, floor: int = 1) -> int:
    """Units per tile: the most whose ``units · per_unit_bytes`` fits
    :data:`TILE_BYTES`, at least ``floor`` and at most ``n_units``."""
    n_units = int(n_units)
    fit = int(TILE_BYTES // max(float(per_unit_bytes), 1.0))
    return max(min(int(floor), n_units), min(n_units, fit))
