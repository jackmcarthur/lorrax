"""Shared runtime configuration for minimax-based screening and sigma workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimaxConfig:
    """Canonical minimax settings shared by χ₀ screening and Σ^c windows.

    One frozen, all-scalar (hashable) currency for screening quadratures.
    Dynamic Sigma's denominator-box rule has its own service-owned controls.
    """

    target_error: float = 1.0e-6
    max_nodes: int = 64
    regenerate_tables: bool = False
    energy_reference: str | float | int | None = "midgap"

    @property
    def use_shipped_tables(self) -> bool:
        return not bool(self.regenerate_tables)

