"""Shared runtime configuration for minimax-based screening and sigma workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimaxConfig:
    """Canonical minimax settings shared by screening and PPM construction."""

    target_error: float = 1.0e-6
    max_nodes: int = 64
    regenerate_tables: bool = False
    energy_reference: str | float | int | None = "midgap"

    @property
    def use_shipped_tables(self) -> bool:
        return not bool(self.regenerate_tables)


@dataclass(frozen=True)
class SigmaQuadratureConfig:
    """Quadrature controls specific to Sigma^c window construction."""

    target_error: float = 1.0e-6
    max_nodes: int = 64
    crossing_max_nodes: int = 500
    crossing_eps_q: float = 1.0e-3
    regenerate_tables: bool = False

    @property
    def use_shipped_tables(self) -> bool:
        return not bool(self.regenerate_tables)


