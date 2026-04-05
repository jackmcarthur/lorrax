"""Shared runtime configuration for minimax-based screening and sigma workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def minimax_config_from_params(params: dict[str, Any]) -> MinimaxConfig:
    """Build a shared minimax config from parsed GW input parameters."""

    return MinimaxConfig(
        target_error=float(params.get("minimax_target_error", 1.0e-6)),
        max_nodes=int(params.get("minimax_max_nodes", 64)),
        regenerate_tables=bool(params.get("regenerate_minimax_tables", False)),
        energy_reference=params.get("minimax_energy_reference", "midgap"),
    )


def sigma_quadrature_config_from_params(
    params: dict[str, Any],
    *,
    crossing_eps_q: float = 1.0e-3,
) -> SigmaQuadratureConfig:
    """Build Sigma^c quadrature settings from parsed GW input parameters."""

    max_nodes = int(params.get("ppm_sigma_max_nodes", 64))
    return SigmaQuadratureConfig(
        target_error=float(params.get("ppm_sigma_target_error", 1.0e-6)),
        max_nodes=max_nodes,
        crossing_max_nodes=max(500, max_nodes),
        crossing_eps_q=float(crossing_eps_q),
        regenerate_tables=bool(params.get("regenerate_minimax_tables", False)),
    )
