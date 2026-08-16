"""Shared runtime configuration for minimax-based screening and sigma workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimaxConfig:
    """Canonical minimax settings shared by χ₀ screening and Σ^c windows.

    One frozen, all-scalar (hashable) currency for both quadrature
    consumers.  Screening builds it with an ``energy_reference``; the
    Σ^c-window path builds it with the crossing-quadrature controls
    (``crossing_max_nodes`` / ``crossing_eps_q``).  Each side leaves the
    other's knobs at their defaults — the merged class carries the union
    so the Σ-quad defaults live at exactly one site.
    """

    target_error: float = 1.0e-6
    max_nodes: int = 64
    regenerate_tables: bool = False
    #: THE energy zero the band energies are shifted by before the Laplace
    #: fit.  No deck key: the ``minimax_energy_reference`` one was deleted
    #: in 0.1.0 because the shift is algebraically neutral for chi0/W (only
    #: E_c - E_v enters), so no value of it could change a result.  The
    #: field stays because ``resolve_minimax_energy_reference`` accepts a
    #: float or a callable and a future consumer on a non-neutral integrand
    #: would need it; "midgap" is its one shipped value.
    energy_reference: str | float | int | None = "midgap"
    crossing_max_nodes: int = 500
    crossing_eps_q: float = 1.0e-3

    @property
    def use_shipped_tables(self) -> bool:
        return not bool(self.regenerate_tables)


