from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np


DenomName = Literal["antiresonant", "resonant_below", "crossing", "resonant_above"]
QuadratureKind = Literal["GL", "HGL"]
DampingKind = Literal["exponential", "oscillatory"]


@dataclass(frozen=True)
class PoleBlock:
    """Poles plus matrix residues in canonical X/Y layouts."""

    psi_X: jax.Array
    psi_Y: jax.Array
    energies: jax.Array
    mask: jax.Array
    label: str = ""


@dataclass(frozen=True)
class DenomType:
    """One denominator branch with one quadrature grid."""

    name: DenomName
    quadrature_kind: QuadratureKind
    damping_kind: DampingKind
    sign: int
    omega_indices: np.ndarray
    omega_values: np.ndarray
    E_gap_eff: float
    E_bw_eff: float
    z_lm: float
    tau: np.ndarray
    w: np.ndarray


@dataclass(frozen=True)
class WindowExecutionPlan:
    """Execution plan for one window pair."""

    window_index: int
    window_pair: object
    denom_types: tuple[DenomType, ...]
    E_gap: float
    E_bw: float


@dataclass(frozen=True)
class IntegrationPlan:
    """Top-level integration plan for one chi run."""

    omega: np.ndarray
    policy: str
    window_plans: tuple[WindowExecutionPlan, ...]
