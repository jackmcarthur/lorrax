"""Shared dynamic-Sigma branch records and broadening policy.

GN/HL-PPM no longer owns a dynamic window builder.  A PPM fit is persisted as
a one-pole MPA store and the MPA planner constructs every production and pane
control window.  This module retains only the small records/helpers consumed
by that shared planner plus the common regularization resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np

from .efermi import (
    OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
    band_in_occupation_window,
    occupation_weight_floor,
)
from .minimax_screening import MinimaxNodes


class SigmaRegularization(NamedTuple):
    """Effective dynamic-Sigma broadening and its provenance.

    All numeric fields are in Ry except properties named ``*_ev``.
    """

    requested_ry: float
    resolved_ry: float
    floor_ry: float
    floor_policy: str
    ansatz: str

    @property
    def raised(self) -> bool:
        return self.resolved_ry > self.requested_ry

    @property
    def requested_ev(self) -> float:
        from common.units import RYD_TO_EV
        return self.requested_ry * RYD_TO_EV

    @property
    def resolved_ev(self) -> float:
        from common.units import RYD_TO_EV
        return self.resolved_ry * RYD_TO_EV

    def describe(self) -> str:
        """Return the canonical one-line broadening receipt."""
        from common.units import RYD_TO_EV
        del RYD_TO_EV
        return (f"  Σ broadening ξ: {self.resolved_ev:.4f} eV "
                f"(requested {self.requested_ev:.4f} eV, ansatz "
                f"{self.ansatz}, {self.floor_policy})")


def resolve_sigma_regularization(*, requested_ry: float, ansatz: str) -> SigmaRegularization:
    """Resolve the Sigma broadening once for every dynamic ansatz.

    The deck's ``sigma_regularization_ev`` IS the broadening the kernel runs
    at, for every ansatz: the executor inserts it exactly once as
    ``exp(-eta t)`` on every time node.  Retired 2026-09-02 (owner ruling):
    the ``auto`` conditioning floor that raised GN/HL-PPM's xi to
    ``2 omega_max/(24 - 2 edge)`` -- 1.43 eV on a +-15 eV grid, 8.6 eV on
    CrI3's -- belonged to the deleted HGL sine-table executor (its tables
    stopped at A = 24) and made the broadening a function of the omega grid.
    The receipt type and the h5 stamp are kept so a run's xi stays
    auditable.
    """
    requested = float(requested_ry)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("sigma_regularization_ev must be finite and positive")
    return SigmaRegularization(
        requested_ry=requested, resolved_ry=requested, floor_ry=0.0,
        floor_policy="literal",
        ansatz=str(getattr(ansatz, "value", ansatz)).strip().lower())


def sigma_regularization_for_config(config) -> SigmaRegularization:
    """Resolve broadening from a ``LorraxConfig`` without importing it."""
    from common.units import RYD_TO_EV

    return resolve_sigma_regularization(
        requested_ry=float(config.sigma.regularization_ev) / RYD_TO_EV,
        ansatz=config.compute_mode)


@dataclass(frozen=True)
class _SigmaWindow:
    """One window consumed by the shared MPA dynamic-Sigma executor."""

    name: str
    nodes: MinimaxNodes
    mask_A: np.ndarray
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str
    prefactor: float
    mask_B_mode: str = "all"
    mask_B_threshold: float | None = None
    crossing_kind: str | None = None
    max_error: float | None = None
    provenance: str | None = None
    E_min: float | None = None
    E_max: float | None = None
    B_lo: float | None = None
    B_hi: float | None = None
    omega_indices: np.ndarray | None = None

    @property
    def n_tau(self) -> int:
        return int(self.nodes.t.shape[0])

    @property
    def project_code(self) -> int:
        """Return the projection code consumed by the shared accumulator."""
        if self.project == "full":
            return 0
        if self.project == "imag":
            return 1
        raise ValueError(
            f"Unknown window projection {self.project!r}; "
            "expected 'full' or 'imag'.")


class _SigmaBranch(NamedTuple):
    """One causal branch of the dynamic ``Sigma_c(omega)`` sum.

    Arrays ``E_A`` and ``base_mask_A`` have shape ``(nk, nb)``.  The optional
    ``band_weight`` has the same shape and carries fractional-occupation
    weights when available.
    """

    tag: str
    E_A: jax.Array
    base_mask_A: jax.Array
    space: str
    neg_omega_half: bool
    omega_abs: np.ndarray
    omega_idx: np.ndarray
    band_weight: jax.Array | None = None


def _omega_clusters(
    omega_abs,
    gap_ry: float,
    *,
    max_span_ry: float | None = None,
):
    """Split ``|omega|`` at real gaps, optionally capping cluster span."""
    gap = float(gap_ry)
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError("omega cluster gap must be finite and positive")
    span = None if max_span_ry is None else float(max_span_ry)
    if span is not None and (not np.isfinite(span) or span <= 0.0):
        raise ValueError("omega cluster maximum span must be finite and positive")

    omega = np.asarray(omega_abs, dtype=np.float64)
    if omega.ndim != 1:
        raise ValueError("omega_abs must be one-dimensional")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega_abs contains a non-finite value")
    if not omega.size:
        return []

    order = np.argsort(omega, kind="stable")
    breaks = np.nonzero(np.diff(omega[order]) > gap)[0]
    gap_pieces = np.split(order, breaks + 1)
    pieces = []
    for piece in gap_pieces:
        if span is None:
            pieces.append(piece)
            continue
        start = 0
        while start < piece.size:
            omega_start = float(omega[piece[start]])
            stop = start + 1
            while (stop < piece.size
                   and float(omega[piece[stop]]) - omega_start <= span):
                stop += 1
            pieces.append(piece[start:stop])
            start = stop
    return [
        (np.sort(piece),
         float(np.min(omega[piece])),
         float(np.max(omega[piece])))
        for piece in pieces
    ]


def _iter_branches(
    *,
    omega_pos: np.ndarray,
    idx_pos: np.ndarray,
    omega_neg_abs: np.ndarray,
    idx_neg: np.ndarray,
    E_cond: jax.Array,
    H_val: jax.Array,
    cond_mask: jax.Array,
    val_mask: jax.Array,
    cond_weight: jax.Array | None = None,
    val_weight: jax.Array | None = None,
    weight_floor: float = 0.0,
) -> list[_SigmaBranch]:
    """Enumerate the four causal branches, skipping empty omega halves."""

    def _narrow(mask, weight):
        if weight is None:
            return mask
        return mask & band_in_occupation_window(weight, weight_floor)

    cond_mask = _narrow(cond_mask, cond_weight)
    val_mask = _narrow(val_mask, val_weight)
    branches: list[_SigmaBranch] = []
    if omega_pos.size:
        branches += [
            _SigmaBranch(
                tag="ω≥E_F cond", E_A=E_cond,
                base_mask_A=cond_mask, space="cond",
                neg_omega_half=False, omega_abs=omega_pos,
                omega_idx=idx_pos, band_weight=cond_weight),
            _SigmaBranch(
                tag="ω≥E_F val", E_A=H_val,
                base_mask_A=val_mask, space="val",
                neg_omega_half=False, omega_abs=omega_pos,
                omega_idx=idx_pos, band_weight=val_weight),
        ]
    if omega_neg_abs.size:
        branches += [
            _SigmaBranch(
                tag="ω<E_F cond", E_A=E_cond,
                base_mask_A=cond_mask, space="cond",
                neg_omega_half=True, omega_abs=omega_neg_abs,
                omega_idx=idx_neg, band_weight=cond_weight),
            _SigmaBranch(
                tag="ω<E_F val", E_A=H_val,
                base_mask_A=val_mask, space="val",
                neg_omega_half=True, omega_abs=omega_neg_abs,
                omega_idx=idx_neg, band_weight=val_weight),
        ]
    return branches


def branches_for_omega_grid(
    omega_grid_ry,
    *,
    E_cond: jax.Array,
    H_val: jax.Array,
    cond_mask: jax.Array,
    val_mask: jax.Array,
    cond_weight: jax.Array | None = None,
    val_weight: jax.Array | None = None,
    occupation_window_threshold: float = OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
) -> list[_SigmaBranch]:
    """Split a signed omega grid and enumerate its causal branches.

    Energy and mask construction remains with the pole-model caller.  A
    supplied fractional occupation weight is narrowed at the shared
    occupation threshold; the insulating boolean-mask path is unchanged.
    """
    omega = np.asarray(omega_grid_ry, np.float64)
    idx_pos = np.where(omega >= 0.0)[0]
    idx_neg = np.where(omega < 0.0)[0]
    return _iter_branches(
        omega_pos=omega[idx_pos],
        idx_pos=idx_pos,
        omega_neg_abs=-omega[idx_neg],
        idx_neg=idx_neg,
        E_cond=E_cond,
        H_val=H_val,
        cond_mask=cond_mask,
        val_mask=val_mask,
        cond_weight=cond_weight,
        val_weight=val_weight,
        weight_floor=occupation_weight_floor(occupation_window_threshold),
    )


def window_mask_B_bounds(window: _SigmaWindow) -> tuple[float, float]:
    """Return one shared window's pole selector as ``(lo, hi]`` bounds."""
    b_lo = getattr(window, "B_lo", None)
    b_hi = getattr(window, "B_hi", None)
    if b_lo is not None or b_hi is not None:
        if b_lo is None or b_hi is None:
            raise ValueError("A scalar B pane requires both B_lo and B_hi")
        lo, hi = float(b_lo), float(b_hi)
        if not lo < hi:
            raise ValueError(f"Invalid B pane ({lo!r}, {hi!r}]")
        return lo, hi

    mode = str(window.mask_B_mode)
    if mode == "all":
        return (-np.inf, np.inf)
    if window.mask_B_threshold is None:
        raise ValueError(f"mask_B_mode={mode!r} requires a mask_B_threshold")
    threshold = float(window.mask_B_threshold)
    if mode == "le_t":
        return (-np.inf, threshold)
    if mode == "gt_t":
        return (threshold, np.inf)
    raise ValueError(f"Unknown mask_B_mode={mode!r}")
