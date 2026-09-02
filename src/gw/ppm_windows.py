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


# The historical HGL pane control becomes poorly conditioned above this
# dimensionless bandwidth.  The resolver retains that published broadening
# policy; window construction itself now belongs solely to the MPA route.
_CROSSING_A_MAX = 24.0


def crossing_regularization_floor(
    omega_max_ry: float,
    edge_factor: float,
) -> float:
    """Return the historical HGL conditioning floor for ``xi`` in Ry."""
    denom = _CROSSING_A_MAX - 2.0 * float(edge_factor)
    if denom <= 1.0 or float(omega_max_ry) <= 0.0:
        return 0.0
    return 2.0 * float(omega_max_ry) / denom


_HGL_CROSSING_ANSATZE = frozenset({"gn_ppm", "hl_ppm"})


def hgl_partition_required(
    omega_grid_ry,
    regularization_width_ry: float,
    edge_factor: float,
) -> bool:
    """Return whether a grid exceeds the historical HGL capacity."""
    omega = np.asarray(omega_grid_ry, dtype=np.float64)
    omega_max = float(np.max(np.abs(omega))) if omega.size else 0.0
    xi = float(regularization_width_ry)
    edge = float(edge_factor)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("regularization_width_ry must be finite and positive")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("edge_factor must be finite and non-negative")
    if not np.isfinite(omega_max):
        raise ValueError("omega_grid_ry contains a non-finite value")
    A_core = 2.0 * (omega_max + edge * xi) / xi
    capacity = _CROSSING_A_MAX * (
        1.0 + 8.0 * np.finfo(np.float64).eps)
    return A_core > capacity


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
        head = (f"  Σ broadening ξ: {self.resolved_ev:.4f} eV "
                f"(requested {self.requested_ev:.4f} eV, ansatz "
                f"{self.ansatz}, floor {self.floor_ry * RYD_TO_EV:.4f} eV "
                f"[{self.floor_policy}])")
        if self.raised:
            head += " — RAISED to the floor"
        return head


def resolve_sigma_regularization(
    *,
    requested_ry: float,
    omega_grid_ry,
    edge_factor: float,
    ansatz: str,
    floor_ev=None,
) -> SigmaRegularization:
    """Resolve the effective broadening once for every dynamic ansatz.

    An explicit non-negative ``floor_ev`` applies to every ansatz.  ``auto``
    retains the historical HGL conditioning floor for GN/HL-PPM and is zero
    for MPA.  The causal executor receives only the resolved value.
    """
    from common.units import RYD_TO_EV

    requested = float(requested_ry)
    omega = np.asarray(omega_grid_ry, dtype=np.float64)
    omega_max_ry = float(np.max(np.abs(omega))) if omega.size else 0.0
    name = str(getattr(ansatz, "value", ansatz)).strip().lower()

    explicit = not (
        floor_ev is None or str(floor_ev).strip().lower() == "auto")
    if explicit:
        floor_ry = float(floor_ev) / RYD_TO_EV
        if floor_ry < 0.0:
            raise ValueError(
                "sigma_regularization_floor_ev must be >= 0 or 'auto'; "
                f"got {floor_ev!r}.")
        policy = "explicit"
    elif name in _HGL_CROSSING_ANSATZE:
        floor_ry = crossing_regularization_floor(omega_max_ry, edge_factor)
        policy = "auto"
    else:
        floor_ry = 0.0
        policy = "auto"

    return SigmaRegularization(
        requested_ry=requested,
        resolved_ry=max(requested, floor_ry),
        floor_ry=floor_ry,
        floor_policy=policy,
        ansatz=name,
    )


def sigma_regularization_for_config(config) -> SigmaRegularization:
    """Resolve broadening from a ``LorraxConfig`` without importing it."""
    from common.units import RYD_TO_EV

    sigma_cfg = config.sigma
    return resolve_sigma_regularization(
        requested_ry=float(sigma_cfg.regularization_ev) / RYD_TO_EV,
        omega_grid_ry=np.asarray(config.omega_grid_ry, dtype=np.float64),
        edge_factor=float(sigma_cfg.window_edge_factor),
        ansatz=config.compute_mode,
        floor_ev=getattr(sigma_cfg, "regularization_floor_ev", None),
    )


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
