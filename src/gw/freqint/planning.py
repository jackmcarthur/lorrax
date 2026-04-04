from __future__ import annotations

import math

import numpy as np
from scipy.special import roots_laguerre

from ..hgl_quadrature import hgl_nodes_weights, n_tau_hgl
from .types import DenomType, IntegrationPlan, WindowExecutionPlan


def _window_bounds(win) -> tuple[float, float]:
    E_gap = float(win.cond_window.start_energy - win.val_window.end_energy)
    E_bw = float(win.cond_window.end_energy - win.val_window.start_energy)
    return E_gap, E_bw


def _gl_quadrature_for_interval(E_gap_eff: float, E_bw_eff: float, epsq: float) -> tuple[np.ndarray, np.ndarray, float]:
    tiny = 1e-12
    E_gap_eff = max(float(E_gap_eff), tiny)
    E_bw_eff = max(float(E_bw_eff), E_gap_eff + tiny)
    alpha = math.sqrt(E_bw_eff / E_gap_eff)
    n_tau = int(np.ceil(alpha * (0.4 - 0.3 * np.log(epsq))))
    n_tau = max(3, min(300, n_tau))
    tau_i, w_i = roots_laguerre(n_tau)
    z_lm = 1.0 / math.sqrt(E_gap_eff * E_bw_eff)
    return tau_i, w_i, z_lm


def _resonant_branch(omega_abs: float, E_gap: float, E_bw: float) -> str:
    if omega_abs < E_gap:
        return "resonant_below"
    if omega_abs <= E_bw:
        return "crossing"
    return "resonant_above"


def _append_gl(
    denoms: list[DenomType],
    *,
    name: str,
    sign: int,
    omega_indices: np.ndarray,
    omega_values: np.ndarray,
    E_gap_eff: float,
    E_bw_eff: float,
    epsq: float,
) -> None:
    tau_i, w_i, z_lm = _gl_quadrature_for_interval(E_gap_eff, E_bw_eff, epsq)
    denoms.append(
        DenomType(
            name=name,  # type: ignore[arg-type]
            quadrature_kind="GL",
            damping_kind="exponential",
            sign=sign,
            omega_indices=omega_indices,
            omega_values=omega_values,
            E_gap_eff=float(E_gap_eff),
            E_bw_eff=float(E_bw_eff),
            z_lm=float(z_lm),
            tau=tau_i,
            w=w_i,
        )
    )


def _build_chi_window_plan_per_omega(win, omega: np.ndarray, E_gap: float, E_bw: float) -> WindowExecutionPlan:
    denoms: list[DenomType] = []
    for idx, om in enumerate(omega):
        omega_abs = float(abs(np.real(om)))
        omega_idx = np.asarray([idx], dtype=np.int32)
        omega_val = np.asarray([omega_abs], dtype=np.float64)

        _append_gl(
            denoms,
            name="antiresonant",
            sign=-1,
            omega_indices=omega_idx,
            omega_values=omega_val,
            E_gap_eff=E_gap + omega_abs,
            E_bw_eff=E_bw + omega_abs,
            epsq=float(win.epsq),
        )

        branch = _resonant_branch(omega_abs, E_gap, E_bw)
        if branch == "resonant_below":
            _append_gl(
                denoms,
                name="resonant_below",
                sign=-1,
                omega_indices=omega_idx,
                omega_values=omega_val,
                E_gap_eff=E_gap - omega_abs,
                E_bw_eff=E_bw - omega_abs,
                epsq=float(win.epsq),
            )
            continue

        if branch == "resonant_above":
            _append_gl(
                denoms,
                name="resonant_above",
                sign=1,
                omega_indices=omega_idx,
                omega_values=omega_val,
                E_gap_eff=omega_abs - E_bw,
                E_bw_eff=omega_abs - E_gap,
                epsq=float(win.epsq),
            )
            continue

        n_hgl = int(n_tau_hgl(float(win.z_lm), E_bw, float(win.epsq)))
        tau_hgl, w_hgl = hgl_nodes_weights(n_hgl)
        denoms.append(
            DenomType(
                name="crossing",
                quadrature_kind="HGL",
                damping_kind="oscillatory",
                sign=1,
                omega_indices=omega_idx,
                omega_values=omega_val,
                E_gap_eff=E_gap,
                E_bw_eff=E_bw,
                z_lm=float(win.z_lm),
                tau=tau_hgl,
                w=w_hgl,
            )
        )

    return WindowExecutionPlan(
        window_index=-1,
        window_pair=win,
        denom_types=tuple(denoms),
        E_gap=E_gap,
        E_bw=E_bw,
    )


def _build_chi_window_plan_shared_conservative(
    win, omega: np.ndarray, E_gap: float, E_bw: float
) -> WindowExecutionPlan:
    """Group omegas by denominator type, one shared quadrature per group."""
    denoms: list[DenomType] = []
    omega_abs = np.abs(np.real(omega))
    all_idx = np.arange(omega.shape[0], dtype=np.int32)

    # Antiresonant always exists for all omega >= 0 in the reduced formulation.
    anti_gap = float(np.min(E_gap + omega_abs))
    anti_bw = float(np.max(E_bw + omega_abs))
    _append_gl(
        denoms,
        name="antiresonant",
        sign=-1,
        omega_indices=all_idx,
        omega_values=omega_abs.astype(np.float64),
        E_gap_eff=anti_gap,
        E_bw_eff=anti_bw,
        epsq=float(win.epsq),
    )

    below_mask = omega_abs < E_gap
    if np.any(below_mask):
        below_idx = np.where(below_mask)[0].astype(np.int32)
        below_omega = omega_abs[below_mask].astype(np.float64)
        _append_gl(
            denoms,
            name="resonant_below",
            sign=-1,
            omega_indices=below_idx,
            omega_values=below_omega,
            E_gap_eff=float(np.min(E_gap - below_omega)),
            E_bw_eff=float(np.max(E_bw - below_omega)),
            epsq=float(win.epsq),
        )

    above_mask = omega_abs > E_bw
    if np.any(above_mask):
        above_idx = np.where(above_mask)[0].astype(np.int32)
        above_omega = omega_abs[above_mask].astype(np.float64)
        _append_gl(
            denoms,
            name="resonant_above",
            sign=1,
            omega_indices=above_idx,
            omega_values=above_omega,
            E_gap_eff=float(np.min(above_omega - E_bw)),
            E_bw_eff=float(np.max(above_omega - E_gap)),
            epsq=float(win.epsq),
        )

    cross_mask = (~below_mask) & (~above_mask)
    if np.any(cross_mask):
        cross_idx = np.where(cross_mask)[0].astype(np.int32)
        cross_omega = omega_abs[cross_mask].astype(np.float64)
        n_hgl = int(n_tau_hgl(float(win.z_lm), E_bw, float(win.epsq)))
        tau_hgl, w_hgl = hgl_nodes_weights(n_hgl)
        denoms.append(
            DenomType(
                name="crossing",
                quadrature_kind="HGL",
                damping_kind="oscillatory",
                sign=1,
                omega_indices=cross_idx,
                omega_values=cross_omega,
                E_gap_eff=E_gap,
                E_bw_eff=E_bw,
                z_lm=float(win.z_lm),
                tau=tau_hgl,
                w=w_hgl,
            )
        )

    return WindowExecutionPlan(
        window_index=-1,
        window_pair=win,
        denom_types=tuple(denoms),
        E_gap=E_gap,
        E_bw=E_bw,
    )


def build_chi_window_plan(
    win,
    omega: np.ndarray,
    *,
    policy: str = "per_omega",
) -> WindowExecutionPlan:
    """Build denominator plan for one chi window pair."""
    E_gap, E_bw = _window_bounds(win)
    if policy == "per_omega":
        return _build_chi_window_plan_per_omega(win, omega, E_gap, E_bw)
    if policy == "shared_conservative":
        return _build_chi_window_plan_shared_conservative(win, omega, E_gap, E_bw)
    raise ValueError(
        f"Unknown policy={policy!r}. Supported policies: 'per_omega', 'shared_conservative'."
    )


def build_chi_integration_plan(
    windows,
    omega,
    *,
    policy: str = "per_omega",
) -> IntegrationPlan:
    """Build full chi integration plan for all windows."""
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    window_plans: list[WindowExecutionPlan] = []
    for iwin, win in enumerate(windows):
        wplan = build_chi_window_plan(win, omega_arr, policy=policy)
        window_plans.append(
            WindowExecutionPlan(
                window_index=iwin,
                window_pair=wplan.window_pair,
                denom_types=wplan.denom_types,
                E_gap=wplan.E_gap,
                E_bw=wplan.E_bw,
            )
        )
    return IntegrationPlan(
        omega=omega_arr,
        policy=policy,
        window_plans=tuple(window_plans),
    )
