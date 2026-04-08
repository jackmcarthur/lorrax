from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..archive import integrands
from ..archive import w_isdf_dynamic as legacy_dynamic
from .planning import build_chi_integration_plan
from .types import IntegrationPlan


def _normalize_omega_input(omega) -> tuple[np.ndarray, bool]:
    if np.isscalar(omega):
        return np.asarray([float(omega)], dtype=np.float64), True
    arr = np.asarray(omega, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"omega must be scalar or 1D, got shape {arr.shape}")
    return arr, False


def _ensure_window_band_ranges_for_bundle(wf_bundle, windows) -> None:
    s = wf_bundle.slices
    enk_v_host = np.asarray(jax.device_get(wf_bundle.enk[:, s.v_slice]))
    enk_c_host = np.asarray(jax.device_get(wf_bundle.enk[:, s.c_slice]))
    for win in windows:
        if getattr(win, "_has_band_ranges", False):
            continue
        legacy_dynamic._ensure_window_band_ranges(win, enk_v_host, enk_c_host)


def _compute_window_denom(
    *,
    wf_bundle,
    win,
    denom,
    nkx: int,
    nky: int,
    nkz: int,
):
    s = wf_bundle.slices
    psi_vTX = wf_bundle.x(s.v_slice)
    psi_vY = wf_bundle.y(s.v_slice)
    psi_cX = wf_bundle.y(s.c_slice)
    psi_cTY = wf_bundle.x(s.c_slice)
    enk_v = wf_bundle.enk[:, s.v_slice]
    enk_c = wf_bundle.enk[:, s.c_slice]

    val_start = jnp.asarray(win.val_band_start, dtype=jnp.int32)
    cond_start = jnp.asarray(win.cond_band_start, dtype=jnp.int32)
    val_mask = jnp.asarray(win.val_band_mask, dtype=jnp.bool_)
    cond_mask = jnp.asarray(win.cond_band_mask, dtype=jnp.bool_)
    tau = jnp.asarray(denom.tau, dtype=jnp.float64)
    w = jnp.asarray(denom.w, dtype=jnp.float64)
    omega_values = jnp.asarray(denom.omega_values, dtype=jnp.float64)

    kwargs = dict(
        psi_vTX=psi_vTX,
        psi_vY=psi_vY,
        psi_cX=psi_cX,
        psi_cTY=psi_cTY,
        enk_v=enk_v,
        enk_c=enk_c,
        val_start=val_start,
        cond_start=cond_start,
        val_mask=val_mask,
        cond_mask=cond_mask,
        max_val_len=int(win.max_val_len),
        max_cond_len=int(win.max_cond_len),
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        vmin=float(win.val_window.start_energy),
        vmax=float(win.val_window.end_energy),
        cmin=float(win.cond_window.start_energy),
        cmax=float(win.cond_window.end_energy),
        tau_i=tau,
        w_i=w,
        gamma=float(denom.z_lm),
        omega_values=omega_values,
    )

    if denom.quadrature_kind == "HGL":
        return integrands.compute_hgl_batch(
            psi_vTX=psi_vTX,
            psi_vY=psi_vY,
            psi_cX=psi_cX,
            psi_cTY=psi_cTY,
            enk_v=enk_v,
            enk_c=enk_c,
            val_start=val_start,
            cond_start=cond_start,
            val_mask=val_mask,
            cond_mask=cond_mask,
            max_val_len=int(win.max_val_len),
            max_cond_len=int(win.max_cond_len),
            nkx=nkx,
            nky=nky,
            nkz=nkz,
            tau_hgl=tau,
            w_hgl=w,
            gamma=float(denom.z_lm),
            omega_values=omega_values,
        )

    if denom.name in ("antiresonant", "resonant_below"):
        return integrands.compute_gl_standard_batch(
            denom_name=denom.name,
            sign=int(denom.sign),
            **kwargs,
        )
    if denom.name == "resonant_above":
        return integrands.compute_gl_reverse_batch(**kwargs)
    raise ValueError(f"Unsupported denominator branch {denom.name!r}")


def plan_chi_from_bundle(
    wf_bundle,
    windows,
    omega,
    *,
    policy: str = "shared_conservative",
) -> IntegrationPlan:
    """Plan-only helper for chi integration."""
    omega_arr, _ = _normalize_omega_input(omega)
    _ = wf_bundle  # kept for signature parity with execute API
    return build_chi_integration_plan(windows, omega_arr, policy=policy)


def execute_chi_from_bundle(
    wf_bundle,
    windows,
    omega,
    meta,
    mesh_xy,
    *,
    policy: str = "shared_conservative",
    return_plan: bool = False,
):
    """Separate chi execution path with denom-plan dispatch and tau scans."""
    del mesh_xy
    omega_arr, scalar_input = _normalize_omega_input(omega)
    plan = build_chi_integration_plan(windows, omega_arr, policy=policy)
    _ensure_window_band_ranges_for_bundle(wf_bundle, windows)

    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nrmu = int(wf_bundle.psi_x.shape[2])
    chi_total = jnp.zeros((omega_arr.shape[0], nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)

    for wplan in plan.window_plans:
        win = wplan.window_pair
        if int(getattr(win, "max_val_len", 0)) == 0 or int(getattr(win, "max_cond_len", 0)) == 0:
            continue
        for denom in wplan.denom_types:
            if denom.omega_indices.size == 0:
                continue
            chi_bucket = _compute_window_denom(
                wf_bundle=wf_bundle,
                win=win,
                denom=denom,
                nkx=nkx,
                nky=nky,
                nkz=nkz,
            )
            idx = jnp.asarray(denom.omega_indices, dtype=jnp.int32)
            chi_total = chi_total.at[idx].add(chi_bucket)

    neg_mask = jnp.asarray(omega_arr < 0.0, dtype=jnp.bool_).reshape((-1, 1, 1, 1, 1, 1, 1, 1))
    chi_total = jnp.where(neg_mask, jnp.conj(chi_total), chi_total)
    chi_out = chi_total[0] if scalar_input else chi_total
    if return_plan:
        return chi_out, plan
    return chi_out
