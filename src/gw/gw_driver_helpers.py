"""Small driver-side helpers to keep ``gw_jax.py`` focused on orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from common.units import RYD_TO_EV
from .gw_config import LorraxConfig


@dataclass(frozen=True)
class PPMSigmaRuntimeOptions:
    """Resolved PPM sigma options parsed once in the GW driver."""

    omega_p_ry: float
    ppm_fallback: float
    omega_grid_ev: np.ndarray
    omega_grid_ry: np.ndarray
    sigma_regularization_ry: float
    sigma_edge_factor: float
    sigma_omega_batch_size: int
    sigma_omega_accumulation: str
    ppm_sigma_scale: float
    ppm_sigma_flip_neg: bool
    ppm_invalid_mode: str
    sigma_debug_split_contrib: bool
    sigma_freq_debug_output: bool
    fermi_reference: str
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool
    ppm_sigma_debug_static_norm: bool
    ppm_static_cohsex_check: bool
    sigma_debug_quadrature: bool
    sigma_debug_quadrature_samples: int
    sigma_kij_h5_path: str
    write_w_copies_debug: bool
    w_copies_debug_file: str
    sigma_freq_debug_file: str
    use_ffi_io: bool = False


def _resolve_input_path(input_dir: str, path: str) -> str:
    if path and (not os.path.isabs(path)):
        return os.path.join(input_dir, path)
    return path


def build_ppm_sigma_runtime_options(
    config: LorraxConfig, *, input_dir: str
) -> PPMSigmaRuntimeOptions:
    """Build PPM sigma runtime options from a ``LorraxConfig``."""

    if config.sigma_omega_step_ev <= 0.0:
        raise ValueError("sigma_omega_step_ev must be > 0.")
    if config.sigma_omega_max_ev < config.sigma_omega_min_ev:
        raise ValueError("sigma_omega_max_ev must be >= sigma_omega_min_ev.")
    if config.fermi_reference not in ("vbm", "midgap"):
        raise ValueError("fermi_reference must be 'vbm' or 'midgap'.")

    n_omega = int(np.floor(
        (config.sigma_omega_max_ev - config.sigma_omega_min_ev)
        / config.sigma_omega_step_ev + 0.5
    )) + 1
    omega_grid_ev = (
        config.sigma_omega_min_ev
        + config.sigma_omega_step_ev * np.arange(n_omega, dtype=np.float64)
    )
    omega_grid_ry = omega_grid_ev / RYD_TO_EV
    sigma_regularization_ry = config.sigma_regularization_ev / RYD_TO_EV

    return PPMSigmaRuntimeOptions(
        omega_p_ry=float(config.ppm_omega_p),
        ppm_fallback=float(config.ppm_fallback_omega),
        omega_grid_ev=omega_grid_ev,
        omega_grid_ry=omega_grid_ry,
        sigma_regularization_ry=sigma_regularization_ry,
        sigma_edge_factor=float(config.sigma_window_edge_factor),
        sigma_omega_batch_size=int(max(1, config.sigma_omega_batch_size)),
        sigma_omega_accumulation=str(config.sigma_omega_accumulation).strip().lower(),
        ppm_sigma_scale=float(config.ppm_sigma_scale),
        ppm_sigma_flip_neg=bool(config.ppm_sigma_flip_neg),
        ppm_invalid_mode=str(config.ppm_invalid_mode).strip().lower(),
        sigma_debug_split_contrib=bool(config.sigma_debug_split_contrib),
        sigma_freq_debug_output=bool(config.sigma_freq_debug_output),
        fermi_reference=str(config.fermi_reference).strip().lower(),
        sigma_at_dft_extrapolate=bool(config.sigma_at_dft_extrapolate),
        sigma_at_dft_energies=bool(config.sigma_at_dft_energies),
        ppm_sigma_debug_static_norm=bool(config.ppm_sigma_debug_static_norm),
        ppm_static_cohsex_check=bool(config.ppm_static_cohsex_check),
        sigma_debug_quadrature=bool(config.sigma_debug_quadrature),
        sigma_debug_quadrature_samples=int(config.sigma_debug_quadrature_samples),
        sigma_kij_h5_path=_resolve_input_path(input_dir, str(config.sigma_kij_h5_file or "").strip()),
        write_w_copies_debug=bool(config.write_w_copies_debug),
        w_copies_debug_file=_resolve_input_path(input_dir, str(config.w_copies_debug_file or "").strip()),
        sigma_freq_debug_file=_resolve_input_path(input_dir, str(config.sigma_freq_debug_file or "").strip()),
        use_ffi_io=bool(config.use_ffi_io),
    )
