"""Small driver-side helpers to keep ``gw_jax.py`` focused on orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

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
    sigma_munu_h5_path: str
    sigma_kij_h5_path: str
    write_w_copies_debug: bool
    w_copies_debug_file: str
    sigma_freq_debug_file: str
    use_ffi_io: bool = False


def _resolve_input_path(input_dir: str, path: str) -> str:
    if path and (not os.path.isabs(path)):
        return os.path.join(input_dir, path)
    return path


def build_ppm_sigma_runtime_options(config, *, input_dir: str, ryd2ev: float) -> PPMSigmaRuntimeOptions:
    """Build PPM sigma runtime options from LorraxConfig (or legacy params dict)."""

    def _g(key, default=None):
        """Get from config object (attribute) or dict (key)."""
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    omega_p_ry = float(_g("ppm_omega_p", 2.0))
    ppm_fallback = float(_g("ppm_fallback_omega", 2.0))
    omega_min_ev = float(_g("sigma_omega_min_ev", -5.0))
    omega_max_ev = float(_g("sigma_omega_max_ev", 5.0))
    omega_step_ev = float(_g("sigma_omega_step_ev", 0.25))
    sigma_regularization_ev = float(_g("sigma_regularization_ev", 0.25))
    sigma_edge_factor = float(_g("sigma_window_edge_factor", 1.5))
    sigma_omega_batch_size = int(max(1, _g("sigma_omega_batch_size", 4)))
    sigma_omega_accumulation = str(_g("sigma_omega_accumulation", "auto")).strip().lower()
    ppm_sigma_scale = float(_g("ppm_sigma_scale", 1.0))
    ppm_sigma_flip_neg = bool(_g("ppm_sigma_flip_neg", False))
    ppm_invalid_mode = str(_g("ppm_invalid_mode", "static_limit")).strip().lower()
    sigma_debug_split_contrib = bool(_g("sigma_debug_split_contrib", False))
    sigma_freq_debug_output = bool(_g("sigma_freq_debug_output", True))
    fermi_reference = str(_g("fermi_reference", "midgap")).strip().lower()
    sigma_at_dft_extrapolate = bool(_g("sigma_at_dft_extrapolate", False))
    sigma_at_dft_energies = bool(_g("sigma_at_dft_energies", False))
    ppm_sigma_debug_static_norm = bool(_g("ppm_sigma_debug_static_norm", False))
    ppm_static_cohsex_check = bool(_g("ppm_static_cohsex_check", False))
    sigma_debug_quadrature = bool(_g("sigma_debug_quadrature", False))
    sigma_debug_quadrature_samples = int(_g("sigma_debug_quadrature_samples", 200))
    sigma_munu_h5_path = str(_g("sigma_munu_h5_file", "") or "").strip()
    sigma_kij_h5_path = str(_g("sigma_kij_h5_file", "") or "").strip()
    write_w_copies_debug = bool(_g("write_w_copies_debug", False))
    w_copies_debug_file = str(_g("w_copies_debug_file", "w_copies_debug.h5") or "").strip()
    sigma_freq_debug_file = str(_g("sigma_freq_debug_file", "sigma_freq_debug.dat") or "").strip()

    if omega_step_ev <= 0.0:
        raise ValueError("sigma_omega_step_ev must be > 0.")
    if omega_max_ev < omega_min_ev:
        raise ValueError("sigma_omega_max_ev must be >= sigma_omega_min_ev.")
    if fermi_reference not in ("vbm", "midgap"):
        raise ValueError("fermi_reference must be 'vbm' or 'midgap'.")

    n_omega = int(np.floor((omega_max_ev - omega_min_ev) / omega_step_ev + 0.5)) + 1
    omega_grid_ev = omega_min_ev + omega_step_ev * np.arange(n_omega, dtype=np.float64)
    omega_grid_ry = omega_grid_ev / ryd2ev
    sigma_regularization_ry = sigma_regularization_ev / ryd2ev

    return PPMSigmaRuntimeOptions(
        omega_p_ry=omega_p_ry,
        ppm_fallback=ppm_fallback,
        omega_grid_ev=omega_grid_ev,
        omega_grid_ry=omega_grid_ry,
        sigma_regularization_ry=sigma_regularization_ry,
        sigma_edge_factor=sigma_edge_factor,
        sigma_omega_batch_size=sigma_omega_batch_size,
        sigma_omega_accumulation=sigma_omega_accumulation,
        ppm_sigma_scale=ppm_sigma_scale,
        ppm_sigma_flip_neg=ppm_sigma_flip_neg,
        ppm_invalid_mode=ppm_invalid_mode,
        sigma_debug_split_contrib=sigma_debug_split_contrib,
        sigma_freq_debug_output=sigma_freq_debug_output,
        fermi_reference=fermi_reference,
        sigma_at_dft_extrapolate=sigma_at_dft_extrapolate,
        sigma_at_dft_energies=sigma_at_dft_energies,
        ppm_sigma_debug_static_norm=ppm_sigma_debug_static_norm,
        ppm_static_cohsex_check=ppm_static_cohsex_check,
        sigma_debug_quadrature=sigma_debug_quadrature,
        sigma_debug_quadrature_samples=sigma_debug_quadrature_samples,
        sigma_munu_h5_path=_resolve_input_path(input_dir, sigma_munu_h5_path),
        sigma_kij_h5_path=_resolve_input_path(input_dir, sigma_kij_h5_path),
        write_w_copies_debug=write_w_copies_debug,
        w_copies_debug_file=_resolve_input_path(input_dir, w_copies_debug_file),
        sigma_freq_debug_file=_resolve_input_path(input_dir, sigma_freq_debug_file),
        use_ffi_io=bool(_g("use_ffi_io", False)),
    )
