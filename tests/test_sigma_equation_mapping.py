import numpy as np

from gw_isdf.ppm_sigma import _build_single_sigma_window


def _eval_single_window_scalar(
    windows,
    omega_nonneg_ry: np.ndarray,
    s_value: float,
    *,
    scale: float = 1.0,
    omega_sign_flip: int = 1,
) -> np.ndarray:
    """Scalar analog of the single-window sigma accumulation path.

    This mirrors the code path used in ``ppm_sigma._convolve_sigma_branch_kij``
    for sign-definite windows, including the global ``-1`` carried by
    ``get_sigma_static_mu_nu_jax``.
    """
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    out = np.zeros_like(omega_nonneg_ry, dtype=np.float64)
    for iw, omega in enumerate(omega_nonneg_ry):
        acc = 0.0
        for win in windows:
            z = 0.0 + 0.0j
            for t_node, alpha_node in zip(win.t_nodes, win.alpha):
                sigma_tau = -np.exp(-1j * (s_value - (win.E_ref_A + win.E_ref_B)) * t_node)
                coeff = (
                    alpha_node
                    * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
                    * np.exp(1j * float(win.omega_sign) * float(omega_sign_flip) * omega * t_node)
                )
                z += coeff * sigma_tau
            projected = np.imag(z) if win.project == "imag" else np.real(z)
            acc += float(scale) * float(win.prefactor) * float(projected)
        out[iw] = acc
    return out


def test_single_window_kernel_sign_maps_to_expected_denominators():
    """Equation-level check for sign-definite kernels.

    Verifies:
      kernel_sign=+1 -> 1/(omega - S)
      kernel_sign=-1 -> 1/(omega + S)
    in the non-crossing regime.
    """
    e_a = np.array([0.8, 1.2, 1.5], dtype=np.float64)
    e_b = np.array([0.4, 0.7, 1.1], dtype=np.float64)
    mask_a = np.ones_like(e_a, dtype=bool)
    mask_b = np.ones_like(e_b, dtype=bool)
    omega = np.linspace(0.0, 0.2, 9, dtype=np.float64)
    s_value = 1.65

    wins_minus = _build_single_sigma_window(
        E_A=e_a,
        E_B=e_b,
        base_mask_A=mask_a,
        base_mask_B=mask_b,
        omega_nonneg_ry=omega,
        kernel_sign=+1,
        target_error=1.0e-10,
        max_nodes=120,
    )
    wins_plus = _build_single_sigma_window(
        E_A=e_a,
        E_B=e_b,
        base_mask_A=mask_a,
        base_mask_B=mask_b,
        omega_nonneg_ry=omega,
        kernel_sign=-1,
        target_error=1.0e-10,
        max_nodes=120,
    )

    approx_minus = _eval_single_window_scalar(wins_minus, omega, s_value)
    approx_plus = _eval_single_window_scalar(wins_plus, omega, s_value)
    exact_minus = 1.0 / (omega - s_value)
    exact_plus = 1.0 / (omega + s_value)

    rel_minus = np.max(np.abs(approx_minus - exact_minus)) / np.max(np.abs(exact_minus))
    rel_plus = np.max(np.abs(approx_plus - exact_plus)) / np.max(np.abs(exact_plus))

    assert rel_minus < 1.0e-8
    assert rel_plus < 1.0e-8


def test_negative_frequency_routing_matches_equation_level_denominators():
    """Equation-level check for omega<E_F routing used in ``compute_sigma_c_ppm_omega_grid``.

    Current routing for omega_rel<0 uses:
      cond branch: kernel=-1, scale=-1  -> 1/(omega_rel - S)
      val  branch: kernel=+1, scale=-1  -> 1/(omega_rel + S)
    """
    e_a = np.array([0.8, 1.2, 1.5], dtype=np.float64)
    e_b = np.array([0.4, 0.7, 1.1], dtype=np.float64)
    mask_a = np.ones_like(e_a, dtype=bool)
    mask_b = np.ones_like(e_b, dtype=bool)
    omega_abs = np.array([0.02, 0.06, 0.10, 0.14, 0.18], dtype=np.float64)
    omega_rel_neg = -omega_abs
    s_value = 1.65

    wins_plus = _build_single_sigma_window(
        E_A=e_a,
        E_B=e_b,
        base_mask_A=mask_a,
        base_mask_B=mask_b,
        omega_nonneg_ry=omega_abs,
        kernel_sign=-1,
        target_error=1.0e-10,
        max_nodes=120,
    )
    wins_minus = _build_single_sigma_window(
        E_A=e_a,
        E_B=e_b,
        base_mask_A=mask_a,
        base_mask_B=mask_b,
        omega_nonneg_ry=omega_abs,
        kernel_sign=+1,
        target_error=1.0e-10,
        max_nodes=120,
    )

    approx_cond_neg = _eval_single_window_scalar(wins_plus, omega_abs, s_value, scale=-1.0)
    approx_val_neg = _eval_single_window_scalar(wins_minus, omega_abs, s_value, scale=-1.0)
    exact_cond_neg = 1.0 / (omega_rel_neg - s_value)
    exact_val_neg = 1.0 / (omega_rel_neg + s_value)

    rel_cond = np.max(np.abs(approx_cond_neg - exact_cond_neg)) / np.max(np.abs(exact_cond_neg))
    rel_val = np.max(np.abs(approx_val_neg - exact_val_neg)) / np.max(np.abs(exact_val_neg))

    assert rel_cond < 1.0e-8
    assert rel_val < 1.0e-8
