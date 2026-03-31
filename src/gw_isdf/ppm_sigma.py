"""GN-PPM construction from W(0), W(i*omega_p) and sigma_c frequency integration.

This module is designed to reuse the existing GW helpers rather than duplicate
FFT/projection logic:
  - chi/W evaluation reuses ``w_isdf.compute_chi0`` and ``w_isdf.solve_w_from_chi_q_jax``.
  - sigma projection/convolution reuses callables supplied by ``gw_jax``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import h5py

from .minimax_screening import (
    build_static_minimax_window_pair,
    extract_gn_ppm_parameters_from_Wc,
    _load_docs_minimax_module,
    solve_laplace_minimax_interval,
    solve_laplace_minimax_imag_interval,
    solve_phase_minimax_bandwidth,
)
from . import w_isdf


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    W0_q: jax.Array
    Wiwp_q: jax.Array
    Wc0_mu_nu: jax.Array
    B_mu_nu: jax.Array
    Omega_mu_nu: jax.Array
    valid_mask_mu_nu: jax.Array
    unfulfilled_fraction: float
    n_nodes_static: int
    w0_rel_error: float | None = None
    w0_abs_error: float | None = None
    w0_ref_norm: float | None = None


@dataclass(frozen=True)
class SigmaOmegaResult:
    omega_ry: np.ndarray
    omega_ev: np.ndarray
    sigma_c_kij: jax.Array | None
    sigma_c_plus_kij: jax.Array | None = None
    sigma_c_minus_kij: jax.Array | None = None
    sigma_c_invalid_static_kij: jax.Array | None = None
    sigma_munu_h5_path: str | None = None
    sigma_kij_h5_path: str | None = None


@dataclass(frozen=True)
class _SigmaWindow:
    name: str
    t_nodes: np.ndarray
    alpha: np.ndarray
    mask_A: np.ndarray
    mask_B: np.ndarray
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str
    prefactor: float
    x_min: float | None = None
    x_max: float | None = None
    crossing_A: float | None = None
    crossing_kind: str | None = None
    t_cut: float | None = None
    z_edge: float | None = None


def _mu_nu_sharding(mesh_xy: Mesh) -> NamedSharding:
    # (mu, nu, kx, ky, kz) with mu/nu split across x/y mesh axes.
    return NamedSharding(mesh_xy, P("x", "y", None, None, None))


def _extract_mu_nu_q_layout(W_q: jax.Array) -> jax.Array:
    # (nkx, nky, nkz, 1, mu, 1, nu) -> (mu, nu, nkx, nky, nkz)
    return W_q[:, :, :, 0, :, 0, :].transpose(3, 4, 0, 1, 2)


def _summarize_hermitian_qmunu(name: str, W_q: jax.Array, print_fn=print) -> None:
    """Emit Hermiticity diagnostics for W_q with layout (kx,ky,kz,1,mu,1,nu)."""
    try:
        W_host = np.asarray(jax.device_get(W_q), dtype=np.complex128)
    except Exception:
        return
    if W_host.ndim != 7:
        print_fn(f"[{name}] unexpected shape {W_host.shape}; expected (nkx,nky,nkz,1,mu,1,nu)")
        return
    W_flat = W_host[:, :, :, 0, :, 0, :].reshape(-1, W_host.shape[4], W_host.shape[6])
    herm_resid = float(np.max(np.abs(W_flat - np.conj(np.swapaxes(W_flat, -2, -1)))))
    diag = np.diagonal(W_flat, axis1=-2, axis2=-1)
    diag_im = float(np.max(np.abs(np.imag(diag))))
    diag_min = float(np.min(np.real(diag)))
    diag_max = float(np.max(np.real(diag)))
    print_fn(
        f"  [{name}] hermitian residual={herm_resid:.3e} "
        f"max|Im diag|={diag_im:.3e} diag range=[{diag_min:.4e}, {diag_max:.4e}]"
    )


def compute_w0_wiwp_and_ppm_from_minimax(
    V_qmunu: jax.Array,
    wf_bundle,
    meta,
    mesh_xy: Mesh,
    *,
    omega_p_ry: float = 2.0,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    minimax_energy_reference: str | float | int | None = "midgap",
    minimax_energy_reference_fn: Callable[[jax.Array, jax.Array], float] | None = None,
    fallback_omega: float = 2.0,
    head_correction_fn: Callable[[jax.Array, jax.Array, complex], tuple[jax.Array, jax.Array]] | None = None,
    print0=print,
) -> PPMBuildResult:
    """Build GN-PPM parameters from minimax W(0) and W(i*omega_p)."""

    omega_p_ry = float(omega_p_ry)
    if omega_p_ry <= 0.0:
        raise ValueError("omega_p_ry must be > 0.")

    s = wf_bundle.slices
    psi_vTX = wf_bundle.x(s.v_slice)
    psi_vY = wf_bundle.y(s.v_slice)
    psi_cX = wf_bundle.y(s.c_slice)
    psi_cTY = wf_bundle.x(s.c_slice)
    enk_v = wf_bundle.enk[:, s.v_slice]
    enk_c = wf_bundle.enk[:, s.c_slice]
    e_ref = w_isdf.resolve_minimax_energy_reference(
        enk_v,
        enk_c,
        reference=minimax_energy_reference,
        reference_fn=minimax_energy_reference_fn,
    )

    _windows_minimax, quad = build_static_minimax_window_pair(
        enk_v,
        enk_c,
        target_error=target_error,
        max_nodes=max_nodes,
        print_fn=print0,
    )

    # Fresh minimax for chi0(i*omega_p): approximates x/(x^2+omega_p^2)
    # Fresh minimax for chi0(i*omega_p): approximates x/(x^2+omega_p^2)
    # directly, instead of cos-reweighting the static nodes (which is wrong).
    quad_imag = solve_laplace_minimax_imag_interval(
        quad.x_min, quad.x_max, omega_p_ry,
        target_error=target_error,
        max_nodes=max_nodes,
    )
    R = quad_imag.x_max / quad_imag.x_min
    omega_hat = omega_p_ry / quad_imag.x_min
    print0(
        f"  Minimax imag-freq window (ωp={omega_p_ry:.4f} Ry): "
        f"x=[{quad_imag.x_min:.6e}, {quad_imag.x_max:.6e}] Ry, "
        f"R={R:.2f}, ω̂={omega_hat:.2f}, "
        f"nodes={quad_imag.node_count}, fit_err~{quad_imag.max_error:.3e}"
    )

    chi0_q = w_isdf.compute_chi0_minimax(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, quad, meta, mesh_xy,
        energy_reference=e_ref,
    )
    chii_q = w_isdf.compute_chi0_minimax(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, quad_imag, meta, mesh_xy,
        energy_reference=e_ref,
    )
    W0_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chi0_q, meta, mesh_xy)
    Wiwp_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chii_q, meta, mesh_xy)
    W0_q.block_until_ready()
    Wiwp_q.block_until_ready()

    V_head = V_qmunu
    if head_correction_fn is not None:
        V_head, W0_q = head_correction_fn(V_qmunu, W0_q, 0.0 + 0.0j)
        _, Wiwp_q = head_correction_fn(V_qmunu, Wiwp_q, 1j * float(omega_p_ry))

    _summarize_hermitian_qmunu("W(0)", W0_q, print0)
    _summarize_hermitian_qmunu(f"W(iωp={omega_p_ry:.3f} Ry)", Wiwp_q, print0)

    # Build W^c = W(with W_head) - V(with V_head).
    nkx, nky, nkz = W0_q.shape[0], W0_q.shape[1], W0_q.shape[2]
    n_rmu = W0_q.shape[4]
    V_q = jnp.asarray(V_head)[0, 0, 0].reshape(nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
    Wc0_q = W0_q - V_q
    Wci_q = Wiwp_q - V_q


    ppm = extract_gn_ppm_parameters_from_Wc(
        Wc0_q,
        Wci_q,
        omega_p=omega_p_ry,
        fallback_omega=float(fallback_omega),
    )
    unfulfilled = float(ppm.unfulfilled_fraction)

    # GN poles are for W^c, so W^c(t) is a direct exponential sum.
    Omega = jnp.asarray(ppm.omega_qmunu).transpose(3, 4, 0, 1, 2)
    B = jnp.asarray(ppm.b_qmunu).transpose(3, 4, 0, 1, 2)
    valid_mask = jnp.asarray(ppm.valid_qmunu).transpose(3, 4, 0, 1, 2)
    Wc0_mu_nu = Wc0_q[:, :, :, 0, :, 0, :].transpose(3, 4, 0, 1, 2)
    mu_shard = _mu_nu_sharding(mesh_xy)

    with mesh_xy:
        Omega = jax.lax.with_sharding_constraint(Omega, mu_shard)
        B = jax.lax.with_sharding_constraint(B, mu_shard)
        valid_mask = jax.lax.with_sharding_constraint(valid_mask, mu_shard)
        Wc0_mu_nu = jax.lax.with_sharding_constraint(Wc0_mu_nu, mu_shard)

    # PPM consistency check: W^c(0) ≈ ± 2 B / Omega (elementwise).
    w0_rel_error = None
    w0_abs_error = None
    w0_ref_norm = None
    w0_rel_error_neg = None
    try:
        w0_target = Wc0_q[:, :, :, 0, :, 0, :].transpose(3, 4, 0, 1, 2)
        w0_pred = jnp.where(Omega != 0.0, (2.0 * B) / Omega, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        w0_pred_neg = -w0_pred
        diff = w0_pred - w0_target
        diff_neg = w0_pred_neg - w0_target
        w0_abs_error = float(jnp.max(jnp.abs(diff)))
        w0_abs_error_neg = float(jnp.max(jnp.abs(diff_neg)))
        w0_ref_norm = float(jnp.max(jnp.abs(Wc0_q)))
        w0_rel_error = w0_abs_error / max(w0_ref_norm, 1.0e-16)
        w0_rel_error_neg = w0_abs_error_neg / max(w0_ref_norm, 1.0e-16)
    except Exception:
        pass

    print0(
        "  GN-PPM from W^c(0), W^c(iωp): "
        f"ωp={omega_p_ry:.6f} Ry, unfulfilled={100.0 * unfulfilled:.2f}%"
    )
    if w0_rel_error is not None:
        print0(
            f"  PPM W^c(0) check (+2B/Ω): max|Δ|={w0_abs_error:.3e}, "
            f"rel={w0_rel_error:.3e} (ref max|W^c(0)|={w0_ref_norm:.3e})"
        )
    if w0_rel_error_neg is not None:
        print0(
            f"  PPM W^c(0) check (-2B/Ω): max|Δ|={w0_abs_error_neg:.3e}, "
            f"rel={w0_rel_error_neg:.3e}"
        )
    return PPMBuildResult(
        omega_p=omega_p_ry,
        W0_q=W0_q,
        Wiwp_q=Wiwp_q,
        Wc0_mu_nu=Wc0_mu_nu,
        B_mu_nu=B,
        Omega_mu_nu=Omega,
        valid_mask_mu_nu=valid_mask,
        unfulfilled_fraction=unfulfilled,
        n_nodes_static=quad.node_count,
        w0_rel_error=w0_rel_error,
        w0_abs_error=w0_abs_error,
        w0_ref_norm=w0_ref_norm,
    )


def build_ppm_w_time_q(
    B_mu_nu: jax.Array,
    Omega_mu_nu: jax.Array,
    t_node: complex,
    mask_B: jax.Array,
    E_ref_B: float,
    mesh_xy: Mesh,
) -> jax.Array:
    """Build masked PPM correlation interaction W^c(t) in q-space."""
    mu_shard = _mu_nu_sharding(mesh_xy)
    phase = jnp.exp(-1j * (Omega_mu_nu - jnp.asarray(E_ref_B, dtype=jnp.float64)) * t_node)
    Wc_t = jnp.where(mask_B, B_mu_nu * phase, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    with mesh_xy:
        Wc_t = jax.lax.with_sharding_constraint(Wc_t, mu_shard)
    return Wc_t


def _build_single_sigma_window(
    *,
    E_A: np.ndarray,
    E_B: np.ndarray,
    base_mask_A: np.ndarray,
    base_mask_B: np.ndarray,
    omega_nonneg_ry: np.ndarray,
    kernel_sign: int,
    target_error: float,
    max_nodes: int,
) -> list[_SigmaWindow]:
    A_vals = E_A[base_mask_A]
    B_vals = E_B[base_mask_B]
    if A_vals.size == 0 or B_vals.size == 0:
        return []
    S_min = float(np.min(A_vals) + np.min(B_vals))
    S_max = float(np.max(A_vals) + np.max(B_vals))
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    x_min = max(S_min, 1.0e-12)
    if kernel_sign == -1:
        x_max = max(S_max + omega_max, x_min * (1.0 + 1.0e-9))
    else:
        x_max = max(S_max, x_min * (1.0 + 1.0e-9))
    q = solve_laplace_minimax_interval(x_min, x_max, target_error=target_error, max_nodes=max_nodes)

    # get_sigma_mu_nu_fn already includes a global "-1" factor from convolution.
    docs_prefactor = -1.0 if kernel_sign == +1 else 1.0
    prefactor = -docs_prefactor
    return [
        _SigmaWindow(
            name="single",
            t_nodes=np.asarray(-1j * q.tau, dtype=np.complex128),
            alpha=np.asarray(q.alpha, dtype=np.float64),
            mask_A=np.asarray(base_mask_A, dtype=bool),
            mask_B=np.asarray(base_mask_B, dtype=bool),
            E_ref_A=float(np.min(A_vals)),
            E_ref_B=float(np.min(B_vals)),
            omega_sign=int(kernel_sign),
            project="full",
            prefactor=float(prefactor),
            x_min=float(x_min),
            x_max=float(x_max),
        )
    ]


def _build_three_sigma_windows(
    *,
    E_A: np.ndarray,
    E_B: np.ndarray,
    base_mask_A: np.ndarray,
    base_mask_B: np.ndarray,
    omega_nonneg_ry: np.ndarray,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
) -> list[_SigmaWindow]:
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    xi = max(float(regularization_width_ry), 1.0e-12)
    z_edge = float(edge_factor) * xi
    T = omega_max + z_edge
    windows: list[_SigmaWindow] = []

    for name in ("core", "a_stripe", "b_slab"):
        if name == "core":
            mA = base_mask_A & (E_A <= T)
            mB = base_mask_B & (E_B <= T)
        elif name == "a_stripe":
            mA = base_mask_A & (E_A > T)
            mB = base_mask_B & (E_B <= T)
        else:
            mA = base_mask_A
            mB = base_mask_B & (E_B > T)
        if not np.any(mA) or not np.any(mB):
            continue

        A_vals = E_A[mA]
        B_vals = E_B[mB]
        S_min = float(np.min(A_vals) + np.min(B_vals))
        S_max = float(np.max(A_vals) + np.max(B_vals))
        E_ref_A = float(np.min(A_vals))
        E_ref_B = float(np.min(B_vals))

        if name == "core":
            A_core = max(2.0 * T / xi, 1.0e-8)
            q_cross = solve_phase_minimax_bandwidth(
                A_core,
                target_error=target_error,
                max_nodes=crossing_max_nodes,
                eps_q=crossing_eps_q,
                target_kind="hgl",
            )
            t_nodes = np.asarray(q_cross.tau / xi, dtype=np.complex128)
            alpha = np.asarray(q_cross.alpha / xi, dtype=np.float64)
            project = "imag"
            docs_prefactor = 1.0
        else:
            x_min = max(S_min - (T - z_edge), z_edge, 1.0e-12)
            x_max = max(S_max, x_min * (1.0 + 1.0e-9))
            q = solve_laplace_minimax_interval(x_min, x_max, target_error=target_error, max_nodes=max_nodes)
            t_nodes = np.asarray(-1j * q.tau, dtype=np.complex128)
            alpha = np.asarray(q.alpha, dtype=np.float64)
            project = "full"
            docs_prefactor = -1.0

        windows.append(
            _SigmaWindow(
                name=name,
                t_nodes=t_nodes,
                alpha=alpha,
                mask_A=np.asarray(mA, dtype=bool),
                mask_B=np.asarray(mB, dtype=bool),
                E_ref_A=E_ref_A,
                E_ref_B=E_ref_B,
                omega_sign=+1,
                project=project,
                # get_sigma_mu_nu_fn carries an extra global -1.
                prefactor=float(-docs_prefactor),
                x_min=float(x_min) if name != "core" else None,
                x_max=float(x_max) if name != "core" else None,
                crossing_A=float(A_core) if name == "core" else None,
                crossing_kind="hgl" if name == "core" else None,
                t_cut=float(T),
                z_edge=float(z_edge),
            )
        )
    return windows


def _convolve_sigma_branch(
    *,
    omega_nonneg_ry: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
    B_mu_nu: jax.Array,
    Omega_mu_nu: jax.Array,
    base_mask_B: jax.Array,
    kernel_sign: int,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    psi_coh_rmuT_X: jax.Array,
    psi_coh_rmu_Y: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
    mesh_xy: Mesh,
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array],
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array],
    omega_sign_flip: int = 1,
    log_tag: str = "",
    print0=print,
) -> tuple[jax.Array, list[_SigmaWindow]]:
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])
    if n_omega == 0:
        shape = (0, psi_coh_rmuT_X.shape[1], psi_coh_rmuT_X.shape[2], psi_coh_rmuT_X.shape[1], psi_coh_rmuT_X.shape[2], nkx, nky, nkz)
        return jnp.zeros(shape, dtype=jnp.complex128), []

    E_A_host = np.asarray(jax.device_get(E_A), dtype=np.float64)
    base_A_host = np.asarray(jax.device_get(base_mask_A), dtype=bool)
    E_B_host = np.asarray(jax.device_get(Omega_mu_nu), dtype=np.float64)
    base_B_host = np.asarray(jax.device_get(base_mask_B), dtype=bool)

    if kernel_sign == +1 and float(np.max(omega_nonneg_ry)) > 1.0e-14:
        windows = _build_three_sigma_windows(
            E_A=E_A_host,
            E_B=E_B_host,
            base_mask_A=base_A_host,
            base_mask_B=base_B_host,
            omega_nonneg_ry=omega_nonneg_ry,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error,
            max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
        )
    else:
        windows = _build_single_sigma_window(
            E_A=E_A_host,
            E_B=E_B_host,
            base_mask_A=base_A_host,
            base_mask_B=base_B_host,
            omega_nonneg_ry=omega_nonneg_ry,
            kernel_sign=kernel_sign,
            target_error=target_error,
            max_nodes=max_nodes,
        )

    if not windows:
        shape = (n_omega, psi_coh_rmuT_X.shape[1], psi_coh_rmuT_X.shape[2], psi_coh_rmuT_X.shape[1], psi_coh_rmuT_X.shape[2], nkx, nky, nkz)
        return jnp.zeros(shape, dtype=jnp.complex128), windows

    eye_nb = jnp.eye(E_A.shape[1], dtype=jnp.complex128)
    omega_vec = jnp.asarray(omega_nonneg_ry, dtype=jnp.float64)
    mu_shard = _mu_nu_sharding(mesh_xy)
    acc_total = jnp.zeros(
        (n_omega, psi_coh_rmuT_X.shape[1], psi_coh_rmuT_X.shape[2], psi_coh_rmuT_X.shape[1], psi_coh_rmuT_X.shape[2], nkx, nky, nkz),
        dtype=jnp.complex128,
    )
    with mesh_xy:
        acc_total = jax.lax.with_sharding_constraint(acc_total, P(None, None, "x", None, "y", None, None, None))

    for win in windows:
        mask_A = jnp.asarray(win.mask_A)
        mask_B = jnp.asarray(win.mask_B)
        acc_win = jnp.zeros_like(acc_total)
        with mesh_xy:
            acc_win = jax.lax.with_sharding_constraint(acc_win, P(None, None, "x", None, "y", None, None, None))

        for t_node, alpha_node in zip(win.t_nodes, win.alpha):
            phase_A = jnp.exp(-1j * (E_A - jnp.asarray(win.E_ref_A, dtype=jnp.float64)) * t_node)
            weights_kn = jnp.where(mask_A, phase_A, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
            Gij = eye_nb[None, :, :] * weights_kn[:, :, None]
            G_k = get_G_mu_nu_fn(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij)
            W_t_q = build_ppm_w_time_q(B_mu_nu, Omega_mu_nu, t_node, mask_B, win.E_ref_B, mesh_xy)
            with mesh_xy:
                G_R = get_G_R_fn(G_k, nkx, nky, nkz)
                W_t_q = jax.lax.with_sharding_constraint(W_t_q, mu_shard)
                sigma_tau = get_sigma_mu_nu_fn(G_R, W_t_q, nk_tot, bispinor)
            omega_kernel = jnp.exp(1j * float(win.omega_sign) * float(omega_sign_flip) * omega_vec * t_node)
            alpha_eff = complex(alpha_node) * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
            acc_win = acc_win + jnp.asarray(alpha_eff, dtype=jnp.complex128) * omega_kernel[:, None, None, None, None, None, None, None] * sigma_tau[None, ...]

        if win.project == "full":
            projected = acc_win
        elif win.project == "imag":
            projected = jnp.imag(acc_win)
        else:
            projected = jnp.real(acc_win)
        acc_total = acc_total + jnp.asarray(win.prefactor, dtype=jnp.float64) * projected.astype(jnp.complex128)

    tag = f"{log_tag} " if log_tag else ""
    if kernel_sign == +1:
        n_core = sum(1 for w in windows if w.name == "core")
        n_ext = sum(1 for w in windows if w.name in ("a_stripe", "b_slab"))
        print0(f"  {tag}Σ^- windows: core={n_core}, exterior={n_ext}")
    else:
        n_tot = len(windows)
        n_nodes = sum(int(w.alpha.shape[0]) for w in windows)
        print0(f"  {tag}Σ^+ windows: count={n_tot}, total nodes={n_nodes}")
    if debug_quadrature:
        for w in windows:
            print0(
                f"  [quad] {tag}{w.name}: nodes={int(w.alpha.shape[0])}, "
                f"project={w.project}, pref={w.prefactor:+.1f}"
            )
    return acc_total, windows


def _convolve_sigma_branch_kij(
    *,
    omega_nonneg_ry: np.ndarray,
    omega_global_idx: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
    B_mu_nu: jax.Array,
    Omega_mu_nu: jax.Array,
    base_mask_B: jax.Array,
    kernel_sign: int,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    psi_coh_rmuT_X: jax.Array,
    psi_coh_rmu_Y: jax.Array,
    psi_proj_rmu_X: jax.Array,
    psi_proj_rmuT_Y: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
    mesh_xy: Mesh,
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array],
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array],
    get_sigma_kij_channels_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    omega_sign_flip: int = 1,
    log_tag: str = "",
    print0=print,
    omega_batch_size: int = 4,
    stream_writer: Callable[[np.ndarray, jax.Array], None] | None = None,
    scale: float = 1.0,
    debug_quadrature: bool = False,
    debug_quadrature_samples: int = 200,
    efermi_vac: float | None = None,
    axis_kind: str = "",
) -> tuple[jax.Array, list[_SigmaWindow]]:
    """Convolve sigma in one pass, accumulating directly in (k,i,j) space.

    The exact reduced-storage path contracts each complex τ-node contribution
    to two band-space channels [K[Re X_tau], K[Im X_tau]] and then mixes those
    channels with scalar frequency coefficients. This preserves the required
    per-window Re/Im projection without storing Σ_k(μ,ν,ω).
    """
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])
    if n_omega == 0:
        nk_proj = int(psi_proj_rmu_X.shape[0])
        nb_proj = int(psi_proj_rmu_X.shape[1])
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    E_A_host = np.asarray(jax.device_get(E_A), dtype=np.float64)
    base_A_host = np.asarray(jax.device_get(base_mask_A), dtype=bool)
    E_B_host = np.asarray(jax.device_get(Omega_mu_nu), dtype=np.float64)
    base_B_host = np.asarray(jax.device_get(base_mask_B), dtype=bool)

    if kernel_sign == +1 and float(np.max(omega_nonneg_ry)) > 1.0e-14:
        windows = _build_three_sigma_windows(
            E_A=E_A_host,
            E_B=E_B_host,
            base_mask_A=base_A_host,
            base_mask_B=base_B_host,
            omega_nonneg_ry=omega_nonneg_ry,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error,
            max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
        )
    else:
        windows = _build_single_sigma_window(
            E_A=E_A_host,
            E_B=E_B_host,
            base_mask_A=base_A_host,
            base_mask_B=base_B_host,
            omega_nonneg_ry=omega_nonneg_ry,
            kernel_sign=kernel_sign,
            target_error=target_error,
            max_nodes=max_nodes,
        )

    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])
    if not windows:
        return jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows

    if debug_quadrature:
        docs_mod = _load_docs_minimax_module()
        nsamp = max(50, int(debug_quadrature_samples))
        for win in windows:
            mask_A = np.asarray(win.mask_A, dtype=bool)
            mask_B = np.asarray(win.mask_B, dtype=bool)
            count_A = int(np.sum(mask_A))
            count_B = int(np.sum(mask_B))
            total_A = int(mask_A.size)
            total_B = int(mask_B.size)
            nband = None
            if mask_A.ndim == 2:
                nband = int(np.sum(np.any(mask_A, axis=0)))
            if count_A > 0:
                A_vals = np.asarray(E_A_host[mask_A], dtype=np.float64)
                A_min = float(np.min(A_vals))
                A_max = float(np.max(A_vals))
            else:
                A_min = None
                A_max = None
            if count_B > 0:
                B_vals = np.asarray(E_B_host[mask_B], dtype=np.float64)
                B_min = float(np.min(B_vals))
                B_max = float(np.max(B_vals))
            else:
                B_min = None
                B_max = None
            if win.x_min is not None and win.x_max is not None:
                x_grid = np.exp(np.linspace(np.log(win.x_min), np.log(win.x_max), nsamp))
                approx = np.zeros_like(x_grid)
                for tau, alpha in zip(win.t_nodes, win.alpha):
                    tau_l = float(np.abs(np.imag(tau)))
                    approx += float(alpha) * np.exp(-tau_l * x_grid)
                err = np.max(np.abs(1.0 / x_grid - approx))
                print0(
                    f"  [quad] {log_tag}{win.name}: nodes={int(win.alpha.shape[0])}, "
                    f"max|1/x-approx|={err:.3e} on [{win.x_min:.3e}, {win.x_max:.3e}]"
                )
            elif win.crossing_A is not None:
                u_grid = np.linspace(0.0, float(win.crossing_A), nsamp)
                tau_orig = np.asarray(np.real(win.t_nodes), dtype=np.float64) * float(regularization_width_ry)
                alpha_orig = np.asarray(win.alpha, dtype=np.float64) * float(regularization_width_ry)
                approx = np.zeros_like(u_grid)
                for tau, alpha in zip(tau_orig, alpha_orig):
                    approx += float(alpha) * np.sin(float(tau) * u_grid)
                target = docs_mod.G_hgl(u_grid) if win.crossing_kind == "hgl" else docs_mod.G_fermi(u_grid)
                err = np.max(np.abs(target - approx))
                print0(
                    f"  [quad] {log_tag}{win.name}: nodes={int(win.alpha.shape[0])}, "
                    f"max|G-approx|={err:.3e} on [0, {win.crossing_A:.3e}]"
                )
            band_note = f", bands={nband}" if nband is not None else ""
            if A_min is not None and B_min is not None:
                print0(
                    f"  [mask] {log_tag}{win.name}: A={count_A}/{total_A}{band_note} "
                    f"[{A_min:.3e},{A_max:.3e}] "
                    f"B={count_B}/{total_B} [{B_min:.3e},{B_max:.3e}]"
                )
            else:
                print0(
                    f"  [mask] {log_tag}{win.name}: A={count_A}/{total_A}{band_note} "
                    f"B={count_B}/{total_B}"
                )
            if efermi_vac is not None and A_min is not None:
                if axis_kind == "cond":
                    ec_min = A_min + float(efermi_vac)
                    ec_max = A_max + float(efermi_vac)
                    print0(
                        f"  [mask] {log_tag}{win.name}: Ec_vac [{ec_min:.3e},{ec_max:.3e}]"
                    )
                elif axis_kind == "val":
                    ev_min = float(efermi_vac) - A_max
                    ev_max = float(efermi_vac) - A_min
                    print0(
                        f"  [mask] {log_tag}{win.name}: Ev_vac [{ev_min:.3e},{ev_max:.3e}]"
                    )
            if win.t_cut is not None:
                print0(f"  [mask] {log_tag}{win.name}: T={win.t_cut:.3e}, z_edge={float(win.z_edge):.3e}")

    eye_nb = jnp.eye(E_A.shape[1], dtype=jnp.complex128)
    omega_vec = jnp.asarray(omega_nonneg_ry, dtype=jnp.float64)
    acc_total = None
    if stream_writer is None:
        acc_total = jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128)

    for win in windows:
        mask_A = jnp.asarray(win.mask_A)
        mask_B = jnp.asarray(win.mask_B)
        acc_win = None
        if acc_total is not None:
            acc_win = jnp.zeros_like(acc_total)

        for t_node, alpha_node in zip(win.t_nodes, win.alpha):
            phase_A = jnp.exp(-1j * (E_A - jnp.asarray(win.E_ref_A, dtype=jnp.float64)) * t_node)
            weights_kn = jnp.where(mask_A, phase_A, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
            Gij = eye_nb[None, :, :] * weights_kn[:, :, None]
            G_k = get_G_mu_nu_fn(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij)
            W_t_q = build_ppm_w_time_q(B_mu_nu, Omega_mu_nu, t_node, mask_B, win.E_ref_B, mesh_xy)
            with mesh_xy:
                G_R = get_G_R_fn(G_k, nkx, nky, nkz)
                sigma_tau = get_sigma_mu_nu_fn(G_R, W_t_q, nk_tot, bispinor)
            sigma_tau_kij_ri = get_sigma_kij_channels_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_tau)
            sigma_tau_kij_re = sigma_tau_kij_ri[0]
            sigma_tau_kij_im = sigma_tau_kij_ri[1]
            if debug_quadrature:
                re_mag = float(jnp.max(jnp.abs(sigma_tau_kij_re)))
                im_mag = float(jnp.max(jnp.abs(sigma_tau_kij_im)))
                print0(f"  [diag] {log_tag}{win.name}: |K[Re(σ)]|={re_mag:.4e}  |K[Im(σ)]|={im_mag:.4e}")
            alpha_eff = complex(alpha_node) * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
            omega_sign = float(win.omega_sign) * float(omega_sign_flip)
            pref = jnp.asarray(win.prefactor * scale, dtype=jnp.float64)
            if acc_win is not None:
                omega_kernel = jnp.exp(1j * omega_sign * omega_vec * t_node)
                coeff = jnp.asarray(alpha_eff, dtype=jnp.complex128) * omega_kernel
                coeff_re = jnp.real(coeff)[:, None, None, None]
                coeff_im = jnp.imag(coeff)[:, None, None, None]
                if win.project == "full":
                    # Full complex: c * K[σ] = (c_r + i*c_i) * (K[Re(σ)] + i*K[Im(σ)])
                    sigma_tau_kij_full = sigma_tau_kij_re[None, ...] + 1j * sigma_tau_kij_im[None, ...]
                    contrib = (coeff_re + 1j * coeff_im) * sigma_tau_kij_full
                elif win.project == "imag":
                    contrib = coeff_re * sigma_tau_kij_im[None, ...] + coeff_im * sigma_tau_kij_re[None, ...]
                else:
                    contrib = coeff_re * sigma_tau_kij_re[None, ...] - coeff_im * sigma_tau_kij_im[None, ...]
                acc_win = acc_win + pref * contrib.astype(jnp.complex128)
            else:
                # Stream in omega chunks without repeating τ-node work.
                for ibeg in range(0, n_omega, int(max(1, omega_batch_size))):
                    iend = min(ibeg + int(max(1, omega_batch_size)), n_omega)
                    idx = omega_global_idx[ibeg:iend]
                    omega_batch = omega_vec[ibeg:iend]
                    omega_kernel_batch = jnp.exp(1j * omega_sign * omega_batch * t_node)
                    coeff = jnp.asarray(alpha_eff, dtype=jnp.complex128) * omega_kernel_batch
                    coeff_re = jnp.real(coeff)[:, None, None, None]
                    coeff_im = jnp.imag(coeff)[:, None, None, None]
                    if win.project == "full":
                        sigma_tau_kij_full = sigma_tau_kij_re[None, ...] + 1j * sigma_tau_kij_im[None, ...]
                        batch_proj = (coeff_re + 1j * coeff_im) * sigma_tau_kij_full
                    elif win.project == "imag":
                        batch_proj = coeff_re * sigma_tau_kij_im[None, ...] + coeff_im * sigma_tau_kij_re[None, ...]
                    else:
                        batch_proj = coeff_re * sigma_tau_kij_re[None, ...] - coeff_im * sigma_tau_kij_im[None, ...]
                    stream_writer(idx, pref * batch_proj.astype(jnp.complex128))

        if acc_win is not None:
            acc_total = acc_total + acc_win

    tag = f"{log_tag} " if log_tag else ""
    if kernel_sign == +1:
        n_core = sum(1 for w in windows if w.name == "core")
        n_ext = sum(1 for w in windows if w.name in ("a_stripe", "b_slab"))
        print0(f"  {tag}Σ^- windows: core={n_core}, exterior={n_ext}")
    else:
        n_tot = len(windows)
        n_nodes = sum(int(w.alpha.shape[0]) for w in windows)
        print0(f"  {tag}Σ^+ windows: count={n_tot}, total nodes={n_nodes}")
    if acc_total is None:
        # Nothing accumulated in-memory for streaming path.
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows
    return acc_total, windows


def compute_sigma_c_ppm_omega_grid(
    *,
    psi_coh_rmuT_X: jax.Array,
    psi_coh_rmu_Y: jax.Array,
    psi_proj_rmu_X: jax.Array,
    psi_proj_rmuT_Y: jax.Array,
    enk_full: jax.Array,
    occ_full: jax.Array,
    B_mu_nu: jax.Array,
    Omega_mu_nu: jax.Array,
    Wc0_mu_nu: jax.Array | None = None,
    valid_mask_mu_nu: jax.Array | None = None,
    omega_values_ry: np.ndarray,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
    mesh_xy: Mesh,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    crossing_max_nodes: int = 500,
    crossing_eps_q: float = 1.0e-3,
    regularization_width_ry: float = 0.018374661087827496,  # 0.25 eV
    edge_factor: float = 1.5,
    omega_batch_size: int = 4,
    omega_accumulation: str = "auto",
    sigma_munu_h5_path: str | None = None,
    sigma_kij_h5_path: str | None = None,
    sigma_scale: float = 1.0,
    sigma_flip_neg: bool = False,
    invalid_mode: str = "static_limit",
    debug_split_contrib: bool = False,
    fermi_reference: str = "midgap",
    debug_quadrature: bool = False,
    debug_quadrature_samples: int = 200,
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array] | None = None,
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array] | None = None,
    get_sigma_kij_channels_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    print0=print,
) -> SigmaOmegaResult:
    """Compute frequency-dependent correlation self-energy Σ^c_kij(ω) with GN-PPM windows.

    Notes
    -----
    To avoid repeating τ-node work, Σ(t) is always contracted to band space
    before applying exp(iωt). When memory is constrained, set
    ``omega_accumulation='kij_stream'`` and provide ``sigma_kij_h5_path`` to
    stream Σ_kij(ω) to disk in ω-chunks.
    """
    if None in (get_G_mu_nu_fn, get_G_R_fn, get_sigma_mu_nu_fn, get_sigma_kij_channels_fn):
        raise ValueError("All reusable sigma helper callables must be provided.")
    nk = int(nkx * nky * nkz)
    if nk != int(enk_full.shape[0]):
        raise ValueError(f"enk_full shape mismatch: expected first dim {nk}, got {enk_full.shape[0]}")

    omega_req = np.asarray(omega_values_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_values_ry must be a 1D non-empty array.")
    omega_batch_size = int(max(1, omega_batch_size))
    omega_accumulation = str(omega_accumulation).strip().lower()
    if omega_accumulation not in ("auto", "kij", "kij_stream"):
        raise ValueError("omega_accumulation must be one of: auto, kij, kij_stream.")

    # Input omega grid is interpreted as relative to E_F (default driver behavior).
    omega_rel_req = omega_req
    idx_pos = np.where(omega_rel_req >= 0.0)[0]
    idx_neg = np.where(omega_rel_req < 0.0)[0]
    omega_pos = np.asarray(omega_rel_req[idx_pos], dtype=np.float64)
    omega_neg_abs = np.asarray(-omega_rel_req[idx_neg], dtype=np.float64)

    fermi_reference = str(fermi_reference).strip().lower()
    if fermi_reference not in ("vbm", "midgap"):
        raise ValueError("fermi_reference must be 'vbm' or 'midgap'.")
    occ_mask = occ_full > 0.5
    unocc_mask = ~occ_mask
    vbm = jnp.max(jnp.where(occ_mask, enk_full, -1.0e30))
    if fermi_reference == "midgap":
        cbm = jnp.min(jnp.where(unocc_mask, enk_full, 1.0e30))
        has_unocc = jnp.any(unocc_mask)
        efermi = jnp.where(has_unocc, 0.5 * (vbm + cbm), vbm)
    else:
        efermi = vbm
    E_cond = jnp.maximum(enk_full - efermi, 0.0)
    H_val = jnp.maximum(efermi - enk_full, 0.0)
    cond_mask = ~occ_mask
    val_mask = occ_mask
    Omega_abs = jnp.maximum(jnp.real(Omega_mu_nu), 0.0)
    B_corr = jnp.asarray(B_mu_nu, dtype=jnp.complex128)
    Omega_abs = jnp.asarray(Omega_abs, dtype=jnp.float64)
    B_mask = Omega_abs > 1.0e-14
    invalid_mode = str(invalid_mode).strip().lower()
    if invalid_mode not in ("fixed_2ry", "static_limit"):
        raise ValueError("invalid_mode must be 'fixed_2ry' or 'static_limit'.")
    if valid_mask_mu_nu is None:
        valid_mask = jnp.ones_like(B_mask, dtype=bool)
    else:
        valid_mask = jnp.asarray(valid_mask_mu_nu, dtype=bool)
    invalid_mask = B_mask & (~valid_mask)
    b_total_mask_host = np.asarray(B_mask | invalid_mask, dtype=bool)
    invalid_mask_host = np.asarray(invalid_mask, dtype=bool)
    if invalid_mode == "static_limit":
        B_mask = B_mask & valid_mask

    omega_step_ev = float(omega_req[1] - omega_req[0]) * 13.6056980659 if omega_req.size > 1 else 0.0
    print0(
        "  Σc(ω) grid: "
        f"{float(np.min(omega_req) * 13.6056980659):.3f} .. {float(np.max(omega_req) * 13.6056980659):.3f} eV, "
        f"Nω={omega_req.size}, Δω={omega_step_ev:.3f} eV, ξ={float(regularization_width_ry * 13.6056980659):.3f} eV"
    )
    if print0 is not None:
        occ_mask_host = np.asarray(occ_mask, dtype=bool)
        unocc_mask_host = np.asarray(unocc_mask, dtype=bool)
        enk_host = np.asarray(enk_full, dtype=np.float64)
        if np.any(unocc_mask_host):
            ec_min = float(np.min(enk_host[unocc_mask_host]))
            ec_max = float(np.max(enk_host[unocc_mask_host]))
        else:
            ec_min = float("nan")
            ec_max = float("nan")
        if np.any(occ_mask_host):
            ev_min = float(np.min(enk_host[occ_mask_host]))
            ev_max = float(np.max(enk_host[occ_mask_host]))
        else:
            ev_min = float("nan")
            ev_max = float("nan")
        omega_mask = np.asarray(B_mask, dtype=bool)
        omega_vals = np.asarray(Omega_abs[omega_mask], dtype=np.float64)
        if omega_vals.size:
            om_min = float(np.min(omega_vals))
            om_max = float(np.max(omega_vals))
        else:
            om_min = float("nan")
            om_max = float("nan")
        print0(
            "  Σc(ω) axes (vacuum): "
            f"Ev=[{ev_min:.6e},{ev_max:.6e}] Ry, "
            f"Ec=[{ec_min:.6e},{ec_max:.6e}] Ry, "
            f"Omega=[{om_min:.6e},{om_max:.6e}] Ry"
        )
        n_invalid = int(np.sum(invalid_mask_host, dtype=np.int64))
        if n_invalid:
            n_total_modes = int(np.sum(b_total_mask_host, dtype=np.int64))
            print0(
                f"  GN invalid modes: {n_invalid}/{n_total_modes} "
                f"({100.0 * n_invalid / max(n_total_modes, 1):.2f}%), "
                f"policy={invalid_mode}"
            )
    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])
    kij_bytes = float(omega_req.size * nk_proj * nb_proj * nb_proj * 16)
    use_kij_accum = omega_accumulation == "kij"
    use_kij_stream = omega_accumulation == "kij_stream"
    if omega_accumulation == "auto":
        use_kij_accum = (sigma_kij_h5_path is None) and (kij_bytes <= 0.5 * 1024**3)
        use_kij_stream = not use_kij_accum
    if use_kij_stream and not sigma_kij_h5_path:
        print0("  WARNING: omega_accumulation=kij_stream without sigma_kij_h5_path; falling back to kij.")
        use_kij_stream = False
        use_kij_accum = True
    if debug_split_contrib and use_kij_stream:
        print0("  WARNING: sigma_debug_split_contrib requires in-memory kij accumulation; disabling split.")
        debug_split_contrib = False
    if sigma_munu_h5_path:
        print0("  NOTE: sigma_munu_h5_path requested; mu-nu streaming is independent of kij accumulation.")
    if use_kij_accum:
        print0(f"  Σc(ω) accumulation: kij (single-pass), est={kij_bytes / (1024**2):.1f} MiB")
    if use_kij_stream:
        print0(f"  Σc(ω) streaming: kij_stream (ω-chunk={omega_batch_size})")

    n_omega = int(omega_req.size)
    sigma_kij_host = None if use_kij_stream else np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)
    sigma_plus_host = None
    sigma_minus_host = None
    sigma_invalid_static_host = None
    if debug_split_contrib and not use_kij_stream:
        sigma_plus_host = np.zeros_like(sigma_kij_host)
        sigma_minus_host = np.zeros_like(sigma_kij_host)

    stream_path = None
    h5_stream = None
    dset_sigma_munu = None
    if sigma_munu_h5_path:
        print0("  WARNING: sigma_munu_h5_path ignored; Σ(ω) now accumulates in band space before exp(iωt).")
        sigma_munu_h5_path = None

    kij_stream_path = None
    h5_kij = None
    dset_sigma_kij = None
    if use_kij_stream and sigma_kij_h5_path:
        if jax.process_count() != 1:
            print0("  WARNING: sigma_kij_h5_path requested with multi-process JAX; disabling kij streaming.")
            use_kij_stream = False
            use_kij_accum = True
            sigma_kij_host = np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)
        elif jax.process_index() == 0:
            kij_stream_path = str(sigma_kij_h5_path)
            kij_dir = os.path.dirname(os.path.abspath(kij_stream_path))
            if kij_dir:
                os.makedirs(kij_dir, exist_ok=True)
            k_chunks = max(1, min(4, nk_proj))
            o_chunks = max(1, min(omega_batch_size, n_omega))
            chunks = (o_chunks, k_chunks, nb_proj, nb_proj)
            h5_kij = h5py.File(kij_stream_path, "w")
            h5_kij.create_dataset("omega_ry", data=np.asarray(omega_req, dtype=np.float64))
            h5_kij.create_dataset("omega_ev", data=np.asarray(omega_req * 13.6056980659, dtype=np.float64))
            dset_sigma_kij = h5_kij.create_dataset(
                "sigma_c_kij_ry",
                shape=(n_omega, nk_proj, nb_proj, nb_proj),
                dtype=np.complex128,
                chunks=chunks,
                fillvalue=0.0,
            )
            h5_kij.attrs["layout"] = "omega,k,i,j"
            h5_kij.attrs["note"] = "Σ_c(kij,ω) streamed in ω-chunks; τ-node work is not repeated."


    try:
        if use_kij_accum or use_kij_stream:
            def _accumulate_kij_stream(global_idx: np.ndarray, contrib_batch: jax.Array) -> None:
                if dset_sigma_kij is None:
                    return
                idx = np.asarray(global_idx, dtype=np.int64)
                buf = dset_sigma_kij[idx]
                buf = buf + np.asarray(jax.device_get(contrib_batch), dtype=np.complex128)
                dset_sigma_kij[idx] = buf

            def _convolve_kij(
                omega_eval: np.ndarray,
                omega_idx: np.ndarray,
                kernel_cond: int,
                kernel_val: int,
                tag: str,
                scale: float = 1.0,
                omega_sign_flip_cond: int = 1,
                omega_sign_flip_val: int = 1,
            ) -> jax.Array | None:
                sigma_cond, _ = _convolve_sigma_branch_kij(
                    omega_nonneg_ry=omega_eval,
                    omega_global_idx=omega_idx,
                    E_A=E_cond,
                    base_mask_A=cond_mask,
                    B_mu_nu=B_corr,
                    Omega_mu_nu=Omega_abs,
                    base_mask_B=B_mask,
                    kernel_sign=kernel_cond,
                    regularization_width_ry=regularization_width_ry,
                    edge_factor=edge_factor,
                    target_error=target_error,
                    max_nodes=max_nodes,
                    crossing_eps_q=crossing_eps_q,
                    crossing_max_nodes=crossing_max_nodes,
                    psi_coh_rmuT_X=psi_coh_rmuT_X,
                    psi_coh_rmu_Y=psi_coh_rmu_Y,
                    psi_proj_rmu_X=psi_proj_rmu_X,
                    psi_proj_rmuT_Y=psi_proj_rmuT_Y,
                    nkx=nkx,
                    nky=nky,
                    nkz=nkz,
                    nk_tot=nk_tot,
                    bispinor=bispinor,
                    mesh_xy=mesh_xy,
                    get_G_mu_nu_fn=get_G_mu_nu_fn,
                    get_G_R_fn=get_G_R_fn,
                    get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
                    get_sigma_kij_channels_fn=get_sigma_kij_channels_fn,
                    omega_sign_flip=omega_sign_flip_cond,
                    log_tag=tag,
                    print0=print0,
                    omega_batch_size=omega_batch_size,
                    stream_writer=_accumulate_kij_stream if use_kij_stream else None,
                    scale=scale,
                    debug_quadrature=debug_quadrature,
                    debug_quadrature_samples=debug_quadrature_samples,
                    efermi_vac=float(efermi),
                    axis_kind="cond",
                )
                sigma_val, _ = _convolve_sigma_branch_kij(
                    omega_nonneg_ry=omega_eval,
                    omega_global_idx=omega_idx,
                    E_A=H_val,
                    base_mask_A=val_mask,
                    B_mu_nu=B_corr,
                    Omega_mu_nu=Omega_abs,
                    base_mask_B=B_mask,
                    kernel_sign=kernel_val,
                    regularization_width_ry=regularization_width_ry,
                    edge_factor=edge_factor,
                    target_error=target_error,
                    max_nodes=max_nodes,
                    crossing_eps_q=crossing_eps_q,
                    crossing_max_nodes=crossing_max_nodes,
                    psi_coh_rmuT_X=psi_coh_rmuT_X,
                    psi_coh_rmu_Y=psi_coh_rmu_Y,
                    psi_proj_rmu_X=psi_proj_rmu_X,
                    psi_proj_rmuT_Y=psi_proj_rmuT_Y,
                    nkx=nkx,
                    nky=nky,
                    nkz=nkz,
                    nk_tot=nk_tot,
                    bispinor=bispinor,
                    mesh_xy=mesh_xy,
                    get_G_mu_nu_fn=get_G_mu_nu_fn,
                    get_G_R_fn=get_G_R_fn,
                    get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
                    get_sigma_kij_channels_fn=get_sigma_kij_channels_fn,
                    omega_sign_flip=omega_sign_flip_val,
                    log_tag=tag,
                    print0=print0,
                    omega_batch_size=omega_batch_size,
                    stream_writer=_accumulate_kij_stream if use_kij_stream else None,
                    scale=scale,
                    debug_quadrature=debug_quadrature,
                    debug_quadrature_samples=debug_quadrature_samples,
                    efermi_vac=float(efermi),
                    axis_kind="val",
                )
                if use_kij_stream:
                    return None
                if debug_split_contrib:
                    return sigma_cond, sigma_val
                return sigma_cond + sigma_val
            # ω_rel >= 0: Σ^- gets crossing treatment, Σ^+ sign-definite.
            if omega_pos.size:
                sigma_pos = _convolve_kij(
                    omega_pos,
                    idx_pos,
                    kernel_cond=+1,
                    kernel_val=-1,
                    tag="ω>=E_F",
                    scale=float(sigma_scale),
                    omega_sign_flip_cond=1,
                    omega_sign_flip_val=1,
                )
                if sigma_pos is not None:
                    if debug_split_contrib:
                        sigma_cond, sigma_val = sigma_pos
                        sigma_kij_host[idx_pos] = np.asarray(jax.device_get(sigma_cond + sigma_val), dtype=np.complex128)
                        sigma_minus_host[idx_pos] = np.asarray(jax.device_get(sigma_cond), dtype=np.complex128)
                        sigma_plus_host[idx_pos] = np.asarray(jax.device_get(sigma_val), dtype=np.complex128)
                    else:
                        sigma_kij_host[idx_pos] = np.asarray(jax.device_get(sigma_pos), dtype=np.complex128)

            # ω_rel < 0: evaluate with |ω_rel| and swapped branch kernels.
            # Keep omega kernel signs unchanged (omega_sign_flip=+1). For w=|ω_rel|:
            #   D_minus = ω_rel - S_c = -(w + S_c)
            #   D_plus  = ω_rel + S_v = -(w - S_v)
            # In this implementation, kernel_sign=+1 realizes 1/(w-S) and
            # kernel_sign=-1 realizes 1/(w+S), so both routed branches require
            # the same extra global minus.
            if omega_neg_abs.size:
                neg_scale = -float(sigma_scale)
                if sigma_flip_neg:
                    # Optional debug-only sign flip knob.
                    neg_scale = float(sigma_scale)
                sigma_neg = _convolve_kij(
                    omega_neg_abs,
                    idx_neg,
                    kernel_cond=-1,
                    kernel_val=+1,
                    tag="ω<E_F",
                    scale=neg_scale,
                    omega_sign_flip_cond=1,
                    omega_sign_flip_val=1,
                )
                if sigma_neg is not None:
                    if debug_split_contrib:
                        sigma_cond, sigma_val = sigma_neg
                        sigma_kij_host[idx_neg] = np.asarray(jax.device_get(sigma_cond + sigma_val), dtype=np.complex128)
                        sigma_minus_host[idx_neg] = np.asarray(jax.device_get(sigma_cond), dtype=np.complex128)
                        sigma_plus_host[idx_neg] = np.asarray(jax.device_get(sigma_val), dtype=np.complex128)
                    else:
                        sigma_kij_host[idx_neg] = np.asarray(jax.device_get(sigma_neg), dtype=np.complex128)

            if invalid_mode == "static_limit" and Wc0_mu_nu is not None and np.any(invalid_mask_host):
                Wc0_invalid = jnp.where(
                    invalid_mask,
                    jnp.asarray(Wc0_mu_nu, dtype=jnp.complex128),
                    jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
                )
                occ_diag = jnp.where(occ_mask, 1.0 + 0.0j, 0.0 + 0.0j)
                Gij_occ = jnp.einsum(
                    "kn,nm->knm",
                    occ_diag.astype(jnp.complex128),
                    jnp.eye(int(occ_mask.shape[1]), dtype=jnp.complex128),
                    optimize=True,
                )
                with mesh_xy:
                    G_occ_k = get_G_mu_nu_fn(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij_occ)
                    G_occ_R = get_G_R_fn(G_occ_k, nkx, nky, nkz)
                    sigma_occ_munu = get_sigma_mu_nu_fn(G_occ_R, Wc0_invalid, nk_tot, bispinor)
                sigma_occ_kij = get_sigma_kij_channels_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_occ_munu)[0]

                with mesh_xy:
                    G_RI_k = get_G_mu_nu_fn(
                        psi_coh_rmuT_X,
                        psi_coh_rmu_Y,
                        jnp.broadcast_to(
                            jnp.eye(int(occ_mask.shape[1]), dtype=jnp.complex128)[None, :, :],
                            (nk, int(occ_mask.shape[1]), int(occ_mask.shape[1])),
                        ),
                    )
                    G_RI_R = get_G_R_fn(G_RI_k, nkx, nky, nkz)
                    sigma_ri_munu = get_sigma_mu_nu_fn(G_RI_R, Wc0_invalid, nk_tot, bispinor)
                sigma_ri_kij = get_sigma_kij_channels_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_ri_munu)[0]

                sigma_invalid_static = sigma_occ_kij - 0.5 * sigma_ri_kij
                sigma_invalid_static_host = np.asarray(jax.device_get(sigma_invalid_static), dtype=np.complex128)
                print0(
                    f"  GN invalid-mode static correction: max|Σ|="
                    f"{float(np.max(np.abs(sigma_invalid_static_host))):.6e} Ry"
                )
                if use_kij_stream and sigma_kij_h5_path:
                    with h5py.File(sigma_kij_h5_path, "r+") as h5_add:
                        dset = h5_add["sigma_c_kij_ry"]
                        for ibeg in range(0, n_omega, int(max(1, omega_batch_size))):
                            iend = min(ibeg + int(max(1, omega_batch_size)), n_omega)
                            dset[ibeg:iend] = dset[ibeg:iend] + sigma_invalid_static_host[None, ...]
                else:
                    sigma_kij_host = sigma_kij_host + sigma_invalid_static_host[None, ...]
        else:
            raise RuntimeError("Internal error: no valid Σc(ω) accumulation path selected.")
    finally:
        if h5_stream is not None:
            h5_stream.close()
        if h5_kij is not None:
            h5_kij.close()

    sigma_kij_req = None if sigma_kij_host is None else jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    sigma_plus_req = None if sigma_plus_host is None else jnp.asarray(sigma_plus_host, dtype=jnp.complex128)
    sigma_minus_req = None if sigma_minus_host is None else jnp.asarray(sigma_minus_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * 13.6056980659, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
        sigma_c_plus_kij=sigma_plus_req,
        sigma_c_minus_kij=sigma_minus_req,
        sigma_c_invalid_static_kij=(None if sigma_invalid_static_host is None else jnp.asarray(sigma_invalid_static_host, dtype=jnp.complex128)),
        sigma_munu_h5_path=stream_path,
        sigma_kij_h5_path=kij_stream_path,
    )


def compute_sigma_c_ppm_laplace(
    *,
    psi_coh_rmuT_X: jax.Array,
    psi_coh_rmu_Y: jax.Array,
    psi_proj_rmu_X: jax.Array,
    psi_proj_rmuT_Y: jax.Array,
    enk_full: jax.Array,
    occ_full: jax.Array,
    B_mu_nu: jax.Array,
    Omega_mu_nu: jax.Array,
    omega_eval_ry: float,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
    mesh_xy: Mesh,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    sigma_scale: float = 1.0,
    sigma_flip_neg: bool = False,
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array] | None = None,
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array] | None = None,
    get_sigma_kij_channels_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    print0=print,
) -> jax.Array:
    """Compatibility wrapper returning Σ^c_kij at one target frequency."""
    out = compute_sigma_c_ppm_omega_grid(
        psi_coh_rmuT_X=psi_coh_rmuT_X,
        psi_coh_rmu_Y=psi_coh_rmu_Y,
        psi_proj_rmu_X=psi_proj_rmu_X,
        psi_proj_rmuT_Y=psi_proj_rmuT_Y,
        enk_full=enk_full,
        occ_full=occ_full,
        B_mu_nu=B_mu_nu,
        Omega_mu_nu=Omega_mu_nu,
        omega_values_ry=np.asarray([float(omega_eval_ry)], dtype=np.float64),
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        nk_tot=nk_tot,
        bispinor=bispinor,
        mesh_xy=mesh_xy,
        target_error=target_error,
        max_nodes=max_nodes,
        sigma_scale=sigma_scale,
        sigma_flip_neg=sigma_flip_neg,
        get_G_mu_nu_fn=get_G_mu_nu_fn,
        get_G_R_fn=get_G_R_fn,
        get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
        get_sigma_kij_channels_fn=get_sigma_kij_channels_fn,
        print0=print0,
    )
    return out.sigma_c_kij[0]
