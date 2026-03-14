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
    solve_phase_minimax_bandwidth,
)
from . import w_isdf


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    W0_q: jax.Array
    Wiwp_q: jax.Array
    B_mu_nu: jax.Array
    Omega_mu_nu: jax.Array
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


def _mu_nu_sharding(mesh_xy: Mesh) -> NamedSharding:
    # (mu, nu, kx, ky, kz) with mu/nu split across x/y mesh axes.
    return NamedSharding(mesh_xy, P("x", "y", None, None, None))


def _extract_mu_nu_q_layout(W_q: jax.Array) -> jax.Array:
    # (nkx, nky, nkz, 1, mu, 1, nu) -> (mu, nu, nkx, nky, nkz)
    return W_q[:, :, :, 0, :, 0, :].transpose(3, 4, 0, 1, 2)


def compute_w0_wiwp_and_ppm_from_minimax(
    V_qmunu: jax.Array,
    wf_bundle,
    meta,
    mesh_xy: Mesh,
    *,
    omega_p_ry: float = 2.0,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
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

    windows_minimax, quad = build_static_minimax_window_pair(
        enk_v,
        enk_c,
        target_error=target_error,
        max_nodes=max_nodes,
        print_fn=print0,
    )
    w0 = windows_minimax[0]
    wiw = w0.with_imag_freq_modulation(omega_p_ry)

    chi0_q = w_isdf.compute_chi0(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, [w0], meta, mesh_xy,
    )
    chii_q = w_isdf.compute_chi0(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, [wiw], meta, mesh_xy,
    )
    W0_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chi0_q, meta, mesh_xy)
    Wiwp_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chii_q, meta, mesh_xy)
    W0_q.block_until_ready()
    Wiwp_q.block_until_ready()

    V_head = V_qmunu
    if head_correction_fn is not None:
        V_head, W0_q = head_correction_fn(V_qmunu, W0_q, 0.0 + 0.0j)
        _, Wiwp_q = head_correction_fn(V_qmunu, Wiwp_q, 1j * float(omega_p_ry))

    # Build W^c(0) and W^c(iωp) without head: W^c = W - V.
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
    mu_shard = _mu_nu_sharding(mesh_xy)

    with mesh_xy:
        Omega = jax.lax.with_sharding_constraint(Omega, mu_shard)
        B = jax.lax.with_sharding_constraint(B, mu_shard)

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
        B_mu_nu=B,
        Omega_mu_nu=Omega,
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
            project="real",
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
            project = "real"
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
            omega_kernel = jnp.exp(1j * float(win.omega_sign) * omega_vec * t_node)
            alpha_eff = complex(alpha_node) * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
            acc_win = acc_win + jnp.asarray(alpha_eff, dtype=jnp.complex128) * omega_kernel[:, None, None, None, None, None, None, None] * sigma_tau[None, ...]

        projected = jnp.imag(acc_win) if win.project == "imag" else jnp.real(acc_win)
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
    get_sigma_kij_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    log_tag: str = "",
    print0=print,
    omega_batch_size: int = 4,
    stream_writer: Callable[[np.ndarray, jax.Array], None] | None = None,
    scale: float = 1.0,
    debug_quadrature: bool = False,
    debug_quadrature_samples: int = 200,
) -> tuple[jax.Array, list[_SigmaWindow]]:
    """Convolve sigma in one pass, accumulating directly in (k,i,j) space."""
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
            if win.x_min is not None and win.x_max is not None:
                x_grid = np.exp(np.linspace(np.log(win.x_min), np.log(win.x_max), nsamp))
                approx = np.zeros_like(x_grid)
                for tau, alpha in zip(win.t_nodes, win.alpha):
                    tau_l = float(np.abs(np.imag(tau)))
                    approx += float(alpha) * np.exp(-tau_l * x_grid)
                err = np.max(np.abs(1.0 / x_grid - approx))
                print0(f"  [quad] {log_tag}{win.name}: max|1/x-approx|={err:.3e} on [{win.x_min:.3e}, {win.x_max:.3e}]")
            elif win.crossing_A is not None:
                u_grid = np.linspace(0.0, float(win.crossing_A), nsamp)
                tau_orig = np.asarray(np.real(win.t_nodes), dtype=np.float64) * float(regularization_width_ry)
                alpha_orig = np.asarray(win.alpha, dtype=np.float64) * float(regularization_width_ry)
                approx = np.zeros_like(u_grid)
                for tau, alpha in zip(tau_orig, alpha_orig):
                    approx += float(alpha) * np.sin(float(tau) * u_grid)
                target = docs_mod.G_hgl(u_grid) if win.crossing_kind == "hgl" else docs_mod.G_fermi(u_grid)
                err = np.max(np.abs(target - approx))
                print0(f"  [quad] {log_tag}{win.name}: max|G-approx|={err:.3e} on [0, {win.crossing_A:.3e}]")

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
            sigma_tau_kij = get_sigma_kij_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_tau)
            # Project to band space BEFORE applying exp(iωt).
            alpha_eff = complex(alpha_node) * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
            if acc_win is not None:
                omega_kernel = jnp.exp(1j * float(win.omega_sign) * omega_vec * t_node)
                contrib = jnp.asarray(alpha_eff, dtype=jnp.complex128) * omega_kernel[:, None, None, None] * sigma_tau_kij[None, ...]
                acc_win = acc_win + contrib
            else:
                # Stream in omega chunks without repeating τ-node work.
                for ibeg in range(0, n_omega, int(max(1, omega_batch_size))):
                    iend = min(ibeg + int(max(1, omega_batch_size)), n_omega)
                    idx = omega_global_idx[ibeg:iend]
                    omega_batch = omega_vec[ibeg:iend]
                    omega_kernel_batch = jnp.exp(1j * float(win.omega_sign) * omega_batch * t_node)
                    batch = jnp.asarray(alpha_eff, dtype=jnp.complex128) * omega_kernel_batch[:, None, None, None] * sigma_tau_kij[None, ...]
                    projected = jnp.imag(batch) if win.project == "imag" else jnp.real(batch)
                    batch_proj = (win.prefactor * scale) * projected.astype(jnp.complex128)
                    stream_writer(idx, batch_proj)

        if acc_win is not None:
            projected = jnp.imag(acc_win) if win.project == "imag" else jnp.real(acc_win)
            acc_total = acc_total + jnp.asarray(win.prefactor * scale, dtype=jnp.float64) * projected.astype(jnp.complex128)

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
    debug_quadrature: bool = False,
    debug_quadrature_samples: int = 200,
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array] | None = None,
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array] | None = None,
    get_sigma_kij_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
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
    if None in (get_G_mu_nu_fn, get_G_R_fn, get_sigma_mu_nu_fn, get_sigma_kij_fn):
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

    occ_mask = occ_full > 0.5
    efermi = jnp.max(jnp.where(occ_mask, enk_full, -1.0e30))
    E_cond = jnp.maximum(enk_full - efermi, 0.0)
    H_val = jnp.maximum(efermi - enk_full, 0.0)
    cond_mask = ~occ_mask
    val_mask = occ_mask
    Omega_abs = jnp.maximum(jnp.real(Omega_mu_nu), 0.0)
    B_corr = jnp.asarray(B_mu_nu, dtype=jnp.complex128)
    Omega_abs = jnp.asarray(Omega_abs, dtype=jnp.float64)
    B_mask = Omega_abs > 1.0e-14

    print0(
        "  Σc(ω) grid: "
        f"{float(np.min(omega_req) * 13.6056980659):.3f} .. {float(np.max(omega_req) * 13.6056980659):.3f} eV, "
        f"Nω={omega_req.size}, ξ={float(regularization_width_ry * 13.6056980659):.3f} eV"
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
    if sigma_munu_h5_path:
        print0("  NOTE: sigma_munu_h5_path requested; mu-nu streaming is independent of kij accumulation.")
    if use_kij_accum:
        print0(f"  Σc(ω) accumulation: kij (single-pass), est={kij_bytes / (1024**2):.1f} MiB")
    if use_kij_stream:
        print0(f"  Σc(ω) streaming: kij_stream (ω-chunk={omega_batch_size})")

    def _run_sigma_pair(omega_eval: np.ndarray, kernel_cond: int, kernel_val: int, tag: str) -> jax.Array:
        sigma_cond, _ = _convolve_sigma_branch(
            omega_nonneg_ry=omega_eval,
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
            nkx=nkx,
            nky=nky,
            nkz=nkz,
            nk_tot=nk_tot,
            bispinor=bispinor,
            mesh_xy=mesh_xy,
            get_G_mu_nu_fn=get_G_mu_nu_fn,
            get_G_R_fn=get_G_R_fn,
            get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
            log_tag=tag,
            print0=print0,
        )
        sigma_val, _ = _convolve_sigma_branch(
            omega_nonneg_ry=omega_eval,
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
            nkx=nkx,
            nky=nky,
            nkz=nkz,
            nk_tot=nk_tot,
            bispinor=bispinor,
            mesh_xy=mesh_xy,
            get_G_mu_nu_fn=get_G_mu_nu_fn,
            get_G_R_fn=get_G_R_fn,
            get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
            log_tag=tag,
            print0=print0,
            )
        return sigma_cond + sigma_val

    n_omega = int(omega_req.size)
    sigma_kij_host = None if use_kij_stream else np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)

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

            def _convolve_kij(omega_eval: np.ndarray, omega_idx: np.ndarray, kernel_cond: int, kernel_val: int, tag: str, scale: float = 1.0) -> jax.Array | None:
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
                    get_sigma_kij_fn=get_sigma_kij_fn,
                    log_tag=tag,
                    print0=print0,
                    omega_batch_size=omega_batch_size,
                    stream_writer=_accumulate_kij_stream if use_kij_stream else None,
                    scale=scale,
                    debug_quadrature=debug_quadrature,
                    debug_quadrature_samples=debug_quadrature_samples,
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
                    get_sigma_kij_fn=get_sigma_kij_fn,
                    log_tag=tag,
                    print0=print0,
                    omega_batch_size=omega_batch_size,
                    stream_writer=_accumulate_kij_stream if use_kij_stream else None,
                    scale=scale,
                    debug_quadrature=debug_quadrature,
                    debug_quadrature_samples=debug_quadrature_samples,
                )
                if use_kij_stream:
                    return None
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
                )
                if sigma_pos is not None:
                    sigma_kij_host[idx_pos] = np.asarray(jax.device_get(sigma_pos), dtype=np.complex128)

            # ω_rel < 0: flip kernels with |ω_rel| (no conjugation shortcut).
            if omega_neg_abs.size:
                neg_scale = float(sigma_scale) if sigma_flip_neg else -float(sigma_scale)
                sigma_neg = _convolve_kij(
                    omega_neg_abs,
                    idx_neg,
                    kernel_cond=-1,
                    kernel_val=+1,
                    tag="ω<E_F",
                    scale=neg_scale,
                )
                if sigma_neg is not None:
                    sigma_kij_host[idx_neg] = np.asarray(jax.device_get(sigma_neg), dtype=np.complex128)
        else:
            raise RuntimeError("Internal error: no valid Σc(ω) accumulation path selected.")
    finally:
        if h5_stream is not None:
            h5_stream.close()
        if h5_kij is not None:
            h5_kij.close()

    sigma_kij_req = None if sigma_kij_host is None else jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * 13.6056980659, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
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
    get_sigma_kij_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
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
        get_sigma_kij_fn=get_sigma_kij_fn,
        print0=print0,
    )
    return out.sigma_c_kij[0]
