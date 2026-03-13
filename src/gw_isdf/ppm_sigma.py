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
    extract_gn_ppm_parameters,
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
    V_mu_nu: jax.Array
    unfulfilled_fraction: float
    n_nodes_static: int


@dataclass(frozen=True)
class SigmaOmegaResult:
    omega_ry: np.ndarray
    omega_ev: np.ndarray
    sigma_c_kij: jax.Array
    sigma_munu_h5_path: str | None = None


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
    # Canonical GN extraction: build poles from Π(0), Π(iωp), not directly from W ratios.
    # This mirrors the documented formulation and BerkeleyGW-style treatment.
    ppm = extract_gn_ppm_parameters(
        V_qmunu,
        chi0_q,
        chii_q,
        omega_p=omega_p_ry,
        fallback_omega=float(fallback_omega),
    )
    unfulfilled = float(ppm.unfulfilled_fraction)

    W0_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chi0_q, meta, mesh_xy)
    Wiwp_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chii_q, meta, mesh_xy)
    W0_q.block_until_ready()
    Wiwp_q.block_until_ready()

    # Keep GN poles in Π-space. W^c(t) is assembled as V @ Π(t) @ V at each node.
    V_local = jnp.asarray(V_qmunu)[0, 0, 0].transpose(3, 4, 0, 1, 2)
    Omega = jnp.asarray(ppm.omega_qmunu).transpose(3, 4, 0, 1, 2)
    B = jnp.asarray(ppm.b_qmunu).transpose(3, 4, 0, 1, 2)
    mu_shard = _mu_nu_sharding(mesh_xy)

    with mesh_xy:
        Omega = jax.lax.with_sharding_constraint(Omega, mu_shard)
        B = jax.lax.with_sharding_constraint(B, mu_shard)
        V_local = jax.lax.with_sharding_constraint(V_local, mu_shard)

    print0(
        "  GN-PPM from W(0), W(iωp): "
        f"ωp={omega_p_ry:.6f} Ry, unfulfilled={100.0 * unfulfilled:.2f}%"
    )
    return PPMBuildResult(
        omega_p=omega_p_ry,
        W0_q=W0_q,
        Wiwp_q=Wiwp_q,
        B_mu_nu=B,
        Omega_mu_nu=Omega,
        V_mu_nu=V_local,
        unfulfilled_fraction=unfulfilled,
        n_nodes_static=quad.node_count,
    )


def build_ppm_w_time_q(
    B_mu_nu: jax.Array,
    Omega_mu_nu: jax.Array,
    V_mu_nu: jax.Array,
    t_node: complex,
    mask_B: jax.Array,
    E_ref_B: float,
    mesh_xy: Mesh,
) -> jax.Array:
    """Build masked PPM correlation interaction W^c(t)=V Π(t) V in q-space."""
    mu_shard = _mu_nu_sharding(mesh_xy)
    phase = jnp.exp(-1j * (Omega_mu_nu - jnp.asarray(E_ref_B, dtype=jnp.float64)) * t_node)
    pi_t = jnp.where(mask_B, B_mu_nu * phase, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    # Convert to q-major for local matrix products on each q-block.
    v_q = V_mu_nu.transpose(2, 3, 4, 0, 1)
    pi_q = pi_t.transpose(2, 3, 4, 0, 1)
    w_q = jnp.einsum("...ab,...bc,...cd->...ad", v_q, pi_q, v_q, optimize=True)
    Wc_t = w_q.transpose(3, 4, 0, 1, 2)
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
    V_mu_nu: jax.Array,
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
            W_t_q = build_ppm_w_time_q(B_mu_nu, Omega_mu_nu, V_mu_nu, t_node, mask_B, win.E_ref_B, mesh_xy)
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
    V_mu_nu: jax.Array,
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
    sigma_munu_h5_path: str | None = None,
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array] | None = None,
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array] | None = None,
    get_sigma_kij_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    print0=print,
) -> SigmaOmegaResult:
    """Compute frequency-dependent correlation self-energy Σ^c_kij(ω) with GN-PPM windows."""
    if None in (get_G_mu_nu_fn, get_G_R_fn, get_sigma_mu_nu_fn, get_sigma_kij_fn):
        raise ValueError("All reusable sigma helper callables must be provided.")
    nk = int(nkx * nky * nkz)
    if nk != int(enk_full.shape[0]):
        raise ValueError(f"enk_full shape mismatch: expected first dim {nk}, got {enk_full.shape[0]}")

    omega_req = np.asarray(omega_values_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_values_ry must be a 1D non-empty array.")
    omega_batch_size = int(max(1, omega_batch_size))

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
    print0(f"  Σc(ω) batching: ω-batch={omega_batch_size}")

    def _run_sigma_pair(omega_eval: np.ndarray, kernel_cond: int, kernel_val: int, tag: str) -> jax.Array:
        sigma_cond, _ = _convolve_sigma_branch(
            omega_nonneg_ry=omega_eval,
            E_A=E_cond,
            base_mask_A=cond_mask,
            B_mu_nu=B_corr,
            Omega_mu_nu=Omega_abs,
            V_mu_nu=V_mu_nu,
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
            V_mu_nu=V_mu_nu,
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
    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])
    sigma_kij_host = np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)

    stream_path = None
    h5_stream = None
    dset_sigma_munu = None
    if sigma_munu_h5_path:
        if jax.process_count() != 1:
            print0("  WARNING: sigma_munu_h5_path requested with multi-process JAX; disabling streamed mu-nu dump.")
        elif jax.process_index() == 0:
            stream_path = str(sigma_munu_h5_path)
            stream_dir = os.path.dirname(os.path.abspath(stream_path))
            if stream_dir:
                os.makedirs(stream_dir, exist_ok=True)
            ns = int(psi_coh_rmuT_X.shape[1])
            n_rmu = int(psi_coh_rmuT_X.shape[2])
            mu_chunk = max(1, min(32, n_rmu))
            chunks = (1, ns, mu_chunk, ns, mu_chunk, 1, 1, 1)
            h5_stream = h5py.File(stream_path, "w")
            h5_stream.create_dataset("omega_ry", data=np.asarray(omega_req, dtype=np.float64))
            h5_stream.create_dataset("omega_ev", data=np.asarray(omega_req * 13.6056980659, dtype=np.float64))
            dset_sigma_munu = h5_stream.create_dataset(
                "sigma_c_munu_ry",
                shape=(n_omega, ns, n_rmu, ns, n_rmu, nkx, nky, nkz),
                dtype=np.complex128,
                chunks=chunks,
            )
            h5_stream.attrs["layout"] = "omega,s1,mu,s2,nu,kx,ky,kz"
            h5_stream.attrs["note"] = "Streamed in omega batches to avoid full Nw*Nmu^2 GPU residency."

    def _accumulate_batch(global_idx: np.ndarray, sigma_munu_batch: jax.Array) -> None:
        sigma_kij_batch = jax.vmap(
            lambda s_munu: get_sigma_kij_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, s_munu),
            in_axes=0,
        )(sigma_munu_batch)
        sigma_kij_host[np.asarray(global_idx, dtype=np.int64)] = np.asarray(
            jax.device_get(sigma_kij_batch), dtype=np.complex128
        )
        if dset_sigma_munu is not None:
            dset_sigma_munu[np.asarray(global_idx, dtype=np.int64)] = np.asarray(
                jax.device_get(sigma_munu_batch), dtype=np.complex128
            )

    try:
        # ω_rel >= 0: Σ^- gets crossing treatment, Σ^+ sign-definite.
        if omega_pos.size:
            for ibeg in range(0, omega_pos.size, omega_batch_size):
                iend = min(ibeg + omega_batch_size, omega_pos.size)
                idx_batch = idx_pos[ibeg:iend]
                omega_batch = omega_rel_req[idx_batch]
                sigma_batch = _run_sigma_pair(omega_batch, kernel_cond=+1, kernel_val=-1, tag="ω>=E_F")
                _accumulate_batch(idx_batch, sigma_batch)

        # ω_rel < 0: flip kernels with |ω_rel| (no conjugation shortcut).
        if omega_neg_abs.size:
            # For ω_rel = -|ω|:
            #   1/(ω_rel - S_c) = -1/(|ω| + S_c)  -> -(plus-kernel)
            #   1/(ω_rel + S_v) = -1/(|ω| - S_v)  -> -(minus-kernel)
            # so the flipped-kernel result carries an overall minus sign.
            for ibeg in range(0, omega_neg_abs.size, omega_batch_size):
                iend = min(ibeg + omega_batch_size, omega_neg_abs.size)
                idx_batch = idx_neg[ibeg:iend]
                omega_batch_abs = -omega_rel_req[idx_batch]
                sigma_batch = -_run_sigma_pair(omega_batch_abs, kernel_cond=-1, kernel_val=+1, tag="ω<E_F")
                _accumulate_batch(idx_batch, sigma_batch)
    finally:
        if h5_stream is not None:
            h5_stream.close()

    sigma_kij_req = jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * 13.6056980659, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
        sigma_munu_h5_path=stream_path,
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
    V_mu_nu: jax.Array,
    omega_eval_ry: float,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
    mesh_xy: Mesh,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
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
        V_mu_nu=V_mu_nu,
        omega_values_ry=np.asarray([float(omega_eval_ry)], dtype=np.float64),
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        nk_tot=nk_tot,
        bispinor=bispinor,
        mesh_xy=mesh_xy,
        target_error=target_error,
        max_nodes=max_nodes,
        get_G_mu_nu_fn=get_G_mu_nu_fn,
        get_G_R_fn=get_G_R_fn,
        get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
        get_sigma_kij_fn=get_sigma_kij_fn,
        print0=print0,
    )
    return out.sigma_c_kij[0]
