"""GN-PPM construction from W(0), W(i*omega_p) and sigma_c frequency integration.

This module reuses existing GW helpers:
  - chi/W evaluation: w_isdf.compute_chi0 and w_isdf.solve_w_from_chi_q_jax
  - G builder: greens_function_kernel.build_G
  - band projection: projection_kernel.project_ri
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

from common import jax_profile
from .minimax_config import MinimaxConfig, SigmaQuadratureConfig
from .minimax_screening import (
    build_static_minimax_window_pair,
    fit_gn_ppm_from_wc_pair,
    solve_laplace_minimax_interval,
    solve_laplace_minimax_imag_interval,
    solve_phase_minimax_bandwidth,
)
from . import w_isdf


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    W0_q: jax.Array           # (nq, μ, μ) flat-q
    Wiwp_q: jax.Array         # (nq, μ, μ) flat-q
    B_q: jax.Array            # (nq, μ, μ) PPM amplitude
    Omega_q: jax.Array        # (nq, μ, μ) PPM pole frequency
    valid_mask_q: jax.Array   # (nq, μ, μ)
    unfulfilled_fraction: float
    n_nodes_static: int


@dataclass(frozen=True)
class SigmaOmegaResult:
    omega_ry: np.ndarray
    omega_ev: np.ndarray
    sigma_c_kij: jax.Array | None      # (n_omega, nk, nb, nb) or None if streamed
    sigma_kij_h5_path: str | None = None


@dataclass(frozen=True)
class _SigmaWindow:
    name: str
    t_nodes: np.ndarray        # (n_tau,)
    alpha: np.ndarray          # (n_tau,)
    mask_A: np.ndarray         # (nk, nb)
    mask_B: np.ndarray | None  # (nk, nb)
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str               # "full", "real", or "imag"
    prefactor: float
    mask_B_mode: str = "explicit"
    mask_B_threshold: float | None = None
    crossing_kind: str | None = None


def _to_host_np(a, dtype=np.complex128, *, tiled: bool = False):
    """Gather a possibly sharded array to host."""
    try:
        return np.asarray(
            jax.experimental.multihost_utils.process_allgather(a, tiled=tiled),
            dtype=dtype,
        )
    except Exception:
        return np.asarray(jax.device_get(a), dtype=dtype)


def _to_host_scalar(a, dtype=float):
    np_dtype = np.dtype(dtype)
    gathered = _to_host_np(jnp.asarray(a), dtype=np_dtype, tiled=False)
    return dtype(np.asarray(gathered).reshape(-1)[0])


def _masked_stats_device(values: jax.Array, mask: jax.Array) -> tuple[int, int, float | None, float | None]:
    """Return total size, masked count, and masked min/max."""
    total = int(np.prod(values.shape))
    count = int(_to_host_scalar(jnp.sum(mask, dtype=jnp.int64), int))
    if count == 0:
        return total, 0, None, None
    min_val = float(_to_host_scalar(jnp.min(jnp.where(mask, values, jnp.inf)), float))
    max_val = float(_to_host_scalar(jnp.max(jnp.where(mask, values, -jnp.inf)), float))
    return total, count, min_val, max_val


def _materialize_window_mask_B(
    window: _SigmaWindow,
    *,
    base_mask_B: jax.Array,
    Omega_q: jax.Array,
) -> jax.Array:
    """Build one window's B-side selector lazily on device."""
    mode = str(window.mask_B_mode)
    if mode == "explicit":
        if window.mask_B is None:
            raise ValueError("window.mask_B must be provided when mask_B_mode='explicit'.")
        return jnp.asarray(window.mask_B, dtype=bool)
    if mode == "all":
        return jnp.asarray(base_mask_B, dtype=bool)
    threshold = jnp.asarray(window.mask_B_threshold, dtype=Omega_q.dtype)
    if mode == "le_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_q <= threshold)
    if mode == "gt_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_q > threshold)
    raise ValueError(f"Unknown mask_B_mode={mode!r}")


_sigma_tau_channel_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
_sigma_channel_pipeline_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}


def _project_sigma_channels(
    coeff_re: jax.Array,
    coeff_im: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Apply one window projection to precomputed real/imag sigma channels."""

    def _full(_):
        sigma_full = sigma_tau_kij_re[None, ...] + 1j * sigma_tau_kij_im[None, ...]
        return (coeff_re + 1j * coeff_im) * sigma_full

    def _imag(_):
        return coeff_re * sigma_tau_kij_im[None, ...] + coeff_im * sigma_tau_kij_re[None, ...]

    def _real(_):
        return coeff_re * sigma_tau_kij_re[None, ...] - coeff_im * sigma_tau_kij_im[None, ...]

    return jax.lax.switch(project_code, (_full, _imag, _real), operand=None)


@jax.jit
def _accumulate_window_channels_jit(
    acc_win: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    omega_vec: jax.Array,
    t_node: jax.Array,
    alpha_eff: jax.Array,
    omega_sign: jax.Array,
    pref: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    omega_kernel = jnp.exp(1j * omega_sign * omega_vec * t_node)
    coeff = alpha_eff * omega_kernel
    coeff_re = jnp.real(coeff)[:, None, None, None]
    coeff_im = jnp.imag(coeff)[:, None, None, None]
    contrib = _project_sigma_channels(
        coeff_re, coeff_im, sigma_tau_kij_re, sigma_tau_kij_im, project_code
    )
    return acc_win + pref * contrib.astype(jnp.complex128)


@jax.jit
def _project_stream_batch_jit(
    omega_batch: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    t_node: jax.Array,
    alpha_eff: jax.Array,
    omega_sign: jax.Array,
    pref: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    omega_kernel = jnp.exp(1j * omega_sign * omega_batch * t_node)
    coeff = alpha_eff * omega_kernel
    coeff_re = jnp.real(coeff)[:, None, None, None]
    coeff_im = jnp.imag(coeff)[:, None, None, None]
    contrib = _project_sigma_channels(
        coeff_re, coeff_im, sigma_tau_kij_re, sigma_tau_kij_im, project_code
    )
    return pref * contrib.astype(jnp.complex128)


def _get_sigma_channel_pipeline(
    *,
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
) -> Callable[..., jax.Array]:
    """Return a jit-compatible sigma-channel pipeline with device-local FFTs."""

    pipeline_key = (id(mesh_xy), nkx, nky, nkz, nk_tot, bispinor)
    if pipeline_key in _sigma_channel_pipeline_cache:
        return _sigma_channel_pipeline_cache[pipeline_key]

    from common.fft_helpers import (
        make_jittable_local_fftn_3d,
        make_jittable_local_ifftn_3d,
    )

    w_isdf._ensure_compilation_cache()
    _G_spec = P(None, None, None, None, 'x', None, 'y')
    _V_spec = P(None, None, None, 'x', 'y')
    _G_shard = NamedSharding(mesh_xy, _G_spec)
    _V_shard = NamedSharding(mesh_xy, _V_spec)
    _G_ifftn = make_jittable_local_ifftn_3d(mesh_xy, _G_spec, _G_spec, norm='ortho', axes=(0, 1, 2))
    _G_fftn = make_jittable_local_fftn_3d(mesh_xy, _G_spec, _G_spec, norm='ortho', axes=(0, 1, 2))
    _V_ifftn = make_jittable_local_ifftn_3d(mesh_xy, _V_spec, _V_spec, norm='ortho', axes=(0, 1, 2))
    nk = nkx * nky * nkz
    inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))

    def _fft_flat_G(x_k, fft_fn):
        x_3d = jax.lax.with_sharding_constraint(x_k.reshape(nkx, nky, nkz, *x_k.shape[1:]), _G_shard)
        return fft_fn(x_3d).reshape(nk, *x_k.shape[1:])

    def _fft_flat_V(x_k):
        x_3d = jax.lax.with_sharding_constraint(x_k.reshape(nkx, nky, nkz, *x_k.shape[1:]), _V_shard)
        return _V_ifftn(x_3d).reshape(nk, *x_k.shape[1:])

    from .greens_function_kernel import build_G as _build_G
    from .projection_kernel import project_ri as _project_channels

    @jax.jit
    def _sigma_channel_pipeline(
        psi_coh_rmuT_X, psi_coh_rmu_Y, psi_proj_rmu_X, psi_proj_rmuT_Y,
        Gij, W_q,
    ):
        """Σ_kij = project[ FFT[ G(R) · W(R) / √Nk ] ].  All flat-k.

        W_q is (nq, μ, μ) flat-q — same layout as all other flat-k arrays.
        """
        G_k = _build_G(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij=Gij)
        G_R = _fft_flat_G(G_k, _G_ifftn)
        V_R = _fft_flat_V(W_q)[:, None, :, None, :]  # (nk,1,μ,1,μ) broadcast to G shape
        sigma_k = _fft_flat_G(G_R * V_R * inv_sqrt_nk, _G_fftn)
        return _project_channels(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_k)

    _sigma_channel_pipeline_cache[pipeline_key] = _sigma_channel_pipeline
    return _sigma_channel_pipeline


def _get_sigma_tau_channel_kernel(
    *,
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
) -> Callable[..., jax.Array]:
    """Return a cached tau-node sigma builder with jittable local FFTs."""

    cache_key = (id(mesh_xy), nkx, nky, nkz, nk_tot, bispinor)
    if cache_key in _sigma_tau_channel_kernel_cache:
        return _sigma_tau_channel_kernel_cache[cache_key]

    w_isdf._ensure_compilation_cache()
    q_mu_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    sigma_channel_pipeline = _get_sigma_channel_pipeline(
        mesh_xy=mesh_xy, nkx=nkx, nky=nky, nkz=nkz,
        nk_tot=nk_tot, bispinor=bispinor,
    )

    @jax.jit
    def _build_tau_operands(
        E_A, mask_A, B_q, Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node, eye_nb,
    ):
        phase_A = jnp.exp(-1j * (E_A - E_ref_A) * t_node)
        weights_kn = jnp.where(mask_A, phase_A, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        Gij = eye_nb[None, :, :] * weights_kn[:, :, None]

        phase_B = jnp.exp(-1j * (Omega_q - E_ref_B) * t_node)
        W_t_q = jnp.where(mask_B, B_q * phase_B, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        W_t_q = jax.lax.with_sharding_constraint(W_t_q, q_mu_shard)
        return Gij, W_t_q

    @jax.jit
    def _tau_channel_step(
        psi_coh_rmuT_X, psi_coh_rmu_Y,
        psi_proj_rmu_X, psi_proj_rmuT_Y,
        E_A, mask_A, B_q, Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node, eye_nb,
    ):
        Gij, W_t_q = _build_tau_operands(
            E_A, mask_A, B_q, Omega_q, mask_B,
            E_ref_A, E_ref_B, t_node, eye_nb,
        )
        return sigma_channel_pipeline(
            psi_coh_rmuT_X, psi_coh_rmu_Y,
            psi_proj_rmu_X, psi_proj_rmuT_Y,
            Gij, W_t_q,
        )

    _sigma_tau_channel_kernel_cache[cache_key] = _tau_channel_step
    return _tau_channel_step


# ---------------------------------------------------------------------------
#  PPM construction
# ---------------------------------------------------------------------------

def fit_gn_ppm(
    W0_q: jax.Array,
    Wiwp_q: jax.Array,
    V_q: jax.Array,
    omega_p: float,
    mesh_xy: Mesh,
    *,
    fallback_omega: float = 2.0,
    n_nodes_static: int = 0,
    print_fn=None,
) -> PPMBuildResult:
    """Fit GN-PPM pole parameters from precomputed W(0) and W(iωp).

    All input arrays are flat-q (nq, μ, μ).  Returns PPMBuildResult with
    B_q, Omega_q, valid_mask_q sharded as P(None, 'x', 'y').
    """
    import time as _t
    omega_p = float(omega_p)
    t0 = _t.perf_counter()

    Wc0_q = W0_q - V_q
    Wci_q = Wiwp_q - V_q
    omega_qmunu, b_qmunu, valid_qmunu, unfulfilled = fit_gn_ppm_from_wc_pair(
        Wc0_q, Wci_q, 1j * complex(omega_p), fallback_omega=float(fallback_omega))

    q_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    Omega = jax.lax.with_sharding_constraint(jnp.asarray(omega_qmunu), q_shard)
    B = jax.lax.with_sharding_constraint(jnp.asarray(b_qmunu), q_shard)
    valid_mask = jax.lax.with_sharding_constraint(jnp.asarray(valid_qmunu), q_shard)
    t1 = _t.perf_counter()

    if print_fn is not None:
        print_fn(f"  GN-PPM fit: {t1-t0:.2f}s, ωp={omega_p:.4f} Ry, "
                 f"unfulfilled={100.0 * unfulfilled:.2f}%")

    return PPMBuildResult(
        omega_p=omega_p,
        W0_q=W0_q,
        Wiwp_q=Wiwp_q,
        B_q=B,
        Omega_q=Omega,
        valid_mask_q=valid_mask,
        unfulfilled_fraction=unfulfilled,
        n_nodes_static=n_nodes_static,
    )



# ---------------------------------------------------------------------------
#  Minimax window construction
# ---------------------------------------------------------------------------

def _build_single_sigma_window(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_count: int,
    mask_B_min: float | None,
    mask_B_max: float | None,
    omega_nonneg_ry: np.ndarray,
    kernel_sign: int,
    target_error: float,
    max_nodes: int,
    use_shipped_tables: bool,
) -> list[_SigmaWindow]:
    A_vals = E_A[base_mask_A]
    if A_vals.size == 0 or mask_B_count == 0 or mask_B_min is None or mask_B_max is None:
        return []
    S_min = float(np.min(A_vals) + mask_B_min)
    S_max = float(np.max(A_vals) + mask_B_max)
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    x_min = max(S_min, 1.0e-12)
    if kernel_sign == -1:
        x_max = max(S_max + omega_max, x_min * (1.0 + 1.0e-9))
    else:
        x_max = max(S_max, x_min * (1.0 + 1.0e-9))
    q = solve_laplace_minimax_interval(
        x_min, x_max,
        target_error=target_error,
        max_nodes=max_nodes,
        use_shipped_tables=use_shipped_tables,
    )
    prefactor = 1.0 if kernel_sign == +1 else -1.0
    return [
        _SigmaWindow(
            name="single",
            t_nodes=np.asarray(-1j * q.tau, dtype=np.complex128),
            alpha=np.asarray(q.alpha, dtype=np.float64),
            mask_A=np.asarray(base_mask_A, dtype=bool),
            mask_B=None,
            E_ref_A=float(np.min(A_vals)),
            E_ref_B=float(mask_B_min),
            omega_sign=int(kernel_sign),
            project="full",
            prefactor=float(prefactor),
            mask_B_mode="all",
        )
    ]


def _build_three_sigma_windows(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_all_count: int,
    mask_B_le_count: int,
    mask_B_le_min: float | None,
    mask_B_le_max: float | None,
    mask_B_gt_count: int,
    mask_B_gt_min: float | None,
    mask_B_gt_max: float | None,
    omega_nonneg_ry: np.ndarray,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    use_shipped_tables: bool,
) -> list[_SigmaWindow]:
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    xi = max(float(regularization_width_ry), 1.0e-12)
    z_edge = float(edge_factor) * xi
    T = omega_max + z_edge
    windows: list[_SigmaWindow] = []

    for name in ("core", "a_stripe", "b_slab"):
        if name == "core":
            mA = base_mask_A & (E_A <= T)
            mask_B_mode = "le_t"
            count_B = mask_B_le_count
            B_min = mask_B_le_min
            B_max = mask_B_le_max
        elif name == "a_stripe":
            mA = base_mask_A & (E_A > T)
            mask_B_mode = "le_t"
            count_B = mask_B_le_count
            B_min = mask_B_le_min
            B_max = mask_B_le_max
        else:
            mA = base_mask_A
            mask_B_mode = "gt_t"
            count_B = mask_B_gt_count
            B_min = mask_B_gt_min
            B_max = mask_B_gt_max
        if not np.any(mA) or count_B == 0 or B_min is None or B_max is None:
            continue

        A_vals = E_A[mA]
        E_ref_A = float(np.min(A_vals))
        E_ref_B = float(B_min)

        if name == "core":
            A_core = max(2.0 * T / xi, 1.0e-8)
            q_cross = solve_phase_minimax_bandwidth(
                A_core,
                target_error=target_error,
                max_nodes=crossing_max_nodes,
                eps_q=crossing_eps_q,
                target_kind="hgl",
                use_shipped_tables=use_shipped_tables,
            )
            t_nodes = np.asarray(q_cross.tau / xi, dtype=np.complex128)
            alpha = np.asarray(q_cross.alpha / xi, dtype=np.float64)
            project = "imag"
            prefactor = -1.0
        else:
            S_min = float(np.min(A_vals) + B_min)
            S_max = float(np.max(A_vals) + B_max)
            x_min = max(S_min - (T - z_edge), z_edge, 1.0e-12)
            x_max = max(S_max, x_min * (1.0 + 1.0e-9))
            q = solve_laplace_minimax_interval(
                x_min, x_max,
                target_error=target_error,
                max_nodes=max_nodes,
                use_shipped_tables=use_shipped_tables,
            )
            t_nodes = np.asarray(-1j * q.tau, dtype=np.complex128)
            alpha = np.asarray(q.alpha, dtype=np.float64)
            project = "full"
            prefactor = +1.0

        windows.append(
            _SigmaWindow(
                name=name,
                t_nodes=t_nodes,
                alpha=alpha,
                mask_A=np.asarray(mA, dtype=bool),
                mask_B=None,
                E_ref_A=E_ref_A,
                E_ref_B=E_ref_B,
                omega_sign=+1,
                project=project,
                prefactor=float(prefactor),
                mask_B_mode=mask_B_mode,
                mask_B_threshold=float(T),
                crossing_kind="hgl" if name == "core" else None,
            )
        )
    return windows


# ---------------------------------------------------------------------------
#  Sigma convolution
# ---------------------------------------------------------------------------

def _convolve_sigma_branch_kij(
    *,
    omega_nonneg_ry: np.ndarray,
    omega_global_idx: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    kernel_sign: int,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    wfns,
    mesh_xy: Mesh,
    meta,
    omega_sign_flip: int = 1,
    log_tag: str = "",
    print_fn=print,
    omega_batch_size: int = 4,
    stream_writer: Callable[[np.ndarray, jax.Array], None] | None = None,
    scale: float = 1.0,
    use_shipped_minimax_tables: bool = True,
) -> tuple[jax.Array, list[_SigmaWindow]]:
    """Convolve sigma for one branch (cond or val), accumulating in (k,i,j) space."""

    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])
    s = wfns.slices
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nk_tot = int(meta.nk_tot)
    bispinor = bool(meta.bispinor)
    psi_coh_rmuT_X = wfns.xn(s.full)
    psi_coh_rmu_Y = wfns.yr(s.full)
    psi_proj_rmu_X = wfns.xr(s.sigma)
    psi_proj_rmuT_Y = wfns.yn(s.sigma)
    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])

    if n_omega == 0:
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    E_A_host = _to_host_np(E_A, dtype=np.float64, tiled=False)
    base_A_host = _to_host_np(base_mask_A, dtype=bool, tiled=False)

    _, mask_B_all_count, mask_B_all_min, mask_B_all_max = _masked_stats_device(
        Omega_q, base_mask_B
    )
    if kernel_sign == +1 and float(np.max(omega_nonneg_ry)) > 1.0e-14:
        omega_max = float(np.max(omega_nonneg_ry))
        xi = max(float(regularization_width_ry), 1.0e-12)
        z_edge = float(edge_factor) * xi
        T = omega_max + z_edge
        le_mask_B = base_mask_B & (Omega_q <= T)
        gt_mask_B = base_mask_B & (Omega_q > T)
        _, mask_B_le_count, mask_B_le_min, mask_B_le_max = _masked_stats_device(Omega_q, le_mask_B)
        _, mask_B_gt_count, mask_B_gt_min, mask_B_gt_max = _masked_stats_device(Omega_q, gt_mask_B)
        windows = _build_three_sigma_windows(
            E_A=E_A_host,
            base_mask_A=base_A_host,
            mask_B_all_count=mask_B_all_count,
            mask_B_le_count=mask_B_le_count,
            mask_B_le_min=mask_B_le_min,
            mask_B_le_max=mask_B_le_max,
            mask_B_gt_count=mask_B_gt_count,
            mask_B_gt_min=mask_B_gt_min,
            mask_B_gt_max=mask_B_gt_max,
            omega_nonneg_ry=omega_nonneg_ry,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error,
            max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
            use_shipped_tables=bool(use_shipped_minimax_tables),
        )
    else:
        windows = _build_single_sigma_window(
            E_A=E_A_host,
            base_mask_A=base_A_host,
            mask_B_count=mask_B_all_count,
            mask_B_min=mask_B_all_min,
            mask_B_max=mask_B_all_max,
            omega_nonneg_ry=omega_nonneg_ry,
            kernel_sign=kernel_sign,
            target_error=target_error,
            max_nodes=max_nodes,
            use_shipped_tables=bool(use_shipped_minimax_tables),
        )

    if not windows:
        return jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows

    # Per-window summary
    for win in windows:
        A_vals = E_A_host[win.mask_A]
        kind = "crossing" if win.crossing_kind else "Laplace"
        print_fn(
            f"    {log_tag} window \"{win.name}\" ({kind}): "
            f"{int(win.alpha.shape[0])} nodes, err<{target_error:.0e}, "
            f"E_A=[{float(np.min(A_vals)):.4f}, {float(np.max(A_vals)):.4f}] Ry, "
            f"project={win.project}"
        )

    eye_nb = jnp.eye(E_A.shape[1], dtype=jnp.complex128)
    omega_vec = jnp.asarray(omega_nonneg_ry, dtype=jnp.float64)
    tau_channel_step = _get_sigma_tau_channel_kernel(
        mesh_xy=mesh_xy, nkx=nkx, nky=nky, nkz=nkz,
        nk_tot=nk_tot, bispinor=bispinor,
    )
    acc_total = None
    if stream_writer is None:
        acc_total = jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128)

    branch_label = log_tag if log_tag else f"kernel_sign={kernel_sign:+d}"
    total_tau_nodes = sum(int(win.alpha.shape[0]) for win in windows)
    from common.progress import LoopProgress
    progress = LoopProgress(
        total_tau_nodes, print_fn, title=f"sigma[{branch_label}]",
        item_name="tau node", max_updates=10)

    with jax_profile.annotation(f"sigma_branch[{branch_label}]"):
        for win_idx, win in enumerate(windows):
            with jax_profile.step_annotation(
                "sigma_window",
                step_num=win_idx,
                detail=f"{branch_label}:{win.name}:n{int(win.alpha.shape[0])}",
            ):
                mask_A = jnp.asarray(win.mask_A)
                mask_B = _materialize_window_mask_B(
                    win, base_mask_B=base_mask_B, Omega_q=Omega_q,
                )
                E_ref_A_j = jnp.asarray(win.E_ref_A, dtype=jnp.float64)
                E_ref_B_j = jnp.asarray(win.E_ref_B, dtype=jnp.float64)
                project_code = {"full": 0, "imag": 1}.get(win.project, 2)
                project_code_j = jnp.asarray(project_code, dtype=jnp.int32)
                acc_win = jnp.zeros_like(acc_total) if acc_total is not None else None

                for t_node, alpha_node in zip(win.t_nodes, win.alpha):
                    t_node_j = jnp.asarray(t_node, dtype=jnp.complex128)
                    with mesh_xy:
                        sigma_tau_kij_ri = tau_channel_step(
                            psi_coh_rmuT_X, psi_coh_rmu_Y,
                            psi_proj_rmu_X, psi_proj_rmuT_Y,
                            E_A, mask_A, B_q, Omega_q, mask_B,
                            E_ref_A_j, E_ref_B_j, t_node_j, eye_nb,
                        )
                    sigma_tau_kij_ri[0].block_until_ready()
                    progress.step()
                    sigma_tau_kij_re = sigma_tau_kij_ri[0]
                    sigma_tau_kij_im = sigma_tau_kij_ri[1]
                    alpha_eff = complex(alpha_node) * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
                    omega_sign = float(win.omega_sign) * float(omega_sign_flip)
                    alpha_eff_j = jnp.asarray(alpha_eff, dtype=jnp.complex128)
                    omega_sign_j = jnp.asarray(omega_sign, dtype=jnp.float64)
                    pref = jnp.asarray(win.prefactor * scale, dtype=jnp.float64)
                    if acc_win is not None:
                        acc_win = _accumulate_window_channels_jit(
                            acc_win, sigma_tau_kij_re, sigma_tau_kij_im,
                            omega_vec, t_node_j, alpha_eff_j,
                            omega_sign_j, pref, project_code_j,
                        )
                    else:
                        for ibeg in range(0, n_omega, int(max(1, omega_batch_size))):
                            iend = min(ibeg + int(max(1, omega_batch_size)), n_omega)
                            idx = omega_global_idx[ibeg:iend]
                            omega_batch = omega_vec[ibeg:iend]
                            batch_proj = _project_stream_batch_jit(
                                omega_batch, sigma_tau_kij_re, sigma_tau_kij_im,
                                t_node_j, alpha_eff_j, omega_sign_j,
                                pref, project_code_j,
                            )
                            stream_writer(idx, batch_proj)

                if acc_win is not None:
                    acc_total = acc_total + acc_win

    progress.finish()

    if acc_total is None:
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows
    return acc_total, windows


# ---------------------------------------------------------------------------
#  Top-level sigma driver
# ---------------------------------------------------------------------------

def compute_sigma_c_ppm_omega_grid(
    wfns,
    ppm,
    meta,
    mesh_xy: Mesh,
    ppm_options,
    *,
    sigma_window_quad: SigmaQuadratureConfig | None = None,
    print_fn=print,
) -> SigmaOmegaResult:
    """Compute Σ^c_kij(ω) via GN-PPM windowed minimax integration."""

    s = wfns.slices
    psi_proj_rmu_X = wfns.xr(s.sigma)
    enk_full = wfns.enk[:, s.full]
    occ_full = wfns.occ[:, s.full]
    B_q = ppm.B_q
    Omega_q = ppm.Omega_q
    valid_mask_q = getattr(ppm, 'valid_mask_q', None)
    omega_values_ry = ppm_options.omega_grid_ry

    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nk = int(nkx * nky * nkz)

    # Quadrature config
    if sigma_window_quad is not None:
        target_error = float(sigma_window_quad.target_error)
        max_nodes = int(sigma_window_quad.max_nodes)
        crossing_max_nodes = int(sigma_window_quad.crossing_max_nodes)
        crossing_eps_q = float(sigma_window_quad.crossing_eps_q)
        use_shipped_minimax_tables = bool(sigma_window_quad.use_shipped_tables)
    else:
        target_error, max_nodes = 1e-6, 64
        crossing_max_nodes, crossing_eps_q = 500, 1e-3
        use_shipped_minimax_tables = True

    regularization_width_ry = getattr(ppm_options, 'sigma_regularization_ry', 0.018374661087827496)
    edge_factor = getattr(ppm_options, 'sigma_edge_factor', 1.5)
    omega_batch_size = getattr(ppm_options, 'sigma_omega_batch_size', 4)
    omega_accumulation = getattr(ppm_options, 'sigma_omega_accumulation', 'auto')
    sigma_kij_h5_path = getattr(ppm_options, 'sigma_kij_h5_path', None)
    fermi_reference = getattr(ppm_options, 'fermi_reference', 'midgap')

    if nk != int(enk_full.shape[0]):
        raise ValueError(f"enk_full shape mismatch: expected first dim {nk}, got {enk_full.shape[0]}")

    omega_req = np.asarray(omega_values_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_values_ry must be a 1D non-empty array.")
    omega_batch_size = int(max(1, omega_batch_size))
    omega_accumulation = str(omega_accumulation).strip().lower()
    if omega_accumulation not in ("auto", "kij", "kij_stream"):
        raise ValueError("omega_accumulation must be one of: auto, kij, kij_stream.")

    # Split omega grid into positive and negative relative to Fermi level
    idx_pos = np.where(omega_req >= 0.0)[0]
    idx_neg = np.where(omega_req < 0.0)[0]
    omega_pos = np.asarray(omega_req[idx_pos], dtype=np.float64)
    omega_neg_abs = np.asarray(-omega_req[idx_neg], dtype=np.float64)

    # Fermi reference and band masks
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

    # PPM mode masking
    Omega_abs = jnp.maximum(jnp.real(Omega_q), 0.0).astype(jnp.float64)
    B_corr = jnp.asarray(B_q, dtype=jnp.complex128)
    B_mask = Omega_abs > 1.0e-14

    if valid_mask_q is None:
        valid_mask = jnp.ones_like(B_mask, dtype=bool)
    else:
        valid_mask = jnp.asarray(valid_mask_q, dtype=bool)
    invalid_mask = B_mask & (~valid_mask)
    n_total_modes = int(np.asarray(jax.device_get(jnp.sum(B_mask, dtype=jnp.int64)), dtype=np.int64))
    n_invalid = int(np.asarray(jax.device_get(jnp.sum(invalid_mask, dtype=jnp.int64)), dtype=np.int64))
    B_mask = B_mask & valid_mask

    ryd2ev = 13.6056980659
    omega_step_ev = float(omega_req[1] - omega_req[0]) * ryd2ev if omega_req.size > 1 else 0.0
    print_fn(
        f"  Σc(ω) grid: "
        f"{float(np.min(omega_req)) * ryd2ev:.3f}..{float(np.max(omega_req)) * ryd2ev:.3f} eV, "
        f"Nω={omega_req.size}, Δω={omega_step_ev:.3f} eV, "
        f"ξ={float(regularization_width_ry) * ryd2ev:.3f} eV"
    )
    if n_invalid:
        print_fn(
            f"  GN invalid modes: {n_invalid}/{n_total_modes} "
            f"({100.0 * n_invalid / max(n_total_modes, 1):.2f}%)"
        )

    # Accumulation mode
    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])
    kij_bytes = float(omega_req.size * nk_proj * nb_proj * nb_proj * 16)
    use_kij_accum = omega_accumulation == "kij"
    use_kij_stream = omega_accumulation == "kij_stream"
    if omega_accumulation == "auto":
        use_kij_accum = (sigma_kij_h5_path is None) and (kij_bytes <= 0.5 * 1024**3)
        use_kij_stream = not use_kij_accum
    if use_kij_stream and not sigma_kij_h5_path:
        use_kij_stream = False
        use_kij_accum = True

    n_omega = int(omega_req.size)

    # Stream mode is a fine-grained read-modify-write accumulator that
    # fires once per (tau_node × omega_batch); at multi-process scale
    # every call is a collective MPI-IO or rank-0 h5py round-trip, and
    # there are hundreds of them — so it's a real perf problem under
    # the current structure.  Until we refactor to accumulate on GPU
    # and stream out at branch granularity, fall back to the accum
    # path in multi-process runs.
    use_ffi_io = bool(getattr(ppm_options, 'use_ffi_io', False))
    if use_kij_stream and jax.process_count() != 1:
        use_kij_stream = False
        use_kij_accum = True

    sigma_kij_host = None if use_kij_stream else np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)

    # Single-process stream-mode file setup.  The accumulator pattern
    # itself is unchanged from pre-SlabIO (rank-0 h5py); the final
    # sigma_mnk.h5 copy-over is already migrated via
    # write_sigma_omega_h5 in gw_jax.py.
    kij_stream_path = None
    h5_kij = None
    dset_sigma_kij = None
    if use_kij_stream and sigma_kij_h5_path and jax.process_index() == 0:
        kij_stream_path = str(sigma_kij_h5_path)
        kij_dir = os.path.dirname(os.path.abspath(kij_stream_path))
        if kij_dir:
            os.makedirs(kij_dir, exist_ok=True)
        k_chunks = max(1, min(4, nk_proj))
        o_chunks = max(1, min(omega_batch_size, n_omega))
        h5_kij = h5py.File(kij_stream_path, "w")
        h5_kij.create_dataset("omega_ry", data=np.asarray(omega_req, dtype=np.float64))
        h5_kij.create_dataset("omega_ev", data=np.asarray(omega_req * ryd2ev, dtype=np.float64))
        dset_sigma_kij = h5_kij.create_dataset(
            "sigma_c_kij_ry",
            shape=(n_omega, nk_proj, nb_proj, nb_proj),
            dtype=np.complex128,
            chunks=(o_chunks, k_chunks, nb_proj, nb_proj),
            fillvalue=0.0,
        )
        h5_kij.attrs["layout"] = "omega,k,i,j"

    try:
        if not (use_kij_accum or use_kij_stream):
            raise RuntimeError("Internal error: no valid Σc(ω) accumulation path selected.")

        def _accumulate_kij_stream(global_idx: np.ndarray, contrib_batch: jax.Array) -> None:
            if dset_sigma_kij is None:
                return
            idx = np.asarray(global_idx, dtype=np.int64)
            buf = dset_sigma_kij[idx]
            buf = buf + np.asarray(jax.device_get(contrib_batch), dtype=np.complex128)
            dset_sigma_kij[idx] = buf

        common_branch_kwargs = dict(
            B_q=B_corr,
            Omega_q=Omega_abs,
            base_mask_B=B_mask,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error,
            max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
            wfns=wfns,
            mesh_xy=mesh_xy,
            meta=meta,
            print_fn=print_fn,
            omega_batch_size=omega_batch_size,
            stream_writer=_accumulate_kij_stream if use_kij_stream else None,
            use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
        )

        # ω_rel >= 0: Σ^- (crossing) for cond, Σ^+ (sign-definite) for val
        if omega_pos.size:
            sigma_cond_pos, _ = _convolve_sigma_branch_kij(
                omega_nonneg_ry=omega_pos, omega_global_idx=idx_pos,
                E_A=E_cond, base_mask_A=cond_mask, kernel_sign=+1,
                omega_sign_flip=1, log_tag="ω≥E_F cond", scale=1.0,
                **common_branch_kwargs,
            )
            sigma_val_pos, _ = _convolve_sigma_branch_kij(
                omega_nonneg_ry=omega_pos, omega_global_idx=idx_pos,
                E_A=H_val, base_mask_A=val_mask, kernel_sign=-1,
                omega_sign_flip=1, log_tag="ω≥E_F val", scale=1.0,
                **common_branch_kwargs,
            )
            if not use_kij_stream:
                sigma_kij_host[idx_pos] = np.asarray(
                    jax.device_get(sigma_cond_pos + sigma_val_pos), dtype=np.complex128)

        # ω_rel < 0: evaluate with |ω_rel| and swapped branch kernels, global -1 scale
        if omega_neg_abs.size:
            sigma_cond_neg, _ = _convolve_sigma_branch_kij(
                omega_nonneg_ry=omega_neg_abs, omega_global_idx=idx_neg,
                E_A=E_cond, base_mask_A=cond_mask, kernel_sign=-1,
                omega_sign_flip=1, log_tag="ω<E_F cond", scale=-1.0,
                **common_branch_kwargs,
            )
            sigma_val_neg, _ = _convolve_sigma_branch_kij(
                omega_nonneg_ry=omega_neg_abs, omega_global_idx=idx_neg,
                E_A=H_val, base_mask_A=val_mask, kernel_sign=+1,
                omega_sign_flip=1, log_tag="ω<E_F val", scale=-1.0,
                **common_branch_kwargs,
            )
            if not use_kij_stream:
                sigma_kij_host[idx_neg] = np.asarray(
                    jax.device_get(sigma_cond_neg + sigma_val_neg), dtype=np.complex128)
    finally:
        if h5_kij is not None:
            h5_kij.close()

    ryd2ev = 13.6056980659
    sigma_kij_req = None if sigma_kij_host is None else jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * ryd2ev, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
        sigma_kij_h5_path=kij_stream_path,
    )
