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

from common.minimax import G_hgl as _G_hgl, G_fermi as _G_fermi
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
    mask_B: np.ndarray | None
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str
    prefactor: float
    mask_B_mode: str = "explicit"
    mask_B_threshold: float | None = None
    mask_B_count: int | None = None
    mask_B_total: int | None = None
    mask_B_min: float | None = None
    mask_B_max: float | None = None
    x_min: float | None = None
    x_max: float | None = None
    crossing_A: float | None = None
    crossing_kind: str | None = None
    t_cut: float | None = None
    z_edge: float | None = None


def _mu_nu_sharding(mesh_xy: Mesh) -> NamedSharding:
    # (mu, nu, kx, ky, kz) with mu/nu split across x/y mesh axes.
    return NamedSharding(mesh_xy, P("x", "y", None, None, None))

def _to_host_np(a, dtype=np.complex128, *, tiled: bool = False):
    """Gather a possibly sharded array to host without assuming single-process JAX."""
    try:
        return np.asarray(
            jax.experimental.multihost_utils.process_allgather(a, tiled=tiled),
            dtype=dtype,
        )
    except Exception:
        return np.asarray(jax.device_get(a), dtype=dtype)


def _to_host_scalar(a, dtype=float):
    """Fetch one replicated scalar to host, tolerating multihost arrays."""

    np_dtype = np.dtype(dtype)
    gathered = _to_host_np(jnp.asarray(a), dtype=np_dtype, tiled=False)
    return dtype(np.asarray(gathered).reshape(-1)[0])


def _masked_stats_device(values: jax.Array, mask: jax.Array) -> tuple[int, int, float | None, float | None]:
    """Return total size, masked count, and masked min/max using only scalar host fetches."""

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
    Omega_mu_nu: jax.Array,
) -> jax.Array:
    """Build one window's B-side selector lazily on device."""

    mode = str(window.mask_B_mode)
    if mode == "explicit":
        if window.mask_B is None:
            raise ValueError("window.mask_B must be provided when mask_B_mode='explicit'.")
        return jnp.asarray(window.mask_B, dtype=bool)
    if mode == "all":
        return jnp.asarray(base_mask_B, dtype=bool)
    threshold = jnp.asarray(window.mask_B_threshold, dtype=Omega_mu_nu.dtype)
    if mode == "le_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_mu_nu <= threshold)
    if mode == "gt_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_mu_nu > threshold)
    raise ValueError(f"Unknown mask_B_mode={mode!r}")


def _qmunu_to_munu_local(arr_qmunu: jax.Array, *, mesh_xy: Mesh) -> jax.Array:
    """Transpose ``(nkx,nky,nkz,mu,nu) -> (mu,nu,nkx,nky,nkz)`` and keep mu/nu tiled."""

    mu_shard = _mu_nu_sharding(mesh_xy)
    with mesh_xy:
        arr_munu = jnp.asarray(arr_qmunu).transpose(3, 4, 0, 1, 2)
        return jax.lax.with_sharding_constraint(arr_munu, mu_shard)

_sigma_tau_channel_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
_sigma_channel_pipeline_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}


def _summarize_hermitian_qmunu(name: str, W_q: jax.Array, print_fn=print) -> None:
    """Emit Hermiticity diagnostics for W_q with layout (kx,ky,kz,1,mu,1,nu)."""
    if W_q.ndim != 7:
        print_fn(f"[{name}] unexpected shape {W_q.shape}; expected (nkx,nky,nkz,1,mu,1,nu)")
        return
    W_munu = jnp.asarray(W_q[:, :, :, 0, :, 0, :], dtype=jnp.complex128)
    herm_resid = _to_host_scalar(
        jnp.max(jnp.abs(W_munu - jnp.swapaxes(jnp.conj(W_munu), -2, -1))),
        float,
    )
    diag = jnp.diagonal(W_munu, axis1=-2, axis2=-1)
    diag_im = _to_host_scalar(jnp.max(jnp.abs(jnp.imag(diag))), float)
    diag_real = jnp.real(diag)
    diag_min = _to_host_scalar(jnp.min(diag_real), float)
    diag_max = _to_host_scalar(jnp.max(diag_real), float)
    print_fn(
        f"  [{name}] hermitian residual={herm_resid:.3e} "
        f"max|Im diag|={diag_im:.3e} diag range=[{diag_min:.4e}, {diag_max:.4e}]"
    )


def _project_sigma_channels(
    coeff_re: jax.Array,
    coeff_im: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Apply one window projection to precomputed real/imag sigma channels."""

    def _full(_):
        sigma_tau_kij_full = sigma_tau_kij_re[None, ...] + 1j * sigma_tau_kij_im[None, ...]
        return (coeff_re + 1j * coeff_im) * sigma_tau_kij_full

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
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    get_sigma_kij_channels_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> Callable[..., jax.Array]:
    """Return a jit-compatible sigma-channel pipeline with device-local FFTs."""

    pipeline_key = (
        id(mesh_xy),
        int(nkx),
        int(nky),
        int(nkz),
        int(nk_tot),
        bool(bispinor),
        id(get_G_mu_nu_fn),
        id(get_sigma_kij_channels_fn),
    )
    if pipeline_key in _sigma_channel_pipeline_cache:
        return _sigma_channel_pipeline_cache[pipeline_key]

    from common.fft_helpers import (
        make_jittable_local_fftn_3d,
        make_jittable_local_ifftn_3d,
    )

    w_isdf._ensure_compilation_cache()
    mu_shard = _mu_nu_sharding(mesh_xy)
    sigma_7d_shard = NamedSharding(mesh_xy, P(None, "x", None, "y", None, None, None))
    sigma_7d_spec = P(None, "x", None, "y", None, None, None)
    local_ifftn_7d = make_jittable_local_ifftn_3d(
        mesh_xy, sigma_7d_spec, sigma_7d_spec, norm="ortho", axes=(-3, -2, -1)
    )
    local_fftn_7d = make_jittable_local_fftn_3d(
        mesh_xy, sigma_7d_spec, sigma_7d_spec, norm="ortho", axes=(-3, -2, -1)
    )
    nk_scale = jnp.asarray(-1.0 / np.sqrt(float(nk_tot)), dtype=jnp.float64)

    @jax.jit
    def _build_G_7d(
        psi_coh_rmuT_X: jax.Array,
        psi_coh_rmu_Y: jax.Array,
        Gij: jax.Array,
    ) -> jax.Array:
        G_k = get_G_mu_nu_fn(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij)
        G_7d = G_k.transpose(1, 2, 3, 4, 0).reshape(
            G_k.shape[1], G_k.shape[2], G_k.shape[3], G_k.shape[4], nkx, nky, nkz
        )
        return jax.lax.with_sharding_constraint(G_7d, sigma_7d_shard)

    @jax.jit
    def _expand_W_7d(W_q: jax.Array) -> jax.Array:
        W_q = jax.lax.with_sharding_constraint(W_q, mu_shard)
        W_7d = W_q[None, :, None, :, :, :, :]
        return jax.lax.with_sharding_constraint(W_7d, sigma_7d_shard)

    @jax.jit
    def _build_sigma_R(G_R: jax.Array, W_R: jax.Array) -> jax.Array:
        if bispinor:
            gamma0_diag = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
            gamma0_vertex = gamma0_diag[:, None] * gamma0_diag[None, :]
            gamma0_vertex = gamma0_vertex[:, None, :, None, None, None, None]
            G_R = G_R * gamma0_vertex
        sigma_R = G_R * W_R * nk_scale
        return jax.lax.with_sharding_constraint(sigma_R, sigma_7d_shard)

    @jax.jit
    def _project_channels(
        psi_proj_rmu_X: jax.Array,
        psi_proj_rmuT_Y: jax.Array,
        sigma_k: jax.Array,
    ) -> jax.Array:
        return get_sigma_kij_channels_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_k)

    @jax.jit
    def _sigma_channel_pipeline(
        psi_coh_rmuT_X: jax.Array,
        psi_coh_rmu_Y: jax.Array,
        psi_proj_rmu_X: jax.Array,
        psi_proj_rmuT_Y: jax.Array,
        Gij: jax.Array,
        W_q: jax.Array,
    ) -> jax.Array:
        G_7d = _build_G_7d(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij)
        W_7d = _expand_W_7d(W_q)
        G_R = local_ifftn_7d(G_7d)
        W_R = local_ifftn_7d(W_7d)
        sigma_R = _build_sigma_R(G_R, W_R)
        sigma_k = local_fftn_7d(sigma_R)
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
    get_G_mu_nu_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    get_G_R_fn: Callable[[jax.Array, int, int, int], jax.Array],
    get_sigma_mu_nu_fn: Callable[[jax.Array, jax.Array, int, bool], jax.Array],
    get_sigma_kij_channels_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> Callable[..., jax.Array]:
    """Return a cached tau-node sigma builder with jittable local FFTs."""

    cache_key = (
        id(mesh_xy),
        int(nkx),
        int(nky),
        int(nkz),
        int(nk_tot),
        bool(bispinor),
        id(get_G_mu_nu_fn),
        id(get_sigma_kij_channels_fn),
    )
    if cache_key in _sigma_tau_channel_kernel_cache:
        return _sigma_tau_channel_kernel_cache[cache_key]

    # Intentionally unused here: the tau-node path reconstructs the sigma FFT
    # pipeline directly from get_G_mu_nu / projection helpers.
    del get_G_R_fn, get_sigma_mu_nu_fn

    w_isdf._ensure_compilation_cache()
    mu_shard = _mu_nu_sharding(mesh_xy)
    sigma_channel_pipeline = _get_sigma_channel_pipeline(
        mesh_xy=mesh_xy,
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        nk_tot=nk_tot,
        bispinor=bispinor,
        get_G_mu_nu_fn=get_G_mu_nu_fn,
        get_sigma_kij_channels_fn=get_sigma_kij_channels_fn,
    )

    @jax.jit
    def _build_tau_operands(
        E_A: jax.Array,
        mask_A: jax.Array,
        B_mu_nu: jax.Array,
        Omega_mu_nu: jax.Array,
        mask_B: jax.Array,
        E_ref_A: jax.Array,
        E_ref_B: jax.Array,
        t_node: jax.Array,
        eye_nb: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        phase_A = jnp.exp(-1j * (E_A - E_ref_A) * t_node)
        weights_kn = jnp.where(mask_A, phase_A, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        Gij = eye_nb[None, :, :] * weights_kn[:, :, None]

        phase_B = jnp.exp(-1j * (Omega_mu_nu - E_ref_B) * t_node)
        W_t_q = jnp.where(mask_B, B_mu_nu * phase_B, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        W_t_q = jax.lax.with_sharding_constraint(W_t_q, mu_shard)
        return Gij, W_t_q

    @jax.jit
    def _tau_channel_step(
        psi_coh_rmuT_X: jax.Array,
        psi_coh_rmu_Y: jax.Array,
        psi_proj_rmu_X: jax.Array,
        psi_proj_rmuT_Y: jax.Array,
        E_A: jax.Array,
        mask_A: jax.Array,
        B_mu_nu: jax.Array,
        Omega_mu_nu: jax.Array,
        mask_B: jax.Array,
        E_ref_A: jax.Array,
        E_ref_B: jax.Array,
        t_node: jax.Array,
        eye_nb: jax.Array,
    ) -> jax.Array:
        Gij, W_t_q = _build_tau_operands(
            E_A,
            mask_A,
            B_mu_nu,
            Omega_mu_nu,
            mask_B,
            E_ref_A,
            E_ref_B,
            t_node,
            eye_nb,
        )
        return sigma_channel_pipeline(
            psi_coh_rmuT_X,
            psi_coh_rmu_Y,
            psi_proj_rmu_X,
            psi_proj_rmuT_Y,
            Gij,
            W_t_q,
        )

    _sigma_tau_channel_kernel_cache[cache_key] = _tau_channel_step
    return _tau_channel_step


def compute_w0_wiwp_and_ppm_from_minimax(
    V_qmunu: jax.Array,
    wf_bundle,
    meta,
    mesh_xy: Mesh,
    *,
    minimax_config: MinimaxConfig | None = None,
    omega_p_ry: float = 2.0,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    use_shipped_minimax_tables: bool = True,
    minimax_energy_reference: str | float | int | None = "midgap",
    minimax_energy_reference_fn: Callable[[jax.Array, jax.Array], float] | None = None,
    fallback_omega: float = 2.0,
    print0=print,
) -> PPMBuildResult:
    """Build GN-PPM parameters from minimax W(0) and W(i*omega_p)."""
    import time as _ppm_time

    if minimax_config is not None:
        target_error = float(minimax_config.target_error)
        max_nodes = int(minimax_config.max_nodes)
        use_shipped_minimax_tables = bool(minimax_config.use_shipped_tables)
        if minimax_energy_reference_fn is None:
            minimax_energy_reference = minimax_config.energy_reference

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
        minimax_config=minimax_config,
        target_error=target_error,
        max_nodes=max_nodes,
        use_shipped_tables=bool(use_shipped_minimax_tables),
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

    _t_chi0 = _ppm_time.perf_counter()
    chi0_q = w_isdf.compute_chi0_minimax(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, quad, meta, mesh_xy,
        energy_reference=e_ref,
    )
    chi0_q.block_until_ready()
    print0(f"  [TIMING-PPM] chi(0): {_ppm_time.perf_counter() - _t_chi0:.3f}s")
    _t_chii = _ppm_time.perf_counter()
    chii_q = w_isdf.compute_chi0_minimax(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, quad_imag, meta, mesh_xy,
        energy_reference=e_ref,
    )
    chii_q.block_until_ready()
    print0(f"  [TIMING-PPM] chi(iωp): {_ppm_time.perf_counter() - _t_chii:.3f}s")
    _t_w0 = _ppm_time.perf_counter()
    W0_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chi0_q, meta, mesh_xy)
    W0_q.block_until_ready()
    print0(f"  [TIMING-PPM] W(0): {_ppm_time.perf_counter() - _t_w0:.3f}s")
    _t_wi = _ppm_time.perf_counter()
    Wiwp_q = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chii_q, meta, mesh_xy)
    Wiwp_q.block_until_ready()
    print0(f"  [TIMING-PPM] W(iωp): {_ppm_time.perf_counter() - _t_wi:.3f}s")

    nkx, nky, nkz = W0_q.shape[0], W0_q.shape[1], W0_q.shape[2]
    n_rmu = W0_q.shape[4]
    V_q = jnp.asarray(V_qmunu)[0, 0, 0].reshape(nkx, nky, nkz, 1, n_rmu, 1, n_rmu)

    _summarize_hermitian_qmunu("W(0)", W0_q, print0)
    _summarize_hermitian_qmunu(f"W(iωp={omega_p_ry:.3f} Ry)", Wiwp_q, print0)

    # Build W^c = W(with W_head) - V(with V_head).
    Wc0_q = W0_q - V_q
    Wci_q = Wiwp_q - V_q


    Wc0_qmunu = Wc0_q[:, :, :, 0, :, 0, :]
    Wci_qmunu = Wci_q[:, :, :, 0, :, 0, :]
    _t_fit = _ppm_time.perf_counter()
    omega_qmunu, b_qmunu, valid_qmunu, unfulfilled = fit_gn_ppm_from_wc_pair(
        Wc0_qmunu,
        Wci_qmunu,
        1j * complex(omega_p_ry),
        fallback_omega=float(fallback_omega),
    )
    print0(f"  [TIMING-PPM] GN fit: {_ppm_time.perf_counter() - _t_fit:.3f}s")

    # GN poles are for W^c, so W^c(t) is a direct exponential sum.
    _t_layout = _ppm_time.perf_counter()
    Omega = _qmunu_to_munu_local(omega_qmunu, mesh_xy=mesh_xy)
    B = _qmunu_to_munu_local(b_qmunu, mesh_xy=mesh_xy)
    valid_mask = _qmunu_to_munu_local(valid_qmunu, mesh_xy=mesh_xy)
    Wc0_mu_nu = _qmunu_to_munu_local(Wc0_qmunu, mesh_xy=mesh_xy)
    print0(f"  [TIMING-PPM] q->mu,nu layout: {_ppm_time.perf_counter() - _t_layout:.3f}s")

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

def _build_single_sigma_window(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_total: int,
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
        x_min,
        x_max,
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
            mask_B_total=int(mask_B_total),
            mask_B_count=int(mask_B_count),
            mask_B_min=float(mask_B_min),
            mask_B_max=float(mask_B_max),
            x_min=float(x_min),
            x_max=float(x_max),
        )
    ]


def _build_three_sigma_windows(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_total: int,
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
        S_min = float(np.min(A_vals) + B_min)
        S_max = float(np.max(A_vals) + B_max)
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
            # Explicit sign in front of core contribution.
            prefactor = -1.0
        else:
            x_min = max(S_min - (T - z_edge), z_edge, 1.0e-12)
            x_max = max(S_max, x_min * (1.0 + 1.0e-9))
            q = solve_laplace_minimax_interval(
                x_min,
                x_max,
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
                mask_B_total=int(mask_B_total),
                mask_B_count=int(count_B),
                mask_B_min=float(B_min),
                mask_B_max=float(B_max),
                x_min=float(x_min) if name != "core" else None,
                x_max=float(x_max) if name != "core" else None,
                crossing_A=float(A_core) if name == "core" else None,
                crossing_kind="hgl" if name == "core" else None,
                t_cut=float(T),
                z_edge=float(z_edge),
            )
        )
    return windows

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
    use_shipped_minimax_tables: bool = True,
) -> tuple[jax.Array, list[_SigmaWindow]]:
    """Convolve sigma in one pass, accumulating directly in (k,i,j) space.

    The exact reduced-storage path contracts each complex τ-node contribution
    to two band-space channels [K[Re X_tau], K[Im X_tau]] and then mixes those
    channels with scalar frequency coefficients. This preserves the required
    per-window Re/Im projection without storing Σ_k(μ,ν,ω).
    """
    import time as _btime
    _bprof = bool(os.environ.get("LORRAX_PROFILE_PPM", ""))
    _b0 = _btime.perf_counter()
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])
    if n_omega == 0:
        nk_proj = int(psi_proj_rmu_X.shape[0])
        nb_proj = int(psi_proj_rmu_X.shape[1])
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    E_A_host = _to_host_np(E_A, dtype=np.float64, tiled=False)
    base_A_host = _to_host_np(base_mask_A, dtype=bool, tiled=False)
    if _bprof:
        _b1 = _btime.perf_counter()
        print0(f"  [TIMING-BRANCH] {log_tag} device_get pre-loop: {_b1-_b0:.3f}s")

    mask_B_total, mask_B_all_count, mask_B_all_min, mask_B_all_max = _masked_stats_device(
        Omega_mu_nu, base_mask_B
    )
    if kernel_sign == +1 and float(np.max(omega_nonneg_ry)) > 1.0e-14:
        omega_max = float(np.max(omega_nonneg_ry))
        xi = max(float(regularization_width_ry), 1.0e-12)
        z_edge = float(edge_factor) * xi
        T = omega_max + z_edge
        le_mask_B = base_mask_B & (Omega_mu_nu <= T)
        gt_mask_B = base_mask_B & (Omega_mu_nu > T)
        _, mask_B_le_count, mask_B_le_min, mask_B_le_max = _masked_stats_device(Omega_mu_nu, le_mask_B)
        _, mask_B_gt_count, mask_B_gt_min, mask_B_gt_max = _masked_stats_device(Omega_mu_nu, gt_mask_B)
        windows = _build_three_sigma_windows(
            E_A=E_A_host,
            base_mask_A=base_A_host,
            mask_B_total=mask_B_total,
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
            mask_B_total=mask_B_total,
            mask_B_count=mask_B_all_count,
            mask_B_min=mask_B_all_min,
            mask_B_max=mask_B_all_max,
            omega_nonneg_ry=omega_nonneg_ry,
            kernel_sign=kernel_sign,
            target_error=target_error,
            max_nodes=max_nodes,
            use_shipped_tables=bool(use_shipped_minimax_tables),
        )

    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])
    if not windows:
        return jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows

    if _bprof:
        n_nodes_total = int(sum(int(win.alpha.shape[0]) for win in windows))
        print0(
            f"  [TIMING-BRANCH] {log_tag} windows built: "
            f"count={len(windows)}, total_nodes={n_nodes_total}, "
            f"kernel_sign={kernel_sign:+d}, axis={axis_kind or 'unknown'}"
        )

    if debug_quadrature:
        nsamp = max(50, int(debug_quadrature_samples))
        for win in windows:
            mask_A = np.asarray(win.mask_A, dtype=bool)
            count_A = int(np.sum(mask_A))
            total_A = int(mask_A.size)
            count_B = int(win.mask_B_count or 0)
            total_B = int(win.mask_B_total or mask_B_total)
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
            B_min = win.mask_B_min
            B_max = win.mask_B_max
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
                target = _G_hgl(u_grid) if win.crossing_kind == "hgl" else _G_fermi(u_grid)
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
    tau_channel_step = _get_sigma_tau_channel_kernel(
        mesh_xy=mesh_xy,
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        nk_tot=nk_tot,
        bispinor=bispinor,
        get_G_mu_nu_fn=get_G_mu_nu_fn,
        get_G_R_fn=get_G_R_fn,
        get_sigma_mu_nu_fn=get_sigma_mu_nu_fn,
        get_sigma_kij_channels_fn=get_sigma_kij_channels_fn,
    )
    acc_total = None
    if stream_writer is None:
        acc_total = jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128)

    import time as _time
    _prof_enabled = bool(os.environ.get("LORRAX_PROFILE_PPM", ""))
    _prof_t = {"tau_channel": 0.0, "omega_acc": 0.0, "total": 0.0}
    _prof_n = 0
    _prof_total_start = _time.perf_counter()
    _first_tau_traced = False
    _first_tau_logged = False
    _first_omega_logged = False

    branch_label = log_tag if log_tag else f"kernel_sign={kernel_sign:+d}"
    with jax_profile.annotation(f"sigma_branch[{branch_label}]"):
        for win_idx, win in enumerate(windows):
            with jax_profile.step_annotation(
                "sigma_window",
                step_num=win_idx,
                detail=f"{branch_label}:{win.name}:n{int(win.alpha.shape[0])}",
            ):
                mask_A = jnp.asarray(win.mask_A)
                mask_B = _materialize_window_mask_B(
                    win,
                    base_mask_B=base_mask_B,
                    Omega_mu_nu=Omega_mu_nu,
                )
                E_ref_A_j = jnp.asarray(win.E_ref_A, dtype=jnp.float64)
                E_ref_B_j = jnp.asarray(win.E_ref_B, dtype=jnp.float64)
                project_code = {"full": 0, "imag": 1}.get(win.project, 2)
                project_code_j = jnp.asarray(project_code, dtype=jnp.int32)
                acc_win = None
                if acc_total is not None:
                    acc_win = jnp.zeros_like(acc_total)

                for tau_idx, (t_node, alpha_node) in enumerate(zip(win.t_nodes, win.alpha)):
                    if _prof_enabled:
                        _t0 = _time.perf_counter()
                        if not _first_tau_logged:
                            print0(
                                f"  [TIMING-BRANCH] {log_tag} first tau start: "
                                f"window={win.name}, tau_idx={tau_idx}, "
                                f"nodes={int(win.alpha.shape[0])}, project={win.project}"
                            )
                            _first_tau_logged = True
                    t_node_j = jnp.asarray(t_node, dtype=jnp.complex128)
                    if _prof_enabled and not _first_tau_traced:
                        with jax_profile.trace_section("ppm_sigma_first_tau"):
                            with mesh_xy:
                                sigma_tau_kij_ri = tau_channel_step(
                                    psi_coh_rmuT_X,
                                    psi_coh_rmu_Y,
                                    psi_proj_rmu_X,
                                    psi_proj_rmuT_Y,
                                    E_A,
                                    mask_A,
                                    B_mu_nu,
                                    Omega_mu_nu,
                                    mask_B,
                                    E_ref_A_j,
                                    E_ref_B_j,
                                    t_node_j,
                                    eye_nb,
                                )
                        _first_tau_traced = True
                    else:
                        with mesh_xy:
                            sigma_tau_kij_ri = tau_channel_step(
                                psi_coh_rmuT_X,
                                psi_coh_rmu_Y,
                                psi_proj_rmu_X,
                                psi_proj_rmuT_Y,
                                E_A,
                                mask_A,
                                B_mu_nu,
                                Omega_mu_nu,
                                mask_B,
                                E_ref_A_j,
                                E_ref_B_j,
                                t_node_j,
                                eye_nb,
                            )
                    sigma_tau_kij_re = sigma_tau_kij_ri[0]
                    sigma_tau_kij_im = sigma_tau_kij_ri[1]
                    if _prof_enabled:
                        sigma_tau_kij_re.block_until_ready()
                        sigma_tau_kij_im.block_until_ready()
                        _t1 = _time.perf_counter()
                        _prof_t["tau_channel"] += _t1 - _t0
                        _t4 = _t1
                        if _prof_n == 0:
                            print0(
                                f"  [TIMING-BRANCH] {log_tag} first tau returned: "
                                f"{_t1 - _t0:.3f}s"
                            )
                    if debug_quadrature:
                        re_mag = float(jnp.max(jnp.abs(sigma_tau_kij_re)))
                        im_mag = float(jnp.max(jnp.abs(sigma_tau_kij_im)))
                        print0(f"  [diag] {log_tag}{win.name}: |K[Re(σ)]|={re_mag:.4e}  |K[Im(σ)]|={im_mag:.4e}")
                    alpha_eff = complex(alpha_node) * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_node)
                    omega_sign = float(win.omega_sign) * float(omega_sign_flip)
                    alpha_eff_j = jnp.asarray(alpha_eff, dtype=jnp.complex128)
                    omega_sign_j = jnp.asarray(omega_sign, dtype=jnp.float64)
                    pref = jnp.asarray(win.prefactor * scale, dtype=jnp.float64)
                    if acc_win is not None:
                        if _prof_enabled and not _first_omega_logged:
                            print0(
                                f"  [TIMING-BRANCH] {log_tag} first omega_acc start: "
                                f"window={win.name}, tau_idx={tau_idx}, Nω={n_omega}"
                            )
                        acc_win = _accumulate_window_channels_jit(
                            acc_win,
                            sigma_tau_kij_re,
                            sigma_tau_kij_im,
                            omega_vec,
                            t_node_j,
                            alpha_eff_j,
                            omega_sign_j,
                            pref,
                            project_code_j,
                        )
                        if _prof_enabled:
                            acc_win.block_until_ready()
                            _t5 = _time.perf_counter()
                            _prof_t["omega_acc"] += _t5 - _t4
                            if not _first_omega_logged:
                                print0(
                                    f"  [TIMING-BRANCH] {log_tag} first omega_acc returned: "
                                    f"{_t5 - _t4:.3f}s"
                                )
                                _first_omega_logged = True
                    _prof_n += 1
                    if acc_win is None:
                        # Stream in omega chunks without repeating τ-node work.
                        for ibeg in range(0, n_omega, int(max(1, omega_batch_size))):
                            iend = min(ibeg + int(max(1, omega_batch_size)), n_omega)
                            idx = omega_global_idx[ibeg:iend]
                            omega_batch = omega_vec[ibeg:iend]
                            batch_proj = _project_stream_batch_jit(
                                omega_batch,
                                sigma_tau_kij_re,
                                sigma_tau_kij_im,
                                t_node_j,
                                alpha_eff_j,
                                omega_sign_j,
                                pref,
                                project_code_j,
                            )
                            stream_writer(idx, batch_proj)

                if acc_win is not None:
                    acc_total = acc_total + acc_win

    _prof_total_end = _time.perf_counter()
    _prof_t["total"] = _prof_total_end - _prof_total_start
    if _prof_enabled and _prof_n > 0:
        print0(f"  [PROFILE] {log_tag} tau-node loop: {_prof_n} iterations, {_prof_t['total']:.3f}s total")
        print0(f"    tau_channel: {_prof_t['tau_channel']:8.3f}s ({100*_prof_t['tau_channel']/_prof_t['total']:5.1f}%)  avg {_prof_t['tau_channel']/_prof_n:.3f}s/iter")
        print0(f"    omega_acc: {_prof_t['omega_acc']:8.3f}s ({100*_prof_t['omega_acc']/_prof_t['total']:5.1f}%)  avg {_prof_t['omega_acc']/_prof_n:.3f}s/iter")
        _overhead = _prof_t['total'] - sum(v for k, v in _prof_t.items() if k != 'total')
        print0(f"    overhead:  {_overhead:8.3f}s ({100*_overhead/_prof_t['total']:5.1f}%)")

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
    quadrature_config: SigmaQuadratureConfig | None = None,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    use_shipped_minimax_tables: bool = True,
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
    ppm_static_cohsex_check: bool = False,
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
    if None in (get_G_mu_nu_fn, get_sigma_kij_channels_fn):
        raise ValueError("get_G_mu_nu_fn and get_sigma_kij_channels_fn must be provided.")
    if quadrature_config is not None:
        target_error = float(quadrature_config.target_error)
        max_nodes = int(quadrature_config.max_nodes)
        crossing_max_nodes = int(quadrature_config.crossing_max_nodes)
        crossing_eps_q = float(quadrature_config.crossing_eps_q)
        use_shipped_minimax_tables = bool(quadrature_config.use_shipped_tables)
    import time as _sgtime
    _sg_t0 = _sgtime.perf_counter()
    _sg_prof = bool(os.environ.get("LORRAX_PROFILE_PPM", ""))
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

    if ppm_static_cohsex_check:
        # Force static COH limit: E_A = 0 for all bands, omega = [0].
        # With B/Omega = -Wc0/2, the result should exactly reproduce static COH:
        #   Sigma_COH = -0.5 * (1/Nk) Sigma_q I_{k-q} * Wc0_q
        print0("  *** PPM STATIC COHSEX CHECK: zeroing E_cond, H_val, omega=[0] ***")
        E_cond = jnp.zeros_like(E_cond)
        H_val = jnp.zeros_like(H_val)
        omega_req = np.array([0.0], dtype=np.float64)
        omega_rel_req = omega_req
        idx_pos = np.array([0], dtype=np.intp)
        idx_neg = np.array([], dtype=np.intp)
        omega_pos = np.array([0.0], dtype=np.float64)
        omega_neg_abs = np.array([], dtype=np.float64)
    invalid_mode = str(invalid_mode).strip().lower()
    if invalid_mode not in ("fixed_2ry", "static_limit"):
        raise ValueError("invalid_mode must be 'fixed_2ry' or 'static_limit'.")
    if valid_mask_mu_nu is None:
        valid_mask = jnp.ones_like(B_mask, dtype=bool)
    else:
        valid_mask = jnp.asarray(valid_mask_mu_nu, dtype=bool)
    invalid_mask = B_mask & (~valid_mask)
    n_total_modes = int(np.asarray(jax.device_get(jnp.sum(B_mask, dtype=jnp.int64)), dtype=np.int64))
    n_invalid = int(np.asarray(jax.device_get(jnp.sum(invalid_mask, dtype=jnp.int64)), dtype=np.int64))
    if invalid_mode == "static_limit":
        B_mask = B_mask & valid_mask

    omega_step_ev = float(omega_req[1] - omega_req[0]) * 13.6056980659 if omega_req.size > 1 else 0.0
    print0(
        "  Σc(ω) grid: "
        f"{float(np.min(omega_req) * 13.6056980659):.3f} .. {float(np.max(omega_req) * 13.6056980659):.3f} eV, "
        f"Nω={omega_req.size}, Δω={omega_step_ev:.3f} eV, ξ={float(regularization_width_ry * 13.6056980659):.3f} eV"
    )
    if print0 is not None:
        if _sg_prof:
            _sg_diag0 = _sgtime.perf_counter()
            print0("  [TIMING-SIG] post-grid diag: reducing occupied / unoccupied windows on device")
        if bool(np.asarray(jax.device_get(jnp.any(unocc_mask)), dtype=bool)):
            ec_min = float(np.asarray(jax.device_get(jnp.min(jnp.where(unocc_mask, enk_full, jnp.inf))), dtype=np.float64))
            ec_max = float(np.asarray(jax.device_get(jnp.max(jnp.where(unocc_mask, enk_full, -jnp.inf))), dtype=np.float64))
        else:
            ec_min = float("nan")
            ec_max = float("nan")
        if bool(np.asarray(jax.device_get(jnp.any(occ_mask)), dtype=bool)):
            ev_min = float(np.asarray(jax.device_get(jnp.min(jnp.where(occ_mask, enk_full, jnp.inf))), dtype=np.float64))
            ev_max = float(np.asarray(jax.device_get(jnp.max(jnp.where(occ_mask, enk_full, -jnp.inf))), dtype=np.float64))
        else:
            ev_min = float("nan")
            ev_max = float("nan")
        if _sg_prof:
            _sg_diag1 = _sgtime.perf_counter()
            print0(f"  [TIMING-SIG] post-grid diag energy summary built in {_sg_diag1 - _sg_diag0:.3f}s")
            print0("  [TIMING-SIG] post-grid diag: reducing omega abs on device")
        if n_total_modes:
            omega_min_dev = jnp.min(jnp.where(B_mask, Omega_abs, jnp.inf))
            omega_max_dev = jnp.max(jnp.where(B_mask, Omega_abs, -jnp.inf))
            om_min = float(np.asarray(jax.device_get(omega_min_dev), dtype=np.float64))
            om_max = float(np.asarray(jax.device_get(omega_max_dev), dtype=np.float64))
        else:
            om_min = float("nan")
            om_max = float("nan")
        if _sg_prof:
            _sg_diag4 = _sgtime.perf_counter()
            print0(f"  [TIMING-SIG] post-grid diag omega summary built in {_sg_diag4 - _sg_diag1:.3f}s")
        if _sg_prof:
            _sg_diag6 = _sgtime.perf_counter()
            print0(f"  [TIMING-SIG] post-grid diag summary built in {_sg_diag6 - _sg_diag4:.3f}s")
        print0(
            "  Σc(ω) axes (vacuum): "
            f"Ev=[{ev_min:.6e},{ev_max:.6e}] Ry, "
            f"Ec=[{ec_min:.6e},{ec_max:.6e}] Ry, "
            f"Omega=[{om_min:.6e},{om_max:.6e}] Ry"
        )
        if n_invalid:
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


    if _sg_prof:
        _sg_t_setup = _sgtime.perf_counter()
        print0(f"  [TIMING-SIG] setup: {_sg_t_setup - _sg_t0:.1f}s")

    try:
        if use_kij_accum or use_kij_stream:
            sigma_channel_pipeline = _get_sigma_channel_pipeline(
                mesh_xy=mesh_xy,
                nkx=nkx,
                nky=nky,
                nkz=nkz,
                nk_tot=nk_tot,
                bispinor=bispinor,
                get_G_mu_nu_fn=get_G_mu_nu_fn,
                get_sigma_kij_channels_fn=get_sigma_kij_channels_fn,
            )

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
                if _sg_prof:
                    print0(
                        f"  [TIMING-SIG] entering _convolve_kij[{tag}]: "
                        f"Nω={int(omega_eval.size)}, cond_sign={kernel_cond:+d}, val_sign={kernel_val:+d}"
                    )
                    _sg_t_cond0 = _sgtime.perf_counter()
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
                    use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
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
                if _sg_prof:
                    _sg_t_cond1 = _sgtime.perf_counter()
                    print0(
                        f"  [TIMING-SIG] _convolve_kij[{tag}] cond branch returned: "
                        f"{_sg_t_cond1 - _sg_t_cond0:.3f}s"
                    )
                    _sg_t_val0 = _sgtime.perf_counter()
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
                    use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
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
                if _sg_prof:
                    _sg_t_val1 = _sgtime.perf_counter()
                    print0(
                        f"  [TIMING-SIG] _convolve_kij[{tag}] val branch returned: "
                        f"{_sg_t_val1 - _sg_t_val0:.3f}s"
                    )
                if use_kij_stream:
                    return None
                if debug_split_contrib:
                    return sigma_cond, sigma_val
                return sigma_cond + sigma_val
            if _sg_prof:
                _sg_t_conv_start = _sgtime.perf_counter()
                print0(f"  [TIMING-SIG] pre-convolution: {_sg_t_conv_start - _sg_t0:.1f}s")
            # ω_rel >= 0: Σ^- gets crossing treatment, Σ^+ sign-definite.
            if _sg_prof: _sg_ts = _sgtime.perf_counter()
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
                if _sg_prof:
                    _sg_te = _sgtime.perf_counter()
                    print0(f"  [TIMING-SIG] convolve_kij(pos) returned: {_sg_te - _sg_ts:.1f}s")
                if sigma_pos is not None:
                    if debug_split_contrib:
                        sigma_cond, sigma_val = sigma_pos
                        sigma_kij_host[idx_pos] = np.asarray(jax.device_get(sigma_cond + sigma_val), dtype=np.complex128)
                        if _sg_prof: print0(f"  [TIMING-SIG] device_get(pos kij): {_sgtime.perf_counter() - _sg_te:.1f}s")
                        sigma_minus_host[idx_pos] = np.asarray(jax.device_get(sigma_cond), dtype=np.complex128)
                        sigma_plus_host[idx_pos] = np.asarray(jax.device_get(sigma_val), dtype=np.complex128)
                        if _sg_prof: print0(f"  [TIMING-SIG] device_get(pos split): {_sgtime.perf_counter() - _sg_te:.1f}s")
                    else:
                        sigma_kij_host[idx_pos] = np.asarray(jax.device_get(sigma_pos), dtype=np.complex128)

            if _sg_prof: _sg_ts2 = _sgtime.perf_counter()
            # ω_rel < 0: evaluate with |ω_rel| and swapped branch kernels.
            # Keep omega kernel signs unchanged (omega_sign_flip=+1). For w=|ω_rel|:
            #   D_minus = ω_rel - S_c = -(w + S_c)
            #   D_plus  = ω_rel + S_v = -(w - S_v)
            # In this implementation, kernel_sign=+1 realizes 1/(w-S) and
            # kernel_sign=-1 realizes 1/(w+S), so both routed branches require
            # the same extra global minus.
            if omega_neg_abs.size:
                sigma_neg = _convolve_kij(
                    omega_neg_abs,
                    idx_neg,
                    kernel_cond=-1,
                    kernel_val=+1,
                    tag="ω<E_F",
                    # Optional debug-only sign flip knob on the ω<E_F block.
                    scale=(-float(sigma_scale) if not sigma_flip_neg else float(sigma_scale)),
                    omega_sign_flip_cond=1,
                    omega_sign_flip_val=1,
                )
                if _sg_prof:
                    _sg_te2 = _sgtime.perf_counter()
                    print0(f"  [TIMING-SIG] convolve_kij(neg) returned: {_sg_te2 - _sg_ts2:.1f}s")
                if sigma_neg is not None:
                    if debug_split_contrib:
                        sigma_cond, sigma_val = sigma_neg
                        sigma_kij_host[idx_neg] = np.asarray(jax.device_get(sigma_cond + sigma_val), dtype=np.complex128)
                        if _sg_prof: print0(f"  [TIMING-SIG] device_get(neg kij): {_sgtime.perf_counter() - _sg_te2:.1f}s")
                        sigma_minus_host[idx_neg] = np.asarray(jax.device_get(sigma_cond), dtype=np.complex128)
                        sigma_plus_host[idx_neg] = np.asarray(jax.device_get(sigma_val), dtype=np.complex128)
                        if _sg_prof: print0(f"  [TIMING-SIG] device_get(neg split): {_sgtime.perf_counter() - _sg_te2:.1f}s")
                    else:
                        sigma_kij_host[idx_neg] = np.asarray(jax.device_get(sigma_neg), dtype=np.complex128)

            if _sg_prof:
                _sg_t_conv_end = _sgtime.perf_counter()
                print0(f"  [TIMING-SIG] convolution total: {_sg_t_conv_end - _sg_t_conv_start:.1f}s")
            if invalid_mode == "static_limit" and Wc0_mu_nu is not None and n_invalid > 0:
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
                    sigma_occ_kij = sigma_channel_pipeline(
                        psi_coh_rmuT_X,
                        psi_coh_rmu_Y,
                        psi_proj_rmu_X,
                        psi_proj_rmuT_Y,
                        Gij_occ,
                        Wc0_invalid,
                    )[0]
                    sigma_ri_kij = sigma_channel_pipeline(
                        psi_coh_rmuT_X,
                        psi_coh_rmu_Y,
                        psi_proj_rmu_X,
                        psi_proj_rmuT_Y,
                        jnp.broadcast_to(
                            jnp.eye(int(occ_mask.shape[1]), dtype=jnp.complex128)[None, :, :],
                            (nk, int(occ_mask.shape[1]), int(occ_mask.shape[1])),
                        ),
                        Wc0_invalid,
                    )[0]

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
