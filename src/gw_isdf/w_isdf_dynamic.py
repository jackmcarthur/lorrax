"""
Dynamic W(ω) routines with per-window JIT compilation.

These routines use dynamic window slicing and compile separate kernels
per (max_val_len, max_cond_len) pair.

For the streamlined static χ₀/W pipeline, see w_isdf.py.
"""
import os
from pathlib import Path
import math
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.experimental import compilation_cache as jax_compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import numpy as np
from scipy.special import roots_laguerre

from common import Meta, jax_profile

if TYPE_CHECKING:
    from .wavefunction_bundle import WavefunctionBundle


# ============================================================================
# Mesh registry for sharding specs (shared with w_isdf.py)
# ============================================================================

_MESH_SHARD_REGISTRY: dict[int | str, dict[str, NamedSharding | None]] = {}
_COMPILATION_CACHE_READY = False
_ENERGY_TOL = 1e-6


def _ensure_compilation_cache():
    global _COMPILATION_CACHE_READY
    if _COMPILATION_CACHE_READY:
        return
    cache_dir = os.environ.get("ISDF_JAX_CACHE_DIR")
    if cache_dir is None:
        base_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base_cache, "isdf_jax_compilation")
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        jax_compilation_cache.set_cache_dir(cache_path)
        _COMPILATION_CACHE_READY = True
    except Exception:
        _COMPILATION_CACHE_READY = True


def _register_mesh_shardings(mesh_xy: Mesh | None) -> str | int:
    """Ensure shardings for a mesh are cached and return the registry key."""
    if mesh_xy is None:
        key = "none"
        if key not in _MESH_SHARD_REGISTRY:
            _MESH_SHARD_REGISTRY[key] = {
                "xt": None, "y_shard": None, "x_shard": None, "yt": None,
                "y": None, "out": None, "chiR": None, "gv": None, "gc": None,
            }
        return key

    key = id(mesh_xy)
    if key not in _MESH_SHARD_REGISTRY:
        _MESH_SHARD_REGISTRY[key] = {
            "xt": NamedSharding(mesh_xy, P(None, None, 'x', None)),
            "y_shard": NamedSharding(mesh_xy, P(None, None, None, 'y')),
            "x_shard": NamedSharding(mesh_xy, P(None, None, None, 'x')),
            "yt": NamedSharding(mesh_xy, P(None, None, 'y', None)),
            "out": NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y')),
            "chiR": NamedSharding(mesh_xy, P('x', 'y', None, None, None)),
            "gv": NamedSharding(mesh_xy, P(None, None, 'x', None, 'y')),
            "gc": NamedSharding(mesh_xy, P(None, None, 'y', None, 'x')),
        }
    return key


# ============================================================================
# Window slicing utilities
# ============================================================================

def _band_slice_metadata(
    energies: np.ndarray,
    emin: float,
    emax: float,
    upper_inclusive: bool = True,
    lower_relaxed: bool = False,
):
    """Return per-k slice metadata covering all bands within [emin, emax]."""
    if energies is None or energies.size == 0:
        nk = 0 if energies is None else energies.shape[0]
        zero = np.zeros(nk, dtype=np.int32)
        mask = np.zeros((nk, 1), dtype=bool)
        return zero, zero, mask, 0
    lower_bound = emin - (_ENERGY_TOL if lower_relaxed else 0.0)
    if upper_inclusive:
        upper_bound = emax + _ENERGY_TOL
        mask = (energies >= lower_bound) & (energies <= upper_bound)
    else:
        mask = (energies >= lower_bound) & (energies < emax)
    nk, nb = mask.shape
    starts = np.zeros(nk, dtype=np.int32)
    spans = np.zeros(nk, dtype=np.int32)
    slice_masks: list[np.ndarray] = []
    max_span = 0
    for k in range(nk):
        idx = np.nonzero(mask[k])[0]
        if idx.size == 0:
            slice_masks.append(np.zeros(0, dtype=bool))
            continue
        start = int(idx[0])
        end = int(idx[-1])
        span = end - start + 1
        starts[k] = start
        spans[k] = span
        slice_masks.append(mask[k, start:start + span])
        if span > max_span:
            max_span = span
    if max_span == 0:
        clipped_start = np.zeros_like(starts)
        return clipped_start, spans, np.zeros((nk, 1), dtype=bool), 0

    slice_cap = max(nb - max_span, 0)
    clipped_start = np.minimum(starts, slice_cap).astype(np.int32)
    offsets = (starts - clipped_start).astype(np.int32)
    mask_arr = np.zeros((nk, max_span), dtype=bool)
    for k, sl in enumerate(slice_masks):
        if sl.size:
            off = offsets[k]
            mask_arr[k, off:off + sl.size] = sl
    return clipped_start, spans, mask_arr, max_span


def _ensure_window_band_ranges(win, enk_v_host: np.ndarray, enk_c_host: np.ndarray):
    """Populate per-window band metadata for efficient slicing."""
    if getattr(win, "_has_band_ranges", False):
        return
    val_upper = getattr(win.val_window, "upper_inclusive", True)
    val_lower_relaxed = getattr(win.val_window, "index", 0) == 0
    val_start, val_len, val_mask, max_val_len = _band_slice_metadata(
        enk_v_host, win.val_window.start_energy, win.val_window.end_energy,
        upper_inclusive=val_upper, lower_relaxed=val_lower_relaxed,
    )
    cond_upper = getattr(win.cond_window, "upper_inclusive", True)
    cond_lower_relaxed = getattr(win.cond_window, "index", 0) == 0
    cond_start, cond_len, cond_mask, max_cond_len = _band_slice_metadata(
        enk_c_host, win.cond_window.start_energy, win.cond_window.end_energy,
        upper_inclusive=cond_upper, lower_relaxed=cond_lower_relaxed,
    )
    win.val_band_start = val_start.astype(np.int32)
    win.val_band_len = val_len.astype(np.int32)
    win.val_band_mask = val_mask
    win.cond_band_start = cond_start.astype(np.int32)
    win.cond_band_len = cond_len.astype(np.int32)
    win.cond_band_mask = cond_mask
    win.max_val_len = int(max_val_len)
    win.max_cond_len = int(max_cond_len)
    win._has_band_ranges = True


def _slice_along_axis(arr: jax.Array, starts: jax.Array, max_len: int, axis: int):
    """Slice each k-row of arr along the given axis using per-row start indices."""
    if max_len <= 0:
        raise ValueError("max_len must be positive for slicing")
    if axis == 0:
        raise ValueError("Axis 0 (k-dimension) is not sliceable via _slice_along_axis")
    axis_row = axis - 1
    max_len = int(max_len)
    axis_size = arr.shape[axis]
    if max_len > axis_size:
        raise ValueError(f"Requested slice length {max_len} exceeds axis {axis} size {axis_size}")
    max_start_allowed = max(axis_size - max_len, 0)
    starts = jnp.clip(starts, 0, max_start_allowed)

    def _slice_one(row, start_idx):
        return jax.lax.dynamic_slice_in_dim(row, start_idx, max_len, axis=axis_row)

    return jax.vmap(_slice_one, in_axes=(0, 0), out_axes=0)(arr, starts)


def _as_int_array(data: np.ndarray | jax.Array | None, nk: int) -> jax.Array:
    if data is None:
        return jnp.zeros((nk,), dtype=jnp.int32)
    return jnp.asarray(data, dtype=jnp.int32)


def _gl_quadrature_for_interval(E_gap_eff: float, E_bw_eff: float, epsq: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build GL quadrature for a positive denominator interval [E_gap_eff, E_bw_eff].

    Uses the CTSP Appendix-A sizing relation with the branch-local gap/bandwidth.
    """
    tiny = 1e-12
    E_gap_eff = max(float(E_gap_eff), tiny)
    E_bw_eff = max(float(E_bw_eff), E_gap_eff + tiny)
    alpha = math.sqrt(E_bw_eff / E_gap_eff)
    n_tau = int(np.ceil(alpha * (0.4 - 0.3 * np.log(epsq))))
    n_tau = max(3, min(300, n_tau))
    tau_i, w_i = roots_laguerre(n_tau)
    z_lm = 1.0 / math.sqrt(E_gap_eff * E_bw_eff)
    return tau_i, w_i, z_lm


# ============================================================================
# Window-specific chi kernel (causes JIT recompilation per window shape)
# ============================================================================

@lru_cache(maxsize=None)
def _get_chi_kernel_windowed(mesh_key, nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int):
    """Window-sliced chi kernel - compiles per (max_val_len, max_cond_len) pair."""
    shards = _MESH_SHARD_REGISTRY[mesh_key]
    xt_shard = shards["xt"]
    y_shard = shards["y_shard"]
    x_shard = shards["x_shard"]
    yt_shard = shards["yt"]
    out_shard = shards["out"]
    chiR_shard = shards["chiR"]
    gv_shard = shards["gv"]
    gc_shard = shards["gc"]

    @partial(
        jax.jit,
        static_argnames=("nkx", "nky", "nkz", "max_val_len", "max_cond_len"),
        in_shardings=(
            xt_shard, y_shard, x_shard, yt_shard,
            None, None,
            None, None, None, None, None, None, None,
            None, None, None, None, None, None,
        ),
        out_shardings=out_shard,
    )
    def _compute(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        vmin, vmax, cmin, cmax, tau_i, z_lm, w_i,
        val_start, val_len, val_mask, cond_start, cond_len, cond_mask,
        nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int,
    ) -> jax.Array:
        nrmu_local = psi_vTX.shape[2]
        if max_val_len == 0 or max_cond_len == 0 or tau_i.shape[0] == 0:
            chi_empty = jnp.zeros((nkx, nky, nkz, 1, nrmu_local, 1, nrmu_local), dtype=jnp.complex128)
            chi_empty = jax.lax.with_sharding_constraint(chi_empty, out_shard)
            return chi_empty

        psi_vTX_win = _slice_along_axis(psi_vTX, val_start, max_val_len, axis=3)
        psi_vY_win = _slice_along_axis(psi_vY, val_start, max_val_len, axis=1)
        psi_cX_win = _slice_along_axis(psi_cX, cond_start, max_cond_len, axis=1)
        psi_cTY_win = _slice_along_axis(psi_cTY, cond_start, max_cond_len, axis=3)
        enk_v_win = _slice_along_axis(enk_v, val_start, max_val_len, axis=1)
        enk_c_win = _slice_along_axis(enk_c, cond_start, max_cond_len, axis=1)

        val_mask_f = val_mask.astype(psi_vTX_win.dtype)
        cond_mask_f = cond_mask.astype(psi_cX_win.dtype)
        psi_vTX_win = psi_vTX_win * val_mask_f[:, None, None, :]
        psi_vY_win = psi_vY_win * val_mask_f[:, :, None, None]
        psi_cX_win = psi_cX_win * cond_mask_f[:, :, None, None]
        psi_cTY_win = psi_cTY_win * cond_mask_f[:, None, None, :]
        val_mask_complex = val_mask.astype(jnp.complex128)
        cond_mask_complex = cond_mask.astype(jnp.complex128)

        quad_w = -2.0 * z_lm * w_i * jnp.exp(-(z_lm * (cmin - vmax) - 1.0) * tau_i)

        def _k_to_R(g_k: jax.Array, flip_sign: bool) -> jax.Array:
            g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            return jax.lax.cond(
                flip_sign,
                lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho'),
                lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho'),
                g_fft,
            )

        def tau_body(itau, chi_R_acc):
            tau = tau_i[itau]
            exp_v = jnp.exp(-z_lm * tau * (vmax - enk_v_win)) * val_mask_complex
            exp_c = jnp.exp(-z_lm * tau * (enk_c_win - cmin)) * cond_mask_complex
            w_v = exp_v.astype(jnp.complex128)
            w_c = exp_c.astype(jnp.complex128)
            Gv_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_vTX_win), w_v, psi_vY_win, optimize=True)
            Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cTY_win), w_c, psi_cX_win, optimize=True)
            Gv_k = jax.lax.with_sharding_constraint(Gv_k, gv_shard)
            Gc_k = jax.lax.with_sharding_constraint(Gc_k, gc_shard)
            Gv_R = _k_to_R(Gv_k, flip_sign=False)
            Gc_R = _k_to_R(Gc_k, flip_sign=True)
            chi_tau = jnp.einsum('ambnxyz, bnamxyz-> mnxyz', Gc_R, Gv_R, optimize=True)
            return chi_R_acc + quad_w[itau] * chi_tau

        chi_R = jnp.zeros((nrmu_local, nrmu_local, nkx, nky, nkz), dtype=jnp.complex128)
        chi_R = jax.lax.with_sharding_constraint(chi_R, chiR_shard)
        chi_R = jax.lax.fori_loop(0, tau_i.shape[0], tau_body, chi_R)
        chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm='ortho')
        chi_q = chi_q.transpose(2, 3, 4, 0, 1)
        chi_q = chi_q[:, :, :, None, :, None, :]
        chi_q = jax.lax.with_sharding_constraint(chi_q, out_shard)
        return chi_q

    return _compute


# ============================================================================
# HGL kernel for dynamic χ(ω) with energy crossings
# ============================================================================

@lru_cache(maxsize=None)
def _get_chi_hgl_kernel(mesh_key, nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int):
    """HGL kernel for χ(ω) in crossing regions."""
    shards = _MESH_SHARD_REGISTRY[mesh_key]
    xt_shard = shards["xt"]
    y_shard = shards["y_shard"]
    x_shard = shards["x_shard"]
    yt_shard = shards["yt"]
    out_shard = shards["out"]
    chiR_shard = shards["chiR"]
    gv_shard = shards["gv"]
    gc_shard = shards["gc"]

    @partial(
        jax.jit,
        static_argnames=("nkx", "nky", "nkz", "max_val_len", "max_cond_len"),
        in_shardings=(
            xt_shard, y_shard, x_shard, yt_shard,
            None, None, None, None,
            None, None, None, None, None, None,
        ),
        out_shardings=(out_shard, out_shard),
    )
    def _compute_hgl_tile(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        tau, gamma,
        val_start, val_len, val_mask, cond_start, cond_len, cond_mask,
        nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int,
    ) -> tuple[jax.Array, jax.Array]:
        nrmu_local = psi_vTX.shape[2]
        if max_val_len == 0 or max_cond_len == 0:
            zero = jnp.zeros((nkx, nky, nkz, 1, nrmu_local, 1, nrmu_local), dtype=jnp.float64)
            zero = jax.lax.with_sharding_constraint(zero, out_shard)
            return zero, zero

        psi_vTX_win = _slice_along_axis(psi_vTX, val_start, max_val_len, axis=3)
        psi_vY_win = _slice_along_axis(psi_vY, val_start, max_val_len, axis=1)
        psi_cX_win = _slice_along_axis(psi_cX, cond_start, max_cond_len, axis=1)
        psi_cTY_win = _slice_along_axis(psi_cTY, cond_start, max_cond_len, axis=3)
        enk_v_win = _slice_along_axis(enk_v, val_start, max_val_len, axis=1)
        enk_c_win = _slice_along_axis(enk_c, cond_start, max_cond_len, axis=1)

        val_mask_f = val_mask.astype(psi_vTX_win.dtype)
        cond_mask_f = cond_mask.astype(psi_cX_win.dtype)
        psi_vTX_win = psi_vTX_win * val_mask_f[:, None, None, :]
        psi_vY_win = psi_vY_win * val_mask_f[:, :, None, None]
        psi_cX_win = psi_cX_win * cond_mask_f[:, :, None, None]
        psi_cTY_win = psi_cTY_win * cond_mask_f[:, None, None, :]
        val_mask_complex = val_mask.astype(jnp.complex128)
        cond_mask_complex = cond_mask.astype(jnp.complex128)

        def _k_to_R(g_k: jax.Array, flip_sign: bool) -> jax.Array:
            g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            return jax.lax.cond(
                flip_sign,
                lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho'),
                lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho'),
                g_fft,
            )

        phase_v = jnp.exp(1j * gamma * tau * enk_v_win) * val_mask_complex
        phase_c = jnp.exp(1j * gamma * tau * enk_c_win) * cond_mask_complex
        Gv_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_vTX_win), phase_v, psi_vY_win, optimize=True)
        Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cTY_win), phase_c, psi_cX_win, optimize=True)
        Gv_k = jax.lax.with_sharding_constraint(Gv_k, gv_shard)
        Gc_k = jax.lax.with_sharding_constraint(Gc_k, gc_shard)
        Gv_R = _k_to_R(Gv_k, flip_sign=False)
        Gc_mR = _k_to_R(Gc_k, flip_sign=True)
        product_R = jnp.einsum('ambnxyz, bnamxyz-> mnxyz', jnp.conj(Gc_mR), Gv_R, optimize=True)
        P_plus_R = product_R.real
        P_cross_R = -product_R.imag
        P_plus_q = jnp.fft.fftn(P_plus_R, axes=(-3, -2, -1), norm='ortho')
        P_cross_q = jnp.fft.fftn(P_cross_R, axes=(-3, -2, -1), norm='ortho')
        P_plus_q = P_plus_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
        P_cross_q = P_cross_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
        P_plus_q = jax.lax.with_sharding_constraint(P_plus_q, out_shard)
        P_cross_q = jax.lax.with_sharding_constraint(P_cross_q, out_shard)
        return P_plus_q, P_cross_q

    return _compute_hgl_tile


# ============================================================================
# GL kernel for χ(ω) (non-crossing)
# ============================================================================

@lru_cache(maxsize=None)
def _get_chi_omega_gl_kernel(mesh_key, nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int):
    """GL kernel for χ(ω) with standard (vmax/cmin) propagator shifts."""
    shards = _MESH_SHARD_REGISTRY[mesh_key]
    xt_shard = shards["xt"]
    y_shard = shards["y_shard"]
    x_shard = shards["x_shard"]
    yt_shard = shards["yt"]
    out_shard = shards["out"]
    chiR_shard = shards["chiR"]
    gv_shard = shards["gv"]
    gc_shard = shards["gc"]

    @partial(
        jax.jit,
        static_argnames=("nkx", "nky", "nkz", "max_val_len", "max_cond_len"),
        in_shardings=(
            xt_shard, y_shard, x_shard, yt_shard,
            None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None,
        ),
        out_shardings=out_shard,
    )
    def _compute(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        vmin, vmax, cmin, cmax, tau_i, z_lm, w_i, omega,
        omega_shift_sign, overall_sign,
        val_start, val_len, val_mask, cond_start, cond_len, cond_mask,
        nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int,
    ) -> jax.Array:
        nrmu_local = psi_vTX.shape[2]
        if max_val_len == 0 or max_cond_len == 0 or tau_i.shape[0] == 0:
            chi_empty = jnp.zeros((nkx, nky, nkz, 1, nrmu_local, 1, nrmu_local), dtype=jnp.complex128)
            chi_empty = jax.lax.with_sharding_constraint(chi_empty, out_shard)
            return chi_empty

        psi_vTX_win = _slice_along_axis(psi_vTX, val_start, max_val_len, axis=3)
        psi_vY_win = _slice_along_axis(psi_vY, val_start, max_val_len, axis=1)
        psi_cX_win = _slice_along_axis(psi_cX, cond_start, max_cond_len, axis=1)
        psi_cTY_win = _slice_along_axis(psi_cTY, cond_start, max_cond_len, axis=3)
        enk_v_win = _slice_along_axis(enk_v, val_start, max_val_len, axis=1)
        enk_c_win = _slice_along_axis(enk_c, cond_start, max_cond_len, axis=1)

        val_mask_f = val_mask.astype(psi_vTX_win.dtype)
        cond_mask_f = cond_mask.astype(psi_cX_win.dtype)
        psi_vTX_win = psi_vTX_win * val_mask_f[:, None, None, :]
        psi_vY_win = psi_vY_win * val_mask_f[:, :, None, None]
        psi_cX_win = psi_cX_win * cond_mask_f[:, :, None, None]
        psi_cTY_win = psi_cTY_win * cond_mask_f[:, None, None, :]
        val_mask_complex = val_mask.astype(jnp.complex128)
        cond_mask_complex = cond_mask.astype(jnp.complex128)

        gamma = z_lm
        E_gap = cmin - vmax
        E_gap_eff = E_gap + omega_shift_sign * omega
        quad_w = overall_sign * gamma * w_i * jnp.exp(-(gamma * E_gap_eff - 1.0) * tau_i)

        def _k_to_R(g_k: jax.Array, flip_sign: bool) -> jax.Array:
            g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            return jax.lax.cond(
                flip_sign,
                lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho'),
                lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho'),
                g_fft,
            )

        def tau_body(itau, chi_R_acc):
            tau = tau_i[itau]
            exp_v = jnp.exp(-gamma * tau * (vmax - enk_v_win)) * val_mask_complex
            exp_c = jnp.exp(-gamma * tau * (enk_c_win - cmin)) * cond_mask_complex
            w_v = exp_v.astype(jnp.complex128)
            w_c = exp_c.astype(jnp.complex128)
            Gv_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_vTX_win), w_v, psi_vY_win, optimize=True)
            Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cTY_win), w_c, psi_cX_win, optimize=True)
            Gv_k = jax.lax.with_sharding_constraint(Gv_k, gv_shard)
            Gc_k = jax.lax.with_sharding_constraint(Gc_k, gc_shard)
            Gv_R = _k_to_R(Gv_k, flip_sign=False)
            Gc_R = _k_to_R(Gc_k, flip_sign=True)
            chi_tau = jnp.einsum('ambnxyz, bnamxyz-> mnxyz', Gc_R, Gv_R, optimize=True)
            return chi_R_acc + quad_w[itau] * chi_tau

        chi_R = jnp.zeros((nrmu_local, nrmu_local, nkx, nky, nkz), dtype=jnp.complex128)
        chi_R = jax.lax.with_sharding_constraint(chi_R, chiR_shard)
        chi_R = jax.lax.fori_loop(0, tau_i.shape[0], tau_body, chi_R)
        chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm='ortho')
        chi_q = chi_q.transpose(2, 3, 4, 0, 1)
        chi_q = chi_q[:, :, :, None, :, None, :]
        chi_q = jax.lax.with_sharding_constraint(chi_q, out_shard)
        return chi_q

    return _compute


# ============================================================================
# GL kernel for χ(ω) with reversed (vmin/cmax) propagator shifts
# ============================================================================

@lru_cache(maxsize=None)
def _get_chi_omega_gl_reverse_kernel(mesh_key, nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int):
    """GL kernel for resonant χ(ω) when ω > E_bw (positive denominator)."""
    shards = _MESH_SHARD_REGISTRY[mesh_key]
    xt_shard = shards["xt"]
    y_shard = shards["y_shard"]
    x_shard = shards["x_shard"]
    yt_shard = shards["yt"]
    out_shard = shards["out"]
    chiR_shard = shards["chiR"]
    gv_shard = shards["gv"]
    gc_shard = shards["gc"]

    @partial(
        jax.jit,
        static_argnames=("nkx", "nky", "nkz", "max_val_len", "max_cond_len"),
        in_shardings=(
            xt_shard, y_shard, x_shard, yt_shard,
            None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None,
        ),
        out_shardings=out_shard,
    )
    def _compute(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        vmin, vmax, cmin, cmax, tau_i, z_lm, w_i, omega,
        val_start, val_len, val_mask, cond_start, cond_len, cond_mask,
        nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int,
    ) -> jax.Array:
        nrmu_local = psi_vTX.shape[2]
        if max_val_len == 0 or max_cond_len == 0 or tau_i.shape[0] == 0:
            chi_empty = jnp.zeros((nkx, nky, nkz, 1, nrmu_local, 1, nrmu_local), dtype=jnp.complex128)
            chi_empty = jax.lax.with_sharding_constraint(chi_empty, out_shard)
            return chi_empty

        psi_vTX_win = _slice_along_axis(psi_vTX, val_start, max_val_len, axis=3)
        psi_vY_win = _slice_along_axis(psi_vY, val_start, max_val_len, axis=1)
        psi_cX_win = _slice_along_axis(psi_cX, cond_start, max_cond_len, axis=1)
        psi_cTY_win = _slice_along_axis(psi_cTY, cond_start, max_cond_len, axis=3)
        enk_v_win = _slice_along_axis(enk_v, val_start, max_val_len, axis=1)
        enk_c_win = _slice_along_axis(enk_c, cond_start, max_cond_len, axis=1)

        val_mask_f = val_mask.astype(psi_vTX_win.dtype)
        cond_mask_f = cond_mask.astype(psi_cX_win.dtype)
        psi_vTX_win = psi_vTX_win * val_mask_f[:, None, None, :]
        psi_vY_win = psi_vY_win * val_mask_f[:, :, None, None]
        psi_cX_win = psi_cX_win * cond_mask_f[:, :, None, None]
        psi_cTY_win = psi_cTY_win * cond_mask_f[:, None, None, :]
        val_mask_complex = val_mask.astype(jnp.complex128)
        cond_mask_complex = cond_mask.astype(jnp.complex128)

        gamma = z_lm
        E_bw = cmax - vmin
        quad_w = gamma * w_i * jnp.exp(-(gamma * (omega - E_bw) - 1.0) * tau_i)

        def _k_to_R(g_k: jax.Array, flip_sign: bool) -> jax.Array:
            g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            return jax.lax.cond(
                flip_sign,
                lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho'),
                lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho'),
                g_fft,
            )

        def tau_body(itau, chi_R_acc):
            tau = tau_i[itau]
            exp_v = jnp.exp(-gamma * tau * (enk_v_win - vmin)) * val_mask_complex
            exp_c = jnp.exp(-gamma * tau * (cmax - enk_c_win)) * cond_mask_complex
            w_v = exp_v.astype(jnp.complex128)
            w_c = exp_c.astype(jnp.complex128)
            Gv_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_vTX_win), w_v, psi_vY_win, optimize=True)
            Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cTY_win), w_c, psi_cX_win, optimize=True)
            Gv_k = jax.lax.with_sharding_constraint(Gv_k, gv_shard)
            Gc_k = jax.lax.with_sharding_constraint(Gc_k, gc_shard)
            Gv_R = _k_to_R(Gv_k, flip_sign=False)
            Gc_R = _k_to_R(Gc_k, flip_sign=True)
            chi_tau = jnp.einsum('ambnxyz, bnamxyz-> mnxyz', Gc_R, Gv_R, optimize=True)
            return chi_R_acc + quad_w[itau] * chi_tau

        chi_R = jnp.zeros((nrmu_local, nrmu_local, nkx, nky, nkz), dtype=jnp.complex128)
        chi_R = jax.lax.with_sharding_constraint(chi_R, chiR_shard)
        chi_R = jax.lax.fori_loop(0, tau_i.shape[0], tau_body, chi_R)
        chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm='ortho')
        chi_q = chi_q.transpose(2, 3, 4, 0, 1)
        chi_q = chi_q[:, :, :, None, :, None, :]
        chi_q = jax.lax.with_sharding_constraint(chi_q, out_shard)
        return chi_q

    return _compute


# ============================================================================
# Window-specific chi computation wrappers
# ============================================================================

def get_chi_lm_Yt_jax_windowed(
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    win, meta: Meta, mesh_xy: Mesh | None = None,
):
    """Compute chi_lm for a specific window (causes JIT recompilation per window shape)."""
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nrmu = psi_vTX.shape[2]
    nk = psi_vTX.shape[0]
    max_val_len = int(getattr(win, "max_val_len", 0))
    max_cond_len = int(getattr(win, "max_cond_len", 0))
    val_band_start = _as_int_array(getattr(win, "val_band_start", None), nk)
    val_band_len = _as_int_array(getattr(win, "val_band_len", None), nk)
    val_band_mask = jnp.asarray(getattr(win, "val_band_mask", None), dtype=jnp.bool_)
    cond_band_start = _as_int_array(getattr(win, "cond_band_start", None), nk)
    cond_band_len = _as_int_array(getattr(win, "cond_band_len", None), nk)
    cond_band_mask = jnp.asarray(getattr(win, "cond_band_mask", None), dtype=jnp.bool_)

    vmin = jnp.asarray(win.val_window.start_energy)
    vmax = jnp.asarray(win.val_window.end_energy)
    cmin = jnp.asarray(win.cond_window.start_energy)
    cmax = jnp.asarray(win.cond_window.end_energy)
    tau_i = jnp.asarray(win.tau_i, dtype=jnp.float64)
    z_lm = jnp.asarray(win.z_lm, dtype=jnp.complex128)
    w_i = jnp.asarray(win.w_i, dtype=jnp.complex128)

    _ensure_compilation_cache()
    mesh_key = _register_mesh_shardings(mesh_xy)
    kernel = _get_chi_kernel_windowed(mesh_key, nkx, nky, nkz, max_val_len, max_cond_len)

    with jax_profile.annotation(f"chi0_kernel_legacy[v{max_val_len}_c{max_cond_len}]"):
        chi_q = kernel(
            psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
            vmin, vmax, cmin, cmax, tau_i, z_lm, w_i,
            val_band_start, val_band_len, val_band_mask,
            cond_band_start, cond_band_len, cond_band_mask,
            nkx, nky, nkz, max_val_len, max_cond_len,
        )
    chi_q = jax.lax.with_sharding_constraint(chi_q, NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y')))
    return chi_q


def get_chi0_jax_windowed(
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, meta: Meta, mesh_xy: Mesh | None = None,
):
    """Sum chi over windows using window-specific kernels (legacy interface)."""
    enk_v_host = None
    enk_c_host = None
    for win in windows:
        if getattr(win, "_has_band_ranges", False):
            continue
        if enk_v_host is None:
            enk_v_host = np.asarray(jax.device_get(enk_v))
        if enk_c_host is None:
            enk_c_host = np.asarray(jax.device_get(enk_c))
        _ensure_window_band_ranges(win, enk_v_host, enk_c_host)

    chi_sum = None
    for w_i, win in enumerate(windows):
        step_detail = f"v{win.max_val_len}_c{win.max_cond_len}"
        with jax_profile.step_annotation("chi0_window_legacy", step_num=w_i, detail=step_detail):
            chi_win = get_chi_lm_Yt_jax_windowed(psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c, win, meta, mesh_xy)
            chi_sum = chi_win if chi_sum is None else (chi_sum + chi_win)
    return chi_sum


def _compute_chi_omega_hgl(
    psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
    win, omega: float, meta: Meta, mesh_xy: Mesh,
) -> jax.Array:
    """Compute χ(ω) for one window using HGL quadrature (crossing region)."""
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nrmu = psi_vTX.shape[2]
    
    gamma = jnp.asarray(win.z_lm, dtype=jnp.float64)
    tau_hgl = jnp.asarray(win.tau_hgl, dtype=jnp.float64)
    w_hgl = jnp.asarray(win.w_hgl, dtype=jnp.float64)
    omega_jax = jnp.asarray(abs(omega), dtype=jnp.float64)
    
    _ensure_compilation_cache()
    mesh_key = _register_mesh_shardings(mesh_xy)
    hgl_kernel = _get_chi_hgl_kernel(mesh_key, nkx, nky, nkz, win.max_val_len, win.max_cond_len)
    out_shard = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    
    val_band_start = jnp.asarray(win.val_band_start, dtype=jnp.int32)
    val_band_len = jnp.asarray(win.val_band_len, dtype=jnp.int32)
    val_band_mask = jnp.asarray(win.val_band_mask, dtype=jnp.bool_)
    cond_band_start = jnp.asarray(win.cond_band_start, dtype=jnp.int32)
    cond_band_len = jnp.asarray(win.cond_band_len, dtype=jnp.int32)
    cond_band_mask = jnp.asarray(win.cond_band_mask, dtype=jnp.bool_)
    
    chi_sum = jnp.zeros((nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)
    
    for u in range(win.ntau_hgl):
        tau_u = tau_hgl[u]
        w_u = w_hgl[u]
        
        P_plus_q, P_cross_q = hgl_kernel(
            psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
            tau_u, gamma,
            val_band_start, val_band_len, val_band_mask,
            cond_band_start, cond_band_len, cond_band_mask,
            nkx, nky, nkz, win.max_val_len, win.max_cond_len,
        )
        
        phase = gamma * tau_u * omega_jax
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)
        chi_tau = -gamma * w_u * (cos_phase * P_cross_q - sin_phase * P_plus_q)
        chi_sum = chi_sum + chi_tau
    
    return jax.lax.with_sharding_constraint(chi_sum, out_shard)


def _compute_chi_omega_gl(
    psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
    win, omega: float, meta: Meta, mesh_xy: Mesh, omega_shift_sign: float, overall_sign: float,
    tau_i: np.ndarray, w_i: np.ndarray, z_lm: float,
) -> jax.Array:
    """Compute one GL branch contribution for χ(ω) in standard (vmax/cmin) form."""
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    
    tau_i = jnp.asarray(tau_i, dtype=jnp.float64)
    z_lm = jnp.asarray(z_lm, dtype=jnp.float64)
    w_i = jnp.asarray(w_i, dtype=jnp.float64)
    omega_jax = jnp.asarray(omega, dtype=jnp.float64)
    
    val_band_start = jnp.asarray(win.val_band_start, dtype=jnp.int32)
    val_band_len = jnp.asarray(win.val_band_len, dtype=jnp.int32)
    val_band_mask = jnp.asarray(win.val_band_mask, dtype=jnp.bool_)
    cond_band_start = jnp.asarray(win.cond_band_start, dtype=jnp.int32)
    cond_band_len = jnp.asarray(win.cond_band_len, dtype=jnp.int32)
    cond_band_mask = jnp.asarray(win.cond_band_mask, dtype=jnp.bool_)
    
    _ensure_compilation_cache()
    mesh_key = _register_mesh_shardings(mesh_xy)
    kernel = _get_chi_omega_gl_kernel(mesh_key, nkx, nky, nkz, win.max_val_len, win.max_cond_len)
    
    chi_q = kernel(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        jnp.asarray(win.val_window.start_energy, dtype=jnp.float64),
        jnp.asarray(win.val_window.end_energy, dtype=jnp.float64),
        jnp.asarray(win.cond_window.start_energy, dtype=jnp.float64),
        jnp.asarray(win.cond_window.end_energy, dtype=jnp.float64),
        tau_i, z_lm, w_i, omega_jax,
        jnp.asarray(omega_shift_sign, dtype=jnp.float64),
        jnp.asarray(overall_sign, dtype=jnp.float64),
        val_band_start, val_band_len, val_band_mask,
        cond_band_start, cond_band_len, cond_band_mask,
        nkx, nky, nkz, win.max_val_len, win.max_cond_len,
    )
    return chi_q


def _compute_chi_omega_gl_reverse(
    psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
    win, omega: float, meta: Meta, mesh_xy: Mesh,
    tau_i: np.ndarray, w_i: np.ndarray, z_lm: float,
) -> jax.Array:
    """Compute resonant GL branch for ω > E_bw using reversed edge shifts."""
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)

    tau_i = jnp.asarray(tau_i, dtype=jnp.float64)
    z_lm = jnp.asarray(z_lm, dtype=jnp.float64)
    w_i = jnp.asarray(w_i, dtype=jnp.float64)
    omega_jax = jnp.asarray(abs(omega), dtype=jnp.float64)

    val_band_start = jnp.asarray(win.val_band_start, dtype=jnp.int32)
    val_band_len = jnp.asarray(win.val_band_len, dtype=jnp.int32)
    val_band_mask = jnp.asarray(win.val_band_mask, dtype=jnp.bool_)
    cond_band_start = jnp.asarray(win.cond_band_start, dtype=jnp.int32)
    cond_band_len = jnp.asarray(win.cond_band_len, dtype=jnp.int32)
    cond_band_mask = jnp.asarray(win.cond_band_mask, dtype=jnp.bool_)

    _ensure_compilation_cache()
    mesh_key = _register_mesh_shardings(mesh_xy)
    kernel = _get_chi_omega_gl_reverse_kernel(mesh_key, nkx, nky, nkz, win.max_val_len, win.max_cond_len)

    chi_q = kernel(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        jnp.asarray(win.val_window.start_energy, dtype=jnp.float64),
        jnp.asarray(win.val_window.end_energy, dtype=jnp.float64),
        jnp.asarray(win.cond_window.start_energy, dtype=jnp.float64),
        jnp.asarray(win.cond_window.end_energy, dtype=jnp.float64),
        tau_i, z_lm, w_i, omega_jax,
        val_band_start, val_band_len, val_band_mask,
        cond_band_start, cond_band_len, cond_band_mask,
        nkx, nky, nkz, win.max_val_len, win.max_cond_len,
    )
    return chi_q


def _compute_chi_omega_antiresonant_gl(
    psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
    win, omega: float, meta: Meta, mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute antiresonant branch: -1 / (ω + ΔE), always GL for ω >= 0.
    """
    omega_abs = abs(omega)
    E_gap = win.cond_window.start_energy - win.val_window.end_energy
    E_bw = win.cond_window.end_energy - win.val_window.start_energy
    tau_i, w_i, z_lm = _gl_quadrature_for_interval(E_gap + omega_abs, E_bw + omega_abs, win.epsq)
    return _compute_chi_omega_gl(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        win, omega, meta, mesh_xy,
        omega_shift_sign=+1.0,
        overall_sign=-1.0,
        tau_i=tau_i,
        w_i=w_i,
        z_lm=z_lm,
    )


def _compute_chi_omega_resonant(
    psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
    win, omega: float, meta: Meta, mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute resonant branch: +1 / (ω - ΔE).

    For scalar ω >= 0:
      - ω < E_gap  : GL with denominator -(ΔE - ω)
      - E_gap ≤ ω ≤ E_bw: HGL regularization
      - ω > E_bw   : GL with denominator +(ω - ΔE)
    """
    omega_abs = abs(omega)
    E_gap = win.cond_window.start_energy - win.val_window.end_energy
    E_bw = win.cond_window.end_energy - win.val_window.start_energy

    if omega_abs < E_gap:
        tau_i, w_i, z_lm = _gl_quadrature_for_interval(E_gap - omega_abs, E_bw - omega_abs, win.epsq)
        return _compute_chi_omega_gl(
            psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
            win, omega_abs, meta, mesh_xy,
            omega_shift_sign=-1.0,
            overall_sign=-1.0,
            tau_i=tau_i,
            w_i=w_i,
            z_lm=z_lm,
        )

    if omega_abs <= E_bw:
        win.init_hgl_quadrature()
        return _compute_chi_omega_hgl(
            psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
            win, omega_abs, meta, mesh_xy,
        )

    tau_i, w_i, z_lm = _gl_quadrature_for_interval(omega_abs - E_bw, omega_abs - E_gap, win.epsq)
    return _compute_chi_omega_gl_reverse(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        win, omega_abs, meta, mesh_xy,
        tau_i=tau_i,
        w_i=w_i,
        z_lm=z_lm,
    )


def frequency_integration(
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, omega: float, meta: Meta, mesh_xy: Mesh,
):
    """
    Scalar-ω CTSP frequency integration:
      χ(ω) = χ_antiresonant^(GL)(ω) + χ_resonant^(GL/HGL)(ω)
    """
    omega_eval = abs(float(omega))
    enk_v_host = None
    enk_c_host = None
    for win in windows:
        if getattr(win, "_has_band_ranges", False):
            continue
        if enk_v_host is None:
            enk_v_host = np.asarray(jax.device_get(enk_v))
        if enk_c_host is None:
            enk_c_host = np.asarray(jax.device_get(enk_c))
        _ensure_window_band_ranges(win, enk_v_host, enk_c_host)

    chi_sum = None

    for win in windows:
        if win.max_val_len == 0 or win.max_cond_len == 0:
            continue

        chi_anti = _compute_chi_omega_antiresonant_gl(
            psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
            win, omega_eval, meta, mesh_xy,
        )
        chi_res = _compute_chi_omega_resonant(
            psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
            win, omega_eval, meta, mesh_xy,
        )
        chi_win = chi_anti + chi_res
        chi_sum = chi_win if chi_sum is None else (chi_sum + chi_win)

    if chi_sum is None:
        nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
        nrmu = psi_vTX.shape[2]
        chi_sum = jnp.zeros((nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)

    if omega < 0:
        return jnp.conj(chi_sum)
    return chi_sum


def get_chi_omega_jax(
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, omega: float, meta: Meta, mesh_xy: Mesh,
):
    """Compatibility wrapper around the scalar-ω frequency integration path."""
    return frequency_integration(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, windows, omega, meta, mesh_xy,
    )


def get_w_omega_jax(
    V_qmunu: jax.Array,
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, omega: float, meta: Meta, mesh_xy: Mesh,
):
    """Compute W(ω) = V / (1 - V χ(ω)) for a single frequency."""
    from . import w_isdf  # Import the main module for W solve
    
    omega_abs = abs(omega)
    enk_v_host = np.asarray(jax.device_get(enk_v))
    enk_c_host = np.asarray(jax.device_get(enk_c))
    
    for win in windows:
        _ensure_window_band_ranges(win, enk_v_host, enk_c_host)
        E_gap = win.cond_window.start_energy - win.val_window.end_energy
        E_bw = win.cond_window.end_energy - win.val_window.start_energy
        if (omega_abs >= E_gap) and (omega_abs <= E_bw):
            win.init_hgl_quadrature()
    
    chi_omega = get_chi_omega_jax(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, windows, omega, meta, mesh_xy
    )
    
    W_omega = w_isdf.solve_w_from_chi_q_jax(V_qmunu, chi_omega, meta, mesh_xy)
    
    return W_omega, chi_omega


def get_chi_omega_jax_from_bundle(
    wf_bundle: "WavefunctionBundle",
    windows,
    omega: float,
    meta: Meta,
    mesh_xy: Mesh,
):
    """Compute dynamic χ(ω) directly from canonical WavefunctionBundle storage."""
    s = wf_bundle.slices
    return get_chi_omega_jax(
        wf_bundle.x(s.v_slice), wf_bundle.y(s.v_slice),
        wf_bundle.y(s.c_slice), wf_bundle.x(s.c_slice),
        wf_bundle.enk[:, s.v_slice], wf_bundle.enk[:, s.c_slice],
        windows, omega, meta, mesh_xy,
    )


def frequency_integration_from_bundle(
    wf_bundle: "WavefunctionBundle",
    windows,
    omega: float,
    meta: Meta,
    mesh_xy: Mesh,
):
    """Bundle-based wrapper for scalar-ω frequency integration."""
    s = wf_bundle.slices
    return frequency_integration(
        wf_bundle.x(s.v_slice), wf_bundle.y(s.v_slice),
        wf_bundle.y(s.c_slice), wf_bundle.x(s.c_slice),
        wf_bundle.enk[:, s.v_slice], wf_bundle.enk[:, s.c_slice],
        windows, omega, meta, mesh_xy,
    )


def get_w_omega_jax_from_bundle(
    V_qmunu: jax.Array,
    wf_bundle: "WavefunctionBundle",
    windows,
    omega: float,
    meta: Meta,
    mesh_xy: Mesh,
):
    """Compute dynamic W(ω) from canonical bundle slices."""
    s = wf_bundle.slices
    return get_w_omega_jax(
        V_qmunu,
        wf_bundle.x(s.v_slice), wf_bundle.y(s.v_slice),
        wf_bundle.y(s.c_slice), wf_bundle.x(s.c_slice),
        wf_bundle.enk[:, s.v_slice], wf_bundle.enk[:, s.c_slice],
        windows, omega, meta, mesh_xy,
    )
