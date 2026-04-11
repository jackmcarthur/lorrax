"""
Static χ₀ and W computation with JAX.

Streamlined pipeline for COHSEX:
- Single universal chi kernel with energy masking (no per-window JIT recompilation)
- Two-stage resharding for W solve following load_wfns pattern
- χ computed with P(..., μ_X, ..., ν_Y) sharding
- V, χ resharded to P(q_XY, μ, ν) for Dyson solve

For legacy dynamic W(ω) with window-specific kernels, see archive/w_isdf_dynamic.py.
"""
import os
import math
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.experimental import compilation_cache as jax_compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import numpy as np

from common import Meta, jax_profile
from .minimax_config import MinimaxConfig
from .minimax_screening import (
    build_static_minimax_window_pair,
    extract_gn_ppm_parameters_from_Wc,
)

if TYPE_CHECKING:
    from .wavefunction_bundle import Wavefunctions


# ============================================================================
# Cache and sharding registry
# ============================================================================

_COMPILATION_CACHE_READY = False
_chi_kernel_cache: dict = {}
_chi_minimax_kernel_cache: dict = {}
_w_solve_cache: dict = {}


def _ensure_compilation_cache():
    """Enable JAX persistent compilation cache."""
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
    except Exception:
        pass
    _COMPILATION_CACHE_READY = True


# ============================================================================
# χ₀ kernel with energy masking (single JIT for all windows)
# ============================================================================

def _get_chi_kernel(mesh_xy: Mesh, nkx: int, nky: int, nkz: int):
    """
    Get or create universal chi kernel for given mesh and k-grid.
    
    Single JIT compilation regardless of window configuration.
    Uses energy masks to select band contributions.
    """
    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_kernel_cache:
        return _chi_kernel_cache[cache_key]
    
    # Define shardings
    psi_XT = NamedSharding(mesh_xy, P(None, None, 'x', None))   # (nk, ns, μ_X, nb)
    psi_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))    # (nk, nb, ns, μ_Y)
    chi_out = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    chi_R = NamedSharding(mesh_xy, P('x', 'y', None, None, None))
    G_v = NamedSharding(mesh_xy, P(None, None, 'x', None, 'y'))  # (nk, ns, μ_X, ns, μ_Y)
    G_c = NamedSharding(mesh_xy, P(None, None, 'y', None, 'x'))
    
    @partial(jax.jit, static_argnames=("nkx", "nky", "nkz"))
    def _chi_kernel(
        psi_val_xn: jax.Array,     # (nk, ns, μ_X, nb_v) valence, bands fast, μ on X
        psi_val_yr: jax.Array,     # (nk, nb_v, ns, μ_Y) valence, centroids fast, μ on Y
        psi_cond_yr: jax.Array,    # (nk, nb_c, ns, μ_Y) conduction, centroids fast, μ on Y
        psi_cond_xn: jax.Array,    # (nk, ns, μ_X, nb_c) conduction, bands fast, μ on X
        enk_v: jax.Array,      # (nk, nb_v)
        enk_c: jax.Array,      # (nk, nb_c)
        val_mask: jax.Array,   # (nk, nb_v) True if band in window
        cond_mask: jax.Array,  # (nk, nb_c) True if band in window
        vmax: jax.Array,       # scalar: upper valence window edge
        cmin: jax.Array,       # scalar: lower conduction window edge
        tau_i: jax.Array,      # (ntau,) quadrature nodes
        z_lm: jax.Array,       # scalar: γ = z_lm
        w_i: jax.Array,        # (ntau,) quadrature weights
        nkx: int, nky: int, nkz: int,
    ) -> jax.Array:
        """Compute static χ₀ with energy masking."""
        n_rmu = psi_val_xn.shape[2]
        ntau = tau_i.shape[0]
        
        # Handle empty case
        def empty_chi():
            chi = jnp.zeros((nkx, nky, nkz, 1, n_rmu, 1, n_rmu), dtype=jnp.complex128)
            return jax.lax.with_sharding_constraint(chi, chi_out)
        
        if ntau == 0:
            return empty_chi()
        
        # Quadrature prefactor: -2γ z_lm w_i exp(-(γ(E_gap) - 1)τ)
        E_gap = cmin - vmax
        quad_w = -2.0 * z_lm * w_i * jnp.exp(-(z_lm * E_gap - 1.0) * tau_i)
        
        def _k_to_R(g_k: jax.Array, flip_sign: bool) -> jax.Array:
            """G(k) → G(±R) via FFT."""
            # g_k: (nk, ns, μ, ns, ν) → reshape to (nkx, nky, nkz, ns, μ, ns, ν)
            g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            return jax.lax.cond(
                flip_sign,
                lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho'),
                lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho'),
                g_fft,
            )
        
        def tau_body(itau, chi_R_acc):
            tau = tau_i[itau]
            
            # Valence: exp(-γτ(E_vmax - E_v)) * mask_v
            # Use jnp.where to avoid computing exp() for masked bands (which could overflow)
            delta_v = vmax - enk_v  # positive for bands in window
            exp_v_raw = jnp.exp(-z_lm * tau * delta_v)
            exp_v = jnp.where(val_mask, exp_v_raw, jnp.zeros_like(exp_v_raw))  # (nk, nb_v) complex
            
            # Conduction: exp(-γτ(E_c - E_cmin)) * mask_c
            delta_c = enk_c - cmin  # positive for bands in window
            exp_c_raw = jnp.exp(-z_lm * tau * delta_c)
            exp_c = jnp.where(cond_mask, exp_c_raw, jnp.zeros_like(exp_c_raw))  # (nk, nb_c) complex
            
            # G_v(k) = Σ_n exp_v[n] |ψ_v^n⟩⟨ψ_v^n|  (xn conj side, yr direct side)
            Gv_k = jnp.einsum('ksxn,kn,knty->ksxty', jnp.conj(psi_val_xn), exp_v.astype(jnp.complex128), psi_val_yr, optimize=True)

            # G_c(k) = Σ_m exp_c[m] |ψ_c^m⟩⟨ψ_c^m|  (xn conj side, yr direct side)
            Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cond_xn), exp_c.astype(jnp.complex128), psi_cond_yr, optimize=True)
            
            # Apply sharding constraints
            Gv_k = jax.lax.with_sharding_constraint(Gv_k, G_v)
            Gc_k = jax.lax.with_sharding_constraint(Gc_k, G_c)
            
            # FFT to R-space
            Gv_R = _k_to_R(Gv_k, flip_sign=False)   # IFFT for valence
            Gc_mR = _k_to_R(Gc_k, flip_sign=True)   # FFT for conduction (gives -R)
            
            # Contract: χ(R) = Σ_{s,s'} conj(G_c)_{-R}^{ss'} · G_v_R^{s's}
            # (a,μ,b,ν,x,y,z) × (b,ν,a,μ,x,y,z) → (μ,ν,x,y,z)
            chi_tau = jnp.einsum('ambnxyz,bnamxyz->mnxyz', Gc_mR, Gv_R, optimize=True)
            
            return chi_R_acc + quad_w[itau] * chi_tau
        
        # Accumulate χ(R) over τ
        chi_R_init = jnp.zeros((n_rmu, n_rmu, nkx, nky, nkz), dtype=jnp.complex128)
        chi_R_init = jax.lax.with_sharding_constraint(chi_R_init, chi_R)
        chi_R_final = jax.lax.fori_loop(0, ntau, tau_body, chi_R_init)
        
        # FFT to q-space: χ(R) → χ(q)
        chi_q = jnp.fft.fftn(chi_R_final, axes=(-3, -2, -1), norm='ortho')
        
        # Reshape: (μ, ν, qx, qy, qz) → (qx, qy, qz, 1, μ, 1, ν)
        chi_q = chi_q.transpose(2, 3, 4, 0, 1)
        chi_q = chi_q[:, :, :, None, :, None, :]
        chi_q = jax.lax.with_sharding_constraint(chi_q, chi_out)
        
        return chi_q
    
    _chi_kernel_cache[cache_key] = _chi_kernel
    return _chi_kernel


def compute_chi0(
    psi_val_xn: jax.Array, psi_val_yr: jax.Array,
    psi_cond_yr: jax.Array, psi_cond_xn: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, meta: Meta, mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute static χ₀(q) by summing over all window pairs.
    
    Uses a single universal kernel with energy masking - no per-window JIT.
    
    Args:
        psi_val_xn:  (nk, ns, μ_X, nb_v) valence, bands fast, μ on X
        psi_val_yr:  (nk, nb_v, ns, μ_Y) valence, centroids fast, μ on Y
        psi_cond_yr: (nk, nb_c, ns, μ_Y) conduction, centroids fast, μ on Y
        psi_cond_xn: (nk, ns, μ_X, nb_c) conduction, bands fast, μ on X
        enk_v: (nk, nb_v) valence energies
        enk_c: (nk, nb_c) conduction energies
        windows: list of WindowPair objects
        meta: Meta with nkx, nky, nkz
        mesh_xy: 2D device mesh
    
    Returns:
        chi_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu) with μ_X, ν_Y sharding
    """
    _ensure_compilation_cache()
    
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nk = psi_val_xn.shape[0]
    nb_v = enk_v.shape[1]
    nb_c = enk_c.shape[1]
    n_rmu = psi_val_xn.shape[2]
    
    # Get energies on host for mask computation
    enk_v_host = np.asarray(jax.device_get(enk_v))
    enk_c_host = np.asarray(jax.device_get(enk_c))
    
    # Get the universal kernel
    kernel = _get_chi_kernel(mesh_xy, nkx, nky, nkz)
    
    # Output sharding
    chi_out = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    chi_sum = jnp.zeros((nkx, nky, nkz, 1, n_rmu, 1, n_rmu), dtype=jnp.complex128)
    chi_sum = jax.lax.with_sharding_constraint(chi_sum, chi_out)
    
    for win in windows:
        # Compute energy masks for this window
        vmin = win.val_window.start_energy
        vmax = win.val_window.end_energy
        cmin = win.cond_window.start_energy
        cmax = win.cond_window.end_energy
        
        # Handle window boundary inclusivity to avoid double-counting
        val_upper_incl = getattr(win.val_window, "upper_inclusive", True)
        val_lower_relaxed = getattr(win.val_window, "index", 0) == 0
        cond_upper_incl = getattr(win.cond_window, "upper_inclusive", True)
        cond_lower_relaxed = getattr(win.cond_window, "index", 0) == 0
        
        # Valence mask
        val_lower = vmin - (1e-6 if val_lower_relaxed else 0.0)
        if val_upper_incl:
            val_mask = (enk_v_host >= val_lower) & (enk_v_host <= vmax + 1e-6)
        else:
            val_mask = (enk_v_host >= val_lower) & (enk_v_host < vmax)
        
        # Conduction mask
        cond_lower = cmin - (1e-6 if cond_lower_relaxed else 0.0)
        if cond_upper_incl:
            cond_mask = (enk_c_host >= cond_lower) & (enk_c_host <= cmax + 1e-6)
        else:
            cond_mask = (enk_c_host >= cond_lower) & (enk_c_host < cmax)
        
        # Skip if no bands in window
        if not val_mask.any() or not cond_mask.any():
            continue
        
        # Convert to JAX arrays
        val_mask_jax = jnp.asarray(val_mask, dtype=jnp.bool_)
        cond_mask_jax = jnp.asarray(cond_mask, dtype=jnp.bool_)
        
        # Get quadrature params
        tau_i = jnp.asarray(win.tau_i, dtype=jnp.float64)
        z_lm = jnp.asarray(win.z_lm, dtype=jnp.complex128)
        w_i = jnp.asarray(win.w_i, dtype=jnp.complex128)
        vmax_j = jnp.asarray(vmax, dtype=jnp.float64)
        cmin_j = jnp.asarray(cmin, dtype=jnp.float64)
        
        with jax_profile.annotation(f"chi0_window"):
            chi_win = kernel(
                psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                enk_v, enk_c,
                val_mask_jax, cond_mask_jax,
                vmax_j, cmin_j,
                tau_i, z_lm, w_i,
                nkx, nky, nkz,
            )
        
        chi_sum = chi_sum + chi_win
    
    return chi_sum


# ============================================================================
# χ₀ kernel — direct minimax interface (no WindowPair indirection)
# ============================================================================

def _get_chi_minimax_kernel(mesh_xy: Mesh, nkx: int, nky: int, nkz: int):
    """Build chi0 kernel with jittable device-local FFTs on replicated k axes."""
    from common.fft_helpers import (
        make_jittable_local_fftn_3d,
        make_jittable_local_ifftn_3d,
    )

    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    chi_out = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    chi_R_shard = NamedSharding(mesh_xy, P('x', 'y', None, None, None))
    G_v_5d = NamedSharding(mesh_xy, P(None, None, 'x', None, 'y'))
    G_c_5d = NamedSharding(mesh_xy, P(None, None, 'y', None, 'x'))
    G_v_7d = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    G_c_7d = NamedSharding(mesh_xy, P(None, None, None, None, 'y', None, 'x'))

    G_v_t_spec = P(None, 'x', None, 'y', None, None, None)
    G_c_t_spec = P(None, 'y', None, 'x', None, None, None)
    chi_R_spec = P('x', 'y', None, None, None)

    local_ifftn_Gv = make_jittable_local_ifftn_3d(
        mesh_xy, G_v_t_spec, G_v_t_spec, norm='ortho', axes=(-3, -2, -1)
    )
    local_fftn_Gc = make_jittable_local_fftn_3d(
        mesh_xy, G_c_t_spec, G_c_t_spec, norm='ortho', axes=(-3, -2, -1)
    )
    local_fftn_chi = make_jittable_local_fftn_3d(
        mesh_xy, chi_R_spec, chi_R_spec, norm='ortho', axes=(-3, -2, -1)
    )

    @partial(jax.jit, static_argnames=("nkx", "nky", "nkz"))
    def _build_G(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v, enk_c,
        tau_scalar, vmax, cmin,
        nkx: int, nky: int, nkz: int,
    ):
        exp_v = jnp.exp(-tau_scalar * (vmax - enk_v))
        exp_c = jnp.exp(-tau_scalar * (enk_c - cmin))

        Gv_k = jnp.einsum('ksxn,kn,knty->ksxty',
                           jnp.conj(psi_val_xn), exp_v.astype(jnp.complex128), psi_val_yr,
                           optimize=True)
        Gc_k = jnp.einsum('ksxm,km,kmty->ksxty',
                           jnp.conj(psi_cond_xn), exp_c.astype(jnp.complex128), psi_cond_yr,
                           optimize=True)

        Gv_k = jax.lax.with_sharding_constraint(Gv_k, G_v_5d)
        Gc_k = jax.lax.with_sharding_constraint(Gc_k, G_c_5d)

        Gv_7 = jax.lax.with_sharding_constraint(
            Gv_k.reshape(nkx, nky, nkz, *Gv_k.shape[1:]), G_v_7d)
        Gc_7 = jax.lax.with_sharding_constraint(
            Gc_k.reshape(nkx, nky, nkz, *Gc_k.shape[1:]), G_c_7d)

        return Gv_7.transpose(3, 4, 5, 6, 0, 1, 2), Gc_7.transpose(3, 4, 5, 6, 0, 1, 2)

    @partial(jax.jit, donate_argnums=(0,))
    def _tau_step(
        chi_R_acc,
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v, enk_c,
        tau_scalar, prefactor_scalar, vmax, cmin,
    ):
        Gv_t, Gc_t = _build_G(
            psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
            enk_v, enk_c,
            tau_scalar, vmax, cmin,
            nkx, nky, nkz,
        )
        Gv_R = local_ifftn_Gv(Gv_t)
        Gc_mR = local_fftn_Gc(Gc_t)
        chi_tau = jnp.einsum('ambnxyz,bnamxyz->mnxyz', Gc_mR, Gv_R, optimize=True)
        return chi_R_acc + prefactor_scalar * chi_tau

    @jax.jit
    def _chi_R_to_q(chi_R_final):
        chi_q = local_fftn_chi(chi_R_final)
        chi_q = chi_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
        return jax.lax.with_sharding_constraint(chi_q, chi_out)

    def _chi_kernel(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v, enk_c,
        tau_i, prefactor_i,
        vmax, cmin,
        nkx, nky, nkz,
    ):
        n_rmu = psi_val_xn.shape[2]
        ntau = len(tau_i)

        if ntau == 0:
            chi = jnp.zeros((nkx, nky, nkz, 1, n_rmu, 1, n_rmu), dtype=jnp.complex128)
            return jax.lax.with_sharding_constraint(chi, chi_out)

        # Accumulator: directly sharded
        chi_R_shape = (n_rmu, n_rmu, nkx, nky, nkz)
        def _chi_zeros(idx):
            sh = tuple((s.stop - s.start) if s.stop is not None else d
                       for s, d in zip(idx, chi_R_shape))
            return np.zeros(sh, dtype=np.complex128)
        chi_R_acc = jax.make_array_from_callback(chi_R_shape, chi_R_shard, _chi_zeros)

        for itau in range(ntau):
            chi_R_acc = _tau_step(
                chi_R_acc,
                psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                enk_v, enk_c,
                tau_i[itau], prefactor_i[itau], vmax, cmin,
            )

        return _chi_R_to_q(chi_R_acc)

    _chi_minimax_kernel_cache[cache_key] = _chi_kernel
    return _chi_kernel


def compute_chi0_minimax(
    psi_val_xn: jax.Array, psi_val_yr: jax.Array,
    psi_cond_yr: jax.Array, psi_cond_xn: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    quad, meta, mesh_xy: Mesh,
    energy_reference: float | None = None,
) -> jax.Array:
    """Compute χ₀ from a LaplaceMinimaxQuadrature directly.

    quad.tau and quad.alpha approximate either:
      1/x  (static)  or  x/(x²+ωp²)  (imaginary-frequency)
    on [x_min, x_max] where x = E_c - E_v.

    The physical chi0 is:
      χ₀ = -2 Σ_ℓ α_ℓ Σ_{v,c} |M_vc|² exp(-τ_ℓ (E_c - E_v))

    A uniform energy shift is optional via ``energy_reference`` and is applied
    to both valence and conduction energies before building the minimax factors.
    Because only differences enter, this is algebraically invariant; the knob is
    provided so callers can keep the implementation explicitly aligned with their
    chosen global zero of energy (e.g. midgap or VBM).
    """
    _ensure_compilation_cache()

    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)

    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    alpha = np.asarray(quad.alpha, dtype=np.float64)
    prefactor = -2.0 * alpha * np.exp(-tau * E_gap)

    kernel = _get_chi_minimax_kernel(mesh_xy, nkx, nky, nkz)
    return kernel(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v - jnp.asarray(eref, dtype=enk_v.dtype),
        enk_c - jnp.asarray(eref, dtype=enk_c.dtype),
        jnp.asarray(tau, dtype=jnp.float64),
        jnp.asarray(prefactor, dtype=jnp.float64),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
        nkx, nky, nkz,
    )


# ============================================================================
# W solve with two-stage resharding (following load_wfns pattern)
# ============================================================================

def _get_w_solve_fn(mesh_xy: Mesh, nq: int, n_rmu: int):
    """
    Get or create W solve function with q-parallel shard_map.

    Strategy:
    1. Reshard V, χ from P(..., μ_X, ..., ν_Y) to P((q_XY), μ, ν)
    2. shard_map: each device solves ONLY its local q-points via fori_loop
    3. Reshard W back to P(..., μ_X, ..., ν_Y)

    The old code used a global fori_loop over all nq on every device,
    all-gathering each (μ,ν) slice — 16× redundant work on 16 GPUs.
    """
    from jax.experimental.shard_map import shard_map

    cache_key = (id(mesh_xy), nq, n_rmu)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    q_flat_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    W_out = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    rep_3d = NamedSharding(mesh_xy, P(None, None, None))

    q_spec = P(('x', 'y'), None, None)

    @jax.jit
    def _solve_w(V_q: jax.Array, chi_q: jax.Array, pref: jax.Array) -> jax.Array:
        nkx, nky, nkz = chi_q.shape[0], chi_q.shape[1], chi_q.shape[2]
        nq_local = nkx * nky * nkz
        n = chi_q.shape[4]

        # Flatten and reshard to q-parallel
        V_flat = V_q[0, 0, 0].reshape(nq_local, n, n)
        V_flat = jax.lax.with_sharding_constraint(V_flat, rep_3d)
        V_flat = jax.lax.with_sharding_constraint(V_flat, q_flat_shard)

        chi_flat = chi_q[:, :, :, 0, :, 0, :].reshape(nq_local, n, n)
        chi_flat = pref * chi_flat
        chi_flat = jax.lax.with_sharding_constraint(chi_flat, rep_3d)
        chi_flat = jax.lax.with_sharding_constraint(chi_flat, q_flat_shard)

        # Pad to multiple of total devices for clean shard_map
        total_devices = mesh_xy.devices.size
        nq_padded = ((nq_local + total_devices - 1) // total_devices) * total_devices
        pad_amount = nq_padded - nq_local
        if pad_amount > 0:
            V_flat = jnp.pad(V_flat, ((0, pad_amount), (0, 0), (0, 0)))
            chi_flat = jnp.pad(chi_flat, ((0, pad_amount), (0, 0), (0, 0)))
            V_flat = jax.lax.with_sharding_constraint(V_flat, q_flat_shard)
            chi_flat = jax.lax.with_sharding_constraint(chi_flat, q_flat_shard)

        # Each device solves its local q-shard independently (no communication)
        def _local_solve(V_local, chi_local):
            """Solve W = (I - V χ)^{-1} V for each local q-point."""
            nq_dev = V_local.shape[0]
            def solve_one(iq, W_acc):
                A = jnp.eye(n, dtype=V_local.dtype) - V_local[iq] @ chi_local[iq]
                lu, piv = jsp_linalg.lu_factor(A)
                W_iq = jsp_linalg.lu_solve((lu, piv), V_local[iq])
                return W_acc.at[iq].set(W_iq)
            return jax.lax.fori_loop(0, nq_dev, solve_one, jnp.zeros_like(V_local))

        W_flat = shard_map(
            _local_solve, mesh=mesh_xy,
            in_specs=(q_spec, q_spec), out_specs=q_spec,
        )(V_flat, chi_flat)

        # Remove padding and reshape back
        if pad_amount > 0:
            W_flat = W_flat[:nq_local]
        W_flat = jax.lax.with_sharding_constraint(W_flat, rep_3d)
        W_kqmn = W_flat.reshape(nkx, nky, nkz, n, n)
        W_out_arr = W_kqmn[:, :, :, None, :, None, :]
        W_out_arr = jax.lax.with_sharding_constraint(W_out_arr, W_out)

        return W_out_arr

    _w_solve_cache[cache_key] = _solve_w
    return _solve_w


def solve_w_from_chi_q_jax(
    V_qmunu: jax.Array,
    chi_q: jax.Array,
    meta: Meta,
    mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute screened interaction W_q = (I - V χ)^{-1} V from a precomputed χ(q).
    
    Uses two-stage resharding following load_wfns pattern:
    - χ input: P(None, None, None, None, 'x', None, 'y')
    - Reshard to q-parallel for solve
    - W output: same as χ
    
    Args:
        V_qmunu: (1, 1, 1, nkx, nky, nkz, n_rmu, n_rmu)
        chi_q:   (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
        meta: Meta with k-grid info
        mesh_xy: 2D device mesh
    
    Returns:
        W_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu) with μ_X, ν_Y sharding
    """
    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nq = nkx * nky * nkz
    n_rmu = chi_q.shape[4]
    
    # Normalization prefactor
    Nk = max(1, nq)
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    pref = jnp.asarray(2.0 / (math.sqrt(float(Nk)) * float(nspin) * float(nspinor)), dtype=jnp.complex128)
    
    # Get cached solve function
    solve_fn = _get_w_solve_fn(mesh_xy, nq, n_rmu)
    
    with jax_profile.annotation("W_solve"):
        W_q = solve_fn(V_qmunu, chi_q, pref)
    
    return W_q


def compute_screening(
    V_qmunu,
    wf_bundle,
    meta,
    mesh_xy,
    *,
    minimax_config: MinimaxConfig | None = None,
    ppm_omega_p=None,
    ppm_fallback_omega=2.0,
    tensors_filename=None,
    print0=print,
):
    """Compute static screened Coulomb W via minimax quadrature.

    Optionally also extracts GN-PPM parameters if ppm_omega_p is set.
    """
    import time

    if minimax_config is None:
        raise ValueError("minimax_config is required.")

    print0("")
    print0(f"  Minimax screening path enabled at ω = 0.0000 eV (0.000000 Ry)")

    s = wf_bundle.slices
    psi_val_xn = wf_bundle.xn(s.val)
    psi_val_yr = wf_bundle.yr(s.val)
    psi_cond_xn = wf_bundle.xn(s.cond)
    psi_cond_yr = wf_bundle.yr(s.cond)
    enk_v = wf_bundle.enk[:, s.val]
    enk_c = wf_bundle.enk[:, s.cond]
    e_ref = resolve_minimax_energy_reference(
        enk_v, enk_c,
        reference=minimax_config.energy_reference,
    )

    _windows_minimax, quad = build_static_minimax_window_pair(
        enk_v, enk_c,
        minimax_config=minimax_config,
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
        use_shipped_tables=bool(minimax_config.use_shipped_tables),
        print_fn=print0,
    )

    t_min0 = time.perf_counter()
    chi_omega = compute_chi0_minimax(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v, enk_c, quad, meta, mesh_xy,
        energy_reference=e_ref,
    )
    W_q = solve_w_from_chi_q_jax(V_qmunu, chi_omega, meta, mesh_xy)
    chi_omega.block_until_ready()
    W_q.block_until_ready()
    t_min = time.perf_counter() - t_min0
    chi_max = float(jnp.max(jnp.abs(chi_omega)))
    print0(f"  |χ(0)|_max = {chi_max:.6e}   (minimax, {quad.node_count} nodes)")
    print0(f"  minimax χ→W time: {t_min:9.3f} s")

    if ppm_omega_p is not None:
        omega_p = float(ppm_omega_p)
        if omega_p <= 0.0:
            raise ValueError("ppm_omega_p must be > 0 when GN-PPM extraction is requested.")
        from .minimax_screening import solve_laplace_minimax_imag_interval
        quad_imag = solve_laplace_minimax_imag_interval(
            quad.x_min, quad.x_max, omega_p,
            target_error=float(minimax_config.target_error),
            max_nodes=int(minimax_config.max_nodes),
        )
        print0(
            f"  Minimax imag-freq window (ωp={omega_p:.4f} Ry): "
            f"x=[{quad_imag.x_min:.6e}, {quad_imag.x_max:.6e}] Ry, "
            f"R={quad_imag.x_max / quad_imag.x_min:.2f}, "
            f"nodes={quad_imag.node_count}, fit_err~{quad_imag.max_error:.3e}"
        )
        chi_iwp = compute_chi0_minimax(
            psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
            enk_v, enk_c, quad_imag, meta, mesh_xy,
            energy_reference=e_ref,
        )
        chi_iwp.block_until_ready()
        W_iwp = solve_w_from_chi_q_jax(V_qmunu, chi_iwp, meta, mesh_xy)
        W_iwp.block_until_ready()
        nkx, nky, nkz = W_q.shape[0], W_q.shape[1], W_q.shape[2]
        n_rmu = W_q.shape[4]
        V_q = V_qmunu[0, 0, 0].reshape(nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
        ppm = extract_gn_ppm_parameters_from_Wc(
            W_q - V_q, W_iwp - V_q,
            omega_p=omega_p, fallback_omega=float(ppm_fallback_omega),
        )
        print0(
            f"  GN-PPM extracted from minimax W^c:"
            f" ω_p={omega_p:.6f} Ry, unfulfilled={100.0 * ppm.unfulfilled_fraction:.2f}%"
        )

    if tensors_filename is not None and os.path.exists(tensors_filename):
        from file_io import write_w0_qmunu_to_h5
        W0_qmunu = W_q[..., 0, :, 0, :][None, None, None, :, :, :, :, :]
        write_w0_qmunu_to_h5(tensors_filename, W0_qmunu)

    return W_q
def resolve_minimax_energy_reference(
    enk_v: jax.Array,
    enk_c: jax.Array,
    *,
    reference: str | float | int | None = "midgap",
    reference_fn: Callable[[jax.Array, jax.Array], float] | None = None,
) -> float:
    """Resolve the minimax energy reference used to shift band energies.

    This shift is algebraically neutral for χ0/W (only E_c-E_v enters), but
    exposing it at the top-level minimax pipeline keeps reference conventions
    explicit and synchronized with sigma paths.
    """
    if reference_fn is not None:
        return float(reference_fn(enk_v, enk_c))

    if reference is None:
        return 0.0
    if isinstance(reference, (int, float)):
        return float(reference)

    ref = str(reference).strip().lower()
    if ref in ("none", "raw", "zero"):
        return 0.0

    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64)
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64)
    vbm_ref = float(np.max(enk_v_host))
    cbm_ref = float(np.min(enk_c_host))

    if ref == "midgap":
        return 0.5 * (vbm_ref + cbm_ref)
    if ref == "vbm":
        return vbm_ref
    if ref == "cbm":
        return cbm_ref
    raise ValueError(f"Unknown minimax energy reference '{reference}'. Expected midgap/vbm/cbm/none or float.")
