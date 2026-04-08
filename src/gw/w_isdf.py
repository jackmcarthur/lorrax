"""
Static χ₀ and W computation with JAX.

Streamlined pipeline for COHSEX:
- Single universal chi kernel with energy masking (no per-window JIT recompilation)
- Two-stage resharding for W solve following load_wfns pattern
- χ computed with P(..., μ_X, ..., ν_Y) sharding
- V, χ resharded to P(q_XY, μ, ν) for Dyson solve

For dynamic W(ω) with window-specific kernels, see w_isdf_dynamic.py.
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
    from .wavefunction_bundle import WavefunctionBundle


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
        psi_vTX: jax.Array,    # (nk, ns, μ, nb_v) valence, μ sharded X
        psi_vY: jax.Array,     # (nk, nb_v, ns, μ) valence, μ sharded Y
        psi_cX: jax.Array,     # (nk, nb_c, ns, μ) conduction, μ sharded X
        psi_cTY: jax.Array,    # (nk, ns, μ, nb_c) conduction, μ sharded Y
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
        n_rmu = psi_vTX.shape[2]
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
            
            # G_v(k) = Σ_n exp_v[n] |ψ_v^n⟩⟨ψ_v^n|
            # psi_vTX: (nk, ns, μ, nb_v), psi_vY: (nk, nb_v, ns, μ)
            Gv_k = jnp.einsum('ksxn,kn,knty->ksxty', jnp.conj(psi_vTX), exp_v.astype(jnp.complex128), psi_vY, optimize=True)
            
            # G_c(k) = Σ_m exp_c[m] |ψ_c^m⟩⟨ψ_c^m|
            # Note: for conduction, psi_cX: (nk, nb_c, ns, μ), psi_cTY: (nk, ns, μ, nb_c)
            Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cTY), exp_c.astype(jnp.complex128), psi_cX, optimize=True)
            
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
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, meta: Meta, mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute static χ₀(q) by summing over all window pairs.
    
    Uses a single universal kernel with energy masking - no per-window JIT.
    
    Args:
        psi_vTX: (nk, ns, μ, nb_v) valence, μ sharded on X axis
        psi_vY:  (nk, nb_v, ns, μ) valence, μ sharded on Y axis
        psi_cX:  (nk, nb_c, ns, μ) conduction, μ sharded on X axis
        psi_cTY: (nk, ns, μ, nb_c) conduction, μ sharded on Y axis
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
    nk = psi_vTX.shape[0]
    nb_v = enk_v.shape[1]
    nb_c = enk_c.shape[1]
    n_rmu = psi_vTX.shape[2]
    
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
                psi_vTX, psi_vY, psi_cX, psi_cTY,
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
    """Build chi0 kernel with Python tau loop and eager shard_map FFTs.

    MEMORY-CRITICAL: shard_map inside @jax.jit hangs on multi-node (16+ GPUs).
    Solution: split the tau step into three phases:
      1. @jax.jit: einsum to build G_v(k), G_c(k) + reshape + transpose
      2. EAGER (Python level): shard_map FFT (physically local, no JIT)
      3. @jax.jit: contract G_v(R) × G_c(-R) → χ(R) + accumulate

    The shard_map FFT called eagerly works because it's not inside any JIT —
    it physically breaks the sharded array into per-device local pieces,
    FFTs each locally, and reassembles. XLA never sees the global array.
    """
    from common.fft_helpers import make_sharded_fftn_3d, make_sharded_ifftn_3d

    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    chi_out = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    chi_R_shard = NamedSharding(mesh_xy, P('x', 'y', None, None, None))
    G_v_5d = NamedSharding(mesh_xy, P(None, None, 'x', None, 'y'))
    G_c_5d = NamedSharding(mesh_xy, P(None, None, 'y', None, 'x'))
    G_v_7d = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    G_c_7d = NamedSharding(mesh_xy, P(None, None, None, None, 'y', None, 'x'))

    # Eager shard_map FFTs (called OUTSIDE JIT — this is the key).
    G_v_t_spec = P(None, 'x', None, 'y', None, None, None)
    G_c_t_spec = P(None, 'y', None, 'x', None, None, None)
    chi_R_spec = P('x', 'y', None, None, None)

    sharded_ifftn_Gv = make_sharded_ifftn_3d(mesh_xy, G_v_t_spec, G_v_t_spec,
                                              norm='ortho', axes=(-3, -2, -1))
    sharded_fftn_Gc = make_sharded_fftn_3d(mesh_xy, G_c_t_spec, G_c_t_spec,
                                            norm='ortho', axes=(-3, -2, -1))
    sharded_fftn_chi = make_sharded_fftn_3d(mesh_xy, chi_R_spec, chi_R_spec,
                                             norm='ortho', axes=(-3, -2, -1))

    # Phase 1: JIT'd einsum only. Reshape/transpose done eagerly in Python
    # to avoid SPMD partitioner compilation hangs on 16+ GPUs.
    @jax.jit
    def _build_G_k(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c,
        tau_scalar, vmax, cmin,
    ):
        exp_v = jnp.exp(-tau_scalar * (vmax - enk_v))
        exp_c = jnp.exp(-tau_scalar * (enk_c - cmin))

        Gv_k = jnp.einsum('ksxn,kn,knty->ksxty',
                           jnp.conj(psi_vTX), exp_v.astype(jnp.complex128), psi_vY,
                           optimize=True)
        Gc_k = jnp.einsum('ksxm,km,kmty->ksxty',
                           jnp.conj(psi_cTY), exp_c.astype(jnp.complex128), psi_cX,
                           optimize=True)

        return Gv_k, Gc_k

    # Phase 3: JIT'd contraction + accumulation (no FFT inside JIT)
    @partial(jax.jit, donate_argnums=(0,))
    def _contract_and_accumulate(chi_R_acc, Gv_R, Gc_mR, prefactor_scalar):
        chi_tau = jnp.einsum('ambnxyz,bnamxyz->mnxyz', Gc_mR, Gv_R, optimize=True)
        return chi_R_acc + prefactor_scalar * chi_tau

    @jax.jit
    def _chi_R_reshape(chi_R_q):
        """Reshape chi from (μ,ν,nkx,nky,nkz) → (nkx,nky,nkz,1,μ,1,ν)."""
        chi_q = chi_R_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
        return jax.lax.with_sharding_constraint(chi_q, chi_out)

    def _chi_kernel(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c,
        tau_i, prefactor_i,
        vmax, cmin,
        nkx, nky, nkz,
    ):
        n_rmu = psi_vTX.shape[2]
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
            # Phase 1: JIT'd einsum only
            Gv_k, Gc_k = _build_G_k(
                psi_vTX, psi_vY, psi_cX, psi_cTY,
                enk_v, enk_c,
                tau_i[itau], vmax, cmin,
            )

            # Reshape + transpose in Python (avoids SPMD partitioner issues)
            Gv_t = Gv_k.reshape(nkx, nky, nkz, *Gv_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            Gc_t = Gc_k.reshape(nkx, nky, nkz, *Gc_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
            del Gv_k, Gc_k

            # Phase 2: EAGER shard_map FFT (physically local, no JIT)
            Gv_R = sharded_ifftn_Gv(Gv_t)
            Gc_mR = sharded_fftn_Gc(Gc_t)
            del Gv_t, Gc_t

            # Phase 3: JIT'd contraction + accumulation
            chi_R_acc = _contract_and_accumulate(
                chi_R_acc, Gv_R, Gc_mR, prefactor_i[itau])
            del Gv_R, Gc_mR

        # Final R→q FFT: eager shard_map, then JIT'd reshape
        chi_R_q = sharded_fftn_chi(chi_R_acc)
        return _chi_R_reshape(chi_R_q)

    _chi_minimax_kernel_cache[cache_key] = _chi_kernel
    return _chi_kernel


def compute_chi0_minimax(
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
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
        psi_vTX, psi_vY, psi_cX, psi_cTY,
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
    Get or create W solve function with two-stage resharding.
    
    Strategy (from load_wfns):
    1. Reshard V, χ from P(..., μ_X, ..., ν_Y) to P((q_XY), μ, ν) for q-parallel solve
    2. fori_loop over q with replicated (μ, ν) solve per q
    3. Reshard W back to P(..., μ_X, ..., ν_Y)
    """
    cache_key = (id(mesh_xy), nq, n_rmu)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]
    
    # Shardings
    q_flat_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))            # (nq, μ, ν) q-sharded
    W_out = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))   # W output
    rep_shard = NamedSharding(mesh_xy, P(None, None))                           # replicated (μ, ν)
    rep_3d = NamedSharding(mesh_xy, P(None, None, None))                        # replicated (nq, μ, ν)

    @jax.jit
    def _solve_w(V_q: jax.Array, chi_q: jax.Array, pref: jax.Array) -> jax.Array:
        """
        Solve W = V (I - V χ)^{-1} = (I - V χ)^{-1} V via LU per q.

        Three-stage resharding to avoid XLA involuntary rematerialization:
        1. χ, V: P(..., μ_X, ν_Y) → replicated → P(q_XY, μ, ν)
        2. Solve per q with replicated matrices
        3. W: P(q_XY, μ, ν) → replicated → P(..., μ_X, ν_Y)
        """
        # Extract shapes
        nkx, nky, nkz = chi_q.shape[0], chi_q.shape[1], chi_q.shape[2]
        nq_local = nkx * nky * nkz
        n = chi_q.shape[4]  # n_rmu

        # Flatten V: (1, 1, 1, nkx, nky, nkz, μ, ν) → (nq, μ, ν)
        # Two-stage resharding: first replicate, then shard on q
        V_flat = V_q[0, 0, 0].reshape(nq_local, n, n)
        V_flat = jax.lax.with_sharding_constraint(V_flat, rep_3d)
        V_flat = jax.lax.with_sharding_constraint(V_flat, q_flat_shard)

        # Flatten χ: (nkx, nky, nkz, 1, μ, 1, ν) → (nq, μ, ν)
        chi_flat = chi_q[:, :, :, 0, :, 0, :].reshape(nq_local, n, n)
        chi_flat = pref * chi_flat
        chi_flat = jax.lax.with_sharding_constraint(chi_flat, rep_3d)
        chi_flat = jax.lax.with_sharding_constraint(chi_flat, q_flat_shard)

        # fori_loop over q with replicated solve
        def solve_body(iq, W_acc):
            # All-gather V[iq] and chi[iq] to replicated
            V_iq = jax.lax.with_sharding_constraint(V_flat[iq], rep_shard)
            chi_iq = jax.lax.with_sharding_constraint(chi_flat[iq], rep_shard)

            # Build A = I - V χ
            I = jnp.eye(n, dtype=V_iq.dtype)
            A = I - V_iq @ chi_iq

            # Solve A W = V via LU
            lu, piv = jsp_linalg.lu_factor(A)
            W_iq = jsp_linalg.lu_solve((lu, piv), V_iq)

            return W_acc.at[iq].set(W_iq)

        # Initialize output with same sharding
        W_flat = jnp.zeros_like(V_flat)
        W_flat = jax.lax.with_sharding_constraint(W_flat, q_flat_shard)
        W_flat = jax.lax.fori_loop(0, nq_local, solve_body, W_flat)

        # Reshape back: (nq, μ, ν) → (nkx, nky, nkz, 1, μ, 1, ν)
        # Two-stage resharding: first replicate, then apply output sharding
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


def get_static_w_q_jax(
    V_qmunu: jax.Array,
    chi_q: jax.Array,
    meta: Meta,
    mesh_xy: Mesh,
) -> jax.Array:
    """Backward-compatible alias for static W solve from χ(q)."""
    return solve_w_from_chi_q_jax(V_qmunu, chi_q, meta, mesh_xy)


# ============================================================================
# Combined χ₀ → W pipeline
# ============================================================================

def compute_chi0_and_w(
    V_qmunu: jax.Array,
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, meta: Meta, mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute static χ₀ and screened interaction W.
    
    Streamlined pipeline:
    1. χ₀(q) via universal kernel with energy masking
    2. W(q) via two-stage resharding + Dyson solve
    
    Returns:
        chi_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
        W_q:   (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
    """
    chi_q = compute_chi0(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, windows, meta, mesh_xy,
    )
    
    W_q = solve_w_from_chi_q_jax(V_qmunu, chi_q, meta, mesh_xy)
    
    return chi_q, W_q


def get_chi0_jax_from_bundle(
    wf_bundle: "WavefunctionBundle",
    windows,
    meta: Meta,
    mesh_xy: Mesh | None = None,
) -> jax.Array:
    """Compute static χ₀ directly from canonical WavefunctionBundle storage."""
    if mesh_xy is None:
        raise ValueError("mesh_xy is required for bundle-based chi0 evaluation")

    s = wf_bundle.slices
    psi_vTX = wf_bundle.x(s.v_slice)
    psi_vY = wf_bundle.y(s.v_slice)
    psi_cX = wf_bundle.y(s.c_slice)
    psi_cTY = wf_bundle.x(s.c_slice)
    enk_v = wf_bundle.enk[:, s.v_slice]
    enk_c = wf_bundle.enk[:, s.c_slice]
    return compute_chi0(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, windows, meta, mesh_xy,
    )


# ============================================================================
# Legacy interface aliases (for backward compatibility)
# ============================================================================

def get_chi0_jax(
    psi_vTX: jax.Array, psi_vY: jax.Array,
    psi_cX: jax.Array, psi_cTY: jax.Array,
    enk_v: jax.Array, enk_c: jax.Array,
    windows, meta: Meta, mesh_xy: Mesh | None = None,
) -> jax.Array:
    """Compute static χ₀ (alias for compute_chi0)."""
    return compute_chi0(
        psi_vTX, psi_vY, psi_cX, psi_cTY,
        enk_v, enk_c, windows, meta, mesh_xy,
    )


# Re-export legacy functions for W(ω)
def get_w_omega_jax(*args, **kwargs):
    """Dynamic W(ω) - delegates to the dynamic implementation."""
    from .w_isdf_dynamic import get_w_omega_jax as dynamic_w_omega
    return dynamic_w_omega(*args, **kwargs)


def get_chi_omega_jax(*args, **kwargs):
    """Dynamic χ(ω) - delegates to the dynamic implementation."""
    from .w_isdf_dynamic import get_chi_omega_jax as dynamic_chi_omega
    return dynamic_chi_omega(*args, **kwargs)


def frequency_integration(*args, **kwargs):
    """Scalar-ω CTSP frequency integration - delegates to dynamic implementation."""
    from .w_isdf_dynamic import frequency_integration as dynamic_frequency_integration
    return dynamic_frequency_integration(*args, **kwargs)


def get_w_omega_jax_from_bundle(*args, **kwargs):
    """Dynamic W(ω) from canonical WavefunctionBundle."""
    from .w_isdf_dynamic import get_w_omega_jax_from_bundle as dynamic_w_omega_bundle
    return dynamic_w_omega_bundle(*args, **kwargs)


def get_chi_omega_jax_from_bundle(*args, **kwargs):
    """Dynamic χ(ω) from canonical WavefunctionBundle."""
    from .w_isdf_dynamic import get_chi_omega_jax_from_bundle as dynamic_chi_omega_bundle
    return dynamic_chi_omega_bundle(*args, **kwargs)


def frequency_integration_from_bundle(*args, **kwargs):
    """Scalar-ω CTSP frequency integration from canonical WavefunctionBundle."""
    from .w_isdf_dynamic import frequency_integration_from_bundle as dynamic_frequency_integration_bundle
    return dynamic_frequency_integration_bundle(*args, **kwargs)


def compute_screening(
    V_qmunu,
    wf_bundle,
    window_pairs,
    meta,
    mesh_xy,
    *,
    omega=0.0,
    screening_method="minimax",
    minimax_config: MinimaxConfig | None = None,
    minimax_target_error=1.0e-6,
    minimax_max_nodes=64,
    use_shipped_minimax_tables=True,
    minimax_energy_reference="midgap",
    minimax_energy_reference_fn=None,
    ppm_omega_p=None,
    ppm_fallback_omega=2.0,
    validate_static=True,
    tensors_filename=None,
    print0=print,
):
    """Compute screened Coulomb W(ω) from V and wavefunctions.

    Supports two screening paths:
    - ``screening_method='minimax'`` (default): canonical minimax quadrature
      from ``common.minimax`` on a single static window.
    - ``screening_method='ctsp'``: legacy CTSP dynamic path.

    Parameters
    ----------
    V_qmunu : jax.Array
        Bare Coulomb in ISDF basis.
    wf_bundle : WavefunctionBundle
        Canonical wavefunction bundle.
    window_pairs : list or None
        Energy windows for CTSP quadrature. Required only for ``ctsp`` mode.
    meta : Meta
        System metadata.
    mesh_xy : Mesh
        Device mesh.
    omega : float
        Evaluation frequency in Ry (0.0 for static COHSEX).
    screening_method : {'minimax', 'ctsp'}
        Select minimax static path or legacy CTSP path.
    minimax_config : MinimaxConfig or None
        Optional shared minimax configuration object. When provided, it overrides
        the individual minimax scalar arguments below.
    minimax_target_error : float
        Max 1/x fit error target for canonical minimax quadrature.
    minimax_max_nodes : int
        Upper bound for minimax node search.
    use_shipped_minimax_tables : bool
        If True, allow canonical minimax quadratures to be loaded from the
        bundled table catalog before falling back to the exact solver.
    minimax_energy_reference : {'midgap','vbm','cbm','none'} or float
        Uniform energy shift reference for minimax χ0/W. This does not change
        the physical result, but keeps reference conventions explicit.
    minimax_energy_reference_fn : callable or None
        Optional override returning the reference energy from (enk_v, enk_c).
    ppm_omega_p : float or None
        If set in minimax mode, also extract GN-PPM parameters using chi(0)
        and chi(i*omega_p) built from the same minimax nodes.
    ppm_fallback_omega : float
        Fallback Omega value (Ry) for unfulfilled GN modes.
    validate_static : bool
        In CTSP mode, if True and omega==0 compare dynamic and static paths.
    tensors_filename : str or None
        If not None and file exists, save W0_qmunu to it.
    print0 : callable
        Rank-0 print function.

    Returns
    -------
    W_q : jax.Array
        Screened Coulomb interaction.
    """
    import time

    ryd2ev = 13.605693122994
    omega = float(omega)
    omega_ev = omega * ryd2ev
    method = str(screening_method).strip().lower()
    if method not in {"minimax", "ctsp"}:
        raise ValueError(f"Unknown screening_method={screening_method!r}; expected 'minimax' or 'ctsp'.")

    if method == "minimax" and abs(omega) > 1.0e-14:
        raise NotImplementedError(
            "screening_method='minimax' currently supports only omega=0. "
            "Use screening_method='ctsp' for finite-frequency screening."
        )

    print0("")
    if abs(omega) <= 1e-14 and method == "minimax":
        print0(f"  Minimax screening path enabled at ω = 0.0000 eV ({omega:.6f} Ry)")
    elif abs(omega) <= 1e-14:
        print0(f"  CTSP screening path enabled at ω = 0.0000 eV ({omega:.6f} Ry)")
    else:
        print0(f"  [DEBUG] CTSP screening at ω = {omega_ev:.4f} eV ({omega:.6f} Ry)")

    if method == "minimax":
        if minimax_config is not None:
            minimax_target_error = float(minimax_config.target_error)
            minimax_max_nodes = int(minimax_config.max_nodes)
            use_shipped_minimax_tables = bool(minimax_config.use_shipped_tables)
            minimax_energy_reference = minimax_config.energy_reference
        s = wf_bundle.slices
        psi_vTX = wf_bundle.x(s.v_slice)
        psi_vY = wf_bundle.y(s.v_slice)
        psi_cX = wf_bundle.y(s.c_slice)
        psi_cTY = wf_bundle.x(s.c_slice)
        enk_v = wf_bundle.enk[:, s.v_slice]
        enk_c = wf_bundle.enk[:, s.c_slice]
        e_ref = resolve_minimax_energy_reference(
            enk_v,
            enk_c,
            reference=minimax_energy_reference,
            reference_fn=minimax_energy_reference_fn,
        )

        _windows_minimax, quad = build_static_minimax_window_pair(
            enk_v,
            enk_c,
            minimax_config=minimax_config,
            target_error=float(minimax_target_error),
            max_nodes=int(minimax_max_nodes),
            use_shipped_tables=bool(use_shipped_minimax_tables),
            print_fn=print0,
        )

        t_min0 = time.perf_counter()
        chi_omega = compute_chi0_minimax(
            psi_vTX, psi_vY, psi_cX, psi_cTY,
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
            # Fresh minimax for chi0(i*omega_p)
            from .minimax_screening import solve_laplace_minimax_imag_interval
            quad_imag = solve_laplace_minimax_imag_interval(
                quad.x_min, quad.x_max, omega_p,
                target_error=float(minimax_target_error),
                max_nodes=int(minimax_max_nodes),
            )
            R_imag = quad_imag.x_max / quad_imag.x_min
            omega_hat = omega_p / quad_imag.x_min
            print0(
                f"  Minimax imag-freq window (ωp={omega_p:.4f} Ry): "
                f"x=[{quad_imag.x_min:.6e}, {quad_imag.x_max:.6e}] Ry, "
                f"R={R_imag:.2f}, ω̂={omega_hat:.2f}, "
                f"nodes={quad_imag.node_count}, fit_err~{quad_imag.max_error:.3e}"
            )
            chi_iwp = compute_chi0_minimax(
                psi_vTX, psi_vY, psi_cX, psi_cTY,
                enk_v, enk_c, quad_imag, meta, mesh_xy,
                energy_reference=e_ref,
            )
            chi_iwp.block_until_ready()
            W_iwp = solve_w_from_chi_q_jax(V_qmunu, chi_iwp, meta, mesh_xy)
            W_iwp.block_until_ready()
            # Fit GN-PPM to W^c(0) and W^c(iωp) (no head).
            nkx, nky, nkz = W_q.shape[0], W_q.shape[1], W_q.shape[2]
            n_rmu = W_q.shape[4]
            V_q = V_qmunu[0, 0, 0].reshape(nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
            Wc0_q = W_q - V_q
            Wci_q = W_iwp - V_q
            ppm = extract_gn_ppm_parameters_from_Wc(
                Wc0_q,
                Wci_q,
                omega_p=omega_p,
                fallback_omega=float(ppm_fallback_omega),
            )
            print0(
                "  GN-PPM extracted from minimax W^c:"
                f" ω_p={omega_p:.6f} Ry, unfulfilled={100.0 * ppm.unfulfilled_fraction:.2f}%"
            )
    else:
        if window_pairs is None:
            raise ValueError("window_pairs are required when screening_method='ctsp'.")
        t_dyn0 = time.perf_counter()
        W_q, chi_omega = get_w_omega_jax_from_bundle(
            V_qmunu, wf_bundle, window_pairs, omega, meta, mesh_xy,
        )
        W_q.block_until_ready()
        chi_omega.block_until_ready()
        t_dyn = time.perf_counter() - t_dyn0
        chi_max = float(jnp.max(jnp.abs(chi_omega)))
        print0(f"  |χ(ω)|_max = {chi_max:.6e}")

        # Validate CTSP dynamic path against legacy static path at ω=0
        if validate_static and abs(omega) <= 1e-14:
            t_static0 = time.perf_counter()
            chi0_static = get_chi0_jax_from_bundle(wf_bundle, window_pairs, meta, mesh_xy)
            W_q_static = get_static_w_q_jax(V_qmunu, chi0_static, meta, mesh_xy)
            chi0_static.block_until_ready()
            W_q_static.block_until_ready()
            t_static = time.perf_counter() - t_static0

            chi_err_abs = float(jnp.max(jnp.abs(chi_omega - chi0_static)))
            chi_ref = max(float(jnp.max(jnp.abs(chi0_static))), 1e-16)
            chi_err_rel = chi_err_abs / chi_ref
            W_err_abs = float(jnp.max(jnp.abs(W_q - W_q_static)))
            W_ref = max(float(jnp.max(jnp.abs(W_q_static))), 1e-16)
            W_err_rel = W_err_abs / W_ref

            print0("  χ/W path comparison at ω=0 (dynamic vs static):")
            print0(f"    dynamic time: {t_dyn:9.3f} s")
            print0(f"    static time:  {t_static:9.3f} s")
            print0(f"    χ max abs diff: {chi_err_abs:.6e}, rel diff: {chi_err_rel:.6e}")
            print0(f"    W max abs diff: {W_err_abs:.6e}, rel diff: {W_err_rel:.6e}")

    if tensors_filename is not None and os.path.exists(tensors_filename):
        from file_io import write_w0_qmunu_to_h5
        W0_qmunu = W_q[..., 0, :, 0, :]
        W0_qmunu = W0_qmunu[None, None, None, :, :, :, :, :]
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
