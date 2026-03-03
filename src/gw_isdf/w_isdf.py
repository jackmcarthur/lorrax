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

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.experimental import compilation_cache as jax_compilation_cache
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import numpy as np

from isdf.common import Meta, jax_profile


# ============================================================================
# Cache and sharding registry
# ============================================================================

_COMPILATION_CACHE_READY = False
_chi_kernel_cache: dict = {}
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
    
    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']
    P_total = Pr * Pc
    
    # Shardings
    chi_in = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))  # χ input
    V_in = NamedSharding(mesh_xy, P(None, None, None, None, None, 'x', 'y'))    # V input (same for last 2 dims)
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


def get_static_w_q_jax(
    V_qmunu: jax.Array,
    chi_q: jax.Array,
    S_qmunu: jax.Array | None,
    meta: Meta,
    mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute static W_q = (I - V χ)^{-1} V.
    
    Uses two-stage resharding following load_wfns pattern:
    - χ input: P(None, None, None, None, 'x', None, 'y')
    - Reshard to q-parallel for solve
    - W output: same as χ
    
    Args:
        V_qmunu: (1, 1, 1, nkx, nky, nkz, n_rmu, n_rmu)
        chi_q:   (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
        S_qmunu: overlap matrix (not yet implemented) or None
        meta: Meta with k-grid info
        mesh_xy: 2D device mesh
    
    Returns:
        W_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu) with μ_X, ν_Y sharding
    """
    if S_qmunu is not None:
        raise NotImplementedError("Whitening (S_qmunu) not yet implemented in refactored W solve")
    
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
    
    W_q = get_static_w_q_jax(V_qmunu, chi_q, None, meta, mesh_xy)
    
    return chi_q, W_q


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
