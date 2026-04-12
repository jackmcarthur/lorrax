"""Static χ₀ and W computation using ISDF + minimax quadrature.

All inter-function arrays use flat k/q indices: chi(nq, μ, μ), V(nq, μ, μ), W(nq, μ, μ).
The 3D k-grid only appears inside FFT helpers.
"""
import os
import math
from functools import partial
from pathlib import Path
from typing import Callable

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

def _get_chi_kernel(mesh_xy: Mesh, kgrid: tuple[int, int, int]):
    """Get or create chi0 kernel.  Returns flat-q χ₀(nq, μ, μ)."""
    nkx, nky, nkz = kgrid
    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_kernel_cache:
        return _chi_kernel_cache[cache_key]

    nk = nkx * nky * nkz
    _kg = (nkx, nky, nkz)

    # FFT helpers: flat-k ↔ flat-R, no 7D arrays visible to callers
    G_shard = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
    chi_shard = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

    def _fft_flat(x_flat, shard, fft_fn):
        x_3d = jax.lax.with_sharding_constraint(x_flat.reshape(*_kg, *x_flat.shape[1:]), shard)
        return fft_fn(x_3d, axes=(0, 1, 2), norm='ortho').reshape(nk, *x_flat.shape[1:])

    def _G_ifftn(g_k): return _fft_flat(g_k, G_shard, jnp.fft.ifftn)
    def _G_fftn(g_k):  return _fft_flat(g_k, G_shard, jnp.fft.fftn)
    def _chi_fftn(c_R): return _fft_flat(c_R, chi_shard, jnp.fft.fftn)

    @jax.jit
    def _chi_kernel(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v, enk_c, val_mask, cond_mask, vmax, cmin, tau_i, z_lm, w_i,
    ):
        n_rmu = psi_val_xn.shape[2]
        ntau = tau_i.shape[0]
        if ntau == 0:
            return jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128)

        E_gap = cmin - vmax
        quad_w = -2.0 * z_lm * w_i * jnp.exp(-(z_lm * E_gap - 1.0) * tau_i)

        from .greens_function_kernel import build_G as _build_G

        def tau_body(itau, chi_R_acc):
            tau = tau_i[itau]
            phases_v = jnp.where(val_mask,
                jnp.exp(-z_lm * tau * (vmax - enk_v)), jnp.zeros_like(enk_v))
            phases_c = jnp.where(cond_mask,
                jnp.exp(-z_lm * tau * (enk_c - cmin)), jnp.zeros_like(enk_c))

            # conj() because build_G conjugates psi_yr (ν side) but chi0 needs
            # conjugation on psi_xn (μ side). For real phases, conj(G) = G†.
            Gv_k = jnp.conj(_build_G(psi_val_xn, psi_val_yr, phases=phases_v))
            Gc_k = jnp.conj(_build_G(psi_cond_xn, psi_cond_yr, phases=phases_c))

            Gv_R = _G_ifftn(Gv_k)   # G_v(+R)
            Gc_mR = _G_fftn(Gc_k)    # G_c(-R)

            # χ(R,μ,ν) = Σ_{s,s'} G_c(-R,s',ν,s,μ) · G_v(R,s,μ,s',ν)
            # G shapes: (nR, s, μ, s, μ) — R is leading index (batch dim)
            chi_tau = jnp.einsum('Rambn,Rbnam->Rmn', Gc_mR, Gv_R, optimize=True)
            return chi_R_acc + quad_w[itau] * chi_tau

        chi_R = jax.lax.fori_loop(0, ntau, tau_body,
            jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128))
        return _chi_fftn(chi_R)

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
        chi_q: (nq, n_rmu, n_rmu) flat-q
    """
    _ensure_compilation_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    nk = kgrid[0] * kgrid[1] * kgrid[2]
    n_rmu = psi_val_xn.shape[2]
    enk_v_host = np.asarray(jax.device_get(enk_v))
    enk_c_host = np.asarray(jax.device_get(enk_c))
    kernel = _get_chi_kernel(mesh_xy, kgrid)
    chi_sum = jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128)
    
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
            )
        
        chi_sum = chi_sum + chi_win
    
    return chi_sum


# ============================================================================
# χ₀ kernel — direct minimax interface (no WindowPair indirection)
# ============================================================================

def _get_chi_minimax_kernel(mesh_xy: Mesh, kgrid: tuple[int, int, int]):
    """Build chi0 kernel with device-local FFTs.  Returns flat-q χ₀(nq, μ, μ)."""
    from common.fft_helpers import make_jittable_local_fftn_3d, make_jittable_local_ifftn_3d

    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    # Flat FFT helpers using device-local shard_map (production path)
    _Gv_spec = P(None, None, None, None, 'x', None, 'y')
    _Gc_spec = P(None, None, None, None, 'y', None, 'x')
    _chi_spec = P(None, None, None, 'x', 'y')
    _Gv_shard = NamedSharding(mesh_xy, _Gv_spec)
    _Gc_shard = NamedSharding(mesh_xy, _Gc_spec)
    _chi_shard = NamedSharding(mesh_xy, _chi_spec)

    _raw_ifftn_Gv = make_jittable_local_ifftn_3d(mesh_xy, _Gv_spec, _Gv_spec, norm='ortho', axes=(0, 1, 2))
    _raw_fftn_Gc = make_jittable_local_fftn_3d(mesh_xy, _Gc_spec, _Gc_spec, norm='ortho', axes=(0, 1, 2))
    _raw_fftn_chi = make_jittable_local_fftn_3d(mesh_xy, _chi_spec, _chi_spec, norm='ortho', axes=(0, 1, 2))

    def _Gv_ifftn(g_k):
        return _raw_ifftn_Gv(jax.lax.with_sharding_constraint(
            g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]), _Gv_shard)).reshape(nk, *g_k.shape[1:])

    def _Gc_fftn(g_k):
        return _raw_fftn_Gc(jax.lax.with_sharding_constraint(
            g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]), _Gc_shard)).reshape(nk, *g_k.shape[1:])

    def _chi_fftn_local(c_R):
        return _raw_fftn_chi(jax.lax.with_sharding_constraint(
            c_R.reshape(nkx, nky, nkz, *c_R.shape[1:]), _chi_shard)).reshape(nk, *c_R.shape[1:])

    from .greens_function_kernel import build_G as _build_G_mm

    @jax.jit
    def _build_G(psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                 enk_v, enk_c, tau_scalar, vmax, cmin):
        phases_v = jnp.exp(-tau_scalar * (vmax - enk_v))
        phases_c = jnp.exp(-tau_scalar * (enk_c - cmin))
        Gv_k = jnp.conj(_build_G_mm(psi_val_xn, psi_val_yr, phases=phases_v))
        Gc_k = jnp.conj(_build_G_mm(psi_cond_xn, psi_cond_yr, phases=phases_c))
        return Gv_k, Gc_k

    @partial(jax.jit, donate_argnums=(0,))
    def _tau_step(chi_R_acc, psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                  enk_v, enk_c, tau_scalar, prefactor_scalar, vmax, cmin):
        Gv_k, Gc_k = _build_G(psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                               enk_v, enk_c, tau_scalar, vmax, cmin)
        Gv_R = _Gv_ifftn(Gv_k)
        Gc_mR = _Gc_fftn(Gc_k)
        chi_tau = jnp.einsum('Rambn,Rbnam->Rmn', Gc_mR, Gv_R, optimize=True)
        return chi_R_acc + prefactor_scalar * chi_tau

    def _chi_kernel(psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                    enk_v, enk_c, tau_i, prefactor_i, vmax, cmin):
        n_rmu = psi_val_xn.shape[2]
        ntau = len(tau_i)
        if ntau == 0:
            return jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128)

        chi_R_shape = (nk, n_rmu, n_rmu)
        def _chi_zeros(idx):
            sh = tuple((s.stop - s.start) if s.stop is not None else d
                       for s, d in zip(idx, chi_R_shape))
            return np.zeros(sh, dtype=np.complex128)
        chi_R_acc = jax.make_array_from_callback(
            chi_R_shape, NamedSharding(mesh_xy, P(('x', 'y'), None, None)), _chi_zeros)

        for itau in range(ntau):
            chi_R_acc = _tau_step(
                chi_R_acc, psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                enk_v, enk_c, tau_i[itau], prefactor_i[itau], vmax, cmin)

        return _chi_fftn_local(chi_R_acc)

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
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    prefactor = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)

    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid)
    return kernel(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v - jnp.asarray(eref, dtype=enk_v.dtype),
        enk_c - jnp.asarray(eref, dtype=enk_c.dtype),
        jnp.asarray(tau, dtype=jnp.float64),
        jnp.asarray(prefactor, dtype=jnp.float64),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )


# ============================================================================
# W solve with two-stage resharding (following load_wfns pattern)
# ============================================================================

def _get_w_solve_fn(mesh_xy: Mesh, nq: int, n_rmu: int):
    """W = (I - V χ)⁻¹ V via q-parallel shard_map.  All arrays flat-q: (nq, μ, μ)."""
    from jax.experimental.shard_map import shard_map

    cache_key = (id(mesh_xy), nq, n_rmu)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    q_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    rep_3d = NamedSharding(mesh_xy, P(None, None, None))
    q_spec = P(('x', 'y'), None, None)

    @jax.jit
    def _solve_w(V_flat: jax.Array, chi_flat: jax.Array, pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ).  Returns W: (nq, μ, μ)."""
        nq_local = V_flat.shape[0]
        n = V_flat.shape[1]
        chi_scaled = pref * chi_flat

        # Reshard to q-parallel
        V_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(V_flat, rep_3d), q_shard)
        chi_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(chi_scaled, rep_3d), q_shard)

        # Pad to device count for clean shard_map
        total_devices = mesh_xy.devices.size
        nq_padded = ((nq_local + total_devices - 1) // total_devices) * total_devices
        pad = nq_padded - nq_local
        if pad > 0:
            V_q = jax.lax.with_sharding_constraint(
                jnp.pad(V_q, ((0, pad), (0, 0), (0, 0))), q_shard)
            chi_q = jax.lax.with_sharding_constraint(
                jnp.pad(chi_q, ((0, pad), (0, 0), (0, 0))), q_shard)

        def _local_solve(V_local, chi_local):
            nq_dev = V_local.shape[0]
            def solve_one(iq, W_acc):
                A = jnp.eye(n, dtype=V_local.dtype) - V_local[iq] @ chi_local[iq]
                lu, piv = jsp_linalg.lu_factor(A)
                return W_acc.at[iq].set(jsp_linalg.lu_solve((lu, piv), V_local[iq]))
            return jax.lax.fori_loop(0, nq_dev, solve_one, jnp.zeros_like(V_local))

        W_flat = shard_map(
            _local_solve, mesh=mesh_xy,
            in_specs=(q_spec, q_spec), out_specs=q_spec,
        )(V_q, chi_q)

        if pad > 0:
            W_flat = W_flat[:nq_local]
        return jax.lax.with_sharding_constraint(W_flat, rep_3d)

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
    All arrays flat-q: V(nq, μ, μ), chi(nq, μ, μ) → W(nq, μ, μ).
    """
    nq = int(meta.nk_tot)
    n_rmu = chi_q.shape[1]
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    pref = jnp.asarray(2.0 / (math.sqrt(float(max(1, nq))) * float(nspin) * float(nspinor)),
                        dtype=jnp.complex128)
    solve_fn = _get_w_solve_fn(mesh_xy, nq, n_rmu)
    with jax_profile.annotation("W_solve"):
        return solve_fn(V_qmunu, chi_q, pref)


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

    _, quad = build_static_minimax_window_pair(
        enk_v, enk_c,
        minimax_config=minimax_config,
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
        use_shipped_tables=bool(minimax_config.use_shipped_tables),
        print_fn=print0,
    )

    # Flatten V to (nq, μ, μ) for Dyson solve
    V_flat = jnp.asarray(V_qmunu)[0, 0, 0].reshape(-1, V_qmunu.shape[-2], V_qmunu.shape[-1])

    t_min0 = time.perf_counter()
    # chi0 and W are all flat-q: (nq, μ, μ)
    chi0_q = compute_chi0_minimax(
        psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
        enk_v, enk_c, quad, meta, mesh_xy, energy_reference=e_ref)
    W_q = solve_w_from_chi_q_jax(V_flat, chi0_q, meta, mesh_xy)
    chi0_q.block_until_ready()
    W_q.block_until_ready()
    t_min = time.perf_counter() - t_min0
    print0(f"  |χ(0)|_max = {float(jnp.max(jnp.abs(chi0_q))):.6e}   (minimax, {quad.node_count} nodes)")
    print0(f"  minimax χ→W time: {t_min:9.3f} s")

    if ppm_omega_p is not None:
        omega_p = float(ppm_omega_p)
        if omega_p <= 0.0:
            raise ValueError("ppm_omega_p must be > 0 when GN-PPM extraction is requested.")
        from .minimax_screening import solve_laplace_minimax_imag_interval
        quad_imag = solve_laplace_minimax_imag_interval(
            quad.x_min, quad.x_max, omega_p,
            target_error=float(minimax_config.target_error),
            max_nodes=int(minimax_config.max_nodes))
        print0(
            f"  Minimax imag-freq window (ωp={omega_p:.4f} Ry): "
            f"x=[{quad_imag.x_min:.6e}, {quad_imag.x_max:.6e}] Ry, "
            f"R={quad_imag.x_max / quad_imag.x_min:.2f}, "
            f"nodes={quad_imag.node_count}, fit_err~{quad_imag.max_error:.3e}")
        chi_iwp = compute_chi0_minimax(
            psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
            enk_v, enk_c, quad_imag, meta, mesh_xy, energy_reference=e_ref)
        chi_iwp.block_until_ready()
        W_iwp = solve_w_from_chi_q_jax(V_flat, chi_iwp, meta, mesh_xy)
        W_iwp.block_until_ready()
        # PPM extraction: W^c = W - V, all flat-q (nq, μ, μ)
        ppm = extract_gn_ppm_parameters_from_Wc(
            W_q - V_flat, W_iwp - V_flat,
            omega_p=omega_p, fallback_omega=float(ppm_fallback_omega))
        print0(
            f"  GN-PPM extracted from minimax W^c:"
            f" ω_p={omega_p:.6f} Ry, unfulfilled={100.0 * ppm.unfulfilled_fraction:.2f}%")

    if tensors_filename is not None and os.path.exists(tensors_filename):
        from file_io import write_w0_qmunu_to_h5
        kgrid_tuple = (int(meta.nkx), int(meta.nky), int(meta.nkz))
        W0_7d = W_q.reshape(*kgrid_tuple, W_q.shape[1], W_q.shape[2])
        write_w0_qmunu_to_h5(tensors_filename, W0_7d[None, None, None, :, :, :, :, :])

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
