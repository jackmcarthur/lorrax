"""Static χ₀ and W computation using ISDF + minimax quadrature.

All inter-function arrays use flat k/q indices: chi(nq, μ, μ), V(nq, μ, μ), W(nq, μ, μ).
The 3D k-grid only appears inside FFT helpers.
"""
import os
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
from .minimax_screening import build_static_minimax_window_pair


# ============================================================================
# Cache and sharding registry
# ============================================================================

_chi_minimax_kernel_cache: dict = {}
_w_solve_cache: dict = {}


# Thin wrapper around the shared activator so in-place callers in this
# module keep working.  See common.jax_compile_cache.
def _ensure_compilation_cache():
    from common.jax_compile_cache import ensure_jax_compile_cache
    ensure_jax_compile_cache()


# ============================================================================
# χ₀ kernel — minimax quadrature
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
            chi_R_shape, NamedSharding(mesh_xy, P(None, 'x', 'y')), _chi_zeros)

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

        # Pad to device count then reshard to q-parallel
        total_devices = mesh_xy.devices.size
        nq_padded = ((nq_local + total_devices - 1) // total_devices) * total_devices
        pad = nq_padded - nq_local
        V_padded = jnp.pad(V_flat, ((0, pad), (0, 0), (0, 0))) if pad > 0 else V_flat
        chi_padded = jnp.pad(chi_scaled, ((0, pad), (0, 0), (0, 0))) if pad > 0 else chi_scaled
        V_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(V_padded, rep_3d), q_shard)
        chi_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(chi_padded, rep_3d), q_shard)

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


def solve_w(V_q, chi0_q, meta, mesh_xy):
    """W(q) = (I − V χ₀)⁻¹ V  via q-parallel Dyson solve.

    All arrays flat-q: V(nq, μ, μ), χ₀(nq, μ, μ) → W(nq, μ, μ).
    """
    nq = int(meta.nk_tot)
    n_rmu = chi0_q.shape[1]
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    pref = jnp.asarray(2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor)),
                        dtype=jnp.complex128)
    solve_fn = _get_w_solve_fn(mesh_xy, nq, n_rmu)
    with jax_profile.annotation("W_solve"):
        return solve_fn(V_q, chi0_q, pref)


# Backward-compatible alias
solve_w_from_chi_q_jax = solve_w



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


# ---------------------------------------------------------------------------
#  Top-level screening helpers (used directly by gw_jax.main)
# ---------------------------------------------------------------------------

def flatten_V_qmunu(V_qmunu):
    """Strip (1,npol,npol) leading axes and flatten k-grid → flat-q (nq, μ, μ)."""
    return jnp.asarray(V_qmunu)[0, 0, 0].reshape(-1, V_qmunu.shape[-2], V_qmunu.shape[-1])


def build_static_quadrature(wfns, minimax_config, *, print_fn=None):
    """Build static minimax quadrature and energy reference from wavefunction bundle.

    Returns (quad, e_ref) where quad is a LaplaceMinimaxQuadrature for 1/x
    on the band-energy interval, and e_ref is the global energy zero.
    """
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    e_ref = resolve_minimax_energy_reference(
        enk_v, enk_c, reference=minimax_config.energy_reference)
    _, quad = build_static_minimax_window_pair(
        enk_v, enk_c, minimax_config=minimax_config, print_fn=print_fn)
    return quad, e_ref


def build_imag_quadrature(quad, omega_p, minimax_config, *, print_fn=None):
    """Build imaginary-frequency minimax quadrature for x/(x²+ωp²).

    Uses the same energy interval as the static quadrature.
    """
    from .minimax_screening import solve_laplace_minimax_imag_interval
    quad_imag = solve_laplace_minimax_imag_interval(
        quad.x_min, quad.x_max, float(omega_p),
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
    )
    if print_fn is not None:
        R = quad_imag.x_max / quad_imag.x_min
        print_fn(
            f"  PPM imag-freq quadrature (ωp={float(omega_p):.4f} Ry): "
            f"R={R:.1f}, nodes={quad_imag.node_count}, err~{quad_imag.max_error:.1e}")
    return quad_imag


def compute_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=0.0):
    """Compute χ₀(q) from wavefunction bundle and minimax quadrature.

    Returns flat-q array (nq, μ, μ).  Thin wrapper around compute_chi0_minimax.
    """
    s = wfns.slices
    return compute_chi0_minimax(
        wfns.xn(s.val), wfns.yr(s.val),
        wfns.yr(s.cond), wfns.xn(s.cond),
        wfns.enk[:, s.val], wfns.enk[:, s.cond],
        quad, meta, mesh_xy, energy_reference=energy_reference,
    )

