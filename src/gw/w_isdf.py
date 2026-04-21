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
    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    # Flat-k FFT helpers — callers see only (nk, *trail) arrays.
    # Gv_R label 'Rbnam' (n at axis 4, m at axis 6) with spec below pins
    # n on 'x', m on 'y'.  Gc_mR label 'Rambn' (m at axis 4, n at axis 6)
    # pins m on 'y', n on 'x'.  The einsum 'Rambn,Rbnam->Rmn' contracts out
    # the replicated (a, b, n) axes and leaves output sharded (m='y', n='x').
    # chi_R_acc MUST agree: (m='y', n='x') == P(None, 'y', 'x').  Earlier
    # code had P(None, 'x', 'y') which forced XLA to materialize the full
    # c128[64,2400,2400] replicated buffer every tau step to permute the
    # sharding — 23.6 GB temp that OOMs at Si 4x4x4 60Ry μ=2400.
    _Gv_spec = P(None, None, None, None, 'x', None, 'y')
    _Gc_spec = P(None, None, None, None, 'y', None, 'x')
    _chi_spec = P(None, None, None, 'y', 'x')

    _Gv_ifftn      = make_flat_k_ifftn(mesh_xy, kgrid, _Gv_spec,  norm='ortho')
    _Gc_fftn       = make_flat_k_fftn( mesh_xy, kgrid, _Gc_spec,  norm='ortho')
    _chi_fftn_local = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm='ortho')

    from .greens_function_kernel import build_G as _build_G_mm

    # 5-D flat-k output shardings matching the 7-D FFT input specs above
    # (μ_X on 'x', μ_Y on 'y' for Gv; swapped for Gc).  Explicit constraints
    # on the einsum output force XLA to pick a sharded GEMM lowering rather
    # than a replicated cuBLAS call that materializes the full 64×2400×2400
    # buffer per device.
    _Gv_out_flatk = P(None, None, 'x', None, 'y')
    _Gc_out_flatk = P(None, None, 'y', None, 'x')
    _chi_R_spec = P(None, 'y', 'x')

    @jax.jit
    def _build_G(psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                 enk_v, enk_c, tau_scalar, vmax, cmin):
        phases_v = jnp.exp(-tau_scalar * (vmax - enk_v))
        phases_c = jnp.exp(-tau_scalar * (enk_c - cmin))
        Gv_k = jax.lax.with_sharding_constraint(
            _build_G_mm(psi_val_xn, psi_val_yr, phases=phases_v),
            NamedSharding(mesh_xy, _Gv_out_flatk))
        Gc_k = jax.lax.with_sharding_constraint(
            _build_G_mm(psi_cond_xn, psi_cond_yr, phases=phases_c),
            NamedSharding(mesh_xy, _Gc_out_flatk))
        return jnp.conj(Gv_k), jnp.conj(Gc_k)

    @partial(jax.jit, donate_argnums=(0,))
    def _tau_step(chi_R_acc, psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                  enk_v, enk_c, tau_scalar, prefactor_scalar, vmax, cmin):
        Gv_k, Gc_k = _build_G(psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
                               enk_v, enk_c, tau_scalar, vmax, cmin)
        Gv_R = _Gv_ifftn(Gv_k)
        Gc_mR = _Gc_fftn(Gc_k)
        chi_tau = jax.lax.with_sharding_constraint(
            jnp.einsum('Rambn,Rbnam->Rmn', Gc_mR, Gv_R, optimize=True),
            NamedSharding(mesh_xy, _chi_R_spec))
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
            chi_R_shape, NamedSharding(mesh_xy, _chi_R_spec), _chi_zeros)

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
    # Intermediate sharding for the reshard from P(None,'x','y') → q_shard.
    # Routing through P('x',None,'y') (x parks on nq, y stays on μ₂) lets
    # SPMD plan it as two single-axis all_to_alls instead of the
    # "Involuntary full rematerialization" it falls into when asked to
    # un-shard both x and y on μ simultaneously via a fully replicated
    # intermediate.  Measured at Si 4×4×4 60Ry (nq=64, μ=1200, 2×2 mesh):
    #   via rep_3d: peak 2.95 GB/dev (temp 2.21 GB) — SPMD Involuntary Remat
    #   via P('x',None,'y'): peak 1.11 GB/dev (temp 0.37 GB)  -- 62% reduction
    reshard_mid = NamedSharding(mesh_xy, P('x', None, 'y'))
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
            jax.lax.with_sharding_constraint(V_padded, reshard_mid), q_shard)
        chi_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(chi_padded, reshard_mid), q_shard)

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


def _get_w_solve_fn_low_mem(mesh_xy: Mesh, nq: int, n_rmu: int, dtype):
    """Low-mem W-solve: symmetric Cholesky path.

    Trick (algebraic identity (I − AB)⁻¹ A = A (I − BA)⁻¹):
        v = X X†        (Cholesky of v)
        W = (I − v χ)⁻¹ v = X (I − X† χ X)⁻¹ X† = X H⁻¹ X†
    where H = I − X† χ X is Hermitian PD (χ ≼ 0 for static response).
    All distributed ops use the working Cholesky + trisolve kernels.

    We use this because cuSOLVERMp 0.6.0 has a 2D-grid bug in getrf/getrs
    (see ffi.cusolvermp.batched_distributed_solve_lu).  Also potentially
    2× faster than LU: 2× potrf + 1× potrs vs getrf+getrs, though the
    three extra matmuls can dominate at small N.
    """
    from ffi.cusolvermp import (
        batched_distributed_cholesky,
        batched_distributed_potrs,
        cholesky_handle_to_natural_L,
    )
    from jax.sharding import NamedSharding

    cache_key = ("low_mem", id(mesh_xy), nq, n_rmu, dtype)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    Py = int(mesh_xy.shape["y"])
    nat = NamedSharding(mesh_xy, P(None, "x", "y"))
    I_q = jnp.broadcast_to(jnp.eye(n_rmu, dtype=dtype)[None, :, :],
                            (nq, n_rmu, n_rmu))

    @jax.jit
    def _solve_w_low(V_q, chi0_q, pref):
        chi_scaled = pref * chi0_q
        V_q = jax.lax.with_sharding_constraint(V_q, nat)
        chi_scaled = jax.lax.with_sharding_constraint(chi_scaled, nat)
        # 1. X = chol(V)   [batched distributed potrf]
        X_handle = batched_distributed_cholesky(V_q, mesh=mesh_xy)
        # 2. Materialize X and X† as regular jax.Arrays.
        X = cholesky_handle_to_natural_L(X_handle)
        X_dagger = jnp.conj(jnp.swapaxes(X, -1, -2))
        # 3. T = X† χ X — two distributed matmuls.
        T = X_dagger @ chi_scaled @ X
        # 4. H = I − T  (Hermitian, expected PD)
        H = jax.lax.with_sharding_constraint(I_q - T, nat)
        # 5. L_H = chol(H)
        L_H_handle = batched_distributed_cholesky(H, mesh=mesh_xy)
        # 6. Y = potrs(L_H, X†) = H⁻¹ X†.  Two workarounds for
        #    cuSOLVERMp 0.6.0 bugs on 2D grids:
        #      (a) potrs requires B sharded P(None, 'x', 'y') — X_dagger
        #          is P(None, 'y', 'x') after the swapaxes.
        #      (b) potrs returns wrong answers when NRHS ≤ N.  Pad X_dagger
        #          along the RHS column axis so NRHS > N, then slice
        #          the result back to (nq, N, N).  Padding cost is
        #          O(1/N) relative; negligible at N ≥ 640.
        pad = 2 * Py   # keep NRHS divisible by Py; pick smallest passing
        X_dagger_padded = jnp.pad(
            X_dagger, ((0, 0), (0, 0), (0, pad)), mode="constant")
        X_dagger_padded = jax.lax.with_sharding_constraint(
            X_dagger_padded, NamedSharding(mesh_xy, P(None, "x", "y")))
        Y_padded = batched_distributed_potrs(
            L_H_handle, X_dagger_padded, mesh=mesh_xy)
        Y = Y_padded[:, :, :n_rmu]
        # 7. W = X Y
        W = X @ Y
        return jax.lax.with_sharding_constraint(W, nat)

    _w_solve_cache[cache_key] = _solve_w_low
    return _solve_w_low


def solve_w(V_q, chi0_q, meta, mesh_xy, *, memory_mode: str = "high_mem"):
    """W(q) = (I − V χ₀)⁻¹ V  via q-parallel Dyson solve.

    All arrays flat-q: V(nq, μ, μ), χ₀(nq, μ, μ) → W(nq, μ, μ).

    memory_mode:
        "high_mem": q-parallel reshard + local LU on each rank (existing
            path; legal for any mesh; uses one all-gather + all-scatter).
        "low_mem": symmetric Cholesky formulation W = X H⁻¹ X†; no
            reshard to q-parallel, but matmuls can reshard internally
            (JAX-planned).  Requires χ such that I − X†χX is PD.
    """
    nq = int(meta.nk_tot)
    n_rmu = chi0_q.shape[1]
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    pref = jnp.asarray(2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor)),
                        dtype=jnp.complex128)
    mode = (memory_mode or "high_mem").lower()
    if mode == "low_mem":
        solve_fn = _get_w_solve_fn_low_mem(mesh_xy, nq, n_rmu, V_q.dtype)
    else:
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

