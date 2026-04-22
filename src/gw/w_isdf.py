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
    from common.fft_helpers import make_flat_k_fftn

    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    cache_key = (id(mesh_xy), nkx, nky, nkz)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    # Flat-k FFT helpers — callers see only (nk, *trail) arrays.
    #
    # Historical form had Gv via ifftn (sign +ikR) and Gc via fftn (sign -ikR),
    # with einsum 'Rambn, Rbnam -> Rmn' swapping the μ_m/μ_n positions across
    # the two operands.  That forced Gc (or Gv) to reshard its μ sharding to
    # make the contracted index consistent, and landed chi_R in
    # P(None, 'y', 'x') — which then had to reshard AGAIN at the hand-off to
    # the W-solve (which consumes chi in P(None, 'x', 'y')).
    #
    # We exploit G's per-k Hermitian property ``G_k(μ,ν) = G_k(ν,μ)*``.  After
    # FT, ``G_R(μ,ν) = G_{-R}(ν,μ)*``, so running Gv's k→R with the SAME sign
    # as Gc's (both fftn, not one fft + one ifft) gives a Gv_R that equals
    # ``conj(original_Gv_R)`` with (μ_m, μ_n) swapped to the Gc-natural order.
    # The chi0 einsum then collapses to an element-wise product + spin sum:
    #
    #    chi_R(m,n) = Σ_{a,b} Gc_R(a,m,b,n) · conj(Gv_R(a,m,b,n))
    #
    # identical index order on both operands, no reshard.  Verified to
    # machine precision against the original formulation.
    #
    # Both Gs now share their natural 5-D sharding P(_, _, 'x', _, 'y')
    # (μ_first on x from psi_xn, μ_second on y from psi_yr).  chi_R inherits
    # P(_, 'x', 'y') naturally — aligned with V for W-solve, so the post-chi0
    # reshard into the fused W-solve drops out too.
    _G_spec         = P(None, None, None, None, 'x', None, 'y')    # 7-D FFT form
    _G_out_flatk    = P(None, None, 'x', None, 'y')                # 5-D flat-k form
    _chi_spec       = P(None, None, None, 'x', 'y')                # 5-D chi FFT form
    _chi_R_spec     = P(None, 'x', 'y')                            # flat-k chi

    _Gv_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _Gc_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _chi_fftn_local = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm='ortho')

    from .greens_function_kernel import build_G as _build_G_mm

    # Shardings for psi inputs (carried at the caller's natural layout; 'x' on
    # the μ_X axis for the _xn variant, 'y' on the μ_Y axis for the _yr one).
    _psi_xn_spec = P(None, None, 'x', None)   # (nk, s, μ_X, nb)
    _psi_yr_spec = P(None, None, None, 'y')   # (nk, nb, s, μ_Y)
    # Scalars / 1-D arrays replicated across all devices.
    _rep0 = P()             # scalar
    _rep1 = P(None)         # (nb,) band-indexed

    _G_k_shard = NamedSharding(mesh_xy, _G_out_flatk)
    _chi_R_shard = NamedSharding(mesh_xy, _chi_R_spec)

    @partial(jax.jit,
             in_shardings=(NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0)),
             out_shardings=(_G_k_shard, _G_k_shard))
    def _build_G(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
                 enk_v, enk_c, tau_scalar, vmax, cmin):
        phases_v = jnp.exp(-tau_scalar * (vmax - enk_v))
        phases_c = jnp.exp(-tau_scalar * (enk_c - cmin))
        Gv_k = jax.lax.with_sharding_constraint(
            _build_G_mm(psi_v_xn, psi_v_yr, phases=phases_v), _G_k_shard)
        Gc_k = jax.lax.with_sharding_constraint(
            _build_G_mm(psi_c_xn, psi_c_yr, phases=phases_c), _G_k_shard)
        return jnp.conj(Gv_k), jnp.conj(Gc_k)

    @partial(jax.jit,
             in_shardings=(NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep1),       # tau_arr
                            NamedSharding(mesh_xy, _rep1),       # prefactor_arr
                            NamedSharding(mesh_xy, _rep0),       # vmax
                            NamedSharding(mesh_xy, _rep0)),      # cmin
             out_shardings=_chi_R_shard,
             static_argnums=())
    def _chi_scan(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
                  enk_v, enk_c, tau_arr, prefactor_arr, vmax, cmin):
        """Full τ sweep in one jit: lax.scan walks the minimax nodes, builds
        Gv/Gc per τ, element-wise contracts to chi_R, accumulates, then one
        final FFT to chi_q.  All collectives and dispatch happen inside a
        single compiled graph — no Python loop over τ, no per-τ jit call."""
        n_rmu = psi_v_xn.shape[2]
        chi_R_zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128), _chi_R_shard)

        def _body(chi_R_acc, xs):
            tau_scalar, prefactor_scalar = xs
            Gv_k, Gc_k = _build_G(psi_v_xn, psi_v_yr,
                                    psi_c_yr, psi_c_xn,
                                    enk_v, enk_c, tau_scalar, vmax, cmin)
            Gv_R = _Gv_fftn(Gv_k)
            Gc_R = _Gc_fftn(Gc_k)
            # chi_R(m, n) = Σ_{a,b} Gc_R(a,m,b,n) · conj(Gv_R(a,m,b,n))
            chi_tau = jax.lax.with_sharding_constraint(
                jnp.einsum('Rambn,Rambn->Rmn',
                           Gc_R, jnp.conj(Gv_R), optimize=True),
                _chi_R_shard)
            return chi_R_acc + prefactor_scalar * chi_tau, None

        final_R, _ = jax.lax.scan(_body, chi_R_zero, (tau_arr, prefactor_arr))
        return _chi_fftn_local(final_R)

    def _chi_kernel(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
                    enk_v, enk_c, tau_i, prefactor_i, vmax, cmin):
        n_rmu = psi_v_xn.shape[2]
        if len(tau_i) == 0:
            return jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128)
        return _chi_scan(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
                         enk_v, enk_c, tau_i, prefactor_i, vmax, cmin)

    # Expose the underlying jit so ``precompile_chi0`` can call
    # ``.lower(*args).compile()`` to AOT-warm the in-process cache and
    # separate compile time from exec time in timing reports.
    _chi_kernel.compiled_fn = _chi_scan
    _chi_minimax_kernel_cache[cache_key] = _chi_kernel
    return _chi_kernel


def compute_chi0_minimax(
    psi_v_xn: jax.Array, psi_v_yr: jax.Array,
    psi_c_yr: jax.Array, psi_c_xn: jax.Array,
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
        psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
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
    """Low-mem W-solve: fused cuBLASMp + cuSOLVERMp FFI.

    One distributed FFI call runs the entire symmetric Cholesky Dyson
    solve inside the device:
        v = X X†                 (cusolverMp potrf)
        H = I − X† (pref·χ) X    (2 cublasMp gemms + identity kernel)
        L_H = chol(H)            (cusolverMp potrf)
        W = X H⁻¹ X†             (2 cublasMp trsms + 1 cublasMp gemm)

    No JAX-level intermediates, no opportunity for XLA to reshard or
    rematerialize — all matmuls are distributed-native in-device.
    """
    from ffi.cublasmp import batched_fused_w_solve
    from jax.sharding import NamedSharding

    cache_key = ("low_mem_fused", id(mesh_xy), nq, n_rmu, dtype)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    nat = NamedSharding(mesh_xy, P(None, "x", "y"))

    def _solve_w_low(V_q, chi0_q, pref):
        # The fused FFI reads pref as a compile-time complex scalar attr,
        # so pref must be a Python scalar here (not a jnp array).
        # chi0_q now comes out of compute_chi0 in P(None, 'x', 'y')
        # (same as V_q) — no reshard needed at the hand-off.
        V_q    = jax.lax.with_sharding_constraint(V_q, nat)
        chi0_q = jax.lax.with_sharding_constraint(chi0_q, nat)
        return batched_fused_w_solve(V_q, chi0_q, pref, mesh=mesh_xy)

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
    pref_scalar = 2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor))
    mode = (memory_mode or "high_mem").lower()
    if mode == "low_mem":
        # Fused FFI consumes pref as a Python scalar (compile-time attr).
        solve_fn = _get_w_solve_fn_low_mem(mesh_xy, nq, n_rmu, V_q.dtype)
        with jax_profile.annotation("W_solve"):
            return solve_fn(V_q, chi0_q, complex(pref_scalar))
    else:
        solve_fn = _get_w_solve_fn(mesh_xy, nq, n_rmu)
        pref_jnp = jnp.asarray(pref_scalar, dtype=jnp.complex128)
        with jax_profile.annotation("W_solve"):
            return solve_fn(V_q, chi0_q, pref_jnp)


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


def precompile_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=None):
    """AOT lower+compile of the χ₀ minimax kernel at the real input
    shapes/shardings — warms the JAX in-process cache so the first
    ``compute_chi0`` call is execution-only.  Call inside a dedicated
    ``timing.section('chi0_W.chi.compile')`` block to separate compile
    from exec in the end-of-run timing report.
    """
    _ensure_compilation_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    eref = 0.0 if energy_reference is None else float(energy_reference)
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax
    tau = np.asarray(quad.tau, dtype=np.float64)
    if len(tau) == 0:
        return  # compute_chi0 falls through to a static-zeros path — nothing to compile
    prefactor = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)

    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid)
    scan_jit = getattr(kernel, "compiled_fn", None)
    if scan_jit is None:
        return

    scan_jit.lower(
        wfns.xn(s.val), wfns.yr(s.val),
        wfns.yr(s.cond), wfns.xn(s.cond),
        enk_v - jnp.asarray(eref, dtype=enk_v.dtype),
        enk_c - jnp.asarray(eref, dtype=enk_c.dtype),
        jnp.asarray(tau, dtype=jnp.float64),
        jnp.asarray(prefactor, dtype=jnp.float64),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    ).compile()


def precompile_solve_w(V_q, chi0_q, meta, mesh_xy, *, memory_mode: str = "high_mem"):
    """AOT lower+compile of the W-solve jit.  See ``precompile_chi0``.

    No-op on the ``low_mem`` path: that wrapper is a plain Python call
    into the FFI primitive ``batched_fused_w_solve``, which has no XLA
    compile of its own — first-call latency there is device setup, not
    XLA compile.
    """
    _ensure_compilation_cache()
    nq = int(meta.nk_tot)
    n_rmu = chi0_q.shape[1]
    mode = (memory_mode or "high_mem").lower()
    if mode == "low_mem":
        # Still prime the wrapper cache so the FFI handle is allocated;
        # actual work happens inside the single FFI call on first invoke.
        _get_w_solve_fn_low_mem(mesh_xy, nq, n_rmu, V_q.dtype)
        return
    solve_fn = _get_w_solve_fn(mesh_xy, nq, n_rmu)
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    pref_scalar = 2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor))
    pref_jnp = jnp.asarray(pref_scalar, dtype=jnp.complex128)
    solve_fn.lower(V_q, chi0_q, pref_jnp).compile()

