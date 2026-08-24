"""Static χ₀ and W computation using ISDF + minimax quadrature.

All inter-function arrays use flat k/q indices: chi(nq, μ, μ), V(nq, μ, μ), W(nq, μ, μ).
The 3D k-grid only appears inside FFT helpers.

W Dyson solve — exactly TWO plans (input key ``w_dyson_solver``):

``local`` (default)
    q-parallel shard_map: q's scattered ``P(('x','y'),None,None)``, one
    dense pivoted LU per q on the owning rank, W constrained back out to
    ``P(None,'x','y')`` through a staged relayout.  Fast at moderate P;
    every rank must hold whole (μ, μ) tiles for its q's.
``distributed``
    2-D-sharded backsolve: A_q = 1 − V_q·χ_q formed by stacked block
    GEMMs with every operand at ``P(None,'x','y')``, factored and solved
    through the ``distrib_la`` plan door (ScaLAPACK ``pzgetrf`` /
    ``pzgetrs`` on CPU meshes, cuSOLVERMp on CUDA).  No rank ever
    materialises a full (μ, μ) tile — the memory ceiling that matters at
    thousands of low-memory processes.  W lands natively in
    ``P(None,'x','y')`` (no relayout).

Two-face carrier (``low_mem_bands = true``, ``wfns.layout == "face"``):
:func:`_get_chi_minimax_kernel` dispatches to
:func:`_get_chi_minimax_kernel_legacy` (untouched) or
:func:`_get_chi_minimax_kernel_face`, which builds G via a single planned
``distrib_la.gemm_plan`` shared by Gv/Gc (the val/cond split becomes a
band-identity mask over the FULL loaded extent, not two differently-sized
slices — report §3) and reuses the legacy FFT/contraction stages
unchanged (G's output shape/sharding does not depend on layout).
:func:`_chi_layout_operands`/:func:`_chi_face_kwargs` are the one place
:func:`compute_chi0`/:func:`_chi0_multi_kernel_args`/
:func:`_chi0_contour_kernel_args` turn a bundle into kernel operands, so
those callers no longer extract legacy views (``.xn()``/``.yr()``)
themselves.

Exact finite-occupation response (``feat/metal-response-face-2026-08-23``,
``docs/architecture/fractional_chi0_response_face.md``): also ported.
``_get_chi_fractional_contour_kernel`` dispatches to
``_get_chi_fractional_contour_kernel_legacy`` (untouched) or
``_get_chi_fractional_contour_kernel_face`` — a substitution of operands
onto the SAME ``build_G_tau(layout='face', ...)`` mechanism the ordinary
minimax kernel already uses, since its two Green's functions are each a
one-particle, band_weight-diagonal contraction.  The ordered-pair kernel
(``_fractional_pair_scan`` / :func:`_get_chi_static_fractional_gamma_kernel`
/ :func:`_get_chi_fractional_q_kernel`) needed a genuinely new mechanism —
its divided-difference weight depends JOINTLY on both band indices and
cannot collapse to a one-particle GEMM — and gets one:
``_fractional_pair_scan_face`` reconstructs each band tile it touches from
the persistent ``psi_mun``/``psi_nmu`` via a masked-gather + ``psum`` on
BOTH mesh axes (``isdf.core._z_q_face``'s idiom, generalized), never a
resident single-axis copy.  See the design doc for the full derivation and
why ``distrib_la.gemm_plan``/``GemmPlan.local_call`` do not apply here.
"""
import os
from dataclasses import dataclass
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
from common.jax_compile_cache import ensure_jax_compile_cache
from runtime.padding import round_up, solve_at_logical
from .efermi import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                     band_in_occupation_window, occupation_weight_floor)
from .minimax_screening import MinimaxNodes


# ============================================================================
# Cache and sharding registry
# ============================================================================

_chi_minimax_kernel_cache: dict = {}
_w_solve_cache: dict = {}

# The static metal fallback streams one small ordered band-pair tile at a
# time.  This is deliberately a compile-time constant: changing it is a
# memory/performance choice, not physics or an input-file option.
_STATIC_FRACTIONAL_PAIR_TILE = 8



# ============================================================================
# χ₀ kernel — minimax quadrature
# ============================================================================

def _get_chi_minimax_kernel(mesh_xy: Mesh, kgrid: tuple[int, int, int],
                            n_out: int = 1, *,
                            complex_contour: bool = False,
                            layout: str = "legacy", face_shape=None,
                            right_face_shape=None,
                            vertex_pair: bool = False,
                            vertex_identity=(False, False)):
    """Build chi0 kernel with device-local FFTs.  Returns flat-q χ₀(nq, μ, μ).

    ``n_out`` (static): number of χ outputs accumulated over the SAME τ
    sweep.  ``n_out=1`` is the historical kernel, untouched.  ``n_out>=2``
    (the probe-χ₀ reuse path, ``ppm_probe_chi_reuse=auto``) takes
    ``nodes.alpha`` of shape ``(n_out, L)`` and returns an ``n_out``-tuple
    of χ's — the per-τ G-build/FFT/contraction tensors are computed once
    and each output is its own weighted accumulation.

    ``complex_contour=True`` carries complex time through the same spatial
    contraction.  The cache key keeps that convention separate from the
    bit-locked real-time static path.

    ``layout`` (static): ``'legacy'`` (default) is the exact kernel this
    module has always built — see :func:`_get_chi_minimax_kernel_legacy`.
    ``'face'`` builds the two-face carrier's G-construction instead (one
    planned ``distrib_la.gemm_plan`` per kernel, shared by Gv/Gc since
    face G no longer slices a band window — see report §3); requires
    ``face_shape=(nk, nb_full, n_rmu, nspinor)``.  Its ordinary scalar
    branch preserves the established FFT/contraction sequence; the optional
    four-current face branch changes only the open-spin trace to insert the
    canonical Lorentz vertices.  See :func:`_get_chi_minimax_kernel_face`.
    """
    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    n_out = int(n_out)
    # ffi_dial_key(): the make_flat_k_fftn factories below read
    # LORRAX_FFT_FFI at FACTORY time, so the dials must be part of this
    # cache key or a mid-process flag flip serves the stale backend
    # (flat-k FFT service contract, docs/dev/flat_k_fft_service.md).
    from ffi import ffi_dial_key
    complex_contour = bool(complex_contour)
    if layout not in ("legacy", "face"):
        raise ValueError(
            f"_get_chi_minimax_kernel: layout must be 'legacy' or 'face', "
            f"got {layout!r}")
    vertex_pair = bool(vertex_pair)
    vertex_identity = tuple(bool(x) for x in vertex_identity)
    if len(vertex_identity) != 2:
        raise ValueError("vertex_identity must be (left_identity,right_identity)")
    if vertex_pair and n_out != 1:
        raise ValueError(
            "four-current vertex chi currently supports one static output "
            "per tau sweep")
    if vertex_pair and layout != "face":
        raise ValueError(
            "four-current vertex chi requires the canonical face "
            "wavefunction layout")
    cache_key = (id(mesh_xy), kgrid, ffi_dial_key(), n_out,
                 complex_contour, layout, face_shape, right_face_shape,
                 vertex_pair, vertex_identity)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    if layout == "legacy":
        kernel = _get_chi_minimax_kernel_legacy(
            mesh_xy, kgrid, nk, n_out, complex_contour)
    else:
        if face_shape is None:
            raise ValueError(
                "_get_chi_minimax_kernel(layout='face') requires "
                "face_shape=(nk, nb_full, n_rmu, nspinor)")
        kernel = _get_chi_minimax_kernel_face(
            mesh_xy, kgrid, nk, n_out, complex_contour, face_shape,
            right_face_shape=right_face_shape,
            vertex_pair=vertex_pair,
            vertex_identity=vertex_identity)
    _chi_minimax_kernel_cache[cache_key] = kernel
    return kernel


def _get_chi_minimax_kernel_legacy(mesh_xy, kgrid, nk, n_out, complex_contour):
    """The exact pre-``low_mem_bands`` kernel body, moved verbatim out of
    :func:`_get_chi_minimax_kernel` so that dispatcher could gain a
    layout branch without touching this one at all.  Caching is now the
    dispatcher's job — this function only builds."""
    from common.fft_helpers import make_flat_k_fftn

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
    from .wavefunction_bundle import (
        G_FFT7D_SPEC as _G_spec,
        G_FLATK_SPEC as _G_out_flatk,
        CHI_Q_SPEC as _chi_spec,
        CHI_R_SPEC as _chi_R_spec,
        PSI_XN_SPEC as _psi_xn_spec,
        PSI_YR_SPEC as _psi_yr_spec,
    )
    _Gv_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _Gc_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _chi_fftn_local = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm='ortho')

    from .greens_function_kernel import build_G_tau
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
    def _build_Gv_Gc(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
                    enk_v, enk_c, tau_scalar, vmax, cmin):
        # The returned conjugate changes a raw builder time t into conj(t).
        # Complex contour nodes therefore need raw t_c=conj(τ), while the
        # locked real path continues to receive t_c=τ exactly as before.
        t_c = jnp.conj(tau_scalar) if complex_contour else tau_scalar
        Gv_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_v_xn, psi_v_yr, enk_v, -tau_scalar, e_ref=vmax),
            _G_k_shard)
        Gc_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_c_xn, psi_c_yr, enk_c, t_c, e_ref=cmin),
            _G_k_shard)
        # Hermitian-swap conj (see FFT-convention block comment above) —
        # belongs at the call site, NOT inside build_G_tau.
        return jnp.conj(Gv_k), jnp.conj(Gc_k)

    # MinimaxNodes pytree (t, alpha) — both replicated across devices.
    # n_out>=2: alpha is (n_out, L); P() replicates every axis.
    _nodes_shard = MinimaxNodes(
        t=NamedSharding(mesh_xy, _rep1),
        alpha=NamedSharding(mesh_xy, _rep1 if n_out == 1 else P()),
    )

    if n_out >= 2:
        _chi_R_out = tuple(_chi_R_shard for _ in range(n_out))

        @partial(jax.jit,
                 in_shardings=(_nodes_shard,
                                NamedSharding(mesh_xy, _psi_xn_spec),
                                NamedSharding(mesh_xy, _psi_yr_spec),
                                NamedSharding(mesh_xy, _psi_yr_spec),
                                NamedSharding(mesh_xy, _psi_xn_spec),
                                NamedSharding(mesh_xy, _rep1),
                                NamedSharding(mesh_xy, _rep1),
                                NamedSharding(mesh_xy, _rep0),      # vmax
                                NamedSharding(mesh_xy, _rep0)),     # cmin
                 out_shardings=_chi_R_out,
                 static_argnums=())
        def minimax_tau_integrate_chi_multi(
            nodes, psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
            enk_v, enk_c, vmax, cmin,
        ):
            """n_out-output sibling of ``minimax_tau_integrate_chi``: ONE τ
            sweep (identical per-node Gv/Gc + FFTs + contraction), n_out
            weighted accumulators, one R→q FFT per output.  Bit-parity
            with the single kernel is NOT contractual (different XLA
            program); the consumer (probe-χ₀ reuse) is gated on the
            quadrature-error contract instead."""
            n_rmu = psi_v_xn.shape[2]
            zero = jax.lax.with_sharding_constraint(
                jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128),
                _chi_R_shard)
            acc0 = tuple(zero for _ in range(n_out))

            def _body(accs, xs):
                t_scalar, alpha_col = xs        # alpha_col: (n_out,)
                tau_kernel = (t_scalar if complex_contour else
                              jnp.real(t_scalar).astype(jnp.float64))
                Gv_k, Gc_k = _build_Gv_Gc(psi_v_xn, psi_v_yr,
                                          psi_c_yr, psi_c_xn,
                                          enk_v, enk_c, tau_kernel, vmax, cmin)
                Gv_R = _Gv_fftn(Gv_k)
                Gc_R = _Gc_fftn(Gc_k)
                chi_tau = jax.lax.with_sharding_constraint(
                    jnp.einsum('Rambn,Rambn->Rmn',
                               Gc_R, jnp.conj(Gv_R), optimize=True),
                    _chi_R_shard)
                return tuple(a + alpha_col[i] * chi_tau
                             for i, a in enumerate(accs)), None

            final_R, _ = jax.lax.scan(
                _body, acc0, (nodes.t, jnp.transpose(nodes.alpha)), unroll=1)
            return tuple(_chi_fftn_local(f) for f in final_R)

        return minimax_tau_integrate_chi_multi

    @partial(jax.jit,
             in_shardings=(_nodes_shard,
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep0),       # vmax
                            NamedSharding(mesh_xy, _rep0)),      # cmin
             out_shardings=_chi_R_shard,
             static_argnums=())
    def minimax_tau_integrate_chi(
        nodes, psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
        enk_v, enk_c, vmax, cmin,
    ):
        """Full τ sweep accumulating χ_R, then one R→q FFT.

        Sibling of ``ppm_sigma.minimax_tau_integrate_sigma`` — takes a
        ``MinimaxNodes`` pytree in the same slot.  For chi0 the nodes
        arrive with purely-real τ (``time_axis='real'``) and complex α
        whose Im part is zero; ``alpha`` already includes the chi0
        prefactor ``-2·α_quad·exp(-τ·E_gap)`` so the scan body only
        scales the per-τ contraction.

        For each τ node: build Gv, Gc via build_G_tau; FFT both to R;
        element-wise contract (Σ_{a,b} Gc_R · conj(Gv_R)) into chi_R;
        accumulate weighted by α; final back-FFT to q.  All collectives
        and dispatch happen inside one compiled graph — no Python loop.
        """
        n_rmu = psi_v_xn.shape[2]
        chi_R_zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128), _chi_R_shard)

        def _body(chi_R_acc, xs):
            t_scalar, alpha_scalar = xs
            # ``t`` arrives complex (pytree dtype); chi0's Laplace quad
            # places it with Im=0.  Cast to float64 so _build_Gv_Gc's
            # float64 tau signature — and build_G_tau's downstream exp —
            # stay on the exact numerical path that produced the locked
            # MoS2 3×3 regression hash.
            tau_kernel = (t_scalar if complex_contour else
                          jnp.real(t_scalar).astype(jnp.float64))
            Gv_k, Gc_k = _build_Gv_Gc(psi_v_xn, psi_v_yr,
                                      psi_c_yr, psi_c_xn,
                                      enk_v, enk_c, tau_kernel, vmax, cmin)
            Gv_R = _Gv_fftn(Gv_k)
            Gc_R = _Gc_fftn(Gc_k)
            # chi_R(m, n) = Σ_{a,b} Gc_R(a,m,b,n) · conj(Gv_R(a,m,b,n))
            chi_tau = jax.lax.with_sharding_constraint(
                jnp.einsum('Rambn,Rambn->Rmn',
                           Gc_R, jnp.conj(Gv_R), optimize=True),
                _chi_R_shard)
            # α is complex; its Im part is zero for the chi0 Laplace
            # window.  Multiplying complex·complex is identical to
            # float·complex at the hardware level when Im(α)=0.
            return chi_R_acc + alpha_scalar * chi_tau, None

        final_R, _ = jax.lax.scan(
            _body, chi_R_zero, (nodes.t, nodes.alpha), unroll=1)
        return _chi_fftn_local(final_R)

    # Minimax quadrature always delivers ≥1 node — the compiled scan
    # handles any n≥1 without a short-circuit wrapper.
    return minimax_tau_integrate_chi


def _get_chi_minimax_kernel_face(mesh_xy, kgrid, nk, n_out, complex_contour,
                                 face_shape, *, right_face_shape=None,
                                 vertex_pair=False,
                                 vertex_identity=(False, False)):
    """Face-layout sibling of :func:`_get_chi_minimax_kernel_legacy`.

    G construction is the only part that forks (module-level docstring):
    ``psi_mun``/``psi_nmu`` cover the FULL [b0,b4) band range (obstacle
    #3 — a legal face matrix cannot be sliced to val/cond), so the
    val/cond split that legacy expresses as two DIFFERENTLY-SIZED band
    slices becomes two full-extent G builds gated by a band-IDENTITY
    ``mask`` (``Wavefunctions.band_mask`` — this is exactly
    ``build_G_tau``'s own pre-existing ``mask`` parameter, designed for
    Σ's val/cond selector; chi0 simply becomes its second user).  ONE
    ``distrib_la.gemm_plan`` serves BOTH Gv and Gc, since both now share
    the same (mu*ns, nb_full, mu*ns) shape — the legacy kernel needed two
    only because its val/cond slices differ in size.  FFT/contraction
    stages below are copied from the legacy body UNCHANGED (same specs,
    same einsum) because G's output shape/sharding is layout-independent.
    """
    from common.fft_helpers import make_flat_k_fftn
    from distrib_la import gemm_plan
    from .wavefunction_bundle import (
        G_FFT7D_SPEC as _G_spec,
        G_FLATK_SPEC as _G_out_flatk,
        CHI_Q_SPEC as _chi_spec,
        CHI_R_SPEC as _chi_R_spec,
        PSI_MUN_SPEC as _psi_mun_spec,
        PSI_NMU_SPEC as _psi_nmu_spec,
    )
    from .greens_function_kernel import build_G_tau

    nk_shape, nb_full, n_rmu_left, ns = (int(v) for v in face_shape)
    if right_face_shape is None:
        right_face_shape = face_shape
    nk_right, nb_right, n_rmu_right, ns_right = (
        int(v) for v in right_face_shape)
    if nk_shape != nk:
        raise ValueError(
            f"_get_chi_minimax_kernel_face: face_shape nk={nk_shape} does "
            f"not match kgrid's prod={nk}")
    if (nk_right, nb_right, ns_right) != (nk, nb_full, ns):
        raise ValueError(
            "_get_chi_minimax_kernel_face: left/right endpoint bundles "
            "must share (nk, nb_full, nspinor); got "
            f"{face_shape} and {right_face_shape}")

    _Gv_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _Gc_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _chi_fftn_local = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm='ortho')

    _rep0 = P()
    _rep1 = P(None)
    _rep2 = P(None, None)
    _G_k_shard = NamedSharding(mesh_xy, _G_out_flatk)
    _chi_R_shard = NamedSharding(mesh_xy, _chi_R_spec)
    _psi_mun_shard = NamedSharding(mesh_xy, _psi_mun_spec)
    _psi_nmu_shard = NamedSharding(mesh_xy, _psi_nmu_spec)

    # ONE planned GEMM, built here (eagerly, once) and shared by every Gv
    # and Gc build this kernel ever does — see distrib_la.gemm_plan's own
    # "hoist this call out of every per-k/per-tau loop" instruction.
    g_plan = gemm_plan(
        mesh_xy, m=n_rmu_left * ns, k=nb_full,
        n=n_rmu_right * ns, nq=nk, dtype=jnp.complex128)

    @partial(jax.jit,
             in_shardings=(_psi_mun_shard, _psi_nmu_shard,
                            NamedSharding(mesh_xy, _rep2),   # mask_v
                            NamedSharding(mesh_xy, _rep2),   # mask_c
                            NamedSharding(mesh_xy, _rep2),   # enk_full
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0)),
             out_shardings=(_G_k_shard, _G_k_shard))
    def _build_Gv_Gc(psi_mun_left, psi_nmu_right,
                    mask_v, mask_c, enk_full,
                    tau_scalar, vmax, cmin):
        t_c = jnp.conj(tau_scalar) if complex_contour else tau_scalar
        Gv_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_mun_left, psi_nmu_right, enk_full,
                       -tau_scalar, e_ref=vmax,
                       mask=mask_v, layout="face", gemm=g_plan),
            _G_k_shard)
        Gc_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_mun_left, psi_nmu_right, enk_full,
                       t_c, e_ref=cmin,
                       mask=mask_c, layout="face", gemm=g_plan),
            _G_k_shard)
        return jnp.conj(Gv_k), jnp.conj(Gc_k)

    _nodes_shard = MinimaxNodes(
        t=NamedSharding(mesh_xy, _rep1),
        alpha=NamedSharding(mesh_xy, _rep1 if n_out == 1 else P()),
    )

    if n_out >= 2:
        _chi_R_out = tuple(_chi_R_shard for _ in range(n_out))

        @partial(jax.jit,
                 in_shardings=(_nodes_shard, _psi_mun_shard, _psi_nmu_shard,
                                NamedSharding(mesh_xy, _rep2),
                                NamedSharding(mesh_xy, _rep2),
                                NamedSharding(mesh_xy, _rep2),
                                NamedSharding(mesh_xy, _rep0),
                                NamedSharding(mesh_xy, _rep0)),
                 out_shardings=_chi_R_out,
                 static_argnums=())
        def minimax_tau_integrate_chi_multi(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
        ):
            zero = jax.lax.with_sharding_constraint(
                jnp.zeros((nk, n_rmu_left, n_rmu_right),
                          dtype=jnp.complex128),
                _chi_R_shard)
            acc0 = tuple(zero for _ in range(n_out))

            def _body(accs, xs):
                t_scalar, alpha_col = xs
                tau_kernel = (t_scalar if complex_contour else
                              jnp.real(t_scalar).astype(jnp.float64))
                Gv_k, Gc_k = _build_Gv_Gc(psi_mun, psi_nmu, mask_v, mask_c,
                                          enk_full, tau_kernel, vmax, cmin)
                Gv_R = _Gv_fftn(Gv_k)
                Gc_R = _Gc_fftn(Gc_k)
                chi_tau = jax.lax.with_sharding_constraint(
                    jnp.einsum('Rambn,Rambn->Rmn',
                               Gc_R, jnp.conj(Gv_R), optimize=True),
                    _chi_R_shard)
                return tuple(a + alpha_col[i] * chi_tau
                             for i, a in enumerate(accs)), None

            final_R, _ = jax.lax.scan(
                _body, acc0, (nodes.t, jnp.transpose(nodes.alpha)), unroll=1)
            return tuple(_chi_fftn_local(f) for f in final_R)

        return minimax_tau_integrate_chi_multi

    if vertex_pair:
        from common.gamma_matrices import gamma_double_contract
    left_identity, right_identity = vertex_identity

    def _single_impl(
        nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
        vertex_operands,
    ):
        chi_R_zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_rmu_left, n_rmu_right),
                      dtype=jnp.complex128), _chi_R_shard)

        def _body(chi_R_acc, xs):
            t_scalar, alpha_scalar = xs
            tau_kernel = (t_scalar if complex_contour else
                          jnp.real(t_scalar).astype(jnp.float64))
            Gv_k, Gc_k = _build_Gv_Gc(psi_mun, psi_nmu, mask_v, mask_c,
                                      enk_full, tau_kernel, vmax, cmin)
            Gv_R = _Gv_fftn(Gv_k)
            Gc_R = _Gc_fftn(Gc_k)
            if vertex_operands is None:
                chi_tau_raw = jnp.einsum(
                    'Rambn,Rambn->Rmn',
                    Gc_R, jnp.conj(Gv_R), optimize=True)
            else:
                perm_l, phase_l, perm_r, phase_r = vertex_operands
                chi_tau_raw = gamma_double_contract(
                    jnp.conj(Gv_R), Gc_R,
                    perm_L=None if left_identity else perm_l,
                    phase_L=None if left_identity else phase_l,
                    # Right endpoint orientation: the trace uses
                    # Gamma_B[c,d], whereas the helper's row form is
                    # Gamma_B[d,c].  Canonical alpha matrices are Hermitian
                    # monomials, so conjugating the row phase transposes it.
                    perm_R=None if right_identity else perm_r,
                    phase_R=(None if right_identity else jnp.conj(phase_r)),
                    spin_axes=(1, 3))
            chi_tau = jax.lax.with_sharding_constraint(
                chi_tau_raw, _chi_R_shard)
            return chi_R_acc + alpha_scalar * chi_tau, None

        final_R, _ = jax.lax.scan(
            _body, chi_R_zero, (nodes.t, nodes.alpha), unroll=1)
        return _chi_fftn_local(final_R)

    _base_in = (_nodes_shard, _psi_mun_shard, _psi_nmu_shard,
                NamedSharding(mesh_xy, _rep2),
                NamedSharding(mesh_xy, _rep2),
                NamedSharding(mesh_xy, _rep2),
                NamedSharding(mesh_xy, _rep0),
                NamedSharding(mesh_xy, _rep0))

    if vertex_pair:
        @partial(jax.jit,
                 in_shardings=_base_in + tuple(
                     NamedSharding(mesh_xy, _rep1) for _ in range(4)),
                 out_shardings=_chi_R_shard)
        def minimax_tau_integrate_chi_vertex(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
            perm_l, phase_l, perm_r, phase_r,
        ):
            return _single_impl(
                nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full,
                vmax, cmin, (perm_l, phase_l, perm_r, phase_r))

        return minimax_tau_integrate_chi_vertex

    @partial(jax.jit, in_shardings=_base_in, out_shardings=_chi_R_shard)
    def minimax_tau_integrate_chi(
        nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
    ):
        return _single_impl(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full,
            vmax, cmin, None)

    return minimax_tau_integrate_chi




def _get_chi_fractional_contour_kernel(
    mesh_xy: Mesh, kgrid: tuple[int, int, int], n_out: int,
    *, layout: str = "legacy", face_shape=None,
):
    """Retarded finite-occupation chi0 on one positive-time sweep.

    ``layout`` (static): ``'legacy'`` (default) is the exact kernel this
    module has always built — see
    :func:`_get_chi_fractional_contour_kernel_legacy`.  ``'face'`` builds
    the two-face carrier's version — a substitution of operands onto the
    SAME ``build_G_tau(layout='face', ...)`` mechanism the ordinary
    minimax kernel already uses (this kernel's two Green's functions are
    each a one-particle, ``band_weight``-diagonal contraction, so no new
    distributed algorithm is needed here — see
    :func:`_get_chi_fractional_contour_kernel_face` and
    ``docs/architecture/fractional_chi0_response_face.md``); requires
    ``face_shape=(nk, nb_full, n_rmu, nspinor)``.  Mirrors
    ``_get_chi_minimax_kernel``'s own legacy/face dispatcher split — the
    cache-management lines below moved here from the (now pure-builder)
    legacy body, exactly as that split's own precedent.
    """
    from ffi import ffi_dial_key

    grid = tuple(int(n) for n in kgrid)
    n_out = int(n_out)
    if n_out < 1:
        raise ValueError("fractional contour chi0 requires at least one output")
    if layout not in ("legacy", "face"):
        raise ValueError(
            f"_get_chi_fractional_contour_kernel: layout must be 'legacy' "
            f"or 'face', got {layout!r}")
    cache_key = ("fractional_contour", id(mesh_xy), grid, ffi_dial_key(),
                 n_out, layout, face_shape)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    if layout == "legacy":
        kernel = _get_chi_fractional_contour_kernel_legacy(mesh_xy, grid, n_out)
    else:
        if face_shape is None:
            raise ValueError(
                "_get_chi_fractional_contour_kernel(layout='face') requires "
                "face_shape=(nk, nb_full, n_rmu, nspinor)")
        kernel = _get_chi_fractional_contour_kernel_face(
            mesh_xy, grid, n_out, face_shape)
    _chi_minimax_kernel_cache[cache_key] = kernel
    return kernel


def _get_chi_fractional_contour_kernel_legacy(
    mesh_xy: Mesh, kgrid: tuple[int, int, int], n_out: int,
):
    """The exact pre-``low_mem_bands`` kernel body, moved verbatim out of
    :func:`_get_chi_fractional_contour_kernel` (only the cache-key/lookup/
    store lines were removed — that bookkeeping is now the dispatcher's
    job, mirroring ``_get_chi_minimax_kernel_legacy``'s own split) so the
    dispatcher could gain a layout branch without touching this one at
    all.

    At each node this builds only

        A_q(t) = sum_ab f_a (1-f_b) exp[-i(E_b-E_a)t] X_ab(q)

    from two single-band Green sums.  The other Keldysh product is not a
    second pair build: Hermiticity of the density vertex gives

        B_q(t) = A_{-q}(-t)^T.

    With the FFT convention used here, FFT_R[conj(A_R(t))](q) is exactly
    conj(A_{-q}(t)) = A_{-q}(-t)^T.  This is an explicit full-grid
    q-orientation identity, not time-reversal symmetry, and is therefore
    valid for broken-TRS spinor calculations.  The two products meet only
    after both band sums have disappeared.
    """
    from common.fft_helpers import make_flat_k_fftn
    from .greens_function_kernel import build_G_tau
    from .wavefunction_bundle import (
        G_FFT7D_SPEC,
        G_FLATK_SPEC,
        CHI_Q_SPEC,
        CHI_R_SPEC,
        PSI_XN_SPEC,
        PSI_YR_SPEC,
    )

    grid = tuple(int(n) for n in kgrid)
    nk = int(np.prod(grid))
    n_out = int(n_out)

    G_fftn = make_flat_k_fftn(
        mesh_xy, grid, G_FFT7D_SPEC, norm="ortho")
    chi_fftn = make_flat_k_fftn(
        mesh_xy, grid, CHI_Q_SPEC, norm="ortho")
    G_shard = NamedSharding(mesh_xy, G_FLATK_SPEC)
    chi_R_shard = NamedSharding(mesh_xy, CHI_R_SPEC)
    rep2 = NamedSharding(mesh_xy, P(None, None))
    rep1 = NamedSharding(mesh_xy, P(None))
    rep0 = NamedSharding(mesh_xy, P())

    @partial(
        jax.jit,
        in_shardings=(
            rep1, rep0,
            NamedSharding(mesh_xy, PSI_XN_SPEC),
            NamedSharding(mesh_xy, PSI_YR_SPEC),
            NamedSharding(mesh_xy, PSI_YR_SPEC),
            NamedSharding(mesh_xy, PSI_XN_SPEC),
            rep2, rep2, rep2, rep2, rep0,
        ),
        out_shardings=tuple(chi_R_shard for _ in range(n_out)),
    )
    def integrate(
        time_nodes,
        projection_rows,
        psi_f_xn,
        psi_f_yr,
        psi_u_yr,
        psi_u_xn,
        enk_f,
        enk_u,
        occ_f,
        occ_u,
        energy_reference,
    ):
        n_mu = psi_f_xn.shape[2]
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_mu, n_mu), dtype=jnp.complex128),
            chi_R_shard,
        )
        initial = tuple(zero for _ in range(n_out))

        def body(accumulators, node):
            time, projection = node
            tau = jnp.asarray(1j, dtype=jnp.complex128) * time
            Gf_k = jax.lax.with_sharding_constraint(
                jnp.conj(build_G_tau(
                    psi_f_xn,
                    psi_f_yr,
                    enk_f,
                    -tau,
                    e_ref=energy_reference,
                    band_weight=occ_f,
                )),
                G_shard,
            )
            Gu_k = jax.lax.with_sharding_constraint(
                jnp.conj(build_G_tau(
                    psi_u_xn,
                    psi_u_yr,
                    enk_u,
                    -tau,
                    e_ref=energy_reference,
                    band_weight=1.0 - occ_u,
                )),
                G_shard,
            )
            Gf_R = G_fftn(Gf_k)
            Gu_R = G_fftn(Gu_k)
            A_R = jax.lax.with_sharding_constraint(
                jnp.einsum(
                    "Rambn,Rambn->Rmn",
                    Gu_R,
                    jnp.conj(Gf_R),
                    optimize=True,
                ),
                chi_R_shard,
            )
            reverse_R = jnp.conj(A_R)
            updated = tuple(
                accumulators[i]
                - 1j * projection[i] * A_R
                + 1j * projection[i] * reverse_R
                for i in range(n_out)
            )
            return updated, None

        final_R, _ = jax.lax.scan(
            body,
            initial,
            (time_nodes, jnp.transpose(projection_rows)),
            unroll=1,
        )
        return tuple(chi_fftn(value) for value in final_R)

    return integrate


def _get_chi_fractional_contour_kernel_face(
    mesh_xy: Mesh, kgrid: tuple[int, int, int], n_out: int, face_shape,
):
    """Face-layout sibling of
    :func:`_get_chi_fractional_contour_kernel_legacy`.  Same Keldysh
    identity (see that function's docstring); the two per-node Green's
    functions (``Gf`` weighted by ``occ_f``, ``Gu`` weighted by
    ``occ_u``) are each an ordinary ONE-PARTICLE ``build_G_tau``
    contraction — this kernel is therefore a substitution of operands
    (``psi_mun``/``psi_nmu`` instead of the four legacy views) onto the
    SAME mechanism the ordinary minimax kernel's face port already ships,
    not a new algorithm.  ONE ``distrib_la.gemm_plan`` (shared by both
    ``Gf`` and ``Gu``, mirroring ``_get_chi_minimax_kernel_face``'s own
    ``g_plan`` shared by ``Gv``/``Gc``) is built ONCE here, eagerly.

    Legacy slices ψ down to the ``f_slice``/``u_slice`` occupation-support
    window (a genuine cost cut — fewer bands enter the contraction); a
    face carrier cannot be band-sliced (obstacle #3), so the CALLER
    (:func:`_chi0_fractional_contour_args`) instead precomputes the FINAL
    ``occ_f``/``occ_u`` band_weight values — ``occ`` and ``1-occ``
    respectively, legacy's own two conventions — zero-weighted outside
    that same window before calling here — "weight, don't window", the
    SAME convention ``isdf.core._c_q_face`` uses for its own L/R window
    (see the design doc).  ``occ_u`` therefore already IS ``1-occ``, not
    a raw occupation this function must itself invert — applying a
    SECOND ``1-x`` here reintroduces exactly the bug this comment is
    warning against (see ``integrate``'s own comment, found via the Na
    production-shape harness).  This function always runs the FULL
    ``nb_full`` contraction; it does not know or care that the caller
    has already zeroed part of the weight.
    """
    from common.fft_helpers import make_flat_k_fftn
    from distrib_la import gemm_plan
    from .greens_function_kernel import build_G_tau
    from .wavefunction_bundle import (
        G_FFT7D_SPEC,
        G_FLATK_SPEC,
        CHI_Q_SPEC,
        CHI_R_SPEC,
        PSI_MUN_SPEC,
        PSI_NMU_SPEC,
    )

    grid = tuple(int(n) for n in kgrid)
    nk = int(np.prod(grid))
    n_out = int(n_out)
    nk_shape, nb_full, n_rmu, ns = (int(v) for v in face_shape)
    if nk_shape != nk:
        raise ValueError(
            f"_get_chi_fractional_contour_kernel_face: face_shape "
            f"nk={nk_shape} does not match kgrid's prod={nk}")

    G_fftn = make_flat_k_fftn(
        mesh_xy, grid, G_FFT7D_SPEC, norm="ortho")
    chi_fftn = make_flat_k_fftn(
        mesh_xy, grid, CHI_Q_SPEC, norm="ortho")
    G_shard = NamedSharding(mesh_xy, G_FLATK_SPEC)
    chi_R_shard = NamedSharding(mesh_xy, CHI_R_SPEC)
    rep2 = NamedSharding(mesh_xy, P(None, None))
    rep1 = NamedSharding(mesh_xy, P(None))
    rep0 = NamedSharding(mesh_xy, P())
    psi_mun_shard = NamedSharding(mesh_xy, PSI_MUN_SPEC)
    psi_nmu_shard = NamedSharding(mesh_xy, PSI_NMU_SPEC)

    # ONE planned GEMM, built here (eagerly, once) and shared by every Gf
    # and Gu build this kernel ever does — mirrors
    # _get_chi_minimax_kernel_face's own g_plan.
    g_plan = gemm_plan(mesh_xy, m=n_rmu * ns, k=nb_full, n=n_rmu * ns,
                       nq=nk, dtype=jnp.complex128)

    @partial(
        jax.jit,
        in_shardings=(
            rep1, rep0,
            psi_mun_shard, psi_nmu_shard,
            rep2, rep2, rep2, rep0,
        ),
        out_shardings=tuple(chi_R_shard for _ in range(n_out)),
    )
    def integrate(
        time_nodes,
        projection_rows,
        psi_mun,
        psi_nmu,
        enk_full,
        occ_f,
        occ_u,
        energy_reference,
    ):
        # occ_f/occ_u are the FINAL band_weight values, applied directly
        # (no further transform here) -- the caller
        # (_chi0_fractional_contour_args) has ALREADY done both jobs: the
        # physical weight (occ_f is occ; occ_u is 1-occ -- legacy's own
        # Gu_k convention, inverted BEFORE masking, not after) AND the
        # "weight, don't window" support indicator (zero outside
        # f_slice/u_slice).  Do NOT write `1.0 - occ_u` here: occ_u is
        # already `(1-occ)*indicator`, and re-subtracting from 1.0 would
        # turn the correctly-zeroed OUTSIDE-the-window positions into 1.0
        # instead of 0.0, silently pulling every band outside u_slice
        # into Gu_k's contraction (found and fixed 2026-08-23 via the Na
        # production-shape harness: max_rel 0.19 -> 1e-16 at nb_full=48,
        # 98.7% of the (mu,mu) output entries above 1% of the max diff --
        # not a localized bug, every u_slice-adjacent band was wrong).
        # enk_full is the bundle's own full [b0,b4) energy table, shared
        # by both roles -- legacy's enk_f/enk_u are always the SAME array
        # sliced twice, so this is not a new sharing, only a name
        # simplification.
        n_mu = psi_mun.shape[2]
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_mu, n_mu), dtype=jnp.complex128),
            chi_R_shard,
        )
        initial = tuple(zero for _ in range(n_out))

        def body(accumulators, node):
            time, projection = node
            tau = jnp.asarray(1j, dtype=jnp.complex128) * time
            Gf_k = jax.lax.with_sharding_constraint(
                jnp.conj(build_G_tau(
                    psi_mun,
                    psi_nmu,
                    enk_full,
                    -tau,
                    e_ref=energy_reference,
                    band_weight=occ_f,
                    layout="face",
                    gemm=g_plan,
                )),
                G_shard,
            )
            Gu_k = jax.lax.with_sharding_constraint(
                jnp.conj(build_G_tau(
                    psi_mun,
                    psi_nmu,
                    enk_full,
                    -tau,
                    e_ref=energy_reference,
                    band_weight=occ_u,
                    layout="face",
                    gemm=g_plan,
                )),
                G_shard,
            )
            Gf_R = G_fftn(Gf_k)
            Gu_R = G_fftn(Gu_k)
            A_R = jax.lax.with_sharding_constraint(
                jnp.einsum(
                    "Rambn,Rambn->Rmn",
                    Gu_R,
                    jnp.conj(Gf_R),
                    optimize=True,
                ),
                chi_R_shard,
            )
            reverse_R = jnp.conj(A_R)
            updated = tuple(
                accumulators[i]
                - 1j * projection[i] * A_R
                + 1j * projection[i] * reverse_R
                for i in range(n_out)
            )
            return updated, None

        final_R, _ = jax.lax.scan(
            body,
            initial,
            (time_nodes, jnp.transpose(projection_rows)),
            unroll=1,
        )
        return tuple(chi_fftn(value) for value in final_R)

    return integrate



# ============================================================================
# W solve — plan 1 of 2: LOCAL (q-parallel per-q dense LU)
# ============================================================================

def _get_w_solve_fn_local(mesh_xy: Mesh, nq: int, n_rmu: int,
                          n_rmu_logical: int | None = None):
    """W = (I - V χ)⁻¹ V via q-parallel shard_map.  All arrays flat-q: (nq, μ, μ).

    The LOCAL plan: q's are scattered over all devices
    (``P(('x','y'),None,None)``) and each rank runs one dense pivoted LU
    (``lu_factor``/``lu_solve``) per owned q.  LU is the right inner
    solve: A is SQUARE and generically well conditioned (it is I minus a
    term whose spectral radius is < 1 wherever the RPA screening is
    physical — an eigenvalue of Vχ₀ reaching 1 is a plasmon instability,
    not a numerical one).  One factorisation, one triangular pair of
    solves.

    ``n_rmu_logical``: when smaller than ``n_rmu`` (μ-padded inputs),
    the per-q pivoted LU is μ-SLICED to the logical extent and the W
    pad rows/cols are zero-filled after (their exact value: V pad rows
    are zero).  Load-bearing for device-count invariance — LU at the
    padded extent regroups partial sums per pad extent, and the
    resulting 1e-8-rel W wobble is amplified to eV on near-pole GN-PPM
    bands (reports/device_invariance_2026-07-08/ROOT_CAUSE.md, charge
    manifestation).  At zero pad the slice/fill are no-ops.
    """
    from common.shard_map import shard_map

    n_log = int(n_rmu_logical) if n_rmu_logical is not None else int(n_rmu)
    if n_log > int(n_rmu):
        raise ValueError(
            f"_get_w_solve_fn_local: n_rmu_logical={n_log} exceeds extent {n_rmu}")
    mu_pad = int(n_rmu) - n_log

    cache_key = ("local", id(mesh_xy), nq, n_rmu, n_log)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    q_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    # ── W COMES OUT 2-D SHARDED: W_q(μ_X, ν_Y) ────────────────────────
    # This used to be ``rep_3d = P(None, None, None)`` — a full
    # all-gather of the whole (nq, μ, μ) stack onto every rank, and the
    # last replicated O(nq·μ²) object in the production path (scorecard
    # J.2 #3: nq·μ²·16 per rank, ×2 for the static+probe pair, break-μ
    # ≈ 4.4 k, re-paid every SC iteration).  Nothing wanted it: every
    # consumer either is layout-agnostic (sigma_dispatch, sc_iteration,
    # gw_jax) or immediately re-imposes exactly this layout —
    # ``symmetry_maps.unfold_isdf_operator`` (P(None,'x','y') in and out),
    # ``cohsex_sigma._convolve``'s 5-D V_FFT5D_SPEC = P(None,None,None,
    # 'x','y'), ``ppm_sigma.fit_ppm``'s q_shard, and
    # ``head_wing_schur`` which literally undid the replication by hand.
    # The ONLY q-index anywhere is ``screening.py``'s ``W[0]``
    # hermiticity gate, which is a two-reduction check on one tile.
    # The distributed plan has always returned P(None,'x','y'), so the
    # whole downstream chain is already proven on this layout.
    #
    # Collective-wise the final constraint changes from an ALL-GATHER
    # (every rank ends holding nq·μ²·16 B) to an ALL-TO-ALL (nq·μ²·16/P
    # per rank).  The values are untouched — the shard_map above
    # computes the same numbers either way and this is pure data
    # movement — so the change is bit-exact by construction.
    nat_3d = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    # Intermediate sharding for the reshard from P(None,'x','y') → q_shard.
    # Routing through P('x',None,'y') (x parks on nq, y stays on μ₂) lets
    # SPMD plan it as two single-axis all_to_alls instead of the
    # "Involuntary full rematerialization" it falls into when asked to
    # un-shard both x and y on μ simultaneously via a fully replicated
    # intermediate.  Measured at Si 4×4×4 60Ry (nq=64, μ=1200, 2×2 mesh):
    #   via a fully replicated P(None,None,None) intermediate:
    #                          peak 2.95 GB/dev (temp 2.21 GB) — Involuntary Remat
    #   via P('x',None,'y'): peak 1.11 GB/dev (temp 0.37 GB)  -- 62% reduction
    reshard_mid = NamedSharding(mesh_xy, P('x', None, 'y'))
    q_spec = P(('x', 'y'), None, None)

    # ``chi_flat`` is donated (position 1): the caller releases χ₀ right
    # after this call (module contract, same as the distributed plan —
    # the ``del chi0_q_solve`` inside ``screening.py``'s ``W.exec``
    # timing block).  ``V_flat`` is NOT donated — V is reused
    # by COHSEX Σ_SX, Σ_COH, Σ_X and the PPM fit's Wc = W - V step.
    @partial(jax.jit, donate_argnums=(1,))
    def _solve_w(V_flat: jax.Array, chi_flat: jax.Array, pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ).  Returns W: (nq, μ, μ)."""
        nq_local = V_flat.shape[0]
        n = V_flat.shape[1]
        chi_scaled = pref * chi_flat

        # Pad to device count then reshard to q-parallel
        total_devices = mesh_xy.devices.size
        nq_padded = round_up(nq_local, total_devices)
        pad = nq_padded - nq_local
        V_padded = jnp.pad(V_flat, ((0, pad), (0, 0), (0, 0))) if pad > 0 else V_flat
        chi_padded = jnp.pad(chi_scaled, ((0, pad), (0, 0), (0, 0))) if pad > 0 else chi_scaled
        V_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(V_padded, reshard_mid), q_shard)
        chi_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(chi_padded, reshard_mid), q_shard)

        def _local_solve(V_local, chi_local):
            nq_dev = V_local.shape[0]

            def _dyson_log(V_log, chi_log):
                # Solve at the LOGICAL μ extent (see _get_w_solve_fn_local
                # docstring; slice/zero-refill via solve_at_logical).
                # V/χ pad rows are exact zeros, so the sliced system IS
                # the logical Dyson system; the W pad block is exactly
                # zero (A_pad = I, RHS_pad = 0).
                A = jnp.eye(n_log, dtype=V_log.dtype) - V_log @ chi_log
                lu, piv = jsp_linalg.lu_factor(A)
                return jsp_linalg.lu_solve((lu, piv), V_log)

            def solve_one(iq, W_acc):
                W_row = solve_at_logical(
                    _dyson_log, n_log, (V_local[iq], chi_local[iq]),
                    pad_axes=(-2, -1))
                return jax.lax.dynamic_update_slice(
                    W_acc, W_row[None, :, :], (iq, 0, 0))
            return jax.lax.fori_loop(0, nq_dev, solve_one, jnp.zeros_like(V_local))

        W_flat = shard_map(
            _local_solve, mesh=mesh_xy,
            in_specs=(q_spec, q_spec), out_specs=q_spec,
        )(V_q, chi_q)

        if pad > 0:
            W_flat = W_flat[:nq_local]
        # Land W on P(None,'x','y') through the SAME single-axis staging the
        # input reshard uses, in reverse:
        #     q-parallel [px·py,1,1] -> P('x',None,'y') [px,1,py]
        #                            -> P(None,'x','y') [1,px,py]
        # Asking SPMD for the composite in ONE step makes it
        # replicate-then-partition: MEASURED on a real 2×2 CUDA mesh and on
        # CPU, `[SPMD] Involuntary full rematerialization ... cannot go from
        # {devices=[4,1,1]} to {devices=[1,2,2]} ... op_name=
        # "jit(_solve_w)/shard_map"`, 1 per rank, where the base (replicated)
        # output produced none.  That transient is the whole nq·μ² object
        # this change exists to stop materialising, so the staging is
        # load-bearing, not cosmetic.  Each stage moves ONE mesh axis, which
        # is a single all_to_all — the same reasoning (and the same
        # `reshard_mid`) as the 62 %-peak-reduction note above.
        W_flat = jax.lax.with_sharding_constraint(W_flat, reshard_mid)
        return jax.lax.with_sharding_constraint(W_flat, nat_3d)

    _w_solve_cache[cache_key] = _solve_w
    return _solve_w


# ============================================================================
# W solve — plan 2 of 2: DISTRIBUTED (2-D-sharded stacked-GEMM backsolve)
# ============================================================================

def _get_w_solve_fn_distributed(mesh_xy: Mesh, nq: int, n_rmu: int,
                                n_rmu_logical: int,
                                distrib_la_batched_route: str = "auto"):
    """W = solve(A, V), A = (1 − pref·V·χ₀), everything 2-D sharded.

    The DISTRIBUTED plan — the scale-out route for thousands of
    low-memory processes, in the same architectural family as the
    ζ-fit's distributed rank-truncate tier
    (:func:`isdf.core._factor_c_q_distributed_rank_truncate`):

    1. **A build** — per q-block, ``A = I − V·(pref·χ)`` as a 2-D block
       GEMM inside ``shard_map``: rank (x, y) all-gathers V's row block
       along 'y' (full k for its i rows, μ·μ/Px per rank) and χ's column
       block along 'x' (full k for its j columns, μ·μ/Py per rank),
       multiplies locally, and subtracts from its identity tile.  The
       gathers are STRUCTURAL — inside shard_map the partitioner cannot
       hoist them into a full-stack gather (the per_q-tier lesson,
       quality pattern #4).  The q loop is chunked HOST-side so one
       collective instruction never exceeds ``LORRAX_COLLECTIVE_CHUNK_MB``
       (the AF transport bound; separate XLA executions cannot be
       re-combined by a compiler pass).
    2. **Factor + backsolve** — ONE resolved
       :class:`distrib_la.Plan` for ``solve_lu`` with
       ``backend='distributed'`` (ScaLAPACK ``pzgetrf``/``pzgetrs`` on a
       CPU mesh, cuSOLVERMp on CUDA — ``resolve._DISTRIBUTED_DEFAULT``),
       consuming the block-cyclic tiles where they already live.

    **No rank ever materialises a full (μ, μ) tile**: inputs, A, the LU
    factors and W all stay ``P(None,'x','y')`` (per-rank blocks of
    μ/Px × μ/Py; the largest per-rank transient is the μ·μ/min(Px,Py)
    gathered GEMM operand).  W lands natively in ``P(None,'x','y')`` —
    no relayout, unlike the local plan.

    Padding contract, and why it is exact: V and χ pad rows/cols are
    exact zeros (the bilinear-in-zero-padded-ψ contract), so at the
    PADDED extent ``A = [[A_log, 0], [0, I]]`` and ``RHS = [[V_log], [0]]``
    hold EXACTLY — the identity-embedded block-diagonal system whose
    solution is ``[[W_log], [0]]``; partial pivoting cannot mix the
    blocks (every pad column is a unit vector, every pad row is zero in
    the logical columns).  Therefore W's pad rows/cols leave the solve as
    exact zeros without a separate post-solve mask graph.  Unlike the local
    plan the LOGICAL
    block is formed/factored at the padded extent, so W here carries the
    ≤1e-8-rel pad-extent regrouping wobble — which is subsumed by the
    block-cyclic factorisation's own non-bit-identity; this plan's
    numerical contract is the Dyson residual (``LORRAX_W_RESIDUAL_CHECK``),
    not bit-identity with the local plan.

    Geometry/capability failures (host lib absent, non-square or 1-D
    mesh, n not divisible, process coverage) RAISE at resolve time with
    the resolver's own message — an explicitly requested distributed
    solve never silently downgrades to the local plan (quality pattern
    #6/#8).
    """
    n_ext = int(n_rmu)
    n_log = int(n_rmu_logical)
    if n_log > n_ext:
        raise ValueError(
            f"_get_w_solve_fn_distributed: n_rmu_logical={n_log} exceeds "
            f"extent {n_ext}")

    # The distributed closure factors the padded operator and is independent
    # of the logical prefix.  Keep n_log only for the public range check;
    # including it here would compile duplicate solve programs for identical
    # padded shapes.
    cache_key = ("distributed", id(mesh_xy), nq, n_ext,
                 str(distrib_la_batched_route))
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    from common.shard_map import shard_map
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import plan as linalg_plan
    # House chunking pattern — single source (scorecard AF): one emitted
    # collective's payload is bounded by LORRAX_COLLECTIVE_CHUNK_MB.
    # TODO(release): promote _chunk_q/_chunk_log to a public home (e.g.
    # common/collectives.chunk_q/chunk_log) and import them publicly from
    # both isdf/core and here — gw physics code should not reach into
    # another package's underscore-private namespace, and _chunk_log's
    # module-global dedup set is cross-package shared mutable state with
    # no public contract (audit fix/zq 2026-07-28, _idx 29; needs an
    # isdf/core edit, outside this fix's file set).
    from isdf.core import _chunk_q, _chunk_log

    # Every guard fires HERE (vocabulary, platform, capability, process
    # coverage, mesh geometry — including the 1-D-mesh cusolvermp refusal
    # — and divisibility) — resolve.py's ladder, with its own messages.
    # ``distributed`` maps to the platform default (ScaLAPACK on cpu,
    # cuSOLVERMp on CUDA) in ONE place.  No compensating ``p.is_native``
    # re-check is needed: an explicit request that cannot be honored
    # raises at resolve time (the former silent 1-D-mesh degenerate-to-
    # native was removed; audit fix/zq 2026-07-28), so a returned plan is
    # always the distributed backend it names.
    p = linalg_plan(
        "solve_lu", mesh_xy, backend="distributed", n=n_ext,
        batched_route=distrib_la_batched_route)

    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    if jax.process_index() == 0:
        print(f"  [W solve] w_dyson_solver=distributed -> {p.describe()}",
              flush=True)

    # The two collectives ``_a_local`` emits, per q (2-D block GEMM):
    #   all_gather('y')  V   (μ/Px, μ/Py) -> (μ/Px, μ)  = μ²/Px · 16 B
    #   all_gather('x')  χ   (μ/Px, μ/Py) -> (μ, μ/Py)  = μ²/Py · 16 B
    # The BIGGER of the two sets the q-block (see ``_chunk_q``).
    per_q_coll = max(n_ext * (n_ext // px), n_ext * (n_ext // py)) * 16

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(P(None, 'x', 'y'), P(None, 'x', 'y')),
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def _a_local(V_loc, chi_loc):
        # A[q,i,j] = δ_ij − Σ_k V[q,i,k]·χs[q,k,j] on my (i on 'x',
        # j on 'y') tile.  Classic 2-D block GEMM pairing — same shape
        # of communication as ``isdf.core._distributed_pinv_apply``.
        V_row = jax.lax.all_gather(V_loc, 'y', axis=2, tiled=True)
        chi_col = jax.lax.all_gather(chi_loc, 'x', axis=1, tiled=True)
        prod = jnp.einsum('qik,qkj->qij', V_row, chi_col)
        i0 = jax.lax.axis_index('x') * (n_ext // px)
        j0 = jax.lax.axis_index('y') * (n_ext // py)
        eye_tile = jnp.equal(
            i0 + jnp.arange(n_ext // px)[:, None],
            j0 + jnp.arange(n_ext // py)[None, :]).astype(V_loc.dtype)
        return eye_tile[None, :, :] - prod

    @partial(jax.jit, donate_argnums=(2,), out_shardings=nat)
    def _a_chunk(V_blk, chi_blk, A_acc, q0):
        return jax.lax.dynamic_update_slice(
            A_acc, _a_local(V_blk, chi_blk), (q0, 0, 0))

    # χ is donated here (module contract: the caller releases χ₀ after
    # solve_w — see screening.py's ``del chi0_q_solve``).
    _scale = jax.jit(lambda c, pref: pref * c,
                     donate_argnums=(0,), out_shardings=nat)
    _zeros_like = jax.jit(jnp.zeros_like, out_shardings=nat)
    # RHS must be a FRESH buffer, never an alias of the caller's V —
    # the FFI backsolve DONATES both operands (docs/dev/linalg_ffi.md
    # "Sharp edges") and V is still needed by Σ_SX/Σ_COH/Σ_X and the
    # PPM fit's Wc = W − V.
    _copy = jax.jit(jnp.copy, out_shardings=nat)

    def _solve_w_dist(V_flat: jax.Array, chi_flat: jax.Array,
                      pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ) at P(None,'x','y').  Returns W
        (nq, μ, μ) at P(None,'x','y').  chi_flat's buffer is consumed."""
        nq_local = int(V_flat.shape[0])
        qb = _chunk_q(nq_local, per_q_coll)
        _chunk_log('W Dyson A-build (GEMM)', nq_local, qb, per_q_coll)
        chi_scaled = _scale(chi_flat, pref)
        A = _zeros_like(V_flat)
        # Host-level q-block loop: ONE XLA execution per block, so the
        # emitted all_gather payloads are bounded by construction and
        # cannot be re-combined by a compiler pass (AF note in
        # isdf/core).  At most two compiled shapes (full + remainder).
        for q0 in range(0, nq_local, qb):
            q1 = min(q0 + qb, nq_local)
            A = _a_chunk(V_flat[q0:q1], chi_scaled[q0:q1], A, q0)
        B = _copy(V_flat)
        # ONE plan call for the whole stack: one descriptor, one
        # workspace; A and B are donated into the FFI.
        W = p.batched(A, B)
        # House falsy vocabulary — same parse (and same rationale comment)
        # as common/collectives.py's LORRAX_CHECK_REPLICA fix (workstream
        # AT): the narrow "0"/""/"false" tuple this replaced meant
        # LORRAX_W_RESIDUAL_CHECK=off/no/False silently ENABLED the
        # diagnostic — which must be OFF when taking collective-table
        # probes (docs/dev/env_vars.md).  (audit fix/zq 2026-07-28)
        if os.environ.get("LORRAX_W_RESIDUAL_CHECK", "0").strip().lower() \
                not in ("", "0", "false", "no", "off"):
            _w_residual_report(V_flat, chi_scaled, W, n_ext)
        return W

    _w_solve_cache[cache_key] = _solve_w_dist
    return _solve_w_dist


def _w_residual_report(V_flat, chi_scaled, W, n_ext, n_check: int = 4):
    """Direct Dyson residual ‖(1−Vχ)W − V‖/‖V‖ on the first few q.

    THE strict numerical contract of the distributed plan (a
    block-cyclic LU is not bit-comparable to the local per-q LU; the
    residual is what certifies the solve — quality pattern #6, "test
    what executes").  Diagnostic-only, opt-in via
    ``LORRAX_W_RESIDUAL_CHECK=1``; never on in the traced production
    path, so the collective-table gate is taken with it OFF.
    """
    ns = min(int(V_flat.shape[0]), int(n_check))

    @jax.jit
    def _res(V_s, chi_s, W_s):
        A_s = jnp.eye(n_ext, dtype=V_s.dtype)[None, :, :] - V_s @ chi_s
        num = jnp.linalg.norm((A_s @ W_s - V_s).reshape(ns, -1), axis=1)
        den = jnp.linalg.norm(V_s.reshape(ns, -1), axis=1)
        return num / den

    r = np.asarray(jax.device_get(_res(V_flat[:ns], chi_scaled[:ns], W[:ns])))
    if jax.process_index() == 0:
        vals = "  ".join(f"q{iq}={v:.3e}" for iq, v in enumerate(r))
        print(f"  [W solve] Dyson residual |(1-Vchi)W - V|/|V| ({ns} q): "
              f"{vals}  max={r.max():.3e}", flush=True)


def _w_solve_pref_scalar(meta) -> float:
    """The 2/(√N_k · n_spin · n_spinor) prefactor in front of χ₀ in the
    Dyson solve.  Same value for both plans; pulled out so the dispatch
    helper below isn't the only place it's computed."""
    nq = int(meta.nk_tot)
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    return 2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor))


def _resolve_w_solve_fn(meta, mesh_xy, *, n_rmu, dyson_solver=None,
                        distrib_la_batched_route: str = "auto"):
    """Return ``(solve_fn, pref)`` for the requested W plan.

    Single source of truth for the two-plan dispatch.  Both ``solve_w``
    and ``precompile_solve_w`` go through this helper — the dispatch
    logic exists in one place.

    ``dyson_solver`` (input key ``w_dyson_solver``) selects the plan:

    ``local`` (default; ``auto`` is an alias)
        per-q pivoted LU inside the q-parallel shard_map —
        :func:`_get_w_solve_fn_local`.
    ``distributed``
        the 2-D-sharded stacked-GEMM backsolve through the linalg plan
        facade — :func:`_get_w_solve_fn_distributed`.  Refuses loudly at
        resolve time when the mesh/build cannot run it; never silently
        downgrades.

    W comes out ``P(None,'x','y')`` on BOTH — that is the module's
    output contract, not a per-plan detail.
    """
    from .gw_config import normalize_w_dyson_solver
    dyson = normalize_w_dyson_solver(dyson_solver)
    nq = int(meta.nk_tot)
    pref_scalar = _w_solve_pref_scalar(meta)

    # ``meta.n_rmu`` is a HARD read: a soft getattr fallback here
    # silently restored the padded-extent LU for any meta-like object
    # missing the field (opt-out-by-omission, PADDING_AUDIT item 3).
    # Synthetic-meta callers must carry n_rmu (= the logical extent).
    n_log = int(meta.n_rmu)

    if dyson == "distributed":
        solve_fn = _get_w_solve_fn_distributed(
            mesh_xy, nq, n_rmu, n_log, distrib_la_batched_route)
    else:
        solve_fn = _get_w_solve_fn_local(
            mesh_xy, nq, n_rmu, n_rmu_logical=n_log)
    return solve_fn, jnp.asarray(pref_scalar, dtype=jnp.complex128)


def solve_w(V_q, chi0_q, meta, mesh_xy, *, dyson_solver=None,
            distrib_la_batched_route: str = "auto"):
    """W(q) = (I − V χ₀)⁻¹ V  via a Dyson solve.  **W comes out sharded.**

    All arrays flat-q: V(nq, μ, μ), χ₀(nq, μ, μ) → W(nq, μ, μ).

    **Output contract:** ``W`` is ``P(None, 'x', 'y')`` — 2-D sharded
    W_q(μ_X, ν_Y) — on both plans, and stays that way into its
    consumers (Σ_SX/Σ_COH's 5-D FFT spec, the PPM fit, the IBZ unfold,
    the restart writer).

    ``dyson_solver`` (input key ``w_dyson_solver``) picks one of the
    TWO plans — see :func:`_resolve_w_solve_fn`:

    - ``local`` (default): q-parallel reshard + per-q dense LU via
      shard_map.  Legal on any mesh; each rank holds whole (μ, μ)
      tiles for its q's.
    - ``distributed``: 2-D-sharded stacked-GEMM backsolve through the
      distrib_la plan door (ScaLAPACK on CPU, cuSOLVERMp on CUDA).
      No rank ever materialises a full (μ, μ) tile — the P→∞ memory
      ceiling.  Slower than ``local`` at moderate P; that is priced and
      accepted (the point is the per-rank memory ceiling, not speed).

    ``chi0_q``'s buffer is CONSUMED (donated) on both plans — the
    caller must drop its reference after this call.
    """
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy, n_rmu=chi0_q.shape[1], dyson_solver=dyson_solver,
        distrib_la_batched_route=distrib_la_batched_route)
    with jax_profile.annotation("W_solve"):
        return solve_fn(V_q, chi0_q, pref)


def _chi_face_kwargs(wfns) -> dict:
    """``{}`` under ``layout='legacy'``; the ``layout='face'`` +
    ``face_shape`` kwargs :func:`_get_chi_minimax_kernel` needs otherwise.
    Thin alias for :func:`gw.wavefunction_bundle.face_kernel_kwargs`, the
    shared owner (mirrors ``cohsex_sigma._face_kwargs``) — kept under this
    name so this module's own call sites did not need to change."""
    from .wavefunction_bundle import face_kernel_kwargs
    return face_kernel_kwargs(wfns)


def _chi_layout_operands(wfns, eref):
    """The ψ/energy operand tuple :func:`_get_chi_minimax_kernel`'s
    returned kernel expects AFTER ``nodes``, dispatched on ``wfns.layout``
    — the one place :func:`compute_chi0`, :func:`_chi0_multi_kernel_args`
    and :func:`_chi0_contour_kernel_args` turn a bundle into kernel
    operands, so those three callers stop extracting legacy views
    (``.xn()``/``.yr()``) themselves.  Pair with :func:`_chi_face_kwargs`
    at the SAME call site to build the matching kernel — the two tuples
    differ in length (legacy's four windowed ψ views vs face's two
    full-extent ψ copies plus two band masks) because the two kernel
    bodies' own signatures do
    (``_get_chi_minimax_kernel_legacy``/``_face``), not because this
    function special-cases anything beyond that.  ``vmax``/``cmin`` are
    NOT included — every caller appends those itself, since their host-
    side reduction is layout-independent and already computed before this
    is called.
    """
    s = wfns.slices
    eref_j = jnp.asarray(eref, dtype=wfns.enk.dtype)
    if wfns.layout == "legacy":
        enk_v = wfns.enk[:, s.val] - eref_j
        enk_c = wfns.enk[:, s.cond] - eref_j
        return (wfns.xn(s.val), wfns.yr(s.val),
               wfns.yr(s.cond), wfns.xn(s.cond), enk_v, enk_c)
    mask_v = wfns.band_mask(s.val)
    mask_c = wfns.band_mask(s.cond)
    enk_full = wfns.enk - eref_j
    return (wfns.psi_mun, wfns.psi_nmu, mask_v, mask_c, enk_full)


def compute_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=0.0):
    """Compute χ₀(q) from a wavefunction bundle and minimax quadrature.

    Returns flat-q array (nq, μ, μ).

    ``quad.tau`` and ``quad.alpha`` approximate either 1/x (static) or
    x/(x²+ωp²) (imaginary-frequency) on [x_min, x_max] where x = E_c - E_v.
    The physical χ₀ is::

        χ₀ = -2 Σ_ℓ α_ℓ Σ_{v,c} |M_vc|² exp(-τ_ℓ (E_c - E_v))

    A uniform energy shift via ``energy_reference`` is applied to both
    valence and conduction energies before building the minimax factors.
    Because only differences enter, this is algebraically invariant; the
    knob lets callers align the global zero (e.g. midgap, VBM, CBM).
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    # Fold the chi0 prefactor (-2 · exp(-τ·E_gap)) into α so the τ-scan
    # body can apply a single weighted add per node.  ``MinimaxNodes``
    # carries both in complex128; τ has Im=0 for the Laplace quad.
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid, **_chi_face_kwargs(wfns))
    return kernel(
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )


def compute_chi0_block(
    wfns_left, wfns_right, quad, meta, mesh_xy, *,
    vertex_left: int, vertex_right: int, energy_reference=0.0,
):
    """Compute one rectangular four-current response block ``chi_AB``.

    Returns ``(nq, mu_left, mu_right)`` at ``P(None,'x','y')``.  Endpoint
    Green functions are built by the same :func:`build_G_tau` path as scalar
    charge response; only its two centroid operands may have different
    extents.  The final open-spin trace routes through the canonical
    :func:`common.gamma_matrices.gamma_double_contract` for a non-scalar
    vertex pair.

    ``quad.tau`` and ``quad.alpha`` approximate either 1/x (static) or
    x/(x²+ωp²) (imaginary-frequency) on [x_min, x_max] where x = E_c - E_v.
    The physical χ₀ is::

        χ₀_AB = -2 Σ_ℓ α_ℓ Σ_{v,c}
                 M^A_vc M^B_cv exp(-τ_ℓ (E_c - E_v))

    A uniform energy shift via ``energy_reference`` is applied to both
    valence and conduction energies before building the minimax factors.
    Because only differences enter, this is algebraically invariant; the
    knob lets callers align the global zero (e.g. midgap, VBM, CBM).
    """
    A, B = int(vertex_left), int(vertex_right)
    if not (0 <= A <= 3 and 0 <= B <= 3):
        raise ValueError(
            f"chi Lorentz vertices must be in {{0,1,2,3}}; got ({A},{B})")
    if wfns_left.layout != wfns_right.layout:
        raise ValueError(
            "compute_chi0_block endpoint layouts differ: "
            f"{wfns_left.layout!r} vs {wfns_right.layout!r}")
    if wfns_left.layout != "face":
        raise ValueError(
            "compute_chi0_block four-current response requires "
            "layout='face' (low_mem_bands=true); the incumbent legacy "
            "scalar chi kernel remains unchanged")
    if wfns_right.slices != wfns_left.slices or tuple(
            wfns_right.enk.shape) != tuple(wfns_left.enk.shape):
        raise ValueError(
            "chi endpoint bundles must share band slices and energy-table "
            f"shape; got left={wfns_left.slices}/{wfns_left.enk.shape}, "
            f"right={wfns_right.slices}/{wfns_right.enk.shape}")
    ns_left = int(wfns_left.psi_mun.shape[1])
    ns_right = int(wfns_right.psi_nmu.shape[2])
    if ns_left != 4 or ns_right != 4:
        raise ValueError(
            "four-current gamma vertices require four-component bispinors; "
            f"got endpoint spinor extents ({ns_left},{ns_right})")

    if wfns_left is wfns_right and A == 0 and B == 0:
        return compute_chi0(
            wfns_left, quad, meta, mesh_xy,
            energy_reference=energy_reference)

    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

    s = wfns_left.slices
    enk_v = wfns_left.enk[:, s.val]
    enk_c = wfns_left.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    # Fold the chi0 prefactor (-2 · exp(-τ·E_gap)) into α so the τ-scan
    # body can apply a single weighted add per node.  ``MinimaxNodes``
    # carries both in complex128; τ has Im=0 for the Laplace quad.
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    from .wavefunction_bundle import face_kernel_kwargs
    left_shape = face_kernel_kwargs(wfns_left)["face_shape"]
    right_shape = face_kernel_kwargs(wfns_right)["face_shape"]
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, layout="face", face_shape=left_shape,
        right_face_shape=right_shape, vertex_pair=True,
        vertex_identity=(A == 0, B == 0))
    mask_v = wfns_left.band_mask(s.val)
    mask_c = wfns_left.band_mask(s.cond)
    enk_full = wfns_left.enk - jnp.asarray(
        eref, dtype=wfns_left.enk.dtype)
    args = (
        nodes, wfns_left.psi_mun, wfns_right.psi_nmu,
        mask_v, mask_c, enk_full,
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )
    from common.gamma_matrices import gamma_perm_phase
    perm_l, phase_l = gamma_perm_phase(A)
    perm_r, phase_r = gamma_perm_phase(B)
    return kernel(*args, perm_l, phase_l, perm_r, phase_r)


_WARD_SUBTRACTED_NO_PAIR = "ward_subtracted_no_pair"

STATIC_PHOTON_LONGWAVE_APPROXIMATION = (
    "experimental_no_pair_bubble_screened_breit_q0_nearest_shell_v2"
)
STATIC_PHOTON_HALL_SOURCE = "absent_uncertified_zero"


@dataclass(frozen=True)
class PhotonG0Vectors:
    """Canonical per-channel G=0 vectors in the two centroid shardings.

    ``x_by_channel[A]`` and ``y_by_channel[A]`` both have global shape
    ``(nq_full, layout.padded_extent(A))``.  They contain the same
    ``zeta_tilde(q,mu,G=0)`` values, respectively sharded ``P(None,'x')``
    and ``P(None,'y')``.  Rows follow ``SymMaps.kvecs_asints`` exactly.

    This is only the consumer contract.  The Coulomb/current producer remains
    the owner of how zeta is evaluated and unfolded; this module must not grow
    a second G-vector, FFT, or symmetry path.
    """

    x_by_channel: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    y_by_channel: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    provenance: str


@dataclass(frozen=True)
class StaticPhotonLongWaveCoefficients:
    r"""Nearest-shell coefficients of the static packed photon response.

    The coefficient form is the storage contract; finite-q body-sized samples
    are never retained.  For in-plane Cartesian ``q`` (1/bohr),

    ``R(q) = q_a H_direct[a] + q_a q_b Q_direct[a,b]``.

    ``H_direct`` is structurally zero under
    ``hall_topological_source=absent_uncertified_zero``.  A nearest-shell odd
    CT/TC signal is not a dc-Hall certificate: cubic spatial dispersion
    aliases into its linear fit on one radius.  The odd fit is used only in
    the shell-closure diagnostic until a separately certified Hall/Chern
    source owns this coefficient.

    and both wings are linear, ``q_a Y_x[a]`` and ``q_b Z_y[b]``.
    ``Q_direct[a,b]`` is coordinate-symmetric; its antisymmetric part is
    unidentifiable because ``q_a q_b`` annihilates it, so this record does
    not pretend to report one.  The
    no-reshard wing convention uses static Hermiticity explicitly:

    ``R[A,B]   = sum_IJ g_A[I] P[I,J] conj(g_B[J])``
    ``Y[A,I_x] = conj(sum_J P[I,J] conj(g_A[J]))``
    ``Z[J_y,B] = conj(sum_I g_B[I] P[I,J])``.

    These are the incumbent qsgw-head orientations ``Y W Z``.  Taking the
    conjugates supplies the desired opposite body orientation without a
    centroid-vector redistribution; ``hermiticity_residual`` certifies that
    premise on every projected shell/Gamma vector used by the fit.
    """

    layout: object
    H_direct: jax.Array                 # (2, 4, 4), replicated
    Q_direct: jax.Array                 # (2, 2, 4, 4), replicated
    Y_x: jax.Array                      # (2, 4, N_packed), x-sharded
    Z_y: jax.Array                      # (2, N_packed, 4), y-sharded
    shell_indices: tuple[int, ...]
    shell_directions_xy: tuple[tuple[float, float], ...]
    shell_radii_bohr_inv: tuple[float, ...]
    linear_rank: int
    quadratic_rank: int
    # Maximum fitted Lorentz/body-block shell closure diagnostics.  Analytic
    # charge replacements are excluded; one shell cannot certify convergence.
    direct_relative_residual: float
    Y_relative_residual: float
    Z_relative_residual: float
    uniform_A_residual: float
    hermiticity_residual: float
    g0_copy_residual: float
    g0_provenance: str
    charge_response_source: str
    hall_topological_source: str
    current_contact: str
    approximation: str = STATIC_PHOTON_LONGWAVE_APPROXIMATION


_photon_longwave_fit_cache: dict = {}


def _same_named_sharding(array, mesh_xy, spec) -> bool:
    """Whether ``array`` is already on ``mesh_xy`` with exactly ``spec``."""
    sharding = getattr(array, "sharding", None)
    if not isinstance(sharding, NamedSharding) or sharding.spec != spec:
        return False
    return (
        tuple(sharding.mesh.axis_names) == tuple(mesh_xy.axis_names)
        and np.array_equal(sharding.mesh.devices, mesh_xy.devices)
    )


def _require_named_sharding(array, mesh_xy, spec, name) -> None:
    if not _same_named_sharding(array, mesh_xy, spec):
        actual = getattr(array, "sharding", None)
        raise ValueError(
            f"{name} must arrive on the production mesh with sharding "
            f"{spec}; got {actual!r}. This response fit does not reshard "
            "body-sized or centroid-sized inputs implicitly."
        )


def _nearest_paired_inplane_shell(sym, kgrid, bvec_cart_bohr):
    """Smallest complete +/-q radial-shell union identifying H and Q.

    ``SymMaps.kqfull_map[0]`` supplies -q; there is no second q lookup.  The
    next radius is included only when (as on a square axis shell) the nearest
    radius alone cannot identify the symmetric Q_xy coefficient.
    """
    grid = np.asarray(kgrid, dtype=np.int64)
    steps = np.asarray(sym.kvecs_asints, dtype=np.int64)
    nq = int(np.prod(grid))
    if grid.shape != (3,) or np.any(grid <= 0) or steps.shape != (nq, 3):
        raise ValueError("nearest-shell fit requires one full positive kgrid")
    steps = np.mod(steps, grid[None, :])
    gamma = np.flatnonzero(np.all(steps == 0, axis=1))
    if not np.array_equal(gamma, np.asarray([0])):
        raise ValueError("full-BZ q order must contain Gamma exactly at row 0")
    bvec = np.asarray(bvec_cart_bohr, dtype=np.float64)
    if bvec.shape != (3, 3) or not np.all(np.isfinite(bvec)):
        raise ValueError("bvec_cart_bohr must be finite (3,3), rows in 1/bohr")
    # ``fine_q_cart`` is the sole FFT-index -> first-BZ Cartesian-q service.
    # SymMaps rows need not be C ordered, so map each integer row into that
    # service's flattened FFT box rather than respelling either convention.
    from .head_densify import fine_q_cart
    fft_flat = np.ravel_multi_index(steps.T, tuple(int(n) for n in grid))
    q_box, q_raw_box = fine_q_cart(
        bvec, tuple(int(n) for n in grid), return_unwrapped=True)
    q_cart = q_box.reshape(-1, 3)[fft_flat]
    q_raw = q_raw_box.reshape(-1, 3)[fft_flat]
    candidate = np.flatnonzero(
        np.any(steps != 0, axis=1) & (steps[:, 2] == 0))
    # The persisted one-leg coefficient is literally zeta(q,G=0).  If the
    # Voronoi service changed q by a reciprocal vector, the singular leg is
    # G* rather than G=0 and using this vector would silently omit its Umklapp
    # transformation.  Refuse until that canonical producer supplies G*.
    wrap_scale = np.maximum(
        np.maximum(np.linalg.norm(q_cart, axis=1),
                   np.linalg.norm(q_raw, axis=1)), 1.0)
    wrapped_row = np.linalg.norm(q_cart - q_raw, axis=1) > (
        512.0 * np.finfo(float).eps * wrap_scale)
    radii = np.linalg.norm(q_cart[candidate, :2], axis=1)
    keep = radii > 0.0
    candidate, radii = candidate[keep], radii[keep]
    if not candidate.size:
        raise ValueError("nearest-shell fit found no nonzero in-plane q")

    def _design(q):
        linear = q
        quadratic = np.stack(
            (q[:, 0] ** 2, 2.0 * q[:, 0] * q[:, 1], q[:, 1] ** 2), axis=1)
        return linear, quadratic

    def _rank(a):
        s = np.linalg.svd(a, compute_uv=False)
        tol = 128.0 * max(a.shape) * np.finfo(float).eps * float(s[0])
        return int(np.sum(s > tol)), tol

    levels = []
    for radius in np.sort(radii):
        if not any(np.isclose(radius, old, rtol=1e-11,
                              atol=1e-12 * old) for old in levels):
            levels.append(float(radius))
    selected = np.zeros(candidate.size, dtype=bool)
    for radius in levels:
        selected |= np.isclose(
            radii, radius, rtol=1e-11, atol=1e-12 * radius)
        shell = np.asarray(candidate[selected], dtype=np.int32)
        q_shell = np.asarray(q_cart[shell, :2], dtype=np.float64)
        linear, quadratic = _design(q_shell)
        linear_rank, linear_tol = _rank(linear)
        quadratic_rank, quadratic_tol = _rank(quadratic)
        if (linear_rank, quadratic_rank) == (2, 3):
            break
    if (linear_rank, quadratic_rank) != (2, 3):
        raise ValueError(
            "nearest-shell active-direction design is rank deficient: "
            f"linear={linear_rank}/2, quadratic={quadratic_rank}/3")
    if np.any(wrapped_row[shell]):
        bad = shell[wrapped_row[shell]]
        raise ValueError(
            "nearest-shell photon fit found a selected q whose first-BZ "
            "Voronoi representative changes the literal G=0 leg: rows="
            f"{bad.tolist()}. The canonical V producer must persist the "
            "corresponding argmin-|q+G| one-leg vector before this skew-grid "
            "shell can be fitted.")

    try:
        negative = np.asarray(sym.kqfull_map[0], dtype=np.int64)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("SymMaps.kqfull_map[0] (-q) is required") from exc
    if negative.shape != (nq,) or np.any((negative < 0) | (negative >= nq)):
        raise ValueError("SymMaps.kqfull_map[0] is not a full-BZ permutation")
    position = np.full(nq, -1, dtype=np.int32)
    position[shell] = np.arange(shell.size, dtype=np.int32)
    partners = negative[shell]
    partner_position = position[partners]
    if np.any(partner_position < 0):
        raise ValueError("nearest complete shell is missing a -q partner")
    if (np.any(partners == shell) or np.any(negative[partners] != shell)
            or not np.array_equal(
                steps[partners], np.mod(-steps[shell], grid[None, :]))):
        raise ValueError("SymMaps -q pairs are ambiguous on the fitted shell")
    scale = max(float(np.max(np.abs(q_cart[shell]))), 1.0)
    tol = 512.0 * np.finfo(float).eps * scale
    if (np.max(np.abs(q_cart[shell, 2])) > tol
            or not np.allclose(q_cart[partners, :2], -q_shell,
                               rtol=0.0, atol=tol)):
        raise ValueError("fitted +/-q shell is not Cartesian in-plane")

    pair_rows = np.eye(shell.size)[partner_position]
    linear_weights = np.linalg.pinv(
        linear, rcond=linear_tol / np.linalg.norm(linear, 2)
    ) @ (0.5 * (np.eye(shell.size) - pair_rows))
    quadratic_weights = np.linalg.pinv(
        quadratic, rcond=quadratic_tol / np.linalg.norm(quadratic, 2)
    ) @ (0.5 * (np.eye(shell.size) + pair_rows))
    shell_radii = np.linalg.norm(q_shell, axis=1)
    return (
        shell, q_shell, q_shell / shell_radii[:, None], shell_radii,
        linear_weights, quadratic_weights, linear_rank, quadratic_rank,
    )


def _static_charge_longwave_inputs(charge_response, layout, mesh_xy):
    """Select the unique analytic omega=0 charge S/Y/Z without gathering."""
    p_charge = layout.padded_extent(0)
    if charge_response is None:
        rep = NamedSharding(mesh_xy, P(None, None))
        sh_x = NamedSharding(mesh_xy, P(None, "x"))
        sh_y = NamedSharding(mesh_xy, P("y", None))
        return (
            jax.device_put(np.zeros((2, 2), np.complex128), rep),
            jax.device_put(np.zeros((2, p_charge), np.complex128), sh_x),
            jax.device_put(np.zeros((p_charge, 2), np.complex128), sh_y),
            False,
            "nearest_shell_only",
        )

    omegas = tuple(complex(z) for z in charge_response.omegas)
    zero_rows = [i for i, z in enumerate(omegas) if z == 0.0j]
    if len(zero_rows) != 1:
        raise ValueError(
            "analytic charge replacement requires exactly one literal "
            f"omega=0 qsgw_head row; found {zero_rows} in {omegas}")
    index = zero_rows[0]
    if charge_response.Y_x is None or charge_response.Z_y is None:
        raise ValueError(
            "analytic charge replacement requires the existing qsgw_head "
            "S, Y_x, and Z_y arrays")
    S = charge_response.S_direct
    Y = charge_response.Y_x
    Z = charge_response.Z_y
    if tuple(S.shape) != (len(omegas), 3, 3):
        raise ValueError(
            f"qsgw_head S_direct shape {S.shape} != ({len(omegas)},3,3)")
    if tuple(Y.shape) != (len(omegas), 3, p_charge):
        raise ValueError(
            "qsgw_head Y_x does not match the padded charge centroid "
            f"extent: got {Y.shape}, expected ({len(omegas)},3,{p_charge})")
    if tuple(Z.shape) != (len(omegas), p_charge, 3):
        raise ValueError(
            "qsgw_head Z_y does not match the padded charge centroid "
            f"extent: got {Z.shape}, expected ({len(omegas)},{p_charge},3)")
    _require_named_sharding(Y, mesh_xy, P(None, None, "x"),
                            "charge_response.Y_x")
    _require_named_sharding(Z, mesh_xy, P(None, "y", None),
                            "charge_response.Z_y")
    S_xy_raw = np.asarray(
        jax.device_get(S[index, :2, :2]), np.complex128)
    # q_a q_b sees only the coordinate-symmetric part; retaining an
    # arbitrary antisymmetric representative would contradict Q's contract.
    S_xy_host = 0.5 * (S_xy_raw + S_xy_raw.T)
    if not np.all(np.isfinite(S_xy_host)):
        raise ValueError("qsgw_head static S contains non-finite entries")
    S_xy = jax.device_put(
        S_xy_host, NamedSharding(mesh_xy, P(None, None)))
    return (
        S_xy,
        Y[index, :2, :],
        Z[index, :, :2],
        True,
        "qsgw_head.IterationHeadResponse@omega=0",
    )


def _get_static_photon_longwave_fit_kernel(mesh_xy, layout):
    """One compiled shape family for projection, parity split, and fit."""
    # Logical padding lengths do not enter the compiled program.  Reusing one
    # kernel for every layout with the same physical carrier prevents a second
    # JAX executable whose only distinction is masked metadata.
    key = (id(mesh_xy), tuple(int(n) for n in layout.padded_extents))
    hit = _photon_longwave_fit_cache.get(key)
    if hit is not None:
        return hit
    from common.shard_map import shard_map

    layout.assert_mesh(mesh_xy)
    linear_mask = np.fromfunction(
        lambda A, B: (A == 0) != (B == 0), (4, 4), dtype=int)
    local_widths = tuple(
        int(n) // int(layout.mesh_side) for n in layout.padded_extents)
    local_offsets = tuple(int(layout.local_offset(A)) for A in range(4))

    def _local(
        chi_local, g0_x_local, g0_y_local, shell_indices,
        q_shell_xy, linear_weights, quadratic_weights, prefactor,
        inverse_volume,
        charge_S, charge_Y_packed_x, charge_Z_packed_y,
        charge_mask_x, charge_mask_y, use_charge,
    ):
        dtype = chi_local.dtype
        real_dtype = jnp.real(chi_local).dtype
        x_coord = jax.lax.axis_index("x")
        y_coord = jax.lax.axis_index("y")
        prefactor = jnp.asarray(prefactor, dtype=dtype)
        inverse_volume = jnp.asarray(inverse_volume, dtype=dtype)

        def _abs2(value):
            return jnp.square(jnp.abs(value))

        def _head_body_block_norms(rows):
            """Squared norms of (head channel, packed body channel)."""
            return jnp.stack([
                jnp.sum(
                    _abs2(rows[:, offset:offset + width]), axis=1)
                for offset, width in zip(local_offsets, local_widths)
            ], axis=1)

        def _body_head_block_norms(columns):
            """Squared norms of (packed body channel, head channel)."""
            return jnp.stack([
                jnp.sum(
                    _abs2(columns[offset:offset + width, :]), axis=0)
                for offset, width in zip(local_offsets, local_widths)
            ], axis=0)

        p_gamma = chi_local[0] * prefactor
        gx_gamma = g0_x_local[0]
        gy_gamma = g0_y_local[0]

        r0_natural = jax.lax.psum(
            jnp.einsum(
                "ai,ij,bj->ab", gx_gamma, p_gamma,
                jnp.conj(gy_gamma), optimize=True),
            ("x", "y"))
        t0 = jax.lax.psum(
            jnp.einsum(
                "ij,aj->ai", p_gamma, jnp.conj(gy_gamma), optimize=True),
            "y")
        u0 = jax.lax.psum(
            jnp.einsum("bi,ij->jb", gx_gamma, p_gamma,
                       optimize=True),
            "x")

        def _project_shell_row(shell_position):
            """Project one q row; never form shell x packed-body storage."""
            q_index = shell_indices[shell_position]
            p_row = chi_local[q_index] * prefactor
            gx_row = g0_x_local[q_index]
            gy_row = g0_y_local[q_index]

            # Direct head and both one-sided projections.  Static
            # Hermiticity converts the native body-head x carrier into the
            # canonical head-body Y (and conversely for Z) by conjugation,
            # without a body-axis redistribution.
            r_row = jax.lax.psum(
                jnp.einsum(
                    "ai,ij,bj->ab", gx_row, p_row, jnp.conj(gy_row),
                    optimize=True),
                ("x", "y"))
            t_row = jax.lax.psum(
                jnp.einsum(
                    "ij,aj->ai", p_row, jnp.conj(gy_row), optimize=True),
                "y")
            u_row = jax.lax.psum(
                jnp.einsum("bi,ij->jb", gx_row, p_row, optimize=True),
                "x")

            # The projected body response g P conj(g) carries no cell-volume
            # factor.  A small direct photon head is paired with vcoul's raw
            # D(q), so its H/Q coefficients own the same sole 1/Omega as
            # qsgw_head.S_direct.  Wings remain unscaled; their later Y W Z
            # fold owns its separate 1/Omega.
            r_row = (r_row - r0_natural) * inverse_volume
            y_row = jnp.conj(t_row - t0)
            z_row = jnp.conj(u_row - u0)
            return (
                r_row, y_row, z_row, t_row, u_row,
                ~jnp.all(jnp.isfinite(p_row)),
            )

        n_shell = shell_indices.shape[0]
        shell_positions = jnp.arange(n_shell, dtype=jnp.int32)
        fit_initial = (
            jnp.zeros((2, 4, 4), dtype=dtype),
            jnp.zeros((3, 4, 4), dtype=dtype),
            jnp.zeros((2, 4, g0_x_local.shape[-1]), dtype=dtype),
            jnp.zeros((2, g0_y_local.shape[-1], 4), dtype=dtype),
            jnp.asarray(False),
        )

        def _fit_row(carry, shell_position):
            H_acc, q_three_acc, Y_acc, Z_acc, nonfinite_acc = carry
            r_row, y_row, z_row, _, _, nonfinite_row = (
                _project_shell_row(shell_position))
            lw = linear_weights[:, shell_position]
            qw = quadratic_weights[:, shell_position]
            return (
                H_acc + lw[:, None, None] * r_row[None, :, :],
                q_three_acc + qw[:, None, None] * r_row[None, :, :],
                Y_acc + lw[:, None, None] * y_row[None, :, :],
                Z_acc + lw[:, None, None] * z_row[None, :, :],
                nonfinite_acc | nonfinite_row,
            ), None

        (H_shell, q_three, Y, Z, sampled_nonfinite), _ = jax.lax.scan(
            _fit_row, fit_initial, shell_positions, unroll=1)
        H_shell = jnp.where(
            jnp.asarray(linear_mask)[None, :, :], H_shell, 0)
        Q = jnp.zeros((2, 2, 4, 4), dtype=dtype)
        Q = Q.at[0, 0].set(q_three[0])
        Q = Q.at[0, 1].set(q_three[1])
        Q = Q.at[1, 0].set(q_three[1])
        Q = Q.at[1, 1].set(q_three[2])
        Q = Q.at[:, :, 0, 0].set(jnp.where(
            use_charge, charge_S, Q[:, :, 0, 0]))
        Y = Y.at[:, 0, :].set(jnp.where(
            use_charge & charge_mask_x[:, 0, :],
            charge_Y_packed_x[:, 0, :], Y[:, 0, :]))
        Z = Z.at[:, :, 0].set(jnp.where(
            use_charge & charge_mask_y[:, 0, :],
            charge_Z_packed_y[:, 0, :], Z[:, :, 0]))

        zero_real = jnp.asarray(0.0, dtype=real_dtype)
        zero_blocks = jnp.zeros((4, 4), dtype=real_dtype)
        diagnostic_initial = (
            zero_blocks, zero_blocks, zero_blocks,
            zero_blocks, zero_blocks, zero_blocks,
            zero_blocks, zero_blocks, zero_blocks,
            zero_real, zero_real,
        )

        def _diagnose_row(carry, shell_position):
            (
                r_error, r_norm, r_prediction_norm,
                y_error, y_norm, y_prediction_norm,
                z_error, z_norm, z_prediction_norm,
                herm_error, herm_scale,
            ) = carry
            r_row, y_row, z_row, t_row, u_row, _ = (
                _project_shell_row(shell_position))
            q_row = q_shell_xy[shell_position]
            r_prediction = (
                jnp.einsum("a,aAB->AB", q_row, H_shell, optimize=True)
                + jnp.einsum(
                    "a,b,abAB->AB", q_row, q_row, Q, optimize=True))
            y_prediction = jnp.einsum(
                "a,aAI->AI", q_row, Y, optimize=True)
            z_prediction = jnp.einsum(
                "a,aJB->JB", q_row, Z, optimize=True)
            return (
                r_error + _abs2(r_prediction - r_row),
                r_norm + _abs2(r_row),
                r_prediction_norm + _abs2(r_prediction),
                y_error + _head_body_block_norms(y_prediction - y_row),
                y_norm + _head_body_block_norms(y_row),
                y_prediction_norm + _head_body_block_norms(y_prediction),
                z_error + _body_head_block_norms(z_prediction - z_row),
                z_norm + _body_head_block_norms(z_row),
                z_prediction_norm + _body_head_block_norms(z_prediction),
                jnp.maximum(
                    herm_error,
                    jnp.max(jnp.abs(t_row - jnp.conj(
                        jnp.swapaxes(u_row, 0, 1))))),
                jnp.maximum(
                    herm_scale,
                    jnp.maximum(jnp.max(jnp.abs(t_row)),
                                jnp.max(jnp.abs(u_row)))),
            ), None

        (
            r_error, r_norm, r_prediction_norm,
            y_error, y_norm, y_prediction_norm,
            z_error, z_norm, z_prediction_norm,
            hermiticity_error, hermiticity_scale,
        ), _ = jax.lax.scan(
            _diagnose_row, diagnostic_initial, shell_positions, unroll=1)
        y_error, y_norm, y_prediction_norm = (
            jax.lax.psum(value, "x") for value in (
                y_error, y_norm, y_prediction_norm))
        z_error, z_norm, z_prediction_norm = (
            jax.lax.psum(value, "y") for value in (
                z_error, z_norm, z_prediction_norm))

        # Diagnose closure per Lorentz/body-channel block, excluding exactly
        # the (0,0) entries replaced by the canonical analytic charge S/Y/Z.
        # This is deliberately not an acceptance threshold: on one radius,
        # q^3 aliases into H and q^4 into Q.  A nested-shell/Richardson result
        # is required before this statistic can claim q->0 convergence.
        fitted_blocks = jnp.ones((4, 4), dtype=bool).at[0, 0].set(~use_charge)

        def _maximum_block_residual(error_sq, sample_sq, prediction_sq):
            denominator = jnp.maximum(
                jnp.maximum(jnp.sqrt(sample_sq), jnp.sqrt(prediction_sq)),
                jnp.finfo(real_dtype).tiny)
            relative = jnp.sqrt(error_sq) / denominator
            return jnp.max(jnp.where(fitted_blocks, relative, 0.0))

        direct_residual = _maximum_block_residual(
            r_error, r_norm, r_prediction_norm)
        y_residual = _maximum_block_residual(
            y_error, y_norm, y_prediction_norm)
        z_residual = _maximum_block_residual(
            z_error, z_norm, z_prediction_norm)

        uniform_A = jnp.max(
            jnp.abs(r0_natural[1:, 1:] * inverse_volume))

        # On diagonal mesh ranks the x/y copies carry the same global mu
        # slice.  Those ranks collectively cover every slice once, so this
        # scalar certifies the copies without redistributing either vector.
        diagonal_rank = x_coord == y_coord
        g0_copy_error = jnp.asarray(0.0, dtype=real_dtype)
        g0_copy_scale = jnp.asarray(0.0, dtype=real_dtype)
        g0_nonfinite = jnp.asarray(False)
        difference = jnp.max(jnp.abs(g0_x_local - g0_y_local))
        magnitude = jnp.maximum(
            jnp.max(jnp.abs(g0_x_local)), jnp.max(jnp.abs(g0_y_local)))
        g0_copy_error = jnp.where(diagonal_rank, difference, 0.0)
        g0_copy_scale = jnp.where(diagonal_rank, magnitude, 0.0)
        finite_here = (
            jnp.all(jnp.isfinite(g0_x_local))
            & jnp.all(jnp.isfinite(g0_y_local)))
        g0_nonfinite = jnp.where(diagonal_rank, ~finite_here, False)

        hermiticity_error = jnp.maximum(
            hermiticity_error,
            jnp.max(jnp.abs(t0 - jnp.conj(jnp.swapaxes(u0, 0, 1)))))
        hermiticity_scale = jnp.maximum(
            hermiticity_scale,
            jnp.maximum(jnp.max(jnp.abs(t0)), jnp.max(jnp.abs(u0))))
        hermiticity_error = jax.lax.pmax(
            jnp.where(diagonal_rank, hermiticity_error, 0.0), ("x", "y"))
        hermiticity_scale = jax.lax.pmax(
            jnp.where(diagonal_rank, hermiticity_scale, 0.0), ("x", "y"))
        g0_copy_error = jax.lax.pmax(g0_copy_error, ("x", "y"))
        g0_copy_scale = jax.lax.pmax(g0_copy_scale, ("x", "y"))
        g0_nonfinite = jax.lax.pmax(
            g0_nonfinite.astype(jnp.int32), ("x", "y"))
        sampled_nonfinite = jax.lax.pmax(
            (sampled_nonfinite
             | ~jnp.all(jnp.isfinite(p_gamma))).astype(jnp.int32),
            ("x", "y"))
        analytic_nonfinite = (
            ~jnp.all(jnp.isfinite(charge_S))
            | jax.lax.pmax(
                (~jnp.all(jnp.isfinite(charge_Y_packed_x))).astype(jnp.int32),
                "x").astype(bool)
            | jax.lax.pmax(
                (~jnp.all(jnp.isfinite(charge_Z_packed_y))).astype(jnp.int32),
                "y").astype(bool)
        ) & use_charge
        # A one-radius odd fit cannot distinguish the q-linear topological
        # Hall/Chern term from q-cubic spatial dispersion.  Keep H out of the
        # production completion until an independently certified source owns
        # it; the fitted H_shell above survives only through the closure
        # diagnostic.
        H = jnp.zeros_like(H_shell)
        return (
            H, Q, Y, Z,
            direct_residual, y_residual, z_residual,
            uniform_A, hermiticity_error, hermiticity_scale,
            g0_copy_error, g0_copy_scale,
            g0_nonfinite | sampled_nonfinite | analytic_nonfinite,
        )

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(
            P(None, "x", "y"),
            P(None, None, "x"), P(None, None, "y"),
            P(None), P(None, None), P(None, None), P(None, None), P(), P(),
            P(None, None), P(None, None, "x"),
            P(None, None, "y"), P(None, None, "x"),
            P(None, None, "y"), P(),
        ),
        out_specs=(
            P(None, None, None), P(None, None, None, None),
            P(None, None, "x"), P(None, "y", None),
            P(), P(), P(), P(), P(), P(), P(), P(), P(),
        ),
        check_vma=False,
    ))
    _photon_longwave_fit_cache[key] = kernel
    return kernel


def fit_static_photon_longwave_coefficients(
    chi_packed,
    g0_vectors: PhotonG0Vectors,
    layout,
    meta,
    mesh_xy,
    *,
    sym,
    bvec_cart_bohr,
    sys_dim: int,
    omega_ry=0.0,
    material_class: str,
    current_contact: str = _WARD_SUBTRACTED_NO_PAIR,
    charge_response=None,
) -> StaticPhotonLongWaveCoefficients:
    r"""Fit the experimental static slab q->0 photon response coefficients.

    ``chi_packed`` is the raw packed output of :func:`compute_photon_chi0`.
    This function forms the Dyson-normalized response
    ``P = _w_solve_pref_scalar(meta) * chi_packed`` exactly once, then
    projects only Gamma and the nearest complete +/-q shell.  It never reads
    WFN/G-vectors, performs a q lookup or symmetry operation, repacks an
    operator, folds wings, or solves Dyson.  Full packed samples die inside
    the one sharded kernel; only coefficient arrays leave it.

    The fitted powers follow the static field expansion: an odd linear direct
    term is possible only in CT/TC, but the one-shell value is diagnostic and
    the returned H is zero until a separately certified Hall/Chern source is
    available.  The ordinary quadratic Q is retained in all CC/CT/TC/TT
    blocks, including magnetoelectric CT/TC.  Every body wing is linear; all
    intercepts are zero through explicit subtraction of the projected Gamma
    value.
    ``charge_response``, when supplied, must be the existing
    ``qsgw_head.IterationHeadResponse`` with one literal static row.  Its
    in-plane S/Y/Z replace the fitted charge entries without gathering their
    centroid axes.
    """
    from .photon_layout import (
        PhotonBasisLayout, pack_photon_channel_vectors,
    )

    if not isinstance(layout, PhotonBasisLayout):
        raise TypeError("layout must be PhotonBasisLayout")
    layout.assert_mesh(mesh_xy)
    if int(sys_dim) != 2:
        raise ValueError(
            "nearest-shell photon long-wave fit is slab-only (sys_dim=2); "
            f"got {sys_dim}")
    if complex(omega_ry) != 0.0j:
        raise ValueError(
            "static Ward powers cannot be used for a dynamic photon "
            f"response; omega_ry={complex(omega_ry)!r}")
    if str(material_class).strip().lower() != "insulator":
        raise ValueError(
            "nearest-shell static photon fit is restricted to gapped "
            f"insulators; material_class={material_class!r}")
    if current_contact != _WARD_SUBTRACTED_NO_PAIR:
        raise ValueError(
            "nearest-shell static photon fit requires the explicit "
            f"{_WARD_SUBTRACTED_NO_PAIR!r} contact; got {current_contact!r}")
    if not isinstance(g0_vectors, PhotonG0Vectors):
        raise TypeError("g0_vectors must be PhotonG0Vectors")
    try:
        from .v_q_bispinor import G0_DIAGONAL_FULL_BZ_V1
    except ImportError as exc:
        raise RuntimeError(
            "photon long-wave fitting requires the canonical G=0 producer "
            "contract in gw.v_q_bispinor") from exc
    if g0_vectors.provenance != G0_DIAGONAL_FULL_BZ_V1:
        raise ValueError(
            "ambiguous photon G=0 provenance: expected "
            f"{G0_DIAGONAL_FULL_BZ_V1!r}, got {g0_vectors.provenance!r}")

    nq = int(meta.nk_tot)
    expected_shape = (nq, layout.packed_extent, layout.packed_extent)
    if tuple(chi_packed.shape) != expected_shape:
        raise ValueError(
            f"packed photon chi shape {chi_packed.shape} != {expected_shape}")
    _require_named_sharding(
        chi_packed, mesh_xy, P(None, "x", "y"), "chi_packed")
    if len(g0_vectors.x_by_channel) != 4 or len(
            g0_vectors.y_by_channel) != 4:
        raise ValueError("PhotonG0Vectors requires exactly four channels")
    for A, (gx, gy) in enumerate(zip(
            g0_vectors.x_by_channel, g0_vectors.y_by_channel)):
        expected = (nq, layout.padded_extent(A))
        if tuple(gx.shape) != expected or tuple(gy.shape) != expected:
            raise ValueError(
                f"photon g0 channel {A} shapes {gx.shape}/{gy.shape} != "
                f"{expected}")
        _require_named_sharding(gx, mesh_xy, P(None, "x"),
                                f"g0_vectors.x_by_channel[{A}]")
        _require_named_sharding(gy, mesh_xy, P(None, "y"),
                                f"g0_vectors.y_by_channel[{A}]")

    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    if int(np.prod(kgrid)) != nq or int(getattr(sym, "nk_tot", nq)) != nq:
        raise ValueError(
            "meta and SymMaps do not describe one full-BZ q axis: "
            f"kgrid={kgrid}, meta.nk_tot={nq}, "
            f"sym.nk_tot={getattr(sym, 'nk_tot', None)}")
    (
        shell, q_shell, directions, shell_radii,
        linear_weights, quadratic_weights, linear_rank, quadratic_rank,
    ) = _nearest_paired_inplane_shell(sym, kgrid, bvec_cart_bohr)
    charge_S, charge_Y, charge_Z, use_charge, charge_source = (
        _static_charge_longwave_inputs(charge_response, layout, mesh_xy))
    g0_x_packed = pack_photon_channel_vectors(
        tuple(g0_vectors.x_by_channel), layout, mesh_xy, axis_name="x")
    g0_y_packed = pack_photon_channel_vectors(
        tuple(g0_vectors.y_by_channel), layout, mesh_xy, axis_name="y")

    # The same layout owner embeds the analytic charge wing into the packed
    # carrier.  Transverse rows are explicit zeros; w_isdf never spells a
    # channel offset or packed-order constructor.
    zero_x = tuple(jax.device_put(
        np.zeros((2, layout.padded_extent(A)), dtype=np.complex128),
        NamedSharding(mesh_xy, P(None, "x"))) for A in range(1, 4))
    zero_y = tuple(jax.device_put(
        np.zeros((2, layout.padded_extent(A)), dtype=np.complex128),
        NamedSharding(mesh_xy, P(None, "y"))) for A in range(1, 4))
    charge_Y_packed = pack_photon_channel_vectors(
        (charge_Y,) + zero_x, layout, mesh_xy, axis_name="x")
    charge_Z_rows = jnp.swapaxes(charge_Z, 0, 1)
    charge_Z_packed = pack_photon_channel_vectors(
        (charge_Z_rows,) + zero_y, layout, mesh_xy, axis_name="y")
    charge_mask_x = pack_photon_channel_vectors(
        (jax.device_put(
            np.ones((2, layout.padded_extent(0)), dtype=np.complex128),
            NamedSharding(mesh_xy, P(None, "x"))),) + zero_x,
        layout, mesh_xy, axis_name="x") != 0
    charge_mask_y = pack_photon_channel_vectors(
        (jax.device_put(
            np.ones((2, layout.padded_extent(0)), dtype=np.complex128),
            NamedSharding(mesh_xy, P(None, "y"))),) + zero_y,
        layout, mesh_xy, axis_name="y") != 0

    rep1 = NamedSharding(mesh_xy, P(None))
    rep2 = NamedSharding(mesh_xy, P(None, None))
    shell_device = jax.device_put(shell, rep1)
    q_device = jax.device_put(q_shell, rep2)
    linear_device = jax.device_put(linear_weights, rep2)
    quadratic_device = jax.device_put(quadratic_weights, rep2)
    prefactor = jax.device_put(
        np.asarray(_w_solve_pref_scalar(meta), dtype=np.float64),
        NamedSharding(mesh_xy, P()))
    cell_volume = float(meta.cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError(
            f"photon long-wave fit requires positive cell volume; got "
            f"{cell_volume}")
    inverse_volume = jax.device_put(
        np.asarray(1.0 / cell_volume, dtype=np.float64),
        NamedSharding(mesh_xy, P()))
    use_charge_device = jax.device_put(
        np.asarray(use_charge, dtype=np.bool_), NamedSharding(mesh_xy, P()))
    results = _get_static_photon_longwave_fit_kernel(mesh_xy, layout)(
        chi_packed,
        g0_x_packed,
        g0_y_packed,
        shell_device,
        q_device,
        linear_device,
        quadratic_device,
        prefactor,
        inverse_volume,
        charge_S,
        charge_Y_packed,
        charge_Z_packed,
        charge_mask_x,
        charge_mask_y,
        use_charge_device,
    )
    H, Q, Y, Z = results[:4]
    # These nine values are replicated scalars already.  Copy the tiny tuple
    # directly instead of compiling a second program merely to stack it.
    diagnostics = np.asarray(
        [np.asarray(value).item() for value in jax.device_get(results[4:])],
        dtype=np.float64,
    )
    (
        direct_residual, y_residual, z_residual, uniform_A,
        hermiticity_error, hermiticity_scale,
        g0_copy_error, g0_copy_scale, nonfinite,
    ) = diagnostics.tolist()
    if int(nonfinite) != 0:
        raise ValueError(
            "nearest-shell photon response or canonical g0 vectors contain "
            "non-finite sampled values")
    g0_tolerance = (
        256.0 * np.finfo(np.float64).eps * max(g0_copy_scale, 1.0))
    if g0_copy_error > g0_tolerance:
        raise ValueError(
            "photon g0 x/y carriers are not copies of the same canonical "
            f"vectors: max_abs={g0_copy_error:.6e}, "
            f"tolerance={g0_tolerance:.6e}")
    hermiticity_tolerance = 1.0e-10 * max(hermiticity_scale, 1.0)
    if hermiticity_error > hermiticity_tolerance:
        raise ValueError(
            "static packed response is not Hermitian in the projected "
            f"G=0/body subspace: max_abs={hermiticity_error:.6e}, "
            f"tolerance={hermiticity_tolerance:.6e}")
    if uniform_A != 0.0:
        raise ValueError(
            "post-subtraction static uniform-A residual is nonzero in the "
            f"projected TT Gamma head: max_abs={uniform_A:.6e}")

    return StaticPhotonLongWaveCoefficients(
        layout=layout,
        H_direct=H,
        Q_direct=Q,
        Y_x=Y,
        Z_y=Z,
        shell_indices=tuple(int(i) for i in shell),
        shell_directions_xy=tuple(
            tuple(float(v) for v in row) for row in directions),
        shell_radii_bohr_inv=tuple(float(r) for r in shell_radii),
        linear_rank=int(linear_rank),
        quadratic_rank=int(quadratic_rank),
        direct_relative_residual=float(direct_residual),
        Y_relative_residual=float(y_residual),
        Z_relative_residual=float(z_residual),
        uniform_A_residual=float(uniform_A),
        hermiticity_residual=float(hermiticity_error),
        g0_copy_residual=float(g0_copy_error),
        g0_provenance=g0_vectors.provenance,
        charge_response_source=charge_source,
        hall_topological_source=STATIC_PHOTON_HALL_SOURCE,
        current_contact=current_contact,
    )


@partial(jax.jit, donate_argnums=(0,))
def _subtract_static_tt_contact(chi_tt):
    """Bring-up Ward proxy ``Pi_para(q)-Pi_para(q=0)`` for one TT block.

    Full flat-q uses the production C-order convention with Gamma at index
    zero (the same body convention consumed by ``qsgw_head.py``'s
    ``finalize_iteration_head_samples``); this is not a second q lookup.
    """
    corrected = chi_tt - chi_tt[0:1]
    # Make the q=0 cancellation structural rather than roundoff-dependent.
    return corrected.at[0].set(jnp.zeros_like(corrected[0]))


def compute_photon_chi0(
    wfns_charge, wfns_transverse, quad, meta, mesh_xy, layout, *,
    current_contact: str = _WARD_SUBTRACTED_NO_PAIR,
    energy_reference=0.0,
):
    """Build and locally pack all sixteen static CC/CT/TC/TT blocks.

    Only one response block and the donated packed accumulator are live at a
    time.  The three transverse channels reuse ``wfns_transverse`` and differ
    only by their gamma vertex; no T1/T2/T3 wavefunction copies are made.
    """
    from .photon_layout import pack_photon_operator

    layout.assert_mesh(mesh_xy)
    if current_contact != _WARD_SUBTRACTED_NO_PAIR:
        raise ValueError(
            "full static photon response currently requires "
            f"current_contact={_WARD_SUBTRACTED_NO_PAIR!r}; "
            f"got {current_contact!r}")
    if (wfns_charge.layout, wfns_transverse.layout) != ("face", "face"):
        raise ValueError(
            "full four-current response requires layout='face' for both "
            "charge and transverse endpoint bundles (low_mem_bands=true); "
            f"got {wfns_charge.layout!r}/{wfns_transverse.layout!r}")
    n_c = int(wfns_charge.psi_mun.shape[2])
    n_t = int(wfns_transverse.psi_mun.shape[2])
    if (n_c != layout.padded_extent(0) or
            n_t != layout.padded_extent(1)):
        raise ValueError(
            "photon layout padded extents do not match wavefunction "
            f"bundles: layout C/T=({layout.padded_extent(0)},"
            f"{layout.padded_extent(1)}), wfns C/T=({n_c},{n_t})")
    nq = int(meta.nk_tot)
    families = (wfns_charge, wfns_transverse,
                wfns_transverse, wfns_transverse)

    def get_block(A, B):
        chi_ab = compute_chi0_block(
            families[A], families[B], quad, meta, mesh_xy,
            vertex_left=A, vertex_right=B,
            energy_reference=energy_reference)
        if A and B:
            # Experimental no-pair Ward completion: TT only.  CC/CT/TC are
            # left untouched and no diamagnetic contact is invented.
            return _subtract_static_tt_contact(chi_ab)
        return chi_ab

    return pack_photon_operator(get_block, nq, layout, mesh_xy)


@dataclass(frozen=True)
class StaticPhotonResponse:
    """Experimental full-body static photon response and provenance."""

    layout: object
    V_packed: jax.Array
    W_packed: jax.Array
    current_contact: str
    head_completion: object | None = None
    approximation: str = "experimental_no_pair_bubble_screened_breit_v1"


def compute_static_photon_response(
    wfns_charge, wfns_transverse, quad, bispinor_v_q_path,
    meta, mesh_xy, *,
    sym=None,
    wfn=None,
    config=None,
    charge_response=None,
    current_contact: str = _WARD_SUBTRACTED_NO_PAIR,
    energy_reference=0.0,
    dyson_solver: str = "distributed",
    distrib_la_batched_route: str = "auto",
) -> StaticPhotonResponse:
    """Build the packed static photon body and optional coupled slab q=0 cell.

    ``head_correction=off`` preserves the finite-body bring-up path.  ``full``
    fits the nearest complete in-plane shell, replaces its charge sector from
    the incumbent symmetry-aware qsgw-head response, solves the headless body,
    and then applies the bounded bordered-Dyson reconstruction.  The sole
    vcoul sampler supplies every mini-BZ draw and bare propagator.  No photon
    body, shell-by-centroid tensor, G-vector transform, or symmetry operation
    is rebuilt here.
    """
    from .gw_config import HeadCorrection
    from .photon_layout import (
        PhotonBasisLayout, pack_photon_channel_vectors, pack_photon_operator,
    )
    from .v_q_bispinor import BispinorVqReader

    if str(dyson_solver).strip().lower() != "distributed":
        raise ValueError(
            "full static photon response requires dyson_solver='distributed'")
    head_policy = (HeadCorrection.OFF if config is None
                   else config.head.correction)
    if head_policy not in (HeadCorrection.OFF, HeadCorrection.FULL):
        raise ValueError(
            "full static photon response accepts only head_correction=off "
            f"or full; got {head_policy!r}")
    coupled_head = head_policy is HeadCorrection.FULL
    if coupled_head and (sym is None or wfn is None or config is None):
        raise ValueError(
            "coupled photon head completion requires sym, wfn, and config")
    if coupled_head and charge_response is None:
        raise ValueError(
            "coupled photon head completion requires the incumbent static "
            "qsgw-head charge response; refusing a second charge-head fit")

    g0_vectors = None
    with BispinorVqReader(bispinor_v_q_path, mesh_xy) as reader:
        if int(reader.n_q_total) != int(meta.nk_tot):
            raise ValueError(
                "full photon response requires full-BZ body blocks: "
                f"V reader has nq={reader.n_q_total}, meta.nk_tot={meta.nk_tot}")
        layout = PhotonBasisLayout.from_centroid_extents(
            reader.n_rmu_C, reader.n_rmu_T, mesh_xy)
        V_packed = pack_photon_operator(
            reader.get_tile, reader.n_q_total, layout, mesh_xy)
        if coupled_head:
            g0_x = tuple(reader.get_g0(A) for A in range(4))
            if any(vector is None for vector in g0_x):
                raise ValueError(
                    "coupled photon head requires all four canonical G=0 "
                    "channel vectors in v_q_bispinor.h5")
            jax.block_until_ready(g0_x)
            y_sharding = NamedSharding(mesh_xy, P(None, "y"))
            # This is the one deliberate one-index redistribution.  The
            # photon body never moves; both native centroid orientations are
            # needed for directed head/body projections and rank-4 updates.
            g0_y = tuple(jax.device_put(vector, y_sharding)
                         for vector in g0_x)
            jax.block_until_ready(g0_y)
            g0_vectors = PhotonG0Vectors(
                x_by_channel=g0_x,
                y_by_channel=g0_y,
                provenance=reader.g0_channel_provenance,
            )

    if jax.process_index() == 0:
        print(
            "  [photon response] EXPERIMENTAL no-pair bubble-screened "
            f"Breit; current_contact={current_contact}; "
            f"head_correction={head_policy.value}",
            flush=True)
    chi_packed = compute_photon_chi0(
        wfns_charge, wfns_transverse, quad, meta, mesh_xy, layout,
        current_contact=current_contact,
        energy_reference=energy_reference)

    coefficients = None
    g0_X = g0_Y = None
    if coupled_head:
        from ffi import _services
        _services.ensure_on_path()
        import vcoul

        geometry = vcoul.CoulombGeometry.from_wfn(wfn)
        coefficients = fit_static_photon_longwave_coefficients(
            chi_packed, g0_vectors, layout, meta, mesh_xy,
            sym=sym,
            bvec_cart_bohr=geometry.bvec,
            sys_dim=int(config.sys_dim),
            material_class=config.mpa.material_class,
            current_contact=current_contact,
            charge_response=charge_response,
        )
        g0_X = pack_photon_channel_vectors(
            g0_vectors.x_by_channel, layout, mesh_xy, axis_name="x")[0]
        g0_Y = pack_photon_channel_vectors(
            g0_vectors.y_by_channel, layout, mesh_xy, axis_name="y")[0]
        jax.block_until_ready((g0_X, g0_Y))
        # The packed Gamma rows are now independent device buffers.  Do not
        # retain four per-channel x/y copies across the distributed LU.
        del g0_vectors, g0_x, g0_y

    W_packed = solve_w(
        V_packed, chi_packed, meta, mesh_xy,
        dyson_solver="distributed",
        distrib_la_batched_route=distrib_la_batched_route)
    # The bare Hartree/exchange stage that follows does not depend on W and
    # could otherwise begin allocating its Green/operator workspaces while
    # the asynchronous distributed LU still owns A, RHS and donated chi.
    # Finish the response here, inside its timing/lifetime boundary.
    W_packed.block_until_ready()
    # This packed path bypasses screening._gate_w, so apply its two valid
    # static stage invariants here through the shared sanity owner.  Full-q
    # scalar reciprocity is deliberately not borrowed: Lorentz current
    # channels transform as vectors and require their own derived relation.
    from common import sanity
    sanity.refuse_nonfinite("static packed photon W", W_packed)
    if not sanity.check_hermitian(
            "static packed photon W[q=0]", W_packed[0], rtol=1.0e-6,
            always=True):
        raise ValueError(
            "static packed photon W[q=0] failed the canonical Hermiticity "
            "gate before coupled head/body folding")

    completion = None
    approximation = "experimental_no_pair_bubble_screened_breit_v1"
    if coupled_head:
        from .head_correction import complete_static_slab_photon_q0

        kernel = vcoul.get_kernel(config.sys_dim)
        sample_chunks = vcoul.iter_minibz_photon_samples(
            kernel, geometry, meta.kgrid,
            analytic_sphere=config.head.analytic_q0_sphere,
        )
        V_packed, W_packed, completion = complete_static_slab_photon_q0(
            V_packed, W_packed, coefficients, g0_X, g0_Y, sample_chunks,
            cell_volume=geometry.cell_volume,
            mesh_xy=mesh_xy,
        )
        # The completion mutates only the Gamma rows through one bare and nine
        # screened packed rank-4 updates.  Its tiny Dyson residual certifies
        # the 4x4 solve, not the left/right/conjugation convention of those
        # outer products, so reapply the shared construction invariants to the
        # corrected rows before Sigma can consume them.
        sanity.refuse_nonfinite(
            "completed static packed photon V[q=0]", V_packed[0])
        sanity.refuse_nonfinite(
            "completed static packed photon W[q=0]", W_packed[0])
        for name, operator in (
                ("V", V_packed[0]), ("W", W_packed[0])):
            if not sanity.check_hermitian(
                    f"completed static packed photon {name}[q=0]",
                    operator, rtol=1.0e-6, always=True):
                raise ValueError(
                    "coupled photon head/body completion produced a "
                    f"non-Hermitian {name}[q=0] row")
        approximation = coefficients.approximation
        if jax.process_index() == 0:
            print(
                "  [photon q0] coupled slab head/wings complete: "
                f"shell_rows={len(coefficients.shell_indices)}, "
                f"fit_residuals=(direct={coefficients.direct_relative_residual:.3e}, "
                f"Y={coefficients.Y_relative_residual:.3e}, "
                f"Z={coefficients.Z_relative_residual:.3e}), "
                f"Dyson={completion.max_dyson_relative_residual:.3e}",
                flush=True,
            )
            print(
                "  [photon q0 provenance] "
                f"approximation={coefficients.approximation}; "
                f"shell_indices={coefficients.shell_indices}; "
                f"directions_xy={coefficients.shell_directions_xy}; "
                f"radii_bohr^-1={coefficients.shell_radii_bohr_inv}; "
                f"design_ranks=({coefficients.linear_rank},"
                f"{coefficients.quadratic_rank}); "
                f"max_fitted_block_residuals=({coefficients.direct_relative_residual:.6e},"
                f"{coefficients.Y_relative_residual:.6e},"
                f"{coefficients.Z_relative_residual:.6e}); "
                f"g0={coefficients.g0_provenance}; "
                f"charge={coefficients.charge_response_source}; "
                f"hall_topological={coefficients.hall_topological_source}; "
                f"contact={coefficients.current_contact}",
                flush=True,
            )
        # The response object owns corrected V/W and compact evidence only;
        # retaining Y/Z here would pin two centroid-sized wing carriers through
        # all three Sigma contractions despite having no downstream consumer.
        del coefficients, g0_X, g0_Y
    return StaticPhotonResponse(
        layout=layout, V_packed=V_packed, W_packed=W_packed,
        current_contact=current_contact,
        head_completion=completion,
        approximation=approximation,
    )


def _chi0_multi_kernel_args(wfns, tau, alpha_rows, energy_reference):
    """Shared host prep for the multi-output χ₀ paths (compute + precompile).

    ``tau``: (L,) node vector (the fused static∪extra union on the probe-
    reuse path).  ``alpha_rows``: (n_out, L) RAW quadrature weights, one
    row per output, all on ``tau``.  Row 0 is normally the static weights
    (zero-padded onto any extra nodes — zero-weight nodes add exact
    zeros); further rows are probe representations on the same nodes.
    The χ₀ prefactor ``-2·exp(-τ·E_gap)`` folds into every row exactly as
    the single-output path folds it into its one α vector.
    """
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax
    tau = np.asarray(tau, dtype=np.float64)
    alpha_rows = np.asarray(alpha_rows, dtype=np.float64)
    if alpha_rows.ndim != 2 or alpha_rows.shape[1] != tau.shape[0]:
        raise ValueError(
            f"chi0 multi: alpha_rows shape {alpha_rows.shape} does not "
            f"match tau nodes ({tau.shape[0]},) — every row must be a "
            f"weight vector on quad.tau.")
    alpha_chi = -2.0 * alpha_rows * np.exp(-tau * E_gap)[None, :]
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )
    args = (
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )
    return args, alpha_rows.shape[0]


def compute_chi0_multi(wfns, tau, alpha_rows, meta, mesh_xy, *,
                       energy_reference=0.0):
    """χ₀ at several weight vectors over ONE τ sweep — see
    ``_get_chi_minimax_kernel(n_out>=2)``.  Returns an ``n_out``-tuple of
    flat-q (nq, μ, μ) arrays, one per row of ``alpha_rows``."""
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_multi_kernel_args(
        wfns, tau, alpha_rows, energy_reference)
    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid, n_out=n_out,
                                     **_chi_face_kwargs(wfns))
    return kernel(*args)


def precompile_chi0_multi(wfns, tau, alpha_rows, meta, mesh_xy, *,
                          energy_reference=None):
    """AOT lower+compile sibling of :func:`precompile_chi0` for the
    multi-output kernel."""
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    if len(np.asarray(tau)) == 0:
        return
    args, n_out = _chi0_multi_kernel_args(
        wfns, tau, alpha_rows, energy_reference)
    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid, n_out=n_out,
                                     **_chi_face_kwargs(wfns))
    kernel.lower(*args).compile()


def _chi0_contour_alpha_rows(tau, weight_rows, frequency_sign, z_values,
                             E_gap):
    """Complete contour weights for both independent-particle resolvents.

    ``frequency_sign=+1`` represents ``-1/(Delta-z)`` and ``-1`` represents
    ``-1/(Delta+z)``.  The device kernel evolves ``Delta-E_gap``, so this
    host-side coefficient supplies the omitted gap and requested frequency.
    """
    tau = np.asarray(tau, dtype=np.complex128)
    weight_rows = np.asarray(weight_rows, dtype=np.complex128)
    frequency_sign = np.asarray(frequency_sign)
    z_values = np.asarray(z_values, dtype=np.complex128)
    if (tau.ndim != 1 or z_values.ndim != 1 or z_values.size == 0 or
            frequency_sign.shape != tau.shape or
            weight_rows.shape != (z_values.size, tau.size)):
        raise ValueError(
            "chi0 contour requires tau/sign (L,), z (n_out,), and "
            "weight_rows (n_out,L)")
    if not np.all(np.isin(frequency_sign, (-1, 1))):
        raise ValueError("chi0 contour frequency_sign must contain only +/-1")
    exponent = -tau[None, :] * (
        float(E_gap) - frequency_sign[None, :] * z_values[:, None])
    return -weight_rows * np.exp(exponent)


def _chi0_contour_kernel_args(wfns, tau, weight_rows, frequency_sign,
                              z_values, energy_reference):
    """Prepare complex-frequency rows and the existing sharded operands."""
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    tau = np.asarray(tau, dtype=np.complex128)
    alpha_rows = _chi0_contour_alpha_rows(
        tau, weight_rows, frequency_sign, z_values, cmin - vmax)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(
            alpha_rows[0] if alpha_rows.shape[0] == 1 else alpha_rows,
            dtype=jnp.complex128),
    )
    args = (
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )
    return args, alpha_rows.shape[0]


def compute_chi0_contour(wfns, tau, weight_rows, frequency_sign, z_values,
                         meta, mesh_xy, *, energy_reference=0.0):
    """Evaluate several complex-frequency chi0 values in one node sweep.

    The scalar contour arrays select the two ``Delta +/- z`` resolvents.  All
    Green-function construction, FFTs, contraction, and sharding are the same
    operations used by :func:`compute_chi0`.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_contour_kernel_args(
        wfns, tau, weight_rows, frequency_sign, z_values, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=n_out, complex_contour=True,
        **_chi_face_kwargs(wfns))
    return kernel(*args)


def precompile_chi0_contour(wfns, tau, weight_rows, frequency_sign,
                            z_values, meta, mesh_xy, *,
                            energy_reference=None):
    """AOT sibling of :func:`compute_chi0_contour`."""
    if len(np.asarray(tau)) == 0:
        return
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_contour_kernel_args(
        wfns, tau, weight_rows, frequency_sign, z_values, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=n_out, complex_contour=True,
        **_chi_face_kwargs(wfns))
    kernel.lower(*args).compile()




def _occupation_support_slices(
        occupations,
        occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT):
    """Smallest contiguous f and (1-f) band supports without truncation.

    THIS IS THE ONE PLACE χ₀'s TWO GREEN'S FUNCTIONS GET THEIR BANDS, and
    unlike the Σ planner's mask it is a genuine COST cut: the returned slices
    index ``wfns.xn``/``yr``, so a band outside them is absent from the
    ``build_G_tau`` contraction rather than merely multiplied by a small
    weight.  ``occupation_support_bandwidth`` reads the same two slices to
    size the damped-line rule, so widening them also buys quadrature nodes.

    ``occupation_window_threshold`` is the OCCUPANCY at which a band leaves a
    support; the cut is on the branch WEIGHT — ``f`` on the occupied side,
    ``1 − f`` on the empty side, matching ``band_weight=occ_f`` and
    ``band_weight=1.0 - occ_u`` in the kernel — at the floor
    ``1 − threshold``, by MAGNITUDE.  Nothing is clipped: MP1 occupations
    overshoot [0, 1] and a wrong-side band's NEGATIVE weight is kept by
    ``abs`` exactly as the historical rule kept it (the argument is at
    ``gw.efermi.band_in_occupation_window``).  Partially occupied bands
    belong to both slices, as before.

    ``threshold = 1.0`` gives floor 0.0 and restores the historical exact
    rule (``occ != 0`` / ``occ != 1``) bit-for-bit; an insulating table, whose
    weights are exactly 0 or 1, gives the same two slices at EVERY threshold,
    since ``abs(1) > floor`` and ``abs(0) > floor`` are threshold-independent
    on [0.5, 1.0].
    """
    occ = np.asarray(jax.device_get(occupations), dtype=np.float64)
    if occ.ndim != 2:
        raise ValueError(
            "fractional contour occupations must have shape (nk, nb), got "
            + str(occ.shape))
    floor = occupation_weight_floor(occupation_window_threshold)
    f_support = np.any(band_in_occupation_window(occ, floor), axis=0)
    u_support = np.any(band_in_occupation_window(1.0 - occ, floor), axis=0)
    if not np.any(f_support) or not np.any(u_support):
        raise ValueError(
            "fractional contour chi0 needs at least one band clearing the "
            f"occupation window on each side (threshold "
            f"{float(occupation_window_threshold)!r} ⇒ |weight| > {floor!r}); "
            "raise occupation_window_threshold toward 1.0 to widen the "
            "supports, or check the occupation table")
    f_idx = np.flatnonzero(f_support)
    u_idx = np.flatnonzero(u_support)
    return (
        slice(int(f_idx[0]), int(f_idx[-1]) + 1),
        slice(int(u_idx[0]), int(u_idx[-1]) + 1),
    )


def _chi0_fractional_contour_args(
    wfns,
    time_nodes,
    weight_rows,
    z_values,
    occupations,
    energy_reference,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """Prepare the exact finite-occupation positive-time response.

    Layout-dispatched (``docs/architecture/fractional_chi0_response_
    face.md``).  ``legacy``: windowed ψ views + windowed energies/
    occupations, the historical body, unchanged.  ``face``: the FULL
    [b0,b4) face carrier (``psi_mun``/``psi_nmu`` cannot be band-sliced —
    obstacle #3) plus occupations WEIGHTED to zero exactly outside the
    SAME two contiguous support windows legacy slices to — "weight, don't
    window" — which reproduces legacy's windowed contraction bit-for-bit
    up to summation-order roundoff (the contraction is bilinear in ψ).
    """
    time_nodes = np.asarray(time_nodes, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.complex128)
    if time_nodes.ndim != 1 or time_nodes.size == 0:
        raise ValueError(
            "fractional contour time_nodes must be a nonempty 1-D array")
    if not np.all(np.isfinite(time_nodes)) or np.any(time_nodes < 0.0):
        raise ValueError(
            "fractional contour time_nodes must be finite and nonnegative")
    if z_values.ndim != 1 or z_values.size == 0:
        raise ValueError(
            "fractional contour z_values must be a nonempty 1-D array")
    if np.any(z_values.imag <= 0.0):
        raise ValueError(
            "fractional retarded contour requires Im(z) > 0; the exact static "
            "divided-difference limit is a separate quadrature target")

    weights = np.asarray(weight_rows, dtype=np.complex128)
    if weights.ndim == 1:
        if weights.shape != time_nodes.shape:
            raise ValueError(
                "fractional contour 1-D weights must match time_nodes")
        weights = np.broadcast_to(
            weights[None, :], (z_values.size, time_nodes.size))
    if weights.shape != (z_values.size, time_nodes.size):
        raise ValueError(
            "fractional contour weights must have shape (n_z, n_time)")
    projection_rows = weights * np.exp(
        1j * z_values[:, None] * time_nodes[None, :])

    occ_full = wfns.occ if occupations is None else occupations
    if tuple(occ_full.shape) != tuple(wfns.enk.shape):
        raise ValueError(
            "fractional contour occupation shape {} does not match energies "
            "{}".format(occ_full.shape, wfns.enk.shape))
    f_slice, u_slice = _occupation_support_slices(
        occ_full, occupation_window_threshold)
    eref = 0.0 if energy_reference is None else float(energy_reference)
    if wfns.layout == "legacy":
        args = (
            jnp.asarray(time_nodes, dtype=jnp.float64),
            jnp.asarray(projection_rows, dtype=jnp.complex128),
            wfns.xn(f_slice),
            wfns.yr(f_slice),
            wfns.yr(u_slice),
            wfns.xn(u_slice),
            wfns.enk[:, f_slice],
            wfns.enk[:, u_slice],
            occ_full[:, f_slice],
            occ_full[:, u_slice],
            jnp.asarray(eref, dtype=jnp.float64),
        )
        return args, z_values.size
    # face: "weight, don't window" -- occ_full multiplied by a {0,1}
    # indicator that is 1 exactly inside f_slice/u_slice, 0 outside,
    # reproducing legacy's windowed sum bit-for-bit (see this function's
    # docstring).  enk stays the bundle's FULL [b0,b4) table, unsliced,
    # shared by both roles.  occ_f_face/occ_u_face are the FINAL
    # band_weight values the face kernel applies directly (see
    # _get_chi_fractional_contour_kernel_face's own comment on why the
    # 1-occ inversion for the Gu role must happen BEFORE masking, not
    # after -- masking a RAW occupation then inverting turns the
    # correctly-excluded outside-window positions into 1.0 instead of
    # 0.0).
    nb_full = int(wfns.slices.nb_full)
    idx = np.arange(nb_full)
    f_ind = jnp.asarray(
        (idx >= f_slice.start) & (idx < f_slice.stop), dtype=occ_full.dtype)
    u_ind = jnp.asarray(
        (idx >= u_slice.start) & (idx < u_slice.stop), dtype=occ_full.dtype)
    occ_f_face = occ_full * f_ind[None, :]
    occ_u_face = (1.0 - occ_full) * u_ind[None, :]
    args = (
        jnp.asarray(time_nodes, dtype=jnp.float64),
        jnp.asarray(projection_rows, dtype=jnp.complex128),
        wfns.psi_mun,
        wfns.psi_nmu,
        wfns.enk,
        occ_f_face,
        occ_u_face,
        jnp.asarray(eref, dtype=jnp.float64),
    )
    return args, z_values.size


def compute_chi0_contour_fractional(
    wfns,
    time_nodes,
    weight_rows,
    z_values,
    meta,
    mesh_xy,
    *,
    occupations=None,
    energy_reference=0.0,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """Evaluate retarded finite-occupation chi0 at complex frequencies.

    weight_rows contains the positive real-time quadrature weights; this
    routine supplies exp(i*z*t) and both exact Keldysh terms.  It does not
    implement z=0: the gapless static limit contains the finite divided
    difference -df/dE and requires its own certified integration rule.

    ``occupation_window_threshold`` is the OCCUPANCY at which a band leaves
    one of the two Green's-function supports; it MUST be the same value the
    caller gave ``occupation_support_bandwidth``, or the damped-line rule is
    sized for transitions the band slices no longer contain.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_fractional_contour_args(
        wfns,
        time_nodes,
        weight_rows,
        z_values,
        occupations,
        energy_reference,
        occupation_window_threshold,
    )
    kernel = _get_chi_fractional_contour_kernel(
        mesh_xy, kgrid, n_out, **_chi_face_kwargs(wfns))
    values = kernel(*args)
    return values[0] if n_out == 1 else values


def _fractional_pair_scan(
    psi_x_a, psi_y_a, psi_x_b, psi_y_b, energy_a, energy_b,
    occ_a, occ_b, surface_a, surface_b, z_values, *, nb_logical, tile,
):
    """Ordered-pair scan shared by the static and direct-frequency kernels.

    The ``a`` operands ride at k; the ``b`` operands ride at whatever k row
    the caller supplied — the same arrays for the Gamma kernel, the k−q
    rolled arrays for the finite-q kernel.  TASTE 6 judgment (the same
    ruling the Gamma kernel carries): the ordered band-pair index exists
    only inside one tile step beside the centroid axes; the per-rank
    transient is the two density tiles,
    ``nk·(nmu_x/P_x + nmu_y/P_y)·tile²·16 B`` per step (Na 48b, 8³ k,
    tile 8: tens of MB).  At z=0 it uses the analytic divided difference;
    at nonzero z it evaluates ``(f_a-f_b)/(E_a-E_b+z)`` directly.  The
    latter is the exact escape hatch for the ONE shifted-origin metal MPA
    sample whose damped contour would require about a million nodes; it is
    not a general full-frequency route.  The certified separable rational-f
    service (docs/theory/finite-occupation-screening.md §static) is the
    staged scaling path behind the same public API.
    """
    nk, nspinor, nmu_x, nb = psi_x_a.shape
    nmu_y = psi_y_a.shape[2]
    nb_pad = ((int(nb) + tile - 1) // tile) * tile
    pad = nb_pad - int(nb)
    pad4 = ((0, 0), (0, 0), (0, 0), (0, pad))
    pad2 = ((0, 0), (0, pad))
    pa_x_full = jnp.pad(psi_x_a, pad4)
    pb_x_full = jnp.pad(psi_x_b, pad4)
    pa_y_full = jnp.pad(psi_y_a, pad4)
    pb_y_full = jnp.pad(psi_y_b, pad4)
    ea_full = jnp.pad(energy_a, pad2)
    eb_full = jnp.pad(energy_b, pad2)
    fa_full = jnp.pad(occ_a, pad2)
    fb_full = jnp.pad(occ_b, pad2)
    sa_full = jnp.pad(surface_a, pad2)
    sb_full = jnp.pad(surface_b, pad2)
    z = jnp.asarray(z_values, dtype=jnp.complex128)
    ntiles = nb_pad // tile

    def _pair_tile(accumulator, flat_index):
        ia = (flat_index // ntiles) * tile
        ib = (flat_index % ntiles) * tile
        pa_x = jax.lax.dynamic_slice(
            pa_x_full, (0, 0, 0, ia), (nk, nspinor, nmu_x, tile))
        pb_x = jax.lax.dynamic_slice(
            pb_x_full, (0, 0, 0, ib), (nk, nspinor, nmu_x, tile))
        pa_y = jax.lax.dynamic_slice(
            pa_y_full, (0, 0, 0, ia), (nk, nspinor, nmu_y, tile))
        pb_y = jax.lax.dynamic_slice(
            pb_y_full, (0, 0, 0, ib), (nk, nspinor, nmu_y, tile))
        ea = jax.lax.dynamic_slice(ea_full, (0, ia), (nk, tile))
        eb = jax.lax.dynamic_slice(eb_full, (0, ib), (nk, tile))
        fa = jax.lax.dynamic_slice(fa_full, (0, ia), (nk, tile))
        fb = jax.lax.dynamic_slice(fb_full, (0, ib), (nk, tile))
        sa = jax.lax.dynamic_slice(sa_full, (0, ia), (nk, tile))
        sb = jax.lax.dynamic_slice(sb_full, (0, ib), (nk, tile))

        de = ea[:, :, None] - eb[:, None, :]
        df = fa[:, :, None] - fb[:, None, :]
        scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(ea[:, :, None]),
                        jnp.abs(eb[:, None, :])),
        )
        separated = (
            jnp.abs(de) > 64.0 * jnp.finfo(jnp.float64).eps * scale)
        # surface_weight is -df/dE.  The average is exact for a truly
        # degenerate pair and is the stable midpoint limit for a pair
        # closer than floating-point energy resolution.
        diagonal_limit = -0.5 * (sa[:, :, None] + sb[:, None, :])
        static_divided = jnp.where(
            separated, df / jnp.where(separated, de, 1.0),
            diagonal_limit)
        dynamic = df[None, :, :, :] / (
            de[None, :, :, :] + z[:, None, None, None])
        weights = jnp.where(
            (z == 0.0)[:, None, None, None],
            static_divided[None, :, :, :], dynamic)
        ga = ia + jnp.arange(tile)
        gb = ib + jnp.arange(tile)
        logical = (
            (ga[:, None] < int(nb_logical))
            & (gb[None, :] < int(nb_logical))
        )[None, :, :]
        weights = jnp.where(logical[None, :, :, :], weights, 0.0)

        # d_ab(mu) = sum_s psi_a(mu) conj(psi_b(mu)).  The spinor
        # component is summed here, so scalar, two-component and future
        # four-component wavefunctions share this exact kernel.
        density_x = jnp.einsum(
            "ksma,ksmb->kmab", pa_x, jnp.conj(pb_x), optimize=True)
        density_y = jnp.einsum(
            "ksna,ksnb->knab", pa_y, jnp.conj(pb_y), optimize=True)
        contribution = jnp.einsum(
            "zkab,kmab,knab->zmn", weights, density_x,
            jnp.conj(density_y), optimize=True)
        return accumulator + contribution, None

    zero = jnp.zeros((z.size, nmu_x, nmu_y), dtype=jnp.complex128)
    chi, _ = jax.lax.scan(
        _pair_tile, zero, jnp.arange(ntiles * ntiles), unroll=1)
    return chi / jnp.sqrt(jnp.asarray(nk, jnp.float64))


def _get_chi_static_fractional_gamma_kernel(
    mesh_xy: Mesh, *, nb_logical: int, pair_tile: int,
):
    """Exact q=0 divided-difference body, streamed by band-pair tiles.

    The output centroid axes are the process mesh axes.  Bands and k points
    remain replicated, so every rank performs the same number of pair-tile
    steps for its unique ``(mu_x, nu_y)`` output tile.  No rank forms an
    ``(nk, nb, nb, n_mu)`` transition-density array.
    """
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_XN_SPEC, PSI_YN_SPEC

    tile = int(pair_tile)
    key = ("static_fractional_gamma", id(mesh_xy), int(nb_logical), tile)
    hit = _chi_minimax_kernel_cache.get(key)
    if hit is not None:
        return hit

    def _local(psi_xn, psi_yn, energies, occupations, surface_weight):
        return _fractional_pair_scan(
            psi_xn, psi_yn, psi_xn, psi_yn, energies, energies,
            occupations, occupations, surface_weight, surface_weight,
            jnp.zeros((1,), dtype=jnp.complex128),
            nb_logical=nb_logical, tile=tile)

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(PSI_XN_SPEC, PSI_YN_SPEC, P(None, None), P(None, None),
                  P(None, None)),
        out_specs=P(None, "x", "y"),
        check_vma=False,
    ))
    _chi_minimax_kernel_cache[key] = kernel
    return kernel


def _get_chi_fractional_q_kernel(
    mesh_xy: Mesh, *, nb_logical: int, pair_tile: int, n_z: int,
):
    """Finite-q direct ordered-pair kernel: b rides at k−q.

    The caller supplies the flat ``k → k−q`` map for one stored q row;
    every b-side operand (both densities, energies, occupations, surface
    weights) is rolled by it before the shared ordered-pair scan.  The
    map is replicated and the ψ k axis is replicated on this mesh, so the
    gather is rank-local — no collectives are added over the Gamma kernel.
    """
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_XN_SPEC, PSI_YN_SPEC

    tile = int(pair_tile)
    key = ("direct_fractional_q", id(mesh_xy), int(nb_logical), tile,
           int(n_z))
    hit = _chi_minimax_kernel_cache.get(key)
    if hit is not None:
        return hit

    def _local(psi_xn, psi_yn, kminq_idx, energies, occupations,
               surface_weight, z_values):
        pb_x = jnp.take(psi_xn, kminq_idx, axis=0)
        pb_y = jnp.take(psi_yn, kminq_idx, axis=0)
        eb = jnp.take(energies, kminq_idx, axis=0)
        fb = jnp.take(occupations, kminq_idx, axis=0)
        sb = jnp.take(surface_weight, kminq_idx, axis=0)
        return _fractional_pair_scan(
            psi_xn, psi_yn, pb_x, pb_y, energies, eb,
            occupations, fb, surface_weight, sb,
            z_values,
            nb_logical=nb_logical, tile=tile)

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(PSI_XN_SPEC, PSI_YN_SPEC, P(None), P(None, None),
                  P(None, None), P(None, None), P(None)),
        out_specs=P(None, "x", "y"),
        check_vma=False,
    ))
    _chi_minimax_kernel_cache[key] = kernel
    return kernel


# ============================================================================
# Exact finite-occupation response — face-layout ordered-pair kernel
# ============================================================================
#
# See docs/architecture/fractional_chi0_response_face.md for the full
# derivation.  ``_fractional_pair_scan`` above is FROZEN — this session
# does not edit it, matching every other legacy/face split in this
# codebase (isdf.core._c_q_legacy/_c_q_face, _z_q_legacy/_z_q_face,
# greens_function_kernel._legacy_build_G/_face_build_G).  The per-pair
# physics below (divided-difference weight, two density contractions, the
# final zmn einsum) is intentionally re-typed rather than shared through a
# helper both functions call, for the same reason.


def _fractional_pair_scan_face(
    psi_mun_a, psi_nmu_a, psi_mun_b, psi_nmu_b, energy_a, energy_b,
    occ_a, occ_b, surface_a, surface_b, z_values, *,
    nb_full, nb_logical, tile,
):
    """Face-layout sibling of :func:`_fractional_pair_scan`.

    Runs INSIDE a shard_map body whose in_specs give this rank only its
    own local shard of the persistent face carrier (``PSI_MUN_SPEC``/
    ``PSI_NMU_SPEC``) — never a resident, band-replicated single-axis
    copy.  The divided-difference weight depends JOINTLY on both band
    indices' energies/occupations, so it cannot collapse to the
    one-particle G GEMM (see the design doc); instead, each band TILE
    this scan touches is reconstructed on demand from the persistent
    carrier via a masked-gather + ``psum`` on BOTH mesh axes —
    ``isdf.core._z_q_face``'s idiom, generalized from one axis (its own
    ``psum('y')`` X-operand reconstruction) to both, since this kernel
    needs BOTH ψ orientations (μ-on-X from ``psi_mun``, μ-on-Y from
    ``psi_nmu``) at BOTH pair-index roles.

    ``psi_mun_a``/``psi_nmu_a`` and ``psi_mun_b``/``psi_nmu_b`` are the
    (already rank-local) face-carrier shards for the "a" and "b" pair-
    index roles respectively — the SAME array for both, at Gamma; the
    caller's own k−q-rolled copy for the finite-q kernel (rolling a
    REPLICATED k axis is a rank-local ``jnp.take``, unaffected by layout).
    ``energy_*``/``occ_*``/``surface_*`` are ``(nk, nb_full)`` REPLICATED
    (already zero-padded by the caller up to ``nb_full`` if its own
    window was narrower — safe, since any padded position is
    ``>= nb_logical`` and hence excluded by the mask below regardless).

    Tiling: nested scans, outer over the "a" band tile (reconstructed
    ONCE per outer step, reused across the whole inner sweep), inner over
    "b" (reconstructed fresh every step) — bounding the resident working
    set to O(tile) band-widths at any instant.  See the design doc's
    "Tiling choice" for the communication-cost accounting and why a
    ``ppermute`` ring was considered and deferred, not needed at the
    scale this session gated.
    """
    nk = psi_mun_a.shape[0]
    ns = psi_mun_a.shape[1]
    nmu_x_loc = psi_mun_a.shape[2]
    nmu_y_loc = psi_nmu_a.shape[3]
    shard_w_y = psi_mun_a.shape[3]   # psi_mun's own 'y'-shard width (bands)
    shard_w_x = psi_nmu_a.shape[1]   # psi_nmu's own 'x'-shard width (bands)
    y_idx = jax.lax.axis_index('y')
    x_idx = jax.lax.axis_index('x')

    def _gather_mun(psi_mun_local, g_lo):
        """(nk, s, mu_X_loc, tile) un-conjugated, present on every rank —
        masked-gather + psum('y') from psi_mun's local shard (bands on
        'y').  psi_mun's own axis order (nk, s, mu, n) already matches
        PSI_XN_SPEC's (nk, s, mu_X, n) -- no reorder needed."""
        p = jnp.arange(tile, dtype=jnp.int32)
        global_band = g_lo + p
        owner = global_band // shard_w_y
        owns = owner == y_idx
        local_idx = jnp.clip(
            global_band - y_idx * shard_w_y, 0, shard_w_y - 1)
        gathered = jnp.take(psi_mun_local, local_idx, axis=3)
        gathered = jnp.where(owns[None, None, None, :], gathered, 0)
        return jax.lax.psum(gathered, 'y')

    def _gather_nmu(psi_nmu_local, g_lo):
        """(nk, s, mu_Y_loc, tile) un-conjugated, present on every rank —
        masked-gather + psum('x') from psi_nmu's local shard (bands on
        'x'), then a LOCAL (no-comm, bounded-size — this tile is `tile`
        bands wide, not nb_full) axis reorder: psi_nmu stores (nk, n, s,
        mu), band axis SECOND, so the post-gather (nk, tile, s, mu_Y_loc)
        needs one transpose to match PSI_YN_SPEC's (nk, s, mu, n) order."""
        p = jnp.arange(tile, dtype=jnp.int32)
        global_band = g_lo + p
        owner = global_band // shard_w_x
        owns = owner == x_idx
        local_idx = jnp.clip(
            global_band - x_idx * shard_w_x, 0, shard_w_x - 1)
        gathered = jnp.take(psi_nmu_local, local_idx, axis=1)
        gathered = jnp.where(owns[None, :, None, None], gathered, 0)
        gathered = jax.lax.psum(gathered, 'x')
        return jnp.transpose(gathered, (0, 2, 3, 1))

    nb_pad = ((int(nb_full) + tile - 1) // tile) * tile
    pad = nb_pad - int(nb_full)
    pad2 = ((0, 0), (0, pad))
    ea_full = jnp.pad(energy_a, pad2)
    eb_full = jnp.pad(energy_b, pad2)
    fa_full = jnp.pad(occ_a, pad2)
    fb_full = jnp.pad(occ_b, pad2)
    sa_full = jnp.pad(surface_a, pad2)
    sb_full = jnp.pad(surface_b, pad2)
    z = jnp.asarray(z_values, dtype=jnp.complex128)
    ntiles = nb_pad // tile

    def _pair_contribution(pa_x, pb_x, pa_y, pb_y, ea, eb, fa, fb, sa, sb,
                           ga, gb):
        # EXACT mirror of _fractional_pair_scan's per-pair body -- see
        # that function for the physics derivation/comments.
        de = ea[:, :, None] - eb[:, None, :]
        df = fa[:, :, None] - fb[:, None, :]
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(ea[:, :, None]), jnp.abs(eb[:, None, :])))
        separated = jnp.abs(de) > 64.0 * jnp.finfo(jnp.float64).eps * scale
        diagonal_limit = -0.5 * (sa[:, :, None] + sb[:, None, :])
        static_divided = jnp.where(
            separated, df / jnp.where(separated, de, 1.0), diagonal_limit)
        dynamic = df[None, :, :, :] / (de[None, :, :, :] + z[:, None, None, None])
        weights = jnp.where(
            (z == 0.0)[:, None, None, None], static_divided[None, :, :, :],
            dynamic)
        logical = (
            (ga[:, None] < int(nb_logical)) & (gb[None, :] < int(nb_logical))
        )[None, :, :]
        weights = jnp.where(logical[None, :, :, :], weights, 0.0)
        density_x = jnp.einsum(
            "ksma,ksmb->kmab", pa_x, jnp.conj(pb_x), optimize=True)
        density_y = jnp.einsum(
            "ksna,ksnb->knab", pa_y, jnp.conj(pb_y), optimize=True)
        return jnp.einsum(
            "zkab,kmab,knab->zmn", weights, density_x, jnp.conj(density_y),
            optimize=True)

    zero = jnp.zeros((z.size, nmu_x_loc, nmu_y_loc), dtype=jnp.complex128)

    def _outer(acc, ia_step):
        ia = ia_step * tile
        ga = ia + jnp.arange(tile)
        a_x = _gather_mun(psi_mun_a, ia)
        a_y = _gather_nmu(psi_nmu_a, ia)
        ea = jax.lax.dynamic_slice(ea_full, (0, ia), (nk, tile))
        fa = jax.lax.dynamic_slice(fa_full, (0, ia), (nk, tile))
        sa = jax.lax.dynamic_slice(sa_full, (0, ia), (nk, tile))

        def _inner(acc_inner, ib_step):
            ib = ib_step * tile
            gb = ib + jnp.arange(tile)
            b_x = _gather_mun(psi_mun_b, ib)
            b_y = _gather_nmu(psi_nmu_b, ib)
            eb = jax.lax.dynamic_slice(eb_full, (0, ib), (nk, tile))
            fb = jax.lax.dynamic_slice(fb_full, (0, ib), (nk, tile))
            sb = jax.lax.dynamic_slice(sb_full, (0, ib), (nk, tile))
            contribution = _pair_contribution(
                a_x, b_x, a_y, b_y, ea, eb, fa, fb, sa, sb, ga, gb)
            return acc_inner + contribution, None

        acc_inner, _ = jax.lax.scan(
            _inner, jnp.zeros_like(acc), jnp.arange(ntiles), unroll=1)
        return acc + acc_inner, None

    chi, _ = jax.lax.scan(_outer, zero, jnp.arange(ntiles), unroll=1)
    return chi / jnp.sqrt(jnp.asarray(nk, jnp.float64))


def _get_chi_static_fractional_gamma_kernel_face(
    mesh_xy: Mesh, *, nb_full: int, nb_logical: int, pair_tile: int,
):
    """Face-layout sibling of :func:`_get_chi_static_fractional_gamma_kernel`.
    See that function's docstring for the physics; only the operand
    source differs — the persistent face carrier, never a resident
    band-replicated copy (:func:`_fractional_pair_scan_face`)."""
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_MUN_SPEC, PSI_NMU_SPEC

    tile = int(pair_tile)
    key = ("static_fractional_gamma_face", id(mesh_xy), int(nb_full),
           int(nb_logical), tile)
    hit = _chi_minimax_kernel_cache.get(key)
    if hit is not None:
        return hit

    def _local(psi_mun, psi_nmu, energies, occupations, surface_weight):
        return _fractional_pair_scan_face(
            psi_mun, psi_nmu, psi_mun, psi_nmu, energies, energies,
            occupations, occupations, surface_weight, surface_weight,
            jnp.zeros((1,), dtype=jnp.complex128),
            nb_full=nb_full, nb_logical=nb_logical, tile=tile)

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(PSI_MUN_SPEC, PSI_NMU_SPEC, P(None, None), P(None, None),
                  P(None, None)),
        out_specs=P(None, "x", "y"),
        check_vma=False,
    ))
    _chi_minimax_kernel_cache[key] = kernel
    return kernel


def _get_chi_fractional_q_kernel_face(
    mesh_xy: Mesh, *, nb_full: int, nb_logical: int, pair_tile: int,
    n_z: int,
):
    """Face-layout sibling of :func:`_get_chi_fractional_q_kernel`.  The
    k−q roll is unaffected by layout (a rank-local ``jnp.take`` on the
    REPLICATED k axis, same as legacy's own roll on ``PSI_XN_SPEC``'s
    replicated k axis); only the band-tile reconstruction differs."""
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_MUN_SPEC, PSI_NMU_SPEC

    tile = int(pair_tile)
    key = ("direct_fractional_q_face", id(mesh_xy), int(nb_full),
           int(nb_logical), tile, int(n_z))
    hit = _chi_minimax_kernel_cache.get(key)
    if hit is not None:
        return hit

    def _local(psi_mun, psi_nmu, kminq_idx, energies, occupations,
               surface_weight, z_values):
        psi_mun_b = jnp.take(psi_mun, kminq_idx, axis=0)
        psi_nmu_b = jnp.take(psi_nmu, kminq_idx, axis=0)
        eb = jnp.take(energies, kminq_idx, axis=0)
        fb = jnp.take(occupations, kminq_idx, axis=0)
        sb = jnp.take(surface_weight, kminq_idx, axis=0)
        return _fractional_pair_scan_face(
            psi_mun, psi_nmu, psi_mun_b, psi_nmu_b, energies, eb,
            occupations, fb, surface_weight, sb, z_values,
            nb_full=nb_full, nb_logical=nb_logical, tile=tile)

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(PSI_MUN_SPEC, PSI_NMU_SPEC, P(None), P(None, None),
                  P(None, None), P(None, None), P(None)),
        out_specs=P(None, "x", "y"),
        check_vma=False,
    ))
    _chi_minimax_kernel_cache[key] = kernel
    return kernel


def compute_chi0_static_fractional_gamma(
    wfns,
    energies_kn_ry,
    occupations_kn,
    surface_weight_kn,
    meta,
    mesh_xy,
    *,
    nb_logical: int,
):
    r"""Return the exact static fractional-occupation chi0 at Gamma.

    The ordered-pair kernel evaluates

    ``(f_ka-f_kb)/(E_ka-E_kb)``

    and uses ``df/dE`` on the degenerate diagonal.  The supplied surface
    table owns that diagonal limit; the QSGW metal path supplies periodic
    tetrahedron weights, while off-diagonal pairs retain the carried MP1
    occupations.  The returned ``(1,n_mu,n_mu)`` array has the historical
    raw-chi normalization expected by :func:`solve_w`.

    This direct tiled implementation is the exact finite-band fallback.  A
    future certified separable divided-difference minimax target can replace
    its internals without changing this API or the Dyson/head callers.
    """
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if e.ndim != 2 or f.shape != e.shape or surface.shape != e.shape:
        raise ValueError(
            "static fractional chi requires matching (nk,nb) energies, "
            f"occupations, and surface weights; got {e.shape}, {f.shape}, "
            f"{surface.shape}")
    if int(e.shape[0]) != int(meta.nk_tot):
        raise ValueError(
            f"static fractional chi has nk={e.shape[0]}, expected "
            f"meta.nk_tot={meta.nk_tot}")
    if not (0 < int(nb_logical) <= int(e.shape[1])):
        raise ValueError(
            f"static fractional chi needs 0 < nb_logical <= {e.shape[1]}; "
            f"got {nb_logical}")
    if wfns.layout == "legacy":
        if int(wfns.psi_xn.shape[-1]) < int(e.shape[1]):
            raise ValueError(
                "centroid wavefunctions do not cover the static bands")
        psi_x = wfns.psi_xn[..., : int(e.shape[1])]
        psi_y = wfns.psi_yn[..., : int(e.shape[1])]
        return _get_chi_static_fractional_gamma_kernel(
            mesh_xy,
            nb_logical=int(nb_logical),
            pair_tile=_STATIC_FRACTIONAL_PAIR_TILE,
        )(psi_x, psi_y, e, f, surface)
    # face: the persistent carrier spans the FULL [b0,b4) window and
    # cannot be band-sliced (obstacle #3) -- pad the caller's
    # energies/occupations/surface table up to nb_full instead (any
    # padded position is >= nb_logical, so the pair kernel's own
    # nb_logical mask excludes it regardless).
    nb_full = int(wfns.slices.nb_full)
    if nb_full < int(e.shape[1]):
        raise ValueError(
            "compute_chi0_static_fractional_gamma(layout='face'): the "
            f"loaded face carrier covers only {nb_full} bands, fewer than "
            f"the {e.shape[1]} the caller's tables provide")
    bpad = nb_full - int(e.shape[1])
    pad2 = ((0, 0), (0, bpad))
    e_full = jnp.pad(e, pad2)
    f_full = jnp.pad(f, pad2)
    surface_full = jnp.pad(surface, pad2)
    return _get_chi_static_fractional_gamma_kernel_face(
        mesh_xy,
        nb_full=nb_full,
        nb_logical=int(nb_logical),
        pair_tile=_STATIC_FRACTIONAL_PAIR_TILE,
    )(wfns.psi_mun, wfns.psi_nmu, e_full, f_full, surface_full)


def occupation_support_bandwidth(
        energies_kn_ry, occupations_kn,
        occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT):
    """Largest transition energy over the occupation supports, Ry.

    ``max(E over the (1-f) support) − min(E over the f support)`` over the
    SAME two slices :func:`_occupation_support_slices` hands the χ₀ kernel,
    so the rule bandwidth and the bands it must resolve can never disagree —
    which is why the threshold is an argument here rather than a second
    default.  An MP1 overshoot band at a support edge is included, by
    magnitude.  This — not ``quad.x_max`` — sizes the damped-line rule
    bandwidth on metal plans, where the occupied and empty supports overlap.
    """
    e = np.asarray(jax.device_get(energies_kn_ry), dtype=np.float64)
    f_slice, u_slice = _occupation_support_slices(
        occupations_kn, occupation_window_threshold)
    return float(np.max(e[:, u_slice]) - np.min(e[:, f_slice]))


def compute_chi0_static_fractional(
    wfns,
    meta,
    mesh_xy,
    *,
    occupation_state,
    kminq_rows,
    nb_logical=None,
):
    """Exact static finite-occupation chi0 for every stored q row.

    The finite-q generalization of
    :func:`compute_chi0_static_fractional_gamma`: for wedge row j the b
    side of every ordered pair rides at ``k − q_j`` through the caller's
    precomputed flat map ``kminq_rows[j]`` (``common.kq_mapping``), and
    the divided difference ``(f_a(k)−f_b(k−q))/(E_a(k)−E_b(k−q))`` uses
    the analytic MP1 ``−df/dE`` midpoint limit on accidentally degenerate
    pairs.  This is the literal static member of the shared ordered-pair
    evaluator; the metal MPA shifted-origin slot instead calls
    :func:`compute_chi0_direct_fractional` at its stamped nonzero ``z``.
    Returns ``(n_q, n_mu, n_mu)``
    wedge rows in the raw-chi normalization expected by :func:`solve_w`,
    sharded ``P(None, 'x', 'y')``.
    """
    return compute_chi0_direct_fractional(
        wfns, np.asarray([0.0j], dtype=np.complex128), meta, mesh_xy,
        occupation_state=occupation_state, kminq_rows=kminq_rows,
        nb_logical=nb_logical)


def compute_chi0_direct_fractional(
    wfns,
    z_values,
    meta,
    mesh_xy,
    *,
    occupation_state,
    kminq_rows,
    nb_logical=None,
):
    """Exact finite-occupation chi0 at selected complex frequencies.

    This is the ordered-pair escape hatch for isolated points at which the
    damped-contour evaluator is unaffordable.  It shares the static kernel's
    band-pair scan and distributed centroid output.  A zero entry uses the
    MP1 divided-difference limit; every nonzero entry is evaluated at its
    literal complex coordinate.  With one frequency the returned shape is
    ``(n_q,n_mu,n_mu)``; otherwise it is ``(n_z,n_q,n_mu,n_mu)``.
    """
    from gw.efermi import mp1_negative_derivative

    family = getattr(occupation_state, "smearing_family", None)
    if family != "mp1":
        raise ValueError(
            "GATE static_fractional_needs_mp1: direct fractional chi "
            "requires OccupationState.smearing_family == 'mp1' "
            f"(analytic -df/dE diagonal); got {family!r}. A step-function "
            "static body is the insulating compute_chi0 kernel's job. "
            "FALSE case: smearing_family == 'mp1'.")
    e = jnp.asarray(wfns.enk, dtype=jnp.float64)
    f = jnp.asarray(occupation_state.f_kn, dtype=jnp.float64)
    if f.shape != e.shape:
        raise ValueError(
            f"static fractional chi occupations {f.shape} do not match "
            f"energies {e.shape}")
    if int(e.shape[0]) != int(meta.nk_tot):
        raise ValueError(
            f"static fractional chi has nk={e.shape[0]}, expected "
            f"meta.nk_tot={meta.nk_tot}")
    nb = int(e.shape[1])
    nb_log = nb if nb_logical is None else int(nb_logical)
    if not (0 < nb_log <= nb):
        raise ValueError(
            f"static fractional chi needs 0 < nb_logical <= {nb}; got "
            f"{nb_logical}")
    kmq = np.asarray(kminq_rows, dtype=np.int32)
    if kmq.ndim != 2 or kmq.shape[1] != int(e.shape[0]):
        raise ValueError(
            "static fractional chi kminq_rows must have shape (n_q, nk="
            f"{e.shape[0]}); got {kmq.shape}")
    z = np.asarray(z_values, dtype=np.complex128)
    if z.ndim != 1 or not z.size or not np.all(np.isfinite(z)):
        raise ValueError(
            "direct fractional chi z_values must be a finite nonempty vector")
    surface = mp1_negative_derivative(
        e, float(occupation_state.mu_ry),
        float(occupation_state.smearing_width_ry))
    if wfns.layout == "legacy":
        if int(wfns.psi_xn.shape[-1]) < nb:
            raise ValueError(
                "centroid wavefunctions do not cover the direct bands")
        psi_x = wfns.psi_xn[..., :nb]
        psi_y = wfns.psi_yn[..., :nb]
        kernel = _get_chi_fractional_q_kernel(
            mesh_xy, nb_logical=nb_log,
            pair_tile=_STATIC_FRACTIONAL_PAIR_TILE, n_z=z.size)
        rows = [
            kernel(psi_x, psi_y, jnp.asarray(row), e, f, surface, jnp.asarray(z))
            for row in kmq
        ]
        values = jnp.stack(rows, axis=1)
        return values[0] if z.size == 1 else values
    # face: wfns.enk is already (nk, nb_full) -- e/f/surface above are
    # ALREADY at the full loaded extent for this call site (they are
    # wfns.enk/occupation_state.f_kn/its own derivative, not a caller-
    # narrowed sub-window), but pad defensively for the general case
    # rather than assume it (mirrors the Gamma wrapper's own guard).
    nb_full = int(wfns.slices.nb_full)
    if nb_full < nb:
        raise ValueError(
            "compute_chi0_direct_fractional(layout='face'): the loaded "
            f"face carrier covers only {nb_full} bands, fewer than the "
            f"{nb} the caller's energies/occupations table provides")
    bpad = nb_full - nb
    pad2 = ((0, 0), (0, bpad))
    e_full = jnp.pad(e, pad2)
    f_full = jnp.pad(f, pad2)
    surface_full = jnp.pad(surface, pad2)
    kernel = _get_chi_fractional_q_kernel_face(
        mesh_xy, nb_full=nb_full, nb_logical=nb_log,
        pair_tile=_STATIC_FRACTIONAL_PAIR_TILE, n_z=z.size)
    rows = [
        kernel(wfns.psi_mun, wfns.psi_nmu, jnp.asarray(row), e_full, f_full,
               surface_full, jnp.asarray(z))
        for row in kmq
    ]
    values = jnp.stack(rows, axis=1)
    return values[0] if z.size == 1 else values


def precompile_chi0_contour_fractional(
    wfns,
    time_nodes,
    weight_rows,
    z_values,
    meta,
    mesh_xy,
    *,
    occupations=None,
    energy_reference=0.0,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """AOT sibling of compute_chi0_contour_fractional."""
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_fractional_contour_args(
        wfns,
        time_nodes,
        weight_rows,
        z_values,
        occupations,
        energy_reference,
        occupation_window_threshold,
    )
    _get_chi_fractional_contour_kernel(
        mesh_xy, kgrid, n_out, **_chi_face_kwargs(wfns)).lower(*args).compile()


def precompile_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=None):
    """AOT lower+compile of the χ₀ minimax kernel at the real input
    shapes/shardings — warms the JAX in-process cache so the first
    ``compute_chi0`` call is execution-only.  Call inside a dedicated
    ``timing.section('chi0_W.chi.compile')`` block to separate compile
    from exec in the end-of-run timing report.
    """
    ensure_jax_compile_cache()
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
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid, **_chi_face_kwargs(wfns))
    kernel.lower(
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    ).compile()


def precompile_solve_w(V_q, chi0_q, meta, mesh_xy, *, dyson_solver=None,
                       distrib_la_batched_route: str = "auto"):
    """AOT lower+compile of the W-solve jit.  See ``precompile_chi0``.

    Goes through the same ``_resolve_w_solve_fn`` dispatch as
    :func:`solve_w` so both paths agree on which jit to compile.
    """
    ensure_jax_compile_cache()
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy, n_rmu=chi0_q.shape[1], dyson_solver=dyson_solver,
        distrib_la_batched_route=distrib_la_batched_route)
    # The DISTRIBUTED plan is a plain function around chunked jits + one
    # FFI call, not a single jit, so there is nothing to lower here —
    # the first real call builds the BLACS descriptor and compiles its
    # own modules (scorecard L §5, amortised from call 2; the ζ tier
    # behaves the same way).
    if not hasattr(solve_fn, "lower"):
        return
    solve_fn.lower(V_q, chi0_q, pref).compile()
