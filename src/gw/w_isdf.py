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
                            layout: str = "legacy", face_shape=None):
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
    ``face_shape=(nk, nb_full, n_rmu, nspinor)``.  Every OTHER stage
    (FFT to R, the χ contraction, FFT back to q) is byte-identical either
    way — G's output shape/sharding does not depend on layout — so only
    the G-build half of this factory forks; see
    :func:`_get_chi_minimax_kernel_face`.
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
    cache_key = (id(mesh_xy), kgrid, ffi_dial_key(), n_out,
                 complex_contour, layout, face_shape)
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
            mesh_xy, kgrid, nk, n_out, complex_contour, face_shape)
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
                                 face_shape):
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

    nk_shape, nb_full, n_rmu, ns = (int(v) for v in face_shape)
    if nk_shape != nk:
        raise ValueError(
            f"_get_chi_minimax_kernel_face: face_shape nk={nk_shape} does "
            f"not match kgrid's prod={nk}")

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
    g_plan = gemm_plan(mesh_xy, m=n_rmu * ns, k=nb_full, n=n_rmu * ns,
                       nq=nk, dtype=jnp.complex128)

    @partial(jax.jit,
             in_shardings=(_psi_mun_shard, _psi_nmu_shard,
                            NamedSharding(mesh_xy, _rep2),   # mask_v
                            NamedSharding(mesh_xy, _rep2),   # mask_c
                            NamedSharding(mesh_xy, _rep2),   # enk_full
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0)),
             out_shardings=(_G_k_shard, _G_k_shard))
    def _build_Gv_Gc(psi_mun, psi_nmu, mask_v, mask_c, enk_full,
                    tau_scalar, vmax, cmin):
        t_c = jnp.conj(tau_scalar) if complex_contour else tau_scalar
        Gv_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_mun, psi_nmu, enk_full, -tau_scalar, e_ref=vmax,
                       mask=mask_v, layout="face", gemm=g_plan),
            _G_k_shard)
        Gc_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_mun, psi_nmu, enk_full, t_c, e_ref=cmin,
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
                jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128),
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

    @partial(jax.jit,
             in_shardings=(_nodes_shard, _psi_mun_shard, _psi_nmu_shard,
                            NamedSharding(mesh_xy, _rep2),
                            NamedSharding(mesh_xy, _rep2),
                            NamedSharding(mesh_xy, _rep2),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0)),
             out_shardings=_chi_R_shard,
             static_argnums=())
    def minimax_tau_integrate_chi(
        nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
    ):
        chi_R_zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128), _chi_R_shard)

        def _body(chi_R_acc, xs):
            t_scalar, alpha_scalar = xs
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
            return chi_R_acc + alpha_scalar * chi_tau, None

        final_R, _ = jax.lax.scan(
            _body, chi_R_zero, (nodes.t, nodes.alpha), unroll=1)
        return _chi_fftn_local(final_R)

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
    the logical columns).  W's pad rows/cols are masked to exact zeros
    after the solve (same contract as ``solve_at_logical``'s
    zero-refill on the local plan).  Unlike the local plan the LOGICAL
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

    cache_key = ("distributed", id(mesh_xy), nq, n_ext, n_log,
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

    if n_log < n_ext:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=P(None, 'x', 'y'), out_specs=P(None, 'x', 'y'),
                 check_vma=False)
        def _mask_pads_local(W_loc):
            # W pad rows/cols → exact zeros (they are already exact by
            # the block-diagonal argument above; the mask makes the
            # contract structural, mirroring the ζ tier's pad-row mask).
            i0 = jax.lax.axis_index('x') * (n_ext // px)
            j0 = jax.lax.axis_index('y') * (n_ext // py)
            ri = (i0 + jnp.arange(n_ext // px)) < n_log
            cj = (j0 + jnp.arange(n_ext // py)) < n_log
            return jnp.where(ri[None, :, None] & cj[None, None, :], W_loc, 0)
        _mask_pads = jax.jit(_mask_pads_local, donate_argnums=(0,))
    else:
        _mask_pads = None

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
        if _mask_pads is not None:
            W = _mask_pads(W)
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
    ``face_shape`` kwargs :func:`_get_chi_minimax_kernel` needs otherwise
    — read off ``psi_mun``'s own shape rather than threaded in by every
    caller, since the bundle already carries it (mirrors
    ``cohsex_sigma._face_kwargs``, the same pattern for the COHSEX
    factory)."""
    if wfns.layout != "face":
        return {}
    nk, ns, mu, nb_full = wfns.psi_mun.shape
    return {"layout": "face", "face_shape": (nk, wfns.slices.nb_full, mu, ns)}


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
