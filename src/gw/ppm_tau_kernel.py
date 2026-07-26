"""Device τ-kernel unit for the Σ_c(ω) GN-PPM integration.

The single-tau integrand kernel plus its cache/AOT machinery:

    σ^τ_nmk(τ) = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)       = diag[ e^{-i(E_A - E_ref_A)·τ} ] · mask_A           (A = val or cond)
    W(τ)       = Σ_μν  B_q · e^{-i(Ω_q - E_ref_B)·τ}  · mask_B      (PPM pole sum)

This is the only Σ_PPM file where SPMD / sharding / HLO expertise is required —
the reduce-scatter layout doc and the deferred scan / collective-flush notes all
live here.

The module-level kernel caches (`_sigma_tau_kernel_cache`,
`_sigma_kij_kernel_cache`) are co-located with the factories that read them.
This is load-bearing: ``precompile_sigma`` (the AOT prewarm called from
``ppm_pipeline``) must hit the *same* cache dicts as the runtime path, or the
first per-τ dispatch pays a full compile inside execution.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common.jax_compile_cache import ensure_jax_compile_cache


_sigma_tau_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
_sigma_kij_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}


def _make_project_ri_reduce_scatter(mesh_xy: Mesh) -> Callable[..., jax.Array]:
    """Build a shard_map'd ψ* σ ψ that reduce-scatters the output.

    Drop-in replacement for ``wavefunction_bundle.project_ri`` at the tail of
    ``_sigma_kij_kernel``.  Preserves the math exactly:

        Σ_mn(k) = Σ_{s, μ} Σ_{s', μ'}  ψ*_m(k, s, μ) · σ(k, s, μ, s', μ')
                                        · ψ_n(k, s', μ')

    Input sharding (global → per-rank):
        ψ_xr  P(None, None, None, 'x')       (nk, m, s, μ_X)
        σ     P(None, None, 'x', None, 'y')  (nk, s, μ_X, s', μ_Y)
        ψ_yn  P(None, None, 'y', None)       (nk, s', μ_Y, n)

    Output sharding:
        (sigma_re, sigma_im) each at  P(None, 'x', 'y')   (nk, m_X, n_Y)

    Returns the re/im parts as a tuple rather than a single (2, nk, m, n)
    stack — avoids the tuple-unpack at the caller (which would trigger a
    gather+broadcast pjit pair for a sharded array and blocks on
    is_fully_addressable in multi-process mode).

    Comms inside:  the two implicit psums of the original einsum
        psum(x)       over the μ_X contraction axis
        psum(y)       over the μ_Y contraction axis
    become:
        psum_scatter(x, scatter_dim=m)   — reduces μ_X AND scatters m on x
        psum_scatter(y, scatter_dim=n)   — reduces μ_Y AND scatters n on y

    Same NCCL byte volume as the original pair of psums (on-ring LL128), but
    the output is sharded (m_X, n_Y) so every downstream coeff·σ multiply
    stays local — which is the whole point.  A downstream Σ_c(ω, k, m, n)
    accumulator that keeps this layout end-to-end holds a per-rank buffer of
    (n_b/p_x)·(n_b/p_y)·n_ω·n_k — ~100× smaller than a replicated Σ_μν, which
    is the scaling argument for shipping this layout end-to-end.

    Deferred follow-up if that on-GPU sharded accumulator is ever wired:
        (a) m-chunking at add-τ so σ^τ arrives one m-strip at a time rather
            than a full (m_full, n_Y/p) shard (default chunk = 1 tile = m/p);
            needed when (m, n, k, ω) per-rank stops fitting.
        (b) τ batching via lax.scan over a stacked τ axis — previously tried
            and reverted (regressed sigma_ppm ~80% at MoS2 3×3: multiple n_τ
            compiles, no amortization, lost async-dispatch overlap).  Re-add
            only when per-τ Python dispatch cost exceeds those costs.
        (c) collective-flush SlabIO variant (stage many τ on GPU, one
            parallel-HDF5 write at window close).

    Requires m % p_x == 0 and n % p_y == 0.  Padding at the caller is the
    cleanest place to handle non-divisibility (TODO when we hit that).
    """
    from jax.experimental.shard_map import shard_map

    in_specs = (
        P(None, None, None, 'x'),          # psi_xr  : (nk, m, s, μ_X)
        P(None, None, 'x', None, 'y'),     # sigma_k : (nk, s, μ_X, s', μ_Y)
        P(None, None, 'y', None),          # psi_yn  : (nk, s', μ_Y, n)
    )
    # 2-tuple output: re part, im part.  Each (nk, m_X, n_Y) sharded.
    out_specs = (P(None, 'x', 'y'), P(None, 'x', 'y'))

    def _local(psi_xr_local, sigma_k_local, psi_yn_local):
        # Per-channel reduce-scatter: do re and im independently so the
        # output is a tuple of two (nk, m/p_x, n/p_y) arrays rather than a
        # stacked (2, nk, m, n) that callers would have to slice.
        def _one_channel(sigma_real_or_imag):
            # 'kmsx' × 'ksxty' -> 'kmty'  (contracts s, local μ_X)
            left_partial = jnp.einsum(
                'kmsx,ksxty->kmty',
                jnp.conj(psi_xr_local), sigma_real_or_imag, optimize=True)
            # psum_scatter(x, scatter_dim=m=1) → (nk, m/p_x, s', μ/p_y)
            left_rs = jax.lax.psum_scatter(
                left_partial, 'x', scatter_dimension=1, tiled=True)
            # 'kmty' × 'ktyn' -> 'kmn'  (contracts s', local μ_Y)
            result_partial = jnp.einsum(
                'kmty,ktyn->kmn', left_rs, psi_yn_local, optimize=True)
            # psum_scatter(y, scatter_dim=n=2) → (nk, m/p_x, n/p_y)
            return jax.lax.psum_scatter(
                result_partial, 'y', scatter_dimension=2, tiled=True,
            ).astype(jnp.complex128)

        return (_one_channel(jnp.real(sigma_k_local)),
                _one_channel(jnp.imag(sigma_k_local)))

    _sm = shard_map(_local, mesh=mesh_xy,
                    in_specs=in_specs, out_specs=out_specs,
                    check_rep=False)

    # Guard the divisibility this kernel requires (see docstring): the two
    # psum_scatters split m over p_x and n over p_y, so an indivisible sigma
    # band window would otherwise crash cryptically deep inside psum_scatter
    # (or, with a future non-tiled variant, misalign silently).  Convert that
    # into a clear, actionable failure that names the fix.  No behaviour change
    # for valid (divisible) inputs — identity passthrough.  meta.py rounds
    # b_id_4 to world_size but NOT the sigma band window (b3-b0), so this is a
    # real, reachable precondition, not a tautology.
    p_x, p_y = mesh_xy.shape['x'], mesh_xy.shape['y']

    def _project_ri_reduce_scatter(psi_xr, sigma_k, psi_yn):
        m, n = psi_xr.shape[1], psi_yn.shape[3]
        assert m % p_x == 0 and n % p_y == 0, (
            f"sigma reduce-scatter needs the band window divisible by the "
            f"mesh: m={m} must be a multiple of p_x={p_x} and n={n} of "
            f"p_y={p_y}. Pad the sigma band window (b3-b0) up to a multiple "
            f"of p_x*p_y at the caller (meta.py rounds b_id_4 but NOT the "
            f"sigma window).")
        return _sm(psi_xr, sigma_k, psi_yn)

    return _project_ri_reduce_scatter


def _get_sigma_kij_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
) -> Callable[..., jax.Array]:
    """Return a jit-compatible sigma-kij kernel with device-local FFTs.

    The tail project (ψ* σ ψ → Σ_mn) uses the reduce-scatter variant
    (_make_project_ri_reduce_scatter) so the emitted σ^τ is sharded
    (m_X, n_Y) without any downstream reshuffle.
    """

    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    pipeline_key = (id(mesh_xy), kgrid)
    if pipeline_key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[pipeline_key]

    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
    from .wavefunction_bundle import G_FFT7D_SPEC as _G_spec, V_FFT5D_SPEC as _V_spec

    ensure_jax_compile_cache()
    _G_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, _G_spec, norm='ortho')
    _G_fftn  = make_flat_k_fftn( mesh_xy, kgrid, _G_spec, norm='ortho')
    _V_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, _V_spec, norm='ortho')
    inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))

    from .greens_function_kernel import build_G_tau

    _project_ri_rs = _make_project_ri_reduce_scatter(mesh_xy)

    @partial(jax.jit, donate_argnums=(8,))
    def _sigma_kij_kernel(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_ref_A, t_node, W_q,
    ):
        """Σ_kij = project_rs[ FFT[ G(R) · W(R) / √Nk ] ].  All flat-k.

        W_q is (nq, μ, μ) flat-q — same layout as all other flat-k arrays.
        ``W_q`` is **donated**: it's built fresh each τ by ``_build_W_t_q``
        and only consumed here, so XLA can reuse its buffer for the
        ``V_R = _V_ifftn(W_q)`` output instead of allocating a separate
        intermediate.

        G(t) = build_G_tau(psi, E_A, 1j·t_node, e_ref=E_ref_A, mask=mask_A),
        i.e. the unified ISDF-basis G builder with pure-imaginary t
        (real-time evolution).  Output (Σ_ri) emerges (m_X, n_Y)-sharded
        from the final shard_map.
        """
        G_k = build_G_tau(
            psi_coh_xn, psi_coh_yr, E_A, 1j * t_node,
            e_ref=E_ref_A, mask=mask_A,
        )
        G_R = _G_ifftn(G_k)
        V_R = _V_ifftn(W_q)[:, None, :, None, :]  # (nk,1,μ,1,μ) broadcast to G shape
        sigma_k = _G_fftn(G_R * V_R * inv_sqrt_nk)
        return _project_ri_rs(psi_proj_xr, sigma_k, psi_proj_yn)

    _sigma_kij_kernel_cache[pipeline_key] = _sigma_kij_kernel
    return _sigma_kij_kernel


def _get_sigma_tau_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
) -> Callable[..., jax.Array]:
    """Return a cached tau-node sigma builder with jittable local FFTs."""

    kgrid = tuple(int(x) for x in kgrid)
    cache_key = (id(mesh_xy), kgrid)
    if cache_key in _sigma_tau_kernel_cache:
        return _sigma_tau_kernel_cache[cache_key]

    ensure_jax_compile_cache()
    q_mu_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    sigma_kij_kernel = _get_sigma_kij_kernel(mesh_xy=mesh_xy, kgrid=kgrid)

    @jax.jit
    def _build_W_t_q(B_q, Omega_q, mask_B, E_ref_B, t_node):
        """W(τ) = Σ_q B_q · exp(-i·(Ω_q - E_ref_B)·τ) · mask_B.

        (A-side G now built inside sigma_kij_kernel via build_G_tau, so
        the tau-operand helper only shapes the PPM-pole-sum B-side.)
        """
        phase_B = jnp.exp(-1j * (Omega_q - E_ref_B) * t_node)
        W_t_q = jnp.where(mask_B, B_q * phase_B,
                          jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        return jax.lax.with_sharding_constraint(W_t_q, q_mu_shard)

    @jax.jit
    def _tau_kernel(
        psi_coh_xn, psi_coh_yr,
        psi_proj_xr, psi_proj_yn,
        E_A, mask_A, B_q, Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node,
    ):
        W_t_q = _build_W_t_q(B_q, Omega_q, mask_B, E_ref_B, t_node)
        return sigma_kij_kernel(
            psi_coh_xn, psi_coh_yr,
            psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_t_q,
        )

    _sigma_tau_kernel_cache[cache_key] = _tau_kernel
    return _tau_kernel


def precompile_sigma(wfns, ppm, meta, mesh_xy: Mesh) -> None:
    """AOT lower + compile the per-τ sigma kernel.

    Parallel to :func:`w_isdf.precompile_chi0` / ``precompile_solve_w``:
    lower the cached ``_tau_kernel`` at the real input shapes/shardings
    and eagerly ``.compile()`` it so the first per-τ dispatch inside
    ``compute_sigma_c_ppm_omega_grid`` is execution-only.  Call inside
    a dedicated ``timing.section('sigma.compile')`` block to split
    compile from exec in the end-of-run timing report.

    The kernel is shape-invariant across the four ω-sign × cond/val
    branches (ψ / E_A / mask_A / B_q / Ω_q / mask_B / scalars all have
    fixed shape+dtype+sharding; only values change per window) — so
    one AOT compile covers every branch.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    tau_kernel = _get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid)

    s = wfns.slices
    psi_coh_xn  = wfns.xn(s.full)
    psi_coh_yr  = wfns.yr(s.full)
    psi_proj_xr = wfns.xr(s.sigma)
    psi_proj_yn = wfns.yn(s.sigma)
    # Mesh-pad the QP band window EXACTLY as ``ppm_sigma._run_sigma_branch``
    # does at runtime.  This is load-bearing twice over: the reduce-scatter
    # projector asserts m % p_x == 0 / n % p_y == 0 (so an unpadded AOT
    # lowering fires the guard here, which is where 7874338 died), and the AOT
    # signature must match the runtime one shape-for-shape or pjit silently
    # re-traces and the precompile buys nothing.
    from .ppm_sigma import pad_sigma_window
    psi_proj_xr, psi_proj_yn, _nb_real = pad_sigma_window(
        psi_proj_xr, psi_proj_yn, mesh_xy)

    # Representative non-ψ inputs — values don't matter for AOT, only
    # the full `(shape, dtype, sharding, committed-ness)` tuple must
    # match the runtime signature or pjit re-traces.  Specifically:
    #   * E_A at runtime comes from ``_prepare_sigma_state`` (jit output)
    #     — committed to the mesh as ``NamedSharding(P(None, None))``.
    #     Must device_put the dummy to match, otherwise pjit sees
    #     ``UnspecifiedValue`` vs ``P(None, None)`` and re-compiles.
    #   * mask_A, scalars: at runtime go through ``jnp.asarray(numpy_val)``
    #     which stays uncommitted — leave as plain jnp to match.
    #   * mask_B inherits Ω_q's sharding, same as ``_materialize_window_mask_B``.
    nb_full = int(psi_coh_xn.shape[-1])
    rep_2d  = NamedSharding(mesh_xy, P(None, None))
    E_A     = jax.device_put(
        jnp.zeros((int(meta.nk_tot), nb_full), dtype=jnp.float64), rep_2d)
    mask_A  = jnp.ones((int(meta.nk_tot), nb_full), dtype=bool)
    mask_B  = jnp.ones_like(ppm.Omega_q, dtype=bool)
    E_ref_A = jnp.asarray(0.0, dtype=jnp.float64)
    E_ref_B = jnp.asarray(0.0, dtype=jnp.float64)
    t_node  = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    tau_kernel.lower(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, ppm.B_q, ppm.Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node,
    ).compile()
