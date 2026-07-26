"""GN-PPM construction from W(0), W(iω_p) and Σ_c(ω) frequency integration.

What this module computes
-------------------------

    Σ^c_nm(k, ω) = Σ_{branches} Σ_{windows} Σ_τ  α(τ) · e^{i·ω_sign·ω·τ}
                                                 · project[ σ^τ_nmk(τ) ]
                                                 · pref

where ``ω_sign`` and ``pref`` are the per-window signs the branch's physics
fixes: +1/−1 in the ω-kernel for the (ω̃ − S)/(ω̃ + S) denominator, and a
prefactor that already carries both the Laplace-vs-crossing sign and the −1
that the −ω half contributes (folded in at window-build time — there is no
separate ``scale`` factor).

Per branch the τ nodes are placed by a minimax quadrature chosen from the
range of E_A = E_c − E_F (cond) or E_F − E_v (val) and the PPM pole
frequencies Ω_q.  Each τ node fires one sharded GPU kernel (σ^τ) that
evaluates the single-tau integrand:

    σ^τ_nmk(τ) = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)       = diag[ e^{-i(E_A - E_ref_A)·τ} ] · mask_A           (A = val or cond)
    W(τ)       = Σ_μν  B_q · e^{-i(Ω_q - E_ref_B)·τ}  · mask_B      (PPM pole sum)

The ω-dependence is *linear* in τ (only the exp(iω·τ) kernel involves ω),
so every τ contribution contributes to all ω in one shot.

Module family (post-WS3 split)
------------------------------

This file is the driver; the three single-concern units it orchestrates live
alongside it (acyclic: driver → stages → engine):

    ppm_windows.py       host-side branch + window construction (leaf; the
                         _SigmaWindow / _SigmaBranch vocabulary, the four-branch
                         Σc(−ω) decomposition, the minimax window builders).
    ppm_tau_kernel.py    the device τ-kernel unit + AOT precompile + caches.
    ppm_accumulators.py  the single numpy ω-projector + one async-D2H
                         accumulator with a memory-tile / streamed-h5 sink.

This driver retains the physics prologue (PPM fit + physics-state prep) plus the
τ-loop orchestration that binds window × kernel × accumulator, and reads as the
8-stage teleology verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import h5py

from common import jax_profile
from common.units import RYD_TO_EV
from .gw_config import PPMConfig
from .minimax_config import MinimaxConfig
from .minimax_screening import (
    MinimaxNodes,
    fit_gn_ppm_from_wc_pair,
)
from .ppm_windows import (
    _SigmaWindow,
    _iter_branches,
    _build_windows_for_branch,
    _materialize_window_mask_B,
    _to_host_np,
    _CROSSING_A_MAX,
    crossing_regularization_floor,
)
from .ppm_tau_kernel import _get_sigma_tau_kernel
from .ppm_accumulators import (
    _AccumMode,
    _select_accum_mode,
    _SigmaAccumulator,
    _TauAccumulator,
    _MemoryTileSink,
    _H5Sink,
)


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    Wc0_q: jax.Array          # (nq, μ, μ) static W^c(0) = W(0) − V; the data
                              # seam for the invalid-pole static-COHSEX term
                              # (ppm_invalid_mode="static_limit", BGW mode 3).
                              # Identity: Wc0 = −2·B_q/Ω_q elementwise.
    B_q: jax.Array            # (nq, μ, μ) PPM amplitude
    Omega_q: jax.Array        # (nq, μ, μ) PPM pole frequency
    valid_mask_q: jax.Array   # (nq, μ, μ)
    unfulfilled_fraction: float
    n_nodes_static: int


@dataclass(frozen=True)
class SigmaOmegaResult:
    omega_ry: np.ndarray
    omega_ev: np.ndarray
    sigma_c_kij: jax.Array | None      # (n_omega, nk, nb, nb) or None if streamed
    sigma_kij_h5_path: str | None = None


# ---------------------------------------------------------------------------
#  Physics-state prep — single jit that collapses the scattered trace-time
#  jnp operations the driver used to emit (Fermi level, band masks, PPM
#  pole masks, invalid-count tallies).
# ---------------------------------------------------------------------------

class _SigmaPhysicsState(NamedTuple):
    efermi: jax.Array          # scalar
    E_cond: jax.Array          # (nk, nb_full)  max(enk - efermi, 0)
    H_val: jax.Array           # (nk, nb_full)  max(efermi - enk, 0)
    cond_mask: jax.Array       # (nk, nb_full)  bool
    val_mask: jax.Array        # (nk, nb_full)  bool
    B_corr: jax.Array          # (nq, μ, μ)     c128, ready-to-contract B_q
    Omega_abs: jax.Array       # (nq, μ, μ)     f64,  max(Re Ω_q, 0)
    B_mask: jax.Array          # (nq, μ, μ)     bool, B_mask_raw & valid
    invalid_mask: jax.Array    # (nq, μ, μ)     bool, logical modes with Ω²<0
    n_total_modes: jax.Array   # scalar int64
    n_invalid: jax.Array       # scalar int64


@jax.jit
def _prepare_sigma_state(
    enk_full: jax.Array,
    occ_full: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    valid_mask_q: jax.Array,
    use_midgap: jax.Array,
    keep_invalid: jax.Array,
) -> _SigmaPhysicsState:
    """Derive Fermi level + derived energy/PPM arrays in one fused trace.

    Replaces ~9 eager jnp ops previously emitted at trace time by the sigma
    driver.  ``use_midgap`` is a traced bool scalar; the caller passes
    ``jnp.asarray(fermi_reference == 'midgap')``.  ``valid_mask_q`` is always
    a real bool array (the caller substitutes ``jnp.ones_like(...)`` when
    no mask is available), so the helper doesn't branch on None.

    ``keep_invalid`` is a traced bool implementing ``ppm_invalid_mode`` (BGW
    ``invalid_gpp_mode``) for poles with fitted ``Omega^2 < 0``: False = drop
    them from the τ-pole sum (``B_mask &= valid``; BGW mode 0 / "zero", and
    also the pole-sum half of "static_limit" / BGW mode 3 — the caller adds
    the analytic static-COHSEX term for the modes flagged by
    ``invalid_mask``); True = keep the fit's fallback pole at
    ``fallback_omega`` (default 2 Ry; BGW mode 2 / "2ry").

    μ-pad safety is structural, not per-consumer: pad modes are born DEAD
    at the fit (``fit_gn_ppm_from_wc_pair(n_mu_logical=...)`` zeroes their
    Ω, hence B = 0 and valid = False), so ``B_mask_raw = Ω > 1e-14``
    excludes them here — and in every other Ω/B consumer — with no mask
    argument (ROOT_CAUSE.md 2026-07-08; PADDING_AUDIT item 3).
    """
    occ_mask = occ_full > 0.5
    unocc_mask = ~occ_mask

    vbm = jnp.max(jnp.where(occ_mask, enk_full, -1.0e30))
    cbm = jnp.min(jnp.where(unocc_mask, enk_full, 1.0e30))
    has_unocc = jnp.any(unocc_mask)
    midgap_candidate = jnp.where(has_unocc, 0.5 * (vbm + cbm), vbm)
    efermi = jnp.where(use_midgap, midgap_candidate, vbm)

    E_cond = jnp.maximum(enk_full - efermi, 0.0)
    H_val = jnp.maximum(efermi - enk_full, 0.0)

    Omega_abs = jnp.maximum(jnp.real(Omega_q), 0.0).astype(jnp.float64)
    B_corr = jnp.asarray(B_q, dtype=jnp.complex128)
    B_mask_raw = Omega_abs > 1.0e-14
    valid = jnp.asarray(valid_mask_q, dtype=bool)
    # ppm_invalid_mode: keep_invalid=False drops Omega^2<0 poles (BGW mode 0);
    # keep_invalid=True keeps the fit's fallback pole (BGW mode 2).
    B_mask = B_mask_raw & (valid | keep_invalid)
    invalid_mask = B_mask_raw & (~valid)

    return _SigmaPhysicsState(
        efermi=efermi,
        E_cond=E_cond, H_val=H_val,
        cond_mask=unocc_mask, val_mask=occ_mask,
        B_corr=B_corr, Omega_abs=Omega_abs, B_mask=B_mask,
        invalid_mask=invalid_mask,
        n_total_modes=jnp.sum(B_mask_raw, dtype=jnp.int64),
        n_invalid=jnp.sum(invalid_mask, dtype=jnp.int64),
    )


# ---------------------------------------------------------------------------
#  PPM construction
# ---------------------------------------------------------------------------

def fit_ppm(
    W0_q: jax.Array,
    Wprobe_q: jax.Array,
    V_q: jax.Array,
    probe_omega: complex,
    mesh_xy: Mesh,
    *,
    fallback_omega: float = 2.0,
    n_nodes_static: int = 0,
    print_fn=None,
    model_label: str = "PPM",
    n_mu_logical: int,
) -> PPMBuildResult:
    """Fit two-point PPM pole parameters from precomputed W(0) and W(probe).

    Model-agnostic over the pole-fit ansatz: the same algebra serves
    both Godby-Needs (purely imaginary ``probe_omega = i·ωp``) and
    Hybertsen-Louie (real ``probe_omega = Ω`` above all transitions).

    All input arrays are flat-q (nq, μ, μ).  Returns PPMBuildResult with
    B_q, Omega_q, valid_mask_q sharded as P(None, 'x', 'y').

    ``n_mu_logical`` (REQUIRED, = ``meta.n_rmu``): logical centroid
    count.  The fitted tensors keep the padded extent, but pad modes are
    born DEAD (Ω = B = 0, valid = False) and the ``unfulfilled``
    fraction counts logical modes only — see ``fit_gn_ppm_from_wc_pair``.
    """
    import time as _t
    z = complex(probe_omega)
    t0 = _t.perf_counter()

    Wc0_q = W0_q - V_q
    Wci_q = Wprobe_q - V_q
    omega_qmunu, b_qmunu, valid_qmunu, unfulfilled = fit_gn_ppm_from_wc_pair(
        Wc0_q, Wci_q, z, fallback_omega=float(fallback_omega),
        n_mu_logical=int(n_mu_logical))

    q_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    Omega = jax.lax.with_sharding_constraint(jnp.asarray(omega_qmunu), q_shard)
    B = jax.lax.with_sharding_constraint(jnp.asarray(b_qmunu), q_shard)
    valid_mask = jax.lax.with_sharding_constraint(jnp.asarray(valid_qmunu), q_shard)
    Wc0_q = jax.lax.with_sharding_constraint(Wc0_q, q_shard)
    t1 = _t.perf_counter()

    # ω_p in PPMBuildResult historically meant the imaginary-axis magnitude;
    # carry the probe magnitude there for diagnostics.  Downstream Σ kernels
    # consume only B_q, Omega_q (the *fitted* pole frequency), so the probe
    # magnitude is for logging / restart provenance only.
    probe_mag = float(abs(z))

    if print_fn is not None:
        kind = "iωp" if abs(z.real) < 1.0e-12 else "Ω"
        print_fn(
            f"  {model_label} fit: {t1-t0:.2f}s, {kind}={probe_mag:.4f} Ry, "
            f"unfulfilled={100.0 * unfulfilled:.2f}%")

    return PPMBuildResult(
        omega_p=probe_mag,
        Wc0_q=Wc0_q,
        B_q=B,
        Omega_q=Omega,
        valid_mask_q=valid_mask,
        unfulfilled_fraction=unfulfilled,
        n_nodes_static=n_nodes_static,
    )


# ---------------------------------------------------------------------------
#  Sigma convolution — the device-side τ loop.  Its host-side counterpart
#  (window construction) lives in ppm_windows; the two halves share no state
#  beyond the window list itself.
# ---------------------------------------------------------------------------

def minimax_tau_integrate_sigma(
    nodes: MinimaxNodes,
    *,
    build_sigma_tau: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    add_tau: Callable[..., None],
    E_ref_sum: float,
    progress=None,
) -> None:
    """One window's τ integration for Σ^c(ω).

    Sibling of ``w_isdf.minimax_tau_integrate_chi`` — both take a
    ``MinimaxNodes`` pytree in the same slot.  chi0 can run its τ sweep
    inside one ``lax.scan`` because its body emits no collective; sigma
    stays a Python τ loop because its per-τ body emits NCCL and a
    monolithic scan regressed MoS2 3×3 by ~80%.

    Parameters
    ----------
    nodes
        Window-local τ nodes (complex128 ``t`` and ``alpha``).  For
        Laplace windows ``t = -1j·τ_real``; for crossing windows
        ``t = τ_real / ξ``.
    build_sigma_tau
        Callable ``t_j -> (σ_re, σ_im)`` that bundles G(τ)·W(τ), the
        FFT round-trip and ψ-projection for one τ scalar.  Closes over
        the window-pinned args (psi, masks, E_ref_A/B, B_q, Ω_q) so
        the signature here reads parallel to chi0's builders.
    add_tau
        Callable invoked per τ with ``(σ_re, σ_im, t_c, α_eff_c)``.
        ``t_c`` and ``α_eff_c`` are Python complex scalars (already on
        host — they were the numpy values we used to build ``t_j``).
        Host-side accumulators can use them directly; GPU-side
        accumulators wrap them as jax scalars themselves.
    E_ref_sum
        ``E_ref_A + E_ref_B`` for this window — absorbed into α per τ as
        ``α_eff = α · exp(-i · E_ref_sum · t)`` so the Laplace kernel
        sees non-negative (E_A, Ω_q) arguments.
    progress
        Optional ``LoopProgress``-like object whose ``.step()`` is called
        after each τ dispatch.
    """
    t_host = np.asarray(jax.device_get(nodes.t), dtype=np.complex128)
    alpha_host = np.asarray(jax.device_get(nodes.alpha), dtype=np.complex128)
    alpha_eff_host = alpha_host * np.exp(-1j * float(E_ref_sum) * t_host)

    for i in range(int(nodes.t.shape[0])):
        t_c = complex(t_host[i])
        alpha_eff_c = complex(alpha_eff_host[i])
        # σ^τ is returned as a (re, im) tuple so the crossing window's HGL
        # quadrature (which keeps only Im[coeff·σ]) doesn't carry a complex
        # σ^τ through the FFT stack — would double HBM + collective traffic.
        sigma_re, sigma_im = build_sigma_tau(
            jnp.asarray(t_c, dtype=jnp.complex128))
        if progress is not None:
            progress.step()
        add_tau(sigma_re, sigma_im, t_c, alpha_eff_c)


def _integrate_tau_windows_for_branch(
    *,
    windows: list[_SigmaWindow],
    accumulator: _SigmaAccumulator,
    E_A: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    psi_coh_xn: jax.Array,
    psi_coh_yr: jax.Array,
    psi_proj_xr: jax.Array,
    psi_proj_yn: jax.Array,
    tau_kernel: Callable[..., jax.Array],
    log_tag: str,
    print_fn,
) -> None:
    """Walk windows; for each, dispatch ``minimax_tau_integrate_sigma``
    with closures that bind this window's (psi, masks, E_ref, kernel) and
    feed the window's σ^τ into the accumulator.  The result lands either
    in per-rank host tiles or streamed to H5 — that decision is the
    accumulator's sink, not this loop's.  See
    _TauAccumulator + _MemoryTileSink / _H5Sink.
    """
    from common.progress import LoopProgress

    branch_label = log_tag if log_tag else "sigma"
    total_tau_nodes = sum(win.n_tau for win in windows)
    progress = LoopProgress(
        total_tau_nodes, print_fn, title=f"sigma[{branch_label}]",
        item_name="tau node", max_updates=10)

    with jax_profile.annotation(f"sigma_branch[{branch_label}]"):
        for win_idx, win in enumerate(windows):
            with jax_profile.step_annotation(
                "sigma_window", step_num=win_idx,
                detail=f"{branch_label}:{win.name}:n{win.n_tau}",
            ):
                mask_A_j    = jnp.asarray(win.mask_A)
                mask_B_j    = _materialize_window_mask_B(
                    win, base_mask_B=base_mask_B, Omega_q=Omega_q)
                E_ref_A_j   = jnp.asarray(win.E_ref_A, dtype=jnp.float64)
                E_ref_B_j   = jnp.asarray(win.E_ref_B, dtype=jnp.float64)

                def build_sigma_tau(t_j):
                    return tau_kernel(
                        psi_coh_xn, psi_coh_yr,
                        psi_proj_xr, psi_proj_yn,
                        E_A, mask_A_j, B_q, Omega_q, mask_B_j,
                        E_ref_A_j, E_ref_B_j, t_j,
                    )

                accumulator.begin_window(win)
                minimax_tau_integrate_sigma(
                    win.nodes,
                    build_sigma_tau=build_sigma_tau,
                    add_tau=accumulator.add_tau,
                    E_ref_sum=win.E_ref_A + win.E_ref_B,
                    progress=progress,
                )
                accumulator.end_window()

    progress.finish()


# ---------------------------------------------------------------------------
# Sigma band-window mesh padding
# ---------------------------------------------------------------------------

def pad_sigma_window(psi_proj_xr, psi_proj_yn, mesh_xy):
    """Zero-pad the sigma band window up to a multiple of ``p_x·p_y``.

    ``ppm_tau_kernel._make_project_ri_reduce_scatter`` reduce-scatters m over
    ``'x'`` and n over ``'y'``, and ``_MemoryTileSink`` holds Sigma_c(w,k,m,n)
    at ``P(None, None, 'x', 'y')`` — so BOTH need ``m % p_x == 0`` and
    ``n % p_y == 0``.  ``common/meta.py`` rounds ``b_id_4`` (the FULL window)
    to ``world_size`` but never the sigma window ``b3-b0``, so an indivisible
    QP window is reachable and fired on MoS2 12x12 (m=n=70, mesh 8x10).

    Padding is the fix the guard itself prescribes, and it is exact: every
    output element ``Sigma[k,m,n]`` is an INDEPENDENT contraction
    ``psi*_m . sigma . psi_n``, so appending bands adds output rows/columns
    without perturbing any existing one.  The pad rows are exactly zero, so
    the pad block of Sigma is exactly zero too — and it is stripped by
    :func:`strip_sigma_window` before Sigma leaves the branch, so nothing
    downstream (host buffer, eqp write) ever sees the padded extent.

    Mirrors the established zero-pad-band contract used by the wfn loader
    (``load_psi_gflat_padded``) and htransform (``band_pad_to``).

    Returns ``(xr_padded, yn_padded, nb_real)``; a no-op (identity, same
    buffers) when the window already divides.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    nb_real = int(psi_proj_xr.shape[1])
    div = p_x * p_y
    nb_pad = -(-nb_real // div) * div        # round up
    if nb_pad == nb_real:
        return psi_proj_xr, psi_proj_yn, nb_real
    extra = nb_pad - nb_real
    # psi_xr : (nk, m, s, mu_X) at P(None,None,None,'x') -> band axis 1
    # psi_yn : (nk, s, mu_Y, n) at P(None,None,'y',None) -> band axis 3
    # Neither band axis is mesh-sharded, so both pads are rank-local.
    xr_p = jnp.pad(psi_proj_xr, ((0, 0), (0, extra), (0, 0), (0, 0)))
    yn_p = jnp.pad(psi_proj_yn, ((0, 0), (0, 0), (0, 0), (0, extra)))
    return xr_p, yn_p, nb_real


def strip_sigma_window(sigma_kij, nb_real: int):
    """Drop the :func:`pad_sigma_window` pad block from a (..., m, n) Sigma.

    The pad rows/cols are exactly zero (bilinear in zero-padded psi); this is
    the single seam where the padded extent stops.  No-op when unpadded.
    """
    if sigma_kij is None or int(sigma_kij.shape[-1]) == int(nb_real):
        return sigma_kij
    return sigma_kij[..., :nb_real, :nb_real]


def _run_sigma_branch(
    *,
    omega_nonneg_ry: np.ndarray,
    omega_global_idx: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    space: str,
    neg_omega_half: bool,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    wfns,
    mesh_xy: Mesh,
    meta,
    log_tag: str = "",
    print_fn=print,
    omega_batch_size: int = 4,
    stream_writer: Callable[[np.ndarray, jax.Array], None] | None = None,
    use_shipped_minimax_tables: bool = True,
) -> tuple[jax.Array, list[_SigmaWindow]]:
    """Orchestrator for one branch (cond or val × pos or neg ω half).

    Reads as a physics outline:
        windows = _build_windows_for_branch(...)          # host
        acc     = _integrate_tau_windows_for_branch(...)  # device
    """
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])

    s = wfns.slices
    psi_coh_xn = wfns.xn(s.full)
    psi_coh_yr = wfns.yr(s.full)
    psi_proj_xr = wfns.xr(s.sigma)
    psi_proj_yn = wfns.yn(s.sigma)
    nk_proj = int(psi_proj_xr.shape[0])
    # Mesh-pad the QP band window: the reduce-scatter projector and the
    # Sigma_c tile sink both need m % p_x == 0 / n % p_y == 0 (see
    # pad_sigma_window).  ``nb_proj`` stays the REAL window everywhere the
    # caller can see; only the in-branch machinery runs at ``nb_pad``.
    psi_proj_xr, psi_proj_yn, nb_proj = pad_sigma_window(
        psi_proj_xr, psi_proj_yn, mesh_xy)
    nb_pad = int(psi_proj_xr.shape[1])

    if n_omega == 0:
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    windows = _build_windows_for_branch(
        omega_nonneg_ry=omega_nonneg_ry,
        E_A=E_A, base_mask_A=base_mask_A,
        Omega_q=Omega_q, base_mask_B=base_mask_B,
        space=space, neg_omega_half=neg_omega_half,
        regularization_width_ry=regularization_width_ry,
        edge_factor=edge_factor,
        target_error=target_error, max_nodes=max_nodes,
        crossing_eps_q=crossing_eps_q, crossing_max_nodes=crossing_max_nodes,
        use_shipped_minimax_tables=use_shipped_minimax_tables,
        log_tag=log_tag, print_fn=print_fn,
    )
    if not windows:
        return jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    omega_vec = jnp.asarray(omega_nonneg_ry, dtype=jnp.float64)
    tau_kernel = _get_sigma_tau_kernel(
        mesh_xy=mesh_xy,
        kgrid=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
    )

    # One async-D2H accumulator; the sink decides where a finished window goes.
    # Both sinks consume the SAME per-shard host tiles produced by the single
    # numpy projector (copy_to_host_async + a short deque overlap GPU-τ_{k+lag}
    # with the numpy-τ_k accumulate).
    if stream_writer is None:
        # Σ_c(ω,k,m,n) lives as per-rank numpy tiles matching σ(τ)'s
        # (m_X, n_Y) sharding — the full (n_ω,n_k,n_b,n_b) buffer never
        # exists on any GPU until the final device assembly at finalize().
        sink: _MemoryTileSink | _H5Sink = _MemoryTileSink(
            shape=(n_omega, nk_proj, nb_pad, nb_pad),
            sharding=NamedSharding(mesh_xy, P(None, None, 'x', 'y')),
        )
    else:
        # Single-process streamed: assemble each window on host and RMW it to
        # the h5 dataset ω-batched (n_windows RMW, not n_τ).
        assert nb_pad == nb_proj, (
            f"streamed sigma sink cannot carry a padded QP window "
            f"(nb_pad={nb_pad} != nb_proj={nb_proj}); KIJ_STREAM is "
            f"single-process only, where the mesh is 1x1 and no pad exists.")
        sink = _H5Sink(
            writer=stream_writer,
            omega_global_idx=omega_global_idx,
            omega_batch_size=int(max(1, omega_batch_size)),
            # KIJ_STREAM is single-process only (ppm_accumulators
            # _select_accum_mode: n_proc != 1 -> KIJ_HOST), and a 1x1 mesh
            # cannot pad -- so the real and padded extents coincide here.
            # Assert it rather than rely on the coincidence.
            full_spatial_shape=(nk_proj, nb_proj, nb_proj),
        )
    accumulator: _SigmaAccumulator = _TauAccumulator(
        omega_vec=omega_vec, sink=sink)

    _integrate_tau_windows_for_branch(
        windows=windows, accumulator=accumulator,
        E_A=E_A, B_q=B_q, Omega_q=Omega_q, base_mask_B=base_mask_B,
        psi_coh_xn=psi_coh_xn, psi_coh_yr=psi_coh_yr,
        psi_proj_xr=psi_proj_xr, psi_proj_yn=psi_proj_yn,
        tau_kernel=tau_kernel,
        log_tag=log_tag, print_fn=print_fn,
    )

    acc_total = accumulator.finalize()
    if acc_total is None:
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows
    # Strip the mesh pad block (exactly zero) — the ONE seam where the padded
    # QP window stops.  Everything above this line ran at nb_pad; everything
    # below (host Sigma buffer, eqp write) sees only the real nb_proj.
    return strip_sigma_window(acc_total, nb_proj), windows


def _compute_invalid_static_sigma(
    wfns,
    Wc0_q: jax.Array,
    invalid_mask: jax.Array,
    meta,
    mesh_xy: Mesh,
) -> np.ndarray:
    """Static-COHSEX Σ for the invalid PPM poles (BGW ``invalid_gpp_mode=3``).

    BGW's default treatment of a pole with fitted ``Ω² < 0`` sets
    ``ω̃ → 1/TOL_ZERO`` (mtxel_cor.f90:788/838), which is the Ω→∞ limit of
    the full dynamical pole: for that mode's ``W_static = W^c(0)·mask``,

        occupied   l:  ssx → −I_ε,  sch → −½·I_ε   ⇒  −W_static + ½·W_static
        unoccupied l:                sch → −½·I_ε   ⇒            + ½·W_static

    i.e. the mode is treated within static COHSEX: a screened-exchange
    term over occupied states plus the Coulomb-hole over the full RI
    window.  (Ω→∞ can NOT be pushed through the τ-integral — ``B ∝ Ω``
    makes ``B·e^{−iΩτ}`` non-integrable — hence this analytic,
    ω-independent term instead.)  Equivalently, per intermediate state:
    occ → −½·W^c(0) (= B/Ω), unocc → +½·W^c(0) — the exact Ω→∞ limit of
    the two-branch pole sum ``B/(ω−E_l∓Ω)``.

    Reuses the two static COHSEX contraction kernels verbatim
    (``cohsex_sigma._make_cohsex_kernels``) with the masked static
    ``W^c(0)`` as the screening operand:

        Σ_static = sigma_sx(G_occ, W_static) + sigma_coh(W_static − 0)
                 = −⟨G_occ·W_static⟩ + ½·⟨G_RI·W_static⟩

    matching design note GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md §8
    (Σ_occ − ½·Σ_RI in its sign convention).  μ-pad safety is inherited
    from ``invalid_mask`` (pad modes are born dead at the fit, so they
    are never flagged invalid and ``W_static`` is exactly zero there).

    Returns the replicated host tensor (nk, nb_sigma, nb_sigma) in Ry,
    to be added to Σ_c at EVERY ω (the term is ω-independent).
    """
    from .cohsex_sigma import _make_cohsex_kernels, build_Gij

    sigma_sx_k, sigma_coh_k, _ = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot))
    Gij = build_Gij(meta, mesh_xy)
    rep = NamedSharding(mesh_xy, P(None, None, None))

    with mesh_xy:
        W_static = jnp.where(
            jnp.asarray(invalid_mask, dtype=bool),
            jnp.asarray(Wc0_q, dtype=jnp.complex128),
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )
        sig_static = (
            sigma_sx_k(wfns, Gij, W_static)
            + sigma_coh_k(wfns, W_static, jnp.zeros_like(W_static))
        )
        sig_static = jax.lax.with_sharding_constraint(sig_static, rep)
        sig_static.block_until_ready()

    # Replicated (None,None,None) ⇒ every process's first addressable shard
    # IS the full tensor.  (_to_host_np's process_allgather would STACK a
    # fully-replicated array across processes into (nproc, nk, nb, nb).)
    return np.asarray(sig_static.addressable_data(0), dtype=np.complex128)


# ---------------------------------------------------------------------------
#  Top-level sigma driver
# ---------------------------------------------------------------------------

def compute_sigma_c_ppm_omega_grid(
    wfns,
    ppm,
    meta,
    mesh_xy: Mesh,
    *,
    ppm_cfg: PPMConfig,
    quad: MinimaxConfig,
    omega_grid_ry: np.ndarray,
    sigma_kij_h5_path: str | None,
    print_fn=print,
) -> SigmaOmegaResult:
    """Compute Σ^c_kij(ω) via GN-PPM windowed minimax integration.

    Config seam (WS2): scalar knobs are read by direct attribute access
    off the validated frozen ``ppm_cfg`` (no ``getattr(..., default)`` —
    a stale/typo'd name must raise, not silently default); the derived
    ω-grid and the input_dir-resolved h5 path arrive as explicit data
    arguments.  ``ppm_cfg``/``quad`` never travel below this driver.
    """

    s = wfns.slices
    psi_proj_xr = wfns.xr(s.sigma)
    enk_full = wfns.enk[:, s.full]
    occ_full = wfns.occ[:, s.full]
    B_q = ppm.B_q
    Omega_q = ppm.Omega_q
    valid_mask_q = ppm.valid_mask_q
    omega_values_ry = omega_grid_ry

    # Flat nk is used throughout this driver; (nkx, nky, nkz) only flows
    # into the kernel factory (tau_kernel) below — it's already the
    # kernel's cache key, so we don't unpack kgrid here at the driver.
    nk = int(meta.nk_tot)

    # Quadrature config (required — one merged MinimaxConfig instance).
    target_error = float(quad.target_error)
    max_nodes = int(quad.max_nodes)
    crossing_max_nodes = int(quad.crossing_max_nodes)
    crossing_eps_q = float(quad.crossing_eps_q)
    use_shipped_minimax_tables = bool(quad.use_shipped_tables)

    # Scalar knobs — direct reads off the validated frozen PPMConfig.
    regularization_width_ry = float(ppm_cfg.regularization_ev) / RYD_TO_EV
    edge_factor = float(ppm_cfg.window_edge_factor)

    # Crossing-quadrature conditioning floor: raise ξ if the Σ_c ω-grid is wide
    # enough that the HGL core window would be ill-conditioned (Σ|α| ~ 1e5,
    # amplifying the mesh-sensitive per-τ operand → device-dependent Σ_c blow-up
    # + O(1e3) eV Im).  See ppm_windows.crossing_regularization_floor.
    omega_max_ry = float(np.max(np.abs(np.asarray(omega_values_ry, dtype=np.float64))))
    xi_floor = crossing_regularization_floor(omega_max_ry, edge_factor)
    if regularization_width_ry < xi_floor:
        print_fn(
            f"  Σc crossing conditioning: ξ raised "
            f"{regularization_width_ry * RYD_TO_EV:.3f} → {xi_floor * RYD_TO_EV:.3f} eV "
            f"(A_core capped at {_CROSSING_A_MAX:.0f}; the requested ξ would make the "
            f"HGL crossing quadrature ill-conditioned)")
        regularization_width_ry = xi_floor
    omega_batch_size = int(ppm_cfg.omega_batch_size)
    omega_accumulation = ppm_cfg.omega_accumulation
    fermi_reference = ppm_cfg.fermi_reference
    invalid_mode = ppm_cfg.invalid_mode

    if nk != int(enk_full.shape[0]):
        raise ValueError(f"enk_full shape mismatch: expected first dim {nk}, got {enk_full.shape[0]}")

    omega_req = np.asarray(omega_values_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_values_ry must be a 1D non-empty array.")
    omega_batch_size = int(max(1, omega_batch_size))

    # Split omega grid into positive and negative relative to Fermi level
    idx_pos = np.where(omega_req >= 0.0)[0]
    idx_neg = np.where(omega_req < 0.0)[0]
    omega_pos = np.asarray(omega_req[idx_pos], dtype=np.float64)
    omega_neg_abs = np.asarray(-omega_req[idx_neg], dtype=np.float64)

    # fermi_reference / omega_accumulation are validated + normalized at
    # PPMConfig construction; used directly here (fermi → traced bool below).

    # ppm_invalid_mode (BGW ``invalid_gpp_mode``): how to treat poles whose
    # fitted Omega^2 came out < 0.  'zero'/'skip' drop them (BGW mode 0);
    # '2ry' keeps the fit's fallback_omega pole (default 2 Ry, BGW mode 2);
    # 'static_limit'/'infinity' (BGW mode 3 = BGW's and LORRAX's default)
    # drops them from the τ-pole sum AND adds the analytic ω-independent
    # static-COHSEX term for those modes (see _compute_invalid_static_sigma);
    # 'imaginary' (BGW mode 1) needs a complex-Omega path.
    invalid_mode = str(invalid_mode).strip().lower()
    if invalid_mode == "imaginary":
        raise NotImplementedError(
            "ppm_invalid_mode='imaginary' (BGW mode 1) needs a complex-Omega path.")
    if invalid_mode not in ("zero", "skip", "2ry", "static_limit", "infinity"):
        raise ValueError(
            f"ppm_invalid_mode must be zero/skip/2ry/static_limit/infinity; got {invalid_mode!r}")
    keep_invalid = invalid_mode == "2ry"
    invalid_static = invalid_mode in ("static_limit", "infinity")

    # Derive Fermi level, energy/band masks, and PPM pole masks in one fused trace.
    # valid_mask_q=None → all-true mask at the caller so the jit sees a real array.
    # (μ-pad modes need no mask here: they are born with Ω = 0 at the fit
    # and drop out of B_mask_raw structurally — see _prepare_sigma_state.)
    if valid_mask_q is None:
        valid_mask_q = jnp.ones(Omega_q.shape, dtype=bool)
    state = _prepare_sigma_state(
        enk_full, occ_full, B_q, Omega_q, valid_mask_q,
        jnp.asarray(fermi_reference == "midgap", dtype=bool),
        jnp.asarray(keep_invalid, dtype=bool),
    )
    efermi = state.efermi
    E_cond = state.E_cond
    H_val = state.H_val
    cond_mask = state.cond_mask
    val_mask = state.val_mask
    B_corr = state.B_corr
    Omega_abs = state.Omega_abs
    B_mask = state.B_mask
    n_total_modes = int(jax.device_get(state.n_total_modes))
    n_invalid = int(jax.device_get(state.n_invalid))

    omega_step_ev = float(omega_req[1] - omega_req[0]) * RYD_TO_EV if omega_req.size > 1 else 0.0
    print_fn(
        f"  Σc(ω) grid: "
        f"{float(np.min(omega_req)) * RYD_TO_EV:.3f}..{float(np.max(omega_req)) * RYD_TO_EV:.3f} eV, "
        f"Nω={omega_req.size}, Δω={omega_step_ev:.3f} eV, "
        f"ξ={float(regularization_width_ry) * RYD_TO_EV:.3f} eV"
    )
    if n_invalid:
        print_fn(
            f"  GN invalid modes: {n_invalid}/{n_total_modes} "
            f"({100.0 * n_invalid / max(n_total_modes, 1):.2f}%)"
        )
        # Per-q localization of the invalid poles (diagnostic; see
        # reports/bgw_invalid_mode_refs_2026-07-08 — the ISDF invalid
        # population sits on different (pair, q) structure than BGW's).
        n_invalid_q = np.asarray(jax.device_get(
            jnp.sum(state.invalid_mask, axis=(1, 2), dtype=jnp.int64)))
        print_fn(
            "  GN invalid modes per q: "
            f"min={int(n_invalid_q.min())} max={int(n_invalid_q.max())} "
            f"counts={np.array2string(n_invalid_q, max_line_width=100, threshold=64)}"
        )

    # ppm_invalid_mode='static_limit': ω-independent static-COHSEX term for
    # the invalid poles (their dynamical poles were dropped via B_mask above).
    # Computed once here, added to Σ_c at every ω on whichever accumulation
    # path is active (host tensor add / streamed h5 RMW — same values, so
    # kij↔kij_stream parity is preserved).
    sigma_static_host = None
    if invalid_static and n_invalid:
        sigma_static_host = _compute_invalid_static_sigma(
            wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy)
        print_fn(
            "  GN invalid modes → static COHSEX: max|Σ_static| = "
            f"{float(np.max(np.abs(sigma_static_host))) * RYD_TO_EV:.4f} eV "
            f"(diag max {float(np.max(np.abs(np.diagonal(sigma_static_host, axis1=1, axis2=2)))) * RYD_TO_EV:.4f} eV)"
        )

    # Decide accumulation mode + allocate any backing storage.
    nk_proj = int(psi_proj_xr.shape[0])
    nb_proj = int(psi_proj_xr.shape[1])
    n_omega = int(omega_req.size)
    kij_bytes = float(n_omega * nk_proj * nb_proj * nb_proj * 16)
    accum_mode = _select_accum_mode(
        omega_accumulation,
        sigma_kij_h5_path=sigma_kij_h5_path,
        kij_bytes=kij_bytes,
        n_proc=int(jax.process_count()),
    )
    streaming = (accum_mode == _AccumMode.KIJ_STREAM)

    sigma_kij_host = (
        None if streaming
        else np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)
    )

    # Single-process stream-mode file setup.  The accumulator pattern
    # itself is unchanged from pre-SlabIO (rank-0 h5py); the final
    # sigma_mnk.h5 copy-over now lives in
    # ``file_io.copy_sigma_kij_h5_to_omega_h5`` (called by gw_jax.main).
    kij_stream_path = None
    h5_kij = None
    dset_sigma_kij = None
    if streaming and jax.process_index() == 0:
        kij_stream_path = str(sigma_kij_h5_path)
        kij_dir = os.path.dirname(os.path.abspath(kij_stream_path))
        if kij_dir:
            os.makedirs(kij_dir, exist_ok=True)
        k_chunks = max(1, min(4, nk_proj))
        o_chunks = max(1, min(omega_batch_size, n_omega))
        h5_kij = h5py.File(kij_stream_path, "w")
        h5_kij.create_dataset("omega_ry", data=np.asarray(omega_req, dtype=np.float64))
        h5_kij.create_dataset("omega_ev", data=np.asarray(omega_req * RYD_TO_EV, dtype=np.float64))
        dset_sigma_kij = h5_kij.create_dataset(
            "sigma_c_kij_ry",
            shape=(n_omega, nk_proj, nb_proj, nb_proj),
            dtype=np.complex128,
            chunks=(o_chunks, k_chunks, nb_proj, nb_proj),
            fillvalue=0j,  # h5py>=3.13 rejects a float fill on a complex dtype
        )
        h5_kij.attrs["layout"] = "omega,k,i,j"

    try:
        def _accumulate_kij_stream(global_idx: np.ndarray, contrib_batch: np.ndarray) -> None:
            # Called by _H5Sink once per (window × ω-batch) with an already-host
            # numpy buffer (the assembled window slab), read-modify-write add.
            if dset_sigma_kij is None:
                return
            idx = np.asarray(global_idx, dtype=np.int64)
            buf = dset_sigma_kij[idx]
            buf = buf + np.asarray(contrib_batch, dtype=np.complex128)
            dset_sigma_kij[idx] = buf

        common_branch_kwargs = dict(
            B_q=B_corr,
            Omega_q=Omega_abs,
            base_mask_B=B_mask,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error,
            max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
            wfns=wfns,
            mesh_xy=mesh_xy,
            meta=meta,
            print_fn=print_fn,
            omega_batch_size=omega_batch_size,
            stream_writer=_accumulate_kij_stream if streaming else None,
            use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
        )

        # Enumerate the 4 branches (ω sign × cond/val), skipping empty ω halves.
        # See _iter_branches for how each branch's physical identity fixes its
        # denominator/prefactor signs (no ±1 sign fields are carried).
        branches = _iter_branches(
            omega_pos=omega_pos, idx_pos=idx_pos,
            omega_neg_abs=omega_neg_abs, idx_neg=idx_neg,
            E_cond=E_cond, H_val=H_val,
            cond_mask=cond_mask, val_mask=val_mask,
        )

        # Run each branch and fold its Σc directly into the host tensor at its
        # global ω indices.  cond and val of a given ω-half share those indices,
        # so the second branch's `+=` sums cond+val there — same values, same
        # traversal order (cond before val, per-branch device reduction then
        # host add) as the old per-ω-half dict, minus the tuple-of-ints key.
        # Streaming writes straight to the h5 via the branch's _H5Sink; both
        # cond and val RMW-add into the same dataset, so no host fold is needed.
        for br in branches:
            sigma_kij, _ = _run_sigma_branch(
                omega_nonneg_ry=br.omega_abs, omega_global_idx=br.omega_idx,
                E_A=br.E_A, base_mask_A=br.base_mask_A,
                space=br.space, neg_omega_half=br.neg_omega_half,
                log_tag=br.tag,
                **common_branch_kwargs,
            )
            if not streaming:
                # the reduce-scatter project (_make_project_ri_reduce_scatter)
                # returns Σ sharded (m_X, n_Y), so the host copy needs a
                # cross-process gather rather than jax.device_get; _to_host_np
                # falls back to device_get for single-process / replicated.
                idx = np.asarray(br.omega_idx, dtype=np.int64)
                sigma_kij_host[idx] = (
                    sigma_kij_host[idx]
                    + _to_host_np(sigma_kij, dtype=np.complex128, tiled=False))

        # static_limit: fold the ω-independent invalid-pole static-COHSEX
        # term into Σ_c at every ω (host add / streamed ω-batched h5 RMW —
        # identical values on both paths).
        if sigma_static_host is not None:
            if not streaming:
                sigma_kij_host += sigma_static_host[None, ...]
            else:
                for ibeg in range(0, n_omega, omega_batch_size):
                    idx = np.arange(
                        ibeg, min(ibeg + omega_batch_size, n_omega),
                        dtype=np.int64)
                    _accumulate_kij_stream(
                        idx,
                        np.broadcast_to(
                            sigma_static_host,
                            (idx.size, *sigma_static_host.shape)))
    finally:
        if h5_kij is not None:
            h5_kij.close()

    sigma_kij_req = None if sigma_kij_host is None else jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * RYD_TO_EV, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
        sigma_kij_h5_path=kij_stream_path,
    )
