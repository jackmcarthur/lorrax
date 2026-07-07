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
    ppm_accumulators.py  the ω-projection pair + the two accumulators.

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
)
from .ppm_tau_kernel import _get_sigma_tau_kernel
from .ppm_accumulators import (
    _AccumMode,
    _select_accum_mode,
    _SigmaAccumulator,
    _HostOmegaAccumulator,
    _StreamedH5Accumulator,
)


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    W0_q: jax.Array           # (nq, μ, μ) flat-q
    Wiwp_q: jax.Array         # (nq, μ, μ) flat-q
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
    them (``B_mask &= valid``; BGW mode 0 / "zero"); True = keep the fit's
    fallback pole at ``fallback_omega`` (default 2 Ry; BGW mode 2 / "2ry").
    The static-COHSEX mode (BGW 3) is handled/rejected at the caller.
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
) -> PPMBuildResult:
    """Fit two-point PPM pole parameters from precomputed W(0) and W(probe).

    Model-agnostic over the pole-fit ansatz: the same algebra serves
    both Godby-Needs (purely imaginary ``probe_omega = i·ωp``) and
    Hybertsen-Louie (real ``probe_omega = Ω`` above all transitions).

    All input arrays are flat-q (nq, μ, μ).  Returns PPMBuildResult with
    B_q, Omega_q, valid_mask_q sharded as P(None, 'x', 'y').
    """
    import time as _t
    z = complex(probe_omega)
    t0 = _t.perf_counter()

    Wc0_q = W0_q - V_q
    Wci_q = Wprobe_q - V_q
    omega_qmunu, b_qmunu, valid_qmunu, unfulfilled = fit_gn_ppm_from_wc_pair(
        Wc0_q, Wci_q, z, fallback_omega=float(fallback_omega))

    q_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    Omega = jax.lax.with_sharding_constraint(jnp.asarray(omega_qmunu), q_shard)
    B = jax.lax.with_sharding_constraint(jnp.asarray(b_qmunu), q_shard)
    valid_mask = jax.lax.with_sharding_constraint(jnp.asarray(valid_qmunu), q_shard)
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
        W0_q=W0_q,
        Wiwp_q=Wprobe_q,
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
    on-GPU (host ω accumulator) or streamed to H5 — that decision
    is the accumulator's, not this loop's.  See
    _HostOmegaAccumulator / _StreamedH5Accumulator.
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
    nb_proj = int(psi_proj_xr.shape[1])

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

    if stream_writer is None:
        # Σ_c(ω,k,m,n) lives as per-rank numpy tiles matching σ(τ)'s
        # (m_X, n_Y) sharding — the full (n_ω,n_k,n_b,n_b) buffer never
        # exists on any GPU.  copy_to_host_async + a short deque overlap
        # GPU-τ_{k+lag} with numpy-τ_k accumulate.
        accumulator: _SigmaAccumulator = _HostOmegaAccumulator(
            shape=(n_omega, nk_proj, nb_proj, nb_proj),
            gpu_mesh=mesh_xy,
            omega_vec=omega_vec,
        )
    else:
        accumulator = _StreamedH5Accumulator(
            writer=stream_writer,
            omega_vec=omega_vec,
            omega_global_idx=omega_global_idx,
            omega_batch_size=int(max(1, omega_batch_size)),
        )

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
    return acc_total, windows


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
    # '2ry' keeps the fit's fallback_omega pole (default 2 Ry, BGW mode 2).
    # 'static_limit'/'infinity' (BGW mode 3, the BGW default) needs a static
    # -1/2 Wc0 term not yet wired; 'imaginary' (BGW mode 1) needs complex Omega.
    invalid_mode = str(invalid_mode).strip().lower()
    if invalid_mode in ("static_limit", "infinity"):
        raise NotImplementedError(
            f"ppm_invalid_mode={invalid_mode!r} (BGW invalid_gpp_mode=3, static COHSEX) "
            "needs the static -1/2*Wc0 Coulomb-hole term (Wc0 = B_q/Omega_q) added to the "
            "diagonal Sigma_c; not yet wired. Use 'zero' (drop, BGW mode 0) or '2ry' "
            "(keep the fallback_omega pole, BGW mode 2).")
    if invalid_mode == "imaginary":
        raise NotImplementedError(
            "ppm_invalid_mode='imaginary' (BGW mode 1) needs a complex-Omega path.")
    if invalid_mode not in ("zero", "skip", "2ry"):
        raise ValueError(
            f"ppm_invalid_mode must be zero/skip/2ry/static_limit/infinity; got {invalid_mode!r}")
    keep_invalid = invalid_mode == "2ry"

    # Derive Fermi level, energy/band masks, and PPM pole masks in one fused trace.
    # valid_mask_q=None → all-true mask at the caller so the jit sees a real array.
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
        def _accumulate_kij_stream(global_idx: np.ndarray, contrib_batch: jax.Array) -> None:
            if dset_sigma_kij is None:
                return
            idx = np.asarray(global_idx, dtype=np.int64)
            buf = dset_sigma_kij[idx]
            buf = buf + np.asarray(jax.device_get(contrib_batch), dtype=np.complex128)
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

        # Sum cond+val per ω-half before gathering to host.  Preserves the
        # original traversal order so reduction ordering stays bit-identical.
        per_half: dict[tuple, jax.Array] = {}
        for _bi, br in enumerate(branches):
            sigma_kij, _ = _run_sigma_branch(
                omega_nonneg_ry=br.omega_abs, omega_global_idx=br.omega_idx,
                E_A=br.E_A, base_mask_A=br.base_mask_A,
                space=br.space, neg_omega_half=br.neg_omega_half,
                log_tag=br.tag,
                **common_branch_kwargs,
            )
            key = tuple(br.omega_idx.tolist())
            per_half[key] = (per_half[key] + sigma_kij) if key in per_half else sigma_kij

        if not streaming:
            # the reduce-scatter project (_make_project_ri_reduce_scatter)
            # returns Σ sharded (m_X, n_Y), so the
            # host copy needs a cross-process gather rather than jax.device_get.
            # _to_host_np falls back to device_get for single-process / replicated.
            for key, total in per_half.items():
                idx = np.asarray(key, dtype=np.int64)
                sigma_kij_host[idx] = _to_host_np(total, dtype=np.complex128, tiled=False)
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
