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
                         accumulator with the memory-tile sink.

This driver retains the physics prologue (PPM fit + physics-state prep) plus the
τ-loop orchestration that binds window × kernel × accumulator, and reads as the
8-stage teleology verbatim.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, NamedTuple
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common import jax_profile, timing
from common.units import RYD_TO_EV
from .gw_config import DynamicSigmaConfig, PPMConfig
from .minimax_config import MinimaxConfig
from .minimax_screening import (
    MinimaxNodes,
    fit_gn_ppm_from_wc_pair,
)
from .ppm_windows import (
    _SigmaWindow,
    branches_for_omega_grid,
    _build_windows_for_branch,
    window_mask_B_bounds,
    _to_host_np,
    _CROSSING_A_MAX,
    crossing_regularization_floor,
    resolve_sigma_regularization,
)
from .ppm_tau_kernel import _get_sigma_tau_kernel, get_sigma_spatial_kernel
from .ppm_accumulators import (
    _SigmaAccumulator,
    _TauAccumulator,
    _MemoryTileSink,
)
from .wavefunction_bundle import face_kernel_kwargs


def _face_g_plan(mesh_xy: Mesh, face_shape):
    """One ``distrib_la.gemm_plan`` for a face-layout G build, at the shape
    ``greens_function_kernel._face_build_G`` requires — mirrors
    ``cohsex_sigma._make_cohsex_kernels_face``'s identical construction
    (single source of truth: same GEMM shape, same convention, both
    named as "built ONCE by the caller" in ``build_G``'s own docstring).
    Used only by the invalid-pole static-limit term below
    (:func:`_compute_invalid_static_sigma` /
    :func:`_invalid_static_coh_by_bracket`), which runs O(1) times per
    Σ_c(ω) evaluation — NOT per τ — so building this plan inline here
    (rather than caching it, the way ``ppm_tau_kernel``'s per-τ hot-loop
    factories do) is proportionate to its call frequency."""
    from distrib_la import gemm_plan
    nk, nb_full, n_rmu, ns = (int(v) for v in face_shape)
    mu_s = n_rmu * ns
    return gemm_plan(mesh_xy, m=mu_s, k=nb_full, n=mu_s, nq=nk,
                     dtype=jnp.complex128)


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
    # (n_bracket, n_omega, nk, nb, nb).  THE LEADING AXIS IS THE BAND-COUNT
    # AXIS and it is CUMULATIVE: element ``i`` is Σ_c summed over bands
    # ``[0, band_counts[i])``, so ``sigma_c_kij[-1]`` is the ordinary
    # full-band Σ_c and is what every downstream consumer takes.  Length 1
    # in the ordinary case (``band_counts == (nband,)``), 3 under
    # ``sigma_band_extrapolation`` — one shape, one code path, no branch.
    #
    # Layout of the TRAILING four axes is carried BY THE ARRAY'S OWN
    # SHARDING (single source of truth): replicated/uncommitted under
    # sigma_omega_layout=replicated (historical), or
    # P(..., None, None, 'x', 'y') band-tiled under sigma_omega_layout=sharded —
    # consumers branch via qsgw_utils.is_band_sharded_sigma_omega.
    sigma_c_kij: jax.Array
    #: The LOGICAL band count each leading-axis element sums to.  Aligned
    #: with ``sigma_c_kij``'s axis 0; the extrapolation reads both together
    #: and nothing else needs either.
    band_counts: tuple[int, ...] = ()
    #: ``(n_bracket, nk, nb, nb)`` Ry, or ``None``.  The CUMULATIVE Coulomb-hole
    #: half of the ``ppm_invalid_mode="static_limit"`` term at each of
    #: ``band_counts`` — the part of Σ_c that is a static Coulomb hole and is
    #: therefore folded in as a CONSTANT rather than extrapolated.  Present
    #: only when the extrapolation is running (``len(band_counts) > 1``) and
    #: the run has invalid poles to treat.  It is NOT a component to be added
    #: to ``sigma_c_kij`` — that fold already happened; it is the evidence for
    #: how much of ``S_inf`` was never extrapolated
    #: (``band_extrapolation.static_limit_tail_ruling``).
    static_coh_at_counts: np.ndarray | None = None


class _SigmaBranchTiles(NamedTuple):
    """One branch's Σ_c as per-rank HOST tiles — the single-gather tail seam.

    Produced by ``_run_sigma_branch`` on the memory-tile (KIJ_HOST) path in
    place of the old device-assembled, branch-stripped jax.Array (comms fix,
    2026-07-28; evidence: AQ 4962c/P=64 gw.log branch tails — 4× device
    re-upload + 4× 64-process allgather of the full Σ slab, ~17-18 s of the
    Σ stage).  ``tiles[d]`` is (n_ω_branch, nk, m_pad/p_x, n_pad/p_y) numpy
    at global 4-D index ``tile_index[d]``; the driver sums branches at their
    global ω indices and gathers ONCE at stage end.  The mesh pad block is
    still attached (stripped once, after the gather).
    """
    tiles: list                      # list[np.ndarray], one per addressable shard
    tile_index: list                 # list[tuple[slice, ...]] 5-D global indices
    devices: list                    # owning jax devices, aligned with tiles
    spatial_padded: tuple            # (n_brk, nk_proj, m_pad, n_pad) padded extents
                                     # (the ω axis sits BETWEEN n_brk and nk_proj
                                     # in the assembled cube, so it is not here)
    sharding: NamedSharding          # P(None, None, None, 'x', 'y'), 5-D global
    nb_real: int                     # real QP window extent (pre-pad), for strip


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


#: Env escape hatch for :func:`assert_gapped_occupations_for_ppm`.  It is an
#: ENV knob and not a deck key for the same reason ``LORRAX_BAND_DEGENERACY``
#: is: it is a debugging escape for a deck you are looking at, not a property
#: of a calculation anyone would want recorded in an input file.
#: ``AGENT_PREAMBLE``: never set it to make a gate pass.
_PPM_METAL_ENV = "LORRAX_PPM_ALLOW_CROSSING_BANDS"


def assert_gapped_occupations_for_ppm(occ_full, *, print_fn=print) -> int:
    """Refuse a GN/HL-PPM Σ whose occupation table has a Fermi-crossing band.

    Returns the number of crossing bands (always 0 on the accepted path), so
    a caller or a test reads a NUMBER rather than the absence of an
    exception.

    WHAT IS MEASURED, AND WHY THE DECK KEY CANNOT ANSWER IT.
    ``gw_config._validate_metal_compute_mode`` already refuses
    ``mpa_material_class = metal`` outside ``compute_mode = mpa``.  But
    ``insulator`` is the DEFAULT, so a metallic system run without the key
    reaches this driver with nothing objecting — and the deck key is a
    DECLARATION, while this is a property of the spectrum.  So the
    measurement is on the occupation table:

        band n crosses E_F  <=>  occ[:, n] > 0.5 is not constant over k.

    ``wavefunction_bundle._build_occ`` fills ``occ`` as ``(enk <= efermi)``,
    exactly 0.0/1.0, so on a gapped system every band is uniformly occupied
    or uniformly empty and this is exactly zero.  A crossing band is the one
    thing that makes it nonzero, and it needs no tolerance: the predicate is
    over an integer table.

    WHAT GOES WRONG IF IT RUNS ANYWAY, which is why this refuses rather than
    warns.  ``_prepare_sigma_state`` builds ``vbm = max(enk | occupied)`` and
    ``cbm = min(enk | empty)``; with a crossing band ``vbm > cbm``, so the
    "midgap" ``0.5*(vbm+cbm)`` is not in any gap and the ``fermi_reference``
    the deck chose is meaningless.  It then clips
    ``E_cond = max(enk - efermi, 0)`` and ``H_val = max(efermi - enk, 0)``, so
    a band on the wrong side of that pseudo-Fermi level cannot even be
    REPRESENTED — its dynamic denominator collapses to the ω = 0 edge.  Every
    array keeps its shape and the run completes.  Measured on Na bcc SOC 48b
    (`reports/occupation_threshold_all_paths_2026-08-16/evidence/probe_na.log`,
    JID 57138992): MP1 gives f in [-0.002194, +1.031587] with 150
    negative-lobe and 12 over-one (k,n) entries, and the step split assigns
    every one of them fully to one branch with weight 1.

    This is the SCOPE LIMIT the module has documented in prose since the
    ``TODO(metal-greens)`` at ``_prepare_sigma_state`` was written, enforced.
    The port itself — feeding the iteration's ``OccupationState`` into
    ``branches_for_omega_grid`` as ``cond_weight = 1-f`` / ``val_weight = f``
    — remains open work; a documented limitation that nothing enforces is a
    limitation only the reader has.
    """
    occ = np.asarray(jax.device_get(occ_full))
    if occ.ndim != 2:
        raise ValueError(
            f"assert_gapped_occupations_for_ppm: expected (nk, nb) "
            f"occupations, got shape {occ.shape}")
    filled = occ > 0.5
    crossing = np.flatnonzero(filled.any(axis=0) & ~filled.all(axis=0))
    if crossing.size == 0:
        return 0
    if os.environ.get(_PPM_METAL_ENV, "").strip().lower() in ("1", "true", "on"):
        print_fn(
            f"  *** {_PPM_METAL_ENV} is set: running GN/HL-PPM Sigma on a "
            f"spectrum with {crossing.size} Fermi-crossing band(s) "
            f"{crossing.tolist()[:12]}.  The band split below is a 0/1 step "
            f"at a pseudo-Fermi level that is not in any gap, and E_cond/H_val "
            f"are clipped at zero, so wrong-side states are unrepresentable. "
            f"This is a debugging override, not a supported configuration. ***")
        return int(crossing.size)
    raise ValueError(
        f"GATE ppm_sigma_gapped_occupations: this spectrum has "
        f"{crossing.size} Fermi-crossing band(s) — occupied at some k and "
        f"empty at others — at band index/indices "
        f"{crossing.tolist()[:12]}"
        + (" (first 12 shown)" if crossing.size > 12 else "") + ".\n"
        f"  got:  a metallic occupation table on the GN/HL plasmon-pole "
        f"Sigma driver, whose band split is a hard occ > 0.5 step\n"
        f"  want: a gapped spectrum, i.e. every band uniformly occupied or "
        f"uniformly empty over k\n"
        f"  fix:  run this system with compute_mode = mpa and "
        f"mpa_material_class = metal, which carries the iteration's fixed-N "
        f"MP1 occupation state; or narrow the band window so no band "
        f"crosses E_F\n"
        f"  why:  with a crossing band, vbm > cbm, so the 'midgap' Fermi "
        f"reference this driver derives is not in any gap, and E_cond/H_val "
        f"are clipped at zero so a wrong-side band cannot be represented.  "
        f"Nothing about that changes an array shape or the exit code.\n"
        f"  override: {_PPM_METAL_ENV}=1 (debugging only)\n"
        f"  doc:  docs/theory/metallic-mpa-screening.md")


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
    # TODO(metal-greens): the finite-occupation Green's-function/Sigma
    # decomposition needs particle/hole weights f and 1-f, not f > 0.5.
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

    # Deck-level ε_H measurement (env-gated observability; channel-
    # hermiticity memo §1.3/§3.5): the Laplace-family symmetry diagnostics
    # (σ_R symmetric / σ_I antisymmetric, check L1) hold only to the PPM
    # amplitude's INHERITED hermiticity residual
    # ε_H = max_q |B_q − B_q†| / max|B| — inherited from the un-Hermitized
    # LU Dyson solve, gated in production only at q=0 / rtol 1e-6.  Measure
    # it, don't assume it.  The channel MERGE itself needs no hermiticity
    # (bilinearity), so this is diagnostic, not a gate; rtol=1.0 keeps the
    # HL probe (legitimately non-Hermitian B) from warning.
    if os.environ.get("LORRAX_PPM_HERM_DIAG", "0").strip().lower() in (
            "1", "true", "yes", "on"):
        from common import sanity
        _pf = print_fn if print_fn is not None else (lambda *a, **k: None)
        sanity.check_hermitian(f"{model_label} B_q (eps_H, all q)", B,
                               rtol=1.0, verbose=True, print_fn=_pf)
        sanity.check_hermitian(f"{model_label} Omega_q (symmetry, all q)",
                               Omega, rtol=1.0, verbose=True, print_fn=_pf)

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
        the signature here reads parallel to chi0's builders.  For
        Laplace windows the tuple is ``(X, None)`` with
        X = ψ†σψ = S_R + i·S_I from the single-chain kernel (the default
        and only Laplace channel plan); the accumulator consumes X
        directly.  Crossing windows deliver the (σ_re, σ_im) pair.
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
        # Crossing windows return σ^τ as a (re, im) tuple: the crossing
        # window's HGL quadrature consumes Im[coeff·σ] = Re(c)·S_I +
        # Im(c)·S_R with independent real ω-weights, so both channels must
        # ship.  Laplace windows return (X, None) — one complex tile, half
        # the projection GEMMs / collective payload / D2H bytes; legal
        # because their consumer forms only c·X (bilinearity).
        #
        # Per-τ timing sub-rows (instrumentation, 2026-07-28; evidence: AQ
        # 4962c/P=64 — 'sigma.exec 272.040' hid 176 uniform 1.51 s τ
        # dispatches with no finer attribution):
        #   sigma.tau.dispatch    the τ-kernel call.  Fused path: async
        #                         submit, ~0 host time.  With
        #                         LORRAX_SIGMA_TAU_TIMING=1 the staged
        #                         kernel emits blocking per-stage children
        #                         (w_phase / G_build / G_ifft / V_ifft /
        #                         GW_mult_fft / project_rs) under this row.
        #   sigma.tau.host_accum  add_tau: async-D2H drain of τ_{i-lag} +
        #                         the numpy ω-projection.  On the fused
        #                         path this row also absorbs the wait for
        #                         device compute (the deque's lag), so it
        #                         UPPER-bounds host-side work.
        # Overhead when nothing is enabled: two timing.section enter/exits
        # per τ (~µs) — scale-neutral (independent of n_atoms, N_μ, nk, P,
        # backend); the design-envelope τ counts (hundreds) keep this in
        # the sub-ms range per stage.
        with timing.section("sigma.tau.dispatch"):
            sigma_re, sigma_im = build_sigma_tau(
                jnp.asarray(t_c, dtype=jnp.complex128))
        if progress is not None:
            progress.step()
        with timing.section("sigma.tau.host_accum"):
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
    tau_kernel_x: Callable[..., jax.Array],
    log_tag: str,
    print_fn,
) -> None:
    """Walk windows; for each, dispatch ``minimax_tau_integrate_sigma``
    with closures that bind this window's (psi, masks, E_ref, kernel) and
    feed the window's σ^τ into the accumulator.  The result lands in
    per-rank host tiles — see _TauAccumulator + _MemoryTileSink.

    Channel-plan dispatch (owner ruling 2026-07-28; made the default and
    only path by owner order the same day): Laplace windows
    (project="full", project_code=0) ALWAYS run ``tau_kernel_x``, the
    merged single-complex-chain kernel — X = ψ†σψ, consumed directly by
    the accumulator as ``(X, None)`` — because their ω-consumer forms only
    c·(S_R+i·S_I) = c·X (bilinearity; channel-plan doc in
    ppm_tau_kernel._make_project_ri_reduce_scatter).
    Crossing windows (project="imag") ALWAYS dispatch ``tau_kernel``, the
    two-channel kernel, unchanged: their consumer weights S_R and S_I
    independently and X under-determines the pair.
    """
    from common.progress import LoopProgress

    branch_label = log_tag if log_tag else "sigma"
    total_tau_nodes = sum(win.n_tau for win in windows)
    progress = LoopProgress(
        total_tau_nodes, print_fn, title=f"sigma[{branch_label}]",
        item_name="tau node", max_updates=10)

    # One profiler SESSION per branch, first window only, active only when
    # ISDF_JAX_PROFILE_DIR is set (jax_profile.trace_section no-ops
    # otherwise — zero overhead in production).  The annotation/
    # step_annotation hooks below were already wired but inert without a
    # session; this is the missing session starter the AQ analysis called
    # out (2026-07-28): a perfetto trace of one window per branch is what
    # separates dot self-time from reduce-scatter wait inside the single
    # jit__tau_kernel module, which no timing.section row can.  First
    # window only, to bound trace size at any n_τ (scale-neutral).
    def _trace_tag(label: str) -> str:
        return "".join(c if (c.isascii() and c.isalnum()) else "_"
                       for c in label)

    with jax_profile.annotation(f"sigma_branch[{branch_label}]"):
        for win_idx, win in enumerate(windows):
            trace_ctx = (
                jax_profile.trace_section(
                    "sigma_tau_" + _trace_tag(branch_label))
                if win_idx == 0 else nullcontext())
            with trace_ctx, jax_profile.step_annotation(
                "sigma_window", step_num=win_idx,
                detail=f"{branch_label}:{win.name}:n{win.n_tau}",
            ):
                with timing.section("sigma.window_operands"):
                    mask_A_j    = jnp.asarray(win.mask_A)
                    # The B-side Ω window is TWO SCALARS, not a
                    # (nq, μ_pad, μ_pad) bool tile built here and pinned
                    # across the whole τ scan.  The predicate is rebuilt
                    # from Ω_q inside _build_W_t_q, where it fuses into
                    # the select that was already there.
                    _om_lo, _om_hi = window_mask_B_bounds(win)
                    Om_lo_j     = jnp.asarray(_om_lo, dtype=jnp.float64)
                    Om_hi_j     = jnp.asarray(_om_hi, dtype=jnp.float64)
                    E_ref_A_j   = jnp.asarray(win.E_ref_A, dtype=jnp.float64)
                    E_ref_B_j   = jnp.asarray(win.E_ref_B, dtype=jnp.float64)

                # Laplace windows dispatch the merged X kernel; crossing
                # windows the two-channel kernel (see docstring).
                use_merged_x = win.project_code == 0
                kern = tau_kernel_x if use_merged_x else tau_kernel

                def build_sigma_tau(t_j):
                    out = kern(
                        psi_coh_xn, psi_coh_yr,
                        psi_proj_xr, psi_proj_yn,
                        E_A, mask_A_j, B_q, Omega_q, base_mask_B,
                        Om_lo_j, Om_hi_j,
                        E_ref_A_j, E_ref_B_j, t_j,
                    )
                    # Merged kernel emits the single complex X = ψ†σψ; the
                    # accumulator reads (X, None).  Two-channel kernel emits
                    # the (σ_re, σ_im) tuple unchanged.
                    return (out, None) if use_merged_x else out

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
    """Zero-pad the sigma band window: m to a multiple of ``p_x``, n of ``p_y``.

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

    **The two axes are padded INDEPENDENTLY, and that is the whole point.**
    The precondition is ``m % p_x == 0`` AND ``n % p_y == 0`` — two separate
    one-axis constraints, because ``m`` is reduce-scattered over ``'x'``
    only and ``n`` over ``'y'`` only.  Rounding *both* up to a multiple of
    the PRODUCT ``p_x·p_y`` (what this used to do) satisfies them, but pays
    for it in the largest object the Σ branch carries: Σ_c(ω, k, m, n) and
    every per-τ tile that feeds it scale as ``m_pad · n_pad``.  On MoS₂
    12×12 at P=80 (8×10, window 70) that is 80×80 = 6400 where 72×70 = 5040
    suffices — **1.27× of the Σ_c tile, the host accumulate and the D2H
    copy, for nothing** — and on a square mesh it is far worse: at P=64
    (8×8) the product rule demands 128×128 = 16384 against 72×72 = 5184,
    i.e. **3.16×**.  Since Y.4 recommends square meshes for two independent
    reasons, the product rule was on a collision course with the mesh shape
    the campaign is moving to.

    Exactness is unchanged by the split: every output element
    ``Sigma[k,m,n]`` is an independent contraction, so the pad extent on
    one axis cannot perturb any element of the other, and the pad block is
    exactly zero either way.

    Returns ``(xr_padded, yn_padded, nb_real)``; a no-op (identity, same
    buffers) on whichever axis already divides.  The caller reads the two
    padded extents back off the returned arrays' shapes — they are no
    longer equal in general.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    nb_real = int(psi_proj_xr.shape[1])
    m_pad = -(-nb_real // p_x) * p_x          # round up to p_x  (reduce-scatter 'x')
    n_pad = -(-nb_real // p_y) * p_y          # round up to p_y  (reduce-scatter 'y')
    # psi_xr : (nk, m, s, mu_X) at P(None,None,None,'x') -> band axis 1
    # psi_yn : (nk, s, mu_Y, n) at P(None,None,'y',None) -> band axis 3
    # Neither band axis is mesh-sharded, so both pads are rank-local.
    xr_p = (psi_proj_xr if m_pad == nb_real else
            jnp.pad(psi_proj_xr,
                    ((0, 0), (0, m_pad - nb_real), (0, 0), (0, 0))))
    yn_p = (psi_proj_yn if n_pad == nb_real else
            jnp.pad(psi_proj_yn,
                    ((0, 0), (0, 0), (0, 0), (0, n_pad - nb_real))))
    return xr_p, yn_p, nb_real


def assert_sharded_sigma_window_divides_mesh(nb_proj: int, mesh_xy, *,
                                             ansatz: str):
    """THE precondition for a mesh-SHARDED ``Sigma_c(w,k,m,n)`` cube.

    ONE owner for a contract two ansaetze reach at the same seam.  Before
    2026-08-22 the GN/HL-PPM branch refused an indivisible sigma band
    window by name here while the MPA executor
    (``gw.mpa.sigma._integrate_sigma_batches``) padded with
    :func:`pad_sigma_window`, accumulated into a
    ``P(None,None,'x','y')`` array and stripped it again -- silently, and
    with no divisibility check anywhere in that module.  Two contracts at
    one seam is one contract too many: either pad+strip is safe on the
    sharded consumer path, in which case the PPM refusal should cite the
    proof and go, or it is not, in which case MPA must refuse too.  It is
    not, and the reason is in :func:`strip_sigma_window`.

    WHY PAD+STRIP IS NOT SAFE ON A SHARDED CUBE.  ``pad_sigma_window`` is
    exact -- every ``Sigma[k,m,n]`` is an independent contraction, so the
    pad block is exactly zero and stripping it loses nothing.  That
    argument is about VALUES and it holds here too.  What does not hold is
    the LAYOUT: the accumulator is born at ``P(None,None,'x','y')`` on the
    PADDED extents (``m_pad % p_x == 0`` by construction), and
    ``strip_sigma_window`` then slices it back to ``nb_real`` on both
    trailing axes.  When ``nb_real`` does not divide the mesh the result is
    an array whose declared sharding no longer divides its own shape, and
    every downstream consumer that reads the layout off the array
    (``qsgw_utils.is_band_sharded_sigma_omega``, the QSGW Hermitize, the
    SlabIO write) inherits that.  The PPM refusal already names the
    Hermitize; this makes the same statement once, for both.

    Raises ``ValueError`` naming the window, the mesh and the fix.  No-op
    when both axes divide.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    nb = int(nb_proj)
    if nb % p_x == 0 and nb % p_y == 0:
        return
    raise ValueError(
        f"{ansatz}: a mesh-sharded Sigma_c(w,k,m,n) cube requires the "
        f"sigma band window to divide the mesh on BOTH axes: nb={nb}, "
        f"mesh {p_x}x{p_y} (nb%p_x={nb % p_x}, nb%p_y={nb % p_y}).  The "
        f"pad/strip pair is exact in VALUE but leaves a sharded array "
        f"whose declared P(None,None,'x','y') no longer divides its own "
        f"shape, and the QSGW Hermitize needs a square unpadded extent.  "
        f"Choose nval+ncond divisible by both mesh extents"
        + (", or use sigma_omega_layout = replicated."
           if ansatz.endswith("ppm") else
           " (compute_mode = mpa has no replicated cube plan)."))


def strip_sigma_window(sigma_kij, nb_real: int):
    """Drop the :func:`pad_sigma_window` pad block from a (..., m, n) Sigma.

    The pad rows/cols are exactly zero (bilinear in zero-padded psi); this is
    the single seam where the padded extent stops.  No-op when unpadded.

    BOTH trailing extents are tested: since ``pad_sigma_window`` pads m and n
    independently, one axis can be at the real extent while the other is
    padded (mesh 8×10, window 70 → m=72, n=70).  Testing only the last axis
    would have returned an m-padded Σ untouched.
    """
    if sigma_kij is None:
        return sigma_kij
    if (int(sigma_kij.shape[-2]) == int(nb_real)
            and int(sigma_kij.shape[-1]) == int(nb_real)):
        return sigma_kij
    return sigma_kij[..., :nb_real, :nb_real]


def _run_sigma_branch(
    *,
    omega_nonneg_ry: np.ndarray,
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
    brackets: tuple[tuple[int, int], ...],
    log_tag: str = "",
    print_fn=print,
    use_shipped_minimax_tables: bool = True,
) -> tuple['_SigmaBranchTiles | None', list[_SigmaWindow]]:
    """Orchestrator for one branch (cond or val × pos or neg ω half).

    Reads as a physics outline:
        windows = _build_windows_for_branch(...)          # host
        acc     = _integrate_tau_windows_for_branch(...)  # device

    Returns a :class:`_SigmaBranchTiles` of per-rank HOST tiles still
    carrying the mesh pad — the driver sums branches on host and performs
    the single end-of-stage gather + strip (comms fix 2026-07-28, see
    _SigmaBranchTiles).  Empty branches return ``None``.
    """
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])

    s = wfns.slices
    face_kwargs = face_kernel_kwargs(wfns)
    if wfns.layout == "legacy":
        psi_coh_xn = wfns.xn(s.full)
        psi_coh_yr = wfns.yr(s.full)
        psi_proj_xr = wfns.xr(s.sigma)
        psi_proj_yn = wfns.yn(s.sigma)
        nk_proj = int(psi_proj_xr.shape[0])
        # Mesh-pad the QP band window: the reduce-scatter projector and the
        # Sigma_c tile sink both need m % p_x == 0 / n % p_y == 0 (see
        # pad_sigma_window).  ``nb_proj`` stays the REAL window everywhere
        # the caller can see; only the in-branch machinery runs at
        # ``nb_pad``.
        psi_proj_xr, psi_proj_yn, nb_proj = pad_sigma_window(
            psi_proj_xr, psi_proj_yn, mesh_xy)
        # m and n are padded to DIFFERENT extents in general (m→p_x, n→p_y).
        m_pad = int(psi_proj_xr.shape[1])
        n_pad = int(psi_proj_yn.shape[3])
    else:
        # Face carrier (2026-08-22): psi_mun/psi_nmu span the FULL
        # [b0,b4) loaded extent and cannot be sliced/windowed to the Σ
        # band window s.sigma (report obstacle #3) — used UNSLICED for
        # both the G-build ("coh") and projection ("proj") roles, same
        # operand identity as cohsex_sigma's own face kernels.  The
        # in-branch accumulator therefore runs at ``nb_full`` (already
        # mesh-divisible, BandSlices.b4's own invariant — no padding
        # needed), NOT the physical Σ window.  ``nb_proj`` still carries
        # the FINAL desired extent (nb_sigma): the driver's
        # ``strip_sigma_window`` call slices the fully-gathered/replicated
        # host array down to it after the τ loop, which is valid at ANY
        # extent (a plain host slice, not a sharding-divisibility
        # constraint) — see ``compute_sigma_c_ppm_omega_grid``'s
        # ``assert tile_meta.nb_real == nb_proj``.
        psi_coh_xn = wfns.psi_mun
        psi_coh_yr = wfns.psi_nmu
        psi_proj_xr = wfns.psi_nmu
        psi_proj_yn = wfns.psi_mun
        nk_proj = int(meta.nk_tot)
        nb_proj = int(s.nb_sigma)
        m_pad = n_pad = int(s.nb_full)

    if n_omega == 0:
        return None, []

    with timing.section("sigma.windows"):
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
        return None, []

    omega_vec = jnp.asarray(omega_nonneg_ry, dtype=jnp.float64)
    tau_kernel = _get_sigma_tau_kernel(
        mesh_xy=mesh_xy,
        kgrid=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
        brackets=brackets,
        **face_kwargs,
    )
    # Merged Laplace-plan sibling kernel (the default and only path for
    # project="full" windows — owner order 2026-07-28); crossing windows
    # keep tau_kernel, the two-channel kernel, unchanged.
    tau_kernel_x = _get_sigma_tau_kernel(
        mesh_xy=mesh_xy,
        kgrid=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
        merged_x=True,
        brackets=brackets,
        **face_kwargs,
    )

    # One async-D2H accumulator over the memory-tile sink: Σ_c(ω,k,m,n)
    # lives as per-rank numpy tiles matching σ(τ)'s (m_X, n_Y) sharding —
    # the full (n_ω,n_k,n_b,n_b) buffer never exists on any GPU until the
    # final device assembly at finalize().  (copy_to_host_async + a short
    # deque overlap GPU-τ_{k+lag} with the numpy-τ_k accumulate.)
    n_brk = len(brackets)
    sink = _MemoryTileSink(
        shape=(n_brk, n_omega, nk_proj, m_pad, n_pad),
        sharding=NamedSharding(mesh_xy, P(None, None, None, 'x', 'y')),
    )
    accumulator: _SigmaAccumulator = _TauAccumulator(
        omega_vec=omega_vec, sink=sink)

    _integrate_tau_windows_for_branch(
        windows=windows, accumulator=accumulator,
        E_A=E_A, B_q=B_q, Omega_q=Omega_q, base_mask_B=base_mask_B,
        psi_coh_xn=psi_coh_xn, psi_coh_yr=psi_coh_yr,
        psi_proj_xr=psi_proj_xr, psi_proj_yn=psi_proj_yn,
        tau_kernel=tau_kernel, tau_kernel_x=tau_kernel_x,
        log_tag=log_tag, print_fn=print_fn,
    )

    # Branch tail.  'sigma.finalize' is the timing row the AQ analysis asked
    # for (2026-07-28): the old tail hid a 4-5 s dead span per branch inside
    # the branch elapsed (deque drain + device re-upload + full-slab
    # process_allgather).  On the memory-tile path the tail is the branch's
    # ONLY pipeline flush (2026-08-01: end_window no longer drains — the
    # deque persists across windows, so the last ``lag`` τ's of the branch
    # drain here under this row; everything else overlaps).
    with timing.section("sigma.finalize"):
        tiles, tile_index, tile_devices = accumulator.finalize_host_tiles()
    # The mesh pad block stays attached here; the driver strips it ONCE
    # after the single end-of-stage gather (pad rows are exactly zero, so
    # summing padded branch tiles then stripping equals stripping each
    # branch — see pad_sigma_window/strip_sigma_window).
    return _SigmaBranchTiles(
        tiles=tiles,
        tile_index=tile_index,
        devices=tile_devices,
        spatial_padded=(n_brk, nk_proj, m_pad, n_pad),
        sharding=NamedSharding(mesh_xy, P(None, None, None, 'x', 'y')),
        nb_real=nb_proj,
    ), windows


def _compute_invalid_static_sigma(
    wfns,
    Wc0_q: jax.Array,
    invalid_mask: jax.Array,
    meta,
    mesh_xy: Mesh,
    occupation_state=None,
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

    Reuses the canonical Sigma spatial kernel with the masked static
    ``W^c(0)`` as the screening operand:

        Σ_static = sigma_sx(G_occ, W_static) + sigma_coh(W_static − 0)
                 = −⟨G_occ·W_static⟩ + ½·⟨G_RI·W_static⟩

    matching design note GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md §8
    (Σ_occ − ½·Σ_RI in its sign convention).  μ-pad safety is inherited
    from ``invalid_mask`` (pad modes are born dead at the fit, so they
    are never flagged invalid and ``W_static`` is exactly zero there).

    Returns the replicated host tensor (nk, nb_sigma, nb_sigma) in Ry,
    to be added to Σ_c at EVERY ω (the term is ω-independent).

    ``occupation_state`` is the iteration's
    :class:`gw.efermi.OccupationState` (``None`` ⇒ the integer projector,
    bit-exact insulating behaviour).  It matters here for the same reason
    it matters in ``cohsex_sigma``: the screened-exchange half of this
    term runs over the OCCUPIED manifold, so on a metal the Fermi-shell
    bands must enter with their fractional weights.
    """
    from common.collectives import gather_to_host
    from .cohsex_sigma import build_Gij, _occ_diag_full
    from .greens_function_kernel import build_G

    Gij = build_Gij(meta, mesh_xy, occupation_state)
    face_kwargs = face_kernel_kwargs(wfns)
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=meta.kgrid, merged_x=True, **face_kwargs)
    s = wfns.slices
    g_plan = (_face_g_plan(mesh_xy, face_kwargs["face_shape"])
             if wfns.layout == "face" else None)

    with mesh_xy:
        W_static = jnp.where(
            jnp.asarray(invalid_mask, dtype=bool),
            jnp.asarray(Wc0_q, dtype=jnp.complex128),
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )
        W_prep = spatial.prep_w(W_static)

        if wfns.layout == "legacy":
            psi_xr, psi_yn, nb_real = pad_sigma_window(
                wfns.xr(s.sigma), wfns.yn(s.sigma), mesh_xy)
        else:
            # Face: project over the FULL nb_full extent (report §3), then
            # strip to nb_sigma below — same "weight, don't window" +
            # late-window pattern as cohsex_sigma's own face kernels.
            psi_xr, psi_yn = wfns.psi_nmu, wfns.psi_mun
            nb_real = int(s.nb_sigma)

        # The shared spatial kernel returns -<G.W>.  Gather each tiny sharded
        # band tensor before building the next centroid-square G: this makes
        # the one-G-at-a-time memory bound structural instead of leaving XLA
        # free to overlap the occupied and RI contractions.  The production
        # fused-FFI route additionally keeps the R-space G tile inside its
        # bounded handler rather than materialising the decomposed FFT chain.
        if wfns.layout == "legacy":
            G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)
        else:
            nb_full = int(s.nb_full)
            phases = _occ_diag_full(Gij, s.nb_sigma, nb_full)
            G_occ = build_G(wfns.psi_mun, wfns.psi_nmu, phases=phases,
                            layout="face", gemm=g_plan)
        sig_sx = spatial.conv_project(psi_xr, psi_yn, G_occ, W_prep)
        sx_host = np.asarray(strip_sigma_window(
            gather_to_host(sig_sx), nb_real), dtype=np.complex128)
        del G_occ, sig_sx

        if wfns.layout == "legacy":
            G_ri = build_G(wfns.xn(s.sigma_sum), wfns.yr(s.sigma_sum))
        else:
            mask = wfns.band_mask(s.sigma_sum).astype(jnp.complex128)
            G_ri = build_G(wfns.psi_mun, wfns.psi_nmu, phases=mask,
                           layout="face", gemm=g_plan)
        sig_ri = spatial.conv_project(psi_xr, psi_yn, G_ri, W_prep)
        ri_host = np.asarray(strip_sigma_window(
            gather_to_host(sig_ri), nb_real), dtype=np.complex128)
        del G_ri, sig_ri

    # shared_conv(G_RI, W_static) = -<G_RI.W_static>, while static COH is
    # +1/2<G_RI.W_static>; hence the minus one-half below.
    return sx_host - 0.5 * ri_host


def _invalid_static_coh_by_bracket(
    wfns,
    Wc0_q: jax.Array,
    invalid_mask: jax.Array,
    meta,
    mesh_xy: Mesh,
    brackets,
) -> np.ndarray:
    """The static-limit term's OWN band-count series, one point per bracket.

    WHY THIS EXISTS — THE ONE PLACE THE PPM-ONLY GUARD CANNOT REACH.
    ``gw.sigma_dispatch``'s guard keeps the band extrapolation away from a
    static Coulomb hole because the ``1/N → 0`` limit is wrong for one (it
    ANTI-converges: 94.9 → 288.2 meV MAE as nband goes 60 → 124, overshooting
    BerkeleyGW's exact closure by ~340 meV — ``gw.band_extrapolation``'s module
    docstring owns that measurement).  That guard is per-``compute_mode``, and
    ``ppm_invalid_mode = "static_limit"`` — the SHIPPING DEFAULT — puts a
    static Coulomb hole inside a Σ whose ``compute_mode`` genuinely IS
    ``gn_ppm``.  The guard cannot see it because it is per-MODE, one logical
    ISDF mode at a time, underneath the seam the guard checks.

    WHAT IS AND IS NOT BAND-COUNT DEPENDENT.  The static-limit term is
    ``Σ_static = −⟨G_occ·W_static⟩ + ½⟨G_RI·W_static⟩``.  The first half runs
    over OCCUPIED states through ``Gij`` and is band-count independent for any
    ``nband ≥ nelec``.  The second — the Coulomb hole — runs over ``s.full``
    with no occupation projector, so it carries exactly the slowly convergent
    unoccupied tail this module exists to worry about.  Only the second half is
    measured here, and that is not a simplification: the extrapolation is an
    AFFINE estimator with ``sum(c) == 1``, so any band-count-INDEPENDENT part
    passes through it unchanged and cancels identically out of
    ``S_inf − S(N₃)``.  The occupied half therefore cannot contribute to the
    diagnostic even in principle.

    IT IS FREE, WHICH IS WHY IT CAN BE ON BY DEFAULT.  ``G_RI`` is a plain band
    sum, so the brackets PARTITION it the same way they partition Σ_c, and
    ``n_brk`` contractions over disjoint sub-ranges cost the same total flops
    as the one contraction over the whole range that runs anyway.  The price is
    one extra pass of the COH channel, not ``n_brk`` extra passes.

    Returns the replicated host tensor ``(n_brk, nk, nb_sigma, nb_sigma)`` in
    Ry, holding the DISJOINT per-bracket contributions — the caller cumulates
    them to get Σ_COH at each of the plan's band counts, in the same order and
    by the same rule that turns the Σ_c brackets into band counts.
    """
    from common.collectives import gather_to_host
    from .greens_function_kernel import build_G

    face_kwargs = face_kernel_kwargs(wfns)
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=meta.kgrid, merged_x=True, **face_kwargs)
    s = wfns.slices
    g_plan = (_face_g_plan(mesh_xy, face_kwargs["face_shape"])
             if wfns.layout == "face" else None)

    out = []
    with mesh_xy:
        W_static = jnp.where(
            jnp.asarray(invalid_mask, dtype=bool),
            jnp.asarray(Wc0_q, dtype=jnp.complex128),
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )
        W_prep = spatial.prep_w(W_static)
        if wfns.layout == "legacy":
            psi_xr, psi_yn, nb_real = pad_sigma_window(
                wfns.xr(s.sigma), wfns.yn(s.sigma), mesh_xy)
        else:
            psi_xr, psi_yn = wfns.psi_nmu, wfns.psi_mun
            nb_real = int(s.nb_sigma)
        for lo, hi in brackets:
            if wfns.layout == "legacy":
                G_ri = build_G(
                    wfns.xn(slice(int(lo), int(hi))),
                    wfns.yr(slice(int(lo), int(hi))))
            else:
                # "Weight, don't window" — the SAME bracket-as-mask
                # substitute the tau kernel's own _bracketed_face uses:
                # psi_mun/psi_nmu cannot be sliced to an arbitrary band
                # sub-range, so the bracket becomes a band-range mask
                # applied as a phase weight instead.
                mask = wfns.band_mask(
                    slice(int(lo), int(hi))).astype(jnp.complex128)
                G_ri = build_G(wfns.psi_mun, wfns.psi_nmu, phases=mask,
                               layout="face", gemm=g_plan)
            sig_ri = spatial.conv_project(psi_xr, psi_yn, G_ri, W_prep)
            ri_host = np.asarray(strip_sigma_window(
                gather_to_host(sig_ri), nb_real), dtype=np.complex128)
            out.append(-0.5 * ri_host)
            del G_ri, sig_ri
    return np.stack(out, axis=0)


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
    sigma_cfg: DynamicSigmaConfig,
    quad: MinimaxConfig,
    omega_grid_ry: np.ndarray,
    ansatz: str,
    occupation_state=None,
    plan: 'BandBracketPlan | None' = None,
    print_fn=print,
) -> SigmaOmegaResult:
    """Compute Σ^c_kij(ω) via GN-PPM windowed minimax integration.

    Config seam (WS2): scalar knobs are read by direct attribute access
    off the validated frozen ``ppm_cfg`` (no ``getattr(..., default)`` —
    a stale/typo'd name must raise, not silently default); the derived
    ω-grid arrives as an explicit data argument.  ``ppm_cfg``/``quad``
    never travel below this driver.

    ``occupation_state`` is forwarded to the invalid-pole static-COHSEX
    term only (the sole occupation projector this driver builds); the
    dynamic branches take their occupations from ``wfns.occ``.  ``None``
    is the insulating default and is bit-exact.

    ``plan`` is the band-bracket plan (:mod:`gw.band_extrapolation`).
    ``None`` means the trivial one — a single bracket over every band —
    which is the ordinary Σ_c and is bit-identical to the un-bracketed
    code this replaced.  A three-bracket plan makes the τ kernel build
    three DISJOINT Green's functions per τ against one W(τ), and the
    returned cube's leading axis carries their CUMULATIVE sums.

    The quadrature, the ω grid, E_ref_A/E_ref_B, W(τ) and the ISDF
    representation are all built from the FULL band range BEFORE the
    bracket loop and shared verbatim by every bracket — which is what
    makes the three points differ by band count and nothing else.
    """
    from .band_extrapolation import (
        assert_brackets_match_ols_abscissae, trivial_plan)

    s = wfns.slices
    if plan is None:
        # The Σ count, not the loaded extent — see the comment at the
        # ``plan_band_brackets`` call in ``ppm_pipeline`` for why these are
        # different numbers on a split deck and the same one otherwise.
        plan = trivial_plan(int(s.nb_sigma_sum), int(s.b2 - s.b0),
                            int(meta.b_id_4_sigma_user or s.b4) - int(s.b0))
    # THE PARTITION IS ENTERED ON THE NEXT LINE.  Checked here as well as at
    # the ``ppm_pipeline`` plan seam because this is where ``plan.bounds``
    # stops being a description and becomes the slices ``_run_sigma_branch``
    # takes of the ψ/E/mask operands — and because a plan reaching this
    # function from anywhere else (a future caller, a fixture) gets the same
    # guarantee.  ``psi_coh_*`` below are built over ``s.full``, the LOADED
    # extent, so nothing about the slicing itself would complain if the
    # brackets ran past the Σ band sum: it would silently sum χ-only bands.
    assert_brackets_match_ols_abscissae(
        plan, s, meta=meta, where="ppm_sigma bracket partition")
    brackets = plan.bounds
    n_brk = plan.n_brackets
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

    # Ansatz-neutral grid/window knobs come from DynamicSigmaConfig; the PPM
    # object contributes only its invalid-pole policy.
    regularization_width_ry = float(sigma_cfg.regularization_ev) / RYD_TO_EV
    edge_factor = float(sigma_cfg.window_edge_factor)
    # The ansatz NAME comes from ``compute_mode`` -- the CANONICAL axis --
    # so the shared xi resolver and the sharded-window precondition decide
    # the same way here and at the sigma_mnk.h5 writer, which reads
    # ``config.compute_mode`` directly.  Deliberately NOT ``ppm_cfg.model``:
    # that is one of the five legacy self-energy-axis keys, and every other
    # runtime consumer in the tree already reads ``compute_mode.ppm_model``.
    ansatz_name = str(getattr(ansatz, "value", ansatz)).strip().lower()
    regularization_floor_ev = getattr(
        sigma_cfg, "regularization_floor_ev", None)

    # Crossing-quadrature conditioning floor: raise ξ if the Σ_c ω-grid is wide
    # enough that the HGL core window would be ill-conditioned (Σ|α| ~ 1e5,
    # amplifying the mesh-sensitive per-τ operand → device-dependent Σ_c blow-up
    # + O(1e3) eV Im).
    #
    # RESOLVED BY THE SHARED RESOLVER, not here.  ``ppm_windows.
    # resolve_sigma_regularization`` is the one place any ansatz decides its
    # effective ξ, and it is a pure function of (requested ξ, ω grid, edge
    # factor, ansatz, floor policy) — so the Σ_c(ω) HDF5 writer re-derives the
    # SAME number from the same config and stamps it, instead of the resolved
    # value living only in this local and a print.  Before 2026-08-22 MPA
    # passed ``regularization_ev`` straight through while this raised it
    # silently: 1.90x apart on the sodium 48b deck, 5.7x on a +/-15 eV window.
    _xi = resolve_sigma_regularization(
        requested_ry=regularization_width_ry,
        omega_grid_ry=np.asarray(omega_values_ry, dtype=np.float64),
        edge_factor=edge_factor,
        ansatz=ansatz_name,
        floor_ev=regularization_floor_ev,
    )
    print_fn(_xi.describe())
    if _xi.raised:
        print_fn(
            f"    (A_core capped at {_CROSSING_A_MAX:.0f}; the requested ξ "
            f"would make the HGL crossing quadrature ill-conditioned)")
    regularization_width_ry = _xi.resolved_ry
    fermi_reference = sigma_cfg.fermi_reference
    invalid_mode = ppm_cfg.invalid_mode

    if nk != int(enk_full.shape[0]):
        raise ValueError(f"enk_full shape mismatch: expected first dim {nk}, got {enk_full.shape[0]}")

    omega_req = np.asarray(omega_values_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_values_ry must be a 1D non-empty array.")

    # fermi_reference is validated + normalized at config construction;
    # used directly here (fermi → traced bool below).

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

    # THE OCCUPATION SPLIT THIS DRIVER CAN HONOUR IS A GAPPED ONE.  Measured
    # here, on the spectrum, because the deck key cannot see it: config's
    # ``_validate_metal_compute_mode`` refuses ``mpa_material_class = metal``
    # outside MPA, but ``insulator`` is the DEFAULT, so a metallic system run
    # without the key reaches this driver and nothing objects.
    assert_gapped_occupations_for_ppm(occ_full, print_fn=print_fn)

    # Derive Fermi level, energy/band masks, and PPM pole masks in one fused trace.
    # valid_mask_q=None → all-true mask at the caller so the jit sees a real array.
    # (μ-pad modes need no mask here: they are born with Ω = 0 at the fit
    # and drop out of B_mask_raw structurally — see _prepare_sigma_state.)
    if valid_mask_q is None:
        valid_mask_q = jnp.ones(Omega_q.shape, dtype=bool)
    with timing.section("sigma.state"):
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
    # Computed once here, added to Σ_c at every ω (host tensor add, or
    # tile-local on the sharded layout — same values on both).
    sigma_static_host = None
    static_coh_at_counts = None
    if invalid_static and n_invalid:
        sigma_static_host = _compute_invalid_static_sigma(
            wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy,
            occupation_state=occupation_state)
        print_fn(
            "  GN invalid modes → static COHSEX: max|Σ_static| = "
            f"{float(np.max(np.abs(sigma_static_host))) * RYD_TO_EV:.4f} eV "
            f"(diag max {float(np.max(np.abs(np.diagonal(sigma_static_host, axis1=1, axis2=2)))) * RYD_TO_EV:.4f} eV)"
        )
        # THE CONTAMINANT'S OWN BAND-COUNT SERIES — measured, not assumed.
        # Only when the band extrapolation is actually running (n_brk > 1):
        # with one bracket there is no fit to contaminate and no reader to
        # inform, and the extra COH pass would be paid for nothing.
        if n_brk > 1:
            static_coh_at_counts = np.cumsum(
                _invalid_static_coh_by_bracket(
                    wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy,
                    brackets),
                axis=0)

    # Host-tile accumulation is the only mode (``kij_stream`` REMOVED
    # 2026-07-31).  ``nk_proj``/``nb_proj`` are read off ``meta``/``s``
    # directly (2026-08-22) rather than off a ``wfns.xr(s.sigma)`` probe
    # slice — the FINAL Σ_c(ω,k,m,n) extent this driver publishes is
    # ALWAYS (nk_tot, nb_sigma, nb_sigma) regardless of which internal
    # accumulator extent ``_run_sigma_branch`` used to get there (nb_sigma
    # itself under legacy; the mesh-divisible nb_full under face — see
    # that function's own docstring), and reading it off ``wfns.xr``
    # crashed under ``layout='face'`` (that accessor refuses by name;
    # this WAS the exact crash site the low_mem_bands_dynamic_ppm_unported
    # refusal's discovery run hit first).
    nk_proj = int(meta.nk_tot)
    nb_proj = int(s.nb_sigma)
    n_omega = int(omega_req.size)
    kij_bytes = float(n_omega * nk_proj * nb_proj * nb_proj * 16)

    # Σ_c(ω) end-of-stage layout (wk_REL ω-cube sharding).  "sharded" keeps
    # the per-rank (m_X, n_Y) host tiles where the stacked psum_scatter left
    # them and publishes them as ONE P(None,None,'x','y')-sharded jax.Array
    # on the EXISTING mesh — the full-cube reconstruction gather (the
    # P-independent n_ω·nk·nb²·16 B 'sigma.host_gather' collective,
    # 2751 MB/rank at nb=512) is elided, and every consumer reads the tiles
    # at their native sharding.  Movement-only: outputs are bit-identical
    # (A/B gated).  Announced here per doctrine 3.
    sharded_layout = (str(sigma_cfg.omega_layout) == "sharded")
    if sharded_layout and wfns.layout == "face":
        # NOT PORTED (2026-08-22).  The sharded-output tail below
        # (`sigma.tile_finalize`) asserts the per-rank tiles it received
        # are ALREADY at the nb_proj=nb_sigma extent with no pad
        # (`tile_meta.spatial_padded == (n_brk, nk_proj, nb_proj, nb_proj)`)
        # — true under legacy (pad_sigma_window rounds UP to nb_proj's own
        # mesh-divisible ceiling) but false under face whenever
        # nb_full != nb_sigma (the internal accumulator runs at nb_full,
        # per `_run_sigma_branch`'s own docstring).  `sigma_omega_layout
        # = sharded` is production-reachable only for
        # `mpa_material_class = metal` (gw_config.py), which is already
        # refused under `low_mem_bands = true`
        # (`low_mem_bands_metal_material_class_unported`) — so this is a
        # defensive backstop for a deck that sets the layout explicitly
        # without going through that path, not a gap in the metal port.
        raise NotImplementedError(
            "compute_sigma_c_ppm_omega_grid: sigma_omega_layout = 'sharded' "
            "is not ported for low_mem_bands = true, layout = 'face' — the "
            "sharded end-of-stage tail assumes the branch tiles already sit "
            "at the nb_sigma extent with no pad, which face's internal "
            "nb_full-extent accumulator does not generally satisfy.  Use "
            "sigma_omega_layout = replicated (the default) under "
            "low_mem_bands = true, or low_mem_bands = false for a sharded "
            "Σ_c(ω) cube.")
    if sharded_layout:
        # ONE owner for this precondition; the MPA executor calls the same
        # function (doctrine 3 / pattern #6 -- refuse with the fix named,
        # never fall back silently).
        assert_sharded_sigma_window_divides_mesh(
            nb_proj, mesh_xy, ansatz=ansatz_name)
        print_fn(
            "  Σc layout: sharded — Σ_c(ω,k,m,n) stays (m_X, n_Y)-tiled on "
            "the existing mesh; the end-of-stage full-cube replication "
            f"gather ({kij_bytes / 1e6:.2f} MB/rank) is ELIDED "
            "(sigma_omega_layout=sharded).")

    sigma_kij_host = (
        None if sharded_layout
        else np.zeros((n_brk, n_omega, nk_proj, nb_proj, nb_proj),
                      dtype=np.complex128)
    )

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
        use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
        brackets=brackets,
    )

    # Enumerate the 4 branches (ω sign × cond/val), skipping empty ω halves.
    # See _iter_branches for how each branch's physical identity fixes its
    # denominator/prefactor signs (no ±1 sign fields are carried).
    branches = branches_for_omega_grid(
        omega_req,
        E_cond=E_cond, H_val=H_val,
        cond_mask=cond_mask, val_mask=val_mask,
    )

    # Run each branch and fold its Σc tiles into per-rank HOST tile
    # accumulators at the branch's global ω indices.  cond and val of a
    # given ω-half share those indices, so the second branch's `+=` sums
    # cond+val there — same values, same traversal order (cond before
    # val), same pairwise-add order per element as the old per-branch
    # allgather + host fold, so the result is bit-identical.
    #
    # Comms fix — scorecard AK.9's second named lever, the P-independent
    # per-branch `_to_host_np(sigma_kij, tiled=False)` full-slab gather
    # onto EVERY rank (≈237 MB/branch at 606c/b160, grows as nb²).
    # (2026-07-28; evidence: AQ 4962c/P=64 run run_AQ_c4962_p64_mpi;
    # measured at that shape by job 7878038: finalize+gather now
    # 0.24 s total vs ~1-4 s before — small at nb=128; the target is
    # the nb² growth term.)  Σ_c already lives as per-rank host numpy tiles in
    # exactly the (m_X, n_Y) sharding the reduce-scatter projector
    # emitted; the old tail re-uploaded them to device and
    # process_allgather'd the FULL (n_ω_branch, nk, nb, nb) slab to all
    # ranks once PER BRANCH.  Now the tiles stay on host across
    # branches and ONE gather runs at the end of the stage
    # ('sigma.host_gather' below).  Design-envelope scaling: the moved
    # object is the QP-band-window slab (n_ω · nk · nb_σ² · 16 B),
    # independent of N_μ; per-rank tiles shrink as 1/P, and the 4→1
    # replication cut is flat in n_atoms / nk / N_μ / P on CPU and GPU
    # backends — nothing here is tuned to the rehearsal shape.
    tile_acc = None     # per-shard host accumulators, full ω extent
    tile_meta = None    # first branch's _SigmaBranchTiles (layout metadata)
    sigma_kij_sharded = None  # sharded-layout result (set in tile_finalize)
    for br in branches:
        branch_tiles, _ = _run_sigma_branch(
            omega_nonneg_ry=br.omega_abs,
            E_A=br.E_A, base_mask_A=br.base_mask_A,
            space=br.space, neg_omega_half=br.neg_omega_half,
            log_tag=br.tag,
            **common_branch_kwargs,
        )
        if branch_tiles is None:
            continue
        idx = np.asarray(br.omega_idx, dtype=np.int64)
        if tile_acc is None:
            tile_meta = branch_tiles
            # The bracket axis leads and the branch's ω axis sits BEHIND it
            # (a branch owns a subset of ω but every bracket), so the full-ω
            # accumulator keeps t's axis 0 and widens axis 1.
            tile_acc = [
                np.zeros((t.shape[0], n_omega) + t.shape[2:],
                         dtype=np.complex128)
                for t in branch_tiles.tiles]
        else:
            # All branches run the same ψ window on the same mesh, so
            # their tile layouts must agree — a mismatch here is a bug,
            # not a configuration (QUALITY_PATTERNS #7: guard it).
            assert (branch_tiles.tile_index == tile_meta.tile_index
                    and branch_tiles.spatial_padded == tile_meta.spatial_padded), (
                f"sigma branch tile layout drifted across branches: "
                f"{branch_tiles.spatial_padded} vs {tile_meta.spatial_padded}")
        with timing.section("sigma.branch_fold"):
            for d, t in enumerate(branch_tiles.tiles):
                tile_acc[d][:, idx] += t

    # Single end-of-stage gather: assemble the global padded Σ_c from the
    # per-rank host tiles and reconstruct it on every rank's host — ONCE,
    # replacing the old 4 per-branch device round trips + allgathers.
    # (Every rank needs the full tensor: downstream head injection /
    # diag(Σ_c) interpolation / sigma_mnk write all read the replicated
    # host copy, and the final jnp.asarray(sigma_kij_host) must agree
    # across processes.)  Skipped entirely on the sharded layout, whose
    # tail is the 'sigma.tile_finalize' block below.
    if not sharded_layout and tile_acc is not None:
        assert tile_meta.nb_real == nb_proj
        with timing.section("sigma.host_gather"):
            # spatial_padded is (n_brk, nk, m_pad, n_pad); the ω axis is
            # INSERTED at position 1, behind the leading bracket axis.
            _sp = tuple(int(v) for v in tile_meta.spatial_padded)
            padded_shape = (_sp[0], n_omega) + _sp[1:]
            if int(jax.process_count()) == 1:
                # Every shard is addressable (single process, any device
                # count — no shard-0 assumption, Bug C): pure host
                # assembly, no device hop at all.
                full_pad = np.zeros(padded_shape, dtype=np.complex128)
                for t, ix in zip(tile_acc, tile_meta.tile_index):
                    full_pad[ix] = t
            else:
                arrays = [jax.device_put(t, d)
                          for t, d in zip(tile_acc, tile_meta.devices)]
                gathered = jax.make_array_from_single_device_arrays(
                    padded_shape, tile_meta.sharding, arrays)
                full_pad = _to_host_np(
                    gathered, dtype=np.complex128, tiled=True)
            # Strip the mesh pad block (exactly zero) — the ONE seam
            # where the padded QP window stops; everything below (host
            # Σ buffer, eqp write) sees only the real nb_proj extent.
            sigma_kij_host += strip_sigma_window(full_pad, nb_proj)

    # Sharded-layout tail: NO reconstruction collective.  The per-rank
    # host tiles (already branch-summed at their global ω indices, in
    # the same element order as the replicated path) get the
    # static-COHSEX invalid-pole term added RANK-LOCALLY, are placed
    # back on their owning local devices (device_put of a process-local
    # buffer — no collective), and are published as ONE
    # P(None,None,'x','y')-sharded jax.Array on the existing mesh.
    # Consumers (head injection, diag extraction, QSGW build,
    # sigma_mnk.h5 SlabIO write) read this array at its native
    # sharding.  The timing row exists to PROVE the tail stays ~0 s
    # (it replaces 'sigma.host_gather', which does not run here).
    if sharded_layout:
        with timing.section("sigma.tile_finalize"):
            gshape = (n_brk, n_omega, nk_proj, nb_proj, nb_proj)
            if tile_acc is None:
                # No branch produced tiles (all-empty branches): a zero
                # Σ_c, mirroring the replicated path's untouched zeros
                # buffer.  Same metadata idiom as
                # _MemoryTileSink.host_tiles()'s empty path.
                sharding = NamedSharding(
                    mesh_xy, P(None, None, None, 'x', 'y'))
                devices = list(sharding.addressable_devices)
                dmap = sharding.devices_indices_map(gshape)
                local_shape = sharding.shard_shape(gshape)
                tile_acc = [np.zeros(local_shape, dtype=np.complex128)
                            for _ in devices]
                tile_index = [tuple(dmap[d]) for d in devices]
            else:
                assert tile_meta.nb_real == nb_proj
                # The divisibility refusal above guarantees the mesh pad
                # resolved to identity — padded extents ARE the real
                # extents (pattern #7: assert it, don't assume it).
                assert tuple(int(s) for s in tile_meta.spatial_padded) \
                    == (n_brk, nk_proj, nb_proj, nb_proj), (
                    f"sharded Σ layout saw a padded window "
                    f"{tile_meta.spatial_padded} despite the "
                    f"divisibility guard (nb={nb_proj})")
                sharding = tile_meta.sharding
                devices = tile_meta.devices
                tile_index = tile_meta.tile_index
            # static_limit fold, rank-local — same per-element order as
            # the replicated path (branch sum first, then the static
            # term); tile_index[d] is (bracket-slice, ω-slice, k-slice,
            # m-slice, n-slice) into the global cube.
            #
            # THE TERM GOES INTO BRACKET 0 ONLY, not into every bracket.
            # Adding it to every bracket would multiply it by the bracket
            # index under the cumulative sum; adding it to the first puts it
            # into all three cumulative sums exactly once.
            #
            # ⚠ THE REASON IS **NOT** THAT THE TERM IS BAND-COUNT
            # INDEPENDENT.  It is not.  This comment used to claim it was
            # ("no unoccupied tail"), and that claim is false: half of
            # Σ_static is ``+½⟨G_RI·W_static⟩`` with ``G_RI`` a sum over
            # ``s.full`` and NO occupation projector
            # (``cohsex_sigma.sigma_coh``), i.e. precisely the Coulomb hole's
            # slowly convergent unoccupied tail.  MEASURED — see the
            # ``static-limit`` block the extrapolation report now prints.
            #
            # The real reason is the same one the PPM-only guard in
            # ``sigma_dispatch`` is built on: **a static Coulomb hole must
            # not be run through this estimator.**  Its ``1/N → 0`` limit
            # ANTI-converges (94.9 → 288.2 meV MAE as nband goes 60 → 124,
            # ~340 meV past BerkeleyGW's exact closure —
            # ``band_extrapolation``'s module docstring).  Folding the term
            # in as a CONSTANT is what keeps it out of the fitted slope: the
            # estimator is affine with ``sum(c) == 1``, so a constant passes
            # into ``S_inf`` 1:1 and contributes nothing to ``A``,
            # ``Δ_tail``, ``Δ_model``, the residual or the verdict.  That is
            # the right treatment and it is deliberate.
            #
            # What it COSTS is that this term is then pinned at ``N₃`` and
            # never extrapolated, so ``S_inf`` carries a band-truncated
            # static Coulomb hole that the reported extrapolation bar does
            # not cover.  That residual is invisible by construction — a
            # constant cancels out of every diagnostic above — which is why
            # it is measured separately and reported by name
            # (``_invalid_static_coh_by_bracket`` →
            # ``band_extrapolation.static_limit_tail_ruling``) instead of
            # being asserted away in a comment, as it was here.
            if sigma_static_host is not None:
                for d, ix5 in enumerate(tile_index):
                    tile_acc[d][0] += sigma_static_host[tuple(ix5[2:])][None, ...]
            arrays = [jax.device_put(t, dev)
                      for t, dev in zip(tile_acc, devices)]
            sigma_kij_sharded = jax.make_array_from_single_device_arrays(
                gshape, sharding, arrays)

    # static_limit: fold the ω-independent invalid-pole static-COHSEX
    # term into Σ_c at every ω (host add; the sharded layout folded it
    # tile-locally above — identical values).
    # (Bracket 0 only — see the sharded twin above for why.)
    if sigma_static_host is not None and not sharded_layout:
        sigma_kij_host[0] += sigma_static_host[None, ...]

    # ── THE BRACKETS BECOME BAND COUNTS ─────────────────────────────────
    # Up to here the leading axis holds DISJOINT band brackets; a
    # cumulative sum along it turns them into Σ_c at each of the plan's
    # band counts, with the last element the ordinary full-band Σ_c.  At
    # n_brk = 1 the cumulative sum is the identity on the values, which is
    # what keeps the default path bit-identical.  Done in place on the
    # replicated path so the default run pays no second copy of the cube.
    if sharded_layout:
        # Axis 0 is replicated in the Σ sharding, so the scan is entirely
        # shard-local; pin the output sharding so it cannot drift off the
        # layout every consumer downstream reads.
        sigma_kij_req = jax.jit(
            lambda a: jnp.cumsum(a, axis=0),
            out_shardings=sigma_kij_sharded.sharding)(sigma_kij_sharded)
    else:
        np.cumsum(sigma_kij_host, axis=0, out=sigma_kij_host)
        sigma_kij_req = jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * RYD_TO_EV, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
        band_counts=tuple(int(c) for c in plan.counts),
        static_coh_at_counts=static_coh_at_counts,
    )
