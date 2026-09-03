"""Two-point GN/HL-PPM fit and its adapter to shared dynamic Sigma.

The PPM-specific work ends after fitting one ``(Omega, B)`` pair for every
``(q, mu, nu)``.  :func:`compute_sigma_c_ppm_omega_grid` resolves invalid-pole
policy, writes those fields as a finalized one-pole MPA store, and delegates
all dynamic windows, box rules, cache use, tau execution and accumulation to
``gw.mpa.sigma``.  Only the static-COHSEX completion for invalid PPM poles
remains outside that store because it is frequency independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common import timing
from common.units import RYD_TO_EV
from .gw_config import DynamicSigmaConfig, PPMConfig
from .minimax_screening import (
    GN_PPM_EXTREME_TAIL_DIVISOR,
    fit_gn_ppm_from_wc_pair,
)
from .ppm_windows import (
    branches_for_omega_grid,
    resolve_sigma_regularization,
)
from .ppm_tau_kernel import get_sigma_spatial_kernel
from .wavefunction_bundle import face_kernel_kwargs

if TYPE_CHECKING:
    from .band_extrapolation import BandBracketPlan


def _face_g_plan(mesh_xy: Mesh, face_shape):
    """One ``distrib_la.gemm_plan`` for a face-layout G build, at the shape
    ``greens_function_kernel._build_G_face`` requires — mirrors
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
    B_q: jax.Array            # (nq, μ, μ) PPM amplitude: (R_+ + R_-)/2
    Omega_q: jax.Array        # (nq, μ, μ) PPM pole frequency
    valid_mask_q: jax.Array   # (nq, μ, μ)
    unfulfilled_fraction: float
    n_nodes_static: int
    #: The odd residue D = (R_+ - R_-)/2 of the ordered-orientation fit on a
    #: measured-broken-TR deck.  Conduction branches consume R_+ = B + D;
    #: valence branches consume R_- = B - D.  ``None`` is the incumbent
    #: single-residue model.
    B_odd_q: jax.Array | None = None
    probe_hermiticity_residual: float | None = None
    odd_even_residue_ratio: float | None = None


def _residue_for_space(space: str, B_q, B_odd_q=None):
    """The pole residue one Σ branch evolves: ``R_+`` (cond) or ``R_-`` (val).

    ``Σ_c(E) = Σ_occ ψψ†⊙R_-/(E−ε+Ω) + Σ_emp ψψ†⊙R_+/(E−ε−Ω)`` — the
    conduction (empty) A-space carries the +Ω residue, the valence
    (occupied) A-space the −Ω one (derivation §4, verified against the
    imaginary-axis contour in ``tests/test_gnppm_ordered_orientations.py``).
    With ``B_odd_q=None`` both branches get ``B_q`` and nothing about the
    incumbent path changes, not even the object identity.  A zero-valued
    ``B_odd_q`` (the explicit broken-TR debug A/B arm) likewise gives
    ``R_+ = R_- = B_q`` while retaining the odd-channel observability seam.
    """
    if space not in ("cond", "val"):
        raise ValueError(
            f"_residue_for_space: space must be 'cond' or 'val'; got {space!r}")
    if B_odd_q is None:
        return B_q
    return B_q + B_odd_q if space == "cond" else B_q - B_odd_q


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
    #: Optional single-frequency outer block ``Sigma_c(E_F; k, P, U)``.
    #: It is produced only for the self-consistent fixed-index map.  The
    #: ordinary P x P omega cube above is unchanged, preserving the
    #: iteration-1/one-shot contract.
    sigma_c_fermi_pu_kij: jax.Array | None = None
    #: Optional exact ordered-residue contribution on the SAME omega grid:
    #: ``Sigma_c[B,D] - Sigma_c[B,D=0]``.  GN-PPM carries this beside the
    #: result in ``PPMOutputs`` because of its band-bracket pipeline; MPA has
    #: no bracket tail and carries it here.  ``None`` is the TRS/incumbent
    #: route and preserves its object graph.
    sigma_c_odd_kij: jax.Array | None = None
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
    #: Measured fit-level ``max|D|/max|B|`` for an ordered MPA store.  Kept
    #: here so the shared driver report does not re-read or re-fit poles.
    odd_even_residue_ratio: float | None = None




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
    q_neg_index: np.ndarray | None = None,
    coarsen_extreme_tails: bool = False,
    ordered_orientations: bool = False,
) -> PPMBuildResult:
    """Fit two-point PPM pole parameters from precomputed W(0) and W(probe).

    Model-agnostic over the pole-fit ansatz: the same algebra serves
    both Godby-Needs (purely imaginary ``probe_omega = i·ωp``) and
    Hybertsen-Louie (real ``probe_omega = Ω`` above all transitions).

    ``ordered_orientations=True`` (GN on a measured-broken-TR deck) splits
    the probe into Hermitian and anti-Hermitian halves and returns the odd
    residue ``D`` in ``B_odd_q``.

    All input arrays are flat-q (nq, μ, μ).  Returns PPMBuildResult with
    B_q, Omega_q, valid_mask_q sharded as P(None, 'x', 'y').

    ``n_mu_logical`` (REQUIRED, = ``meta.n_rmu``): logical centroid
    count.  The fitted tensors keep the padded extent, but pad modes are
    born DEAD (Ω = B = 0, valid = False) and the ``unfulfilled``
    fraction counts logical modes only — see ``fit_gn_ppm_from_wc_pair``.
    ``q_neg_index`` is the public symmetry service's canonical full-grid
    involution, passed from the driver rather than rebuilt at this layer.  It
    is required only when ``coarsen_extreme_tails`` is true.

    ``coarsen_extreme_tails=True`` is the fixed user-ruled GN-only policy.  It
    preserves affected ``Wc(0)`` elements but changes their finite poles and
    ``1/z^2`` moments, so it is not strict BGW pole parity.  It is explicit
    here because the same algebra also fits HL poles, whose real-axis
    two-point model must not silently inherit this GN policy.
    """
    import time as _t
    z = complex(probe_omega)
    t0 = _t.perf_counter()

    Wc0_q = W0_q - V_q
    Wci_q = Wprobe_q - V_q
    fit = fit_gn_ppm_from_wc_pair(
         Wc0_q, Wci_q, z, fallback_omega=float(fallback_omega),
         n_mu_logical=int(n_mu_logical),
         q_neg_index=q_neg_index,
         coarsen_extreme_tails=bool(coarsen_extreme_tails),
         ordered_orientations=bool(ordered_orientations),
         print_fn=print_fn if print_fn is not None else print)

    q_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    Omega = jax.lax.with_sharding_constraint(
        jnp.asarray(fit.omega_qmunu), q_shard)
    B = jax.lax.with_sharding_constraint(jnp.asarray(fit.B_qmunu), q_shard)
    B_odd = (None if fit.B_odd_qmunu is None else
             jax.lax.with_sharding_constraint(
                 jnp.asarray(fit.B_odd_qmunu), q_shard))
    valid_mask = jax.lax.with_sharding_constraint(
        jnp.asarray(fit.valid_qmunu), q_shard)
    Wc0_q = jax.lax.with_sharding_constraint(Wc0_q, q_shard)
    t1 = _t.perf_counter()

    probe_hermiticity_residual = None
    odd_even_residue_ratio = None
    if B_odd is not None:
        d_max = float(jax.device_get(jnp.max(jnp.abs(B_odd))))
        b_max = float(jax.device_get(jnp.max(jnp.abs(B))))
        odd_even_residue_ratio = d_max / b_max if b_max > 0.0 else d_max
        probe_scale = float(jax.device_get(jnp.max(jnp.abs(Wprobe_q))))
        probe_anti = float(jax.device_get(jnp.max(jnp.abs(
            Wprobe_q - jnp.conj(jnp.swapaxes(Wprobe_q, -1, -2))))))
        probe_hermiticity_residual = (
            probe_anti / probe_scale if probe_scale > 0.0 else probe_anti)
        if print_fn is not None:
            print_fn(
                f"  {model_label} ORDERED residues (measured-broken-TR deck): "
                f"R± = B ± D; max|D|/max|B| = "
                f"{odd_even_residue_ratio:.3e}; W(iω_p) Hermiticity "
                f"residual = {probe_hermiticity_residual:.3e}.")

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
        if B_odd is not None:
            sanity.check_hermitian(f"{model_label} D_q (odd residue, all q)",
                                   B_odd, rtol=1.0, verbose=True,
                                   print_fn=_pf)

    # ω_p in PPMBuildResult historically meant the imaginary-axis magnitude;
    # carry the probe magnitude there for diagnostics.  Downstream Σ kernels
    # consume only B_q, Omega_q (the *fitted* pole frequency), so the probe
    # magnitude is for logging / restart provenance only.
    probe_mag = float(abs(z))

    if print_fn is not None:
        kind = "iωp" if abs(z.real) < 1.0e-12 else "Ω"
        print_fn(
            f"  {model_label} fit: {t1-t0:.2f}s, {kind}={probe_mag:.4f} Ry, "
            f"unfulfilled={100.0 * fit.unfulfilled_fraction:.2f}%")
        print_fn(
            f"  {model_label} pole census: valid={fit.n_valid}, "
            f"Omega=[{fit.omega_min_raw:.8e}, {fit.omega_max_raw:.8e}] Ry, "
            f"min |Wc(0)-Wc(probe)|/max(|Wc(0)|,|Wc(probe)|)="
            f"{fit.pair_relative_separation_min:.8e}")
        if coarsen_extreme_tails:
            budget = fit.n_valid // GN_PPM_EXTREME_TAIL_DIVISOR
            print_fn(
                "  GN user-ruled fitted-pole tail policy: "
                f"low={fit.n_tail_low}/{budget}, "
                f"high={fit.n_tail_high}/{budget}; "
                f"[{fit.omega_min_raw:.8e}, {fit.omega_max_raw:.8e}] -> "
                f"[{fit.omega_min_after:.8e}, {fit.omega_max_after:.8e}] Ry; "
                f"anchor={fit.tail_anchor_omega:.8e} Ry; boundary key groups "
                "are atomic before orbit closure; physical partner orbits "
                "remain unsplit and closure may further undershoot"
            )
            print_fn(
                "  GN tail semantics: affected Wc(0) is preserved by "
                "B'=-Wc(0)*Omega'/2; the 1/z^2 moment changes; this is not "
                "BGW finite-pole parity; exact panes remain downstream."
            )

    return PPMBuildResult(
        omega_p=probe_mag,
        Wc0_q=Wc0_q,
        B_q=B,
        Omega_q=Omega,
        valid_mask_q=valid_mask,
        unfulfilled_fraction=fit.unfulfilled_fraction,
        n_nodes_static=n_nodes_static,
        B_odd_q=B_odd,
        probe_hermiticity_residual=probe_hermiticity_residual,
        odd_even_residue_ratio=odd_even_residue_ratio,
    )


# ---------------------------------------------------------------------------
#  Sigma convolution — the device-side τ loop.  Its host-side counterpart
#  (window construction) lives in ppm_windows; the two halves share no state
#  beyond the window list itself.
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Sigma band-window mesh padding
# ---------------------------------------------------------------------------

def pad_sigma_window(psi_proj_xr, psi_proj_yn, mesh_xy):
    """Zero-pad the sigma band window: m to a multiple of ``p_x``, n of ``p_y``.

    ``ppm_tau_kernel._make_project_ri_reduce_scatter`` reduce-scatters m over
    ``'x'`` and n over ``'y'``, and the shared device accumulator holds
    Sigma_c(w,k,m,n) at ``P(None, None, 'x', 'y')`` — so BOTH need ``m % p_x == 0`` and
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


def _make_strip_sharded_sigma_window(mesh_xy: Mesh, ndim: int, nb_real: int):
    """The device-array arm of :func:`strip_sigma_window`.

    Ordinary numpy-style trailing-axis indexing (``sigma_kij[..., :nb_real,
    :nb_real]``) is illegal on a LIVE ``P(...,'x','y')``-sharded axis whenever
    the sliced extent would not itself divide the mesh --
    :func:`assert_sharded_sigma_window_divides_mesh` exists to refuse that
    case before this function is ever reached; ``nb_real`` arriving here is
    always mesh-divisible on both trailing axes.

    Given that precondition, this applies EXACTLY the mechanism
    :func:`gw.wavefunction_bundle.pack_band_window` already uses on its own
    ψ pair -- ``jax.lax.slice_in_dim`` on each mesh-sharded axis followed by
    ``jax.lax.with_sharding_constraint`` back onto the canonical
    ``P(...,'x','y')`` spec, run inside a cached ``jax.jit`` -- to Σ_c's OWN
    trailing (m, n) axes instead of ψ's band axis.  Same idiom, different
    tensor; not a second mechanism (2026-08-23, the MPA split-Σ-window fix).
    """
    axis_m, axis_n = ndim - 2, ndim - 1
    spec = P(*((None,) * axis_m), 'x', 'y')

    @jax.jit
    def strip(sigma_kij):
        out = jax.lax.slice_in_dim(sigma_kij, 0, nb_real, axis=axis_m)
        out = jax.lax.slice_in_dim(out, 0, nb_real, axis=axis_n)
        return jax.lax.with_sharding_constraint(
            out, NamedSharding(mesh_xy, spec))
    return strip


def _strip_sharded_sigma_window_kernel(mesh_xy: Mesh, ndim: int, nb_real: int):
    """One compiled repack kernel per ``(mesh, ndim, nb_real)`` -- the same
    ``common.wfn_transforms._cached_jit`` idiom
    ``wavefunction_bundle._pack_band_window_kernel`` uses, so a caller that
    revisits the same (layout, window) does not re-trace."""
    from common.wfn_transforms import _cached_jit
    return _cached_jit(
        'strip_sharded_sigma_window', (id(mesh_xy), ndim, nb_real),
        lambda: _make_strip_sharded_sigma_window(mesh_xy, ndim, nb_real))


def strip_sigma_window(sigma_kij, nb_real: int, *, mesh_xy: Mesh | None = None):
    """Publish the leading logical window of a wider (..., m, n) Sigma.

    Under legacy the wider tail is :func:`pad_sigma_window`'s exact-zero
    mesh pad (bilinear in zero-padded psi).  Under the face carrier it can
    contain real higher-band matrix elements; selecting the leading block
    is still exact because every output ``Sigma[k,m,n]`` is an independent
    contraction.  This is the single seam where either internal carrier
    stops.  No-op when the input already has the logical extent.

    BOTH trailing extents are tested: since ``pad_sigma_window`` pads m and n
    independently, one axis can be at the real extent while the other is
    padded (mesh 8×10, window 70 → m=72, n=70).  Testing only the last axis
    would have returned an m-padded Σ untouched.

    ``mesh_xy`` (2026-08-23): pass this when ``sigma_kij`` is a LIVE
    ``jax.Array`` still carrying a mesh-partitioned ``P(...,'x','y')``
    sharding on its trailing axes -- both dynamic-Sigma face finalizers use
    it today: ``gw.mpa.sigma._integrate_sigma_batches`` directly, and this
    module after its per-rank HOST tiles have been reassembled as a live
    sharded array.  It is the EXPLICIT switch to the mesh-aware repack kernel
    (:func:`_strip_sharded_sigma_window_kernel`), not an implicit sniff of
    ``sigma_kij``'s type: every existing host/numpy caller, and the unit
    tests that build a bare single-device ``jnp.asarray`` cube, never pass
    it and take the ordinary trailing-axis slice below completely
    unchanged.  A genuinely mesh-sharded array arriving with ``mesh_xy =
    None`` refuses loudly instead of silently mis-indexing.
    """
    if sigma_kij is None:
        return sigma_kij
    nb_real = int(nb_real)
    if (int(sigma_kij.shape[-2]) == nb_real
            and int(sigma_kij.shape[-1]) == nb_real):
        return sigma_kij
    # ``mesh_xy`` is the caller's explicit signal that ``sigma_kij`` is a
    # LIVE mesh-sharded jax.Array, not an implicit type sniff: existing
    # host/numpy AND plain single-device jax.Array callers (the unit tests
    # in tests/test_sigma_window_pad.py build bare ``jnp.asarray`` cubes
    # with no mesh at all) never pass it and take the ordinary slice below
    # completely unchanged.
    if mesh_xy is not None:
        kernel = _strip_sharded_sigma_window_kernel(
            mesh_xy, int(sigma_kij.ndim), nb_real)
        return kernel(sigma_kij)
    if isinstance(sigma_kij, jax.Array):
        sharding = getattr(sigma_kij, "sharding", None)
        if (isinstance(sharding, NamedSharding)
                and sharding.spec[-2] is not None
                and sharding.spec[-1] is not None):
            # Defensive backstop for a FUTURE caller that forgets mesh_xy,
            # not a path any current caller reaches: a genuinely
            # mesh-sharded array here would silently mis-shard under the
            # plain slice below (the exact hazard assert_sharded_sigma_
            # window_divides_mesh exists to prevent upstream) rather than
            # loudly refuse.
            raise ValueError(
                "strip_sigma_window: sigma_kij is mesh-sharded "
                f"(spec={tuple(sharding.spec)}) on its trailing axes but "
                "mesh_xy was not given -- ordinary indexing is illegal on "
                "a P(...,'x','y')-sharded axis whose sliced extent would "
                "not itself divide the mesh; pass mesh_xy so the legal "
                "slice+reshard kernel (pack_band_window's own mechanism) "
                "can run.")
    return sigma_kij[..., :nb_real, :nb_real]






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

def _resolve_ppm_band_plan(wfns, meta, plan, *, where):
    """One band-bracket validation seam for both transition executors."""
    from .band_extrapolation import (
        assert_brackets_match_ols_abscissae, trivial_plan)

    s = wfns.slices
    if plan is None:
        plan = trivial_plan(
            int(s.nb_sigma_sum), int(s.b2 - s.b0),
            int(meta.b_id_4_sigma_user or s.b4) - int(s.b0))
    assert_brackets_match_ols_abscissae(
        plan, s, meta=meta, where=where)
    return plan




@jax.jit
def _ppm_as_one_pole_store_fields(
    state: _SigmaPhysicsState,
    B_odd_q=None,
):
    """Policy-resolved PPM poles in the MPA ``(p,q,mu,nu)`` convention.

    Both ansatzes use

    ``Wc(z) = 2*Omega*B/(z**2 - Omega**2)`` and therefore
    ``Wc(0) = -2*B/Omega``.  No residue rescaling or sign flip is needed.
    ``state.B_mask`` has already applied ``ppm_invalid_mode``; absent modes
    are encoded by the MPA store's ordinary zero-residue convention and their
    frequency is zeroed as well.  An ordered GN fit's ``B_odd_q=D`` is masked
    by the same policy and stored beside ``B`` so the shared executor consumes
    ``R_+=B+D`` on conduction rows and ``R_-=B-D`` on valence rows.  GN/HL
    poles retained here are positive real frequencies, i.e. causal MPA poles
    with zero fitted width.
    """
    live = jnp.asarray(state.B_mask, dtype=bool)
    if B_odd_q is not None and tuple(B_odd_q.shape) != tuple(state.B_corr.shape):
        raise ValueError(
            "ordered PPM B_odd_q must match B_q; got "
            f"{tuple(B_odd_q.shape)} and {tuple(state.B_corr.shape)}")
    Omega = jnp.where(live, state.Omega_abs, 0.0).astype(jnp.complex128)
    B = jnp.where(live, state.B_corr, 0.0 + 0.0j).astype(jnp.complex128)
    D = (None if B_odd_q is None else
         jnp.where(live, jnp.asarray(B_odd_q), 0.0 + 0.0j)
         .astype(jnp.complex128)[None, ...])
    return Omega[None, ...], B[None, ...], D


def _add_static_ppm_term(sigma_c_kij, sigma_static_host, mesh_xy):
    """Add one static invalid-pole term to every cumulative band count."""
    from common.collectives import device_put_process_local

    static_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    static = device_put_process_local(
        np.asarray(sigma_static_host, dtype=np.complex128), static_sharding)
    return jax.jit(
        lambda sigma, term: sigma + term[None, None, ...],
        out_shardings=sigma_c_kij.sharding)(sigma_c_kij, static)


def compute_sigma_c_ppm_omega_grid(
    wfns,
    ppm,
    meta,
    mesh_xy: Mesh,
    *,
    ppm_cfg: PPMConfig,
    sigma_cfg: DynamicSigmaConfig,
    mpa_cfg,
    omega_grid_ry: np.ndarray,
    ansatz: str,
    fit_store_path: str,
    screening_diagrams,
    quadrature_cache_dir: str | None = None,
    occupation_state=None,
    plan: 'BandBracketPlan | None' = None,
    include_fermi_pu: bool = False,
    print_fn=print,
) -> SigmaOmegaResult:
    """Compute GN/HL-PPM Sigma_c through the shared MPA dynamic route.

    The PPM fit remains the two-point algebra in :func:`fit_ppm`.  This stage
    applies the established invalid-pole policy, writes that result as a
    finalized one-pole MPA store, and then delegates window construction,
    denominator-box rules, cache lookup, pole batching, the tau executor and
    omega accumulation to :func:`gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`.

    ``plan`` retains the PPM band-convergence contract: disjoint brackets are
    evaluated by the shared spatial kernel and returned as cumulative band
    counts.  The optional static-COHSEX invalid-pole term remains separate
    from the dynamic store and is added exactly once to each cumulative count.
    """
    from .mpa.sigma import compute_sigma_c_mpa_omega_grid

    s = wfns.slices
    plan = _resolve_ppm_band_plan(
        wfns, meta, plan, where="ppm_sigma shared MPA bracket partition")

    enk_full = wfns.enk[:, s.full]
    occ_full = wfns.occ[:, s.full]
    omega_req = np.asarray(omega_grid_ry, dtype=np.float64)
    if omega_req.ndim != 1 or not omega_req.size:
        raise ValueError("omega_grid_ry must be a nonempty vector")
    if int(meta.nk_tot) != int(enk_full.shape[0]):
        raise ValueError(
            "enk_full shape mismatch: expected first dim "
            f"{int(meta.nk_tot)}, got {enk_full.shape[0]}")

    invalid_mode = str(ppm_cfg.invalid_mode).strip().lower()
    if invalid_mode == "imaginary":
        raise NotImplementedError(
            "ppm_invalid_mode='imaginary' (BGW mode 1) needs a "
            "complex-Omega path.")
    if invalid_mode not in (
            "zero", "skip", "2ry", "static_limit", "infinity"):
        raise ValueError(
            "ppm_invalid_mode must be zero/skip/2ry/static_limit/infinity; "
            f"got {invalid_mode!r}")
    keep_invalid = invalid_mode == "2ry"
    invalid_static = invalid_mode in ("static_limit", "infinity")

    assert_gapped_occupations_for_ppm(occ_full, print_fn=print_fn)
    valid_mask = ppm.valid_mask_q
    if valid_mask is None:
        valid_mask = jnp.ones(ppm.Omega_q.shape, dtype=bool)
    with timing.section("sigma.state"):
        state = _prepare_sigma_state(
            enk_full, occ_full, ppm.B_q, ppm.Omega_q, valid_mask,
            jnp.asarray(sigma_cfg.fermi_reference == "midgap", dtype=bool),
            jnp.asarray(keep_invalid, dtype=bool))

    regularization_width_ry = (
        float(sigma_cfg.regularization_ev) / RYD_TO_EV)
    ansatz_name = str(getattr(ansatz, "value", ansatz)).strip().lower()
    resolved_xi = resolve_sigma_regularization(
        requested_ry=regularization_width_ry, ansatz=ansatz_name)
    print_fn(resolved_xi.describe())
    regularization_width_ry = resolved_xi.resolved_ry

    n_total_modes, n_invalid = map(
        int, jax.device_get((state.n_total_modes, state.n_invalid)))
    if n_invalid:
        print_fn(
            f"  PPM invalid modes: {n_invalid}/{n_total_modes} "
            f"({100.0 * n_invalid / max(n_total_modes, 1):.2f}%)")
        n_invalid_q = np.asarray(jax.device_get(
            jnp.sum(state.invalid_mask, axis=(1, 2), dtype=jnp.int64)))
        print_fn(
            "  PPM invalid modes per q: "
            f"min={int(n_invalid_q.min())} max={int(n_invalid_q.max())} "
            f"counts={np.array2string(n_invalid_q, max_line_width=100, threshold=64)}")

    sigma_static_host = None
    static_coh_at_counts = None
    if invalid_static and n_invalid:
        sigma_static_host = _compute_invalid_static_sigma(
            wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy,
            occupation_state=occupation_state)
        print_fn(
            "  PPM invalid modes -> static COHSEX: max|Sigma_static| = "
            f"{float(np.max(np.abs(sigma_static_host))) * RYD_TO_EV:.4f} eV")
        if plan.n_brackets > 1:
            static_coh_at_counts = np.cumsum(
                _invalid_static_coh_by_bracket(
                    wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy,
                    plan.bounds),
                axis=0)

    Omega_p, B_p, B_odd_p = _ppm_as_one_pole_store_fields(
        state, ppm.B_odd_q)
    from file_io.mpa_store import write_complete_pole_store_collective

    diagram_value = str(getattr(
        screening_diagrams, "value", screening_diagrams))
    write_complete_pole_store_collective(
        fit_store_path, Omega_p, B_p,
        B_odd_p=B_odd_p,
        mesh_xy=mesh_xy,
        n_mu_logical=int(meta.n_rmu),
        energy_unit="Ry",
        provenance={
            "fit_protocol": "two_point_ppm",
            "pole_model": ansatz_name,
            "ppm_invalid_mode": invalid_mode,
            "screening_diagrams": diagram_value,
            "certification_basis": "algebraic_no_linear_solve",
            "probe_frequency_ry": float(ppm.omega_p),
            "unfulfilled_fraction": float(ppm.unfulfilled_fraction),
        },
        certification={
            "condition_max_allowed": 1.0,
            "backward_error_max_allowed": 1.0,
        },
        occupation_state=occupation_state,
    )
    print_fn(
        f"  {ansatz_name} fit -> MPA store: one pole per ISDF pair at "
        f"{fit_store_path}; invalid policy={invalid_mode}")

    branches = branches_for_omega_grid(
        omega_req,
        E_cond=state.E_cond,
        H_val=state.H_val,
        cond_mask=state.cond_mask,
        val_mask=state.val_mask)
    result = compute_sigma_c_mpa_omega_grid(
        wfns, fit_store_path, meta, mesh_xy,
        omega_grid_ry=omega_req,
        efermi_ry=float(jax.device_get(state.efermi)),
        regularization_width_ry=regularization_width_ry,
        edge_factor=float(sigma_cfg.window_edge_factor),
        quadrature_eps=float(sigma_cfg.quadrature_eps),
        quadrature_reduction_seconds=float(
            sigma_cfg.quadrature_reduction_seconds),
        quadrature_cache_dir=quadrature_cache_dir,
        omega_grid_step_ry=float(sigma_cfg.omega_step_ev) / RYD_TO_EV,
        pole_batch_size=int(mpa_cfg.pole_batch_size),
        expected_screening_diagrams=screening_diagrams,
        sigma_branches=branches,
        band_brackets=plan.bounds,
        band_counts=plan.counts,
        include_fermi_pu=include_fermi_pu,
        print_fn=print_fn)
    sigma_c_kij = result.sigma_c_kij
    if sigma_static_host is not None:
        sigma_c_kij = _add_static_ppm_term(
            sigma_c_kij, sigma_static_host, mesh_xy)
    return SigmaOmegaResult(
        omega_ry=result.omega_ry,
        omega_ev=result.omega_ev,
        sigma_c_kij=sigma_c_kij,
        sigma_c_fermi_pu_kij=result.sigma_c_fermi_pu_kij,
        sigma_c_odd_kij=result.sigma_c_odd_kij,
        band_counts=result.band_counts,
        static_coh_at_counts=static_coh_at_counts,
        odd_even_residue_ratio=ppm.odd_even_residue_ratio)
