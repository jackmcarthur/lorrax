"""Screening-frequency planner + executor.

Each Σ scheme declares the (ω, role) pairs at which W must be available;
a generic :func:`compute_screening` turns that declaration into a
``{role → W_q}`` dict that the Σ build consumes.  Decouples *which* W's
are needed from *how* Σ is built so adding a new self-energy scheme is
a two-line change:

    1. Extend :func:`screening_requests_for` with the (ω, role) tuples
       the new scheme needs.
    2. Add a dispatch case in :func:`gw.sigma_dispatch.compute_sigma_xc`
       that reads ``W_by_role[role]`` and produces a
       :class:`~gw.sigma_dispatch.SigmaResult`.

No retrofitting of the SC iteration map or main driver is required.

Conventional roles
------------------
``"static"``  W(ω = 0).  Universal — every screened scheme needs it.
``"probe"``   Probe-frequency W for two-point fits:

              * GN-PPM:  ω = i · ω_p   (imag axis)
              * HL-PPM:  Ω = ω_p       (real axis, above all transitions)

Future schemes will introduce their own role labels (e.g.
``"imag_<i>"`` for full-imaginary-axis CD, or ``"real_grid"`` for direct
real-axis Σ on a frequency mesh).  Roles are symbolic — the Σ builder
looks them up by string, so no enum/registry retrofit is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from common import jax_profile
import common.timing as timing
from .gw_config import ComputeMode, env_bool


# ---------------------------------------------------------------------------
# Stage cadence — the screening stage is not allowed to be silent
# ---------------------------------------------------------------------------
# Everything between the end of the ζ-fit and the first ``Started
# sigma[...]`` line used to print NOTHING: the χ₀ build, the Dyson solve
# and — for the dynamic modes — the *entire* probe-ω W ran with no output
# at all, so a healthy run was indistinguishable from a hang (paid for
# three times: AC.2, AF.4c, the 2.5 h silent c2406 screening stage).
#
# The cadence is now the UNIFIED timing infrastructure: every phase below
# is a ``timing.section(..., announce=True, label=...)``, which prints a
# timestamped enter line (so the last line on screen names what the run
# is inside) and an exit line with the wall time — same formatter, same
# rank-0 gate as ``LORRAX_TIMING_TRACE``, and the same node names in the
# end-of-run stage table as before.  ``LoopProgress`` supplies the
# bar/ETA cadence over the W roles themselves, in the same format as the
# ζ chunk loop and the Σ τ sweep.  Print-only: no array is touched and
# every phase body is unchanged.


# ---------------------------------------------------------------------------
# Request type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningRequest:
    """One W evaluation a Σ scheme needs.

    Attributes
    ----------
    omega_ry : complex
        Frequency at which W must be available, in Rydberg.  Pure real
        (``omega_ry.imag == 0``) lands on the real axis;
        pure imaginary on the Matsubara/Wick-rotated axis.  Both branches
        are supported by :func:`compute_screening` via the existing
        ``build_real_quadrature`` / ``build_imag_quadrature`` helpers in
        :mod:`gw.minimax_screening`.
    role : str
        Symbolic label the Σ builder uses to look up this W in the
        screening output dict.  See module docstring for conventions.
    """

    omega_ry: complex
    role: str


# ---------------------------------------------------------------------------
# Per-scheme declarations
# ---------------------------------------------------------------------------

def screening_requests_for(
    mode: ComputeMode,
    config,
) -> list[ScreeningRequest]:
    """Single source of truth for which W's each Σ scheme needs.

    Returns an empty list for unscreened schemes (``X_ONLY``); a single
    static request for COHSEX; static + probe for the PPM schemes.

    EVERY MODE IS NAMED HERE.  This
    function is the first place a compute mode turns into work, so a mode
    reaching it without a branch of its own would get its screening plan
    from whichever ``if`` happened to sit last — which is how a new ansatz
    silently becomes a run of an old one.  MPA returns no independent
    requests here because its named model stage owns the shared
    double-parallel frequency walk.  Returning the PPM pair would be wrong
    in a way no downstream stage could detect.

    To add a new scheme, extend the dispatch here AND add a
    corresponding case to ``compute_sigma_xc`` that reads the W's by
    role label.
    """
    if mode is ComputeMode.X_ONLY:
        return []  # bare exchange — no screening
    static = ScreeningRequest(omega_ry=0.0 + 0.0j, role="static")
    if mode is ComputeMode.COHSEX:
        return [static]
    if mode is ComputeMode.GN_PPM:
        return [static, ScreeningRequest(
            omega_ry=1j * float(config.ppm.omega_p), role="probe")]
    if mode is ComputeMode.HL_PPM:
        return [static, ScreeningRequest(
            omega_ry=complex(float(config.ppm.omega_p), 0.0), role="probe")]
    if mode is ComputeMode.MPA:
        # MPA owns a shared multi-frequency disk walk, not independent W
        # requests.  ``gw.mpa.model.build_mpa_fit`` is called by the driver
        # directly; the gw_config.UNIMPLEMENTED_MODES row that used to gate
        # that path has been deleted.
        return []
    raise ValueError(
        f"screening_requests_for: unknown compute mode {mode!r}")


# ---------------------------------------------------------------------------
# Static W with the IBZ fast path
# ---------------------------------------------------------------------------

# Per-process dedup for the LORRAX_FORCE_FULL_BZ announcement: one tagged
# line per process per run, not one per role / SC iteration.
_FORCE_FULL_BZ_ANNOUNCED = False


def compute_static_w(
    wfns,
    V_q: jax.Array,
    quad,
    *,
    e_ref: float,
    sym,
    centroid_indices,
    config,
    meta,
    mesh_xy,
    role: str = "static",
    force_full_bz: bool = False,
    section: str = "chi0_W",
    fused_probe_chi=None,
    chi0_override: jax.Array | None = None,
    gamma_chi_override: jax.Array | None = None,
    head_channel=None,
):
    """W = (1 − Vχ₀)⁻¹V on the full BZ, solved on the IBZ wedge when legal.

    One cadence for EVERY W role: AOT compile split (chi.compile /
    chi.exec / W.compile / W.exec), ``block_until_ready`` discipline, and
    the χ₀ donation contract are defined here exactly once.  The static
    role runs the IBZ cascade; the probe roles call this same function
    with ``force_full_bz=True`` (see :func:`compute_screening`'s
    frozen-golden note), so the stage rows stay directly comparable and a
    cadence/donation change cannot fork between the two paths.

    IBZ cascade for the Dyson solve: V_q and χ₀_q are sliced to IBZ rows
    before :func:`gw.w_isdf.solve_w` so the per-q Cholesky/LU factor runs
    only on ``n_q_ibz`` blocks; W_q comes out at IBZ shape and is unfolded
    back to the full BZ via the SAME helper V_q uses (same physics — W is
    bilinear in centroids and rotates by centroid double-permute + L-phase
    + TRS conj under sym).  Explicit-full-BZ debug bypass
    (``LORRAX_FORCE_FULL_BZ=1``) matches the V_q gate and is ANNOUNCED
    when it flips this decision; IBZ activation otherwise depends only on
    orbit-closure of the centroid set, resolved and ANNOUNCED once per run
    in ``gw.qgrid_symmetry`` (reached through ``_resolve_ibz_q_list``).
    Until 2026-08-08 this site passed ``verbose=False`` into that helper
    and a non-closed centroid set dropped the Dyson solve from
    ``n_q_ibz`` blocks to ``n_q_full`` in total silence.

    Parameters
    ----------
    wfns
        ``Wavefunctions`` bundle (DFT or current-QP basis).
    V_q : (nq, μ, μ) jax.Array
        Bare Coulomb in flat-q ISDF basis, ``P(None, 'x', 'y')``.
    quad, e_ref
        Minimax quadrature (static, or a single-frequency probe quad from
        ``build_imag_quadrature`` / ``build_real_quadrature``) + energy
        reference from ``minimax_screening.build_static_quadrature``.
    sym, centroid_indices
        Symmetry tables + ISDF centroid set for the IBZ resolve (unused
        when the full-BZ route is taken).
    config, meta, mesh_xy
        Standard driver scaffolding.
    role
        Label used in the announced cadence lines (``W[<role>] ...``).
    force_full_bz
        Skip the IBZ cascade unconditionally (the probe roles' deliberate
        frozen-golden route).  Independent of the ``LORRAX_FORCE_FULL_BZ``
        debug env bypass, which additionally forces the static role.
    section
        Timing/trace node name — ``chi0_W`` (static) or ``chi0_W_probe``,
        so the two roles keep separate rows in the end-of-run stage table.
    fused_probe_chi
        Probe-χ₀ reuse (``ppm_probe_chi_reuse=auto``): a tuple
        ``(tau_full, alpha_static_row, alpha_probe_row)`` from
        ``minimax_screening.refit_imag_alpha_augmented``.  The χ build
        runs the multi-output kernel over ``tau_full`` — ONE τ sweep, two
        accumulators — and the function returns ``(W_q, chi0_probe_q)``
        with ``chi0_probe_q`` the full-BZ probe χ₀ (never IBZ-sliced,
        never donated here).  Row 0 is the static weights zero-padded
        onto the extra nodes, so the static χ is numerically the static
        quadrature.
    chi0_override
        Skip the χ build entirely and use this precomputed full-BZ χ₀
        (the probe role consuming the reused χ).  Mutually exclusive with
        ``fused_probe_chi``.  The override buffer IS donated into the
        Dyson solve, matching the computed-χ contract.

    Returns
    -------
    W_q : (nq_full, μ, μ) jax.Array
        Screened Coulomb on the full BZ, ``P(None, 'x', 'y')``.
        With ``fused_probe_chi``: ``(W_q, chi0_probe_q)``.
    """
    from .w_isdf import (
        compute_chi0,
        precompile_chi0,
        precompile_solve_w,
        solve_w,
    )

    # LORRAX_FORCE_FULL_BZ is a numerics-affecting debug bypass: it
    # reroutes the W Dyson solve (and the V_q / gw_init IBZ writes, which
    # read the same flag) from the IBZ wedge to the full BZ — a route with
    # a documented ~0.1 meV path-dependence in the downstream PPM fit
    # (see compute_screening's docstring).  Doctrine 5: env may grant
    # capability but must not SILENTLY select policy — so engaging it
    # announces loudly at this decision site.  Env state can differ per
    # rank, so the affected process speaks, tagged with its rank id
    # (audit fix/zq 2026-07-28; previously this read was fully silent —
    # verbose=False below suppresses even the IBZ resolve's own print).
    # ONE grammar, shared with the four other readers of this knob
    # (``gw/gw_init.py`` x3, ``gw/v_q_g_flat.py``).  ``bool(int(...))``
    # accepted decimal digits only, which for a knob whose whole purpose is
    # to be typed by hand at a shell prompt meant ``=true``/``=on``/``=yes``
    # raised ``invalid literal for int()`` from inside the W solve, and
    # ``=2`` silently meant "on".  ``env_bool`` also ANNOUNCES an
    # unrecognised token instead of resolving it in either direction
    # silently, which matters here more than anywhere: this is the one
    # numerics-affecting reader of the five.
    _env_force_full_bz = env_bool('LORRAX_FORCE_FULL_BZ', False)
    if _env_force_full_bz and not force_full_bz:
        global _FORCE_FULL_BZ_ANNOUNCED
        if not _FORCE_FULL_BZ_ANNOUNCED:
            _FORCE_FULL_BZ_ANNOUNCED = True
            print(
                f"  [screening rank {jax.process_index()}] "
                f"LORRAX_FORCE_FULL_BZ=1: BYPASSING the IBZ wedge — the "
                f"static-W Dyson solve (and the V_q/gw_init IBZ writes) "
                f"run on the FULL BZ.  Numerics scope: ~0.1 meV "
                f"PPM-fit path-dependence vs the IBZ route; debug only.",
                flush=True)
    use_ibz_w_requested = not (force_full_bz or _env_force_full_bz)
    if use_ibz_w_requested and getattr(sym, 'q_irr_full_idx', None) is not None:
        from .v_q_g_flat import _resolve_ibz_q_list
        (_, q_irr_frac, full_to_irr_idx, full_to_irr_sym,
         sym_perm, L_table, use_ibz_w) = _resolve_ibz_q_list(
            sym=sym, centroid_indices=centroid_indices,
            kgrid=tuple(meta.kgrid),
            fft_grid=tuple(meta.fft_grid),
            context=f"W[{role}] Dyson solve q-grid reduction")
    else:
        use_ibz_w = False

    nq_solve = (int(np.asarray(sym.q_irr_full_idx).shape[0]) if use_ibz_w
                else int(meta.nk_tot))
    _mu = int(meta.n_rmu)
    _w = f"W[{role}]"

    with timing.section(f"gw_jax.{section}"):
        with jax_profile.trace_section(section):
            # Split compile vs exec for χ₀ and W.  Each section's
            # wall time is read off the end-of-run timing report
            # under ``gw_jax.<section>.{chi,W}.{compile,exec}``
            # (section = chi0_W for the static role, chi0_W_probe for
            # the probe roles).  The
            # explicit ``block_until_ready`` inside the exec sections
            # is load-bearing: it (a) pins chi.exec / W.exec wall time
            # to the actual dispatched compute (not just the host
            # dispatch), and (b) drops the last Python reference to
            # χ₀ before the W-solve call so XLA can donate that
            # buffer.  Do NOT use ``_chi_sec.watch(...)`` here — it
            # keeps a bound ``block_until_ready`` method alive on the
            # section object past W-solve, which blocks donation.
            if chi0_override is not None and fused_probe_chi is not None:
                raise ValueError(
                    "compute_static_w: chi0_override and fused_probe_chi "
                    "are mutually exclusive.")
            chi0_extra_q = None
            if chi0_override is not None:
                # Probe-χ₀ reuse consumer: the χ was accumulated inside the
                # static role's τ sweep.  Announced row keeps the stage
                # table honest about where the probe's χ time went.
                with timing.section(
                        "chi.reused", announce=True,
                        label=f"{_w} chi0 REUSED from the static tau sweep "
                              f"(ppm_probe_chi_reuse)"):
                    chi0_q = chi0_override
            elif fused_probe_chi is not None:
                from .w_isdf import compute_chi0_multi, precompile_chi0_multi
                _tau_full, _a_static, _a_probe = fused_probe_chi
                _alpha_rows = np.stack([
                    np.asarray(_a_static, dtype=np.float64),
                    np.asarray(_a_probe, dtype=np.float64)])
                _n_static = int(np.asarray(quad.tau).shape[0])
                _n_full = int(np.asarray(_tau_full).shape[0])
                with timing.section("chi.compile", announce=True,
                                    label=f"{_w} chi0 compile (dual-output)"):
                    precompile_chi0_multi(wfns, _tau_full, _alpha_rows, meta,
                                          mesh_xy, energy_reference=e_ref)
                with timing.section(
                        "chi.exec", announce=True,
                        label=f"{_w} chi0 build "
                              f"({_n_static}+{_n_full - _n_static} tau "
                              f"nodes, {int(meta.nk_tot)} q, mu={_mu}, "
                              f"fused probe accumulator)"):
                    chi0_q, chi0_extra_q = compute_chi0_multi(
                        wfns, _tau_full, _alpha_rows, meta, mesh_xy,
                        energy_reference=e_ref)
                    chi0_q.block_until_ready()
                    chi0_extra_q.block_until_ready()
            else:
                with timing.section("chi.compile", announce=True,
                                    label=f"{_w} chi0 compile"):
                    precompile_chi0(wfns, quad, meta, mesh_xy,
                                    energy_reference=e_ref)
                with timing.section(
                        "chi.exec", announce=True,
                        label=f"{_w} chi0 build "
                              f"({len(np.asarray(quad.tau))} tau nodes, "
                              f"{int(meta.nk_tot)} q, mu={_mu})"):
                    chi0_q = compute_chi0(wfns, quad, meta, mesh_xy,
                                          energy_reference=e_ref)
                    chi0_q.block_until_ready()
            if gamma_chi_override is not None:
                gamma = jnp.asarray(gamma_chi_override)
                expected = (1, int(meta.n_rmu), int(meta.n_rmu))
                if tuple(gamma.shape) != expected:
                    raise ValueError(
                        "Gamma chi override must have shape "
                        f"{expected}, got {gamma.shape}")
                with timing.section(
                        "chi.gamma_static", announce=True,
                        label=f"{_w} exact static fractional Gamma body"):
                    chi0_q = chi0_q.at[0].set(gamma[0])
                    chi0_q.block_until_ready()
            # IBZ slice on V_q and χ₀_q.  Both retain the canonical
            # ``P(None, 'x', 'y')`` sharding; the helper locks it in.
            if use_ibz_w:
                from ffi import _services
                _services.ensure_on_path()
                from symmetry_maps import slice_q_full_to_ibz
                _nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))
                with timing.section(
                        "W.slice_to_ibz", announce=True,
                        label=f"{_w} IBZ slice "
                              f"({int(meta.nk_tot)} q -> {nq_solve} q)"):
                    V_q_solve = slice_q_full_to_ibz(
                        V_q, sym.q_irr_full_idx, out_sharding=_nat)
                    chi0_q_solve = slice_q_full_to_ibz(
                        chi0_q, sym.q_irr_full_idx, out_sharding=_nat)
                    del chi0_q
                    chi0_q_solve.block_until_ready()
            else:
                V_q_solve = V_q
                chi0_q_solve = chi0_q
            with timing.section("W.compile", announce=True,
                                label=f"{_w} Dyson compile"):
                precompile_solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy,
                                   dyson_solver=config.backend.w_dyson_solver,
                                   distrib_la_batched_route=(
                                       getattr(config.backend,
                                               "distrib_la_batched_route",
                                               "auto")))
            if (head_channel is None
                    or str(getattr(head_channel, "mode", "off")) == "off"):
                with timing.section(
                        "W.exec", announce=True,
                        label=f"{_w} Dyson solve ({nq_solve} q, mu={_mu}, "
                              f"{'IBZ wedge' if use_ibz_w else 'full BZ'})"):
                    W_q_solve = solve_w(
                        V_q_solve, chi0_q_solve, meta, mesh_xy,
                        dyson_solver=config.backend.w_dyson_solver,
                        distrib_la_batched_route=(
                            getattr(config.backend,
                                    "distrib_la_batched_route", "auto")))
                    # χ₀ is donated inside solve_w — the reference is
                    # now invalid.  Do NOT touch ``chi0_q_solve`` after this.
                    del chi0_q_solve
                    W_q_solve.block_until_ready()
            else:
                # ── Head-channel Coulomb placement (mc_average_placement) ──
                # TWO single-V Dyson solves, then ONE real scalar per q-cell.
                # ``V_body`` has the q != 0 head channel removed; ``V_bare``
                # has it at the UNAVERAGED v_c.  Their difference IS the head
                # channel with wings (gw/head_channel.py derives the identity
                # against head_wing_schur's Schur reduction), so
                #     W = W_body0 + r (W_bare − W_body0),  r = <v>/v_c
                # places the mini-BZ average on W's head scalar AFTER the
                # solve — BerkeleyGW's placement — while every solve here is
                # still single-V and therefore Hermitian by congruence.
                # Γ is untouched: its head mask is identically zero, so both
                # arms and the scalar are the production objects there.
                from . import head_channel as _hc
                q_idx = (np.asarray(jax.device_get(sym.q_irr_full_idx),
                                    dtype=np.int64)
                         if use_ibz_w else None)
                V_body_solve, V_bare_solve = _hc.build_v_arms(
                    V_q_solve, head_channel, mesh_xy, q_index=q_idx)
                with timing.section(
                        "W.exec", announce=True,
                        label=f"{_w} Dyson solve x2 (head-channel placement "
                              f"'{head_channel.mode}', {nq_solve} q, "
                              f"mu={_mu}, "
                              f"{'IBZ wedge' if use_ibz_w else 'full BZ'})"):
                    # χ₀ is DONATED by solve_w, and we need it twice — so the
                    # body arm gets a copy and the bare arm gets the original.
                    # The copy is the price of the second solve; it is one
                    # (nq, μ, μ) transient, live only across the first LU.
                    W_body0 = solve_w(
                        V_body_solve, chi0_q_solve.copy(), meta, mesh_xy,
                        dyson_solver=config.backend.w_dyson_solver,
                        distrib_la_batched_route=(
                            getattr(config.backend,
                                    "distrib_la_batched_route", "auto")))
                    del V_body_solve
                    W_bare = solve_w(
                        V_bare_solve, chi0_q_solve, meta, mesh_xy,
                        dyson_solver=config.backend.w_dyson_solver,
                        distrib_la_batched_route=(
                            getattr(config.backend,
                                    "distrib_la_batched_route", "auto")))
                    del chi0_q_solve, V_bare_solve
                    W_q_solve = _hc.combine_head_channel(
                        W_body0, W_bare, head_channel, q_index=q_idx)
                    del W_body0, W_bare
                    W_q_solve.block_until_ready()
            # IBZ → full-BZ unfold (centroid double-permute + L-phase
            # + TRS conj) — same helper V_q uses.  Σ_COH/SX still
            # iterate over the full BZ in the k-q sums.
            if use_ibz_w:
                from ffi import _services
                _services.ensure_on_path()
                from symmetry_maps import unfold_isdf_operator
                with timing.section(
                        "W.unfold_to_full_bz", announce=True,
                        label=f"{_w} IBZ -> full-BZ unfold "
                              f"({nq_solve} q -> {int(meta.nk_tot)} q)"):
                    n_sym_spatial = int(
                        np.asarray(sym_perm).shape[0]) // 2
                    # W's PRE-UNFOLD BLOCK, offered to whoever is writing
                    # the restart — same contract, same reason, same
                    # no-op-outside-a-scope as the V site in v_q_g_flat.
                    # This is the array ``persist_w0_and_head`` stores when
                    # the resolution says wedge; slicing the unfolded W
                    # instead would be a different array whose equality to
                    # this one depends on the op-selection policy.
                    from .restart_q_storage import deposit_pre_unfold
                    deposit_pre_unfold(
                        "W0_qmunu", W_q_solve,
                        n_rmu_logical=int(meta.n_rmu),
                        q_irr_frac=q_irr_frac, irr_idx_q=full_to_irr_idx,
                        sym_idx_q=full_to_irr_sym, sym_perm=sym_perm,
                        L_table=L_table, n_sym_spatial=n_sym_spatial)
                    W_q = unfold_isdf_operator(
                        W_q_solve,
                        irr_idx=full_to_irr_idx,
                        sym_idx=full_to_irr_sym,
                        sym_perm=sym_perm, L_table=L_table,
                        q_irr_frac=q_irr_frac,
                        mesh_xy=mesh_xy,
                        n_sym_spatial=n_sym_spatial)
                    del W_q_solve
                    W_q.block_until_ready()
            else:
                W_q = W_q_solve
    if fused_probe_chi is not None:
        return W_q, chi0_extra_q
    return W_q


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def compute_screening(
    wfns,
    V_q: jax.Array,
    requests: list[ScreeningRequest],
    *,
    quad,                     # static minimax quad from build_static_quadrature
    e_ref: float,
    sym,
    centroid_indices,
    config,
    meta,
    mesh_xy,
    print_fn: Callable = print,
    head_channel=None,
    iteration_head_response=None,
) -> dict[str, jax.Array]:
    """Evaluate W at each requested frequency.

    Returns ``{role: W_q}``.  The static role uses the prebuilt minimax
    quadrature ``quad`` and runs through :func:`compute_static_w` — the
    IBZ fast path (slice → per-q Dyson solve on the wedge → unfold).
    Non-static roles build a single-frequency quadrature on the fly
    using the existing :func:`gw.minimax_screening.build_imag_quadrature` /
    :func:`gw.minimax_screening.build_real_quadrature` helpers (chosen by whether
    ``omega_ry`` is on the imag or real axis) and solve on the full BZ
    directly: the nonlinear PPM fit downstream has a documented ~0.1 meV
    q-set path-dependence (see ``test_ibz_full_bz_equivalence``), so the
    probe W stays on the frozen-golden full-BZ path until that is
    re-frozen deliberately.

    The static minimax interval ``[x_min, x_max]`` is reused for both
    branches — both probe-quad builders take the same ``quad`` argument
    as the interval source.

    Caller is responsible for matching roles to its Σ build's
    expectations; an unrequested role lookup is a KeyError.
    """
    from .minimax_screening import (
        build_imag_quadrature,
        build_real_quadrature,
        refit_imag_alpha_augmented,
    )

    from common.progress import LoopProgress

    W_by_role: dict[str, jax.Array] = {}
    # X_ONLY declares no screening requests — return before any cadence
    # print.  A Started/Finished "screening (chi0 -> W)" pair around zero
    # work is an observable that does not discriminate (QUALITY_PATTERNS
    # addendum); the old ``max(1, len(requests))`` clamp printed exactly
    # that.  (audit fix/zq 2026-07-28)
    if not requests:
        return W_by_role
    # ── Probe-χ₀ reuse planning (``ppm_probe_chi_reuse=auto``) ─────────
    # GN probes only (single imag-axis request + a static request): plan
    # the probe χ₀ on the STATIC quadrature's τ nodes plus the minimal
    # augmentation from the dedicated probe quadrature's node set
    # (``refit_imag_alpha_augmented`` — weights-only refits alone plateau
    # ~1e-4 because the probe integrand is the Laplace transform of
    # cos(ωp·t), which the 1/x static grid cannot resolve in the τ tail;
    # measured, job 7885097).  The probe χ₀ then accumulates as a second
    # weighted sum inside ONE fused τ sweep — every shared node's
    # G-build/FFT/contraction tensors are computed once for both
    # frequencies, and only the k extras cost new compute (scorecard BC:
    # the dedicated probe pass duplicated ~40% of the screening stage).
    # Same quadrature-error contract, different bits — hence
    # deck-key-gated, default off (gw_config).
    fused_plan = None
    chi0_probe_reused = None
    _reuse_mode = str(getattr(config.ppm, "probe_chi_reuse", "off"))
    if _reuse_mode == "auto":
        _imag_probes = [
            r for r in requests
            if r.role != "static"
            and abs(complex(r.omega_ry).imag) > 0.0
            and abs(complex(r.omega_ry).real) == 0.0]
        _has_static = any(r.role == "static" for r in requests)
        if len(_imag_probes) == 1 and _has_static:
            _wp = abs(complex(_imag_probes[0].omega_ry).imag)
            _quad_ded = build_imag_quadrature(
                quad, _wp, config.minimax_config, print_fn=print_fn)
            _target = float(config.minimax_config.target_error)
            _gate = max(float(_quad_ded.max_error), _target)
            _tau_full, _a_static, _a_probe, _k, _err = \
                refit_imag_alpha_augmented(
                    quad, _quad_ded, _wp, gate_error=_gate)
            fused_plan = (_tau_full, _a_static, _a_probe)
            _n_s = int(np.asarray(quad.tau).shape[0])
            _n_d = int(np.asarray(_quad_ded.tau).shape[0])
            print_fn(
                f"  probe chi0 reuse (ppm_probe_chi_reuse=auto): fused "
                f"sweep = {_n_s} static + {_k} extra nodes (dedicated "
                f"pass would cost {_n_d}); probe representation err "
                f"{_err:.1e} vs dedicated {_quad_ded.max_error:.1e} "
                f"(gate {_gate:.1e}).")
        elif len(requests) > 1:
            print_fn(
                "  probe chi0 reuse (ppm_probe_chi_reuse=auto): no single "
                "imag-axis probe request (HL/real-axis probes always take "
                "the dedicated path) — reuse inactive.")
    # Bar/ETA cadence over the W roles; every phase inside announces
    # itself via ``timing.section(..., announce=True)`` (see the stage
    # cadence note at the top of this module).
    bar = LoopProgress(
        len(requests), print_fn,
        title="screening (chi0 -> W)", item_name="W role").start()
    for req in requests:
        _w = f"W[{req.role}]"
        if req.role == "static":
            if fused_plan is not None:
                W_static, chi0_probe_reused = compute_static_w(
                    wfns, V_q, quad, e_ref=e_ref,
                    sym=sym, centroid_indices=centroid_indices,
                    config=config, meta=meta, mesh_xy=mesh_xy,
                    role=req.role, fused_probe_chi=fused_plan,
                    head_channel=head_channel,
                    gamma_chi_override=(
                        iteration_head_response.static_chi_body_gamma
                        if iteration_head_response is not None else None))
            else:
                W_static = compute_static_w(
                    wfns, V_q, quad, e_ref=e_ref,
                    sym=sym, centroid_indices=centroid_indices,
                    config=config, meta=meta, mesh_xy=mesh_xy,
                    role=req.role, head_channel=head_channel,
                    gamma_chi_override=(
                        iteration_head_response.static_chi_body_gamma
                        if iteration_head_response is not None else None))
            with timing.section("W.gate", announce=True,
                                label=f"{_w} finiteness + hermiticity gate"):
                _gate_w(W_static, req, print_fn=print_fn,
                        kgrid=tuple(meta.kgrid))
            W_by_role[req.role] = W_static
            bar.step()
            continue
        # Pick imag or real axis by which component of ω is non-zero.
        on_imag = abs(req.omega_ry.imag) > 0.0
        on_real = abs(req.omega_ry.real) > 0.0
        if on_imag and on_real:
            raise ValueError(
                f"compute_screening: complex-axis ω={req.omega_ry!r} "
                f"not supported — ω must be pure real or pure imag.")
        if on_imag and chi0_probe_reused is not None:
            # Probe-χ₀ reuse: the χ was accumulated inside the static
            # role's τ sweep above (requests are static-first by
            # construction).  Same cadence function, χ phase replaced by
            # the announced ``chi.reused`` row; the Dyson solve keeps the
            # frozen-golden full-BZ route unchanged.
            W = compute_static_w(
                wfns, V_q, quad, e_ref=e_ref,
                sym=sym, centroid_indices=centroid_indices,
                config=config, meta=meta, mesh_xy=mesh_xy,
                role=req.role, force_full_bz=True, section="chi0_W_probe",
                chi0_override=chi0_probe_reused,
                head_channel=head_channel)
            chi0_probe_reused = None   # donated into solve_w — dead ref
            with timing.section("W.gate", announce=True,
                                label=f"{_w} finiteness + hermiticity gate"):
                _gate_w(W, req, print_fn=print_fn, kgrid=tuple(meta.kgrid))
            W_by_role[req.role] = W
            bar.step()
            continue
        if on_imag:
            quad_used = build_imag_quadrature(
                quad, abs(req.omega_ry.imag),
                config.minimax_config, print_fn=print_fn)
        else:
            quad_used = build_real_quadrature(
                quad, abs(req.omega_ry.real),
                config.minimax_config, print_fn=print_fn)

        # The probe-ω W runs through the SAME cadence function as the
        # static role (compute_static_w: chi.compile → chi.exec →
        # W.compile → W.exec, with the AOT split, block_until_ready
        # discipline and χ₀ donation defined once), on the FULL BZ
        # (force_full_bz=True — not the IBZ wedge; see the docstring's
        # frozen-golden note) and under its own ``chi0_W_probe`` timing
        # node so the two roles stay separate-but-comparable rows in the
        # stage table.  This replaces a statement-for-statement copy of
        # the static body that had already forced one coordinated
        # two-site edit (the w_dyson_solver rename) — data-movement-only
        # consolidation; acceptance criterion is the 785c bit-gate
        # (run_800c md5 baseline).  (audit fix/zq 2026-07-28)
        W = compute_static_w(
            wfns, V_q, quad_used, e_ref=e_ref,
            sym=sym, centroid_indices=centroid_indices,
            config=config, meta=meta, mesh_xy=mesh_xy,
            role=req.role, force_full_bz=True, section="chi0_W_probe",
            head_channel=head_channel)
        with timing.section("W.gate", announce=True,
                            label=f"{_w} finiteness + hermiticity gate"):
            _gate_w(W, req, print_fn=print_fn)
        W_by_role[req.role] = W
        bar.step()

    bar.finish()
    return W_by_role


def compute_screening_model(
    mode,
    wfns,
    V_q,
    *,
    quad,
    e_ref,
    sym,
    centroid_indices,
    config,
    meta,
    mesh_xy,
    run_dir,
    label,
    wfn=None,
    head_resolver=None,
    head_channel=None,
    mpa_plan=None,
    iteration_head_response=None,
    occupation_state=None,
    static_only=False,
    print_fn=print,
):
    """Build the screening representation consumed by one Sigma ansatz.

    Ordinary modes return their in-memory ``{role: W_q}`` mapping.  MPA
    returns one disk-backed fit path under ``"mpa_fit"``; its shared
    frequency walk cannot be represented as independent screening requests.
    ``static_only`` is the pre-SC restart seed used by ordinary modes; MPA
    rebuilds its model inside each iteration and therefore returns nothing.
    """
    if mode is ComputeMode.MPA:
        if static_only:
            return {}
        if head_resolver is None:
            raise ValueError("MPA screening requires a head resolver")
        from .mpa.model import build_mpa_fit
        fit_path, iteration_head = build_mpa_fit(
            run_dir, label, wfns=wfns, wfn=wfn, V_q=V_q, quad=quad, sym=sym,
            centroid_indices=centroid_indices, head_resolver=head_resolver,
            config=config, meta=meta, mesh_xy=mesh_xy,
            energy_reference=e_ref, plan=mpa_plan,
            iteration_head_response=iteration_head_response,
            occupation_state=occupation_state,
            head_channel=head_channel,
            print_fn=print_fn)
        result = {"mpa_fit": fit_path}
        if iteration_head is not None:
            result["iteration_head"] = iteration_head
        return result

    requests = screening_requests_for(mode, config)
    if static_only:
        requests = [request for request in requests
                    if request.role == "static"]
    return compute_screening(
        wfns, V_q, requests, quad=quad, e_ref=e_ref, sym=sym,
        centroid_indices=centroid_indices, config=config, meta=meta,
        mesh_xy=mesh_xy, print_fn=print_fn, head_channel=head_channel,
        iteration_head_response=iteration_head_response)


def _gate_w(W, req: ScreeningRequest, *, print_fn: Callable = print,
            kgrid=None) -> None:
    """Stage gate on one solved W — the Dyson solve is the fragile seam.

    ``W = (1 − Vχ₀)⁻¹V`` is the only place in the GW flow where a matrix
    *inverse* of a near-singular object is taken at production scale, and
    it is where the distributed back-solve work lives.  A flat-mesh column
    sharding bug in exactly this solve produced NaN with ``rc=0`` and was
    caught only downstream by counting floats in ``eqp0.dat``.  One
    finiteness reduction plus one hermiticity residual here names the
    stage instead.

    ``W`` is Hermitian for the two frequencies LORRAX evaluates (ω = 0 and
    ω = iω_p, both on axes where χ₀ is Hermitian); the real-axis HL probe
    is *not* Hermitian in general, so hermiticity is only asserted on the
    imaginary/zero-frequency branch.

    THE HERMITICITY CHECK IS NOT THE ONE THE BSE NEEDS, and on its own it
    is how this seam stayed green while broken.  ``W[q=0]`` passes
    :func:`check_hermitian` at 1e-15 on every fixture measured while
    ``W_q = conj(W_{−q})`` — the condition the BSE kernel's own hermiticity
    reduces to, equivalently "``W_R`` is real" — fails at 9.1e-4
    (armA_base480, 2026-08-07).  The two are independent: per-q hermiticity
    is neither necessary nor sufficient for the reciprocity.  Worse, the
    old gate's ``q=0`` restriction makes the *right* property untestable
    too, since ``−0 == 0`` collapses it to "``W[0]`` is real", which holds
    at 1.9e-11 regardless.  Both are now checked, and the reciprocity one
    is checked over the whole flat-q axis.
    """
    from common import sanity

    label = f"W[{req.role}]"
    sanity.check_finite(label, W, print_fn=print_fn)
    if abs(complex(req.omega_ry).real) == 0.0:
        # ω on the imaginary axis (including ω=0): W is Hermitian.
        # rtol is generous — this catches structural mixing, not roundoff.
        sanity.check_hermitian(f"{label}[q=0]", W[0], rtol=1e-6,
                               print_fn=print_fn)
        # The load-bearing property, over ALL q.
        #
        # SCOPE.  This sits inside the ω-real == 0 branch deliberately.  On
        # the imaginary axis (ω = 0 and ω = iω_p) χ₀ — and hence W — is real
        # in R-space for a time-reversal-symmetric system, so reciprocity is
        # a true property there.  It is NOT true of the real-axis HL probe:
        # a dynamical W obeys Kramers-Kronig, carries a genuine W'' , and
        # legitimately violates W_q = conj(W_{−q}).  Gating that branch
        # would be checking something false by construction.
        #
        # TOLERANCE, derived from the MEASURED floor rather than from eps.
        # These tiles span |A| ∈ [1.4, 3.8e6] (dyn. range ~50x on the median,
        # max/min ~2.7e6), so an ‖·‖-relative floor is NOT eps: the residual
        # is set by cancellation among large intermediates.  The empirical
        # floor is the orbit-closed IBZ arm, where the unfold builds W_{−q}
        # from W_q by symmetry and reciprocity therefore holds BY
        # CONSTRUCTION — whatever residual it shows is pure arithmetic.
        # MEASURED there (armB_orbit504, 2026-08-07): 1.13e-7 on this exact
        # statistic.  Its magnitude signature confirms round-off — the
        # per-element relative residual FALLS with |A| (4.9e-7 at <p10 to
        # 3.1e-8 at >p99) while the absolute residual rises.
        # 1e-5 is ~90x above that floor and ~400x below the smallest real
        # break measured (7.8e-4), so it can neither cry wolf nor miss one.
        if kgrid is not None:
            sanity.check_q_conjugate_reciprocity(
                f"{label}[all q]", W, kgrid, rtol=1e-5, print_fn=print_fn)


__all__ = [
    "ScreeningRequest",
    "screening_requests_for",
    "compute_static_w",
    "compute_screening",
    "compute_screening_model",
]
