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

import dataclasses

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common import collectives, jax_profile
import common.timing as timing
from . import quadrature_log
from .gw_config import (
    ComputeMode, ScreeningDiagrams, coerce_screening_diagrams,
)


# ---------------------------------------------------------------------------
# Stage cadence — the screening stage is not allowed to be silent
# ---------------------------------------------------------------------------
# Everything between the end of the ζ-fit and the first ``Started
# sigma[...]`` line used to print NOTHING: the χ₀ build, the Dyson solve
# and — for the dynamic modes — the *entire* probe-ω W ran with no output
# at all, so a healthy run was indistinguishable from a hang (paid for
# three times: AC.2, AF.4c, the 2.5 h silent c2406 screening stage).
#
# The cadence uses the unified timing infrastructure. Potentially long
# compile/build/solve phases announce entry and exit; cheap slices, unfolds
# and gates are measured in the final tree without printing two lines on every
# self-consistency iteration. ``LoopProgress`` supplies cadence over the W
# roles themselves. Print-only: no array is touched.


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
    gamma_chi_override: jax.Array | None = None,
    head_channel=None,
    ordered_orientations: bool = False,
):
    """W = (1 − Vχ₀)⁻¹V, solved and KEPT on the IBZ wedge when legal.

    ``ordered_orientations`` (probe roles on a measured-broken-TR deck):
    build χ₀ through :func:`gw.w_isdf.compute_chi0_imag_ordered`, which
    keeps the time-reversal-odd channel the even Laplace completion deletes.
    Needs ``quad`` built with ``with_odd_kernel=True`` and the full BZ
    (``force_full_bz``); ``False`` is the incumbent path, bit-for-bit.

    One cadence for EVERY W role: AOT compile split (chi.compile /
    chi.exec / W.compile / W.exec), ``block_until_ready`` discipline, and
    the χ₀ donation contract are defined here exactly once.  The static
    role runs the IBZ cascade; the probe roles call this same function
    with ``force_full_bz=True`` (see :func:`compute_screening`'s
    frozen-golden note), so the stage rows stay directly comparable and a
    cadence/donation change cannot fork between the two paths.

    IBZ cascade for the Dyson solve: V_q and χ₀_q are sliced to IBZ rows
    before :func:`gw.w_isdf.solve_w` so the per-q Cholesky/LU factor runs
    only on ``n_q_ibz`` blocks; W_q comes out at IBZ shape and STAYS there,
    returned as a ``symmetry_maps.QirrOperator`` carrying the unfold tables
    the SAME helper V_q uses (W is bilinear in centroids and rotates by
    centroid double-permute + L-phase + TRS conj under sym).  The Σ
    convolutions unfold it on their transform's load
    (``ffi.fft.make_kfft_klead_unfold``); a consumer that needs the full
    zone calls ``.unfold(mesh_xy)``.  IBZ activation depends on
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
        frozen-golden route).
    section
        Timing/trace node name — ``chi0_W`` (static) or ``chi0_W_probe``,
        so the two roles keep separate rows in the end-of-run stage table.

    Returns
    -------
    W_q : symmetry_maps.QirrOperator
        Screened Coulomb on the IBZ wedge (``values`` ``(n_q_ibz, μ, μ)`` at
        ``P(None, 'x', 'y')``), or on the whole zone with trivial tables
        when the q grid does not reduce (``force_full_bz``, an open
        centroid set).
    """
    from .w_isdf import (
        compute_chi0,
        precompile_chi0,
        precompile_solve_w,
        solve_w,
    )

    # The run's q wedge (the one bare V is held on), and the measured-TRS
    # policy that chose its unfold rows.  TIME REVERSAL IS MEASURED, NEVER
    # ASSUMED — one policy object, shared with bare V and ladder W, and no
    # TRS branch at this site (``gw.qgrid_symmetry.qgrid_trs_policy_for``).
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import QirrOperator
    from .v_q_g_flat import q_wedge
    wedge = (None if force_full_bz else q_wedge(
        sym=sym, centroid_indices=centroid_indices, meta=meta,
        context=f"W[{role}] Dyson solve q-grid reduction"))
    use_ibz_w = wedge is not None

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
            if ordered_orientations:
                # ORDERED ORIENTATIONS (measured-broken-TR deck, imaginary
                # probe): the kernel's own orientation and its q-negated
                # conjugate partner each get their own resolvent weight, so
                # χ₀(iω_p) keeps its anti-Hermitian, magnetisation-odd
                # channel.  Full BZ only: the q-negation involution is a
                # full-grid statement.
                from .w_isdf import (
                    compute_chi0_imag_ordered, precompile_chi0_imag_ordered)
                if use_ibz_w:
                    raise RuntimeError(
                        "GATE chi0_imag_ordered_full_bz: ordered response "
                        "received an IBZ q cascade.\n"
                        f"  got:  use_ibz_w = true, role = {role!r}\n"
                        "  want: force_full_bz = true for this role\n"
                        "  why:  the ordered kernel pairs each q with its "
                        "explicit -q row; that involution is not present "
                        "on the irreducible q axis")
                from ffi import _services
                _services.ensure_on_path()
                from symmetry_maps import q_negation_index
                _q_neg = q_negation_index(tuple(int(v) for v in meta.kgrid))
                with timing.section(
                        "chi.compile", announce=True,
                        label=f"{_w} chi0 compile (ordered orientations)"):
                    precompile_chi0_imag_ordered(
                        wfns, quad, meta, mesh_xy, energy_reference=e_ref)
                with timing.section(
                        "chi.exec", announce=True,
                        label=f"{_w} chi0 build ORDERED ORIENTATIONS "
                              f"(TR-odd channel kept; "
                              f"{len(np.asarray(quad.tau))} tau nodes = "
                              f"{len(np.asarray(quad.tau)) - int(quad.n_odd_extra)}"
                              f" even + {int(quad.n_odd_extra)} odd, "
                              f"{int(meta.nk_tot)} q, mu={_mu})"):
                    chi0_q = compute_chi0_imag_ordered(
                        wfns, quad, meta, mesh_xy, q_neg_index=_q_neg,
                        energy_reference=e_ref)
                    chi0_q.block_until_ready()
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
                # The fractional-Gamma producer returns the same canonical
                # product-padded carrier as chi0_q.  ``meta.n_rmu`` is the
                # logical prefix and is deliberately smaller on geometries
                # such as P=36 Bi (2070 -> 2088), so validating against it
                # rejects the correct producer and would make the following
                # sharded update shape-incompatible.
                expected = (1,) + tuple(int(n) for n in chi0_q.shape[-2:])
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
            # V is held on the wedge already (its representative rows);
            # χ₀ is sliced to them.  Both keep ``P(None, 'x', 'y')``.
            if use_ibz_w:
                from symmetry_maps import slice_q_full_to_ibz
                _nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))
                with timing.section("W.slice_to_ibz"):
                    V_q_solve = jax.device_put(
                        QirrOperator.of(V_q).restrict(
                            QirrOperator(values=None, **wedge[0])).values, _nat)
                    chi0_q_solve = slice_q_full_to_ibz(
                        chi0_q, sym.q_irr_full_idx, out_sharding=_nat)
                    del chi0_q
                    chi0_q_solve.block_until_ready()
            else:
                # The full-zone solve (probe roles, an unreduced q grid)
                # reads V at every q, transient.
                V_q_solve = QirrOperator.of(V_q).unfold(mesh_xy)
                chi0_q_solve = chi0_q
            with timing.section("W.compile", announce=True,
                                label=f"{_w} Dyson compile"):
                precompile_solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy,
                                   dyson_solver=config.backend.w_dyson_solver,
                                   distrib_la_batched_route=(
                                       getattr(config.backend,
                                               "distrib_la_batched_route",
                                               "batch_reshard")))
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
                                    "distrib_la_batched_route", "batch_reshard")))
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
                                    "distrib_la_batched_route", "batch_reshard")))
                    del V_body_solve
                    W_bare = solve_w(
                        V_bare_solve, chi0_q_solve, meta, mesh_xy,
                        dyson_solver=config.backend.w_dyson_solver,
                        distrib_la_batched_route=(
                            getattr(config.backend,
                                    "distrib_la_batched_route", "batch_reshard")))
                    del chi0_q_solve, V_bare_solve
                    W_q_solve = _hc.combine_head_channel(
                        W_body0, W_bare, head_channel, q_index=q_idx)
                    del W_body0, W_bare
                    W_q_solve.block_until_ready()
            # The wedge W and the tables of its IBZ → full-BZ unfold
            # (centroid double-permute + L-phase + TRS conj), the same
            # wedge V is held on; nothing is unfolded here.
            if use_ibz_w:
                tables, policy = wedge
                q_irr_frac, full_to_irr_idx = tables["q_irr_frac"], tables["irr_idx"]
                sym_perm, L_table = tables["sym_perm"], tables["L_table"]
                unfold_sym, n_sym_spatial = tables["sym_idx"], tables["n_sym_spatial"]
                with timing.section("W.wedge_tables"):
                    cov = policy.measure_covariance(
                        W_q_solve, q_irr_frac=q_irr_frac,
                        q_irr_full_idx=sym.q_irr_full_idx,
                        sym_mats_k=sym.sym_mats_k, sym_perm=sym_perm,
                        L_table=L_table)
                    W_q_solve, removed = policy.project_fixed_q(
                        W_q_solve, sym.q_irr_full_idx)
                    if jax.process_index() == 0:
                        from common import sanity
                        sanity.report_parent_covariance(
                            f"W[{_w}] IBZ parents", cov, removed=removed)
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
                        sym_idx_q=unfold_sym, sym_perm=sym_perm,
                        L_table=L_table, n_sym_spatial=n_sym_spatial,
                        mu_basis=getattr(meta, 'mu_basis', None))
                    W_q = QirrOperator(values=W_q_solve, **tables)
                    del W_q_solve
            else:
                W_q = QirrOperator.whole_zone(W_q_solve)
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
    using the existing :func:`gw.minimax_screening.build_imag_probe_response_rule` (GN probe) /
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

    LIVE-SET BOUND ACROSS ROLES.  Each role's χ₀ build (the
    ``minimax_tau_integrate_chi`` τ-scan in ``gw.w_isdf``) needs an
    ``O(nq·μ²/P)`` scratch arena that is legitimately unchunked over q —
    the flat-k FFT it runs needs the whole q/k axis local on every rank —
    and is therefore the same size for every role regardless of τ-node
    count (a `lax.scan`'s compiled buffer graph is trip-count-independent).
    An EARLIER role's completed W must not still be a live on-device array
    while a LATER role pays that cost: measured on the production 9x9x1
    deck, the static role's chi0 build/Dyson solve completed and the
    probe role's IDENTICAL build then OOM'd with the static role's W still
    resident (KNOWN_LORRAX_ISSUES.md, "GN-PPM probe chi0 has no bounded
    two-role live-set plan at 81 q", 2026-08-20; exact repeated request
    27,262,284,032 B on two independent jobs).  So every role's W but the
    LAST is spilled to host RAM (:func:`common.collectives.spill_to_host`)
    the moment its own gate passes, and restored
    (:func:`~common.collectives.restore_from_host`) only after every
    requested role has been built — a role-serialized schedule, not a
    q-chunked χ₀ kernel (which the FFT constraint above rules out short of
    redesigning the flat-k FFT service).  Bit-exact: a host round trip
    moves bits, it does not touch them.  A single-role scheme (e.g.
    COHSEX) never spills — the loop only ever holds ONE role at a time
    regardless.
    """
    from .minimax_screening import (
        build_imag_probe_response_rule,
        build_real_quadrature,
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
    # ORDERED ORIENTATIONS on the imaginary-axis probe: exactly when the
    # deck's MEASURED time-reversal verdict is false. Missing symmetry tables
    # refuse in ``_trs_verdict`` rather than selecting a branch. See
    # ``docs/dev/notes/DERIVATION_gnppm_nonhermitian.md``.
    _tr_odd = _trs_verdict(sym) is False
    # Bar/ETA cadence over the W roles; long phases inside also announce via
    # ``timing.section(..., announce=True)`` (see the note at module top).
    bar = LoopProgress(
        len(requests), print_fn,
        title="screening (chi0 -> W)", item_name="W role").start()
    n_requests = len(requests)

    def _store_role(role: str, idx: int, W: jax.Array) -> None:
        """Bound the cross-role live set (see the docstring's LIVE-SET
        BOUND note): every role's gated W but the LAST is immediately
        spilled to host RAM, freeing its device buffer before the NEXT
        role's chi0/W build can compete with it for HBM.  Restored
        uniformly, after the whole loop, right before this function
        returns."""
        if idx == n_requests - 1:
            W_by_role[role] = W
        else:
            W_by_role[role] = dataclasses.replace(
                W, values=collectives.spill_to_host(W.values))

    for idx, req in enumerate(requests):
        _w = f"W[{req.role}]"
        if req.role == "static":
            W_static = compute_static_w(
                wfns, V_q, quad, e_ref=e_ref,
                sym=sym, centroid_indices=centroid_indices,
                config=config, meta=meta, mesh_xy=mesh_xy,
                role=req.role, head_channel=head_channel,
                gamma_chi_override=(
                    iteration_head_response.static_chi_body_gamma
                    if iteration_head_response is not None else None))
            with timing.section("W.gate"):
                # The reciprocity half of the gate reads W at −q: the
                # full zone, transient, for this check only.
                _gate_w(W_static.unfold(mesh_xy), req, print_fn=print_fn,
                        kgrid=tuple(meta.kgrid),
                        trs_allowed=_trs_verdict(sym))
            _store_role(req.role, idx, W_static)
            bar.step()
            continue
        # Pick imag or real axis by which component of ω is non-zero.
        on_imag = abs(req.omega_ry.imag) > 0.0
        on_real = abs(req.omega_ry.real) > 0.0
        if on_imag and on_real:
            raise ValueError(
                f"compute_screening: complex-axis ω={req.omega_ry!r} "
                f"not supported — ω must be pure real or pure imag.")
        if on_imag:
            # GN-PPM probe: times placed by the analytic response rule the
            # shared-pole bank uses (owner 2026-09-22), not a runtime minimax.
            quad_used = build_imag_probe_response_rule(
                quad, abs(req.omega_ry.imag),
                config.minimax_config, print_fn=print_fn,
                with_odd_kernel=_tr_odd)
            if _tr_odd:
                print_fn(
                    f"  {_w}: measured time-reversal verdict is BROKEN — "
                    "χ₀(iω_p) is built with ORDERED particle-hole "
                    "orientations (TR-odd anti-Hermitian channel kept; "
                    "w_isdf.compute_chi0_imag_ordered).  W(iω_p) is then "
                    "legitimately non-Hermitian; its residual is reported "
                    "by the gate, not refused.")
        else:
            quad_used = build_real_quadrature(
                quad, abs(req.omega_ry.real),
                config.minimax_config, print_fn=print_fn)
            if _tr_odd:
                print_fn(
                    f"  {_w}: measured time-reversal verdict is BROKEN, "
                    "but a REAL-axis probe cannot carry the TR-odd "
                    "residue (W^c(z)^H = W^c(conj z) is Hermitian at real "
                    "z), so this HL probe keeps the incumbent even "
                    "orientation completion; the odd channel of χ₀(Ω) is "
                    "NOT represented here (KNOWN_LORRAX_ISSUES, lane M).")
        quadrature_log.record_minimax(
            "imag" if on_imag else "real", quad_used,
            omega_ry=abs(req.omega_ry.imag if on_imag else req.omega_ry.real),
            target=config.minimax_config.target_error)

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
            head_channel=head_channel,
            ordered_orientations=bool(_tr_odd and on_imag))
        with timing.section("W.gate"):
            _gate_w(W.unfold(mesh_xy), req, print_fn=print_fn,
                    trs_allowed=_trs_verdict(sym))
        _store_role(req.role, idx, W)
        bar.step()

    bar.finish()
    # Restore anything spilled during the loop.  Every entry is spilled at
    # most once (the last role's W never was), so this is at most
    # ``n_requests - 1`` host round trips — negligible beside the χ₀/W
    # compute they guard, and zero for a single-role scheme.
    with timing.section("W.restore_spilled_roles"):
        for role, val in W_by_role.items():
            if isinstance(val.values, collectives.HostSpill):
                W_by_role[role] = dataclasses.replace(
                    val, values=collectives.restore_from_host(val.values))
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
    wfn_fingerprint_binding=None,
    charge_zeta_identity=None,
    wfns_transverse=None,
    bispinor_v_q_path=None,
    mu_bases=None,
    photon_static_reference=None,
    photon_g0_vectors=None,
    photon_head_cache=None,
    photon_head_rotation=None,
    head_resolver=None,
    head_channel=None,
    mpa_plan=None,
    iteration_head_response=None,
    occupation_state=None,
    material_class,
    static_only=False,
    tensors_filename=None,
    print_fn=print,
):
    """Build the screening representation consumed by one Sigma ansatz.

    Ordinary modes return their in-memory ``{role: W_q}`` mapping.  MPA
    returns one disk-backed fit path under ``"mpa_fit"``; its shared
    frequency walk cannot be represented as independent screening requests.
    ``static_only`` is an explicit request for just the ordinary static role;
    MPA has no independent static-role representation and returns nothing.

    THE ``screening_diagrams`` FORK LIVES HERE AND NOWHERE ELSE.  The role
    plan (:func:`screening_requests_for`) and the Σ dispatch are identical
    for every value — the same Σ ansatz wants W at the same frequencies
    either way; only WHICH W satisfies the request changes.  ``w_rpa``
    therefore reaches exactly the statements it reached before the fork
    existed, which is what keeps the frozen Si pins and the 0.644 meV BGW
    cell bit-identical.

    ``w_bse`` and ``w_rpa_resolvent`` reach the SAME stage helper
    (:func:`gw.screening_bse.compute_screening_ladder`) with ``include_w``
    set to whether the ladder's direct rung is in the operator — one
    matvec builder with a kernel switch (``bse.bse_ring_comm.
    build_bse_ring_matvec_full``'s own ``include_W``), not a second
    facade.

    ``tensors_filename`` is the ISDF restart file.  It is unused by the
    ``w_rpa`` path and REQUIRED by ``w_bse``, which persists the RPA W(0)
    into it and then hands the path to the BSE ladder facade; a caller
    that cannot supply it gets a named refusal rather than a fallback.
    """
    diagrams = coerce_screening_diagrams(
        getattr(config.screening, "diagrams", ScreeningDiagrams.W_RPA))
    trs_allowed = _trs_verdict(sym)
    if diagrams is ScreeningDiagrams.W_BSE and not trs_allowed:
        raise RuntimeError(
            "GATE w_bse_requires_measured_trs: screening_diagrams = w_bse "
            "uses a time-reversal pair gauge for the ladder's anti-resonant "
            "channel, but SymMaps.trs_allowed is false.  The measured DFT "
            "reference therefore does not license that construction; use "
            "screening_diagrams = w_rpa (or w_rpa_resolvent) until the "
            "ladder owns a general ordered-response construction.")
    if mode is ComputeMode.MPA:
        if static_only:
            return {}
        if config.sigma.w_model == "shared_pole":
            from .shared_pole_screening import screen_shared_poles
            return screen_shared_poles(
                wfns, V_q, meta, config, mesh_xy=mesh_xy, sym=sym,
                centroid_indices=centroid_indices, run_dir=run_dir, label=label,
                wfn=wfn, wfn_fingerprint_binding=wfn_fingerprint_binding,
                tensors_filename=tensors_filename, occupation_state=occupation_state,
                print_fn=print_fn, head_resolver=head_resolver, mpa_plan=mpa_plan,
                iteration_head_response=iteration_head_response,
                material_class=material_class, wfns_transverse=wfns_transverse,
                photon_static_reference=photon_static_reference,
                bispinor_v_q_path=bispinor_v_q_path, mu_bases=mu_bases,
                photon_g0_vectors=photon_g0_vectors,
                photon_head_cache=photon_head_cache,
                photon_head_rotation=photon_head_rotation)
        reuse_path = getattr(config.mpa, "fit_reuse_file", None)
        if reuse_path is not None:
            if mpa_plan is None:
                if quad is None:
                    raise ValueError(
                        "mpa_fit_reuse_file needs either the live MPA plan "
                        "or the current screening quadrature")
                from .mpa.model import make_mpa_plan
                mpa_plan = make_mpa_plan(
                    config, quad, material_class=material_class)
            from .mpa.model import validate_reused_mpa_fit
            fit_path = validate_reused_mpa_fit(
                reuse_path, config=config, live_plan=mpa_plan, sym=sym,
                centroid_indices=centroid_indices, meta=meta,
                mesh_xy=mesh_xy, wfn=wfn,
                wfn_fingerprint_binding=wfn_fingerprint_binding,
                charge_zeta_identity=charge_zeta_identity,
                charge_zeta_source_path=tensors_filename,
                occupation_state=occupation_state,
                material_class=material_class, print_fn=print_fn)
            return {"mpa_fit": fit_path, "mpa_fit_reused": True}
        if quad is None:
            raise ValueError(
                "MPA screening has no quadrature and no certified-fit reuse "
                "provider served an existing fit. A normal screening build "
                "requires quad; a reuse harness must return its already-"
                "resolved path as {'mpa_fit': path}.")
        if head_resolver is None:
            raise ValueError("MPA screening requires a head resolver")
        from .mpa.model import build_mpa_fit
        # ``wc_source = None`` is the RPA seam default (``mpa.model._solve_wc``,
        # the Dyson solve of the sampled chi).  Under w_bse the same per-z,
        # per-wedge-q Wc(z) slabs come off the ladder resolvent instead; the
        # sample plan, the Pade fit and the SlabIO store lifecycle are the
        # same objects either way.
        wc_source = None
        if diagrams is ScreeningDiagrams.W_BSE:
            from .screening_bse import make_ladder_wc_source
            wc_source = make_ladder_wc_source(
                wfns, V_q, quad=quad, e_ref=e_ref, sym=sym,
                centroid_indices=centroid_indices, config=config, meta=meta,
                mesh_xy=mesh_xy, tensors_filename=tensors_filename,
                head_resolver=head_resolver, head_channel=head_channel,
                print_fn=print_fn)
        fit_path, iteration_head = build_mpa_fit(
            run_dir, label, wfns=wfns, wfn=wfn, V_q=V_q, quad=quad, sym=sym,
            wfn_fingerprint_binding=wfn_fingerprint_binding,
            charge_zeta_identity=charge_zeta_identity,
            charge_zeta_source_path=tensors_filename,
            centroid_indices=centroid_indices, head_resolver=head_resolver,
            config=config, meta=meta, mesh_xy=mesh_xy,
            energy_reference=e_ref, plan=mpa_plan,
            iteration_head_response=iteration_head_response,
            occupation_state=occupation_state,
            material_class=material_class,
            head_channel=head_channel,
            wc_source=wc_source, print_fn=print_fn)
        result = {"mpa_fit": fit_path}
        if iteration_head is not None:
            result["iteration_head"] = iteration_head
        return result

    if diagrams in (ScreeningDiagrams.W_BSE, ScreeningDiagrams.W_RPA_RESOLVENT):
        from .screening_bse import compute_screening_ladder
        return compute_screening_ladder(
            mode, wfns, V_q, quad=quad, e_ref=e_ref, sym=sym,
            centroid_indices=centroid_indices, config=config, meta=meta,
            mesh_xy=mesh_xy, tensors_filename=tensors_filename,
            head_resolver=head_resolver, head_channel=head_channel,
            static_only=static_only, print_fn=print_fn,
            include_w=(diagrams is ScreeningDiagrams.W_BSE))
    if diagrams is not ScreeningDiagrams.W_RPA:
        # EXHAUSTIVENESS, not defensive programming.  A member added to
        # ScreeningDiagrams without a branch here would otherwise fall into
        # the RPA path below and produce a complete, plausible W under
        # another diagram set's name — the failure ``compute_mode``'s
        # dispatch audit (16e0c9c0) closed on its own axis.
        raise ValueError(
            f"compute_screening_model: screening_diagrams = "
            f"{diagrams.value!r} has no branch here.  Every member of "
            f"gw_config.ScreeningDiagrams needs one "
            f"({', '.join(d.value for d in ScreeningDiagrams)}); it is "
            f"refused rather than served the RPA path under its name.")

    requests = screening_requests_for(mode, config)
    if static_only:
        requests = [request for request in requests
                    if request.role == "static"]
    return compute_screening(
        wfns, V_q, requests, quad=quad, e_ref=e_ref, sym=sym,
        centroid_indices=centroid_indices, config=config, meta=meta,
        mesh_xy=mesh_xy, print_fn=print_fn, head_channel=head_channel,
        iteration_head_response=iteration_head_response)


def driver_persists_w0(mode, config) -> bool:
    """Does the DRIVER own the W0 restart flush for this run?

    Two runs answer no, for opposite reasons, and the driver should not
    have to know either:

    * ``compute_mode = mpa`` — there is no ``{0, probe}`` head grid to
      stamp beside a W sampled on the double-parallel plan, so
      ``gw_output.persist_w0_and_head`` refuses that shape by name
      (gw_output.py, the ``is_dynamic``-without-``ppm_model`` branch) and
      the driver has never called it.
    * ``screening_diagrams = w_bse`` or ``w_rpa_resolvent`` — the stage
      helper has ALREADY persisted, because the RPA W(0) it wrote is the
      restart-handoff input the ladder facade reads back (both arms run
      the identical ``prepare_ladder_restart``).  ``W_by_role["static"]``
      at this point is the RESOLVENT'S W — the ladder W under ``w_bse``,
      the resolvent-RPA W under ``w_rpa_resolvent`` — and re-persisting
      either would stamp it into ``W0_qmunu``, where every downstream
      consumer (BSE, downfold, a later restart) reads that dataset as the
      DYSON-route RPA static screened Coulomb.  Silently writing a
      different operator (or even a value-close but distinctly-computed
      one) under an established dataset's name is the provenance failure
      class (QUALITY_PATTERNS #10), so the answer is "already done", not
      "do it again".

    Lives here rather than in ``gw_jax`` so the driver keeps one call and
    no mode/diagram arithmetic, and so the two reasons sit next to the
    fork that creates the second one.
    """
    if mode is ComputeMode.MPA:
        return False
    diagrams = coerce_screening_diagrams(
        getattr(config.screening, "diagrams", ScreeningDiagrams.W_RPA))
    return diagrams is ScreeningDiagrams.W_RPA


#: Why a non-Hermitian ``W(iω_p)`` is the answer rather than the bug, on a
#: deck whose measured time-reversal verdict is false.  Passed to
#: ``sanity.report_hermitian_residual``'s ``cause``, which is the one seam
#: for "this residual is a quantity, not a violation" — see ``_gate_w``.
_TR_ODD_W_HERMITICITY_CAUSE = (
    "TR-odd anti-Hermitian part of W(iω_p).  Hermiticity of W on the "
    "imaginary axis away from ω = 0 is a time-reversal consequence, not a "
    "construction invariant: χ₀(r,r';iω) is real for any system once both "
    "particle-hole orientations are summed, but its real antisymmetric "
    "part 2ω Σ Im[conj(ρ_vc(r)) ρ_vc(r')]/(ω²+Δ²) cancels only when the "
    "reverse transition set is the conjugate of the forward one.  This "
    "deck's measured verdict is trs_allowed = False, so the residual above "
    "is that component — odd in the magnetisation, zero at ω = 0 — and NOT "
    "an index/shard mixing fault.  Judge it against the same run's ω = 0 "
    "role, which is gated unconditionally."
)


def _trs_verdict(sym) -> bool:
    """Return the deck's required MEASURED time-reversal verdict.

    ``SymMaps.trs_allowed`` comes from ``WfnLoader.trs_holds``, which
    ``density_symmetry_check`` obtained from the occupied DFT subspaces —
    the same single measurement ``gw/qgrid_symmetry.qgrid_trs_policy_for``
    consumes, and read the same way. Missing symmetry tables are not a
    physical answer and no longer select an incumbent TRS branch.
    """
    if sym is None or not hasattr(sym, "trs_allowed"):
        raise ValueError(
            "GATE screening_needs_measured_trs: screening requires "
            "SymMaps.trs_allowed; no measured verdict was supplied.")
    return bool(sym.trs_allowed)


def _gate_w(W, req: ScreeningRequest, *, print_fn: Callable = print,
            kgrid=None, trs_allowed: bool) -> None:
    """Stage gate on one solved W — the Dyson solve is the fragile seam.

    ``W = (1 − Vχ₀)⁻¹V`` is the only place in the GW flow where a matrix
    *inverse* of a near-singular object is taken at production scale, and
    it is where the distributed back-solve work lives.  A flat-mesh column
    sharding bug in exactly this solve produced NaN with ``rc=0`` and was
    caught only downstream by counting floats in ``eqp0.dat``.  One
    finiteness reduction plus one hermiticity residual here names the
    stage instead.

    ``W`` is Hermitian at ω = 0 for any system; the real-axis HL probe is
    *not* Hermitian in general, so hermiticity is never asserted on that
    branch.  At ω = iω_p it is a TIME-REVERSAL consequence rather than a
    construction invariant — see the SCOPE note at the check itself, and
    ``trs_allowed``, which carries this deck's measured verdict.

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

    if not isinstance(trs_allowed, (bool, np.bool_)):
        raise ValueError(
            "GATE W_gate_needs_measured_trs: _gate_w requires the boolean "
            "SymMaps.trs_allowed verdict; got "
            f"{trs_allowed!r}.")
    label = f"W[{req.role}]"
    sanity.check_finite(label, W, print_fn=print_fn)
    if abs(complex(req.omega_ry).real) == 0.0:
        # HERMITICITY, and the one frequency at which it is unconditional.
        #
        # rtol is generous — this catches structural mixing, not roundoff.
        #
        # SCOPE.  With BOTH particle-hole orientations summed, χ₀(r,r';iω)
        # is REAL for any system, but it is Hermitian in (r,r') only when
        # time reversal holds.  Writing the ordered pair sum out, a v→c
        # transition of gap Δ contributes
        #
        #     [2ω Im(P) − 2Δ Re(P)] / (ω² + Δ²),    P = |ρ_vc⟩⟨ρ_vc|
        #
        # whose second term is real symmetric and whose FIRST term is real
        # ANTIsymmetric.  Im(P) cancels pairwise between k and −k exactly
        # when the reverse transition set is the conjugate of the forward
        # one, i.e. under Θ; on a magnet it survives, is odd in the
        # magnetisation, and carries the TR-odd dynamic screening.  It also
        # carries the factor ω, so it vanishes IDENTICALLY at ω = 0 whatever
        # the deck.  Derivation and a random-state model:
        # reports/four_current_head_frequency_audit_2026-09-01 §F2 (relative
        # anti-Hermitian part 2.9e-1 at z=2i without Θ, 2e-16 with it, 1e-17
        # at z=0 either way).
        #
        # So ω = 0 is gated for every deck, and ω = iω_p is gated only where
        # the deck's MEASURED verdict says Hermiticity is owed.  Elsewhere
        # the same residual is REPORTED, with its number, through the same
        # reducer — the framing ``check_q_conjugate_reciprocity`` below
        # already uses for its own TRS-independent statistic.
        #
        # WHAT THIS DOES NOT COST.  The probe role's W comes off the SAME
        # ``compute_static_w`` → ``solve_w`` path, on the same tiles and the
        # same mesh, as the static role in the same run — and the static
        # role's gate is unconditional.  A column-sharding fault in the
        # Dyson back-solve (the failure this gate was built for: NaN with
        # rc=0, caught downstream by counting floats in eqp0.dat) therefore
        # still refuses, through the ω = 0 role, on a magnet.  What changes
        # is only which of the two roles is allowed to call a residual a
        # defect.  Measured on today's builder the reported number is
        # ~machine-zero anyway: ``w_isdf._complete_static_vertex_orientations``
        # forms ``forward_R + conj(forward_R)``, i.e. χ_q + conj(χ_{−q}),
        # which is Hermitian for every q with no Θ assumption — so the
        # TR-odd channel is absent from χ₀(iω_p) BEFORE W is solved (four
        # bispinor SOC CrI3 GN-PPM runs, all `Time reversal: BROKEN`, all
        # silent at rtol=1e-6: runs 129/142/150/157).  That is a defect in
        # the χ₀ completion, registered separately; this gate must not
        # assert a property only that defect is supplying.
        if abs(complex(req.omega_ry).imag) == 0.0 or trs_allowed:
            sanity.check_hermitian(f"{label}[q=0]", W[0], rtol=1e-6,
                                   print_fn=print_fn)
        else:
            sanity.check_hermitian(f"{label}[q=0]", W[0], rtol=1e-6,
                                   print_fn=print_fn, measurement=True,
                                   cause=_TR_ODD_W_HERMITICITY_CAUSE)
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
        # floor is the orbit-closed IBZ arm, MEASURED there (armB_orbit504,
        # 2026-08-07): 1.13e-7 on this exact statistic.  Its magnitude
        # signature confirms round-off — the per-element relative residual
        # FALLS with |A| (4.9e-7 at <p10 to 3.1e-8 at >p99) while the
        # absolute residual rises.  1e-5 is ~90x above that floor and ~400x
        # below the smallest real break measured (7.8e-4), so it can neither
        # cry wolf nor miss one.
        #
        # THE FLOOR IS A MEASUREMENT, NOT AN IDENTITY.  This comment used to
        # justify itself with "the unfold builds W_{−q} from W_q by symmetry
        # and reciprocity therefore holds BY CONSTRUCTION".  It does not.
        # The unfold applies a SPATIAL operation and reciprocity is a
        # conjugation statement; the two coincide only if the finite ISDF ζ
        # basis is point-group covariant, which is measured FALSE by 1.2e-02
        # on the Na 8×8×8 SOC c464 deck.  1.13e-7 is that deck's covariance,
        # not an arithmetic floor.
        #
        # AND THIS GATE IS BLIND AT q == −q, where the condition collapses to
        # "W_q is real" and reads ~1e-17 regardless.  The discriminating
        # statistic is the little-group covariance of the IBZ parents,
        # measured at the unfold site above via
        # ``QgridTrsPolicy.measure_covariance`` and reported by
        # ``sanity.report_parent_covariance``.  Do not tighten this rtol to
        # compensate: it is measuring a projection.
        #
        # TRS SCOPE.  This statistic is NOT gated on the measured
        # time-reversal verdict and must not be: W_{−q} = conj(W_q) follows
        # from the conjugation-equivariance of the ISDF fit (the pair
        # densities at −q are the conjugates of those at +q with bra and ket
        # relabelled) and from v(|q+G|) being real, neither of which needs Θ.
        # On a magnet the relation simply stops being imposed by the unfold —
        # q and −q are independently solved — so this gate becomes an
        # independent measurement rather than an identity, which is strictly
        # more informative.
        if kgrid is not None:
            sanity.check_q_conjugate_reciprocity(
                f"{label}[all q]", W, kgrid, rtol=1e-5, print_fn=print_fn)


__all__ = [
    "ScreeningRequest",
    "screening_requests_for",
    "compute_static_w",
    "compute_screening",
    "compute_screening_model",
    "driver_persists_w0",
]
