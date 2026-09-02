"""Host-side branch + window construction for the Σ_c(ω) GN-PPM integration.

This is the leaf of the Σ_PPM module family: it turns (E_A, Ω_q stats, ω-grid,
quadrature params) into host-side ``_SigmaBranch`` / ``_SigmaWindow`` lists.  It
imports only ``minimax_screening`` (the quadrature engine), jax, and numpy — no
GPU kernels, no accumulators, no config objects.  The driver (``ppm_sigma``) and
the accumulators (``ppm_accumulators``) import *from* this module; nothing here
imports back.

Branch decomposition
--------------------

Four branches span ω ∈ ℝ: {conduction/empty, valence/occupied} A-space ×
{+ω, −ω} half.  Each branch is a definite-sign Laplace problem; a branch
carries only its physical identity ``(space, neg_omega_half)`` — the signs
that make Σ_c correct are written inline where the physics puts them, not
stored as ±1 fields.  The pole S = E_A + Ω enters the Σ_c denominator either
as (ω̃ − S), when S can coincide with a grid ω and the crossing (HGL)
quadrature is needed, or as the sign-definite (ω̃ + S), a single Laplace
window.  On the +ω half the conduction (empty) A-space is the crossing one;
on the −ω half — where Σ_c is evaluated at |ω| with an overall −1 baked into
each window's prefactor — the valence (occupied) A-space is.

The −ω branches are computed explicitly, term by term; they are NOT the
conjugate of the +ω result.  (The often-quoted *global* identity
Σ_c(−ω) = −[Σ_c(ω)]* is false — the ω→−ω reflection holds only per pole
term, which is exactly what swaps the crossing role between the conduction
and valence A-spaces above.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .efermi import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                     band_in_occupation_window, occupation_weight_floor)
from .minimax_screening import (
    MinimaxNodes,
    solve_laplace_minimax_interval,
    solve_phase_minimax_bandwidth,
)


# ---------------------------------------------------------------------------
#  Crossing-quadrature conditioning (the load-bearing fix for the conduction
#  Σ_c instability — reports/gw_ppm_sigma_regularization_2026-07-20).
#
#  The HGL "core" crossing window fits its regularization target with a minimax
#  sin-sum over the dimensionless bandwidth  A_core = 2·T/ξ = 2·ω_max/ξ + 2·edge
#  (T = ω_max + edge·ξ).  The sin-fit is well-conditioned only for MODEST A: the
#  exact solver's weights Σ|α_hat| jump from ~2–3 at A≲24 to ~3e4 (A=83, ξ=0.25)
#  and ~2e5 (A=43, ξ=0.5) — an ill-conditioning that is *not even monotone in ξ*.
#  Those O(1e4–1e5) weights are near-cancelling, so they amplify ANY perturbation
#  of the per-τ operand σ(τ) by Σ|α|; σ(τ) carries the (large, ~1e4 in the ISDF
#  centroid basis, and mesh-sensitive) screened W, so on a multi-device mesh the
#  amplified perturbations do NOT cancel → Σ_c blows up to O(1e5) eV (device/mesh
#  dependent, gap-inverting) with O(1e3) eV imaginary parts even on one device.
#
#  Fix: floor ξ so A_core ≤ _CROSSING_A_MAX (well-conditioned regime).  ξ is a
#  *broadening*; the QP self-energy is evaluated off-pole (at E_DFT, away from the
#  plasmon poles), so a coarser near-pole broadening does not move the QP energies.
#  The floor scales with ω_max, so it only engages when the Σ_c ω-grid is wide
#  enough to force an ill-conditioned crossing; a narrow grid keeps the user's ξ.
_CROSSING_A_MAX = 24.0     # dimensionless bandwidth ceiling (Σ|α_hat| ~ 2–3 below)

#: A reducible sign-definite inverse-Laplace pane may span at most eight binary
#: octaves in its denominator, ``x_max/x_min <= 2**8``.  A zero-width Ω pane
#: continues over the existing E/ω selectors rather than sending an
#: over-wide request to minimax.  This ceiling neither blesses a table nor
#: changes a pole, and is independent of the caller's node budget.
_SIGN_DEFINITE_PANE_MAX_RANGE = 2.0 ** 8


def crossing_regularization_floor(omega_max_ry: float, edge_factor: float) -> float:
    """Minimum ξ (Ry) so the HGL core bandwidth A_core ≤ _CROSSING_A_MAX.

    A_core = 2·ω_max/ξ + 2·edge ≤ A_max  ⇔  ξ ≥ 2·ω_max/(A_max − 2·edge).
    Returns 0.0 when no floor is meaningful (degenerate edge/ω_max).
    """
    denom = _CROSSING_A_MAX - 2.0 * float(edge_factor)
    if denom <= 1.0 or float(omega_max_ry) <= 0.0:
        return 0.0
    return 2.0 * float(omega_max_ry) / denom


#: Ansätze whose Σ_c crossing quadrature is the HGL sin-fit that
#: :func:`crossing_regularization_floor` was derived for, and therefore the
#: ones ``sigma_regularization_floor_ev = auto`` raises ξ for.  MPA's
#: crossing family is a positive real-time rule with its own node ceiling
#: and error budget (``gw.mpa.sigma_windows``), so the HGL bandwidth
#: derivation says nothing about it and ``auto`` leaves its ξ alone.  An
#: EXPLICIT floor applies to every ansatz — that is the knob a cross-ansatz
#: comparison uses to equalise ξ.
_HGL_CROSSING_ANSATZE = frozenset({"gn_ppm", "hl_ppm"})


def hgl_partition_required(
    omega_grid_ry,
    regularization_width_ry: float,
    edge_factor: float,
) -> bool:
    """Whether the resolved grid exceeds the incumbent HGL capacity."""
    omega = np.asarray(omega_grid_ry, dtype=np.float64)
    omega_max = float(np.max(np.abs(omega))) if omega.size else 0.0
    xi = float(regularization_width_ry)
    edge = float(edge_factor)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("regularization_width_ry must be finite and positive")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("edge_factor must be finite and non-negative")
    if not np.isfinite(omega_max):
        raise ValueError("omega_grid_ry contains a non-finite value")
    A_core = 2.0 * (omega_max + edge * xi) / xi
    capacity = _CROSSING_A_MAX * (
        1.0 + 8.0 * np.finfo(np.float64).eps)
    return A_core > capacity


class SigmaRegularization(NamedTuple):
    """The EFFECTIVE Σ broadening ξ, and where it came from.

    THE ONE PLACE the Σ regularization is resolved, for every ansatz.
    Before 2026-08-22 GN-PPM silently raised the deck's
    ``sigma_regularization_ev`` to a window-dependent conditioning floor
    while MPA passed the same key straight through, so the two ran at
    different broadenings on the same deck with nothing in either output
    tying the number back to the key the operator set.  Measured on the
    sodium 48b deck (grid ±5 eV, edge 1.5): GN-PPM 0.4762 eV against MPA
    0.2500 eV, **1.90×**; on a ±15 eV window the same formula gives 1.4286
    eV, **5.7×**.  Any MPA-vs-GN-PPM comparison made without equalising ξ
    is confounded, and the July MoS2 ξ ladder measured 2.381 eV worth +1016
    meV on the K gap — the floor is not comfortably far from where it goes
    bad.

    Fields are Ry except where named ``_ev``.  ``floor_policy`` is
    ``'auto'`` (the ansatz's own conditioning floor) or ``'explicit'`` (a
    deck-supplied ``sigma_regularization_floor_ev``).
    """

    requested_ry: float
    resolved_ry: float
    floor_ry: float
    floor_policy: str
    ansatz: str

    @property
    def raised(self) -> bool:
        return self.resolved_ry > self.requested_ry

    @property
    def requested_ev(self) -> float:
        from common.units import RYD_TO_EV
        return self.requested_ry * RYD_TO_EV

    @property
    def resolved_ev(self) -> float:
        from common.units import RYD_TO_EV
        return self.resolved_ry * RYD_TO_EV

    def describe(self) -> str:
        """ONE log line, identical in wording for every ansatz.

        A comparison can only assert that two runs share ξ if both runs
        print it the same way; before this the PPM arms printed "ξ raised
        0.250 → 1.429 eV" only when the floor engaged and the MPA arms
        printed "eta=0.2500 eV", so a reader had to know the formula to
        tell whether they matched.
        """
        from common.units import RYD_TO_EV
        head = (f"  Σ broadening ξ: {self.resolved_ev:.4f} eV "
                f"(requested {self.requested_ev:.4f} eV, ansatz "
                f"{self.ansatz}, floor {self.floor_ry * RYD_TO_EV:.4f} eV "
                f"[{self.floor_policy}])")
        if self.raised:
            head += " — RAISED to the floor"
        return head


def resolve_sigma_regularization(
    *,
    requested_ry: float,
    omega_grid_ry,
    edge_factor: float,
    ansatz: str,
    floor_ev=None,
) -> SigmaRegularization:
    """Resolve the effective Σ broadening ξ.  Called by every Σ ansatz.

    Parameters
    ----------
    requested_ry
        ``sigma_regularization_ev`` converted to Ry — what the deck asked
        for.
    omega_grid_ry
        The Σ ω grid.  Only ``max|ω|`` is read; that is what sets the HGL
        core bandwidth ``A_core = 2·ω_max/ξ + 2·edge``.
    edge_factor
        ``sigma_window_edge_factor``.
    ansatz
        ``compute_mode``'s value string (``'gn_ppm'``, ``'hl_ppm'``,
        ``'mpa'``, …).  Decides what ``auto`` means; see
        :data:`_HGL_CROSSING_ANSATZE`.
    floor_ev
        ``sigma_regularization_floor_ev``.  ``None`` / ``'auto'`` selects
        the ansatz's own conditioning floor.  A float (eV) is an EXPLICIT
        floor applied to every ansatz — the knob that equalises ξ across a
        cross-ansatz comparison.  ``0`` is a legal explicit value and means
        "do not raise", which on an HGL ansatz re-opens the ill-conditioned
        regime the floor exists for; it is spellable on purpose so an
        operator can measure that, and it is stamped so nobody can do it by
        accident.

    Returns
    -------
    SigmaRegularization
        Pure function of its arguments, so a consumer that has the same
        config (the Σ_c(ω) HDF5 writer, for instance) can re-derive the
        resolved value instead of having it threaded to it.
    """
    from common.units import RYD_TO_EV

    requested = float(requested_ry)
    omega = np.asarray(omega_grid_ry, dtype=np.float64)
    omega_max_ry = float(np.max(np.abs(omega))) if omega.size else 0.0
    name = str(getattr(ansatz, "value", ansatz)).strip().lower()

    explicit = not (floor_ev is None
                    or str(floor_ev).strip().lower() == "auto")
    if explicit:
        floor_ry = float(floor_ev) / RYD_TO_EV
        if floor_ry < 0.0:
            raise ValueError(
                f"sigma_regularization_floor_ev must be >= 0 or 'auto'; "
                f"got {floor_ev!r}.")
        policy = "explicit"
    elif name in _HGL_CROSSING_ANSATZE:
        floor_ry = crossing_regularization_floor(omega_max_ry, edge_factor)
        policy = "auto"
    else:
        floor_ry = 0.0
        policy = "auto"

    return SigmaRegularization(
        requested_ry=requested,
        resolved_ry=max(requested, floor_ry),
        floor_ry=floor_ry,
        floor_policy=policy,
        ansatz=name,
    )


def sigma_regularization_for_config(config) -> SigmaRegularization:
    """:func:`resolve_sigma_regularization` from a ``LorraxConfig``.

    Attribute reads only — no import of ``gw_config``, so the leaf-module
    rule at the top of this file still holds.  Every consumer that has the
    config (the drivers AND the Σ_c(ω) HDF5 writer) calls THIS, so the
    stamped value and the value the kernel ran at cannot disagree.
    """
    from common.units import RYD_TO_EV

    sigma_cfg = config.sigma
    return resolve_sigma_regularization(
        requested_ry=float(sigma_cfg.regularization_ev) / RYD_TO_EV,
        omega_grid_ry=np.asarray(config.omega_grid_ry, dtype=np.float64),
        edge_factor=float(sigma_cfg.window_edge_factor),
        ansatz=config.compute_mode,
        floor_ev=getattr(sigma_cfg, "regularization_floor_ev", None),
    )


@dataclass(frozen=True)
class _SigmaWindow:
    name: str
    nodes: MinimaxNodes        # (t, alpha) complex128, carries this window's τ points
    mask_A: np.ndarray         # (nk, nb)
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str               # "full" (Laplace) or "imag" (crossing)
    prefactor: float
    mask_B_mode: str = "all"
    mask_B_threshold: float | None = None
    crossing_kind: str | None = None
    #: The ACHIEVED error of the quadrature behind this window, and where
    #: that quadrature came from.  Both trailing and defaulted, so every
    #: existing construction site and the frozen G2 tile flattening (which
    #: names its fields explicitly) are untouched.
    #:
    #: These exist because the per-window log line used to print
    #: ``err<{target_error:.0e}`` -- the bound that was REQUESTED.  A window
    #: whose rule missed its target by four orders printed the same line as
    #: one that met it, which is half of how the minimax module came to be
    #: trusted for the wrong reason.
    max_error: float | None = None
    provenance: str | None = None
    #: Optional scalar pane selectors.  They are execution metadata for the
    #: bounded HGL decomposition, not replacement masks: ``mask_A`` remains
    #: the band-identity/occupation selector, while the canonical
    #: ``build_G_tau(..., E_min, E_max)`` owner applies ``(E_min,E_max]``.
    #: ``omega_indices`` lets the incumbent accumulator project only the
    #: omega cluster this rule certifies and scatter it into the branch tile.
    E_min: float | None = None
    E_max: float | None = None
    B_lo: float | None = None
    B_hi: float | None = None
    omega_indices: np.ndarray | None = None

    @property
    def n_tau(self) -> int:
        return int(self.nodes.t.shape[0])

    @property
    def project_code(self) -> int:
        """Int form of ``project`` — the code branch in _project_tau_onto_omega_np."""
        if self.project == "full":
            return 0
        if self.project == "imag":
            return 1
        raise ValueError(f"Unknown window projection {self.project!r}; "
                         f"expected 'full' or 'imag'.")


# ---------------------------------------------------------------------------
#  Branch enumeration — the four (ω sign × cond/val) calls that together
#  sum to Σ_c(ω).  Split into a NamedTuple so every caller sees the same
#  physics labeling without copy-pasted kwargs.
# ---------------------------------------------------------------------------

class _SigmaBranch(NamedTuple):
    """One branch of the Σ_c(ω) sum.  Four branches cover ω ∈ ℝ.

    A branch stores only its physical identity — which A-space it sums and
    which ω half it covers.  The signs (which way the pole enters the
    denominator, the −1 that the −ω half carries) are derived from
    ``(space, neg_omega_half)`` at each expression that needs them; see
    ``_build_windows_for_branch``.
    """
    tag: str                    # human label ("ω≥E_F cond" etc.) — drives progress output
    E_A: jax.Array              # (nk, nb) energy-above-Fermi for A-space (E_cond or H_val)
    base_mask_A: jax.Array      # (nk, nb) bool — which bands in A-space contribute
    space: str                  # "cond" (empty A-space, E_c−E_F) / "val" (occupied, E_F−E_v)
    neg_omega_half: bool        # True on the −ω half — Σ_c evaluated at |ω|
    omega_abs: np.ndarray       # non-negative ω values to evaluate at (|ω_rel|)
    omega_idx: np.ndarray       # global ω indices these map into
    band_weight: jax.Array | None = None
    # (nk, nb) real, or None.  Fractional-occupation weight (f on the val
    # branch, 1−f on cond) applied in the τ kernel's G synthesis through the
    # build_G_tau(band_weight=...) seam.  None ⇒ the incumbent bool-mask
    # semantics, bit-exact.  Never clipped — MP smearing can overshoot [0,1]
    # (docs/theory/finite-occupation-screening.md).


# ---------------------------------------------------------------------------
#  Shared scalar crossing geometry.  MPA and GN have different pole carriers,
#  but the omega-axis partition is physics-neutral and therefore has one owner
#  here, below both planners.  The GN cell planner is deliberately host-only:
#  it returns scalar interval metadata and never builds a pole-sized mask.
# ---------------------------------------------------------------------------

def _omega_clusters(
    omega_abs,
    gap_ry: float,
    *,
    max_span_ry: float | None = None,
):
    """Split ``|omega|`` values at gaps, optionally capping cluster span.

    Returns ``[(index_array, w_lo, w_hi), ...]`` in ascending-energy order.
    Each index array is sorted back into the caller's original order, which is
    the ordering the Sigma branch accumulator consumes.  ``max_span_ry=None``
    is the incumbent MPA rule exactly: only genuine gaps split a grid.

    The optional span cap is a planning constraint, not a second clustering
    algorithm.  Within each gap-connected component the greedy left-to-right
    packing is the minimum number of scalar intervals of the requested maximum
    width that cover the actual evaluation points.
    """
    gap = float(gap_ry)
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError("omega cluster gap must be finite and positive")
    span = None if max_span_ry is None else float(max_span_ry)
    if span is not None and (not np.isfinite(span) or span <= 0.0):
        raise ValueError("omega cluster maximum span must be finite and positive")

    w = np.asarray(omega_abs, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("omega_abs must be one-dimensional")
    if not np.all(np.isfinite(w)):
        raise ValueError("omega_abs contains a non-finite value")
    if not w.size:
        return []

    order = np.argsort(w, kind="stable")
    breaks = np.nonzero(np.diff(w[order]) > gap)[0]
    gap_pieces = np.split(order, breaks + 1)
    pieces = []
    for piece in gap_pieces:
        if span is None:
            pieces.append(piece)
            continue
        start = 0
        while start < piece.size:
            w_start = float(w[piece[start]])
            stop = start + 1
            while (stop < piece.size
                   and float(w[piece[stop]]) - w_start <= span):
                stop += 1
            pieces.append(piece[start:stop])
            start = stop
    return [
        (np.sort(piece), float(np.min(w[piece])), float(np.max(w[piece])))
        for piece in pieces
    ]


@dataclass(frozen=True)
class HGLCrossingCell:
    """One exact rectangular cell of a clustered GN crossing branch.

    ``omega_indices`` addresses the owning branch's omega vector.  The A and B
    selectors both use the repository-wide ``(lo, hi]`` convention.  ``e_min``
    and ``e_max`` are the actual live A-energy extrema in that selector; they
    are kept separate from ``e_lo`` because the first selector begins at
    ``-inf``.  ``A_dim`` is present only on the crossing shell.
    """

    kind: str
    omega_indices: np.ndarray
    omega_lo: float
    omega_hi: float
    e_lo: float
    e_hi: float
    e_min: float
    e_max: float
    b_lo: float
    b_hi: float
    A_dim: float | None = None


@dataclass(frozen=True)
class HGLCrossingPlan:
    """Pure scalar plan for omega-cluster x A-pane x B-cell tiling."""

    cells: tuple[HGLCrossingCell, ...]
    omega_cluster_count: int
    energy_pane_count: int
    max_A_dim: float
    regularization_width_ry: float
    edge_factor: float


@dataclass(frozen=True)
class _SignDefiniteCell:
    """One exact ``A x Omega x omega`` cell served by a Laplace rule.

    All three selectors reuse metadata already consumed by the canonical
    Sigma executor: ``(E_lo,E_hi]`` by ``build_G_tau``, ``(B_lo,B_hi]`` by
    ``_build_W_t_q``, and explicit evaluation-frequency indices by the
    accumulator.  ``x_min/x_max`` are the actual extrema of the cell, not a
    requested catalog tier.
    """

    E_lo: float
    E_hi: float
    E_min: float
    E_max: float
    B_lo: float
    B_hi: float
    B_count: int
    B_min: float
    B_max: float
    omega_indices: np.ndarray
    omega_min: float
    omega_max: float
    x_min: float
    x_max: float


def _energy_panes(values: np.ndarray, max_span_ry: float):
    """Minimum-width-capped ``(lo, hi]`` cover of actual scalar energies."""
    cap = float(max_span_ry)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError("A-energy pane maximum span must be finite and non-negative")
    unique = np.unique(np.asarray(values, dtype=np.float64))
    if not unique.size:
        return []
    panes = []
    select_lo = -np.inf
    start = 0
    while start < unique.size:
        e_min = float(unique[start])
        stop = int(np.searchsorted(unique, e_min + cap, side="right"))
        # ``unique[start]`` is always inside its own zero-width pane.  Keep the
        # guard explicit because a NaN/overflow here would otherwise loop.
        if stop <= start:
            raise AssertionError("energy-pane packing made no progress")
        e_max = float(unique[stop - 1])
        panes.append((float(select_lo), e_max, e_min, e_max))
        select_lo = e_max
        start = stop
    return panes


def plan_hgl_crossing_cells(
    *,
    omega_abs,
    E_A,
    base_mask_A,
    regularization_width_ry: float,
    edge_factor: float,
    omega_cluster_gap_ry: float,
    omega_max_span_ry: float,
    crossing_A_max: float = _CROSSING_A_MAX,
) -> HGLCrossingPlan:
    """Plan the exact clustered GN crossing decomposition, without execution.

    For an omega cluster ``[w0,w1]``, an A-energy pane ``(e0,e1]`` with
    actual extrema ``[emin,emax]``, and ``z = edge*xi``, the B axis is split
    into three adjacent cells::

        (-inf, w0-emax-z]                 x = omega-E-B >= z
        (w0-emax-z, w1-emin+z]           crossing shell
        (w1-emin+z, +inf)                x = omega-E-B <= -z

    The shell obeys
    ``max|x| <= Delta_omega + Delta_E + z``.  Energy panes are greedily
    packed at the largest width allowed by ``crossing_A_max``; that greedy
    cover is minimal for the chosen omega clustering.  Nothing here chooses a
    quadrature, touches JAX, or changes the incumbent contiguous-grid path.
    """
    xi = float(regularization_width_ry)
    edge = float(edge_factor)
    A_max = float(crossing_A_max)
    omega_span = float(omega_max_span_ry)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("regularization_width_ry must be finite and positive")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("edge_factor must be finite and non-negative")
    if not np.isfinite(A_max) or A_max <= edge:
        raise ValueError("crossing_A_max must be finite and exceed edge_factor")
    capacity = (A_max - edge) * xi
    if (not np.isfinite(omega_span) or omega_span <= 0.0
            or omega_span >= capacity):
        raise ValueError(
            "omega_max_span_ry must be positive and smaller than the HGL "
            f"pane capacity {(capacity):.6g} Ry")

    energies = np.asarray(E_A, dtype=np.float64)
    base = np.asarray(base_mask_A, dtype=bool)
    if energies.shape != base.shape:
        raise ValueError("E_A and base_mask_A must have identical shapes")
    live = energies[base]
    if not live.size:
        return HGLCrossingPlan((), 0, 0, 0.0, xi, edge)
    if not np.all(np.isfinite(live)):
        raise ValueError("live A-energy support contains a non-finite value")

    clusters = _omega_clusters(
        omega_abs, omega_cluster_gap_ry, max_span_ry=omega_span)
    z = edge * xi
    cells = []
    n_panes = 0
    max_A_dim = 0.0
    for omega_indices, w_lo, w_hi in clusters:
        delta_w = w_hi - w_lo
        e_span_cap = A_max * xi - z - delta_w
        if e_span_cap < 0.0:
            raise AssertionError(
                "omega cluster exceeds the HGL capacity after span capping")
        panes = _energy_panes(live, e_span_cap)
        n_panes += len(panes)

        # The energy intervals must tile the live A support exactly once.
        ownership = np.zeros(live.shape, dtype=np.int32)
        for e_lo, e_hi, _e_min, _e_max in panes:
            ownership += ((live > e_lo) & (live <= e_hi)).astype(np.int32)
        if not np.all(ownership == 1):
            raise AssertionError(
                "GATE hgl_A_partition: A panes do not tile live support exactly")

        for e_lo, e_hi, e_min, e_max in panes:
            b_shell_lo = w_lo - e_max - z
            b_shell_hi = w_hi - e_min + z
            A_dim = (delta_w + (e_max - e_min) + z) / xi
            if A_dim > A_max * (1.0 + 8.0 * np.finfo(np.float64).eps):
                raise AssertionError(
                    "GATE hgl_shell_capacity: planned shell exceeds "
                    f"A_dim={A_dim:.16g} > {A_max:.16g}")
            max_A_dim = max(max_A_dim, A_dim)
            common = dict(
                omega_indices=np.asarray(omega_indices, dtype=np.int64),
                omega_lo=w_lo, omega_hi=w_hi,
                e_lo=e_lo, e_hi=e_hi, e_min=e_min, e_max=e_max)
            cells.extend((
                HGLCrossingCell(
                    "positive", b_lo=-np.inf, b_hi=b_shell_lo,
                    **common),
                HGLCrossingCell(
                    "crossing", b_lo=b_shell_lo, b_hi=b_shell_hi,
                    A_dim=A_dim, **common),
                HGLCrossingCell(
                    "negative", b_lo=b_shell_hi, b_hi=np.inf,
                    **common),
            ))

    # Each omega point belongs to exactly one cluster, including repeated
    # values (stable indices, not value equality, own the partition).
    omega = np.asarray(omega_abs, dtype=np.float64)
    omega_ownership = np.zeros(omega.shape, dtype=np.int32)
    for idx, _lo, _hi in clusters:
        omega_ownership[np.asarray(idx, dtype=np.int64)] += 1
    if not np.all(omega_ownership == 1):
        raise AssertionError(
            "GATE hgl_omega_partition: omega clusters do not tile the branch")

    return HGLCrossingPlan(
        cells=tuple(cells), omega_cluster_count=len(clusters),
        energy_pane_count=n_panes, max_A_dim=max_A_dim,
        regularization_width_ry=xi, edge_factor=edge)


def _iter_branches(
    *,
    omega_pos: np.ndarray, idx_pos: np.ndarray,
    omega_neg_abs: np.ndarray, idx_neg: np.ndarray,
    E_cond: jax.Array, H_val: jax.Array,
    cond_mask: jax.Array, val_mask: jax.Array,
    cond_weight: jax.Array | None = None,
    val_weight: jax.Array | None = None,
    weight_floor: float = 0.0,
) -> list[_SigmaBranch]:
    """Enumerate the 4 branches (A-space × ω-half), skipping empty ω halves.

    Each branch is one definite-sign Laplace problem.  The +ω half sums Σ_c
    directly on E_A = E_c−E_F (cond) or E_F−E_v (val); the −ω half evaluates at
    |ω| and carries an overall −1.  No sign is stored here — the branch's
    identity ``(space, neg_omega_half)`` is what the window builder reads to
    place the pole in the denominator (crossing vs sign-definite) and to fold
    the −ω-half −1 into each window's prefactor.

    The two −ω branches are built and integrated explicitly, exactly like the
    +ω ones — they are not conjugated from the +ω result (the global
    Σ_c(−ω) = −[Σ_c(ω)]* identity does not hold; the reflection is per pole
    term, which is why the crossing role moves from cond to val across halves).

    ``weight_floor`` is ``1 − occupation_window_threshold``
    (``gw.efermi.occupation_weight_floor``): a band leaves a branch once its
    branch weight's MAGNITUDE falls to the floor.  It is applied HERE, at the
    one place all four branches are built, so no consumer of ``base_mask_A``
    can plan or execute against a support the deck did not ask for.  A branch
    with no weight — every insulating plan, and today every GN-PPM branch —
    keeps the incumbent bool mask bit-for-bit, because there is no weight to
    threshold; see the note in :func:`branches_for_omega_grid`.
    """
    def _narrow(mask, weight):
        # The identity that makes threshold = 1.0 the exact incumbent rule:
        # floor 0.0 ⇒ abs(w) > 0.0 ⇔ w != 0.0.  None ⇒ the SAME OBJECT back,
        # not merely an equal one, so the bool-mask path stays bit-exact and
        # emits no jnp op at all.
        if weight is None:
            return mask
        return mask & band_in_occupation_window(weight, weight_floor)

    cond_mask = _narrow(cond_mask, cond_weight)
    val_mask = _narrow(val_mask, val_weight)
    branches: list[_SigmaBranch] = []
    if omega_pos.size:
        branches += [
            _SigmaBranch(tag="ω≥E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         space="cond", neg_omega_half=False,
                         omega_abs=omega_pos, omega_idx=idx_pos,
                         band_weight=cond_weight),
            _SigmaBranch(tag="ω≥E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         space="val",  neg_omega_half=False,
                         omega_abs=omega_pos, omega_idx=idx_pos,
                         band_weight=val_weight),
        ]
    if omega_neg_abs.size:
        branches += [
            _SigmaBranch(tag="ω<E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         space="cond", neg_omega_half=True,
                         omega_abs=omega_neg_abs, omega_idx=idx_neg,
                         band_weight=cond_weight),
            _SigmaBranch(tag="ω<E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         space="val",  neg_omega_half=True,
                         omega_abs=omega_neg_abs, omega_idx=idx_neg,
                         band_weight=val_weight),
        ]
    return branches


def branches_for_omega_grid(
    omega_grid_ry,
    *,
    E_cond: jax.Array, H_val: jax.Array,
    cond_mask: jax.Array, val_mask: jax.Array,
    cond_weight: jax.Array | None = None,
    val_weight: jax.Array | None = None,
    occupation_window_threshold: float = OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
) -> list[_SigmaBranch]:
    """Split a signed ω grid at zero and enumerate the four causal branches.

    Only the ω-sign split is shared; the A-space energies/masks stay with
    the caller because the two drivers derive them differently (PPM clips
    E_cond/H_val at zero around an internally derived E_F; MPA keeps signed
    occupation-chosen distances for small-gap/inverted systems).

    ``occupation_window_threshold`` is the OCCUPANCY at which a band leaves a
    Green's-function branch; it is converted here to the weight floor
    ``1 − threshold`` and applied to ``cond_mask``/``val_mask`` against the
    supplied weights.  **It has an effect only where a weight is supplied.**

    THE GN-PPM Σ DRIVER SUPPLIES NONE, and that is a real gap this key cannot
    close on its own: ``ppm_sigma._prepare_sigma_state`` derives its two masks
    as ``occ_full > 0.5`` from ``wfns.occ``, which ``wavefunction_bundle.
    _build_occ`` fills as the STEP array ``(enk <= efermi)`` — so that driver
    has no fractional occupancy anywhere, only an integer split, and there is
    nothing for an occupancy threshold to cut.  Closing it means porting
    fractional occupations into that driver first (its own
    ``TODO(metal-greens)``), after which the threshold governs it through this
    call with no further change.  Validated at this consumer.
    """
    omega = np.asarray(omega_grid_ry, np.float64)
    idx_pos, idx_neg = np.where(omega >= 0.0)[0], np.where(omega < 0.0)[0]
    return _iter_branches(
        omega_pos=omega[idx_pos], idx_pos=idx_pos,
        omega_neg_abs=-omega[idx_neg], idx_neg=idx_neg,
        E_cond=E_cond, H_val=H_val,
        cond_mask=cond_mask, val_mask=val_mask,
        cond_weight=cond_weight, val_weight=val_weight,
        weight_floor=occupation_weight_floor(occupation_window_threshold))


def _to_host_np(a, dtype=np.complex128, *, tiled: bool = False):
    """Gather a possibly sharded array to host."""
    # A globally-sharded (non-fully-addressable) jax.Array can only be gathered
    # with tiled=True.  With tiled=False process_allgather raises
    #   ValueError: Gathering global non-fully-addressable arrays only
    #               supports tiled=True
    # and the device_get fallback below then raises
    #   RuntimeError: Fetching value for `jax.Array` that spans non-addressable
    #                 (non process local) devices is not possible
    # so BOTH paths died and every multi-host PPM sigma run aborted right after
    # the first sigma branch finished.  tiled=True returns the reconstructed
    # *global* array, which is what the callers here want (they add it straight
    # into a host buffer of global shape).  tiled is left as the caller asked
    # for process-local / fully-addressable values, so single-process behaviour
    # is unchanged.
    if not getattr(a, "is_fully_addressable", True):
        tiled = True
    # NO FALLBACK, AND THAT IS THE FIX.  This used to catch bare ``Exception``
    # and retry with ``jax.device_get``, which is the SECOND gather of the
    # same array and cannot succeed where the first failed on a globally
    # sharded operand -- the comment above records both deaths.  What the
    # fallback bought was not recovery but a WORSE error message: the run
    # died one frame later inside device_get, reporting "Fetching value for a
    # jax.Array that spans non-addressable devices" instead of whatever
    # process_allgather actually objected to.  A gather that cannot gather is
    # a failure; it is allowed to say so.
    return np.asarray(
        jax.experimental.multihost_utils.process_allgather(a, tiled=tiled),
        dtype=dtype,
    )


def _to_host_scalar(a, dtype=float):
    np_dtype = np.dtype(dtype)
    gathered = _to_host_np(jnp.asarray(a), dtype=np_dtype, tiled=False)
    return dtype(np.asarray(gathered).reshape(-1)[0])


def _masked_stats_device(values: jax.Array, mask: jax.Array) -> tuple[int, int, float | None, float | None]:
    """Return total size, masked count, and masked min/max."""
    total = int(np.prod(values.shape))
    count = int(_to_host_scalar(jnp.sum(mask, dtype=jnp.int64), int))
    if count == 0:
        return total, 0, None, None
    min_val = float(_to_host_scalar(jnp.min(jnp.where(mask, values, jnp.inf)), float))
    max_val = float(_to_host_scalar(jnp.max(jnp.where(mask, values, -jnp.inf)), float))
    return total, count, min_val, max_val


@jax.jit
def _masked_interval_stats_kernel(values, base_mask, lo, hi):
    """Scalar statistics for one ``(lo, hi]`` pane, without a mask tile."""
    selected = base_mask & (values > lo) & (values <= hi)
    return (
        jnp.sum(selected, dtype=jnp.int64),
        jnp.min(jnp.where(selected, values, jnp.inf)),
        jnp.max(jnp.where(selected, values, -jnp.inf)),
    )


def _masked_interval_stats_device(
    values: jax.Array,
    base_mask: jax.Array,
    lo: float,
    hi: float,
) -> tuple[int, int, float | None, float | None]:
    """As :func:`_masked_stats_device`, for a dynamic scalar interval.

    The pane predicate is born and consumed inside one compiled reduction;
    no pole-sized boolean operand is materialised or retained per window.
    """
    total = int(np.prod(values.shape))
    count_j, min_j, max_j = _masked_interval_stats_kernel(
        values, base_mask,
        jnp.asarray(lo, dtype=jnp.float64),
        jnp.asarray(hi, dtype=jnp.float64),
    )
    count = int(_to_host_scalar(count_j, int))
    if count == 0:
        return total, 0, None, None
    return (
        total,
        count,
        float(_to_host_scalar(min_j, float)),
        float(_to_host_scalar(max_j, float)),
    )


def window_mask_B_bounds(window: _SigmaWindow) -> tuple[float, float]:
    """One window's B-side Ω selector, as the SCALAR bounds ``(lo, hi)``.

    Replaces the old ``_materialize_window_mask_B``, which built a full
    ``(nq, μ_pad, μ_pad)`` boolean tile on device — per window, eagerly,
    outside any jit — and kept it alive as a kernel operand for every τ
    node of that window's scan.  That tile is ~148 MiB/rank at the MoS₂
    4×4 / μ_pad = 24,960 / P = 64 reference shape, resident straight
    through the most memory-intensive stage of Σ.  The window it encodes
    is TWO NUMBERS.  Ship the two numbers; let the predicate be recomputed
    from ``Omega_q`` inside the kernel, where it fuses into the select that
    was already there and never becomes a buffer.

    THE CONVENTION IS HALF-OPEN ON THE LOW SIDE, ``(lo, hi]``, because
    that is what the three existing modes already are, exactly:

        mode "all"   ->  (-inf, +inf)   ->  (Ω > -inf) & (Ω <= +inf)  == all
        mode "le_t"  ->  (-inf, T)      ->  (Ω > -inf) & (Ω <= T)     == Ω <= T
        mode "gt_t"  ->  (T, +inf)      ->  (Ω > T)    & (Ω <= +inf)  == Ω > T

    so ONE data-driven predicate reproduces all three bit-for-bit, with no
    static argument and therefore no extra compile.

    This is now the convention on BOTH sides of Σ.  It used to be the mirror
    of ``greens_function_kernel.windowed_exp_iEt``'s ``[lo, hi)``, so the A
    and B sides assigned a pole sitting exactly on a threshold in opposite
    directions.  That assignment belongs to the window PLAN — which pole goes
    into which certified-quadrature interval — and the plan's owner decided
    it downward: a pane must contain its own supremum, because every rule is
    built at max(Γ) over its pane.  ``windowed_exp_iEt`` was flipped to match
    this side (2026-08-10); the B side is unchanged, and
    ``tests/test_windowed_exp_iEt.py`` gates the two against each other.

    Ω is finite and non-negative by construction (it is
    ``where(good, sqrt(ω²), fallback)`` out of the GN-PPM fit, with
    ``good`` gating on ``isfinite``), so the ±inf sentinels are safe: no
    lane can compare false against both of them.
    """
    if window.B_lo is not None or window.B_hi is not None:
        if window.B_lo is None or window.B_hi is None:
            raise ValueError("A scalar B pane requires both B_lo and B_hi")
        lo, hi = float(window.B_lo), float(window.B_hi)
        if not lo < hi:
            raise ValueError(f"Invalid B pane ({lo!r}, {hi!r}]")
        return lo, hi

    mode = str(window.mask_B_mode)
    if mode == "all":
        return (-np.inf, np.inf)
    if window.mask_B_threshold is None:
        raise ValueError(f"mask_B_mode={mode!r} requires a mask_B_threshold")
    T = float(window.mask_B_threshold)
    if mode == "le_t":
        return (-np.inf, T)
    if mode == "gt_t":
        return (T, np.inf)
    raise ValueError(f"Unknown mask_B_mode={mode!r}")


def sigma_window_product_support(
    window: _SigmaWindow,
    E_A: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    omega_abs: np.ndarray,
) -> dict:
    """Return one physical window's exact scalar product support.

    The executor represents a Cartesian product without materialising it:
    selected intermediate states ``(k,n)`` times selected elementwise PPM
    poles ``(q,mu,nu)`` times this window's external-frequency points.  Since
    the denominator is affine in all three coordinates, the live extrema are
    sufficient to construct all eight real-support corners.  Pole statistics
    remain scalar device reductions; no ``(q,mu,nu)`` host array is gathered.

    Parameters
    ----------
    window
        Physical PPM pane and its ``(lo, hi]`` state/pole selectors.
    E_A
        Positive branch energies, shape ``(nk, nb)`` in Ry.
    Omega_q
        Elementwise PPM pole frequencies, shape ``(nq, nmu, nmu)`` in Ry.
    base_mask_B
        Live-pole selector with the same shape as ``Omega_q``.
    omega_abs
        Non-negative branch frequency grid, shape ``(nomega,)`` in Ry.

    Returns
    -------
    dict
        Host scalar/vector support used by ``gw.sigma_box_plan``.  The pole
        width extrema are exactly zero for the real GN/HL-PPM carrier.
    """
    energy = _to_host_np(E_A, dtype=np.float64, tiled=False)
    selected_A = np.array(window.mask_A, dtype=bool, copy=True)
    if selected_A.shape != energy.shape:
        if selected_A.size != energy.size:
            raise ValueError(
                "Sigma window state selector and E_A have different sizes")
        selected_A = selected_A.reshape(energy.shape)
    if window.E_min is not None:
        selected_A &= energy > float(window.E_min)
    if window.E_max is not None:
        selected_A &= energy <= float(window.E_max)
    selected_A &= np.isfinite(energy)
    states = np.asarray(energy[selected_A], dtype=np.float64)
    if not states.size:
        raise ValueError(f"Sigma window {window.name!r} has no live states")

    B_lo, B_hi = window_mask_B_bounds(window)
    _, pole_count, pole_min, pole_max = _masked_interval_stats_device(
        Omega_q, base_mask_B, B_lo, B_hi)
    if pole_count == 0 or pole_min is None or pole_max is None:
        raise ValueError(f"Sigma window {window.name!r} has no live PPM poles")

    omega = np.asarray(omega_abs, dtype=np.float64)
    if window.omega_indices is None:
        omega_indices = np.arange(omega.size, dtype=np.int64)
    else:
        omega_indices = np.asarray(window.omega_indices, dtype=np.int64)
    if (not omega_indices.size or np.any(omega_indices < 0)
            or np.any(omega_indices >= omega.size)):
        raise ValueError(
            f"Sigma window {window.name!r} has invalid omega indices")

    return {
        "states": states,
        "state_count": int(states.size),
        "pole_stats": ((float(pole_min), float(pole_max), 0.0, 0.0),),
        "pole_count": int(pole_count),
        "omega_indices": omega_indices,
        "omega_abs": np.asarray(omega[omega_indices], dtype=np.float64),
        "B_lo": float(B_lo),
        "B_hi": float(B_hi),
    }


# ---------------------------------------------------------------------------
#  Minimax window construction
# ---------------------------------------------------------------------------


def _deferred_box_nodes() -> MinimaxNodes:
    """Empty carrier between physical-pane planning and the shared box fit.

    A box run must not first request an unrelated shipped minimax rule merely
    to obtain this module's state/pole/omega selectors.  The orchestrator
    replaces this carrier before execution and refuses if an empty rule ever
    survives that handoff.
    """
    empty = jnp.asarray(np.empty(0, dtype=np.complex128))
    return MinimaxNodes(t=empty, alpha=empty)


def _sign_definite_support(
    E_min: float,
    E_max: float,
    B_min: float,
    B_max: float,
    omega_max: float,
    *,
    omega_min: float = 0.0,
    subtract_omega: bool = False,
    x_min_floor: float = 1.0e-12,
) -> tuple[float, float]:
    """Conservative support for one sign-definite ``E + Ω ± ω`` pane."""
    omega_lo = float(omega_min)
    omega_hi = float(omega_max)
    if subtract_omega:
        raw_min = float(E_min) + float(B_min) - omega_hi
        raw_max = float(E_max) + float(B_max) - omega_lo
        if raw_min <= 0.0 or raw_max < raw_min:
            raise AssertionError(
                "GATE sign_definite_support: E + Omega - omega is not "
                f"strictly positive on [{raw_min:.16g}, {raw_max:.16g}]")
    else:
        raw_min = float(E_min) + float(B_min) + omega_lo
        raw_max = float(E_max) + float(B_max) + omega_hi
    x_min = max(raw_min, float(x_min_floor), 1.0e-12)
    x_max = max(
        raw_max,
        x_min * (1.0 + 1.0e-9),
    )
    return x_min, x_max


def _oriented_sign_definite_support(
    E_min: float,
    E_max: float,
    B_min: float,
    B_max: float,
    omega_min: float,
    omega_max: float,
    orientation: str,
) -> tuple[float, float]:
    """Actual positive support for one of the three Sigma Laplace forms."""
    if orientation == "E+B+omega":
        raw_min = float(E_min) + float(B_min) + float(omega_min)
        raw_max = float(E_max) + float(B_max) + float(omega_max)
    elif orientation == "E+B-omega":
        raw_min = float(E_min) + float(B_min) - float(omega_max)
        raw_max = float(E_max) + float(B_max) - float(omega_min)
    elif orientation == "omega-E-B":
        raw_min = float(omega_min) - float(E_max) - float(B_max)
        raw_max = float(omega_max) - float(E_min) - float(B_min)
    else:
        raise ValueError(f"Unknown sign-definite orientation {orientation!r}")
    if raw_min <= 0.0 or raw_max < raw_min:
        raise AssertionError(
            f"GATE sign_definite_support: {orientation} is not strictly "
            f"positive on [{raw_min:.16g}, {raw_max:.16g}]")
    return raw_min, raw_max


def _assert_bounded_laplace_support(x_min: float, x_max: float) -> None:
    """Fail closed before an unpartitioned Sigma range reaches minimax."""
    if (not np.isfinite(x_min) or not np.isfinite(x_max)
            or float(x_min) <= 0.0 or float(x_max) < float(x_min)):
        raise AssertionError(
            "GATE sign_definite_partition: invalid Laplace support reached "
            f"the minimax door ([{float(x_min):.16g}, "
            f"{float(x_max):.16g}])")
    R = float(x_max) / float(x_min)
    limit = _SIGN_DEFINITE_PANE_MAX_RANGE * (
        1.0 + 8.0 * np.finfo(np.float64).eps)
    if R > limit:
        raise AssertionError(
            "GATE sign_definite_partition: unbounded Laplace cell reached "
            f"the minimax door (R={R:.16g} > "
            f"{_SIGN_DEFINITE_PANE_MAX_RANGE:.16g})")


def _build_single_sigma_window(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_count: int,
    mask_B_min: float | None,
    mask_B_max: float | None,
    omega_nonneg_ry: np.ndarray,
    denom_can_cross: bool,
    neg_omega_half: bool,
    target_error: float,
    max_nodes: int,
    use_shipped_tables: bool,
    sign_definite_cells: list[_SignDefiniteCell] | None = None,
    defer_quadrature: bool = False,
) -> list[_SigmaWindow]:
    A_vals = E_A[base_mask_A]
    if A_vals.size == 0 or mask_B_count == 0 or mask_B_min is None or mask_B_max is None:
        return []
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    panes = ([
            (cell.B_lo, cell.B_hi, cell.B_count, cell.B_min, cell.B_max)
            for cell in sign_definite_cells
        ] if sign_definite_cells is not None else
        [(-np.inf, np.inf, mask_B_count, mask_B_min, mask_B_max)])
    windows: list[_SigmaWindow] = []
    for pane_idx, (B_lo, B_hi, count_B, B_min, B_max) in enumerate(panes):
        if count_B == 0:
            continue
        cell = (None if sign_definite_cells is None
                else sign_definite_cells[pane_idx])
        if cell is not None:
            x_min, x_max = cell.x_min, cell.x_max
            E_ref_A = cell.E_min
            E_lo = None if np.isneginf(cell.E_lo) else cell.E_lo
            E_hi = None if np.isposinf(cell.E_hi) else cell.E_hi
            omega_indices = (
                None if cell.omega_indices.size == omega_nonneg_ry.size
                else cell.omega_indices)
        else:
            S_min = float(np.min(A_vals) + B_min)
            S_max = float(np.max(A_vals) + B_max)
            if denom_can_cross:
                x_min = max(S_min, 1.0e-12)
                x_max = max(S_max, x_min * (1.0 + 1.0e-9))
            else:
                x_min, x_max = _sign_definite_support(
                    float(np.min(A_vals)), float(np.max(A_vals)),
                    B_min, B_max, omega_max)
            E_ref_A = float(np.min(A_vals))
            E_lo = E_hi = omega_indices = None
        _assert_bounded_laplace_support(x_min, x_max)
        if defer_quadrature:
            nodes = _deferred_box_nodes()
            max_error = None
            provenance = "uniform denominator box pending"
        else:
            q = solve_laplace_minimax_interval(
                x_min, x_max,
                target_error=target_error,
                max_nodes=max_nodes,
                use_shipped_tables=use_shipped_tables,
            )
            nodes = q.to_minimax_nodes(time_axis='imag')
            max_error = float(q.max_error)
            provenance = q.provenance
        omega_sign = +1 if denom_can_cross else -1
        prefactor = ((1.0 if denom_can_cross else -1.0)
                     * (-1.0 if neg_omega_half else 1.0))
        one_pane = len(panes) == 1
        windows.append(_SigmaWindow(
            name="single" if one_pane else f"single_pane_{pane_idx}",
            nodes=nodes,
            mask_A=np.asarray(base_mask_A, dtype=bool),
            E_ref_A=float(E_ref_A),
            E_ref_B=float(B_min),
            omega_sign=omega_sign,
            project="full",
            prefactor=float(prefactor),
            mask_B_mode="all",
            max_error=max_error,
            provenance=provenance,
            B_lo=None if one_pane else float(B_lo),
            B_hi=None if one_pane else float(B_hi),
            E_min=E_lo,
            E_max=E_hi,
            omega_indices=omega_indices,
        ))
    return windows


def _plan_sign_definite_omega_panes(
    *,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    mask_B_count: int,
    mask_B_min: float,
    mask_B_max: float,
    E_min: float,
    E_max: float,
    omega_max: float,
    omega_min: float = 0.0,
    orientation: str = "E+B+omega",
    support_lo: float = -np.inf,
    support_hi: float = np.inf,
) -> list[tuple[float, float, int, float, float]]:
    """Make exact scalar Ω panes when one sign-definite range is costly.

    Inverse-Laplace range cost is logarithmic in ``R=x_max/x_min``.  A
    non-singleton pane is recursively bounded to eight binary octaves,
    independent of the minimax catalog and node budget; a singleton is handed
    to :func:`_plan_sign_definite_cells` for exact E/evaluation-omega
    continuation.  The two ``E+B`` forms use the balanced continuous-support
    cut; the reversed HGL flank uses its exact midpoint cut.

    Splits use the existing ``(lo, hi]`` convention and scalar reduction.  No
    pole-sized host array or retained mask is formed.  Count conservation
    proves every live Ω lane is owned exactly once.  Giving every disjoint
    pane the original per-denominator L-infinity kernel tolerance preserves
    that tolerance on their disjoint union: the union error is the maximum
    pane error.  This is not a claim that errors in the final residue-weighted
    Sigma matrix sum by maximum; its amplification bound is unchanged from
    the unpartitioned per-denominator contract.
    """
    pending = [(float(support_lo), float(support_hi), int(mask_B_count),
                float(mask_B_min), float(mask_B_max))]
    panes: list[tuple[float, float, int, float, float]] = []
    while pending:
        lo, hi, count, B_min, B_max = pending.pop()
        x_min, x_max = _oriented_sign_definite_support(
            E_min, E_max, B_min, B_max, omega_min, omega_max,
            orientation)
        if (x_max / x_min <= _SIGN_DEFINITE_PANE_MAX_RANGE
                or B_min >= B_max):
            panes.append((lo, hi, count, B_min, B_max))
            continue

        if orientation == "E+B-omega":
            a = float(E_min) - float(omega_max)
            C = float(E_max) - float(omega_min)
            disc = (a - C) ** 2 + 4.0 * (C + B_max) * (a + B_min)
            threshold = 0.5 * (-(a + C) + np.sqrt(max(disc, 0.0)))
        elif orientation == "E+B+omega":
            a = float(E_min) + float(omega_min)
            C = float(E_max) + float(omega_max)
            disc = (a - C) ** 2 + 4.0 * (C + B_max) * (a + B_min)
            threshold = 0.5 * (-(a + C) + np.sqrt(max(disc, 0.0)))
        elif orientation == "omega-E-B":
            threshold = B_min + 0.5 * (B_max - B_min)
        else:
            raise ValueError(
                f"Unknown sign-definite orientation {orientation!r}")
        threshold = max(float(np.nextafter(B_min, B_max)), threshold)
        threshold = min(float(np.nextafter(B_max, B_min)), threshold)
        if not B_min < threshold < B_max:
            # Adjacent finite floats have no representable interior.  The
            # actual minimum is still a legal (lo,hi] boundary: it assigns all
            # equal-min lanes left and every strictly larger lane right.
            threshold = B_min
        if not lo < threshold < hi:
            raise AssertionError(
                "GATE sign_definite_omega_partition: no legal strict cut "
                f"inside support ({lo!r}, {hi!r}]")

        left = _masked_interval_stats_device(
            Omega_q, base_mask_B, lo, threshold)
        right = _masked_interval_stats_device(
            Omega_q, base_mask_B, threshold, hi)
        children = [
            (lo, threshold, left[1], left[2], left[3]),
            (threshold, hi, right[1], right[2], right[3]),
        ]
        live = [p for p in children if p[2] > 0]
        owned = sum(p[2] for p in live)
        if owned != count:
            raise AssertionError(
                "GATE sign_definite_omega_partition: pane counts do not "
                f"conserve live poles ({owned} != {count})")
        if len(live) != 2 or any(p[2] >= count for p in live):
            raise AssertionError(
                "GATE sign_definite_omega_partition: range split made no "
                "strict progress")
        pending.extend(reversed(live))

    panes.sort(key=lambda p: p[0])
    if sum(p[2] for p in panes) != int(mask_B_count):
        raise AssertionError(
            "GATE sign_definite_omega_partition: final panes do not own "
            "every live pole exactly once")
    return panes


def _bisect_discrete_axis(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, tuple[tuple[np.ndarray, float, float], ...]]:
    """Bisect one finite discrete support without losing repeated values."""
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if not indices.size:
        raise AssertionError("cannot bisect empty discrete support")
    selected = values[indices]
    value_min = float(np.min(selected))
    value_max = float(np.max(selected))
    if not value_min < value_max:
        raise AssertionError("discrete-axis bisection needs nonzero support")
    cut = value_min + 0.5 * (value_max - value_min)
    cut = max(float(np.nextafter(value_min, value_max)), cut)
    cut = min(float(np.nextafter(value_max, value_min)), cut)
    if not value_min < cut < value_max:
        # Adjacent finite floats have no representable interior; the actual
        # minimum remains a legal ``(lo,hi]`` cut and keeps its repeats left.
        cut = value_min
    rows = []
    for select in (selected <= cut, selected > cut):
        child = indices[select]
        if child.size:
            child_values = values[child]
            rows.append((child, float(np.min(child_values)),
                         float(np.max(child_values))))
    if (len(rows) != 2
            or any(row[0].size >= indices.size for row in rows)):
        raise AssertionError("discrete-axis bisection made no strict progress")
    return cut, tuple(rows)


def _plan_sign_definite_cells(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    mask_B_count: int,
    mask_B_min: float,
    mask_B_max: float,
    omega_nonneg_ry: np.ndarray,
    orientation: str,
    support_B_lo: float = -np.inf,
    support_B_hi: float = np.inf,
    support_E_lo: float = -np.inf,
    support_E_hi: float = np.inf,
    omega_indices: np.ndarray | None = None,
) -> list[_SignDefiniteCell]:
    """Bound every exact sign-definite Sigma cell to eight octaves.

    The incumbent Omega-pane owner is the first cut for the two ``E+B``
    orientations.  A constant-Omega pane can still be wide because of its
    band-energy or evaluation-frequency support, so this routine continues
    that *same* exact partition over those already-supported scalar selectors.
    The ``omega-E-B`` HGL flank uses the identical recursion, with midpoint
    Omega cuts because its denominator orientation is reversed.

    ``_energy_panes`` and ``_omega_clusters`` remain the one fixed-width
    packing idiom for an HGL crossing shell.  This continuation is
    ratio-adaptive instead: globally crossing those two covers would run every
    energy pane against every frequency pane even where the denominator is
    already cheap.  The shared discrete-axis bisection below therefore acts
    only on a proved over-cap Cartesian cell and stops as soon as that cell is
    bounded.

    No pole, residue, energy, or evaluation point is changed.  Recursive
    children are disjoint Cartesian cells, and terminal singleton support has
    ``R=1``; therefore a finite discrete input must terminate with every
    ``x_max/x_min <= _SIGN_DEFINITE_PANE_MAX_RANGE``.
    """
    energies = np.asarray(E_A, dtype=np.float64)
    base_A = np.asarray(base_mask_A, dtype=bool)
    if energies.shape != base_A.shape:
        raise ValueError("E_A and base_mask_A must have identical shapes")
    selected_A = (base_A & (energies > float(support_E_lo))
                  & (energies <= float(support_E_hi)))
    A_vals = energies[selected_A]
    omega = np.asarray(omega_nonneg_ry, dtype=np.float64)
    idx_all = (np.arange(omega.size, dtype=np.int64)
               if omega_indices is None
               else np.asarray(omega_indices, dtype=np.int64))
    if A_vals.size == 0 or int(mask_B_count) == 0 or idx_all.size == 0:
        return []
    if not np.all(np.isfinite(A_vals)) or not np.all(np.isfinite(omega[idx_all])):
        raise ValueError("sign-definite support contains a non-finite scalar")

    E_min0, E_max0 = float(np.min(A_vals)), float(np.max(A_vals))
    w_min0 = float(np.min(omega[idx_all]))
    w_max0 = float(np.max(omega[idx_all]))

    B_panes = _plan_sign_definite_omega_panes(
        Omega_q=Omega_q,
        base_mask_B=base_mask_B,
        mask_B_count=int(mask_B_count),
        mask_B_min=float(mask_B_min),
        mask_B_max=float(mask_B_max),
        E_min=E_min0,
        E_max=E_max0,
        omega_min=w_min0,
        omega_max=w_max0,
        orientation=orientation,
        support_lo=float(support_B_lo),
        support_hi=float(support_B_hi),
    )

    pending = []
    E_indices0 = np.arange(A_vals.size, dtype=np.int64)
    for B_lo, B_hi, count_B, B_min, B_max in B_panes:
        pending.append((
            float(support_E_lo), float(support_E_hi), E_indices0,
            E_min0, E_max0,
            float(B_lo), float(B_hi), int(count_B), float(B_min), float(B_max),
            np.asarray(idx_all, dtype=np.int64), w_min0, w_max0,
        ))

    cells: list[_SignDefiniteCell] = []
    while pending:
        (E_lo, E_hi, E_idx, E_min, E_max, B_lo, B_hi, count_B, B_min,
         B_max, w_idx, w_min, w_max) = pending.pop()
        x_min, x_max = _oriented_sign_definite_support(
            E_min, E_max, B_min, B_max, w_min, w_max, orientation)
        if x_max / x_min <= _SIGN_DEFINITE_PANE_MAX_RANGE:
            cells.append(_SignDefiniteCell(
                E_lo, E_hi, E_min, E_max,
                B_lo, B_hi, count_B, B_min, B_max,
                np.asarray(w_idx, dtype=np.int64), w_min, w_max,
                x_min, x_max))
            continue

        if B_min != B_max:
            raise AssertionError(
                "GATE sign_definite_partition: B owner returned an over-cap "
                "non-singleton pane")

        # Omega is already singleton here: the incumbent B planner only
        # returns an over-cap pane when B_min == B_max.  Split whichever of
        # the two remaining exact host axes contributes the larger width.
        e_span = E_max - E_min
        w_span = w_max - w_min
        if e_span <= 0.0 and w_span <= 0.0:
            raise AssertionError(
                "GATE sign_definite_partition: singleton Cartesian support "
                f"has irreducible range {x_max / x_min:.16g}")
        if e_span >= w_span and e_span > 0.0:
            cut, rows = _bisect_discrete_axis(A_vals, E_idx)
            bounds = ((E_lo, cut), (cut, E_hi))
            for (child_idx, child_min, child_max), (
                    child_lo, child_hi) in reversed(tuple(zip(rows, bounds))):
                pending.append((
                    child_lo, child_hi, child_idx, child_min, child_max,
                    B_lo, B_hi, count_B, B_min, B_max,
                    w_idx, w_min, w_max))
        else:
            _cut, rows = _bisect_discrete_axis(omega, w_idx)
            for child_idx, child_min, child_max in reversed(rows):
                pending.append((
                    E_lo, E_hi, E_idx, E_min, E_max,
                    B_lo, B_hi, count_B, B_min, B_max,
                    child_idx, child_min, child_max))

    expected = int(A_vals.size) * int(mask_B_count) * int(idx_all.size)
    owned = sum(
        int(np.sum((A_vals > c.E_lo) & (A_vals <= c.E_hi)))
        * int(c.B_count) * int(c.omega_indices.size)
        for c in cells)
    if owned != expected:
        raise AssertionError(
            "GATE sign_definite_partition: Cartesian ownership is not exact "
            f"({owned} != {expected})")
    cells.sort(key=lambda c: (int(c.omega_indices[0]), c.E_lo, c.B_lo))
    return cells


def _scaled_crossing_error_bound(
    regularization_width_ry: float,
    target_error: float,
) -> float:
    """Convert a physical HGL-kernel tolerance to dimensionless units.

    The crossing service approximates a dimensionless target ``G(u)``.  The
    Sigma consumer maps its rule to physical energy units with

    ``t = tau_hat / xi`` and ``alpha = alpha_hat / xi``.

    Hence an error ``eps_hat`` in the service rule becomes
    ``eps_phys = eps_hat / xi``.  The service request must be
    ``eps_hat <= eps_phys * xi``.  Reject invalid values rather than clipping
    the tolerance, since a floor would silently loosen the public physical
    contract.
    """

    xi = float(regularization_width_ry)
    target_error = float(target_error)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError(
            "regularization_width_ry must be finite and positive before "
            f"crossing-rule rescaling; got {xi!r}.")
    if not np.isfinite(target_error) or target_error <= 0.0:
        raise ValueError(
            "target_error must be a finite positive physical absolute "
            f"tolerance; got {target_error!r}.")
    scaled = target_error * xi
    if not np.isfinite(scaled) or scaled <= 0.0:
        raise ValueError(
            "Scaled crossing tolerance is not representable: "
            f"{target_error!r} * {xi!r} = {scaled!r}.")
    return scaled


def _build_three_sigma_windows(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_all_count: int,
    mask_B_le_count: int,
    mask_B_le_min: float | None,
    mask_B_le_max: float | None,
    mask_B_gt_count: int,
    mask_B_gt_min: float | None,
    mask_B_gt_max: float | None,
    omega_nonneg_ry: np.ndarray,
    neg_omega_half: bool,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    use_shipped_tables: bool,
    laplace_cells_by_name: dict[str, list[_SignDefiniteCell]] | None = None,
    defer_quadrature: bool = False,
) -> list[_SigmaWindow]:
    # This is the crossing branch: the pole S can coincide with a grid ω, so
    # the denominator is ω̃ − S and ω enters the kernel as exp(+i·ω·τ)
    # (omega_sign=+1) on every window.  The −ω half carries an overall −1,
    # folded into each window's prefactor via ``neg`` below.
    neg = -1.0 if neg_omega_half else 1.0
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    xi = max(float(regularization_width_ry), 1.0e-12)
    z_edge = float(edge_factor) * xi
    T = omega_max + z_edge
    windows: list[_SigmaWindow] = []

    for name in ("core", "a_stripe", "b_slab"):
        if name == "core":
            mA = base_mask_A & (E_A <= T)
            mask_B_mode = "le_t"
            count_B = mask_B_le_count
            B_min = mask_B_le_min
            B_max = mask_B_le_max
        elif name == "a_stripe":
            mA = base_mask_A & (E_A > T)
            mask_B_mode = "le_t"
            count_B = mask_B_le_count
            B_min = mask_B_le_min
            B_max = mask_B_le_max
        else:
            mA = base_mask_A
            mask_B_mode = "gt_t"
            count_B = mask_B_gt_count
            B_min = mask_B_gt_min
            B_max = mask_B_gt_max
        if not np.any(mA) or count_B == 0 or B_min is None or B_max is None:
            continue

        cells = (None if laplace_cells_by_name is None
                 else laplace_cells_by_name.get(name))
        pane_rows = ([
            (cell.B_lo, cell.B_hi, cell.B_count, cell.B_min, cell.B_max)
            for cell in cells
        ] if cells is not None else
            [(None, None, count_B, B_min, B_max)])
        for pane_idx, (B_lo, B_hi, pane_count, pane_min, pane_max) in enumerate(
                pane_rows):
            if pane_count == 0 or pane_min is None or pane_max is None:
                continue
            cell = None if cells is None else cells[pane_idx]
            A_vals = E_A[mA]
            E_ref_A = (float(np.min(A_vals)) if cell is None
                       else float(cell.E_min))
            E_ref_B = float(pane_min)

            if name == "core":
                A_core = max(2.0 * T / xi, 1.0e-8)
                if (not defer_quadrature
                        and A_core > _CROSSING_A_MAX * (
                        1.0 + 8.0 * np.finfo(np.float64).eps)):
                    raise AssertionError(
                        "GATE hgl_shell_capacity: unpartitioned HGL range "
                        f"A={A_core:.16g} exceeds {_CROSSING_A_MAX:.16g}")
                project = "imag"
                prefactor = -1.0 * neg
                if defer_quadrature:
                    nodes = _deferred_box_nodes()
                    max_error = None
                    provenance = "uniform denominator box pending"
                else:
                    target_error_hat = _scaled_crossing_error_bound(
                        xi, target_error)
                    q_cross = solve_phase_minimax_bandwidth(
                        A_core,
                        target_error=target_error_hat,
                        max_nodes=crossing_max_nodes,
                        eps_q=crossing_eps_q,
                        target_kind="hgl",
                        use_shipped_tables=use_shipped_tables,
                    )
                    raw = q_cross.to_minimax_nodes(
                        time_axis='crossing_hgl')
                    # Crossing scaling: t = τ/ξ, α = α/ξ.
                    nodes = MinimaxNodes(
                        t=raw.t / xi, alpha=raw.alpha / xi)
                    max_error = float(q_cross.max_error) / xi
                    provenance = q_cross.provenance
            else:
                if cell is None:
                    x_min, x_max = _sign_definite_support(
                        float(np.min(A_vals)), float(np.max(A_vals)),
                        float(pane_min), float(pane_max), omega_max,
                        subtract_omega=True, x_min_floor=z_edge)
                else:
                    x_min, x_max = cell.x_min, cell.x_max
                _assert_bounded_laplace_support(x_min, x_max)
                project = "full"
                prefactor = +1.0 * neg
                if defer_quadrature:
                    nodes = _deferred_box_nodes()
                    max_error = None
                    provenance = "uniform denominator box pending"
                else:
                    q = solve_laplace_minimax_interval(
                        x_min, x_max,
                        target_error=target_error,
                        max_nodes=max_nodes,
                        use_shipped_tables=use_shipped_tables,
                    )
                    nodes = q.to_minimax_nodes(time_axis='imag')
                    max_error = float(q.max_error)
                    provenance = q.provenance

            many_panes = len(pane_rows) > 1
            E_lo = (None if cell is None or np.isneginf(cell.E_lo)
                    else cell.E_lo)
            E_hi = (None if cell is None or np.isposinf(cell.E_hi)
                    else cell.E_hi)
            omega_indices = None
            if cell is not None and cell.omega_indices.size != omega_nonneg_ry.size:
                omega_indices = cell.omega_indices
            windows.append(
                _SigmaWindow(
                    name=(f"{name}_pane_{pane_idx}"
                          if many_panes else name),
                    nodes=nodes,
                    mask_A=np.asarray(mA, dtype=bool),
                    E_ref_A=E_ref_A,
                    E_ref_B=E_ref_B,
                    omega_sign=+1,
                    project=project,
                    prefactor=float(prefactor),
                    mask_B_mode=mask_B_mode,
                    mask_B_threshold=float(T),
                    crossing_kind="hgl" if name == "core" else None,
                    max_error=max_error,
                    provenance=provenance,
                    B_lo=float(B_lo) if many_panes else None,
                    B_hi=float(B_hi) if many_panes else None,
                    E_min=E_lo,
                    E_max=E_hi,
                    omega_indices=omega_indices,
                )
            )
    return windows


def _build_partitioned_hgl_windows(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    omega_nonneg_ry: np.ndarray,
    neg_omega_half: bool,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    use_shipped_tables: bool,
    defer_quadrature: bool = False,
) -> tuple[list[_SigmaWindow], HGLCrossingPlan]:
    """Build certified rules for the exact HGL pane plan.

    This is the execution leaf for :func:`plan_hgl_crossing_cells`.  It adds
    no second convolution or accumulator: every returned row is the existing
    :class:`_SigmaWindow` vocabulary, augmented only with scalar A/B bounds
    and its omega-cluster indices.

    The two sign-definite orientations follow directly from
    ``x = omega - E - B``.  ``x < 0`` uses the incumbent ``t=-i*tau``
    placement anchored at the A/B minima.  ``x > 0`` uses its conjugate
    ``t=+i*tau`` placement anchored at the maxima, so both factorised
    exponentials have magnitude at most one.  The shell uses the incumbent
    HGL rule and physical ``eps_hat = xi*eps_phys`` conversion unchanged.
    """
    from common.units import RYD_TO_EV

    xi = float(regularization_width_ry)
    plan = plan_hgl_crossing_cells(
        omega_abs=np.asarray(omega_nonneg_ry, dtype=np.float64),
        E_A=E_A,
        base_mask_A=base_mask_A,
        regularization_width_ry=xi,
        edge_factor=float(edge_factor),
        omega_cluster_gap_ry=1.0,
        omega_max_span_ry=1.0 / RYD_TO_EV,
        crossing_A_max=_CROSSING_A_MAX,
    )
    neg = -1.0 if neg_omega_half else 1.0
    base = np.asarray(base_mask_A, dtype=bool)
    stats_cache: dict[tuple[float, float], tuple[int, int, float | None, float | None]] = {}
    windows: list[_SigmaWindow] = []

    for cell in plan.cells:
        b_key = (float(cell.b_lo), float(cell.b_hi))
        stats = stats_cache.get(b_key)
        if stats is None:
            stats = _masked_interval_stats_device(
                Omega_q, base_mask_B, cell.b_lo, cell.b_hi)
            stats_cache[b_key] = stats
        _, count_B, B_min, B_max = stats
        if count_B == 0 or B_min is None or B_max is None:
            continue

        laplace_cells = None
        if cell.kind != "crossing":
            laplace_cells = _plan_sign_definite_cells(
                E_A=E_A,
                base_mask_A=base,
                Omega_q=Omega_q,
                base_mask_B=base_mask_B,
                mask_B_count=count_B,
                mask_B_min=float(B_min),
                mask_B_max=float(B_max),
                omega_nonneg_ry=np.asarray(omega_nonneg_ry, dtype=np.float64),
                orientation=("E+B-omega" if cell.kind == "negative"
                             else "omega-E-B"),
                support_B_lo=float(cell.b_lo),
                support_B_hi=float(cell.b_hi),
                support_E_lo=float(cell.e_lo),
                support_E_hi=float(cell.e_hi),
                omega_indices=np.asarray(cell.omega_indices, dtype=np.int64),
            )
        pane_rows = (
            [(cell.b_lo, cell.b_hi, count_B, B_min, B_max)]
            if laplace_cells is None else
            [(row.B_lo, row.B_hi, row.B_count, row.B_min, row.B_max)
             for row in laplace_cells]
        )

        for pane_idx, (B_lo, B_hi, pane_count, pane_min, pane_max) in enumerate(
                pane_rows):
            if pane_count == 0 or pane_min is None or pane_max is None:
                continue
            sign_cell = (None if laplace_cells is None
                         else laplace_cells[pane_idx])
            if cell.kind == "crossing":
                if (not defer_quadrature
                        and float(cell.A_dim) > _CROSSING_A_MAX * (
                        1.0 + 8.0 * np.finfo(np.float64).eps)):
                    raise AssertionError(
                        "GATE hgl_shell_capacity: planned HGL range "
                        f"A={float(cell.A_dim):.16g} exceeds "
                        f"{_CROSSING_A_MAX:.16g}")
                E_ref_A = float(cell.e_min)
                E_ref_B = float(pane_min)
                project = "imag"
                prefactor = -1.0 * neg
                crossing_kind = "hgl"
                if defer_quadrature:
                    nodes = _deferred_box_nodes()
                    max_error = None
                    provenance = "uniform denominator box pending"
                else:
                    target_error_hat = _scaled_crossing_error_bound(
                        xi, target_error)
                    q_cross = solve_phase_minimax_bandwidth(
                        float(cell.A_dim),
                        target_error=target_error_hat,
                        max_nodes=crossing_max_nodes,
                        eps_q=crossing_eps_q,
                        target_kind="hgl",
                        use_shipped_tables=use_shipped_tables,
                    )
                    raw = q_cross.to_minimax_nodes(
                        time_axis="crossing_hgl")
                    nodes = MinimaxNodes(
                        t=raw.t / xi, alpha=raw.alpha / xi)
                    max_error = float(q_cross.max_error) / xi
                    provenance = q_cross.provenance
            else:
                x_min, x_max = sign_cell.x_min, sign_cell.x_max
                if cell.kind == "negative":
                    E_ref_A = float(sign_cell.E_min)
                    E_ref_B = float(sign_cell.B_min)
                    prefactor = +1.0 * neg
                    conjugate = False
                elif cell.kind == "positive":
                    E_ref_A = float(sign_cell.E_max)
                    E_ref_B = float(sign_cell.B_max)
                    prefactor = -1.0 * neg
                    conjugate = True
                else:
                    raise AssertionError(
                        f"Unknown HGL cell kind {cell.kind!r}")
                if x_min <= 0.0 or x_max < x_min:
                    raise AssertionError(
                        "GATE hgl_sign_definite_cell: invalid Laplace interval "
                        f"for {cell.kind}: [{x_min:.16g}, {x_max:.16g}]")
                _assert_bounded_laplace_support(x_min, x_max)
                project = "full"
                crossing_kind = None
                if defer_quadrature:
                    nodes = _deferred_box_nodes()
                    max_error = None
                    provenance = "uniform denominator box pending"
                else:
                    q = solve_laplace_minimax_interval(
                        x_min,
                        max(x_max, x_min * (1.0 + 1.0e-9)),
                        target_error=target_error,
                        max_nodes=max_nodes,
                        use_shipped_tables=use_shipped_tables,
                    )
                    raw = q.to_minimax_nodes(time_axis="imag")
                    nodes = (MinimaxNodes(t=-raw.t, alpha=raw.alpha)
                             if conjugate else raw)
                    max_error = float(q.max_error)
                    provenance = q.provenance

            windows.append(_SigmaWindow(
                name=f"pane_{cell.kind}",
                nodes=nodes,
                mask_A=base,
                E_ref_A=E_ref_A,
                E_ref_B=E_ref_B,
                omega_sign=+1,
                project=project,
                prefactor=float(prefactor),
                crossing_kind=crossing_kind,
                max_error=max_error,
                provenance=provenance,
                E_min=(float(cell.e_lo) if sign_cell is None
                       else (None if np.isneginf(sign_cell.E_lo)
                             else float(sign_cell.E_lo))),
                E_max=(float(cell.e_hi) if sign_cell is None
                       else (None if np.isposinf(sign_cell.E_hi)
                             else float(sign_cell.E_hi))),
                B_lo=float(B_lo),
                B_hi=float(B_hi),
                omega_indices=(np.asarray(cell.omega_indices, dtype=np.int64)
                               if sign_cell is None else
                               np.asarray(sign_cell.omega_indices,
                                          dtype=np.int64)),
            ))

    return windows, plan


# ---------------------------------------------------------------------------
#  Sigma convolution — host-side window construction.  The device-side tau
#  loop lives in the driver (ppm_sigma); the two halves have no shared state
#  beyond the window list itself, so they're easy to test independently.
# ---------------------------------------------------------------------------

def _build_windows_for_branch(
    *,
    omega_nonneg_ry: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
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
    use_shipped_minimax_tables: bool,
    log_tag: str,
    print_fn,
    partition_hgl: bool = False,
    defer_quadrature: bool = False,
) -> list[_SigmaWindow]:
    """Host-side window construction for a single branch.

    Gathers E_A and base_mask_A to host, computes masked B-side stats, and
    picks either the three-window crossing+stripe+slab decomposition (when the
    pole S can coincide with a grid ω) or a single sign-definite Laplace window
    (otherwise, or when the ω range is negligible).  Prints a one-line summary
    per returned window.

    Which case applies is a physical fact about the branch, not a stored sign:
    the pole S = E_A + Ω crosses the evaluation point (denominator ω̃ − S) for
    the conduction (empty) A-space on the +ω half and for the valence
    (occupied) A-space on the −ω half; otherwise the denominator is the
    sign-definite ω̃ + S.
    """
    if omega_nonneg_ry.size == 0:
        return []

    E_A_host = _to_host_np(E_A, dtype=np.float64, tiled=False)
    base_A_host = _to_host_np(base_mask_A, dtype=bool, tiled=False)

    _, mask_B_all_count, mask_B_all_min, mask_B_all_max = _masked_stats_device(
        Omega_q, base_mask_B)

    denom_can_cross = (space == "cond") != neg_omega_half
    omega_max = float(np.max(omega_nonneg_ry))
    if denom_can_cross and omega_max > 1.0e-14:
        xi = max(float(regularization_width_ry), 1.0e-12)
        T = omega_max + float(edge_factor) * xi
        if partition_hgl and hgl_partition_required(
                omega_nonneg_ry, xi, edge_factor):
            windows, plan = _build_partitioned_hgl_windows(
                E_A=E_A_host,
                base_mask_A=base_A_host,
                Omega_q=Omega_q,
                base_mask_B=base_mask_B,
                omega_nonneg_ry=omega_nonneg_ry,
                neg_omega_half=neg_omega_half,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor,
                target_error=target_error,
                max_nodes=max_nodes,
                crossing_eps_q=crossing_eps_q,
                crossing_max_nodes=crossing_max_nodes,
                use_shipped_tables=bool(use_shipped_minimax_tables),
                defer_quadrature=defer_quadrature,
            )
            counts = {
                kind: sum(win.name == f"pane_{kind}" for win in windows)
                for kind in ("positive", "crossing", "negative")
            }
            print_fn(
                f"    {log_tag} bounded HGL plan: "
                f"{plan.omega_cluster_count} omega clusters, "
                f"{plan.energy_pane_count} A panes, "
                f"{len(windows)} windows from {len(plan.cells)} planned cells "
                f"(+{counts['positive']}/crossing{counts['crossing']}"
                f"/-{counts['negative']}), max A={plan.max_A_dim:.6f}")
        else:
            _, mask_B_le_count, mask_B_le_min, mask_B_le_max = _masked_stats_device(
                Omega_q, base_mask_B & (Omega_q <= T))
            _, mask_B_gt_count, mask_B_gt_min, mask_B_gt_max = _masked_stats_device(
                Omega_q, base_mask_B & (Omega_q > T))
            laplace_cells_by_name = {}
            for name, mask_A, count_B, B_min, B_max, B_lo, B_hi in (
                ("a_stripe", base_A_host & (E_A_host > T),
                 mask_B_le_count, mask_B_le_min, mask_B_le_max, -np.inf, T),
                ("b_slab", base_A_host,
                 mask_B_gt_count, mask_B_gt_min, mask_B_gt_max, T, np.inf),
            ):
                if (np.any(mask_A) and count_B and B_min is not None
                        and B_max is not None):
                    laplace_cells_by_name[name] = _plan_sign_definite_cells(
                        E_A=E_A_host,
                        base_mask_A=mask_A,
                        Omega_q=Omega_q,
                        base_mask_B=base_mask_B,
                        mask_B_count=count_B,
                        mask_B_min=float(B_min),
                        mask_B_max=float(B_max),
                        omega_nonneg_ry=omega_nonneg_ry,
                        orientation="E+B-omega",
                        support_B_lo=B_lo,
                        support_B_hi=B_hi,
                    )
            windows = _build_three_sigma_windows(
                E_A=E_A_host, base_mask_A=base_A_host,
                mask_B_all_count=mask_B_all_count,
                mask_B_le_count=mask_B_le_count,
                mask_B_le_min=mask_B_le_min, mask_B_le_max=mask_B_le_max,
                mask_B_gt_count=mask_B_gt_count,
                mask_B_gt_min=mask_B_gt_min, mask_B_gt_max=mask_B_gt_max,
                omega_nonneg_ry=omega_nonneg_ry,
                neg_omega_half=neg_omega_half,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor,
                target_error=target_error, max_nodes=max_nodes,
                crossing_eps_q=crossing_eps_q,
                crossing_max_nodes=crossing_max_nodes,
                use_shipped_tables=bool(use_shipped_minimax_tables),
                laplace_cells_by_name=laplace_cells_by_name,
                defer_quadrature=defer_quadrature,
            )
    else:
        A_live = E_A_host[base_A_host]
        sign_definite_cells = None
        if (A_live.size and mask_B_all_count and
                mask_B_all_min is not None and mask_B_all_max is not None):
            # A crossing-capable branch with a negligible omega range skips
            # the HGL shell, but its denominator does not change identity:
            # the executor below still uses omega_sign=+1 and therefore
            # E + Omega - omega.  Keep the exact cell support oriented to the
            # same physical denominator even at the 1e-14 dispatch boundary.
            orientation = ("E+B-omega" if denom_can_cross
                           else "E+B+omega")
            sign_definite_cells = _plan_sign_definite_cells(
                E_A=E_A_host,
                base_mask_A=base_A_host,
                Omega_q=Omega_q,
                base_mask_B=base_mask_B,
                mask_B_count=mask_B_all_count,
                mask_B_min=float(mask_B_all_min),
                mask_B_max=float(mask_B_all_max),
                omega_nonneg_ry=omega_nonneg_ry,
                orientation=orientation,
            )
        windows = _build_single_sigma_window(
            E_A=E_A_host, base_mask_A=base_A_host,
            mask_B_count=mask_B_all_count,
            mask_B_min=mask_B_all_min, mask_B_max=mask_B_all_max,
            omega_nonneg_ry=omega_nonneg_ry,
            denom_can_cross=denom_can_cross,
            neg_omega_half=neg_omega_half,
            target_error=target_error, max_nodes=max_nodes,
            use_shipped_tables=bool(use_shipped_minimax_tables),
            sign_definite_cells=sign_definite_cells,
            defer_quadrature=defer_quadrature,
        )

    for win in windows:
        A_vals = E_A_host[win.mask_A]
        if win.E_min is not None or win.E_max is not None:
            E_lo = -np.inf if win.E_min is None else win.E_min
            E_hi = np.inf if win.E_max is None else win.E_max
            A_vals = A_vals[(A_vals > E_lo) & (A_vals <= E_hi)]
        kind = "crossing" if win.crossing_kind else "Laplace"
        # ACHIEVED, not requested (R2).  The old line said
        # ``err<{target_error:.0e}``, which is what was ASKED for -- so a
        # window served by an uncertified solve that missed its target by
        # orders printed exactly the same string as one served by a
        # certified table that met it.
        achieved = (
            "uniform-box rule pending" if defer_quadrature else
            "err~unrecorded" if win.max_error is None
            else f"err~{win.max_error:.2e} (asked <{target_error:.0e})")
        node_count = ("pending" if defer_quadrature else str(win.n_tau))
        print_fn(
            f"    {log_tag} window \"{win.name}\" ({kind}): "
            f"{node_count} nodes, {achieved}, "
            f"E_A=[{float(np.min(A_vals)):.4f}, {float(np.max(A_vals)):.4f}] Ry, "
            f"project={win.project}"
            f"  [{win.provenance or 'provenance unrecorded'}]"
        )
    return windows
