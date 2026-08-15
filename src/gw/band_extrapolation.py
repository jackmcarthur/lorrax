"""Band-convergence extrapolation of the correlation self-energy Σ_c.

WHAT CONVERGES SLOWLY.  Σ_c's intermediate-state sum runs over every band in
the Green's function,

    Σ_c(ω) = Σ_n  ψ_n ⊗ ψ_n*  ⊗  [W-dependent kernel](ω − E_n),

and its unoccupied tail decays like 1/N: the states above the QP window
contribute a small, same-signed, slowly-vanishing amount that a brute-force
band count only removes by being enormous.  This module evaluates the SAME
sum at three band counts in ONE pass and extrapolates to N → ∞ instead.

THE THREE POINTS COME FROM DISJOINT BRACKETS, NOT THREE RUNS.  The band axis
is cut into three contiguous brackets

    bracket 0   [0, N₁)          occupied + the first 50 % of the conduction bands
    bracket 1   [N₁, N₂)         50 % → 75 %
    bracket 2   [N₂, N₃)         75 % → 100 %

and the τ kernel builds one G(τ) per bracket, contracting each against the
SAME, singly-computed W(τ).  Because the brackets PARTITION the band sum, a
cumulative sum along the bracket axis is the sum at each cut:

    S(N₁) = b₀,   S(N₂) = b₀ + b₁,   S(N₃) = b₀ + b₁ + b₂,

and S(N₃) is — to floating-point associativity — the ordinary full-band Σ_c.
Nothing is computed twice and no band is dropped, which is the property
``tests/test_band_extrapolation.py::test_brackets_partition_the_band_sum``
exists to defend.

WHAT MUST BE HELD FIXED ACROSS THE THREE POINTS, or the fit measures
something other than band convergence:

  * **W is built once.**  χ₀/W and the PPM pole fit happen before any
    bracket exists; only the A-side (Green's function) band range varies.
  * **One ISDF representation.**  Centroids and ζ are fitted once at the
    largest band range; the smaller points are obtained by RESTRICTING the
    band index, never by refitting — a regenerated ISDF basis would mix
    basis error into what is meant to be pure band-sum error.
  * **One quadrature.**  The minimax windows, their τ nodes, E_ref_A/E_ref_B
    and the ω grid are built from the FULL band range and shared verbatim;
    the bracket enters only as a band-index restriction inside the kernel.
  * **One evaluation energy.**  All three points are read off the same ω
    grid at the same E_nk.
  * **Σ_c only.**  Σ_x is a bare-exchange sum over OCCUPIED states; it has no
    slow unoccupied tail and is not extrapolated.

THE FIT HAS EXACTLY TWO FREE PARAMETERS.

    S(N) = S_∞ + A / N_eff,        N_eff = N_occ + N_c = N

fitted by ordinary least squares in 1/N over the three points.  The shifted
three-parameter form ``A/(N − N₀)`` with N₀ free is DELIBERATELY NOT USED:
three parameters against three points interpolates exactly, has zero
residual by construction, and therefore tests nothing.  With two parameters
the third point is a genuine check, which is what the pairwise intercepts
below report.

THE DIAGNOSTICS ARE THE POINT.  A two-parameter fit through three points
always returns a number; whether that number means anything is decided by
whether the three points are already in the asymptotic 1/N regime.  The
exact two-point intercepts

    S_∞^(ij) = (N_j·S_j − N_i·S_i) / (N_j − N_i)

are free (they are the closed-form solution of the same model on a pair) and
they answer exactly that: if the model held, all three would coincide.
``Δ_model`` is their spread about the fit and measures preasymptotic
curvature; ``Δ_tail`` is how much the extrapolation moved the largest
computed point.  ``Δ_model`` comparable to ``Δ_tail`` — or a sign reversal
between the (1,2) and (2,3) intervals — means the correction is not
resolved by these three counts and the extrapolation must not be trusted.

BAND COUNTS, NOT AN ENERGY CUTOFF.  The cut is a number of bands because
JAX's shapes are static: a band count is a compile-time slice, an energy
cutoff is not.  ``mean_energy_ev`` is reported per cut for readers who want
a cutoff-like number, and is a REPORTED quantity only — nothing keys off it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.band_degeneracy import (
    DEGENERACY_TOL_RY,
    BandWindowDegeneracyError,
    snap_cut_to_clean_boundary,
)
from common.units import RYD_TO_EV


#: The two interior sampling fractions of the CONDUCTION band count.  The
#: third point is always the full range, so it needs no fraction.
BRACKET_FRACTIONS: tuple[float, float] = (0.50, 0.75)


class BandExtrapolationRefused(ValueError):
    """``sigma_band_extrapolation`` was requested but cannot be honored.

    Raised by name rather than silently disabling the feature: a run that
    quietly did not extrapolate would report a converged-looking Σ with no
    indication that the thing being tested never happened.
    """


@dataclass(frozen=True)
class BandBracketPlan:
    """Which band ranges the τ kernel sums, and what each cumulative cut means.

    Attributes
    ----------
    bounds : tuple[tuple[int, int], ...]
        Half-open ``(lo, hi)`` band-index brackets, contiguous and covering
        ``[0, nb_padded)`` exactly.  Length 1 in the ordinary case.
    counts : tuple[int, ...]
        ``N_i`` — the LOGICAL band count reached after cumulating brackets
        ``0..i``.  Excludes the mesh pad bands, whose ψ is exactly zero and
        which therefore add nothing to any sum (``common/meta.py``).
    requested : tuple[int, ...]
        The unsnapped targets, for the log line.  Same length as ``counts``.
    n_occ : int
        Occupied bands in the Σ band sum (the ``N_occ`` of ``N_eff``).
    n_cond : int
        Unoccupied bands in the Σ band sum.
    mean_energy_ev : tuple[float, ...]
        Mean band energy over ``[0, N_i)``, in eV relative to nothing in
        particular — a cutoff-flavoured REPORTING number, per the module
        docstring.  Never consumed.
    enabled : bool
        False for the trivial single-bracket plan.
    """

    bounds: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]
    requested: tuple[int, ...]
    n_occ: int
    n_cond: int
    mean_energy_ev: tuple[float, ...]
    enabled: bool

    @property
    def n_brackets(self) -> int:
        return len(self.bounds)


def trivial_plan(nb_padded: int, n_occ: int, nb_logical: int) -> BandBracketPlan:
    """The ordinary (non-extrapolating) plan: ONE bracket over every band.

    This is what the default path runs, and it is a plan rather than a
    ``None`` so that the kernel, the accumulator and the Σ cube carry a
    length-1 leading bracket axis unconditionally — one code path, no
    ``if extrapolating`` fork anywhere below this module.
    """
    nb_padded = int(nb_padded)
    return BandBracketPlan(
        bounds=((0, nb_padded),),
        counts=(int(nb_logical),),
        requested=(int(nb_logical),),
        n_occ=int(n_occ),
        n_cond=int(nb_logical) - int(n_occ),
        mean_energy_ev=(float("nan"),),
        enabled=False,
    )


def plan_band_brackets(
    *,
    enabled: bool,
    enk_ry: np.ndarray,
    n_occ: int,
    nb_logical: int,
    nb_padded: int,
    tol_ry: float = DEGENERACY_TOL_RY,
    fractions: tuple[float, ...] = BRACKET_FRACTIONS,
) -> BandBracketPlan:
    """Build the bracket plan for one Σ stage.

    Parameters
    ----------
    enabled : bool
        The deck's ``sigma_band_extrapolation``.  False returns
        :func:`trivial_plan` — the length-1 axis, bit-identical default.
    enk_ry : (nk, nb) float array
        Band energies in Ry over the Σ band sum's band range, ascending in
        the band axis.  Only ``[:, :nb_logical]`` is read: the mesh pad
        bands carry a sentinel energy and zero ψ.
    n_occ : int
        Occupied bands in the Σ band sum (``b2 - b0``).
    nb_logical, nb_padded : int
        Real and mesh-padded band counts of the sum (``b4_user - b0`` and
        ``b4 - b0``).  The last bracket runs to ``nb_padded`` so the plan
        still covers every band the un-bracketed path summed; the pad bands
        contribute exactly zero, so ``counts`` stays logical.

    Raises
    ------
    BandExtrapolationRefused
        When extrapolation is requested and ``n_cond <= n_occ`` (the tail
        being extrapolated is then shorter than the occupied block and the
        1/N model has no room to be tested), or when degeneracy snapping
        collapses two of the three cuts onto each other.
    """
    n_occ = int(n_occ)
    nb_logical = int(nb_logical)
    nb_padded = int(nb_padded)
    n_cond = nb_logical - n_occ

    if not enabled:
        return trivial_plan(nb_padded, n_occ, nb_logical)

    # ── THE ACTIVATION GATE ─────────────────────────────────────────────
    # Requested-but-impossible REFUSES.  Silently running the ordinary path
    # would produce a log with no extrapolation block and a Σ that looks
    # converged, and the operator would have no way to tell the feature was
    # off (measurement-discipline rule 1: an ignored deck key is how a green
    # A/B comes to measure nothing).
    if n_cond <= n_occ:
        raise BandExtrapolationRefused(
            f"sigma_band_extrapolation = true, but the Σ_c band sum has "
            f"n_cond = {n_cond} unoccupied bands against n_occ = {n_occ} "
            f"occupied ones (nband = {nb_logical}).  The feature extrapolates "
            f"the UNOCCUPIED tail, and it is only meaningful when that tail "
            f"is the larger part of the sum: it requires n_cond > n_occ.  "
            f"Raise the deck's `nband` to at least {2 * n_occ + 1} "
            f"(n_occ is set by the electron count, not by a deck key), or "
            f"set sigma_band_extrapolation = false.")

    e = np.asarray(enk_ry, dtype=np.float64)[:, :nb_logical]
    if e.ndim != 2 or e.shape[1] != nb_logical:
        raise ValueError(
            f"plan_band_brackets: expected (nk, >={nb_logical}) energies, "
            f"got shape {np.shape(enk_ry)}")

    requested = tuple(int(round(n_occ + f * n_cond)) for f in fractions)
    # The interior cuts are snapped so no bracket boundary splits a
    # degenerate multiplet — the same constraint BerkeleyGW enforces on a
    # band-count convergence series, and the same one-number-per-boundary
    # test ``common.band_degeneracy`` already owns.  The top cut is the end
    # of the input and is not a cut at all.
    snapped: list[int] = []
    lo_bound = n_occ + 1
    for req in requested:
        cut = snap_cut_to_clean_boundary(
            e, req, tol_ry=tol_ry, lo=lo_bound, hi=nb_logical - 1)
        snapped.append(int(cut))
        lo_bound = int(cut) + 1
    counts = tuple(snapped) + (nb_logical,)

    if len(set(counts)) != len(counts) or list(counts) != sorted(counts):
        raise BandExtrapolationRefused(
            f"sigma_band_extrapolation: degeneracy snapping collapsed the "
            f"three band counts onto {counts} (requested "
            f"{requested + (nb_logical,)}).  Three DISTINCT, ascending counts "
            f"are required — two coincident points cannot determine a "
            f"two-parameter fit.  The spectrum has a multiplet wide enough to "
            f"swallow the gap between the sampling fractions at nband = "
            f"{nb_logical}; raise nband, or set "
            f"sigma_band_extrapolation = false.")

    bounds = tuple(
        (int(a), int(b)) for a, b in
        zip((0,) + counts[:-1], counts[:-1] + (nb_padded,)))
    mean_e = tuple(
        float(np.mean(e[:, :c])) * RYD_TO_EV for c in counts)
    return BandBracketPlan(
        bounds=bounds,
        counts=counts,
        requested=requested + (nb_logical,),
        n_occ=n_occ,
        n_cond=n_cond,
        mean_energy_ev=mean_e,
        enabled=True,
    )


# ---------------------------------------------------------------------------
#  The fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtrapolationFit:
    """Result of the two-parameter 1/N fit, plus its quality diagnostics.

    Every array field carries the CALLER's trailing shape (e.g. ``(nk, nb)``
    for a band-diagonal Σ_c); only the leading three-point axis is consumed.
    """

    counts: np.ndarray            # (3,) int   — N₁, N₂, N₃
    s_at_counts: np.ndarray       # (3, ...)   — S(N₁), S(N₂), S(N₃)
    s_inf: np.ndarray             # (...)      — least-squares intercept
    amplitude: np.ndarray         # (...)      — A, the 1/N coefficient
    pair_s_inf: dict              # {(i, j): (...)} exact two-point intercepts
    delta_tail: np.ndarray        # (...)      |S_∞^fit − S₃|
    delta_model: np.ndarray       # (...)      max_ij |S_∞^(ij) − S_∞^fit|
    residual: np.ndarray          # (...)      max_i |S_i − model(N_i)|

    def at(self, index) -> "ExtrapolationFit":
        """Restrict every field to a subset of the trailing state axes.

        The fit is elementwise in (k, band) — every state has its own two
        parameters — so a subset of it IS the fit of that subset, and this is
        an indexing operation rather than a refit.  It exists because an
        aggregate over ALL states is the wrong summary: Σ_c at the top of the
        QP window is both large and the worst-converged thing in the run, so
        a max over (k, band) reports that state's tail as if it were the
        calculation's, and its verdict as if it were the calculation's.  The
        states a GW run is FOR are the band edges.
        """
        # ``index`` addresses the TRAILING state axes; ``s_at_counts`` has the
        # three-point axis in front of them, so the tuple has to be SPLICED
        # after a full slice, not passed as one more index.  Wrapping it
        # instead — ``[(slice(None), index)]`` — makes numpy read a (k, n)
        # pair as fancy indexing along axis 1 and returns a (3, 2) array where
        # a scalar was meant; that reached a driver run.
        idx = index if isinstance(index, tuple) else (index,)
        return ExtrapolationFit(
            counts=self.counts,
            s_at_counts=self.s_at_counts[(slice(None),) + idx],
            s_inf=self.s_inf[index],
            amplitude=self.amplitude[index],
            pair_s_inf={k: v[index] for k, v in self.pair_s_inf.items()},
            delta_tail=self.delta_tail[index],
            delta_model=self.delta_model[index],
            residual=self.residual[index],
        )


def fit_band_extrapolation(
    counts, s_at_counts: np.ndarray) -> ExtrapolationFit:
    """Ordinary least squares of ``S(N) = S_∞ + A/N`` over three points.

    Parameters
    ----------
    counts : sequence of int, length 3
        ``N_eff`` at each point.  ``N_eff = N_occ + N_c`` is just the total
        band count of the sum, which is what the brackets cumulate to.
    s_at_counts : (3, ...) complex array
        The CUMULATIVE bracket sums — S(N₁), S(N₂), S(N₃) — with any
        trailing shape.  Complex is carried through: the model is linear, so
        fitting the complex value is exactly fitting Re and Im separately.

    Returns
    -------
    ExtrapolationFit
    """
    N = np.asarray(counts, dtype=np.float64)
    S = np.asarray(s_at_counts)
    if N.ndim != 1 or N.size < 2:
        raise ValueError(f"fit_band_extrapolation: need >=2 counts, got {N}")
    if S.shape[0] != N.size:
        raise ValueError(
            f"fit_band_extrapolation: S leading axis {S.shape[0]} != "
            f"{N.size} counts")

    x = 1.0 / N                                   # the regressor
    n = float(N.size)
    xbar = float(np.mean(x))
    # Ordinary least squares, written out rather than via lstsq so the
    # broadcast over the trailing (k, band) axes is explicit and allocation
    # free.  Sxx is a scalar; the covariance carries the trailing shape.
    Sxx = float(np.sum((x - xbar) ** 2))
    if Sxx <= 0.0:
        raise ValueError(
            "fit_band_extrapolation: the three band counts are degenerate "
            f"in 1/N ({N}) — no slope is determined.")
    xb = x.reshape((-1,) + (1,) * (S.ndim - 1))
    Sbar = np.mean(S, axis=0)
    Sxy = np.sum((xb - xbar) * (S - Sbar), axis=0)
    A = Sxy / Sxx
    s_inf = Sbar - A * xbar

    pair = {}
    for i in range(N.size):
        for j in range(i + 1, N.size):
            # Exact two-point solution of the same model:
            #   N_j·S_j − N_i·S_i = (N_j − N_i)·S_∞
            pair[(i, j)] = (N[j] * S[j] - N[i] * S[i]) / (N[j] - N[i])

    model = s_inf[None, ...] + A[None, ...] * xb
    residual = np.max(np.abs(S - model), axis=0)
    delta_tail = np.abs(s_inf - S[-1])
    delta_model = np.max(
        np.abs(np.stack([p - s_inf for p in pair.values()], axis=0)), axis=0)

    return ExtrapolationFit(
        counts=np.asarray(N, dtype=np.int64),
        s_at_counts=S,
        s_inf=s_inf,
        amplitude=A,
        pair_s_inf=pair,
        delta_tail=delta_tail,
        delta_model=delta_model,
        residual=residual,
    )


def trust_verdict(fit: ExtrapolationFit, *, ratio_warn: float = 0.35) -> str:
    """One line saying whether the three points support the 1/N model.

    Two failure signatures, both read off the REAL part (the part that moves
    a QP energy):

    * **sign reversal** — the (1,2) and (2,3) intervals imply corrections of
      opposite sign, i.e. the sum is not monotonically approaching anything
      on this range;
    * **Δ_model comparable to Δ_tail** — the pairwise intercepts disagree by
      an appreciable fraction of the correction being applied, so the fitted
      correction is smaller than its own model error.

    Reported as prose, not as a refusal: the numbers are the product, and a
    run that stops on a soft quality metric would be worse than one that
    prints the metric.
    """
    d_tail = float(np.max(np.abs(np.real(fit.delta_tail))))
    d_model = float(np.max(np.abs(np.real(fit.delta_model))))
    a12 = np.real(fit.pair_s_inf[(0, 1)] - fit.s_at_counts[1])
    a23 = np.real(fit.pair_s_inf[(1, 2)] - fit.s_at_counts[2])
    reversed_sign = bool(np.any(np.sign(a12) * np.sign(a23) < 0))
    ratio = d_model / d_tail if d_tail > 0.0 else float("inf")

    if reversed_sign:
        return (f"NOT TRUSTWORTHY — the (1,2) and (2,3) intervals imply "
                f"corrections of OPPOSITE SIGN on at least one state: the "
                f"band sum is not monotone in 1/N over these counts, so the "
                f"extrapolation is not measuring a tail.")
    if ratio > ratio_warn:
        return (f"NOT TRUSTWORTHY — Δ_model/Δ_tail = {ratio:.2f} > "
                f"{ratio_warn:.2f}: the pairwise intercepts disagree by an "
                f"appreciable fraction of the correction, so these three "
                f"counts do not resolve the 1/N tail.  Use more bands.")
    return (f"consistent — Δ_model/Δ_tail = {ratio:.2f}; the three pairwise "
            f"intercepts agree to well within the applied correction.")


def format_extrapolation_report(
    plan: BandBracketPlan,
    fit: ExtrapolationFit,
    *,
    states: "list[tuple[str, object]] | None" = None,
    label: str = "Sigma_c",
    unit: str = "eV",
    scale: float = 1.0,
) -> str:
    """The log block.  Numbers first; the layout is deliberately not final.

    The requirement this satisfies is that ONE log carries the full-band
    value and the extrapolated value side by side, with the three band
    counts, the fit parameters and every diagnostic — unambiguously, and
    without the reader having to reconstruct anything.

    ``states`` is ``[(label, index), ...]``: the individual (k, band) states
    to report in full, each on its own row with SIGNED values, plus its own
    verdict.  The aggregate row that follows is a max over every state and is
    deliberately labelled as an envelope, not as a result — on a real deck it
    is dominated by the top of the QP window, whose Σ_c is both the largest
    and the least converged quantity in the run.
    """
    def _sg(a):
        return float(np.real(np.asarray(a))) * scale

    def _mx(a):
        return float(np.max(np.abs(np.real(np.asarray(a))))) * scale

    lines = [
        f"  -- {label} band-convergence extrapolation "
        f"(S(N) = S_inf + A/N, 2 parameters, 3 points) --",
        f"     N_occ = {plan.n_occ}   N_cond = {plan.n_cond}   "
        f"fractions = {BRACKET_FRACTIONS}",
    ]
    for i, (req, got, (lo, hi), me) in enumerate(zip(
            plan.requested, plan.counts, plan.bounds, plan.mean_energy_ev)):
        snap = "" if req == got else f"  (requested {req}, snapped)"
        lines.append(
            f"     N{i + 1} = {got:5d}   bracket [{lo:5d}, {hi:5d})   "
            f"<E> = {me:9.3f} {unit}{snap}")

    for slabel, index in (states or []):
        f1 = fit.at(index)
        lines += [
            f"     [{slabel}]",
            f"       S(N1={plan.counts[0]}) = {_sg(f1.s_at_counts[0]):+12.6f}   "
            f"S(N2={plan.counts[1]}) = {_sg(f1.s_at_counts[1]):+12.6f}   "
            f"S(N3={plan.counts[2]}) = {_sg(f1.s_at_counts[2]):+12.6f} {unit}",
            f"       S(N3) [full {plan.counts[-1]}-band Sigma_c] = "
            f"{_sg(f1.s_at_counts[-1]):+12.6f} {unit}   ->   "
            f"S_inf = {_sg(f1.s_inf):+12.6f} {unit}   "
            f"(A = {_sg(f1.amplitude):+.4f} {unit}*band)",
            f"       S_inf^(12) = {_sg(f1.pair_s_inf[(0, 1)]):+12.6f}   "
            f"S_inf^(23) = {_sg(f1.pair_s_inf[(1, 2)]):+12.6f}   "
            f"S_inf^(13) = {_sg(f1.pair_s_inf[(0, 2)]):+12.6f} {unit}",
            f"       Delta_tail = {_mx(f1.delta_tail):.6f} {unit}   "
            f"Delta_model = {_mx(f1.delta_model):.6f} {unit}   "
            f"residual = {_mx(f1.residual):.3e} {unit}",
            f"       verdict: {trust_verdict(f1)}",
        ]

    lines += [
        f"     [envelope over ALL (k, band) of the QP window -- an upper "
        f"bound, NOT the result: it is set by the top of the window, whose "
        f"Sigma_c is the least converged quantity in the run]",
        f"       max|S(N3)| = {_mx(fit.s_at_counts[-1]):.6f}   "
        f"max|S_inf| = {_mx(fit.s_inf):.6f} {unit}",
        f"       max Delta_tail = {_mx(fit.delta_tail):.6f} {unit}   "
        f"max Delta_model = {_mx(fit.delta_model):.6f} {unit}   "
        f"max residual = {_mx(fit.residual):.3e} {unit}",
        f"       verdict: {trust_verdict(fit)}",
    ]
    return "\n".join(lines)


__all__ = [
    "BRACKET_FRACTIONS",
    "BandBracketPlan",
    "BandExtrapolationRefused",
    "ExtrapolationFit",
    "fit_band_extrapolation",
    "format_extrapolation_report",
    "plan_band_brackets",
    "trivial_plan",
    "trust_verdict",
]
