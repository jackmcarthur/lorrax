"""Downfold: compress V and W from a large ISDF centroid basis to a small one.

WHAT THIS IS
------------
A production GW run fits a LARGE ISDF basis over a wide band window (μ_L
centroids, sized for Σ over hundreds of bands).  A BSE consumer only needs a
few dozen bands and wants the same physics in a SMALL basis (μ_S).  This
module compresses ``V_q[μν]`` and ``W_q[μν]`` from the first to the second so
that the physical observable

    𝒲[a,b] = ∫dr dr' conj(ρ_a(r)) W(r,r') ρ_b(r'),   ρ_a = conj(ψ_m) ψ_n

is preserved on the retained band window.  The theory, the excavation and
every design ruling behind it are in ``DOWNFOLD_DESIGN.md``; the measured
rank ceiling that sizes μ_S is in ``DOWNFOLD_RANK_PROBE.md``.  What follows
is the short version, because the algebra is what the code has to get right.

THE ALGEBRA, IN FOUR LINES
--------------------------
Write ``X^L[μ, a]`` for the ISDF coefficient matrix over the fit window,
``a = (m, n, k)``, and ``X^S`` for the same thing on the small centroid set.
Minimising ``‖(X^S)† W^S X^S − (X^L)† W^L X^L‖_F`` over ``W^S`` gives

    S_SS W^S S_SS = S_cross W^L S_cross†        (the normal equations)
    W^S = T W^L T† ,   T = S_SS⁻¹ S_cross       (a CONGRUENCE)

with the three **pair-density Grams over the fit window**

    S_SS    = X^S (X^S)†      (μ_S, μ_S)
    S_cross = X^S (X^L)†      (μ_S, μ_L)
    S_LL    = X^L (X^L)†      (μ_L, μ_L)

All three are what :func:`isdf.core.c_q_from_psi_sm` already computes — that
kernel IS the pair-density Gram at centroids over whatever band window its ψ
operands were sliced to, and it is rectangular-capable, so feeding it ψ at
the small centroids on its X legs and ψ at the large centroids on its Y legs
returns ``S_cross`` directly.  There is no new kernel here and no second ζ
fit; the whole module is operand assembly plus one solve plus one congruence.

WHY THE SMALL SET IS A SUBSET (CUR), NOT A ROTATION
---------------------------------------------------
The Frobenius-optimal rank-μ_S compression is the natural-auxiliary-function
rotation of Backhouse et al., JCTC (2025) §2.1 — diagonalise S_LL, keep the
top μ_S eigenvectors.  It is strictly better by Eckart–Young and it is
**inadmissible here**, because the rotated coefficients
``(U†X^L)[σ,a] = Σ_ν conj(U[ν,σ]) conj(ψ_m(r_ν)) ψ_n(r_ν)`` DO NOT FACTORISE
into a product of two single-band quantities.  Every LORRAX kernel downstream
contracts the two legs separately — Σ's ``G_μν`` build
(``gw/greens_function_kernel.py``), the BSE's ``_encode_T_A``
(``bse/bse_stack_matvec.py``), the exchange ``M`` matrix
(``bse/bse_serial.py``) — so a rotated basis would need a dense rank-3
coefficient tensor, which is precisely the storage ISDF exists to avoid.

So the small set is a SUBSET of the large one, chosen by pivoted Cholesky
against the retained window (the interpolative decomposition of Lu & Ying,
JCP 302, 329 (2015) §2; Hu, Lin & Yang, JCTC 13, 5420 (2017) §3.2).  Then
``S_cross = S_LL[keep, :]``, ``S_SS = S_LL[keep, keep]``, and the new
ψ-at-centroids are a literal column slice of the ones already on disk.
**Do not "improve" this into a rotation.**  The separable-coefficient
contract is worth more than the Eckart–Young gap.

THE TWO TOLERANCES ARE NOT THE SAME KNOB — READ THIS BEFORE TOUCHING μ_S
-----------------------------------------------------------------------
``select_tol`` (pivoted Cholesky's stopping tolerance, on a residual Schur
diagonal) and ``rcond`` (the truncation of S_SS, on an EIGENVALUE) both look
like "a tolerance" and produce ranks that differ by about 3×.  Measured on
si_bse_debug's 20-band window at pool 3000 (``DOWNFOLD_RANK_PROBE.md`` §7):

    tolerance     PC selection rank     eigenvalue truncation rank
    1e-6                588                       195
    √ε ≈ 1.49e-8       1146                       603
    1e-10              1614                      1140

A user who sets both to 1e-6 is handed 588 selected points of which ~195
carry an eigenvalue above 1e-6 of the largest; the other two thirds are
truncated away by the solve that consumes them.  **μ_S is therefore
validated against the EIGENVALUE rank of the window's Gram, not against
pivoted Cholesky's certificate.**  The certificate is still the first gate —
it is cheap and it catches the gross case — but it is a NECESSARY condition,
not a sufficient one.  :func:`select_cur_centroids` refuses on the
eigenvalue rank and reports both numbers side by side, labelled differently.

The measured ceiling at a 20-band window is ~190 directions on two
chemically different decks (si 196, hBN 185), stable to 2 % as the candidate
pool grows 9–12×, and flat in q (si spread 189–195 over 64 q).  At rcond
1e-6 that is a real ceiling; at 1e-8 and below the "rank" keeps climbing
with the size of the pool and is not a ceiling at all.  Hence the documented
default ``rcond = 1.1e-6``.

CONDITIONING
------------
``S_SS`` is a pair-density Gram over a NARROW window; it is numerically
singular and the rank-truncated Hermitian route is the NORMAL path, not the
exception.  No ridge is applied and no pseudo-inverse is substituted
silently — that would be indistinguishable from a correct projection
downstream.  The truncation criterion is ``common.rank_criterion``'s: a CAP
on how much the pseudo-inverse may amplify round-off, not a search for a gap
(these spectra are smooth and have none).  The R19 anchor travels with every
report: retaining 41 % more rank once moved a 2.2 eV gap by 5000 eV.

THE ERROR BAR THE CODE PRINTS ON EVERY RUN
------------------------------------------
Substituting the solution back gives ``𝒲_S = Π 𝒲 Π`` with Π the orthogonal
projector onto ``row(X^S)``, so the residual is orthogonal to the fit and
Pythagoras holds.  That turns "is the downfold accurate?" into a number
computable from μ×μ traces alone, with no reference calculation and without
ever forming the N×N observable:

    ε_W(q) = sqrt(1 − ‖𝒲_S‖²/‖𝒲‖²),
    ‖𝒲‖²   = tr(W^L S_LL (W^L)† S_LL),
    ‖𝒲_S‖² = tr(W^S S_SS (W^S)† S_SS).

:func:`epsilon_w` is that identity.  Its cost is two GEMMs at μ_L per q — the
same cost class as the congruence itself — and it is worth it.

SHARDING
--------
Three SEQUENTIAL top-level regions, no nesting: the Gram build (one
``shard_map``, ``isdf.core``'s), the solve (rank-local by construction —
``S_SS`` replicated, the μ_L axis sharded), and the congruence (one
``shard_map``, ``common.contract_bands``').  Every congruence output is
sharding-pinned: leaving an ``UΣU†``-shaped einsum's output free once emitted
an all-reduce of the FULL cube where pinning it emitted two all-gathers of
cube/p (``SIGMA_SCALING.md`` §7).

Stage 1 implements the LOCAL/GSPMD plan only.  The 2-D ``distrib_la`` grid
and the q-sharded local-LA alternative are later stages; so is the
asymmetric Σ-serving window (which the probe measures at ~2× μ_S, not the
order of magnitude the design feared) and the ``refit`` mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common import rank_criterion
from common.zeta_projection import (
    least_squares_transfer,
    project_w_between_zeta_bases,
    transfer_operands_from_dense,
)

__all__ = [
    "BandWindow",
    "SelectionReport",
    "pair_density_gram",
    "select_cur_centroids",
    "slice_psi_to_centroids",
    "build_transfer",
    "congruence",
    "transform_head_vector",
    "observable_norm2",
    "epsilon_w",
    "R19_ANCHOR",
    "AUTO_HAZARD",
    "describe_auto_hazard",
    "DEFAULT_RCOND",
]


#: The documented default for ``downfold_rcond``.  ``DOWNFOLD_RANK_PROBE.md``
#: §1: at a 20-band window the pair-density Gram holds ~190 independent
#: directions and the rcond that selects them is 1.1e-6 (si_bse_debug reads
#: 196 at 1e-6, hbn_cohsex_debug 185).  Below 1e-8 the retained "rank" stops
#: being a ceiling and starts tracking the size of the candidate pool.
DEFAULT_RCOND = 1.1e-6

#: Printed beside every margin, because the margin is only meaningful next to
#: a case where loosening the cut destroyed the answer.
R19_ANCHOR = "R19 anchor: +41% retained rank moved a 2.2 eV gap by 5000 eV"

#: THE HAZARD ``mu_small = auto`` CARRIES, in one sentence, stated once so
#: that the driver, the end-of-run summary, ``docs/downfold.md`` and the test
#: that guards them cannot drift apart.  Measured 2026-08-10 on the standard
#: si pipeline walk and written down in ``PIPELINE_HEALTH.md``; the mechanism
#: is in ``DOWNFOLD_S1.md`` §3(c) and the rank probe.  ``auto`` is the
#: EIGENVALUE-RANK CEILING — a statement about how many directions the parent
#: can still represent — and rank completeness at that ceiling does not imply
#: accuracy in any observable unless the parent was over-complete for the
#: window, which nothing in this driver measures.
AUTO_HAZARD = (
    "auto sizes by rank, not by observable accuracy; on a non-redundant "
    "parent this has produced eV-scale errors with epsilon_W silent — "
    "validate against the parent with a direct comparison before trusting "
    "any downfolded result")

#: The banner.  Wide and ugly on purpose: this has to survive being scrolled
#: past in a job log.
_BANG_TITLE = ("!!  WARNING — mu_small = auto SIZED THIS RUN BY RANK, "
               "NOT BY ACCURACY  !!")
_BANG = "  " + "!" * len(_BANG_TITLE)


def describe_auto_hazard(mu_S: int, rcond: float) -> str:
    """The loud block ``mu_small = auto`` prints at SELECTION time.

    Not a refusal — ``auto`` remains an explicit user choice and this driver
    does not overrule explicit choices.  It is the warning that was missing
    when the standard pipeline walk followed the tree's own guidance end to
    end and got a 2.087 eV error in the lowest BSE eigenvalue with nothing
    anywhere refusing and ``eps_W`` reading about one per cent.

    Fires ONLY on ``auto``.  An explicit integer ``mu_small`` prints none of
    this: the user who typed a number has already made the sizing decision
    this block exists to interrupt, and a warning that fires on every run is
    a warning nobody reads.  ``tests/test_downfold.py`` holds both arms.
    """
    return "\n".join([
        _BANG,
        "  " + _BANG_TITLE,
        _BANG,
        f"  [downfold/select] mu_small = auto -> {mu_S}: the number of "
        f"independent pair-density",
        f"      directions the retained window holds at rcond={rcond:g}.  "
        f"That is the CEILING",
        "      the parent can still represent, and the largest value this "
        "driver accepts.  It",
        "      is NOT a recommendation, and it is NOT a statement about the "
        "accuracy of",
        "      anything you will compute from the bundle this run is about "
        "to write.",
        "  [downfold/select] THE MEASURED HAZARD, verbatim:",
        f"      {AUTO_HAZARD}.",
        "  [downfold/select] THE MEASUREMENT (PIPELINE_HEALTH.md, "
        "2026-08-10, si_bse_debug",
        "      4x4x4, a 936-centroid parent on a 0:20 window, following this "
        "tree's own",
        "      documented guidance end to end): auto -> 189 at rcond=1.1e-6 "
        "moved the lowest",
        "      BSE eigenvalue from the parent's 2.3449 eV to 0.2579 eV — "
        "WRONG BY 2.087 eV —",
        "      while eps_W read 1.33e-2 and NOTHING refused.  Tightening to "
        "rcond=1e-8 (auto",
        "      -> 624, a compression of only 1.5x) still left 1.08 eV of "
        "error.  Rank",
        "      completeness at the eigenvalue ceiling implies observable "
        "accuracy only on a",
        "      parent that is OVER-COMPLETE for your window; whether yours "
        "is, this driver",
        "      does not measure and cannot tell you.",
        "  [downfold/select] eps_W IS A TRIPWIRE, NOT AN ACCURACY GATE.  It "
        "reported about one",
        "      per cent in the row that was wrong by 2 eV.  The honest "
        "accuracy instrument is",
        "      the observable comparison against the parent, and auto never "
        "consults it.",
        "  [downfold/select] THERE IS NO ACCURACY-SIZED MODE IN THIS DRIVER "
        "TODAY.  The planned",
        "      one — state a meV bar and let the driver size mu_S by "
        "sweeping it against the",
        "      parent observable — is the stage-4 target-accuracy item on "
        "the downfold",
        "      roadmap and IS NOT BUILT.  Until it lands, auto is where a "
        "sweep STARTS and",
        "      never where it ends, and the comparison this run prints at "
        "the end is the only",
        "      accuracy evidence there is.  Production BSE work wants better "
        "than 1 meV.",
        _BANG,
    ])


# ---------------------------------------------------------------------------
# the fit window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandWindow:
    """The pair-density fit window, as two half-open band ranges.

    ``left`` is the window of the CONJUGATED leg of the pair density
    ``ρ_{mn} = conj(ψ_m) ψ_n`` (m), ``right`` is the window of the plain leg
    (n).  Same convention as ``centroid.pivoted_cholesky``'s
    ``band_range_left`` / ``band_range_right``, whose legacy ``(n_val,
    n_cond)`` spelling is ``left = (0, n_val)``, ``right = (n_val, n_val +
    n_cond)``.

    NOTE ON THE LEG MAPPING INTO ``c_q_from_psi_sm``.  That kernel's ``l``
    operands carry the factor ``Σ_band ψ(μ) conj(ψ(ν))`` and its ``r``
    operands carry ``Σ_band conj(ψ(μ)) ψ(ν)`` (the conjugation lands on
    ``P_l`` *after* the k→R transform, ``isdf/core.py``).  Matching that
    against ``S[μ,ν] = Σ_{mnk} X[μ,a] conj(X[ν,a])`` factorised as
    ``[Σ_m conj(ψ_m(μ)) ψ_m(ν)]·[Σ_n ψ_n(μ) conj(ψ_n(ν))]`` puts the
    ``left`` (m) window on the kernel's **r** legs and the ``right`` (n)
    window on its **l** legs.  It is symmetric in the symmetric case, which
    is stage 1's only case; the mapping is written down here so the
    asymmetric Σ-serving stage does not have to rediscover it.
    """

    left: tuple[int, int]
    right: tuple[int, int]

    @property
    def symmetric(self) -> bool:
        return tuple(self.left) == tuple(self.right)

    @property
    def n_bands_max(self) -> int:
        return max(int(self.left[1]), int(self.right[1]))

    def validate(self, n_bands: int) -> None:
        for tag, (b0, b1) in (("left", self.left), ("right", self.right)):
            if not (0 <= int(b0) < int(b1)):
                raise ValueError(
                    f"downfold: band_range_{tag}=({b0}, {b1}) is not a "
                    f"non-empty half-open range 0 <= b0 < b1.")
            if int(b1) > int(n_bands):
                raise ValueError(
                    f"downfold: band_range_{tag}=({b0}, {b1}) runs past the "
                    f"restart's band extent nb={n_bands}.  The window is an "
                    f"ABSOLUTE band index range into psi_full_y / enk_full, "
                    f"not an offset — a window outside the file's extent "
                    f"makes ψ-at-centroids identically zero and every "
                    f"relative threshold downstream meaningless.")

    def describe(self) -> str:
        if self.symmetric:
            return (f"[{self.left[0]}, {self.left[1]}) symmetric "
                    f"({self.left[1] - self.left[0]} bands)")
        return (f"left [{self.left[0]}, {self.left[1]}) x right "
                f"[{self.right[0]}, {self.right[1]}) ASYMMETRIC")


# ---------------------------------------------------------------------------
# D1 — the Grams
# ---------------------------------------------------------------------------

def pair_density_gram(psi_X, psi_Y, window: BandWindow, *, kgrid, mesh_xy):
    """``S[q, μ_X, μ_Y]`` — the pair-density Gram over ``window``.

    Parameters
    ----------
    psi_X
        ``(nk, μ_X, nb, ns)`` at ``P(None,'x',None,None)`` — the CONJUGATED
        ψ at the centroid set that indexes the Gram's ROW axis (the
        ``psi_rmuT_X`` the restart loader returns).
    psi_Y
        ``(nk, nb, ns, μ_Y)`` at ``P(None,None,None,'y')`` — the plain ψ at
        the centroid set that indexes the COLUMN axis (``psi_rmu_Y``).
    window
        The fit window.  Its band axis is replicated on both operands, so
        slicing it is free.

    Returns ``(nq, μ_X, μ_Y)`` at ``P(None,'x','y')``.

    Feed the same centroid set on both sides for ``S_LL`` / ``S_SS``, and
    the SMALL set on X with the LARGE set on Y for ``S_cross`` — the kernel
    is rectangular-capable (its ``n_col == n_rmu`` docstring line is a
    statement about its one existing caller, not a constraint), so the cross
    Gram comes out at the right sharding with no reshard and no sharded
    fancy-indexing.
    """
    from isdf.core import c_q_from_psi_sm

    (bl0, bl1), (br0, br1) = window.left, window.right
    # left (m) window -> the kernel's r legs; right (n) window -> its l legs.
    # See BandWindow's docstring for why, and do not swap these without
    # re-deriving it.
    return c_q_from_psi_sm(
        psi_X[:, :, br0:br1, :], psi_Y[:, br0:br1, :, :],
        psi_X[:, :, bl0:bl1, :], psi_Y[:, bl0:bl1, :, :],
        kgrid=tuple(int(v) for v in kgrid), mesh_xy=mesh_xy)


# ---------------------------------------------------------------------------
# D2 — centroid selection, and the rank refusal that is the point of it
# ---------------------------------------------------------------------------

@dataclass
class SelectionReport:
    """Everything a run should say about which centroids it kept, and why."""

    mu_large: int
    mu_small: int
    rcond: float
    select_tol: float
    #: #{λ of the POOL Gram > rcond·λ_max} — the honest ceiling.  This is
    #: the number μ_S is validated against.
    eigen_rank_pool: int
    #: pivoted Cholesky's certificate.  Necessary, NOT sufficient; runs
    #: about 3x the eigenvalue rank at the same nominal tolerance.
    select_rank: int
    #: #{λ of S_SS[keep,keep] > rcond·λ_max} — what the transfer solve will
    #: actually retain at q=0.
    eigen_rank_kept: int
    pool_report: rank_criterion.RankReport | None = None
    kept_report: rank_criterion.RankReport | None = None
    requested_auto: bool = False

    def describe(self, indent="  ") -> str:
        lines = [
            f"{indent}[downfold/select] CUR selection from the parent set: "
            f"mu_L={self.mu_large} -> mu_S={self.mu_small}"
            + ("  (mu_small = auto: sized by the certified eigenvalue RANK, "
               "not by observable accuracy — see the warning above)"
               if self.requested_auto else ""),
            f"{indent}[downfold/select] TWO DIFFERENT RANKS, and they are "
            f"not the same knob:",
            f"{indent}[downfold/select]   EIGENVALUE rank of the pool Gram "
            f"at rcond={self.rcond:g}: {self.eigen_rank_pool}   "
            f"<- this is what mu_S is validated against",
            f"{indent}[downfold/select]   pivoted-Cholesky SELECTION "
            f"certificate at tol={self.select_tol:g}: {self.select_rank}   "
            f"<- necessary, NOT sufficient (~3x the eigenvalue rank at the "
            f"same nominal tolerance; RANK_PROBE §7)",
            f"{indent}[downfold/select]   EIGENVALUE rank of the KEPT "
            f"block S_SS[keep,keep] at q=0: {self.eigen_rank_kept} of "
            f"{self.mu_small}",
        ]
        if self.pool_report is not None:
            lines.append(f"{indent}[downfold/select] pool spectrum:")
            lines.append(self.pool_report.describe(indent=indent + "  "))
        return "\n".join(lines)


def _eigen_rank(G_host: np.ndarray, rcond: float, *, label: str,
                n_logical: int | None = None) -> rank_criterion.RankReport:
    """Hermitian eigenvalue spectrum of ``G`` → a :class:`RankReport`.

    ``common.rank_criterion`` verbatim: the criterion is a cap on how much
    the pseudo-inverse amplifies round-off, the margin is the rank inflation
    from loosening the cap by four decades, and the R19 anchor rides beside
    it.  Nothing here looks for a gap; these spectra do not have one.
    """
    n = int(G_host.shape[-1]) if n_logical is None else int(n_logical)
    sub = np.asarray(G_host)[:n, :n]
    ev = np.linalg.eigvalsh(0.5 * (sub + sub.conj().T))
    return rank_criterion.rank_report(
        ev, rcond, label=label, quantity="eigenvalues", n_rows=n)


def select_cur_centroids(
    S_q0, mu_small, *, rcond: float, select_tol: float | None,
    mesh_xy, mu_large_logical: int, print_fn=print,
):
    """Pick μ_S of the parent's centroids against the retained window.

    ``S_q0`` is the q = 0 slice of the pair-density Gram over the fit
    window, ``(μ_L_pad, μ_L_pad)``, sharded ``P('x','y')``.  The q = 0 slice
    is the right selection Gram and the whole selection story: the probe's
    §9a.1 measured the retained rank over all 64 q of si_bse_debug at
    189/193/195 (min/median/max) against 194 at q = 0, and over all 18 q of
    hBN at 182/193/195 against 182 — three and seven per cent spreads, with
    q = 0 the MINIMUM on hBN.  μ_S does not need sizing by a worst q.

    ``mu_small`` may be the string ``"auto"``, meaning "μ_S = the rank
    certified at ``rcond``".  That is a CEILING, not a recommendation: it is
    sized by rank and says nothing about the accuracy of any observable, and
    on a parent that is not over-complete for the window it has produced
    eV-scale BSE errors with ``eps_W`` silent (``PIPELINE_HEALTH.md``,
    2026-08-10).  ``auto`` therefore prints :func:`describe_auto_hazard`
    loudly at selection time; an explicit integer prints nothing.

    REFUSES, loudly and before any of the expensive stages, when the
    requested μ_S exceeds the eigenvalue rank the window actually holds.
    That refusal is not a failure mode, it is the intended behaviour: it is
    the ``isdf_basis_adequacy_at_large_nband.md`` rule ("the ISDF basis must
    be SELECTED against the band window actually consumed") applied on
    purpose instead of by accident, and the accident cost 2.8 eV.

    Returns ``(keep_idx, SelectionReport)`` with ``keep_idx`` a sorted
    ``int64`` array of parent centroid indices.
    """
    from centroid import distribution as dist
    from centroid.pivoted_cholesky import (
        make_sharded_pivoted_cholesky_select, refuse_unless_select_certified,
    )
    from common.collectives import gather_to_host

    if not (0.0 < float(rcond) < 1.0):
        raise ValueError(
            f"downfold_rcond must lie in (0, 1) — it is a RELATIVE threshold "
            f"on lambda/lambda_max, i.e. the reciprocal of the amplification "
            f"cap the truncated pseudo-inverse is allowed to reach.  Got "
            f"{rcond!r}.")

    tol = (float(np.sqrt(np.finfo(np.float64).eps)) if select_tol is None
           else float(select_tol))
    mu_L = int(mu_large_logical)

    # ---- 1. THE CEILING.  Eigenvalue rank of the pool Gram at rcond. ----
    G_host = np.asarray(gather_to_host(S_q0))
    pool_rep = _eigen_rank(G_host, rcond, label="downfold pool Gram (q=0)",
                           n_logical=mu_L)
    ceiling = int(pool_rep.rank_criterion)

    requested_auto = isinstance(mu_small, str) and str(mu_small).strip().lower() == "auto"
    if requested_auto:
        mu_S = ceiling
        # LOUD, at selection time, before the expensive stages — because the
        # thing this warns about is invisible everywhere downstream.  Fires
        # only here; an explicit integer mu_small prints none of it.
        print_fn(describe_auto_hazard(mu_S, float(rcond)))
    else:
        mu_S = int(mu_small)

    if mu_S < 1:
        raise ValueError(f"downfold: mu_small={mu_S} must be at least 1.")
    if mu_S > mu_L:
        raise ValueError(
            f"downfold: mu_small={mu_S} exceeds the parent basis "
            f"mu_L={mu_L}.  A downfold selects a SUBSET of the parent's "
            f"centroids (mode='cur'); it cannot invent new ones.  Either "
            f"lower mu_small or run the parent GW at a larger centroid set.")
    if mu_S > ceiling:
        raise ValueError(
            f"downfold: REFUSING mu_small={mu_S}.  The retained band window "
            f"holds only {ceiling} numerically independent pair-density "
            f"directions at downfold_rcond={rcond:g} — measured, on this "
            f"deck, from the eigenvalue spectrum of the q=0 pair-density "
            f"Gram over the {mu_L}-point parent set.  Asking for {mu_S} is "
            f"asking for {mu_S - ceiling} directions that are not there: "
            f"S_SS would be rank-deficient by that much and the truncated "
            f"pseudo-inverse would discard them, so the '{mu_S}-centroid "
            f"basis' would be a fiction.\n"
            f"  THREE HONEST FIXES, in order of preference:\n"
            f"   (1) set mu_small = {ceiling} (or mu_small = auto, which "
            f"does exactly that) — but note that {ceiling} is a RANK "
            f"ceiling, not an accuracy answer: it is the most this parent "
            f"can represent, and whether it is enough for your observable "
            f"is a question only a comparison against the parent can "
            f"settle;\n"
            f"   (2) widen the retained band window — the ceiling scales as "
            f"roughly nb^0.8 (RANK_PROBE §8), so it is the WINDOW that sets "
            f"it, not the number of centroids;\n"
            f"   (3) loosen downfold_rcond — and read RANK_PROBE §5 first, "
            f"because the rank moves by a factor of twelve across "
            f"1e-4..1e-10 and buying mu_S=500 on a 20-band window means "
            f"keeping directions at lambda/lambda_max = 3.5e-8.  "
            f"{R19_ANCHOR}.\n"
            f"  Do NOT reach for a ridge or a looser pivoted-Cholesky "
            f"tolerance: the selection certificate runs about 3x the "
            f"eigenvalue rank at the same nominal number, so it will happily "
            f"hand you {mu_S} points that the solve then truncates away.")

    # ---- 2. THE SELECTION.  Pivoted Cholesky, with its own refusal. -----
    select_axis = dist.require_axes(mesh_xy, dist.MESH_AXES,
                                    "downfold.select_cur_centroids")
    n_dev = dist.n_shards(mesh_xy, select_axis)
    M_pad = int(S_q0.shape[-1])
    if M_pad % n_dev:
        raise ValueError(
            f"downfold: the parent Gram came back at M_pad={M_pad}, which "
            f"the mesh product {n_dev} does not divide.  The sharded select "
            f"needs an even row split; the restart reader pads mu to "
            f"padded_mu_extent(mu_L, device_count), so this means the mesh "
            f"changed under the array.")
    n_pad = M_pad - mu_L

    G_rows = jax.lax.with_sharding_constraint(
        S_q0, NamedSharding(mesh_xy, P(select_axis, None)))
    select_step = make_sharded_pivoted_cholesky_select(
        mesh_xy, M_pad, mu_S, mesh_axis=select_axis, tol_rel=tol)
    active_init = None
    if n_pad:
        from common.collectives import device_put_process_local
        _act = np.ones((M_pad,), dtype=bool)
        _act[mu_L:] = False
        active_init = device_put_process_local(
            _act, NamedSharding(mesh_xy, P(select_axis)))
    piv, _L, rank, _d_final, _d_taken, _trR, psd_info = select_step(
        G_rows, None, active_init)
    piv_np = np.asarray(jax.device_get(piv))
    psd_host = (float(np.asarray(psd_info[0])), int(np.asarray(psd_info[1])),
                int(np.asarray(psd_info[2])))
    # The SAME refusal ``prune_candidates_by_pivoted_cholesky`` raises, on
    # the same kernel output.  A jitted kernel cannot raise, so it reports
    # and this refuses — LAPACK ``pstrf``'s division of labour.
    refuse_unless_select_certified(
        piv_np, int(rank), psd_host, n_keep=mu_S, M=mu_L, M_pad=M_pad,
        orbit_id=None, d0max=float(np.real(np.diag(G_host)[:mu_L]).max()),
        tol_rel=tol)
    keep_idx = np.sort(piv_np.astype(np.int64))

    # ---- 3. What the SOLVE will actually retain, at q = 0. --------------
    kept_rep = _eigen_rank(G_host[np.ix_(keep_idx, keep_idx)], rcond,
                           label="downfold S_SS (q=0)")

    rep = SelectionReport(
        mu_large=mu_L, mu_small=mu_S, rcond=float(rcond), select_tol=tol,
        eigen_rank_pool=ceiling, select_rank=int(rank),
        eigen_rank_kept=int(kept_rep.rank_criterion),
        pool_report=pool_rep, kept_report=kept_rep,
        requested_auto=requested_auto)
    return keep_idx, rep


def slice_psi_to_centroids(psi_X, psi_Y, keep_idx, mu_small_pad, mesh_xy):
    """``(ψ^S_X, ψ^S_Y)`` — the small basis's coefficients, by column slice.

    "Wavefunction coefficients redefined on a smaller centroid set" costs a
    slice, and that is the whole reason the CUR route is worth its
    Eckart–Young gap.  The extent is zero-padded to ``mu_small_pad`` (the
    device-count-legal μ extent) exactly as the ISDF fit's ``n_rmu_padded``
    contract does: a ZERO ψ row contributes a zero pair density, hence a
    zero row and column of every Gram, hence a zero eigenvalue that the
    truncation drops — the pad is inert, not merely harmless.
    """
    keep = jnp.asarray(np.asarray(keep_idx, dtype=np.int32))
    n_keep = int(keep.shape[0])
    pad = int(mu_small_pad) - n_keep
    if pad < 0:
        raise ValueError(
            f"downfold: mu_small_pad={mu_small_pad} is below the kept count "
            f"{n_keep}; the pad must round UP.  A device-grid round-DOWN "
            f"would make the retained subspace depend on the mesh, which is "
            f"the defect rank_criterion.RankReport.violations() reports.")

    x_spec = NamedSharding(mesh_xy, P(None, "x", None, None))
    y_spec = NamedSharding(mesh_xy, P(None, None, None, "y"))

    def _take_x(a):
        out = jnp.take(a, keep, axis=1)
        if pad:
            out = jnp.pad(out, ((0, 0), (0, pad), (0, 0), (0, 0)))
        return jax.lax.with_sharding_constraint(out, x_spec)

    def _take_y(a):
        out = jnp.take(a, keep, axis=3)
        if pad:
            out = jnp.pad(out, ((0, 0), (0, 0), (0, 0), (0, pad)))
        return jax.lax.with_sharding_constraint(out, y_spec)

    return (jax.jit(_take_x, out_shardings=x_spec)(psi_X),
            jax.jit(_take_y, out_shardings=y_spec)(psi_Y))


# ---------------------------------------------------------------------------
# D3 — the transfer solve
# ---------------------------------------------------------------------------

def build_transfer(S_SS, S_cross, mesh_xy, *, rcond: float,
                   axes=("x", "y"), print_fn=print, announce: bool = True):
    """``T = S_SS⁻¹ S_cross`` — the rank-truncated transfer, both shardings.

    ``S_SS`` arrives at ``P(None,'x','y')`` and is gathered to REPLICATED
    (it is the one replicated object on this path, and it is bounded by μ_S,
    the dimension the whole exercise makes small).  ``S_cross`` arrives at
    ``P(None,'x','y')`` and is resharded to ``P(None,None,'x')`` and
    ``P(None,None,'y')``, because the congruence needs T on both mesh axes
    and deriving the second by resharding the first hands the partitioner an
    all-to-all it plans itself.

    The solve itself is ``common.zeta_projection.least_squares_transfer``
    verbatim — same eigh, same truncation, same κ_eff and zero-rank
    refusals.  Only the two matrices fed to it changed: the ζ-sphere Gram
    became the pair-density Gram, and the ζ cross-overlap became the
    pair-density cross-Gram.  Its own log line is suppressed and this
    function prints the authoritative report instead, built from
    ``common.rank_criterion`` so the margin and the R19 anchor come with it.

    Returns ``(T_x, T_y, per_q_reports)``.
    """
    ax_x, ax_y = axes
    if not (0.0 < float(rcond) < 1.0):
        raise ValueError(
            f"downfold_rcond must lie in (0, 1); got {rcond!r}.")

    gram = jax.lax.with_sharding_constraint(
        S_SS, NamedSharding(mesh_xy, P(None, None, None)))
    cross_x = jax.lax.with_sharding_constraint(
        S_cross, NamedSharding(mesh_xy, P(None, None, ax_x)))
    cross_y = jax.lax.with_sharding_constraint(
        S_cross, NamedSharding(mesh_xy, P(None, None, ax_y)))

    # The per-q truncation report, from the module that owns the criterion.
    # Computed BEFORE the solve: ``least_squares_transfer``'s own eigh is an
    # implementation detail and this must not depend on it.
    ev = np.asarray(jax.device_get(jnp.linalg.eigvalsh(gram)))
    reports = [rank_criterion.rank_report(
        ev[q], rcond, label=f"downfold S_SS q={q}", quantity="eigenvalues",
        n_rows=int(ev.shape[-1])) for q in range(int(ev.shape[0]))]
    if announce:
        _announce_ranks(reports, rcond, int(gram.shape[-1]), print_fn)

    T_x = least_squares_transfer(gram, cross_x, mesh_xy, ax_x, rcond=rcond,
                                 print_fn=lambda *a, **k: None)
    T_y = least_squares_transfer(gram, cross_y, mesh_xy, ax_y, rcond=rcond,
                                 print_fn=lambda *a, **k: None)
    return T_x, T_y, reports


def _announce_ranks(reports, rcond, mu_s_pad, print_fn) -> None:
    ranks = np.array([r.rank_criterion for r in reports], dtype=np.int64)
    kap = np.array([r.kappa_eff if r.kappa_eff is not None else np.nan
                    for r in reports])
    marg = np.array([r.overcomplete_margin for r in reports])
    print_fn(
        f"  [downfold/solve] rank-truncated transfer at "
        f"rcond={rcond:g}: retained rank of S_SS per q "
        f"min/median/max = {ranks.min()}/{int(np.median(ranks))}/"
        f"{ranks.max()} of mu_S_padded={mu_s_pad} over {len(reports)} q.")
    print_fn(
        f"  [downfold/solve] achieved amplification kappa_eff max "
        f"{np.nanmax(kap):.3e} against the cap 1/rcond="
        f"{1.0 / float(rcond):.3e}; overcomplete margin (loosen rcond by "
        f"1e-4) max +{100.0 * np.max(marg):.1f}%.  {R19_ANCHOR} — a LARGE "
        f"margin means the cut is load-bearing and rcond must NOT be "
        f"loosened on this run.")
    viol = {q: r.violations() for q, r in enumerate(reports) if r.violations()}
    if viol:
        for q, vs in list(viol.items())[:4]:
            for v in vs:
                print_fn(f"  [downfold/solve]   ** VIOLATION q={q}: {v}")


# ---------------------------------------------------------------------------
# D4 — the congruence, and the vector that rides along with it
# ---------------------------------------------------------------------------

def congruence(mesh_xy, T_x, T_y, *, axes=("x", "y")):
    """Build ``project(A_L) -> A_S = T A_L T†`` for every linear object.

    ``common.zeta_projection.project_w_between_zeta_bases`` unmodified —
    which is ``contract_bands_block_reshard(extra="none", channels="none")``
    under the identification (k batch → q; μ,ν → μ_L; m,n → μ_S; ψ_left →
    conj(T) on 'x'; ψ_right → T† on 'y').  It inherits the two-stage
    ``psum_scatter``, so no (μ_L, μ_L) tile is ever gathered, and it returns
    at ``P(None,'x','y')`` — the same layout it consumed, so the output is a
    drop-in for any consumer that already eats a sharded W.

    Apply once per LINEAR object: V, W(0), each ω/τ tile, and the ``_nohead``
    twins.  **Not** to a plasmon-pole reduction: ``B_q`` is a residue and
    would transform, but ``Omega_q`` is a pole POSITION per matrix element
    and there is no congruence that maps a table of pole frequencies from
    one basis to another.  Downfold the linear objects and re-fit the PPM /
    MPA model in the small basis; anyone who congruences ``Omega_q`` gets
    numbers that look plausible and are meaningless.
    """
    psi_left, psi_right = jax.jit(
        lambda a, b: transfer_operands_from_dense(a, b, mesh_xy, axes)
    )(T_x, T_y)
    project = project_w_between_zeta_bases(mesh_xy, axes=axes)

    def _apply(A_L):
        return project(A_L, psi_left, psi_right)

    return _apply


def transform_head_vector(g0_L, T_x, q_index: int, mesh_xy, *,
                          axes=("x", "y")):
    """``g0_S = conj(T[q]) · g0_L`` — and the conjugate is LOAD-BEARING.

    ``G0_mu_nu`` in the restart bundle is the single-index object
    ``g0_μ = ζ̃_{q,μ}(G=0)`` (``gw/v_q_g_flat.py``), stored at the one q it
    belongs to.  It is not a two-point object, so it does not take the
    congruence — but it is not a free-standing vector either.  It exists to
    reconstitute the q→0 Coulomb head as the RANK-1 UPDATE that
    ``gw.head_correction.apply_q0_head_rank1`` applies:

        V_head[μ,ν] = s · conj(g0_μ) · g0_ν

    so the only correct transformation is the one that makes the congruence
    of that rank-1 matrix be the SAME rank-1 form in the small basis.
    Writing u = T conj(g0),

        (T V_head T†)[σ,τ] = s · u_σ · conj(u_τ)  ≡  s · conj(g0^S_σ) · g0^S_τ

    which forces ``g0^S = conj(u) = conj(T) g0``.  Using ``T g0`` instead
    gives a vector of the right SIZE and the wrong PHASES, so every shape
    check passes, the run reports a healthy error bar (eps_W never sees the
    head — it is injected downstream of the bundle), and the BSE exchange
    kernel is built from a different rank-1 matrix of the same magnitude.
    MEASURED consequence on si_bse_debug: the lowest exciton moved from
    2.347 eV to 0.211 eV, a factor of eleven, with every gate green.  That
    is what this conjugate is worth, and
    ``test_head_vector_transforms_with_the_conjugate_transfer`` is the twin
    that holds it.

    THE OPEN OWNER ROW TRAVELS WITH IT: "g0_mu placement" is an unresolved
    item on the zeta_loader service ledger.  This function inherits that
    question rather than answering it — what it fixes is the algebra of the
    map, not where the object should live.
    """
    ax_x, _ = axes
    T_q = jax.lax.with_sharding_constraint(
        T_x[int(q_index)], NamedSharding(mesh_xy, P(None, ax_x)))
    g0 = jax.lax.with_sharding_constraint(
        g0_L, NamedSharding(mesh_xy, P(ax_x)))
    out = jnp.einsum("mn,n->m", jnp.conj(T_q), g0, optimize=True)
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh_xy, P("y")))


# ---------------------------------------------------------------------------
# D5 — the Pythagorean residual: the error bar the driver prints every run
# ---------------------------------------------------------------------------

def observable_norm2(A, S, mesh_xy, *, axes=("x", "y")):
    """``tr(A S A† S)`` per q — the observable's squared Frobenius norm.

    ``‖𝒲‖²_F = ‖X† A X‖²_F = tr(A S A† S)`` with ``S = X X†``, so the N×N
    observable (N = n_k·|B_L|·|B_R|, millions) is never formed: two GEMMs at
    μ per q and an elementwise reduction is the whole cost.

    Written as ``Σ_ij (A S)_ij conj((S A)_ij)`` rather than as a trace of a
    product, because that spelling needs no transpose of a sharded (μ, μ)
    tile.  ``A`` and ``S`` are both Hermitian here; the identity
    ``tr(A S A† S) = Σ_ij (AS)_ij conj((SA)_ij)`` uses only ``S = S†``.

    Both intermediates are sharding-PINNED.  ``SIGMA_SCALING.md`` §7: the
    same shape of einsum with its output sharding left free emitted an
    all-reduce of the full cube, identical at P=4 and P=16, where pinning it
    emitted two all-gathers of cube/p and no full-cube buffer.  The
    difference was one ``with_sharding_constraint``.
    """
    ax_x, ax_y = axes
    munu = NamedSharding(mesh_xy, P(None, ax_x, ax_y))
    rep = NamedSharding(mesh_xy, P(None))

    @partial(jax.jit, out_shardings=rep)
    def _norm2(A_, S_):
        AS = jax.lax.with_sharding_constraint(
            jnp.einsum("qij,qjk->qik", A_, S_, optimize=True), munu)
        SA = jax.lax.with_sharding_constraint(
            jnp.einsum("qij,qjk->qik", S_, A_, optimize=True), munu)
        return jnp.real(jnp.sum(AS * jnp.conj(SA), axis=(1, 2)))

    return _norm2(A, S)


def epsilon_w(W_L, S_LL, W_S, S_SS, mesh_xy, *, axes=("x", "y")):
    """``ε_W(q) = sqrt(1 − ‖𝒲_S‖²/‖𝒲‖²)`` — the per-run error bar.

    Substituting the solution into the fit gives ``𝒲_S = Π 𝒲 Π`` with Π the
    orthogonal projector onto ``row(X^S)``, so the residual is orthogonal to
    the fit and Pythagoras holds exactly:
    ``‖𝒲 − 𝒲_S‖² = ‖𝒲‖² − ‖𝒲_S‖²``.  That is what makes this a number the
    code can print on every run with no reference calculation.

    THE RIDGE IS WHAT BREAKS IT.  Add a ridge to ``S_SS`` and the fit stops
    being an orthogonal projection, Pythagoras fails, and this diagnostic
    silently becomes meaningless — which is the executable form of the
    argument for why no ridge is applied anywhere on this path.

    At q = 0 the head divergence contaminates both norms identically, so
    ``ε_W`` stays meaningful there; the ABSOLUTE norms at q = 0 are
    head-dominated and should not be compared across q.

    Returns ``(eps_w, norm2_L, norm2_S)``, all ``(nq,)`` host arrays.
    """
    n2_L = np.asarray(jax.device_get(observable_norm2(W_L, S_LL, mesh_xy,
                                                      axes=axes)))
    n2_S = np.asarray(jax.device_get(observable_norm2(W_S, S_SS, mesh_xy,
                                                      axes=axes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n2_L > 0.0, n2_S / np.where(n2_L > 0.0, n2_L, 1.0),
                         np.nan)
    eps = np.sqrt(np.maximum(0.0, 1.0 - ratio))
    return eps, n2_L, n2_S
