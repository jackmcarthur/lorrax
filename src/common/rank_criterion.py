"""The rank-truncation criterion for every pseudo-inverse in LORRAX.

WHAT THIS MODULE IS FOR
-----------------------
Several places in LORRAX truncate a spectrum before inverting it:

* ``bandstructure/htransform.py`` — SVD of ψ-at-centroids, then ``1/σ``
  (``inv_s``) in the Galerkin projector.
* ``isdf/core.py::_rank_trunc_factor`` — eigh of the charge ``C_q``, then
  ``λ^{-1/2}`` (``zeta_rcond``, production default ``1e-8``).
* ``common/zeta_projection.py::least_squares_transfer`` — eigh of the
  small-basis Gram ``G_S``, then ``λ^{-1}``.

All three spectra are ISDF / Galerkin **overlap** spectra.  They are smooth
by construction and **there is no gap, knee, elbow or plateau in them to
find**.  Any criterion phrased as "cut at the separation" is inapplicable
here.  This module states the criterion that is applicable, implements it,
and carries the measurements that chose it.

THE CRITERION
-------------
    Retain the largest subspace on which the pseudo-inverse amplifies by no
    more than a stated factor κ_cap.

For a spectrum ``σ`` of the operator that is actually inverted,

    keep = { i : σ_i > σ_max / κ_cap }        rtol := 1 / κ_cap

and the achieved amplification is ``κ_eff = σ_max / min(σ_kept) ≤ κ_cap``
by construction.  This is exactly the relative threshold the code already
used — the point of this module is that the relative threshold is **not a
gap-finder**; it is a hard cap on noise amplification, it is well defined on
a perfectly smooth spectrum, and ``rtol`` is the reciprocal of a number with
a physical meaning (how much round-off the inverse is allowed to multiply).

WHY THE FAILURE MODE FIXES THE OBJECTIVE
----------------------------------------
The danger is **not** cutting a real direction.  It is **keeping a near-null
one**, whose pseudo-inverse amplifies noise by ``1/σ``.  Measured on this
code (docs/dev/notes/ladder_rung1_R19_zeta_rcond.md, MoS2 4×4, nb=1024, μ≈10k,
``zeta_rcond`` swept, everything else fixed):

    zeta_rcond   retained rank   eqp0        eqp1
    1e-8              6700       3.1350      3.0710
    1e-10             8290    -206.83    -1039.84
    1e-12             9461   -5049.59     -304.20

Retaining **41 % more rank** moved the answer from wrong-by-2.8-eV to wrong
by 5000 eV on a 2.2 eV DFT gap.  So the criterion must control the
**conditioning of the downstream solve**, not the **fidelity of the
decomposition**.  Every criterion in the standard regularisation toolbox
targets fidelity, and each is therefore refuted by that table:

* **Discrepancy principle** — truncate at the data's noise floor,
  ``δ ≈ ε_mach · σ_max · √n``.  At LORRAX's sizes (n ~ 1e4) that is
  ``δ/σ_max ≈ 2.2e-14``, i.e. it prescribes ``rtol ≈ 1e-14`` — **looser than
  the 1e-12 that measured −5049.59 eV**, and the degradation is monotone in
  that direction.  Refuted by measurement, not by argument.  This is the
  same "real content six decades above the f64 noise floor" reasoning that
  §R19.1 puts on the record as refuted from both ends.
  :func:`noise_floor_rtol` computes it so the report can show where it sits;
  it is a **reference line, never the cut**.
* **Residual / L-curve on the decomposition** — the truncated-SVD residual
  is ``σ_{k+1}``: monotone non-increasing in k, and on a smooth spectrum it
  has no corner.  The L-curve's corner is where ``‖A⁺b‖`` starts to blow up,
  which is a conditioning statement wearing a residual's clothes, and for a
  smooth power-law spectrum its location is set by the noise level — i.e. it
  degenerates to the discrepancy principle and inherits its refutation.
* **Tikhonov + GCV** — GCV minimises predicted error **of the fit**, so it
  again buys fidelity.  Two in-repo measurements say a fidelity-sized ridge
  is already too big: ``factor_c_q``'s 1e-14·trace ridge "biases htransform's
  small-trace G by ~1e-5" (``htransform.py`` §4 note), and ``zeta_ridge``
  "PERTURBS the physical result … hence opt-in, a physics call"
  (``gw_config.py`` note on ``zeta_ridge``).  The ridge that empirically
  *works* on MoS2 6×6 is ε≈1e-4 — a **conditioning**-sized ridge (κ ~ 1e4),
  not a GCV-sized one.
* **Observable convergence** — sweep the criterion parameter and take the
  plateau **in the energy**.  This is the only one whose verdict matches the
  measurements, and it is the one already executed: the ``zeta_rcond``
  default is documented in ``gw_config.py`` as "the LOW end of the
  over-complete recovery plateau", from MoS2 4×4/1204c (1e-10 → MAE 1.4 eV;
  the whole 1e-8…1e-4 plateau → ~0.04 eV) and bulk Si 4×4×4/960c (1e-6 →
  1.021 meV sigTOT drift; 1e-8 → 0.054 meV).  Note what kind of plateau that
  is: **a plateau of the observable as a function of rtol**, not a plateau
  in the spectrum.  It needs no spectral structure and is therefore legal
  under the owner's ruling.

CONCLUSION, and it is the reason this module changes no default: the
relative threshold is already the right criterion and its value is already
fixed by the only admissible evidence class.  What was missing is that the
invariant it enforces was implicit, unverified per run, and silently broken
by a device-grid round-down.  This module makes the invariant explicit
(:func:`select_rank`), verifies it (:func:`RankReport.violations`), and
reports the margin to the R19 cliff on every run
(:attr:`RankReport.overcomplete_margin`).

THE MARGIN, AND HOW TO READ IT
------------------------------
``overcomplete_margin`` = (rank at ``rtol·1e-4``)/(rank at ``rtol``) − 1: the
fractional rank inflation from loosening the cap by four decades.  It needs
no gap — it is a finite difference of a monotone counting function — and it
has a measured anchor: R19's 1e-8 → 1e-12 loosening inflated the rank by
**+41 %** and destroyed the answer.  A margin near zero means the spectrum
genuinely terminates and the cut is nearly free; a margin of tens of percent
means the basis is over-complete, the cut is load-bearing, and ``rtol`` must
NOT be loosened on that run.

UNITS — do not "harmonise" the two thresholds
---------------------------------------------
``htransform`` thresholds **singular values of A** and inverts ``1/σ``;
``isdf/core`` thresholds **eigenvalues of the Gram C** and inverts ``1/λ``.
Both therefore cap the amplification **of the operator each one actually
inverts** at ``1/rtol``.  The two 1e-8's agree as amplification caps even
though λ ~ σ².  Converting one to the other's units would break both.

THE GATE — ``κ_eff ≤ κ_cap`` IS NECESSARY AND NOT SUFFICIENT
------------------------------------------------------------
Everything above sizes the retained set.  :func:`certify` is the part with
AUTHORITY, and it exists because every one of these instruments used to
measure the right number and let the run continue
(``TASTE.md`` 2026-08-15, "an instrument that measures a defect and then
proceeds is not a gate": six of its rows are rank cuts).

``RankReport.violations()`` asks whether the code did what it was told —
``κ_eff ≤ κ_cap``, no device-grid round-down, a finite ``σ_max``, and (since
2026-08-22) no rank above the operator's STRUCTURAL ceiling.  All four
passed on both registered catastrophes.  :func:`certify` asks the different
question: **whether what it was told is a regime anyone has certified.**

    When the criterion BINDS (it discarded at least one direction), the
    ACHIEVED amplification κ_eff must not exceed the site's certified
    ceiling.

:data:`KAPPA_CERTIFIED_GRAM` is 1e8, measured on two decks that share no
code path downstream of the fit (R19's MoS2 rcond ladder — 1e-8 correct,
1e-10 → −206.83 eV, 1e-12 → −5049.59 eV; and Si 4×4×4 at 1776 centroids,
κ_eff 9.7e9…1.0e10, Σ_c MAE 54.4 eV at exit 0).  The BINDS clause is not a
softening: when nothing was discarded the criterion made no choice, the
spectrum ended on its own, and refusing would refuse the input rather than
the policy — which is the Si 960-point anchor set's case at production
settings (768 of 768 retained, and the best BerkeleyGW agreement on record).

WHAT IS REFUTED AS A GATE, so nobody re-proposes it
----------------------------------------------------
**Drop fraction.**  Dead by measurement in this tree, in BOTH directions:
MoS2 production discards 33 % and is right; Si 4×4×4/1776c discards 17 % and
is wrong by 54 eV; Si 960 at rcond 1e-6 discards 34 % and moves the σ-star
spread by 0.005 meV.  Any threshold firing on the 17 % case fires on the
33 % one.  ``TASTE.md`` rule 12 in numbers.  The drop count is REPORTED with
the R19 anchor beside it and gates nothing.

**A plane-wave upper bound on N_μ.**  Not implemented: on the same Si deck
the GOOD 600-centroid arm also exceeds ``ngkmax = 588``, so the naive bound
would refuse a run measured at 0.90 eV MAE.  An open register question, not
a gate calibrated from one payload.

WHAT IS REPORTED INSTEAD — the accuracy statement
--------------------------------------------------
:attr:`RankReport.discarded_weight` = ``Σ_dropped |λ| / Σ_all |λ|``.  For a
charge Gram ``C = P Pᴴ`` this is EXACT: ``Σ λ = tr C = ‖P‖_F²``, so it is
the fraction of pair-density weight the cut throws away.  O(n) after an eigh
that already happened, relative, scale-free, and — unlike a rank count — an
accuracy statement.  Gated only at :data:`DISCARDED_WEIGHT_MAX`, four
decades above anything healthy measured here.

Full derivation, the site register and the two dials:
``docs/dev/rank_truncation_policy.md``.
"""
from __future__ import annotations

import math

__all__ = [
    "select_rank",
    "noise_floor_rtol",
    "RankReport",
    "rank_report",
    # The gate.
    "POLICY_MODES",
    "DEFAULT_POLICY_MODE",
    "POLICY_MODE_ENV",
    "resolve_policy_mode",
    "KAPPA_CERTIFIED_GRAM",
    "KAPPA_INDEFINITE_MAX",
    "DISCARDED_WEIGHT_MAX",
    "RankPolicyError",
    "certify",
    "certify_numbers",
    # Scale-aware independence test for probe / Gram-Schmidt rank decisions.
    "PROBE_RTOL",
    "probe_is_independent",
    # Deferred refusal for cuts that live inside a jit.
    "note_device_finding",
    "pending",
    "raise_if_pending",
]

# f64 unit round-off.  Spelled out rather than imported so this module stays
# numpy-optional at import time.
_EPS64 = 2.220446049250313e-16


def _as_desc_list(spectrum):
    """``spectrum`` as a plain descending list of floats.

    Accepts anything with ``__iter__`` (numpy array, jax array already
    pulled to host, list).  Ascending input (``eigh``) and descending input
    (``svd``) are both fine — this normalises.
    """
    vals = [float(v) for v in spectrum]
    vals.sort(reverse=True)
    return vals


def select_rank(spectrum, rtol, *, ceiling=None):
    """The criterion: how many directions survive an amplification cap.

    Parameters
    ----------
    spectrum : iterable of float
        Spectrum of the operator being pseudo-inverted (singular values, or
        eigenvalues of a Hermitian PSD Gram).  Order does not matter.
    rtol : float
        Reciprocal of the amplification cap, ``rtol = 1/κ_cap``.
    ceiling : int or None
        The operator's **structural** rank ceiling — the number of
        independent directions it can carry no matter what the arithmetic
        says.  ``min(n_rows, n_cols)`` for a rectangular fit; the candidate
        count for a Gram.  ``None`` means the caller has not supplied one,
        which is legal and is reported as such rather than assumed safe.

    Returns the retained count, i.e.
    ``min(ceiling, #{ i : σ_i > σ_max · rtol })``.

    This is deliberately the same arithmetic the call sites already ran.
    Its value is that the *meaning* is now stated once: it is a cap on
    ``κ_eff``, not a search for a separation, so it is well posed on the
    smooth spectra LORRAX actually has.

    WHY THE CEILING EXISTS.  A Gram route diagonalises ``A Aᴴ`` at the LARGER
    dimension, so the null space of a tall rank-deficient ``A`` arrives as
    round-off-sized POSITIVE eigenvalues.  A relative threshold counts them:
    measured on Na bands 1–24, ``A`` is ``(12288, 2032)`` and ``rtol=1e-8``
    selected **2034** — two directions more than the matrix algebraically
    has, and the capacity message that followed contradicted itself
    (``rank=2034``, ``nspinor*n_mu=2032``).  Clamping here makes the number a
    rank again; :meth:`RankReport.violations` refuses an UNCLAMPED overshoot
    so a caller that forgets the ceiling is not silently believed.
    """
    vals = _as_desc_list(spectrum)
    if not vals:
        return 0
    cut = vals[0] * float(rtol)
    n = 0
    for v in vals:
        if v > cut:
            n += 1
        else:
            break
    if ceiling is not None:
        n = min(n, max(0, int(ceiling)))
    return n


def noise_floor_rtol(n_rows, n_cols=None):
    """The discrepancy principle's relative cut, ``ε_mach·√n`` — REFERENCE ONLY.

    ``n`` is the larger matrix dimension (the length of the accumulation
    that generates the round-off).  Reported beside the actual cut so a
    reader can see how many decades of margin the criterion is holding above
    the noise floor **and that holding that margin is the point** — cutting
    AT this line is measured-catastrophic (module docstring, §R19).
    """
    n = int(n_rows) if n_cols is None else max(int(n_rows), int(n_cols))
    return _EPS64 * math.sqrt(max(n, 1))


class RankReport(object):
    """Everything a run should say about one truncation.

    Constructed by :func:`rank_report`.  Attributes:

    ``n_total``            spectrum length
    ``rank_criterion``     rank the criterion selects
    ``rank_used``          rank actually carried downstream (may differ:
                           mesh alignment, caller clamp)
    ``sigma_max``          largest spectral value
    ``sigma_min_kept``     smallest RETAINED value (``None`` if rank 0)
    ``kappa_eff``          ``sigma_max / sigma_min_kept`` — the achieved
                           amplification of the pseudo-inverse
    ``kappa_cap``          ``1/rtol`` — the amplification the criterion asked
                           for
    ``dropped_hi``/``dropped_lo``  spectral range DISCARDED by the criterion
    ``n_dropped_criterion``  directions discarded because σ ≤ σ_max·rtol
    ``n_padded_alignment``   NULL directions ADDED to reach a mesh-legal
                           extent (never discards; 0 when align == 1)
    ``n_dropped_alignment``  directions discarded by a device-grid round-DOWN.
                           MUST be 0 — a non-zero value means the numerics
                           depend on the device grid.
    ``n_dropped_closure``  directions discarded by ``common/spectral_closure``
                           because they were inside the degenerate block the
                           cut straddled.  Supplied by the caller; NOT a
                           violation.  It is a round-down like the one above
                           and is deliberately counted apart from it, because
                           the two have opposite standing: an alignment
                           round-down makes the retained set a function of the
                           DEVICE COUNT, while a closure drop makes it a
                           function of the SPECTRUM — which is the point.  It
                           can only raise ``sigma_min_kept``, so it can only
                           improve ``kappa_eff``, and check 1 stays sharp
                           across it.
    ``noise_rtol``         the discrepancy-principle reference line
    ``rank_at_noise``      rank a noise-floor cut would have retained
    ``overcomplete_margin``  see module docstring
    ``rank_ceiling``       the operator's STRUCTURAL rank ceiling, or
                           ``None`` when the caller supplied none (which is
                           reported, never assumed safe)
    ``rank_unclamped``     what the relative threshold selected BEFORE the
                           ceiling clamp.  Above the ceiling it is a count of
                           round-off, and check 4 refuses it
    ``discarded_weight``   ``Σ_dropped |λ| / Σ_all |λ|`` — the fraction of
                           the operator's trace the cut throws away.  EXACT
                           relative fit weight for a Gram ``C = P Pᴴ``; the
                           accuracy statement a rank count is not
    ``kappa_certified``    the largest ACHIEVED amplification any measurement
                           supports at this site, or ``None`` for
                           uncertified.  Read by :func:`certify`, never by
                           :meth:`violations`
    """

    __slots__ = ("label", "quantity", "rtol", "n_total", "rank_criterion",
                 "rank_used", "sigma_max", "sigma_min_kept", "kappa_eff",
                 "kappa_cap", "dropped_hi", "dropped_lo",
                 "n_dropped_criterion", "n_padded_alignment",
                 "n_dropped_alignment", "n_dropped_closure", "noise_rtol",
                 "rank_at_noise", "overcomplete_margin", "rank_loose",
                 "n_kept", "has_nan",
                 # The gate's inputs and its accuracy statement.
                 "rank_ceiling", "rank_unclamped", "discarded_weight",
                 "kappa_certified")

    def violations(self):
        """List of strings; empty when the truncation is self-consistent.

        Three checks, none of which assumes any spectral structure:

        1. ``κ_eff ≤ κ_cap`` — the invariant the criterion exists to enforce.
           Fires when something downstream RAISED the rank past the cut.
        2. ``n_dropped_alignment == 0`` — the retained set must not depend on
           the device grid.  ``n_dropped_closure`` is EXCLUDED from this
           count by construction: a spectral-closure drop is a round-down
           chosen by the spectrum, not by the mesh, and folding the two
           together would either blind this check or make it fire on the
           repair.  See ``common/spectral_closure``.
        3. ``σ_max`` finite and non-zero — a relative threshold is meaningless
           on a zero/NaN operator.  This is the documented ``nband``-window
           trap where "the SVD of a zero matrix returns rank 0"
           (wk_REL/reference/perlmutter/EXCITON_AND_PERF_SALVAGE.md §1.4).
        """
        out = []
        if self.has_nan:
            out.append(
                "the spectrum contains NaN/Inf — a RELATIVE threshold is "
                "meaningless here, and every retained direction is suspect.")
        if not (self.sigma_max > 0.0) or math.isinf(self.sigma_max) \
                or math.isnan(self.sigma_max):
            out.append(
                "the spectrum's largest value is %r — a RELATIVE threshold "
                "is meaningless here.  Classic cause: the band window lies "
                "outside the file's band extent, so ψ-at-centroids is "
                "identically zero (perlmutter salvage §1.4, the `nband` is "
                "an ABSOLUTE band index trap)." % (self.sigma_max,))
        if self.kappa_eff is not None and self.kappa_cap is not None \
                and self.kappa_eff > self.kappa_cap * (1.0 + 1e-12):
            out.append(
                "achieved amplification kappa_eff=%.3e EXCEEDS the cap "
                "kappa_cap=%.3e (rtol=%.1e).  Something downstream raised "
                "the retained rank past the criterion; the pseudo-inverse "
                "now amplifies round-off by more than was authorised "
                "(§R19: +41%% rank cost 5000 eV)."
                % (self.kappa_eff, self.kappa_cap, self.rtol))
        if self.n_dropped_alignment:
            out.append(
                "%d direction(s) were discarded for a DEVICE-GRID reason, "
                "not a physics one.  The retained subspace must not depend "
                "on the mesh; pad the sharded face instead."
                % (self.n_dropped_alignment,))
        if self.rank_ceiling is not None and self.rank_used > self.rank_ceiling:
            out.append(
                "%d direction(s) are being CARRIED from an operator that can "
                "algebraically hold at most %d.  The excess is null-space "
                "round-off: a Gram route diagonalises A Aᴴ at the LARGER "
                "dimension, so the null space of a tall rank-deficient A "
                "arrives as small POSITIVE eigenvalues and a relative "
                "threshold counts them (measured: (12288, 2032) selected "
                "2034 at rtol 1e-8, and the capacity line that followed "
                "contradicted itself).  A rank above the algebraic ceiling "
                "is not a rank.  Fix: pass ceiling=min(n_rows, n_cols) to "
                "select_rank." % (self.rank_used, self.rank_ceiling))
        return out

    def describe(self, indent="  "):
        """Multi-line diagnostic, one truncation per call. Safe to log always."""
        q = self.quantity
        lines = []
        kap = ("%.3e" % self.kappa_eff) if self.kappa_eff is not None else "n/a"
        smin = (("%.6e" % self.sigma_min_kept)
                if self.sigma_min_kept is not None else "n/a")
        lines.append(
            "%s[rank] %s: kept %d of %d %s  (criterion %d, +%d null pad, "
            "-%d closure-dropped, -%d grid-dropped)"
            % (indent, self.label, self.rank_used, self.n_total, q,
               self.rank_criterion, self.n_padded_alignment,
               self.n_dropped_closure, self.n_dropped_alignment))
        lines.append(
            "%s[rank]   retained %s range %.6e .. %s -> kappa_eff=%s "
            "(cap 1/rtol=%.3e at rtol=%.1e)"
            % (indent, q, self.sigma_max, smin, kap, self.kappa_cap,
               self.rtol))
        n_disc = self.n_total - self.n_kept
        if n_disc:
            lines.append(
                "%s[rank]   discarded %d %s in %.6e .. %.6e  "
                "= %d by the amplification cap + %d by DEGENERACY CLOSURE "
                "+ %d by GRID ALIGNMENT"
                % (indent, n_disc, q, self.dropped_lo, self.dropped_hi,
                   self.n_dropped_criterion, self.n_dropped_closure,
                   self.n_dropped_alignment))
        else:
            lines.append(
                "%s[rank]   discarded 0 %s — the cap bound nothing on this "
                "run (the operator is not over-complete here)"
                % (indent, q))
        lines.append(
            "%s[rank]   margin: loosening rtol by 1e-4 would admit %d more "
            "(+%.1f%%); noise-floor reference eps*sqrt(n)=%.2e would admit "
            "%d more.  R19 anchor: +41%% cost 5000 eV — do NOT loosen when "
            "this margin is large."
            % (indent, max(0, self.rank_loose - self.rank_criterion),
               100.0 * self.overcomplete_margin, self.noise_rtol,
               max(0, self.rank_at_noise - self.rank_criterion)))
        # The ACCURACY statement.  Reported unconditionally beside the rank
        # count precisely because the rank count is not one (TASTE rule 12,
        # and §"WHAT IS REFUTED AS A GATE" above).
        lines.append(
            "%s[rank]   discarded weight %.3e of tr|operator| "
            "(= the relative fit weight the cut throws away; EXACT for a "
            "Gram C = P Pᴴ).  Structural ceiling %s%s."
            % (indent, self.discarded_weight,
               ("none supplied" if self.rank_ceiling is None
                else "%d" % self.rank_ceiling),
               ("" if (self.rank_ceiling is None
                       or self.rank_unclamped <= self.rank_ceiling)
                else ", which CLAMPED the criterion from %d (the excess was "
                     "null-space round-off, not rank)"
                     % (self.rank_unclamped,))))
        if self.kappa_certified is not None and self.kappa_eff is not None:
            # WHY THE REACHABILITY CLAUSE IS HERE.  select_rank retains
            # sigma_i > sigma_max*rtol, so kappa_eff < 1/rtol = kappa_cap
            # ALWAYS.  When kappa_cap <= kappa_certified the certify() arm
            # that compares them is arithmetically incapable of firing, and
            # "achieved 9.9e7 (0.99x of the ceiling)" reads exactly like a
            # measured pass.  Say which it is — same rule the None branch
            # below already follows.
            _inert = (self.kappa_cap is not None
                      and self.kappa_cap <= self.kappa_certified * (1.0 + 1e-12))
            lines.append(
                "%s[rank]   certified kappa ceiling %.3e for this site; "
                "achieved %.3e (%.2gx of it)%s"
                % (indent, self.kappa_certified, self.kappa_eff,
                   self.kappa_eff / self.kappa_certified,
                   ("" if self.n_dropped_criterion else
                    " — but the cap bound NOTHING, so the gate does not "
                    "apply (see certify())")
                   + ("" if not _inert else
                      " — and at rtol=%.1e the criterion's own cap %.3e is "
                      "AT OR BELOW that ceiling, so kappa_eff cannot exceed "
                      "it by construction: this arm's silence is arithmetic, "
                      "not a measurement"
                      % (self.rtol, self.kappa_cap))))
        elif self.kappa_certified is None:
            lines.append(
                "%s[rank]   certified kappa ceiling: NONE for this site — "
                "no measurement in this tree supports one, so the gate warns "
                "rather than refuses.  That is an absence, not a pass."
                % (indent,))
        for v in self.violations():
            lines.append("%s[rank]   ** VIOLATION: %s" % (indent, v))
        return "\n".join(lines)


def rank_report(spectrum, rtol, *, label="truncation",
                quantity="singular values", rank_used=None,
                n_rows=None, n_cols=None, n_dropped_closure=0,
                rank_ceiling=None, kappa_certified=None):
    """Build a :class:`RankReport` from a host-side spectrum.

    ``rank_used`` is the rank actually carried downstream.  Leave it ``None``
    to mean "the criterion's rank was used".  When it is LARGER than the
    criterion's rank the excess is counted as ``n_padded_alignment`` (null
    directions added to reach a mesh-legal extent — harmless, and the fix
    for the round-down defect); when it is SMALLER the deficit is counted as
    ``n_dropped_alignment``, which :meth:`RankReport.violations` reports as a
    defect because it makes the numerics depend on the device grid.

    ``n_dropped_closure`` is how many of any such deficit ``common/
    spectral_closure`` accounts for — the members of a straddled degenerate
    block that the cut dropped.  A caller that ran the closure guard MUST
    pass it, and the amount is attributed to ``n_dropped_closure`` instead of
    ``n_dropped_alignment`` so that check 2 keeps meaning "the mesh changed
    the physics" rather than firing on the symmetry repair.  Anything left
    over after this attribution is still an alignment round-down and still a
    violation, which is what keeps the accounting honest rather than merely
    quiet.

    ``rank_ceiling`` is the operator's STRUCTURAL rank ceiling (see
    :func:`select_rank`).  The report applies the same clamp the call site
    should have applied, keeps the unclamped number in ``rank_unclamped`` so
    the clamp is visible in the log, and refuses (check 4) when ``rank_used``
    exceeds it.

    **Declare a ceiling only on a report whose ``rank_used`` is a SELECTION.**
    A report whose ``rank_used`` is a *carried extent* — one that includes
    mesh-alignment null padding, like ``htransform``'s ``rank`` — legitimately
    exceeds the ceiling by the pad and must pass ``rank_ceiling=None``.  The
    two are different quantities and check 4 is about the first one; the
    padded report's own accounting (``n_padded_alignment``) already covers the
    second.

    ``kappa_certified`` is the largest ACHIEVED amplification any measurement
    supports at this site — ``None`` for uncertified, which :func:`certify`
    downgrades to a warning and says so.  It is deliberately NOT read by
    :meth:`RankReport.violations`: self-consistency and certification are
    different questions and conflating them is how a run that satisfied
    ``κ_eff ≤ κ_cap`` exactly came to report 54.4 eV at exit 0.
    """
    raw = [float(v) for v in spectrum]
    has_nan = any(math.isnan(v) or math.isinf(v) for v in raw)
    vals = _as_desc_list(raw)
    r = RankReport()
    r.has_nan = has_nan
    r.label = label
    r.quantity = quantity
    r.rtol = float(rtol)
    r.n_total = len(vals)
    r.kappa_cap = (1.0 / float(rtol)) if float(rtol) > 0.0 else float("inf")
    r.sigma_max = vals[0] if vals else 0.0
    r.rank_ceiling = None if rank_ceiling is None else int(rank_ceiling)
    r.kappa_certified = (None if kappa_certified is None
                         else float(kappa_certified))
    r.rank_unclamped = select_rank(vals, rtol)
    r.rank_criterion = select_rank(vals, rtol, ceiling=r.rank_ceiling)
    r.rank_used = int(r.rank_criterion if rank_used is None else rank_used)

    # Only the directions that were BOTH selected and carried count as kept.
    n_kept_real = min(r.rank_used, r.rank_criterion, r.n_total)
    r.n_kept = n_kept_real
    r.sigma_min_kept = vals[n_kept_real - 1] if n_kept_real > 0 else None
    r.kappa_eff = (r.sigma_max / r.sigma_min_kept
                   if (r.sigma_min_kept not in (None, 0.0)
                       and r.sigma_max > 0.0) else None)

    r.n_dropped_criterion = r.n_total - r.rank_criterion
    r.n_padded_alignment = max(0, r.rank_used - r.rank_criterion)
    # The closure drop is subtracted FIRST, so what remains in
    # ``n_dropped_alignment`` is the mesh's doing and check 2 stays sharp.
    # Clamped to the deficit actually present: a caller claiming more closure
    # drops than there are missing directions cannot manufacture credit.
    _deficit = max(0, r.rank_criterion - r.rank_used)
    r.n_dropped_closure = min(int(n_dropped_closure), _deficit)
    r.n_dropped_alignment = _deficit - r.n_dropped_closure
    # The discarded band is everything the criterion cut, PLUS anything a
    # round-down took off the bottom of the retained block.
    n_disc = r.n_total - n_kept_real
    if n_disc > 0:
        r.dropped_hi = vals[n_kept_real]
        r.dropped_lo = vals[-1]
    else:
        r.dropped_hi = 0.0
        r.dropped_lo = 0.0
    r.n_dropped_criterion = r.n_total - r.rank_criterion

    nr = r.n_total if n_rows is None else n_rows
    r.noise_rtol = noise_floor_rtol(nr, n_cols)
    r.rank_at_noise = select_rank(vals, r.noise_rtol)
    r.rank_loose = select_rank(vals, float(rtol) * 1e-4)
    r.overcomplete_margin = (
        (r.rank_loose - r.rank_criterion) / float(r.rank_criterion)
        if r.rank_criterion > 0 else 0.0)
    # The accuracy statement: what fraction of the operator's trace the cut
    # discards.  ``|v|`` because the transverse CCT is Hermitian INDEFINITE
    # and both signs are physical there; for a PSD Gram it is the plain
    # trace and the identity ``Σλ = tr C = ‖P‖_F²`` makes this the relative
    # fit weight exactly.
    _tot = math.fsum(abs(v) for v in vals)
    _kept = math.fsum(abs(v) for v in vals[:n_kept_real])
    r.discarded_weight = (
        (_tot - _kept) / _tot if _tot > 0.0 and not has_nan else 0.0)
    return r


# ═══════════════════════════════════════════════════════════════════════
# THE GATE
# ═══════════════════════════════════════════════════════════════════════

#: The three modes, in decreasing authority.  Same vocabulary as
#: ``common/spectral_closure.MODES`` and ``common/band_degeneracy.MODES`` —
#: deliberately, because a user who has met one should not have to learn a
#: second spelling for "this guard is off".
POLICY_MODES = ("refuse", "warn", "off")

#: THE ONE PLACE THE DEFAULT IS DECIDED.  Flipping this line flips every gate
#: in this module; there is no second literal in the tree.
DEFAULT_POLICY_MODE: str = "refuse"

#: NAME of the environment dial the DRIVERS read.  This module never reads it
#: — ``tests/test_layering.py``: an L2 module is a function of its arguments.
#: The name lives here once; each call site does its own ``os.environ`` lookup
#: and hands the answer to :func:`resolve_policy_mode`.
POLICY_MODE_ENV = "LORRAX_RANK_POLICY"

#: Largest ACHIEVED amplification certified for a POSITIVE-DEFINITE overlap
#: Gram (the ζ charge route, the htransform ψ@centroids Gram-eigh).
#:
#: 1e8, measured on two decks sharing no code path downstream of the fit:
#:
#:   * R19 (``docs/dev/notes/ladder_rung1_R19_zeta_rcond.md``), MoS2 4×4,
#:     nb=1024, μ≈10k — rcond 1e-8 (κ ≤ 1e8) gives eqp0 3.1350; 1e-10
#:     (κ ≈ 1e10) gives −206.83; 1e-12 (κ ≈ 1e12) gives −5049.59.
#:   * Si 4×4×4 SYM/SOC 128-band at 1776 centroids (register 2026-08-15) —
#:     κ_eff 9.7e9…1.0e10, Σ_c MAE 54.4 eV, max 100.3 eV, exit 0, no banner.
#:     The same deck at 600 centroids does not truncate at all: MAE 0.90 eV.
#:
#: It is the same boundary ``gw_config`` states from the other side: the
#: ``zeta_rcond`` default sits at the LOW end of the recovery plateau
#: 1e-8…1e-4, i.e. κ_cap ∈ [1e4, 1e8].
KAPPA_CERTIFIED_GRAM: float = 1.0e8

#: Ceiling for an INDEFINITE operator carried on a shifted (ridged) path.
#: A positive ridge on an indefinite matrix moves negative eigenvalues TOWARD
#: zero, so it is not a regularizer; above κ ~ 1e12 it is measured actively
#: harmful (register ``bispinor``, job 7885987).  The instrument on that path
#: is a LOWER bound on κ, which is the right direction for a gate that fires
#: when the number is large.
KAPPA_INDEFINITE_MAX: float = 1.0e12

#: Loose ceiling on :attr:`RankReport.discarded_weight`.  Four decades above
#: anything healthy measured here (the ζ cuts discard ≲1e-5 of the trace even
#: when they discard a third of the RANK — which is the whole reason rank is
#: not the accuracy statement).  It exists to catch a cut eating real weight,
#: not to police ordinary conditioning.
DISCARDED_WEIGHT_MAX: float = 1.0e-3


class RankPolicyError(RuntimeError):
    """A truncation is outside the regime any measurement certifies."""


def resolve_policy_mode(explicit=None) -> str:
    """Validate a policy mode and supply :data:`DEFAULT_POLICY_MODE`.

    ``explicit`` is whatever the driver read — typically
    ``os.environ.get(POLICY_MODE_ENV)``, ``None`` when unset.  A mis-spelled
    mode RAISES rather than falling back to ``off``: a gate silently disarmed
    by a typo is worse than no gate, because the log then reads clean.
    """
    if explicit is None:
        return DEFAULT_POLICY_MODE
    mode = str(explicit).strip().lower()
    if mode not in POLICY_MODES:
        raise ValueError(
            f"{POLICY_MODE_ENV}={mode!r} is not one of {POLICY_MODES}.  A "
            f"mis-spelled policy mode is not silently 'off'.")
    return mode


def certify(report, *, site, mode=None, cause="", fix="", log=None):
    """THE GATE.  Refuse a truncation outside the certified regime.

    ``report`` is a :class:`RankReport`.  ``site`` names the truncation in the
    message.  ``mode`` is a :data:`POLICY_MODES` member (the driver resolves
    it from :data:`POLICY_MODE_ENV`); ``None`` means
    :data:`DEFAULT_POLICY_MODE`.  ``cause``/``fix`` are the site's own
    sentences about what over-completed and what deck key changes it — a
    refusal that names no deck key is a dead end for the operator.

    Returns the list of findings (empty when clean).  Under ``refuse`` a
    non-empty list RAISES :class:`RankPolicyError`; under ``warn`` it is
    logged with ``***`` markers; ``off`` returns it without acting.

    THE TWO FINDINGS, and both are conditioned on the cut having BOUND:

    1. ``κ_eff > report.kappa_certified`` — the pseudo-inverse amplifies
       round-off by more than any measurement at this site supports.
    2. ``discarded_weight > DISCARDED_WEIGHT_MAX`` — the cut is throwing away
       real weight of the operator, not just null directions.

    When ``n_dropped_criterion == 0`` neither applies: the criterion made no
    choice, the spectrum ended on its own, and a refusal here would refuse
    the INPUT rather than the policy.  That is not a loophole — it is the
    case of the Si 960-point anchor set at production settings (768 of 768
    retained), which carries the best BerkeleyGW agreement on record.

    An UNCERTIFIED site (``report.kappa_certified is None``) cannot raise
    finding 1 and says so out loud rather than reporting a clean bill: an
    absence of a ceiling is an absence, not a pass.
    """
    return certify_numbers(
        kappa_eff=report.kappa_eff, n_dropped=report.n_dropped_criterion,
        n_total=report.n_total, discarded_weight=report.discarded_weight,
        kappa_certified=report.kappa_certified, quantity=report.quantity,
        # The criterion's OWN cap, so the gate can tell a measured pass
        # from an arithmetically impossible finding.  See
        # :func:`certify_numbers`.
        kappa_cap=report.kappa_cap,
        site=site, mode=mode, cause=cause, fix=fix, log=log)


def certify_numbers(*, kappa_eff, n_dropped, n_total, discarded_weight,
                    kappa_certified, quantity="directions", site,
                    kappa_cap=None, mode=None, cause="", fix="", log=None):
    """:func:`certify` for a site that already holds the numbers.

    The array-shaped host sites (``common/zeta_projection``, which reduces
    over q before anything reaches host) have the three quantities and no
    single spectrum to build a :class:`RankReport` from.  They call this;
    :func:`certify` calls it too, so there is ONE set of thresholds and ONE
    wording rather than a second copy that drifts.

    ``kappa_eff`` / ``n_dropped`` / ``discarded_weight`` must be the WORST
    over whatever the site reduced (largest κ, largest drop, largest
    weight): a gate reported on an average is a gate that cannot fire.

    ``kappa_cap`` IS THE CRITERION'S OWN CAP (``1/rtol``), AND WITHOUT IT
    THIS GATE CANNOT TELL A PASS FROM A TAUTOLOGY.  :func:`select_rank`
    retains exactly ``σ_i > σ_max·rtol``, so ``σ_min_kept > σ_max·rtol``
    and therefore ``kappa_eff < 1/rtol = kappa_cap`` — ALWAYS, by
    construction.  Finding 1 compares ``kappa_eff`` against
    ``kappa_certified``; when ``kappa_cap ≤ kappa_certified`` that
    comparison is arithmetically incapable of being true and the gate
    returns clean from a path that tested nothing.  That is the case at
    EVERY default in this tree today: ``zeta_rcond = 1e-8`` and
    htransform's ``rtol = 1e-8`` both give ``kappa_cap = 1e8 =
    KAPPA_CERTIFIED_GRAM``.  The arm becomes live exactly when an operator
    loosens the dial below the certified plateau — which is the registered
    catastrophe (rcond 1e-10 / 1e-12, κ_eff 9.7e9 … 1e12) and is what the
    gate is FOR.

    So the inert case is ANNOUNCED rather than reported as a pass, the same
    way ``kappa_certified is None`` already is: "an absence is not a pass",
    and neither is an impossibility.  ``TASTE.md`` 2026-08-06,
    "a check that cannot fail is not evidence", and rule 18's corollary —
    a gate reporting nothing looks identical whether it found nothing or
    checked nothing, so state which.  The discarded-weight arm is
    independent of ``rtol`` and stays live either way.
    """
    m = resolve_policy_mode(mode)
    findings = []
    bound = int(n_dropped) > 0
    _kappa_arm_inert = (
        bound and kappa_certified is not None and kappa_cap is not None
        and float(kappa_cap) <= float(kappa_certified) * (1.0 + 1e-12))
    if _kappa_arm_inert and m != "off":
        emit = log if log is not None else print
        emit(
            "  [rank-policy] %s: the kappa arm is STRUCTURALLY INERT at this "
            "rtol — the criterion's own cap 1/rtol=%.3e is at or below the "
            "certified ceiling %.3e, so kappa_eff < the ceiling by "
            "construction and this arm cannot fire.  Its silence is "
            "arithmetic, not a measurement.  It becomes live only below "
            "rtol=%.1e, which is where both registered catastrophes sat.  "
            "The discarded-weight arm (ceiling %.1e) is rtol-independent "
            "and IS live."
            % (site, float(kappa_cap), float(kappa_certified),
               1.0 / float(kappa_certified), DISCARDED_WEIGHT_MAX))
    if bound and kappa_certified is not None and kappa_eff is not None \
            and float(kappa_eff) > float(kappa_certified):
        findings.append(
            "the truncation BOUND (%d of %d %s discarded) and the achieved "
            "amplification kappa_eff=%.3e is %.3gx the CERTIFIED ceiling "
            "%.3e for this site.  Everything measured at or below the "
            "ceiling is right; everything measured at ~1e10 and above is "
            "wrong by electron-volts (R19: eqp0 3.1350 -> -206.83 -> "
            "-5049.59 across rcond 1e-8/1e-10/1e-12; Si 4x4x4 at 1776 "
            "centroids: Sigma_c MAE 54.4 eV at kappa_eff 9.7e9, exit 0)."
            % (int(n_dropped), int(n_total), quantity, float(kappa_eff),
               float(kappa_eff) / float(kappa_certified),
               float(kappa_certified)))
    if bound and float(discarded_weight) > DISCARDED_WEIGHT_MAX:
        findings.append(
            "the cut discards %.3e of tr|operator| — above the %.1e ceiling. "
            "That is the fraction of the operator's own weight thrown away "
            "(EXACT relative fit weight for a Gram C = P Pdag), so this is "
            "an ACCURACY finding, not a conditioning one: the discarded "
            "directions are carrying real content."
            % (float(discarded_weight), DISCARDED_WEIGHT_MAX))
    if not findings:
        return findings
    head = ("[rank-policy] %s: the truncation is outside the regime any "
            "measurement in this tree certifies." % (site,))
    body = "\n".join("  - " + f for f in findings)
    tail = ""
    if cause:
        tail += "\n  cause : " + cause
    if fix:
        tail += "\n  fix   : " + fix
    tail += ("\n  override: %s=warn continues and leaves a trace in the log; "
             "=off disarms the gate.  Do NOT reach for a looser rtol instead "
             "— it is a physical convergence axis with a measured plateau, "
             "and loosening it by four decades inflated the rank by 41%% and "
             "moved a 2.2 eV gap to -5049 eV (R19)."
             % (POLICY_MODE_ENV,))
    msg = head + "\n" + body + tail
    if m == "refuse":
        raise RankPolicyError(msg)
    if m == "warn":
        emit = log if log is not None else print
        for line in msg.splitlines():
            emit("*** " + line)
    return findings


# ═══════════════════════════════════════════════════════════════════════
# Scale-aware independence — the rank decision inside an orthogonalization
# ═══════════════════════════════════════════════════════════════════════

#: Relative floor for "this probe adds an independent direction".
#:
#: ``sqrt(eps) ≈ 1.49e-8`` — the classical modified-Gram-Schmidt rank
#: threshold, and the same number ``centroid/pivoted_cholesky`` already stops
#: its greedy on.  It is applied RELATIVE to the probe's own scale
#: (:func:`probe_is_independent`), never as an absolute norm.
PROBE_RTOL: float = 1.4901161193847656e-08


def probe_is_independent(norm_after, norm_before, scale=0.0, *,
                         rtol=PROBE_RTOL) -> bool:
    """Did a probe survive projection with enough of itself left to trust?

    ``norm_after``  residual norm after projecting out the accepted columns.
    ``norm_before`` the probe's own norm before projection.
    ``scale``       a norm scale for the FAMILY of probes — typically the
                    largest ``norm_before`` over all of them.  Guards the
                    case where one probe is itself negligible: without it,
                    a probe of norm 1e-30 that projects to 0.9e-30 would
                    "survive" on its own relative test.

    WHY THIS IS NOT AN ABSOLUTE FLOOR, and it matters at production size.
    A probe's coefficient vector scales like ``|psi|`` at ONE sample point,
    which falls like ``1/sqrt(N_mu)``.  An absolute floor is therefore a
    SYSTEM-SIZE-DEPENDENT refusal wearing a numerical constant's clothes: it
    passes on a fixture and refuses on the production deck.  Measured — a
    hard-coded ``1e-6`` in ``bse/bse_w_exact`` discarded every fixed-order
    Kramers probe on a valid fully relativistic LiF WFN that had already
    passed the density-symmetry audit (TRS residual 1.42e-13, 48/48 spatial
    ops at 6.31e-12) and the Theta-closure gate, reporting
    "Kramers probes spanned 0/2 of a TRIM block".
    """
    nb = float(norm_before)
    na = float(norm_after)
    ref = max(nb, float(scale))
    if not (ref > 0.0) or math.isnan(na) or math.isinf(na):
        return False
    return na > float(rtol) * ref


# ═══════════════════════════════════════════════════════════════════════
# Deferred refusal — for cuts that live inside a jit
# ═══════════════════════════════════════════════════════════════════════
#
# Same division of labour ``common/spectral_closure`` documents and
# ``centroid/pivoted_cholesky`` states: a jitted kernel cannot raise, so it
# REPORTS through a host callback and the next host seam REFUSES.  Without
# this the ζ gate would mean one thing at the htransform host site and
# another at the ζ device site, which is exactly the "one seam refuses and
# another whispers" failure ``band_degeneracy`` names.

_PENDING: list[str] = []


def note_device_finding(where, detail) -> None:
    """Record a device-site policy finding for the next host seam.

    Called from ``jax.debug.callback``.  It records; it does not raise,
    because raising out of a host callback inside XLA is not a contract this
    tree relies on.  :func:`raise_if_pending` is the refusal.
    """
    _PENDING.append(f"{where}: {detail}")


def pending() -> list[str]:
    """The findings recorded so far.  Read-only view, for gates."""
    return list(_PENDING)


def raise_if_pending(where="rank truncation", *, mode=None, log=None) -> None:
    """Refuse for any device-site finding recorded since the last call.

    Placed at the first host seam after a jitted truncation.  ALWAYS clears,
    so a later stage cannot inherit an earlier stage's finding.
    """
    m = resolve_policy_mode(mode)
    found, _PENDING[:] = list(_PENDING), []
    if not found or m == "off":
        return
    body = "\n".join("  - " + f for f in found)
    msg = (f"[rank-policy] {where}: {len(found)} truncation(s) fell outside "
           f"the certified regime.\n{body}\n"
           f"  override: {POLICY_MODE_ENV}=warn continues and leaves a trace; "
           f"=off disarms the gate.")
    if m == "refuse":
        raise RankPolicyError(msg)
    emit = log if log is not None else print
    for line in msg.splitlines():
        emit("*** " + line)
