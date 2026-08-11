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
"""
from __future__ import annotations

import math

__all__ = [
    "select_rank",
    "noise_floor_rtol",
    "RankReport",
    "rank_report",
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


def select_rank(spectrum, rtol):
    """The criterion: how many directions survive an amplification cap.

    Parameters
    ----------
    spectrum : iterable of float
        Spectrum of the operator being pseudo-inverted (singular values, or
        eigenvalues of a Hermitian PSD Gram).  Order does not matter.
    rtol : float
        Reciprocal of the amplification cap, ``rtol = 1/κ_cap``.

    Returns the retained count, i.e. ``#{ i : σ_i > σ_max · rtol }``.

    This is deliberately the same arithmetic the call sites already ran.
    Its value is that the *meaning* is now stated once: it is a cap on
    ``κ_eff``, not a search for a separation, so it is well posed on the
    smooth spectra LORRAX actually has.
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
    """

    __slots__ = ("label", "quantity", "rtol", "n_total", "rank_criterion",
                 "rank_used", "sigma_max", "sigma_min_kept", "kappa_eff",
                 "kappa_cap", "dropped_hi", "dropped_lo",
                 "n_dropped_criterion", "n_padded_alignment",
                 "n_dropped_alignment", "n_dropped_closure", "noise_rtol",
                 "rank_at_noise", "overcomplete_margin", "rank_loose",
                 "n_kept", "has_nan")

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
        for v in self.violations():
            lines.append("%s[rank]   ** VIOLATION: %s" % (indent, v))
        return "\n".join(lines)


def rank_report(spectrum, rtol, *, label="truncation",
                quantity="singular values", rank_used=None,
                n_rows=None, n_cols=None, n_dropped_closure=0):
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
    r.rank_criterion = select_rank(vals, rtol)
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
    return r
