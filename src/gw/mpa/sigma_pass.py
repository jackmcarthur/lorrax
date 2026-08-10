"""The MPA self-energy, run one pole at a time through the existing integration.

This is the consumer half of the multipole stage.  ``sigma_routing`` says
which certified rule serves one complex pole on one branch; this module
takes the staged fit store, walks it pole by pole in the pinned order, and
turns each pass's ``(B_p, Omega_p)`` slab into a contribution to
``Sigma_c(omega, k, m, n)`` using the SAME device tau loop, accumulator and
sharded tile sink the two-point plasmon-pole path uses.

THE SHAPE OF THE LOOP, AND WHY IT IS THIS SHAPE
-----------------------------------------------
``THEORY_mpa_implementation`` section 7.5 fixes it::

    for p in fit_driver.pole_pass_order(n_p):          # ascending, pinned
        Omega_p, B_p = mpa_store.read_pole_slice(fit, p)
        for branch in the four (A-space x omega-half) branches:
            for group in the branch's window groups:
                integrate the group's tau nodes into the branch tiles
        accumulate

The correctness lemma is that ``W_c(tau) = sum_p B_p e^{-i Omega_p tau}``
enters every downstream contraction linearly, so a sum over poles is
unchanged when it is re-associated into one pass per pole.  It is exact in
exact arithmetic and NOT bit-exact in floating point, which is why the pass
order is pinned rather than convenient.  ``accumulate_over_pole_passes``
proves the lemma on arrays; this module is the production analogue of it,
and ``tests/test_mpa_sigma_pass.py`` runs the two against each other through
the real window machinery.

THE NARROW-POLE CORNER, AND THE ROUTING DECISION TAKEN FOR IT
-------------------------------------------------------------
The audited first-light field (si_mpa_0808, 81 432 576 poles) has exactly
one corner that the certified complex-pole route refuses: its zeroth
percentile width, ``Gamma = 3.96e-5 eV``, on a crossing branch, where the
composite rule's dimensionless bandwidth ``A = f_max/Gamma`` reaches
``2.4e5``.  That refusal is correct -- resolving a quarter of a million
oscillations is not the right answer at any node count, and raising the cap
by five octaves does not move it.  But a Sigma that refuses is not a Sigma,
and the pole cannot be dropped: it carries weight like any other.

**The decision: a pole whose width is below the two-point path's own
crossing regularization width ``xi`` is routed through the two-point path's
crossing machinery, at that ``xi``, as a real pole.**  The threshold is
``xi`` exactly, and the reason it is exactly ``xi`` and not a tuned number
is the smearing that ``xi`` already represents:

* On the two-point path ``xi`` is a BROADENING, not an error term.  The
  crossing quadrature fits ``G_hgl(u/xi)/xi``, which is ``1/u`` smeared over
  a width ``xi``, and section 5 of the theory guide records that the
  conditioning floor engages on every default run and raises ``xi`` from the
  requested 0.25 eV to 0.476 eV.  So the two-point path resolves nothing
  finer than about half an electron-volt near the crossing.
* A pole at ``Gamma = 4e-5 eV`` is four orders below that.  Convolved with
  the same smearing it is INDISTINGUISHABLE from a real pole: the two
  Lorentzians ``1/(u + i*Gamma)`` and ``1/(u + i*0)`` differ only inside a
  region of width ``Gamma``, and the ``xi`` machinery integrates over a
  region ``xi >> Gamma`` wide.  GN-PPM's crossing treatment is exactly the
  ``Gamma -> 0`` limit of the complex one with the regularization put back,
  so routing such a pole there is not an approximation chosen for
  convenience; it is the same number computed by the route that can compute
  it.  Above ``xi`` the pole's own width IS resolvable and is the only
  broadening in the integrand, and the complex route is the one that is
  right.

Two consequences are worth naming because they are what makes the decision
cheap rather than merely defensible.  First, the split BOUNDS the complex
route's bandwidth: after it, ``Gamma >= xi`` for every pole the composite
rule sees, so ``A = f_max/Gamma <= f_max/xi``, which is the same order the
two-point path's own ``A_core = 2T/xi`` lives at, and the 2.4e5 corner
cannot recur.  Second, it is measured and announced per pass, not assumed:
:func:`format_pass_report` prints the count and the ``|B|`` mass fraction
that took the legacy branch, so a field that is mostly narrow poles is
visible as such instead of silently becoming a plasmon-pole run.

THE SLAB, AND WHAT A WINDOW HAS TO COVER
-----------------------------------------
``route_pole`` plans ONE scalar pole.  A pass is a slab: ``(n_q, N_mu,
N_mu)`` complex poles at once, with a spread in both ``Re Omega`` and
``Gamma``.  A window built for a slab therefore has to serve every element
of it, so each rule is built at the SET's worst parameters rather than at a
representative pole:

``crossing``       truncation is set by the SMALLEST width in the set
                   (slowest decay, longest ``t_max``); the panel resolution
                   by the LARGEST beat frequency.  A larger ``Gamma`` decays
                   faster and is integrated at least as well, so this is
                   conservative in the only direction that matters.
``sign_definite``  decay is set by the SMALLEST Laplace edge ``x_min`` and
                   the oscillation by the LARGEST ``Gamma`` plus the
                   interval's own spread.

One term is new relative to the scalar router and is not an option: the
crossing core's beat frequency gains the slab's own ``Re Omega`` spread.
``route_pole`` may take ``E_ref_B = a_p`` exactly because it plans a single
pole; a slab's ``E_ref_B`` is the set minimum and the residual
``a - E_ref_B`` is a real phase in the integrand.  Dropping it would build a
rule for a narrower band than the integrand occupies.

Because the Laplace interval ratio ``R = x_max/x_min`` is what a
sign-definite rule pays for, the slab is partitioned in ``Re Omega`` into
geometric buckets chosen so that no bucket's ratio exceeds ``r_max``
(default 100, the top of the shipped ``complex_laplace_width`` R grid).  On
the audited field's 0.26 eV -- 6.5 keV span that is three buckets, not
fifteen: the bucket edge grows like ``r_max * x_min``, so each bucket is
about two decades wide.

HOW A PANE SAYS WHICH MODES IT HOLDS, AND WHAT THAT COST
---------------------------------------------------------
A pane's membership is an INDEX SET -- ascending flat indices into the
``(n_q, N_mu, N_mu)`` pole field -- and not a boolean of that shape.  The
difference is not stylistic.  The width clause partitions the
non-crossing branch into ~218 panes on the audited field; at 81.4 MB per
full-size boolean that is **17.8 GB of masks**, which the 2026-08-09
memory row measured as the whole of a pass's excess over the two-point
path (~24 GB resident to describe a 4.55 GB pole slab).  As index sets the
same partition costs one index per LIVE mode across every pane together,
because the panes partition the live set: 4 bytes x 79.2 million wide
modes = 0.32 GB, and the planner's own arithmetic stops touching the
81-million-element field once per pane and starts touching each pane's own
values.

What this change is NOT is a cost fix for the tau loop, and the
measurement says so plainly.  The mask-dependent stage of a tau node --
building ``W(tau) = B e^{-i(Omega - E_ref)tau}`` over the field -- is
4.6 to 11.0 ms across runs at production shapes, against a node of 139 to
175 ms measured on a SOLO A100 (nid001044, 2026-08-09); the rest is the
``G(tau)`` formation, the k-axis transforms and the band projection, none
of which know a mask exists.  The direct form of the same statement: the
kernel measures the same wall at ``mask=all`` as at ``mask=1/218`` and at
``mask=1/2000``.  A pane at 1/218 occupancy therefore costs what a full
one costs; the node runs at 2.4 TFLOP/s fp64, a quarter of that card's
non-tensor peak, so the "0.4 % utilization" of the three-way table is a
fraction of MODES and not a fraction of the machine.  What stands between
this path and the two-point floor is the PANE COUNT, and that is the
owner's registered row (a slab-aware width clause).  Nothing here touches
it: same panes, same membership, same nodes, same weights, bit-identical
Sigma.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not unfold the store's q axis.  A fit store written on the symmetry
wedge carries ``n_q = 8`` where the Sigma kernel's k-q sums want the full
BZ, and unfolding a POLE FIELD is not the same operation as unfolding
``W``: the residues ``B_p`` transform as ``W`` does (centroid double
permute, L-phase, and a conjugation on the time-reversed members) while the
pole POSITIONS only permute -- and what time reversal does to a pole in the
closed fourth quadrant is precisely the question this tree has got wrong
four times in other guises.  :func:`refuse_wedge_pole_slab` therefore
refuses a wedge store by name and says what has to be certified first,
rather than guessing a conjugation and returning a finite, plausible Sigma.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from common.units import RYD_TO_EV

__all__ = [
    "DEFAULT_LAPLACE_RATIO_MAX",
    "MAX_WIDTH_SPLIT_LEAVES",
    "PARTIAL_FORMAT_VERSION",
    "PassRecord",
    "WindowGroup",
    "combine_pass_partials",
    "compute_mpa_sigma_c_omega_grid",
    "flat_index_dtype",
    "format_pass_report",
    "narrow_pole_threshold_ry",
    "plan_branch_groups",
    "refuse_wedge_pole_slab",
    "resolve_pole_subset",
    "resolve_pass_poles",
    "run_pass_branch",
    "split_pass_by_width",
    "write_pass_partial",
]


#: Ceiling on a sign-definite window's Laplace interval ratio
#: ``R = x_max/x_min``.  100 is the fourth of the five rungs the shipped
#: ``complex_laplace_width/v1`` catalog measures (10, 21.54, 46.42, 100,
#: 215.44); the composite route this module actually builds is certified by
#: positivity rather than by that catalog, but sizing the buckets to the
#: catalog's own grid keeps the two routes asking for the same shapes and
#: keeps a future table lookup a substitution rather than a redesign.
DEFAULT_LAPLACE_RATIO_MAX = 100.0

#: Floor on a Laplace edge, in Ry.  Below this the interval is degenerate
#: and the rule builder's own gate says so.
_X_FLOOR_RY = 1.0e-12


@dataclass(frozen=True)
class WindowGroup:
    """One ``_SigmaWindow`` list and the mode subset it is built for.

    The two-point path carries ONE ``base_mask_B`` per branch and lets each
    window refine it by comparing ``Omega_q`` against a threshold.  That
    comparison is a real-number one and does not survive a complex pole
    field, so on this path the refinement is done here, on the host, and
    each group ships an explicit selection with ``mask_B_mode='all'``
    windows.

    THE SELECTION IS AN INDEX SET, NOT A MASK, AND THAT IS THE WHOLE
    POINT OF THIS CLASS.  A pane's membership used to be a full-size
    ``(n_q, N_mu, N_mu)`` boolean -- 81.4 MB on the production deck --
    and the width clause demands ~218 panes on the non-crossing branch,
    so the planner held **17.8 GB of masks** whose every byte but one in
    218 was ``False``.  ``idx_B`` is the same membership written as
    ascending flat indices into that field, so the planner's footprint is
    ``sum(live) * itemsize`` -- one index per live mode across ALL panes
    together, because the panes partition the live set -- rather than
    ``panes * 81.4 MB``.  Nothing about WHICH modes are in a pane changes;
    :meth:`dense_mask_B` reconstructs the identical boolean whenever a
    consumer genuinely needs one, and exactly one such mask is alive at a
    time.

    ``omega_operand`` is what the tau kernel gets in its ``Omega_q`` slot:
    the REAL ``Re Omega`` for a legacy-routed group (which is what makes
    that group a two-point plasmon-pole evaluation and nothing else) and
    the complex ``Omega_p`` for an MPA-routed one.
    """

    name: str
    windows: list
    idx_B: np.ndarray
    field_shape: tuple
    omega_operand: np.ndarray
    n_modes: int
    b_mass: float
    provenance: str

    def dense_mask_B(self):
        """The identical boolean this group used to carry, rebuilt on call.

        Kept because two consumers still want a dense selector and both
        are bounded: the device tau loop, which needs ONE selector
        resident per group and never a second, and the planning tests,
        whose fields are hundreds of modes rather than eighty million.
        It is a method and not an attribute so that materializing 81.4 MB
        is something a caller does on purpose.
        """
        return _dense_from_index(self.idx_B, self.field_shape)


@dataclass
class PassRecord:
    """The ``(pole index, E_ref_B, quadrature provenance)`` triple, per pass.

    Section 7.5's requirement, and the reason it is a requirement: the
    fourteen partial self-energies only add coherently if each pass can say
    which pole it summed and under which rule, so a pass that silently
    refused a window is visible as a gap rather than as a slightly small
    answer.
    """

    pole_index: int
    n_legacy_modes: int = 0
    n_mpa_modes: int = 0
    legacy_b_mass: float = 0.0
    mpa_b_mass: float = 0.0
    n_tau_nodes: int = 0
    groups: list = field(default_factory=list)
    re_omega_min_ev: float = 0.0
    re_omega_max_ev: float = 0.0
    gamma_min_ev: float = 0.0
    gamma_max_ev: float = 0.0


def narrow_pole_threshold_ry(regularization_width_ry, omega_max_ry,
                             edge_factor):
    """``xi`` as the two-point path would actually use it, in Ry.

    NOT the deck's ``sigma_regularization_ev`` on its own: the two-point
    path raises it whenever the requested value would make the HGL crossing
    quadrature ill-conditioned (``ppm_windows.crossing_regularization_floor``
    -- ``A_core = 2*omega_max/xi + 2*edge <= 24``), and the theory guide
    records that this floor engages on EVERY default run, taking 0.25 eV to
    0.476 eV on the production grid.  The threshold this module splits on is
    the width the legacy crossing machinery will really smear by, because
    that is the width a pole has to be narrower than to be indistinguishable
    from a real one.  Reading the deck key instead would put poles between
    0.25 and 0.476 eV on the complex route while the legacy route it is
    being compared against had already smeared them.
    """
    from ..ppm_windows import crossing_regularization_floor

    xi = max(float(regularization_width_ry), 1.0e-12)
    floor = crossing_regularization_floor(float(omega_max_ry),
                                          float(edge_factor))
    return float(max(xi, floor))


def split_pass_by_width(gamma_ry, live_mask, xi_ry):
    """``(narrow, wide)`` boolean masks, split at the smearing width.

    ``narrow`` is ``Gamma < xi`` -- the poles the legacy crossing machinery
    cannot tell from real ones, routed there.  ``wide`` is the rest, routed
    through the certified complex-pole rules.  The boundary is closed on the
    wide side so a pole exactly AT ``xi`` takes the complex route, which is
    the branch that carries its width explicitly; the red twin in
    ``tests/test_mpa_sigma_pass.py`` engineers a pole on each side of the
    boundary and checks it takes the branch it is supposed to.
    """
    g = np.asarray(gamma_ry, dtype=np.float64)
    live = np.asarray(live_mask, dtype=bool)
    xi = float(xi_ry)
    if not (xi > 0.0) or not np.isfinite(xi):
        raise ValueError(
            f"split_pass_by_width: xi={xi_ry!r} is not a finite positive "
            f"smearing width, so there is no threshold to split on.")
    narrow = live & (g < xi)
    return narrow, live & ~narrow


def _host_at_source_shape(a, dtype, to_host):
    """``_to_host_np`` without the process axis it prepends.

    THE DEFECT THIS EXISTS FOR, measured end to end.  On a SINGLE-process
    run ``ppm_windows._to_host_np(x, tiled=False)`` goes through
    ``process_allgather``, which prepends a length-``process_count`` axis:
    a ``(64, 1128, 1128)`` mask comes back ``(1, 64, 1128, 1128)``.  The
    two-point path never noticed because its only scalar consumer,
    ``_to_host_scalar``, ends in ``.reshape(-1)[0]`` and a leading 1 is
    invisible to that.  The pass loop's consumers are not scalars, and the
    first one to touch the array died:

        np.min(a_host, where=live, initial=np.inf)
        ValueError: input operand has more dimensions than allowed by the
        axis remapping

    with ``a_host`` rank 3 and ``live`` rank 4.  So the MPA pass loop could
    not complete a single pole on a real store, which is why this never
    showed up in its own suite -- every cell there hands the planner host
    numpy arrays it built itself and never crosses this seam.

    Reshaping to the SOURCE array's own shape is right in both regimes and
    not only the one that was broken.  Multi-process arrays are not fully
    addressable, so ``_to_host_np`` forces ``tiled=True`` and returns the
    reconstructed GLOBAL array, whose shape already equals the source's;
    the reshape is then the identity and asserts as much.  A genuine
    element-count mismatch raises here rather than silently broadcasting
    somewhere downstream.
    """
    out = np.asarray(to_host(a, dtype=dtype, tiled=False), dtype=dtype)
    want = tuple(int(v) for v in np.shape(a))
    if out.shape == want:
        return out
    n_want = int(np.prod(want)) if want else 1
    if out.size != n_want:
        raise ValueError(
            f"mpa sigma: gathering an array of shape {want} returned "
            f"{out.shape}, which is not a reshape of it ({out.size} "
            f"elements against {n_want}).  That is not the "
            f"process axis this strips; it is a different object.")
    return out.reshape(want)


def refuse_wedge_pole_slab(n_q_store, n_k_tot):
    """Refuse a wedge-shaped pole slab by name, and say what would fix it.

    The Sigma kernel's k-q sums index the FULL Bloch zone.  ``W`` reaches
    that shape through ``symmetry_maps.unfold_isdf_operator`` -- centroid
    double permute, L-phase, and a conjugation on the time-reversed star
    members.  A pole field is two objects under that map and only one of
    them is ``W``: the residues ``B_p`` carry the phase and the
    conjugation, while the pole POSITIONS carry only the permutation, and
    the conjugation is the half nobody has certified.  Time reversal sends
    ``W_c(q, z)`` to a relation between ``W_c(-q, z)`` and a conjugated
    argument, and a fourth-quadrant pole ``a - i*Gamma`` conjugated
    naively becomes ``a + i*Gamma``, which enters the tau stage as
    ``e^{+Gamma*tau}`` and grows -- the failure ``write_head_axis`` already
    refuses by name at the other end of the pipeline.

    So this is refused rather than guessed.  The gate that would retire the
    refusal is cheap and is named here so it is not re-derived: rebuild
    ``W_c(q, z_j)`` from the UNFOLDED poles at the store's own ``2*n_p``
    sample points and compare against the same store's ``W`` unfolded by
    ``unfold_isdf_operator``.  Agreement at the fit's own backward error
    certifies both halves of the pole unfold at once; disagreement on the
    time-reversed members alone localizes it to the conjugation.
    """
    if int(n_q_store) == int(n_k_tot):
        return
    raise NotImplementedError(
        f"MPA Sigma: the fit store's pole axis has n_q={int(n_q_store)} but "
        f"the Sigma kernel sums over the full Bloch zone, n_k_tot="
        f"{int(n_k_tot)}.  This store is on the symmetry wedge, and "
        f"unfolding a POLE FIELD is not the operation that unfolds W: the "
        f"residues B_p carry the centroid permutation, the L-phase and a "
        f"conjugation on the time-reversed star members, while the pole "
        f"positions carry only the permutation -- and a fourth-quadrant "
        f"pole conjugated the wrong way becomes exp(+Gamma*tau), which "
        f"grows.  That conjugation is not certified, so it is refused here "
        f"rather than guessed into a finite, plausible Sigma.  The gate "
        f"that retires this: rebuild W_c(q, z_j) from the unfolded poles at "
        f"the store's own 2*n_p sample points and compare against the same "
        f"store's W unfolded by symmetry_maps.unfold_isdf_operator.  Until "
        f"then, fit a full-BZ store (allocate_fit_store with n_q = "
        f"{int(n_k_tot)}).")


def _laplace_buckets(a_values, *, e_lo, e_hi, omega_max, r_max):
    """Geometric buckets in ``Re Omega`` with bounded Laplace ratio.

    ``a_values`` is the ``Re Omega`` of the modes being bucketed -- the
    gathered values of ONE selection, not the whole field beside a mask.
    Only their min and max enter, which is why the index representation
    costs this function nothing.

    Each bucket's sign-definite window will span ``x_min ~ e_lo + a_lo -
    omega_max`` to ``x_max ~ e_hi + a_hi + omega_max``; the bucket edge is
    the largest ``a_hi`` keeping that ratio at or under ``r_max``.  The
    recursion ``a_hi = r_max*(e_lo + a_lo) - e_hi`` grows geometrically, so
    a field spanning four decades of pole position needs a handful of
    buckets rather than one per octave.

    Returns a list of ``(a_lo, a_hi)`` closed intervals covering the given
    values, or ``[]`` when there are none.
    """
    a = np.asarray(a_values, dtype=np.float64)
    if a.size == 0:
        return []
    a_min = float(np.min(a))
    a_max = float(np.max(a))
    r_max = float(r_max)
    out = []
    lo = a_min
    for _ in range(64):                     # a hard stop, not a schedule
        # The Laplace-ratio edge.  It can come out BELOW ``lo`` -- that
        # happens exactly when the crossing pane has already pushed the
        # window's lower edge onto its ``z_edge`` floor, where ``x_min`` no
        # longer tracks ``a`` and the ratio this line is solving for is not
        # the one the rule will pay.  Stepping geometrically there is what
        # keeps the recursion advancing; the rule builder's own node cap is
        # the backstop, and it refuses by name rather than building
        # something enormous.
        x_min = max(e_lo + lo - omega_max, _X_FLOOR_RY)
        cap = r_max * x_min - e_hi - omega_max
        hi = max(cap, lo * r_max, lo + _X_FLOOR_RY)
        hi = min(hi, a_max)
        out.append((lo, hi))
        if hi >= a_max:
            return out
        lo = np.nextafter(hi, np.inf)
    return out                              # pragma: no cover - 128 decades


#: Ceiling on the number of leaves one bucket's width split may return.
#: A MEMORY guard, not a cost opinion, and its size is set by a
#: measurement: on the real n_p = 1 field the ALIGNED predicate returns
#: 208 leaves for one val-branch bucket, and that number is not a
#: defect -- it is what the slab width clause DEMANDS.  For a pole with
#: Gamma ~ a (the fitter's fourth guard allows Gamma up to a exactly),
#: beta_p = Gamma/(e_lo + a) < 1 per pole always, but a PANE of such
#: poles spanning ratio r has beta ~ r, so clause-satisfying panes need
#: r -> 1 and a continuum of near-45-degree widths partitions into
#: O(100) panes.
#:
#: THE CEILING WAS RE-DERIVED WHEN THE PANES STOPPED BEING MASKS, and
#: the old number was refusing real physics.  Each leaf USED TO BE a
#: full-size (n_q, N_mu, N_mu) bool mask (81.4 MB on the production
#: deck), so 512 of them was ~42 GB and the ceiling was a MEMORY guard
#: sized against a 230 GB node.  A leaf now costs: its slice of the
#: pass's ONE index array (4 bytes per live mode, and the leaves
#: partition the live set, so that total does not grow with the leaf
#: count at all), plus its plan -- one mask_A boolean per window over
#: the A-side shape (nk x nb = 6.4 kB on the production deck), the
#: window's nodes at 32 bytes each, and the group object.  That is
#: ~8-10 kB per pane at production shapes, so 512 panes cost ~5 MB
#: where they used to cost 42 GB, and memory has stopped being the
#: reason for any number here.
#:
#: What the ceiling bounds NOW is the recursion, and through it the
#: cost: a field that DEFEATS the termination argument bisects to the
#: depth-16 cap, and each of those 2^16 leaves is a certified rule
#: BUILT (a table lookup or a minimax solve) and a tau-node run at a
#: measured 0.165 s per node that does not fall when the leaf is small.
#: 8192 = 6.2x the largest demand a real field has produced (pole 7 of
#: the si_mpa_0808 n_p = 8 fit: 1312 panes on one Laplace bucket
#: spanning Gamma = [1.84e-2, 4.49e+2] Ry over 81 360 063 modes, which
#: the 512 ceiling refused and which is not a pathology but the tail
#: term's real width spread), 8x below the 2^16 the recursion can
#: structurally reach, and ~74 MB of plan at production shapes.
#:
#: OWNER ROW, registered here because this constant is where the cost
#: shows up: whether the slab width clause should stay the per-pole
#: envelope (beta <= 1, forcing the O(100)-pane partition and its
#: dispatch cost) or become slab-aware (beta <= r for a width-binned
#: pane, derivation as above) is the routing design's call, not this
#: guard's.
MAX_WIDTH_SPLIT_LEAVES = 8192


def _refuse_width_split_explosion(n_leaves, gamma_values):
    """Refuse a width split that is diverging, with the field statistics.

    ``gamma_values`` is the bucket's OWN widths -- the gathered values of
    the modes being split, which is what the index representation makes
    cheap to hand around; the count in the message is their number.

    The FALSE case this guards was measured, not imagined: the first
    end-to-end MPA Sigma dispatch put the split's termination clause on
    an unreachable floor and the recursion bisected toward its depth-16
    cap -- 2^16 leaves x 81.4 MB of mask each, a 230 GB node dead at
    ~2 800 of them.  The predicate fix at the call site makes that
    geometry terminate; this guard is for the field NOBODY has met yet,
    and it converts "the node died six minutes in" into a one-line
    refusal naming what to look at.

    WHAT THE GUARD NOW BOUNDS.  With panes carried as index sets a
    diverging split no longer runs a node out of memory -- 2^16 leaves
    of an 81-million-mode field still cost one index per mode.  It runs
    it out of TIME instead: each leaf is a certified rule of its own and
    a tau-node run of its own, at a measured 0.165 s per node that does
    not fall when the leaf is small.  Same ceiling, same value, a cost
    that changed its units.
    """
    if int(n_leaves) <= MAX_WIDTH_SPLIT_LEAVES:
        return
    g = np.asarray(gamma_values, dtype=np.float64)
    g_lo = float(np.min(g)) if g.size else np.inf
    g_hi = float(np.max(g)) if g.size else -np.inf
    n_modes = int(g.size)
    raise RuntimeError(
        f"MPA Sigma window planning: the width split of one Laplace "
        f"bucket returned {int(n_leaves)} leaves against a ceiling of "
        f"{MAX_WIDTH_SPLIT_LEAVES}.  Each leaf is a certified rule and a "
        f"tau-node run of its own, at a measured 0.165 s per node that "
        f"does not fall when the leaf is small -- and when the leaves "
        f"were dense masks instead of index sets this same divergence "
        f"was an out-of-memory kill wearing a planning stage's clothes "
        f"(the 2026-08-09 profile measured ~2 800 live masks at a 230 GB "
        f"node's death).  This bucket spans Gamma = [{g_lo:.3e}, "
        f"{g_hi:.3e}] Ry over {n_modes} modes; a split "
        f"this deep means the width clause cannot be satisfied by "
        f"splitting at all (the Laplace edge is pinned far below the "
        f"widths), which is a property of the pole field worth looking "
        f"at, not one worth 2^16 panes.")


#: Ceiling on ``Gamma_hi/Gamma_lo`` within one CROSSING pane.  The
#: crossing core's bandwidth is ``A = f_max/Gamma_lo`` with ``f_max``
#: dominated by the pane ``T = omega_max + edge_factor*Gamma_hi``, so a
#: pane mixing decades of width pays ``A ~ edge*Gamma_hi/Gamma_lo`` --
#: the first fixed-tree verification leg measured A = 2485 needing
#: 12 771 nodes against the 4 096 cap on exactly that mixture.  Binning
#: the widths geometrically at this ratio bounds
#: ``A <~ (2*omega_max + spread)/Gamma_lo + 2*edge*r`` at the two-point
#: order the module docstring promises (``A_core = 2T/xi``): at the
#: production xi and the +/-2 eV window that is ~50, i.e. a few hundred
#: nodes.  4 is two octaves -- small enough to bound A, large enough
#: that a field spanning four decades of width is ~7 panes, not 40.
CROSSING_WIDTH_RATIO_MAX = 4.0


def _geometric_width_bins_sorted(g_sorted, idx_sorted, *,
                                 r_max=CROSSING_WIDTH_RATIO_MAX):
    """:func:`_geometric_width_bins` on a pane already sorted by width.

    Same partition, same order, no full-field temporaries: with
    ``g_sorted`` ascending, ``np.digitize``'s bins are contiguous runs
    and ``searchsorted`` on the same edges names their boundaries.  The
    bins come back as slices of ``idx_sorted``, so the whole partition
    costs one index array, not one full-size boolean per bin.
    """
    if idx_sorted.size == 0:
        return []
    g_lo = float(g_sorted[0])
    g_hi = float(g_sorted[-1])
    if not (g_lo > 0.0) or g_hi <= g_lo * float(r_max):
        return [idx_sorted]
    n_bins = int(np.ceil(np.log(g_hi / g_lo) / np.log(float(r_max))))
    edges = g_lo * float(r_max) ** np.arange(1, n_bins)
    # digitize's bin b is [edges[b-1], edges[b]) -- left-closed -- which
    # is exactly what side='left' names on an ascending array.
    cuts = ([0] + [int(c) for c in np.searchsorted(g_sorted, edges, side="left")]
            + [int(idx_sorted.size)])
    out = []
    for b in range(n_bins):
        lo, hi = cuts[b], cuts[b + 1]
        if hi > lo:
            out.append(idx_sorted[lo:hi])
    return out


def _geometric_width_bins(gamma_ry, mask, *, r_max=CROSSING_WIDTH_RATIO_MAX):
    """Partition ``mask`` into width panes of bounded ratio -- directly.

    ``np.digitize`` on ``log Gamma``, NO recursion: the bin count is
    ``ceil(log_r(Gamma_hi/Gamma_lo))`` by construction and the returned
    masks partition the input, so this cannot diverge the way the
    clause-driven recursion did -- there is no predicate to satisfy,
    only a ratio to respect.  Single-width sets (and empty ones) come
    back whole.

    THE BOOLEAN FACE OF :func:`_geometric_width_bins_sorted`, kept so the
    partition property is testable on a mask the way it was written; the
    planner calls the index-set core directly and never builds one of
    these.
    """
    g = np.asarray(gamma_ry, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    idx, g_v = _sorted_by_width(np.ravel(g), np.flatnonzero(np.ravel(m)))
    return [_dense_from_index(p, m.shape)
            for p in _geometric_width_bins_sorted(g_v, idx, r_max=r_max)]


def _clause_safe_width_split_sorted(a_sorted, g_sorted, idx_sorted, *,
                                    e_lo, omega_max, beta_max, depth=0):
    """:func:`_clause_safe_width_split` on a pane already sorted by width.

    THE SAME RECURSION, THE SAME PREDICATE, THE SAME LEAVES -- and none
    of the full-size booleans.  The split is a threshold in ``Gamma``, so
    on a width-sorted pane it is a CUT POINT: ``searchsorted`` names it,
    the two halves are slices, and a leaf is a slice of one index array
    rather than an 81.4 MB boolean of its own.  That is what turns the
    planner's 218 panes from 17.8 GB into one index per live mode.

    The predicate is character-for-character the one the boolean version
    paid, including its two backstops (the depth cap and the equal-width
    stop), because the leaves it names are the certified panes and this
    change is a representation change.
    """
    if idx_sorted.size == 0:
        return [idx_sorted]
    a_lo = float(np.min(a_sorted))
    g_lo = float(g_sorted[0])
    g_hi = float(g_sorted[-1])
    x_min = max(e_lo + a_lo - omega_max, _X_FLOOR_RY)
    if g_hi <= float(beta_max) * x_min or depth >= 16 or g_hi <= g_lo:
        return [idx_sorted]
    cut = float(np.sqrt(g_lo * g_hi))
    k = int(np.searchsorted(g_sorted, cut, side="right"))   # g <= cut
    if k == 0 or k == idx_sorted.size:
        return [idx_sorted]                 # one width; nothing to separate
    return (_clause_safe_width_split_sorted(
                a_sorted[:k], g_sorted[:k], idx_sorted[:k], e_lo=e_lo,
                omega_max=omega_max, beta_max=beta_max, depth=depth + 1)
            + _clause_safe_width_split_sorted(
                a_sorted[k:], g_sorted[k:], idx_sorted[k:], e_lo=e_lo,
                omega_max=omega_max, beta_max=beta_max, depth=depth + 1))


def _clause_safe_width_split(a_ry, gamma_ry, mask, *, e_lo, omega_max,
                             beta_max, depth=0):
    """Split a mode set by width until every part's width clause is inside.

    ``beta = Gamma_max / x_min`` is a SLAB quantity and can exceed the
    envelope even though it cannot for any single pole: the fitter's fourth
    guard gives ``Gamma_p <= Re Omega_p`` element by element, so
    ``beta_p = Gamma_p/(min E_A + a_p) < 1`` always -- but a slab pairs the
    widest ``Gamma`` in the set with the shallowest ``a`` in the set, and
    those need not belong to the same pole.

    The cure is therefore not to widen the clause and not to bucket harder
    in ``a``: it is to separate the widths, because ``Gamma <= a`` means a
    wide-``Gamma`` sub-bucket automatically has a deep ``a`` and its own
    ``x_min`` rises with it.  Splitting at the geometric mean of the width
    range is the octave split the theory's width buckets describe, and the
    exact ``Gamma_p`` is never rounded by it -- the buckets select rules,
    nothing else.

    THE PREDICATE MUST BE THE BUILDER'S OWN x_min, and ``omega_max`` is
    the caller's way of saying which builder.  The termination test
    below compares ``Gamma_hi`` against ``beta_max * x_min`` with
    ``x_min = max(e_lo + a_lo - omega_max, floor)``; hand it an
    ``omega_max`` the window builder will not actually subtract and the
    predicate lands on the 1e-12 floor whenever ``e_lo + a_lo <
    omega_max``, where NO amount of splitting can satisfy it and the
    recursion runs to its depth cap accumulating one full-size mask per
    leaf.  That is not a hypothetical: it OOM-killed eleven Sigma
    attempts before the call site learned to pass the non-crossing
    builder's own ``omega_max = 0`` and to skip the split entirely on
    crossing branches, whose windows floor x_min at z_edge and cannot
    trip the clause.  The termination ARGUMENT (Gamma <= a) only holds
    with the aligned predicate; the depth cap and the leaf ceiling at
    the call site are the backstops for a field that defeats it.

    THE BOOLEAN FACE OF :func:`_clause_safe_width_split_sorted`.  The
    planner works in index sets and calls the core; this signature is
    kept because it is the one the OOM fixture's cells were written
    against, and those cells are the executable memory of a 230 GB node's
    death -- they must keep testing the real predicate, not a paraphrase
    of it.
    """
    a = np.ravel(np.asarray(a_ry, dtype=np.float64))
    g = np.ravel(np.asarray(gamma_ry, dtype=np.float64))
    m = np.asarray(mask, dtype=bool)
    idx, g_v = _sorted_by_width(g, np.flatnonzero(np.ravel(m)))
    parts = _clause_safe_width_split_sorted(
        a[idx], g_v, idx, e_lo=e_lo, omega_max=omega_max,
        beta_max=beta_max, depth=depth)
    return [_dense_from_index(p, m.shape) for p in parts]


def _stats(arr, mask):
    """``(min, max)`` of ``arr`` over ``mask``; the caller checks emptiness."""
    a = np.asarray(arr, dtype=np.float64)
    return (float(np.min(a, where=mask, initial=np.inf)),
            float(np.max(a, where=mask, initial=-np.inf)))


# ---------------------------------------------------------------------------
#  The compact pane index -- the representation this module's planner runs
#  on, and the reason it no longer needs 17.8 GB to describe 218 panes.
# ---------------------------------------------------------------------------

def flat_index_dtype(n_flat):
    """The narrowest integer that can address ``n_flat`` modes.

    A pane index costs 4 bytes per live mode on any field with fewer than
    2^31 modes, which the production deck (64 x 1128 x 1128 = 81 432 576)
    is by a factor of 26.  Larger fields fall back to 8 bytes rather than
    wrapping, because a silently wrapped index is a pane that quietly
    contains the wrong modes -- the one failure class this whole module
    refuses to leave to chance.
    """
    return np.int32 if int(n_flat) < (1 << 31) else np.int64


def _sorted_by_width(g_flat, idx):
    """``(idx, gamma)`` for one selection, sorted ascending in ``Gamma``.

    Every splitter in this planner cuts on ``Gamma`` -- the clause-safe
    recursion at a geometric mean, the crossing panes at fixed ratio
    edges -- so sorting once turns every later split into a slice.  The
    sort is stable, so equal widths keep their field order and the
    partition is reproducible run to run.
    """
    ix = np.asarray(idx, dtype=np.int64)
    g_v = np.asarray(g_flat, dtype=np.float64)[ix]
    order = np.argsort(g_v, kind="stable")
    return ix[order], g_v[order]


def _dense_from_index(idx, shape):
    """The boolean an index set stands for -- built on purpose, one at a time."""
    shape = tuple(int(x) for x in shape)
    m = np.zeros(int(np.prod(shape)) if shape else 0, dtype=bool)
    m[np.asarray(idx, dtype=np.int64)] = True
    return m.reshape(shape)


def _sigma_window(*, name, plan_t, plan_alpha, mask_A, e_ref_a, e_ref_b,
                  omega_sign, project, prefactor, max_error, provenance):
    """A ``ppm_windows._SigmaWindow`` carrying an MPA rule.

    The two paths share this dataclass on purpose: it is what the device
    tau loop, the accumulator and the omega projector all read, so an MPA
    window that is a ``_SigmaWindow`` reaches the existing integration with
    no branch anywhere below this call.
    """
    from ..minimax_screening import MinimaxNodes
    from ..ppm_windows import _SigmaWindow

    return _SigmaWindow(
        name=name,
        nodes=MinimaxNodes(t=np.asarray(plan_t, dtype=np.complex128),
                           alpha=np.asarray(plan_alpha, dtype=np.complex128)),
        mask_A=np.asarray(mask_A, dtype=bool),
        E_ref_A=float(e_ref_a),
        E_ref_B=float(e_ref_b),
        omega_sign=int(omega_sign),
        project=str(project),
        prefactor=float(prefactor),
        mask_B_mode="all",
        max_error=float(max_error),
        provenance=str(provenance),
    )


def _mpa_groups_for_bucket(
    *, idx, a_v, g_v, E_A_host, base_mask_A_host,
    omega_max, space, neg_omega_half, edge_factor, rel_tol, max_nodes,
):
    """The up-to-three MPA windows serving one ``Re Omega`` bucket.

    ``idx`` is the pane's flat index set and ``a_v``/``g_v`` its gathered
    ``(Re Omega, Gamma)``; every rule below is built from the SET's
    extreme values exactly as before, and the classifier that separates
    the crossing core from the deep slab is the same ``a <= T`` predicate
    -- evaluated on ``a_v``, whose length is the pane's, rather than on
    the whole field beside a mask.
    """
    from . import sigma_routing as R

    neg = -1.0 if neg_omega_half else 1.0
    e_lo, e_hi = _stats(E_A_host, base_mask_A_host)
    groups = []

    if not R.denominator_can_cross(space, bool(neg_omega_half)):
        a_lo, a_hi = float(np.min(a_v)), float(np.max(a_v))
        g_hi = float(np.max(g_v))
        x_min = max(e_lo + a_lo, _X_FLOOR_RY)
        x_max = max(e_hi + a_hi + omega_max, x_min * (1.0 + 1.0e-9))
        t, alpha, rule = R.sign_definite_rule(
            x_min, x_max, g_hi, rel_tol=rel_tol, max_nodes=max_nodes)
        beta = R.beta_for_window(g_hi, x_min)
        _refuse_width_clause(beta, "single", x_min, g_hi)
        win = _sigma_window(
            name="single", plan_t=t, plan_alpha=alpha,
            mask_A=base_mask_A_host, e_ref_a=e_lo, e_ref_b=a_lo,
            omega_sign=-1, project="full", prefactor=-1.0 * neg,
            max_error=rule["rel_tol"],
            provenance=(f"MPA sign-definite composite, {rule['n_panels']} "
                        f"panels, R={x_max / x_min:.3g}, beta={beta:.4f}, "
                        f"kappa0={rule['kappa0']:.6f}"))
        groups.append((("single",), [win], idx))
        return groups

    R.refuse_edge_factor_below_envelope(edge_factor)
    # The pane is sized by the WIDEST width in the bucket, so a pole this
    # classifier sends to the slab satisfies a > omega_max + edge*Gamma_p
    # for its OWN width and genuinely cannot cross.  Over-including in the
    # core is safe (the crossing rule is exact at any offset) and only
    # costs nodes; under-including would put a crossing pole under a
    # sign-definite rule, which is a wrong number rather than a slow one.
    g_hi_all = float(np.max(g_v))
    z_edge = float(edge_factor) * g_hi_all
    T = omega_max + z_edge

    in_core = np.asarray(a_v) <= T
    cross_idx, cross_a, cross_g = idx[in_core], a_v[in_core], g_v[in_core]
    slab_idx, slab_a, slab_g = idx[~in_core], a_v[~in_core], g_v[~in_core]

    if cross_idx.size:
        a_lo, a_hi = float(np.min(cross_a)), float(np.max(cross_a))
        g_lo, g_hi = float(np.min(cross_g)), float(np.max(cross_g))
        core_A = base_mask_A_host & (np.asarray(E_A_host) <= T)
        wins = []
        if core_A.any():
            core_e_lo, _ = _stats(E_A_host, core_A)
            # The beat frequency, with the slab's own Re Omega spread in
            # it -- the one term a scalar router does not have, because a
            # scalar router can put E_ref_B on the pole exactly.
            f_max = (omega_max + max(T - core_e_lo, 0.0) + (a_hi - a_lo))
            t, alpha, rule = R.crossing_rule(
                g_lo, f_max, rel_tol=rel_tol, max_nodes=max_nodes)
            wins.append(_sigma_window(
                name="core", plan_t=t, plan_alpha=alpha, mask_A=core_A,
                e_ref_a=core_e_lo, e_ref_b=a_lo, omega_sign=+1,
                # project="full": the crossing consumer here forms the whole
                # complex product coeff*X and never weights Re and Im by two
                # independent real vectors, which is what licenses the merged
                # one-complex-chain kernel.  A sine-only projection would keep
                # the absorptive part and drop the dispersive one.
                project="full", prefactor=-1.0 * neg,
                max_error=rule["rel_tol"],
                provenance=(f"MPA crossing composite on the exact complex "
                            f"resolvent, {rule['n_panels']} panels, "
                            f"A={rule['a_dim']:.4g}, "
                            f"kappa0={rule['kappa0']:.6f}")))
        stripe_A = base_mask_A_host & (np.asarray(E_A_host) > T)
        if stripe_A.any():
            s_lo, s_hi = _stats(E_A_host, stripe_A)
            x_min = max(s_lo + a_lo - omega_max, z_edge, _X_FLOOR_RY)
            x_max = max(s_hi + a_hi, x_min * (1.0 + 1.0e-9))
            t, alpha, rule = R.sign_definite_rule(
                x_min, x_max, g_hi, rel_tol=rel_tol, max_nodes=max_nodes)
            beta = R.beta_for_window(g_hi, x_min)
            _refuse_width_clause(beta, "a_stripe", x_min, g_hi)
            wins.append(_sigma_window(
                name="a_stripe", plan_t=t, plan_alpha=alpha,
                mask_A=stripe_A, e_ref_a=s_lo, e_ref_b=a_lo,
                omega_sign=+1, project="full", prefactor=+1.0 * neg,
                max_error=rule["rel_tol"],
                provenance=(f"MPA sign-definite composite, "
                            f"{rule['n_panels']} panels, "
                            f"R={x_max / x_min:.3g}, beta={beta:.4f}, "
                            f"kappa0={rule['kappa0']:.6f}")))
        if wins:
            groups.append((tuple(w.name for w in wins), wins, cross_idx))

    if slab_idx.size:
        a_lo, a_hi = float(np.min(slab_a)), float(np.max(slab_a))
        g_hi = float(np.max(slab_g))
        x_min = max(e_lo + a_lo - omega_max, z_edge, _X_FLOOR_RY)
        x_max = max(e_hi + a_hi, x_min * (1.0 + 1.0e-9))
        t, alpha, rule = R.sign_definite_rule(
            x_min, x_max, g_hi, rel_tol=rel_tol, max_nodes=max_nodes)
        beta = R.beta_for_window(g_hi, x_min)
        _refuse_width_clause(beta, "b_slab", x_min, g_hi)
        groups.append((("b_slab",), [_sigma_window(
            name="b_slab", plan_t=t, plan_alpha=alpha,
            mask_A=base_mask_A_host, e_ref_a=e_lo, e_ref_b=a_lo,
            omega_sign=+1, project="full", prefactor=+1.0 * neg,
            max_error=rule["rel_tol"],
            provenance=(f"MPA sign-definite composite, {rule['n_panels']} "
                        f"panels, R={x_max / x_min:.3g}, beta={beta:.4f}, "
                        f"kappa0={rule['kappa0']:.6f}"))], slab_idx))
    return groups


def _refuse_width_clause(beta, where, x_min, gamma):
    """The width clause, checked where the window is built rather than later.

    The rule this module builds is the positive composite one, certified by
    its own ``kappa0 <= 1`` and not by a catalog entry -- so ``beta`` does
    not gate the build.  It is checked anyway, because the audited envelope
    (``beta <= 1`` for any field this fitter can produce, the fitter's own
    fourth guard) is the statement that the slab's conservative statistics
    have not walked outside the pole field the routing was scored on.  A
    slab whose ``beta`` exceeds it is a bucketing failure, not a physics
    one, and it should say so here rather than 4000 tau nodes later.
    """
    from . import sigma_routing as R

    if float(beta) <= R.SHIPPED_WIDTH_BETA_MAX + 1.0e-12:
        return
    raise R.RoutingRefusal(
        f"MPA {where} window: the slab's width clause is beta = "
        f"Gamma_max/x_min = {float(beta):.6f}, above the envelope "
        f"beta_max={R.SHIPPED_WIDTH_BETA_MAX} that the fitter's own fourth "
        f"guard closes at (x_min={float(x_min):.6g} Ry, "
        f"Gamma={float(gamma):.6g} Ry).  For a SINGLE pole this is "
        f"structurally impossible; for a slab it means the bucket mixes a "
        f"shallow Laplace edge with a wide pole.  Narrow the buckets "
        f"(lower mpa_laplace_ratio_max) rather than widening the clause.",
        code="slab_width_clause", beta=float(beta))


def plan_branch_groups(
    *,
    a_ry,
    gamma_ry,
    live_mask,
    E_A_host,
    base_mask_A_host,
    omega_nonneg_ry,
    space,
    neg_omega_half,
    xi_ry,
    edge_factor,
    b_abs=None,
    rel_tol=1.0e-8,
    max_nodes=None,
    laplace_ratio_max=DEFAULT_LAPLACE_RATIO_MAX,
    target_error=1.0e-6,
    laplace_max_nodes=64,
    crossing_eps_q=1.0e-10,
    crossing_max_nodes=200,
    use_shipped_minimax_tables=True,
    log_tag="",
    print_fn=print,
):
    """Every window group one pass owes on one branch.

    Returns ``(groups, stats)``.  ``groups`` is a list of
    :class:`WindowGroup`; ``stats`` records the narrow/wide split so the
    pass announcement can name it.

    The legacy group comes first and is built by the two-point path's own
    ``_build_windows_for_branch``, called with the REAL parts of the narrow
    poles as its ``Omega_q``.  That is not an imitation of the two-point
    treatment; it IS the two-point treatment, applied to the poles the
    smearing cannot distinguish from real ones.
    """
    from ..ppm_windows import _build_windows_for_branch
    import jax.numpy as jnp

    from . import sigma_routing as R

    a = np.asarray(a_ry, dtype=np.float64)
    g = np.asarray(gamma_ry, dtype=np.float64)
    field_shape = tuple(int(x) for x in np.shape(a))
    n_flat = int(np.prod(field_shape)) if field_shape else 0
    ix_dtype = flat_index_dtype(n_flat)
    a_flat = np.ravel(a)
    g_flat = np.ravel(g)
    omega_max = (float(np.max(np.asarray(omega_nonneg_ry, dtype=np.float64)))
                 if np.size(omega_nonneg_ry) else 0.0)
    max_nodes = (R.DEFAULT_MAX_CROSSING_NODES if max_nodes is None
                 else int(max_nodes))
    narrow, wide = split_pass_by_width(g, live_mask, xi_ry)
    mass = (np.abs(np.asarray(b_abs, dtype=np.float64))
            if b_abs is not None else None)
    mass_flat = None if mass is None else np.ravel(mass)

    def _mass(mask):
        return 0.0 if mass is None else float(np.sum(mass, where=mask))

    def _mass_idx(idx):
        return 0.0 if mass_flat is None else float(np.sum(mass_flat[idx]))

    groups = []
    if narrow.any():
        # THE A-SIDE OPERANDS GO DOWN AS HOST NUMPY, NOT BACK ONTO A DEVICE.
        # They were gathered at their source shape a few frames up
        # (``_host_at_source_shape``); ``jnp.asarray`` here used to put them
        # back on THIS PROCESS's device, where they are fully addressable at
        # any process count, and the window builder's own
        # ``process_allgather(tiled=False)`` then prepended an axis of length
        # ``jax.process_count()``.  At one process that axis is 1 and
        # ``build_G_tau``'s reshape absorbed it; at four it is 4 and every
        # rank died on the first tau dispatch (C1, 2026-08-10).  The B-side
        # operands stay on device: their only consumers are the scalar
        # ``_masked_stats_device`` reductions, which end in ``.reshape(-1)[0]``
        # and to which a leading axis of replicated values is invisible.
        wins = _build_windows_for_branch(
            omega_nonneg_ry=np.asarray(omega_nonneg_ry, dtype=np.float64),
            E_A=np.asarray(E_A_host, dtype=np.float64),
            base_mask_A=np.asarray(base_mask_A_host, dtype=bool),
            Omega_q=jnp.asarray(a, dtype=jnp.float64),
            base_mask_B=jnp.asarray(narrow, dtype=bool),
            space=space, neg_omega_half=bool(neg_omega_half),
            regularization_width_ry=float(xi_ry),
            edge_factor=float(edge_factor),
            target_error=float(target_error),
            max_nodes=int(laplace_max_nodes),
            crossing_eps_q=float(crossing_eps_q),
            crossing_max_nodes=int(crossing_max_nodes),
            use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
            log_tag=f"{log_tag} legacy-routed", print_fn=print_fn)
        if wins:
            groups.append(WindowGroup(
                name="legacy", windows=wins,
                idx_B=np.flatnonzero(np.ravel(narrow)).astype(ix_dtype),
                field_shape=field_shape,
                omega_operand=a, n_modes=int(np.count_nonzero(narrow)),
                b_mass=_mass(narrow),
                provenance=("two-point crossing machinery at "
                            f"xi={float(xi_ry) * RYD_TO_EV:.4f} eV; these "
                            "poles are narrower than the smearing and are "
                            "not distinguishable from real ones")))

    omega_complex = a - 1j * g
    # THE ONE GATHER THIS PLANNER PERFORMS.  Everything after it works on
    # the wide set's own values and its own index array: the bucket edges,
    # the width panes, the rule statistics and the group membership.  The
    # dense field is never compared against a mask again, and no pane ever
    # allocates a boolean of its own -- which is the 17.8 GB the 2026-08-09
    # memory row measured, and the reason a pass held ~24 GB to describe a
    # 4.55 GB pole slab.
    wide_idx = np.flatnonzero(np.ravel(wide)).astype(ix_dtype)
    wide_a = a_flat[wide_idx]
    wide_g = g_flat[wide_idx]
    for a_lo, a_hi in _laplace_buckets(
            wide_a, e_lo=_stats(E_A_host, base_mask_A_host)[0],
            e_hi=_stats(E_A_host, base_mask_A_host)[1],
            omega_max=omega_max, r_max=float(laplace_ratio_max)):
        in_bucket = (wide_a >= a_lo) & (wide_a <= a_hi)
        if not in_bucket.any():
            continue
        bucket_idx = wide_idx[in_bucket]
        bucket_g = wide_g[in_bucket]
        e_lo_A = _stats(E_A_host, base_mask_A_host)[0]
        # THE SPLIT RUNS ONLY WHERE A WINDOW WILL PAY ITS CLAUSE, at the
        # x_min THAT WINDOW WILL PAY.  This loop OOM-killed a 230 GB
        # Perlmutter node eleven times before either condition was
        # checked (measured 2026-08-09, memwatch profile at
        # /pscratch/sd/j/jackm/mpa_oom_0809/): the predicate below used
        # x_min = e_lo + a_lo - omega_max, which on the Si production
        # deck is NEGATIVE (0.0255 + ~0.02 - 0.147 Ry), so x_min sat on
        # the 1e-12 floor, beta = Gamma/1e-12 exceeded every clause, and
        # the recursion bisected to its depth-16 cap -- 2^16 leaf masks
        # of (n_q, N_mu, N_mu) bool = 81.4 MB each, 5.3 TB demanded, the
        # node dead at ~230 GB with ~2 800 leaves live, invariant under
        # n_p, layout, batch and the Sigma window because none of those
        # touch the mask shape or the floored predicate.
        #
        # The repair is alignment, not a new heuristic, and it has two
        # halves.  On a branch that CAN cross, every Laplace window the
        # bucket builds floors its x_min at z_edge = edge_factor *
        # Gamma_hi (see _mpa_groups_for_bucket), so beta <= 1/edge_factor
        # STRUCTURALLY -- refuse_edge_factor_below_envelope already
        # guards edge_factor >= 1/beta_max at the build -- and a width
        # split cannot tighten a clause that cannot fire: the bucket
        # passes through whole.  On a branch that CANNOT cross, the
        # single window's own x_min is e_lo + a_lo with NO omega_max
        # subtraction (the denominators are sign-definite; omega only
        # widens x_max), so the predicate uses that same number --
        # omega_max=0.0 below is that alignment, and with it the floor
        # is unreachable and the recursion terminates the way its
        # docstring argues (Gamma <= a per pole, so a wide-width
        # sub-bucket has a deep Laplace edge of its own).
        if R.denominator_can_cross(space, bool(neg_omega_half)):
            # ...but the crossing CORE has a sizing concern of its own,
            # and the first fixed-tree verification leg met it: the pane
            # is set by the bucket's WIDEST width (T = omega_max +
            # edge*Gamma_hi) while the rule's bandwidth divides by its
            # NARROWEST (A = f_max/Gamma_lo), so a bucket mixing decades
            # of width pays A ~ edge*Gamma_hi/Gamma_lo -- measured
            # A = 2485, 12 771 nodes demanded against the 4 096 cap.
            # Geometric width panes bound the ratio, and with it A, at
            # the two-point order; the binning is direct (digitize on
            # log Gamma), so it cannot diverge -- there is no clause to
            # chase, only a ratio to respect.
            b_idx, b_g = _sorted_by_width(g_flat, bucket_idx)
            subs = _geometric_width_bins_sorted(b_g, b_idx)
        else:
            b_idx, b_g = _sorted_by_width(g_flat, bucket_idx)
            subs = _clause_safe_width_split_sorted(
                a_flat[b_idx], b_g, b_idx, e_lo=e_lo_A, omega_max=0.0,
                beta_max=R.SHIPPED_WIDTH_BETA_MAX)
            _refuse_width_split_explosion(len(subs), bucket_g)
        for sub in subs:
            sub_a = a_flat[sub]
            sub_g = g_flat[sub]
            for names, wins, idx in _mpa_groups_for_bucket(
                    idx=sub, a_v=sub_a, g_v=sub_g, E_A_host=E_A_host,
                    base_mask_A_host=base_mask_A_host, omega_max=omega_max,
                    space=space, neg_omega_half=neg_omega_half,
                    edge_factor=edge_factor, rel_tol=rel_tol,
                    max_nodes=max_nodes):
                g_sub_lo, g_sub_hi = float(np.min(sub_g)), float(np.max(sub_g))
                # Ascending flat order inside the group: it is the order a
                # dense mask would have handed the kernel, so the selector
                # this group stands for is the identical object however it
                # is later materialized or gathered.
                idx = np.sort(idx).astype(ix_dtype)
                groups.append(WindowGroup(
                    name=(f"mpa[a={a_lo * RYD_TO_EV:.3g}"
                          f"-{a_hi * RYD_TO_EV:.3g}eV,"
                          f"G={g_sub_lo * RYD_TO_EV:.3g}"
                          f"-{g_sub_hi * RYD_TO_EV:.3g}eV]"
                          f"{'+'.join(names)}"),
                    windows=wins, idx_B=idx, field_shape=field_shape,
                    omega_operand=omega_complex,
                    n_modes=int(idx.size),
                    b_mass=_mass_idx(idx),
                    provenance="; ".join(w.provenance for w in wins)))

    stats = {
        "n_narrow": int(np.count_nonzero(narrow)),
        "n_wide": int(np.count_nonzero(wide)),
        "narrow_b_mass": _mass(narrow),
        "wide_b_mass": _mass(wide),
        "xi_ev": float(xi_ry) * RYD_TO_EV,
        "n_tau": int(sum(w.n_tau for grp in groups for w in grp.windows)),
    }
    return groups, stats


def format_pass_report(records, *, xi_ev):
    """The per-pass announcement, including the legacy-routed count.

    The legacy count is printed whether it is zero or not.  A run in which
    it is zero is a run whose every pole was resolved by the complex route,
    and that is worth reading off the log rather than inferring from the
    absence of a line.
    """
    lines = [
        "",
        "-" * 72,
        f"MPA Sigma: {len(records)} pole passes, narrow-pole threshold "
        f"xi = {float(xi_ev):.4f} eV",
        f"{'pass':>4}  {'Re Omega (eV)':>20}  {'Gamma (eV)':>18}  "
        f"{'legacy':>8}  {'complex':>9}  {'tau':>6}",
    ]
    tot_legacy = tot_mpa = tot_tau = 0
    for r in records:
        lines.append(
            f"{r.pole_index:>4}  {r.re_omega_min_ev:>9.3f}"
            f"..{r.re_omega_max_ev:<9.3f}  {r.gamma_min_ev:>8.2e}"
            f"..{r.gamma_max_ev:<8.2e}  {r.n_legacy_modes:>8d}  "
            f"{r.n_mpa_modes:>9d}  {r.n_tau_nodes:>6d}")
        tot_legacy += r.n_legacy_modes
        tot_mpa += r.n_mpa_modes
        tot_tau += r.n_tau_nodes
    total = tot_legacy + tot_mpa
    frac = (100.0 * tot_legacy / total) if total else 0.0
    lines += [
        f"  legacy-routed modes: {tot_legacy} of {total} ({frac:.2f} %) "
        f"-- poles with Gamma < xi, summed by the two-point crossing "
        f"machinery at that xi",
        f"  tau dispatches: {tot_tau}",
        "-" * 72,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  The pass loop proper -- the same device tau loop the two-point path runs,
#  called once per pole with that pole's slab resident and nothing else.
# ---------------------------------------------------------------------------

def run_pass_branch(
    *,
    groups,
    omega_nonneg_ry,
    E_A,
    B_p,
    psi,
    tau_kernels,
    mesh_xy,
    log_tag,
    print_fn,
):
    """Integrate one pass's window groups on one branch; return host tiles.

    Mirrors ``ppm_sigma._run_sigma_branch``'s tail exactly, and calls its
    ``_integrate_tau_windows_for_branch`` once per group -- the group loop
    is the only structural difference, and it exists because a group is
    what carries an explicit ``mask_B`` and its own ``Omega_q`` operand
    (real for the legacy-routed poles, complex for the rest).
    """
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common import timing
    from ..ppm_accumulators import _MemoryTileSink, _TauAccumulator
    from ..ppm_sigma import _SigmaBranchTiles, _integrate_tau_windows_for_branch

    n_omega = int(np.asarray(omega_nonneg_ry).shape[0])
    if n_omega == 0 or not groups:
        return None

    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, nk_proj, nb_proj = psi
    m_pad = int(psi_proj_xr.shape[1])
    n_pad = int(psi_proj_yn.shape[3])
    sink = _MemoryTileSink(
        shape=(n_omega, nk_proj, m_pad, n_pad),
        sharding=NamedSharding(mesh_xy, P(None, None, 'x', 'y')))
    accumulator = _TauAccumulator(
        omega_vec=jnp.asarray(omega_nonneg_ry, dtype=jnp.float64), sink=sink)

    tau_kernel, tau_kernel_x = tau_kernels
    for grp in groups:
        # ONE selector resident, and only while its group is integrating.
        # The group carries an index set; the tau loop wants the boolean
        # that index set stands for, so it is built here and dropped when
        # the loop returns -- 81.4 MB at a time on the production deck
        # instead of 218 of them held for the whole branch.
        _integrate_tau_windows_for_branch(
            windows=grp.windows, accumulator=accumulator, E_A=E_A,
            B_q=B_p, Omega_q=jnp.asarray(grp.omega_operand),
            base_mask_B=jnp.asarray(grp.dense_mask_B(), dtype=bool),
            psi_coh_xn=psi_coh_xn, psi_coh_yr=psi_coh_yr,
            psi_proj_xr=psi_proj_xr, psi_proj_yn=psi_proj_yn,
            tau_kernel=tau_kernel, tau_kernel_x=tau_kernel_x,
            log_tag=f"{log_tag} {grp.name}", print_fn=print_fn)

    with timing.section("sigma.finalize"):
        tiles, tile_index, tile_devices = accumulator.finalize_host_tiles()
    return _SigmaBranchTiles(
        tiles=tiles, tile_index=tile_index, devices=tile_devices,
        spatial_padded=(nk_proj, m_pad, n_pad),
        sharding=NamedSharding(mesh_xy, P(None, None, 'x', 'y')),
        nb_real=nb_proj)


#: Format version of a per-pass partial Σ_c file.  Stamped by
#: :func:`write_pass_partial` and required by :func:`combine_pass_partials`,
#: because a partial cube is the one artifact in this pipeline that is
#: MEANINGLESS ON ITS OWN and yet has exactly the shape, dtype and units of
#: the finished one.  Nothing downstream could tell a stack of partials
#: written under a changed convention from a stack written under this one,
#: so the convention is named in the bytes.
PARTIAL_FORMAT_VERSION = 1


def resolve_pole_subset(n_p, subset):
    """The poles ONE process integrates, in the pinned ascending order.

    ``None`` (or an empty selection) means every pole, which is the
    single-process production route and the only one that existed before
    the pass loop could be split.  Anything else is validated against
    :func:`~gw.mpa.fit_driver.pole_pass_order` and returned as a subsequence
    of it, NOT in the order the caller happened to type: the pinned order is
    a property of the accumulation, and a caller who writes ``3,0`` is
    asking which poles, not in which order to add them.

    Duplicates and out-of-range indices are refused rather than deduplicated
    or clipped.  The FALSE case each refusal guards is a coverage error in
    the recombination — a pole summed twice or never — and both of those
    produce a finite, smooth, plausible Σ that no shape or Hermiticity
    check can see.
    """
    from .fit_driver import pole_pass_order

    order = pole_pass_order(n_p)
    if subset is None:
        return order
    want = [int(p) for p in subset]
    if not want:
        return order
    bad = sorted({p for p in want if p not in order})
    if bad:
        raise ValueError(
            f"mpa pole subset: pole index/indices {bad} are outside this "
            f"store's pinned pass order {order} (n_p={int(n_p)}).  A subset "
            f"that names a pole the store does not have is a coverage error "
            f"in the recombination, not a smaller run.")
    dup = sorted({p for p in want if want.count(p) > 1})
    if dup:
        raise ValueError(
            f"mpa pole subset: pole index/indices {dup} appear more than "
            f"once.  A pole summed twice is a finite, smooth, plausible Σ "
            f"that no Hermiticity or shape check can see, so it is refused "
            f"here rather than silently deduplicated.")
    return tuple(p for p in order if p in set(want))


def resolve_pass_poles(n_p, pole_subset, group_subset):
    """The poles one leg walks, given both subset keys.

    A WINDOW-GROUP LEG NAMES ITS POLES TWICE and the two must agree.  The
    group spec already says which (pole, branch) pairs the leg owns, so
    the poles are derivable from it; ``mpa_pole_subset`` is still accepted
    beside it because it is the key the pole farm uses and a leg spelling
    both is the normal case.  What is refused is a DISAGREEMENT — a leg
    told to walk pole 3 while owning groups of pole 4 would read one
    pole's slab and integrate none of it, report success, and leave pole
    4's groups uncovered with every other leg reporting success too.
    """
    order = resolve_pole_subset(n_p, pole_subset)
    if group_subset is None:
        return order
    want = sorted({int(p) for (p, _b) in group_subset})
    bad = [p for p in want if p not in resolve_pole_subset(n_p, None)]
    if bad:
        raise ValueError(
            f"mpa_group_subset: pole index/indices {bad} are outside this "
            f"store's pinned pass order (n_p={int(n_p)}).")
    if pole_subset:
        extra = sorted(set(want) - set(order))
        idle = sorted(set(order) - set(want))
        if extra or idle:
            raise ValueError(
                f"this leg's mpa_pole_subset walks poles {list(order)} but "
                f"its mpa_group_subset owns groups of poles {want}"
                + (f"; poles {extra} would never be read" if extra else "")
                + (f"; poles {idle} would be read and never integrated"
                   if idle else "")
                + ".  Both are coverage errors that finish with rc=0 and a "
                  "plausible cube, so the two keys are required to name the "
                  "same poles rather than being reconciled here.")
    return tuple(p for p in resolve_pole_subset(n_p, None) if p in set(want))


def write_pass_partial(path, sigma_c_kij, records, *, n_p, poles,
                       omega_grid_ry, fit_src, group_spec=None, leg_id=None,
                       print_fn=print):
    """Write ONE process's partial Σ_c cube, with the manifest that makes
    it recombinable and the refusals that make it non-mistakable.

    THE OBJECT AND WHY IT NEEDS A MANIFEST.  ``Σ_c`` is a sum over poles
    and the pass loop is that sum written out one term at a time, so a
    process that walks a SUBSET of the poles returns a partial sum with
    the identical shape, dtype and units as the total.  Nothing about the
    array says which poles are in it.  Two stacks of partials — one
    covering the store's poles exactly once, one missing a pole or
    carrying one twice — differ by a smooth, finite, entirely plausible
    self-energy of a few tens of meV, which is the size of the effect this
    campaign is measuring.  So the manifest is not bookkeeping: it is the
    only thing standing between a parallel run and a wrong number that
    looks right.

    Stamped: the format version, the pole indices this cube contains, the
    store's ``n_p``, the ω grid in Ry, the fit store path, and the per-pass
    provenance triples the pass report prints.  :func:`combine_pass_partials`
    checks every one of them.

    ``group_spec`` IS THE SAME STAMP ONE LEVEL DOWN.  When the pass is
    farmed at window-group granularity a pole no longer names a cube:
    several cubes carry pole 3, each holding a different run of its window
    groups, and the pole list alone would then read as "pole 3 counted
    four times" to a coverage check that only knows about poles.  So a
    window-farmed leg stamps the group ranges it integrated, in the
    canonical ``<pole>.<branch>:<lo>-<hi>/<total>`` spelling, and
    :func:`combine_pass_partials` checks coverage at that granularity
    against the farm's manifest.  A cube with no group stamp is a
    whole-pole cube and is checked the way it always was.
    """
    import datetime
    import h5py

    arr = np.asarray(sigma_c_kij)
    om = np.asarray(omega_grid_ry, dtype=np.float64)
    poles = tuple(int(p) for p in poles)
    with h5py.File(str(path), "w") as f:
        f.attrs["mpa_partial_format_version"] = int(PARTIAL_FORMAT_VERSION)
        f.attrs["mpa_partial_poles"] = np.asarray(poles, dtype=np.int64)
        f.attrs["mpa_partial_n_p"] = int(n_p)
        f.attrs["mpa_partial_fit_store"] = str(fit_src)
        if group_spec:
            f.attrs["mpa_partial_group_spec"] = str(group_spec)
        if leg_id:
            f.attrs["mpa_partial_leg_id"] = str(leg_id)
        f.attrs["mpa_partial_written_utc"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat())
        f.create_dataset("omega_grid_ry", data=om)
        f.create_dataset("sigma_c_partial", data=arr.astype(np.complex128),
                         compression=None)
        prov = f.create_group("pass_records")
        for r in records:
            g = prov.create_group(str(int(r.pole_index)))
            g.attrs["n_legacy_modes"] = int(r.n_legacy_modes)
            g.attrs["n_mpa_modes"] = int(r.n_mpa_modes)
            g.attrs["legacy_b_mass"] = float(r.legacy_b_mass)
            g.attrs["mpa_b_mass"] = float(r.mpa_b_mass)
            g.attrs["n_tau_nodes"] = int(r.n_tau_nodes)
            g.attrs["re_omega_min_ev"] = float(r.re_omega_min_ev)
            g.attrs["re_omega_max_ev"] = float(r.re_omega_max_ev)
            g.attrs["gamma_min_ev"] = float(r.gamma_min_ev)
            g.attrs["gamma_max_ev"] = float(r.gamma_max_ev)
            g.attrs["groups"] = np.asarray(
                [str(x) for x in r.groups], dtype=h5py.string_dtype())
    print_fn(
        f"  MPA pass partial written: {path}\n"
        f"    poles {list(poles)} of n_p={int(n_p)}; Σ_c shape "
        f"{tuple(int(x) for x in arr.shape)}; this cube is a PARTIAL SUM "
        f"and is not a self-energy until it is combined with the rest."
        + (f"\n    window-group range: {group_spec}" if group_spec else ""))
    return str(path)


def _group_sort_key(spec):
    """A window-farmed cube's position in the pinned walk.

    Its lowest address: pole ascending, branch in ``_iter_branches`` order,
    then the planner's own group index.  The legs are contiguous runs of
    that walk, so sorting on the lowest address puts the cubes back in the
    order a single process would have summed their contents in — which is
    what "the ascending recombination" means once the split is finer than
    a pole.
    """
    from . import window_farm as WF

    return min((p, WF.BRANCH_KEYS.index(b), lo)
               for (p, b), (lo, _hi, _tot) in spec.items())


def _refuse_group_coverage(group_specs, *, order, manifest):
    """Every planned window group summed exactly once, or a refusal by name.

    The pole-level check cannot see this failure and the leg-level one
    cannot see it either: sixteen legs that all report success can still
    leave a hole, because a leg that dies before writing is caught by the
    manifest while a leg that ran the WRONG RANGE is not.  So the ranges
    themselves are tiled here — per (pole, branch), the union of the legs'
    ``[lo, hi)`` must be exactly ``[0, total)`` with no overlap — and the
    branch totals must agree with the manifest's universe.
    """
    if manifest is None:
        raise ValueError(
            "combine_pass_partials: these cubes are window-farmed (they "
            "carry group ranges) but no manifest was given.  The cubes can "
            "say which groups they hold and cannot say how many legs there "
            "were meant to be, so without the manifest a farm that lost a "
            "leg and a farm that was always this size are the same stack of "
            "files.  Pass the manifest the farm was launched from.")
    universe = {tuple(k.split(".", 1)): int(v)
                for k, v in (manifest.get("universe") or {}).items()}
    universe = {(int(p), b): n for (p, b), n in universe.items()}
    if not universe:
        raise ValueError(
            "combine_pass_partials: the manifest declares no group "
            "universe, so there is nothing to check coverage against.")
    seen = {}
    for path, spec in sorted(group_specs.items()):
        for key, (lo, hi, total) in spec.items():
            if key in universe and total != universe[key]:
                raise ValueError(
                    f"combine_pass_partials: {path} says pole {key[0]} "
                    f"branch {key[1]} has {total} window groups; the "
                    f"manifest says {universe[key]}.  The cube was "
                    f"integrated against a different partition than the one "
                    f"this farm was balanced for.")
            for g in range(lo, hi):
                prev = seen.setdefault((key, g), path)
                if prev != path:
                    raise ValueError(
                        f"combine_pass_partials: pole {key[0]} branch "
                        f"{key[1]} group {g} appears in both {prev} and "
                        f"{path}.  A window group summed twice is a smooth, "
                        f"finite Σ with one group's contribution doubled, "
                        f"which no shape, Hermiticity or magnitude check "
                        f"downstream can see.")
    holes = []
    for key, n in sorted(universe.items()):
        missing = [g for g in range(n) if (key, g) not in seen]
        if missing:
            holes.append(f"pole {key[0]} branch {key[1]}: groups "
                         f"{missing[:8]}{' ...' if len(missing) > 8 else ''} "
                         f"of {n}")
    if holes:
        raise ValueError(
            "combine_pass_partials: the window-farmed cubes do not cover "
            "the pass's window groups exactly once — "
            + "; ".join(holes)
            + ".  Every leg that ran may have succeeded and the missing "
              "groups still be missing; that is what this check is for.")
    poles_seen = sorted({key[0] for (key, _g) in seen})
    if tuple(poles_seen) != tuple(order):
        raise ValueError(
            f"combine_pass_partials: the window-farmed cubes cover poles "
            f"{poles_seen}, not the store's pinned pass order "
            f"{list(order)}.")


def combine_pass_partials(paths, *, n_p, omega_grid_ry, fit_src,
                          manifest=None, print_fn=print):
    """Sum per-pole partial cubes in the PINNED ascending order.

    Returns ``(sigma_c_total, poles, audit)``.

    THE CANONICAL RESULT IS THE PINNED-ORDER ACCUMULATION, and that is the
    whole reason this function exists rather than ``sum(cubes)``.  The
    re-association lemma (``fit_driver.accumulate_over_pole_passes``) is
    exact in exact arithmetic and NOT bit-exact in floating point, so a
    parallel run whose partials are summed in completion order is a
    DIFFERENT floating-point number from the same run summed ascending —
    reproducible only if the order is pinned.  This sums ascending, and
    it MEASURES what the other orders would have cost: ``audit`` carries
    the max-abs difference against a descending recombination and against
    a fixed shuffled one, so the re-association's size on the real cube is
    a reported number rather than an assumed ulp.

    COVERAGE IS CHECKED, NOT ASSUMED.  The union of the partials' pole
    lists must equal ``pole_pass_order(n_p)`` exactly — every pole once,
    no pole twice, none from a different store and none on a different ω
    grid.  Each of those failures produces a finite, smooth Σ that no
    downstream gate can distinguish from the right one; the FALSE case of
    this check is precisely those, and the red twins in
    ``tests/test_mpa_pass_partials.py`` construct each of them.

    AT WINDOW-GROUP GRANULARITY THE POLE IS NO LONGER THE UNIT, and the
    check follows the split down.  When the cubes carry
    ``mpa_partial_group_spec`` stamps, several of them hold the same pole
    and the pole-level check would read that as a duplicate; so coverage
    is then asked of the (pole, branch, group) addresses, which must tile
    each branch's ``[0, n_groups)`` exactly once, and a MANIFEST is
    required — because the partials themselves cannot say how many legs
    there were supposed to be, and "every group I was given is accounted
    for" is precisely the sentence a farm that silently lost a leg can
    also say.  ``manifest`` is a farm manifest as
    ``gw.mpa.window_farm.write_manifest`` writes it; the completeness
    refusal runs before a single cube is added.
    """
    import h5py

    from .fit_driver import pole_pass_order
    from . import window_farm as WF

    order = pole_pass_order(n_p)
    om_want = np.asarray(omega_grid_ry, dtype=np.float64)
    paths = sorted(str(p) for p in paths)
    if manifest is not None:
        WF.refuse_incomplete(manifest, print_fn=print_fn)
        declared = {str(leg["output"]) for leg in manifest["legs"]}
        stray = sorted(set(paths) - declared)
        if stray:
            raise ValueError(
                f"combine_pass_partials: {len(stray)} cube(s) were handed to "
                f"the merge that the manifest never declared: {stray}.  A "
                f"cube from a previous farm has the same shape, dtype and "
                f"units as one from this farm and adds a whole extra pass "
                f"to the sum; the manifest is the list of what belongs and "
                f"this is not on it.")
        paths = [str(leg["output"]) for leg in manifest["legs"]]
    loaded = []                     # (pole, path, cube) with one row per pole
    group_specs = {}                # path -> parsed group spec, when stamped
    for path in paths:
        with h5py.File(path, "r") as f:
            version = int(f.attrs.get("mpa_partial_format_version", -1))
            if version != PARTIAL_FORMAT_VERSION:
                raise ValueError(
                    f"combine_pass_partials: {path} declares partial format "
                    f"version {version}, not {PARTIAL_FORMAT_VERSION}.  A "
                    f"partial cube has the shape, dtype and units of a "
                    f"finished one, so a stack written under a different "
                    f"convention is refused by its stamp rather than "
                    f"summed.")
            store = str(f.attrs.get("mpa_partial_fit_store", ""))
            if store != str(fit_src):
                raise ValueError(
                    f"combine_pass_partials: {path} was integrated against "
                    f"fit store {store!r}, not {str(fit_src)!r}.  Summing "
                    f"partials from two stores returns a self-energy of no "
                    f"screening at all.")
            n_p_file = int(f.attrs.get("mpa_partial_n_p", -1))
            if n_p_file != int(n_p):
                raise ValueError(
                    f"combine_pass_partials: {path} was written from a store "
                    f"with n_p={n_p_file}, against n_p={int(n_p)} here.")
            om = np.asarray(f["omega_grid_ry"][()], dtype=np.float64)
            if om.shape != om_want.shape or not np.array_equal(om, om_want):
                raise ValueError(
                    f"combine_pass_partials: {path} carries a different Σ ω "
                    f"grid than this run.  The partials are summed elementwise "
                    f"on that axis, so a mismatched grid adds two different "
                    f"frequencies together.")
            spec_txt = str(f.attrs.get("mpa_partial_group_spec", "") or "")
            if spec_txt:
                group_specs[path] = WF.parse_group_subset(spec_txt)
            cube = np.asarray(f["sigma_c_partial"][()], dtype=np.complex128)
            for p in np.asarray(f.attrs["mpa_partial_poles"]).tolist():
                loaded.append((int(p), path, cube))
                cube = None         # one cube per FILE; poles share it

    # One file may carry several poles; the cube belongs to the file, and
    # the sum is over FILES taken in the ascending order of their lowest
    # pole.  Group them back up so a file is added exactly once.
    by_path = {}
    for p, path, cube in loaded:
        entry = by_path.setdefault(path, {"poles": [], "cube": None})
        entry["poles"].append(p)
        if cube is not None:
            entry["cube"] = cube

    if group_specs:
        if len(group_specs) != len(by_path):
            plain = sorted(set(by_path) - set(group_specs))
            raise ValueError(
                f"combine_pass_partials: this stack mixes window-farmed "
                f"cubes with whole-pole cubes ({plain} carry no group "
                f"stamp).  A whole-pole cube and the group-range cubes of "
                f"the same pole sum that pole twice, which is a finite, "
                f"smooth Σ wrong by one pole's worth; the two farm "
                f"granularities are not combinable and are refused rather "
                f"than reconciled.")
        _refuse_group_coverage(group_specs, order=order, manifest=manifest)
        ordered = sorted(by_path.items(),
                         key=lambda kv: _group_sort_key(group_specs[kv[0]]))
    else:
        covered = sorted(p for e in by_path.values() for p in e["poles"])
        if tuple(covered) != order:
            missing = sorted(set(order) - set(covered))
            twice = sorted({p for p in covered if covered.count(p) > 1})
            raise ValueError(
                f"combine_pass_partials: the partials do not cover the "
                f"store's pass order exactly once.  Pinned order {order}; "
                f"found {covered}; missing {missing}; duplicated {twice}.  A "
                f"missing pole and a doubled pole both return a finite, "
                f"smooth Σ of the same shape as the right one — the "
                f"difference is tens of meV, which is the size of the effect "
                f"being measured — so coverage is refused here rather than "
                f"checked downstream.")
        ordered = sorted(by_path.items(), key=lambda kv: min(kv[1]["poles"]))
    total = np.zeros_like(ordered[0][1]["cube"])
    for _, e in ordered:                       # THE PINNED ORDER
        total = total + e["cube"]

    rev = np.zeros_like(total)
    for _, e in reversed(ordered):
        rev = rev + e["cube"]
    shuffled = np.zeros_like(total)
    idx = list(range(len(ordered)))
    idx = idx[1::2] + idx[0::2]                # a fixed, order-changing walk
    for i in idx:
        shuffled = shuffled + ordered[i][1]["cube"]

    scale = float(np.max(np.abs(total))) if total.size else 0.0
    audit = {
        "n_files": len(ordered),
        "poles": order,
        "max_abs_sigma_ry": scale,
        "reassoc_descending_max_abs_ry": float(np.max(np.abs(total - rev)))
        if total.size else 0.0,
        "reassoc_shuffled_max_abs_ry": float(np.max(np.abs(total - shuffled)))
        if total.size else 0.0,
    }
    audit["reassoc_descending_rel"] = (
        audit["reassoc_descending_max_abs_ry"] / scale if scale else 0.0)
    audit["reassoc_shuffled_rel"] = (
        audit["reassoc_shuffled_max_abs_ry"] / scale if scale else 0.0)
    print_fn(
        f"  MPA pass recombination: {audit['n_files']} partial cubes "
        f"covering poles {list(order)}, summed in the PINNED ascending "
        f"order.\n"
        f"    re-association against a descending sum: "
        f"{audit['reassoc_descending_max_abs_ry']:.3e} Ry max-abs "
        f"({audit['reassoc_descending_rel']:.3e} relative)\n"
        f"    re-association against a shuffled sum:   "
        f"{audit['reassoc_shuffled_max_abs_ry']:.3e} Ry max-abs "
        f"({audit['reassoc_shuffled_rel']:.3e} relative)\n"
        f"    the ascending sum is the canonical result; the two numbers "
        f"above are what the order is worth, measured rather than assumed.")
    return total, order, audit


def compute_mpa_sigma_c_omega_grid(
    wfns, fit_src, meta, mesh_xy, *, ppm_cfg, quad, omega_grid_ry,
    laplace_ratio_max=DEFAULT_LAPLACE_RATIO_MAX, rel_tol=1.0e-8,
    allow_partial=False, pole_subset=None, group_subset=None,
    group_digests=None, census_out=None, census_sha="", plan_store=None,
    plan_verify=False, print_fn=print,
):
    """``Sigma_c(omega, k, m, n)`` from a staged multipole fit store.

    Reads as the physics outline section 7.5 asks for: one pass per pole,
    ascending, each pass a full four-branch Sigma integration over that
    pole's ``(B_p, Omega_p)`` slab and nothing else resident.

    Returns ``(sigma_c_kij, records)`` -- the replicated host tensor in Ry
    and the per-pass provenance the design requires each pass to record.

    ``pole_subset`` restricts the walk to some of the store's poles, in the
    pinned order regardless of how it is written.  It exists so the passes
    can be spread across devices -- one process per pole, each writing its
    own partial cube, recombined by :func:`combine_pass_partials` in the
    pinned ascending order, which is the canonical result.  ``None`` is the
    single-process production route and walks every pole.  A subset run
    returns a PARTIAL SUM with the shape, dtype and units of a finished
    self-energy, which is exactly why the partial writer stamps a manifest
    and the combiner refuses anything that does not cover the store's pass
    order exactly once.

    ``group_subset`` SPLITS THE SAME WALK ONE LEVEL FINER, and it is the
    key that lets a pass fill more devices than it has poles.  It names,
    per (pole, branch), a half-open range of the planner's window groups;
    the planner is called with the identical arguments and its output is
    sliced, so the groups, their membership, their rules and their node
    counts are the ones the unsplit walk would have used, and the leg
    integrates them in the order the unsplit walk would have.  A (pole,
    branch) the subset does not name is not planned at all — by linearity
    its contribution to THIS leg is zero, and skipping the plan is what
    keeps a sixteen-leg farm from paying the planning cost sixteen times.

    ``census_out`` is the other half of the mechanism and does no
    integration: it plans every branch of the poles it walks, writes the
    per-group tau-dispatch table that the farm balancer needs, and stops.
    The balance cannot be struck without it — ``n_tau`` per group is only
    known once the rules are built — and it is the one thing in this path
    that must be measured before any of it is scheduled.

    ``plan_store`` IS WHAT STOPS THE PLAN FROM MULTIPLYING WITH THE FARM.
    The planner is a pure function of one pole's slab, the branch's A-side
    arrays, the ω half-grid and a handful of scalars; nothing it reads is
    the split.  So a census run with ``plan_store`` set writes each
    (pole, branch) plan to an artifact addressed by those very inputs, and
    an integrating leg with the same key set LOADS the groups it owns
    instead of re-deriving them — which is the ~65 s per pole-touch that
    §9.5 of the 16-GPU plan measured a sixteen-leg farm paying sixteen
    times for eight poles' worth of work.  The artifact is addressed by a
    digest over every planner input, so a plan built from anything else is
    a file this leg does not ask for; a plan that is simply absent is
    refused by name rather than quietly re-planned.  ``plan_verify``
    additionally plans the branch the old way and compares the two, which
    is the gate that establishes the loaded plan IS the computed one.
    """
    import time

    import jax
    import jax.numpy as jnp

    from common import timing
    from file_io import mpa_store
    from . import plan_store as PS
    from . import window_farm as WF
    from ..ppm_sigma import (
        _iter_branches, _prepare_sigma_state, pad_sigma_window,
        strip_sigma_window)
    from ..ppm_tau_kernel import _get_sigma_tau_kernel
    from ..ppm_windows import _to_host_np

    omega_req = np.asarray(omega_grid_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_grid_ry must be a 1D non-empty array.")
    omega_max_ry = float(np.max(np.abs(omega_req)))
    edge_factor = float(ppm_cfg.window_edge_factor)
    xi_ry = narrow_pole_threshold_ry(
        float(ppm_cfg.regularization_ev) / RYD_TO_EV, omega_max_ry,
        edge_factor)

    ledger = mpa_store.fit_completion_ledger(fit_src)
    n_p = int(ledger["n_p"])
    # WHAT THIS POLE FIELD IS A FIT TO, asked before any of it is read.
    # W(z) = v + W_c(z) and only W_c has poles: v is frequency-
    # independent, it is already counted in Sigma_x, and its tau
    # transform is a delta at tau = 0, so a pole field carrying it feeds
    # the bare Coulomb interaction into the G(tau) W(tau) convolution at
    # every node.  The 2026-08-09 bridge gate is what that reads as --
    # Sigma_c = -130.651 eV where Godby-Needs on the same two samples
    # gives +0.6754 eV -- and no gate on this side of the seam can see
    # it, which is why the answer is an attr the fit driver stamped and
    # not something measured here.
    mpa_store.require_correlation_part(
        ledger.get("screening_content"),
        where="compute_mpa_sigma_c_omega_grid", source=str(fit_src))
    refuse_wedge_pole_slab(int(ledger["n_q"]), int(meta.nk_tot))

    s = wfns.slices
    psi_proj_xr, psi_proj_yn, nb_proj = pad_sigma_window(
        wfns.xr(s.sigma), wfns.yn(s.sigma), mesh_xy)
    psi = (wfns.xn(s.full), wfns.yr(s.full), psi_proj_xr, psi_proj_yn,
           int(wfns.xr(s.sigma).shape[0]), nb_proj)
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    tau_kernels = (_get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid),
                   _get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid,
                                         merged_x=True))

    idx_pos = np.where(omega_req >= 0.0)[0]
    idx_neg = np.where(omega_req < 0.0)[0]
    n_omega = int(omega_req.size)
    tile_acc = None
    tile_meta = None
    records = []

    # THE PINNED ORDER.  Ascending pole index: the store's leading axis,
    # the fitter's own sort, and the better-conditioned accumulation
    # direction.  The re-association this loop performs is exact in exact
    # arithmetic and not bit-exact in floating point, so the order being
    # pinned is what makes the result reproducible run to run.  A
    # ``pole_subset`` narrows WHICH poles this process walks and never
    # the order it walks them in -- ``resolve_pole_subset`` returns a
    # subsequence of the pinned order, so a split run and a whole run
    # associate the same way within each process.
    walk = resolve_pass_poles(n_p, pole_subset, group_subset)
    census_rows, census_digests = [], {}
    # THE PER-LEG FIXED TERM, ACCOUNTED RATHER THAN INFERRED.  §9.5 of the
    # 16-GPU plan had to back it out of a census leg's total (106-115 s
    # minus a ~42 s bring-up) because nothing in the run said where the
    # time went.  These four counters and the line they print at the end
    # are that measurement, per leg, in every log — read, plan, load, and
    # the address hash the plan store pays for its own staleness guard.
    fixed = {"read_s": 0.0, "plan_s": 0.0, "load_s": 0.0, "addr_s": 0.0,
             "verify_s": 0.0, "poles": 0, "branches": 0, "loaded": 0}
    plan_dir = str(plan_store or "") or None
    if plan_dir is not None and census_out is not None:
        os.makedirs(plan_dir, exist_ok=True)
    plan_written = {}
    if census_out is not None:
        print_fn(
            f"  MPA Σ: CENSUS ONLY over poles {list(walk)} -- every branch "
            f"is planned and NOTHING is integrated.  This run produces the "
            f"window-group table a farm balances on and no self-energy.")
    elif len(walk) != n_p or group_subset is not None:
        print_fn(
            f"  MPA Σ: this process integrates poles {list(walk)} of "
            f"{n_p} -- a PARTIAL sum over the pole axis.  Its Σ_c is not "
            f"a self-energy until every other pole's partial is added."
            + (f"\n    window-group range: "
               f"{WF.format_group_subset(group_subset)}"
               if group_subset is not None else ""))
    for p in walk:
        t_read = time.perf_counter()
        with timing.section("mpa.pass_read"):
            Omega_p, B_p = mpa_store.read_pole_slice(
                fit_src, p, allow_partial=allow_partial)
        a_host = np.asarray(np.real(Omega_p), dtype=np.float64)
        g_host = np.abs(np.asarray(np.imag(Omega_p), dtype=np.float64))
        b_abs = np.abs(np.asarray(B_p))
        fixed["read_s"] += time.perf_counter() - t_read
        fixed["poles"] += 1
        print_fn(f"  MPA pass read: pole {int(p)} slab in "
                 f"{time.perf_counter() - t_read:.2f} s")
        # The complex operand the MPA-routed groups take, built once per
        # pole rather than once per branch, and only on the route that
        # needs it: a loaded plan names which of the two slab operands
        # each group takes and the loader rebuilds it from these arrays.
        omega_complex_host = (a_host - 1j * g_host
                              if (plan_dir is not None
                                  and census_out is None) else None)
        state = _prepare_sigma_state(
            jnp.asarray(wfns.enk[:, s.full]), jnp.asarray(wfns.occ[:, s.full]),
            jnp.asarray(B_p, dtype=jnp.complex128),
            jnp.asarray(a_host, dtype=jnp.float64),
            jnp.ones(a_host.shape, dtype=bool),
            jnp.asarray(str(ppm_cfg.fermi_reference) == "midgap", dtype=bool),
            jnp.asarray(True, dtype=bool))
        live = _host_at_source_shape(state.B_mask, bool, _to_host_np)
        # The pole-level half of the plan address, taken once for the four
        # branches that share these arrays -- and here rather than beside
        # the read, because ``live`` is the store's B mask and comes from
        # ``_prepare_sigma_state``, not from the slab.  Four times over
        # the hash cost 12.56 s a pole against 3.14 s once (2026-08-10);
        # the four branches genuinely read the same Re Omega, Gamma, live
        # mask and |B|, and only E_A, its mask and the ω half-grid differ.
        slab_dig = None
        if plan_dir is not None:
            t_slab = time.perf_counter()
            slab_dig = PS.slab_digest(
                a_ry=a_host, gamma_ry=g_host, live_mask=live, b_abs=b_abs)
            fixed["addr_s"] += time.perf_counter() - t_slab
        rec = PassRecord(
            pole_index=int(p),
            re_omega_min_ev=float(np.min(a_host, where=live, initial=np.inf))
            * RYD_TO_EV,
            re_omega_max_ev=float(np.max(a_host, where=live, initial=0.0))
            * RYD_TO_EV,
            gamma_min_ev=float(np.min(g_host, where=live, initial=np.inf))
            * RYD_TO_EV,
            gamma_max_ev=float(np.max(g_host, where=live, initial=0.0))
            * RYD_TO_EV)

        B_dev = jnp.asarray(state.B_corr)
        for br in _iter_branches(
                omega_pos=np.asarray(omega_req[idx_pos], dtype=np.float64),
                idx_pos=idx_pos,
                omega_neg_abs=np.asarray(-omega_req[idx_neg],
                                         dtype=np.float64),
                idx_neg=idx_neg,
                E_cond=state.E_cond, H_val=state.H_val,
                cond_mask=state.cond_mask, val_mask=state.val_mask):
            bkey = WF.branch_key(br.space, br.neg_omega_half)
            # A BRANCH THIS LEG DOES NOT OWN IS NOT PLANNED, and that is
            # the difference between a sixteen-leg farm that pays the
            # planner once per pole and one that pays it once per leg.
            # Skipping is exact rather than approximate: this leg's
            # contribution to a branch it holds no groups of is zero by
            # linearity, and the leg that does hold them plans them.
            if (census_out is None and group_subset is not None
                    and (int(p), bkey) not in group_subset):
                continue
            E_A_host = _host_at_source_shape(br.E_A, np.float64, _to_host_np)
            mask_A_host = _host_at_source_shape(
                br.base_mask_A, bool, _to_host_np)
            fixed["branches"] += 1

            def _plan_this_branch():
                """``plan_branch_groups`` at this (pole, branch), verbatim."""
                with timing.section("mpa.windows"):
                    return plan_branch_groups(
                        a_ry=a_host, gamma_ry=g_host, live_mask=live,
                        E_A_host=E_A_host, base_mask_A_host=mask_A_host,
                        omega_nonneg_ry=br.omega_abs, space=br.space,
                        neg_omega_half=br.neg_omega_half, xi_ry=xi_ry,
                        edge_factor=edge_factor, b_abs=b_abs, rel_tol=rel_tol,
                        laplace_ratio_max=laplace_ratio_max,
                        target_error=float(quad.target_error),
                        laplace_max_nodes=int(quad.max_nodes),
                        crossing_eps_q=float(quad.crossing_eps_q),
                        crossing_max_nodes=int(quad.crossing_max_nodes),
                        use_shipped_minimax_tables=bool(
                            quad.use_shipped_tables),
                        log_tag=f"p{p} {br.tag}", print_fn=print_fn)

            plan_addr, plan_file = None, None
            if plan_dir is not None:
                t_addr = time.perf_counter()
                plan_addr = PS.branch_address(
                    source_sha=census_sha, fit_store=fit_src, n_p=n_p,
                    pole=p, bkey=bkey, slab=slab_dig,
                    arrays={
                        "E_A_host": E_A_host,
                        "base_mask_A_host": mask_A_host,
                        "omega_nonneg_ry": np.asarray(br.omega_abs,
                                                      dtype=np.float64),
                    },
                    scalars={
                        "xi_ry": float(xi_ry),
                        "edge_factor": float(edge_factor),
                        "rel_tol": float(rel_tol),
                        "laplace_ratio_max": float(laplace_ratio_max),
                        "target_error": float(quad.target_error),
                        "laplace_max_nodes": int(quad.max_nodes),
                        "crossing_eps_q": float(quad.crossing_eps_q),
                        "crossing_max_nodes": int(quad.crossing_max_nodes),
                        "use_shipped_minimax_tables": bool(
                            quad.use_shipped_tables),
                        "space": str(br.space),
                        "neg_omega_half": bool(br.neg_omega_half),
                    })
                fixed["addr_s"] += time.perf_counter() - t_addr
                plan_file = PS.plan_path(plan_dir, pole=p, bkey=bkey,
                                         address=plan_addr)

            loaded_lo = None
            if plan_dir is not None and census_out is None:
                # THE LOAD ROUTE.  The groups this leg owns come off disk;
                # the ones it does not are never read at all, which is why
                # the artifact stores each group's index set as its own
                # dataset rather than one array per branch.
                if not os.path.exists(plan_file):
                    PS.refuse_missing_plan(plan_dir, pole=p, bkey=bkey,
                                           address=plan_addr)
                t_load = time.perf_counter()
                header = PS.read_plan_header(plan_file)
                rng = WF.group_range(group_subset, pole=p, bkey=bkey,
                                     n_groups=header["n_groups"])
                if rng is None:                  # pragma: no cover - guarded
                    continue                     # above by the branch skip
                lo_i, hi_i, total_i = rng
                WF.check_partition(
                    n_groups=header["n_groups"],
                    digest_got=WF.group_plan_digest_from_rows(header["rows"]),
                    pole=p, bkey=bkey, total=total_i, lo=lo_i, hi=hi_i,
                    digest=(group_digests or {}).get(f"{int(p)}.{bkey}"),
                    source="the stored plan")
                groups = PS.read_branch_plan(
                    plan_file, lo=lo_i, hi=hi_i, a_ry=a_host,
                    omega_complex=omega_complex_host)
                stats = header["stats"]
                loaded_lo = lo_i
                dt_load = time.perf_counter() - t_load
                fixed["load_s"] += dt_load
                fixed["loaded"] += 1
                print_fn(
                    f"  MPA plan p{p} {br.tag} ({bkey}): LOADED groups "
                    f"[{lo_i}, {hi_i}) of {total_i} in {dt_load:.2f} s "
                    f"(address {plan_addr[:12]}, digest "
                    f"{header['group_plan_digest']})")
                if plan_verify:
                    # THE GATE, RUN WHERE THE REAL INPUTS ARE.  Plan the
                    # branch the old way and fingerprint both to the last
                    # bit: the τ nodes, the weights, the A-masks, both
                    # reference energies, the prefactors and the index
                    # sets.  A verification that only compared group
                    # counts would pass for a plan whose every rule had
                    # moved.
                    t_ver = time.perf_counter()
                    fresh, fresh_stats = _plan_this_branch()
                    got = WF.full_plan_digest(fresh[lo_i:hi_i])
                    want = WF.full_plan_digest(groups)
                    whole = WF.full_plan_digest(fresh)
                    dt_ver = time.perf_counter() - t_ver
                    # The same seconds under two names on purpose: a
                    # verifying leg is the only one that pays BOTH routes,
                    # so it is also the cleanest A/B of them, and the
                    # summary line says which is a subset of which.
                    fixed["verify_s"] += dt_ver
                    fixed["plan_s"] += dt_ver
                    ok = (got == want)
                    stats_same = all(
                        float(fresh_stats[k]) == float(stats[k])
                        for k in ("n_narrow", "n_wide", "narrow_b_mass",
                                  "wide_b_mass", "n_tau"))
                    print_fn(
                        f"  PLAN-VERIFY p{p} {bkey}: loaded {want} vs "
                        f"freshly planned {got} over groups "
                        f"[{lo_i}, {hi_i}) -- "
                        f"{'BIT-IDENTICAL' if ok else '*** DIFFERENT ***'}"
                        f"; whole-branch fresh digest {whole}; planner "
                        f"stats {'match' if stats_same else '*** DIFFER ***'}"
                        f"; fresh plan cost {dt_ver:.2f} s against "
                        f"{dt_load:.2f} s to load")
                    if not ok:
                        raise ValueError(
                            f"mpa_plan_verify: the plan loaded for pole {p} "
                            f"branch {bkey} is not the plan this tree "
                            f"computes from the same inputs.  The artifact "
                            f"is addressed by a digest over every planner "
                            f"input, so this cannot be a stale store; it is "
                            f"a serialization defect, and every leg that "
                            f"loaded it integrated certified rules that are "
                            f"not the ones the planner built.")
            else:
                t_plan = time.perf_counter()
                groups, stats = _plan_this_branch()
                fixed["plan_s"] += time.perf_counter() - t_plan
                print_fn(f"  MPA plan p{p} {br.tag} ({bkey}): PLANNED "
                         f"{len(groups)} groups in "
                         f"{time.perf_counter() - t_plan:.2f} s")
                if plan_dir is not None and census_out is not None:
                    PS.write_branch_plan(
                        plan_file, groups, address=plan_addr, pole=p,
                        bkey=bkey, source_sha=census_sha, fit_store=fit_src,
                        n_p=n_p, stats=stats, a_ry=a_host,
                        omega_complex=a_host - 1j * g_host,
                        print_fn=print_fn)
                    plan_written[f"{int(p)}.{bkey}"] = {
                        "path": plan_file, "address": plan_addr}
            rec.n_legacy_modes = stats["n_narrow"]
            rec.n_mpa_modes = stats["n_wide"]
            rec.legacy_b_mass = stats["narrow_b_mass"]
            rec.mpa_b_mass = stats["wide_b_mass"]
            if census_out is not None:
                census_rows += WF.census_rows_from_groups(
                    groups, pole=p, bkey=bkey, branch_tag=br.tag)
                census_digests[f"{int(p)}.{bkey}"] = WF.group_plan_digest(
                    groups)
                print_fn(
                    f"  MPA census p{p} {br.tag} ({bkey}): {len(groups)} "
                    f"window groups, {stats['n_tau']} tau dispatches")
                continue
            if loaded_lo is None:
                groups, group_lo = WF.select_branch_groups(
                    groups, pole=p, bkey=bkey, spec=group_subset,
                    digest=(group_digests or {}).get(f"{int(p)}.{bkey}"))
            else:
                group_lo = loaded_lo
            if not groups:
                continue
            rec.n_tau_nodes += int(sum(int(w.n_tau) for g in groups
                                       for w in g.windows))
            rec.groups += [g.name for g in groups]
            del group_lo
            branch_tiles = run_pass_branch(
                groups=groups, omega_nonneg_ry=br.omega_abs, E_A=br.E_A,
                B_p=B_dev, psi=psi, tau_kernels=tau_kernels,
                mesh_xy=mesh_xy, log_tag=f"p{p} {br.tag}", print_fn=print_fn)
            if branch_tiles is None:
                continue
            if tile_acc is None:
                tile_meta = branch_tiles
                tile_acc = [np.zeros((n_omega,) + t.shape[1:],
                                     dtype=np.complex128)
                            for t in branch_tiles.tiles]
            with timing.section("mpa.branch_fold"):
                for d, t in enumerate(branch_tiles.tiles):
                    tile_acc[d][np.asarray(br.omega_idx, dtype=np.int64)] += t
        records.append(rec)
        del omega_complex_host

    # THE LINE §10 IS WRITTEN FROM.  One greppable row per leg, so the
    # fixed term is a measurement in the log rather than a subtraction
    # performed afterwards by whoever reads it.
    print_fn(
        f"  MPA-FIXED-TERM: poles={fixed['poles']} branches="
        f"{fixed['branches']} read={fixed['read_s']:.2f}s "
        f"plan={fixed['plan_s']:.2f}s load={fixed['load_s']:.2f}s "
        f"({fixed['loaded']} branches) address_hash={fixed['addr_s']:.2f}s "
        f"verify={fixed['verify_s']:.2f}s (a subset of plan) total_fixed="
        f"{fixed['read_s'] + fixed['plan_s'] + fixed['load_s'] + fixed['addr_s']:.2f}s")

    if census_out is not None:
        WF.write_census(
            census_out, census_rows, fit_store=str(fit_src), n_p=n_p,
            sha=str(census_sha), digests=census_digests,
            extra={"poles": [int(x) for x in walk],
                   "omega_grid_ry": [float(x) for x in omega_req],
                   "plans": plan_written})
        print_fn(
            f"  MPA pass census written: {census_out}\n"
            f"    {len(census_rows)} window groups over poles "
            f"{list(walk)}, "
            f"{sum(int(r['n_tau']) for r in census_rows)} tau dispatches.  "
            f"NO Σ was integrated by this run.")
        return None, records

    if tile_acc is None:
        raise RuntimeError(
            "MPA Sigma: no branch produced a tile, so no self-energy was "
            "accumulated.  A pole field with no live modes reads back as "
            "zero, which is indistinguishable from a converged dark "
            "channel; it is refused instead.")

    with timing.section("mpa.host_gather"):
        padded = (n_omega,) + tuple(int(x) for x in tile_meta.spatial_padded)
        full_pad = np.zeros(padded, dtype=np.complex128)
        if int(jax.process_count()) == 1:
            for t, ix in zip(tile_acc, tile_meta.tile_index):
                full_pad[ix] = t
        else:                                    # pragma: no cover - multihost
            arrays = [jax.device_put(t, d)
                      for t, d in zip(tile_acc, tile_meta.devices)]
            full_pad = np.asarray(_to_host_np(
                jax.make_array_from_single_device_arrays(
                    padded, tile_meta.sharding, arrays), tiled=False))
    sigma_c = strip_sigma_window(full_pad, tile_meta.nb_real)
    print_fn(format_pass_report(records, xi_ev=xi_ry * RYD_TO_EV))
    return jnp.asarray(sigma_c, dtype=jnp.complex128), records
