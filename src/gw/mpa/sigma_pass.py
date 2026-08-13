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
        Omega_p, B_p = mpa_store.read_pole_slice(       # wedge -> full BZ
            fit, p, unfold=unfold_q, mesh_xy=mesh_xy)
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

THE STORE'S q AXIS, WEDGE OR FULL
----------------------------------
A fit store written on the symmetry wedge carries ``n_q = 8`` where this
module's k-q sums want the full BZ, and until 2026-08-10 that store was
refused by name: unfolding a POLE FIELD is not the same operation as
unfolding ``W``, and what time reversal does to a pole in the closed
fourth quadrant was the half nobody had certified.

It is certified now, and the answer removed the hazard instead of
managing it.  Time reversal acts on a ``(mu, nu)`` operator as the PAIR
TRANSPOSE at the same frequency; the elementwise conjugate the W unfold
applied is the Hermitian shorthand for that swap, and ``W_c`` at a
complex sample is not Hermitian.  Under the corrected rule the unfold
multiplies each element by a FREQUENCY-INDEPENDENT scalar, so ``Omega_p``
carries the permutation alone and ``B_p`` the permutation and the phase,
with no conjugation anywhere -- ``Im Omega_p < 0`` is preserved by
construction and ``e^{+Gamma*tau}`` is unreachable.
:func:`resolve_pole_q_axis` decides which zone the store is on and the
pass loop reads its slices with ``unfold=`` set accordingly; the map
itself lives in ``file_io.mpa_store.unfold_pole_field`` and its rule in
``symmetry_maps.unfold_isdf_operator``, so this module still does not
own an opinion about the symmetry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

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
    "resolve_pole_q_axis",
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
    idx_B: np.ndarray | None
    field_shape: tuple
    omega_operand: np.ndarray
    n_modes: int
    b_mass: float
    provenance: str
    selector_bounds_B: np.ndarray | None = None

    def dense_mask_B(self, omega_complex=None):
        """The identical boolean this group used to carry, rebuilt on call.

        Kept because two consumers still want a dense selector and both
        are bounded: the device tau loop, which needs ONE selector
        resident per group and never a second, and the planning tests,
        whose fields are hundreds of modes rather than eighty million.
        It is a method and not an attribute so that materializing 81.4 MB
        is something a caller does on purpose.
        """
        if self.selector_bounds_B is not None:
            if omega_complex is None:
                omega_complex = self.omega_operand
            return _sector_selector_host(omega_complex,
                                         self.selector_bounds_B)
        if self.idx_B is None:
            raise ValueError(f"window group {self.name!r} has no selector")
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


def resolve_pole_q_axis(ledger, n_k_tot):
    """Does this pole store need unfolding to reach the Sigma kernel's zone?

    Returns ``True`` when the store is on the symmetry wedge and its
    slices must be read with ``unfold=True``.  Refuses, by name, a store
    whose q extent is neither the full zone nor a declared wedge of it.

    THE REFUSAL THIS REPLACES, AND WHY IT IS GONE.  Until 2026-08-10 this
    was ``refuse_wedge_pole_slab``, which turned a wedge store away
    outright: unfolding a pole field is not the operation that unfolds
    ``W``, the residues were thought to carry a conjugation on the
    time-reversed star members, and *"a fourth-quadrant pole conjugated
    the wrong way becomes exp(+Gamma*tau), which grows"*.  The named
    retirement gate was to rebuild ``W_c(q, z_j)`` from the unfolded
    poles at the store's own ``2*n_p`` samples and compare against the
    same store's ``W`` unfolded by ``unfold_isdf_operator``.

    That gate was run, and it retired the refusal by DISSOLVING its
    premise rather than by clearing it.  There is no conjugation.  Time
    reversal acts on a ``(mu, nu)`` operator as the PAIR TRANSPOSE at the
    same frequency -- ``O(-q, z)_{mu nu} = O(q, z)_{nu mu}`` -- and the
    elementwise conjugate the unfold applied was the Hermitian shorthand
    for that swap, exact for every static object the map had been
    certified on and wrong by O(1) for a ``W_c`` slab at a complex
    sample.  Under the corrected rule the unfold's action on any element
    is a FREQUENCY-INDEPENDENT scalar times another element's value, so
    ``Omega_p`` permutes and ``B_p`` permutes and takes a phase, and
    NOTHING is conjugated: ``Im Omega_p < 0`` survives the unfold by
    construction, and ``exp(+Gamma*tau)`` is not reachable from here.
    See ``file_io.mpa_store.unfold_pole_field`` for the map and
    ``symmetry_maps.unfold_isdf_operator`` for the rule.
    """
    n_q = int(ledger["n_q"])
    n_k_tot = int(n_k_tot)
    if n_q == n_k_tot:
        return False
    n_q_full = int(ledger["n_q_full"])
    if ledger["q_storage"] == "ibz" and n_q_full == n_k_tot:
        return True
    raise ValueError(
        f"MPA Sigma: the fit store's pole axis has n_q={n_q} but the "
        f"Sigma kernel sums over the full Bloch zone, n_k_tot={n_k_tot}. "
        f"The store declares q_storage={ledger['q_storage']!r} over a "
        f"zone of {n_q_full} points, so it is neither that zone nor a "
        f"wedge of it, and unfolding it would be guessing which q each "
        f"row is.  A wedge store reaches the full zone through "
        f"mpa_store.read_pole_slice(..., unfold=True), which needs the "
        f"W(omega) file's unfold tables stamped beside the poles "
        f"(mpa_store.stamp_fit_unfold_tables); a store fitted against a "
        f"DIFFERENT k-grid than this run's is not a store this Sigma can "
        f"use at all.")


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
    # THE BOUNDARY CONVENTION, AND IT IS UPPER-CLOSED: bin b is
    # ``(edges[b-1], edges[b]]``, which is what ``side='right'`` names on
    # an ascending array.  One sentence decides it -- A PANE'S CERTIFIED
    # PARAMETER IS ITS SUPREMUM, SO A PANE MUST CONTAIN ITS SUPREMUM.
    # Every rule this module builds is built at the pane's LARGEST width
    # (``g_hi = max(g_v)`` in _mpa_groups_for_bucket) and the crossing
    # pane at its largest ``Re Omega`` (the ``a_v <= T`` predicate, which
    # has always been upper-closed), so a pole sitting exactly on a bin
    # edge belongs to the interval whose certificate was built AT its own
    # width: covered inclusively, by exactly one rule, never by neither
    # and never by both.  The binned clause's bound is closed at the top
    # for the same reason (``beta <= r``, inclusive), and the catalog's
    # own ``beta_covers`` compares inclusively, so the three agree.
    #
    # IT USED TO BE ``side='left'`` and the two halves of the pane
    # assignment path disagreed: the width axis was ``[lo, hi)`` while
    # Sigma's B-side predicate was ``(lo, hi]``, so a pole exactly on a
    # threshold was assigned in OPPOSITE directions by the two.  Only a
    # width that lands exactly on ``g_lo * r**k`` in float64 can tell the
    # difference -- which is not never, because ``r = 4`` makes those
    # edges exact binary shifts -- so the flag-off byte-identity gate is
    # also the measurement of whether any production pole sits on one.
    cuts = ([0]
            + [int(c) for c in np.searchsorted(g_sorted, edges, side="right")]
            + [int(idx_sorted.size)])
    out = []
    for b in range(n_bins):
        lo, hi = cuts[b], cuts[b + 1]
        if hi > lo:
            out.append(idx_sorted[lo:hi])
    return out


def _refuse_mis_binned_pane(g_lo, g_hi, r_max, *, where, beta=None):
    """A pane whose widths do not fit the bin it claims.  Refuse it.

    THE FALSE CASE THIS GUARD IS FOR is not a bug in the binning; it is a
    pane reaching the rule builder by some route that did not bin it, or
    binned it at a different ratio than the one the catalog entry was
    fetched for.  Either way the derivation that licenses ``beta <= r``
    has stopped holding -- it needs ``Gamma_hi/Gamma_lo <= r`` and
    nothing else supplies it -- so the certificate on the entry is
    certifying a band the pane does not sit inside.  That is a wrong
    number rather than a slow one, which is why it is checked at the
    pane and not inferred from the beta afterwards.

    The comparison is inclusive at ``r`` and carries one ulp of slack,
    because the bin edges are ``g_lo * r**k`` and a pole sitting exactly
    on one is IN the bin by this module's upper-closed convention.
    """

    lo, hi, r = float(g_lo), float(g_hi), float(r_max)
    if not (lo > 0.0) or not np.isfinite(hi):
        from . import sigma_routing as R
        raise R.RoutingRefusal(
            f"MPA {where} window: the pane's width range [{lo:.6g}, "
            f"{hi:.6g}] Ry is not a positive finite interval, so no bin "
            f"ratio can be formed for it.",
            code="mis_binned_pane")
    ratio = hi / lo
    if ratio <= r * (1.0 + 1.0e-12):
        return float(ratio)
    from . import sigma_routing as R
    raise R.RoutingRefusal(
        f"MPA {where} window: the binned-width clause was asked for at "
        f"bin ratio r={r:g}, but this pane's widths span "
        f"Gamma_hi/Gamma_lo = {ratio:.6g} (Gamma in [{lo:.6g}, {hi:.6g}] "
        f"Ry)"
        + ("" if beta is None else f", beta={float(beta):.6g}")
        + ".  The clause's edge IS the bin ratio: beta <= "
        "Gamma_hi/Gamma_lo holds only because pade_fit's fourth guard "
        "puts Gamma_p <= a_p on every pole, so a pane wider than r has "
        "no derivation putting its beta under r and the band certificate "
        "fetched for r does not cover it.  This is a pane that reached "
        "the rule builder without being binned, or binned at a different "
        "ratio than the entry was fetched for -- not a physics corner.",
        code="mis_binned_pane", beta=None if beta is None else float(beta))


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


#: Below this many modes the width sort stays on the host: the device
#: round trip costs more than a numpy sort of a small selection, and the
#: planner's own unit fields are hundreds of modes.  Above it the sort is
#: the planner's dominant term (a stable argsort of ~80 million float64 is
#: seconds of single-threaded numpy, against ~50 ms of GPU sort plus its
#: transfers), which is what makes the branch plan ~16 s.
_DEVICE_SORT_MIN_MODES = 1 << 20


def _sorted_by_width(g_flat, idx):
    """``(idx, gamma)`` for one selection, sorted ascending in ``Gamma``.

    Every splitter in this planner cuts on ``Gamma`` -- the clause-safe
    recursion at a geometric mean, the crossing panes at fixed ratio
    edges -- so sorting once turns every later split into a slice.  The
    sort is stable, so equal widths keep their field order and the
    partition is reproducible run to run.

    THE SORT RUNS ON THE DEVICE WHEN THE SELECTION IS LARGE, AND THAT IS A
    BIT-IDENTICAL SUBSTITUTION RATHER THAN A NUMERICAL ONE.  A stable sort
    of a fixed key vector is a UNIQUELY DETERMINED permutation -- ties
    resolve to ascending source position by definition of stability, and
    the keys here are the same float64 values gathered from the same
    array -- so ``jnp.argsort(..., stable=True)`` and
    ``np.argsort(..., kind='stable')`` cannot return different orders.
    Nothing is summed, nothing is rounded, and no reduction is
    re-associated: the planner downstream of this call sees the identical
    index array, hence the identical panes, hence the identical certified
    rules.  ``tests/test_mpa_jax_native.py`` pins the two against each
    other, including on a selection engineered to be mostly ties.

    The device path is skipped, silently and without changing the answer,
    when the selection is small or when jax cannot serve it (no backend,
    or an allocator that refuses the transient).  A fallback that returns
    the same permutation is a performance decision and not a correctness
    one, which is the only reason it is allowed to be silent.
    """
    ix = np.asarray(idx, dtype=np.int64)
    g_all = np.asarray(g_flat, dtype=np.float64)
    if ix.size >= _DEVICE_SORT_MIN_MODES:
        out = _sorted_by_width_device(g_all, ix)
        if out is not None:
            return out
    g_v = g_all[ix]
    order = np.argsort(g_v, kind="stable")
    return ix[order], g_v[order]


def _sorted_by_width_device(g_all, ix):
    """:func:`_sorted_by_width`'s gather-and-stable-sort, on the device.

    Returns ``None`` rather than raising if the device cannot serve the
    transient, because the host path computes the same permutation and a
    planner that refuses to plan because a sort did not fit would be
    trading a correct answer for a faster one.
    """
    try:
        import jax
        import jax.numpy as jnp

        if not jax.default_backend():                # pragma: no cover
            return None
        ix_d = jnp.asarray(ix)
        g_v = jnp.take(jnp.asarray(g_all), ix_d, axis=0)
        order = jnp.argsort(g_v, stable=True)
        ix_s = jnp.take(ix_d, order, axis=0)
        g_s = jnp.take(g_v, order, axis=0)
        return (np.asarray(jax.device_get(ix_s), dtype=np.int64),
                np.asarray(jax.device_get(g_s), dtype=np.float64))
    except Exception:                                # pragma: no cover
        # Any device-side refusal (OOM on the transient, no backend, an
        # unsupported dtype) falls back to the host, which is the same
        # permutation.  It is caught here rather than at the call site so
        # the planner never learns that the sort has two implementations.
        return None


def _dense_from_index(idx, shape):
    """The boolean an index set stands for -- built on purpose, one at a time."""
    shape = tuple(int(x) for x in shape)
    m = np.zeros(int(np.prod(shape)) if shape else 0, dtype=bool)
    m[np.asarray(idx, dtype=np.int64)] = True
    return m.reshape(shape)


_SECTOR_SELECTOR_SIZE = 6


def _validate_sector_selector_bounds(bounds):
    """Validate ``(a_gt,a_le,g_ge,g_gt,g_lt,g_le)`` scalar membership."""
    out = np.asarray(bounds, dtype=np.float64)
    if out.shape != (_SECTOR_SELECTOR_SIZE,) or np.isnan(out).any():
        raise ValueError(
            "MPA compact selector must be six binary64 bounds without NaN")
    if not out[0] < out[1] or not max(out[2], out[3], 0.0) < min(
            out[4], out[5]):
        raise ValueError("MPA compact selector describes an empty domain")
    return out


def _sector_selector_from_ag(a, gamma, bounds):
    b = bounds
    return ((a > b[0]) & (a <= b[1])
            & (gamma >= b[2]) & (gamma > b[3])
            & (gamma < b[4]) & (gamma <= b[5]))


def _sector_selector_host(omega_complex, bounds):
    omega = np.asarray(omega_complex)
    if omega.dtype.kind != "c":
        raise ValueError("MPA compact selector requires complex Omega")
    gamma = -np.imag(omega)
    live = np.real(omega) > 1.0e-14
    if np.any(gamma[live] < 0.0):
        raise ValueError("MPA pole field contains upper-half-plane poles")
    return _sector_selector_from_ag(
        np.real(omega), gamma, _validate_sector_selector_bounds(bounds))


_SECTOR_SELECTOR_FNS = {}


def _sector_selector_device(omega_complex, bounds, mesh_xy):
    """Evaluate one scalar pole selector at the pole field's sharding."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    pole_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    scalar_sharding = NamedSharding(mesh_xy, P())
    key = (pole_sharding, scalar_sharding)
    fn = _SECTOR_SELECTOR_FNS.get(key)
    if fn is None:
        fn = jax.jit(
            lambda omega, b: _sector_selector_from_ag(
                jnp.real(omega), -jnp.imag(omega), b),
            in_shardings=(pole_sharding, scalar_sharding),
            out_shardings=pole_sharding)
        _SECTOR_SELECTOR_FNS[key] = fn
    return fn(omega_complex, jax.device_put(
        _validate_sector_selector_bounds(bounds), scalar_sharding))


# ---------------------------------------------------------------------------
#  The group selector, on the device that consumes it
# ---------------------------------------------------------------------------
#
#  THE COST THIS REPLACES, WHICH IS A DISPATCH COST AND NOT A KERNEL ONE.
#  ``run_pass_branch`` used to hand ``_integrate_tau_windows_for_branch`` a
#  freshly built HOST boolean per group -- ``_dense_from_index`` allocates
#  the full ``(n_q, N_mu, N_mu)`` field (81.4 MB on the production deck),
#  scatters the group's index set into it, and ``jnp.asarray`` then copies
#  all 81.4 MB across PCIe.  918 groups over an n_p = 8 pass is 74 GB of
#  host materialisation and 74 GB of pageable H2D to describe a partition
#  whose whole index representation is 0.32 GB -- which is the memory the
#  compact pane index was built to stop spending, spent again one level
#  down at dispatch time.
#
#  The selector is the same boolean.  It is built where it is read, from
#  the index set the planner already produced, by one scatter whose entire
#  H2D payload is 4 bytes per live mode.
#
#  WHY THE INDEX IS PADDED, AND WHY TO A POWER OF TWO.  ``jax.jit`` keys on
#  shape, and a group's live-mode count is a property of the physics: 918
#  groups have ~918 distinct sizes, so an unpadded selector would compile
#  918 XLA modules per pass and populate a persistent cache with entries no
#  second run can hit.  Padding to the next power of two bounds the number
#  of distinct signatures at ``log2(n_flat)`` -- 27 on any field this code
#  can address, a handful in practice -- and the pad entries are the
#  out-of-range index ``n_flat``, which ``mode="drop"`` discards.  The
#  capacity ladder is a function of the GROUP's mode count and the FIELD's
#  size, both of which every rank computes identically from the same
#  replicated pole slab: no signature here depends on a process's rank or
#  on the device count, which is the multi_slice lesson stated as a
#  constraint rather than remembered as an incident.
#
#  It carries no host callback, so a persistent compile cache serves it on
#  the second run (FIX_warmcache.md's sink pattern: values in, values out,
#  nothing traced that reaches the host).

def _selector_capacity(n_live):
    """The padded index length a group of ``n_live`` modes compiles at."""
    n = int(n_live)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _make_selector_fn():
    """``(idx_padded, n_flat) -> flat bool selector``, jitted, cached.

    A module-level ``jax.jit`` object rather than a fresh one per call, so
    the trace cache is shared by every group of every branch of every pole
    in a process -- the second group of a given capacity is a cache hit and
    the second RUN is a persistent-cache hit.
    """
    import jax
    import jax.numpy as jnp

    global _SELECTOR_FN
    if _SELECTOR_FN is None:
        def _scatter(idx_padded, n_flat):
            return (jnp.zeros((n_flat,), dtype=bool)
                    .at[idx_padded].set(True, mode="drop"))

        _SELECTOR_FN = jax.jit(_scatter, static_argnums=(1,))
    return _SELECTOR_FN


_SELECTOR_FN = None


def group_selector_device(idx_B, field_shape):
    """The dense selector ``idx_B`` stands for, built on the device.

    Bit-for-bit :meth:`WindowGroup.dense_mask_B` -- a boolean that is
    ``True`` exactly at the group's flat indices -- and the equality is
    exact rather than approximate because a scatter of ``True`` into zeros
    has no arithmetic in it.  ``tests/test_mpa_jax_native.py`` holds the
    two against each other on a field with duplicate-free ascending
    indices, an empty group and a full one.
    """
    import jax.numpy as jnp

    shape = tuple(int(x) for x in field_shape)
    n_flat = int(np.prod(shape)) if shape else 0
    idx = np.asarray(idx_B, dtype=np.int64)
    cap = _selector_capacity(idx.size)
    if idx.size < cap:
        # The pad is the one index that cannot be a mode: ``n_flat``.
        pad = np.full((cap - idx.size,), n_flat, dtype=np.int64)
        idx = np.concatenate([idx, pad])
    flat = _make_selector_fn()(jnp.asarray(idx), n_flat)
    return flat.reshape(shape)


class _BranchOperandCache:
    """One named-sharded device copy of each pole ``omega_operand``.

    THE OPERAND IS A FIELD, NOT A GROUP PROPERTY.  Every MPA-routed group
    of a pass shares one ``Omega_p = a - i*Gamma`` array and every
    legacy-routed group shares one ``a``; the planner hands each group a
    REFERENCE to the same object.  Uploading per group re-copied 1.3 GB of
    complex128 (or 0.65 GB of float64) across PCIe once per group -- 918
    times on an n_p = 8 pass, to put the identical bytes on the identical
    device.  Keyed by object identity, which is exact here: the planner's
    arrays are alive for the whole loop because the groups hold them, so
    no id can be recycled underneath this cache.

    The cache is created once per pole and shared by all four branches.
    Production placement is ``P(None, 'x', 'y')`` through
    ``device_put_process_local``: every rank read the identical global pole
    slab, so no equality all-gather is required.  ``mesh_xy=None`` keeps the
    small standalone planning cells on their ordinary local JAX device.
    """

    def __init__(self, mesh_xy=None):
        self._by_id = {}
        if mesh_xy is None:
            self._sharding = None
        else:
            from jax.sharding import NamedSharding, PartitionSpec as P

            self._sharding = NamedSharding(mesh_xy, P(None, "x", "y"))

    def seed(self, host_array, device_array):
        """Record an array already placed by the pole-state preparation."""
        self._by_id[id(host_array)] = (host_array, device_array)

    def device(self, host_array):
        import jax.numpy as jnp

        key = id(host_array)
        hit = self._by_id.get(key)
        if hit is None:
            # The host array is kept alive by the cache entry itself, which
            # is what makes ``id`` a legal key for the loop's lifetime.
            if self._sharding is None:
                device_array = jnp.asarray(host_array)
            else:
                from common.collectives import device_put_process_local

                device_array = device_put_process_local(
                    host_array, self._sharding)
            hit = (host_array, device_array)
            self._by_id[key] = hit
        return hit[1]


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
    binned_width_clause=None,
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

    # THE BINNED CLAUSE IS FOR THE NON-CROSSING BRANCHES, AND ONLY THEM.
    # A branch that can cross floors every Laplace window's ``x_min`` at
    # ``z_edge = edge_factor * Gamma_hi``, so ``beta <= 1/edge_factor``
    # holds STRUCTURALLY there -- the audit found zero of 81 million
    # poles outside the per-pole clause on those branches -- and the
    # width clause cannot fire, so there is nothing for a wider clause to
    # buy.  Their panes are already binned, at ``CROSSING_WIDTH_RATIO_MAX``,
    # for an unrelated reason (bounding the crossing core's bandwidth
    # ``A = f_max/Gamma_lo``), and that ratio is NOT the flag: fetching a
    # band certificate at the flag's ``r`` for a pane binned at the
    # crossing constant is exactly the mis-binning the guard refuses, and
    # it did, on the first run of this branch's digest tool at ``r = 2``.
    # So the crossing branches keep the per-pole clause whatever the flag
    # says, which also makes them byte-identical with the flag ON.
    _binned = (None if R.denominator_can_cross(space, bool(neg_omega_half))
               else binned_width_clause)

    def _clause(x_min, x_max, g_hi_pane, g_lo_pane):
        """The width clause this window owes, in the form it is checked.

        ``None`` when the binned clause is off or the branch is a
        crossing one, which is the shipped path and the byte-identical
        one.  Otherwise the tuple ``_refuse_width_clause`` needs to fetch
        a band certificate: the ratio, the PANE's own width range (not
        the window's ``g_hi`` alone -- the derivation needs both ends),
        and the request the entry has to cover, whose ``R`` is this
        window's own interval ratio because that is what the catalog
        rescales by.
        """
        if _binned is None:
            return None
        return (float(_binned), float(g_lo_pane),
                float(g_hi_pane), float(x_max) / float(x_min),
                float(rel_tol), int(max_nodes))

    g_lo_pane = float(np.min(g_v))

    if not R.denominator_can_cross(space, bool(neg_omega_half)):
        a_lo, a_hi = float(np.min(a_v)), float(np.max(a_v))
        g_hi = float(np.max(g_v))
        x_min = max(e_lo + a_lo, _X_FLOOR_RY)
        x_max = max(e_hi + a_hi + omega_max, x_min * (1.0 + 1.0e-9))
        t, alpha, rule = R.sign_definite_rule(
            x_min, x_max, g_hi, rel_tol=rel_tol, max_nodes=max_nodes)
        beta = R.beta_for_window(g_hi, x_min)
        _refuse_width_clause(beta, "single", x_min, g_hi,
                             binned=_clause(x_min, x_max, g_hi, g_lo_pane))
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
            _refuse_width_clause(beta, "a_stripe", x_min, g_hi,
                                 binned=_clause(x_min, x_max, g_hi,
                                                g_lo_pane))
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
        _refuse_width_clause(beta, "b_slab", x_min, g_hi,
                             binned=_clause(x_min, x_max, g_hi, g_lo_pane))
        groups.append((("b_slab",), [_sigma_window(
            name="b_slab", plan_t=t, plan_alpha=alpha,
            mask_A=base_mask_A_host, e_ref_a=e_lo, e_ref_b=a_lo,
            omega_sign=+1, project="full", prefactor=+1.0 * neg,
            max_error=rule["rel_tol"],
            provenance=(f"MPA sign-definite composite, {rule['n_panels']} "
                        f"panels, R={x_max / x_min:.3g}, beta={beta:.4f}, "
                        f"kappa0={rule['kappa0']:.6f}"))], slab_idx))
    return groups


def _refuse_width_clause(beta, where, x_min, gamma, *, binned=None):
    """The width clause, checked where the window is built rather than later.

    The rule this module builds is the positive composite one, certified by
    its own ``kappa0 <= 1`` and not by a catalog entry -- so ``beta`` does
    not gate the build.  It is checked anyway, because the audited envelope
    (``beta <= 1`` for any field this fitter can produce, the fitter's own
    fourth guard) is the statement that the slab's conservative statistics
    have not walked outside the pole field the routing was scored on.  A
    slab whose ``beta`` exceeds it is a bucketing failure, not a physics
    one, and it should say so here rather than 4000 tau nodes later.

    ``binned`` ARMS THE SECOND CLAUSE, and when it is None -- the default,
    and the flag's default -- not one line below the first ``return``
    executes and this function is character for character the one that
    shipped.  When it is present it is
    ``(r, g_lo, g_hi, range_value, target_error, max_nodes)``: the pane's
    bin ratio and width range, and the request the band certificate has to
    cover.  Three things then have to be true and each is checked here
    rather than assumed:

    1. the pane really is binned at ``r`` (``_refuse_mis_binned_pane``) --
       the derivation ``beta <= r`` has no other support;
    2. ``beta <= r``, inclusive, which is that derivation's conclusion and
       the catalog clause's edge;
    3. a certified entry EXISTS for ``(R, r, tier)``.  This is the
       lookup-and-refuse discipline arriving on the Sigma path: the pane
       is allowed its wider clause only where something certified the band
       it spans, and a request one hair outside the certified grid refuses
       BY NAME with the catalog's own message rather than being served by
       a narrower entry.
    """
    from . import sigma_routing as R

    if binned is None:
        if float(beta) <= R.SHIPPED_WIDTH_BETA_MAX + 1.0e-12:
            return
        raise R.RoutingRefusal(
            f"MPA {where} window: the slab's width clause is beta = "
            f"Gamma_max/x_min = {float(beta):.6f}, above the envelope "
            f"beta_max={R.SHIPPED_WIDTH_BETA_MAX} that the fitter's own "
            f"fourth guard closes at (x_min={float(x_min):.6g} Ry, "
            f"Gamma={float(gamma):.6g} Ry).  For a SINGLE pole this is "
            f"structurally impossible; for a slab it means the bucket "
            f"mixes a shallow Laplace edge with a wide pole.  Narrow the "
            f"buckets (lower mpa_laplace_ratio_max) rather than widening "
            f"the clause.",
            code="slab_width_clause", beta=float(beta))

    r, g_lo, g_hi, range_value, target_error, max_nodes = binned
    _refuse_mis_binned_pane(g_lo, g_hi, r, where=where, beta=beta)
    if float(beta) > float(r) * (1.0 + 1.0e-12):
        raise R.RoutingRefusal(
            f"MPA {where} window: the binned-width clause is beta = "
            f"Gamma_hi/x_min = {float(beta):.6f}, above its own edge "
            f"r={float(r):g} (x_min={float(x_min):.6g} Ry, "
            f"Gamma={float(gamma):.6g} Ry).  The bin ratio IS the clause "
            f"edge: pade_fit's fourth guard gives Gamma_p <= a_p, so "
            f"x_min = min(E_A) + a_lo >= Gamma_lo and beta <= "
            f"Gamma_hi/Gamma_lo <= r.  A beta above r with the pane "
            f"correctly binned means x_min is NOT min(E_A) + a_lo on this "
            f"window -- which is the crossing branches' z_edge floor, "
            f"where the clause is 1/edge_factor and tighter still.",
            code="binned_width_clause", beta=float(beta))
    got = R.binned_width_entry(
        range_value=range_value, beta=float(beta), bin_ratio=float(r),
        target_error=float(target_error), max_nodes=int(max_nodes))
    if getattr(got, "code", None) is None:
        return got
    raise R.RoutingRefusal(
        f"MPA {where} window: the binned-width clause was asked for at "
        f"r={float(r):g}, R={float(range_value):.6g}, tier "
        f"{float(target_error):.0e}, beta={float(beta):.6f}, and the "
        f"certified catalog refused it.\n"
        f"{got.message}\n"
        f"  the planner emits a width-BINNED pane only where an entry "
        f"certifies the band that pane spans, so this refusal is the "
        f"clause working rather than a table being missing by accident.  "
        f"Set the binned_width_clause flag off to fall back to the "
        f"per-pole width clause, which costs panes and refuses nothing.",
        code="binned_width_no_entry", beta=float(beta))


def _plan_sector_branch_groups(
    *, a, g, live_mask, E_A_host, base_mask_A_host, omega_nonneg_ry,
    omega_max, omega_complex,
    space, neg_omega_half, b_abs, xi_ry, edge_factor, rel_tol, max_nodes,
    target_error, laplace_max_nodes, crossing_eps_q, crossing_max_nodes,
    use_shipped_minimax_tables, log_tag, print_fn,
):
    """Fixed GN geometry with sector rules on every sign-definite piece.

    The pole field stays coupled as ``(Re Omega_p, Gamma_p)`` when radial
    bounds are formed.  Widths therefore choose the scalar integrand but never
    create a pane.  Only a genuinely crossing core is divided into geometric
    width bands, which retain each pole's exact ``exp(-Gamma_p*t)`` damping.
    """
    from . import sigma_routing as R

    field_shape = tuple(int(x) for x in np.shape(a))
    ix_dtype = flat_index_dtype(int(np.prod(field_shape)))
    a_flat = np.ravel(np.asarray(a, dtype=np.float64))
    g_flat = np.ravel(np.asarray(g, dtype=np.float64))
    narrow, wide = split_pass_by_width(g, live_mask, xi_ry)
    idx_live = np.flatnonzero(np.ravel(wide)).astype(ix_dtype)
    idx_all = np.flatnonzero(
        np.ravel(np.asarray(live_mask, dtype=bool))).astype(ix_dtype)
    mass_flat = (None if b_abs is None
                 else np.ravel(np.abs(np.asarray(b_abs, dtype=np.float64))))
    e_all_lo, e_all_hi = _stats(E_A_host, base_mask_A_host)
    neg = -1.0 if neg_omega_half else 1.0
    groups = []

    def _mass(idx):
        return 0.0 if mass_flat is None else float(np.sum(mass_flat[idx]))

    def _group(name, idx, windows):
        idx = np.sort(np.asarray(idx, dtype=ix_dtype))
        groups.append(WindowGroup(
            name=name, windows=windows, idx_B=idx, field_shape=field_shape,
            omega_operand=omega_complex, n_modes=int(idx.size),
            b_mass=_mass(idx),
            provenance="; ".join(w.provenance for w in windows)))

    def _sector_window(name, idx, mask_A, e_lo, e_hi, *, omega_sign,
                       prefactor):
        av, gv = a_flat[idx], g_flat[idx]
        if omega_sign < 0:
            x_lo = e_lo + av
            x_hi = e_hi + av + omega_max
        else:
            x_lo = e_lo + av - omega_max
            x_hi = e_hi + av
        radial_min = float(np.min(np.hypot(x_lo, gv)))
        radial_max = float(np.max(np.hypot(x_hi, gv)))
        t, alpha, rule = R.sector_sign_definite_rule(
            radial_min, radial_max, rel_tol=rel_tol, max_nodes=max_nodes)
        return _sigma_window(
            name=name, plan_t=t, plan_alpha=alpha, mask_A=mask_A,
            e_ref_a=e_lo, e_ref_b=float(np.min(av)),
            omega_sign=omega_sign, project="full", prefactor=prefactor,
            max_error=rule["error_bound"],
            provenance=(f"MPA pi/4 sector sinc, {rule['n_nodes']} nodes, "
                        f"|d|={radial_min:.4g}..{radial_max:.4g} Ry, "
                        f"R={rule['radial_ratio']:.3g}, "
                        f"bound={rule['error_bound']:.2e}, "
                        f"kappa0={rule['kappa0']:.6f}"))

    # Speed-first compatibility bridge: preserve the accepted Gamma<xi
    # substitution exactly while removing the 44,842-node sign-definite pane
    # explosion.  These modes cost only 0.9% of the audited nodes.  The strict
    # fitted-width crossing limit remains a separate, explicitly scored
    # physics change rather than being smuggled into a performance A/B.
    if narrow.any():
        from ..ppm_windows import _build_windows_for_branch
        import jax.numpy as jnp

        wins = _build_windows_for_branch(
            omega_nonneg_ry=np.asarray(
                omega_nonneg_ry, dtype=np.float64),
            E_A=np.asarray(E_A_host, dtype=np.float64),
            base_mask_A=np.asarray(base_mask_A_host, dtype=bool),
            space=space, neg_omega_half=bool(neg_omega_half),
            Omega_q=jnp.asarray(a, dtype=jnp.float64),
            base_mask_B=jnp.asarray(narrow, dtype=bool),
            regularization_width_ry=float(xi_ry),
            edge_factor=float(edge_factor), target_error=float(target_error),
            max_nodes=int(laplace_max_nodes),
            crossing_eps_q=float(crossing_eps_q),
            crossing_max_nodes=int(crossing_max_nodes),
            use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
            log_tag=f"{log_tag} legacy-routed", print_fn=print_fn)
        if wins:
            idx = np.flatnonzero(np.ravel(narrow)).astype(ix_dtype)
            groups.append(WindowGroup(
                name="legacy", windows=wins, idx_B=idx,
                field_shape=field_shape, omega_operand=np.asarray(a),
                n_modes=int(idx.size), b_mass=_mass(idx),
                provenance=("accepted two-point Gamma<xi substitution kept "
                            "fixed for the speed-first sector A/B")))

    if idx_live.size == 0:
        return groups, {
            "n_narrow": int(np.count_nonzero(narrow)), "n_wide": 0,
            "narrow_b_mass": _mass(idx_all), "wide_b_mass": 0.0,
            "xi_ev": float(xi_ry) * RYD_TO_EV,
            "n_tau": int(sum(w.n_tau for q in groups for w in q.windows)),
            "n_panes": int(len(groups)), "binned_width_clause": None,
            "windowing": "sector",
        }

    crossing = R.denominator_can_cross(space, bool(neg_omega_half))
    if not crossing:
        win = _sector_window(
            "single", idx_live, np.asarray(base_mask_A_host, dtype=bool),
            e_all_lo, e_all_hi, omega_sign=-1, prefactor=-1.0 * neg)
        _group("sector:single", idx_live, [win])
    else:
        T = float(omega_max)
        shallow = idx_live[a_flat[idx_live] <= T]
        deep = idx_live[a_flat[idx_live] > T]
        core_A = (np.asarray(base_mask_A_host, dtype=bool)
                  & (np.asarray(E_A_host) <= T))
        stripe_A = (np.asarray(base_mask_A_host, dtype=bool)
                    & (np.asarray(E_A_host) > T))

        if shallow.size and core_A.any():
            core_e_lo, _ = _stats(E_A_host, core_A)
            sorted_idx, sorted_g = _sorted_by_width(g_flat, shallow)
            for band, idx in enumerate(_geometric_width_bins_sorted(
                    sorted_g, sorted_idx)):
                av, gv = a_flat[idx], g_flat[idx]
                f_max = (omega_max + max(T - core_e_lo, 0.0)
                         + float(np.max(av) - np.min(av)))
                t, alpha, rule = R.crossing_rule(
                    float(np.min(gv)), f_max, rel_tol=rel_tol,
                    max_nodes=max_nodes)
                win = _sigma_window(
                    name="core", plan_t=t, plan_alpha=alpha,
                    mask_A=core_A, e_ref_a=core_e_lo,
                    e_ref_b=float(np.min(av)), omega_sign=+1,
                    project="full", prefactor=-1.0 * neg,
                    max_error=rule["rel_tol"],
                    provenance=(f"MPA exact-width crossing core, "
                                f"{rule['n_nodes']} nodes, "
                                f"Gamma={np.min(gv):.4g}.."
                                f"{np.max(gv):.4g} Ry, "
                                f"A={rule['a_dim']:.4g}"))
                _group(f"sector:core:g{band}", idx, [win])

        if shallow.size and stripe_A.any():
            stripe_lo, stripe_hi = _stats(E_A_host, stripe_A)
            win = _sector_window(
                "a_stripe", shallow, stripe_A, stripe_lo, stripe_hi,
                omega_sign=+1, prefactor=+1.0 * neg)
            _group("sector:a_stripe", shallow, [win])

        if deep.size:
            win = _sector_window(
                "b_slab", deep, np.asarray(base_mask_A_host, dtype=bool),
                e_all_lo, e_all_hi, omega_sign=+1,
                prefactor=+1.0 * neg)
            _group("sector:b_slab", deep, [win])

    live_mass = _mass(idx_live)
    narrow_idx = np.flatnonzero(np.ravel(narrow)).astype(ix_dtype)
    return groups, {
        "n_narrow": int(narrow_idx.size),
        "n_wide": int(idx_live.size),
        "narrow_b_mass": _mass(narrow_idx),
        "wide_b_mass": live_mass,
        "xi_ev": float(xi_ry) * RYD_TO_EV,
        "n_tau": int(sum(w.n_tau for grp in groups for w in grp.windows)),
        "n_panes": int(len(groups)),
        "binned_width_clause": None,
        "windowing": "sector",
    }


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
    binned_width_clause=None,
    target_error=1.0e-6,
    laplace_max_nodes=64,
    crossing_eps_q=1.0e-10,
    crossing_max_nodes=200,
    use_shipped_minimax_tables=True,
    windowing="pane", omega_complex=None,
    log_tag="",
    print_fn=print,
):
    """Every window group one pass owes on one branch.

    Returns ``(groups, stats)``.  ``groups`` is a list of
    :class:`WindowGroup`; ``stats`` records the narrow/wide split so the
    pass announcement can name it.

    ``binned_width_clause`` is the flag, and it DEFAULTS OFF.  ``None``
    means the shipped per-pole width clause: the non-crossing branches
    bisect each Laplace bucket in width until every leaf's ``beta`` fits
    under 1, which is what produces 218 panes on a typical branch of the
    audited field and 1312 on pole 7, and a pane is a certified rule and
    a tau-node run of its own.  A float ``r`` (2 or 4, the ratios the
    catalog is qualified at) means the BINNED clause instead: bin the
    widths geometrically at ratio ``r`` and emit one pane per (window,
    bin), with the clause edge at ``r`` rather than 1 -- but only where a
    band certificate exists for the request, which
    ``_refuse_width_clause`` fetches per window and refuses by name
    without.  Nothing else moves: the mandatory-refit guard, all four fit
    guards, the Laplace bucketing, the crossing branches' own width
    binning and every rule builder are untouched, and with the flag off
    this function's plan is byte-identical to the one that shipped.

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
    if windowing == "sector":
        omega_complex = (a - 1j * g if omega_complex is None
                         else np.asarray(omega_complex,
                                         dtype=np.complex128))
        return _plan_sector_branch_groups(
            a=a, g=g, live_mask=live_mask, E_A_host=E_A_host,
            base_mask_A_host=base_mask_A_host,
            omega_nonneg_ry=omega_nonneg_ry, omega_max=omega_max,
            omega_complex=omega_complex,
            space=space, neg_omega_half=neg_omega_half, b_abs=b_abs,
            xi_ry=xi_ry, edge_factor=edge_factor, rel_tol=rel_tol,
            max_nodes=max_nodes, target_error=target_error,
            laplace_max_nodes=laplace_max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
            use_shipped_minimax_tables=use_shipped_minimax_tables,
            log_tag=log_tag, print_fn=print_fn)
    if windowing != "pane":
        raise ValueError(
            f"plan_branch_groups: windowing={windowing!r}; expected 'pane' "
            "or 'sector'.")
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

    omega_complex = (a - 1j * g if omega_complex is None
                     else np.asarray(omega_complex, dtype=np.complex128))
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
    # The A-side edges are a property of the BRANCH, not of a bucket: they
    # were being recomputed three times per bucket off arguments no loop
    # iteration touches.  Cheap either way (E_A is nk x nb), hoisted
    # because a loop-invariant read of loop-invariant data is exactly the
    # kind of thing that stops being cheap when a field grows.
    e_lo_A, e_hi_A = _stats(E_A_host, base_mask_A_host)
    for a_lo, a_hi in _laplace_buckets(
            wide_a, e_lo=e_lo_A, e_hi=e_hi_A,
            omega_max=omega_max, r_max=float(laplace_ratio_max)):
        in_bucket = (wide_a >= a_lo) & (wide_a <= a_hi)
        if not in_bucket.any():
            continue
        bucket_idx = wide_idx[in_bucket]
        bucket_g = wide_g[in_bucket]
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
        elif binned_width_clause is not None:
            # THE BINNED CLAUSE, ON.  The same direct binning the crossing
            # branches have always used, at the flag's ratio, in place of
            # the clause-driven bisection -- so the pane count becomes
            # ceil(log_r(Gamma_hi/Gamma_lo)) by construction rather than
            # whatever satisfying beta <= 1 costs.  It cannot diverge for
            # the same reason the crossing one cannot: there is no
            # predicate to chase, only a ratio to respect, and the leaf
            # ceiling below is left in place as the backstop it always
            # was rather than removed because this arm makes it quiet.
            b_idx, b_g = _sorted_by_width(g_flat, bucket_idx)
            subs = _geometric_width_bins_sorted(
                b_g, b_idx, r_max=float(binned_width_clause))
            _refuse_width_split_explosion(len(subs), bucket_g)
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
                    max_nodes=max_nodes,
                    binned_width_clause=binned_width_clause):
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
        # The two numbers the owner's ruling is made on, per branch, so a
        # run announces its own pane collapse instead of it having to be
        # counted out of a log afterwards.
        "n_panes": int(len(groups)),
        "binned_width_clause": (None if binned_width_clause is None
                                else float(binned_width_clause)),
        "windowing": "pane",
    }
    return groups, stats


def format_pass_report(records, *, xi_ev, windowing="pane"):
    """The per-pass announcement, including the legacy-routed count.

    The legacy count is printed whether it is zero or not.  A run in which
    it is zero is a run whose every pole was resolved by the complex route,
    and that is worth reading off the log rather than inferring from the
    absence of a line.
    """
    lines = [
        "",
        "-" * 72,
        (f"MPA Sigma: {len(records)} pole passes, windowing={windowing}"
         + (f", narrow-pole threshold xi = {float(xi_ev):.4f} eV"
            if windowing == "pane" else
            ", fixed sign-definite sectors; accepted Gamma<xi crossing "
            "substitution retained for the speed/BGW-parity milestone")),
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
        (f"  legacy-routed modes: {tot_legacy} of {total} ({frac:.2f} %) "
         + ("-- poles with Gamma < xi, summed by the two-point crossing "
            "machinery at that xi" if windowing == "pane" else
            "-- sector windowing retains this accepted compatibility route "
            "only for the narrow modes")),
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
    operand_cache=None,
    selector_omega=None,
):
    """Integrate one pass's window groups on one branch; return host tiles.

    Mirrors ``ppm_sigma._run_sigma_branch``'s tail exactly, and calls its
    ``_integrate_tau_windows_for_branch`` once per group -- the group loop
    is the only structural difference, and it exists because a group is
    what carries an explicit ``mask_B`` and its own ``Omega_q`` operand
    (real for the legacy-routed poles, complex for the rest).

    WHERE THE GROUP'S TWO OPERANDS COME FROM, AND WHY IT IS NOT THE HOST.
    Both are now built on the device that reads them: the selector by
    :func:`group_selector_device` (a jitted scatter of the group's index
    set) and the pole operand by :class:`_BranchOperandCache` (one upload
    per distinct field, not one per group).  The call below is otherwise
    the call that was here before -- same keywords, same dtypes, same
    order, same integrator -- because the correctness argument for this
    module is that a window group reaches ``ppm_sigma``'s tau loop
    UNCHANGED, and a performance refactor that moved that seam would owe
    the shared-kernel digest gate a re-anchor.  It does not move it: the
    bytes in both operands are what the host path produced, and
    ``tests/test_mpa_jax_native.py`` holds them against it.
    """
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common import timing
    from ..ppm_accumulators import _DeviceTauAccumulator
    from ..ppm_sigma import _SigmaBranchTiles, _integrate_tau_windows_for_branch

    n_omega = int(np.asarray(omega_nonneg_ry).shape[0])
    if n_omega == 0 or not groups:
        return None

    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, nk_proj, nb_proj = psi
    m_pad = int(psi_proj_xr.shape[1])
    n_pad = int(psi_proj_yn.shape[3])
    output_sharding = NamedSharding(mesh_xy, P(None, None, 'x', 'y'))
    accumulator = _DeviceTauAccumulator(
        omega_vec=jnp.asarray(omega_nonneg_ry, dtype=jnp.float64),
        shape=(n_omega, nk_proj, m_pad, n_pad), sharding=output_sharding)

    tau_kernel, tau_kernel_x = tau_kernels
    operands = (_BranchOperandCache(mesh_xy) if operand_cache is None
                else operand_cache)
    for grp in groups:
        # ONE selector resident, and only while its group is integrating.
        # The group carries an index set; the tau loop wants the boolean
        # that index set stands for, so it is built here -- on the device,
        # from the index -- and dropped when the loop returns.
        with timing.section("mpa.group_operands"):
            omega_q_dev = operands.device(grp.omega_operand)
            mask_b_dev = (
                _sector_selector_device(
                    operands.device(selector_omega),
                    grp.selector_bounds_B, mesh_xy)
                if grp.selector_bounds_B is not None else
                group_selector_device(grp.idx_B, grp.field_shape))
        _integrate_tau_windows_for_branch(
            windows=grp.windows, accumulator=accumulator, E_A=E_A,
            B_q=B_p, Omega_q=omega_q_dev,
            base_mask_B=mask_b_dev,
            psi_coh_xn=psi_coh_xn, psi_coh_yr=psi_coh_yr,
            psi_proj_xr=psi_proj_xr, psi_proj_yn=psi_proj_yn,
            tau_kernel=tau_kernel, tau_kernel_x=tau_kernel_x,
            log_tag=f"{log_tag} {grp.name}", print_fn=print_fn)
        del omega_q_dev, mask_b_dev

    with timing.section("sigma.finalize"):
        tiles, tile_index, tile_devices = accumulator.finalize_host_tiles()
    return _SigmaBranchTiles(
        tiles=tiles, tile_index=tile_index, devices=tile_devices,
        spatial_padded=(nk_proj, m_pad, n_pad),
        sharding=output_sharding,
        nb_real=nb_proj)


def _extract_shared_crossing(groups, a_real, omega_complex):
    """Remove coalescible exact/HGL cores from one loaded branch plan."""
    exact, hgl, kept = [], [], []
    for group in groups:
        if (group.selector_bounds_B is not None
                and str(group.name).startswith("sector:core:")
                and len(group.windows) == 1
                and group.windows[0].project == "full"):
            exact.append(group)
            continue
        core = [window for window in group.windows
                if window.crossing_kind == "hgl"]
        if core:
            if len(core) != 1 or group.selector_bounds_B is None:
                raise ValueError("shared HGL requires one compact core window")
            window = core[0]
            bounds = _validate_sector_selector_bounds(
                group.selector_bounds_B).copy()
            if (window.project != "imag" or window.mask_B_mode != "le_t"
                    or window.mask_B_threshold is None
                    or not np.isfinite(float(window.mask_B_threshold))
                    or group.omega_operand is not a_real):
                raise ValueError(
                    "shared HGL core does not carry its real-pole a<=T "
                    "physics")
            bounds[1] = min(bounds[1], float(window.mask_B_threshold))
            hgl.append((window, bounds))
            side = [item for item in group.windows if item is not window]
            if side:
                kept.append(replace(group, windows=side))
            continue
        kept.append(group)

    if exact:
        if len(exact) != 2:
            raise ValueError("shared exact crossing requires two Gamma bands")
        first = exact[0].windows[0]
        bounds = [_validate_sector_selector_bounds(g.selector_bounds_B)
                  for g in exact]
        candidate = bounds[0].copy()
        if (not np.array_equal(bounds[1][:3], candidate[:3])
                or not np.isneginf(bounds[0][3])
                or not np.isposinf(bounds[0][4])
                or not np.isposinf(bounds[1][4])
                or not np.isposinf(bounds[1][5])
                or bounds[0][5] != bounds[1][3]):
            raise ValueError("exact crossing Gamma bands are not contiguous")
        candidate[3] = -np.inf
        candidate[4] = np.inf
        candidate[5] = max(float(b[5]) for b in bounds)
        for group in exact:
            window = group.windows[0]
            same = (
                np.array_equal(window.nodes.t, first.nodes.t)
                and np.array_equal(window.nodes.alpha, first.nodes.alpha)
                and np.array_equal(window.mask_A, first.mask_A)
                and float(window.E_ref_A) == float(first.E_ref_A)
                and int(window.omega_sign) == int(first.omega_sign)
                and window.project == first.project == "full"
                and float(window.prefactor) == float(first.prefactor)
                and window.mask_B_mode == first.mask_B_mode == "all"
                and window.mask_B_threshold is None
                and first.mask_B_threshold is None
                and window.crossing_kind == first.crossing_kind is None
                and group.omega_operand is omega_complex)
            if not same:
                raise ValueError(
                    "exact crossing bands differ in non-gauge physics")
        exact = [(first, candidate)]

    if exact and hgl:
        if len(hgl) != 1:
            raise ValueError("shared crossing requires one HGL core")
        exact_bounds = exact[0][1]
        hgl_bounds = hgl[0][1]
        xi = float(exact_bounds[2])
        if (not np.isfinite(xi) or xi <= 0.0
                or not np.isneginf(exact_bounds[3])
                or not np.isposinf(exact_bounds[4])
                or not np.isposinf(exact_bounds[5])
                or not np.isneginf(hgl_bounds[2])
                or not np.isneginf(hgl_bounds[3])
                or float(hgl_bounds[4]) != xi
                or not np.isposinf(hgl_bounds[5])):
            raise ValueError(
                "exact and HGL crossing selectors do not meet at xi")
    return kept, exact, hgl


def _common_shared_window(pieces, kind):
    """Validate p0/p1 common-rule physics and return its pole selectors."""
    if len(pieces) != 2 or [piece[0] for piece in pieces] != [0, 1]:
        raise ValueError(f"shared {kind} requires poles 0 and 1")
    first = pieces[0][1]
    for _pole, window, _bounds in pieces[1:]:
        same = (
            np.array_equal(window.nodes.t, first.nodes.t)
            and np.array_equal(window.nodes.alpha, first.nodes.alpha)
            and np.array_equal(window.mask_A, first.mask_A)
            and float(window.E_ref_A) == float(first.E_ref_A)
            and int(window.omega_sign) == int(first.omega_sign)
            and str(window.project) == str(first.project)
            and float(window.prefactor) == float(first.prefactor))
        if not same:
            raise ValueError(f"shared {kind} windows differ across poles")
    return replace(first, E_ref_B=0.0), np.stack(
        [piece[2] for piece in pieces])


def _selector_bounds_overlap(left, right):
    """Whether two validated compact selector rectangles intersect."""
    left = _validate_sector_selector_bounds(left)
    right = _validate_sector_selector_bounds(right)
    if max(left[0], right[0]) >= min(left[1], right[1]):
        return False
    gamma_lo = max(0.0, left[2], left[3], right[2], right[3])
    gamma_hi = min(left[4], left[5], right[4], right[5])
    if gamma_lo != gamma_hi:
        return gamma_lo < gamma_hi
    return all(
        gamma_lo >= bounds[2] and gamma_lo > bounds[3]
        and gamma_lo < bounds[4] and gamma_lo <= bounds[5]
        for bounds in (left, right))


def _extract_shared_sides(groups, a_real, omega_complex):
    """Remove real-pole compatibility windows and refine their selectors."""
    del omega_complex
    kept, sides, side_bounds = [], [], []
    for group in groups:
        candidates = [
            window for window in group.windows
            if window.project == "full" and window.crossing_kind is None
        ]
        if not candidates or group.selector_bounds_B is None:
            kept.append(group)
            continue
        if group.omega_operand is not a_real:
            if np.asarray(group.omega_operand).dtype.kind != "f":
                kept.append(group)
                continue
            raise ValueError(
                "shared compatibility side does not carry its real pole")
        base = _validate_sector_selector_bounds(
            group.selector_bounds_B).copy()
        remaining = [
            window for window in group.windows
            if not (window.project == "full"
                    and window.crossing_kind is None)
        ]
        group_bounds = []
        for window in candidates:
            bounds = base.copy()
            if window.mask_B_mode == "le_t":
                if (window.mask_B_threshold is None
                        or not np.isfinite(float(window.mask_B_threshold))):
                    raise ValueError(
                        "shared compatibility stripe needs a finite threshold")
                bounds[1] = min(bounds[1],
                                float(window.mask_B_threshold))
            elif window.mask_B_mode == "gt_t":
                if (window.mask_B_threshold is None
                        or not np.isfinite(float(window.mask_B_threshold))):
                    raise ValueError(
                        "shared compatibility slab needs a finite threshold")
                bounds[0] = max(bounds[0],
                                float(window.mask_B_threshold))
            elif (window.mask_B_mode != "all"
                  or window.mask_B_threshold is not None):
                raise ValueError(
                    "shared compatibility side has an unsupported selector")
            bounds = _validate_sector_selector_bounds(bounds)
            if any(_selector_bounds_overlap(bounds, previous)
                   for previous in side_bounds):
                raise ValueError("compatibility side selectors overlap")
            side_bounds.append(bounds)
            group_bounds.append(bounds)
            sides.append((window, bounds))
        ordered = sorted(group_bounds, key=lambda item: float(item[0]))
        covers = (
            len(ordered) == 1
            or (float(ordered[0][0]) == float(base[0])
                and float(ordered[-1][1]) == float(base[1])
                and all(float(left[1]) == float(right[0])
                        for left, right in zip(ordered, ordered[1:]))))
        if not covers:
            raise ValueError(
                "compatibility side selectors do not cover their group")
        if remaining:
            kept.append(replace(group, windows=remaining))
    return kept, sides


def _extract_shared_noncross(groups, omega_complex):
    """Remove the six ordinary finite-width classes eligible for pole sharing."""
    names = {"sector:single", "sector:a_stripe", "sector:b_slab"}
    kept, shared = [], []
    for group in groups:
        if (str(group.name) not in names or len(group.windows) != 1
                or not str(group.windows[0].provenance or "").startswith(
                    "NON-PRODUCTION shared noncross frontier;")):
            kept.append(group)
            continue
        if (group.selector_bounds_B is None
                or group.omega_operand is not omega_complex):
            raise ValueError(
                f"shared noncross group {group.name!r} does not carry its "
                "compact finite-width pole physics")
        window = group.windows[0]
        piece = str(group.name).partition(":")[2]
        if (str(window.name) != piece or window.project != "full"
                or window.crossing_kind is not None
                or window.mask_B_mode != "all"
                or window.mask_B_threshold is not None):
            raise ValueError(
                f"shared noncross group {group.name!r} has incompatible "
                "window physics")
        shared.append((window, _validate_sector_selector_bounds(
            group.selector_bounds_B).copy()))
    return kept, shared


def _array_bytes_equal(left, right):
    """Direct byte equality for one quadrature vector; no plan identity."""
    left, right = np.asarray(left), np.asarray(right)
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes(order="C") == right.tobytes(order="C"))


def _same_side_rule(left, right):
    return (
        _array_bytes_equal(left.nodes.t, right.nodes.t)
        and _array_bytes_equal(left.nodes.alpha, right.nodes.alpha)
        and np.array_equal(left.mask_A, right.mask_A)
        and float(left.E_ref_A) == float(right.E_ref_A)
        and int(left.omega_sign) == int(right.omega_sign)
        and str(left.project) == str(right.project)
        and float(left.prefactor) == float(right.prefactor)
    )


def _shared_side_classes(collected, n_p):
    """Infer direct-rule batches and validate the six side partitions."""
    if int(n_p) != 8:
        raise ValueError("shared compatibility schedule requires eight poles")
    classes = []
    for bkey in ("pos_cond", "pos_val", "neg_cond", "neg_val"):
        for pole, window, bounds in collected.get(bkey, []):
            if window.project != "full" or window.crossing_kind is not None:
                raise ValueError("compatibility side has non-Laplace physics")
            match = next((item for item in classes
                          if item[0] == bkey
                          and item[1] == window.name
                          and _same_side_rule(item[2][0][1], window)), None)
            if match is None:
                match = (bkey, str(window.name), [])
                classes.append(match)
            if any(old_pole == pole
                   for old_pole, _window, _bounds in match[2]):
                raise ValueError(
                    f"compatibility side repeats pole {pole} on "
                    f"{bkey}.{window.name}")
            match[2].append((int(pole), window, bounds))

    all_poles = tuple(range(8))
    expected = (
        ("neg_cond", "single", all_poles),
        ("pos_val", "single", all_poles),
        ("neg_val", "a_stripe", (0, 1)),
        ("pos_cond", "a_stripe", (0, 1)),
        ("neg_val", "b_slab", all_poles),
        ("pos_cond", "b_slab", all_poles),
    )
    found = {}
    for bkey, name, pieces in classes:
        pieces.sort(key=lambda item: item[0])
        found.setdefault((bkey, name), []).append(pieces)
    if set(found) != {(bkey, name) for bkey, name, _poles in expected}:
        raise ValueError(
            "loaded sector plans do not contain the six frozen "
            f"compatibility-side classes; found {sorted(found)}")

    out = []
    for bkey, name, expected_poles in expected:
        batches = found[(bkey, name)]
        actual = [piece[0] for batch in batches for piece in batch]
        if (len(actual) != len(set(actual))
                or tuple(sorted(actual)) != expected_poles):
            raise ValueError(
                f"compatibility side {bkey}.{name} does not partition "
                f"poles {list(expected_poles)}; found {sorted(actual)}")
        batches.sort(key=lambda batch: batch[0][0])
        for batch in batches:
            out.append((
                bkey, name, replace(batch[0][1], E_ref_B=0.0),
                np.asarray([piece[0] for piece in batch], dtype=np.int32),
                np.stack([piece[2] for piece in batch])))
    return out


def _shared_noncross_classes(collected, n_p):
    """Infer the six finite-width pole classes from their direct rules."""
    if int(n_p) != 8:
        raise ValueError("shared noncross schedule requires eight poles")
    expected = (
        ("neg_cond", "single", tuple(range(8))),
        ("pos_val", "single", tuple(range(8))),
        ("neg_val", "a_stripe", (0, 1)),
        ("pos_cond", "a_stripe", (0, 1)),
        ("neg_val", "b_slab", tuple(range(8))),
        ("pos_cond", "b_slab", tuple(range(8))),
    )
    out = []
    for bkey, name, poles in expected:
        pieces = sorted(
            (piece for piece in collected.get(bkey, [])
             if str(piece[1].name) == name), key=lambda item: item[0])
        if tuple(piece[0] for piece in pieces) != poles:
            raise ValueError(
                f"shared noncross {bkey}.{name} requires poles "
                f"{list(poles)}")
        first = pieces[0][1]
        if any(not _same_side_rule(first, piece[1]) for piece in pieces[1:]):
            raise ValueError(
                f"shared noncross {bkey}.{name} rules differ across poles")
        out.append((
            bkey, name, replace(first, E_ref_B=0.0),
            np.asarray(poles, dtype=np.int32),
            np.stack([piece[2] for piece in pieces])))
    if sum(len(pieces) for pieces in collected.values()) != sum(
            len(row[2]) for row in expected):
        raise ValueError("loaded plans contain unexpected shared noncross work")
    return out


def _side_batch_sweeps(classes, batch_poles):
    """Map global side-class poles onto one staged pole batch."""
    batch_poles = tuple(map(int, batch_poles))
    if len(batch_poles) != len(set(batch_poles)):
        raise ValueError("staged side batch repeats a pole")
    local = {pole: index for index, pole in enumerate(batch_poles)}
    sweeps = []
    for bkey, name, window, poles, bounds in classes:
        take = [index for index, pole in enumerate(map(int, poles))
                if pole in local]
        if not take:
            continue
        global_poles = np.asarray(
            [int(poles[index]) for index in take], dtype=np.int32)
        local_poles = np.asarray(
            [local[int(pole)] for pole in global_poles], dtype=np.int32)
        sweeps.append((bkey, name, window, global_poles, local_poles,
                       np.asarray(bounds)[take]))
    return sweeps


def _stack_shared_pole_batch(shared_poles, poles, *, consume):
    """Stack one host pole batch, optionally releasing its slab owners."""
    poles = tuple(map(int, poles))
    try:
        values = [shared_poles[pole] for pole in poles]
    except KeyError as exc:
        raise ValueError(f"shared pole {int(exc.args[0])} is missing") from exc
    B_stack = np.stack([value[0] for value in values])
    Omega_stack = np.stack([value[1] for value in values])
    if consume:
        for pole in poles:
            del shared_poles[pole]
    return B_stack, Omega_stack


def run_shared_crossing_branch(
    *, window, pole_indices, selector_bounds, omega_nonneg_ry, E_A, B_poles,
    Omega_poles, psi, tau_kernel, mesh_xy, log_tag, print_fn,
):
    """Integrate one common crossing rule after summing its poles in W(tau)."""
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common import timing
    from common.progress import LoopProgress
    from ..ppm_accumulators import _DeviceTauAccumulator
    from ..ppm_sigma import _SigmaBranchTiles, minimax_tau_integrate_sigma

    n_omega = int(np.asarray(omega_nonneg_ry).size)
    if n_omega == 0:
        return None
    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, nk_proj, nb_proj = psi
    shape = (n_omega, nk_proj, int(psi_proj_xr.shape[1]),
             int(psi_proj_yn.shape[3]))
    sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    accumulator = _DeviceTauAccumulator(
        jnp.asarray(omega_nonneg_ry, dtype=jnp.float64),
        shape=shape, sharding=sharding)
    mask_A = jnp.asarray(window.mask_A)

    def build_sigma_tau(t_node):
        out = tau_kernel(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, B_poles, Omega_poles,
            jnp.asarray(pole_indices, dtype=jnp.int32),
            jnp.asarray(selector_bounds, dtype=jnp.float64),
            jnp.asarray(window.E_ref_A), jnp.asarray(0.0), t_node)
        return (out, None) if window.project_code == 0 else out

    progress = LoopProgress(window.n_tau, print_fn, title=f"sigma[{log_tag}]",
                            item_name="shared tau node", max_updates=10)
    accumulator.begin_window(window)
    minimax_tau_integrate_sigma(
        window.nodes, build_sigma_tau=build_sigma_tau,
        add_tau=accumulator.add_tau, E_ref_sum=window.E_ref_A,
        progress=progress)
    accumulator.end_window()
    progress.finish()
    with timing.section("sigma.finalize"):
        tiles, indices, devices = accumulator.finalize_host_tiles()
    return _SigmaBranchTiles(
        tiles, indices, devices, shape[1:], sharding, nb_proj)


#: Format version of a per-pass partial Σ_c file.  Stamped by
#: :func:`write_pass_partial` and required by :func:`combine_pass_partials`,
#: because a partial cube is the one artifact in this pipeline that is
#: MEANINGLESS ON ITS OWN and yet has exactly the shape, dtype and units of
#: the finished one.  Nothing downstream could tell a stack of partials
#: written under a changed convention from a stack written under this one,
#: so the convention is named in the bytes.
PARTIAL_FORMAT_VERSION = 3


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
                       omega_grid_ry, fit_src, fit_id, source_identity,
                       group_spec=None, leg_id=None, windowing="pane",
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
    fit_id = str(fit_id or "")
    source_identity = str(source_identity or "")
    if not fit_id:
        raise ValueError(
            "write_pass_partial: the fit store has no allocation identity.  "
            "A reusable path cannot distinguish these poles from a later "
            "fit written there; migrate the completed legacy store with "
            "mpa_store.stamp_legacy_fit_id before producing partials.")
    if not source_identity or source_identity == "unknown":
        raise ValueError(
            "write_pass_partial: source identity is unknown.  A partial "
            "must name the exact clean commit or dirty-tree digest whose "
            "quadrature it integrated.")
    with h5py.File(str(path), "w") as f:
        f.attrs["mpa_partial_format_version"] = int(PARTIAL_FORMAT_VERSION)
        f.attrs["mpa_partial_poles"] = np.asarray(poles, dtype=np.int64)
        f.attrs["mpa_partial_n_p"] = int(n_p)
        f.attrs["mpa_partial_fit_store"] = str(fit_src)
        f.attrs["mpa_partial_fit_id"] = fit_id
        f.attrs["mpa_partial_source_identity"] = source_identity
        f.attrs["mpa_partial_windowing"] = str(windowing)
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


def combine_pass_partials(paths, *, n_p, omega_grid_ry, fit_src, fit_id,
                          source_identity, windowing="pane",
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
    fit_id = str(fit_id or "")
    source_identity = str(source_identity or "")
    if not fit_id:
        raise ValueError(
            "combine_pass_partials: the current fit store has no allocation "
            "identity.  Its path may have been reused since these cubes "
            "were written; migrate a completed legacy store explicitly.")
    if not source_identity or source_identity == "unknown":
        raise ValueError(
            "combine_pass_partials: source identity is unknown, so the "
            "quadrature implementation that produced these cubes cannot be "
            "matched to this run.")
    paths = sorted(str(p) for p in paths)
    if manifest is not None:
        manifest_sha = str(manifest.get("sha", ""))
        if manifest_sha != source_identity:
            raise ValueError(
                "combine_pass_partials: the farm manifest declares source "
                f"{manifest_sha!r}, but this run is {source_identity!r}.  "
                "Group ranges and their cubes belong to the exact planner "
                "tree that declared them and cannot cross source states.")
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
            partial_fit_id = str(f.attrs.get("mpa_partial_fit_id", ""))
            if partial_fit_id != fit_id:
                raise ValueError(
                    f"combine_pass_partials: {path} belongs to fit allocation "
                    f"{partial_fit_id!r}, but {str(fit_src)!r} now names "
                    f"allocation {fit_id!r}.  A fit path can be reused; its "
                    "old partials cannot be combined with the new poles.")
            partial_source = str(
                f.attrs.get("mpa_partial_source_identity", ""))
            if partial_source != source_identity:
                raise ValueError(
                    f"combine_pass_partials: {path} was integrated by source "
                    f"{partial_source!r}, not this run's {source_identity!r}.  "
                    "Partial cubes from different quadrature source states "
                    "cannot be mixed.")
            partial_windowing = str(
                f.attrs.get("mpa_partial_windowing", ""))
            if partial_windowing != str(windowing):
                raise ValueError(
                    f"combine_pass_partials: {path} was integrated with MPA "
                    f"windowing={partial_windowing!r}, not {str(windowing)!r}. "
                    "The pane and sector plans are different quadrature "
                    "partitions and cannot be summed into one self-energy.")
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
    plan_verify=False, binned_width_clause=None, windowing="pane",
    print_fn=print,
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
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common import timing
    from common.collectives import barrier, device_put_process_local
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
    windowing = str(windowing).strip().lower()
    if windowing not in ("pane", "sector"):
        raise ValueError(
            f"MPA Sigma windowing={windowing!r}; expected 'pane' or 'sector'.")
    if windowing == "sector" and binned_width_clause is not None:
        raise ValueError(
            "MPA Sigma: mpa_binned_width_clause cannot be combined with "
            "windowing='sector'.  The sector plan has no sign-definite width "
            "panes for that clause to bin.")
    edge_factor = float(ppm_cfg.window_edge_factor)
    xi_ry = narrow_pole_threshold_ry(
        float(ppm_cfg.regularization_ev) / RYD_TO_EV, omega_max_ry,
        edge_factor)
    print_fn(
        f"  MPA Sigma windowing: {windowing} -- "
        + ("fixed GN branch geometry, pi/4 sector sinc on sign-definite "
           "pieces, accepted Gamma<xi compatibility in the crossing core"
           if windowing == "sector" else
           "width-pane planner with the Gamma<xi two-point substitution"))

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
    unfold_q = resolve_pole_q_axis(ledger, int(meta.nk_tot))
    if unfold_q:
        # ANNOUNCED, because a run whose screening reached the full zone
        # through a symmetry map should say so in its own log rather
        # than leave it to be inferred from a store path.
        print_fn(
            f"  MPA Σ: the pole store is on the symmetry WEDGE — "
            f"{ledger['n_q']} of {ledger['n_q_full']} q — and its slices "
            f"are unfolded per pole (permutation and L-phase on B_p, "
            f"permutation alone on Ω_p; nothing conjugated, so the "
            f"fourth quadrant is preserved).  Tables {ledger['table_hash']}.")

    s = wfns.slices
    psi_proj_xr, psi_proj_yn, nb_proj = pad_sigma_window(
        wfns.xr(s.sigma), wfns.yn(s.sigma), mesh_xy)
    psi = (wfns.xn(s.full), wfns.yr(s.full), psi_proj_xr, psi_proj_yn,
           int(wfns.xr(s.sigma).shape[0]), nb_proj)
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    tau_kernels = (_get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid),
                   _get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid,
                                         merged_x=True))
    pole_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    print_fn(
        "  MPA pole placement: B_p, Re Omega_p and complex Omega_p are "
        "staged once per pole at P(None,'x','y') and reused by all four "
        "branches.")

    idx_pos = np.where(omega_req >= 0.0)[0]
    idx_neg = np.where(omega_req < 0.0)[0]
    n_omega = int(omega_req.size)
    tile_acc = None
    tile_meta = None
    records = []

    def _fold_branch(branch, branch_tiles):
        nonlocal tile_acc, tile_meta
        if branch_tiles is None:
            return
        if tile_acc is None:
            tile_meta = branch_tiles
            tile_acc = [np.zeros((n_omega,) + tile.shape[1:],
                                 dtype=np.complex128)
                        for tile in branch_tiles.tiles]
        with timing.section("mpa.branch_fold"):
            for device, tile in enumerate(branch_tiles.tiles):
                tile_acc[device][np.asarray(
                    branch.omega_idx, dtype=np.int64)] += tile

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
    share_crossing = False
    shared_exact, shared_hgl, shared_sides, shared_noncross = {}, {}, {}, {}
    shared_poles, shared_branches = {}, {}
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
    plain_sector_plans = bool(
        plan_dir is not None and windowing == "sector"
        and census_out is None
        and os.path.isfile(os.path.join(
            plan_dir, "plan_p0.pos_cond.h5")))
    if plain_sector_plans:
        required = [os.path.join(
            plan_dir, f"plan_p{pole}.{branch}.h5")
            for pole in range(n_p) for branch in WF.BRANCH_KEYS]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "plain production sector plan is incomplete; missing: "
                + ", ".join(missing))
        if (group_subset is not None
                or tuple(walk) != tuple(resolve_pole_subset(n_p, None))):
            raise ValueError(
                "plain production sector plans share crossing work across "
                "poles 0 and 1 and require one complete in-process walk")
        share_crossing = True
    if plain_sector_plans and plan_verify:
        raise ValueError("production sector plans do not support re-planning")
    if ((census_out is not None
         or (plan_dir is not None and not plain_sector_plans))
            and (not str(census_sha) or str(census_sha) == "unknown")):
        raise ValueError(
            "compute_mpa_sigma_c_omega_grid: a census or plan store "
            "requires the exact clean-commit or dirty-tree source identity; "
            "an empty/unknown identity would let distinct planners address "
            "the same artifact.")
    if plan_dir is not None and census_out is not None:
        if int(jax.process_index()) == 0:
            os.makedirs(plan_dir, exist_ok=True)
        barrier("mpa_plan_store_ready", print_fn=print_fn)
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
                fit_src, p, unfold=unfold_q, mesh_xy=mesh_xy,
                allow_partial=allow_partial)
        a_host = np.asarray(np.real(Omega_p), dtype=np.float64)
        g_host = -np.asarray(np.imag(Omega_p), dtype=np.float64)
        if np.any(g_host[a_host > 1.0e-14] < 0.0):
            raise ValueError(
                f"MPA pole {p} contains live upper-half-plane frequencies")
        b_abs = np.abs(np.asarray(B_p))
        fixed["read_s"] += time.perf_counter() - t_read
        fixed["poles"] += 1
        print_fn(f"  MPA pass read: pole {int(p)} slab in "
                 f"{time.perf_counter() - t_read:.2f} s")
        # The two pole-frequency operands are one physical field.  Keep one
        # host identity for the four planners so the pole-level device cache
        # can stage each representation at most once.
        omega_complex_host = a_host - 1j * g_host
        B_p_device = device_put_process_local(
            np.asarray(B_p, dtype=np.complex128), pole_sharding)
        a_device = device_put_process_local(a_host, pole_sharding)
        state = _prepare_sigma_state(
            jnp.asarray(wfns.enk[:, s.full]), jnp.asarray(wfns.occ[:, s.full]),
            B_p_device, a_device, jnp.ones_like(a_device, dtype=bool),
            jnp.asarray(str(ppm_cfg.fermi_reference) == "midgap", dtype=bool),
            jnp.asarray(True, dtype=bool))
        live = _host_at_source_shape(state.B_mask, bool, _to_host_np)
        B_dev = state.B_corr
        branches = tuple(_iter_branches(
            omega_pos=np.asarray(omega_req[idx_pos], dtype=np.float64),
            idx_pos=idx_pos,
            omega_neg_abs=np.asarray(-omega_req[idx_neg], dtype=np.float64),
            idx_neg=idx_neg,
            E_cond=state.E_cond, H_val=state.H_val,
            cond_mask=state.cond_mask, val_mask=state.val_mask))
        # The branch objects retain precisely the A-side arrays they use and
        # B_dev retains the corrected residue.  The rest of ``state`` is
        # planner evidence already gathered to ``live``; keeping it through
        # thousands of tau dispatches needlessly pins full pole fields.
        del state, B_p_device
        # The pole-level half of the plan address, taken once for the four
        # branches that share these arrays -- and here rather than beside
        # the read, because ``live`` is the store's B mask and comes from
        # ``_prepare_sigma_state``, not from the slab.  Four times over
        # the hash cost 12.56 s a pole against 3.14 s once (2026-08-10);
        # the four branches genuinely read the same Re Omega, Gamma, live
        # mask and |B|, and only E_A, its mask and the ω half-grid differ.
        slab_dig = None
        if plan_dir is not None and not plain_sector_plans:
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

        pole_operands = _BranchOperandCache(mesh_xy)
        pole_operands.seed(a_host, a_device)
        groups = None
        branch_tiles = None
        for br in branches:
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
                        binned_width_clause=binned_width_clause,
                        target_error=float(quad.target_error),
                        laplace_max_nodes=int(quad.max_nodes),
                        crossing_eps_q=float(quad.crossing_eps_q),
                        crossing_max_nodes=int(quad.crossing_max_nodes),
                        use_shipped_minimax_tables=bool(
                            quad.use_shipped_tables),
                        windowing=windowing,
                        omega_complex=omega_complex_host,
                        log_tag=f"p{p} {br.tag}", print_fn=print_fn)

            plan_addr, plan_file = None, None
            if plain_sector_plans:
                plan_addr = "production"
                plan_file = os.path.join(
                    plan_dir, f"plan_p{int(p)}.{bkey}.h5")
            elif plan_dir is not None:
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
                        "windowing": windowing,
                        # THE CLAUSE IS PART OF THE PLAN, SO IT IS PART OF
                        # THE ADDRESS.  The binned width clause changes
                        # pane membership when it is on; a cached plan
                        # keyed without it would serve a clause-off plan
                        # to a clause-on leg and the leg would never know.
                        # -1.0 is the off sentinel (the knob is a positive
                        # ratio when set), so the shipped default keeps a
                        # stable address.
                        "binned_width_clause": (
                            -1.0 if binned_width_clause is None
                            else float(binned_width_clause)),
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
                    if plain_sector_plans:
                        raise FileNotFoundError(
                            f"missing production frequency plan {plan_file}")
                    PS.refuse_missing_plan(
                        plan_dir, pole=p, bkey=bkey, address=plan_addr)
                t_load = time.perf_counter()
                header = PS.read_plan_header(plan_file)
                if plain_sector_plans:
                    if (header["pole"] != int(p)
                            or header["branch"] != bkey
                            or header["n_p"] != n_p):
                        raise ValueError(
                            f"production plan {plan_file} describes a "
                            "different pole, branch, or pole count")
                    lo_i, hi_i = 0, int(header["n_groups"])
                    total_i = hi_i
                else:
                    rng = WF.group_range(
                        group_subset, pole=p, bkey=bkey,
                        n_groups=header["n_groups"])
                    if rng is None:              # pragma: no cover - guarded
                        continue
                    lo_i, hi_i, total_i = rng
                    WF.check_partition(
                        n_groups=header["n_groups"],
                        digest_got=WF.group_plan_digest_from_rows(
                            header["rows"]),
                        pole=p, bkey=bkey, total=total_i, lo=lo_i, hi=hi_i,
                        digest=(group_digests or {}).get(
                            f"{int(p)}.{bkey}"), source="the stored plan")
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
                    f"[{lo_i}, {hi_i}) of {total_i} in {dt_load:.2f} s"
                    + (" (plain production plan)" if plain_sector_plans else
                       f" (address {plan_addr[:12]}, digest "
                       f"{header['group_plan_digest']})"))
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
                    if int(jax.process_index()) == 0:
                        PS.write_branch_plan(
                            plan_file, groups, address=plan_addr, pole=p,
                            bkey=bkey, source_sha=census_sha,
                            fit_store=fit_src, n_p=n_p, stats=stats,
                            a_ry=a_host,
                            omega_complex=omega_complex_host,
                            print_fn=print_fn)
                    barrier(f"mpa_plan_p{int(p)}_{bkey}",
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
            if share_crossing:
                exact, hgl = [], []
                if int(p) in (0, 1) and bkey in ("pos_cond", "neg_val"):
                    groups, exact, hgl = _extract_shared_crossing(
                        groups, a_host, omega_complex_host)
                    if exact:
                        shared_exact.setdefault(bkey, []).append(
                            (int(p), exact[0][0], exact[0][1]))
                    if hgl:
                        if len(hgl) != 1:
                            raise ValueError(
                                "one HGL core is required per branch")
                        shared_hgl.setdefault(bkey, []).append(
                            (int(p), hgl[0][0], hgl[0][1]))
                groups, noncross = _extract_shared_noncross(
                    groups, omega_complex_host)
                if noncross:
                    shared_noncross.setdefault(bkey, []).extend(
                        (int(p), window, bounds)
                        for window, bounds in noncross)
                groups, sides = _extract_shared_sides(
                    groups, a_host, omega_complex_host)
                if sides:
                    shared_sides.setdefault(bkey, []).extend(
                        (int(p), window, bounds)
                        for window, bounds in sides)
                if exact or hgl or sides or noncross:
                    shared_poles.setdefault(int(p), (
                        np.asarray(B_p, dtype=np.complex128),
                        np.asarray(omega_complex_host,
                                   dtype=np.complex128)))
                    shared_branches.setdefault(bkey, br)
            rec.n_tau_nodes += int(sum(int(w.n_tau) for g in groups
                                       for w in g.windows))
            rec.groups += [g.name for g in groups]
            del group_lo
            branch_tiles = run_pass_branch(
                groups=groups, omega_nonneg_ry=br.omega_abs, E_A=br.E_A,
                B_p=B_dev, psi=psi, tau_kernels=tau_kernels,
                mesh_xy=mesh_xy, log_tag=f"p{p} {br.tag}",
                print_fn=print_fn, operand_cache=pole_operands,
                selector_omega=omega_complex_host)
            if branch_tiles is None:
                continue
            _fold_branch(br, branch_tiles)
        records.append(rec)
        # finalize_host_tiles has drained every tau/D2H use of this pole.
        # Drop its device and host owners before the next slab is read so the
        # advertised one-pole residency is also the actual HBM high-water.
        del (B_dev, a_device, pole_operands, groups, branch_tiles, branches,
             br, Omega_p, B_p, a_host, g_host, b_abs, live,
             omega_complex_host)

    if share_crossing:
        from ..ppm_tau_kernel import _get_sigma_shared_tau_kernel

        expected = {"pos_cond", "neg_val"}
        if (set(shared_exact) != expected or set(shared_hgl) != expected
                or set(shared_poles) != set(range(n_p))):
            raise ValueError(
                "loaded sector plans do not contain complete shared "
                "exact/HGL/side work for the full pole stack")
        side_classes = _shared_side_classes(shared_sides, n_p)
        noncross_classes = (_shared_noncross_classes(shared_noncross, n_p)
                            if shared_noncross else [])
        stacked_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
        crossing_poles = (0, 1)
        B_host, Omega_host = _stack_shared_pole_batch(
            shared_poles, crossing_poles, consume=False)
        B_crossing = device_put_process_local(B_host, stacked_sharding)
        Omega_crossing = device_put_process_local(Omega_host,
                                                  stacked_sharding)
        del B_host, Omega_host
        for kind, collected, merged_x, real_pole in (
                ("exact", shared_exact, True, False),
                ("HGL", shared_hgl, False, True)):
            kernel = _get_sigma_shared_tau_kernel(
                mesh_xy=mesh_xy, kgrid=kgrid,
                merged_x=merged_x, real_pole=real_pole)
            for bkey in ("pos_cond", "neg_val"):
                window, bounds = _common_shared_window(
                    collected[bkey], kind)
                branch = shared_branches[bkey]
                _fold_branch(branch, run_shared_crossing_branch(
                    window=window, pole_indices=np.asarray((0, 1)),
                    selector_bounds=bounds,
                    omega_nonneg_ry=branch.omega_abs, E_A=branch.E_A,
                    B_poles=B_crossing, Omega_poles=Omega_crossing, psi=psi,
                    tau_kernel=kernel, mesh_xy=mesh_xy,
                    log_tag=f"shared {kind} p0+p1 {branch.tag}",
                    print_fn=print_fn))
                records[0].n_tau_nodes += int(window.n_tau)
                records[0].groups.append(f"shared-{kind.lower()}:{bkey}")
        del B_crossing, Omega_crossing

        side_kernel = _get_sigma_shared_tau_kernel(
            mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True, real_pole=True)
        noncross_kernel = _get_sigma_shared_tau_kernel(
            mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True, real_pole=False)
        for batch_poles in ((0, 1, 2, 3), (4, 5, 6, 7)):
            B_host, Omega_host = _stack_shared_pole_batch(
                shared_poles, batch_poles, consume=False)
            B_batch = device_put_process_local(B_host, stacked_sharding)
            Omega_batch = device_put_process_local(
                Omega_host, stacked_sharding)
            del B_host, Omega_host
            for (bkey, name, window, global_poles, local_poles,
                 bounds) in _side_batch_sweeps(side_classes, batch_poles):
                branch = shared_branches[bkey]
                _fold_branch(branch, run_shared_crossing_branch(
                    window=window, pole_indices=local_poles,
                    selector_bounds=bounds,
                    omega_nonneg_ry=branch.omega_abs, E_A=branch.E_A,
                    B_poles=B_batch, Omega_poles=Omega_batch, psi=psi,
                    tau_kernel=side_kernel, mesh_xy=mesh_xy,
                    log_tag=(f"shared side {bkey}.{name} "
                             f"rank{window.n_tau} "
                             f"poles{list(map(int, global_poles))}"),
                    print_fn=print_fn))
                records[0].n_tau_nodes += int(window.n_tau)
                records[0].groups.append(
                    f"shared-side:{bkey}.{name}:rank{int(window.n_tau)}")
            for (bkey, name, window, global_poles, local_poles,
                 bounds) in _side_batch_sweeps(noncross_classes,
                                               batch_poles):
                branch = shared_branches[bkey]
                _fold_branch(branch, run_shared_crossing_branch(
                    window=window, pole_indices=local_poles,
                    selector_bounds=bounds,
                    omega_nonneg_ry=branch.omega_abs, E_A=branch.E_A,
                    B_poles=B_batch, Omega_poles=Omega_batch, psi=psi,
                    tau_kernel=noncross_kernel, mesh_xy=mesh_xy,
                    log_tag=(f"shared noncross {bkey}.{name} "
                             f"rank{window.n_tau} "
                             f"poles{list(map(int, global_poles))}"),
                    print_fn=print_fn))
                records[0].n_tau_nodes += int(window.n_tau)
                records[0].groups.append(
                    f"shared-noncross:{bkey}.{name}:"
                    f"rank{int(window.n_tau)}")
            del B_batch, Omega_batch
            for pole in batch_poles:
                del shared_poles[int(pole)]
        if shared_poles:
            raise ValueError(
                f"unconsumed shared pole slabs: {sorted(shared_poles)}")
        del shared_poles

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
        if int(jax.process_index()) == 0:
            WF.write_census(
                census_out, census_rows, fit_store=str(fit_src), n_p=n_p,
                sha=str(census_sha), digests=census_digests,
                extra={"poles": [int(x) for x in walk],
                       "omega_grid_ry": [float(x) for x in omega_req],
                       "windowing": windowing,
                       "plans": plan_written})
        barrier("mpa_pass_census_written", print_fn=print_fn)
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
    print_fn(format_pass_report(
        records, xi_ev=xi_ry * RYD_TO_EV, windowing=windowing))
    return jnp.asarray(sigma_c, dtype=jnp.complex128), records
