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

from dataclasses import dataclass, field

import numpy as np

from common.units import RYD_TO_EV

__all__ = [
    "DEFAULT_LAPLACE_RATIO_MAX",
    "MAX_WIDTH_SPLIT_LEAVES",
    "PassRecord",
    "WindowGroup",
    "compute_mpa_sigma_c_omega_grid",
    "format_pass_report",
    "narrow_pole_threshold_ry",
    "plan_branch_groups",
    "refuse_wedge_pole_slab",
    "run_pass_branch",
    "split_pass_by_width",
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
    each group ships an explicit mask with ``mask_B_mode='all'`` windows.

    ``omega_operand`` is what the tau kernel gets in its ``Omega_q`` slot:
    the REAL ``Re Omega`` for a legacy-routed group (which is what makes
    that group a two-point plasmon-pole evaluation and nothing else) and
    the complex ``Omega_p`` for an MPA-routed one.
    """

    name: str
    windows: list
    mask_B: np.ndarray
    omega_operand: np.ndarray
    n_modes: int
    b_mass: float
    provenance: str


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


def _laplace_buckets(a_ry, mask, *, e_lo, e_hi, omega_max, r_max):
    """Geometric buckets in ``Re Omega`` with bounded Laplace ratio.

    Each bucket's sign-definite window will span ``x_min ~ e_lo + a_lo -
    omega_max`` to ``x_max ~ e_hi + a_hi + omega_max``; the bucket edge is
    the largest ``a_hi`` keeping that ratio at or under ``r_max``.  The
    recursion ``a_hi = r_max*(e_lo + a_lo) - e_hi`` grows geometrically, so
    a field spanning four decades of pole position needs a handful of
    buckets rather than one per octave.

    Returns a list of ``(a_lo, a_hi)`` closed intervals covering the masked
    values, or ``[]`` when the mask is empty.
    """
    a = np.asarray(a_ry, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    a_min = float(np.min(a, where=m, initial=np.inf))
    a_max = float(np.max(a, where=m, initial=-np.inf))
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
#: Not a tuning knob: with the predicate aligned to the builder's own
#: x_min (see the call site) a real field terminates in a handful of
#: leaves, and each leaf is a full-size ``(n_q, N_mu, N_mu)`` boolean
#: mask -- 81.4 MB on the Si production deck -- held live in the
#: returned list.  64 leaves is ~5 GB of masks, an order below any
#: node's memory; a field that asks for more is a field whose width
#: clause cannot be satisfied by splitting, and it should say so in one
#: line rather than accumulate masks until the OOM killer says it in
#: none.
MAX_WIDTH_SPLIT_LEAVES = 64


def _refuse_width_split_explosion(n_leaves, mask, gamma_ry):
    """Refuse a width split that is diverging, with the field statistics.

    The FALSE case this guards was measured, not imagined: the first
    end-to-end MPA Sigma dispatch put the split's termination clause on
    an unreachable floor and the recursion bisected toward its depth-16
    cap -- 2^16 leaves x 81.4 MB of mask each, a 230 GB node dead at
    ~2 800 of them.  The predicate fix at the call site makes that
    geometry terminate; this guard is for the field NOBODY has met yet,
    and it converts "the node died six minutes in" into a one-line
    refusal naming what to look at.
    """
    if int(n_leaves) <= MAX_WIDTH_SPLIT_LEAVES:
        return
    g = np.asarray(gamma_ry, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    g_lo, g_hi = _stats(g, m)
    raise RuntimeError(
        f"MPA Sigma window planning: the width split of one Laplace "
        f"bucket returned {int(n_leaves)} leaves against a ceiling of "
        f"{MAX_WIDTH_SPLIT_LEAVES}.  Each leaf holds a full-size "
        f"(n_q, N_mu, N_mu) boolean mask, so a diverging split is an "
        f"out-of-memory kill wearing a planning stage's clothes -- the "
        f"2026-08-09 profile measured ~2 800 live masks at a 230 GB "
        f"node's death.  This bucket spans Gamma = [{g_lo:.3e}, "
        f"{g_hi:.3e}] Ry over {int(np.count_nonzero(m))} modes; a split "
        f"this deep means the width clause cannot be satisfied by "
        f"splitting at all (the Laplace edge is pinned far below the "
        f"widths), which is a property of the pole field worth looking "
        f"at, not one worth 2^16 masks.")


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
    """
    a = np.asarray(a_ry, dtype=np.float64)
    g = np.asarray(gamma_ry, dtype=np.float64)
    a_lo, _ = _stats(a, mask)
    g_lo, g_hi = _stats(g, mask)
    x_min = max(e_lo + a_lo - omega_max, _X_FLOOR_RY)
    if g_hi <= float(beta_max) * x_min or depth >= 16 or g_hi <= g_lo:
        return [mask]
    cut = float(np.sqrt(g_lo * g_hi))
    lower = mask & (g <= cut)
    upper = mask & (g > cut)
    if not lower.any() or not upper.any():
        return [mask]                       # one width; nothing to separate
    return (_clause_safe_width_split(a, g, lower, e_lo=e_lo,
                                     omega_max=omega_max, beta_max=beta_max,
                                     depth=depth + 1)
            + _clause_safe_width_split(a, g, upper, e_lo=e_lo,
                                       omega_max=omega_max,
                                       beta_max=beta_max, depth=depth + 1))


def _stats(arr, mask):
    """``(min, max)`` of ``arr`` over ``mask``; the caller checks emptiness."""
    a = np.asarray(arr, dtype=np.float64)
    return (float(np.min(a, where=mask, initial=np.inf)),
            float(np.max(a, where=mask, initial=-np.inf)))


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
    *, a_ry, gamma_ry, bucket_mask, E_A_host, base_mask_A_host,
    omega_max, space, neg_omega_half, edge_factor, rel_tol, max_nodes,
):
    """The up-to-three MPA windows serving one ``Re Omega`` bucket."""
    from . import sigma_routing as R

    neg = -1.0 if neg_omega_half else 1.0
    e_lo, e_hi = _stats(E_A_host, base_mask_A_host)
    groups = []

    if not R.denominator_can_cross(space, bool(neg_omega_half)):
        a_lo, a_hi = _stats(a_ry, bucket_mask)
        _, g_hi = _stats(gamma_ry, bucket_mask)
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
        groups.append((("single",), [win], bucket_mask))
        return groups

    R.refuse_edge_factor_below_envelope(edge_factor)
    # The pane is sized by the WIDEST width in the bucket, so a pole this
    # classifier sends to the slab satisfies a > omega_max + edge*Gamma_p
    # for its OWN width and genuinely cannot cross.  Over-including in the
    # core is safe (the crossing rule is exact at any offset) and only
    # costs nodes; under-including would put a crossing pole under a
    # sign-definite rule, which is a wrong number rather than a slow one.
    _, g_hi_all = _stats(gamma_ry, bucket_mask)
    z_edge = float(edge_factor) * g_hi_all
    T = omega_max + z_edge

    cross = bucket_mask & (np.asarray(a_ry) <= T)
    slab = bucket_mask & ~cross

    if cross.any():
        a_lo, a_hi = _stats(a_ry, cross)
        g_lo, g_hi = _stats(gamma_ry, cross)
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
            groups.append((tuple(w.name for w in wins), wins, cross))

    if slab.any():
        a_lo, a_hi = _stats(a_ry, slab)
        _, g_hi = _stats(gamma_ry, slab)
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
                        f"kappa0={rule['kappa0']:.6f}"))], slab))
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
    omega_max = (float(np.max(np.asarray(omega_nonneg_ry, dtype=np.float64)))
                 if np.size(omega_nonneg_ry) else 0.0)
    max_nodes = (R.DEFAULT_MAX_CROSSING_NODES if max_nodes is None
                 else int(max_nodes))
    narrow, wide = split_pass_by_width(g, live_mask, xi_ry)
    mass = (np.abs(np.asarray(b_abs, dtype=np.float64))
            if b_abs is not None else None)

    def _mass(mask):
        return 0.0 if mass is None else float(np.sum(mass, where=mask))

    groups = []
    if narrow.any():
        wins = _build_windows_for_branch(
            omega_nonneg_ry=np.asarray(omega_nonneg_ry, dtype=np.float64),
            E_A=jnp.asarray(E_A_host, dtype=jnp.float64),
            base_mask_A=jnp.asarray(base_mask_A_host, dtype=bool),
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
                name="legacy", windows=wins, mask_B=narrow,
                omega_operand=a, n_modes=int(np.count_nonzero(narrow)),
                b_mass=_mass(narrow),
                provenance=("two-point crossing machinery at "
                            f"xi={float(xi_ry) * RYD_TO_EV:.4f} eV; these "
                            "poles are narrower than the smearing and are "
                            "not distinguishable from real ones")))

    omega_complex = a - 1j * g
    for a_lo, a_hi in _laplace_buckets(
            a, wide, e_lo=_stats(E_A_host, base_mask_A_host)[0],
            e_hi=_stats(E_A_host, base_mask_A_host)[1],
            omega_max=omega_max, r_max=float(laplace_ratio_max)):
        bucket = wide & (a >= a_lo) & (a <= a_hi)
        if not bucket.any():
            continue
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
            subs = [bucket]
        else:
            subs = _clause_safe_width_split(
                a, g, bucket, e_lo=e_lo_A, omega_max=0.0,
                beta_max=R.SHIPPED_WIDTH_BETA_MAX)
            _refuse_width_split_explosion(len(subs), bucket, g)
        for sub in subs:
            for names, wins, mask in _mpa_groups_for_bucket(
                    a_ry=a, gamma_ry=g, bucket_mask=sub, E_A_host=E_A_host,
                    base_mask_A_host=base_mask_A_host, omega_max=omega_max,
                    space=space, neg_omega_half=neg_omega_half,
                    edge_factor=edge_factor, rel_tol=rel_tol,
                    max_nodes=max_nodes):
                g_sub_lo, g_sub_hi = _stats(g, sub)
                groups.append(WindowGroup(
                    name=(f"mpa[a={a_lo * RYD_TO_EV:.3g}"
                          f"-{a_hi * RYD_TO_EV:.3g}eV,"
                          f"G={g_sub_lo * RYD_TO_EV:.3g}"
                          f"-{g_sub_hi * RYD_TO_EV:.3g}eV]"
                          f"{'+'.join(names)}"),
                    windows=wins, mask_B=mask,
                    omega_operand=omega_complex,
                    n_modes=int(np.count_nonzero(mask)),
                    b_mass=_mass(mask),
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
        _integrate_tau_windows_for_branch(
            windows=grp.windows, accumulator=accumulator, E_A=E_A,
            B_q=B_p, Omega_q=jnp.asarray(grp.omega_operand),
            base_mask_B=jnp.asarray(grp.mask_B, dtype=bool),
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


def compute_mpa_sigma_c_omega_grid(
    wfns, fit_src, meta, mesh_xy, *, ppm_cfg, quad, omega_grid_ry,
    laplace_ratio_max=DEFAULT_LAPLACE_RATIO_MAX, rel_tol=1.0e-8,
    allow_partial=False, print_fn=print,
):
    """``Sigma_c(omega, k, m, n)`` from a staged multipole fit store.

    Reads as the physics outline section 7.5 asks for: one pass per pole,
    ascending, each pass a full four-branch Sigma integration over that
    pole's ``(B_p, Omega_p)`` slab and nothing else resident.

    Returns ``(sigma_c_kij, records)`` -- the replicated host tensor in Ry
    and the per-pass provenance the design requires each pass to record.
    """
    import jax
    import jax.numpy as jnp

    from common import timing
    from file_io import mpa_store
    from ..ppm_sigma import (
        _iter_branches, _prepare_sigma_state, pad_sigma_window,
        strip_sigma_window)
    from ..ppm_tau_kernel import _get_sigma_tau_kernel
    from ..ppm_windows import _to_host_np
    from .fit_driver import pole_pass_order

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
    # pinned is what makes the result reproducible run to run.
    for p in pole_pass_order(n_p):
        with timing.section("mpa.pass_read"):
            Omega_p, B_p = mpa_store.read_pole_slice(
                fit_src, p, allow_partial=allow_partial)
        a_host = np.asarray(np.real(Omega_p), dtype=np.float64)
        g_host = np.abs(np.asarray(np.imag(Omega_p), dtype=np.float64))
        b_abs = np.abs(np.asarray(B_p))
        state = _prepare_sigma_state(
            jnp.asarray(wfns.enk[:, s.full]), jnp.asarray(wfns.occ[:, s.full]),
            jnp.asarray(B_p, dtype=jnp.complex128),
            jnp.asarray(a_host, dtype=jnp.float64),
            jnp.ones(a_host.shape, dtype=bool),
            jnp.asarray(str(ppm_cfg.fermi_reference) == "midgap", dtype=bool),
            jnp.asarray(True, dtype=bool))
        live = _host_at_source_shape(state.B_mask, bool, _to_host_np)
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
            E_A_host = _host_at_source_shape(br.E_A, np.float64, _to_host_np)
            mask_A_host = _host_at_source_shape(
                br.base_mask_A, bool, _to_host_np)
            with timing.section("mpa.windows"):
                groups, stats = plan_branch_groups(
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
                    use_shipped_minimax_tables=bool(quad.use_shipped_tables),
                    log_tag=f"p{p} {br.tag}", print_fn=print_fn)
            rec.n_legacy_modes = stats["n_narrow"]
            rec.n_mpa_modes = stats["n_wide"]
            rec.legacy_b_mass = stats["narrow_b_mass"]
            rec.mpa_b_mass = stats["wide_b_mass"]
            rec.n_tau_nodes += stats["n_tau"]
            rec.groups += [g.name for g in groups]
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
