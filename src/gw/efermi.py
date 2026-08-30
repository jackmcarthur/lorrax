"""Fermi level and occupations from an eigenvalue set.

ON DEVICE, FOR THE SYNC AND NOT THE ARITHMETIC.  The sort and the
cumulative sum are microseconds on a ``(nk, nb)`` input either way, but
``fermi_level_step`` runs every self-consistent iteration on ``E`` fresh
off the eigh (``sc_iteration.gw_iteration_map``,
``rebuild_hartree_dft_basis``), so a numpy body cost a readback of the
whole eigenvalue block and a pipeline sync mid-iteration.  The body is one
``jax.jit`` kernel over ``E``; the only thing that crosses to the host is
a 3-vector (E_F, the degeneracy flag, the boundary energy), at the one
point the caller needs a Python float anyway (``rotate_wavefunctions``
takes ``efermi: float``).

THE DEGENERATE REFUSAL IS A RETURNED FLAG, NOT A RAISE.  A ``raise``
cannot fire inside a traced kernel, so the kernel computes the condition
and ships it in that same 3-vector and the host wrapper raises the same
``ValueError``.  ``checkify`` was rejected: it would force every caller
into the checkify transform to recover a condition that already crosses
on the transfer carrying E_F.  The refusal itself is load-bearing —
``qsgw_density.distributed_eigh_bands``'s gauge argument depends on it, ρ
being invariant under the arbitrary unitary inside a degenerate subspace
only because f is constant across that subspace.

OPERANDS.  ``E_kn`` is the device operand.  ``kweights`` is a k-table of
``n_k`` numbers, built once per run, and stays host numpy — passed into
the kernel as an argument, so no recompilation per call.

WHY THIS IS ITS OWN MODULE.  ``sc_iteration._midgap_efermi`` computes a
midgap E_F from a FIXED integer ``n_occ``, i.e. it assumes band
``n_occ`` is the last occupied one at EVERY k and ignores the k-weights.
That is right for a gapped insulator on a uniform grid and wrong for a
metal, for a semimetal, and for any k-set with non-uniform weights — which
is the IBZ, i.e. the case the self-consistent density has to run on.  This
module answers the general question and the self-consistency code calls it
rather than growing a second convention inline.

UNITS: BAND OCCUPANCY, NOT ELECTRONS.  Every quantity here counts
*occupied bands*, weighted by ``kweights``:

    Σ_k w_k Σ_n f_nk  ==  n_occ_bands

The spin degeneracy ``f_spin`` (2 for a spin-restricted scalar
calculation, 1 for a spinor one — ``psp.get_DFT_mtxels.
spin_degeneracy_factor``) multiplies BOTH the electron count and the
occupancy, so it cancels out of E_F exactly.  Carrying it here would be an
invitation to apply it twice; it belongs in ρ's normalisation, where it
does not cancel.

MP1 APIs below count physical electrons.  ``WfnLoader.num_electrons`` is
the weighted WFN occupation count; do not substitute
``WfnLoader.nelec=max(ifmax)``, which is only a band boundary in a metal.
Pass the canonical
``spin_degeneracy_factor(wfn)`` as ``state_capacity``: 1 for a fully
relativistic spinor WFN and 2 for a spin-restricted scalar WFN.  Their
``broadening_ry`` follows BerkeleyGW ``occ_broadening``: the MP1
argument is ``(E-mu)/(2*broadening_ry)``.  Matching QE therefore uses
``broadening_ry=degauss/2``.

TWO OCCUPANCY RULES LIVE HERE AND THEY ARE NOT THE SAME RULE.
``occupation_window_threshold`` (:func:`band_in_occupation_window`) decides
which bands are worth putting in a Green's-function branch — a per-consumer
band-set question.  ``occupation_clamp_tol``
(:func:`clamp_occupation_tail`) decides whether a meaningless value exists
in the table at all, and it is applied once, at evaluation, so every
consumer sees the same support.  Neither subsumes the other and neither
should be collapsed into the other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp


__all__ = [
    "OCCUPATION_CLAMP_TOL_DEFAULT",
    "OCCUPATION_CLAMP_TOL_MAX",
    "OCCUPATION_WINDOW_THRESHOLD_DEFAULT",
    "MP1_LOBE_EXTREMUM",
    "OccupationState", "assert_fixed_n", "assert_wfn_occupation_consistency",
    "band_in_occupation_window", "clamp_occupation_tail", "fermi_level_step",
    "mp1_negative_derivative", "mp1_occupations",
    "occupation_clamp_tol", "occupation_weight_floor",
    "occupied_band_count", "resolve_sigma_efermi_ry",
    "solve_mp1_occupations", "step_occupations",
]


#: The three legal ``fermi_reference`` values and what each NAMES.  Kept
#: beside the resolver so the vocabulary and its interpretation cannot
#: drift apart (``gw_config.DynamicSigmaConfig.__post_init__`` validates
#: membership; this decides what membership MEANS).
SIGMA_FERMI_REFERENCES = ("vbm", "midgap", "mp1_fixed_n")


def resolve_sigma_efermi_ry(fermi_reference, *, occupation_state, wfn):
    """THE energy the Sigma omega axis is measured from, and where it came from.

    Returns ``(efermi_ry, provenance)`` -- the second value is the string
    ``file_io.write_sigma_omega_h5`` stamps as
    ``omega_reference_provenance``, so a post-hoc consumer never has to
    guess (audit A2 measured a 2.79 eV mis-sampling of Sigma_c(omega) on
    the sodium metal deck from exactly that guess).

    WHY THIS IS A FUNCTION AND NOT THREE INLINE BRANCHES.  The MPA Sigma
    branch used to read ``occupation_state.mu_ry or wfn.efermi`` directly
    and never look at ``fermi_reference`` at all, while the PPM branch
    honored it.  ``wfn.efermi`` happens to BE the midgap
    (``wfn_loader.loader`` sets ``0.5*(vbm + cbm)``), so the default
    ``midgap`` agreed by coincidence and only ``vbm`` was silently wrong --
    which is the worst shape for a defect to have, because the one deck
    that would expose it is the one nobody runs by default.

      ``mp1_fixed_n``  the fixed-N MP1 chemical potential of THIS
                       iteration.  Requires an ``occupation_state``; a
                       metal has one and ``gw_config`` refuses the
                       combination without it.
      ``midgap``       ``0.5*(vbm + cbm)`` -- the loader's ``efermi``.
      ``vbm``          the valence-band maximum, ``wfn.vbm``.

    An ``occupation_state`` present with a gap-derived reference is a
    contradiction, not a preference: the state exists because the deck
    solved a fixed-N chemical potential, and measuring Sigma against a
    different reference than the occupations were built at is the
    confound this refusal exists to make unspellable.
    """
    name = str(fermi_reference).strip().lower()
    if name not in SIGMA_FERMI_REFERENCES:
        raise ValueError(
            f"fermi_reference must be one of {SIGMA_FERMI_REFERENCES}; "
            f"got {fermi_reference!r}.")
    if name == "mp1_fixed_n":
        if occupation_state is None:
            raise ValueError(
                "fermi_reference = mp1_fixed_n names the fixed-N MP1 "
                "chemical potential, but no occupation_state reached the "
                "Sigma dispatch.  The QSGW driver passes one; a direct "
                "caller must construct it from the current spectrum.")
        return float(occupation_state.mu_ry), "fixed-N mu"
    if occupation_state is not None:
        raise ValueError(
            f"fermi_reference = {name!r} is a gap-derived reference, but "
            f"this iteration carries a fixed-N occupation_state (mu = "
            f"{float(occupation_state.mu_ry):.6f} Ry).  Measuring Sigma "
            f"against a different reference than the occupations were "
            f"solved at is a confound, not a choice: set fermi_reference "
            f"= mp1_fixed_n.")
    if name == "midgap":
        return float(wfn.efermi), "midgap"
    return float(wfn.vbm), "vbm"


# ---------------------------------------------------------------------------
#  The occupancy threshold that decides whether a band is in a
#  Green's-function branch at all.  ONE predicate, ONE floor, ONE default —
#  every consumer imports from here rather than re-spelling the rule.  The
#  consumers are listed in ``docs/input_reference.md`` under
#  ``occupation_window_threshold``.
# ---------------------------------------------------------------------------

#: Deck default for ``occupation_window_threshold``.  See
#: :func:`occupation_weight_floor` for the occupancy→weight mapping and
#: :func:`band_in_occupation_window` for the rule itself.
OCCUPATION_WINDOW_THRESHOLD_DEFAULT = 0.995


def occupation_weight_floor(occupation_window_threshold):
    """Map the deck's OCCUPANCY threshold onto a branch-WEIGHT floor.

    The deck key is an occupancy because that is how the knob is reasoned
    about ("keep a band until it is 99.5% occupied"), but every cut is on a
    branch WEIGHT, which is ``f`` on the occupied/hole branch and ``1 − f``
    on the empty/electron branch (``gw/mpa/sigma.py`` ``val_weight=f,
    cond_weight=1.0 - f``; ``gw/w_isdf.py`` ``band_weight=occ_f`` and
    ``band_weight=1.0 - occ_u``).  A band at occupancy 0.995 therefore
    carries weight 0.995 in the occupied branch and 0.005 in the empty one,
    and a band at occupancy 0.005 the mirror pair, so ONE floor
    ``1 − threshold`` cuts both branches symmetrically at the same physical
    distance from a filled/empty state.  Occupancy 0.995 ⇒ floor 0.005.

    ``threshold = 1.0`` gives floor 0.0, i.e. ``abs(w) > 0.0`` ≡ ``w != 0.0``
    — the exact incumbent rule, bit-for-bit, at every consumer.  That is the
    deliberate escape hatch and the A/B control for this knob.
    """
    t = float(occupation_window_threshold)
    if not (np.isfinite(t) and 0.5 <= t <= 1.0):
        raise ValueError(
            "occupation_window_threshold must be an occupancy in [0.5, 1.0]; "
            f"got {occupation_window_threshold!r}.  It is the occupancy at "
            "which a band stops counting toward a Green's-function branch; "
            "the weight floor applied is 1 - threshold.  1.0 reproduces the "
            "exact `weight != 0` rule.")
    return 1.0 - t


def band_in_occupation_window(weight, weight_floor):
    """``abs(weight) > weight_floor`` — the one band-inclusion predicate.

    MAGNITUDE, NEVER A ONE-SIDED CUT.  :attr:`OccupationState.f_kn` is never
    clipped (MP1's overshoot beyond [0, 1] is part of the configured
    quadrature — see :func:`mp1_occupations`), so a band just above μ carries
    a NEGATIVE occupied-branch weight, down to about **-0.0355** at the MP1
    lobe minimum — seven times the 0.005 floor the default threshold sets.
    ``weight > floor`` would silently discard every one of them; the exact
    incumbent rule ``weight != 0.0`` kept them, and
    ``mpa: wrong-side fractional states keep their branch's algebra``
    (6d3b6b47) exists to route them correctly.  ``abs`` keeps them and drops
    only what is genuinely negligible.

    EXACT ZEROS STAY EXCLUDED at every threshold, since ``0 > floor`` is
    false for floor ≥ 0.  The reason the exact rule existed is preserved
    rather than traded away: the exact cut excludes only what UNDERFLOWED to
    zero, which for MP1 is ~54 smearing widths out from μ — the -0.53 Ry
    phantom excursion of the first metallic Σ arm (claim 0196), whose
    smallest live weight was the subnormal 2.67e-322.  A fixed occupancy
    threshold instead cuts at a fixed number of smearing widths (0.995 ⇒
    about 4.28), because the MP1 weight depends on ``(E-μ)/(2W)`` alone.

    Works on numpy and jax arrays alike; returns whatever ``abs``/``>`` give
    for the operand type, so a caller keeps its own array library.
    """
    return abs(weight) > weight_floor


# ---------------------------------------------------------------------------
#  The far-tail clamp.  ADDITIVE TO, NOT A REPLACEMENT FOR, the threshold
#  above: they answer different questions.  ``occupation_window_threshold``
#  decides which bands are worth putting in a Green's-function branch;
#  the clamp decides whether a meaningless value exists in the table at all.
# ---------------------------------------------------------------------------

#: MP1's extremum outside [0, 1], reached at ``x = sqrt(3/2) = 1.224745``:
#: ``f = -0.0354579065701`` below mu's mirror point and ``1.0354579065701``
#: above it.  Closed form, not a measurement — ``f'(x) = 0`` for
#: ``f(x) = erfc(x)/2 - x*exp(-x^2)/(2*sqrt(pi))`` is ``2x^2 = 3``.  This
#: number is the ONLY thing standing between the clamp and the overshoot,
#: so it is a named constant and :func:`occupation_clamp_tol` refuses
#: anything that could approach it.
MP1_LOBE_EXTREMUM = 0.0354579065701095

#: Hard ceiling on the clamp tolerance: 35x below :data:`MP1_LOBE_EXTREMUM`.
#: A tolerance at or above the extremum would flatten the whole negative
#: lobe to zero, i.e. delete part of the configured quadrature.  Refusing
#: at 1e-3 makes that unreachable BY CONSTRUCTION rather than by the
#: default happening to be small.
OCCUPATION_CLAMP_TOL_MAX = 1e-3

#: Deck default for ``occupation_clamp_tol``.  Puts the MP1 table's own
#: support at ``|x| < 4.30834``, i.e. ``|E - mu| < 8.6167 * W`` — 1.17 eV at
#: ``W = 0.01 Ry``.  The unclamped table's support edge sits instead at
#: ``x = 27.2971``, which is ``sqrt(-ln(smallest float64 subnormal)) =
#: 27.2844`` to four figures: the radius at which ``exp(-x^2)`` stops being
#: REPRESENTABLE, not any physical scale.  That is 54.59 * W = 7.43 eV at
#: ``W = 0.01 Ry``, 5.4x room-temperature kT.
OCCUPATION_CLAMP_TOL_DEFAULT = 1e-8


def occupation_clamp_tol(value) -> float:
    """Validate the far-tail clamp tolerance and return it as a float.

    ``0.0`` disables the clamp exactly — ``abs(f) < 0.0`` is false for every
    finite ``f`` — which is the A/B control and the bit-for-bit escape hatch,
    the same role ``occupation_window_threshold = 1.0`` plays for the
    threshold.  The upper bound is :data:`OCCUPATION_CLAMP_TOL_MAX`; see
    :func:`clamp_occupation_tail` for why it is where it is.
    """
    t = float(value)
    if not (np.isfinite(t) and 0.0 <= t <= OCCUPATION_CLAMP_TOL_MAX):
        raise ValueError(
            "occupation_clamp_tol must be in [0.0, "
            f"{OCCUPATION_CLAMP_TOL_MAX!r}]; got {value!r}.  It is the "
            "distance from 0 or 1 within which an MP1 occupation is snapped "
            "to exactly 0 or 1.  The ceiling is 35x below MP1's overshoot "
            f"extremum {MP1_LOBE_EXTREMUM!r}, which is part of the "
            "configured quadrature and must never be clamped away; 0.0 "
            "disables the clamp bit-for-bit.")
    return t


def clamp_occupation_tail(f, clamp_tol):
    """Snap the FAR TAIL of an occupation table to exact 0 / 1.

    ``abs(f) < tol -> 0.0`` and ``abs(1 - f) < tol -> 1.0``.  Nothing else
    moves, and nothing moves by more than ``tol``.

    WHY THE TABLE AND NOT EACH CONSUMER.  Left unclamped, an MP1 table's
    support is set by where ``exp(-x^2)`` underflows — ``x = 27.2971``,
    which is ``sqrt(-ln(5e-324))`` to four figures.  That is a float64
    property, not a physical one: it puts nonzero weight on states 54.59
    smearing widths from mu (7.43 eV at ``W = 0.01 Ry``) and it moves if a
    backend flushes subnormals.  Fixing the table once makes every
    downstream support backend-independent, instead of leaving each
    consumer to defend against underflow separately.  At the default
    ``tol = 1e-8`` the support becomes ``|x| < 4.30834``, i.e.
    ``|E - mu| < 8.6167 * W``.

    IT CANNOT REACH THE OVERSHOOT, BY CONSTRUCTION.  ``f_kn`` is never
    clipped: MP1 overshoots [0, 1] near the Fermi surface and that
    overshoot is part of the configured quadrature (measured -0.0316 to
    1.0022 on the Na bcc SOC deck; closed-form extremum
    :data:`MP1_LOBE_EXTREMUM` = 0.03546 at ``x = sqrt(3/2)``).  This is a
    clamp on the VALUE's distance from 0 or 1, and
    :func:`occupation_clamp_tol` caps that distance at 1e-3 — 35x below the
    extremum — so no admissible tolerance can move a value by enough to
    reach it.  The largest change this function can make to any element is
    ``tol``; the smallest change that would damage the overshoot is
    ``MP1_LOBE_EXTREMUM``.  The one place the clamp does fire inside the
    lobe is the isolated zero crossing at ``x = 0.841882``, where ``f``
    passes through 0 and is already within ``tol`` of it over a window
    ``4.5e-8`` wide in ``x``; setting a value of 1e-9 to 0 there is the
    intent, not a violation of it.

    NOT APPLIED TO ``-df/dE``.  :func:`mp1_negative_derivative` is a
    different quantity with different consumers (the Fermi-surface weight),
    and this key says nothing about it.  Deliberate, not an oversight.

    ON AN INSULATING TABLE THIS IS A NO-OP BY CONSTRUCTION: every entry is
    already exactly 0.0 or 1.0, so both ``where`` predicates select the
    identity branch at every element.

    Traced-safe: pure ``jnp`` on float64, no host readback, so it composes
    inside the fixed-N root as well as after it.
    """
    f = jnp.asarray(f, dtype=jnp.float64)
    tol = jnp.asarray(clamp_tol, dtype=jnp.float64)
    f = jnp.where(jnp.abs(f) < tol, 0.0, f)
    return jnp.where(jnp.abs(1.0 - f) < tol, 1.0, f)


# Tolerance for "the cumulative occupancy lands exactly on the target",
# i.e. the gapped case.  Scaled by the target so it is relative.
_EXACT_FILL_RTOL = 1e-9

# Fixed bracket/iteration counts keep one lowering for all per-iteration
# broadening values at fixed (nk, nb).  At either endpoint |x| >= 8, so the
# MP1 tail is below 1e-27 in float64.
_MP1_BRACKET_WIDTHS = 16.0
_MP1_BISECTION_STEPS = 64


def _mp1_values(E, chemical_potential, broadening, clamp_tol):
    """BerkeleyGW v4 MP1 formula on float64 JAX operands, tail-clamped.

    THE CLAMP IS APPLIED HERE, AT THE POINT OF EVALUATION, and therefore
    inside the fixed-N root as well: ``_solve_mp1_kernel``'s ``count(mu)``
    calls this function, so the mu it finds is the one that makes the
    CLAMPED table hit the electron target.  Clamping the table after the
    root instead would round the tail and then fail
    :func:`assert_fixed_n` — the invariant would be asserted against a
    table nobody solved for.  See :func:`clamp_occupation_tail`.
    """
    x = (E - chemical_potential) / (2.0 * broadening)
    gaussian = jnp.exp(-jnp.square(x))
    # For an absurdly small width, avoid leaking the limiting inf*0 as NaN.
    correction = jnp.where(jnp.isfinite(x), x * gaussian, 0.0)
    return clamp_occupation_tail(
        0.5 * (1.0 - jax.lax.erf(x)) - correction / (2.0 * jnp.sqrt(jnp.pi)),
        clamp_tol)


def _mp1_negative_derivative_values(E, chemical_potential, broadening):
    """``-df/dE`` for BerkeleyGW's first-order MP occupation."""
    x = (E - chemical_potential) / (2.0 * broadening)
    gaussian = jnp.exp(-jnp.square(x))
    return ((1.5 - jnp.square(x)) * gaussian
            / (2.0 * broadening * jnp.sqrt(jnp.pi)))


@jax.jit
def _mp1_occupations_kernel(E_kn, chemical_potential, broadening_ry,
                            clamp_tol) -> jax.Array:
    return _mp1_values(
        jnp.asarray(E_kn, dtype=jnp.float64),
        jnp.asarray(chemical_potential, dtype=jnp.float64),
        jnp.asarray(broadening_ry, dtype=jnp.float64),
        jnp.asarray(clamp_tol, dtype=jnp.float64))


def mp1_occupations(E_kn, chemical_potential, broadening_ry,
                    clamp_tol=OCCUPATION_CLAMP_TOL_DEFAULT) -> jax.Array:
    """First-order MP occupations with BerkeleyGW width semantics.

    ``broadening_ry`` is ``occ_broadening`` converted to Ry; the
    argument is ``(E-mu)/(2*broadening_ry)``.  MP1's small overshoot beyond
    [0, 1] is deliberately retained; only the FAR TAIL is snapped to exact
    0/1, at ``clamp_tol`` (:func:`clamp_occupation_tail`).  Pass
    ``clamp_tol=0.0`` for the unclamped table.
    """
    return _mp1_occupations_kernel(
        E_kn, chemical_potential, broadening_ry,
        occupation_clamp_tol(clamp_tol))


@jax.jit
def mp1_negative_derivative(
    E_kn, chemical_potential, broadening_ry,
) -> jax.Array:
    """Return the exact MP1 Fermi-surface weight ``-df/dE`` in Ry^-1.

    First-order Methfessel-Paxton is not a positive distribution: its small
    negative side lobes are part of the configured quadrature and are kept.
    The width convention is identical to :func:`mp1_occupations`.
    """
    return _mp1_negative_derivative_values(
        jnp.asarray(E_kn, dtype=jnp.float64),
        jnp.asarray(chemical_potential, dtype=jnp.float64),
        jnp.asarray(broadening_ry, dtype=jnp.float64))


@jax.jit
def _solve_mp1_kernel(E, w, target, broadening, capacity, clamp_tol):
    """Pure fixed-shape device root plus final occupations.

    ONE ``_mp1_values`` FOR BOTH THE COUNT AND THE TABLE, so the clamp is
    inside the root: mu is chosen to make the clamped table hit ``target``.
    """
    E = jnp.asarray(E, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    target = jnp.asarray(target, dtype=jnp.float64)
    broadening = jnp.asarray(broadening, dtype=jnp.float64)
    capacity = jnp.asarray(capacity, dtype=jnp.float64)
    clamp_tol = jnp.asarray(clamp_tol, dtype=jnp.float64)
    tail = _MP1_BRACKET_WIDTHS * broadening
    bracket0 = (jnp.min(E) - tail, jnp.max(E) + tail)

    def count(mu):
        return capacity * jnp.einsum(
            "k,kn->", w, _mp1_values(E, mu, broadening, clamp_tol))

    def bisect(_iteration, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        below = count(mid) < target
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid))

    # MP1 is locally non-monotone near a level.  Opposite-sign bisection is
    # safeguarded against that derivative; a bare Newton step is not.
    lo, hi = jax.lax.fori_loop(
        0, _MP1_BISECTION_STEPS, bisect, bracket0)
    mu = 0.5 * (lo + hi)
    return mu, _mp1_values(E, mu, broadening, clamp_tol)


def solve_mp1_occupations(
    E_kn, kweights, n_electrons: float, broadening_ry: float, *,
    state_capacity: float,
    clamp_tol: float = OCCUPATION_CLAMP_TOL_DEFAULT,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mu_ry, f_kn)`` satisfying the fixed-electron constraint.

    Solves ``state_capacity * sum_k w_k sum_n f_kn = n_electrons``.
    ``kweights`` are normalized weights for the supplied k-set (normalized
    star multiplicities on an IBZ).  A fully relativistic state has
    ``state_capacity=1``; a restricted scalar state has 2.  Use the
    canonical ``spin_degeneracy_factor(wfn)``.

    ``clamp_tol`` snaps the far tail to exact 0/1 INSIDE the root, so the
    returned ``(mu, f_kn)`` pair satisfies the fixed-electron constraint on
    the clamped table (:func:`clamp_occupation_tail`).  ``0.0`` disables it.

    Validation is host-only and reads no eigenvalues.  Root and occupations
    are one fixed-iteration JAX kernel with no Python loop or scalar readback.
    """
    shape = tuple(int(s) for s in np.shape(E_kn))
    w = np.asarray(kweights, dtype=np.float64)
    if len(shape) != 2 or min(shape, default=0) < 1:
        raise ValueError(
            f"solve_mp1_occupations: E_kn must be nonempty (nk, nb); got {shape}")
    if w.shape != (shape[0],):
        raise ValueError(
            f"solve_mp1_occupations: kweights must be ({shape[0]},); got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("solve_mp1_occupations: kweights must be finite/nonnegative")
    weight_sum = float(w.sum())
    if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"solve_mp1_occupations: kweights must sum to 1; got {weight_sum:.17g}")

    target = float(n_electrons)
    broadening = float(broadening_ry)
    capacity = float(state_capacity)
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("solve_mp1_occupations: broadening_ry must be finite and > 0")
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("solve_mp1_occupations: state_capacity must be finite and > 0")
    maximum = capacity * weight_sum * shape[1]
    if not np.isfinite(target) or not (0.0 < target < maximum):
        raise ValueError(
            f"solve_mp1_occupations: n_electrons={target!r} outside (0, {maximum})")

    return _solve_mp1_kernel(E_kn, w, target, broadening, capacity,
                             occupation_clamp_tol(clamp_tol))


@jax.jit
def _fermi_level_step_kernel(E, w, target):
    """``(E_F, degenerate, E_boundary)`` as one f64 3-vector.

    Both branches are computed and selected with ``jnp.where``: each is a
    handful of scalar ops on already-sorted data, so evaluating both beats
    any control flow.  Out-of-range indices are clamped explicitly next to
    the ``where`` that discards them.
    """
    E = jnp.asarray(E, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    target = jnp.asarray(target, dtype=jnp.float64)
    nb = int(E.shape[1])

    # One weight per (k, n) state, then sort every state by energy.
    # ``jnp.argsort`` is stable, matching the host form's kind="stable":
    # degenerate states keep k-major order, which is what makes the
    # boundary comparison below reproducible.
    w_state = jnp.broadcast_to(w[:, None], E.shape).reshape(-1)
    flat = E.reshape(-1)
    order = jnp.argsort(flat)
    E_sorted = flat[order]
    cum = jnp.cumsum(w_state[order])
    size = int(E_sorted.shape[0])

    # First state at which the running occupancy reaches the target.  THE
    # SEARCH IS AT ``target − tol``, NOT AT ``target``: in the gapped case
    # ``cum`` lands on the target to round-off and which side it lands on
    # depends on the summation order, i.e. on the backend.  One ulp above
    # and ``searchsorted`` returns the next index, where the exact-fill
    # test compares against a partial sum a whole state too high, fails,
    # and the metallic branch puts E_F ~1e-14 into the gap — at the VBM,
    # so ``E < E_F`` drops one band per k from ρ.  Searching at
    # ``target − tol`` removes the cliff without a second branch: any
    # ``cum[j]`` in ``[target − tol, target)`` is the exact-fill state,
    # and if none is there both searches return the same index.
    tol = _EXACT_FILL_RTOL * jnp.maximum(target, 1.0)
    i = jnp.searchsorted(cum, target - tol, side="left")
    over = i >= size
    i_c = jnp.clip(i, 0, size - 1)
    exact = jnp.abs(cum[i_c] - target) <= tol

    # A STEP OCCUPATION CANNOT SPLIT A DEGENERATE MANIFOLD.  E_F has to be
    # placeable strictly between the highest occupied and the lowest
    # unoccupied energy; if the boundary falls inside a set of states that
    # share an energy, ``E < E_F`` takes all of them or none.  Flagged for
    # the caller to refuse — that case needs fractional occupations.
    # Because it is refused, ``nxt`` is the next DISTINCT energy whenever
    # the result is used, which is what the exact-fill midpoint needs.
    boundary = jnp.where(exact, i_c, i_c - 1)
    nxt = boundary + 1
    b_c = jnp.clip(boundary, 0, size - 1)
    n_c = jnp.clip(nxt, 0, size - 1)
    degenerate = ((boundary >= 0) & (nxt < size)
                  & (E_sorted[n_c] == E_sorted[b_c]) & jnp.logical_not(over))

    # EXACT FILL (gapped).  State i is the last occupied one, so E_F must
    # sit ABOVE it: midpoint to the next distinct energy.
    e_exact = jnp.where(nxt >= size,
                        E_sorted[i_c] + 1.0,
                        0.5 * (E_sorted[i_c] + E_sorted[n_c]))

    # PARTIAL FILL (metal).  The step at state i carries the count across
    # the target; interpolate the ENERGY across that step.
    #
    # The realised occupancy is then ``cum[i-1]``, which is BELOW the
    # target by up to one state's weight — a step function can only
    # realise counts that are partial sums of ``kweights``.  That is
    # inherent, not a defect here, and it is why a metal needs fractional
    # occupations.  Callers that care must check with
    # :func:`occupied_band_count`; the shortfall is bounded by
    # ``max(kweights)``.
    im1 = jnp.clip(i_c - 1, 0, size - 1)
    has_below = i_c > 0
    cum_below = jnp.where(has_below, cum[im1], 0.0)
    E_below = jnp.where(has_below, E_sorted[im1], E_sorted[0])
    span = cum[i_c] - cum_below
    frac = jnp.where(span <= 0.0, 0.0, (target - cum_below) / span)
    e_partial = E_below + frac * (E_sorted[i_c] - E_below)

    # Every state occupied and still short — caught by the caller's range
    # check except for round-off; return just above the top.
    e_f = jnp.where(over, E_sorted[size - 1] + 1.0,
                    jnp.where(exact, e_exact, e_partial))
    return jnp.stack([e_f, degenerate.astype(jnp.float64), E_sorted[b_c]])


def fermi_level_step(E_kn, kweights, n_occ_bands: float) -> float:
    """E_F for step (T = 0) occupations, k-weighted.

    Parameters
    ----------
    E_kn : (nk, nb) real
        Eigenvalues, on a device or on the host.  A ``jax.Array`` is not
        read back; only the 3-vector of scalars crosses.  Any consistent
        energy unit; E_F comes back in it.
    kweights : (nk,) real
        k-weights of the SAME k-set as ``E_kn``, summing to 1 over the
        set (the IBZ convention ``WfnLoader.kweights`` uses).  A weighted
        count is the whole reason this is not ``E_sorted[n_occ]``.  Host
        numpy.
    n_occ_bands : float
        Target Σ_k w_k Σ_n f_nk — a BAND count, see the module note.

    Returns
    -------
    float
        Two regimes, and the distinction is not cosmetic:

        * **Partial fill (metal).**  The cumulative occupancy steps ACROSS
          the target at some state.  E_F is linearly interpolated between
          the last state below the target and the first at or above it,
          weighted by how far into that step the target falls.
        * **Exact fill (gapped).**  The cumulative occupancy lands ON the
          target — the target is reached exactly at the VBM.  Returning
          the VBM energy would then make ``E < E_F`` EXCLUDE the VBM and
          undercount by one band, so E_F is placed at the midpoint of the
          VBM and the next state up (the CBM).  That is also the
          convention ``sc_iteration._midgap_efermi`` uses
          (``0.5·(vbm + cbm)``), so the two agree on the case they share.

    Raises
    ------
    ValueError
        When the target falls inside a DEGENERATE MANIFOLD.  No step
        occupation can realise it — ``E < E_F`` takes every state at that
        energy or none — so returning a number would only show up later as
        a wrong electron count.  That case needs fractional occupations.
        Detected inside the kernel and raised here; see the module note.

    ON EXACTNESS.  In the gapped case the realised count equals the target
    exactly.  In the metallic case it does NOT and cannot: a step function
    realises only partial sums of ``kweights``, so the count lands at the
    last partial sum below the target, short by less than
    ``max(kweights)``.  Check with :func:`occupied_band_count` when it
    matters; this is the precise sense in which step occupations are a
    placeholder for the Fermi–Dirac treatment.
    """
    # Shape and range checks stay on the host: shapes are static and
    # ``kweights`` is host numpy, so refusing here reports the numbers
    # rather than a flag to decode.
    shape = tuple(int(s) for s in np.shape(E_kn))
    w = np.asarray(kweights, dtype=np.float64)
    if len(shape) != 2:
        raise ValueError(f"fermi_level_step: E_kn must be (nk, nb); got {shape}")
    if w.shape != (shape[0],):
        raise ValueError(
            f"fermi_level_step: kweights must be (nk,) = ({shape[0]},) to "
            f"match E_kn; got {w.shape}.  Passing full-BZ weights with IBZ "
            f"energies (or the reverse) silently rescales the electron count.")
    target = float(n_occ_bands)
    if not (0.0 < target <= float(w.sum()) * shape[1] + _EXACT_FILL_RTOL):
        raise ValueError(
            f"fermi_level_step: n_occ_bands={target} outside (0, "
            f"sum(w)*nb={float(w.sum()) * shape[1]}]; there are not enough "
            f"bands to hold it.")

    e_f, degenerate, e_boundary = (
        float(v) for v in np.asarray(_fermi_level_step_kernel(E_kn, w, target)))
    if degenerate != 0.0:
        raise ValueError(
            f"fermi_level_step: n_occ_bands={target} puts E_F inside a "
            f"degenerate manifold at E={e_boundary!r} — the "
            f"target is reached partway through a set of states that share "
            f"an energy, so no step occupation can realise it (E < E_F takes "
            f"all of them or none).  This needs fractional occupations.")
    return e_f


def step_occupations(E_kn, e_fermi) -> jax.Array:
    """``f_nk = 1.0 if E_nk < E_F else 0.0``, as float64.

    float64 rather than a boolean or an integer BY DESIGN: these are the
    per-state weights ρ contracts against, and the full finite-temperature
    treatment replaces this function's body with a Fermi–Dirac factor
    without any consumer changing.  Keeping the dtype and the shape fixed
    now is what makes that a one-function change later.

    Returns a ``jax.Array`` so the occupations feed
    ``qsgw_density.rho_from_wfns`` without a round trip; ``e_fermi`` may be
    a Python float or a device scalar.

    Strict ``<`` is correct given :func:`fermi_level_step`, which never
    returns an energy equal to an occupied state's: the exact-fill branch
    places E_F strictly between the VBM and the CBM.
    """
    return (jnp.asarray(E_kn, dtype=jnp.float64)
            < jnp.asarray(e_fermi, dtype=jnp.float64)).astype(jnp.float64)


@dataclass(frozen=True)
class OccupationState:
    """The one per-iteration occupation record every metal consumer reads.

    Frozen cross-agent contract (ARCHITECTURE.md, metal_mpa_plan_2026-08-15):
    chi body, head, Σ_x/Hartree, Σ_c and the provenance stamps all consume
    THIS object, so (f, μ, family, width) never thread loose through five
    signatures — that is the shadow-accounting failure class.  It owns the
    fixed-N invariant: both constructors assert the realised electron count
    against the target (``assert_fixed_n``), and ``__post_init__`` binds
    ``occ_hash`` to the bytes of ``f_kn`` so a stamp can never describe a
    table it was not computed from.

    ``f_kn`` is NEVER clipped — MP1's overshoot beyond [0, 1] is part of the
    configured quadrature (see :func:`mp1_occupations`).  The far tail IS
    snapped to exact 0/1 (:func:`clamp_occupation_tail`), which is a
    different operation on a disjoint part of the table: the clamp happens
    inside :func:`solve_mp1_occupations`'s root, BEFORE this object exists,
    so ``occ_hash`` and the fixed-N assertion both describe the clamped
    table and no consumer sees a value the solver did not solve for.
    """

    f_kn: jax.Array          # (nk, nb), float64, never clipped
    mu_ry: float             # fixed-N chemical potential, or step E_F
    smearing_family: str     # "mp1" | "fixed" ("fixed" = insulating step)
    smearing_width_ry: float # broadening_ry; 0.0 for "fixed"
    n_electrons: float       # physical electron target (capacity-weighted)
    occ_hash: str = ""       # sha256[:16] of f_kn bytes; derived, see below

    def __post_init__(self):
        f = np.asarray(self.f_kn, dtype=np.float64)
        if f.ndim != 2 or min(f.shape) < 1:
            raise ValueError(
                f"OccupationState: f_kn must be nonempty (nk, nb); got {f.shape}")
        if not np.all(np.isfinite(f)):
            raise ValueError("OccupationState: f_kn must be finite")
        if self.smearing_family not in ("mp1", "fixed"):
            raise ValueError(
                "OccupationState: smearing_family must be 'mp1' or 'fixed'; "
                f"got {self.smearing_family!r}")
        width = float(self.smearing_width_ry)
        if self.smearing_family == "fixed" and width != 0.0:
            raise ValueError(
                f"OccupationState: family 'fixed' requires width 0.0; got {width!r}")
        if self.smearing_family == "mp1" and not (np.isfinite(width) and width > 0.0):
            raise ValueError(
                f"OccupationState: family 'mp1' requires width > 0; got {width!r}")
        if not np.isfinite(self.mu_ry):
            raise ValueError(f"OccupationState: mu_ry must be finite; got {self.mu_ry!r}")
        digest = hashlib.sha256(
            np.ascontiguousarray(f).tobytes()).hexdigest()[:16]
        if self.occ_hash and self.occ_hash != digest:
            raise ValueError(
                "OccupationState: supplied occ_hash does not match f_kn "
                f"(got {self.occ_hash!r}, recomputed {digest!r})")
        object.__setattr__(self, "occ_hash", digest)

    @classmethod
    def solve_mp1(cls, E_kn, kweights, n_electrons: float, width_ry: float, *,
                  state_capacity: float,
                  clamp_tol: float = OCCUPATION_CLAMP_TOL_DEFAULT,
                  ) -> "OccupationState":
        """Fixed-N MP1 state via :func:`solve_mp1_occupations` (one solver).

        ``state_capacity`` follows the module convention (1 for a fully
        relativistic spinor state, 2 for a restricted scalar one).  The
        frozen interface sketch omitted it; without it the fixed-N invariant
        is wrong by a factor of 2 on scalar decks.

        ``clamp_tol`` is applied inside the root, so ``assert_fixed_n``
        below tests the table this object actually carries.
        """
        mu, f = solve_mp1_occupations(
            E_kn, kweights, float(n_electrons), float(width_ry),
            state_capacity=float(state_capacity),
            clamp_tol=float(clamp_tol))
        state = cls(f_kn=f, mu_ry=float(mu), smearing_family="mp1",
                    smearing_width_ry=float(width_ry),
                    n_electrons=float(n_electrons))
        assert_fixed_n(state, kweights, state_capacity=float(state_capacity))
        return state

    @classmethod
    def step(cls, E_kn, kweights, n_occ_bands: float, *,
             state_capacity: float = 1.0) -> "OccupationState":
        """Insulating step state via the existing E_F/step machinery.

        Refuses the metallic partial-fill case by name: a step function
        realises only partial sums of the k-weights, so an inexact fill
        means the deck needs :meth:`solve_mp1`, not a silently short count.
        """
        e_f = fermi_level_step(E_kn, kweights, float(n_occ_bands))
        f = step_occupations(E_kn, e_f)
        realized = occupied_band_count(f, kweights)
        target = float(n_occ_bands)
        tol = _EXACT_FILL_RTOL * max(target, 1.0)
        if abs(realized - target) > tol:
            raise ValueError(
                f"OccupationState.step: step occupations realise "
                f"{realized:.12f} of the {target:.12f} requested bands "
                f"(short by {target - realized:.3e}) — a partial fill means "
                "this is a metal; use OccupationState.solve_mp1.")
        state = cls(f_kn=f, mu_ry=float(e_f), smearing_family="fixed",
                    smearing_width_ry=0.0,
                    n_electrons=float(state_capacity) * target)
        assert_fixed_n(state, kweights, state_capacity=float(state_capacity),
                       atol=float(state_capacity) * tol)
        return state


def assert_fixed_n(state: OccupationState, kweights, *,
                   state_capacity: float, atol: float = 1e-10) -> float:
    """The fixed-N invariant: capacity-weighted count equals the target.

    Separate from ``__post_init__`` because the invariant needs ``kweights``
    and ``state_capacity``, which are constructor-scope, not state fields.
    Padding ``f_kn`` with exact-zero bands (the parallel-transport storage
    convention) does not move the count, so padded tables assert too.
    Returns the realised count.
    """
    realized = float(state_capacity) * occupied_band_count(state.f_kn, kweights)
    if abs(realized - float(state.n_electrons)) > atol:
        raise ValueError(
            f"OccupationState fixed-N invariant violated: realised "
            f"{realized:.12f} electrons against target "
            f"{state.n_electrons:.12f} (|Δ| = "
            f"{abs(realized - state.n_electrons):.3e} > {atol:.1e}).  "
            "Mismatched kweights/k-set is the usual cause.")
    return realized


def assert_wfn_occupation_consistency(
    state: OccupationState, wfn_occ_kn, kweights, *,
    state_capacity: float, num_electrons: float,
    occ_atol: float = 1e-6, nelec_atol: float = 1e-8,
) -> float:
    """LORRAX↔WFN smearing consistency at metal-run startup.

    Recomputes the electron count from the solved state and compares the
    solved occupations against the WFN's OWN stored table over the shared
    band window.  A family/width mismatch (the classic one: passing QE's
    ``degauss`` where the BerkeleyGW convention wants ``degauss/2``, module
    header) shows up here as an occupation deviation far above round-off.
    Deliberate deviation from the plan's letter: ``wfn.efermi`` is derived
    by the midgap/VBM convention in the loader and cannot equal a smeared
    metallic μ to 1e-6 Ry even when everything is right, so μ is reported
    by the caller's log rather than asserted against that field.
    Returns the max occupation deviation.
    """
    realized = float(state_capacity) * occupied_band_count(state.f_kn, kweights)
    if abs(realized - float(num_electrons)) > nelec_atol:
        raise ValueError(
            f"occupation consistency: realised {realized:.12f} electrons vs "
            f"WFN num_electrons {float(num_electrons):.12f} "
            f"(|Δ| > {nelec_atol:.1e})")
    ours = np.asarray(state.f_kn, dtype=np.float64)
    theirs = np.asarray(wfn_occ_kn, dtype=np.float64)
    if ours.shape[0] != theirs.shape[0]:
        raise ValueError(
            f"occupation consistency: k-set mismatch {ours.shape[0]} vs "
            f"{theirs.shape[0]}")
    nb = min(ours.shape[1], theirs.shape[1])
    deviation = float(np.max(np.abs(ours[:, :nb] - theirs[:, :nb])))
    if deviation > occ_atol:
        raise ValueError(
            f"occupation consistency: solved occupations deviate from the "
            f"WFN's stored table by {deviation:.3e} (> {occ_atol:.1e}) over "
            f"the first {nb} bands.  Check the smearing family and width — "
            "matching QE 'mp' uses broadening_ry = degauss/2 (module header).")
    return deviation


def occupied_band_count(occ_kn, kweights) -> float:
    """``Σ_k w_k Σ_n f_nk`` — the number :func:`fermi_level_step` targets.

    The round-trip check: feeding this the occupations built from
    ``fermi_level_step``'s output must return the requested count.  Cheap
    enough to assert on every self-consistent iteration, and it is the
    one number that catches a k-weight set that does not match the k-set.
    The contraction runs where ``occ_kn`` lives; only the scalar crosses.
    """
    return float(jnp.einsum("k,kn->",
                            jnp.asarray(kweights, dtype=jnp.float64),
                            jnp.asarray(occ_kn, dtype=jnp.float64)))
