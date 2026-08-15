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

WHY THIS IS ITS OWN MODULE.  ``sc_iteration._diagonalize_and_get_efermi``
computes a midgap E_F from a FIXED integer ``n_occ``, i.e. it assumes band
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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp


__all__ = [
    "OccupationState", "assert_fixed_n", "assert_wfn_occupation_consistency",
    "fermi_level_step", "mp1_negative_derivative", "mp1_occupations",
    "occupied_band_count", "solve_mp1_occupations", "step_occupations",
]

# Tolerance for "the cumulative occupancy lands exactly on the target",
# i.e. the gapped case.  Scaled by the target so it is relative.
_EXACT_FILL_RTOL = 1e-9

# Fixed bracket/iteration counts keep one lowering for all per-iteration
# broadening values at fixed (nk, nb).  At either endpoint |x| >= 8, so the
# MP1 tail is below 1e-27 in float64.
_MP1_BRACKET_WIDTHS = 16.0
_MP1_BISECTION_STEPS = 64


def _mp1_values(E, chemical_potential, broadening):
    """BerkeleyGW v4 MP1 formula on float64 JAX operands."""
    x = (E - chemical_potential) / (2.0 * broadening)
    gaussian = jnp.exp(-jnp.square(x))
    # For an absurdly small width, avoid leaking the limiting inf*0 as NaN.
    correction = jnp.where(jnp.isfinite(x), x * gaussian, 0.0)
    return (0.5 * (1.0 - jax.lax.erf(x))
            - correction / (2.0 * jnp.sqrt(jnp.pi)))


def _mp1_negative_derivative_values(E, chemical_potential, broadening):
    """``-df/dE`` for BerkeleyGW's first-order MP occupation."""
    x = (E - chemical_potential) / (2.0 * broadening)
    gaussian = jnp.exp(-jnp.square(x))
    return ((1.5 - jnp.square(x)) * gaussian
            / (2.0 * broadening * jnp.sqrt(jnp.pi)))


@jax.jit
def mp1_occupations(E_kn, chemical_potential, broadening_ry) -> jax.Array:
    """First-order MP occupations with BerkeleyGW width semantics.

    ``broadening_ry`` is ``occ_broadening`` converted to Ry; the
    argument is ``(E-mu)/(2*broadening_ry)``.  MP1's small overshoot beyond
    [0, 1] is deliberately retained.
    """
    return _mp1_values(
        jnp.asarray(E_kn, dtype=jnp.float64),
        jnp.asarray(chemical_potential, dtype=jnp.float64),
        jnp.asarray(broadening_ry, dtype=jnp.float64))


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
def _solve_mp1_kernel(E, w, target, broadening, capacity):
    """Pure fixed-shape device root plus final occupations."""
    E = jnp.asarray(E, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    target = jnp.asarray(target, dtype=jnp.float64)
    broadening = jnp.asarray(broadening, dtype=jnp.float64)
    capacity = jnp.asarray(capacity, dtype=jnp.float64)
    tail = _MP1_BRACKET_WIDTHS * broadening
    bracket0 = (jnp.min(E) - tail, jnp.max(E) + tail)

    def count(mu):
        return capacity * jnp.einsum(
            "k,kn->", w, _mp1_values(E, mu, broadening))

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
    return mu, _mp1_values(E, mu, broadening)


def solve_mp1_occupations(
    E_kn, kweights, n_electrons: float, broadening_ry: float, *,
    state_capacity: float,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mu_ry, f_kn)`` satisfying the fixed-electron constraint.

    Solves ``state_capacity * sum_k w_k sum_n f_kn = n_electrons``.
    ``kweights`` are normalized weights for the supplied k-set (normalized
    star multiplicities on an IBZ).  A fully relativistic state has
    ``state_capacity=1``; a restricted scalar state has 2.  Use the
    canonical ``spin_degeneracy_factor(wfn)``.

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

    return _solve_mp1_kernel(E_kn, w, target, broadening, capacity)


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
          convention ``sc_iteration._diagonalize_and_get_efermi`` uses
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
    configured quadrature (see :func:`mp1_occupations`).
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
                  state_capacity: float) -> "OccupationState":
        """Fixed-N MP1 state via :func:`solve_mp1_occupations` (one solver).

        ``state_capacity`` follows the module convention (1 for a fully
        relativistic spinor state, 2 for a restricted scalar one).  The
        frozen interface sketch omitted it; without it the fixed-N invariant
        is wrong by a factor of 2 on scalar decks.
        """
        mu, f = solve_mp1_occupations(
            E_kn, kweights, float(n_electrons), float(width_ry),
            state_capacity=float(state_capacity))
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
