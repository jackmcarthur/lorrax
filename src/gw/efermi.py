"""Fermi level and occupations from an eigenvalue set.

Host-side and deliberately unsharded: the input is ``(nk, nb)`` real —
6000 numbers on the MoS₂ 4×4 deck, 10⁵ on a 1000-k Fermi-surface run — so
a sort and a cumulative sum cost microseconds and a distributed version
would be all communication and no arithmetic.

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
does not cancel.  ``WfnLoader.nelec`` is already a band count
(``max(ifmax)``), so it is the right argument to pass unmodified.
"""

from __future__ import annotations

import numpy as np


__all__ = ["fermi_level_step", "step_occupations", "occupied_band_count"]

# Tolerance for "the cumulative occupancy lands exactly on the target",
# i.e. the gapped case.  Scaled by the target so it is relative.
_EXACT_FILL_RTOL = 1e-9


def fermi_level_step(E_kn, kweights, n_occ_bands: float) -> float:
    """E_F for step (T = 0) occupations, k-weighted.

    Parameters
    ----------
    E_kn : (nk, nb) real
        Eigenvalues. Any consistent energy unit; E_F comes back in it.
    kweights : (nk,) real
        k-weights of the SAME k-set as ``E_kn``, summing to 1 over the
        set (the IBZ convention ``WfnLoader.kweights`` uses).  A weighted
        count is the whole reason this is not ``E_sorted[n_occ]``.
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

    ON EXACTNESS.  In the gapped case the realised count equals the target
    exactly.  In the metallic case it does NOT and cannot: a step function
    realises only partial sums of ``kweights``, so the count lands at the
    last partial sum below the target, short by less than
    ``max(kweights)``.  Check with :func:`occupied_band_count` when it
    matters; this is the precise sense in which step occupations are a
    placeholder for the Fermi–Dirac treatment.
    """
    E = np.asarray(E_kn, dtype=np.float64)
    w = np.asarray(kweights, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"fermi_level_step: E_kn must be (nk, nb); got {E.shape}")
    if w.shape != (E.shape[0],):
        raise ValueError(
            f"fermi_level_step: kweights must be (nk,) = ({E.shape[0]},) to "
            f"match E_kn; got {w.shape}.  Passing full-BZ weights with IBZ "
            f"energies (or the reverse) silently rescales the electron count.")
    target = float(n_occ_bands)
    if not (0.0 < target <= float(w.sum()) * E.shape[1] + _EXACT_FILL_RTOL):
        raise ValueError(
            f"fermi_level_step: n_occ_bands={target} outside (0, "
            f"sum(w)*nb={float(w.sum()) * E.shape[1]}]; there are not enough "
            f"bands to hold it.")

    # One weight per (k, n) state, then sort every state by energy.
    w_state = np.repeat(w, E.shape[1])
    order = np.argsort(E.ravel(), kind="stable")
    E_sorted = E.ravel()[order]
    cum = np.cumsum(w_state[order])

    # First state at which the running occupancy reaches the target.
    i = int(np.searchsorted(cum, target, side="left"))
    if i >= E_sorted.size:
        # Every state occupied and still short — caught by the range check
        # above except for round-off; return just above the top.
        return float(E_sorted[-1]) + 1.0

    tol = _EXACT_FILL_RTOL * max(target, 1.0)
    exact = abs(cum[i] - target) <= tol

    # A STEP OCCUPATION CANNOT SPLIT A DEGENERATE MANIFOLD.  E_F has to be
    # placeable strictly between the highest occupied and the lowest
    # unoccupied energy; if the boundary falls inside a set of states that
    # share an energy, no single E_F separates them and ``E < E_F`` either
    # takes the whole manifold or none of it.  Refuse rather than return a
    # number whose only symptom is a wrong electron count.  This is exactly
    # the case that needs fractional occupations.
    boundary = i if exact else i - 1
    nxt = boundary + 1
    if 0 <= boundary < E_sorted.size and nxt < E_sorted.size and \
            E_sorted[nxt] == E_sorted[boundary]:
        raise ValueError(
            f"fermi_level_step: n_occ_bands={target} puts E_F inside a "
            f"degenerate manifold at E={float(E_sorted[boundary])!r} — the "
            f"target is reached partway through a set of states that share "
            f"an energy, so no step occupation can realise it (E < E_F takes "
            f"all of them or none).  This needs fractional occupations.")

    if exact:
        # EXACT FILL (gapped).  State i is the last occupied one, so E_F
        # must sit ABOVE it: midpoint to the next distinct energy.
        if nxt >= E_sorted.size:
            return float(E_sorted[i]) + 1.0
        return 0.5 * (float(E_sorted[i]) + float(E_sorted[nxt]))

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
    cum_below = float(cum[i - 1]) if i > 0 else 0.0
    E_below = float(E_sorted[i - 1]) if i > 0 else float(E_sorted[0])
    span = float(cum[i]) - cum_below
    frac = 0.0 if span <= 0.0 else (target - cum_below) / span
    return E_below + frac * (float(E_sorted[i]) - E_below)


def step_occupations(E_kn, e_fermi: float) -> np.ndarray:
    """``f_nk = 1.0 if E_nk < E_F else 0.0``, as float64.

    float64 rather than a boolean or an integer BY DESIGN: these are the
    per-state weights ρ contracts against, and the full finite-temperature
    treatment replaces this function's body with a Fermi–Dirac factor
    without any consumer changing.  Keeping the dtype and the shape fixed
    now is what makes that a one-function change later.

    Strict ``<`` is correct given :func:`fermi_level_step`, which never
    returns an energy equal to an occupied state's: the exact-fill branch
    places E_F strictly between the VBM and the CBM.
    """
    return (np.asarray(E_kn, dtype=np.float64) < float(e_fermi)).astype(np.float64)


def occupied_band_count(occ_kn, kweights) -> float:
    """``Σ_k w_k Σ_n f_nk`` — the number :func:`fermi_level_step` targets.

    The round-trip check: feeding this the occupations built from
    ``fermi_level_step``'s output must return the requested count.  Cheap
    enough to assert on every self-consistent iteration, and it is the
    one number that catches a k-weight set that does not match the k-set.
    """
    occ = np.asarray(occ_kn, dtype=np.float64)
    w = np.asarray(kweights, dtype=np.float64)
    return float(np.einsum("k,kn->", w, occ))
