"""Three-way band partition for QSGW: protected / non-protected-in-range / out-of-range.

The QSGW iteration map carries ``H_qp_dft`` over the **active subspace**
(``band_slices.sigma`` of the wfn bundle).  Within that subspace, each
band falls into one of three categories per the user-configured
:class:`BandPartition`:

================================  ===========================  =======================================
Category                          ``protected_mask`` element   Diagonal of ``H_qp_dft`` per iteration
================================  ===========================  =======================================
Protected                         ``True``                     Full Σ at QP energy (off-diag mixed in)
Non-protected, in ω-range         ``False``, ``in_range=True`` Diagonal Σ at actual band energy
Non-protected, out of ω-range     ``False``, ``in_range=False`` Scissor extrapolation α·E_DFT + β
================================  ===========================  =======================================

Off-diagonals of ``H_qp_dft`` are kept **only** for protected×protected
pairs.  All other off-diagonals are zeroed each iteration so the
non-protected / out-of-range bands never mix into the protected
subspace's eigenproblem.

Why pre-compute the masks
-------------------------
Both masks are static across iterations (band identity doesn't change)
and small (``nb_active`` booleans each), so they live in :class:`BandPartition`
and are passed as JAX arrays to a single jit'd primitive
:func:`apply_band_partition` that the iteration map calls per step.

Validation
----------
:func:`BandPartition.warn_if_protected_outside_grid` prints a strong
all-caps warning if any protected band is outside the ω-grid — that
would mean the off-diagonal mixing pulls in Σ values that were
extrapolated by edge-clamping rather than properly evaluated, which is
exactly what the partition was designed to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Partition descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandPartition:
    """How each band in the active subspace is treated by the QSGW H build.

    Both masks are 1-D over the active band axis ``nb_active`` and are
    constant across SC iterations.

    Attributes
    ----------
    protected_mask : (nb_active,) bool
        True for bands that get full off-diagonal Σ corrections in
        ``H_qp_dft`` and participate in the basis rotation.
    in_range_mask : (nb_active,) bool
        True for bands whose ``E_DFT`` lies inside ``[ω_min, ω_max]`` at
        every k.  Used to decide between Σ_diag (in-range) and scissor
        (out-of-range) for the *non-protected* bands' diagonal.
    """

    protected_mask: jax.Array
    in_range_mask: jax.Array

    @classmethod
    def all_protected(cls, nb_active: int) -> "BandPartition":
        """Default: every active band is protected, every band in range.
        ``apply_band_partition`` reduces to identity in this case (no
        change to current behaviour).  Used as the default in
        :class:`sc_iteration.SCInputs` so existing code paths are
        unaffected until the partition is configured deliberately."""
        ones = jnp.ones(nb_active, dtype=bool)
        return cls(protected_mask=ones, in_range_mask=ones)

    def report_multiplet_splits(self, enk_full_ry, band_offset, *,
                                label="SC", print_fn=print):
        """Which protected-mask boundaries cut a DEGENERATE MULTIPLET.

        Returns ``(n_split, worst_gap_mev)`` and prints one line per split.

        WHY THIS EXISTS.  ``protected_mask`` comes from
        ``scissor.classify_bands_in_grid``, an ALL-k energy-window predicate,
        so it is not required to be contiguous and its edges are not required
        to fall between multiplets.  Bands degenerate at one k need not be
        degenerate at another, so band ``n`` can be in-grid while its
        multiplet partner ``n+1`` is not — and then
        :func:`apply_band_partition` gives one member full off-diagonal Σ and
        the other a scalar scissor.  Half a multiplet is not a subspace of
        anything: within a degenerate manifold the band label is arbitrary, so
        treating members differently makes the answer depend on an
        eigensolver's ordering, and the result is ``eigh``'d and reported as
        QP energies.

        MEASURED ELSEWHERE ON THIS TREE, which is why this is not theoretical:
        `si_cohsex_debug`'s ζ/Σ band edge slices at a **0.000000 meV** gap,
        and moving it to a clean edge takes the within-star Σ spread from
        ~2 meV to **exactly 0.0000**.

        ``enk_full_ry`` MUST BE THE UNTRUNCATED LADDER, not the active window.
        ``boundary_min_gaps`` returns ``+inf`` at ``b = nb`` by construction,
        so handed a window it cannot see the very edge that made it — the
        whole reason that function now demands ``is_full_spectrum``.
        ``band_offset`` maps active index 0 onto its absolute band.

        REPORT ONLY.  It changes nothing; :meth:`promoted_to_multiplets` is
        the change.
        """
        from common.band_degeneracy import (DEGENERACY_TOL_RY,
                                            boundary_min_gaps)

        mask = np.asarray(self.protected_mask, dtype=bool)
        e = np.asarray(enk_full_ry, dtype=np.float64)
        gaps = boundary_min_gaps(e, is_full_spectrum=True)
        edges = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
        splits = []
        for a in edges:
            b = int(a) + int(band_offset)
            if 0 < b < gaps.size - 1 and gaps[b] <= DEGENERACY_TOL_RY:
                splits.append((int(a), b, float(gaps[b])))
        if not splits:
            print_fn(f"  {label} partition: {int(mask.sum())}/{mask.size} protected; "
                     f"no boundary splits a multiplet")
            return 0, 0.0
        _MEV = 13605.693122994
        print_fn("  " + "=" * 70)
        print_fn(f"  !!! {label} partition: {len(splits)} protected-mask boundary/ies "
                 f"CUT A DEGENERATE MULTIPLET")
        for a, b, g in splits:
            print_fn(f"  !!!   active band {a} (absolute {b}): min gap over k "
                     f"= {g * _MEV:.6f} meV")
        print_fn("  !!! Half a multiplet gets off-diagonal Sigma and half a "
                 "scalar scissor;")
        print_fn("  !!! within a multiplet the band label is arbitrary, so "
                 "this makes H_qp")
        print_fn("  !!! depend on the eigensolver's ordering.")
        print_fn("  " + "=" * 70)
        return len(splits), max(g for _, _, g in splits) * _MEV

    def promoted_to_multiplets(self, enk_full_ry, band_offset, *,
                               label="SC", print_fn=print) -> "BandPartition":
        """This partition with ``protected_mask`` grown to WHOLE multiplets.

        Owner ruling, 2026-08-16: *"I want degenerate spaces degenerate in
        LORRAX."*  A protected mask whose edge falls inside a degenerate
        manifold is promoted OUTWARD until it does not — every member of a
        multiplet that has any protected member becomes protected.

        GROWING, NEVER SHRINKING, and the direction is not arbitrary: the
        alternative (dropping the protected member) would remove off-diagonal
        Σ from a band the ω-grid actually covers, which is a loss of physics
        to buy the same invariance.  Growing admits a band whose Σ is
        edge-clamped — visible, and what
        :meth:`warn_if_protected_outside_grid` is for — rather than silently
        discarding a correction that was properly evaluated.

        THIS MOVES EVERY QSGW NUMBER on any deck where a boundary splits, and
        that is the accepted consequence rather than a defect: more bands
        carry full off-diagonal Σ, so ``H_qp`` differs and so does its
        spectrum.  ``in_range_mask`` is NOT promoted — it answers a different
        question (is this band's Σ on the grid) whose answer is per band and
        genuinely not a multiplet property.
        """
        from common.band_degeneracy import (DEGENERACY_TOL_RY,
                                            boundary_min_gaps)

        mask = np.asarray(self.protected_mask, dtype=bool).copy()
        e = np.asarray(enk_full_ry, dtype=np.float64)
        gaps = boundary_min_gaps(e, is_full_spectrum=True)
        nb = mask.size
        # Group the ACTIVE window into multiplets using absolute boundaries.
        out, start = mask.copy(), 0
        for a in range(1, nb + 1):
            b = a + int(band_offset)
            closes = (a == nb) or (b >= gaps.size - 1) or (
                gaps[b] > DEGENERACY_TOL_RY)
            if closes:
                if mask[start:a].any():
                    out[start:a] = True
                start = a
        n_added = int(out.sum() - mask.sum())
        if n_added:
            print_fn(f"  {label} partition: promoted {n_added} band(s) into whole "
                     f"multiplets — protected {int(mask.sum())} -> "
                     f"{int(out.sum())} of {nb} "
                     f"(owner ruling: degenerate spaces stay degenerate)")
        return BandPartition(protected_mask=jnp.asarray(out),
                             in_range_mask=self.in_range_mask)

    def warn_if_protected_outside_grid(self, *, print_fn=print) -> None:
        """Loud all-caps warning if any protected band is outside the
        ω-grid — that breaks the partition's whole purpose."""
        leak = bool(jnp.any(self.protected_mask & ~self.in_range_mask))
        if leak:
            n_leak = int(jnp.sum(self.protected_mask & ~self.in_range_mask))
            print_fn("=" * 72)
            print_fn(
                f"!!! WARNING: {n_leak} PROTECTED BANDS LIE OUTSIDE THE "
                f"Σ_c(ω) GRID — !!!"
            )
            print_fn(
                "!!! THE QSGW H WILL HAVE OFF-DIAGONAL MIXING WITH BANDS "
                "WHOSE Σ IS UNRELIABLY CLAMPED AT THE GRID EDGE.       !!!"
            )
            print_fn(
                "!!! INCREASE sigma_omega_min/max_ev OR REDUCE THE "
                "PROTECTED BAND RANGE.                                  !!!"
            )
            print_fn("=" * 72)


def build_omega_band_partition(
    e_dft_kn_ry,
    e_dft_full_kn_ry,
    *,
    band_offset: int,
    omega_min_abs_ev: float,
    omega_max_abs_ev: float,
    previous_partition: BandPartition | None = None,
    hysteresis_margin_ev: float = 0.0,
    edge_margin_ev: float = 0.0,
    label: str = "SC",
    print_fn=print,
) -> BandPartition:
    """Build the canonical protected/in-range partition for a Sigma grid.

    The window predicate, full-spectrum multiplet audit, outward promotion,
    and protected-outside-grid warning belong together.  Keeping them in one
    constructor prevents an additional QP ladder from quietly using a
    different protected subspace than the main self-consistent driver.
    """
    from common.units import RYD_TO_EV
    from .scissor import classify_bands_in_grid

    e_dft_ev = np.asarray(e_dft_kn_ry, dtype=np.float64) * RYD_TO_EV
    band_in_grid, _ = classify_bands_in_grid(
        e_dft_ev, float(omega_min_abs_ev), float(omega_max_abs_ev))
    edge = float(edge_margin_ev)
    if edge < 0.0 or not np.isfinite(edge):
        raise ValueError("edge_margin_ev must be finite and nonnegative")
    if edge > 0.0:
        # A SELF-CONSISTENT set needs room to move.  A band whose all-k range
        # ends within ``edge`` of the grid edge is inside today and outside
        # after one QP shift; and its degenerate partners, promoted to keep
        # the multiplet whole, can sit far above the grid with a clamped
        # Sigma that then couples into the trusted block.  Measured on Na
        # (86 bands, top +24 eV): band 14 tops out 1.7 eV under the edge, its
        # Gamma partners 15-23 were promoted, and the k=1 triplet 11-13 moved
        # 6.4 eV in one map (runs/Na/12_sc_observables_eta05_2026-09-04,
        # arms G/H).  The margin is the SC state pad the box rules use.
        with_margin, _ = classify_bands_in_grid(
            e_dft_ev, float(omega_min_abs_ev) + edge,
            float(omega_max_abs_ev) - edge)
        excluded = band_in_grid & ~with_margin
        if excluded.any():
            lo_k = e_dft_ev.min(axis=0); hi_k = e_dft_ev.max(axis=0)
            rows = ", ".join(
                f"{int(n) + int(band_offset) + 1} [{lo_k[n]:.2f}, {hi_k[n]:.2f}]"
                for n in np.flatnonzero(excluded))
            print_fn(
                f"  {label} partition: {int(excluded.sum())} band(s) inside "
                f"the grid but within {edge:.1f} eV of an edge are scissored, "
                f"not self-consistent (band [all-k range, eV]: {rows}; grid "
                f"[{float(omega_min_abs_ev):.2f}, {float(omega_max_abs_ev):.2f}])")
        band_in_grid = with_margin
    in_range = jnp.asarray(band_in_grid, dtype=bool)
    protected = np.asarray(band_in_grid, dtype=bool)
    margin = float(hysteresis_margin_ev)
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("hysteresis_margin_ev must be finite and nonnegative")
    if previous_partition is not None and margin > 0.0:
        previous = np.asarray(
            previous_partition.protected_mask, dtype=bool).reshape(-1)
        if previous.shape != protected.shape:
            raise ValueError(
                "previous partition and current spectrum have different "
                f"band counts ({previous.size} and {protected.size})")
        # Schmitt boundary: a band gains protection at the actual Sigma-grid
        # edge, but one that was already protected is not dropped until every
        # k has crossed an edge by the run-derived deadband.  This keeps an
        # edge band's own off-diagonal mixing from switching the structure of
        # H on and off on alternate maps.
        retained_kn = (
            (e_dft_ev >= float(omega_min_abs_ev) - margin)
            & (e_dft_ev <= float(omega_max_abs_ev) + margin))
        retained = np.all(retained_kn, axis=0)
        protected |= previous & retained
    # A band is in-grid only if EVERY k is in the window, so one stray k
    # discards the whole band and every state it holds inside the window.
    # Report what that costs: on the sodium metal at +-10 eV, bands sitting
    # inside at 90-97% of k took 244 usable (k, band) states down to 58.
    # Silence here reads as "the window is too narrow" when the real answer
    # may be "the window is fine and the all-k rule is throwing it away".
    state_in = np.asarray(
        (e_dft_ev >= float(omega_min_abs_ev))
        & (e_dft_ev <= float(omega_max_abs_ev)), dtype=bool)
    kept = int(band_in_grid.sum()) * int(state_in.shape[0])
    print_fn(
        f"  {label} partition: protected/in-range = "
        f"{int(band_in_grid.sum())}/{int(band_in_grid.size)} bands "
        f"({int(state_in.sum())}/{int(state_in.size)} (k,band) states lie in "
        f"the window; the all-k rule keeps {kept})")
    partition = BandPartition(
        protected_mask=jnp.asarray(protected), in_range_mask=in_range)
    partition.report_multiplet_splits(
        np.asarray(e_dft_full_kn_ry, dtype=np.float64), int(band_offset),
        label=label, print_fn=print_fn)
    partition = partition.promoted_to_multiplets(
        np.asarray(e_dft_full_kn_ry, dtype=np.float64), int(band_offset),
        label=label, print_fn=print_fn)
    # Promotion may deliberately grow across the grid edge.  Warn on the
    # partition that will actually ship, not only on its pre-promotion seed.
    partition.warn_if_protected_outside_grid(print_fn=print_fn)
    return partition


# ---------------------------------------------------------------------------
# Per-iteration mask primitive
# ---------------------------------------------------------------------------

@jax.jit
def apply_band_partition(
    H_full: jax.Array,
    *,
    protected_mask: jax.Array,
    in_range_mask: jax.Array,
    scissor_E_qp_kn: jax.Array,
) -> jax.Array:
    """Apply the three-way band partition to a full QSGW Hamiltonian.

    Parameters
    ----------
    H_full : (nk, nb_active, nb_active) complex
        The full QSGW H = ``kin_ion + V_H + Σ_xc`` in the DFT basis,
        as if every band were protected.
    protected_mask, in_range_mask : (nb_active,) bool
        See :class:`BandPartition`.
    scissor_E_qp_kn : (nk, nb_active) real
        Scissor-extrapolated QP energies E_QP = α·E_DFT + β computed by
        the caller.  Used as the diagonal of ``H_partitioned`` for
        out-of-range bands; ignored for in-range bands.  Pass zeros if
        no scissor is in play (in-range ≡ all bands).

    Returns
    -------
    H_partitioned : (nk, nb_active, nb_active) complex
        ``H_full`` with:
          - off-diagonals zeroed for any (m, n) where m or n is not protected;
          - diagonals replaced by ``scissor_E_qp_kn`` for non-protected
            out-of-range bands; otherwise kept from ``H_full``.

    Identity case
    -------------
    When all bands are protected and in range
    (``BandPartition.all_protected``), this returns ``H_full`` unchanged.
    """
    nk, nb, _ = H_full.shape
    eye = jnp.eye(nb, dtype=H_full.dtype)

    # Off-diagonal keep-mask: 1 where both m and n are protected, else 0.
    p = protected_mask.astype(H_full.dtype)
    offdiag_keep = p[:, None] * p[None, :]
    # Zero the off-diagonal portion outside protected×protected.
    offdiag_part = H_full * (1.0 - eye) * offdiag_keep[None, :, :]

    # The protected class owns its full diagonal even after multiplet
    # promotion across the grid edge. Only the third class is scissored;
    # this is the same protected | in_range set used by convergence.
    diag_full = jnp.diagonal(H_full, axis1=1, axis2=2)            # (nk, nb)
    diag_kept = jnp.where(
        (protected_mask | in_range_mask)[None, :], diag_full,
        scissor_E_qp_kn.astype(H_full.dtype),
    )                                                              # (nk, nb)

    # Reassemble: off-diag matrix + diag(diag_kept).
    return offdiag_part + diag_kept[:, :, None] * eye[None, :, :]


__all__ = [
    "BandPartition", "apply_band_partition", "build_omega_band_partition",
]
