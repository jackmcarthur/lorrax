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

Masks follow DFT reference identities at each k. The classification is rebuilt
from identity-aligned energies each map, with the same energy-dependent pad
that supports the fixed quadrature windows. Sorted QP columns are a readout
of these identities, not coordinates of the Hamiltonian carry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Partition descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandPartition:
    """How each band in the active subspace is treated by the QSGW H build.

    Masks are per ``(k, DFT identity)``; one-dimensional masks are also
    accepted for callers whose classification is identical at every k.

    Attributes
    ----------
    protected_mask : (nk, nb_active) or (nb_active,) bool
        True for bands that get full off-diagonal Σ corrections in
        ``H_qp_dft`` and participate in the basis rotation.
    in_range_mask : (nk, nb_active) or (nb_active,) bool
        True for identities whose current assigned energy lies inside
        ``[ω_min, ω_max]`` at every k.  Used to decide between Σ_diag (in-range) and scissor
        (out-of-range) for the *non-protected* bands' diagonal.
    """

    protected_mask: jax.Array
    in_range_mask: jax.Array
    # An escaping identity keeps its sampled Sigma on this map, then drops
    # on the next accepted map. Trials never commit this memory.
    escaped_mask: jax.Array | None = None
    changed: bool = False

    @classmethod
    def all_protected(cls, nb_active: int) -> "BandPartition":
        """Default: every active band is protected, every band in range.
        ``apply_band_partition`` reduces to identity in this case (no
        change to current behaviour).  Used as the default in
        :class:`sc_iteration.SCInputs` so existing code paths are
        unaffected until the partition is configured deliberately."""
        ones = jnp.ones(nb_active, dtype=bool)
        return cls(protected_mask=ones, in_range_mask=ones)

    def summary(self, energies_ev, indices_kn, *, band_offset, mu_ev):
        """One bounded production line; per-state detail belongs to debug."""
        from .scissor import sc_state_pad_ev

        energies = np.asarray(energies_ev)
        prot = np.broadcast_to(np.asarray(self.protected_mask), energies.shape)
        inr = np.broadcast_to(np.asarray(self.in_range_mask), energies.shape)
        inverse = np.argsort(indices_kn, axis=1)
        differs = np.flatnonzero(np.any(
            (np.take_along_axis(prot, inverse, axis=1) != prot)
            | (np.take_along_axis(inr, inverse, axis=1) != inr), axis=1))
        rows = str(differs[:5].tolist())
        if differs.size > 5:
            rows += f" +{differs.size - 5} more"
        escaped = (np.zeros_like(prot) if self.escaped_mask is None else
                   np.asarray(self.escaped_mask) & prot)
        note = ""
        if escaped.any():
            # Name the largest excursion, not a band-wide companion at a
            # different k whose energy may still be inside the window.
            k, n = np.unravel_index(np.argmax(np.where(
                escaped, np.abs(energies - mu_ev), -np.inf)), energies.shape)
            note = (f"escape band={band_offset + int(n) + 1}, k={int(k)}, "
                    f"E={energies[k, n]:+.6f} eV, "
                    f"pad={float(sc_state_pad_ev(energies[k, n] - mu_ev)):.6f} eV; ")
        return (f"SC partition: {note}protected={int(prot.sum())}/{prot.size}, "
                f"in_range={int(inr.sum())}, escaped={int(escaped.sum())}; "
                f"sorted columns differ at k={rows}")

    def report_multiplet_splits(self, enk_full_ry, band_offset, *,
                                label="SC", print_fn=print, degeneracy_tol_ev=None):
        """Report protected boundaries cutting a reference multiplet at each k."""
        from common.band_degeneracy import DEGENERACY_TOL_RY
        from common.units import RYD_TO_EV

        tolerance_ry = (DEGENERACY_TOL_RY if degeneracy_tol_ev is None else
                        float(degeneracy_tol_ev) / RYD_TO_EV)
        if not np.isfinite(tolerance_ry) or tolerance_ry < 0:
            raise ValueError("degeneracy_tol_ev must be finite and nonnegative")
        e = np.asarray(enk_full_ry, dtype=np.float64)
        mask = np.broadcast_to(np.asarray(self.protected_mask, dtype=bool),
                               (e.shape[0], self.protected_mask.shape[-1]))
        active = e[:, band_offset:band_offset + mask.shape[1]]
        gaps = np.diff(active, axis=1)
        splits = (np.diff(mask.astype(np.int8), axis=1) != 0) & (
            gaps <= tolerance_ry)
        for k, n in zip(*np.nonzero(splits)):
            print_fn(f"  {label} partition: multiplet split at k={k}, "
                     f"bands {n + band_offset + 1}/{n + band_offset + 2}, "
                     f"gap={gaps[k, n] * RYD_TO_EV * 1000:.6f} meV")
        count = int(splits.sum())
        if not count:
            print_fn(f"  {label} partition: no boundary splits a multiplet")
        return count, (float(gaps[splits].max() * RYD_TO_EV * 1000)
                       if count else 0.0)

    def promoted_to_multiplets(self, enk_full_ry, band_offset, *,
                               label="SC", print_fn=print,
                               degeneracy_tol_ev=None) -> "BandPartition":
        """Close protection within each k's reference multiplets independently.

        Adjacent-gap groups use the full reference ladder. Promotion at one
        k does not promote the same label at another k, so degeneracies at
        different k cannot produce a transitive union of unrelated spaces.
        """
        from common.band_degeneracy import DEGENERACY_TOL_RY
        from common.units import RYD_TO_EV

        tolerance_ry = (DEGENERACY_TOL_RY if degeneracy_tol_ev is None else
                        float(degeneracy_tol_ev) / RYD_TO_EV)
        if not np.isfinite(tolerance_ry) or tolerance_ry < 0:
            raise ValueError("degeneracy_tol_ev must be finite and nonnegative")
        e = np.asarray(enk_full_ry, dtype=np.float64)
        nb = self.protected_mask.shape[-1]
        mask = np.broadcast_to(np.asarray(self.protected_mask, dtype=bool),
                               (e.shape[0], nb))
        out = mask.copy()
        for k in range(e.shape[0]):
            groups = np.split(np.arange(e.shape[1]),
                              np.flatnonzero(np.diff(e[k]) > tolerance_ry) + 1)
            for group in groups:
                active = group[(group >= band_offset) & (group < band_offset + nb)]
                local = active - band_offset
                if local.size and mask[k, local].any():
                    if active.size != group.size:
                        raise ValueError(
                            f"{label} protected reference multiplet at k={k}, "
                            f"bands {(group + 1).tolist()} crosses the active window")
                    out[k, local] = True
        n_added = int(out.sum() - mask.sum())
        if n_added:
            print_fn(f"  {label} partition: promoted {n_added} (k,state) members "
                     "to whole reference multiplets")
        in_range = np.broadcast_to(np.asarray(self.in_range_mask, dtype=bool), out.shape)
        return replace(self, protected_mask=jnp.asarray(out),
                       in_range_mask=jnp.asarray(in_range))

    def warn_if_protected_outside_grid(self, *, print_fn=print) -> None:
        """Report protected states outside the requested classification window.

        Hysteresis may legitimately retain these states in padded quadrature
        support; the quadrature planner owns actual coverage certification.
        """
        n = int(jnp.sum(self.protected_mask & ~self.in_range_mask))
        if n:
            print_fn(f"  SC partition: {n} protected (k,state) members outside "
                     "the requested window; quadrature support must cover them")


def build_omega_band_partition(
    e_dft_kn_ry,
    e_dft_full_kn_ry,
    *,
    band_offset: int,
    omega_min_abs_ev: float,
    omega_max_abs_ev: float,
    previous_partition: BandPartition | None = None,
    hysteresis_margin_ev: float = 0.0,
    mu_ev: float | None = None,
    current_indices_kn=None,
    degeneracy_tol_ev: float | None = None,
    label: str = "SC",
    print_fn=print,
) -> BandPartition:
    """Classify reference identities using an all-k window and padded hysteresis.

    Parameters
    ----------
    e_dft_kn_ry : (nk, nb_active) real
        Current energies in Ry, already gathered into DFT identity order.
    e_dft_full_kn_ry : (nk, nb_full) real
        Immutable full DFT reference ladder in Ry for local multiplet closure.
    current_indices_kn : (nk, nb_active) integer, optional
        DFT identity to current sorted QP column, used only for the summary.
    degeneracy_tol_ev : float, optional
        Reference multiplet adjacent-gap tolerance in eV. Defaults to the
        common band-degeneracy threshold; SC supplies its exact tolerance.
    mu_ev : float, optional
        Current chemical potential in eV. Its state-relative energy sets the
        shared quadrature/classification pad. Legacy callers without mu use
        ``hysteresis_margin_ev`` in eV.

    Notes
    -----
    A label enters when its entire k range is inside the requested window.
    Previously protected members survive while the label's entire k range
    remains inside the padded bounds. Reference multiplets close locally.
    """
    from common.units import RYD_TO_EV
    from .scissor import classify_bands_in_grid, sc_state_pad_ev

    e_ev = np.asarray(e_dft_kn_ry, dtype=np.float64) * RYD_TO_EV
    lo, hi = float(omega_min_abs_ev), float(omega_max_abs_ev)
    band_in_grid, _ = classify_bands_in_grid(e_ev, lo, hi)
    in_range = np.broadcast_to(band_in_grid, e_ev.shape).copy()
    protected = in_range.copy()
    margin = float(hysteresis_margin_ev)
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("hysteresis_margin_ev must be finite and nonnegative")
    pad = (np.full(e_ev.shape, margin) if mu_ev is None else
           sc_state_pad_ev(e_ev - float(mu_ev)))
    escaped = np.zeros(e_ev.shape, dtype=bool)
    blocked = np.zeros(e_ev.shape, dtype=bool)
    if previous_partition is not None:
        previous = np.asarray(previous_partition.protected_mask, dtype=bool)
        if previous.shape not in ((e_ev.shape[1],), e_ev.shape):
            raise ValueError("previous partition and current spectrum have different shapes")
        previous = np.broadcast_to(previous, e_ev.shape)
        if previous_partition.escaped_mask is not None:
            blocked = np.asarray(previous_partition.escaped_mask, bool) & ~in_range
            previous = previous & ~blocked
        retained_kn = (e_ev >= lo - pad) & (e_ev <= hi + pad)
        retained = np.all(retained_kn, axis=0)
        escaped = previous & ~retained[None, :]
        protected |= previous
        for k, n in zip(*np.nonzero(previous & ~retained_kn)):
            print_fn(f"  {label} partition: escape band={n + band_offset + 1}, "
                     f"k={k}, E={e_ev[k, n]:+.6f} eV, pad={pad[k, n]:.6f} eV, "
                     f"window=[{lo:+.6f}, {hi:+.6f}] eV")
    partition = BandPartition(jnp.asarray(protected), jnp.asarray(in_range),
                              escaped_mask=jnp.asarray(escaped))
    partition = partition.promoted_to_multiplets(
        e_dft_full_kn_ry, int(band_offset), label=label, print_fn=print_fn,
        degeneracy_tol_ev=degeneracy_tol_ev)
    # A dropped member cannot be repeatedly promoted by its partner. Keep
    # the whole local reference block out until it is entirely in range.
    if blocked.any():
        blocked = np.asarray(BandPartition(
            jnp.asarray(blocked), jnp.zeros_like(partition.in_range_mask)
        ).promoted_to_multiplets(
            e_dft_full_kn_ry, band_offset, label=label, print_fn=print_fn,
            degeneracy_tol_ev=degeneracy_tol_ev).protected_mask)
        partition = replace(partition,
            protected_mask=partition.protected_mask & ~jnp.asarray(blocked),
            in_range_mask=partition.in_range_mask & ~jnp.asarray(blocked),
            escaped_mask=jnp.asarray(escaped | blocked))
    protected = np.asarray(partition.protected_mask)
    all_k = np.flatnonzero(np.all(protected, axis=0)) + int(band_offset) + 1
    print_fn(f"  {label} partition: protected at all k bands={all_k.tolist()}; "
             f"protected {int(protected.sum())}/{protected.size} (k,state)")
    if current_indices_kn is not None:
        indices = np.asarray(current_indices_kn)
        if indices.shape != e_ev.shape or not np.all(
                np.sort(indices, axis=1) == np.arange(e_ev.shape[1])):
            raise ValueError("current_indices_kn must be a per-k permutation of active columns")
        for k in range(e_ev.shape[0]):
            sorted_p = np.zeros(e_ev.shape[1], dtype=bool)
            sorted_i = np.zeros(e_ev.shape[1], dtype=bool)
            sorted_p[indices[k]] = protected[k]
            sorted_i[indices[k]] = in_range[k]
            if k == 0 or not (np.array_equal(sorted_p, protected[k]) and
                             np.array_equal(sorted_i, in_range[k])):
                absolute = lambda mask: (np.flatnonzero(mask) + band_offset + 1).tolist()
                print_fn(f"  {label} partition k={k}: protected identities="
                         f"{absolute(protected[k])}, sorted columns={absolute(sorted_p)}; "
                         f"scissored sorted columns={absolute(~(sorted_p | sorted_i))}")
    partition.report_multiplet_splits(
        e_dft_full_kn_ry, band_offset, label=label, print_fn=print_fn,
        degeneracy_tol_ev=degeneracy_tol_ev)
    partition.warn_if_protected_outside_grid(print_fn=print_fn)
    changed = bool(escaped.any())
    if previous_partition is not None:
        changed |= not (np.array_equal(partition.protected_mask,
                                       previous_partition.protected_mask)
                        and np.array_equal(partition.in_range_mask,
                                           previous_partition.in_range_mask))
    return replace(partition, changed=changed)


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
    protected_mask, in_range_mask : (nk, nb_active) or (nb_active,) bool
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
    offdiag_keep = p[..., :, None] * p[..., None, :]
    # Zero the off-diagonal portion outside protected×protected.
    offdiag_part = H_full * (1.0 - eye) * offdiag_keep

    # The protected class owns its full diagonal even after multiplet
    # promotion across the grid edge. Only the third class is scissored;
    # this is the same protected | in_range set used by convergence.
    diag_full = jnp.diagonal(H_full, axis1=1, axis2=2)            # (nk, nb)
    diag_kept = jnp.where(
        (protected_mask | in_range_mask), diag_full,
        scissor_E_qp_kn.astype(H_full.dtype),
    )                                                              # (nk, nb)

    # Reassemble: off-diag matrix + diag(diag_kept).
    return offdiag_part + diag_kept[:, :, None] * eye[None, :, :]


__all__ = [
    "BandPartition", "apply_band_partition", "build_omega_band_partition",
]
