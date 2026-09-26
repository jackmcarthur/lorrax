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

Masks follow DFT reference identities at each k.  Only fixed-Sigma EQP2
builds a non-trivial partition (:func:`build_omega_band_partition`); the SC
loop keeps every QP-window identity protected (owner rule 2026-09-22), so its
partition is :meth:`BandPartition.all_protected`.
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

    Masks are per ``(k, DFT identity)``; one-dimensional masks are also
    accepted for callers whose classification is identical at every k.

    Attributes
    ----------
    protected_mask : (nk, nb_active) or (nb_active,) bool
        True for bands that get full off-diagonal Σ corrections in
        ``H_qp_dft`` and participate in the basis rotation.
    in_range_mask : (nk, nb_active) or (nb_active,) bool
        True for identities classified inside ``[ω_min, ω_max]`` at every k
        during initialization; SC keeps this mask fixed. Decides between
        Σ_diag and scissor for the *non-protected* bands' diagonal.
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
        return BandPartition(protected_mask=jnp.asarray(out),
                             in_range_mask=jnp.asarray(in_range))


def build_omega_band_partition(
    e_dft_kn_ry,
    e_dft_full_kn_ry,
    *,
    band_offset: int,
    omega_min_abs_ev: float,
    omega_max_abs_ev: float,
    label: str = "EQP2",
    print_fn=print,
) -> BandPartition:
    """Classify reference identities with an all-k window (fixed-Sigma EQP2).

    Parameters
    ----------
    e_dft_kn_ry : (nk, nb_active) real
        Current energies in Ry, already gathered into DFT identity order.
    e_dft_full_kn_ry : (nk, nb_full) real
        Immutable full DFT reference ladder in Ry for local multiplet closure.

    Notes
    -----
    A label is protected and in range when its entire k range is inside the
    requested window. Reference multiplets close locally.
    """
    from common.units import RYD_TO_EV
    from .scissor import classify_bands_in_grid

    e_ev = np.asarray(e_dft_kn_ry, dtype=np.float64) * RYD_TO_EV
    lo, hi = float(omega_min_abs_ev), float(omega_max_abs_ev)
    band_in_grid, _ = classify_bands_in_grid(e_ev, lo, hi)
    in_range = np.broadcast_to(band_in_grid, e_ev.shape).copy()
    protected = in_range.copy()
    partition = BandPartition(jnp.asarray(protected), jnp.asarray(in_range))
    partition = partition.promoted_to_multiplets(
        e_dft_full_kn_ry, int(band_offset), label=label, print_fn=print_fn)
    protected = np.asarray(partition.protected_mask)
    all_k = np.flatnonzero(np.all(protected, axis=0)) + int(band_offset) + 1
    print_fn(f"  {label} partition: protected at all k bands={all_k.tolist()}; "
             f"protected {int(protected.sum())}/{protected.size} (k,state)")
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
