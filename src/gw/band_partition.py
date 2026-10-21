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

Masks follow DFT reference identities at each k.  The SC loop keeps every
QP-window identity protected (owner rule 2026-09-22), so its partition is
:meth:`BandPartition.all_protected` (the fixed-Sigma EQP2 partition that masked
non-protected bands retired with the omega grid).

THE QP MATRIX AND ITS SIGMA READ CLASSES (dynamic SC; owner 2026-09-29: "b3
will count bands as on main yes, and only bands between b0 and b3 will be
rotated amongst each other").  b3 counts bands as on main (nval/ncond, or
``number_bands_protected`` resolved to them); the QP matrix [b0, b3) rotates
among itself and [b3, nband) is the scissored tail (DFT psi, the rigid
conduction scissor, in G and chi only, no Sigma, no mixing).  Nothing here is
energy-dependent in the zeta fit.  Inside the QP matrix every state keeps its
full Sigma row; only where its Sigma_c(omega) is read differs: coarse
("semicore", :func:`semicore_floor`; read at its own energy on held windows
below the near grid, ``qp_support``) or protected (the near grid at the deck
eta).  There is no rotating class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from common.log import get_logger

log = get_logger(__name__)



# ---------------------------------------------------------------------------
# The coarse (semicore) read class (dynamic SC; Sigma_c(omega) quadrature only)
# ---------------------------------------------------------------------------

#: The owner's window rule (+-10 eV of E_F): under ``number_bands_protected``
#: the semicore gap is searched under the requested bands within it.
WINDOW_CLIP_EV = 10.0
#: ``number_bands_protected`` mode only (owner 2026-09-29): an occupied state
#: is semicore when a band gap at least this wide (eV, all k) separates it from
#: the valence manifold above.  That request protects every occupied band, so
#: the class needs its own boundary; 4 eV is 16 deck etas, wide enough that
#: the coarse window never reads a band the near grid resolves.
SEMICORE_GAP_EV = 4.0


class CoarseClass(NamedTuple):
    """The coarse (semicore) read class of an SC run (:func:`semicore_floor`)."""
    coarse_floor_ev: float  # coarse: E < this (absolute eV; -inf = none)
    n_coarse: int           # coarse (k, state) on the loaded k set


def band_gaps_ev(energies_ev):
    """Band-index gaps of a ladder: (lo, hi) with lo[n-1] = max_k E[:, n-1], hi[n-1] = min_k E[:, n].

    A boundary n with hi > lo holds the same n states below it at every k.
    """
    e = np.asarray(energies_ev, float)
    return e.max(axis=0)[:-1], e.min(axis=0)[1:]


def semicore_floor(energies_ev, *, n_below_k, nval, mu_ev, clip_ev, omega_min_rel_ev=None,
                   n_protected=None, semicore_gap_ev=None):
    """The coarse (semicore) class from the DFT ladder: which QP-matrix states
    read Sigma_c(omega) on the coarse windows instead of the near grid.

    ``energies_ev`` (nk, nb) absolute eV on the loaded k set; ``n_below_k``
    the states below mu at each k (``n_occ`` on an insulator).  It sets no
    band count: b3 and the zeta fit are the counted bands, as on main.

    * nval/ncond (owner, 2026-09-29): every occupied state below the minimum
      energy of the lowest requested valence band, ``n_below_k - nval`` at
      each k (``omega_min`` only lowers it); none when ``nval`` covers every
      occupied band.  Energy-based, so a coarse state inside the near grid's
      lower pad reads the near grid.
    * ``n_protected`` (``number_bands_protected``): every occupied band below
      a band gap of at least ``semicore_gap_ev`` under the requested bands
      within ``mu +- clip_ev`` (none without such a gap).
    """
    e = np.asarray(energies_ev, float)
    nk, nb = e.shape
    below = np.broadcast_to(np.asarray(n_below_k, int), (nk,))
    mu = float(mu_ev)
    if n_protected is None:
        lowest = np.clip(below - int(nval), 0, nb - 1)
        floor = float(np.min(e[np.arange(nk), lowest]))
        if omega_min_rel_ev is not None:
            floor = min(floor, mu + float(omega_min_rel_ev))
    else:
        idx = np.arange(nb)[None, :]
        within = (idx < min(int(n_protected), nb)) & (np.abs(e - mu) <= float(clip_ev))
        if not within.any():
            raise ValueError(f"semicore class: no requested state lies within mu +- "
                             f"{float(clip_ev):g} eV; raise number_bands_protected")
        gap_lo, gap_hi = band_gaps_ev(e)
        req_lo = int(np.min(np.where(within, idx, nb)))
        semi = [n for n in range(1, min(int(below.min()), req_lo) + 1)
                if gap_hi[n - 1] - gap_lo[n - 1] >= float(semicore_gap_ev)]
        floor = float(gap_hi[semi[-1] - 1]) if semi else -np.inf
    n_coarse = int(np.count_nonzero(e < floor))
    return CoarseClass(floor if n_coarse else -np.inf, n_coarse)


def coarse_band_report(energies_ev, semicore_kn, *, mu_ev, band_offset=0):
    """One line per band with coarse states: index, DFT range about mu, k count."""
    e = np.asarray(energies_ev, float)
    s = np.asarray(semicore_kn, bool)
    rows = []
    for n in np.flatnonzero(s.any(axis=0)):
        vals = e[s[:, n], n] - float(mu_ev)
        rows.append(f"band {int(n) + int(band_offset) + 1}: [{vals.min():+.3f}, {vals.max():+.3f}] eV "
                    f"on {int(s[:, n].sum())}/{s.shape[0]} k")
    return rows


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
            log.info(f"  {label} partition: multiplet split at k={k}, "
                     f"bands {n + band_offset + 1}/{n + band_offset + 2}, "
                     f"gap={gaps[k, n] * RYD_TO_EV * 1000:.6f} meV")
        count = int(splits.sum())
        if not count:
            log.debug(f"  {label} partition: no boundary splits a multiplet")
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
            log.info(f"  {label} partition: promoted {n_added} (k,state) members "
                     "to whole reference multiplets")
        in_range = np.broadcast_to(np.asarray(self.in_range_mask, dtype=bool), out.shape)
        return BandPartition(protected_mask=jnp.asarray(out),
                             in_range_mask=jnp.asarray(in_range))


__all__ = [
    "BandPartition",
]
