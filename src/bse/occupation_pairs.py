"""Finite-occupation rectangular transition selection and exact retained weights.

Occupation thresholds select overlapping band envelopes only. They never
threshold or replace individual Fermi differences inside those envelopes.
The caller supplies its canonical occupations, energy units and k mapping.
"""
from __future__ import annotations

import numpy as np

from common.band_degeneracy import DEGENERACY_TOL_RY, boundary_min_gaps


def occupation_band_envelopes(energies_ry, occupations, *, threshold=1e-5,
                              degeneracy_tol_ry=DEGENERACY_TOL_RY):
    """Return overlapping occupied/empty envelopes within retained bands.

    Parameters are small host tables ``(nk, nb)``. A hole-side band is kept
    if any k has ``f > threshold``; a particle-side band if any k has
    ``1-f > threshold``. Internal boundaries expand outward to close full
    multiplets using the canonical band-boundary owner. Outer boundaries
    refer to the explicitly retained input spectrum, not missing WFN bands.
    """
    e = np.asarray(energies_ry, dtype=np.float64)
    f = np.asarray(occupations, dtype=np.float64)
    if (e.ndim != 2 or f.shape != e.shape or not np.isfinite(e).all()
            or not np.isfinite(f).all() or np.any((f < 0) | (f > 1))
            or not 0 < threshold < .5 or np.any(np.diff(e, axis=1) < 0)):
        raise ValueError("GATE fd_pair_inputs: require sorted finite energies, matching occupations in[0,1], and0<threshold<.5")
    holes = np.flatnonzero(np.any(f > threshold, axis=0))
    particles = np.flatnonzero(np.any(1-f > threshold, axis=0))
    if not holes.size or not particles.size:
        raise ValueError("GATE fd_pair_empty: both occupation envelopes must be nonempty")
    nb = e.shape[1]
    hi, lo = int(holes[-1]+1), int(particles[0])
    original = (hi, lo)
    gaps = boundary_min_gaps(e, is_full_spectrum=False)
    while hi < nb and gaps[hi] <= degeneracy_tol_ry:
        hi += 1
    while lo > 0 and gaps[lo] <= degeneracy_tol_ry:
        lo -= 1
    return dict(hole_indices=np.arange(hi, dtype=np.int32),
                particle_indices=np.arange(lo, nb, dtype=np.int32),
                threshold=float(threshold), original_hole_stop=original[0],
                original_particle_start=original[1], hole_stop=hi,
                particle_start=lo, retained_band_count=nb,
                multiplet_tolerance_ry=float(degeneracy_tol_ry),
                closure_scope="Outward internal-boundary closure within supplied retained spectrum")
