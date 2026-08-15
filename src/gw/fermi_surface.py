"""Fermi-surface quadrature on a periodic uniform three-dimensional grid."""

from __future__ import annotations

import itertools

import numpy as np


_REF_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0),
     (1.0, 0.0, 0.0),
     (0.0, 1.0, 0.0),
     (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_AXES = np.eye(3, dtype=np.int64)
_TETRA_OFFSETS = tuple(
    np.stack((np.zeros(3, dtype=np.int64),
              _AXES[a],
              _AXES[a] + _AXES[b],
              np.ones(3, dtype=np.int64)))
    for a, b, _c in itertools.permutations(range(3))
)


def _tetra_delta_vertex_weights(energy4, chemical_potential):
    """Integral weights for linear vertex data times delta(E-mu)."""
    energy = np.asarray(energy4, dtype=np.float64)
    mu = float(np.nextafter(float(chemical_potential), np.inf))
    if mu <= float(np.min(energy)) or mu >= float(np.max(energy)):
        return np.zeros(4, dtype=np.float64)
    gradient = energy[1:] - energy[0]
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= 1.0e-15:
        return np.zeros(4, dtype=np.float64)

    points = []
    barycentric = []
    for i, j in _EDGES:
        di = mu - energy[i]
        dj = mu - energy[j]
        if di * dj >= 0.0:
            continue
        t = di / (energy[j] - energy[i])
        lam = np.zeros(4, dtype=np.float64)
        lam[i] = 1.0 - t
        lam[j] = t
        points.append((1.0 - t) * _REF_VERTICES[i] + t * _REF_VERTICES[j])
        barycentric.append(lam)
    if len(points) not in (3, 4):
        return np.zeros(4, dtype=np.float64)

    points = np.asarray(points, dtype=np.float64)
    barycentric = np.asarray(barycentric, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    normal = gradient / gradient_norm
    u = points[0] - centroid
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1.0e-15:
        return np.zeros(4, dtype=np.float64)
    u /= u_norm
    v = np.cross(normal, u)
    angles = np.arctan2((points - centroid) @ v, (points - centroid) @ u)
    order = np.argsort(angles)
    points = points[order]
    barycentric = barycentric[order]

    result = np.zeros(4, dtype=np.float64)
    for i in range(1, len(points) - 1):
        area = 0.5 * float(np.linalg.norm(np.cross(
            points[i] - points[0], points[i + 1] - points[0])))
        result += (
            area / (3.0 * gradient_norm)
            * (barycentric[0] + barycentric[i] + barycentric[i + 1])
        )
    return result


def _uniform_grid_indices(kpoints_crystal, kgrid):
    points = np.mod(np.asarray(kpoints_crystal, dtype=np.float64), 1.0)
    grid = np.asarray(kgrid, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or np.any(grid < 2):
        raise ValueError("need kpoints (nk,3) and a three-dimensional kgrid >= 2")
    scaled = points * grid[None, :]
    # A common shift is defined modulo an integer.  Taking one point's
    # signed residual avoids the 0/1 branch cut that makes a median of
    # ``mod(scaled,1)`` ambiguous on nominally unshifted grids.
    shift = scaled[0] - np.rint(scaled[0])
    residual = scaled - shift[None, :]
    indices = np.mod(np.rint(residual).astype(np.int64), grid[None, :])
    error = np.max(np.abs(residual - np.rint(residual)))
    if error > 2.0e-7:
        raise ValueError(
            f"kpoints are not one uniformly shifted {tuple(grid)} grid; "
            f"maximum integer-coordinate residual is {error:.3e}")
    return indices


def tetrahedron_delta_weights(
    energies_kn,
    kpoints_crystal,
    kgrid,
    chemical_potential,
):
    """Return normalized-BZ weights for ``integral delta(E-mu) g(k) dk``.

    The returned ``weights_kn`` have units inverse to the supplied energies
    and satisfy ``sum_kn weights_kn*g_kn`` for linearly interpolated vertex
    data ``g``.  Every periodic grid cell is split into the six tetrahedra
    sharing its body diagonal.  No smearing or empirical normalization is
    introduced.
    """
    energies = np.asarray(energies_kn, dtype=np.float64)
    grid = np.asarray(kgrid, dtype=np.int64)
    nk_expected = int(np.prod(grid))
    if energies.ndim != 2 or energies.shape[0] != nk_expected:
        raise ValueError(
            f"energies must be ({nk_expected},nb), got {energies.shape}")
    indices = _uniform_grid_indices(kpoints_crystal, grid)
    grid_to_flat = np.full(tuple(grid), -1, dtype=np.int64)
    for flat, ijk in enumerate(indices):
        key = tuple(int(x) for x in ijk)
        if grid_to_flat[key] >= 0:
            raise ValueError(f"duplicate k-grid coordinate {key}")
        grid_to_flat[key] = flat
    if np.any(grid_to_flat < 0):
        raise ValueError("the supplied kpoints do not cover the full periodic grid")

    weights = np.zeros_like(energies)
    cell_jacobian = 1.0 / float(nk_expected)
    nbands = int(energies.shape[1])
    for base in np.ndindex(*(int(x) for x in grid)):
        base_vec = np.asarray(base, dtype=np.int64)
        for offsets in _TETRA_OFFSETS:
            vertices = np.mod(base_vec[None, :] + offsets, grid[None, :])
            flat = np.asarray(
                [grid_to_flat[tuple(int(x) for x in vertex)]
                 for vertex in vertices],
                dtype=np.int64,
            )
            for band in range(nbands):
                local = _tetra_delta_vertex_weights(
                    energies[flat, band], chemical_potential)
                weights[flat, band] += cell_jacobian * local
    return weights


__all__ = ["tetrahedron_delta_weights"]
