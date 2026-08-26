"""Centroid-coordinate loading behind one format and symmetry door.

The text file contains fractional crystal coordinates.  This module owns the
one conversion from those coordinates to the integer FFT-grid rows consumed by
the ISDF kernels.  Orbit closure itself remains the responsibility of the
``symmetry_maps`` service: :func:`load_centroid_basis` calls its public
measurement and returns the resulting structured verdict without printing or
choosing an IBZ/full-BZ policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from symmetry_maps import CentroidClosureVerdict


@dataclass(frozen=True)
class LoadedCentroids:
    """One loaded ISDF centroid basis and its canonical closure verdict.

    ``centroids_frac`` records the coordinates as written;
    ``centroid_indices`` records the periodically wrapped integer FFT-grid
    points the physics actually consumes.  ``closure`` is a
    ``symmetry_maps.CentroidClosureVerdict`` measured on those consumed grid
    points.  Keeping the verdict structured lets each driver render it in its
    own report without putting presentation or q-grid policy in file I/O.
    """

    path: str
    centroids_frac: np.ndarray
    centroid_indices: np.ndarray
    n_rmu: int
    source_n_rmu: int
    closure: CentroidClosureVerdict

    @property
    def orbit_closed(self) -> bool:
        """Whether every spatial symmetry maps the loaded set onto itself."""
        return bool(self.closure.closed)


def _read_centroid_coordinates(centroids_file: str) -> np.ndarray:
    """Read and validate one fractional-coordinate table."""
    with open(centroids_file, "r", encoding="utf-8") as stream:
        if not any(line.strip() and not line.lstrip().startswith("#")
                   for line in stream):
            raise ValueError(f"Centroid file {centroids_file} is empty.")
    centroids_frac = np.loadtxt(centroids_file, ndmin=2)
    centroids_frac = np.asarray(centroids_frac, dtype=np.float64)
    if centroids_frac.ndim != 2 or centroids_frac.shape[1] != 3:
        raise ValueError(
            f"Centroid file {centroids_file} must contain three fractional "
            f"coordinates per row; got shape {centroids_frac.shape}.")
    if centroids_frac.shape[0] == 0:
        raise ValueError(f"Centroid file {centroids_file} is empty.")
    if not np.isfinite(centroids_frac).all():
        raise ValueError(
            f"Centroid file {centroids_file} contains a non-finite "
            "coordinate.")
    return centroids_frac


def _centroid_grid_indices(
    centroids_frac: np.ndarray,
    fft_grid: tuple[int, int, int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(validated_grid, periodically_wrapped_integer_rows)``."""
    grid = np.asarray(fft_grid, dtype=np.int64)
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError(
            f"fft_grid must contain three positive extents; got "
            f"{grid.tolist()}.")
    indices = np.rint(centroids_frac * grid[None, :]).astype(np.int64)
    indices = np.mod(indices, grid[None, :])
    return grid, indices


def validate_centroid_selection(selection, n_parent: int) -> np.ndarray:
    """Validate one ordered parent-to-child centroid row selection."""
    idx = np.asarray(selection)
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise ValueError(
            "centroid selection must be a one-dimensional integer row list; "
            f"got shape={idx.shape}, dtype={idx.dtype}.")
    idx = idx.astype(np.int64, copy=False)
    if idx.size == 0:
        raise ValueError("centroid selection may not be empty.")
    if int(idx.min()) < 0 or int(idx.max()) >= int(n_parent):
        raise ValueError(
            "centroid selection escapes its parent table: "
            f"min/max={int(idx.min())}/{int(idx.max())}, parent rows="
            f"{int(n_parent)}.")
    if np.unique(idx).size != idx.size:
        raise ValueError(
            "centroid selection contains duplicate parent rows; the child "
            "basis must be a strict ordered subset.")
    return idx


def load_centroids(
    centroids_file: str,
    fft_grid: tuple[int, int, int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load centroid coordinates and convert them to periodic FFT-grid rows.

    This is the compatibility surface for consumers that do not hold a
    ``SymMaps`` yet.  Drivers that already constructed the canonical symmetry
    object should use :func:`load_centroid_basis` and retain its closure
    verdict.
    """
    centroids_frac = _read_centroid_coordinates(centroids_file)
    _grid, centroid_indices = _centroid_grid_indices(
        centroids_frac, fft_grid)
    return centroids_frac, centroid_indices, int(centroids_frac.shape[0])


def load_centroid_basis(
    centroids_file: str,
    fft_grid: tuple[int, int, int] | np.ndarray,
    *,
    sym,
    selection=None,
) -> LoadedCentroids:
    """Load centroids and measure orbit closure with the canonical service.

    ``sym`` must be the run's existing ``symmetry_maps.SymMaps``.  This
    function never constructs a second symmetry table and never refuses only
    because a legacy centroid set is not closed: non-closure is a structured
    fact consumed by reporting and by the separate q-grid policy resolver.
    Malformed coordinate/grid inputs still raise at the loading boundary.
    """
    centroids_frac, centroid_indices, source_n_rmu = load_centroids(
        centroids_file, fft_grid)
    if selection is not None:
        selection = validate_centroid_selection(selection, source_n_rmu)
        centroids_frac = centroids_frac[selection]
        centroid_indices = centroid_indices[selection]
    n_rmu = int(centroid_indices.shape[0])
    grid = np.asarray(fft_grid, dtype=np.int64).reshape(3)

    # Import through the service door, never through an implementation
    # submodule.  Measure the grid rows the kernels consume rather than the
    # rounded text spellings, matching ``resolve_qgrid_symmetry`` exactly.
    from symmetry_maps import verify_centroid_orbit_closure

    sym_matrices = np.asarray(sym.sym_matrices)
    n_sym = int(sym_matrices.shape[0])
    translations = np.asarray(sym.translations)
    if translations.ndim != 2 or translations.shape[1] != 3 \
            or int(translations.shape[0]) < n_sym:
        raise ValueError(
            "load_centroid_basis: SymMaps translations and spatial symmetry "
            f"rows disagree: sym_matrices={sym_matrices.shape}, "
            f"translations={translations.shape}.")
    consumed_frac = (
        centroid_indices.astype(np.float64) / grid[None, :].astype(np.float64))
    closure = verify_centroid_orbit_closure(
        consumed_frac,
        sym_matrices,
        tnp=translations[:n_sym],
    )
    return LoadedCentroids(
        path=os.path.abspath(centroids_file),
        centroids_frac=centroids_frac,
        centroid_indices=centroid_indices,
        n_rmu=n_rmu,
        source_n_rmu=source_n_rmu,
        closure=closure,
    )
