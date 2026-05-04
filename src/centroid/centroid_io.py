"""Reader for centroid files produced by ``centroid.kmeans_cli``.

Each file is a header-commented text table of fractional-coordinate
centroids.  Modern files (post-2026-05-02) carry a ``density:`` line in
the comment block telling the reader which weight built them — needed
for the bispinor pipeline, which reads two files (charge + current)
and routes them to different Lorentz channels.

Use this loader rather than ``np.loadtxt`` directly when the caller
wants to validate or branch on the density.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np


DensityLabel = Literal["scalar", "current", "unknown"]


class CentroidFile(NamedTuple):
    coords: np.ndarray   # (n, 3) float64 fractional coords
    density: DensityLabel


_DENSITY_RE = re.compile(r"^#\s*density:\s*(\S+)", re.IGNORECASE)


def read_centroids(path: str | Path) -> CentroidFile:
    """Read a centroid file, returning coords + the recorded density.

    Files written before 2026-05-02 don't have the ``density:`` comment;
    those are returned as ``density='unknown'``.  Callers that need the
    density to dispatch (gw_jax in bispinor mode) should regenerate
    legacy files via ``kmeans_cli`` to embed the new header.
    """
    p = Path(path)
    density: DensityLabel = "unknown"
    with p.open("r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = _DENSITY_RE.match(line)
            if m:
                v = m.group(1).lower()
                density = v if v in ("scalar", "current") else "unknown"  # type: ignore[assignment]
    coords = np.loadtxt(p, dtype=np.float64)
    return CentroidFile(coords=coords, density=density)


__all__ = ["read_centroids", "CentroidFile", "DensityLabel"]
