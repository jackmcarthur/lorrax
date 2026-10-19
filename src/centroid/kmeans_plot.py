"""Matplotlib helpers for k-means / ISDF centroid visualization.

Kept out of ``kmeans_isdf`` so the algorithm module stays free of
matplotlib + scipy dependencies. The CLI imports from here when
``--plot`` is requested.
"""
from __future__ import annotations

import os
# Force headless backend before any matplotlib import. The lorrax shifter
# image lacks PyQt5/PySide2/Tk so interactive backends fail at pyplot
# import; Agg works everywhere and is the right choice for a CLI that
# saves a PNG anyway.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def interpolate_density(rho_np: np.ndarray, zoom_factors=(1, 1, 1)) -> np.ndarray:
    """Bicubic upsample of ρ for smoother plots. Identity if zoom=1."""
    zoom_factors = np.asarray(zoom_factors)
    if np.all(zoom_factors == 1):
        return rho_np
    from scipy.ndimage import zoom
    return zoom(rho_np, zoom_factors, order=3)


def plot_density_and_centroids(
    wfn,
    rho_np: np.ndarray,
    centroids_frac: np.ndarray,
    out: str = "kmeans_centroids.png",
    threshold_frac: float = 0.05,
):
    """3D scatter of ρ(r) > threshold and centroids inside the unit cell.

    ``wfn`` is a WFNReader (used only for ``avec``); ``centroids_frac`` is
    (N_c, 3) in [0, 1)^3. Saves PNG and returns nothing.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rho_np = np.asarray(rho_np)
    threshold = threshold_frac * rho_np.max()
    Nx, Ny, Nz = rho_np.shape
    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, Nx, endpoint=False),
        np.linspace(0, 1, Ny, endpoint=False),
        np.linspace(0, 1, Nz, endpoint=False),
        indexing="ij",
    )
    mask = rho_np > threshold
    pts_frac = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    pts_cart = pts_frac @ wfn.avec
    rho_at_pts = rho_np[mask]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts_cart[:, 0], pts_cart[:, 1], pts_cart[:, 2],
        c=np.log(np.abs(rho_at_pts) - 0.9 * threshold),
        cmap="plasma", alpha=0.09, s=20, marker="s",
        label="Density",
    )
    plt.colorbar(sc, label="Charge density (log)")

    cent_cart = np.asarray(centroids_frac) @ wfn.avec
    ax.scatter(
        cent_cart[:, 0], cent_cart[:, 1], cent_cart[:, 2],
        c="red", s=100, marker="*", label="Centroids",
    )

    # Cell box
    corners_frac = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    corners_cart = corners_frac @ wfn.avec
    edges = [(0, 1), (1, 3), (3, 2), (2, 0),
             (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners_cart[i], corners_cart[j]), "k-", lw=1)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Charge density and centroids")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
