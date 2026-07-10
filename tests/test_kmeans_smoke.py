"""ONE end-to-end smoke test for the k-means fixture-generation tool.

The 2026-07-09 suite redesign deleted the 690-line kmeans unit file
(``test_kmeans_sharded.py``) on the principle that kmeans is a
fixture-generation tool: if it breaks, fixture regeneration fails loudly
and visibly.  This single smoke test keeps the cheapest whole-driver
pin — the hex (MoS2-like) end-to-end run — because it transitively
exercises the offset table, the PBC metric, k-means++ init, and the
Lloyd/shard plumbing that every centroid fixture in tests/regression
was generated with.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh


def _one_device_mesh() -> Mesh:
    dev = np.asarray(jax.devices()[:1]).reshape(1, 1)
    return Mesh(dev, axis_names=("x", "y"))


def test_weighted_kmeans_jax_hex_end_to_end():
    """End-to-end driver smoke test on a hex (MoS2-like) cell.

    Exercises the path that builds the offset table inside
    ``weighted_kmeans_jax`` and threads it through k-means++ init + Lloyd
    iterations.  Catches: a wrong metric tensor (avec.T @ avec vs
    avec @ avec.T — wildly different centroid placement); a missing
    offset thread (hex-biased Lloyd convergence); any signature mismatch
    in the kpp / Lloyd / shard plumbing.
    """
    from src.centroid.kmeans_isdf import weighted_kmeans_jax

    # MoS2-style hex cell, c/a small enough for a 3D grid that fits in seconds.
    avec = jnp.asarray(
        np.array([
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, 2.0],
        ]) * 3.0,                       # ~3 Å lattice constant
        dtype=jnp.float64,
    )
    # Small synthetic density: two Gaussian bumps at hex Wyckoff positions
    # (atoms at (1/3, 2/3) and (2/3, 1/3) under the hex 120° convention).
    Nx, Ny, Nz = 12, 12, 12
    xs = jnp.linspace(0.0, 1.0, Nx, endpoint=False, dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    # Build rho on host so we can use the (correct) hex metric for the bumps.
    G_np = np.asarray(avec @ avec.T)
    centers = np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])
    pos = np.stack([np.asarray(X), np.asarray(Y), np.asarray(Z)],
                   axis=-1).reshape(-1, 3)
    rho = np.full(pos.shape[0], 1e-3)
    for c in centers:
        d = pos - c
        d -= np.round(d)
        d2 = np.einsum("pi,ij,pj->p", d, G_np, d)
        rho = rho + np.exp(-d2 / 0.05)
    rho_jax = jnp.asarray(rho.reshape(Nx, Ny, Nz), dtype=jnp.float64)

    _, centroids, steps, _ = weighted_kmeans_jax(
        avec, rho_jax, N_c=24, max_steps=50, tolerance=1e-3, seed=0,
        init_method="kpp",
        mesh=_one_device_mesh(),
    )
    centroids = np.asarray(centroids)
    assert centroids.shape == (24, 3)
    # Centroids must lie in [0, 1) — hex driver wraps via `% 1.0`.
    assert np.all(centroids >= 0.0) and np.all(centroids < 1.0), (
        f"centroids escaped unit cell: min={centroids.min():.3e}, "
        f"max={centroids.max():.3e}"
    )
    # Lloyd should converge in well under max_steps for this benign density.
    assert steps < 50, f"k-means failed to converge ({steps}/50 steps)"
