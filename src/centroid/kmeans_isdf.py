"""
Weighted k-means clustering for ISDF sampling point selection.

This module implements density-weighted k-means with periodic boundary conditions (PBC).
The clustering uses the minimal image convention for distances, which is critical for
crystalline systems.

PBC Distance Calculation (Minimal Image Convention)
====================================================
For two points with fractional coordinates r1 and r2:

1. Compute fractional displacement: df = r1 - r2
2. Apply minimal image: df_wrapped = df - round(df)
   - This wraps each component to [-0.5, 0.5)
   - Equivalent to finding the closest image among all 27 neighboring cells
3. Compute Cartesian distance: d = |df_wrapped @ avec|
   - Or equivalently using metric tensor: d² = df_wrapped @ G @ df_wrapped
   - Where G = avec.T @ avec is the metric tensor

Why round() gives the minimum over 27 cells:
- Each fractional component df_i can be shifted by any integer n_i
- The closest image has n_i = round(df_i), giving df_i - round(df_i) ∈ [-0.5, 0.5)
- This is the unique image in the first Brillouin zone (Wigner-Seitz cell in reciprocal space)

For non-orthogonal cells, this approximation works well when the cell is "not too skewed".
For highly skewed cells, one should check neighboring images explicitly.
"""

import numpy as np

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

print(f"✓ JAX initialized with device: {jax.devices()[0]}")

from file_io import WFNReader
from common import symmetry_maps

# matplotlib is optional - only needed for plotting
try:
    import matplotlib
    # Try to use an interactive backend if available
    _interactive_backend = False
    for backend in ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'macosx']:
        try:
            matplotlib.use(backend)
            _interactive_backend = True
            break
        except Exception:
            continue
    if not _interactive_backend:
        print("Note: No interactive matplotlib backend available. Plots will be saved to files.")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side effects
    _HAS_MATPLOTLIB = True
except ImportError:
    print("Note: matplotlib not available. Plotting disabled.")
    _HAS_MATPLOTLIB = False
    _interactive_backend = False
    plt = None

from scipy.ndimage import zoom

from .get_charge_density import calculate_charge_density


# =============================================================================
# PBC Distance Utilities (Pure JAX)
# =============================================================================

def precompute_metric_tensor(avec: np.ndarray) -> np.ndarray:
    """Precompute the metric tensor G = A^T @ A for PBC distance calculations.
    
    The metric tensor allows computing squared Cartesian distances directly
    from fractional displacements without explicit coordinate conversion:
    
        d² = df @ G @ df^T
        
    where df is the (wrapped) fractional displacement vector.
    
    This is equivalent to d² = |df @ avec|² but avoids materializing the
    Cartesian displacement vector, saving memory for large arrays.
    
    Args:
        avec: (3, 3) lattice vectors, rows are a1, a2, a3 in Cartesian coords
        
    Returns:
        G: (3, 3) metric tensor
    """
    return avec.T @ avec


@jax.jit
def pbc_distance_sq_batch(
    positions_frac: jnp.ndarray,
    centroids_frac: jnp.ndarray, 
    metric_tensor: jnp.ndarray
) -> jnp.ndarray:
    """Compute squared PBC distances between all positions and all centroids.
    
    Uses the minimal image convention: for each pair, finds the minimum distance
    over all periodic images (equivalent to checking 27 neighboring cells).
    
    Implementation:
        1. Compute fractional displacement: df = pos - cent
        2. Wrap to [-0.5, 0.5): df = df - round(df)  [minimal image]
        3. Compute squared distance: d² = df @ G @ df^T
    
    Args:
        positions_frac: (P, 3) fractional coordinates of grid points
        centroids_frac: (K, 3) fractional coordinates of centroids
        metric_tensor: (3, 3) G = avec^T @ avec
        
    Returns:
        distances_sq: (P, K) squared Cartesian distances with PBC
    """
    # Fractional displacement: shape (P, K, 3)
    delta_frac = positions_frac[:, None, :] - centroids_frac[None, :, :]
    
    # Minimal image convention: wrap to [-0.5, 0.5)
    # This finds the closest periodic image among all 27 neighboring cells
    delta_frac = delta_frac - jnp.round(delta_frac)
    
    # Squared distance using metric tensor: d² = df @ G @ df^T
    # einsum 'pki,ij,pkj->pk' computes this for all (P, K) pairs efficiently
    distances_sq = jnp.einsum('pki,ij,pkj->pk', delta_frac, metric_tensor, delta_frac)
    
    return distances_sq


@jax.jit  
def pbc_distance_sq_single(
    positions_frac: jnp.ndarray,
    centroid_frac: jnp.ndarray,
    metric_tensor: jnp.ndarray
) -> jnp.ndarray:
    """Compute squared PBC distances from all points to a single centroid.
    
    Optimized for k-means++ initialization where we add one centroid at a time.
    
    Args:
        positions_frac: (P, 3) fractional coordinates of grid points
        centroid_frac: (3,) fractional coordinates of single centroid
        metric_tensor: (3, 3) G = avec^T @ avec
        
    Returns:
        distances_sq: (P,) squared distances to the centroid
    """
    # Fractional displacement: shape (P, 3)
    delta_frac = positions_frac - centroid_frac[None, :]
    
    # Minimal image convention
    delta_frac = delta_frac - jnp.round(delta_frac)
    
    # Squared distance: d² = df @ G @ df^T for each point
    distances_sq = jnp.einsum('pi,ij,pj->p', delta_frac, metric_tensor, delta_frac)
    
    return distances_sq


@partial(jax.jit, static_argnames=['n_k'])
def kmeans_update_step(
    positions_frac: jnp.ndarray,
    centroids_frac: jnp.ndarray,
    rho_flat: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    n_k: int
) -> tuple:
    """Single k-means update step: assign labels and compute new centroids.
    
    This is the core k-means iteration, JIT-compiled for speed:
        1. Compute all pairwise PBC distances
        2. Assign each point to nearest centroid
        3. Compute weighted mean of assigned points (in wrapped coordinates)
        4. Update centroid positions
    
    The weighted mean is computed in *displacement space* relative to the current
    centroid, using PBC-wrapped displacements. This ensures the centroid moves
    towards its assigned points correctly even when points wrap around boundaries.
    
    Args:
        positions_frac: (P, 3) fractional coordinates of grid points
        centroids_frac: (K, 3) fractional coordinates of centroids
        rho_flat: (P,) charge density weights
        metric_tensor: (3, 3) G = avec^T @ avec
        n_k: Number of clusters (must match centroids_frac.shape[0])
        
    Returns:
        new_centroids_frac: (K, 3) updated centroid positions
        movement_sq: (K,) squared movement of each centroid (for convergence check)
        labels: (P,) cluster assignment for each point
    """
    # Step 1: Compute all pairwise squared distances with PBC
    distances_sq = pbc_distance_sq_batch(positions_frac, centroids_frac, metric_tensor)
    
    # Step 2: Assign each point to nearest centroid
    # (squared distance preserves ordering, so argmin is the same)
    labels = jnp.argmin(distances_sq, axis=1)  # (P,)
    
    # Step 3: Compute weighted centroid updates
    # One-hot encode labels for masking: (P, K)
    mask = jax.nn.one_hot(labels, n_k, dtype=rho_flat.dtype)
    
    # Compute PBC-wrapped displacements from each point to each centroid
    delta_frac = positions_frac[:, None, :] - centroids_frac[None, :, :]
    delta_frac = delta_frac - jnp.round(delta_frac)  # (P, K, 3)
    
    # Weight displacements by density and assignment mask
    weights = mask * rho_flat[:, None]  # (P, K)
    weighted_delta = weights[:, :, None] * delta_frac  # (P, K, 3)
    
    # Sum weighted displacements and total weights per centroid
    sum_weighted_delta = jnp.sum(weighted_delta, axis=0)  # (K, 3)
    sum_weights = jnp.sum(weights, axis=0)  # (K,)
    
    # Compute average displacement (avoid division by zero)
    # For centroids with no assigned points, keep position unchanged
    avg_delta = jnp.where(
        sum_weights[:, None] > 0,
        sum_weighted_delta / jnp.maximum(sum_weights[:, None], 1e-10),
        0.0
    )
    
    # Step 4: Update centroid positions and wrap to [0, 1)
    new_centroids_frac = (centroids_frac + avg_delta) % 1.0
    
    # Compute movement for convergence check (with PBC!)
    movement_frac = new_centroids_frac - centroids_frac
    movement_frac = movement_frac - jnp.round(movement_frac)  # PBC wrap the movement
    movement_sq = jnp.einsum('ki,ij,kj->k', movement_frac, metric_tensor, movement_frac)
    
    return new_centroids_frac, movement_sq, labels


# =============================================================================
# Visualization utilities
# =============================================================================

def interpolate_density(rho_np, zoom_factors=(1, 1, 1)):
    """Return a zoomed copy of ``rho_np`` using ``scipy.ndimage.zoom``."""
    zoom_factors = np.asarray(zoom_factors)
    if np.all(zoom_factors == 1):
        return rho_np
    return zoom(rho_np, zoom_factors, order=3)


def plot_density_and_centroids(wfn, rho_np, centroids, labels=None):
    """Plot charge density and centroids in 3D."""
    if not _HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not available)")
        return
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Turn off grid and panes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_zlim(0, 4)

    # Plot density points where rho > threshold
    rho_np = np.asarray(rho_np)
    rho_shape = rho_np.shape
    threshold = 0.05 * np.amax(rho_np)
    
    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, rho_shape[0]),
        np.linspace(0, 1, rho_shape[1]),
        np.linspace(0, 1, rho_shape[2]),
        indexing="ij",
    )

    density_mask = rho_np > threshold
    density_points = (
        np.stack([X[density_mask], Y[density_mask], Z[density_mask]], axis=1)
        @ wfn.avec
    )
    density_values = rho_np[density_mask]

    scatter = ax.scatter(
        density_points[:, 0],
        density_points[:, 1],
        density_points[:, 2],
        c=np.log(np.abs(density_values) - 0.9 * threshold),
        cmap="plasma",
        alpha=0.09,
        s=20,
        marker="s",
        label="Density",
        zorder=1,
    )
    plt.colorbar(scatter, label="Charge Density")

    centroids_np = np.asarray(centroids) @ wfn.avec
    ax.scatter(
        centroids_np[:, 0],
        centroids_np[:, 1],
        centroids_np[:, 2],
        c="red",
        s=100,
        marker="*",
        label="Centroids",
        zorder=2,
    )

    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    vertices_cart = vertices @ wfn.avec
    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        ax.plot(
            [vertices_cart[start, 0], vertices_cart[end, 0]],
            [vertices_cart[start, 1], vertices_cart[end, 1]],
            [vertices_cart[start, 2], vertices_cart[end, 2]],
            "k-",
            linewidth=1,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Charge Density and Centroids")
    
    # Save to file and show interactively if possible
    plt.savefig("kmeans_centroids.png", dpi=150, bbox_inches='tight')
    print("Saved plot to kmeans_centroids.png")
    if _interactive_backend:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Main k-means implementation (Pure JAX)
# =============================================================================

def weighted_kmeans_jax(
    avec: jnp.ndarray,
    rho_jax: jnp.ndarray,
    N_k: int = 10,
    max_steps: int = 200,
    tolerance: float = 5e-3,
    seed: int = 0,
) -> tuple:
    """
    Density-weighted k-means clustering with periodic boundary conditions.
    
    Uses k-means++ initialization for better initial centroid placement,
    then iterates Lloyd's algorithm until convergence.
    
    Key features:
    - Minimal image convention for PBC (considers all 27 neighboring cells)
    - Metric tensor for efficient squared distance computation
    - JIT-compiled inner loop for speed
    - Weighted by charge density (centroids concentrate in high-density regions)
    
    Args:
        avec: (3, 3) lattice vectors (rows are a1, a2, a3 in Cartesian coords)
        rho_jax: (Nx, Ny, Nz) charge density on real-space grid
        N_k: Number of cluster centroids (ISDF sampling points)
        max_steps: Maximum k-means iterations
        tolerance: Convergence tolerance for centroid movement (Angstroms)
        seed: Random seed for reproducibility
        
    Returns:
        labels: (P,) cluster assignment for each grid point
        centroids: (N_k, 3) final centroid positions in fractional coordinates
        history: (N_k, max_steps) z-coordinate history (for debugging)
        steps_taken: Number of iterations until convergence
    """
    print(f"Starting weighted k-means with {N_k} clusters...")
    
    # Precompute metric tensor: G = avec^T @ avec
    # This allows d² = df @ G @ df without Cartesian conversion
    metric_tensor = jnp.array(avec.T @ avec, dtype=jnp.float32)
    tolerance_sq = tolerance ** 2  # Compare squared distances for efficiency
    
    # Build grid of fractional positions (float32 to match centroids dtype)
    grid_x, grid_y, grid_z = rho_jax.shape
    X, Y, Z = jnp.meshgrid(
        jnp.linspace(0, 1, grid_x, endpoint=False, dtype=jnp.float32),
        jnp.linspace(0, 1, grid_y, endpoint=False, dtype=jnp.float32),
        jnp.linspace(0, 1, grid_z, endpoint=False, dtype=jnp.float32),
        indexing="ij",
    )
    positions = jnp.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    rho_flat = rho_jax.reshape(-1).astype(jnp.float32)
    n_points = positions.shape[0]
    
    print(f"Grid size: {grid_x}x{grid_y}x{grid_z} = {n_points} points")
    
    # =========================================================================
    # K-means++ initialization
    # =========================================================================
    # Standard k-means++ but with:
    # - Density weighting (prefer high-density regions)
    # - PBC distances (using minimal image convention)
    # =========================================================================
    print("K-means++ initialization...")
    
    rng = np.random.default_rng(seed)
    centroids = jnp.zeros((N_k, 3), dtype=jnp.float32)
    
    # First centroid: random selection weighted by density
    probs = np.array(rho_flat / rho_flat.sum())
    first_idx = rng.choice(n_points, p=probs)
    centroids = centroids.at[0].set(positions[first_idx])

    # Track minimum squared distance to any existing centroid
    min_dist_sq = jnp.full(n_points, jnp.inf, dtype=jnp.float32)
    
    # Add remaining centroids using k-means++ selection
    for k in range(1, N_k):
        if k % 50 == 0:
            print(f"  Selecting centroid {k}/{N_k}")
        
        # Update min_dist_sq with distance to the most recently added centroid
        # This is the key optimization: only compute distance to NEW centroid
        dist_sq_to_new = pbc_distance_sq_single(
            positions, centroids[k-1], metric_tensor
        )
        min_dist_sq = jnp.minimum(min_dist_sq, dist_sq_to_new)
        
        # k-means++ selection: probability ∝ D(x)² × ρ(x)
        # D(x) = distance to nearest existing centroid
        probs = np.array(min_dist_sq * rho_flat)
        probs = probs / probs.sum()
        next_idx = rng.choice(n_points, p=probs)
        centroids = centroids.at[k].set(positions[next_idx])
    
    print("K-means++ initialization complete.")
    
    # =========================================================================
    # Lloyd's algorithm (iterative refinement)
    # =========================================================================
    print("Starting k-means iterations...")

    history = jnp.zeros((N_k, max_steps), dtype=jnp.float32)
    steps_taken = max_steps
    
    for step in range(max_steps):
        # JIT-compiled update step
        new_centroids, movement_sq, labels = kmeans_update_step(
            positions, centroids, rho_flat, metric_tensor, N_k
        )
        
        # Record z-components for debugging/visualization
        history = history.at[:, step].set(new_centroids[:, 2])
        
        # Check convergence: all centroids moved less than tolerance
        max_movement_sq = float(jnp.max(movement_sq))
        
        if step % 20 == 0:
            print(f"  Step {step}: max movement = {np.sqrt(max_movement_sq):.6f} Å")
        
        if max_movement_sq < tolerance_sq:
            print(f"Converged in {step + 1} steps "
                  f"(max movement = {np.sqrt(max_movement_sq):.6f} Å)")
            steps_taken = step + 1
            centroids = new_centroids
            break

        centroids = new_centroids
    else:
        print(f"Reached max steps ({max_steps}) without full convergence")
    
    # Final label assignment
    distances_sq = pbc_distance_sq_batch(positions, centroids, metric_tensor)
    labels = jnp.argmin(distances_sq, axis=1)

    return labels, centroids, history, steps_taken


def snap_centroids_to_grid(
    centroids_frac: np.ndarray,
    fft_grid: np.ndarray,
    deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Snap fractional centroids to the nearest FFT grid points and optionally deduplicate.
    
    When N_k is large relative to the FFT grid, multiple k-means centroids may map 
    to the same grid point. This function:
    1. Converts fractional coords to integer grid indices
    2. Handles periodic wrapping
    3. Removes duplicate grid points (if deduplicate=True)
    
    Args:
        centroids_frac: (N_k, 3) fractional coordinates in [0, 1)
        fft_grid: (3,) FFT grid dimensions [Nx, Ny, Nz]
        deduplicate: If True, remove duplicate grid points
        
    Returns:
        centroid_indices: (N_unique, 3) integer grid indices
        centroids_frac_snapped: (N_unique, 3) fractional coords of snapped centroids
        n_duplicates: Number of duplicate centroids that were removed
    """
    fft_grid = np.asarray(fft_grid)
    centroids_frac = np.asarray(centroids_frac)
    
    # Round to nearest grid point
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int)
    
    # Handle periodic boundary: wrap indices that hit the boundary
    for i in range(3):
        centroid_indices[centroid_indices[:, i] == fft_grid[i], i] = 0
    
    n_original = centroid_indices.shape[0]
    
    if deduplicate:
        # Find unique grid points
        centroid_indices, unique_idx = np.unique(centroid_indices, axis=0, return_index=True)
        n_unique = centroid_indices.shape[0]
        n_duplicates = n_original - n_unique
        
        if n_duplicates > 0:
            print(f"Warning: {n_duplicates} centroids mapped to duplicate grid points "
                  f"({n_original} → {n_unique} unique)")
    else:
        n_duplicates = 0
    
    # Convert back to fractional coordinates (snapped to grid)
    centroids_frac_snapped = centroid_indices.astype(float) / fft_grid
    
    return centroid_indices, centroids_frac_snapped, n_duplicates


def ensure_unique_centroids(
    centroids_frac: np.ndarray,
    fft_grid: np.ndarray,
    rho: np.ndarray = None,
    metric_tensor: np.ndarray = None,
) -> np.ndarray:
    """
    Ensure all centroids map to unique FFT grid points.
    
    If duplicates are found, attempts to redistribute them to nearby 
    unoccupied grid points (weighted by density if provided).
    
    Args:
        centroids_frac: (N_k, 3) fractional coordinates
        fft_grid: (3,) FFT grid dimensions
        rho: Optional (Nx, Ny, Nz) charge density for weighted redistribution
        metric_tensor: Optional (3,3) for PBC distance calculation
        
    Returns:
        centroids_frac_unique: (N_k, 3) fractional coordinates with no duplicates
    """
    fft_grid = np.asarray(fft_grid)
    centroids_frac = np.asarray(centroids_frac)
    N_k = centroids_frac.shape[0]
    
    # Snap to grid
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int) % fft_grid
    
    # Track which grid points are occupied
    occupied = set()
    result_indices = []
    duplicates = []
    
    for i, idx in enumerate(centroid_indices):
        key = tuple(idx)
        if key not in occupied:
            occupied.add(key)
            result_indices.append(idx)
        else:
            duplicates.append(i)
    
    if not duplicates:
        return centroids_frac
    
    print(f"Redistributing {len(duplicates)} duplicate centroids to nearby grid points...")
    
    # Build list of all unoccupied grid points
    all_points = set()
    for ix in range(fft_grid[0]):
        for iy in range(fft_grid[1]):
            for iz in range(fft_grid[2]):
                all_points.add((ix, iy, iz))
    unoccupied = list(all_points - occupied)
    
    if len(unoccupied) < len(duplicates):
        print(f"Warning: Not enough unoccupied grid points ({len(unoccupied)}) "
              f"for all duplicates ({len(duplicates)}). Some centroids will be lost.")
        duplicates = duplicates[:len(unoccupied)]
    
    # Assign duplicates to unoccupied points (greedy: use density if available)
    if rho is not None:
        # Sort unoccupied by density (highest first)
        rho = np.asarray(rho)
        unoccupied_with_density = [(pt, rho[pt]) for pt in unoccupied]
        unoccupied_with_density.sort(key=lambda x: -x[1])
        unoccupied = [pt for pt, _ in unoccupied_with_density]
    
    for dup_idx, new_pt in zip(duplicates, unoccupied):
        result_indices.append(np.array(new_pt))
        occupied.add(new_pt)
    
    result_indices = np.array(result_indices)
    centroids_frac_unique = result_indices.astype(float) / fft_grid
    
    print(f"Redistribution complete: {len(result_indices)} unique centroids")
    return centroids_frac_unique


# =============================================================================
# Main entry point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weighted k-means for ISDF sampling points")
    parser.add_argument("N_k", type=int, nargs="?", default=400,
                        help="Number of clusters/sampling points (default: 400)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--plot-zoom", type=float, default=1.0,
                        help="Zoom factor for density in plot (default: 1.0, higher = finer)")
    parser.add_argument("--no-downsample", action="store_true", help="Use full FFT grid (no zoom)")
    args = parser.parse_args()

    N_k = args.N_k
    print(f"Using N_k = {N_k} clusters")

    # Enable JAX persistent compile cache before any jit compiles.
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as _e:
        print(f"  [jax compile cache] skipped: {_e}", flush=True)

    wfn = WFNReader("WFN.h5")
    sym = symmetry_maps.SymMaps(wfn)

    charge_density = calculate_charge_density(wfn, sym)
    
    # Resize grid to target spacing of ~0.13 Å (gives ~10 points per 1.3 Å core region)
    # For each direction i:
    #   current_spacing = |a_i| / N_i
    #   zoom_factor = current_spacing / target_spacing
    # scipy.ndimage.zoom: new_N = old_N * zoom_factor
    target_spacing = 0.13 * 0.52  # Angstroms
    lattice_lengths = np.linalg.norm(wfn.avec, axis=1)  # |a1|, |a2|, |a3|
    current_spacing = lattice_lengths / np.array(charge_density.shape)
    zoom_factors = current_spacing / target_spacing
    
    # Don't upsample tiny grids, and cap zoom to avoid excessive memory
    if args.no_downsample or any(wfn.fft_grid < 20):
        zoom_factors = np.ones(3)
    zoom_factors = np.clip(zoom_factors, 0.1, 2.0)  # Reasonable bounds

    # Guard: if N_k is a significant fraction of the coarsened grid,
    # the k-means has too few points to cluster meaningfully and centroids
    # spread into vacuum. Skip downsampling in this case so the density
    # contrast is preserved (e.g. molecule in a large box).
    coarsened_size = int(np.prod(np.round(np.array(charge_density.shape) * zoom_factors)))
    if N_k > coarsened_size // 10:
        print(f"N_k={N_k} is >{coarsened_size//10} (10% of coarsened grid {coarsened_size})")
        print(f"Skipping downsampling to preserve density contrast")
        zoom_factors = np.ones(3)

    print(f"Charge density shape: {charge_density.shape}")
    print(f"Lattice lengths: {lattice_lengths} Å")
    print(f"Current spacing: {current_spacing} Å")
    print(f"Target spacing: {target_spacing} Å")
    print(f"Zoom factors: {zoom_factors}")
    rho_np = interpolate_density(charge_density, zoom_factors)
    rho_np_cpu = np.asarray(rho_np)
    avec_np_cpu = np.asarray(wfn.avec)

    rho_jax = jnp.asarray(rho_np_cpu, dtype=jnp.float32)
    avec_jax = jnp.asarray(avec_np_cpu, dtype=jnp.float32)

    _, centroids_jax, _, _ = weighted_kmeans_jax(
        avec_jax, rho_jax, N_k=N_k, seed=args.seed
    )

    centroids_frac = np.array(centroids_jax)
    
    # Snap centroids to the actual FFT grid and handle duplicates
    print(f"\nSnapping {N_k} centroids to FFT grid {wfn.fft_grid}...")
    centroid_indices, centroids_snapped, n_dups = snap_centroids_to_grid(
        centroids_frac, wfn.fft_grid, deduplicate=True
    )
    
    n_unique = centroid_indices.shape[0]
    if n_dups > 0:
        print(f"⚠ {n_dups} duplicates removed. Final count: {n_unique} unique centroids.")
        print(f"  Consider reducing N_k or using a finer FFT grid.")
        # Try to redistribute duplicates to nearby high-density points
        print("Attempting to redistribute duplicates to nearby grid points...")
        centroids_snapped = ensure_unique_centroids(
            centroids_frac, wfn.fft_grid, rho=charge_density
        )
        n_unique = centroids_snapped.shape[0]
        print(f"After redistribution: {n_unique} unique centroids")
    else:
        print(f"✓ All {n_unique} centroids map to unique grid points.")
    
    # Save the snapped (grid-aligned) centroids
    np.savetxt(
        f"centroids_frac_{n_unique}.txt",
        centroids_snapped,
        header=f"x y z (snapped to FFT grid {wfn.fft_grid}, {n_unique} unique points)",
        fmt="%.6f",
        delimiter=" ",
        comments="# ",
    )
    print(f"Saved centroids to centroids_frac_{n_unique}.txt")

    if not args.no_plot:
        if args.plot_zoom != 1.0:
            rho_plot = interpolate_density(charge_density, zoom_factors * args.plot_zoom)
        else:
            rho_plot = rho_np
        plot_density_and_centroids(wfn, rho_plot, centroids_snapped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
