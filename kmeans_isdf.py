import numpy as np
import os

# Configure JAX for four CPU devices so tests run the same everywhere
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from gpu_utils import cp
from wfnreader import WFNReader
import symmetry_maps
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side effects
from scipy.ndimage import zoom

from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test_scripts"))
from test_scripts.get_charge_density import calculate_charge_density
# This script selects ISDF sampling points via a weighted k-means algorithm.
# The density-driven clustering will remain relevant once the self-consistency
# loop is introduced, since new charge densities will require recomputing these
# centroids.

# Functions below keep previously used visualization logic available
# for debugging without affecting runtime when not called.

def interpolate_density(rho_np, zoom_factors=(1, 1, 1)):
    """Return a zoomed copy of ``rho_np`` using ``scipy.ndimage.zoom``."""
    if zoom_factors.all() == 1:
        return rho_np
    return zoom(rho_np, zoom_factors, order=3)


def plot_density_and_centroids(wfn, rho_np, centroids, labels=None):
    """Plot charge density and centroids in 3D."""
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
    # Handle both NumPy and CuPy arrays
    if hasattr(rho_np, 'get'):
        rho_shape = rho_np.shape
        threshold = 0.05 * float(rho_np.max().get())
    else:
        rho_shape = rho_np.shape
        threshold = 0.05 * np.amax(rho_np)
    
    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, rho_shape[0]),
        np.linspace(0, 1, rho_shape[1]),
        np.linspace(0, 1, rho_shape[2]),
        indexing="ij",
    )

    density_mask = rho_np > threshold
    # Convert CuPy arrays to NumPy if needed
    if hasattr(density_mask, 'get'):
        density_mask = density_mask.get()
    density_points = (
        np.stack([X[density_mask], Y[density_mask], Z[density_mask]], axis=1)
        @ wfn.avec
    )
    if hasattr(rho_np, 'get'):
        density_values = rho_np.get()[density_mask]
    else:
        density_values = rho_np[density_mask]

    scatter = ax.scatter(
        density_points[:, 0],
        density_points[:, 1],
        density_points[:, 2],
        c=np.log(np.abs(density_values) - 0.9 * threshold),
        cmap="viridis",
        alpha=0.05,
        s=20,
        marker="s",
        label="Density",
        zorder=1,
    )
    plt.colorbar(scatter, label="Charge Density")

    # Optional Voronoi cell visualization
    # mask0 = (labels == 0)
    # mask1 = (labels == 1)
    # voronoi0_points = positions[mask0].get()
    # ax.scatter(voronoi0_points[:, 0], voronoi0_points[:, 1], voronoi0_points[:, 2],
    #            c='blue', alpha=0.5, s=1, label='Voronoi Cell 0')
    # voronoi1_points = positions[mask1].get()
    # ax.scatter(voronoi1_points[:, 0], voronoi1_points[:, 1], voronoi1_points[:, 2],
    #            c='green', alpha=0.5, s=1, label='Voronoi Cell 1')

    if hasattr(centroids, "get"):
        centroids_np = centroids.get() @ wfn.avec
    else:
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
    plt.show()

def weighted_kmeans_cupy(
    avec,
    rho_cp,
    N_k=10,
    t=20,
    max_steps=200,
    tolerance=5e-3 # slight problem in that 
):
    print("Starting weighted k-means clustering")
    """
    Perform weighted k-means clustering using CuPy with periodic boundary conditions (PBC) in 3D.

    Parameters:
    - avec (cp.ndarray): Lattice vectors (3x3 CuPy array where each row is a lattice vector in Cartesian coordinates).
    - rho (cp.ndarray): Charge density array (3D CuPy array with shape corresponding to the grid size).
    - N_k (int): Number of clusters.
    - t (int): Multiplicative factor for initial centroid candidates (default=20).
    - max_steps (int): Maximum number of iterations (default=2000).
    - tolerance (float): Convergence tolerance based on centroid movement (default=1e-2).

    Returns:
    - centroids_indices (cp.ndarray): Indices of the final centroids in the grid.
    - centroids (cp.ndarray): Coordinates of the final centroids in real space.
    - centroid_z_history (cp.ndarray): Array recording the z-components of centroids at each step.
    - steps_taken (int): Number of steps taken until convergence or reaching max_steps.
    """
    # Define grid sizes based on rho shape
    grid_size_x, grid_size_y, grid_size_z = rho_cp.shape

    # Create synthetic Gaussian data
    #x = cp.linspace(0, 1, grid_size_x)
    #y = cp.linspace(0, 1, grid_size_y)
    #z = cp.linspace(0, 1, grid_size_z)
    #X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    
    # Create a 3D meshgrid of x, y, z values
    X, Y, Z = cp.meshgrid(
        cp.linspace(0,1,grid_size_x),
        cp.linspace(0,1,grid_size_y),
        cp.linspace(0,1,grid_size_z),
        indexing='ij'
    )
    frac_positions = cp.stack((X, Y, Z), axis=-1)  # Shape: (grid_x, grid_y, grid_z, 3)
    positions = frac_positions.reshape(-1, 3)  # Shape: (num_points, 3)
    avec_inv = cp.linalg.inv(avec)

    # Create a Gaussian centered at (0.5, 0.5, 0.5)
    #sigma = 0.1  # Width of Gaussian
    #rho = rho_cp**2
    #rho = cp.exp(-((X-0.5)**2 + (Y-0.5)**2 + (Z-0)**2)/(2*sigma**2)) + cp.exp(-((X-0.5)**2 + (Y-0.5)**2 + (Z-1.)**2)/(2*sigma**2))


    # Replace the random initialization with k-means++
    # Initialize array to store centroids
    centroids_frac = cp.zeros((N_k, 3), dtype=cp.float32)
    
    # Choose first centroid randomly with probability proportional to density
    probs = rho_cp.ravel() / rho_cp.sum()
    first_idx = cp.random.choice(len(positions), size=1, p=probs)[0]
    centroids_frac[0] = positions[first_idx]
    
    print("17. Starting k-means++ initialization loop...")
    
    # Pre-allocate all arrays at maximum size
    delta_frac = cp.zeros((positions.shape[0], N_k, 3), dtype=cp.float32)
    delta_cartesian = cp.zeros_like(delta_frac)
    min_dist_sq = cp.zeros(positions.shape[0], dtype=cp.float32)
    probs = cp.zeros_like(min_dist_sq)
    rho_flat = rho_cp.ravel()
    
    batch_size = 5  # Number of centroids to select per iteration
    
    for k in range(1, N_k, batch_size):
        print(f"{k}", end=' ', flush=True)
        # Calculate distances to existing centroids
        curr_k = min(k + batch_size, N_k)  # Don't exceed N_k
        
        # Use only the portion we need with existing centroids
        delta_frac[:, :k, :] = positions[:, cp.newaxis, :] - centroids_frac[:k, cp.newaxis, :].transpose(1, 0, 2)
        delta_frac[:, :k, :] = delta_frac[:, :k, :] - cp.round(delta_frac[:, :k, :])
        delta_cartesian[:, :k, :] = cp.matmul(delta_frac[:, :k, :], avec)
        min_dist_sq[:] = cp.min(cp.sum(delta_cartesian[:, :k, :]**2, axis=2), axis=1)
        
        # Select batch_size new centroids
        for b in range(k, curr_k):
            probs[:] = min_dist_sq * rho_flat
            probs[:] = probs / probs.sum()
            next_idx = cp.random.choice(len(positions), size=1, p=probs)[0]
            centroids_frac[b] = positions[next_idx]
            
            # Update min_dist_sq with the new centroid if not the last one in batch
            if b < curr_k - 1:
                delta_frac[:, b:b+1, :] = positions[:, cp.newaxis, :] - centroids_frac[b:b+1, cp.newaxis, :].transpose(1, 0, 2)
                delta_frac[:, b:b+1, :] = delta_frac[:, b:b+1, :] - cp.round(delta_frac[:, b:b+1, :])
                delta_cartesian[:, b:b+1, :] = cp.matmul(delta_frac[:, b:b+1, :], avec)
                new_dist_sq = cp.sum(delta_cartesian[:, b:b+1, :]**2, axis=2)
                min_dist_sq[:] = cp.minimum(min_dist_sq, new_dist_sq[:, 0])

    # Initialize array to record z-components of centroids at each step
    centroid_z_history = cp.zeros((N_k, max_steps), dtype=cp.float32)

    # Initialize variable to track steps taken
    steps_taken = max_steps

    # Open the movement log file
    with open('max_centroid_movement.txt', 'w') as movement_file:
        movement_file.write("Step, Max_Movement\n")  # Header

        for step in range(max_steps):
            # Convert positions and centroids to Cartesian coordinates first
            positions_cart = cp.matmul(positions, avec)  # Shape: (P, 3)
            centroids_cart = cp.matmul(centroids_frac, avec)  # Shape: (K, 3)

            # Compute distance vectors in Cartesian coordinates
            delta_cart = positions_cart[:, cp.newaxis, :] - centroids_cart[cp.newaxis, :, :]  # Shape: (P, K, 3)

            # Convert to fractional coordinates for PBC
            delta_frac = cp.matmul(delta_cart, avec_inv)
            
            # Apply minimal image convention in fractional coordinates
            delta_frac = delta_frac - cp.round(delta_frac)
            
            # Convert back to Cartesian for final distances
            delta_cart = cp.matmul(delta_frac, avec)
            
            # Compute Euclidean distances
            distances = cp.linalg.norm(delta_cart, axis=2)  # Shape: (P, K)

            # Assign each point to the nearest centroid
            labels = cp.argmin(distances, axis=1)  # Shape: (P,)

            # Create a mask for each centroid
            mask = cp.equal(labels[:, cp.newaxis], cp.arange(N_k))  # Shape: (P, K)

            # Reshape rho to (P, 1) for broadcasting
            rho_flat = rho_cp.ravel()[:, cp.newaxis]  # Shape: (P, 1)

            # Apply mask to rho to get weights for each centroid
            masked_rho = mask * rho_flat  # Shape: (P, K)

            # Multiply each delta_cartesian by the masked_rho
            weighted_positions = masked_rho[:, :, cp.newaxis] * delta_cart  # Shape: (P, K, 3)

            # Sum weighted positions for each centroid, convert to fractional coordinates
            sum_weighted_frac = weighted_positions.sum(axis=0)   # Shape: (K, 3)

            # Sum weights for each centroid
            sum_weights = masked_rho.sum(axis=0)  # Shape: (K,)

            # Avoid division by zero and do not reinitialize centroids with zero weight
            valid = sum_weights > 0
            new_centroids_frac = centroids_frac.copy()
            new_centroids_frac[valid] = centroids_frac[valid] + cp.matmul(sum_weighted_frac[valid] / sum_weights[valid, cp.newaxis], avec_inv)
            # Wrap fractional centroids to [0, 1)

            # Calculate centroid movement
            centroid_movement = cp.linalg.norm(new_centroids_frac - centroids_frac, axis=1)
            max_movement = cp.max(centroid_movement)
            if hasattr(max_movement, "get"):
                max_movement = max_movement.get()
            movement_file.write(f"{step}, {max_movement}\n")  # Log the maximum movement
            
            new_centroids_frac = new_centroids_frac % 1.0

            # Record the z-components of centroids
            centroid_z_history[:, step] = new_centroids_frac[:, 2]

            # Check for convergence
            if cp.all(centroid_movement < tolerance):
                print(f"Converged in {step} steps.")
                steps_taken = step
                centroids_frac = new_centroids_frac
                break

            # Print every 10th step
            if step % 10 == 0:
                print(f"Step {step}")

            # Update centroids for next iteration
            centroids_frac = new_centroids_frac

        else:
            print(f"Reached max steps ({max_steps}) without full convergence.")

    # Convert final centroid fractional coordinates to Cartesian coordinates
    #centroids = cp.matmul(centroids_frac, avec)  # Shape: (K, 3)

    return labels, centroids_frac, centroid_z_history, steps_taken


def weighted_kmeans_jax(
    avec,
    rho_jax,
    N_k=10,
    max_steps=200,
    tolerance=5e-3,
    seed=0,
):
    """Weighted k-means using JAX on multiple CPU devices."""
    devices = jax.devices()
    n_dev = len(devices)

    grid_x, grid_y, grid_z = rho_jax.shape

    X, Y, Z = jnp.meshgrid(
        jnp.linspace(0, 1, grid_x),
        jnp.linspace(0, 1, grid_y),
        jnp.linspace(0, 1, grid_z),
        indexing="ij",
    )
    positions = jnp.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    avec_inv = jnp.linalg.inv(avec)

    rho_flat = rho_jax.reshape(-1)
    key = jax.random.PRNGKey(seed)
    centroids = jnp.zeros((N_k, 3), dtype=jnp.float32)
    probs = rho_flat / rho_flat.sum()
    key, sub = jax.random.split(key)
    first_idx = jax.random.choice(sub, positions.shape[0], shape=(1,), p=probs)[0]
    centroids = centroids.at[0].set(positions[first_idx])

    min_dist_sq = jnp.zeros(positions.shape[0])
    for k in range(1, N_k):
        diff = positions[:, None, :] - centroids[:k][None, :, :]
        diff = diff - jnp.round(diff)
        dcart = diff @ avec
        dist_sq = jnp.sum(dcart**2, axis=2)
        if k == 1:
            min_dist_sq = dist_sq[:, 0]
        else:
            min_dist_sq = jnp.minimum(min_dist_sq, dist_sq[:, k - 1])
        probs = min_dist_sq * rho_flat
        probs = probs / probs.sum()
        key, sub = jax.random.split(key)
        next_idx = jax.random.choice(sub, positions.shape[0], shape=(1,), p=probs)[0]
        centroids = centroids.at[k].set(positions[next_idx])

    # Shard the flattened real-space grid across devices
    total_points = positions.shape[0]
    indices = np.array_split(np.arange(total_points), n_dev)
    pos_slices = [positions[idx] for idx in indices]
    rho_slices = [rho_flat[idx] for idx in indices]
    pos_sharded = jax.device_put_sharded(pos_slices, devices)
    rho_sharded = jax.device_put_sharded(rho_slices, devices)
    centroids_rep = jax.device_put_replicated(centroids, devices)
    avec_rep = jax.device_put_replicated(avec, devices)
    inv_rep = jax.device_put_replicated(avec_inv, devices)

    @partial(jax.pmap, axis_name="d")
    def partial_sums(pos_slice, rho_slice, cents, a, a_inv):
        pos_cart = pos_slice @ a
        cent_cart = cents @ a
        delta_cart = pos_cart[:, None, :] - cent_cart[None, :, :]
        delta_frac = delta_cart @ a_inv
        delta_frac = delta_frac - jnp.round(delta_frac)
        delta_cart = delta_frac @ a
        dists = jnp.linalg.norm(delta_cart, axis=2)
        labels = jnp.argmin(dists, axis=1)
        mask = jax.nn.one_hot(labels, N_k, dtype=rho_slice.dtype)
        weights = rho_slice[:, None]
        weighted = weights[:, :, None] * delta_cart
        sum_w_pos = jnp.sum(weighted, axis=0)
        sum_w = jnp.sum(mask * weights, axis=0)
        return jax.lax.psum(sum_w_pos, "d"), jax.lax.psum(sum_w, "d")

    history = jnp.zeros((N_k, max_steps), dtype=jnp.float32)
    steps_taken = max_steps
    for step in range(max_steps):
        sum_pos, sum_w = partial_sums(pos_sharded, rho_sharded, centroids_rep, avec_rep, inv_rep)
        total_pos = np.array(sum_pos[0])
        total_w = np.array(sum_w[0])
        cent_np = np.array(centroids)
        valid = total_w > 0
        cent_np[valid] = cent_np[valid] + (total_pos[valid] / total_w[valid, None]) @ np.linalg.inv(np.array(avec))
        movement = np.linalg.norm(cent_np - np.array(centroids), axis=1)
        history = history.at[:, step].set(jnp.array(cent_np % 1.0)[:, 2])
        centroids = jnp.array(cent_np % 1.0, dtype=jnp.float32)
        centroids_rep = jax.device_put_replicated(centroids, devices)
        if np.all(movement < tolerance):
            steps_taken = step
            break

    diff = positions[:, None, :] - centroids[None, :, :]
    diff = diff - jnp.round(diff)
    dcart = diff @ avec
    labels = jnp.argmin(jnp.linalg.norm(dcart, axis=2), axis=1)

    return labels, centroids, history, steps_taken

if __name__ == "__main__":
    wfn = WFNReader("WFNsmall.h5")
    sym = symmetry_maps.SymMaps(wfn)

    charge_density = calculate_charge_density(wfn, sym)
    # making interpolation the default: we could do fourier interpolation 
    # (a good idea actually) but it's probably fine to use the scipy function. 
    # the 3d orbital psuedization radius is 1.3 au; assume we want to sample 10 points in that radius.
    zoom_factors = np.round(np.linalg.norm(wfn.avec, axis=0)/0.13) / charge_density.shape
    print("charge density shape: ", charge_density.shape)
    print("zoom factors: ", zoom_factors)
    rho_np = interpolate_density(charge_density, zoom_factors)  # default no-op
    rho_jax = jnp.asarray(rho_np, dtype=jnp.float32)
    avec_jax = jnp.asarray(wfn.avec, dtype=jnp.float32)

    _, centroids, _, _ = weighted_kmeans_jax(avec_jax, rho_jax, N_k=16)

    centroids_np = np.array(centroids)
    np.savetxt(
        "centroids_frac.txt",
        centroids_np,
        header="x y z",
        fmt="%.6f",
        delimiter=" ",
        comments="# ",
    )

    # Uncomment to visualize the charge density and centroids
    plot_density_and_centroids(wfn, rho_np, centroids)
