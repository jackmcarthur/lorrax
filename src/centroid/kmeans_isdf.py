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

import os
import numpy as np

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map
from functools import partial


# Multi-process bootstrap. When launched under `srun -n N>1` each rank is a
# separate Python process with its own local GPU; `jax.distributed.initialize()`
# stitches them into one multi-host JAX job so `jax.devices()` returns all N
# GPUs and we can build a single `Mesh` spanning them. Matches the pattern in
# ``psp/run_nscf.py`` and ``gw/gw_jax.py``.
_DISTRIBUTED_SENTINEL = "_LORRAX_KMEANS_DISTRIBUTED_DONE"


def _maybe_init_jax_distributed() -> None:
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return
    proc_count = int(
        os.environ.get("JAX_PROCESS_COUNT",
        os.environ.get("JAX_NUM_PROCESSES",
        os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            return
        except Exception:
            pass
        coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
        if coord is None:
            import subprocess
            nodelist = os.environ.get("SLURM_NODELIST")
            if nodelist:
                try:
                    result = subprocess.run(
                        ["scontrol", "show", "hostnames", nodelist],
                        capture_output=True, text=True, check=True)
                    coord = f"{result.stdout.strip().splitlines()[0]}:12355"
                except Exception:
                    pass
            if coord is None:
                host = (os.environ.get("SLURMD_NODENAME") or
                        os.environ.get("HOSTNAME") or "localhost")
                coord = f"{host}:12355"
        proc_id = int(os.environ.get("JAX_PROCESS_INDEX",
                                     os.environ.get("SLURM_PROCID", "0")))
        jax.distributed.initialize(coordinator_address=coord,
                                   num_processes=proc_count,
                                   process_id=proc_id)
    os.environ[_DISTRIBUTED_SENTINEL] = "1"


_maybe_init_jax_distributed()

print(f"✓ JAX initialized: {len(jax.devices())} device(s) "
      f"(local: {len(jax.local_devices())}, proc {jax.process_index()}/"
      f"{jax.process_count()})")

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
        centroids_frac: (C, 3) fractional coordinates of centroids
        metric_tensor: (3, 3) G = avec^T @ avec
        
    Returns:
        distances_sq: (P, C) squared Cartesian distances with PBC
    """
    # Fractional displacement: shape (P, C, 3)
    delta_frac = positions_frac[:, None, :] - centroids_frac[None, :, :]
    
    # Minimal image convention: wrap to [-0.5, 0.5)
    # This finds the closest periodic image among all 27 neighboring cells
    delta_frac = delta_frac - jnp.round(delta_frac)
    
    # Squared distance using metric tensor: d² = df @ G @ df^T
    # einsum 'pci,ij,pcj->pc' computes this for all (P, C) pairs efficiently
    distances_sq = jnp.einsum('pci,ij,pcj->pc', delta_frac, metric_tensor, delta_frac)
    
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


# Default C-block size for the chunked PBC distance pass. Chosen so that for
# typical P ~ 10^5-10^6 and float32 the peak (P, C_BLOCK, 3) tensor stays under
# ~100 MB (e.g. 1e6 * 32 * 3 * 4 B ≈ 380 MB; 1e5 * 32 * 3 * 4 B ≈ 40 MB).
_DEFAULT_C_BLOCK = 32


@partial(jax.jit, static_argnames=['n_c', 'c_block'])
def assign_labels_chunked(
    positions_frac: jnp.ndarray,
    centroids_frac: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    n_c: int,
    c_block: int = _DEFAULT_C_BLOCK,
) -> jnp.ndarray:
    """Nearest-centroid assignment via a C-chunked scan.

    Computes ``argmin_c d²(pos_p, cent_k)`` without ever materializing the full
    (P, C, 3) PBC displacement tensor. Inside the scan we hold at most a
    (P, c_block, 3) slab. PBC minimal image + metric tensor are unchanged.

    Args:
        positions_frac: (P, 3) fractional grid-point coordinates.
        centroids_frac: (C, 3) fractional centroid coordinates. C must be
            divisible by ``c_block`` (pad the caller side if not).
        metric_tensor: (3, 3) G = avec^T @ avec, in whatever units ``avec`` has.
        n_c: Static C (equals ``centroids_frac.shape[0]``).
        c_block: Static C-chunk size. Peak tensor inside the scan is
            (P, c_block, 3).

    Returns:
        labels: (P,) int32 cluster assignment.
    """
    # Pad centroids to a multiple of c_block with NaN rows. NaN-in → NaN-out
    # in the quadratic form below, sanitized to +inf before argmin so they
    # never win. This lets c_block stay at the CUDA-friendly default (32)
    # regardless of whether N_c is divisible — and avoids the old fallback
    # to a tiny c_block (e.g. c_block=25 at N_c=2000 gave 50 % SM occupancy).
    n_chunks = (n_c + c_block - 1) // c_block
    n_c_padded = n_chunks * c_block
    if n_c_padded != n_c:
        pad = jnp.full((n_c_padded - n_c, 3), jnp.nan, dtype=centroids_frac.dtype)
        centroids_frac = jnp.concatenate([centroids_frac, pad], axis=0)
    centroids_blocked = centroids_frac.reshape(n_chunks, c_block, 3)

    # scan carry dtypes must be invariant across iterations; argmin's output
    # dtype depends on whether jax_enable_x64 is on, so we cast explicitly.
    idx_dtype = jnp.int32
    dist_dtype = metric_tensor.dtype

    # Pre-extract metric-tensor scalar entries so the scan body becomes a pure
    # pipeline of element-wise ops (no einsum → no (P·c_block)×3 GEMM). On Si
    # at N_c=2000 the einsum path compiled to `ampere_sgemm_32x128_nn` at 50 %
    # SM occupancy (K=3 is a terrible GEMM inner dim); this form fuses the
    # subtract → round → quadratic-form → reduce into a single elementwise
    # kernel with ~3× better occupancy and no (P, c_block, 3) remat.
    G = metric_tensor
    g00, g01, g02 = G[0, 0], G[0, 1], G[0, 2]
    g11, g12, g22 = G[1, 1], G[1, 2], G[2, 2]

    def body(carry, cent_chunk):
        best_d, best_c, chunk_idx = carry
        # Per-axis PBC-wrapped displacement — 3 separate (P, c_block) tensors,
        # never a (P, c_block, 3) buffer. XLA fuses subtract + round + the
        # quadratic form below into one elementwise kernel.
        dx = positions_frac[:, None, 0] - cent_chunk[None, :, 0]
        dx = dx - jnp.round(dx)
        dy = positions_frac[:, None, 1] - cent_chunk[None, :, 1]
        dy = dy - jnp.round(dy)
        dz = positions_frac[:, None, 2] - cent_chunk[None, :, 2]
        dz = dz - jnp.round(dz)
        # d² = δᵀ G δ for symmetric G:
        #    = G₀₀ dx² + G₁₁ dy² + G₂₂ dz²
        #      + 2 (G₀₁ dx·dy + G₀₂ dx·dz + G₁₂ dy·dz)
        d_chunk = (g00 * dx + g01 * dy + g02 * dz) * dx \
                + (g01 * dx + g11 * dy + g12 * dz) * dy \
                + (g02 * dx + g12 * dy + g22 * dz) * dz
        # NaN sentinel for padded centroids: NaN-in produces NaN-out here.
        d_chunk = jnp.where(jnp.isnan(d_chunk), jnp.inf, d_chunk)
        local_c = jnp.argmin(d_chunk, axis=1).astype(idx_dtype)                # (P,)
        local_d = jnp.take_along_axis(d_chunk, local_c[:, None], axis=1)[:, 0] # (P,)
        global_c = (chunk_idx * c_block + local_c).astype(idx_dtype)
        better = local_d < best_d
        best_d = jnp.where(better, local_d, best_d).astype(dist_dtype)
        best_c = jnp.where(better, global_c, best_c).astype(idx_dtype)
        return (best_d, best_c, chunk_idx + 1), None

    init = (
        jnp.full((positions_frac.shape[0],), jnp.inf, dtype=dist_dtype),
        jnp.zeros((positions_frac.shape[0],), dtype=idx_dtype),
        jnp.asarray(0, dtype=idx_dtype),
    )
    (_, labels, _), _ = lax.scan(body, init, centroids_blocked)
    return labels


def _local_update_accumulators(
    positions_frac: jnp.ndarray,
    centroids_frac: jnp.ndarray,
    rho_flat: jnp.ndarray,
    labels: jnp.ndarray,
    n_c: int,
):
    """Given labels, compute the per-centroid weighted-displacement sums.

    Avoids the (P, C, 3) tensor entirely: gathers the assigned centroid for
    each point (P, 3), computes a single PBC-wrapped displacement (P, 3), and
    uses ``segment_sum`` to reduce P → C. Memory is O(P + C), not O(P·C).

    The PBC minimal image is applied to ``pos - cent[label]`` and only that,
    which matches the original algorithm because ``mask[p, k] = (labels[p]==k)``
    zeroed out every column except the assigned one anyway.
    """
    assigned_cent = centroids_frac[labels]                       # (P, 3)
    delta = positions_frac - assigned_cent                        # (P, 3)
    delta = delta - jnp.round(delta)                              # (P, 3) PBC
    weighted_delta = rho_flat[:, None] * delta                    # (P, 3)
    sum_weighted_delta = jax.ops.segment_sum(
        weighted_delta, labels, num_segments=n_c
    )                                                             # (C, 3)
    sum_weights = jax.ops.segment_sum(
        rho_flat, labels, num_segments=n_c
    )                                                             # (C,)
    return sum_weighted_delta, sum_weights


def _finalize_update(
    centroids_frac: jnp.ndarray,
    sum_weighted_delta: jnp.ndarray,
    sum_weights: jnp.ndarray,
    metric_tensor: jnp.ndarray,
):
    """Turn (sum_weighted_delta, sum_weights) into new centroids + movement."""
    avg_delta = jnp.where(
        sum_weights[:, None] > 0,
        sum_weighted_delta / jnp.maximum(sum_weights[:, None], 1e-10),
        0.0,
    )
    new_centroids_frac = (centroids_frac + avg_delta) % 1.0
    movement_frac = new_centroids_frac - centroids_frac
    movement_frac = movement_frac - jnp.round(movement_frac)
    movement_sq = jnp.einsum('ci,ij,cj->c', movement_frac, metric_tensor, movement_frac)
    return new_centroids_frac, movement_sq


@partial(jax.jit, static_argnames=['n_c', 'c_block'])
def kmeans_update_step(
    positions_frac: jnp.ndarray,
    centroids_frac: jnp.ndarray,
    rho_flat: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    n_c: int,
    c_block: int = _DEFAULT_C_BLOCK,
) -> tuple:
    """Single k-means update step: assign labels and compute new centroids.

    This is the core k-means iteration, JIT-compiled for speed:
        1. Assign each point to nearest centroid (C-chunked PBC distance scan)
        2. For each point, compute PBC-wrapped displacement to its assigned centroid
        3. ``segment_sum`` over labels → (C, 3) accumulator
        4. Update centroid positions, wrap to [0, 1), compute PBC-wrapped movement

    The weighted mean is computed in *displacement space* relative to the
    current centroid, using PBC-wrapped displacements. This ensures the
    centroid moves towards its assigned points correctly even when points
    wrap around boundaries. Mathematically equivalent to the one-hot-mask form
    ``mask[p,k]·(pos_p - cent_k)``, since the mask zeroes all columns except
    the assigned one.

    Peak memory is (P, c_block, 3) inside the distance scan and (P, 3) for
    the mean-update — no (P, C, 3) tensor is ever materialized.

    Args:
        positions_frac: (P, 3) fractional coordinates of grid points.
        centroids_frac: (C, 3) fractional coordinates of centroids. C must be
            divisible by ``c_block``; the caller pads if needed.
        rho_flat: (P,) charge density weights.
        metric_tensor: (3, 3) G = avec^T @ avec. Distances come out in the
            same units ``avec`` has (Å² if avec is in Å, alat² if in alat, etc.).
        n_c: Number of clusters (must match centroids_frac.shape[0]).
        c_block: Static C-chunk size for the distance scan.

    Returns:
        new_centroids_frac: (C, 3) updated centroid positions.
        movement_sq: (C,) squared centroid movement (for convergence check).
        labels: (P,) cluster assignment for each point.
    """
    labels = assign_labels_chunked(
        positions_frac, centroids_frac, metric_tensor, n_c, c_block=c_block
    )
    sum_weighted_delta, sum_weights = _local_update_accumulators(
        positions_frac, centroids_frac, rho_flat, labels, n_c
    )
    new_centroids_frac, movement_sq = _finalize_update(
        centroids_frac, sum_weighted_delta, sum_weights, metric_tensor
    )
    return new_centroids_frac, movement_sq, labels


# =============================================================================
# Multi-device sharded kmeans step
# =============================================================================
# Shard the grid-point axis P across a 1D device mesh named 'x'. Centroids (C, 3),
# the metric tensor, and the final updates are replicated. Inside each shard we
# run the same local computation as the single-device path; the only cross-device
# communication is a `psum` on the (C, 3) weighted-sum and (C,) weight accumulators,
# one collective per iteration. This is the pattern the user asked for: "centroids
# replicated, distance + assignment parallelized along P".

def _kmeans_update_step_sharded_impl(
    positions_frac,
    centroids_frac,
    rho_flat,
    metric_tensor,
    n_c: int,
    c_block: int,
):
    """Per-shard body for `kmeans_update_step_sharded`. See that function.

    Runs inside `shard_map` with P sharded along axis 'x'. Each device sees
    only its P_local slice; the two `psum` collectives combine the per-shard
    `segment_sum` accumulators into globally correct (C, 3) / (C,) tensors.
    """
    # Per-shard local label assignment (P_local, 3) × replicated (C, 3) → (P_local,)
    local_labels = assign_labels_chunked(
        positions_frac, centroids_frac, metric_tensor, n_c, c_block=c_block
    )
    # Per-shard weighted-sum accumulators, then cross-device reduction.
    local_wsum, local_w = _local_update_accumulators(
        positions_frac, centroids_frac, rho_flat, local_labels, n_c
    )
    sum_weighted_delta = lax.psum(local_wsum, axis_name='x')      # (C, 3) replicated
    sum_weights = lax.psum(local_w, axis_name='x')                # (C,) replicated
    new_centroids_frac, movement_sq = _finalize_update(
        centroids_frac, sum_weighted_delta, sum_weights, metric_tensor
    )
    return new_centroids_frac, movement_sq, local_labels


def make_sharded_kmeans_update(mesh: Mesh, n_c: int, c_block: int = _DEFAULT_C_BLOCK):
    """Build a sharded `kmeans_update_step` bound to a device mesh.

    The returned callable expects:
      * positions_frac, rho_flat sharded along ``mesh['x']`` on the P axis.
      * centroids_frac, metric_tensor replicated.

    and returns:
      * new_centroids_frac, movement_sq replicated (identical on every device).
      * labels sharded along ``mesh['x']`` on the P axis (each device sees its
        local slice, matching the input sharding).

    We use ``shard_map`` so the single ``lax.psum`` inside the body is explicit
    and the compiler has complete freedom to fuse the local work.

    Args:
        mesh: 1-axis device mesh, axis name must be 'x'.
        n_c: Static number of centroids (must match centroids_frac.shape[0]).
        c_block: Static C-chunk size.
    """
    if 'x' not in mesh.axis_names:
        raise ValueError(f"Mesh must have an 'x' axis, got {mesh.axis_names}")

    in_specs = (
        PartitionSpec('x', None),   # positions_frac (P, 3) sharded on P
        PartitionSpec(None, None),  # centroids_frac (C, 3) replicated
        PartitionSpec('x'),         # rho_flat (P,) sharded on P
        PartitionSpec(None, None),  # metric_tensor (3, 3) replicated
    )
    out_specs = (
        PartitionSpec(None, None),  # new_centroids replicated
        PartitionSpec(None),        # movement_sq replicated
        PartitionSpec('x'),         # labels sharded on P
    )

    @partial(jax.jit, static_argnames=())
    def step(positions_frac, centroids_frac, rho_flat, metric_tensor):
        return shard_map(
            lambda p, c, r, g: _kmeans_update_step_sharded_impl(p, c, r, g, n_c, c_block),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )(positions_frac, centroids_frac, rho_flat, metric_tensor)

    return step


# =============================================================================
# k-means++ initialization (fully on-device)
# =============================================================================
# The previous init was a N_c-iteration Python loop with per-iteration
# host round-trips: `rng.choice(p=np.array(weights))` + `centroids.at[k].set(...)`.
# The profiler showed that on Si 4×4×4 N_c=128 this dominated wall time
# (~1000 H2D copies, ~6 s of dispatch overhead). Replacing the host-side
# categorical sampler with the Gumbel-max trick lets the whole init run as
# a single `lax.fori_loop`: no per-iteration H2D, one jit dispatch total.
#
# Gumbel-max: for independent g_i ~ Gumbel(0, 1),
#     argmax_i ( log(w_i) + g_i )  ~  Categorical( w / Σw )
# so sampling reduces to a pointwise add + argmax, both of which stay on device.


def _gumbel_sample_argmax(log_weights: jnp.ndarray, key) -> jnp.ndarray:
    """Sample one categorical draw via Gumbel-max on log-weights.

    ``log_weights[i] = -inf`` corresponds to zero probability and is simply
    never selected (since ``-inf + finite = -inf``). Uniform samples are
    clamped away from 0 to avoid ``log(0)`` in the Gumbel inverse-CDF.
    """
    n = log_weights.shape[0]
    # Clamp u ∈ (eps, 1 - eps) so -log(-log(u)) is finite for fp32.
    eps = jnp.finfo(log_weights.dtype).tiny
    u = jax.random.uniform(key, (n,), dtype=log_weights.dtype,
                           minval=eps, maxval=1.0 - eps)
    gumbel = -jnp.log(-jnp.log(u))
    return jnp.argmax(log_weights + gumbel)


@partial(jax.jit, static_argnames=['n_c'])
def kmeans_pp_init(
    positions_frac: jnp.ndarray,
    rho_flat: jnp.ndarray,
    metric_tensor: jnp.ndarray,
    n_c: int,
    key,
) -> jnp.ndarray:
    """Density-weighted k-means++ initialization on device, no H2D per step.

    Draws the first centroid from ``Categorical(ρ)`` and each subsequent one
    from ``Categorical(D²·ρ)`` where D(x) is the PBC-aware distance to the
    nearest already-placed centroid — exactly the standard k-means++ weighting
    used by the previous host-loop implementation. Sampling is done by
    Gumbel-max on ``log(weights)`` so the entire init is a single
    ``lax.fori_loop``.

    PBC minimal image and the metric-tensor distance are unchanged
    (``pbc_distance_sq_single`` is reused verbatim).

    Args:
        positions_frac: (P, 3) fractional coordinates.
        rho_flat:       (P,) density weights (non-negative).
        metric_tensor:  (3, 3) G = avec^T @ avec; units match ``avec``.
        n_c:            static number of centroids.
        key:            JAX PRNGKey.

    Returns:
        centroids_frac: (N_c, 3) fractional centroid coordinates.
    """
    n_points = positions_frac.shape[0]
    dtype = positions_frac.dtype

    # log-weights; clamp at tiny so log(0) never appears.
    tiny = jnp.finfo(dtype).tiny
    log_rho = jnp.log(jnp.maximum(rho_flat.astype(dtype), tiny))

    # First centroid: draw from Categorical(ρ).
    key, sub = jax.random.split(key)
    first_idx = _gumbel_sample_argmax(log_rho, sub)
    first_cent = positions_frac[first_idx]

    centroids = jnp.zeros((n_c, 3), dtype=dtype).at[0].set(first_cent)
    min_d2 = pbc_distance_sq_single(positions_frac, first_cent, metric_tensor)

    def body(c_idx, state):
        centroids, min_d2, key = state
        key, sub = jax.random.split(key)
        # k-means++ weights: D²(x)·ρ(x). On log scale this is additive.
        log_w = jnp.log(jnp.maximum(min_d2, tiny)) + log_rho
        next_idx = _gumbel_sample_argmax(log_w, sub)
        new_cent = positions_frac[next_idx]
        centroids = centroids.at[c_idx].set(new_cent)
        # Running pointwise minimum of squared PBC distance — same
        # incremental update the previous loop did.
        d2_new = pbc_distance_sq_single(positions_frac, new_cent, metric_tensor)
        min_d2 = jnp.minimum(min_d2, d2_new)
        return centroids, min_d2, key

    centroids, _, _ = lax.fori_loop(1, n_c, body, (centroids, min_d2, key))
    return centroids


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

def _pick_c_block(n_c: int, preferred: int = _DEFAULT_C_BLOCK) -> int:
    """Pick the C-chunk size for ``assign_labels_chunked``.

    Returns ``min(preferred, n_c)``. The scan pads centroids to the next
    multiple of ``c_block`` with NaN rows (handled inside the scan body), so
    we no longer need ``c_block`` to divide ``n_c``. Keeping ``c_block`` at
    the default (32) regardless of N_c keeps GEMM tiles a reasonable size
    on the GPU — the previous divisor-based pick dropped to c_block=25 at
    N_c=2000 (80 chunks, 50 % SM occupancy) which was unnecessarily small.
    """
    return min(preferred, n_c)


def _pad_centroids(centroids: jnp.ndarray, n_padded: int) -> jnp.ndarray:
    """Pad centroids with NaN rows up to ``n_padded`` total.

    NaN positions produce NaN distances in the scan body, which are sanitized
    to +inf before argmin, so padded rows never win. Padded rows also never
    get assigned any points (labels cap at the real range), so their
    ``segment_sum`` bucket is 0 and their updated position is NaN — the caller
    discards those trailing rows.
    """
    n_real = centroids.shape[0]
    if n_padded == n_real:
        return centroids
    pad = jnp.full((n_padded - n_real, 3), jnp.nan, dtype=centroids.dtype)
    return jnp.concatenate([centroids, pad], axis=0)


def weighted_kmeans_jax(
    avec: jnp.ndarray,
    rho_jax: jnp.ndarray,
    N_c: int = 10,
    max_steps: int = 200,
    tolerance: float = 5e-3,
    seed: int = 0,
    mesh: Mesh | None = None,
    force_shard: bool = False,
) -> tuple:
    """
    Density-weighted k-means clustering with periodic boundary conditions.

    Uses k-means++ initialization for better initial centroid placement,
    then iterates Lloyd's algorithm until convergence.

    Key features:
    - Minimal image convention for PBC (considers all 27 neighboring cells).
    - Metric tensor for efficient squared distance computation; the tensor
      carries whatever units ``avec`` has, so pass avec in the physical length
      unit you want the tolerance/movement to be in (typically Å).
    - Lloyd's iteration is JIT-compiled and uses ``segment_sum`` for the
      weighted-mean step — the (P, C, 3) displacement tensor is never
      materialized.
    - C-chunked distance scan bounds peak memory to (P, c_block, 3).
    - If ``mesh`` is provided (1-axis mesh named 'x' over the available
      devices), the Lloyd step runs via ``shard_map``: P is sharded on 'x',
      centroids are replicated, and one ``psum`` per step combines the
      per-shard (C, 3) weighted-sum accumulators. Centroids stay
      bit-identical across devices.

    Args:
        avec: (3, 3) lattice vectors (rows are a1, a2, a3). Units determine
            the units of the reported movement / the tolerance; for physical
            Angstrom tolerance pass avec in Å.
        rho_jax: (Nx, Ny, Nz) charge density on real-space grid.
        N_c: Number of cluster centroids (ISDF sampling points).
        max_steps: Maximum k-means iterations.
        tolerance: Convergence tolerance for centroid movement, in the same
            units as ``avec`` (Å if avec is Å).
        seed: Random seed for reproducibility.
        mesh: Optional 1-axis device mesh named 'x' for multi-GPU Lloyd.

    Returns:
        labels: (P,) cluster assignment for each grid point.
        centroids: (N_c, 3) final centroid positions in fractional coordinates.
        history: (N_c, max_steps) z-coordinate history (for debugging).
        steps_taken: Number of iterations until convergence.
    """
    print(f"Starting weighted k-means with {N_c} clusters "
          f"(mesh={'x=' + str(mesh.shape['x']) if mesh is not None else 'single-device'})...")

    # Precompute metric tensor: G = avec^T @ avec. Distances inherit avec's units.
    metric_tensor = jnp.array(avec.T @ avec, dtype=jnp.float32)
    tolerance_sq = tolerance ** 2  # Compare squared distances for efficiency.

    # Build grid of fractional positions (float32 to match centroids dtype).
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

    # Pick c_block as the largest divisor of N_c that is ≤ _DEFAULT_C_BLOCK,
    # so the chunked scan needs no padding. (Padding path works, but divisors
    # are cleaner and equally fast.)
    c_block = _pick_c_block(N_c)
    _n_chunks_scan = (N_c + c_block - 1) // c_block
    _pad = _n_chunks_scan * c_block - N_c
    print(f"C-chunk size for distance scan: {c_block} ({_n_chunks_scan} chunks"
          f"{f', padded with {_pad} NaN sentinels' if _pad else ''})")

    # Auto-gate: for small per-shard P the NCCL launch latency dominates the
    # actual shard compute, so sharding makes things slower. Profile data on
    # Si 4×4×4 (P=110k, 4 GPUs) showed ~4.2 ms max allreduce vs sub-ms local
    # compute, and the sharded run was ~2 s slower overall. Rule of thumb:
    # only shard when each device gets at least ~1e5 points. Pass
    # ``force_shard=True`` to bypass this heuristic (e.g. for benchmarking,
    # or when N_c is large enough that Lloyd compute dominates allreduce
    # latency even with small per-shard P).
    _P_PER_SHARD_MIN = 100_000
    sharded_step = None
    if mesh is not None:
        per_shard = n_points // mesh.shape['x']
        if n_points % mesh.shape['x'] != 0:
            raise ValueError(
                f"n_points={n_points} must be divisible by mesh x-axis "
                f"size {mesh.shape['x']} for even P-sharding"
            )
        if per_shard < _P_PER_SHARD_MIN and not force_shard:
            print(f"[weighted_kmeans_jax] P/{mesh.shape['x']} = {per_shard} "
                  f"< {_P_PER_SHARD_MIN}; falling back to single-device "
                  f"(collective latency would dominate). Pass force_shard=True "
                  f"to override.")
            mesh = None
        else:
            if force_shard and per_shard < _P_PER_SHARD_MIN:
                print(f"[weighted_kmeans_jax] P/{mesh.shape['x']} = {per_shard} "
                      f"< {_P_PER_SHARD_MIN}, but force_shard=True — using mesh anyway.")
            sharded_step = make_sharded_kmeans_update(mesh, N_c, c_block=c_block)
    
    # =========================================================================
    # K-means++ initialization — fully on-device, single jit dispatch.
    # =========================================================================
    # Density-weighted k-means++ with PBC minimal-image distances. Gumbel-max
    # sampling replaces the former host-side `rng.choice(p=...)`, so the whole
    # N_c-step selection runs inside one `lax.fori_loop`. See `kmeans_pp_init`.
    # =========================================================================
    print("K-means++ initialization (on-device fori_loop)...")

    centroids = kmeans_pp_init(
        positions, rho_flat, metric_tensor, N_c, jax.random.PRNGKey(seed)
    )
    # Bring the centroids to host once (tiny, (N_c, 3)) so the Lloyd driver's
    # optional resharding step below can reshape them freely.
    centroids.block_until_ready()
    print("K-means++ initialization complete.")

    # =========================================================================
    # Lloyd's algorithm (iterative refinement)
    # =========================================================================
    print("Starting k-means iterations...")

    # ``history`` is a debug aid — an (N_c, max_steps) record of each
    # centroid's z-coordinate trajectory. The previous implementation updated
    # it every step via ``history[:, step] = np.asarray(new_centroids[:, 2])``,
    # which forced a device→host copy per Lloyd iteration. No caller uses the
    # return value (main() destructures as ``_, centroids, _, _``), so we've
    # returned an empty placeholder here; flip ``_TRACK_HISTORY = True`` to
    # turn it back on when debugging.
    _TRACK_HISTORY = False
    history = (np.zeros((N_c, max_steps), dtype=np.float32)
               if _TRACK_HISTORY else np.zeros((N_c, 0), dtype=np.float32))
    steps_taken = max_steps

    if mesh is not None:
        # Move inputs to the distributed layout now that k-means++ init is done.
        pos_sharding = NamedSharding(mesh, PartitionSpec('x', None))
        rho_sharding = NamedSharding(mesh, PartitionSpec('x'))
        rep_sharding = NamedSharding(mesh, PartitionSpec())
        positions = jax.device_put(positions, pos_sharding)
        rho_flat = jax.device_put(rho_flat, rho_sharding)
        metric_tensor = jax.device_put(metric_tensor, rep_sharding)
        centroids = jax.device_put(centroids, rep_sharding)

    for step in range(max_steps):
        if mesh is not None:
            new_centroids, movement_sq, labels = sharded_step(
                positions, centroids, rho_flat, metric_tensor
            )
        else:
            new_centroids, movement_sq, labels = kmeans_update_step(
                positions, centroids, rho_flat, metric_tensor, N_c, c_block
            )

        if _TRACK_HISTORY:
            history[:, step] = np.asarray(new_centroids[:, 2])
        max_movement_sq = float(jnp.max(movement_sq))

        if step % 20 == 0:
            print(f"  Step {step}: max movement = {np.sqrt(max_movement_sq):.6f} (avec units)")

        if max_movement_sq < tolerance_sq:
            print(f"Converged in {step + 1} steps "
                  f"(max movement = {np.sqrt(max_movement_sq):.6f} (avec units))")
            steps_taken = step + 1
            centroids = new_centroids
            break

        centroids = new_centroids
    else:
        print(f"Reached max steps ({max_steps}) without full convergence")

    # Final label assignment (uses the same C-chunked scan, no (P,C,3) tensor).
    labels = assign_labels_chunked(
        positions, centroids, metric_tensor, N_c, c_block=c_block
    )

    return labels, centroids, history, steps_taken


def snap_centroids_to_grid(
    centroids_frac: np.ndarray,
    fft_grid: np.ndarray,
    deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Snap fractional centroids to the nearest FFT grid points and optionally deduplicate.
    
    When N_c is large relative to the FFT grid, multiple k-means centroids may map 
    to the same grid point. This function:
    1. Converts fractional coords to integer grid indices
    2. Handles periodic wrapping
    3. Removes duplicate grid points (if deduplicate=True)
    
    Args:
        centroids_frac: (N_c, 3) fractional coordinates in [0, 1)
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
        centroids_frac: (N_c, 3) fractional coordinates
        fft_grid: (3,) FFT grid dimensions
        rho: Optional (Nx, Ny, Nz) charge density for weighted redistribution
        metric_tensor: Optional (3,3) for PBC distance calculation
        
    Returns:
        centroids_frac_unique: (N_c, 3) fractional coordinates with no duplicates
    """
    fft_grid = np.asarray(fft_grid)
    centroids_frac = np.asarray(centroids_frac)
    N_c = centroids_frac.shape[0]
    
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

# WFN.h5 convention: ``wfn.alat`` is in Bohr, ``wfn.avec`` rows are unitless
# (components of the primitive vectors in units of alat). Multiplying by
# ``alat * BOHR_TO_ANG`` converts avec rows to Å.
BOHR_TO_ANG = 0.529177210544


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weighted k-means for ISDF sampling points")
    parser.add_argument("N_c", type=int, nargs="?", default=400,
                        help="Number of clusters/sampling points (default: 400)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--plot-zoom", type=float, default=1.0,
                        help="Zoom factor for density in plot (default: 1.0, higher = finer)")
    parser.add_argument("--no-downsample", action="store_true", help="Use full FFT grid (no zoom)")
    parser.add_argument("--shard", action="store_true",
                        help="Use multi-device P-sharded Lloyd step (mesh 'x' "
                             "over all visible jax.devices()).")
    parser.add_argument("--force-shard", action="store_true",
                        help="With --shard, override the per-shard-P auto-gate "
                             "and run the shard_map step even when P/n_dev is "
                             "below the latency-amortization threshold. Useful "
                             "for benchmarking or when N_c is large enough "
                             "that Lloyd compute beats allreduce latency.")
    args = parser.parse_args()
    
    N_c = args.N_c
    print(f"Using N_c = {N_c} clusters")
    
    wfn = WFNReader("WFN.h5")
    sym = symmetry_maps.SymMaps(wfn)

    charge_density = calculate_charge_density(wfn, sym)

    # BGW WFN.h5 stores ``avec`` in units of alat (so rows are typically
    # unit-length) and ``alat`` in Bohr. Convert once to Å and use those
    # Å-basis lattice vectors throughout: this makes ``lattice_lengths``,
    # ``target_spacing``, the kmeans tolerance, and the printed movement all
    # in the same consistent physical unit.
    avec_ang = np.asarray(wfn.avec) * float(wfn.alat) * BOHR_TO_ANG  # (3, 3) in Å

    # Resize grid to target spacing of ~0.13 Å (gives ~10 points per 1.3 Å core region)
    # For each direction i:
    #   current_spacing = |a_i| / N_i
    #   zoom_factor = current_spacing / target_spacing
    # scipy.ndimage.zoom: new_N = old_N * zoom_factor
    target_spacing = 0.13 * 0.52  # Angstroms
    lattice_lengths = np.linalg.norm(avec_ang, axis=1)  # |a1|, |a2|, |a3| in Å
    current_spacing = lattice_lengths / np.array(charge_density.shape)
    zoom_factors = current_spacing / target_spacing
    
    # Don't upsample tiny grids, and cap zoom to avoid excessive memory
    if args.no_downsample or any(wfn.fft_grid < 20):
        zoom_factors = np.ones(3)
    zoom_factors = np.clip(zoom_factors, 0.1, 2.0)  # Reasonable bounds

    # Guard: if N_c is a significant fraction of the coarsened grid,
    # the k-means has too few points to cluster meaningfully and centroids
    # spread into vacuum. Skip downsampling in this case so the density
    # contrast is preserved (e.g. molecule in a large box).
    coarsened_size = int(np.prod(np.round(np.array(charge_density.shape) * zoom_factors)))
    if N_c > coarsened_size // 10:
        print(f"N_c={N_c} is >{coarsened_size//10} (10% of coarsened grid {coarsened_size})")
        print(f"Skipping downsampling to preserve density contrast")
        zoom_factors = np.ones(3)

    print(f"Charge density shape: {charge_density.shape}")
    print(f"Lattice lengths: {lattice_lengths} Å")
    print(f"Current spacing: {current_spacing} Å")
    print(f"Target spacing: {target_spacing} Å")
    print(f"Zoom factors: {zoom_factors}")
    rho_np = interpolate_density(charge_density, zoom_factors)
    rho_np_cpu = np.asarray(rho_np)

    rho_jax = jnp.asarray(rho_np_cpu, dtype=jnp.float32)
    avec_jax = jnp.asarray(avec_ang, dtype=jnp.float32)

    mesh = None
    if args.shard:
        devices = jax.devices()
        if len(devices) < 2:
            print(f"--shard requested but only {len(devices)} JAX device(s) "
                  "visible; running single-device.")
        else:
            mesh = Mesh(np.asarray(devices), ('x',))
            print(f"Sharded Lloyd step: mesh axis 'x' of size {len(devices)}")

    _, centroids_jax, _, _ = weighted_kmeans_jax(
        avec_jax, rho_jax, N_c=N_c, seed=args.seed, mesh=mesh,
        force_shard=args.force_shard,
    )

    centroids_frac = np.array(centroids_jax)
    
    # Snap centroids to the actual FFT grid and handle duplicates
    print(f"\nSnapping {N_c} centroids to FFT grid {wfn.fft_grid}...")
    centroid_indices, centroids_snapped, n_dups = snap_centroids_to_grid(
        centroids_frac, wfn.fft_grid, deduplicate=True
    )
    
    n_unique = centroid_indices.shape[0]
    if n_dups > 0:
        print(f"⚠ {n_dups} duplicates removed. Final count: {n_unique} unique centroids.")
        print(f"  Consider reducing N_c or using a finer FFT grid.")
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
