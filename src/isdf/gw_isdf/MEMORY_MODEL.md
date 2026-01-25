# Memory Model and Chunk Size Optimization

This document describes the memory footprint of the zeta fitting pipeline and derives optimal chunk sizes for memory-constrained systems.

## Quick Start: Using `memory_per_device_gb`

The simplest way to configure memory usage is with the `memory_per_device_gb` parameter:

```ini
[cohsex]
memory_per_device_gb = 7.0   # Use 7 GB per device (for 8 GB GPUs)
```

This will automatically compute optimal `band_chunk`, `z_chunk`, and `q_chunk` sizes.

**Auto-detection (default):** If `memory_per_device_gb = 0` (or omitted):
- **GPU backend:** Queries `nvidia-smi` for total GPU memory, uses 80%
- **CPU backend:** Queries system RAM via `psutil` or `/proc/meminfo`, divides by device count, uses 80%

**CLI override:**
```bash
uv run python -m isdf.gw_isdf.test_chunked_wfn_loading -i input.in --test-zeta-fit --memory 7.0
```

## Notation

| Symbol | Description | Typical Range |
|--------|-------------|---------------|
| $N_k$ | k-points | 1 – 2,000 |
| $N_q$ | q-points | 1 – 200 |
| $N_b$ | bands | 20 – 5,000 |
| $N_\mu$ | ISDF centroids | 200 – 50,000 |
| $N_r = N_x N_y N_z$ | real-space grid | 20,000 – 2M |
| $N_G$ | G-vectors (max per k) | ~$N_r/4$ |
| $N_s$ | spinor components | 2 or 4 |
| $P = P_x \times P_y$ | total devices | 1 – 256 |
| $B_b$ | band chunk size | 16 – 128 |
| $B_z$ | z-chunk (real-space slice) | variable |
| $B_q$ | q-chunk for solve | 1 – $N_q$ |

All sizes in **bytes** using complex128 (16 bytes per element).

---

## Stage-by-Stage Memory Analysis

### Stage 1: G-space Loading (`read_Gvecs_to_devices`)

**Per-process arrays:**

| Array | Shape (local) | Size (bytes) | Lifetime |
|-------|---------------|--------------|----------|
| `psi_Gtot_local` | $(N_k, B_b/P, N_s, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_r$ | persistent until FFT |
| `psi_Gspace_all` | $(N_k, B_b/P, N_s, \max N_G)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_G$ | temporary |
| `gvecs_all` | $(N_k, \max N_G, 3)$ | $4 \cdot N_k \cdot N_G \cdot 3$ (int32) | temporary |

**Note:** Both `fit_zeta_chunked_to_h5` and `get_psi_zchunk` use band chunking via
`load_centroids_band_chunked` and `get_sharded_wfns_centroids`. The legacy function
`read_Gvecs_and_get_sharded_wfns` loads all bands at once and should be avoided
for large systems.

**Peak memory (Stage 1, band-chunked):**
$$M_1 = 16 \cdot N_k \cdot \frac{B_b}{P} \cdot N_s \cdot (N_r + N_G) + 12 \cdot N_k \cdot N_G$$

**Example:** $N_k=100$, $B_b=64$, $P=16$, $N_s=2$, $N_r=500k$, $N_G=125k$:
$$M_1 \approx 16 \cdot 100 \cdot 4 \cdot 2 \cdot 625k = 8 \text{ GB per process}$$

---

### Stage 2: FFT, Phase Correction, and Centroid Gather

During FFT and centroid extraction, there are **two memory phases**:

**Phase 2a: FFT (band-sharded)**

| Array | Shape (per-device shard) | Size (bytes) |
|-------|--------------------------|--------------|
| `psi_G` (input) | $(N_k, B_b/P, N_s, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_r$ |
| `psi_r` (FFT output) | $(N_k, B_b/P, N_s, N_x, N_y, N_z)$ | same as above |
| `phase_spatial` | $(N_k, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot N_r$ |

$$M_{2a} = 2 \cdot 16 \cdot N_k \cdot \frac{B_b}{P} \cdot N_s \cdot N_r + 16 \cdot N_k \cdot N_r$$

**Phase 2b: Centroid gather (optimized - no replication spike)**

The indexed gather for centroids is performed **while bands are still sharded**.
JAX handles the gather efficiently without replicating the full array.

| Array | Shape (per-device) | Size (bytes) |
|-------|-------------------|--------------|
| `psi_rtot` (band-sharded) | $(N_k, B_b/P, N_s, N_r)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_r$ |
| `psi_rmu` (after gather) | $(N_k, B_b/P, N_s, N_\mu)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_\mu$ |

The all-gather of bands happens AFTER the centroid gather, when the array is
much smaller (N_μ instead of N_r).

**Note:** Previous versions used explicit replication before gather, requiring
250× more temporary memory. This was removed after discovering JAX handles
the sharded gather efficiently.

---

### Stage 3: Z-chunk Extraction and Resharding

After z-slice extraction and staged resharding:

| Array | Shape (per-device) | Size (bytes) |
|-------|-------------------|--------------|
| `psi_zchunk_Y` | $(N_k, N_b, N_s, B_z/P_y)$ | $16 \cdot N_k \cdot N_b \cdot N_s \cdot (B_z/P_y)$ |

Where $B_z = N_x \cdot N_y \cdot z\_chunk\_size$ is the flattened z-chunk size.

**Resharding communication buffer:**
During staged reshard, intermediate arrays of size $(N_k, N_b, N_s, B_z/P_y)$ are created.

---

### Stage 4: Centroid Wavefunctions (persistent)

These are computed once and kept for all z-chunks:

| Array | Shape (per-device) | Size (bytes) | Sharding |
|-------|-------------------|--------------|----------|
| `psi_rmu_Y` | $(N_k, N_b, N_s, N_\mu/P_y)$ | $16 \cdot N_k \cdot N_b \cdot N_s \cdot (N_\mu/P_y)$ | `P(None, None, None, 'y')` |
| `psi_rmuT_X` | $(N_k, N_\mu/P_x, N_b, N_s)$ | $16 \cdot N_k \cdot (N_\mu/P_x) \cdot N_b \cdot N_s$ | `P(None, 'x', None, None)` |

**Total centroid memory per device:**
$$M_{\mu} = 16 \cdot N_k \cdot N_b \cdot N_s \cdot N_\mu \cdot \left(\frac{1}{P_x} + \frac{1}{P_y}\right)$$

**Special cases:**
- Square mesh ($P_x = P_y = \sqrt{P}$): $M_\mu = 16 \cdot N_k \cdot N_b \cdot N_s \cdot N_\mu \cdot \frac{2}{\sqrt{P}}$
- 1D mesh ($P_x = 1, P_y = P$): $M_\mu = 16 \cdot N_k \cdot N_b \cdot N_s \cdot N_\mu \cdot (1 + \frac{1}{P})$

**⚠️ For rectangular meshes (especially $P_x=1$ on CPU), the per-device footprint
can be up to 2× larger than the square-mesh formula suggests.**

---

### Stage 5: Pair Density

| Array | Shape (per-device) | Size (bytes) |
|-------|-------------------|--------------|
| `P_k_mumu` | $(N_k, N_s, N_s, N_\mu/P_x, N_\mu/P_y)$ | $16 \cdot N_k \cdot N_s^2 \cdot (N_\mu^2/P)$ |
| `P_k_mu_zchunk` | $(N_k, N_s, N_s, N_\mu/P_x, B_z/P_y)$ | $16 \cdot N_k \cdot N_s^2 \cdot (N_\mu/P_x) \cdot (B_z/P_y)$ |

**This is the array that `max_wfn_chunk_mb` controls:**
$$M_{P} = 16 \cdot N_k \cdot N_s^2 \cdot \frac{N_\mu}{P_x} \cdot \frac{B_z}{P_y}$$

---

### Stage 6: CCT/ZCT FFT Pipeline

| Array | Shape (per-device) | Size (bytes) |
|-------|-------------------|--------------|
| `P_R_mumu` | $(N_s, N_s, N_\mu/P_x, N_\mu/P_y, N_{kx}, N_{ky}, N_{kz})$ | $16 \cdot N_s^2 \cdot (N_\mu^2/P) \cdot N_k$ |
| `C_R` | $(N_\mu/P_x, N_\mu/P_y, N_{kx}, N_{ky}, N_{kz})$ | $16 \cdot (N_\mu^2/P) \cdot N_k$ |
| `C_q` | $(N_q, N_\mu/P_x, N_\mu/P_y)$ | $16 \cdot N_q \cdot (N_\mu^2/P)$ |
| `Z_q` | $(N_q, N_\mu/P_x, B_z/P_y)$ | $16 \cdot N_q \cdot (N_\mu/P_x) \cdot (B_z/P_y)$ |

---

### Stage 7: 2D Blocked Cholesky

| Array | Shape (per-device) | Size (bytes) |
|-------|-------------------|--------------|
| `C_q_tiles` | $(N_q, J/P_x, J/P_y, b, b)$ | $16 \cdot N_q \cdot (J^2/P) \cdot b^2 = 16 \cdot N_q \cdot (N_\mu^2/P)$ |
| `L_q_tiles` | same | same |

Where $J = N_\mu/b$ is the number of tiles and $b$ is the block size.

**Communication buffers during Cholesky:**
- Panel broadcast: $O(N_q \cdot b \cdot N_\mu / \sqrt{P})$ per step
- SYRK panel: $O(N_q \cdot b \cdot N_\mu / \sqrt{P})$

---

### Stage 8: Triangular Solve

**Current (q-by-q all-gather):**

| Array | Shape | Size (bytes) | Notes |
|-------|-------|--------------|-------|
| `L_rep` (replicated) | $(N_\mu, N_\mu)$ | $16 \cdot N_\mu^2$ | **One q at a time** |
| `Z_col` | $(N_q, N_\mu, B_z/P)$ | $16 \cdot N_q \cdot N_\mu \cdot (B_z/P)$ | Column-sharded |
| `zeta_q` | $(N_q, N_\mu, B_z/P)$ | same | Output |

**Peak memory (solve):**
$$M_{\text{solve}} = 16 \cdot N_\mu^2 + 2 \cdot 16 \cdot N_q \cdot N_\mu \cdot \frac{B_z}{P}$$

**With q-chunking ($B_q$ q-points at once):**
$$M_{\text{solve}} = B_q \cdot 16 \cdot N_\mu^2 + 2 \cdot 16 \cdot N_q \cdot N_\mu \cdot \frac{B_z}{P}$$

---

## Communication Buffers

Inter-device communication creates temporary buffers that must be accounted for:

### Staged Reshard (Z-chunk extraction)

When resharding from `P(None, ('x','y'), None, None)` → `P(None, None, None, 'y')`:

| Stage | Operation | Buffer Size |
|-------|-----------|-------------|
| 1 | All-gather bands over X | $(N_k, N_b/P_y, N_s, B_z)$ |
| 2 | All-gather bands over Y + slice | $(N_k, N_b, N_s, B_z/P_y)$ |

**Peak:** Both buffers exist simultaneously during reshard.

$$M_{\text{reshard}} = 16 \cdot N_k \cdot N_b \cdot N_s \cdot B_z \cdot \left(\frac{1}{P_y} + \frac{1}{P_y}\right)$$

### Cholesky Panel Broadcast

During 2D blocked Cholesky, panel rows/columns are broadcast:

$$M_{\text{panel}} = 16 \cdot N_q \cdot b \cdot \frac{N_\mu}{\max(P_x, P_y)}$$

Where $b = N_\mu / J$ is the block size and $J = \text{lcm}(P_x, P_y)$.

### L Matrix Replication (Solve)

During triangular solve, $B_q$ L matrices are replicated:

$$M_{L\_\text{rep}} = B_q \cdot 16 \cdot N_\mu^2$$

This is the main trade-off: larger $B_q$ = more parallelism but more memory.

---

## Memory Bottleneck Summary

At any point, the **peak memory per device** is approximately:

$$M_{\text{peak}} = M_{\mu} + \max(M_{\text{FFT}}, M_P + M_{\text{reshard}}, M_{\text{CCT}}, M_{\text{solve}})$$

Where:
- $M_\mu$ = centroid wavefunctions (persistent)
- $M_{\text{FFT}}$ = FFT workspace (psi_G + psi_r + phase)
- $M_P$ = pair density `P_k(μ, z-chunk)`
- $M_{\text{reshard}}$ = staged reshard buffers
- $M_{\text{CCT}}$ = during CCT/ZCT FFT pipeline
- $M_{\text{solve}}$ = L replication + Z_col + zeta

---

## Optimal Chunk Size Derivation

Given a **memory budget** $M_{\text{budget}}$ per device (e.g., 16 GB for GPU, 64 GB for CPU):

### 1. Band Chunk Size ($B_b$)

**Constraint: FFT workspace (sharded) must fit in budget**

With the optimized centroid gather (no replication), both centroid and z-chunk
extraction have the same constraint - the FFT workspace:

$$2 \cdot 16 \cdot N_k \cdot \frac{B_b}{P} \cdot N_s \cdot N_r \leq M_{\text{budget}} - M_\mu$$

Solving:
$$B_b \leq \frac{(M_{\text{budget}} - M_\mu) \cdot P}{32 \cdot N_k \cdot N_s \cdot N_r}$$

**Note:** This IS divided by $P$ because both FFT input and output are sharded over bands.

### 2. Z-Chunk Size ($B_z$)

Constraint: Pair density fits in budget
$$16 \cdot N_k \cdot N_s^2 \cdot \frac{N_\mu}{P_x} \cdot \frac{B_z}{P_y} \leq M_{\text{budget}} - M_\mu$$

Solving:
$$B_z \leq \frac{(M_{\text{budget}} - M_\mu) \cdot P}{16 \cdot N_k \cdot N_s^2 \cdot N_\mu}$$

### 3. Q-Chunk Size ($B_q$) for Solve

Constraint: Replicated L matrices fit in budget
$$B_q \cdot 16 \cdot N_\mu^2 \leq M_{\text{budget}} - M_\mu - M_{Z}$$

Solving:
$$B_q \leq \frac{M_{\text{budget}} - M_\mu - M_Z}{16 \cdot N_\mu^2}$$

---

## Recommended Memory Budget Allocation

Based on typical workloads, we recommend:

| Component | % of Budget | Purpose |
|-----------|-------------|---------|
| Centroid wfns ($M_\mu$) | 20% | Persistent |
| FFT workspace | 30% | Band-chunked G→r |
| Pair density ($M_P$) | 30% | $P_k(\mu, z\text{-chunk})$ |
| Solve workspace | 15% | $L$ replication, $Z$, $\zeta$ |
| Overhead/buffers | 5% | XLA temporaries |

---

## Automatic Chunk Size Selection

The implementation should:

1. **Query available memory** per device
2. **Compute $M_\mu$** from system parameters
3. **Derive chunk sizes** using formulas above
4. **Validate divisibility** (chunks must divide evenly for sharding)
5. **Report chosen parameters** to user

### Usage Example

```python
from isdf.common.gpu_utils import get_device_memory_gb
from isdf.gw_isdf.cohsex_init import compute_optimal_chunks, print_memory_breakdown

# Auto-detect memory or use explicit budget
memory_gb = get_device_memory_gb(n_devices=4)  # e.g., 7.5 GB

# Compute optimal chunks
chunks = compute_optimal_chunks(
    n_k=100, n_b=500, n_s=2, n_rmu=2000, n_r=500000, n_q=100,
    fft_grid=(80, 80, 80),
    n_devices=4,
    memory_budget_gb=memory_gb,
    p_x=2, p_y=2,
)

# Print detailed breakdown
print_memory_breakdown(chunks, n_b=500, n_r=500000, n_q=100, fft_grid=(80,80,80))

# Use the computed values
band_chunk = chunks['band_chunk']      # e.g., 64
z_chunk = chunks['z_chunk']            # e.g., 8 (z-slices)
z_chunk_r = chunks['z_chunk_r']        # e.g., 51200 (r-points)
q_chunk = chunks['q_chunk']            # e.g., 10
```

### Output Example

```
======================================================================
  MEMORY-OPTIMIZED CHUNK SIZES
======================================================================

  Memory budget: 7.50 GB/device (source: nvidia-smi)
  Device mesh: 2 × 2 = 4 devices

  Parameter                      Value        Total      Per-chunk
  ------------------------------------------------------------
  Band chunk                        64 /   500 bands
  Z-chunk (z-slices)                 8 /    80 slices
  Z-chunk (r-points)             51200 / 500000 points
  Q-chunk                           10 /   100 q-points

  MEMORY ALLOCATION (per device)                     Size (GB)
  ------------------------------------------------------
  [Persistent]
    psi_rmu_Y (centroids, Y-sharded)                    1.600
    psi_rmuT_X (centroids, X-sharded)                   1.600
    ─ Subtotal: centroids                               3.200

  [Stage: FFT + centroid extract]
    psi_G + psi_r + phase                               0.800

  [Stage: Pair density]
    P_k(μ,μ)                                            0.320
    P_k(μ,z-chunk)                                      0.410

  [Stage: CCT/ZCT → solve]
    C_q matrix                                          0.032
    Z_q matrix                                          0.041
    L replicated (for solve)                            0.640
    Z_col + zeta output                                 0.082

  [Communication buffers]
    Cholesky panel broadcast                            0.016
    Staged reshard buffer                               0.205

  ------------------------------------------------------
  PEAK ESTIMATE                                         5.746 GB
  BUDGET                                                7.500 GB
  UTILIZATION                                            76.6 %
======================================================================
```

---

## Example: Large System

**Parameters:**
- $N_k = 100$, $N_q = 100$
- $N_b = 1000$, $N_\mu = 10000$
- $N_r = 500000$ ($\approx 80 \times 80 \times 80$)
- $N_s = 2$
- $P = 16$ devices (4×4 square mesh, so $P_x = P_y = 4$)
- Memory budget: 32 GB per device

**Calculations:**

1. **Centroid memory (using correct sharding formula):**
   $$M_\mu = 16 \times 100 \times 1000 \times 2 \times 10000 \times (1/4 + 1/4) = 16 \text{ GB}$$

2. **Available:** $32 - 16 = 16$ GB

3. **Band chunk size (FFT workspace, divided by P):**
   $$B_b \leq \frac{16 \times 10^9 \times 16}{32 \times 100 \times 2 \times 500000} = 80$$
   
   With the optimized gather (no replication spike), we can use 80 bands per chunk!

4. **Z-chunk size:**
   $$B_z \leq \frac{16 \times 10^9 \times 16}{16 \times 100 \times 4 \times 10000} \approx 40000$$
   
   This allows z_chunk_size ≈ 6 (since $B_z = 80 \times 80 \times 6 = 38400$)

5. **Q-chunk size:**
   $$M_Z = 16 \times 100 \times 10000 \times 40000 / 16 = 40 \text{ GB}$$ (too large!)
   
   Need smaller $B_z$ or accept $B_q = 1$.

---

## Implementation Recommendations

### Current State
- Band chunking: ✅ Implemented (`band_chunk_size`)
- Z-chunking: ✅ Implemented (`z_chunk_size`, `max_wfn_chunk_mb`)
- Q-chunking: ✅ Implemented (`q_chunk_size` parameter)
- Auto chunk sizing: ✅ Implemented (`compute_optimal_chunks()` in cohsex_init.py)
- Memory detection: ✅ Implemented (`get_device_memory_gb()` in gpu_utils.py)
- Memory budget: ✅ Implemented (`memory_per_device_gb` input parameter, `--memory` CLI flag)

### Available Parameters

| Parameter | CLI Flag | Input File Key | Description |
|-----------|----------|----------------|-------------|
| Memory budget | `--memory` | `memory_per_device_gb` | Total memory per device in GB (0=auto) |
| Band chunk | `--band-chunk` | `band_chunk_size` | Bands per FFT batch |
| Z-chunk | `--z-chunk` | `z_chunk_size` | Z-slices per real-space chunk |
| Q-chunk | `--q-chunk` | `q_chunk_size` | Q-points per solve batch |
| P_k budget | — | `max_wfn_chunk_mb` | Legacy: memory for P_k in MB |

### Memory Detection

The system auto-detects available memory:

**GPU (CUDA backend):**
```bash
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

**CPU backend:**
```python
# Try in order:
1. psutil.virtual_memory().total  # Most reliable
2. /proc/meminfo (Linux)          # Fallback
3. 8 GB default                   # Last resort
```

Memory is divided by device count and 80% is used as the budget.

### Q-Chunked Solve

Instead of:
```python
for q in range(N_q):
    L_rep = all_gather(L_q[q])  # 1 matrix at a time
    zeta[q] = solve(L_rep, Z[q])
```

Use:
```python
for q_start in range(0, N_q, B_q):
    q_end = min(q_start + B_q, N_q)
    L_batch = all_gather(L_q[q_start:q_end])  # B_q matrices
    zeta[q_start:q_end] = vmap(solve)(L_batch, Z[q_start:q_end])
```

This trades memory for parallelism, allowing GPU to process multiple q-points simultaneously.

---

## Memory Detection API

```python
from isdf.common.gpu_utils import get_device_memory_gb, get_device_memory_info

# Simple: get memory per device in GB
mem_gb = get_device_memory_gb(n_devices=4)

# Detailed: get backend, source, and device count
info = get_device_memory_info()
# Returns: {
#   'backend': 'gpu',           # or 'cpu'
#   'total_gb': 7.5,            # Memory per device
#   'source': 'nvidia-smi',     # Detection method
#   'n_devices': 4,             # JAX device count
# }
```

## Runtime Memory Monitoring

```python
def print_memory_report(stage: str, arrays: dict[str, jax.Array]):
    """Print memory usage at a pipeline stage."""
    total = 0
    print(f"\n=== Memory Report: {stage} ===")
    for name, arr in arrays.items():
        size_mb = arr.nbytes / 1e6
        local_size_mb = size_mb / jax.device_count()
        total += local_size_mb
        print(f"  {name}: {arr.shape} = {local_size_mb:.1f} MB/device")
    print(f"  TOTAL: {total:.1f} MB/device")
```

---

## References

- JAX memory management: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
- XLA buffer allocation: https://www.tensorflow.org/xla/operation_semantics

