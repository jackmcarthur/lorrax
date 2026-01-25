# Memory Model and Chunk Size Optimization

This document describes the memory footprint of the zeta fitting pipeline and derives optimal chunk sizes for memory-constrained systems.

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
| `psi_Gtot_local` | $(N_k, N_b/P, N_s, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot (N_b/P) \cdot N_s \cdot N_r$ | persistent until FFT |
| `psi_Gspace_all` | $(N_k, N_b/P, N_s, \max N_G)$ | $16 \cdot N_k \cdot (N_b/P) \cdot N_s \cdot N_G$ | temporary |
| `gvecs_all` | $(N_k, \max N_G, 3)$ | $4 \cdot N_k \cdot N_G \cdot 3$ (int32) | temporary |

**Peak memory (Stage 1):**
$$M_1 = 16 \cdot N_k \cdot \frac{N_b}{P} \cdot N_s \cdot (N_r + N_G) + 12 \cdot N_k \cdot N_G$$

**Example:** $N_k=100$, $N_b=1000$, $P=16$, $N_s=2$, $N_r=500k$, $N_G=125k$:
$$M_1 \approx 16 \cdot 100 \cdot 62.5 \cdot 2 \cdot 625k = 125 \text{ GB per process}$$

This is **too large** — hence band chunking is required.

---

### Stage 1b: Band-Chunked G-space Loading

With band chunk size $B_b$:

| Array | Shape (local) | Size (bytes) |
|-------|---------------|--------------|
| `psi_Gtot_local` | $(N_k, B_b/P, N_s, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_r$ |
| `psi_Gspace_all` | $(N_k, B_b/P, N_s, \max N_G)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_G$ |

**Peak memory:**
$$M_{1b} = 16 \cdot N_k \cdot \frac{B_b}{P} \cdot N_s \cdot (N_r + N_G)$$

**Example:** $B_b=64$, same parameters:
$$M_{1b} \approx 16 \cdot 100 \cdot 4 \cdot 2 \cdot 625k = 8 \text{ GB per process}$$

---

### Stage 2: FFT and Phase Correction (`get_sharded_wfns_zchunk_slice`)

During the FFT step:

| Array | Shape (per-device shard) | Size (bytes) |
|-------|--------------------------|--------------|
| `psi_G` (input) | $(N_k, B_b/P, N_s, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot (B_b/P) \cdot N_s \cdot N_r$ |
| `psi_r` (FFT output) | $(N_k, B_b/P, N_s, N_x, N_y, N_z)$ | same as above |
| `phase_spatial` | $(N_k, N_x, N_y, N_z)$ | $16 \cdot N_k \cdot N_r$ |

**Peak memory (Stage 2):**
$$M_2 = 2 \cdot 16 \cdot N_k \cdot \frac{B_b}{P} \cdot N_s \cdot N_r + 16 \cdot N_k \cdot N_r$$

The factor of 2 accounts for input and output of FFT being live simultaneously.

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

| Array | Shape (per-device) | Size (bytes) |
|-------|-------------------|--------------|
| `psi_rmu_Y` | $(N_k, N_b, N_s, N_\mu/P_y)$ | $16 \cdot N_k \cdot N_b \cdot N_s \cdot (N_\mu/P_y)$ |
| `psi_rmuT_X` | $(N_k, N_\mu/P_x, N_b, N_s)$ | $16 \cdot N_k \cdot (N_\mu/P_x) \cdot N_b \cdot N_s$ |

**Total centroid memory:**
$$M_{\mu} = 2 \cdot 16 \cdot N_k \cdot N_b \cdot N_s \cdot \frac{N_\mu}{\sqrt{P}}$$

(assuming $P_x \approx P_y \approx \sqrt{P}$)

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

## Memory Bottleneck Summary

At any point, the **peak memory per device** is approximately:

$$M_{\text{peak}} = M_{\mu} + \max(M_{\text{FFT}}, M_P, M_{\text{CCT}}, M_{\text{solve}})$$

Where:
- $M_\mu$ = centroid wavefunctions (persistent)
- $M_{\text{FFT}}$ = during band-chunked FFT
- $M_P$ = pair density `P_k(μ, z-chunk)`
- $M_{\text{CCT}}$ = during CCT/ZCT FFT pipeline
- $M_{\text{solve}}$ = during triangular solve

---

## Optimal Chunk Size Derivation

Given a **memory budget** $M_{\text{budget}}$ per device (e.g., 16 GB for GPU, 64 GB for CPU):

### 1. Band Chunk Size ($B_b$)

Constraint: FFT memory fits in budget
$$16 \cdot N_k \cdot \frac{B_b}{P} \cdot N_s \cdot N_r \cdot 2 \leq M_{\text{budget}} - M_\mu$$

Solving:
$$B_b \leq \frac{(M_{\text{budget}} - M_\mu) \cdot P}{32 \cdot N_k \cdot N_s \cdot N_r}$$

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

### Pseudocode:

```python
def compute_optimal_chunks(
    N_k, N_b, N_s, N_mu, N_r, N_q, P,
    memory_budget_gb: float,
    target_utilization: float = 0.85,
):
    M_budget = memory_budget_gb * 1e9 * target_utilization
    bytes_per_complex = 16
    
    # Centroid memory (must fit)
    M_mu = 2 * bytes_per_complex * N_k * N_b * N_s * N_mu / sqrt(P)
    M_available = M_budget - M_mu
    
    if M_available < 0:
        raise MemoryError(f"Centroids alone require {M_mu/1e9:.1f} GB")
    
    # Band chunk size (for FFT)
    # psi_r: (N_k, B_b/P, N_s, N_r) × 2 (input + output)
    B_b_max = M_available * P / (2 * bytes_per_complex * N_k * N_s * N_r)
    B_b = min(N_b, max(16, int(B_b_max)))
    
    # Z-chunk size (for pair density)
    # P_k: (N_k, N_s², N_mu/Px, B_z/Py)
    B_z_max = M_available * P / (bytes_per_complex * N_k * N_s**2 * N_mu)
    B_z = min(N_r, max(N_mu, int(B_z_max)))
    
    # Q-chunk size (for solve)
    # L_rep: (B_q, N_mu, N_mu) replicated
    M_Z = bytes_per_complex * N_q * N_mu * B_z / P
    B_q_max = (M_available - M_Z) / (bytes_per_complex * N_mu**2)
    B_q = min(N_q, max(1, int(B_q_max)))
    
    return B_b, B_z, B_q
```

---

## Example: Large System

**Parameters:**
- $N_k = 100$, $N_q = 100$
- $N_b = 1000$, $N_\mu = 10000$
- $N_r = 500000$ ($\approx 80 \times 80 \times 80$)
- $N_s = 2$
- $P = 16$ devices
- Memory budget: 32 GB per device

**Calculations:**

1. **Centroid memory:**
   $$M_\mu = 2 \times 16 \times 100 \times 1000 \times 2 \times 10000 / 4 = 16 \text{ GB}$$

2. **Available:** $32 - 16 = 16$ GB

3. **Band chunk size:**
   $$B_b \leq \frac{16 \times 10^9 \times 16}{2 \times 16 \times 100 \times 2 \times 500000} \approx 80$$

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

### Available Parameters

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| `band_chunk_size` | `--band-chunk` | Bands per FFT batch |
| `z_chunk_size` | `--z-chunk` | Z-slices per real-space chunk |
| `q_chunk_size` | `--q-chunk` | Q-points per solve batch |
| `max_wfn_chunk_mb` | (input file) | Memory budget for P_k in MB |

### Future Enhancements

1. **Add `--memory-budget-gb` CLI flag** to auto-set all chunk sizes
2. **Report memory usage** at each stage during execution

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

## Memory Monitoring Utilities

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

