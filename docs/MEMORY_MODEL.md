# Memory Model and Chunk Size Optimization

The zeta-fitting pipeline allocates tensors in clearly defined stages.  Each
stage has a closed-form expression for its per-device footprint; the chunk
solver simply enforces those expressions without heuristic percentages.  This
document lists every stage, the associated arrays, and the formulas that drive
the automatic sizing in `compute_optimal_chunks`.

For automatic memory detection, `get_device_memory_gb` now uses
`budget = 0.9 * bytes_available`, with:
- `bytes_available = bytes_limit - bytes_in_use` from `jax.memory_stats()` when available
- fallback: `bytes_available = nvidia-smi memory.free`

That detector-side 10% guard band is the only default safety margin; the chunk
solver default is `target_utilization = 0.97` (configurable via
`ISDF_CHUNK_TARGET_UTILIZATION`).

The driver no longer applies an implicit ZCT soft cap.  If needed, set one
explicitly with `ISDF_ZCT_STAGE_CAP_GB` (absolute) or `ISDF_ZCT_STAGE_CAP_FRAC`
(fraction of total GPU memory).

All sizes below use complex128 storage (`bytes_per_complex = 16`).  Mesh axes
follow the code: `'x'` shards the μ/centroid axis, `'y'` shards r-chunks, and
the processor count is `P = p_x * p_y`.

Typical ranges (from production datasets):

| Symbol | Meaning                         | Typical Range |
|--------|---------------------------------|---------------|
| n_k    | total k-points                  | 1 – 2,000     |
| n_b    | union of bands loaded           | 20 – 5,000    |
| n_b^L  | left band count (b0→b3)         | 20 – 2,000    |
| n_b^R  | right band count (b1→b4)        | 20 – 5,000    |
| n_rmu  | ISDF interpolation points       | 200 – 50,000  |
| n_r    | real-space grid (nx*ny*nz)      | 20k – 2M      |
| n_q    | q-points (nkx*nky*nkz)          | 1 – 200       |

## Stage Summary

| Stage | Arrays (per device) | Bytes |
|-------|---------------------|-------|
| **Centroid load** | `psi_rmu_Y (n_k, n_b, n_s, n_rmu/p_y)` and `psi_rmuT_X (n_k, n_rmu/p_x, n_b, n_s)` | `M_full = 16 n_k n_s n_rmu n_b (1/p_x + 1/p_y)` |
| **Centroid copies** | Left+right slices (4 arrays) | `M_cent = 16 n_k n_s n_rmu (n_b^L+n_b^R)(1/p_x+1/p_y)` |
| **FFT workspace** | `psi_G` + `psi_r` + `phase` | `2*16 n_k (B_b/P) n_s n_r + 16 n_k n_r` |
| **C_q build** | `P_l`, `P_r`, `C_q` | `M_cct = M_cent + 2*16 n_k (n_rmu/p_x)(n_rmu/p_y) + 16 n_q (n_rmu/p_x)(n_rmu/p_y)` |
| **Pair density (chunk)** | `psi_xchunk`, `P_l`, `P_r` | `M_pair = base + 16 n_k n_b n_s (B_r/p_y) + 2*16 n_k (n_rmu/p_x)(B_r/p_y)` |
| **ZCT** | `P_l`, `P_r`, FFT temps, `Z_q` | `M_zct = base + 4*16 n_k n_rmu B_r + 16 n_q n_rmu B_r` |
| **Solve** | `Z_col`, triangular-solve temps, `L_q` temp | `M_solve = base + 4*16 n_q n_rmu (B_r/P) + M_L_q` |
| **V_q compute** | μ/ν chunks in r- and G-space | `M_vq ≈ 3 * 16 * μ_chunk * n_r` |

`base` in the chunk stages is `M_cent + M_L_q + cache`, i.e.
persistent centroids, the Cholesky factors, and (optionally) the cached
G-space wavefunctions.

## Band Chunk (`B_b`)

During centroid construction the solver must hold the two union arrays plus the
FFT workspace.  The constraint is

```
M_full + phase + 2 * 16 * n_k * (B_b / P) * n_s * n_r <= M_budget
```

so

```
B_b <= ((M_budget - M_full) - phase) * P / (2 * 16 * n_k * n_s * n_r)
```

`B_b` is clamped between 1 and `n_b`.  If the numerator becomes negative the
system physically cannot fit; the solver raises a descriptive error.

## R-Chunk (`B_r = x_chunk_r`)

`B_r` is the number of contiguous r-points processed per chunk
(`x_chunk * ny * nz`).  Three independent constraints must hold:

1. **Pair density build**

```
base + 16 * (B_r / p_y) * [n_k n_b n_s + 2 n_k (n_rmu / p_x)] <= M_budget
```

2. **ZCT pipeline**

```
base + 16 * B_r * [ (4 n_k + n_q) * n_rmu ] <= M_budget
```

3. **Solve (with q_chunk = 1)**

```
base + 4 * 16 * n_q * n_rmu * (B_r / P) + M_L_q <= M_budget
```

`base = M_cent + M_L_q + cache`.  The analytic upper bounds from these
inequalities provide an initial guess which is then rounded to an integer
number of x-slices, forced to be divisible by `p_y`, and iteratively reduced
until the evaluated stage peaks fall below the budget.  This preserves the
“fill the GPU” behavior without ad-hoc percentages.

## Q-Chunk (`B_q`)

With `B_r` fixed, the triangular solve retains `Z_col` and `zeta`
(`2 * 16 * n_q * n_rmu * (B_r/P)` bytes).  The remaining headroom is available
for replicated Cholesky panels:

```
B_q <= (M_budget - (base + 2 * M_Z_col)) / (16 * n_rmu^2)
```

`B_q` is clamped to `[1, n_q]`.  The solver already ensured that the numerator
is ≥ one `L` matrix when `B_r` was validated, so `B_q=1` is always feasible.

## μ-Chunk for `V_q`

When building `V_q` from the stored zeta HDF5 the code holds

1. `ζ_μ(r)` for the current μ-block,
2. its FFT/weighted counterpart `ζ̃_μ(G)`, and
3. a second ν-block when contracting off-diagonal tiles.

This leads to the simple bound used by the driver:

```
μ_chunk <= available_bytes / (3 * 16 * n_r)
```

The “available bytes” are `effective_budget - M_cent`, because the centroids
remain resident for the rest of the COHSEX calculation.  The CLI now reports
this chunk alongside the others so users can correlate V_q throughput with
memory settings.

## Automatic Sizing Algorithm

1. **Gather inputs**: `{n_k, n_b_full, n_b^L, n_b^R, n_rmu, n_q, fft_grid,
   memory_per_device_gb, mesh}`.

2. **Compute persistent costs**:
   - `M_full`, `M_cent`, `M_L_q`.
   - Validate `M_full + M_cent <= M_budget`.

3. **Band chunk**: Solve inequality above for `B_b`.

4. **C_q stage**: Ensure `M_cent + 2*M_pair(mumu) + M_C_q <= M_budget`.

5. **G-space cache**: Try enabling the cache, evaluate the chunk constraints
   with the cache bytes included, and fall back to “no cache” only if the
   memory system cannot fit even one x-slice otherwise.

6. **R-chunk**:
   - Derive analytic limits from the three inequalities.
   - Convert to integer x-slices, force divisibility by `p_y`, and iteratively
     reduce until the evaluated stage peaks fall below the budget.

7. **Q-chunk**: With `B_r` fixed, compute the available headroom for replicated
   `L` matrices and choose the largest integer chunk that fits.

8. **μ-chunk (V_q driver)**: Use the formula above with the persistent centroid
   cost to size the V_q μ-chunk.  The runtime also clamps this against a fresh
   `get_device_memory_gb` query and takes the minimum budget.

9. **Instrumentation**: The solver returns a `memory_estimate` dictionary with
   the following keys so the CLI can display a detailed breakdown:

   - `centroids_full_gb`, `centroids_gb`, `cached_gspace_gb`
   - `fft_workspace_gb`, `peak_fft_gb`
   - `stage_cct_gb`
   - `psi_xchunk_gb`, `pair_density_xchunk_gb`
   - `Z_q_gb`, `Z_col_gb`, `zeta_gb`, `L_rep_per_q_gb`
   - `stage_pair_gb`, `stage_zct_gb`, `stage_solve_gb`
   - `peak_estimate_gb`, `effective_budget_gb`, `utilization_pct`
   - `centroids_bytes`, `effective_budget_bytes`, `available_vcoul_gb`

These numbers map directly to the table at the top of this document, making it
easy to reason about which stage triggered the peak and how close we are to the
budget.

## Working Backwards from a Failure

When the solver raises an error it always references the offending stage and
provides the required gigabytes.  Typical remedies are:

1. **Centroid copy failure**  
   Reduce `n_b_left`/`n_b_right` (e.g., shrink the sigma window) or increase
   `memory_per_device_gb`.

2. **C_q build failure**  
   Reduce `n_rmu` or `n_q`; these are the only dials that affect the μ×μ stage.

3. **Chunk stage failure**  
   This means even `x_chunk = 1` and `q_chunk = 1` would not fit.  Lower
   `n_rmu`, use more devices, or increase the budget.

4. **V_q μ-chunk failure**  
   Increase the μ chunks or disable caching so more memory is available to the
   V_q builder.

## XProf Workflow

Recommended stack (June 2025 guidance): XProf + TensorBoard memory viewer.

1. Capture:
`uv run python tools/profile_gw_xprof.py -i <input.in> --workdir <run_dir> --logdir ./profiles/xprof --name <tag>`
2. Open UI:
`uv run xprof ./profiles/xprof`
3. Inspect:
Memory Viewer tab for `jit__compute_ZCT_LR(...)` and `jit__solve_all_q(...)`.

Recent production traces show `jit__compute_ZCT_LR` peak dominated by five equal
live buffers (`P_l`, `P_r`, output transpose, and two FFT temporaries), which is
why the ZCT constraint now uses a `4*n_k + n_q` buffer coefficient.
