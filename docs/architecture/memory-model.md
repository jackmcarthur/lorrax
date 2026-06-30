# Memory Model and Chunk Size Optimization

The zeta-fitting pipeline allocates tensors in clearly defined stages.  Each
stage has a closed-form expression for its per-device footprint; the chunk
solver enforces those expressions.  This document lists every stage, the
associated arrays, the formulas that drive the sizing, and the planners
that consume them.

LORRAX has **three** chunk choosers that can run together:

1. **Heuristic (legacy)** — `gw/gw_init.py :: compute_optimal_chunks`.
   Closed-form per-stage inversion of the hand-derived α coefficients
   below.  Always runs.  In the G-flat pipeline it now only authoritatively
   sets `q_chunk`, `q_gather`, `k_chunk`; its `band_chunk` / `chunk_r`
   outputs are overwritten by the G-flat planner before they reach
   `fit_zeta`.
2. **G-flat planner (default for the G-flat ζ + V_q pipeline)** —
   `gw/gflat_memory_model.py :: plan_gflat_chunks`.  Models four named
   per-rank HBM peaks (A: centroid load · B: CCT/Cholesky · C:
   `fit_one_rchunk` · D: `accumulate_rchunk_to_gflat`) and returns a
   `GFlatChunkPlan` with `band_chunk`, `r_chunk`, `n_r_chunks`,
   `gflat_chunk_size`, plus a per-peak breakdown and the binding peak
   name.  Runs unconditionally after the heuristic and overwrites
   `band_chunk` / `chunk_r` / introduces `gflat_chunk_size`.  Its
   `.format()` output is printed at the top of every `fit_zeta`.  See
   [§G-Flat Memory Model](#g-flat-memory-model) below.
3. **AOT memory model (opt-in)** — `gw/aot_memory_model/`.  Per-kernel
   NNLS fits of `memory_analysis()` peaks against dimensional primitives,
   with an **analytic** closed-form chooser and a **20/80** heuristic
   chooser that use those fits.  Enabled with
   `use_aot_chunk_chooser: true` in `cohsex.in` (default `false`).  When
   enabled, overrides the heuristic's `chunk_r` and `band_chunk`; keeps
   the heuristic's `q_chunk`, `q_gather`, `k_chunk`.  See
   [§AOT Memory Model](#aot-memory-model) below and the per-kernel
   artifact JSONs under
   [`src/gw/aot_memory_model/artifacts/`](../../src/gw/aot_memory_model/artifacts/).

Conventions throughout this doc:

| Concept | Value |
|---|---|
| Element size | complex128 → 16 B (`_mem(…) = 16 · ∏dims / shard`) |
| Mesh axes | `'x'` = μ/centroid, `'y'` = r-chunk; `P = p_x · p_y` |
| Budget detection | `common.gpu_utils.get_device_memory_gb = 0.9 · bytes_available`; `bytes_available` from `jax.memory_stats().bytes_limit − bytes_in_use`, falls back to `nvidia-smi memory.free` |
| Target utilization | heuristic `0.97` (`cohsex.in :: chunk_target_utilization`); G-flat planner `0.80` (lower margin reflects bc-loop transient unpredictability) |
| Safety cap (opt-in) | `ISDF_ZCT_STAGE_CAP_GB` / `ISDF_ZCT_STAGE_CAP_FRAC` env vars for the ZCT stage only |

Typical ranges (from production datasets):

| Symbol | Meaning                         | Typical Range |
|--------|---------------------------------|---------------|
| n_k    | total k-points                  | 1 – 2,000     |
| n_b    | union of bands loaded           | 20 – 5,000    |
| n_b^L  | left band count (b0→b3)         | 20 – 2,000    |
| n_b^R  | right band count (b1→b4)        | 20 – 5,000    |
| n_rmu  | ISDF interpolation points       | 200 – 50,000  |
| n_r    | real-space grid (nx·ny·nz)      | 20k – 2M      |
| n_q    | q-points (nkx·nky·nkz)          | 1 – 200       |
| n_q_irr| IBZ q-points after orbit closure| 1 – n_q       |
| ntran  | spatial sym ops (≤ 48)          | 1 – 48        |
| n_bc   | band chunks per kernel call     | 1 – 64        |

## Stage Summary

| Stage | Arrays (per device) | Bytes |
|-------|---------------------|-------|
| **Centroid load (Peak A)** | `psi_rmu_Y (n_k, n_b, n_s, n_rmu/p_y)` and `psi_rmuT_X (n_k, n_rmu/p_x, n_b, n_s)`; transient FFT box | `M_full + 16·n_k·B_b·n_s·n_r·fft_factor / p_xy` |
| **Centroid copies** | Left+right slices (4 arrays) | `M_cent = 16 n_k n_s n_rmu (n_b^L+n_b^R)(1/p_x+1/p_y)` |
| **FFT workspace** | `psi_G` + `psi_r` + 2 FFT staging + `phase` | `4·16 n_k (B_b/P) n_s n_r + 16 n_k n_r` |
| **C_q build (Peak B)** | `P_l`, `P_r`, `C_q`, `L_q` | `M_cct = M_cent + 2·16 n_k (n_rmu/p_x)(n_rmu/p_y) + 16 n_q (n_rmu/p_x)(n_rmu/p_y) + M_L_q` |
| **fit_one_rchunk (Peak C, post-Round-8)** | persistent base + scan-aliased FFT box + 2 rank-5 P-pair carries `(n_k, n_s², n_rmu/p_x, B_r/p_y)` + scan-aliased all_gather slab | see [§R-Chunk](#r-chunk-b_r) |
| **Solve** | `Z_col` input/output, triangular-solve temps, `L_q` temp, replicated `L` panels | `M_solve(B_q) = base + 4·16 n_q n_rmu (B_r/P) + M_L_q + B_q·16 n_rmu²` |
| **accumulate_rchunk_to_gflat (Peak D)** | `gflat_acc`, `zeta_chunk`, scan-aliased FFT box | `M_D = gflat_acc + zeta_chunk + 16·cs·n_r·fft_factor` |
| **V_q compute** | μ/ν chunks in r- and G-space + per-q kernel buffers | `M_vq ≈ 3·16·μ_chunk·n_r` + per-q kernel buffer |
| **V_q unfold (IBZ→full)** | replicated `V_full (n_q_full, n_rmu, n_rmu)` + per-q phase tensor | small relative to per-q kernel — see [§IBZ Cascade Memory](#ibz-cascade-memory) |

`base` in the chunk stages is `M_cent + M_L_q + cache`, i.e.
persistent centroids, the Cholesky factors, and (optionally) the cached
G-space wavefunctions.  The G-space cache can be turned off with
`use_phdf5_gspace: true` (cohsex.in), in which case `read_Gvecs_to_devices`
re-reads the G-space coefficients from the WFN via parallel HDF5 at each
r-chunk — zero device residency for `psiG_cache` at the cost of higher
per-r-chunk I/O.  In the **G-flat path** the host-resident `PsiGStore`
replaces the device-side cache entirely: per-bc tiles live on the host
and an `io_callback` pulls each rank's 1/P band-slab inside the scan body
— see [§ψ(G) host store](#psig-host-store).

`M_L_q` is the on-device Cholesky factor (`16 n_q n_rmu² / P` under
`P(None, 'x', 'y')` sharding plus one replicated panel per concurrent
solve).

## Per-process metadata footprint

Tiny per-rank constants live in host memory and replicated device tables.
Most are negligible relative to the chunked tensors above:

| Datum | Shape | Dtype | Bytes (CrI3 80 Ry, ntran=12, n_rmu=1504) |
|---|---|---|---|
| `sym_perm` (TRS-augmented centroid permutation) | `(2·ntran, n_rmu)` | int32 | `2·12·1504·4` ≈ 144 KB |
| `L_table` (TRS-augmented real-space lattice wrap) | `(2·ntran, n_rmu, 3)` | int8 (float64 promote at use) | `2·12·1504·3·1` ≈ 108 KB |
| `g_index` (per-k sphere) | `(n_k, ngkmax)` | int32 | `36·70000·4` ≈ 10 MB |
| `irr_idx_q / sym_idx_q` (q-folding tables) | `(n_q_full,)` | int32 | `2·n_q_full·4` ≈ <1 MB |
| `q_full_to_irr_sym` | `(n_q_full,)` | int32 | `n_q_full·4` ≈ <1 MB |
| `q_irr_frac` (parent q frac coords) | `(n_q_irr, 3)` | float64 | `24·n_q_irr` ≈ <1 KB |

Worst case (`ntran=48`, `n_rmu=50000`): `L_table` ≈ 14 MB.  Even at the
upper bound it's a negligible per-process overhead — well under 1 GB on
any device.

`L_table` is new in 2026-05-14 (`agent/trs-aware-sym-fix`); it carries
the integer real-space lattice vector by which a centroid image exits
the unit cell, used to build the umklapp phase
`exp(2π i q · (L_μ − L_ν))` in `unfold_v_q`.  TRS rows duplicate
spatial rows (r is fixed under TRS).  See
`reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`.

## Band Chunk (`B_b`)

During centroid construction the solver must hold the FFT workspace plus the
persistent centroid arrays.  The FFT peak is the dominant cost.

### FFT peak memory (measured)

The 3-D `ifftn` decomposes into three 1-D FFTs (x→y→z).  At peak four
copies of the shard `(n_k, B_b/P, n_s, n_r)` coexist — the G-space input,
the real-space output, and two staging buffers from the intermediate
passes.  After the FFT finishes the staging buffers free and the phase
multiply briefly reuses their space, so the peak holds at 4× the shard:

```
M_fft_peak = 4 · 16 · n_k · (B_b / P) · n_s · n_r + 16 · n_k · n_r
```

The second term is the `(n_k, n_r)` phase array (replicated, <1% of the
first term).

**Measured validation** (each in a fresh process, 1 GPU):

| System | Shard (GB) | Peak (GB) | Ratio | 4× pred | Error |
|--------|-----------|-----------|-------|---------|-------|
| Si 24³, nk=64, nb=60 | 1.699 | 6.795 | 4.00× | 6.809 | 0.2% |
| 48³, nk=1, nb=80 | 0.283 | 1.132 | 4.00× | 1.134 | 0.2% |
| MoS2 24×24×80, nk=1, nb=160 | 0.236 | 1.074 | 4.55× | 0.944 | 14% |
| Si 24³, nk=216, nb=10 | 0.956 | 4.059 | 4.25× | 3.870 | 5% |

The 4× model is exact for large shards (>0.3 GB).  For small shards a
fixed overhead of ~0.03–0.1 GB adds ~10–15% above the 4× prediction.
This overhead comes from the cuFFT plan cache, the phase array
broadcast, and JIT compilation metadata.  The G-flat planner uses a
`fft_box_factor = 4.0` constant; the AOT model confirms the same
`β = 4` for the relevant primitive.

### Band chunk constraint

```
M_full + 4 · 16 · n_k · (B_b / P) · n_s · n_r + 16 · n_k · n_r ≤ M_budget
```

so

```
B_b ≤ ((M_budget − M_full) − phase) · P / (4 · 16 · n_k · n_s · n_r)
```

`B_b` is clamped between 1 and `n_b`.  If the numerator becomes negative
the system physically cannot fit; the solver raises a descriptive error.

## R-Chunk (`B_r`)

`B_r` is the number of contiguous r-points per chunk (`x_chunk · ny · nz`).
The **post-Round-8 G-flat path** routes the per-rchunk work through a
single scan-inside-shard_map kernel; the binding allocation is the two
rank-5 P-pair accumulators that live across the bc-scan body.

### Round-8 unified-FFT-pipeline model

The pre-Round-6 model (six independent constraints, of which `reshard`
and `pair` were typically binding) overstated the budget by ~3× at CrI3
80 Ry.  The Round-6/7/8 rewrite (commits `5cadd4b → f567aa0 → c796420`
on `agent/zeta-bc-scan-shardmap`) collapsed the bc-loop FFT-box slot
pile-up from 58 concurrent slots down to **1** by pushing the band-chunk
scan **inside** the shard_map body — the structural fix mandated by
`feedback_path_d_scaffolding_pattern`.

Inside the new `z_q_from_psi_sm._local` (mirror coming for
`c_q_from_psi_sm._local` in the planned Round-9 CCT port):

```
peak_C ≈ persistent_base + 2·M_P_carry + M_FFT_box + M_all_gather_slab
       + M_psi_G_iter
```

where each term is per-rank, c128 (CrI3 6×6 80 Ry, 4×4 mesh, n_rmu=376):

| Term | Shape | Bytes |
|---|---|---|
| `M_P_carry` (×2: L, R) | `(n_k, n_s², n_rmu/p_x, n_zchunk/p_y)` | 3.71 GiB each, both live to γ̃ contract |
| `M_FFT_box` (scan-aliased) | `(n_k, bpd_max_local, n_s, n_r)` + 4× cuFFT scratch | ~5 GiB, 1 slot across all bc iters |
| `M_all_gather_slab` (scan-aliased) | `(n_k, P·bpd_max_local, n_s, n_zchunk)` | ~1.36 GiB pre-r-slice, ~340 MB post-r-slice |
| `M_psi_G_iter` (io_callback) | `(n_k, bpd_max, n_s, ngkmax)` per bc | ~80 MB |
| `persistent_base` | centroids (L+R) + `L_q` | varies |

The key invariant: every transient inside the scan body **aliases to a
single slot** across iterations.  This requires `lax.scan(..., unroll=1)`
and the scan to live INSIDE the `shard_map`, NOT outside (the SPMD
partitioner cannot prove aliasing across a global-sharded carry — see
the `solve_zeta` 88 GB OOM in `isdf_fitting.py:1119-1141`).

### Per-bc streaming scan invariants

The Round-6 rewrite (commit `f567aa0`, BLOCKER fix `c796420`)
established four load-bearing invariants in
`z_q_from_psi_sm._local`.  Violations are silent — they produce wrong
numerics, not crashes:

1. **Per-bc io_callback pull.**  Each iter pulls one bc's bands from
   `PsiGStore._host_tiles` via `io_callback(_slice_local_tile_bc, ...)`
   with a static `out_sds` sized at `bpd_max · ns · ngkmax / P`.  Bands
   are flat-sharded across `('x','y')` host-side, so each rank gets
   `bpd_per_bc = bc_size / P` bands per bc.
2. **IFFT-before-gather invariant.**  Each rank IFFTs its 1/P
   band-slab over the **full** r-chunk before the all_gather.
   Gathering bands first would force a full-bands FFT box per rank
   (~80 GB at CrI3 6×6 80 Ry, infeasible).
3. **Gather-then-slice for r-axis coherence.**  The all_gather over
   `('x','y')` on the band axis stacks contributions from different
   y-ranks; only AFTER the gather can each y-rank `dynamic_slice` its
   per-rank `r_loc = n_zchunk / p_y` slab.  Slicing before the gather
   silently produces wrong numerics (Round 6 Bug B, fixed in
   `f567aa0`).
4. **Symmetric front+back pad on `psi_l_X` / `psi_r_X`** (Round-7
   BLOCKER `c796420`).  XLA's `dynamic_slice_in_dim` **silently clamps**
   out-of-bounds starts to `max(0, axis_size - size)`, returning the
   wrong physical bands.  Bispinor-transverse runs, asymmetric L/R
   windows, and short final bcs all trigger this.  The fix is a
   symmetric front+back pad sized so every per-bc slice lands in-bounds;
   the L/R mask zeroes pad rows so the math is unchanged.  Cost: a few
   extra band rows per einsum; <1 % wall.

### Carry-sizing correctness gotcha

The per-rank scan accumulator's r-dimension is `r_loc = n_zchunk / p_y`,
**not** the full r-chunk.  `out_spec = P(None, 'x', 'y')` requires the
per-rank output shape `(n_q, n_rmu/p_x, n_zchunk/p_y)`; the carry that
feeds the post-scan γ̃ tail must already be at that extent.  Pre-flight
invariant: `n_zchunk % p_y == 0` (the G-flat planner rounds `r_chunk`
down to a multiple of `p_xy`).

### Solve stage

With `B_r` fixed, the triangular solve baseline contains:

- `Z_col` input/output + triangular-solve temps: `4 · 16 · n_q · n_rmu · (B_r/P)`
- one local `L_q` temporary: `M_L_q`
- one replicated `L` panel for `q_chunk=1`: `16 · n_rmu²`

Additional `q_chunk` values add one replicated panel per extra q-point.
The bound is:

```
B_q ≤ 1 + (M_budget − (base + 4·M_Z_col + M_L_q + M_Lrep)) / M_Lrep
```

where `M_Lrep = 16 · n_rmu²`.  `B_q` is clamped to `[1, n_q]`.

The solve stage was deliberately NOT folded into the Round-8 unified
primitive — it's a separate jit (`solve_q`) and its memory footprint is
well-modelled by the AOT `solve_q` fit.

## Q-Chunk (`B_q`)

Bottleneck arrays: replicated Cholesky panels (`B_q` copies of `(n_rmu,
n_rmu)`) plus `Z_col`/`zeta` from the x-chunk stage.  The heuristic's
formula above governs `B_q` selection; the G-flat planner does not
modify it.

## μ-Chunk for `V_q`

Building `V_q` from on-disk zeta holds three `(μ_chunk, n_r)` buffers
concurrently — `ζ_μ(r)`, its weighted G-space counterpart `ζ̃_μ(G)`,
and a second ν-block for off-diagonal tiles — plus the persistent
`V_q (n_rmu, n_rmu)` accumulator (on-GPU via `.at[].set()`, no
device→host sync per block since commit `b0e0f41`).  Available bytes
are `effective_budget − M_cent` because the centroids stay resident
through COHSEX:

```
μ_chunk ≤ available_bytes / (3 · 16 · n_r)
```

The CLI reports this chunk alongside the others.

### Per-q kernel allocation (V_q HWM)

HLO inspection of the V_q stage (Stage A pre-commit audit, P2 findings)
shows the in-memory peak inside the V_q driver is dominated by the per-q
kernel's μ/ν dispatch buffers — `_per_q_kernel` reuses ν-block scratch
across the q-loop but holds two `(μ_chunk, μ_chunk)` accumulators plus
the `V_q (n_rmu, n_rmu)` running sum.  Worst case at the largest q is:

```
peak_vq ≈ M_cent + 3·16·μ_chunk·n_r + 2·16·μ_chunk² + 16·n_rmu²
```

The `unfold_v_q(IBZ→full-BZ)` allocation — replicated `V_full` of shape
`(n_q_full, n_rmu, n_rmu)` plus the per-q phase tensor — is **small**
relative to the per-q kernel.  At Si 4×4×4 with `n_q_full=64`,
`n_rmu=432` the V_full byte count is `64·432²·16 ≈ 191 MB` per device
(sharded `P(None, 'x', 'y')` it drops to ~12 MB per rank on a 4×4 mesh);
the phase tensor adds `64·432·16 ≈ 0.4 MB`.  Both fit inside the per-q
kernel's persistent residence with room to spare.

## IBZ Cascade Memory

The IBZ cascade (commit-pinned 2026-05-11) activates when centroid orbit
closure under the spatial sym ops succeeds.  Effects on memory:

| Quantity | Without IBZ cascade | With IBZ cascade | Savings |
|---|---|---|---|
| `ζ_q.h5` (charge) | `c128[n_q_full, n_rmu, ngkmax]` | `c128[n_q_irr, n_rmu, ngkmax]` | × `n_q_full / n_q_irr` |
| `ζ_q.h5` (Si 4×4×4 80 Ry, ntran=8) | `c128[64, 432, 588]` ≈ 280 MB | `c128[8, 432, 588]` ≈ 35 MB | **× 8.0** |
| `ζ_q.h5` (CrI3 6×6 80 Ry charge) | `c128[36, n_rmu, ~50k]` ≈ 35–80 GB | `c128[6, n_rmu, ~50k]` ≈ 6–13 GB | × 6 |
| `V_q` in memory (any q-set) | `c128[n_q_full, n_rmu, n_rmu] / p_xy` | **same** (unfolded eagerly) | 0 |
| `gflat_acc` in memory | `c128[n_q_full, n_rmu, ngkmax] / p_xy` | `c128[n_q_irr, n_rmu, ngkmax] / p_xy` | × `n_q_full / n_q_irr` |
| `unfold_v_q` umklapp phase | — | `c128[n_q_full, n_rmu]` replicated | new (tiny — ≤10 MB) |

Three points worth keeping straight:

- **Disk-side and Peak-D-side only.**  `V_q` is unfolded inside
  `compute_V_q_..._to_h5` (`gw/v_q_g_flat.py`) eagerly to its full
  `(n_q_full, n_rmu, n_rmu)` shape sharded `P(None, 'x', 'y')`, then
  written to `isdf_tensors.h5` and passed to Σ on the full BZ — so the
  in-memory V_q footprint is identical with and without the cascade.
- **Peak C in-memory footprint is unchanged.**  The fit operates per
  `(r-chunk × bc)`; the IBZ reduction is purely a disk-write
  optimization at `accumulate_rchunk_to_gflat`.  Peak C's HWM model is
  the same as the full-BZ path.
- **`unfold_v_q` transient is small.**  At peak it's
  `2 · 16 · n_q_full · n_rmu² / P` (the umklapp phase plus a permuted
  `V_at_irr` copy), which on Si 4×4×4 is ~1 MB and on CrI3 6×6 is
  ~30 MB.

The cascade is gated on **centroid orbit closure** under the spatial
sym ops.  Regenerate centroids without `--no-orbit` and ensure
`compute_centroid_sym_perm(..., extend_trs=True)` raises no closure
error to activate it.  When inactive (centroid orbit not closed,
bispinor transverse path, or `LORRAX_FORCE_FULL_BZ=1`),
`write_ibz_only_charge = False` and `n_q^disk = n_k_tot`; the planner
sees the full-BZ `gflat_acc` and Peak D ramps proportionally.

Trigger paths to be aware of:

- `LORRAX_FORCE_FULL_BZ=1` env var bypasses the cascade entirely
  (debugging only — useful for isolating residuals from `unfold_v_q`).
- `sym is None` or `centroid_indices is None` — the cascade can't
  activate; falls back to full-BZ iteration.
- `compute_centroid_sym_perm` raises `RuntimeError` on orbit-closure
  failure — falls back to full-BZ with a verbose warning.

The full chain is in `gw/v_q_g_flat.py :: _resolve_ibz_q_list`.

## ψ(G) host store

`PsiGStore` (`common/psi_G_store.py`) holds the G-space wavefunction
coefficients in **pinned host memory**, tiled along the bc axis:

- Tile layout: `(n_k, bc, n_s, ngkmax) c128` per host-tile, one tile
  per band-chunk index.
- Per-process residency: `n_b · n_k · n_s · ngkmax · 16 / total_procs`
  bytes (band-flat-sharded across all ranks).  At Si 4×4×4 25 Ry / 2
  procs: 0.03 GB/proc.  At CrI3 6×6 80 Ry / 16 procs: ~1.5 GB/proc
  (typical).
- Access pattern: `io_callback(_slice_local_tile_bc, out_sds=...)`
  pulls one bc's per-rank-local slab (`bpd_max · n_s · ngkmax`) into
  device memory inside each scan iter.  Single one-shot push to host
  at populate time; many small pulls during ζ-fit.

This is the **single source** for ψ(G) memory residency in the
post-Round-6 pipeline — the previous `psi_G_device_full` device-side
property has been deleted, and the `gflat_to_rchunk` standalone helper
along with it.  See `feedback_iocallback_for_large_caches`.

## G-Flat Memory Model

`gw/gflat_memory_model.py :: plan_gflat_chunks` is the production planner
for `band_chunk` / `chunk_r` / `gflat_chunk_size` on the G-flat ζ + V_q
pipeline.  Four named per-rank HBM peaks, keyed by source-code location:

| Peak | Stage | Persistent | Transient (per scan iter, aliased) |
|---|---|---|---|
| **A** | `load_centroid_wfns` (pre-loop, once per channel) | centroid output being filled `(n_k, n_s, n_rmu, n_b/P)` | ψ(G)→r FFT box `4·16·n_k·B_b·n_s·n_r / (p_x · p_y)`, replicated `(n_k, n_r)` phase table |
| **B** | `CCT + Cholesky` (pre-loop) | centroids (L+R copies) | open-spin `P_l + P_r (n_k, n_s², μ, μ)`, `C_q (n_q, μ, μ)`, `L_q (n_q, μ, μ)` |
| **C** | `fit_one_rchunk` (inside r-chunk loop) | centroids + `L_q` (base) | `slots · 16·n_k·n_s²·μ_loc·r_loc` rank-5 P-pair concurrent slots, scan-aliased FFT box, `Z_q` output |
| **D** | `accumulate_rchunk_to_gflat` (right after each `fit_one_rchunk`) | `gflat_acc (n_q^disk, n_rmu/p_xy, ngkmax)` | `zeta_chunk (n_q^disk, n_rmu/p_xy, B_r)`, per-scan-iter FFT box `cs · n_r · 16 · fft_factor` |

### Peak A — Band-chunked centroid load

`ψ(G) → IFFT → sample at r_μ`.  Runs once per channel (charge + 3
transverse on bispinor).  Persistent: only the centroid output being
filled.  Transient: the ψ(r) FFT box.

```
peak_A = centroid_out_filling + phase_table + fft_box · fft_box_factor
       = 16·n_k·n_s·n_rmu·B_b/p + 16·n_k·n_r + 16·n_k·B_b·n_s·n_r/p_xy · 4
```

### Peak B — CCT + Cholesky

Pair density on (μ, ν) full-grid + C_q FFT + L_q factor.  Persistent
during the pre-loop call: centroids (L+R copies).  Transient: `P_l`,
`P_r` at full μ², `C_q`, `L_q` workspace.

```
peak_B = 2·M_cent + 2·M_P_open_spin + M_C_q + M_L_q
```

### Peak C — fit_one_rchunk

The binding peak on most production runs.  The fused jit holds:

- **Persistent**: centroids (L+R), `L_q`, `gflat_acc`.
- **Transient (`pair_density_slots` concurrent rank-5 buffers)**:
  `c128[n_k, n_s², n_rmu_local, r_chunk_local]`.  Default is
  **backend-aware** — 3 on GPU XLA (`P_l_R_conj`, `P_r_R`, plus one
  XLA scratch — verified in the `module_0510` GPU HLO dump and
  `agent_d_hlo_calibration.md`) and 4 on CPU XLA (one extra
  concurrent slot scheduled by CPU XLA's BufferAssignment heuristic;
  verified at Si μ=384 scalar + bispinor charge + bispinor transverse,
  reports `CPU_OVERHEAD_DECOMP_2026-05-20.md` and
  `CPU_PLANNER_LANDED_2026-05-20.md`).  Resolved at function-call
  time via `_default_pair_density_slots()` in `gflat_memory_model.py`.
  XLA's BufferAssignment reuses these slots for the FFT box and `Z_q`
  intermediate when lifetimes don't overlap on both backends.

```
peak_C ≈ 2·M_cent + M_L_q + slots · 16·n_k·n_s²·μ·B_r/p_xy + M_zeta_out
```

The `pair_density_slots` constant is the **XLA-BufferAssignment-determined**
count of concurrent rank-5 buffers.  Read it from
`module_NNNN.jit__kernel.sm_*.memory-usage-report.txt` as the number of
distinct preallocated-temp slots holding a P-pair-shaped value.  Update
the defaults in `gflat_memory_model._peak_C_fit_one_rchunk` if a future
XLA version changes the BufferAssignment.

### Peak D — accumulate_rchunk_to_gflat

Runs after `fit_one_rchunk` returns; its `P_l`/`P_r` are freed by then.
`zeta_chunk` is the only `fit_one_rchunk` output still live.

```
peak_D = gflat_acc + zeta_chunk + accumulate_fft_box · fft_box_factor
       = 16·n_q_disk·μ·ngkmax/p_xy + 16·n_q_disk·μ·B_r/p_xy
         + 16·gflat_chunk_size·n_r · 4
```

`gflat_acc` is the persistent G-flat ζ accumulator (μ-flat sharded
across mesh).  When the IBZ cascade activates, `n_q_disk = n_q_irr`,
shrinking `gflat_acc` by the `n_q_full / n_q_irr` factor.

### Sample planner output

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 16
  r_chunk            = 8366  (2 chunks)
  gflat_chunk_size   = None
  budget             = 28.00 GB/dev
  HWM estimate       = 24.21 GB/dev (86% of budget) [bottleneck: C_fit_one_rchunk]
  peak breakdown (GB/dev):
    C_fit_one_rchunk........   24.21
    D_accumulate............   14.26
    A_centroid..............    0.97
    B_CCT_chol..............    0.96
```

(Si 4×4×4 80 Ry, 4×4 mesh, charge-only, 28 GB/dev budget; from
`runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym_floor_fix_2026-05-15/gw.out`.)

The four totals appear sorted by descending bytes; the `bottleneck`
field names the binding peak.  Top-level `peak_breakdown` reports the
A/B/C/D totals; the full per-term breakdown (centroids, FFT box, `P_l`,
`P_r`, `L_q`, …) is built into `GFlatChunkPlan.peak_breakdown` keyed by
`{A,B,C,D}.{term}` — accessible programmatically for debugging.

### Algorithm

`plan_gflat_chunks` is deterministic, no iterative search:

1. **Compute persistent footprint** (centroids + `L_q` + `gflat_acc`).
   Validate against the budget at every peak.
2. **Pick `band_chunk` first** — primary lever on Peak A and Peak C
   FFT-box.  Maximize as a power-of-2 divisor of `nb_total` subject to
   the FFT box fitting in 50 % of `target_utilization · budget` minus
   persistent.  Override via `cfg.memory.band_chunk_size` (cohsex.in).
3. **Pick `r_chunk`** — maximize subject to Peak C fitting after
   `band_chunk` is fixed.  Lower-bounded by `n_rmu` (per user spec: the
   eventual Σ_μν output occupies `n_rmu² · n_q · 16` bytes, so paying
   less than `n_rmu` work per chunk is wasted iteration overhead).
   Upper-bounded by `n_rtot`, and `n_rtot / B_r ≤ max_chunks = 64`.
   Rounded *down* to a multiple of `p_xy` so the `(μ_X, r_Y)` sharding
   at the solve output divides cleanly.
4. **Pick `gflat_chunk_size`** — set to one-shot (`gflat_chunk_size =
   None`, meaning `N = n_q^disk · n_rmu_local` rows per call) if Peak D
   fits; else `headroom_D / fft_per_row`, binary-search down.
5. **Compute per-peak breakdowns + HWM**.  HWM = max over A, B, C, D;
   bottleneck = arg-max.  Returned in the plan dataclass; logged on
   rank 0 via `gflat_plan.format()`.

All three chunk sizes have explicit `*_override` parameters that skip
the corresponding sizing step.  No retry loop — the analytic inversion
is one-shot.  If `B_r` would be < 1 the planner emits a descriptive
error naming the binding peak.

### Pair-density slots (`slots`)

Peak C's dominant transient is `slots` concurrent rank-5
`c128[n_k, n_s², n_rmu/p_x, B_r/p_y]` tensors.  XLA's BufferAssignment
fuses lifetimes that don't overlap — verified on MoS2 3×3 bispinor /
2×2 mesh, `slot[1]` holds both a P-pair tensor and the band-chunk FFT
box across non-overlapping windows.  Default `slots = 3`:
`P_l_R_conj`, `P_r_R`, plus one XLA scratch slot.

Re-verify after any kernel change:

```
$ XLA_FLAGS="--xla_dump_to=./hlo --xla_dump_hlo_pass_re=memory-usage-report"
$ uv run python -m gw.gw_jax -i cohsex.in --workdir <run>
$ ls ./hlo/module_*.jit__kernel.sm_*.memory-usage-report.txt
```

Search for the highest-numbered slot holding a `c128[n_k, n_s², ..., ...]`
shape; that's the slot count.  Update
`pair_density_slots_charge` / `pair_density_slots_transverse` in the
planner if it changed.  See `reference_hlo_dump_workflow_lorrax.md` for
the shifter-aware launcher.

## Round-8 efficiency findings

Headline numbers from the Round-6/7/8 push on
`agent/zeta-bc-scan-shardmap` (CrI3 6×6 80 Ry, 16 GPUs):

| Metric | Pre-Round-6 (`5cadd4b`) | Post-Round-8 (`c796420`) |
|---|---:|---:|
| FFT-box concurrent slots (HLO) | 58 | **1** |
| Per-rank preallocated-temp peak | 48.63 GiB | 13–15 GiB |
| `psi_Y_full` materialization | 30 GiB transient | absent |
| `psi_G_device_full` tracer leak | yes | resolved |

**Required composition.**  The kernel nests **four** primitives:
`io_callback × lax.scan × shard_map × lax.all_gather`.  The
io_callback pulls each rank's 1/P bands of one bc per scan iter; the
all_gather reassembles the band axis after the IFFT but before the
einsum (so the per-rank IFFT operates on 1/P bands → ~5 GiB FFT box
instead of full bands → ~80 GB FFT box).  The `unroll=1` pin is
load-bearing; defaulting to JAX's heuristic unroll defeats the entire
fix.  Composition smoke-tested in `tests/test_io_callback_nested.py`.

**Round-9 follow-up.**  Port `c_q_from_psi_sm._local` to mirror
`z_q_from_psi_sm._local` (currently still uses the pre-Round-6
pre-materialized `psi_l_Y`/`psi_r_Y` design, ~4 GB persistent residency
that competes with the chol factor and ψ(G) host caches for the same
budget).  The CCT migration is structurally identical to ZCT — see
`reports/zeta_rchunk_memory_model_2026-05-13/round8_unified_fft_pipeline.md`
§§5 for the per-commit plan.

## Automatic Sizing Algorithm

Run order in `gw_init.fit_zeta`:

1. **Gather inputs**: `{n_k, n_b_full, n_b^L, n_b^R, n_rmu, n_q,
   fft_grid, memory_per_device_gb, mesh}`; compute persistent costs
   `M_full`, `M_cent`, `M_L_q` and validate
   `M_full + M_cent ≤ M_budget`.
2. **Heuristic solver** (`compute_optimal_chunks`) sets `band_chunk`,
   `chunk_r`, `q_chunk`, `q_gather`, `k_chunk` from the per-stage
   inequalities at the top of this doc.
3. **G-flat plan** (`plan_gflat_chunks`) runs unconditionally and
   overwrites `chunks['band_chunk']` / `chunks['chunk_r']` /
   `chunks['gflat_chunk_size']` — see
   [§G-Flat Memory Model](#g-flat-memory-model).
4. **μ-chunk (V_q driver)** sized from the formula above, clamped
   against a fresh `get_device_memory_gb` query.
5. **AOT override (optional)**: if `use_aot_chunk_chooser = true`,
   `choose_chunks_heuristic` or `choose_chunks_analytic` rewrites
   `chunks['chunk_r']` / `chunks['band_chunk']` again.
6. **Instrumentation**: planner returns `peak_estimate_gb`,
   `bottleneck`, per-peak breakdown, and budget; AOT chooser adds
   `ChunkChoice.note`.  Both surfaced on rank 0 of `gw.out`.

## Recipe — planning a run for a given budget

To size a fresh system at a target `memory_per_device_gb` (cohsex.in):

1. **Set the budget** in cohsex.in (`memory_per_device_gb`).
   `get_device_memory_gb` returns `0.9 · bytes_available`; choose
   `28.0` for a 40 GB A100, `56.0`–`72.0` for an 80 GB hbm80g A100,
   `6.0` for an 8 GB local GPU.  `chunk_target_utilization: 0.97`
   (heuristic; G-flat planner uses 0.80 internally).
2. **Pick the mesh** `p_x × p_y = total_GPUs`.  Square-ish meshes
   (e.g. 4×4 on 16 GPUs) minimise both Peak A and Peak C since they
   sit on `p_xy`-sharded buffers.  If `n_rmu_padded % p_xy ≠ 0` the
   centroid loader pads up.
3. **Inspect the centroid footprint first.**
   `M_cent = 16·n_k·n_s·n_rmu·(n_b^L+n_b^R)·(1/p_x + 1/p_y)` is
   non-chunkable; if it exceeds the budget the run is physically
   infeasible.  Remedies: shrink the sigma window, shrink `n_rmu`, or
   grow the mesh.
4. **Run and read the `G-flat memory model` block** of `gw.out`.  If
   `bottleneck = C_fit_one_rchunk` and HWM sits at 80–95 % of budget,
   that's the intended operating point.
5. **If HWM > 100 %** the planner errors with the binding peak named.
   Knobs by peak:
   - **A_centroid** — lower `B_b` via `memory.band_chunk_size`.
   - **B_CCT_chol** — drop `n_rmu` or rebuild centroids with a smaller
     orbit.
   - **C_fit_one_rchunk** — grow mesh, shorten sigma window, or
     override `r_chunk` smaller.  If `slots > 3` in a fresh HLO dump,
     update the planner constant.
   - **D_accumulate** — set `memory.gflat_chunk_size` to a small
     integer (e.g. 512).  Rarely binds.
6. **If HWM < 60 % of budget**, enlarge the GW window or raise
   `memory.per_device_gb` artificially — the planner ramps `B_b` and
   `B_r` to fill it.
7. **Compare HWM to runtime peak.**  `γ = runtime_peak / planner_HWM`
   should land in `[0.7, 1.0]`.  `γ > 1.0` means under-estimate — count
   binding-peak slots in the HLO memory-usage-report and update
   `pair_density_slots_*` or `fft_box_factor`.
8. **Escape hatches**, in order:
   - `LORRAX_FORCE_FULL_BZ=1` — disables the IBZ cascade (debugging).
   - `use_phdf5_gspace: true` — per-rchunk parallel HDF5 reads,
     eliminates `psiG_cache` device residency.  Costs ~2–5× I/O wall.
   - Grow the mesh.  All chunked terms shrink as `1/p_xy`; `M_cent`
     shrinks as `1/p_x + 1/p_y`.

### Worked example: Si 4×4×4 80 Ry

From `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym_floor_fix_2026-05-15/`:

- System: `n_k=64, n_s=2, n_rmu=432, n_rtot=13824, ecutwfc=25 Ry,
  ecutrho=100 Ry`.
- Mesh: 4×4 (16 GPUs), 28 GB/dev budget.
- Planner picks: `band_chunk=16, r_chunk=8366 (2 chunks),
  gflat_chunk_size=None`.
- HWM estimate: **24.21 GB/dev (86 % of budget)**.
- Bottleneck: `C_fit_one_rchunk` — the rank-5 P-pair carries dominate.
- Disk: 8 IBZ q's of 64 full-BZ (`ntran=8`, 8× shrink); `ζ_q.h5`
  carries `(n_q_disk=8, n_rtot=13824, n_rmu=432) · 16 B ≈ 0.77 GB`.
- Runtime peak (measured): close to the 24.21 GB estimate, γ ≈ 0.95.

## XProf Workflow

Recommended stack (June 2025): XProf + TensorBoard memory viewer.

1. Capture:
   `uv run python tools/profile_gw_xprof.py -i <input.in> --workdir <run_dir> --logdir ./profiles/xprof --name <tag>`
2. Open UI: `uv run xprof ./profiles/xprof`
3. Inspect Memory Viewer for `jit__z_q_from_psi_sm(...)` (post-Round-6
   name), `jit__compute_C_q(...)`, and `jit__solve_all_q(...)`.

Recent production traces (CrI3 6×6 80 Ry, 16 A100 / 4×4 mesh,
`agent/zeta-bc-scan-shardmap`): `jit__z_q_from_psi_sm` peak ~13–15 GiB
per rank (two rank-5 carry accumulators at 3.71 GiB each + scan-aliased
FFT box ~5 GiB + scan-aliased post-gather slab ~340 MB);
`jit__solve_all_q` four `Z_col`-sized buffers + `L_q` temp;
`jit__unfold_v_q` per-q `V_full` materialization
(`16 · n_q_full · n_rmu² / p_xy` ≈ 4.5 GiB / rank at CrI3 6×6) plus
one transient `V_at_irr` of the same shape.

## Model Corrections

The post-Round-6 fused kernel `jit__z_q_from_psi_sm` replaced
`jit__compute_ZCT_LR` (formerly the peak binder); its own peak is
~2–3× smaller because the bc-loop is now scan-aliased.  Binding peak
post-Round-8 is the two rank-5 P-pair carries (`P_l_acc`, `P_r_acc`)
which live across the γ̃ contract — `pair_density_slots` (3 on GPU XLA,
4 on CPU XLA) captures this in the G-flat planner.

Pre-Round-6 reference (legacy
`profiles/xprof/cohsex_prod-20260303-112900/...`):
`jit__compute_ZCT_LR` `peakHeapMib ~= 1794`, `jit__solve_all_q`
`~= 1485`, `jit__compute_P_traced` `~= 480-493`.

Recent corrections in `compute_optimal_chunks` (legacy heuristic):

1. ZCT stage coefficient updated from too-optimistic `2·P + Z` to the
   XProf-aligned `4·P + Z` live-set model.
2. Solve stage now explicitly includes replicated `L` panels in both
   feasibility checks and `q_chunk` sizing.
3. Breakdown output now reports ZCT and solve sub-components directly
   (`zct_pair_inputs`, `zct_fft_temps`, `solve_z_io`, etc.).

The G-flat planner supersedes these for `band_chunk` / `chunk_r`; the
heuristic remains the authoritative chooser for `q_chunk` / `q_gather`
/ `k_chunk`.  The **AOT memory model** below independently confirms
the hand-derived coefficients by NNLS-fitting `memory_analysis()`
against dimensional primitives — integer β's match the hand-derived
hypotheses (ZCT `β[PrBr]=4`, load-wfn `β[psi_G]=3`, chi0 τ-step
`β[Gbuf]=4`, `fit_one_rchunk` composite `β[pair]=4`).

## AOT Memory Model

`src/gw/aot_memory_model/` (in-tree, entry points re-exported from the
package `__init__`).

The hand-derived formulas above are hypotheses about each stage's
allocation set.  When XLA schedules something the hand-model missed —
remat, allocator coalescing, a donation failure — the symptom is a
runtime OOM at a `chunk_r` the heuristic claimed would fit.  The AOT
tool addresses this by *measuring* each jit's peak via
`jax.jit(f).lower(specs).compile().memory_analysis()` at many `(sys,
knobs, mesh)` points, then NNLS-fitting a non-negative linear
combination of dimensional primitives:

```
peak(sys, knobs, mesh) = intercept + Σ_i  β_i · T_i(sys, knobs, mesh)
```

Each primitive `T_i` is the bytes of one physical shard (e.g. `(n_k,
n_rmu, B_r)/P`), and `β_i` counts concurrent copies at peak.  A clean
fit gives integer β's with MB-range residual RMS — that's the signal
the primitive set is complete.  Re-run the sweep when you modify a
kernel; artifact JSONs are checked in so every agent sees the same
formulas.

### Architecture

```
aot_memory_model/
├── core.py                 SysDims, MeshSpec, Knobs, AotKernel, aot_measure,
│                           fit_nnls, predict_peak, save/load_fit
├── cost.py                 Analogous FLOPs fit: fit_cost_nnls,
│                           predict_flops_per_call, save/load_cost_fit
├── chooser.py              choose_chunks_analytic (closed-form)
│                           choose_chunks_heuristic (20/80 split)
│                           choose_chunks_aot (grid search)
├── doe.py                  build_doe_axes (one-at-a-time DoE generator)
├── presets.py              points_<kernel>() — per-kernel sample points
├── sweep.py                Rank-0 DoE driver: runs aot_measure + saves JSON
├── predict_cli.py          CLI: `python -m gw.aot_memory_model.predict_cli …`
├── kernels/                One module per covered jit (see table below)
└── artifacts/              Checked-in *__current__{samples,fit,cost_fit}.json
```

### Covered kernels (as of 2026-04-20)

| Kernel | Where in production | KNOBS | PRIMITIVES (fit β → integer count) |
|---|---|---|---|
| `cct_lr` | `common.isdf_fitting.compute_CCT_from_left_right` | — | `Pq` → 4 |
| `zct_lr` | `common.isdf_fitting.compute_ZCT_from_left_right_zchunk` | `chunk_r` | `PrBr` → **4** |
| `pair_density_traced` | `common.isdf_fitting.compute_pair_density_spin_traced` | — | `P`, `psiL`, `psiR` → 1,1,1 |
| `chi0_tau_step` | `gw.w_isdf._get_chi_minimax_kernel._tau_step` | — | `Gbuf`→**4**, `chi`→1, `psi`→1 |
| `solve_q` | `common.isdf_fitting.solve_zeta_from_L_q` | `chunk_r`, `q_chunk` | `Zcol`→3.6, `Lfull_rep`→2.4 (fractional — real-valued fit on a narrow DoE) |
| `slab_write` | `file_io.slab_io.SlabIO.write_slab` | `chunk_r` | `slab` → 1 |
| `load_psi_rchunk_fft` | `common.load_wfns.read_Gvecs_to_devices` + FFT | `k_chunk`, `band_chunk`, `chunk_r` | `psi_G`→**3**, `rchunk_xy`→1 |
| `load_psi_rchunk_reshard` | `common.load_wfns.iter_psi_rchunk_bandwise` reshard | `k_chunk`, `band_chunk`, `chunk_r` | `rchunk_xy`→1, `rchunk_y`→**2** |
| `sigma_kij` | `gw.ppm_sigma._get_sigma_kij_kernel` | — | `Gmid`→2.67, `Vmid`→2.33, `psi_X`→2, `psi_Y`→2, … |
| `vq_mu_chunk` | `gw.compute_vcoul.make_v_munu_chunked_kernel` | `mu_chunk` | `zeta`→**3**, `vphase`→1.5, `out`→1 |
| `fit_one_rchunk` | `common.isdf_fitting.fit_one_rchunk` (driver-level r-chunk body) | `chunk_r`, `band_chunk` | `pair`→**4**, `psiG_cache`→1, `centroid`→9, `Lq_rep`→1 |

Primitives in **bold** confirm the hand-derived α coefficients.  The
`fit_one_rchunk` composite kernel is the one the choosers consult — it
AOT-lowers the entire per-r-chunk driver body (FFT + reshard +
pair-density stream + ZCT + solve) in one HLO, so it captures
*coexisting* buffers that per-stage fits miss.

The post-Round-6 fused kernel is structurally different (single
`z_q_from_psi_sm` shard_map+scan replacing the FFT-reshard-pair-ZCT
chain); the existing `fit_one_rchunk` AOT fit is stale and should be
re-sampled before re-enabling the AOT chooser as the default.  Pending
work item (see
`reports/zeta_rchunk_memory_model_2026-05-13/round8_efficiency_audit.md`
§B5).

### NNLS fits in practice

The `sweep` module drives end-to-end fitting on rank 0 — runs a preset
DoE and writes both `*_samples.json` and `*_fit.json` into
`artifacts/`.  Sweeps must run inside `lxrun` because they need a real
JAX mesh, even though the AOT lower/compile path allocates no GPU
memory.  Programmatic API:

```python
from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs, get_kernel, aot_measure, fit_nnls, save_fit
)
kernel = get_kernel("zct_lr")
samples = [(sys, knobs, mesh,
            aot_measure(kernel, sys, knobs, mesh)["total"])
           for (sys, knobs, mesh) in points]
fit = fit_nnls(kernel, samples)
save_fit(fit, tag="current")
```

### Chooser modes

`gw_init.fit_zeta` consults the AOT chooser when
`cohsex.in :: use_aot_chunk_chooser = true`, picking one of three modes:

**Heuristic (default, `LORRAX_CHOOSER_MODE=heuristic`)** —
`choose_chunks_heuristic`.  No DoE lookups; splits the budget 20/80
between wfn-FFT workspace and the r-chunk body, then sizes against
two calibrated integer β's:

```
wfn_budget    = 0.20 · budget           # k_chunk · bc · (3·16·n_s·n_r/P) ≤ wfn_budget
rchunk_budget = 0.80 · budget           # 4·16·n_k·n_rmu·cr/P + M_persistent ≤ rchunk_budget
```

The 3 is `β[psi_G]=3` from `load_psi_rchunk_fft`; the 4 is `β[pair]=4`
from `zct_lr` + `fit_one_rchunk`.  Prefers `chunk_r` / `band_chunk`
values that divide `n_rtot` / `n_b` so every loop iteration is one
compile shape.

**Analytic (`LORRAX_CHOOSER_MODE=analytic`)** — `choose_chunks_analytic`.
Regroups the `fit_one_rchunk` β·T contributions into four scaling
classes (`PRIMITIVE_CLASSES` in `kernels/fit_one_rchunk.py` assigns
each primitive to `const` / `cr` / `bc` / `crbc`):

```
peak(chunk_r, bc) = α₀ + α_cr·chunk_r + α_bc·bc + α_crbc·(chunk_r·bc) ≤ M
```

Inverted in closed form per candidate `bc`; minimises total FLOPs via
the companion `cost_fit`.  Also enforces the post-jit allgather bound
(`16 · q_gather · n_rmu · cr` bytes per device).

**Grid search** — `choose_chunks_aot`.  Coarse `(cr, bc)` grid +
`predict_peak`.  For tests and manual exploration; not wired into the
driver.

### γ calibration

`memory_analysis()` is an upper-bound *compiler* estimate — XLA
schedules tighter at runtime, and NCCL / cuFFT scratch sits outside
the reported peak.  Each `Fit` carries a scalar
`γ = runtime_peak / aot_predicted` applied uniformly to intercept and
all β·T terms (fresh fits start at `γ = 1.0`).  `gw_init.fit_zeta`
prints `γ` after every ζ-fit so it can be updated manually; a CLI to
roll it into the artifact JSON is still TODO.  Wildly optimistic
prediction vs runtime is the post-Round-6 norm on small systems
(e.g. Si 4×4×4 25 Ry / 2 GPUs measured γ ≈ 0.216 against the stale
fit), since the AOT fits were calibrated on the pre-Round-6 kernel
shape.

### When to trust which chooser

- **G-flat planner** (default for the ζ + V_q pipeline): closed-form,
  microseconds, calibrated by HLO BufferAssignment slot counts.
  Override only if HWM disagrees with runtime peak by more than ~10 %;
  re-fit `pair_density_slots_*` from a fresh HLO memory-usage-report
  and patch `gflat_memory_model._peak_C_fit_one_rchunk` defaults.
- **Legacy heuristic** — α coefficients physically motivated and stable
  across untested (system, mesh) combinations.  Authoritative for
  `q_chunk` / `q_gather` / `k_chunk`.
- **AOT 20/80 heuristic** — one free parameter (`wfn_workspace_frac`)
  and intentionally conservative; useful for diagnosing "am I leaving
  memory on the table?".
- **AOT analytic** — maximally precise within the calibrated DoE range,
  but can overfit on narrow sweeps (e.g. `solve_q` fractional β's from
  7 points) and misses by ~10 % when extrapolating.  Not the default
  until its `fit_one_rchunk` artifact is re-sampled against the
  post-Round-6 fused kernel.

### Status (2026-05-15)

G-flat planner is the default and validated at Si 4×4×4 80 Ry (HWM
24.21 GB predicted, runtime ~5 % off) and CrI3 6×6 80 Ry (HWM 13–15 GiB
measured, ~3× reduction vs pre-Round-6).  AOT model is scaffolded and
calibrated at MoS2 3×3 / Si 4×4×4 but its `fit_one_rchunk` artifact is
**stale** against the post-Round-6 fused kernel; re-sampling is the
[round8_efficiency_audit.md §B5] follow-up.  Known gaps: no multi-node
DoE (some primitives are collinear at fixed `total_devices` — e.g.
`Lq_sharded` vs `Lq_rep` fold into one coefficient — predictions beyond
4 GPUs need validation); `q_chunk` / `q_gather` / `k_chunk` stay on the
hand-heuristic; `γ` is manual, not persisted to JSON, and wired into
the analytic chooser only.  Per-kernel DoE caveats live in the
`.notes` fields of the `*_fit.json` artifacts.

Pending: bispinor-transverse CCT path (Round-9); multi-node DoE sweep;
`γ` persisted to JSON; `solve_q` `q_chunk` AOT-controlled.

See `reports/zeta_rchunk_memory_model_2026-05-13/PATH_D_PICKUP.md` →
`round8_efficiency_audit.md` → `round5_unified_plan.md` for the
mandatory pickup-reading order on this initiative.

## Predicted-vs-realized faithfulness (Round-7 audit, 2026-05-17)

The planner's `HWM_pred` is an **upper bound** on the in-jit transient
peak assuming no XLA buffer aliasing/donation; it is intentionally
conservative.  Round 7 (`agent_n_faithfulness_audit.md`) measured the
spread between three metrics across cs ∈ {50, 100, 500, 1000} on the
production 16-GPU CrI3 80 Ry SOC bispinor:

| metric | what it sees | typical value |
|---|---|---|
| `HWM_pred` (planner) | upper-bound in-jit transient, no aliasing assumed | 22-66 GB/dev (depends on r_chunk) |
| `jax.live_arrays()` sum, sharding-corrected | persistent + post-jit transient (XLA arena, no in-jit transients) | ~5-9 GB/rank |
| `nvidia-smi memory.used` | true HBM, including everything outside the JAX arena | 7.75-8.67 GB/rank |

**Key findings:**

1. **HWM_pred is 7-8× higher than realized nvsmi peak.**  HWM_pred = 66.41
   GB/dev at r=24576, but actual nvsmi observes only 8.67 GB/dev (12 % of
   the 70 GB budget).  XLA's buffer aliasing/donation/remat saves the
   remaining 57.74 GB/dev that the planner cannot see in static analysis.
   The over-prediction errs toward safety: the planner cannot OOM-miss,
   but it can pick chunk sizes more conservatively than needed.

2. **JAX live_arrays view agrees with nvidia-smi to ~4%** when properly
   sharding-corrected (sharded-globals / P + replicated-globals).  The
   model's blind spot (constant ~3.6 GB/rank between raw `live_total/16`
   and nvsmi) is **NOT** cuFFT-related; it's persistent JAX/XLA overhead
   + CUDA context + NCCL collective buffers, which is cs-independent.

3. **`gflat_chunk_size` cap of 100 is empirically very conservative.**
   cs ∈ {50, 100, 500, 1000} all show identical nvsmi peak 8.67 GB/dev —
   cuFFT does not blow up workspace in this range.  The agent_f cs=1414
   hard-OOM remains the cliff, so the cap is safety not waste.  No
   performance gain from raising cs (agent_d M3), so 100 stays.

4. **`device.memory_stats()` returns `None` on the Perlmutter JAX 0.8 /
   CUDA 12.9 stack.**  `peak_bytes_in_use` is unavailable; `_mem_probe`
   in `common/isdf_fitting.py` (commit `6ba1fad`) falls back to
   `nvidia-smi` for the local GPU and tracks a running peak.  This is
   the only per-rank OOM-faithful metric on this stack.

**Trust matrix for memory planning:**

| chunk-sizing question | trustworthy metric | source |
|---|---|---|
| Will this config OOM? | nvsmi_peak < 0.95 × 80 GB | Round-7 X1-X6 all under 9 GB/dev |
| What's the actual persistent state? | live_arrays sum (sharding-corrected) | Round-6 m1 sphere-idx audit |
| What in-jit transient does XLA briefly allocate? | HWM_pred (upper bound, ~7× over) | Round-3 V1-V5 + Round-7 |
| Where's the cuFFT cliff? | known: cs=1414 OOM, cs=1000 OK | Agent F + Round-7 X6 |

See `reports/memory_model_refit_2026-05-17/agent_n_faithfulness_audit.md`
for the full table and per-config nvsmi traces.

## Appendix: Persistent Arrays Verified by `jax.live_arrays()` Probes

The Round-2 refit (commit `38xxxxx`, 2026-05-17) added per-array
accounting to `gflat_memory_model.plan_gflat_chunks` based on
`jax.live_arrays()` probes in `isdf_fitting.py` and `gw_init.py`.
This appendix is the cheat-sheet for future agents: when
`LORRAX_MEM_DEBUG=1` shows an unexpected shape in HBM, grep this
table for the shape and you'll find the planner term to inspect.

Quantitative measurements are for the production CrI3 6×6 80 Ry SOC
bispinor on 16 GPUs (4×4 mesh, ``p_xy=16``, ``nk=36``, ``ns=2``,
``mu=1520``, ``nb=150``, ``ngkmax=59990``, ``n_rtot=1.125M``,
``fft_grid=(75, 75, 200)``). All probe data from
`reports/memory_model_refit_2026-05-17/` (Agents F/G/H/I).

### A. Persistent throughout ζ-fit (alive from `prepare_isdf` through `compute_V_q`)

| live_arrays signature | meta-var formula | per-rank GB | allocation site | sharded? | planner term | smoking gun |
|---|---|---|---|---|---|---|
| ``c128 (nk, mu, nb, ns)`` ×4 buffers per channel (rmuT_X + transposed Y form, for both ψ_l and ψ_r) | ``4 × nk × ns × mu × nb_total × 16 / p_xy`` | 0.066 | ``common/load_wfns.py:474`` (``gflat_to_rmu`` fills psi_rmu_Y/X); transpose copy created at ``common/isdf_fitting.py: fit_zeta_to_h5`` step 1 (slice/divide-by-norms doubles each into a Y-form view) | μ-sharded on ``('x','y')`` | ``{B,C,D}.centroids_persist`` (and ``E.psi_centroids_persistent``) | Agent F probe 1B + Agent G §6 row #1: pre-refit counted ×2, runtime shows ×4 |
| ``c128 (nq, mu, mu)`` | ``nq × mu × mu × 16 / p_xy`` | 0.083 | ``common/isdf_fitting.py: factor_c_q`` (step 3 of ``fit_zeta_to_h5``) | μ-sharded | ``{B,C,D}.L_q`` | Agent F probe 1A row 2: 1.33 GB global / 0.083 GB/rank |
| ``c128 (nq_disk, mu, ngkmax)`` | ``nq_disk × mu × ngkmax × 16 / p_xy`` | 3.283 | ``common/isdf_fitting.py:2443`` (``jnp.zeros`` jit just before r-chunk loop) | μ-sharded | ``C.gflat_acc`` AND ``D.gflat_acc`` (Round-10 / agent_q: resident across the r-chunk loop, fit_one_rchunk and accumulate are separate jits with isolated transient slots so charging both Peak C and Peak D persistent bases is correct, not double-counting) | Agent F probe 1A row 1: 52.52 GB global / 3.28 GB/rank; Round-9b agent_o live_arrays census re-confirmed on Y3_95 |
| ``int32 (nk, nx, ny, nz)`` ×N (post-Round-6: N=1 for both bispinor and charge; post-Round-4 / pre-Round-6: N=3 — three content-distinct numpy sources produced 3 device buffers with identical content but distinct sharding; pre-Round-4: N=8 bispinor, N=3 charge) | ``N × nq × fft_grid_x × fft_grid_y × fft_grid_z × 4`` | 0.162 post-fix (was 1.296 pre-Round-4, 0.486 between Round-4 and Round-6) (REPLICATED) | ``common/gvec_fft_box.py:55`` (``g_index = np.full((nk, nx, ny, nz), ngkmax, dtype=np.int32)``); pre-Round-4 each fresh ``psi_G_store._populate_from_loader`` + each fresh ``gflat_to_rmu`` ``build()`` closure created a new device buffer per channel; **Round-4** (commits d1fcd20 + 94542c2) added per-source caches (``WfnLoader.box_index_dev`` + ``_cached_gindex_dev``) — bounded growth WITHIN each source but loader-side and wfn_transforms-side buffers stayed unbridged (NamedSharding vs SingleDeviceSharding); **Round-6** (commit 9afa11e) routes ``gflat_to_rmu`` through ``WfnLoader.box_index_dev`` via ``shard_map`` in_specs (Manual-mode-compatible), collapsing all three pre-Round-6 sources to one canonical allocation | **REPLICATED — not /p_xy** | ``{A,B,C,D,E}.sphere_idx_replicated`` | Agent H §3 Finding 3 (pre-fix): 2→3→6→7→8 buffers; Round-4 verdict (agent_l_round5_liveverify §2): 3 buffers; **Round-6 verdict (agent_m_round6): 1 buffer** |

### B. fit_one_rchunk transient (alive only after fit returns, freed when accumulate consumes)

| live_arrays signature | meta-var formula | per-rank GB | allocation site | sharded? | planner term | smoking gun |
|---|---|---|---|---|---|---|
| ``c128 (nq_disk, mu, r_chunk)`` | ``nq_disk × mu × r_chunk × 16 / p_xy`` | 1.16 (at r=21232) | ``common/isdf_fitting.py: fit_one_rchunk`` return | μ-sharded | ``D.zeta_chunk`` (transient) | Agent F probe 1B (+18.59 GB vs 1A); freed at probe 1C via ``donate_argnums=(1,)`` |

### C. fit_one_rchunk inside-jit (XLA preallocated-temp; invisible to live_arrays)

| live_arrays signature | meta-var formula | per-rank GB | allocation site | sharded? | planner term | smoking gun |
|---|---|---|---|---|---|---|
| ``c128 (nk, ns, ns, mu_local, r_loc)`` ×3 slots (aliased to P_l_R_conj / P_r_R / FFT box) | ``3 × nk × ns² × mu × r_chunk × 16 / p_xy`` | 14-20 (at r=21232-24576) | ``common/isdf_fitting.py:625-627`` (P_l_acc/P_r_acc) + ``isdf_fitting.py:713-720`` (P_l_R_conj reshape) | μ × r sharded | ``C.P_pair_concurrent_slots`` | Agent D M1: 3 distinct preallocated-temp slots × 20.04 GiB each in module_0438 |

### D. accumulate_rchunk_to_gflat inside-jit

| live_arrays signature | meta-var formula | per-rank GB | allocation site | sharded? | planner term | smoking gun |
|---|---|---|---|---|---|---|
| ``c128 (gflat_chunk_size, nx, ny, nz)`` + ``c128 (gflat_chunk_size, n_rtot)`` flat = 2 box-sized slots | ``factor_D × gflat_chunk_size × n_rtot × 16``, ``factor_D = 2.0`` | 0.036 (cs=1) → 3.6 (cs=100 cap) | ``common/wfn_transforms.py: accumulate_rchunk_to_gflat._kernel`` (lines 1057-1107) | XLA-internal | ``D.accumulate_fft_box`` | Agent D M2 module_0474: 2 box slots × 6.03 GiB at cs=360; Agent D M3 module_0363: 2 × 17 MB at cs=1; factor_D=2.0 confirmed at both |

### E. V_q per-tile transient (allocated/freed per tile in `_compute_V_q_g_flat_one_tile`)

| live_arrays signature | meta-var formula | per-rank GB | allocation site | sharded? | planner term | smoking gun |
|---|---|---|---|---|---|---|
| ``c128 (n_q_ibz, mu, ngkmax)`` (CC or TT diag) | ``n_q_ibz × mu × ngkmax × 16 / p_xy`` | 3.28 | ``gw/v_q_g_flat.py:372-384`` (zeta_L_all pre-loop) | μ-sharded | ``E.zeta_L_all`` | Agent I §2 binding term |
| ``c128 (n_q_ibz, mu, ngkmax)`` second copy (TT off-diagonal only) | ``n_q_ibz × mu × ngkmax × 16 / p_xy`` | 3.28 (off-diag) / 0 (CC + diag) | ``gw/v_q_g_flat.py: same`` | μ-sharded | ``E.zeta_R_all`` | Agent I §2: doubles slab term for ``same_zeta=False`` |
| ``c128 (mu, ngkmax)`` (resharded inside per-q kernel) | ``mu × ngkmax × 16 / p_x`` | 0.365 | ``gw/v_q_g_flat.py: _make_per_q_kernel.fn`` (reshard to ``P('x', None)``) | sharded /p_x (REPLICATED on y) | ``E.zeta_L_on_x_axis`` | Agent I §2 |
| ``c128 (n_q_ibz, mu, mu)`` (V_acc; post-unfold piggybacks same slot) | ``n_q_ibz × mu × mu × 16 / p_xy`` | 0.083 | ``gw/v_q_g_flat.py:372`` | μ-sharded | ``E.V_acc`` + ``E.V_acc_full_BZ`` | Agent H probe P5: post-V_q live_total +1.33 GB global = V_qmunu_CC |
| ``c128 (n_q_full, mu, mu)`` ×{9, 6} (Lorentz mix, bispinor IBZ-T only) | ``{9, 6} × nq × mu × mu × 16 / p_xy`` | 1.22 total | ``gw/v_q_bispinor.py:587-728`` (``unfold_v_q_bispinor_lorentz``) | μ-sharded | ``E.tt_full_in_9_tiles`` + ``E.tt_mixed_6_tiles`` | Agent I §4 |

### How to use this appendix

If `LORRAX_MEM_DEBUG=1` prints a `live_arrays()` row whose shape you
don't recognise:

1. Grep for the shape pattern (e.g. ``(36, 1520, 59990)``) in the
   table above.
2. The "planner term" column tells you which `_peak_*` helper in
   `src/gw/gflat_memory_model.py` models it.
3. The "smoking gun" column points to the report under
   `reports/memory_model_refit_2026-05-17/` that first measured it.

If the live_arrays-observed shape is NOT in the table, it's likely
a new buffer the planner doesn't model — open an issue against
`gflat_memory_model.py` and consider adding it. The procedure is:
(a) identify allocation site via `id(arr.sharding.mesh)` + Python
trace; (b) classify lifetime (alive across which peaks?); (c) add
a term to the appropriate `_peak_*` dict.

