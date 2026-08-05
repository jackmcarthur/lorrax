# HLO findings — `fit_one_rchunk` at CrI3 6×6 80 Ry, planner-free

Run dir: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/`
Kernel HLO: `xla_dump/module_0408.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt` (identical 200.35 GiB total to modules 0297/0299 — same kernel re-compiled at different call sites).

**Framing.** This report uses CrI3 as a single validation point because the run exists. The findings are about *the general memory model*: how to pick chunk sizes from `(n_rtot, n_rmu, nk, ns, p_x, p_y, nb_total, budget)` so the kernel fits any material at any proc grid. The numerical examples below are concrete instances, not targets.

## Framing: replicated intermediates are defects, not resources

Per the user 2026-05-13: **there should be zero replicated intermediates or work done in this entire procedure**. The findings below identify a specific class of defect — unsharded FFT-box buffers materialized in `fit_one_rchunk` — and the magnitudes those defects produce at one scale. The numbers are evidence that the defect exists and how badly it bites; they are NOT a model of an acceptable cost to budget around. The right response is to eliminate the buffers (Path D), not to refine the model that predicts how big they are. The planner-side accommodation landed in `ff5873c` is a stopgap; remove it when Path D lands.

## TL;DR — three findings, in priority order

1. **The band-load FFT box is structurally unsharded on every rank.** The `c128[k_chunk, band_chunk, ns, nx, ny, nz]` buffer takes `k_chunk · band_chunk · ns · n_rtot · 16` bytes *per rank*, with no division by `P`. The current model divides by `p_xy` and is wrong by exactly that factor. The fix is not a tunable — it's a layout fact.

2. **The Python-unrolled bc-loop pins `N_BC × ~3` concurrent FFT-box slots.** XLA's `BufferAssignment` cannot alias them — at CrI3 with `band_chunk=16, nb_total≈310, psig_k_chunk_size=6`, the dump shows **58 slots of 3.22 GiB each** in the preallocated-temp pool. Per-rank cost scales as `N_BC · S_fft · k_chunk · band_chunk · ns · n_rtot · 16` where `S_fft ≈ 3` is the shape-variant slot count.

3. **`pair_density_slots = 3` is correct.** The kernel HLO shows exactly three 14.79-GiB pair-density slots (allocations 31's three largest "slot 0/1/2" rows), each holding the `c128[ns, ns, μ·r_local, nk]` shape and its permutations. The consensus's default value of 3 is verified.

Combined: the current model's `W_wfn ≪ W_zeta` at CrI3 — but in reality `W_wfn × N_BC × P_xy` dominates `W_zeta` by ~3× to 10× depending on geometry. The OOM in this run (200 GiB requested / 60 GiB budget) is fully explained by mechanism (1)+(2).

---

## 1. What XLA actually allocated

Total: **200.35 GiB per rank**. Compare planner's HWM estimate: **51.96 GiB**. Miss: 3.85×.

| Allocation | Size | Role | Notes |
|------------|------|------|-------|
| `allocation 31` (preallocated-temp) | 196.30 GiB | XLA's reused scratch pool | Holds 3 pair-density slots + 58 FFT-box slots + many smaller buffers; lifetimes overlap so no further aliasing |
| `allocation 0` (maybe-live-out) | 3.70 GiB | Output ζ_chunk + a few co-live values | `c128[6, 32, 1125000]` (n_rtot-sized accum) and ζ-output shapes |
| `allocations 1–3` (parameters) | 0.20 GiB | ψ inputs `c128[36, 376, 160, 2]` etc. | μ already μ_X = 376 = 1504/4 → sharded on one mesh axis |
| `allocations 4–9` (constants) | 0.15 GiB | bz constants, phase tables | small |

Inside the preallocated-temp slot list:

| Slot count | Per-rank size | Slot shapes (canonical) | Interpretation |
|------------|---------------|-------------------------|----------------|
| 3          | 14.79 GiB     | `c128[ns, ns, μ_local · r_local, nk]`, `c128[nk, 2r_local, 2μ_local]`, etc. | **Pair-density slots — confirms `pair_density_slots = 3`** |
| 58         | 3.22 GiB      | `c128[k_chunk, band_chunk, ns, nx, ny, nz]`, `c128[k_chunk, 2·band_chunk, n_rtot]`, `c128[k_chunk, band_chunk, ns, ngkmax]` | **Band-load FFT box + intermediates — unsharded, N_BC-unrolled, per-stage** |

Per-rank bytes for one 3.22 GiB FFT-box slot at this geometry:

```
6 · 16 · 2 · 75 · 75 · 200 · 16 B  =  3.456 GB  (matches 3.22 GiB ≈ 3.456 · 10^9 bytes)
```

If the buffer were sharded on `p_xy = 16`, per-rank would be 0.215 GiB. If sharded on one axis (`p_x = 4`), 0.86 GiB. The observed 3.22 GiB confirms **no mesh sharding holds on the FFT-box buffer.**

## 2. The general formula the model should use

For Peak C (inside `fit_one_rchunk`), the W_wfn term is currently in `gflat_memory_model._peak_C_fit_one_rchunk` and is silently aliased into the rank-5 pair-density slots. **That assumption is wrong.** The actual cost (after Agent 1 §2 verified the formula at 0.1% accuracy against `module_0408`):

```
W_wfn_actual  =  (nb_L + nb_R) · S_fft · k_chunk_eff · ns · n_rtot · 16      [per rank, NO /P]

where
  nb_L + nb_R   = sum of L and R band-window widths, i.e. (b3-b0) + (b4-b1).
                  For symmetric GW this is ≈ 2·nb_F. Already canonical in
                  the planner as `nb_total_chunker` (gw_init.fit_zeta:588).
  S_fft         = 3 (verified from HLO slot count, 58 slots / 20 bc-iters ≈ 2.9)
  k_chunk_eff   = psig_k_chunk_size if > 0 else nk
  ns            = nspinor
  n_rtot        = nx · ny · nz
```

(`band_chunk` does **not** appear: `n_bc · band_chunk = nb_L + nb_R` so the
product is band_chunk-invariant. Correction 2026-05-13 per
agent_1_hlo_verify.md §4.1.)

This is a **separate, additive** term to `W_zeta` and `W_pair_density`, not a slot to alias. At any geometry where `n_rtot` is large (≳ 10⁶) and `N_BC` is more than a handful, `W_wfn_actual` will dominate.

The chunker's degrees of freedom against this term are:
- `psig_k_chunk_size ↓` reduces `k_chunk_eff` linearly. **Only knob in the planner that moves this term.**
- Narrowing `nval`/`ncond` reduces `(nb_L + nb_R)` linearly (not a chunker knob — physics).
- Neither `band_chunk`, `r_chunk`, nor `gflat_chunk_size` appears in `W_wfn_actual` — `band_chunk` is band-axis-independent because `N_BC · band_chunk = nb_L + nb_R`; the other two control different peaks. **(Correction 2026-05-13 per agent_1_hlo_verify.md §4.2: an earlier draft of this section claimed "`band_chunk ↑` reduces `N_BC` linearly (the dominant lever)" — that was wrong; only `psig_k_chunk_size` and the band-window widths shift this term.)**

So the algorithm needs to be:

```
Step 1.  W_wfn_actual(band_chunk, k_chunk)  ≤  α · W_pool      (some α ≤ 1)
Step 2.  W_zeta(r_chunk)                     ≤  (1−α) · W_pool   (or share via aliasing if that's confirmed)
Step 3.  W_accum(gflat_chunk_size)           ≤  W_pool            (separate jit)
```

with `α` chosen so the FFT-box term doesn't crowd out `W_zeta`. The two-term tradeoff is **not** scale-free: increasing `band_chunk` decreases `W_wfn_actual` (linearly via `1/N_BC`) but also increases the size of the per-bc FFT box (linearly via `band_chunk`). Net `W_wfn_actual` is monotone-decreasing in `band_chunk` only up to the point where `band_chunk = nb_total` (single iteration); past there, the FFT box itself can blow the budget. There's an interior minimum.

This is fundamentally different from the current model's "max each knob as large as memory allows."

## 3. Structural fix paths (not yet implemented)

| Path | What it does | Pros | Cons |
|------|--------------|------|------|
| A. `lax.fori_loop` over bc | Replace Python-unrolled bc-loop with a JAX-traced loop. XLA sees one iter → one FFT-box slot. | Eliminates `N_BC ×` factor entirely. Cleanest fix. | Requires the bc-loop body to be JIT-pure (no Python-side state). May need refactor in `c_q_from_psi_sm` / `z_q_from_psi_sm`. |
| B. `with_sharding_constraint` at FFT-box creation | Pin the FFT box to a mesh-axis sharding spec at the line that creates it. | If it sticks, recovers `/p_xy` factor. | Per consensus.md §6 trap: at large `n_rtot` XLA may keep both pre- and post-constraint layouts live, doubling peak. Has historically not worked. |
| C. Accept the cost in the model | Add `W_wfn_actual` to the planner as a separate budget term. | One-shot edit. Picks correct chunks. | Doesn't reduce HBM use — just makes the planner pick smaller `band_chunk` to compensate. Less efficient than (A). |

**Recommendation:** start with (C) so the planner stops promising what XLA can't deliver, then pursue (A) as a real-world performance improvement.

## 4. What this means for the existing consensus.md

- **B-1 (unsharded FFT box):** RESOLVED in favor of unsharded. The pathology is real and load-bearing.
- **B-3 (`pair_density_slots`):** RESOLVED — `= 3` is correct. Default in source stays.
- **B-4 (`psi_Y_full` aliasing):** Indirectly resolved — band-axis-related buffers all share FFT-box slots; there's no separate `psi_Y_full` slot in the dump. Aliases cleanly.
- **B-5 / B-6 (where does the planner's pick come from):** EMPIRICALLY confirmed — `plan_gflat_chunks` is the canonical planner; `gw_init.py:617` unconditionally adopts its `r_chunk`. The "12 500" number in `report.md §7` is **wrong**: the actual planner-free pick at the §7 60 GB / `band_chunk=16` config is `r_chunk = 73 328` (16 chunks), HWM-estimated at 51.96 GB. The planner is internally consistent; its outputs are just based on a model that misses the FFT-box term, so it picks chunks that OOM at runtime.

## 6. 2nd HLO dump (2026-05-13, after `ff5873c`) — validates direction, model now over-conservative

Run dir: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_k1_2026-05-13/`.
Same geometry, mesh, hardware as §1. Knob changes: `psig_k_chunk_size = 6 → 1`, `memory_per_device_gb = 60 → 35`. Goal: validate the linear-in-`psig_k_chunk` claim and confirm the post-fix planner refuses-or-picks-feasible at a tighter budget.

**Empirical results:**

| Quantity                       | 1st dump (`psig_k_chunk=6`) | 2nd dump (`psig_k_chunk=1`) | Ratio |
|--------------------------------|------------------------------|------------------------------|-------|
| Planner HWM prediction         | 52 GB / 60 GB budget         | 47 GB / 35 GB budget         | —     |
| XLA actual allocation per rank | 200.35 GiB                   | 23.42 GiB                    | 8.55× drop |
| `r_chunk` (planner-picked)     | 73 328 (16 chunks)           | 17 568 (65 chunks)           | 0.24× |
| Run outcome                    | OOM at fit_one_rchunk        | **completed end-to-end** (~17 min)     |       |

**Two observations:**

1. The fix's *direction* is correct. The run that catastrophically OOMed on the 1st dump now completes cleanly at `psig_k_chunk=1`. The W_wfn term was the binding constraint and `psig_k_chunk` is the working knob.

2. The model is now **~2× over-conservative**. Predicted 47 GB; reality 23 GB. Two causes contribute:
   - `r_chunk` dropped 0.24× as a side-effect of the tighter budget (planner's r_from_budget calculation), so `W_zeta` ∝ `r_chunk` dropped along with it. The 8.55× total reduction = 6× from `psig_k_chunk` (the linear factor we predicted) × 1.5× from XLA aliasing more aggressively at smaller per-slot sizes. **The model assumes "everything live concurrently"; XLA aliases lifetimes much more freely once per-slot bytes are small enough that the allocator has scheduling headroom.**
   - The planner's `band_fft_pool` term still doesn't account for any partial aliasing within the pool itself.

**Implication for further model refinement:** any tightening of the model risks under-predicting again at larger configurations where aliasing doesn't kick in. The right path is **Path D** — collapse the slot count entirely so the model becomes trivially tight rather than fragile.

**Implication for B-5/B-6 (origin of `r_chunk` picks):** definitively answered. `plan_gflat_chunks` produces the picks; `gw_init.py:617` adopts them; the `max_chunks=64` floor does NOT bind at any config we've examined.

## 7. The inverted-priority algorithm (where the model "wants" to go)

The current planner picks `band_chunk` first then `r_chunk`. After all of today's analysis, the right framing is the inverted one:

**Priority order for a general material × proc-grid × budget:**

1. **`r_chunk` target.** Set as large as physics allows — ideally `r_chunk = n_rtot` (single r-chunk). Each r-chunk pays a fixed FFT/I/O tax on both the wavefunction-fetch side (read ψ(G) → IFFT → r-slice) and the accumulator side (write ζ_chunk → G-flat). Fewer r-chunks ⇒ less of that tax. **This is the dominant performance lever.**
2. **`W_zeta`-required headroom.** Given the `r_chunk` target, compute `W_zeta = pair_density_slots · 16 · nk · ns² · μ · r_chunk / p_xy`. This is one of two r_chunk-sensitive terms (the other is `zeta_chunk` post-solve).
3. **`W_wfn`-required headroom.** Currently fixed once `psig_k_chunk_size` is chosen: `(nb_L+nb_R) · S_fft · psig_k_chunk · ns · n_rtot · 16`. **`band_chunk` does not move this term** (it's band-axis-invariant). The only knob is `psig_k_chunk_size`.
4. **Back-solve.** If `W_zeta + W_wfn + B_persist > budget`, decrease `psig_k_chunk_size` first (linearly reduces W_wfn) until either: (a) it fits, or (b) `psig_k_chunk_size` hits a *performance floor* (the cuFFT batch size where GPU thread saturation stops being achieved — unknown today, would need a benchmark). If even `psig_k_chunk_size=1` doesn't fit, back off `r_chunk` from `n_rtot` down.
5. **`gflat_chunk_size`** is independent (Peak D, separate jit). Default to one-shot; bisect down only if Peak D ≥ budget.

Knobs that have **no effect on the bottleneck**:
- `band_chunk`: irrelevant to W_wfn total. Currently it only moves a per-bc FFT-box transient that XLA aliases under fusion.
- `gflat_chunk_size`: only affects Peak D, separate jit.

**Performance floor on `psig_k_chunk_size`:** the user-suggested "FFT workspace big enough to saturate GPU threads" is a real but unknown constraint. The 3D FFT over `c128[..., 75, 75, 200]` (n_rtot ≈ 1.1M) per k-point per band per spinor is small enough that batch fusion across `(k_chunk · band_chunk · ns)` is needed to saturate cuFFT. The minimum effective batch is hardware/library-dependent — a profiling experiment (sweep `psig_k_chunk_size ∈ {1, 2, 3, 4, 6}` at fixed everything else, measure GPU-side wall time of the fetch_psi_rchunk call) would pin it.

**After Path D** this whole priority inversion becomes moot: the slot count collapses to ~3, W_wfn drops to a single small term, `r_chunk` and `psig_k_chunk_size` decouple, and the "largest r_chunk you can afford" planner is straightforward. **Path D is the right next step, not further priority-shuffle refinement of the current planner.**

## 5. Run details

- cohsex.in: `memory_per_device_gb = 60`, `band_chunk_size = 16`, `r_chunk_size = 0`, `gflat_chunk_size = 64`, `psig_k_chunk_size = 6`, `bispinor = false`. (`gflat_chunk_size` was overridden by planner to 555 — the cohsex.in value of 64 only wins when planner can't fit; here the planner picked larger because the gflat term wasn't the bottleneck under its model.)
- Hardware: 4× A100-SXM4-**40GB** (lxalloc didn't request `hbm80g`; the planner believed the user's 60 GB spec).
- **Mesh: 4×4** (`p_x = p_y = 4`, `P_xy = 16`). All "per-rank" byte counts above are at this geometry; the unsharded factor in `W_wfn_actual` means the relevant `P_xy = 16` factor is *absent* from the formula (correction 2026-05-13 per agent_1_hlo_verify.md §4.3).
- OOM at runtime: requested 196.30 GiB on one device. Compile succeeded; allocation failed.
- HLO dump: 3290 files; the three `jit__kernel` modules are the relevant ones (all 200.35 GiB).
- After applying the `_bytes_centroids_LR` fix on `agent/zeta-r-chunk-fixes-2026-05-13` (commit pending), the planner's *predicted* HWM moved from ~46 GB (broken) to ~52 GB — the 4× correction landed in the centroid term as expected, but it's still tiny relative to the unsharded-FFT term.
