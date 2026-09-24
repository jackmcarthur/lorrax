# Memory model

A LORRAX calculation is feasible on `P` ranks exactly when every stage's
per-rank live set fits the device budget. Every large object is tiled over
the whole `x × y` mesh, so its per-rank bytes fall as `1/P`; nothing large is
materialised on fewer than all `P` ranks. This page states, stage by stage,
what is resident per rank, how it scales, which planner prices it, and what
it refuses before dispatch.

| symbol | meaning |
|---|---|
| `N_k` | full-zone k points (= full-zone q points) |
| `Q` | q rows the ζ fit keeps (the IBZ wedge when the grid is reduced) |
| `n_s` | spinor components (1, 2; 4 for the bispinor fit) |
| `μ` | centroid count, mesh-padded |
| `N_b` | bands in the fit window |
| `N_G`, `N_Gψ` | ζ sphere and ψ sphere sizes (`ngkmax`) |
| `N_r` | FFT grid points |
| `P = p_x·p_y` | ranks on the square mesh |

All arrays are complex128 (16 B) unless stated.

## Budget

`memory_per_device_gb > 0` is used as given (decimal GB). At `0` the budget
is the minimum over processes of `common.gpu_utils.get_device_memory_gb()`:
0.90 of the XLA pool limit on GPU (derived from the device total when the
client reports no limit), and 0.90 of host RAM per device on CPU. The minimum
is taken because static tile shapes must agree on every process.

A planner fills `target = budget × utilization`. Utilization defaults to
0.90, 0.85 and 0.78 for `n_s` = 1, 2 and ≥4
(`bfc_fragmentation_target_utilization`): a wider spinor axis makes one larger
contiguous arena, which needs more headroom against allocator fragmentation.
`ISDF_CHUNK_TARGET_UTILIZATION > 0` overrides it, clamped to `[0.85, 1.0]`.

## The unit: one Green's-function tile

```text
G_tile = 16 · N_k · n_s² · μ² / P          bytes per rank
```

`G_tile` is one k-resolved spinor `(μ, ν)` operator (`G`, `W`, `V`) sharded
`P(None, 'x', 'y')`. The screening and Σ stages hold a few of them; the
shared-pole capacity ledger warns above `3U` with `U = G_tile`. The ζ-fit
receipt prints every term as a `G_tile` ratio against a ceiling of
`4·G_tile`, so a reader sees at once whether the fit, rather than the GW
stages, sets the node count.

## Stage inventory

| stage | resident per rank (leading terms) | priced by | refuses |
|---|---|---|---|
| ψ(G) and centroid faces | charge fit: conj ψ(G) on the rank's G slots, `16·N_k·N_b·n_s·N_Gψ/P`. Centroid faces `ψ(r_μ)`: two copies of `16·n_par·n_s·μ·N_b/P` in the face layout (`low_mem_bands = true`, the default; `n_par` raw parent k points) | the ζ-fit planners below | with the fit |
| ζ fit, charge channel (route G) | C factor, one μ-batch working set, the ψ(G) slice; the Z store `16·Q·μ·N_G/P` lives on host or disk | `plan_zeta_route_g` ([§ route G](#route-g-every-ζ-fit)) | `GATE zeta-mubatch-capacity` |
| ζ fit, current channels (bispinor) | the same, at μ_T, with three factors, accumulators and Z stores | `plan_zeta_route_g(n_vertex=3)` ([§ route G](#route-g-every-ζ-fit)) | `GATE zeta-mubatch-capacity` |
| V_q | `V_acc` `16·Q·μ_L·μ_R/P`, one q-tile of ζ rows, G panels | `vq_tile_bytes` ([§ V_q](#vq-g-panels-and-q-tiles)) | `GATE vq_tile_budget` |
| V_q unfold | `16·N_k·μ²/P`, sharded `P(None,'x','y')` | — | — |
| shared-pole screening and Σ | response-bank faces, pencils, eigh workspace, then G and W tiles | the capacity ledger ([shared-pole model](shared_pole_model.md), byte model) | before allocating, when a stage and its named concurrent stages exceed the budget |
| static / GN-PPM screening | χ₀ τ-scan scratch `O(N_k·μ²/P)`, unchunked over q | nothing | — |
| restart write | one sharded tile, `max(16·Q·μ²/P, 16·Q·μ·N_G/P)` | stage F | — |

Replicated per-process metadata (the TRS-augmented centroid permutation and
lattice-wrap tables, `O(n_sym·μ)`; the q-folding tables, `O(N_k)`) is
negligible at every size.

The static and GN-PPM screening path (`gw.screening.compute_screening` → χ₀
→ Dyson) has no planner. Its χ₀ scratch cannot be chunked over q, because the
flat-k transform needs the whole k axis on every rank. The one schedulable
term is bounded: a completed W of an earlier screening role is spilled to host
(`common.collectives.spill_to_host`) while a later role runs, and restored
afterwards.

## Route G: every ζ fit

Each ζ fit forms `Z_q(μ, G)` a batch of centroids at a time and solves
`ζ = C⁻¹ Z` once, in G tiles
([ζ fit by μ-batches](zeta_fit_mubatch.md) owns the algorithm and the
per-object byte table). The charge channel is one fit; the three bispinor
current channels are one fit with `n_v = 3` channels at μ_T. Per rank:

```text
fixed    = n_v · C factor (16·⌈Q/P⌉·μ², q-local tier; 16·Q·μ², replicated tier)
         + centroid faces + sphere and cylinder index tables
Ψ        = 16·N_k·N_b·n_s·⌈N_Gψ/P⌉                        ψ(G), device-resident
work(b)  = max over the batch's three stages (GEMM + all-to-all;
           D cylinder; plane groups) of their live sets     b centroids, b = multiple of P;
                                                            Z rows, Z/k-conv output rows and
                                                            the ζ-cylinder accumulator × n_v
store    = n_v · 16·Q·μ·N_G/P                               host or disk, never the device
```

The rule: with `M_f = target − fixed`, ψ(G) is resident and the batch is
`b = (M_f − Ψ)/c_μ` in multiples of `P`, where `c_μ` is the per-centroid slope
of `work`. Candidate plane-group widths are costed by their collectives
(`gw.comm_model`) and per-group launches, and the cheapest wins. The Z store
stays on host when it fits 0.8 of the node's `MemAvailable` over the processes
on the node (minimum over processes); otherwise it is a slab_io scratch
dataset. The finalize streams it in G tiles sized to a quarter of the target.

- **`GATE zeta-mubatch-capacity`**: the smallest configuration (ψ(G)
  resident, `b = P`, one plane per group) exceeds the target. ψ(G) streaming
  is not implemented. Fix: more ranks or more memory per device.

The receipt (`ISDF μ-batch plan`) prints the route, batch, collectives per
batch against the minimum efficient payload, the Z-store placement, every
term in `G_tile` units, and the HWM estimate.

## Vq G panels and q-tiles

`V_q[μ, ν] = Σ_G conj ζ̃_q(μ, G) · v_q(G) · ζ̃_q(ν, G)`. ζ̃ stays
`P(('x','y'), None)`; each G step all-gathers one `(μ/p_x, g)` panel over
`'y'` and one `(μ/p_y, g)` panel over `'x'` inside a `shard_map`, so no rank
holds a whole `(μ, N_G)` face. One kernel launch contracts a q-tile, a scan
over q around the G scan. `gw.v_q_g_flat.vq_tile_bytes` owns the terms (μ
extents mesh-padded; `[+μ_R]` only for an off-diagonal tile):

```text
resident = 16·Q·μ_L·μ_R/P + 16·Q·μ_L/p_x + 16·Q·μ_L·n_sub/P    V_acc, g0, one-leg columns
per_q    = 16·(μ_L [+ μ_R])·N_G/P + 16·N_G                    ζ rows + replicated v row
work     = 16·(μ_L + μ_R [+ μ_R])·N_G/P                       sliced faces, R permute
         + 2·16·μ_L·μ_R/P                                     V_q carry
         + 16·g·(2·μ_L/p_x + μ_R/p_y)                         L, L·v, R panels
peak     = resident + work + q_tile·per_q
host     = 16·(μ_L [+ μ_R])·N_G/P per q                       phdf5 read staging
```

`_plan_vq_tiles` picks the panel width `g` (`vq_g_chunk_size` if positive,
otherwise the largest width ≤ 4096 whose panel all-gather fits
`LORRAX_COLLECTIVE_CHUNK_MB` and whose panels take at most half of what one q
leaves), then `q_tile` as every q that fits both the device budget and the
host staging budget, balanced across tiles. The budget is 0.9 of live
available device memory, agreed as the minimum across processes because the
tile count fixes the collective reads every rank issues. `GATE vq_tile_budget`
refuses when `resident + work + per_q` alone exceeds it.

For the charge channel, `ZetaG.contract_v` accumulates `V_q` tile by tile as
it forms ζ from the Z store, keeping ζ only at the columns the head consumers
name.

`V_q` is then unfolded to the full zone in memory (`16·N_k·μ²/P`), so the IBZ
reduction shrinks the ζ store and the fit's accumulator by `N_k/Q`, not the
`V_q` held by Σ.

## What the closed forms cannot see

### cuFFT plan scratch

XLA's FFT thunk takes the cuFFT plan workspace from a runtime scratch
allocator, outside buffer assignment, so `compiled.memory_analysis()` and
`jax.live_arrays()` both miss it. It is shape-dependent: at a mixed-radix
`(75, 75, 200)` grid the scratch equals the data, while at `(60, 60, 200)`,
`64³` and `32³` it is zero. The spatial FFT terms are therefore measured, not
assumed:

```text
gflat_memory_model._fft_box_bytes
  → common.fft_helpers.query_fft_peak_bytes     compiles the production helper at the exact kind and norm
    → runtime.aot_memory.aot_kernel_peak_bytes
        compiled.memory_analysis()               → compiled_peak
        cufftMakePlanMany on jaxlib's libcufft   → cufft_scratch
        total = compiled_peak + cufft_scratch
```

On XLA:CPU the scratch is exactly zero, decided from the platform. Each
weaker path announces itself once: a failed probe compile gives a 3× data
bound, no real `Mesh` gives the analytic 4× box bound below, and an
unavailable libcufft gives `cufft_scratch = 0` with `cufft_measured = False`.

### FFT peak memory

The analytic fallback prices one ψ(G)→ψ(r) transform at four copies of the
shard `(N_k, B_b/P, n_s, N_r)`: the input, the output and two staging buffers
of the three 1-D passes. The count is exact for shards above ~0.3 GB and runs
10–15 % low below that, where plan and phase-table overheads show.

### Native handlers

The nvidia-mathdx k-convolution kernels allocate no device workspace beyond
shared memory ([FFI layer](ffi_layout.md#k-convolution-router-and-the-mathdx-family)).
On CPU the host `gw_conv` handler keeps a reused host arena of
`16·N_k·m_x·m_y` bytes and per-thread compact chunks, outside XLA.

## Measured corrections behind the G-flat terms

Each term below exists because a run failed without it; do not remove one
without re-measuring.

1. **`loader_tables`, a P-independent floor.** The WFN loader keeps two
   replicated per-k arrays for its lifetime: an int32 index table and the
   τ-phase row `(N_k, N_Gψ)`. The planner prices them at
   `4·N_k·N_r + 16·N_k·N_Gψ` bytes, whose first term is the dense-box size
   and bounds the `(N_k, N_Gψ)` sphere index from above. They are in the
   floor because adding ranks never shrinks them.
2. **Stage C's gathered ψ(r) slab carries two mesh divisions** (both
   layouts). Each rank computes its `1/P` band block over the full r-chunk,
   then runs `all_to_all('y')` (split r, concatenate bands) and
   `all_gather('x')` (bands). Each rank holds `(n_k, band_chunk, n_s,
   r_chunk/p_y)` (all bands, its r block), twice to cover the short-final-chunk
   compaction, plus its own `band_chunk/P` bands over the full r-chunk, the
   all-to-all source (`n_k` the parents on the parent route). The planner prices
   both divisions, so no r-tile width is admitted whose slab is undivided by the
   mesh.
3. **Stage F takes the larger of two tensors.** The restart write carries
   `V_qμν`/`W0_qμν` `(Q, μ, μ)` and the G-flat ζ tensor `(Q, μ, N_G)`. SlabIO
   writes per-rank hyperslabs, so each costs one sharded tile, and the ζ
   tensor binds whenever `N_G > μ`.

**Pair-density slots.** The Stage-C arena is a BufferAssignment fact, not
shape algebra; do not lower it by inspecting the kernel.

- **Face layout** (`_face_pair_density_slots`): 4 rank-3 equivalents for the
  identity-vertex executable, namely the old and new `Z_R` and the two
  k-IFFT outputs. The current-vertex executable gets `n_s²`, because XLA
  places its scalar spin-pair loop as one open-spin
  `(N_k, n_s, n_s, μ/p_x, r/p_y)` arena.
- **Axis layout** (`_pair_density_slots`): 3 rank-5 buffers on GPU XLA (two
  pair carries and one scratch slot), 4 on CPU XLA.

Re-read the count after any Stage-C kernel change, as the number of distinct
preallocated-temp slots holding a pair-shaped value:

```bash
XLA_FLAGS="--xla_dump_to=./hlo --xla_dump_hlo_pass_re=memory-usage-report" \
  python -m gw.gw_jax -i cohsex.in
ls ./hlo/module_*.memory-usage-report.txt
```

## Communication cost model

At a fixed memory workspace a plan uses as few communication steps as it can.
`gw/comm_model.py` prices one collective that moves `V` bytes per rank among
`n_peers` other ranks as

```text
T = α₀ + α_peer · n_peers + V / β
```

with `V` the all-to-all input or the all-gather output. The model has no
topology term and no intra-/inter-node distinction: LORRAX stays portable
across machines (owner ruling). The constants are per-machine data in
`gw.comm_model.MACHINES`, measured with `tools/comm_model_bench.py`. One
constant set prices every collective kind; it over-prices all-gather.

A planner uses the model through four rules:

1. **Minimum efficient payload.** Every collective is at or above
   `min_efficient_payload(n_peers) = 4·β·(α₀ + α_peer·n_peers)`, the payload
   at which latency is 20 % of the call.
2. **Few calls.** Moving `V` bytes through an `M_buf` buffer takes
   `split_calls(V, M_buf, n_peers)` calls. If the per-call payload falls
   below rule 1, the buffer is too small; do not add calls. Aim for at most
   ~10–20 collectives per batch.
3. **One executable per batch.** A batch's collectives are steps of one
   `lax.scan` inside one jit; a Python-dispatched call costs a host round
   trip per step.
4. **Double buffering.** Overlap step `i+1`'s communication with step `i`'s
   compute: `overlapped_time = max(T_comp, T_comm) + min(T_comp, T_comm)/n_steps`.

`plan_zeta_route_g` prices its loop with `comm_time` and prints collectives
per batch against rule 1. All planners stay single-stage and generic.

## Planning a run

1. **Budget.** Leave `memory_per_device_gb = 0` to detect it, or set it
   below the device (for example 56–72 on an 80 GB A100).
2. **Mesh.** Use a square mesh. Every chunked term and the default
   face-layout centroid copies fall as `1/P`.
3. **Read the receipts** in `gwjax.out`: `ISDF μ-batch plan` for each ζ
   fit (the charge channel, and the current channels with `channels = 3`).
   The `ISDF memory model` A–F receipt prices only dict fields the fits
   still read and is marked not binding. Each names its binder.
4. **On a refusal**, add ranks or memory per device: `GATE
   zeta-mubatch-capacity` names ψ(G) and the smallest batch. For V_q, more
   ranks, fewer centroids, or a smaller ζ sphere; `vq_g_chunk_size` shrinks
   only the panel workspace.
5. **Compare with the run.** Define `γ = runtime peak / planner HWM`; `γ > 1`
   is an under-estimate to investigate. Count Stage-C slots in the HLO
   memory-usage report before changing `_pair_density_slots`, and check the
   log for a `[memory-model]` announcement, which means a term fell back to
   an analytic bound. `tools/profile_gw_xprof.py` captures an XProf trace
   whose modules map onto the stages above.
