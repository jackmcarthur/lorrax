# Memory model

A LORRAX calculation is feasible on `P` ranks exactly when every stage's
per-rank live set fits the device budget. Every large object is tiled over
the whole `x × y` mesh, so its per-rank bytes fall as `1/P`; nothing large is
materialised on fewer than all `P` ranks. This page states, stage by stage,
what is resident per rank, how it scales, and which planner prices it.
No planner refuses on a price (owner 2026-10-01): a stage over its budget
raises one `RuntimeWarning: memory over budget at <stage>` (needed, budget, by
how much; `common.gpu_utils.warn_over_budget`; gwjax.out lists it under
WARNINGS), runs at its smallest size, and OOMs if the device truly lacks the
room.

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

Every planner reads this one number, `common.gpu_utils.device_budget_bytes()`,
which the config sets when the deck resolves. kmeans, htransform, bse and
exciton_bands have no deck key; their planners read the same owner, whose
default is the collective auto-detection above, recorded on the first call.
No planner reads `bytes_limit`, free memory or the card total.

**Streamed loops take a fixed tile, not the budget.** A loop that streams over
k, q, bands, centroids, samples or rows takes the most units whose per-rank
scaling bytes fit `runtime.tiles.TILE_BYTES` (1 GiB; `tile_units`). The tile
comes from the loop's shapes alone, so every rank computes it without a
collective, and no result depends on how much memory a run was given. Two
sizes still follow the budget, through a ledger: the shared-pole response
sample group (`response_bank.response_group_size`, the capacity ledger; a
larger group buys more than 10 % per map) and the Galerkin whole-state
planner (`isdf.galerkin`, whose capacity also bounds its resident rows).

**The χ bank streams instead of splitting into sample groups.** When every
sample's χ carry does not fit one group on the devices and the stream runs on
the row-pass engine (`gw.subtile_stream`; charge and four-current), the
stream runs once with every sample, one program per segment (a row pass, or
one family pair's row pass), and each finished segment goes to SlabIO's
per-rank streamed tier (`file_io.slab_io.StreamedBank`;
`response_bank.response_bank_residence`, the shared-pole bank's device /
host / file rule): host memory when it takes at most half the host budget,
else one O_DIRECT file per rank. The devices hold two segment carries, then
three sample reads; the host stages at most four 64 MiB pinned pieces per
rank. Measured on one node at the P64-local volume (75.6 GB per rank,
4-stripe files): 30 GB/s write, 25 GB/s read per node, every read
digest-checked.

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

## ψ carriers: faces and band panels {#psi-carriers}

The raw-parent carrier holds two orientations of `ψ_{n k̄ s}(r_μ)` on the
`n_par` parents, stored as faces: bands on one mesh axis, centroids on the
other. A Green build never holds a band-complete copy; the Σ projector holds
one for the length of its call:

```text
M_face  = 2·16·n_par·n_s·μ·N_b / P                      resident; μ and bands both tiled
M_axis  = 16·n_par·n_s·μ·N_b·(1/p_x + 1/p_y)
        = 2·16·n_par·n_s·μ·N_b / √P                     one band-complete copy: the Σ projector
M_panel = 2·16·n_par·n_s·μ·(N_b/p_x)·(1/p_x + 1/p_y)    one Green build's two live SUMMA panels
```

With `s = n_par/N_k` (the symmetry reduction) and `r = N_b/μ`, in units of
the Green tile:

```text
M_face / G_tile  = 2·s·r / n_s          independent of P
M_axis / G_tile  = 2·s·r·√P / n_s       grows as √P
M_panel / G_tile = 4·s·r / n_s          independent of P
```

A resident axis copy would overtake one Green tile at `√P = n_s/(2·s·r)`
and then spend the GW feasibility floor (`4·G_tile`) on wavefunctions. For
`μ = 10·N_b` (`r = 0.1`):

| P | n_s=1, s=1 | n_s=2, s=1 | n_s=4, s=1 | n_s=2, s=1/6 |
|---|---|---|---|---|
| 16 | 0.8 | 0.4 | 0.2 | 0.07 |
| 64 | 1.6 | 0.8 | 0.4 | 0.13 |
| 256 | 3.2 | 1.6 | 0.8 | 0.27 |
| 1024 | 6.4 | 3.2 | 1.6 | 0.53 |

(`M_axis` in `G_tile`; `M_face = 0.2·s/n_s` and `M_panel = 0.4·s/n_s` at every P.)

A Green build `G = ψ·diag(f)·ψ†` on faces is a batched 2-D SUMMA
(`distrib_la.panel_matmul`): band panels of at most `N_b/p` columns are
all-gathered (`A` over `y`, `B` over `x`), every k in each exchange, and
multiplied into the rank's own tile, so no reduction follows and no rank holds
a band-complete panel. Two panels are live, the next prefetched; every
Green-building stage reserves one full-k Green tile, which bounds them.
The Σ projector `ψ†·O·ψ` reshards the two
projected-band ψ faces to the band-complete orientation for the call and
contracts each `(μ_x, ν_y)` slab locally, so the μ-sized operator never
moves; its transient is `M_axis` at the projected band count.

## The Green-side stages

χ₀ and Σ(τ) read the parent Greens `T_p = 16·n_par·n_s²·μ²/P` through the
fused k-convolution, which unfolds them on its load, so no full-k Green
exists. A Green of real weights (real node times) reads its antiunitary
partner as `conj(G)` on the load; complex times build a conjugate-face
partner tile (`partner = 1`). Each Green build adds its two live SUMMA
panels `M_panel` ([§ ψ carriers](#psi-carriers)). New bytes per
rank beside what is live:

```text
χ₀ node   = 2·(1 + partner)·(T_p + M_panel) + 16·n_out·N_k·μ²/P + S_11   nothing chunks
Σ(τ) row  = 16·[(2 + partner)·n_par·n_s²·ν + N_k·ν + 2·n_q,irr·ν + n_par·n_s·(N_b + 2·N_bΣ)]
```

`S_11` is mode 11's split-arm scratch, at most one `T_p`
(`greens_function_kernel.chi0_door_scratch`); it is 0 on the single pass.

The q-wedge Σ(τ) kernel (`ppm_tau_kernel._sigma_subtile_kernel`) runs each
rank's tile in row passes of whole centroid orbits (`gw.subtile_stream`):
`Σ(τ) row` is one local μ row's live set (`ν = μ/P_y`: the parent Green and
its partner, mode 7's output, `W_prep`, the pass's rows of `W` and its
partner on the q parents, the pass's ψ rows), and a pass holds the most rows
whose bytes fit the fixed tile. ψ is band-complete once per Σ call,
`16·n_par·n_s·(N_b + N_bΣ)·(μ/P_x + μ/P_y)`. No full-tile Green or `W_prep`
exists. `sigma_spin_block` (the output spin block `d`, a divisor of `n_s`,
whose stored x block fits the fixed tile) still sizes the full-zone Σ kernels
and the static (COHSEX) and PPM spatial kernels.
`price_chi0_node` only prices the χ₀ node: a band chunk of Gv would still
be a whole `(μ, ν)` tile. On the packed bispinor route the static photon
response (`V_packed`, `W_packed`, `2·16·Q·(μ + 3μ_T)²/P`) is deleted after
the static Σ channels read it, before Hartree and the τ sweep.

## Stage inventory

| stage | resident per rank (leading terms) | priced by | over budget |
|---|---|---|---|
| ψ(G) and centroid faces | charge fit: conj ψ(G) on the rank's G slots, `16·N_k·N_b·n_s·N_Gψ/P`. Centroid carrier `ψ(r_μ)`: `M_face` resident; each band contraction adds its transient panels, at most one `G_tile` ([§ ψ carriers](#psi-carriers)) | `plan_zeta_route_g` | with the fit |
| ζ fit, charge channel (route G) | C factor, one μ-batch working set, the ψ(G) slice; the Z store `16·Q·μ·N_G/P` lives on host or disk | `plan_zeta_route_g` ([§ route G](#route-g-zeta-fit)) | warns; smallest μ-batch |
| ζ fit, current channels (bispinor) | the same, at μ_T, with three factors, accumulators and Z stores | `plan_zeta_route_g(n_vertex=3)` ([§ route G](#route-g-zeta-fit)) | warns; smallest μ-batch |
| V_q | `V_acc` `16·Q·μ_L·μ_R/P`, one q-tile of ζ rows, G panels | `vq_tile_bytes` ([§ V_q](#vq-g-panels-and-q-tiles)) | warns; one q |
| V_q unfold | `16·N_k·μ²/P`, sharded `P(None,'x','y')` | — | — |
| shared-pole screening and Σ | response-bank faces, pencils, eigh workspace, then G and W tiles | the capacity ledger ([shared-pole model](shared_pole_model.md), byte model) | warns and admits, when a stage and its named concurrent stages exceed the budget |
| static / GN-PPM screening | the χ₀ node ([§ Green-side](#the-green-side-stages)); the GN fit's q block (XLA's compiled footprint of one q) | `price_chi0_node` (a price, no choice); `_gn_ppm_fit_q_block`: the fixed tile, at least one q | warns; one q (only under `LORRAX_PPM_FIT_ARENA_GIB`) |
| Σ(τ) sweep | the resident pole fields, band-complete ψ, then one row pass ([§ Green-side](#the-green-side-stages)) | `subtile_stream.plan_rows`: the rows within the fixed tile | — |
| matrix-element sweep (V_H, four-current) | the step's slabs, and FFT boxes `(2 + 2·n_comp)·n_s·N_r·16` per band of a band-layout operator | `mtxel_sweep.plan_sweep`: bands in the fewest chunks whose boxes fit the fixed tile | — |
| ψ loader off the fit plan (ζ reuse, current faces) | one band tile of G-flat rows, samples and faces | `gflat_memory_model.loader_band_chunk`: the fixed tile, at least the automatic 16 | warns; one scan row |
| moment bank | `(per_q·w + 16)` faces for a batch of `w` q parents | `response_bank.moment_q_width`: the outputs within the fixed tile | the ledger warns |
| sector Σ(τ) sweep (bispinor) | band-complete ψ, then one row pass: the four-spinor parent Green and partner, the door's Σ rows, W(t)'s pass rows | `subtile_stream.plan_rows` (`mpa.sector_sigma.sector_tau_factory`): the rows within the fixed tile | — |
| direct Γ head (bulk metals) | the compiled per-sample footprint × samples per call, split over every rank | `photon_direct_head.direct_gamma_chunk_plan`: the fixed tile, at least 2¹⁰ samples per rank, at most one 2¹⁷ replicate per call | — |
| head wings | `n_ends` gathered endpoint blocks `16·N_k·n_s·block·N_b` | `qsgw_head.head_wing_mu_block`: the fixed tile, at least 16 centroids | — |
| exciton_bands C_q | P_R and its update, one ψ chunk and its Pk | `vq_interp.build_cq_q_chunk`: q rows within the fixed tile | — |
| restart write | one sharded tile, `max(16·Q·μ²/P, 16·Q·μ·N_G/P)` (SlabIO writes per-rank hyperslabs) | — | — |

Replicated per-process metadata (the TRS-augmented centroid permutation and
lattice-wrap tables, `O(n_sym·μ)`; the q-folding tables, `O(N_k)`) is
negligible at every size.

The χ₀ node cannot be chunked over q, because the k-convolution needs the
whole k axis on every rank. A completed W of an earlier screening role is
spilled to host (`common.collectives.spill_to_host`) while a later role runs,
and restored afterwards.

## Route G: every ζ fit {#route-g-zeta-fit}

Each ζ fit forms `Z_q(μ, G)` a batch of centroids at a time and solves
`ζ = C⁻¹ Z` once, in G tiles
([ζ fit by μ-batches](zeta_fit_mubatch.md) owns the algorithm and the
per-object byte table). The charge channel is one fit; the three bispinor
current channels are one fit with `n_v = 3` channels at μ_T. Per rank:

```text
fixed    = n_v · C factor (16·⌈Q/P⌉·μ², whole tiles on their q owners)
         + centroid faces + sphere and cylinder index tables
Ψ        = 16·N_k·N_b·n_s·⌈N_Gψ/P⌉                        ψ(G), device-resident
work(b)  = max over the batch's three stages (GEMM + all-to-all;
           D cylinder; plane groups) of their live sets     b centroids, b = multiple of P;
                                                            plane stages at c_out ≤ b/P rows;
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

- **Over the target**: when the smallest configuration (ψ(G) resident,
  `b = P`, one plane per group) exceeds the target, it runs with one warning
  line. ψ(G) streaming is not implemented. Fix: more ranks or more memory
  per device.

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
leaves), then `q_tile` as every q whose rows fit the fixed tile and the
host staging budget, balanced across tiles. Both come from the shapes, so every
rank issues the same collective reads; the accumulators are priced, not capped.
When `resident + work + per_q` alone exceeds an explicit `budget_bytes`, or
one q's host staging does not fit, the planner warns and runs one q per tile.

For the charge channel, `ZetaG.contract_v` accumulates `V_q` tile by tile as
it forms ζ from the Z store, keeping ζ only at the columns the head consumers
name.

`V_q` is then unfolded to the full zone in memory (`16·N_k·μ²/P`), so the IBZ
reduction shrinks the ζ store and the fit's accumulator by `N_k/Q`, not the
`V_q` held by Σ.

## What the closed forms cannot see

### The compiled check

A planner's closed form chooses its chunk; the chosen executable is compiled
anyway, and `runtime.aot_memory.check_chunk` reads it before it runs. The
stage is priced as `fixed + chunk·per_unit` new bytes per rank beside what is
live. The executable's figure is `temp + outputs − alias` plus cuFFT plan
scratch (`compiled_new_bytes`), plus what the stage holds beside it that buffer
assignment cannot see: a donated carry the caller allocated, a native
handler's run-time scratch, a lookahead copy of the output. At or below the
closed form, or within the room, the chunk runs unchanged. Above both, the
slope is corrected from that one point, `per_unit = (compiled − fixed)/chunk`,
the chunk solved directly, and compiled once more. There is no bisection; a second figure still over the room warns and runs.
The direct stream's check compiles through the dispatch's own executable cache
(`gw.response_bank._compiled`), so the checked executable is the one that runs.
Reading a figure costs 0.01–0.07 s for the direct stream, 0.02–0.12 s for the
Σ τ window per map, and 0.02–0.04 s once for the ζ batch (Fe 4³ and Fe 8³, P4).
When the samples split into groups, the one-group executable that prices the
temporaries is compiled once at map 0 and never runs (Fe 8³: 2.3 s cold,
0.15 s warm).

| stage | chunk | compiled figure available | what it misses (priced elsewhere) |
|---|---|---|---|
| response direct stream (`gw.response_bank`) | samples per group | yes: temporaries | the donated carry (an argument), mode 11's split-arm scratch (`chi0_door_scratch`) |
| Σ τ window (`gw.mpa.sigma.SynthesisTau.admit`) | row pass | yes: the first window executable | the synthesis GEMM's native workspace (added); the passes are not re-solved |
| ζ μ batch (`gw.isdf_fitting`, route G) | centroids per owner | yes: the batch executable, which the loop then runs | the lookahead batch's rows (its output, added) |

The direct stream's group is the one shared-pole size that follows the
budget: it fits the capacity ledger, then the compiled check above. The ledger
does not own the ψ carriers and other residents live when W is built, so leave
headroom between `memory_per_device_gb` and the card. On a device the stream
and the line selection are two phases, so a group costs its carry plus the
larger phase.

### What compiled statistics miss

Measured on Fe 8³ charge, map 0, P4, A100-40GB, `memory_per_device_gb = 36`
(sandbox `runs/DEV/624_memprice_20260929`). Peaks are the pool high-water of
the stage's section; prices are what the stage-memory table prints.

| stage | closed form | compiled | measured peak |
|---|---|---|---|
| response direct stream (group 16 of 22) | 34.01 (ledger only) | 35.58 (temporaries 9.71 = priced; residents 1.96) | 35.58 |
| Σ τ sweep (`d = 2`) | 13.30 | 14.02 | 13.96 |
| ζ μ batch (320 centroids) | 29.38 new, 29.97 total | 22.68 new | 23.73 total |

With the latency-hiding scheduler flag the direct stream's temporaries grow:
the group of 16 compiled 34.08 GB of new bytes against a 34.04 GB room, the
slope was corrected, and the group of 15 ran once recompiled (priced 34.55,
peak 34.55; main ran 16 and peaked 36.04, over the budget).

Two reserves follow from this.

* **In the pool** (`runtime.aot_memory.RUNTIME_RESERVE_BYTES`): what the pool
  draws beyond every priced and compiled byte. The capacity ledger takes it
  off its budget and the direct stream adds it to its price. At (gpu, P4) the
  direct stream's peak was 0.51 MB above its price (the dispatch's small
  arguments), so the table holds 1 MB. P16 and other meshes are not measured:
  they use the largest measured entry of their platform, announced.
* **Outside the pool**: the CUDA context, NCCL communicators, library handles
  and the mathdx modules peaked at 2.65 GB per rank above the pool's
  reservation (`nvidia-smi` 38 983 MiB against 0.89 × 40 960 MiB), and at
  4.7 GB with a cuSOLVERMp context (`runtime.set_default_env`). The budget
  does not include them. They live in the headroom between
  `memory_per_device_gb` and the card, so leave at least 5 GB of it: on a
  40 GB A100 (42.9 GB) a budget of 36 leaves 6.9 GB. Never set the budget to
  the card size.

When a recompiled chunk is still over its room, `check_chunk` prints one
warning line and the stage runs at that chunk; if the room is really
missing, it fails at allocation.

### Native handlers

The nvidia-mathdx k-convolution kernels allocate no device workspace beyond
shared memory, except the split-arm intermediates of modes 8 and 11, which
XLA's scratch allocator grants ([FFI layer](ffi_layout.md#k-convolution-router-and-the-mathdx-family)).
On CPU the host `gw_conv` handler keeps a reused host arena of
`16·N_k·m_x·m_y` bytes and per-thread compact chunks, outside XLA.

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
   fit (the charge channel, and the current channels with `channels = 3`),
   which names its binder, and `Resident ψ` for the GW carriers.
4. **On a `memory over budget` warning** (or an OOM after one), add ranks
   or memory per device. For the ζ fit the smallest batch keeps ψ(G)
   resident; for V_q, more ranks, fewer centroids, or a smaller ζ sphere;
   `vq_g_chunk_size` shrinks only the panel workspace.
5. **Compare with the run.** `gwjax.out` prints MAJOR-STAGE DEVICE AND HOST
   MEMORY: each stage's device peak (max and min over ranks), the planner's
   price and `γ = peak / price`, or "no planner", the host columns, and the
   section that set the device peak. `γ > 1` is an under-estimate to
   investigate. No planner prices host memory: the `rise` column names the
   stage that grew the process.

## The per-stage receipt

`peak_bytes_in_use` never resets, so a stage below an earlier high-water mark
is invisible in it. The CUDA pool behind XLA's `cuda_async` allocator keeps
`CU_MEMPOOL_ATTR_USED_MEM_HIGH`, which a reset sets back to the bytes in use.
`runtime.xla_memory.pool_high_water` reads and resets it at every
`common.timing` section boundary on the main thread (two driver calls), so
every section has its own peak on every rank. A planner records its price
with `common.gpu_utils.record_stage_price(stage, bytes, section=...)`: the
live bytes plus what it plans. What the pool cannot see: NCCL and library
workspaces outside it (2–3 GB per rank on A100), and work dispatched but not
yet allocated at a boundary, which counts in the next section.

Host memory is read at the same boundaries from
`resource.getrusage(RUSAGE_SELF).ru_maxrss`, the process's resident
high-water mark. It never falls and cannot be reset, so the table prints two
numbers per stage, each the max over ranks: `host GB`, the mark at the
stage's last exit, and `rise`, the sum of what the stage's own intervals
added to the mark. A stage that stays below an earlier host peak shows
`rise +0.00`.
