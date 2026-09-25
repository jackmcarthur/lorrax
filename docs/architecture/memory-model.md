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

## ψ carriers: faces and band panels

The raw-parent carrier holds two orientations of `ψ_{n k̄ s}(r_μ)` on the
`n_par` parents, stored as faces: bands on one mesh axis, centroids on the
other. A band-complete (axis) copy exists only inside one contraction:

```text
M_face = 2·16·n_par·n_s·μ·N_b / P                  resident; μ and bands both tiled
M_axis = 16·n_par·n_s·μ·N_b·(1/p_x + 1/p_y)
       = 2·16·n_par·n_s·μ·N_b / √P                 one band-complete copy, per call
```

With `s = n_par/N_k` (the symmetry reduction) and `r = N_b/μ`, in units of
the Green tile:

```text
M_face / G_tile = 2·s·r / n_s          independent of P
M_axis / G_tile = 2·s·r·√P / n_s       grows as √P
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

(`M_axis` in `G_tile`; `M_face = 0.2·s/n_s` at every P.)

A Green build `G = ψ·diag(f)·ψ†` on faces all-gathers its band panels (`A`
over `y`, `B` over `x`, `distrib_la.panel_matmul`) and multiplies them into
the rank's own tile, so no reduction follows. Every Green-building stage
reserves one full-k Green tile for these transient panels, and the panel
count follows from that reservation: one panel, the complete band extent,
when `M_axis ≤ G_tile`; interleaved band chunks, two live at a time with the
next prefetched, otherwise. The Σ projector `ψ†·O·ψ` reshards the two
projected-band ψ faces to the band-complete orientation for the call and
contracts each `(μ_x, ν_y)` slab locally, so the μ-sized operator never
moves; its transient is `M_axis` at the projected band count.

## Spin-pair streaming

For `n_s > 1` the charge response and the Σ convolutions are elementwise in
the spinor pair, `χ₀ = Σ_ab Gc_ab·conj(Gv_ab)` and
`Σ = Σ_ab ψ*_a (G_ab ⋆ W) ψ_b` with `W` spin-independent, so no stage needs
the `n_s²` Green at once. The typed unfold mixes spinor components, so the
parents move to full k first (the ψ action, `M·N_k/n_par` per rank), and
each `(a, b)` block is one GEMM of the `a` and `b` spinor rows there:

```text
live per block  ≈ 4 · 16·N_k·μ²/P          (Green, transform, product, accumulator)
extra GEMM work = N_k/n_par                full-k blocks instead of parent Greens
```

The stream applies the same one-tile panel rule once per call: when the two
full-k band-complete copies fit one Green tile it gathers them once and the
`2·n_s²` block GEMMs are local; otherwise every block GEMM gathers its own
band panels (`gw.greens_function_kernel.pair_stream_layout`).

The step-occupation χ₀ with the charge vertex, Σ_x and the Coulomb hole
always stream for `n_s > 1`. Their floor drops from the whole-spin
`3·G_tile` (χ₀: Gv, Gc and the unfold transient) or `2·G_tile` (Σ_k and its
convolution transient) to a few `G_tile/n_s²` blocks plus the full-k ψ, at
`N_k/n_par` times the parent-Green GEMM work. Σ_c(τ) streams only when its
whole-spin live set, `2·G_tile`, exceeds the target
(`gw.greens_function_kernel.spin_pairs_needed`); otherwise it keeps the
parent Green and the fused unfold convolution. Fermi-Dirac χ₀, the
vertex-pair currents and Σ^B have no stream and hold the whole-spin set.

## Stage inventory

| stage | resident per rank (leading terms) | priced by | refuses |
|---|---|---|---|
| ψ(G) and centroid faces | charge fit: conj ψ(G) on the rank's G slots, `16·N_k·N_b·n_s·N_Gψ/P`. Centroid carrier `ψ(r_μ)`: `M_face` resident; each band contraction adds its transient panels, at most one `G_tile` ([§ ψ carriers](#ψ-carriers-faces-and-band-panels)) | `plan_zeta_route_g` | with the fit |
| ζ fit, charge channel (route G) | C factor, one μ-batch working set, the ψ(G) slice; the Z store `16·Q·μ·N_G/P` lives on host or disk | `plan_zeta_route_g` ([§ route G](#route-g-every-ζ-fit)) | `GATE zeta-mubatch-capacity` |
| ζ fit, current channels (bispinor) | the same, at μ_T, with three factors, accumulators and Z stores | `plan_zeta_route_g(n_vertex=3)` ([§ route G](#route-g-every-ζ-fit)) | `GATE zeta-mubatch-capacity` |
| V_q | `V_acc` `16·Q·μ_L·μ_R/P`, one q-tile of ζ rows, G panels | `vq_tile_bytes` ([§ V_q](#vq-g-panels-and-q-tiles)) | `GATE vq_tile_budget` |
| V_q unfold | `16·N_k·μ²/P`, sharded `P(None,'x','y')` | — | — |
| shared-pole screening and Σ | response-bank faces, pencils, eigh workspace, then G and W tiles | the capacity ledger ([shared-pole model](shared_pole_model.md), byte model) | before allocating, when a stage and its named concurrent stages exceed the budget |
| static / GN-PPM screening | χ₀ τ-scan: `≈3·G_tile` at `n_s = 1`; for `n_s > 1` the spin pairs stream ([§ spin pairs](#spin-pair-streaming)), four `(a,b)` blocks `16·N_k·μ²/P` (Gv, Gc and their transforms) plus the parents unfolded to full k (`M_face·N_k/n_par` or `M_axis·N_k/n_par`) and the `16·N_k·μ²/P` accumulator; unchunked over q | nothing | — |
| restart write | one sharded tile, `max(16·Q·μ²/P, 16·Q·μ·N_G/P)` (SlabIO writes per-rank hyperslabs) | — | — |

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

### Native handlers

The nvidia-mathdx k-convolution kernels allocate no device workspace beyond
shared memory ([FFI layer](ffi_layout.md#k-convolution-router-and-the-mathdx-family)).
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
4. **On a refusal**, add ranks or memory per device: `GATE
   zeta-mubatch-capacity` names ψ(G) and the smallest batch. For V_q, more
   ranks, fewer centroids, or a smaller ζ sphere; `vq_g_chunk_size` shrinks
   only the panel workspace.
5. **Compare with the run.** Define `γ = runtime peak / planner HWM`; `γ > 1`
   is an under-estimate to investigate. `tools/profile_gw_xprof.py` captures
   an XProf trace whose modules map onto the stages above.
