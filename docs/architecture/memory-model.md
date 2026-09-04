# Memory Model and Chunk Size Optimization

The ζ-fitting pipeline allocates tensors in clearly defined stages.  Its
production planner is `gw/gflat_memory_model.py::plan_gflat_chunks`.  It returns
`band_chunk`, `r_chunk`, `n_r_chunks`, `q_chunk`, and
`gflat_chunk_size`, together with the predicted A–F stage peaks, rank floor,
and binding stage.  `prepare_isdf_and_wavefunctions` makes one charge plan and,
for a bispinor fit, one transverse plan because `N_mu^T` and the transverse
band inventory differ from the charge values.  All three transverse channels
use that transverse plan.  The coupled-μ1–3 admission check in `gw_init.py`
adds its simultaneous live set to the one-channel A–F plan; it does not run a
second chunk-size search.

Vq has one additional execution chunk outside that plan:
`vq_g_chunk_size`.  It chunks the G axis of the per-q contraction.
Zero selects `_pick_g_chunk(ngkmax)`, the largest divisor of `ngkmax` not
exceeding 4096.  This is a fixed divisibility heuristic, not a budget-derived
memory-model result.  The deleted r-space Vq path's `mu_chunk_size` and
`q_batch_size` selectors are not accepted by the live dispatcher.

Conventions throughout this doc:

| Concept | Value |
|---|---|
| Element size | complex128 → 16 B (`_mem(…) = 16 · ∏dims / shard`) |
| Mesh axes | `'x'` = μ/centroid, `'y'` = r-chunk; `P = p_x · p_y` |
| Transverse symbols | `M_T = N_mu^T` (padded transverse extent), `Q = N_q^full`, `R = B_r`, `G = N_G`, all before per-rank division |
| Budget detection | `memory_per_device_gb > 0` is used verbatim; zero calls `common.gpu_utils.get_device_memory_gb`, which reserves headroom from the detected free memory |
| Target utilization | planner default: 0.90 scalar, 0.85 spinor (`nspinor=2`), 0.78 bispinor (`nspinor>=4`); positive `ISDF_CHUNK_TARGET_UTILIZATION` overrides it after clamping to `[0.85, 1.0]` |

Typical ranges (from production datasets):

| Symbol | Meaning                         | Typical Range |
|--------|---------------------------------|---------------|
| n_k    | total k-points                  | 1 – 2,000     |
| n_b    | resident left+right band inventory passed to the planner | 20 – 5,000 |
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
| **Centroid load (Peak A)** | fit-loop persistent floor + compiled centroid-load FFT box | `M_A = persistent + _fft_box_bytes(n_k, B_b, n_s, fft_grid, mesh)` |
| **Centroid copies** | two X-sharded + two Y-sharded copies | `M_cent = 2·16·n_k·n_s·μ·n_b·(1/p_x+1/p_y)` |
| **Stage-A FFT fallback** | used only when the compiled query is unavailable; announced as an under-predicting fallback | `4·16·n_k·B_b·n_s·n_r/P` (no cuFFT-plan term) |
| **C_q build + factor (Peak B)** | `P_l`, `P_r`, full-zone `C_q`, selected-Q factor or pseudo-inverse | `M_B = persistent + 16·K·μ²/P + 2·16·n_k·n_s²·μ²/P` |
| **fit + ζ solve (Peak C)** | larger of the pair-density/r-chunk live set and route-specific solve live set | see [§R-Chunk](#r-chunk-b_r) and [§Solve stage](#solve-stage) |
| **accumulate_rchunk_to_gflat (Peak D)** | `gflat_acc`, `zeta_chunk`, two FFT-box-sized slots | `M_D = persistent + 16·n_q^disk·μ·B_r/P + 2·16·cs·n_r` |
| **Vq contraction (Peak E)** | `V_acc`, one or two full IBZ ζ̃ slabs, and their X/Y resharded faces | see [§Vq G-Chunk](#vq-g-chunk) |
| **Restart write (Peak F)** | larger of the sharded V/W0 tile and sharded G-flat ζ tile | `M_F = E_base + max(16·n_q^irr·μ²/P, 16·n_q^disk·μ·n_G/P)` |
| **Vq unfold (IBZ→full)** | `V_full (n_q_full, μ, μ)` sharded over both μ axes + per-q phase | see [§IBZ Cascade Memory](#ibz-cascade-memory) |

The fit-loop persistent floor is the sum of `L_q`, `gflat_acc`, four
single-axis-sharded centroid copies, replicated loader tables, and the
rectangular ψ(r) cache.  That cache is band-flat over all P ranks and prices
the zero-padded final band chunk exactly.  `PsiGStore` holds source ψ(G)
tiles on the host only while that device cache is built; the r-chunk fit reads
the cache and does not reread or re-FFT ψ(G).

Peak E starts from a different, smaller base because `L_q`, `gflat_acc`, and
the ψ(r) fit cache have been released.  It retains one X- and one
Y-sharded centroid copy for the downstream GW path.

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
broadcast, and JIT compilation metadata.

`fft_box_factor = 4.0` counts **box copies** and is the G-flat planner's
*fallback only* (`_FFT_CUFFT_FACTOR`, used when there is no real `Mesh` to
compile against, and announced when used). It does not model the cuFFT plan
workspace at all — that is a separate term, measured rather than assumed; see
"cuFFT plan scratch (live measurement)" below.

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
The live G-flat path routes the r-chunk work through one
scan-inside-shard_map kernel.  Its binding allocations are usually the
rank-5 pair-density accumulators that live across the band-chunk scan.

### Hoisted-ψ(r) cache model

Production performs each local ψ(G)→ψ(r) transform once in
`build_psi_r_cache_sm`.  Its `io_callback × lax.scan × shard_map` builder
returns `(n_bc, n_k, B_b,padded, n_s, n_r)` with the band axis sharded over
all P ranks.  After `block_until_ready`, the ordinary NumPy ψ(G) host tiles
are released.  No FFT box or host callback is live in the outer r-chunk loop.

For each band chunk, `z_q_from_psi_sm` slices the requested full-r slab from
that cache, then performs the load-bearing data movement:

1. `all_to_all('y')` splits r over `p_y` while concatenating y-owned band
   blocks;
2. `all_gather('x')` completes the band axis after r is already sharded;
3. optional band compaction, the L/R masks, and the two pair-density
   contractions update the scan carries.

The source slab is `c128[n_k, B_b/P, n_s, B_r]`; the gathered slab is
`c128[n_k, B_b, n_s, B_r/p_y]`.  The planner conservatively prices two
gathered slots because short-final-chunk compaction can keep a second copy
live, plus one source slab.  Together with the pair-density and `Z_q` terms,
that is exactly `_stage_C_slope`.  `lax.scan(..., unroll=1)` keeps each of
these per-band-chunk transients to one reusable shape family.

The compatibility-only `psi_r_cache=None` route in `z_q_from_psi_sm` retains
the historical per-r-chunk callback/IFFT for narrow tests.  The production
driver never selects it, so the planner does not price it as an execution
alternative.

Two correctness invariants remain easy to violate:

- r must be scattered on y before the final x band gather; a simple r slice
  before gathering mixes band owners and r owners;
- the L/R centroid arrays require symmetric front and back zero-padding,
  because XLA clamps an out-of-bounds dynamic-slice start rather than
  reporting an error.

The per-rank scan accumulator's r-dimension is `B_r/p_y`, not the full
r-chunk.  `out_spec = P(None, 'x', 'y')` requires
`n_zchunk % p_y == 0`; the planner rounds `r_chunk` to a multiple of `p_xy`.

### Solve stage

The solve holds both the sharded `Z_col` stack and its donated output
accumulator:

```
M_rhs_stacks = 2 · 16 · Q · μ · B_r / P
```

The ordinary charge gather term depends on `distributed_zeta_solve`:

- the replicated gather route uses `q_chunk` as its vmapped compute batch and
  adds `q_chunk · 16 · μ²` bytes;
- the `per_q` gather route deliberately gathers one `(1, μ, μ)` factor
  tile plus its Y-gather row, adds `16 · μ² · (1 + 1/p_y)`, and fixes the
  reported `q_chunk` to one;
- fully distributed routes keep the pseudo-inverse or `FactorToken` 2-D
  sharded and bypass the replicated q-batch loop, so their reported `q_chunk`
  is also one.

For explicit `replicated`, the planner chooses

```
q_chunk = clamp(
    floor((target − persistent − M_rhs_stacks) / (16 · μ²)),
    1, Q)
```

`auto` is conservatively priced by the same replicated formula.  The live
resolver may narrow execution to `per_q`; it cannot make the plan optimistic.
Peak C is `persistent + max(C_fit, C_solve)`.

Consequently `q_chunk` is an active compute-batching choice on one route, not
a universal live-memory cap and not a Vq q chunk.

#### Transverse ridge solve routes

For every route, the ridged matrix enters as
`C_q[Q,M_T_x,M_T_y] = P(None,'x','y')`, and each right-hand side enters
and leaves as `Z_q[Q,M_T_x,R_y] = P(None,'x','y')`.

- **Hoisted local JAX LU.** `factor_c_q` factorizes each logical
  `(M_T,M_T)` tile once, q-parallel when selected, and `solve_zeta` gathers a
  replicated or per-q tile for each r-chunk.  This route can have an
  `O(M_T^2)` whole-tile transient on one rank.
- **Local batch-reshard.** Two `all_to_all` operations change the matrix and
  RHS from `P(None,'x','y')` to `P(('x','y'),None,None)`.  Rank `p` solves
  `ceil(Q/P)` complete matrices with local `jnp.linalg.solve`, then the two
  inverse exchanges restore the face layout.  The leading batch is padded to
  `Q_pad = P·ceil(Q/P)`, `M_T` must tile both mesh axes, and `R` must tile
  `p_y`.  This route deliberately refactorizes in every r-chunk; it does not
  call ScaLAPACK or cuSOLVERMp.  Its measured local-operand HBM floor is

  ```
  M_batch = 3 · 16 · ceil(Q/P) · M_T · (M_T + R).
  ```

  The two exchanges prevent useful input-to-output donation, so the factor
  three is required unless BufferAssignment for the shipping executable proves
  a smaller live count.
- **Fully distributed token.** `distrib_la.factor('solve_lu', ...)` performs
  one batched `getrf` per channel and returns an opaque `FactorToken`.
  `distrib_la.solve(token,Z_q)` performs `getrs` for every r-chunk.  ScaLAPACK
  and cuSOLVERMp both implement this split contract.  The token remains
  2-D-sharded: its leading matrix storage is
  `16·Q·M_T^2/P` bytes per rank, plus lower-order pivot metadata, and an
  RHS or output is `16·Q·M_T·R/P`.  No rank receives a complete matrix.

Thus cuSOLVERMp is not a fused, factor-per-r-chunk route.  The only current
factor-per-r-chunk route is the intentional local batch-reshard route.

#### Coupled mu1-3 transverse live set

The coupled scheduler prepares all three channel factors, builds one shared
`Z_q[3,Q,M_T,R]` per r-chunk, and consumes its slices in the fixed order
μ1, μ2, μ3.  Production keeps three separate `Q`-batch solves rather
than flattening them to one `3Q` batch.  This preserves the accepted batched
solver arithmetic.  It does not stack three solve outputs.

Let `K=N_k`, `S=N_s`, `B=N_b^face`, and
`B_pad=P·ceil(B/P)`.  Relative to the accepted one-channel transverse plan,
the coupled schedule adds the following exact modeled HBM bytes per rank:

```
ΔM_coupled = 2·16·Q·M_T·R/P              # two extra completed Z stacks
             + 16·K·S·M_T·B/p_x            # shared full-spin X face
             + 2·16·Q·M_T^2/P              # two extra factors or raw CCTs
             + 2·16·K·B_pad·S·N_r/P        # only when cache_psi_r=true
```

The three G-flat accumulators are not resident on the GPU together.  Each
local `P(None,('x','y'),None)` shard is parked in process-local host RAM and
only the active channel is restored.  The host requirement used by admission
is conservatively

```
M_host,rank = 3·16·Q·M_T·G/P,
M_host,node = ranks_per_node · M_host,rank.
```

If the transverse basis has `M_T = M_charge/3`, this host accumulator is
exactly the size of the one-channel charge accumulator
`16·Q·M_charge·G/P`; it is not asymptotically larger in any fit dimension.

At runtime the stored q extent may be `N_q^disk`; the admission check uses
full `Q`, so it cannot underprice an IBZ-reduced accumulator.  Coupling is
therefore `O(Q·M_T^2/P + Q·M_T·R/P + K·S·M_T·B/p_x)` in device memory
and `O(Q·M_T·G/P)` in host memory.  The coupled delta adds no additional
whole `M_T^2` tile beyond the selected solve route.  The distributed-token
route has no such tile; local batch-reshard deliberately materializes
`ceil(Q/P)` complete matrices per rank.

Admission in `gw_init.py` applies all of these gates before starting the
coupled schedule:

1. Host spill must be at most 35% of detected node RAM.  Unknown node RAM
   fails this gate.
2. The local batch-reshard route is certified only on CUDA A100, square
   `P in {4,16}`, `M_T <= 16384`, and `M_batch <= 0.50·plan.budget_bytes`.
3. Its projected HBM is the larger of
   `base_peak + ΔM_coupled` and
   `persistent + M_batch + ΔM_coupled`; that value must fit
   `plan.budget_bytes·plan.target_utilization`.
4. If the requested route is `auto` and local batch-reshard does not fit,
   the scheduler tries the fully distributed token route using
   `base_peak + ΔM_coupled`.  An explicit batch-reshard request is never
   silently changed to a distributed numerical route.
5. Failure of either capacity test selects the sequential μ1, μ2, μ3
   schedule.  Partial restart reuse also selects sequential execution so only
   missing channels are fitted; full reuse skips the fit.

## Q-Chunk (`B_q`)

`q_chunk` is therefore an active memory-sized compute batch only for the
replicated route.  The plan report prints the route it priced so a value of one
on `per_q` or `distributed` cannot be mistaken for an execution throttle.

## Vq G-Chunk

The live Vq path never builds r-space μ/ν tiles.  It pre-reads the
mesh-sharded G-flat ζ̃ slab, loops synchronously over q, and scans the G axis
inside each per-q contraction:

```
V_q[μ,ν] += conj(ζ̃_q[μ,G_chunk])
             @ (v_q[G_chunk] · ζ̃_q[ν,G_chunk]).T
```

`vq_g_chunk_size > 0` is the requested G width and must divide the
padded `ngkmax`.  Zero selects the largest divisor `<= 4096`.  The heuristic
is fixed in `gw/v_q_g_flat.py::_pick_g_chunk`; `plan_gflat_chunks` neither
chooses nor reports it.

### Per-q kernel allocation (V_q HWM)

The formulas below use the runtime IBZ extents.  The production planner call
currently supplies conservative full-BZ counts for D–F, so its printed Peak E
replaces `n_q^irr` with `n_q^full`:

```
E_base = one centroid copy / p_x + one centroid copy / p_y
peak_E = E_base
       + 16·n_q^irr·μ²/P
       + (1 or 2)·16·n_q^irr·μ·n_G/P
       + 16·μ·n_G/p_x + 16·μ·n_G/p_y
```

The second slab is present for a bispinor off-diagonal tile.  The two
single-axis terms are the per-q ζ faces reshaped for the X/Y contraction.
The G-chunked matrix-multiply scratch is bounded by `vq_g_chunk_size`, but is
not a term in the current closed-form planner.  Lowering that knob can reduce
the compiled per-q kernel workspace; it cannot shrink the full ζ slabs or
`V_acc`, so compiled-memory inspection remains the authority for that tuning.

## IBZ Cascade Memory

The IBZ cascade (commit-pinned 2026-05-11) activates when centroid orbit
closure under the spatial sym ops succeeds.  Effects on memory:

| Quantity | Without IBZ cascade | With IBZ cascade | Savings |
|---|---|---|---|
| `ζ_q.h5` (charge) | `c128[n_q_full, n_rmu, ngkmax]` | `c128[n_q_irr, n_rmu, ngkmax]` | × `n_q_full / n_q_irr` |
| `ζ_q.h5` (Si 4×4×4 80 Ry, ntran=8) | `c128[64, 432, 588]` ≈ 280 MB | `c128[8, 432, 588]` ≈ 35 MB | **× 8.0** |
| `ζ_q.h5` (CrI3 6×6 80 Ry charge) | `c128[36, n_rmu, ~50k]` ≈ 35–80 GB | `c128[6, n_rmu, ~50k]` ≈ 6–13 GB | × 6 |
| `V_q` in memory (any q-set) | `c128[n_q_full, n_rmu, n_rmu] / p_xy` | **same** (unfolded eagerly) | 0 |
| runtime `gflat_acc` in memory | `c128[n_q_full, n_rmu, ngkmax] / p_xy` | `c128[n_q_irr, n_rmu, ngkmax] / p_xy` | × `n_q_full / n_q_irr` |
| `unfold_v_q` umklapp phase | — | `c128[n_q_full, n_rmu]` replicated | new (tiny — ≤10 MB) |

Four points worth keeping straight:

- **The receipt names both q extents.**  Pair-density, ordered-pair completion,
  and C/Z construction retain full-zone `K`.  The selected post-slice extent
  `Q` prices the persistent factor, G-flat accumulator, two solve RHS stacks,
  accumulation, V contraction, and restart write.  The startup receipt prints
  `selected Q / full-zone K` so this distinction is inspectable.

- **Vq still becomes full-BZ in memory.**  `V_q` is unfolded inside
  `compute_V_q_..._to_h5` (`gw/v_q_g_flat.py`) eagerly to its full
  `(n_q_full, n_rmu, n_rmu)` shape sharded `P(None, 'x', 'y')`, then
  written to `isdf_tensors.h5` and passed to Σ on the full BZ — so the
  in-memory V_q footprint is identical with and without the cascade.
- **The pair-density part of Peak C is unchanged.**  It constructs full-zone
  C/Z and the live Z input at the solve seam is still priced at `K`; only the
  post-slice factor and the two solve RHS/output stacks use `Q`.
- **`unfold_v_q` transient is small.**  At peak it's
  `2 · 16 · n_q_full · n_rmu² / P` (the umklapp phase plus a permuted
  `V_at_irr` copy), which on Si 4×4×4 is ~1 MB and on CrI3 6×6 is
  ~30 MB.

The cascade is gated on **centroid orbit closure** under the spatial
sym ops.  Regenerate centroids without `--no-orbit` and ensure
`compute_centroid_sym_perm(..., extend_trs=True)` raises no closure
error to activate it.  When inactive (centroid orbit not closed,
bispinor transverse path, or `LORRAX_FORCE_FULL_BZ=1`),
`write_ibz_only_charge = False` and the runtime also uses
`n_q^disk = n_k_tot`.

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
coefficients in ordinary NumPy host memory during the one-time ψ(r) cache
build:

- Per-rank layout: one contiguous `(n_k, n_b/P, n_s, ngkmax) c128` array;
  `_slice_local_tile_bc` selects one band chunk from it.
- Per-process residency: `n_b · n_k · n_s · ngkmax · 16 / total_procs`
  bytes (band-flat-sharded across all ranks).  At Si 4×4×4 25 Ry / 2
  procs: 0.03 GB/proc.  At CrI3 6×6 80 Ry / 16 procs: ~1.5 GB/proc
  (typical).
- Access pattern: `io_callback(_slice_local_tile_bc, out_sds=...)`
  pulls one bc's per-rank-local slab (`bpd_max · n_s · ngkmax`) into
  device memory inside each scan iter.  Single one-shot push to host
  at populate time; many small pulls during the cache-build scan.  After
  `block_until_ready`, the host store is closed before the r-chunk loop.

This is the **single source** for ψ(G) memory residency in the
post-Round-6 pipeline — the previous `psi_G_device_full` device-side
property has been deleted, and the `gflat_to_rchunk` standalone helper
along with it.  See `feedback_iocallback_for_large_caches`.

## G-Flat Memory Model

`gw/gflat_memory_model.py::plan_gflat_chunks` is the production planner for
`band_chunk`, `r_chunk`, `q_chunk`, and `gflat_chunk_size` on the G-flat
ζ + Vq pipeline.  Six named per-rank HBM peaks are keyed by source-code
location:

| Peak | Stage | Persistent | Transient (per scan iter, aliased) |
|---|---|---|---|
| **A** | `load_centroid_wfns` (pre-loop, once per channel) | centroid output being filled `(n_k, n_s, n_rmu, n_b/P)` | ψ(G)→r FFT box `4·16·n_k·B_b·n_s·n_r / (p_x · p_y)`, replicated `(n_k, n_r)` phase table |
| **B** | `CCT + factor` (pre-loop) | centroids (L+R copies) | open-spin `P_l + P_r (n_k, n_s², μ, μ)`, `C_q (n_q, μ, μ)`, factor or pseudo-inverse `(n_q, μ, μ)` |
| **C** | `fit_one_rchunk` + ζ solve (inside r-chunk loop) | centroids + `L_q` (base) | larger of the pair-density/r-chunk live set and route-specific RHS/output + factor-gather live set |
| **D** | `accumulate_rchunk_to_gflat` (right after each `fit_one_rchunk`) | `gflat_acc (n_q^disk, n_rmu/p_xy, ngkmax)` | `zeta_chunk (n_q^disk, n_rmu/p_xy, B_r)`, per-scan-iter FFT box `cs · n_r · 16 · fft_factor` |
| **E** | G-flat Vq contraction (post-fit) | one X- and one Y-sharded centroid copy | `V_acc`, one/two IBZ ζ̃ slabs, X/Y-resharded ζ̃ faces |
| **F** | restart tensor write | same post-fit centroid base | larger of the sharded V/W0 tile and G-flat ζ tile |

### Peak A — Band-chunked centroid load

`ψ(G) → IFFT → sample at r_μ`.  Runs once per channel (charge + 3
transverse on bispinor).  For planning, the whole fit-loop persistent floor
is charged at this peak, including the completed rectangular ψ(r) cache.
The transient is the compiled ψ(G)→ψ(r) FFT box.

```
peak_A = persistent_total
       + _fft_box_bytes(n_k, B_b, n_s, fft_grid, mesh_xy, P)
```

### Peak B — CCT + factor

Pair density on the full (μ, ν) grid + Cq FFT + factorization.  The
planner charges the same fit-loop persistent floor plus the full pair-density
and Cq transients:

```
peak_B = persistent_total
       + 16·n_q·μ²/P
       + 2·16·n_k·n_s²·μ²/P
```

Under `low_mem_bands=true` this stage's band contraction is a distributed
SUMMA GEMM over the two-face carrier rather than a rank-local einsum over
single-axis ψ — see
[`zeta_fit_face_psi_cct.md`](zeta_fit_face_psi_cct.md) for the staging and
why it reuses `gw.greens_function_kernel`'s GEMM-seam convention.  The
transient shape above is largely unchanged (still one open-spin pair
density per side); the persistent floor changes as `_persistent_bytes`'s
`psi_copies` docstring now states.

### Peak C — fit_one_rchunk

The binding peak on most production runs.  The pair-density phase holds:

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
  time via `_pair_density_slots()` in `gflat_memory_model.py`.
  XLA's BufferAssignment reuses these slots among pair intermediates and
  `Z_q` when their lifetimes do not overlap.

```
peak_C = persistent_total + max(C_slope · B_r, solve_transient)
```

`C_slope` is `_stage_C_slope`: pair-density slots + sharded `Z_q` +
the gathered ψ(r) slabs described above.  `solve_transient` is the
route-specific inventory in [§Solve stage](#solve-stage).

The `pair_density_slots` constant is the **XLA-BufferAssignment-determined**
count of concurrent rank-5 buffers.  Read it from
`module_NNNN.jit__kernel.sm_*.memory-usage-report.txt` as the number of
distinct preallocated-temp slots holding a P-pair-shaped value.  Update
the default returned by `gflat_memory_model._pair_density_slots` if a future
XLA version changes the BufferAssignment.

### Peak D — accumulate_rchunk_to_gflat

Runs after `fit_one_rchunk` returns; its `P_l`/`P_r` are freed by then.
`zeta_chunk` is the only `fit_one_rchunk` output still live.

```
peak_D = persistent_total
       + 16·n_q_disk·μ·B_r/p_xy
       + 16·gflat_chunk_size·n_r · 2
```

`gflat_acc` is already part of `persistent_total`.  It is the G-flat ζ
accumulator (μ-flat sharded across the mesh).  The runtime object can use
`n_q_irr` under the cascade, but the production planner call currently prices
`n_q_disk = Q`, the selected post-slice extent disclosed by the receipt.

### Peaks E and F — post-fit Vq and tensor write

These peaks use `E_base`, not the fit-loop persistent floor.  Peak E is the
full-slab inventory in [§Vq G-Chunk](#vq-g-chunk).  Peak F adds the larger
of the selected-Q sharded V/W0 tensor and G-flat ζ tensor.  SlabIO writes
per-rank hyperslabs; the deleted
all-gather writer is not an alternative modeled here.

### Sample planner output

`GFlatChunkPlan.format()` prints all resolved values (`band_chunk`,
`r_chunk`, `q_chunk`, `gflat_cs`, `P_min`), the budget and persistent floor,
then the six A–F peaks sorted by descending bytes.  The `bottleneck` field
names the largest modeled peak.  These are total bytes per rank; there is no
second hidden heuristic report to reconcile with them.

### Algorithm

`plan_gflat_chunks` is deterministic.  It uses closed-form inversions plus
short discrete ladders for mesh-compatible band chunks and the rank floor:

1. **Compute persistent footprint** (centroids + loader tables + `L_q` +
   `gflat_acc`).
   Add the full-grid ψ(r) cache, band-flat sharded over all ranks (including
   its uniform final-chunk pad), and validate the resulting floor against the
   budget at every peak.  The cache has no μ axis and is never replicated.
   The pad is the static `lax.scan` output shape, not an accidental estimate:
   a 50-band window at bc16 carries 64 slots (28% overhead), measured/priced
   as 7.25 rather than 5.66 GB at P=1 on Si 80 Ry.  A ragged last item would
   require a second cache/slice executable family and is not a trivial memory
   correction.
2. **Pick `band_chunk` first** — primary lever on Peak A and Peak C.
   The shipping no-key value is the owner-selected 16.  Its pre-AOT P=4 Si
   80 Ry premise (33 ms steady z_q at bc16 versus 46 ms full-window) was
   refuted by the final-tree AOT A/B (31 ms versus 21 ms, respectively), so
   the default is policy rather than a current performance claim.  It is
   passed through `_bump_bc(16)`, so the mesh floor and logical-window cap
   still apply.  An explicit deck value `0` opts into the planner ladder: try
   the full logical ζ-fit window first so the pair GEMM has one K dimension
   and does not read/modify/write its rank-5 carry between band chunks; if it
   does not fit the measured FFT-box + Stage-C guard, fall back through the
   historical power-of-two family.  Any positive deck value remains an
   override.  Every transport tail is zero-masked; the physics window is
   unchanged.
3. **Pick `r_chunk`** — maximize subject to Peak C fitting after
   `band_chunk` is fixed.  Automatic sizing is lower-bounded by `n_rmu` (the
   eventual Σ_μν output occupies `n_rmu² · n_q · 16` bytes, so paying
   less than `n_rmu` work per chunk is wasted iteration overhead).
   Upper-bounded by `n_rtot`, and `n_rtot / B_r ≤ max_chunks = 64`.
   Rounded *down* to a multiple of `p_xy` so the `(μ_X, r_Y)` sharding
   at the solve output divides cleanly.  A positive `r_chunk_size` is an
   explicit override and may select a smaller mesh-compatible extent.
4. **Pick `gflat_chunk_size`** from Peak D headroom, clamp to the live cap
   of 100 rows, floor at 4, and round down to a multiple of 4.
5. **Pick `q_chunk`** for the replicated route from headroom after two
   sharded full-q RHS/output stacks, charging one replicated `(μ,μ)` factor
   per batched q.  `auto` is priced the same way; `per_q` and `distributed`
   report one because they bypass that batch.
6. **Compute A–F peaks + HWM**.  A–D use the fit-loop persistent floor;
   E–F use the smaller post-fit centroid base.  HWM is their maximum.

Positive `band_chunk_size`, `r_chunk_size`, and `gflat_chunk_size` deck
values become overrides.  `q_chunk` has no deck override.  The separate
`vq_g_chunk_size` selector is resolved later by the Vq driver and is not part
of `GFlatChunkPlan`.

### Pair-density slots (`slots`)

Peak C's dominant transient is `slots` concurrent rank-5
`c128[n_k, n_s², n_rmu/p_x, B_r/p_y]` tensors.  XLA's BufferAssignment
fuses lifetimes that do not overlap.  Default `slots = 3`:
`P_l_R_conj`, `P_r_R`, plus one XLA scratch slot.

Do not reduce this to the two conv_kpair input carries by inspection of the
CUDA kernel.  The enclosing scan/custom-call module still owns a third
pair-sized BufferAssignment slot.  Measured 2026-08-18: a route-inferred
two-slot trial on the MoS2 bispinor fixture planned r=72,304 at 23.40 GB, but
the executable requested 31,984,978,688 bytes and OOMed.  Route-aware
accounting therefore needs compiled-module evidence and is not a trivial
subtraction of the old post-pair intermediates.

Re-verify after any kernel change:

```
$ XLA_FLAGS="--xla_dump_to=./hlo --xla_dump_hlo_pass_re=memory-usage-report"
$ uv run python -m gw.gw_jax -i cohsex.in --workdir <run>
$ ls ./hlo/module_*.jit__kernel.sm_*.memory-usage-report.txt
```

Search for the highest-numbered slot holding a `c128[n_k, n_s², ..., ...]`
shape; that's the slot count.  Update
`gflat_memory_model._pair_density_slots` if it changed.  See
`reference_hlo_dump_workflow_lorrax.md` for
the shifter-aware launcher.

## Automatic Sizing Algorithm

Run order in `gw_init.prepare_isdf_and_wavefunctions`:

1. **Gather inputs**: `{n_k, n_b, n_rmu, n_q, fft_grid, ngkmax,
   memory_per_device_gb, mesh}`; build the persistent inventory and search
   for `P_min`.  A requested mesh below that floor is reported as an
   infeasible warning by `gw_init` rather than silently changing the mesh.
2. **One G-flat plan** (`plan_gflat_chunks`) resolves `band_chunk`,
   `r_chunk`, `q_chunk`, and `gflat_chunk_size`, plus the rank floor and
   A–F peaks.
3. **Fit and solve** consume that plan.  The requested ζ-solve route is
   already represented in Peak C and printed beside `q_chunk`.
4. **Vq G chunk** resolves independently: explicit `vq_g_chunk_size`, or
   the largest divisor of `ngkmax` no greater than 4096.
5. **Instrumentation**: the planner returns `peak_estimate_gb`,
   `bottleneck`, per-peak breakdown, and budget, surfaced on rank 0 of `gw.out`.

## Recipe — planning a run for a given budget

To size a fresh system at a target `memory_per_device_gb` (cohsex.in):

1. **Set the budget** in cohsex.in (`memory_per_device_gb`).
   `get_device_memory_gb` returns `0.9 · bytes_available`; choose
   `28.0` for a 40 GB A100, `56.0`–`72.0` for an 80 GB hbm80g A100,
   `6.0` for an 8 GB local GPU.  The planner default is `ns²`-aware
   (`common.gpu_utils.bfc_fragmentation_target_utilization`): 0.90 scalar,
   0.85 spinor `ns=2`, 0.78 bispinor
   `ns=4` — larger `ns²` means a bigger single contiguous Stage-C arena,
   which needs more headroom against BFC fragmentation.
2. **Pick the mesh** `p_x × p_y = total_GPUs`.  Square-ish meshes
   (e.g. 4×4 on 16 GPUs) minimise both Peak A and Peak C since they
   sit on `p_xy`-sharded buffers.  If `n_rmu_padded % p_xy ≠ 0` the
   centroid loader pads up.
3. **Inspect the centroid footprint first.**
   `M_cent = 2·16·n_k·n_s·μ·n_b·(1/p_x + 1/p_y)` is
   non-chunkable; if it exceeds the budget the run is physically
   infeasible.  Remedies: shrink the sigma window, shrink `n_rmu`, or
   grow the mesh.
4. **Run and read the `ISDF memory model` block** of `gw.out`; it names the
   binding peak, `P_min`, target utilization, and total budget separately.
5. **If HWM exceeds the budget or `P_min > P`**, treat the plan as
   infeasible.  `gw_init` warns for the rank-floor case; it does not retry on
   a different mesh.  Knobs by peak:
   - **A_centroid** — lower `B_b` via `band_chunk_size`.
   - **B_CCT_chol** — drop `n_rmu` or rebuild centroids with a smaller
     orbit.
   - **C_fit_one_rchunk** — grow mesh, shorten sigma window, or
     lower `r_chunk_size`.  If `slots > 3` in a fresh HLO dump,
     update the planner constant.
   - **D_accumulate** — lower `gflat_chunk_size`; the automatic
     value is at most 100 and a multiple of four.
   - **E_v_q** — the full ζ slab or `V_acc` usually binds.  More ranks,
     fewer centroids, or an IBZ-reduced ζ file shrink those terms.
     Lowering `vq_g_chunk_size` only shrinks the unmodeled inner
     contraction workspace, not the Peak-E slab inventory.
   - **F_tensor_write** — more ranks, fewer centroids, or a smaller G sphere.
6. **If HWM leaves substantial headroom**, first inspect which defaults or
   explicit overrides bound `band_chunk` and `r_chunk`.  Increase an
   execution chunk only after compiling that exact shape; do not invent a
   larger physical memory budget to force a choice.
7. **Compare HWM to compiled and runtime peaks.**  Define
   `γ = runtime_peak / planner_HWM`; `γ > 1.0` is an under-estimate and
   must be investigated.  Count binding-peak slots in the HLO
   memory-usage-report before changing `_pair_density_slots`.  Do **not**
   reach for `fft_box_factor` first:
   since 2026-07-30 it is only the fallback bound, and a run on a real mesh
   does not use it (the FFT box is compiled and measured, cuFFT plan
   workspace included).  Check the log for a `[memory-model]` announcement —
   if one is there, the term really did fall back and the reason is printed.

   **KNOWN OPEN (2026-07-30):** the ladder campaign measured `γ > 1` —
   the planner under-predicting true peak by **1.5–14 %** — independently of
   the FFT term.  A GB-vs-GiB unit error had masked this and the earlier
   "model needs no fixes" verdict was withdrawn.  The FFT-box half of that
   gap is now measured rather than assumed; the remainder is **not
   diagnosed**.  Treat `HWM` as an estimate, not a bound, until it is.
8. **Escape hatches**, in order:
   - `LORRAX_FORCE_FULL_BZ=1` — disables the IBZ cascade (debugging).
   - Grow the mesh.  All chunked terms shrink as `1/p_xy`; `M_cent`
     shrinks as `1/p_x + 1/p_y`.

## XProf Workflow

Recommended stack (June 2025): XProf + TensorBoard memory viewer.

1. Capture:
   `uv run python tools/profile_gw_xprof.py -i <input.in> --workdir <run_dir> --logdir ./profiles/xprof --name <tag>`
2. Open UI: `uv run xprof ./profiles/xprof`
3. Match modules to the `zeta_fit.build_psi_r_cache`, r-chunk pair-density,
   ζ-solve, and Vq timing sections.  The cache builder owns the ψ FFT; the
   r-chunk pair-density module must not contain it.

Older Round-8 traces named the pair module `jit__z_q_from_psi_sm` and included
a ~5 GiB scan-aliased FFT box.  They predate the ψ(r)-cache hoist and are
historical evidence only; adding that FFT box to current Peak C double-counts
work now owned by the one-time cache-builder module.

## Model Corrections

The current production path hoists ψ(G)→ψ(r) out of the r-chunk loop.  Peak C
therefore contains cache slices, band/r collectives, pair-density carries,
and the solve — not an FFT box.  The two visible rank-5 carries
(`P_l_acc`, `P_r_acc`) plus XLA scratch remain the binding family;
`pair_density_slots` (3 on GPU XLA, 4 on CPU XLA) captures the measured
BufferAssignment count.  The same conservative bound remains active around a
`conv_kpair` custom call.

Pre-Round-6 reference (legacy
`tests/profiles/xprof/cohsex_prod-20260303-112900/...` — the blobs are tracked
under `tests/`, not at the repo root this line used to name):
`jit__compute_ZCT_LR` `peakHeapMib ~= 1794`, `jit__solve_all_q`
`~= 1485`, `jit__compute_P_traced` `~= 480-493`.

The former `compute_optimal_chunks` handoff is gone.  Its `q_gather` and
`k_chunk` outputs are not live planner fields, and its r-space Vq μ chunk is
not a supported execution path.  Current corrections belong directly in
`plan_gflat_chunks`: the padded ψ(r) cache, Stage-C gathered-ψ divisions,
measured FFT workspace, post-fit Peak E, and sharded restart-write Peak F are
all charged there once.

## cuFFT plan scratch (live measurement)

The closed-form shape algebra above cannot see the cuFFT plan *workspace*. It is
not merely "hard to see": the plan workspace is **not in XLA's buffer assignment
at all** — jaxlib's `FftThunk` takes it from a runtime scratch allocator at
execution time — so `compiled.memory_analysis()` and `jax.live_arrays()` both
report a number that excludes it. Any peak built from `memory_analysis()` alone
is a systematic *low bound* for a kernel containing an FFT.

**The one honest path (as of 2026-07-30):**

```
gw/gflat_memory_model.py::_fft_box_bytes          # Stage-A FFT-box term
  -> common/fft_helpers.py::query_fft_peak_bytes  # compile exact kind + norm
     -> runtime/aot_memory.py::aot_kernel_peak_bytes
          compiled.memory_analysis()      -> compiled_peak
          parse fft ops from as_text()    -> FftSpec per op
          cufftMakePlanMany on jaxlib's own libcufft -> cufft_scratch
        total = compiled_peak + cufft_scratch
```

`query_fft_peak_bytes` requires the caller to state the transform kind and
normalization, then compiles **the same helper production runs**. Stage A asks
for `make_sharded_ifftn_3d(..., norm='ortho')`, exactly matching the WFN
spatial loader: a `shard_map`'d device-local rank-3 `jnp.fft.ifftn`, one cuFFT
plan per rank. Before 2026-07-30 the query compiled the per-axis
`custom_partitioning` form instead — three rank-1 plans that no production path
ever builds — and read only `memory_analysis()`, while its own docstring
promised the result "includes cuFFT scratch". Both defects are fixed; the
per-axis form has been deleted rather than left as a modelling-only path. The
query cache includes kind and norm because the normalization scale can change
XLA's live buffers even when cuFFT happens to choose the same plan.

The query itself was calibrated in `scripts/profiling/aot_cufft_sanity.py`
using the `(75, 75, 200)` CrI3 grid and the batched FFT shape from the old
r-space Vq driver.  That driver is deleted; these numbers validate the cuFFT
workspace query, not a live Vq FFT:

| batch rows | `compiled_peak` | observed |
|---|---|---|
| 8 | 40.84 GB | ran fine |
| 12 | 61.22 GB | ran fine |
| 13 | **66.32 GB** | **cuFFT plan creation FAILED** — so the true peak crossed 80 GB, i.e. >13.7 GB of it was invisible to `memory_analysis()` |
| 18 | 91.0 GB | OOM at runtime |

**Direct measurement of the gap** (job 7882062, Quadro RTX 5000, jax 0.9.1,
`(16, 75, 75, 200)` c128 — a 16-row calibration batch that fits a 16 GB
card):

| quantity | value |
|---|---|
| `memory_analysis()` only — the pre-fix number | 0.576 GB |
| `cufft_scratch` (queried, `cufft_measured=True`) | 0.288 GB |
| `aot_kernel_peak_bytes(...).total` | **0.864 GB** |
| runtime `device.memory_stats()['peak_bytes_in_use']` after executing | **0.864 GB** |

The pre-fix number was **50 % low**; the new total matched the runtime peak to
**0.0 %**. The same job showed the query discriminates rather than returning a
constant: at `(75, 75, 200)` (mixed radix, 75 = 3·5², 200 = 2³·5²) the scratch
is exactly 1× the data, while `(60, 60, 200)`, `(64, 64, 64)` and
`(32, 32, 32)` all report **0** — which independently corroborates the MoS2
observation in `aot_cufft_sanity.py` that "cuFFT scratch is essentially zero on
MoS2". The cliff is shape-dependent, which is exactly why a constant factor
cannot stand in for the query.

**Where the cuFFT term is 0 and that is correct:** XLA:CPU has no cuFFT plans,
so the scratch term there is an exact zero and `cufft_measured` stays `True`.
That decision is made from the **platform**, not the HLO — measured on jax
0.9.1 (job 7882062), XLA:CPU keeps the `fft` op in `compiled.as_text()` exactly
as XLA:GPU does, so "the HLO has an fft op" does *not* imply cuFFT. (An earlier
draft of this fix inferred it from the HLO and made every CPU run print a bogus
low-bound warning.) If a compiled FFT shows **no** fft op at all, that is
parser blindness and is announced on either platform.

XLA:CPU's own Ducc/pocketfft scratch is a different and much smaller quantity;
this path does not model it and does not claim to.

**Fallbacks announce.** Every path that returns a weaker number than the one
advertised prints once, from the rank it happened on, via
`runtime.aot_memory.announce_once`: probe compile failure (→ 3×data analytic
bound), no real `Mesh` (→ the analytic `fft_box_factor = 4.0` box-copy bound),
libcufft unavailable (→ `cufft_scratch = 0` with `cufft_measured = False`).

There is no offline fit / DOE / preset framework: the former
`src/gw/aot_memory_model/` package was removed 2026-07-02 (dead-by-clobber —
`plan_gflat_chunks` always overwrote its chunk picks; see
`reports/memplanner_cleanup_2026-07-02/PLAN.md`).

### NOT modelled: the LORRAX FFI handler arenas (the required flat-k FFT service)

A third invisible allocation exists and this planner does **not** account for
it. The flat-k FFT service is the required default (2026-08-01); its handlers in
`src/ffi/cpp/mklfft` (MKL DFTI descriptors) and `src/ffi/cpp/cufft` (cuFFT
plans + the NVRTC-compiled fused multiply) hold their own workspace *outside
both* XLA's buffer assignment and the `cufftMakePlanMany` query above — the
query sizes the plan XLA's own FFT thunk would build, not a plan the FFI
library builds for itself. The FFI workstream reports **~100 MB/rank**.

That figure is theirs, not measured here, so it is recorded rather than
folded into a term: adding an unverified constant to the model is how the
`fft_box_factor` story started. Two things make it tolerable for now — it is
a per-process constant rather than a shape-dependent term that grows with the
chunk knobs, and
`common.gpu_utils.bfc_fragmentation_target_utilization` already withholds
10–22 % of the budget (0.90 / 0.85 / 0.78 by `ns`), which is 0.8–1.8 GB on
an 8 GB card and far more on production cards. It is still an unmodelled term
and should be closed by an arena-size query on the FFI side, not by a constant
here.

## Measured corrections behind the G-flat terms

Three terms in `gw/gflat_memory_model.py` exist because a run died without
them. The code carries a one-line pointer to this section; the measurements
live here so the planner source stays short. **Do not remove these terms
without re-measuring** — each was added after a specific job failed.

### 1. `loader_tables` — a P-independent floor (`_persistent_bytes`)

The WFN loader stages two REPLICATED per-k arrays once and keeps them for its
lifetime: the sparse-G→FFT-box index `(nk, nx, ny, nz) int32` (staged by
`WfnLoader.box_index_dev`) and the τ-phase row `(nk, ngkmax) c128`
(`_ensure_phdf5_static`). Small — 121 MB at MoS2 12×12 — but **P-independent**:
adding nodes never shrinks it, so it belongs in the floor rather than nowhere.

It is modelled at all because these two arrays used to cost `P ×` their size as
a *transient* on top: JAX's hidden `device_put` → `assert_equal` all-gather,
7.7 GB/rank at P=64 and 17.4 GB projected at P=144. That is what made the
planner read **0.48× of the measured node peak** at 606 centroids / P=64. The
transient was cured in `common.collectives.device_put_process_local`; the
residency term stays so the floor remains honest.

### 2. Stage C's gathered ψ(r) slab — two mesh divisions (`_stage_C_slope`)

`z_q_from_psi_sm` computes each rank's 1/P band block over the FULL r-chunk,
then `all_to_all('y', split r, concat bands)` + `all_gather('x', bands)`. Each
rank therefore holds `(nk, band_chunk, ns, cr/p_y)` — all bands, only its
r-block — plus a smaller slab of its own `band_chunk/p_xy` bands over the full
r-chunk (the all-to-all source, unavoidable: other y-ranks need other r-blocks
of exactly those bands).

Before that, the gather ran over `('x','y')` at the full r-chunk with the
r-slice applied *afterwards*, giving `nk·band_chunk·ns·cr·16` with **no mesh
division on either axis**: 129 GB/rank per copy at MoS2 12×12 (nk=144,
band_chunk=160, ns=2, cr=174960) — 9.4× everything else in this slope combined.
Omitting it from the model is what let the planner pick `r_chunk = n_rtot` and
ask XLA for a single 271 GB allocation (**job 7874236**, RESOURCE_EXHAUSTED).
The all-to-all then removed the `p_y` factor outright: 5.2× on this slope at the
8×10 mesh.

### 3. Stage F — the restart-tensor write takes the LARGER of two tensors

SlabIO writes per-rank hyperslabs, so the live planner charges one sharded
tile. The former h5py allgather backend materialized the whole tensor on every
rank and copied it again to host; that backend and the corresponding
`slab_io_replicates` planner arm were deleted. The historical measurement
below explains why that replicated arm could not remain a production option.

TWO tensors cross this seam and the model must take the larger:

| tensor | shape | family |
|---|---|---|
| `V_qmunu` / `W0_qmunu` | `(n_q_ibz, μ, μ)` | μ² |
| the G-flat ζ tensor | `(n_q_disk, μ, ngkmax)` | μ·ngkmax |

Whenever `ngkmax > μ` (true at every centroid count below ~ngkmax) the G-flat
write is the binder, and the old μ²-only term under-predicted it. MEASURED
(wk_Y probe, 1998 centroids / μ_pad=2048 / ngkmax=8603 / P=64, run dir
`runs/d_1998_P64_rep`): the largest collective in the whole run is one
all-gather of `nq·μ_pad·ngkmax·16 = 144·2048·8603·16 = 40,594,046,976 B`, to
the byte, on top of 60.7 GB already live — while the term reported
`2·144·2048²·16 = 19.33 GB`, i.e. **2.10× too small**. The planner named the
right binder at the wrong size. Corrected term, both copies, at nq=144 /
ngkmax=8603 (MoS2 12×12): μ_pad=288 → 11.42 GB, 640 → 25.37 GB,
2048 → 81.19 GB — each 2× a collective the probe has actually seen in a dump.

## Historical Round-7 faithfulness audit (not current calibration)

This section preserves the 2026-05-17 measurement record.  It predates the
current ψ(r) cache accounting, Peaks E/F, compiled Stage-A FFT query, and
current Stage-C kernel.  Its conclusion that the then-current estimate was a
7–8× upper bound must not be applied to today's `GFlatChunkPlan`; the newer
ladder above instead found a 1.5–14% under-prediction on its measured domain.
For a current decision, compare the current plan with the compiled module and
runtime peak at the exact lowered shapes.

The old planner's `HWM_pred` treated the in-jit transient as an upper bound
assuming no XLA buffer aliasing/donation.  Round 7
(`agent_n_faithfulness_audit.md`) measured the
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

**Trust matrix recorded for that archived tree:**

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
`LORRAX_DEBUG_PRINT=1` shows an unexpected shape in HBM, grep this
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

If `LORRAX_DEBUG_PRINT=1` prints a `live_arrays()` row whose shape you
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
