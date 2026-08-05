# ζ-basis W projection — large basis → small basis

Owner directive 2026-07-29: *"fix/parallelize with good strong scaling the
test code to compute overlaps of a large zeta basis like 10k zeta functions
to a small one representing only a few val and cond bands like 500 zeta
functions and project e.g. W_μlarge,νlarge to W_μsmall,νsmall."*

Workstream dir: `/scratch2/08271/jackmc/lorrax_setup/wk_REL`.
Tree: `/work2/08271/jackmc/frontera/lorrax` @ `1a52d51`
(branch `fix/zq-band-gather-device-invariance`), **uncommitted** per the
directive.

---

## 0. What existed

**Nothing.** The "test code to fix" does not exist and never did. An
exhaustive sweep of the working tree, all 200+ local/remote branches and the
full git object database found no code that builds ζ in two different-sized
bases, no cross-basis overlap, and no W basis change:

* the three named candidates are unrelated —
  `src/common/isdf_zeta_mode_test.py` compares the `high_mem`/`low_mem`
  ζ **solve** modes at ONE basis size; `src/common/w_solve_modes_test.py`
  compares the two W **Dyson** plans at one n_μ;
  `tests/test_zeta_mesh_invariance.py` is a device-count invariance gate
  (n_μ fixed inside every worker).
* `grep`/`git log -S` for `zeta_overlap|zeta_small|mu_small|n_rmu_small|
  transfer.?matrix|congruence|two bases|cross.?overlap|project_w` → zero
  hits anywhere in history.
* the ONLY place the codebase reasons about two different-`n_rmu` ζ files
  coexisting is `gw/gw_init.py::_check_zeta_h5_matches_basis`, which
  **refuses** the combination (commit `63f3e2d`).

So: nothing to fix, nothing that was merely serial. Everything below is new.

Two pieces of existing machinery were reused rather than reinvented:

* `common.contract_bands.contract_bands_block_reshard` — the two-sided
  congruence (§2). **The primitive applied, unchanged.**
* `gw/v_q_g_flat.py::_make_per_q_kernel` — already accepts two ζ buffers
  with *different* `n_rmu_L`/`n_rmu_R` (the bispinor charge×current
  off-diagonal tiles) and computes
  `Σ_G conj(ζ̃_L) · v(q+G) · ζ̃_R`. It is the read of record for the
  **metric convention**: the ζ overlap is that contraction with
  `v(q+G) → 1` (§1). Its sharding (`P('x',None)` × `P('y',None)`, G
  chunked and replicated) is *not* what the overlap wants at μ_L=10k —
  see §3.2.

---

## 1. The algebra, and the metric convention (read before touching it)

`W_q(r,r') ≈ Σ_{μν} ζ_{q,μ}(r) W[q,μ,ν] conj(ζ_{q,ν}(r'))` in both bases,
so the basis change is a **congruence with a rectangular transfer matrix**

    W_S = T · W_L · T†,        T = T[q, μ_S, μ_L]                      (★)

**Metric.** Fixed by what `gw.v_q_g_flat` already does:
`V_q[μ,ν] = Σ_G conj(ζ̃_μ(G)) v(q+G) ζ̃_ν(G)` over the per-q G-sphere, with
`ζ̃ = FFT[e^{-iq·r} ζ]` at `norm="backward"`
(`common.wfn_transforms.accumulate_rchunk_to_gflat`). The overlap is the
*same* contraction with unit weight:

    O[q, μ_S, μ_L] = Σ_G conj(ζ̃^S(G)) ζ̃^L(G)                          (†)

Three things this pins down:

1. **The `e^{iq·r}` phase cancels.** ζ = phase × cell-periodic z (manual
   §5.1); both bases sit at the same q, so the on-disk G-flat
   (cell-periodic) buffers are directly the right operands. No phase
   bookkeeping, and this is *why* the two bases must share the FFT grid /
   sphere — a mismatch is refused, not interpolated.
2. **It is the sphere-truncated, N_r-scaled r-grid overlap.** At
   `norm="backward"`, Parseval gives `Σ_G conj(z̃^S) z̃^L = N_r Σ_r conj(z^S)
   z^L` over the full box; the production sphere truncates it exactly as
   V_q is truncated. Same metric V_q uses — not a different one.
3. **Therefore the transfer is NOT the raw overlap.** The least-squares
   transfer

       T = G_S^{-1} O,     G_S[q] = Σ_G conj(ζ̃^S) ζ̃^S                 (‡)

   * minimizes `‖W_L(r,r') − W_S(r,r')‖_F` over the small basis;
   * is **invariant under metric → c·metric** (both G_S and O carry c), so
     the N_r factor and the sphere truncation cannot leak a scale into
     W_S — a bare congruence with O scales as `c²`;
   * is **exact** whenever the field is representable in the small basis.

   The bare-overlap congruence is still available (`mode="raw"`) and
   announces what it is. The choice is *measured*, not asserted: the
   `roundtrip` gate runs both (§4.3).

`G_S` is (n_q, μ_S, μ_S) — bounded by the dimension the whole exercise
makes small. It is the ONLY replicated object in the chain, announced with
its byte size at build time (65 MB at n_q=16, μ_S=504).

---

## 2. Did the `contract_bands` primitive apply? — YES, unmodified

`docs/dev/staged_reshard_primitive.md` §5 asked for a second consumer.
This is one, and it is a clean drop-in with **no change to the primitive**:

| primitive | here |
|---|---|
| `k` (batch) | `q` (momentum transfer) |
| `μ, ν` (contracted) | `μ_L, ν_L` — the LARGE basis |
| `m, n` (output tile) | `μ_S, ν_S` — the SMALL basis |
| `ψ_left  (nk,m,s,μ)` `P(None,None,None,'x')` | `conj(T)` on 'x' |
| `O (nk,s,μ,s',ν)` `P(None,None,'x',None,'y')` | `W_L` — free reshape of the production `P(None,'x','y')` |
| `ψ_right (nk,s',ν,n)` `P(None,None,'y',None)` | `T†` on 'y' |
| `s, s'` | size 1 (ζ is spin-independent, manual §5.2) |
| returns `(nk,m,n)` `P(None,'x','y')` | `W_S`, same layout as W_L |

`out[k,m,n] = Σ conj(ψ_left)[k,m,s,μ] O[k,s,μ,s',ν] ψ_right[k,s',ν,n]`
with `ψ_left = conj(T)`, `ψ_right = T†` is exactly `T W_L T†`.

Inherited for free: the two-stage psum_scatter (no `(μ,μ)` tile on any
rank), the large-payload-on-node-local-axis policy, the f64-split
de-promotion, the vendor-BLAS GEMM dial (`LORRAX_BANDS_GEMM_FFI`, AUTO),
the impl=mpi world-collective-first warm-up, and the actionable
divisibility refusals. `extra` is unused — q is already the primitive's
k axis, so an ω/τ batch remains free for a future caller.

**Not** covered by the primitive, and therefore written here: building `T`
itself (§3.2), which is an outer-product-class contraction over the r/G
axis — the same class §5 of the primitive doc flags for `vq_interp`
("needs a structural sibling, not this primitive"). That sibling is
`zeta_overlap_block_reshard`, written in the same idiom (explicit
collectives inside one `shard_map`, `check_rep=False`).

---

## 3. What was built

`src/common/zeta_projection.py` (library) and
`src/common/zeta_projection_test.py` (gates + scaling driver), following
the `isdf_zeta_mode_test.py` / `w_solve_modes_test.py` convention the
owner pointed at.

### 3.1 Doctrine-1 ledger

| object | global shape / spec | per-rank |
|---|---|---|
| `W_L` | (n_q, μ_L, μ_L) `P(None,'x','y')` | ∝ μ_L²/P ↓ |
| `ζ̃^L` | (n_q, μ_L, n_G) `P(None,None,('x','y'))` | ∝ μ_L·n_G/P ↓ |
| `ζ̃^S` | (n_q, μ_S, n_G) `P(None,None,('x','y'))` | ∝ μ_S·n_G/P ↓ |
| `T_left` | (n_q, μ_S, 1, μ_L) `P(None,None,None,'x')` | ∝ μ_S·μ_L/p_x ↓ |
| `T_right` | (n_q, 1, μ_L, μ_S) `P(None,None,'y',None)` | ∝ μ_S·μ_L/p_y ↓ |
| `G_S` | (n_q, μ_S, μ_S) REPLICATED | bounded ∝ μ_S² |
| `W_S` | (n_q, μ_S, μ_S) `P(None,'x','y')` | ∝ μ_S²/P ↓ |

Everything except `G_S` falls with P; `G_S` is the small basis's own Gram
and can never reach μ_L². Measured VmHWM confirmation: §5.

### 3.2 Building `T` — the r/G axis IS the parallel axis

`zeta_overlap_block_reshard` shards **G flat over the whole mesh** and
produces BOTH shardings of `O` in one pass from the same rank-local
partial:

    loc  = Σ_{G_local} conj(ζ^S_chunk) ζ^L           rank-local
    O_x  = psum(psum_scatter(loc, 'x', dim=μ_L), 'y')
    O_y  = psum(psum_scatter(loc, 'y', dim=μ_L), 'x')

Each chain *completes* the G reduction (scatter-axis group, then the
orthogonal group) while *tiling* μ_L, so the result lands already sharded
and nothing of size (μ_S, μ_L) global, let alone (μ_L, μ_L), is ever
rank-resident. `loc` — the only (n_q, chunk, μ_L) transient — is bounded
independently of P by `mu_s_chunk` (default: ≤256 MB).

**Superseded for scaling by `zeta_overlap_single_axis` (§5.4)** — that
plan shards μ_L as well as G, needs one `psum` per output instead of a
4-collective chain per μ_S chunk, and is the one that actually strong-
scales. The one-pass plan is retained as the parity reference (the two
compute the same matrix by structurally different collectives, which is
now an in-line gate at every scaling point) and for callers who can only
supply ζ in one layout.

Deriving `O_y` from `O_x` by `with_sharding_constraint` was rejected: that
hands the partitioner an all-to-all it plans itself, the exact failure mode
QUALITY_PATTERNS §4 records. Cost of the honest version: the μ_S chunking
means 4 collectives per chunk instead of 4 total (the AK.9 stacking policy
is *not* applied here, deliberately — stacking would require the full
(n_q, μ_S, μ_L) partial and defeat the transient bound). Trade recorded,
not hidden.

The LS solve `G_S T = O` is a batched Cholesky back-substitution done
**inside a shard_map** with `G_S` replicated and μ_L sharded ⇒ rank-local
by construction, no collective.

### 3.3 Interface note / honest limitation

The overlap wants ζ **G-sharded**; `file_io.ZetaLoader.read_zeta_G_slab`
delivers ζ **μ-flat-sharded** (`P(None,('x','y'),None)`), which is what the
V_q driver wants. A production wiring therefore needs one μ→G reshard
(a single all-to-all of the ζ operand) or a second overlap plan with μ_L
sharded and G chunk-replicated — the latter would make ζ^S replicated over
G (μ_S·n_G per rank, NOT falling with P), so the reshard is the better
trade. **This wiring was not built and not measured** — the driver
generates ζ operands directly at the required sharding.

---

## 4. Correctness gates

Job **7879506** — `zproj_gate.sbatch` re-run on the FINAL tree, after the
two-pass overlap plan and the hermiticity/parity gates were added, so this
is the green record that matches the shipped code (7879477 was the same
suite on the pre-two-pass tree, identical numbers). Queue `small`, 1 node,
4 emulated devices, one process, ~2 minutes. **All cells rc=0.**
Predecessor 7879471 recorded below because it earned its keep.

### 4.1 `dense` — whole chain vs an independent dense numpy reference

n_q=3, μ_L=192, μ_S=48, n_G=256. Gathers ζ^S, ζ^L, W_L and recomputes
`T = solve(G_ref, O_ref)`, `W_ref = T W_L T†` entirely in numpy.

```
W_L device vs host formula   rel = 1.522e-16   (gate 1e-14)
W_L hermiticity              rel = 1.522e-16   (gate 1e-14)
W_S vs dense numpy reference rel = 5.111e-15   (gate 1e-12)   <-- THE gate
W_S hermiticity              rel = 4.560e-16   (gate 1e-12)
cond(G_S) max over q             = 2.422e+01
PASS
```

Also asserted in the same cell: the synthetic W_L on device equals its
host closed form (which is what licenses that closed form as the reference
in the production-scale `selection` gate), and W_L / W_S hermiticity.

### 4.2 `selection` — the scale-free kernel gate

`T` = a selection matrix (rows of the identity picking μ_S of the μ_L
indices) ⇒ `W_S` must equal the sub-block `W_L[idx, idx]`. Needs no dense
reference and no gather of any (μ_L, μ_L) object, so it **runs at full
production scale** alongside the scaling points.

```
μ_L=192, μ_S=48 (unit):  W_S vs W_L[idx, idx]  rel = 7.641e-17   (gate 1e-12)
```
(the production-scale value is `3.037e-16` at every P — §5.5 — the same
cell runs after every scaling point via `--with-selection`).

### 4.3 `roundtrip` — the physically meaningful invariance

Build ζ_S as *exact* linear combinations `ζ_S = A ζ_L` (the small basis
then spans a subspace of the large one), embed an arbitrary small-basis
`W_S0` into the large basis exactly — `W_L = Aᵀ W_S0 conj(A)`, which is
what the ansatz gives when the field is representable — and project back.

Algebra: `O = conj(A) G_L`, `G_S = conj(A) G_L Aᵀ`, so
`T Aᵀ = (conj(A) G_L Aᵀ)^{-1} conj(A) G_L Aᵀ = 1` and
`W_S = (T Aᵀ) W_S0 (T Aᵀ)† = W_S0` **exactly**. This is the defining
property of the LS transfer, and it is precisely what the bare-overlap
congruence does not have — the cell runs both and reports the gap.

n_q=2, μ_L=128, μ_S=32, n_G=256, `A` random (32×128), cond(G_S)=15.9:

```
mode=ls    rel |W_S - W_S0| = 1.540e-15     <-- exact, as derived
mode=raw   rel |W_S - W_S0| = 9.451e+03     <-- off by ~4 decades
PASS
```

The `raw` row is the point: the bare-overlap congruence is not merely
less accurate, it is off by the metric scale squared. This is the measured
justification for `mode="ls"` being the default.

### 4.4 `meshinv` — same answer on 1×4 / 4×1 / 2×2

The synthetic operands are P-independent by construction (built from small
replicated seed vectors inside a shard_map, from the rank's own axis
indices), so the dense cell must give the same W_S on every mesh shape.
The `scale` cell prints `Re Σ`, `Im Σ` and `‖W_S‖²` checksums for the same
reason — the strong-scaling sweep doubles as a mesh-invariance gate.

```
mesh 2x2   W_S vs dense numpy reference rel = 5.111e-15   cond(G_S)=2.422e+01
mesh 1x4   W_S vs dense numpy reference rel = 4.932e-15   cond(G_S)=2.422e+01
mesh 4x1   W_S vs dense numpy reference rel = 5.113e-15   cond(G_S)=2.422e+01
```
Three mesh shapes, one numpy reference, all ≤5.2e-15 ⇒ cross-mesh
agreement ≤1e-14. (1x4 and 4x1 also exercise the degenerate p_x=1 / p_y=1
cases, where one of the two psum_scatters becomes a no-op.)

### 4.5 `refuse` — announce-or-refuse, no silent fallbacks

All six refusals fire with the fix named (job 7879471, re-run in 7879477):

| gate | verdict |
|---|---|
| inverted mesh axis order (minor ≠ 'y') | REFUSED |
| μ_S indivisible by p_x / p_y | REFUSED (primitive's message, with the ζ-specific hint appended) |
| n_G indivisible by p_x·p_y | REFUSED |
| ζ_S / ζ_L on different (n_q, n_G) | REFUSED — "refit one of them on the other's grid" |
| singular small-basis Gram | REFUSED — no ridge, no pinv substituted |
| bad `mode` | REFUSED |

### 4.6 The gate that earned its keep (job 7879471)

The first synthetic ζ used a purely multiplicative hash
`(a·μ + b·G + c·μ·G) mod 2^20`. That makes the phase *difference* between
two ζ's linear in G with slope ~1/2^20 — over n_G ≲ 4k every ζ is nearly
parallel to every other, and `G_S` is numerically singular. The
positive-definiteness refusal caught it immediately and printed
`Gram diagonal min/max = 6.447e+01/6.447e+01` (a constant diagonal — the
tell). Fixed with a bit-mixing hash carrying an `s·t` cross term;
measured `cond(G_S) ≈ 20` at both gate sizes. **A silent ridge would have
turned this into a plausible-looking wrong W_S.**

---

## 4.7 Transport finding: impl=mpi refuses a standalone driver's grouped clique

The first production-scale attempt (**job 7879482**, all three points,
rc=1) died at the first grouped `psum_scatter` with

    UNKNOWN: Buffer Definition Event: MPI: Communicator requested from a
    thread that is not the one MPI was initialized from.

**with `contract_bands.ensure_grouped_collectives_ready()` having run** —
i.e. the world-collective-first contract of RESHARD_OVERHEAD_MEMO §4.2
was satisfied and the failure happened anyway.

**Probe job 7879485** (P=4, 2 nodes, one fresh process per variant,
`zproj_mpiprobe.py` / `zproj_mpiprobe.sbatch`) settles it:

| variant | transport | warm-up before the grouped chain | verdict |
|---|---|---|---|
| `mpi_none` | mpi | none | **FAIL** (rc=4) |
| `mpi_sgd` | mpi | `sync_global_devices` (what the helper does) | **FAIL** |
| `mpi_psum` | mpi | `lax.psum` over BOTH axes inside shard_map, caller's mesh | **FAIL** |
| `mpi_ag` | mpi | world all-gather on the caller's mesh | **FAIL** |
| `mpi_both` | mpi | psum + sgd | **FAIL** |
| `gloo_none` | gloo (ib0 pin) | none | **PASS** (value MATCH) |
| `gloo_sgd` | gloo (ib0 pin) | sync_global_devices | **PASS** |

The chain probed is byte-for-byte the one `zeta_overlap_block_reshard`
issues (`psum_scatter('x') + psum('y')` and the mirror), at tiny shapes,
so this is not a size or memory effect.

Reading: jaxlib's `MpiCollectives::CreateCommunicators` gates on
`MPI_Is_thread_main()`, and XLA:CPU executes the collective from its
thread pool. **No warm-up reachable from a standalone driver fixes it.**

Two things this does NOT say:

* it does not contradict production. `gw.gw_jax` **does** run grouped
  collectives under impl=mpi at P=64 (job 7879010, 8/8 passes rc=0), so
  the configuration is satisfiable — by something in the full driver
  process that a standalone module does not reproduce. **What that is was
  not chased down**; it is a real open item for whoever owns the memo §4.2
  claim, because the helper that encodes it is measured insufficient.
* it does not make gloo the "right" transport in general — only the one
  that is *measured green* for this chain, here and in memo §4.3
  (job 7878883 step 1c, the same staged reduce-scatter chain on gloo/ib0
  at P=64, rc=0, full suite).

Consequences, both recorded honestly:

1. `common.zeta_projection.ensure_world_clique_ready()` still issues the
   memo's world-clique collective, but its main job is now to **announce**
   the known-bad configuration on rank 0 instead of letting the caller meet
   an `UNKNOWN` error mid-run. It is deliberately **not** a refusal —
   production runs impl=mpi successfully and a refusal would break it.
2. **The scaling numbers in §5 are gloo/ib0 numbers.** Per memo §4.3 gloo
   moves ~0.52 GB/s effective per rank on this fabric, which is part of the
   measured domain of every efficiency figure below, and the single largest
   caveat on them.

---

## 5. Strong scaling

**Jobs 7879499** (P = 4/16/64, ONE 32-node allocation so all three points
sit on identical hardware) and **7879504** (P = 144, 72 nodes).
Transport gloo/ib0 (§4.7). `LORRAX_BANDS_GEMM_FFI` AUTO-ON — the
vendor-BLAS batched-GEMM handler resolved and engaged at every point.

**Fixed problem at every P** (strong scaling — the problem does not grow):

| | |
|---|---|
| n_q | 16 (MoS2 4×4×1 full-BZ q count) |
| μ_L | 10008 (the real `c10000` deck's 10015 centroids, rounded DOWN to a multiple of 24 = lcm(8,12) so ONE problem serves p ∈ {2,4,8,12}) |
| μ_S | 504 |
| n_G | 4032 (multiple of 576 = lcm(4,16,64,144)) |
| W_L | 25.64 GB global, `P(None,'x','y')` — 1.60 GB per q |
| ζ_L / ζ_S | 10.33 / 0.52 GB global |
| T / W_S | 1.29 / 0.065 GB global |
| congruence work | 6.787 TFLOP total (`8·n_q·(μ_L²μ_S + μ_Lμ_S²)`) |
| reps | 10 barrier-aligned repetitions; `min` quoted |

**Operands are SYNTHETIC — see §7.1.** Shapes, shardings, dtypes and q
count are the real deck's; ζ and W_L values are analytic.

### 5.1 The congruence `W_S = T W_L T†` — the headline table

| P | mesh | nodes | T_proj min (s) | speedup | **efficiency** | GF/s/rank | RS MB/rank | VmHWM max (GB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2×2 | 2 | **2.6028** | 1.00× | **100 %** (ref) | 652 | 645.6 | **12.954** |
| 16 | 4×4 | 8 | **1.1052** | 2.36× | **58.9 %** | 384 | 322.8 | **6.949** |
| 64 | 8×8 | 32 | **0.4988** | 5.22× | **32.6 %** | 213 | 161.4 | **3.671** |
| 144 | 12×12 | 72 | **0.3284** | 7.93× | **22.0 %** | 144 | 107.6 | **3.091** |

Wall time falls monotonically 2.60 → 0.33 s (7.9× on 36× the ranks) and
the answer is IDENTICAL at all four P (§5.5).

### 5.2 Where the efficiency goes — arithmetic and wire fall at different rates

* **arithmetic** falls as 1/P: 1.697 / 0.424 / 0.106 / 0.047 TFLOP per rank;
* **the stage-1 reduce-scatter payload falls only as 1/p_x** — it is
  `n_q·(μ_L/p_x)·μ_S` — 645.6 / 322.8 / 161.4 / 107.6 MB per rank, i.e.
  6× less traffic for 36× more ranks.

At gloo/ib0's measured ~0.52 GB/s effective per rank (memo §4.3) those
payloads cost ≈1.24 / 0.62 / 0.31 / 0.21 s, against ≈1.36 / 0.49 /
0.19 / 0.12 s of GEMM. So the point is **compute-bound at P=4 and
wire-bound from P=64 on**, and the efficiency curve is exactly the two
lines crossing. The 144 GF/s/rank at P=144 is not a GEMM regression —
it is the same GEMM with ~2/3 of the wall in the collective.

This is a property of the **transport**, not of the primitive: the mpi
collectives are unavailable to a standalone driver (§4.7), and gloo's
~0.52 GB/s is the measured domain of every efficiency number above.

### 5.3 Memory — the doctrine-1 evidence

/proc `VmHWM`, read on every rank:

| P | VmHWM max (GB) | VmHWM med (GB) | vs P=4 | W_L share (GB) |
|---:|---:|---:|---:|---:|
| 4 | **12.954** | 12.871 | 1.00× | 6.41 |
| 16 | **6.949** | 6.741 | 1.86× | 1.60 |
| 64 | **3.671** | 3.455 | 3.53× | 0.40 |
| 144 | **3.091** | 3.055 | 4.19× | 0.18 |

**The per-rank footprint FALLS monotonically with P** — 12.95 → 3.09 GB.
That is the doctrine-1 evidence: a replicated `(n_q, μ_L, μ_L)` W_L
would be 25.6 GB on *every* rank at *every* P, and 1.60 GB even for a
single q; no rank ever holds one.

It falls 4.2× for 36× ranks, not 36×, and the residual is accounted for:

```
per-rank model            P=4      P=16     P=64     P=144
W_L            25.64/P    6.41     1.60     0.40     0.18
zeta_L         10.33/P    2.58     0.65     0.16     0.07
T_x+T_y, psi, O (x3)      3.87     1.94     0.97     0.65
G_S           replicated  0.065    0.065    0.065    0.065
`loc` transient (1-pass)  0.253    0.253    0.253    0.253   <- P-INDEPENDENT
-------------------------------------------------------------
model sum                13.2      4.5      1.8      1.2
measured VmHWM           12.95     6.95     3.67     3.09
```

Two P-independent terms are named rather than hand-waved: the Python/XLA
runtime floor (~1.5-2 GB) and the one-pass overlap's μ_L-full `loc`
partial (253 MB) — the latter is removed by the two-pass plan (§5.4).

**Collective-table verdict (P=64, cache-cold rank-0 HLO dump, AY.2
satisfied — `ISDF_JAX_CACHE_DIR=""`, fresh dump dir):**

```
NO collective carries a full (mu,mu) tile (mu=10008).
```

### 5.4 The transfer build: the one-pass plan is flat, the two-pass plan scales

| P | overlap 1-pass (s) | overlap 2-pass (s) | 2-pass speedup | 2-pass efficiency |
|---:|---:|---:|---:|---:|
| 4 | 6.935 | **5.263** | 1.00× | 100 % |
| 16 | 6.813 | **2.434** | 2.16× | 54.0 % |
| 64 | 8.162 | **2.224** | 2.37× | 14.8 % |
| 144 | 6.795 | **1.262** | 4.17× | 11.6 % |

The one-pass `zeta_overlap_block_reshard` is **flat in P** (6.9 → 6.8 →
8.2 → 6.8 s over a 36× rank increase) even though its arithmetic falls as
1/P. The cause was predicted from the code before it was measured: only
**G** is sharded there, so the rank-local partial
`loc[q, μ_S_chunk, μ_L]` is **μ_L-FULL** and the reduce-scatter it feeds
carries `n_q·chunk·μ_L` bytes **independently of P** (≈4.4 GB per rank at
every P).

`zeta_overlap_single_axis` shards μ_L **and** G, so the rank-local partial
*is* the output shard: ONE `psum` per output instead of a 4-collective
chain per μ_S chunk, payload `n_q·μ_S·μ_L/p_mu` which falls with the mesh,
and no μ_S chunking at all. Measured **5.26 → 1.26 s (4.2×)**, and it is
already faster than the one-pass at every P including P=4. Its efficiency
also decays (gloo again — the payload falls only as 1/p_mu), but it
*scales*, which the one-pass does not.

The price: ζ must be supplied in two layouts (μ on 'x'/G on 'y' and μ on
'y'/G on 'x') — one reshard, one re-read, or (driver: ζ is analytic) one
regeneration.

The `ls` and `gram` rows (3.4 / 7.7 / 17.1 / 13.8 s and 0.22 / 0.18 /
0.23 / 0.27 s) carry JIT compile and an eager replicated Cholesky, are
single-shot and barrier-aligned, and are **not repeated measurements** —
they are not a scaling curve and should not be read as one. Only the
`project` row is a repeated measurement.

### 5.5 P-invariance: the scaling sweep doubles as a correctness gate

The synthetic operands are P-independent by construction (built inside a
`shard_map` from the rank's own axis indices, as a pure function of the
GLOBAL (q, μ, G) index), so `W_S` must not depend on P. Measured, all
four points:

```
P=4    Re Σ W_S = 7.979485925820e+04   ‖W_S‖² = 2.792053096821e+07
P=16   Re Σ W_S = 7.979485925820e+04   ‖W_S‖² = 2.792053096821e+07
P=64   Re Σ W_S = 7.979485925820e+04   ‖W_S‖² = 2.792053096821e+07
P=144  Re Σ W_S = 7.979485925820e+04   ‖W_S‖² = 2.792053096821e+07
```

Identical to all 13 printed digits across a 36× range of P and four mesh
shapes. Plus, at every point:

| P | overlap plan parity (O_x / O_y / G_S) | W_S hermiticity | selection gate |
|---:|---|---:|---:|
| 4 | 3.819e-16 / 3.819e-16 / 4.014e-16 | 4.388e-16 | 3.037e-16 |
| 16 | 4.528e-16 / 4.528e-16 / 5.017e-16 | 3.843e-16 | 3.037e-16 |
| 64 | 7.526e-16 / 6.866e-16 / 1.003e-15 | 3.727e-16 | 3.037e-16 |
| 144 | 7.065e-16 / 7.497e-16 / 1.003e-15 | 3.801e-16 | 3.037e-16 |

(plan parity = the one-pass overlap vs the structurally independent
two-pass plan — different sharding, different collectives, no chunking.)

---

## 6. A silent wrong answer at P=4 — first sighting

> **SUPERSEDED BY §9.** This section records the FIRST sighting and the
> reasoning done at the time, when it had not yet reproduced. It HAS
> since reproduced, been measured as a rate, and been localized to
> `psum_scatter` — read §9 for the current state. Kept because the
> elimination chain here (kernel, chunk loop, overlap plan) is what made
> the later localization quick, and because "not reproduced" was a
> conclusion I drew and then had to withdraw.


The P=4 point of job **7879491** returned a `W_S` that is not Hermitian:

```
7879491 P=4    Re Σ W_S = 7.853599816794e+04   Im Σ W_S = 6.253062298730e+02   ‖W_S‖² = 3.070159311873e+07
7879491 P=16   Re Σ W_S = 7.979485925820e+04   Im Σ W_S = -2.27e-12            ‖W_S‖² = 2.792053096821e+07
7879491 P=64   Re Σ W_S = 7.979485925820e+04   Im Σ W_S = -2.27e-12            ‖W_S‖² = 2.792053096821e+07
```

`W_S = T W_L T†` with `W_L = W_L†` is Hermitian for **any** T, and the sum
of a Hermitian matrix is exactly real. `Im Σ = 6.25e+02` against
`Re Σ = 7.85e+04` is a proof of corruption, not a tolerance question.
P=16 and P=64 agreed with each other to all 13 printed digits.

### What it was NOT (each ruled out by measurement)

1. **Not the congruence kernel.** The `selection` gate drives the same
   `contract_bands_block_reshard` call with a transfer that never touches
   ζ: `rel = 3.037e-16` at P=4 **and** at P=16, bit-identical.
2. **Not the μ_S chunk loop.** The unit gates auto-resolve the overlap to
   ONE chunk; production resolves to six — a genuine coverage hole. Job
   **7879496** swept `--mu-s-chunk ∈ {auto, 7, 11, 16, 25}` (1 → 7 chunks)
   against the dense numpy reference on 4 **and** 16 emulated devices:
   **every chunk count returns `rel = 5.111e-15` / `4.927e-15`,
   bit-identical.** The chunk algebra is correct.
3. **Not the overlap plan.** Job **7879499** re-ran the identical
   configuration with a new in-line parity gate comparing the one-pass
   overlap against the structurally independent two-pass plan
   (`zeta_overlap_single_axis`: different sharding, different collectives,
   no chunking). At P=4:

   ```
   overlap plan parity  O_x 3.819e-16  O_y 3.819e-16  G_S 4.014e-16   PASS
   W_S hermiticity      rel = 4.388e-16                               PASS
   ROW P=4 ... chk_re=7.979485925820e+04  chk_n2=2.792053096821e+07
   ```

   — the same P=4, same code, same data, now **correct**, and its
   checksum now matches P=16/P=64 to all digits.

### What it therefore was

A **non-deterministic, silent corruption** in a multi-process gloo run:
same binary, same operands, same mesh, one run wrong and the next right.
The affected module is the one that issues 24 grouped collectives
(6 μ_S chunks × 2 chains × 2 collectives) at 253 MB payloads over 2-rank
replica groups. It was **not** reproduced and is **not** root-caused.

### What was done about it

* **Two in-line gates now run at every production-scale point**, so this
  condition is caught by the run that provokes it rather than inferred
  afterwards from a checksum: `W_S` Hermiticity (a property that holds for
  any T, so it tests only the machinery) and one-pass-vs-two-pass overlap
  parity (two structurally different collective patterns computing the
  same matrix). Both are cheap — one jitted reduction each, no gather.
* `zeta_overlap_single_axis` issues **one** collective per output instead
  of 24, which shrinks the exposure as a side effect of fixing the
  scaling (§5.4).

**Standing recommendation:** do not run a production ζ-projection over
gloo without the Hermiticity gate enabled. A rerun that agrees is
evidence; a single run is not.

## 7. Honest limits

### 7.1 The scaling operands are SYNTHETIC — and the real-ζ inventory, corrected

Timing and memory depend only on shapes, shardings and dtypes, and those
are the real deck's: n_q=16 and the MoS2 4×4×1 1024-band geometry, with
μ_L=10008 taken from that deck's own `c10000` centroid file (10015
points). What is synthetic is the ζ **values** (a bit-mixing analytic
hash, cond(G_S)≈20) and W_L (`f_i conj(f_j) g(|i−j|)`, exactly
Hermitian). **No accuracy claim about real ISDF bases is made from the
scaling runs**, and conditioning-dependent behaviour does not transfer.

**Correction to an earlier claim in these notes.** An initial search
(`find -maxdepth 2` under `mos2_4x4_test`) reported no ζ on disk. That
was wrong — ζ lives at `<run>/tmp/zeta_q.h5`, one level deeper. The real
inventory (172 files; those with an actual `zeta_q_G` dataset):

| deck | μ | dataset | size |
|---|---:|---|---:|
| MoS2 12×12 80 Ry | 2406 | `zeta_q_G {144, 2406, 8603}` | 45 GB (`lorrax_mos2_12x12/run_A_c2406_b400_AF/tmp`) |
| MoS2 12×12 80 Ry | 606 | `zeta_q_G {144, 606, 8603}` | 12 GB (`lorrax_mos2_12x12/sweep_c606/tmp`, ×4 copies) |
| MoS2 12×12 80 Ry | 276 | `zeta_q_G {144, 276, 8603}` | 5.2 GB (`mos2_80ry_12x12/tmp`) |

The MoS2 **4×4** files (`run_L5_b1024_c10000/tmp/zeta_q.h5` and the other
c7000/c10000/c15000/c25000 runs) contain **only `g0_mu` and the headers —
`zeta_q_G` is NOT FOUND in them**, so there is no real ζ at μ≈10k
anywhere. The 12×12 deck does, however, carry **three real bases of
different size on the SAME grid, q set and G-sphere** (144 q,
ngkmax=8603) — i.e. a genuine (μ_L, μ_S) pair is available:
**(2406, 606)**, (2406, 276) or (606, 276).

**That real-data run was NOT made** — the budget went to the P=4
corruption of §6 and to the P=144 point. It is the single highest-value
follow-up, and it is a small one. What it needs:

* the ZetaLoader wiring of §7.2 (the only missing code);
* μ padding: 2406 = 2·3·401 and 606 = 2·3·101 are divisible by 2 and 6
  but **not** by 4 or 8, and n_G = 8603 is odd — so the module's zero-pad
  refusals will fire and the caller must pad (`n_rmu_padded` is already
  the on-disk contract; `ZetaLoader` zero-fills past `ngk[q]`). A 6×6 or
  2×3-shaped mesh takes the padding to zero for μ;
* it would exercise the metric on real ζ and report a real `cond(G_S)` —
  the one number the synthetic study cannot supply.

### 7.2 The ZetaLoader wiring is not built

The overlap consumes ζ G-sharded (one-pass) or μ+G sharded (two-pass);
`ZetaLoader.read_zeta_G_slab` delivers ζ μ-flat-sharded
(`P(None,('x','y'),None)`), which is what the V_q driver wants. A
production consumer needs one μ→G reshard, or a second read at the other
layout. Neither was written nor measured.

### 7.3 gloo/ib0 only

The mpi collectives are measured unusable from a standalone driver
(§4.7) — five warm-up variants, all fail. Every efficiency number carries
gloo's ~0.52 GB/s per-rank effective bandwidth as its measured domain.
A faster transport moves the compute/wire crossover right and would
report better efficiency at P≥64; that run was not made. **Production
`gw_jax` runs impl=mpi successfully**, so the gap is in the standalone
path, and closing it is an open item against the memo §4.2 claim.

### 7.4 The P=4 corruption is not root-caused

§6. Seen once, not reproduced, gates added. Do not treat a single
ζ-projection run over gloo as evidence without the Hermiticity gate.

### 7.5 HLO pins are partial

The colltable doctrine verdict at P=64 was obtained cache-cold
(§5.3). What was **not** done: a `tests/`-level HLO pin in the
§3.3 style (collective count / payload shapes / forbidden promotion
converts) for the new `shard_map` bodies. `contract_bands` has one
(`tests/test_contract_bands.py`); `zeta_projection`'s own overlap
kernels do not.

### 7.6 `G_S` is replicated

`(n_q, μ_S, μ_S)`, 65 MB at these sizes — the one object that does not
fall with P. Bounded by the SMALL basis by construction, but a caller who
wants μ_S in the thousands must revisit `least_squares_transfer`.

### 7.7 `mode="raw"` is exposed and is wrong for most purposes

It carries the metric normalization squared — the round-trip gate
measures the gap at 9.45e+03. Kept because the bare overlap is a
legitimate object to want; it announces itself.

### 7.8 The `ls` / `gram` stage timings are single-shot

They include JIT compile and an eager replicated Cholesky and were not
repeated. Only the `project` row (10 reps) and the overlap A/B are
measurements.

---

## 8. Artifacts

| path | what |
|---|---|
| `src/common/zeta_projection.py` | the library (NEW, uncommitted) |
| `src/common/zeta_projection_test.py` | gates + scaling driver (NEW, uncommitted) |
| `wk_REL/harness/zproj_gate.sbatch` | unit gates (dense / selection / roundtrip / meshinv / refusals) |
| `wk_REL/harness/zproj_scale.sbatch` | the P = 4/16/64 sweep |
| `wk_REL/harness/zproj_p144.sbatch` | the P = 144 point |
| `wk_REL/zproj_mpiprobe.{py,sbatch}` | the transport probe of §4.7 |
| `wk_REL/harness/zproj_chunk.sbatch` | the μ_S-chunk localization of §6 |

| job | what | verdict |
|---:|---|---|
| 7879471 | unit gates, first attempt | caught the degenerate synthetic ζ (§4.6) |
| 7879477 | unit gates | ALL PASS |
| 7879482 | first production sweep, impl=mpi | rc=1 ×3, grouped clique refused |
| 7879485 | transport probe, 7 variants | mpi FAILS ×5, gloo PASSES ×2 |
| 7879488 | gloo sweep, first | timings only (checksum gather bug) |
| 7879491 | gloo sweep | P=16/64 clean, **P=4 corrupt** (§6) |
| 7879496 | μ_S-chunk localization, 7 cells | chunk algebra exonerated |
| 7879499 | gloo sweep + parity + hermiticity gates | **P=4/16/64 ALL PASS** |
| 7879504 | P=144 | **PASS** |
| 7879506 | unit gates re-run on the FINAL tree | **ALL PASS** |


---

# FOLLOW-UPS (owner directive after commit 6bf28bc)

## 9. Item 2 — the P=4 silent wrong answer: REPRODUCED and LOCALIZED

**§6 said "seen once, not reproduced". That is now superseded: it
reproduces, and it is localized to one specific collective chain.**

Method (`--cell provoke`, `wk_REL/harness/zproj_provoke.sbatch`): the exact P=4
configuration, every operand held FIXED, **both** overlap plans
recomputed every rep. Two comparisons per rep:

* **cross-plan** — the one-pass result vs the structurally independent
  two-pass result (different sharding, different collectives, no
  chunking). Floating-point reassociation between the two plans is
  measured at **7.246e-14**; anything above ~1e-9 is corruption.
* **drift** — each rep vs its own rep 0, at **EXACT ZERO** (identical
  deterministic operands, so any bit difference is corruption).

Sampling on two axes, because the original sighting was one process and
one execution: 20 reps in-process × 5 independent `srun` launches
(fresh gloo rendezvous each).

### 9.1 The reproduction

Job **7879519**, process 0, **first execution**:

```
rep 0: cross-plan |1p-2p|   O_x 7.246e-14   O_y 1.636e+02      <-- O_y CORRUPT
rep 1: drift vs rep 0       1p.O_x 0.0   1p.O_y 1.635521e+02
                            2p.O_x 0.0   2p.O_y 0.0
```

Read that carefully — it is a complete localization:

* **`1p.O_y` is the only thing wrong.** `1p.O_x` is bit-identical
  across reps (exactly 0.0 drift) and agrees with the two-pass plan to
  7.2e-14.
* **Both two-pass outputs are bit-identical** across reps (exactly 0.0).
  The two-pass plan — one `psum` per output, no `psum_scatter`, no
  chunking — has not been observed to corrupt at all.
* rep 1's `1p.O_y` differs from rep 0's by the same 1.6e+02 that rep 0
  differed from the two-pass reference by, i.e. **rep 0 was the corrupt
  one and rep 1 was correct** — an intermittent, per-execution fault.

### 9.2 Which collective

The one-pass plan runs two chains off the same rank-local partial:

```
O_x :  psum_scatter(loc, 'x', dim=μ_L)  then  psum(·, 'y')     CLEAN
O_y :  psum_scatter(loc, 'y', dim=μ_L)  then  psum(·, 'x')     CORRUPT
```

On the 2×2 mesh at 2 ranks/node the replica groups are
`'y' = {0,1},{2,3}` (consecutive → **intra-node**) and
`'x' = {0,2},{1,3}` (stride-2 → **inter-node**). So the failing chain is
the one whose **`psum_scatter` runs on the node-local axis**, at a
253 MB payload, in a module that issues six of them.

**This deserves attention beyond this module**: `contract_bands`'s
policy §3.2 deliberately puts the LARGE reduce-scatter on the node-local
axis. Under gloo that is exactly the collective implicated here. Caveat,
stated so it is not over-read: `contract_bands`'s own reduce-scatter has
shown **no** corruption in any of these runs — the selection gate, which
drives it, returned `3.037e-16` bit-identically at P = 4/16/64/144.

### 9.3 Incidence

Job **7879526** (5 processes × 20 reps; the per-rep cross-plan detector,
which catches a corrupt rep directly rather than inferring it from drift):

| process | O_x wrong | O_y wrong | verdict |
|---|---|---|---|
| 0 | 0/20 | 0/20 | clean |
| 1 | **2/20** | **1/20** | corrupt |
| 2 | **0/20** | **2/20** | corrupt |
| 3 | 0/20 | 0/20 | clean |
| 4 | **1/20** | 0/20 | corrupt |

**6 corrupt chain-executions in 100 (6.0 %)**, and **3 of 5 process
lifetimes affected (60 %)**. Intermittent, not deterministic.

**Both mesh axes are hit equally — O_x 3, O_y 3.** §9.2's "only the O_y
chain" was based on the single first sighting and is TOO STRONG; it is
corrected here: **either chain of the one-pass plan can be corrupted.**
What has never been hit, in any run, is the two-pass plan
(`zeta_overlap_single_axis`, one `psum` per output, no `psum_scatter`,
no chunking) — 100/100 clean.

The magnitudes are of order the answer itself (gaps 4.7e+01 … 1.24e+02
against a 7.2e-14 floating-point floor) and differ every time — a
partial-sum / race signature, not a deterministic offset.

**Caveat on these numbers — see §12.** Job 7879526 ran with `PYTHONPATH`
pointing at the LIVE worktree while that tree was being edited between
its processes. The edits were additive and confined to `cell_real`, and
the four kernel functions the provocation exercises are byte-identical
to the committed `6bf28bc` — but that is an after-the-fact argument, and
a reproducibility experiment must not rest on one. Job **7879540**
re-runs the identical experiment from the FROZEN snapshot
`srcsnap_20260729_041040_6bf28bc` and is the authoritative number.



### 9.5 AUTHORITATIVE: frozen, provenance-verified snapshot (job 7879540)

Snapshot `srcsnap_20260729_041040_6bf28bc`. `.py` manifest verified at
job start, mid-run, **and after the run** (sources immutable for the full
27 min); every `.pyc` proven compiled from that snapshot's own sources
(§13.2). 5 processes × 20 reps = 100 executions of each plan.

| process | 1p.O_x wrong | 1p.O_y wrong |
|---|---|---|
| 0 | 0/20 | **2/20** |
| 1 | 0/20 | **1/20** |
| 2 | 0/20 | **1/20** |
| 3 | 0/20 | 0/20 |
| 4 | 0/20 | **1/20** |

**5 corrupt of 100 executions (5.0 %); 4 of 5 process lifetimes (80 %).**

The drift check disambiguates which plan is at fault — in every one of
the five events:

```
1p.O_x 0.000000e+00   1p.O_y <1.3e+02 …>   2p.O_x 0.000000e+00   2p.O_y 0.000000e+00
```

i.e. **the two-pass plan and the one-pass O_x chain are EXACTLY
bit-stable (0.0, not "small"), and only the one-pass O_y chain
corrupts.** Across both runs the two-pass plan is clean in **200/200**
executions.

### 9.5.1 The localization, sharpened: always output segment 0

Every event, in both runs, changes the shard held by exactly the ranks
that hold **block 0** of the scattered μ_L axis, with an identical wrong
value on both holders:

```
7879540 (O_y, spec P(None,None,'y')): ranks [0,2] = (x=0,y=0),(x=1,y=0)  -> block y=0
7879526 (O_x, spec P(None,None,'x')): ranks [0,1] = (x=0,y=0),(x=0,y=1)  -> block x=0
correct baseline 2.090585055562361e+10 -> 2.0965e10 / 2.0970e10 / 2.2016e10 / 2.2352e10 / 2.2890e10
```

**11 of 11 events land on segment 0**, with a different wrong value each
time. That is the actionable signature for whoever debugs the transport:
it is the FIRST output segment of `psum_scatter` — the one that would
carry a rank's own/local contribution — not a random shard, and not the
following all-reduce (which would not leave the two holders identical).

### 9.5.2 What NOT to conclude — the axis

The frozen run is O_y 5 / O_x 0; the live-tree run was O_x 3 / O_y 3.
Combined: 8 O_y / 3 O_x of 11 events, which under a fair split is
~23 % two-tailed — **not significant**. The two runs ran on different
nodes. So: **no axis preference is claimed**, and the earlier §9.2
"only the O_y chain" remains withdrawn. What IS consistent across every
event and both runs is (a) the one-pass plan, (b) `psum_scatter`, and
(c) output segment 0.

### 9.6 Upstream report

Written up for a jax/gloo maintainer with no access to this cluster:
**`wk_REL/docs/UPSTREAM_gloo_psum_scatter_corruption.md`**, with the
standalone reproducer `wk_REL/probes/gloo_psum_scatter_repro.py` (no project
imports) and its harness `wk_REL/harness/zproj_repro.sbatch`.

Structure: summary / environment / reproducer / evidence (rate table,
segment-0 finding, identical-value-on-both-holders, controls) / ruled out
/ NOT established. The §6 "NOT established" list is deliberately long —
no axis preference (with the ~23 % two-tailed arithmetic), no root cause,
one shape and one mesh only, no comparison against the mpi backend, one
gloo transport.

### 9.4 Standing guidance

* The **two-pass plan** (`zeta_overlap_single_axis`) is the default to
  use: it is faster, it strong-scales, and it is the one that has never
  been observed to corrupt.
* The **in-line gates stay permanently**: `W_S` Hermiticity (a property
  that holds for *any* transfer, so it tests only the machinery) and
  one-pass-vs-two-pass parity. Both are one jitted reduction, no gather.
* `zeta_overlap_block_reshard` should be treated as a parity reference,
  not a production path, until this is understood.

---

## 10. Item 3 — proposed amendment to the primitive doc

Wording drafted for the coordinator to land:
**`wk_REL/docs/PROPOSED_staged_reshard_primitive_3.5.md`**.

It appends to `docs/dev/staged_reshard_primitive.md` §3.5 (a) the
7879485 probe table showing all five impl=mpi warm-up variants failing
for a standalone consumer while both gloo variants pass, (b) the
statement that the contract is satisfiable — production `gw_jax` does it
(7879010, 8/8 rc=0) — but by something the standalone path does not
reproduce, and unexplained, and (c) an explicit warning NOT to read the
result as "standalone consumers should use gloo", since the two
transports have disjoint known-bad regions (gloo/ib0 died reproducibly
at P=64 under the distributed tiers; gloo also carries this chain
cleanly at P=4…144; and §9's silent corruption is a gloo sighting).
It also proposes a one-sentence change to
`ensure_grouped_collectives_ready`'s docstring, which currently promises
sufficiency.

I did **not** edit `contract_bands.py` or the primitive doc myself —
`contract_bands` is the shared GW/BSE entry point and the ladder
workstream is live against it.

---

## 11. Item 1 — REAL ζ

### 11.1 Inventory and the sphere-identity refusal

Three production ζ on ONE deck (MoS2 12×12 80 Ry, 144 q full BZ,
FFTgrid 36×36×135, ngkmax 8603, all `zeta_is_done=TRUE`, opened
read-only):

| μ | file | size |
|---:|---|---:|
| 2406 | `lorrax_mos2_12x12/run_A_c2406_b400_AF/tmp/zeta_q.h5` | 45 GB |
| 606 | `lorrax_mos2_12x12/sweep_c606/tmp/zeta_q.h5` | 12 GB |
| 276 | `lorrax_mos2_12x12/run1/tmp/zeta_q.h5` | 5.2 GB |

**A refusal that had to be added.** The G-flat on-disk G axis is a
**per-q** sphere (`isdf_header/gvec_components`, `(n_q,3,ngkmax)`), so
two files can agree on ngkmax, ngk, FFTgrid and n_q and still index
different G's at the same array position — in which case the overlap is
silently meaningless. Measured: the three files above share
`gvec_components` **bit-for-bit** (md5 `977abc2179e8…`), while a FOURTH
μ=276 file (`mos2_80ry_12x12/tmp/zeta_q.h5`) matches on every other
header field and has a **different** `gvec_components`. Nothing in the
shapes catches it. `assert_zeta_bases_compatible` now does, and refuses.

### 11.2 Loader wiring (§7.2 closed)

`read_zeta_G_slab` flat-shards μ over the WHOLE mesh
(`P(None,('x','y'),None)`), so the READ needs μ divisible by **P** — a
stricter condition than the overlap's (divisible by p_x and by p_y
separately) and one that 2406 % 4 = 2 violates. Handled with the
loader's own `valid_mu`: read a P-divisible μ window zero-filled past
the logical extent, then slice the pad off BEFORE the reshard, so no
zero ζ row ever reaches the Gram. G is zero-padded 8603 → 8604 (inert).
One reshard from the loader layout to the overlap layout; measured
read+reshard 4.9–7.1 s for 4 q.

### 11.3 THE finding: real ISDF Grams are numerically singular

```
μ_S=606 : cond(G_S) 1.7e18 … 5.7e18 ; eigenvalues -4.53e-05 … 6.93e+11
μ_S=276 : cond(G_S) 5.2e04 … 1.4e05 ; eigenvalues  9.10e+06 … 1.24e+12
```

The 606-centroid basis is linearly dependent on its own sphere — the
negative eigenvalue is round-off on a PSD matrix. **The Cholesky
refusal fired and named the fix rather than producing a plausible wrong
W_S.** This is invisible to the synthetic study (cond ≈ 20) and is
exactly why the run had to be made on real ζ. It is also not a surprise
in context: the ISDF FIT fights the same degeneracy with
`charge_zeta_solve=rank_truncate` and the RCOND dials.

`mode="ls_trunc"` + explicit `rcond` was therefore added — Hermitian
eigendecomposition, keep λ > rcond·λ_max, announce the retained rank:

```
rcond=1e-10   μ_S=606 -> retained 581/581/582  (95.9% of nominal)
              μ_S=276 -> retained 276/276/276  (100%)
```

So ~25 of the 606 production ζ carry no independent information on the
sphere. **The retained rank is physics and is announced, never silently
applied.**

### 11.4 Real-data gates — FINAL (job 7879544, snapshot `srcsnap_20260729_041801_0534350`)

All three pairs, `mode=ls_trunc rcond=1e-10`, numpy reference applying
the SAME truncation, **all rc=0**:

| pair | μ_L→μ_S | ratio | cond(G_S) | retained rank | **W_S vs dense numpy ref** | hermiticity | selection |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 2406→606 | 3.97 | 5.7e18 | 581–582/606 | **8.592e-13** | 1.017e-15 | 3.037e-16 |
| B | 2406→276 | 8.72 | 1.4e05 | 276/276 | **1.260e-12** | 4.774e-16 | 1.519e-16 |
| C | 606→276 | 2.20 | ~1e05 | 276/276 | **1.287e-11** | 8.880e-16 | 1.522e-16 |

Gate 1e-9. **Three independent (μ_L, μ_S) ratios on production ISDF ζ,
each against an independent dense numpy reference.** The correctness
suite is no longer synthetic-only.

Pair A is the important row: it is the rank-deficient case
(cond 5.7e18), it needs the truncated transfer, and it now agrees with a
truncation-matched reference to **8.6e-13**. Its earlier `9.996e-01`
(job 7879536) was **my reference being wrong** — `np.linalg.solve` on a
cond-5.7e18 matrix — not the kernel. Recorded because a gate that fails
for the wrong reason is worse than no gate: it very nearly got written
up as a kernel defect.

For comparison, the same pairs with the plain Cholesky transfer (only
possible for the well-conditioned pairs) gave **1.33e-13** (B) and
**2.96e-12** (C) — so the truncation costs about an order of magnitude
of agreement and buys the rank-deficient case, which is the whole point.

### 11.5 LS vs bare overlap on real data — the owner's question

```
max|W_S raw| / max|W_S ls| :  2.74e+16 (A)   3.69e+17 (B)   1.89e+17 (C)
synthetic equivalent       :  round-trip gap 9.45e+03
```

**The gap behaves exactly as predicted and is far larger on real data.**
`raw` carries the ζ metric's normalization SQUARED; synthetically that
metric was O(1)-ish, whereas the real metric is `N_r` × sphere
truncation with `N_r = 36·36·135 = 174 960`, so the squared scale lands
around 1e17. This is the quantitative confirmation that `mode="ls"` is
not a stylistic preference — the bare congruence is wrong by ~17 orders
of magnitude on production data.

### 11.6 The round-trip does not transfer to real bases

`round-trip mode=ls` = 1.06 (B), 2.37 (C), 1.07 (A) — FAIL. This is
**not** a kernel defect: the representability argument needs the field
to be exactly representable, and its numerical realization needs the
LARGE basis to be well conditioned. At μ_L=2406 the large Gram is itself
strongly rank-deficient, so "exactly representable" is not a numerically
meaningful category and round-off in the degenerate directions is
amplified without bound. On synthetic ζ (cond ≈ 20) the same gate gives
1.54e-15. Recorded as a **diagnostic, not a gate, on real data** — with
the reason, rather than being quietly dropped.


---

## 12. Source-snapshot discipline (cross-campaign hazard)

Propagated from the ladder workstream: jobs submitted with `SRCDIR`
pointing at a live worktree that was being edited.

**It applies to me, and it did.** Jobs 7879519 / 7879526 (the
provocation) and 7879528…7879536 (real ζ) all ran with
`PYTHONPATH=<live tree>/src`, and I edited that tree between their
processes. For a reproducibility experiment this is disqualifying on its
face: an intermittent result and a source change are indistinguishable
after the fact.

What the evidence says (recorded because it is evidence, not because it
is sufficient): the four functions the provocation exercises —
`zeta_overlap_block_reshard`, `zeta_overlap_single_axis`,
`zeta_gram_replicated`, `zeta_gram_single_axis` — are **byte-identical
to the committed `6bf28bc`**, and every edit made during those runs was
additive and confined to `cell_real` / `least_squares_transfer`. That is
the same "module semantics plus luck" argument the ladder agent made,
and it is not a substitute for discipline.

**Infrastructure now in place** (matching the ladder pattern):

* `wk_REL/srcsnap_<UTC>_<githash>/` — `rsync` of `src/` excluding
  `__pycache__`, **verified byte-equal to the tree** by `diff -r` at
  creation (`VERIFY.diff` empty), plus `MANIFEST.sha256` over all 230
  `.py` files and a `PROVENANCE.txt` recording git HEAD and the dirty-file
  list;
* `wk_REL/snapshots/pointers/CURRENT_SRCSNAP` — the pointer file;
* `zproj_provoke.sbatch` and `zproj_realT.sbatch` now resolve
  `SNAP=$(cat CURRENT_SRCSNAP)` **once at job start**, echo the snapshot
  id into the log, **re-verify the manifest at job start** (fatal if the
  snapshot mutated since creation), and set `SRC=$SNAP/src`.

Current snapshot: **`srcsnap_20260729_041040_6bf28bc`**
(`zeta_projection_test.py` sha256 `f44d633879c5aafba706fbd854b6ccf4…`).

**In flight, both snapshot-frozen:**

| job | experiment | snapshot |
|---|---|---|
| **7879540** | provocation, 5×20 reps — the authoritative incidence | `srcsnap_20260729_041040_6bf28bc` (verified at job start) |
| **7879544** | real ζ, all three pairs, truncation-matched reference | `srcsnap_20260729_041801_0534350` (passed explicitly via `ZPROJ_SRCSNAP`) |

`CURRENT_SRCSNAP` was deliberately NOT repointed while 7879540 was live;
the sbatch now honours an explicit `ZPROJ_SRCSNAP` override so a second
experiment can be pinned to a different snapshot without racing the
pointer file. The 7879526 tally in §9.3 stands as provisional.

(The repo HEAD moved from `6bf28bc` to `0534350` during this work; both
snapshots record the HEAD and the dirty-file list they were taken at.)

Standing rule adopted for this workstream: any run whose value depends
on repeated identical execution reads from a snapshot, and the snapshot
id is quoted with the numbers.

### 12.1 Live hazard found in the shared pointer itself

`wk_REL` is shared, and at least two other workstreams created snapshots
in it while I was working — `srcsnap_rel10k_20260729_041117_e63bc8a`
(04:11:17, i.e. BETWEEN my snapshot #1 at 04:10:40 and my job 7879540's
submission at 04:15) and `srcsnap_isdf_window_0408`. **A single shared
`CURRENT_SRCSNAP` pointer file in a shared directory is not safe**: had
that workstream repointed it, my "frozen" job would silently have run
another workstream's source, which is a worse failure than the one the
snapshot discipline exists to prevent — it would look disciplined and
be wrong.

It did not happen here (7879540's log resolves my snapshot, and the
pointer still reads mine), but that is luck again. Mitigations now in
place and recommended as the workstream convention:

1. the sbatch resolves `${ZPROJ_SRCSNAP:-$(cat CURRENT_SRCSNAP)}`, so a
   run can be **pinned explicitly** and never touch the shared pointer —
   job 7879544 was launched this way;
2. the snapshot path is echoed into the log AND its `MANIFEST.sha256` is
   **re-verified at job start** (fatal on mismatch), so a run that read
   the wrong or a mutated snapshot says so in its own log rather than
   being reconstructed afterwards;
3. snapshot directory names carry the workstream, the UTC timestamp and
   the git hash.

Recommendation for the campaign: make the pointer file per-workstream
(`CURRENT_SRCSNAP_<ws>`), or drop the pointer entirely in favour of the
explicit env pin. The verify-at-job-start step should be mandatory
either way — it is three lines and it converts a silent
wrong-source run into a hard failure.


---

## 13. The `.pyc` hole in the snapshot discipline — audit result

Coordinator-propagated from the 10k workstream: a snapshot built with
`cp -a`/`rsync` can ship `src/**/__pycache__/*.pyc` with preserved
mtimes, CPython accepts that bytecode, and a `*.py`-only manifest leaves
it outside the integrity guarantee. **My snapshots had `.pyc` in them.**

### 13.1 Both snapshots DID contain .pyc — by a different mechanism

```
srcsnap_20260729_041040_6bf28bc   11 .pyc,  2 __pycache__,  360 files, manifest 230 (.py only)
srcsnap_20260729_041801_0534350   35 .pyc,  6 __pycache__,  384 files, manifest 230 (.py only)
```

My `rsync` DID exclude `__pycache__` correctly. Every `.pyc` mtime is
**04:17:16–04:17:35**, i.e. AFTER snapshot creation (04:10:49) and after
job 7879540 started. **The bytecode was written INTO the snapshot by its
own consumers at run time** — the snapshot was not immutable for its
lifetime. Different mechanism from the one reported, same exposure
class, and it means a "frozen" snapshot silently stops being frozen the
moment a job imports from it.

### 13.2 Provenance of 7879540: ESTABLISHED

Every `.pyc` header parsed (PEP 552) and its embedded source
mtime+size compared against the snapshot's own `.py`:

```
snapshot #1: 11 .pyc examined, 0 stale, 0 hash-unchecked, 0 orphan
snapshot #2: 35 .pyc examined, 0 stale, 0 hash-unchecked, 0 orphan
```

All are `flags=0` — **timestamp-based, which CPython re-validates on
every import**, not the dangerous `flags=0b01` hash-UNCHECKED kind that
is never validated. And the two modules under test are among them and
both match exactly:

```
common/zeta_projection.py       pyc stamp 1785315941 / 47441  ==  .py 1785315941 / 47441
common/zeta_projection_test.py  pyc stamp 1785316138 / 43844  ==  .py 1785316138 / 43844
```

(snapshot #2's `zeta_projection_test` stamp differs — 1785316669 / 45163
— correctly, since that snapshot carries the truncation-matched
reference.)

The `.py` manifest also still verifies mid-run, so the sources were
immutable across the whole job.

**Verdict: the bytecode 7879540 executed was compiled from that
snapshot's own sources, by that job's first process, and would have been
rejected by CPython had the source differed. The incidence numbers
stand. No re-run needed.**

### 13.3 The coverage gap was real and material

The manifest hashed 230 `.py` of **360 files**. Uncovered:

```
31 .npz   <-- common/minimax_assets quadrature tables: these CHANGE RESULTS
28 .md    24 .cc   12 .sh   11 .pyc   10 .h   7 .txt   1 .json  1 .f90  1 .cu  ...
```

The `.npz` are the point: a `*.py`-only manifest would not have noticed a
swapped minimax quadrature table. This gap existed independently of the
`.pyc` issue.

### 13.4 Fix forward — adopted the 10k helpers, plus one addition

Rebuilt with the shared `srcpin_snapshot`:

```
srcsnap_zetaproj_20260729_043205_b4c7bca
  0 .pyc          349 files hashed (was 230)      31 .npz now covered
```

`zproj_provoke.sbatch` / `zproj_realT.sbatch` now `source
srcpin_resolve.sh` and call `srcpin_resolve` (EXPLICIT `SRCSNAP`, **rc=90
on unset — no pointer fallback**) and `srcpin_verify_end`. The shared
`CURRENT_SRCSNAP` is no longer read; my pointer is per-workstream
(`CURRENT_SRCSNAP_ZETAPROJ`) and is documentation only.

**One addition contributed back:** `export PYTHONDONTWRITEBYTECODE=1` in
the job environment. Excluding `__pycache__` at build time is necessary
but **not sufficient** — as §13.1 shows, the consumers put it back. With
this set, the snapshot stays byte-stable for its whole life and
`srcpin_verify_end` over an all-files manifest becomes meaningful rather
than vacuously passing on files that did not exist at capture.
