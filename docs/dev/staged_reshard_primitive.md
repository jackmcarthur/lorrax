# The staged band-projection reshard primitive (`common.contract_bands`)

*API documentation for `contract_bands_block_reshard` — written for future
agents who will consume it or build more advanced resharding primitives on
its pattern.  Every design choice below carries its evidence pointer; when
you extend this module, hold your change to the same standard
(QUALITY_PATTERNS §4/§9: the optimized HLO is the only ground truth, and
every performance claim ships with its measured domain).*

Primary sources: `src/common/contract_bands.py` (the implementation and
its docstrings), `wk_REL/RESHARD_OVERHEAD_MEMO.md` (the measurement memo
this module encodes), `wk_REL/lgemm_notes.md` (the de-promotion lever),
`wk_REL/contract_bands_notes.md` (this primitive's gate record, jobs
7879008/7879010).  Owner directive 2026-07-28.

## 1. The API contract

```python
from common.contract_bands import (
    contract_bands_block_reshard,      # THE factory
    bands_gemm_ffi_enabled,            # LORRAX_BANDS_GEMM_FFI (factory-time read)
    ensure_grouped_collectives_ready,  # impl=mpi world-first warm-up (§3.5)
)

project = contract_bands_block_reshard(
    mesh_xy,                       # 2-D Mesh; minor axis MUST be axes[1] (§3.2)
    channels="none",               # "none" | "split_reim"
    extra="none",                  # "none" | "leading" | "minor"
    axes=("x", "y"),               # (mu/m axis, nu/n axis)
    divisibility_hint="",          # appended to the refusal message
)
out = project(psi_left, O, psi_right)
```

Mathematics (fixed; `conj` is applied to `psi_left` inside — pass a
pre-conjugated array if you need the unconjugated form):

    out[extra?, k, m, n] = Σ_{s,μ} Σ_{s',ν}  conj(ψ_left)[k,m,s,μ]
                           · O[extra?, k,s,μ,s',ν] · ψ_right[k,s',ν,n]

Operand layout and shardings (global shapes → `PartitionSpec`s; `s`/`s'`
are spinor axes, size 1 is fine):

| operand | shape | spec |
|---|---|---|
| `psi_left` | `(nk, m, s, μ)` | `P(None, None, None, 'x')` |
| `O` | `(nk, s, μ, s', ν)` | `P(None, None, 'x', None, 'y')` |
| `O`, `extra="leading"` | `(E, nk, s, μ, s', ν)` | `P(None, None, None, 'x', None, 'y')` |
| `O`, `extra="minor"` | `(nk, s, μ, s', ν, E)` | `P(None, None, 'x', None, 'y', None)` |
| `psi_right` | `(nk, s', ν, n)` | `P(None, None, 'y', None)` |
| returns | `(nk, m_X, n_Y)` (+E leading/minor) | `P(None, 'x', 'y')` (+`None`) |
| returns, `channels="split_reim"` | `(S_R, S_I)` tuple, each `(nk, m_X, n_Y)` | `P(None, 'x', 'y')` each |

Plans:

* `channels="none"` — one contraction chain at `O`'s own dtype.  Complex
  `O` → single complex chain (the Σ merged-Laplace plan); REAL `O` → the
  de-promoted f64-split chain (§3.3) automatically.
* `channels="split_reim"` — `O` must be complex; it is split elementwise
  into `(Re O, Im O)` BEFORE projection and each real channel rides its
  own de-promoted chain.  This is the Σ two-channel `(S_R, S_I)` plan for
  consumers that weight the channels independently (crossing windows).
  The channel PAIR is the stacked collective payload, so `extra` is
  refused here — stack `(Re O, Im O)` yourself as a real
  `extra="leading"` operand if you need an additional batch axis.
* `extra` — position of a caller-owned stack axis (Σ channels, BSE trial
  blocks, a τ/ω batch).  Both orders are first-class so the choice stays
  a measurement (§3.8).

Refusal conditions (all raise BEFORE any collective, with the fix named
in the message — the broken-promise pattern, QUALITY_PATTERNS §6):

1. `m % p_x != 0` or `n % p_y != 0` — divisibility (§3.6).
2. `mesh.axis_names[-1] != axes[1]` — inverted mesh (§3.2).
3. `channels="split_reim"` with `extra != "none"`, or with a real `O`.
4. Operand-extent mismatches vs `O` (rank and per-axis, message names
   which axis disagrees).
5. Under an EXPLICIT `LORRAX_BANDS_GEMM_FFI=1`: non-CPU mesh;
   missing/unloadable handler (quotes the `probe_target` reason);
   `extra="minor"`; non-f64/c128 operand dtypes (§3.4).  The AUTO default
   (unset/`auto`, 2026-07-29) never refuses — it quietly keeps the native
   lowering wherever the dial cannot apply.

Callable contract: the returned `project` is a `shard_map`'d function —
jit it or trace it into a larger kernel.  The FACTORY must be invoked
synchronously on every rank (it may issue the §3.5 warm-up collective);
kernel-build sites already satisfy this.  Factory-time env reads
(`bands_gemm_ffi_enabled()`) must be part of any consumer's kernel cache
key — see `gw.ppm_tau_kernel`'s `pipeline_key` for the pattern.

## 2. Raison d'être: the (m, n, k) object is never on one rank

The naive lowering of ψ†·O·ψ over a block-sharded O either gathers a
(μ, μ) tile, or materializes the full (m, n, k) product, or lets the SPMD
partitioner pick its own collectives (which historically produced
replicated intermediates and gathers LARGER than the ones they replaced
— QUALITY_PATTERNS §4).  On the LORRAX scaling target — thousands of
low-memory processes, no N_μ² tile permitted on any single rank — all
three are structural failures, not perf bugs.

The primitive therefore stages the contraction so every intermediate
larger than the final tile exists only as a rank-local shard:

    right    = contract (s', ν_local)  of O_local with ψ_right_local
    right_rs = psum_scatter(right, 'y', scatter_dim=n)   # LARGE payload
    left     = contract (s, μ_local)   of conj(ψ_left_local) with right_rs
    out      = psum_scatter(left, 'x', scatter_dim=m)    # small payload

Each `psum_scatter` COMPLETES a μ/ν contraction (the reduction) while
TILING a band axis (the scatter) in the same collective — that fusion is
why staging is required: a single-stage form would need either the full
reduction result (m, n, k replicated) or a (μ, μ) gather somewhere.  The
output emerges `(m_X, n_Y)`-sharded, so every downstream coeff·Σ multiply
stays local; a consumer keeping this layout end-to-end holds a per-rank
buffer ~100× smaller than a replicated Σ_μν (ppm_tau_kernel factory
docstring).  Gate evidence that the doctrine holds in production: the
colltable verdict "NO collective carries a full (mu,mu) tile" at μ=4962
and μ=2475, jobs 7878942/7878977/7879010.

Structural, not advisory: the contractions and collectives live inside
one `shard_map` (`check_rep=False`), where the partitioner cannot hoist
or re-plan them (the per_q lesson, QUALITY_PATTERNS §4).

## 3. Special techniques (each with its evidence)

### 3.1 Stacked-channel single-collective payloads (AK.9)

Every channel/extra slice rides ONE `psum_scatter` per mesh axis, stacked
on the channel axis — never one collective per slice.  Bit-exact by
construction: reduce-scatter sums elementwise over the same replica
groups in the same rank order, so concatenation cannot change values.
Evidence: SPEEDUP_SCORECARD AK.9 (4→2 collectives/τ, project_rs
84.7→47.6 s staged); measured α≈0 caveat — at ~80 MB payloads message
HALVING buys no wall on either transport (memo Sec. 4.5: both stacks are
bandwidth-shaped with ~zero per-call latency), so the stacking's value is
message hygiene and the call-count-heavy BSE regime, claim-scoped.

### 3.2 Large-payload-on-node-local-axis replica-group policy

The ν/'y'-side contraction runs FIRST, so the LARGE partial — the m-full
`(k, s, μ_loc, n)` block — reduce-scatters over the mesh axis whose
replica groups are CONSECUTIVE ranks (node-local pairs at 2 ranks/node on
the production layout); only the small final block rides the strided,
zero-SHM-locality axis.  Evidence: HLO module_0912 replica groups
({8x..8x+7} consecutive for 'y' vs {y, y+8, ...} stride-8 for 'x');
payloads 2×40.9 MB vs 2×0.52 MB post-swap; owner-approved axis-order swap
2026-07-28.  The primitive REFUSES a mesh whose minor axis is not
`axes[1]` rather than silently shipping the big payload over strided
groups — the BSE inner loop shipped exactly that inversion for months
(memo Sec. 6.1 finding #1), which is why this is a refusal and not a
docstring note.  Assumption stated in-code: device order in the mesh
follows process order (true for every production mesh built here).

### 3.3 f64-split de-promotion lowering (the XLA c128-promotion pathology)

XLA (CPU and GPU) lowers a mixed f64 × c128 dot by CONVERTING the real
operand to c128 — at production shape a ~400 MB materialization per
channel — and then runs a full complex GEMM at 2× the mathematically
required flops.  HLO-proven: the promoted
`c128[16,1248,128] dot(%convert_bitcast_fusion ...)` pair in the
reshard-ubench dump `module_0009.jit__project_ri_reduce_scatter`; Eigen
zgemm 295 GF/s vs 1263 GF/s BLAS for the same contraction (memo
Sec. 4.4).  Whenever a real operand would meet a complex one in the
large right GEMM, this module splits the COMPLEX operand into f64 parts
and issues pure-f64 dgemms + one `lax.complex` recombine.  Measured:
project_rs 43.2→38.7 s at nb=128 (job 7878942).  The split is NOT
applied to genuinely complex × complex chains — tried and REFUTED (Eigen
dgemm ~172 GF/s is per-flop BELOW its zgemm's 295; regression measured
job 7878942, patch archived at `wk_REL/lgemm_full_2026-07-28.patch`).

HLO-pin test methodology (copy it for any lowering claim): compile on
the 4-emulated-device mesh and assert on the optimized text — (a) zero
rank≥2 `convert(f64[...])→c128` anywhere, (b) dot dtype/shape classes,
(c) collective count/dtype/payload shapes exactly.
`tests/test_contract_bands.py` and `tests/test_projection_lgemm.py` are
the reference implementations of the pattern.

### 3.4 The gated FFI vendor-BLAS GEMM body (`LORRAX_BANDS_GEMM_FFI`; AUTO default)

Even de-promoted, XLA:CPU's Eigen dots run well below the node's BLAS
rate (thread-pool probe, job 7879008: scaling with the client pool is
near-linear — NOT a pool-wiring defect; the bare dot saturates 1.6–1.9×
below MKL at 28 threads, and the in-module production rate is a further
~2× below the bare dot).  The dial therefore routes ONLY the large right
contraction through the vendor-BLAS GEMM host handler
`lorrax_mklblas_gemm_batch` (`src/ffi/mklblas/cpp/gemm_batch_ffi.cc`):
row-major NN `C[i] = A[i] @ B[i % BB]` — the B-cycling broadcast is what
serves both the plain per-k batch and the extra-stacked batch with the
k-only ψ — dgemm/zgemm dispatched on buffer dtype, vendor-internal
threading under the workstream-AW `MklThreadScope` pin
(`LORRAX_MKLBLAS_THREADS` auto|off|N, strict grammar; dlsym'd — a no-op
on a non-MKL BLAS).  Collectives, channel algebra and the small left
dots (measured 1.6e-3 of the right's flops) are untouched.

**Vendor portability (2026-07-29):** `cblas_?gemm_batch` is an MKL
extension of CBLAS.  The build probes for it with CMake
`check_symbol_exists` against the resolved BLAS headers + link line
(`src/ffi/common/cpp/host/CMakeLists.txt`, mklblas block): present → the
batched entry (preferred); absent → a portable loop of plain
`cblas_{d,z}gemm` calls (standard CBLAS — each GEMM threaded internally
by the vendor).  The handler therefore **works in principle with Intel
MKL or Cray LibSci (batched entry when available, plain-GEMM loop
otherwise); tested with Intel only so far.**

**Default is AUTO (owner order 2026-07-29 — capability detection, not
policy, doctrine #8):** unset/`auto` turns the FFI body ON when the
platform is CPU AND the handler resolves in the host .so, announced once
on rank 0; on CUDA the auto is OFF **silently by design** (XLA:GPU's dot
lowering already dispatches cuBLAS — optimal); auto also quietly keeps
the XLA plan for `extra="minor"` and for non-f64/c128 dtypes (the BSE
fp32-GMRES class).  `LORRAX_BANDS_GEMM_FFI=0` disables.

Rules for an EXPLICIT `=1` (env grants capability loudly; never a silent
downgrade — QUALITY_PATTERNS §8):

* announce once on rank 0; REFUSE with the probe reason if the host .so
  lacks the handler;
* REFUSE on non-CPU meshes — **the CUDA path is deliberately untouched**
  (XLA:GPU's dot lowering already dispatches cuBLAS, which is optimal);
* REFUSE `extra="minor"` (the contracted axis is not reachable by a
  strided batched GEMM without a full-tile transpose copy);
* REFUSE non-f64/c128 dtypes (the handler dispatches on the buffer
  dtype; the message names the fix);
* read at FACTORY time → consumers key their kernel caches on it
  (`bands_gemm_ffi_enabled()` resolves auto, `bands_gemm_ffi_mode()`
  exposes the raw grammar).

Measured (job 7879010, P=64, cache-cold, coll=mpi): nb=128 staged
project_rs 29.407→19.622 s (−33%), prod sigma.exec 58.313→49.224;
nb=256 composed 20.565→14.162 / 35.234→29.979.  Parity exact-0 .dat,
h5 ≤2.5e-14 eV, reduce-scatter payloads byte-equal off-vs-on.  Build:
`config/frontera/build_ffi_host.sh` (the TU rides the existing MKL link
line; see the mklblas block in `src/ffi/common/cpp/host/CMakeLists.txt`).

**Configure-time probe caveat (2026-07-29, cost a gate cycle):** the
`check_symbol_exists` probe links a try_compile EXECUTABLE, where `ld`
defaults to `--no-allow-shlib-undefined` and demands the *whole*
shared-library closure resolve — which the real target (a `.so`) never
has to.  On Frontera that produced a FALSE NEGATIVE twice (jobs
7879278/7879281 both compiled the slow plain-GEMM loop against an MKL
that *has* the batched entry): first from BLACS's open `MPI_*`
references, then — after adding `libmpi` to close them — from `libmpi`'s
own `fi_*@FABRIC_*` references (libfabric is on `LD_LIBRARY_PATH` at run
time, not on the link search path).  The probe therefore sets
`CMAKE_REQUIRED_LINK_OPTIONS=-Wl,--allow-shlib-undefined`, which
tolerates undefined symbols *inside the dependency libraries* while
still hard-failing on an undefined reference from the probe's own
object — exactly the question being asked.  **If you port this to a
non-GNU linker and the flag is rejected, the probe simply fails and the
portable plain-GEMM loop is compiled: slower, never wrong.**  Read the
configure log line (`mklblas: GEMM host handler ON — batched entry …`
vs `… — plain cblas_?gemm loop …`) rather than assuming.

**Platform reach of this module (CPU/GPU sweep, 2026-07-29).**  The
primitive itself is backend-neutral and *supported on both platforms*:
the `shard_map` bodies are `jnp.einsum` + `jax.lax.psum_scatter` +
`jax.lax.complex` only, so a CUDA mesh lowers the same staged chain
through NCCL reduce-scatter.  The three platform-conditional pieces, and
what each does on GPU:

* **the FFI GEMM body** — never lowered on GPU.  `lorrax_mklblas_gemm_batch`
  is registered in `ffi_loader`'s **host table only**, so even a forced
  probe on CUDA reports *unknown target*; auto resolves OFF from
  `JAX_PLATFORMS` before any probe, the factory re-checks
  `mesh.devices.flat[0].platform`, and an explicit `=1` refuses.  A GPU
  user never sees this dial engage and never needs to unset it — the
  native `dot` → cuBLAS lowering is what runs, by design.
* **`ensure_grouped_collectives_ready()`** — a no-op on GPU: it returns
  early unless `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, a CPU-backend
  variable (§3.5).  GPU meshes have no equivalent ordering contract here.
* **the mesh-minor-axis refusal** (§3.2) — *stays* on GPU and is stated
  in device-order terms, not CPU terms: only the LAST mesh axis has
  consecutive-rank replica groups on the standard process-ordered device
  layout, so that is where the large partial must reduce-scatter.  On a
  multi-node GPU mesh built row-major from `jax.devices()` the minor axis
  is likewise the intra-node one, so the same `axes=(major, minor)` rule
  applies; a hand-permuted GPU device mesh will hit the refusal and must
  pass `axes` matching its own layout.

The one GPU gap to be honest about: **no GPU execution of this primitive
has been run.**  Everything above is a code + platform-table reading, not
a measurement — the campaign's meshes are all CPU.  Nothing GPU-specific
is *claimed* to be fast, only to resolve sanely.

### 3.5 The impl=mpi world-collective-first warm-up contract

Under `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, XLA creates grouped
cliques lazily and its `MPI_Is_thread_main` check passes only after a
WORLD-clique collective has first-touched the runtime; a process whose
FIRST collective is a grouped psum_scatter dies deterministically with
`UNKNOWN: MPI: Communicator requested from a thread ...`.  This is an
ORDER contract, not an async one — `JAX_CPU_ENABLE_ASYNC_DISPATCH=false`
does NOT avoid it.  Evidence: job 7878862 step 1 (52/64 ranks dead at
the first grouped collective), 7878883 probe (world-psum-first fixes it;
async-off does not) — memo Sec. 4.2.  Production never sees it only
because early sync barriers happen to be world collectives.

The factory calls `ensure_grouped_collectives_ready()` — one
world-spanning `common.collectives.barrier` collective, once per
process, only under impl=mpi multi-process (~60 ms warm, ~2 s cold) — so
every consumer (tests, benches, future BSE drivers) inherits the
contract instead of rediscovering the failure.  Standalone code that
issues grouped collectives WITHOUT this primitive should call the helper
directly.  The L3 FFI direct-MPI tier deletes the contract entirely
(explicit communicator lifecycle) and slots behind this same interface.

### 3.6 Divisibility guards / resolve-time refusals

The two psum_scatters tile m over p_x and n over p_y; an indivisible
window crashes cryptically deep inside psum_scatter (job 7874338 died
there).  The wrapper refuses at trace time with the fix named: pad m and
n INDEPENDENTLY per axis (Σ callers: `gw.ppm_sigma.pad_sigma_window`),
never to the p_x·p_y product (up to 3.16× tile waste — audit fix/zq).
`divisibility_hint` lets a caller append site-specific context (Σ
appends the meta.py b_id_4 note).  General rule for extensions: every
precondition a collective needs is checked BEFORE the collective, with
the caller's fix in the message.

### 3.7 In-place / donation practices

The FFI GEMM declares NO `input_output_aliases`: a `(BA, M, N)` GEMM
output can never legally alias a `(BA, M, K)`/`(BB, K, N)` operand
buffer.  Contrast the mklfft handlers (`{0:0}` — shape-preserving
transforms, where aliasing is the terminal form of donation).  Do not
cargo-cult aliases onto shape-changing ops.  Donation of the O operand
by callers remains useful (the Σ staged path donates `sigma_k`), but
remember donation is inert inside fused jits (scorecard audit row (d)) —
it acts only at top-level dispatch.

### 3.8 extra/omega-dim order: the measured verdict

Both stack orders are first-class in the API so the choice stays a
measurement.  Bench (job 7879008 cell `ubench`,
`wk_REL/cbands_ubench.py`, production local shapes, 2×2 emulated mesh):
MINOR wins by 1.2% (E=2) and 2.5% (E=4) — noise-class; per-E cost is
flat in both orders (no stacking penalty).  Measured domain: single
node, shared 28-core pool, SHM collectives — this prices the local
GEMM/stack lowering (71% of the production row), not the wire.  Default
is `"leading"`: it matches the production stacked-collective payload
tables byte-for-byte AND is required by the FFI GEMM plan (§3.4).
Numbers recorded in `wk_REL/contract_bands_notes.md` §4.

## 4. How to gate any future change to this module

Gate order is fixed: unit gates first, A/B only on PASS (never past a
failed unit gate).

1. **Value-parity classes.**  Know which class your change is in and say
   so: (a) bit-exact (pure data movement: stacking, indexing) — assert
   byte equality; (b) value-level identical (any reassociation: GEMM
   splits/merges, contraction-order swaps, BLAS backends) — gate at
   1e-12 (unit: 1e-14 at small shapes) and do NOT claim bit-exactness;
   the h5 tensor compare's expected band is the ULP class (~1e-14 eV at
   these decks).
2. **HLO pins** (§3.3 methodology) on the 4-device pattern: collective
   count/dtype/payload shapes, forbidden-promotion converts, dot/
   custom-call classes.  A lowering claim without a compiled-HLO
   assertion is unverified (QUALITY_PATTERNS §4).
3. **Colltable** on the production dumps: reduce-scatter payloads
   byte-equal to the recorded tables (chmerge tables at both nb), zero
   all-gathers in the τ modules, and the "NO collective carries a full
   (mu,mu) tile" verdict at every μ.
4. **Cache-cold rule (AY.2)**: collective-table and HLO gates are valid
   ONLY from a cache-cold compile (`ISDF_JAX_CACHE_DIR=""`, fresh dump
   dir) — a warm cache leaves the rank-0 dump incomplete and the gate
   silently vacuous.
5. **Restart-gated A/B at production scale** (AC.4 harness, J.7): both
   nb shapes, staged + prod rows, .dat parity vs the frozen baseline
   dirs, jobids recorded in the workstream notes with the measured
   domain (QUALITY_PATTERNS §9).
6. Factory-time env flags added here MUST be appended to every
   consumer's kernel cache key (grep for `pipeline_key`).

Reference gate runs to imitate: jobs 7879008 (unit) and 7879010 (A/B) —
scripts `wk_REL/cbands_gate.sbatch` / `cbands_ab.sbatch`.

## 5. Intended second consumer: the BSE tree

The BSE inner loop is the highest-value adopter — ~14× sigma's grouped
collective count per solve with the same payload class, and its decode
currently ships the LARGE payload on the strided axis (the exact
inversion §3.2 refuses).  The precise, per-site adoption map — reviewed
by no one yet, NOT wired — is `wk_REL/contract_bands_notes.md` §6:

* `bse_stack_matvec._w_stack` decode (bse_stack_matvec.py:126-131): a
  per-trial primitive call is the movement-only axis fix; prerequisite
  is producing the conv output in k-leading layout, which is the SAME
  layout the flat-k FFT-FFI backend reads natively — the two adoptions
  compose.  Trial-stacking the collectives (4b→4) needs a two-phase
  primitive variant (stage-1 GEMM inside the scan, stacked collectives
  after) — an API decision that belongs to the owner, flagged there.
* `bse_ring_comm._apply_W_from_T` (TDA and non-TDA): **ADOPTED
  2026-07-29** (owner order) — clean `extra="leading"` drop-in: those
  paths already materialize the b-stacked T, so the stack axis is free,
  and the adoption converts the partitioner-chosen collectives of the
  historical einsum pair into the structural stacked psum_scatter chain
  (large payload on the node-local 'y' groups, one collective per mesh
  axis for all b trials).  Rank-local transposes bridge the conv layout
  (`(b,M,N,t,s,k)` → the primitive's k-leading O) and ψ_v; they compose
  with the future flat-k conv layout of the 6.1 route (a).  Value-level
  identical (1e-12 gate class).  The fp32-GMRES path's complex64
  operands ride the primitive's XLA lowering (the FFI dial's dtype
  boundary — auto quietly falls back, explicit `=1` refuses).  Both
  builders lower unchanged on a CUDA mesh (the body is `einsum` +
  `psum_scatter`; the GEMM dial is CPU-only and simply stays off there).
* **Not** part of that adoption, and easy to over-read: the same
  2026-07-29 sweep routed BSE's remaining raw `jnp.fft` calls through
  `fft_helpers.local_ifftn3`/`local_fftn3`.  Those are **aliases of
  `jnp.fft`** — call-site hygiene ("one source for the local FFT"), NOT
  a backend switch.  `LORRAX_FFT_FFI` only reaches `make_flat_k_*` sites
  and BSE has none, so BSE is unaffected by that flag (and by the cuFFT
  mirror) on **both** platforms until a real flat-k adoption lands.  The
  aliases being dtype-agnostic is load-bearing: BSE's fp32-GMRES FFTs
  are complex64 and the flat-k FFI handlers are c128-only on cpu *and*
  CUDA, so routing them at the flat-k layer would have to refuse them.
* **Layout warning for whoever wires §6.1 route (a)** (measured
  2026-07-29, jobs 7879363/7879370; HLO dumps in
  `wk_REL/fftlayout_hlo/`).  BSE's conv FFTs are minor-most and XLA:CPU
  gives them **zero** transposes/copies — 0 bytes moved at all six real
  call sites, sharded and unsharded.  A k-major operand is a different
  story: the same 24.92 MB tile through the same `fft` moves **124.60 MB
  in transpose+copy (5.0×)**, and the fft's operand is a fusion literally
  named `%transpose_copy_fusion`.  Consequence: relabelling BSE to
  k-leading is only correct *together with* `LORRAX_FFT_FFI` (whose
  handler reads k-major via stride descriptors).  The XLA-path variant of
  that relabel — §6.1 route (b) — is **refuted**: it buys the layout cost
  with none of the relief.  Do not stage route (a) as "layout first,
  backend later".
* `vq_interp.make_eval_vq`: honestly NOT a contract_bands instance
  (outer-product class — the contracted axis is replicated, the sharded
  axes are outputs).  Needs a structural sibling, not this primitive.

If you build that sibling (or the two-phase variant, or the L3 FFI
direct-MPI collective tier that slots behind this interface): keep the
§2 doctrine and the §3.2/§3.5 contracts, write the §3.3-style HLO pins
first, and record your numbers with their domain next to these.
