# RESHARD_OVERHEAD_MEMO — the multi-stage (mu_X,nu_Y)->(m_X,n_Y) projection reshard as a reusable primitive

**VERDICT (A): the reshard's 43.2 s is NOT collective-bound — 71% of it is
the local projection GEMM running 4.3x below the node's BLAS roofline
(XLA:CPU promotes the f64 channel operands to c128 and runs Eigen zgemm at
295 GF/s vs 1263 GF/s measured BLAS).  Ordered levers, all
microbench-sized: (1) BLAS-grade projection GEMM (FFI MKL or f64-split
relowering): 43.2 -> 19.8 s; (2) FFI direct-MPI reduce-scatter chain
(deletes the jax-mpi shim's measured 1.86x staging factor): -> 15.5 s;
(3) strip overlap at the FFI tier (58% of comm hidden, measured): ->
12.9 s.  Collectives after (2)+(3) sit at the measured Intel-MPI floor;
skew is additive and small (~2 s); jax-level pipelining/double-buffering
is dead on XLA:CPU (runtime serializes).  Full accounting: Sec. 5.**

Workstream: wk_REL follow-on (2026-07-28).  Tree: /work2/08271/jackmc/frontera/lorrax @ 5918cf6.
Author scope: analysis + measurement only — no production-code changes; one dev-queue
microbench job (7878862) + a BSE-tree audit (findings only).

---

## 1. Context and current measured state (all numbers from existing artifacts)

The pattern under study is `gw/ppm_tau_kernel.py::_project_ri_local` (built by
`_make_project_ri_reduce_scatter`, ppm_tau_kernel.py:81-264): the psi* sigma psi
projection that turns the (mu_X, nu_Y)-sharded 399 MB/rank sigma_k tile into an
(m_X, n_Y)-sharded Sigma_mn, multi-staged deliberately so no (m, n, k)- or
(mu, mu)-sized object is ever materialized on one rank.  Structure per tau:

    right = einsum('ksxty,ktyn->ksxn')            (contract s', mu_Y-local)
    psum_scatter(stack[re,im], 'y', dim=n)        payload 2x40.9 MB, consecutive-rank groups
    left  = einsum('kmsx,ksxn->kmn')              (contract s, mu_X-local)
    psum_scatter(stack[re,im], 'x', dim=m)        payload 2x0.52 MB, stride-8 groups

Production cell: AQ 4962c / P=64 (8x8 mesh, 32 nodes x 2 ranks), nb=128,
mu_pad=4992 (mu_loc=624), nk=16, ns=2, 176 tau; certified AS.7 mpi-collectives
stack (JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi via thread-MULTIPLE MPIwrapper,
Intel-MPI mlx/UCX provider).

Current state after the FFT-FFI landing (commit 5918cf6, job 7878727):

| row (pass c prod / pass d staged) | value |
|---|---|
| sigma.exec (production, FFI fused) | **71.906 s** |
| staged dispatch total (pass d) | 65.729 s |
| sigma.tau.w_phase | 5.683 s |
| sigma.tau.G_build | 10.971 s |
| sigma.tau.GW_conv_ffi | 4.992 s |
| **sigma.tau.project_rs** | **43.246 s = 245.7 ms/tau = 66% of dispatch** |
| production d2h_wait (absorbs device wait) | 61.6 + 6.0 s |

History of movement-only levers already applied to this site (sigma_perf_results.md):
AK.9 stacking (4->2 collectives/tau, project_rs 84.7->47.6 s staged), axis-order
swap (large payload moved to the node-local 'y' groups; parity-clean,
no measurable win at this shape), monolithic fusion tried and REFUTED (HLO
transpose regression), FFT trio removed from the wall by the MKL-DFTI FFI
(191.9 -> 4.99 s).  The projection is now the sigma wall and is slated as a
CORE reusable pattern for the BSE routines (Section 6).

## 2. The floor, computed (paper model — to be replaced by measured values in Sec. 4)

### 2.1 Wire floor at the exact payloads/replica-groups

Mesh 8x8 over 32 nodes x 2 ranks; device (x,y) = process 8x+y (HLO module_0912
replica groups: 'y' = {8x..8x+7} consecutive = 4 nodes as node-local pairs;
'x' = {y, y+8, ..., y+56} stride-8 = 8 distinct nodes).  Frontera HDR-100
NIC ~= 12.5 GB/s/node, shared by the node's 2 ranks.

Ring reduce-scatter of per-rank payload S over g ranks moves (g-1)/g x S per
rank.  Per tau:

- 'y' leg: S = 81.79 MB (the re/im stack), g=8 over 4 nodes; with node-local
  pairs adjacent in the ring, each node's NIC carries 7 x S/8 = 71.6 MB
  -> 5.7 ms at 12.5 GB/s.
- 'x' leg: S = 1.05 MB, g=8, no locality; each node carries 2 ranks' worth
  = 1.84 MB -> 0.15 ms.  Serial after 'y' (data dependency).

Wire floor ~= 5.9 ms/tau -> **1.03 s / 176 tau** ("~1-2 s" allowing protocol
inefficiency).  Direct-MPI measured value: phase M (Sec. 4).

### 2.2 GEMM + cast floor at the production local shapes

Per rank per tau: right-einsum 2 channels x 16 batches x (1248x1248)@(1248x128)
= 2.55e10 real FLOPs if computed real x complex (4 flops/MAC); 5.1e10 if XLA
promotes the real operand to c128 (phase G measures which).  At a realistic
28-thread ~1.2-1.5 TF/s f64: **19-38 ms/tau**.  The jnp.real/jnp.imag channel
extracts stream ~1.2 GB/tau: ~10 ms at ~110 GB/s/rank.  Left-einsum negligible
(1.6e-3 of right).  GEMM+cast floor ~= **30-50 ms/tau = 5.3-8.8 s / 176 tau**
(phase G pins it).

### 2.3 The gap to explain

measured 245.7 ms/tau  vs  floor ~= (6 + 30-50) ms/tau
=> **~190-210 ms/tau (~35 +- 2 s / run) unattributed** — the microbench's job:
split it between (a) per-call shim overhead of the jax/XLA-CPU collective
path (thunk + rendezvous + MPIwrapper + copy-in/out), (b) wait-from-skew
(cross-rank spread of the pre-collective stages), (c) genuine wire/algorithm
inefficiency above the ring model.

## 3. Microbench design (job 7878862; scripts reshard_ubench{_jax,_mpi}.py, .sbatch)

Node-count MATCHED to production: 32 nodes x 2 ranks = P=64, dev queue.
Justification: the 'y' groups' SHM/wire mix (node-local pairs inside 4-node
groups) and the 2-ranks-per-NIC contention are exactly the production
collective's environment; fewer nodes (more ranks/node) would change both and
invalidate the wire attribution.  Payloads, replica groups, dtype (c128),
stacking, scatter dims, and the 176-rep count all match the production HLO.

Phases (jax step = the AS.7 mpi-collectives cell, fresh compile-cache dir so
the rank-0 HLO dump is complete per the AY.2 cache-cold rule):

- **F** the production projector itself, imported from the tree
  (`_make_project_ri_reduce_scatter(mesh)`), synthetic sharded data at exact
  shapes/specs: free-run x176 + barrier-aligned x64.  Closes the loop: if
  F_aligned reproduces ~245 ms/tau the whole cost lives in the projector body;
  if it is much cheaper, production pays skew/queueing the bench can quantify.
- **A** the BARE stacked psum_scatter chain (einsums replaced by a slice that
  preserves the y->x data dependency): aligned x176, free-run x176 (does the
  runtime pipeline back-to-back collectives?), per-axis splits, and a 5-point
  y-payload size sweep (10-164 MB) for an alpha/beta (per-call latency /
  bandwidth) fit -> shim+wire per-call cost, separated.
- **B** chain + injected per-rank sleep jitter U[0,delta],
  delta in {0,5,10,25,50,100} ms x40 reps — is skew absorbed additively
  (wall ~= E[max sleep] + bare) or amplified by the collective path?
  delta bracket chosen to cover the measured tau-kernel imbalance; the
  in-job measured value comes from phase G (below) since the production
  timing table is rank-0-only (sigma_perf_candidates.json names that gap).
- **D** the UNSTACKED pre-AK.9 variant (4 collectives/tau) x64 — what stacking
  is worth at exact shapes (calibrates the per-collective fixed cost from a
  second direction).
- **E** overlap probe: 20 reps of (in-flight y-scatter || independent local
  GEMM) vs each alone — does XLA:CPU overlap an issued collective with local
  compute at all?  Gates levers (i)/(ii): if the runtime serializes, strip
  pipelining inside jax buys ~nothing and the lever moves to the FFI tier.
- **G** per-rank spread of the production right-einsum at exact local shapes
  (f64 x c128, both channels), barrier-aligned x40 with per-rep per-rank lists
  -> the measured compute imbalance that phase B's delta is judged against;
  also pins the GEMM floor (and the promotion question) by measurement.
- **M** (separate srun step, pure mpi4py, NO jax): the direct-MPI floor.
  Same groups via comm.Split (y=rank//8 consecutive, x=rank%8 stride-8), same
  payloads, MPI_Reduce_scatter_block c128 SUM: aligned x176, free x176,
  per-axis, Allreduce comparison, size sweep, and two Ireduce_scatter overlap
  probes (Wait-only vs Test-polling between GEMM chunks) — sizes lever (iii)
  (an FFI direct-MPI collective adds only the ~10-100 us jax.ffi call cost on
  top, mklfft experience) and the strip-pipelined variant of it.

## 4. Results

### 4.1 Direct-MPI floor (job 7878862 step 2 — pure mpi4py, COMPLETE, rc=0)

All values rank-0, aligned = world-barrier before every rep; 176 reps for the
chain, mlx provider, exact payloads/groups.

| phase | mean (ms) | p50 | p90 | max |
|---|---|---|---|---|
| M_chain_aligned (y-RS 81.8 MB + x-RS 1.05 MB) | **24.77** | 24.57 | 25.07 | 31.4 |
| M_chain_free (no barriers, back-to-back) | 25.08/rep | — | — | — |
| M_y_only (81.8 MB RS over consecutive-8) | 23.85 | 23.84 | 23.89 | 24.2 |
| M_x_only (1.05 MB RS over stride-8) | 0.33 | 0.33 | 0.39 | 0.58 |
| M_y_allreduce (81.8 MB AR, comparison) | 30.82 | 30.15 | 30.75 | 58.3 |
| M_sweep 10.2/20.4/40.9/81.8/163.6 MB | 2.91 / 5.84 / 12.00 / 24.08 / 53.75 | | | |
| M_gemm_ref (numpy zgemm at right-einsum shape) | 44.9 | | | |
| M_overlap_wait (IRS + full GEMM + Wait) | 59.84 | | | |
| M_overlap_poll (IRS + chunked GEMM + Test) | 54.80 | | | |

Derived:
- **alpha/beta fit of the y-leg: slope 0.331 ms/MB (3.0 GB/s effective
  per-rank), intercept ~0 (-0.5 ms)** — the DIRECT-MPI reduce-scatter has
  NO measurable per-call latency; it is bandwidth-shaped end to end.
- Effective NIC utilisation: per-rank (7/8)x81.8 MB in 23.9 ms = 3.0 GB/s out;
  x2 ranks/node = 6.0 GB/s/NIC direction = 48% of HDR-100 — Intel MPI's
  Reduce_scatter_block runs within ~4x of the ideal-ring wire model
  (reduction arithmetic + algorithm choice), and reduce-scatter beats
  allreduce by 1.29x as expected.
- **Direct-MPI floor for the production chain: 24.8 ms/tau = 4.36 s / 176 tau.**
- Overlap: naive IRS+Wait hides only 13% (Intel MPI without async progress);
  **Test-polling between GEMM chunks hides 58% of the comm** (13.9 of
  23.8 ms) — the strip-pipelined direct-MPI design has real headroom on this
  fabric without any progress-thread machinery.

### 4.2 A shim finding before any timing: the XLA mpi-collectives thread contract

The first jax-step attempt (job 7878862 step 1) FAILED deterministically
(52/64 ranks) at the FIRST grouped collective:

    JaxRuntimeError: UNKNOWN: MPI: Communicator requested from a thread that
    is not the one MPI was initialized from. Multiple threads/devices per
    process are not yet supported.

Mechanism (upstream xla/backends/cpu/collectives/mpi_collectives.cc, string
present in the installed libjax_common.so): `CreateCommunicators` refuses
unless `MPI_Is_thread_main` — communicators for each clique are created
LAZILY at first collective execution (cpu_cliques.cc, process-wide cache,
"potentially from multiple threads").  The 7878883 probe cell RESOLVED the
mechanism empirically — it is an ORDER contract, not an async-dispatch one:

- probe (default dispatch, impl=mpi): world-clique psum FIRST -> grouped
  chain then works on all 64 ranks (world 2.0 s cold, grouped chain 1.03 s
  cold, 2nd call 60 ms) — rc=0, zero FAILs;
- step 1b (async dispatch OFF, grouped collective first): SAME failure —
  JAX_CPU_ENABLE_ASYNC_DISPATCH=false does NOT satisfy the check;
- i.e. a GROUPED clique can only be created after a WORLD-clique collective
  has first-touched the mpi collectives runtime.  Production never sees
  this only because every production run happens to issue world-clique
  collectives (sync barriers, gathers) before its first grouped
  reduce-scatter.

This is a REUSABILITY finding for the primitive: the impl=mpi shim carries
an undocumented "world-collective-first" init-order contract (same family
as the AS.4b FUNNELED race); any standalone consumer (tests, BSE drivers,
microbenches) must warm a world-clique collective first or dies with an
UNKNOWN error at the first grouped psum_scatter.  The L0 helper should
encode that warm-up; the L3 FFI direct-MPI tier removes the contract
entirely (explicit communicator lifecycle).

### 4.3 jax-side phases — gloo/ib0 cell (job 7878883 step 1c, rc=0, full suite)

Steady-state values (median of per-rank means; the first-epoch
A_chain_aligned row carries gloo connection-establishment transients —
mean 413 ms, p90 768 ms — quote the B_skew0 and sweep rows for steady
state):

| phase (gloo/ib0) | ms/rep | note |
|---|---|---|
| chain steady (B_skew0 = barrier-aligned) | **171.3** | y+x stacked chain |
| A_y_aligned (81.8 MB RS) | 157.8 | |
| A_x_aligned (1.05 MB RS) | 2.9 | |
| A_chain_free (back-to-back) | 174.2 | = aligned: NO pipelining gain |
| D_unstacked (4 collectives) | 166.6 | = stacked: alpha ~ 0 on gloo too |
| A_sweep 10.2->163.6 MB | 20.3/40.5/79.3/158.2/317.8 | **slope 1.94 ms/MB = 0.52 GB/s eff/rank, intercept 0.3 ms** |
| **G_gemm_spread (pure local right-einsum, jax)** | **172.9** | collectives-INDEPENDENT |
| F_proj_aligned (production projector) | 450.0 | |
| F_proj_free | 439.8 | 77.4 s / 176 reps |
| E overlap frac (in-flight RS vs local GEMM) | **-0.04** | XLA:CPU fully SERIALIZES |

Skew injection (B, chain; added wall vs E[max sleep] over 64 ranks):

| delta (ms) | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| added wall (ms) | 3.1 | 6.4 | 18.7 | 40.9 | 87.2 |
| E[max sleep] | 4.9 | 9.8 | 24.6 | 49.2 | 98.5 |

-> the chain absorbs injected skew SUB-additively (0.63-0.89 x E[max]); no
amplification, no timeout behavior.  Measured compute imbalance (G, exact
production GEMM, 64 ranks x 40 aligned reps): cross-rank spread p50 16.5 ms
/ p90 21.9 / max 26.4 on a 173 ms GEMM (~10%); mean-of-maxima 184.8 vs
mean-of-means 173.1 => the straggler tax if a collective gates every tau is
**~12 ms/tau ~ 2.1 s/run** — real but small.

### 4.4 THE headline: the projection GEMM, not the collectives, owns the gap

Two independent measurements of the SAME contraction at the SAME local
shapes (16 batches of (1248x1248)@(1248x128), both channels):

| implementation | ms/tau | effective GF/s |
|---|---|---|
| jax/XLA:CPU jnp.einsum (f64 x c128, production code path) | **172.9** | ~295 |
| numpy BLAS zgemm (venv, 28 threads, same flops) | **40.4** | ~1263 |

The production per-tau projection cost decomposes (impl=mpi production,
245.7 ms/tau) as ~173 ms local GEMM + ~25 ms direct-MPI-floor collectives +
~47 ms residual (re/im cast extracts ~2x200 MB, left einsum, stack copies,
shim-over-floor, straggler tax ~12) — **the collectives are ~10-25% of the
row; the XLA:CPU GEMM is ~70% and is running 4.3x below the BLAS roofline
of the very same machine**.  (mpi-shim chain cost measured in the 7878907
cell below; probe datum: warm chain single-shot 60 ms.)

Mechanism, HLO-proven (bench dump module_0009.jit__project_ri_reduce_scatter,
the tree's own projector): XLA **promotes each f64 channel operand to c128**
(`convert(%real/%imag)` fused into `%convert_bitcast_fusion`, a ~400 MB
materialization per channel) and issues full complex dots
`%dot,%dot.1 = c128[16,1248,128] dot(...)` — i.e. **2x the mathematically
required flops through Eigen's zgemm at ~295 GF/s**, where the same
promoted zgemm through the venv BLAS runs at ~1263 GF/s.  Two exits, in
order of invasiveness:
  (a) movement-only re-lowering INSIDE jax: decompose the COMPLEX psi
      operand into re/im f64 parts -> 4 (batched) f64 dgemms per tau, no
      promotion copies, no change to the OWNER-HELD re/im channel algebra
      (only the complex operand's representation changes; each channel's
      einsum stays per-channel).  Sized: 2.55e10 real flops at Eigen dgemm
      rates; must be measured — Eigen f64 dot is typically several x its
      zgemm.
  (b) the FFT-FFI precedent verbatim (ffi_fft_proto_notes.md): an FFI MKL
      GEMM handler (MklThreadScope, input_output_aliases) reading the tile
      in place — guaranteed BLAS-grade (~40 ms/tau measured), backend
      dial stays two-plan.

### 4.5 jax-side phases — impl=mpi cell (job 7878907, rc=0 all steps — the production stack)

| phase (impl=mpi, world-barrier-first) | ms/rep | note |
|---|---|---|
| A_chain_aligned (bare stacked chain) | **49.9** | vs 25.4 direct-MPI |
| A_chain_free | 50.2 | = aligned: no pipelining gain |
| A_y_aligned / A_x_aligned | 48.5 / 1.10 | |
| D_unstacked (4 collectives, pre-AK.9) | 48.2 | = stacked (alpha ~ 0) |
| A_sweep 10.2->163.6 MB | 6.0/12.1/25.2/49.6/101.1 | **slope 0.620 ms/MB (1.61 GB/s eff/rank), intercept -0.5 ms** |
| G_gemm_spread (pure local) | **173.5** | = gloo cell: collectives-independent |
| **F_proj_aligned / F_proj_free** | **235.0 / 244.6** | **production project_rs = 245.7: REPRODUCED to 0.5%** |
| E overlap frac | **-0.15** | XLA:CPU serializes comm vs compute |

Skew injection (chain, added wall vs E[max sleep]): 3.2/7.1/21.3/45.4/94.1 ms
at delta=5/10/25/50/100 vs E[max] 4.9/9.8/24.6/49.2/98.5 — additive
absorption again (0.65-0.96x), no amplification.  G cross-rank spread:
p50 18.2 ms, straggler tax (mean-of-maxima - mean-of-means) 12.3 ms/tau.

Shim characterization: the jax mpi shim is ALSO bandwidth-shaped (intercept
~0 — there is NO large per-call latency to amortize), but at **1.86x the
raw-MPI byte cost** (0.620 vs 0.334 ms/MB — consistent with one extra
full-payload staging copy each way).  Stacking (AK.9) and unstacking are
time-neutral at this payload on BOTH stacks — alpha ~ 0 means message-count
halving buys no wall here; its value stays the message-hygiene/nb^2-growth
argument (claim-scope per house rule).

## 5. Gap decomposition and VERDICT

The free-run production projector in the bench (244.6 ms/tau) reproduces
the production project_rs row (245.7 ms/tau = 43.2 s / 176 tau) to 0.5%,
so the bench decomposition IS the production decomposition:

| component | ms/tau | x176 (s) | share | evidence |
|---|---|---|---|---|
| right-einsum GEMM (XLA promoted zgemm) | 173.5 | 30.5 | **71%** | G, both cells |
| stacked psum_scatter chain via jax-mpi shim | 49.9 | 8.8 | 20% | A_chain |
| — of which raw Intel-MPI reduce-scatter | (25.4) | (4.5) | 10% | M_chain, 3 jobs |
| — of which shim staging overhead | (24.5) | (4.3) | 10% | A minus M |
| casts (jnp.real/imag) + left dots + stacks | 11.6 | 2.0 | 5% | F_al - G - A |
| dispatch-queue / straggler skew | 9.6 | 1.7 | 4% | F_free - F_al; G spread 12.3 |
| **total** | **244.6** | **43.0** | 100% | vs 245.7 production |
| (ideal-ring wire model, reference) | 5.9 | 1.0 | — | Sec 2.1 |

The task's working hypothesis (shim per-call latency + skew dominate) is
REFUTED: both collective stacks have ~zero per-call latency (intercepts
~0), skew is absorbed additively and is worth only ~2 s, and the wire is
10%.  **The gap owner is the local projection GEMM, which XLA:CPU runs at
295 GF/s (promoted c128 Eigen dot + 400 MB/channel convert copies) where
the same machine's BLAS runs the identical contraction at 1263 GF/s.**

**VERDICT: (A) — win identified.**  Measured-basis projections at
nb=128/P=64 (each lever independently measured above; combined by
substitution into the table):

| lever stack | project_rs (s /176 tau) | sigma.exec (s, est) |
|---|---|---|
| today (5918cf6) | 43.2 | 71.9 |
| + L-GEMM at BLAS rate (173.5 -> 40.4 ms) | 19.8 | ~48 |
| + L3 FFI direct-MPI chain (49.9 -> 25.4 ms) | 15.5 | ~44 |
| + L1 strip overlap at FFI tier (58% of 25.4 hidden) | 12.9 | ~41 |

Saturation statement for the collectives after L3+L1: the remaining
~11-25 ms/tau sits at the measured Intel-MPI Reduce_scatter_block floor,
4.3x the ideal-ring model — algorithm/reduction-bound (I_MPI_ADJUST
tuning territory, diminishing returns).  The GEMM after L-GEMM sits at
the measured BLAS roofline of the node.  Cross-tau double buffering (L2)
and jax-level strip-pipelining are DEAD on XLA:CPU — the runtime
serializes an in-flight collective against local compute (overlap
-0.04/-0.15, both cells); overlap exists only at the FFI tier (58%
measured via Test-polling).

Measured domain (claim-scope): MoS2 4x4 shapes (nk=16, ns=2, mu_pad=4992,
nb=128), 8x8 mesh on 32 nodes x 2 ranks, mlx provider, jax 0.9.1/XLA:CPU,
synthetic data, cache-cold compiles.  The GEMM share GROWS with nb
(∝ nb·mu^2/P vs comm ∝ nb·mu/sqrt(P)) — at nb=256 project_rs is already
39% of the sigma wall (sigma_perf_results.md), so L-GEMM's relative value
rises with the size campaign.

## 6. BSE audit: every (mu,nu)->(m,n)-class multi-stage reshard in src/bse + bandstructure

Method: exhaustive sweep (grep psum_scatter/psum/all_gather/process_allgather/
shard_map/with_sharding_constraint/einsum + per-site read) over src/bse/,
src/bandstructure/{bse_setup,htransform}.py; key sites re-verified by hand in
the source.  Reference pattern features: (1) shard_map interior,
(2) psum_scatter (tiled) per mesh axis — output stays sharded,
(3) LARGE payload on the node-local 'y' groups, (4) stacked channels
(one collective per axis).

### 6.1 Production-path sites, ranked by expected cost in a production BSE run

**#1 — bse_stack_matvec.py::_w_stack decode (THE BSE inner loop) — has the
structure, misses two of the four levers.**
`src/bse/bse_stack_matvec.py:126-131` (decode collectives), `:110-112` (encode
all_gathers), `:134` (lax.scan over trials), shard_map at `:137-143`.
This is the W-term of the default TDA matvec (bse_lanczos.solve_bse_sharded,
bse_feast TDA, exciton_bands all route here).  Per trial per matvec:

    encode:  all_gather(X_b, 'y')  +  all_gather(R, 'x')     R = (c_full, nk, ns, nu_loc)
    decode:  psum_scatter(einsum, 'x', dim=c)   payload (c_full, nu_loc, ns, nk)  <- LARGE
             psum_scatter(einsum, 'y', dim=v)   payload (c_loc,  v_full, ns, nk)  <- small

Feature scoring: (1) YES shard_map; (2) YES psum_scatter x2; (3) **NO —
INVERTED: the large (c_full, nu_loc, ns, nk) payload rides the stride-8,
zero-locality 'x' groups; the small one rides node-local 'y'** — exactly the
pre-swap ordering the sigma kernel abandoned (ppm_tau_kernel.py:110-117); the
contraction order (mu first vs nu first) is the same movement-only choice
sigma already made, transposed here to (c vs v); (4) **NO — no stacking: the
scan body fires 2 all_gathers + 2 psum_scatters PER TRIAL** (module docstring
:28-29 records "collectives run per trial inside the scan body" as a chosen
memory-for-comm trade); the trial axis is the natural stack axis, so a
block-b solve pays 4b collectives/matvec where 4 (stacked) would move the
same bytes.  Call-count arithmetic (bse_lanczos.solve_bse_sharded:105-240,
defaults max_iter=200, block/subspace width b=1-4 riding the stack axis):
per solve ~ n_iter x b x (2 all_gather + 2 psum_scatter) ~ 200 x 4 x 6 =
**~4800 grouped collectives — ~14x sigma's 352 per run** — with mu-scale
payloads (nu_loc = mu/8; the in-code T-tensor figure is 655 MB/rank at the
audit shape).  Whatever per-call cost Sec. 4 measures for the sigma chain
multiplies by ~14 here; this site inherits the sigma gap decomposition
almost verbatim, at higher call counts, and is the single highest-value
consumer of a shared optimized primitive.

**#2 — bse_ring_comm.py::_apply_W_from_T (TDA :358-372 and non-TDA :592-606)
— pre-optimization variant, still live on secondary paths.**
Plain `jax.jit(in_shardings=..., out_shardings=...)` einsum pair; NO shard_map,
NO explicit collectives — XLA's SPMD partitioner chooses the collectives for
the mu-sum ('x'), nu-sum ('y') and the c->x / v->y reshard, and materializes a
c-replicated (b, c_full, nu_loc, ns, nk) intermediate by construction.
Features: none of the four.  QUALITY_PATTERNS #4 (optimizer-defeats-intent)
applies: its collective table is whatever XLA picked, unverified.  Live
callers: bse_feast.estimate_spectral_bounds_sharded (:816), bse_kpm.py:129,
absorption_haydock.py:208, bse_pseudopoles.py:234, and the non-TDA family
(bse_w_exact.py:102, bse_nontda.py:76).  bse_stack_matvec's docstring (:43-54)
already marks it for retirement onto #1's encode/decode.

**#3 — vq_interp.py::make_eval_vq (:1024-1056) — (mu,mu) tile assembly by
with_sharding_constraint only.**
The dual reshard A_x = wsc(A, P('x',None)) / A_y = wsc(A, P('y',None)) then
V = V_SR + A_x^H A_y lands a (mu_X, nu_Y) tile with no replicated (mu,mu) —
correct scaling intent, but plain-jit: XLA chooses two all-gathers of the
(mu, nG) factor per Q-point, eager Python loop over Q upstream
(exciton_bands.py:636-661) plus an eager per-Q hermitize (:630-631) that
costs an all-to-all transpose of the (mu,mu) tile each iteration.  Features:
none of (1)(2)(4); per-Q call count makes it the third-ranked cost.

**#4 — bse_ring_comm.py density snapshot family — the GOOD copy plus a
pre-optimization sibling in the same function.**
`build_density_snapshot_operator(scatter_nu_on_y=True)` :892-916 mirrors the
sigma kernel by name ("mirror of ppm_tau_kernel reduce-scatter", :899-903):
shard_map + ONE psum_scatter('y') fusing the nu-contraction with the nu->y
tiling.  Features (1)(2) yes, (3)/(4) n/a (single small collective/channel).
LIVE for W(omega) (bse_w_exact:108,358; w_omega_chain:341).  The DEFAULT
branch scatter_nu_on_y=False (:897-899) is lax.psum -> y-replicated output,
consumed by bse_pseudopoles.py:140-158 one basis-column at a time with a
device_get per column — the per-channel/per-column anti-pattern; diagnostic
tier today.

**#5 — bandstructure/htransform.py + bse_setup.py (mu,nb)-class reshards —
de-replicated but partitioner-chosen.**
`htransform.py::_kpath_batch` :916-943 and `bse_setup.py::_fourier` :222-260:
(bs, rank, rank) contraction pinned P(None,'x','y') then resharded
P(('x','y'),None,None) — the big replication bugs were fixed (comments cite
11.4-51 GiB/device history), the reshard-before-transpose ordering lever is
applied, but the all-to-alls are with_sharding_constraint-implied, not
structural (no shard_map/psum_scatter).  One-shot per run -> medium cost.
`htransform.py::streaming_galerkin_solve` :171-180 replicates the
(nk*nb, ns*mu) psi block on every rank for the dense SVD — acknowledged
in-code as the FFI seam to swap "when n_mu scales up"; it is the one live
site in this family that materializes a mu-sized object per rank by design
(watch it against doctrine 1 at BSE-production mu).  Note: htransform.py:19
imports shard_map and never uses it.

**#6 — smaller live sites, acceptable as-is.**
bse_ring_comm.apply_V_ring :261-271 and the transition-seed generator
:755-788: mixed psum('y') + psum_scatter('x') inside shard_map on small
(b x mu_loc) payloads.  bse_feast.build_preconditioner_diagonal_sharded
:134-161: plain-jit (mu,nu)-tile contractions, one-shot setup.
_encode_T_gather(_B) :306-308/:490-522: 2x all_gather variant, opt-in
low_mem=False only.

### 6.2 Non-production sites (debug/verification — no action, listed for completeness)

bse_serial.py:59-68 (full (mu,mu) on one device, verification oracle);
bse_ring_comm.ring_matvec_correctness_check:1071-1084;
bse_w_exact.main compare branches :640-777 (full-tile device_gets);
bse_nontda._materialize_A_B :104-138 (dense NxN host, size-guarded);
vq_interp run_gates/run_nulls/build_hdir/refit (:477-490, :1191-1250, :1375);
exciton_bands.gate_htransform_vs_stored :747-757 (host gather inside a gate).
bse_simple.py :106-173 is opt-in and carries a **stale docstring claim**
(:27-30 says it uses the psum_scatter trick; it does not — no psum_scatter
exists in the file): flag for the doc-hygiene pass.

### 6.3 Audit verdict

The CURRENT optimized pattern exists in the BSE tree only as the single-axis
snapshot mirror (#4-good).  The BSE inner loop (#1) has the right skeleton
(shard_map + 2x psum_scatter, no replicated big object) but ships the large
payload on the wrong mesh axis and pays 4 unstacked collectives per trial
inside a scan; the secondary matvecs (#2) and the V_Q assembly (#3) predate
the pattern entirely (partitioner-chosen collectives).  Nothing in the
production BSE path gathers a (mu,mu) tile to one rank (doctrine 1 holds);
the exposure is per-call overhead x call count, which is exactly what the
Sec. 4 measurement prices.  No BSE code was edited.

## 7. Recommendation — ordered lever list (final)

Order by measured value at nb=128/P=64, with the BSE multiplier in mind
(Sec. 6.1: the BSE matvec runs ~14x sigma's grouped-collective count and
the same class of projection GEMMs per iteration):

1. **L-GEMM** — BLAS-grade projection GEMM (save ~23.4 s/run at nb=128;
   grows ∝ nb): try (a) the movement-only f64-split relowering inside jax
   first (decompose the COMPLEX psi operand to re/im f64 — kills the
   promotion copies and the zgemm; does NOT touch the owner-held re/im
   channel algebra), measure; if Eigen dgemm still lags BLAS, (b) FFI MKL
   GEMM handler (the FFT-FFI playbook: MklThreadScope, in-place aliases,
   announce-or-refuse, two-plan).  Gate: 1e-12 value parity (dot-order
   changes are value-level, not bit-level) + HLO shows no c128 promotion
   of the channel operands + per-tau staged row A/B.
2. **L0** — the common primitive wrapping the pattern for GW+BSE (below).
3. **L3** — FFI direct-MPI reduce-scatter chain (save 4.3 s/run at sigma;
   x14 call count at BSE; deletes the shim's 1.86x staging factor AND the
   world-collective-first init-order contract of Sec. 4.2).
4. **L1** — strip overlap at the FFI tier only (save ~2.6 s; measured 58%
   comm hiding via Test-polling; XLA-level overlap measured dead).
5. **L4** — BSE adoption (axis swap + trial stacking + ring retirement).
6. NOT recommended: jax-level cross-tau double buffering / strip
   pipelining (L2) — refuted by the overlap probe on this backend.

Lever details, gates, and envelope scaling:

- **L0 — the common primitive** (enabler, not itself a speedup):
  `common/collectives.py` grows a `stacked_projection_reduce_scatter` helper
  — ONE source for the pattern: channel/trial stacking on a fresh leading
  axis, the axis-order policy ("the LARGE partial reduce-scatters over the
  consecutive-rank mesh axis; only the small final block rides the strided
  axis"), the divisibility guard, and an HLO-pin test in the style of
  `test_check_hermitian_sharded_no_full_gather` (compiled-HLO predicate: rs
  count, payload shapes, replica groups — cache-cold per AY.2).  First
  consumers: gw/ppm_tau_kernel._project_ri_local (drop-in) and
  bse_stack_matvec._w_stack decode.  Backend-neutral by construction
  (shard_map + psum_scatter lowers on CPU and GPU); the direct-MPI tier
  (L3) slots BEHIND this interface as a gated plan, satisfying the
  two-plans-per-family doctrine.  Envelope: pure refactor — flat in
  n_atoms/N_mu/nb/P; no new object classes.
- **L1 — strip-pipelining** (deferred comment ppm_tau_kernel.py:207-213(a)):
  chunk the right-einsum on m (or n) and overlap chunk k+1's GEMM with chunk
  k's in-flight 'y' scatter.  Gate: phase E (XLA-runtime overlap) / phase M
  overlap probes (direct-MPI Test-polling already hides 58% of comm at this
  shape).  Envelope: hides min(comm, gemm); per rank comm ~ nb*mu/sqrt(P)
  while gemm ~ nb*mu^2/P — at growing mu/P the GEMM grows faster than the
  wire, so FULL comm hiding becomes easier at scale, and the lever is flat
  in n_atoms and backend-portable only via L3 (XLA:CPU cannot express the
  overlap if phase E measures serialization).
- **L2 — cross-tau double buffering**: dispatch tau i+1's W-phase/G-build/
  conv while tau i's projection collectives drain (production already
  free-runs dispatch; the question is whether the runtime actually overlaps
  — phase A free-vs-aligned and phase E answer it).  Envelope: hides
  per-tau skew up to one tau's compute; flat in problem size; interacts
  with the Sec. 4.2 thread contract (async dispatch is the mechanism that
  trips the mpi shim today).
- **L3 — FFI direct-MPI reduce-scatter chain**: a host FFI handler
  (mklfft/scalapack pattern, input_output_aliases={0:0} in-place, MPI
  communicators split once per replica-group set and cached, the AW
  MklThreadScope lesson applied to its progress loop) executing the stacked
  y-RS + x-RS at the measured 24.8 ms/tau floor, optionally with the
  Test-polling strip overlap (measured: hides 58% of comm behind the GEMM).
  Gate: value-parity 1e-12 vs the XLA path at production shapes + the
  collective table showing the custom-call replaced the reduce-scatters +
  the AS.4c-style rep ledger (the mpi4py/XLA-MPI coexistence is certified,
  but the handler must own its communicators, NOT reuse XLA's).  Envelope:
  cost is bandwidth-shaped (alpha ~ 0 measured), so it scales with payload
  ~ nb*mu/sqrt(P) per rank; the SHIM cost it deletes is per-call, so the
  absolute win scales with CALL COUNT — largest exactly where the BSE
  matvec lives (~14x sigma's call count, Sec. 6.1).  No new replicated
  object (doctrine 1); square-mesh assumption only via the group split.
- **L4 — BSE adoption** (movement-only, the sigma precedent, NO edits made
  here): (a) bse_stack_matvec decode axis swap — large payload to the
  consecutive-rank axis (the exact owner-approved swap sigma already
  carries); (b) trial stacking — hoist the per-trial collectives out of the
  lax.scan body onto a stacked leading axis, 4b -> 4 collectives/matvec at
  identical bytes (AK.9 argument verbatim; the scan's memory bound is
  preserved by keeping the T-tensor inside the scan and stacking ONLY the
  small decode payloads); (c) retire the ring/gather/simple matvecs onto
  the L0 helper (their own docstring already schedules this); (d) fix the
  stale bse_simple docstring claim (Sec. 6.2).

## 8. Artifacts

- Jobs (`REL_reshard_ubench`, dev queue, 32N x 2 = P=64, PHY25006):
  - 7878862 — step 2 direct-MPI floor COMPLETE (first sample); step 1
    failed at the grouped-clique thread check (the Sec. 4.2 finding).
  - 7878883 — probe cell (order-contract resolved), async-off cell
    (refuted the async hypothesis), gloo/ib0 FULL SUITE (Sec. 4.3), MPI
    floor sample 2.
  - 7878907 — impl=mpi FULL SUITE, world-barrier-first (Sec. 4.5), MPI
    floor sample 3.  All rc=0.
- Logs/dumps: wk_REL/reshard_ubench.<jobid>.out and
  wk_REL/reshard_ubench_run.<jobid>/ (jax_probe.log, jax_bench.log,
  jax_gloo.log, mpi_bench.log, hlo_dump_{probe,mpiA,gloo}/ — the dumped
  jit__project_ri_reduce_scatter modules carry the replica-group and
  payload evidence, cache-cold).
- Scripts: wk_REL/harness/reshard_ubench.sbatch, reshard_ubench_jax.py,
  reshard_ubench_mpi.py, reshard_ubench_parse.py.
- MPI-floor replication across jobs: chain 24.77 / 25.36 / 25.02 ms —
  stable.
- Baselines quoted: run_AQ_c4962_p64_mpi (272.040 s era),
  run_SIGMA_fft_fft1_{c,d} (71.906 s / staged rows, project_rs 43.246 s),
  wk_REL/docs/sigma_perf_results.md, ffi_fft_proto_notes.md,
  SPEEDUP_SCORECARD.md AS.4/AS.7/AY.4, QUALITY_PATTERNS.md.
