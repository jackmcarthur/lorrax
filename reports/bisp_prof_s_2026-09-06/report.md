# BISP-PROF-S — parent-route Sigma profiling

Branch `perf/bisp-prof-s-2026-09-06`, unmerged. Six production fixes are pushed through `ebc98918`; all pass their printed-digit gates and the combined P4 physics gate. All measurements use authorized campaign pool57966610. Original P is `9f569c4bf75bad40e4f5895946874b4c503e4410`; read-only F is `e1559a071e244b4f049c924781b668d9e1560739`.

The principal regression is compilation and eager planning in static/bare Sigma. The instrumented MoS2 static caller falls from151.373s to13.542s (F11.448s); compilation events fall1135→160 and compiler work120.740→9.988s. These caller scopes exclude surrounding Sigma work. Native baseline P is already faster than F for the selected warm static block and dynamic node. Complete-deck and fixed-rule after tables appear below.
## Objective and preregistered candidates

Reproduce PERF2's host/device separation for the five supplied bispinor comparisons, then change only measured Sigma bottlenecks. Measurements use P4, one rank/GPU, BFC@0.85, cold persistent compile cache, `LORRAX_DEBUG_PRINT=1`, identical copied P-donor `tmp/` within each P/F pair. Original deck `restart=false` is retained; every copied input has a SHA256 receipt. Actual selected quadrature nodes and digests must agree before comparing dynamic execution. A copied cache alone is not proof of an identical schedule.

Candidate order, registered before any new measurement or production edit:

1. **Restore JIT lifetime.** `photon_blocks_full_q` creates a new jitted zero initializer per output and a new jitted `add` per source block per term. Predict high compilation/dispatch cost relative to warm restoration. Measure per-output restore host time, compile receipts and native restore/mix ranges; ablate compilation lifetime without changing summation order or symmetry ownership. Restore storage remains O(nk mu_left mu_right/P), one output at a time.
2. **Static vertex specialization and repeated faces.** P keys static kernels by plan identity and vertex pair; F by endpoint shape. Predict more compiled bodies even at equal C/T extents. Compare first/warm blocks and body/head counts. Consider reuse of each family's unfolded face, retaining canonical `plan.unfold_face` and vertex ownership; quantify lifetime before keeping full-k faces. No new kernel-side bispinor branch.
3. **Head duplication.** P separately rebuilds G for the body and its q0-only diagnostic; F shares G within a block call. Test a shared G only after measuring head cost; retain exact diagnostic rows.
4. **Dynamic completed-G transport.** The CC tau path builds G on parents then rotates/transports an open-spin mu-square operator; it does not use the static child-face algorithm. Price transport versus parent GEMM savings, band-row selection and band unfold at fixed nodes. Do not infer a tau regression from total Sigma wall.
5. **I/O seams and bare packing.** Count existing compiled pack/unpack calls per family and bare TT packing before changing boundaries. No all-block full-q cache: retaining many mu-square bodies would violate the streaming memory objective.

Only accept a candidate after `tools/eqp_ab.py --tol-uev 0` on eqp0/eqp1 and printed-digit sigCC/sigTT/sigCT identity on the same source/deck baseline; then the combined P4 regression leg. Plan/scaling reasoning is not a substitute for optimized HLO and native device evidence.

## Before any source change

The following are **harvested checkpoint measurements**, not newly measured pinned-P baselines. Raw paths and step IDs are retained in [checkpoint_receipts.txt](checkpoint_receipts.txt). The enclosing sandbox root is `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14`; paths in that receipt file refer to the original read-only runs. Their runtime wrappers select BFC@0.85/P4. Checkpoint P arms predate the final P pin, so their stage walls motivate the experiment but do not certify the new baseline.

| deck / original P,F directories | P rule / tau / other s | F rule / tau / other s | P/F total run s |
|---|---:|---:|---:|
| Si COHSEX 66,49 | — / — / 47.41 | — / — / 11.35 | 74.29 / 56.32 |
| Si GN 56,48 | 6.14 / 4.01 / 55.90 | 6.32 / 4.84 / 20.96 | 91.46 / 76.97 |
| MoS2 full static 73,11 | — / — / 150.14 | — / — / 18.40 | 198.64 / 70.98 |
| MoS2 packed bare 74,12 | — / — / 149.95 | — / — / 18.58 | 182.66 / 85.17 |
| MoS2 GN eps5 80,79 | 77.76 / 3.14 / 185.40 | 84.57 / 2.65 / 27.46 | 338.98 / 165.86 |

The MoS2 Sigma regression is dominated by **Sigma other: +157.94 s**; rule planning actually costs P 6.81 s less. Tau sweep differs by +0.49 s. Si P's tau sweep is 0.83 s faster than F. These aggregate host rows cannot establish per-node steady-state latency or a ns=4 spin-rotation bottleneck.

All four historical dynamic arms report missing `tmp/sigma_quadrature_rules` at lookup and rebuilding rules in memory. The directories now contain the resulting caches; new baselines copy the same P cache into both arms. Historical progress rows end at 29 nodes for both Si arms and 67 for both MoS2 arms; this is the displayed sweep count, not a digest comparison. Compiler count/time receipts are absent from these non-debug checkpoint logs beyond the cache-off banner. Their compile-vs-execute split is **unmeasured**.

Prepared local evidence directories:

- `runs/Si/100_bisp_parent_route_2026-09-05/prof_s/01_P_cohsex_baseline/`, `02_F_cohsex_baseline/`, `03_P_gn_baseline/`, `04_F_gn_baseline/`.
- `runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/05_P_full_static_baseline/`, `06_F_full_static_baseline/`, `07_P_packed_bare_baseline/`, `08_F_packed_bare_baseline/`, `09_P_dynamic_eps5_baseline/`, `10_F_dynamic_eps5_baseline/`.

Each has `manifest.yaml`, `cohsex.in`, `rankwrap.sh`, `driver.sh`, `run.sh`, copied `tmp/`, and `inputs.sha256`. All ten baseline arms have completed on authorized pool 57966610. `run.sh JID` pins every dispatch explicitly and refuses to overwrite an attempt. Source checks compare `src/services` against the code pin inside each rank. Host baselines should precede separately copied profiler variants; Nsight instrumentation walls are not baseline timing evidence.

## Valid rank-0 device captures

Static and dynamic paired captures are complete; their reduced receipts appear below. Dynamic pairs use common certified schedules. We use PERF2's `cudaProfilerStart/Stop` inside `runtime.run_main_and_finalize`, with `jax.effects_barrier()` before stop. The prior installed nsys path is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys`; it is available and produced valid CUDA captures in this allocation. Capture rank0 while all four ranks execute the same science. Export `nvtx_gpu_proj_sum`, `nvtx_kern_sum`, `cuda_gpu_kern_sum`; verify actual CUDA records, then isolate one warm static block and one warm tau node. Native kernel sums and projected spans must remain separate; no nested-range summation.

## Saved Sigma device profile

Static and dynamic P/F native timings and kernel classes are reported below. Fused spin/gather costs are bounded by their enclosing native range, not presented as separately measured kernels.

### HLO collective census per stage

Static optimized dumps have been analyzed with `tools/hlo/analyze_hlo_dump.py` and PERF2's collective census; results below. Count async starts once and identify loop multipliers. HLO-zero does not establish communication-free distributed GEMM FFI. The parent orchestrator report explicitly says the literal final-psum-only criterion is not met; this lane has not disproved it. In particular, do not relabel transposed antiunitary transport as an ordinary local gather without inspecting the generated program.

## Owner's boundary-accounting table

**Preregistered source census at the pinned P/F trees, not measured module or kernel counts.** Let B be selected Lorentz outputs, T=3 terms, H=1 when head diagnostics are active (otherwise 0), K=nk_full, Q=n_parent, N=nb, s=ns, M_C/M_T=packed endpoint extents and p_x p_y=P. A source-level call may be optimized away or fused; the measured costs are reported in the native and host tables below; these source counts remain the pre-change reference.

| boundary | F | P source count | cost/scaling and measurement owed |
|---|---|---|---|
| Static compiled callable identity | endpoint shape class, vertices outside kernel; at most four ordered C/T classes | B(1+H) cached callable identities, keyed additionally by both plan ids and `(A,B)`; terms reuse these objects | 16/32 identities for full static without/with heads, 15/30 current-only, 9 bare TT. Actual executable count may include shape/dtype specializations; first/warm and compile seconds owed. |
| Static child faces before G | already full-k carriers; vertex insertion outside block | 2 unfolds per G; T B(1+H) G calls | 192 face unfolds for full static with heads; 180 current-only with heads; 18 bare TT without heads. One G build per block call, not two: the two operands are faces. Face gather traffic O(K N s M/P); spin rotation arithmetic O(K N s² M/P). |
| Static G and projection | G and projection on full K | static G still full K after child unfold; projection on Q selected rows | G O(K N s² M_left M_right/P); parent projection savings depend on Q/K. Body and head rebuild G separately in P; F reuses G for the diagnostic. |
| Static head kernel | q0 convolution shares block G | separate q0-only cached kernel and two more face unfolds | No quantitative saving claimed; head G and body G are numerically the same inputs within a term. |
| Full-q restore/mix | full-q packed photon body already available at Sigma entry | source restores/output: CC=1; each CT/TC=3; each TT=9 | Per full 16-block term: 1+6×3+9×9=100 restores; 300 over X/SX/COH. Current selection: 99/term,297 total. Bare TT:81 total. Per output/source O(K M_left M_right/P), times source-count; expensive host closure churn candidate. |
| Restore JIT creation | no parent `photon_blocks_full_q` loop | one fresh `add` JIT per restore plus one zero JIT per output | 300+48 creations full static;297+45 current;81+9 bare. Not a measured compilation count. No persistent restore cache exists in this function. |
| Static band unfold | projection already full K | after summing parent blocks, 3 total +3 sector +9 head-sector +3 head-total calls when heads present | 18 calls with heads;6 without; bare caller1. O(K N²) band matrix with producer sharding; final replication is separate. No per-block band unfold in static contraction. |
| Dynamic CC G transport | full-k G directly | one completed open-spin G transport per tau/bracket | Source path `build_G_tau`→`build_G`→`plan.unfold_operator`, not `plan.unfold_face`; O(K s² M_C²/P) payload, spin rotation O(K s³ M_C²/P). Exact kernel cost and collectives owed. |
| Dynamic projection seam | full K band projection | select Q rows, project Q, unfold band operator each tau/bracket | O(K N²) unfold plus small row-index gather; `k_unfold_plan` is in cached factory key, not freshly allocated per node here. No evidence of plan-identity cache misses yet. |
| Bare TT input pack | reader feeds individual TT tiles to scalar exchange kernel | `sigma_x_bispinor` constructs full packed `PhotonBasisLayout(M_T,M_T)` and zero C/mixed blocks before shared consumer | Dense packed carrier O(Q (4M_T)²/P), including zero C slots. At small shapes allocation/packing may matter; retain streaming/memory constraints. |
| I/O pack/unpack per family | canonical carrier | existing `PackedCentroidBasis` owner at canonical seams | O(Q or K times M_left M_right/P) per operator, or O(Q N s M/P) per face; exact frequency and ms require trace. No new conversion owner proposed. |

**Correction to brief:** `PHOTON_BLOCKS_CURRENT` enumerates all fifteen non-CC pairs (`a or b`), not twelve; it includes nine TT plus six CT/TC. The code makes no structural zero-block skip on the packed-bare route. The two supplied Si decks select incumbent charge-screened plus nine-block bare Sigma^B, not the MoS2 sixteen-block packed path.

**Reuse memory:** retaining both full-k faces for both families, with face specs sharded on both band and centroid axes, costs nominal complex128 carrier bytes/rank `32 K N s (M_C+M_T)/P`. This is the carrier footprint, not incremental allocator peak; upstream aliases/lifetimes matter. Retaining all vertex variants adds another multiplier and is not preregistered. Retaining G per block costs `16 K s² M_left M_right/P` bytes/rank per G and is substantially more expensive. Real extents and HLO peak bytes must be reported before selecting face reuse. No rank may acquire a full mu-square body.

### Restore attribution shared with BISP-PROF-ZW

`photon_blocks_full_q` is consumed by Sigma; ZW's report confirms this attribution. Charge its restores and mixing once to Sigma. This lane changes only that helper in `w_isdf.py`, reusing the symmetry service's existing executable cache. Screening and zeta implementations remain ZW-owned. No cross-lane file edits are requested; ZW can integrate findings from this report.

## Architectural proposals

The ranking below uses measured host savings or explicit device bounds per added complexity. Gates and raw step receipts are in the production and native tables below. Let K=nk, Q=n_parent, N=nb, s=ns and P=p_x p_y. The MoS2 gate has K9,Q3,N80,s4,M_C192,M_T100 (logical T98),P4. All payload figures are complex128 bytes/rank; allocator peaks are identified separately.

1. **Accept: reuse GEMM plans and existing restore executables.** These remove factory planning54.071s and restore closure compilation51.527s in the original instrumented caller. Reuse existing cache entries by endpoint shape and mesh; stream each restored/mixed output through the existing symmetry owners in unchanged C,D order. Delete eager per-vertex GEMM construction, loop-local restore JITs and zero initializers. Production caller151.373→97.582→39.204s. Additional retained mu-square data:0 bytes; only callable/plan metadata lives longer. One G remains `16 K s² M_left M_right/P` (max21,233,664 bytes); one full-q scalar block remains `16 K M_left M_right/P` (max1,327,104). Direct owner-cache reuse wins over the separate full-output cache ablation. Gates M23/M24: exact90 rows and both EQP files.

2. **Accept: one executable per endpoint class, with canonical weight placement.** Occupied and COH weights differed in sharding despite identical shape; normalizing the small(K,N) operand avoids COH retracing. Pass canonical gamma permutation/phase arrays as operands so C/C,C/T,T/C,T/T each compile once. The Python loop streams48 combined calls; it does not stack G. The original64 core executables become4. Weight normalization39.204→25.644s; after shared-G, vertex reuse19.531→13.542s. Extra replicated weights at most16KN=11,520 bytes/rank; carrier/G payload unchanged. Delete vertex-specialized cache keys and separate weight layouts. M33/M37 exact gates. **Reject literal sixteen-G batching:** heterogeneous all-G payload139,428,864 bytes/rank versus maximum single21,233,664, scaling `16 K s² (M_C+3M_T)²/P`. No measured dispatch saving justifies that lifetime; shape reuse obtains the compilation benefit without it.

3. **Accept: share G between body and head.** Original warm TT Green ranges cost1.055104 and1.040000ms separately. Feed both convolutions/projections from one G within the same compiled call, returning two small band operators. Delete the separate q0 kernel and duplicate face unfolds. G builds96→48; face unfolds192→96 for16 blocks×3 terms with heads. Production caller25.644→19.531s. One G payload before/after21,233,664 bytes max; two parent band outputs require at most2×16QN²/P=153,600 bytes/rank, subject to actual output sharding. M34 exact gate; native post-change allocator peak is measured separately.

4. **Accept with measured memory tradeoff: two supplied2×2 spin blocks.** Si's dynamic open-spin transport costs14.938586ms, including9.646109ms dense spin GEMMs and5.147709ms surrounding layout kernels. At the symmetry owner, inspect the supplied static4×4 action for exact off-block zeros, then apply its two blocks; the action already contains the lower-block determinant sign. Generic4×4 actions retain the dense path. Delete dense application only for this structurally proven case; do not reconstruct a symmetry rule. Ablation38.633553→30.433975ms/node (21.2%). **Peak memory rises2,020,215,829→2,487,859,989 bytes/rank**, +467,644,160, approximately half a full Si G, scaling O(Ks²M_C²/P); face payload is unchanged. This is an explicit speed/memory tradeoff, not a memory reduction. CPU42 tests the negative lower-block sign, a red twin and generic off-block fallback; fixed-certificate I13/I20 is the production identity gate. Generic elementwise4×4 ablation is rejected:39.206614ms/node.

5. **Reject for this implementation: resident family child faces once per term.** The original TT pre-G range containing both unfolds, spin/gamma and G initialization is0.032032ms, versus about1.05ms for the G it feeds. Even multiplying that enclosing range by96 original G calls yields only3.075072ms/run; this is a TT-class extrapolation, not a measured all-class total. After body/head sharing the invocation opportunity halves. Keeping parents requires2,242,560 bytes/rank; children add6,727,680, combined8,970,240, scaling `32(Q+K)Ns(M_C+M_T)/P`. Flow would unfold four unvertexed family faces once per term and apply canonical gamma at each streamed consumer; it deletes repeated transport, but requires additional operand/lifetime changes for little measured device opportunity. No larger-deck benefit is asserted without a new gate.

6. **Reject full-run dense restore caching; retain streaming plus owner cache.** Original300 restores/mixes cost6.607031ms projected GPU in total, versus95.0077s profiled compiler work. Caching all V/W outputs would retain17,428,608 bytes/rank versus one scalar output max1,327,104 (temporary overlap can add another block), scaling `32 K(M_C+3M_T)²/P`. Term-interleaved V/W streaming could bound retained data at2,654,208 bytes, but changes subtraction/accumulation ordering and would need a fresh printed-digit gate. It does not attack the remaining dominant compiler/setup work after owner reuse. Delete no data path now; do not add a second restore cache.

7. **Defer seam fusion and alternative face layout.** Measured Sigma bare-input packing is32 calls/4.186365ms projected on Si; MoS2 band unfold18 calls/1.543712ms, outside the block kernel. MoS2 pre-convolution layout is0.013216ms per selected body. A fused consumer/writer would consume the existing basis permutation at its adjacent reshard and delete the standalone conversion; a new face layout must delete both replaced adapters and preserve the GEMM operand contract. Payload before/after remains `32QNs(M_C+M_T)/P` for parent faces and O(KN²) for band seams; no allocator saving is proven. The observed per-unit millisecond opportunity does not justify a new carrier contract in this lane. Canonical files stay P-independent. Unattributed seam calls remain unattributed, not charged to Sigma.

8. **Accept ZW's coupled-current tail; do not duplicate it here.** ZW now measures51.002→39.314ms/current tile, saving11.688ms (22.9%), with128 fewer transports and32 fewer transforms. Its existing tile kernel streams one left child projector/IFFT through three channel updates, deleting the repeated outer channel tail while preserving each channel's order. At real-grid tile R3844, three full-k outputs cost41,515,200 bytes/rank and solved q-IBZ outputs13,838,400; HLO peak452,208,158→452,172,878. Scaling is O(K M_T R/P). The three C_q/Gram factors remain distinct: current Gram payload120,000 bytes/rank/channel,360,000 for three. A coupled solve would require a new hoisted-factor/pivot contract and could save at most the entire9.683ms/tile current-solve budget; dispatch savings are smaller. Adopt ZW's `f811f734` after its prerequisites. Fresh-fit gate24/25, step `lx-Xg4-021818-1673870-3826` on57966610, proves four zeta files bit exact and downstream90 EQP/sector rows exact. This is **ZW's measurement**, from [its report](/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_zw_codex_20260906/reports/bisp_prof_zw_2026-09-06/report.md); local Sigma source does not implement a second tail.

Every implemented change retains mu-square distribution on all P ranks, canonical P-agnostic files and one owner per symmetry rule. **The pre-existing MoS2 dynamic kernel still has two collective-permute starts**, detailed below; the literal final-psum-only requirement remains unmet. Static parent core/restores have no explicit HLO collectives, but distributed GEMM FFI broadcasts remain native communication. No additional collective is permitted by these changes. No remaining slowdown is called inherent to a small deck: the larger Si dynamic case is measured, but no P16 result exists under the one-node authorization.

## Ablations

Completed run-local arithmetic-preserving experiments appear in the unprofiled accounting table below. These experiments preceded production changes; their run-local patches and pinned source checks remain on disk. Failed experiment M13 initialized JAX before the communicator; it is retained as a failed variant, registered in KNOWN_SANDBOX_ERRORS.md, and corrected in M19.

## Verification and current disposition

All ten pinned baseline arms and both static captures completed. Static captures and successful ablations pass tolerance-zero eqp0/eqp1 and all 90 complete printed state rows in sigma_diag.dat, including CC/TT/CT. Dynamic captures are complete. All six production fixes have passed their incremental gates and are pushed. The combined P4 physics gate and fixed-rule block-spin gates pass. Repository bookkeeping checks are reported at closure.

Pool 57966610 is explicitly authorized by the user and shared with this campaign. Every arm uses one node sequentially. The earlier failed allocation is historical infrastructure evidence, not a current blocker. All writes, including local run/issue/claim registers, stay within this worktree; shared sandbox ledgers are read-only under the lane's explicit scope.

## Authorized measurements — 2026-09-06

User authorized BISP-orch pool57966610. Arms run sequentially, one P4 node each,
BFC@0.85, source arithmetic pinned as above. This section records the before-change measurements.

| unprofiled full-static MoS2 | P05 | F06 |
|---|---:|---:|
| Sigma other s |150.63|18.24|
| screening support s |31.55|13.86|
| whole run s |204.16|55.13|
| rank0 XLA compile count |1712|597|
| rank0 compiler work s |147.90|27.74|
| step, both exit0 |lx-Xg4-011831-1203588-3371|lx-Xg4-012232-1236210-4898|

Evidence: local `runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/`
`05_P_full_static_baseline` and `06_F_full_static_baseline`, each `gwjax.out`,
`driver.rank0.log` and `driver.1.log`. The compiler receipts cover the **whole
run**, not Sigma alone; no elapsed-wall subtraction is claimed as steady state.
Profile variants11/12 add boundary-specific compile-counter deltas and synchronized
block receipts plus rank0 native Nsight capture; their walls will not replace
unprofiled stage timing. F/P are cross-source performance controls, not the
same-source printed-digit gate used to accept a fix.

Static parser limitation is registered in local KNOWN_SANDBOX_ERRORS.md: the
provided parser demands dynamic rule/tau rows even on a completed COHSEX report.
The static table above is transcribed directly from each production artifact.

ZW's report confirms photon_blocks_full_q has only Sigma-side consumers; all
restore/mix costs here belong to Sigma. It remains unmodified in this lane.

### Static capture receipts (11 P / 12 F)

Both profiler legs completed exit0, with native nsys reports and CSV exports.
The terminal wait during P capture was transient compile-agreement skew while
rank0 flushed profiler work; it did not end in timeout or failure. Do not use
profiled stage walls as unprofiled performance estimates.

| host boundary in capture | P11 | F12 |
|---|---:|---:|
| contraction compiled bodies |64 (32 body +32 head)|4 combined body/head|
| body calls / cold calls / warm calls |48 /32 /16|48 /4 /44 (combined)|
| body cold median ms |570.739|640.200 (combined)|
| body warm median ms |2.795|6.731 (combined)|
| head cold / warm median ms |595.276 /2.532|included|
| restore V calls / compiles / compiler s |16 /120 /31.382|absent|
| restore W calls / compiles / compiler s |16 /116 /31.315|absent|
| restore W−V calls / compiles / compiler s |16 /120 /32.311|absent|
| complete static caller compile count / compiler s |1135 /254.837|141 /24.275|

The body boundary records33 compilation events for32 cold body calls: one
additional supporting executable is included; **64 contraction bodies** is the
32 body +32 head specialization count, not a relabelling of65 boundary events.
All48 restores compile, including W and W−V passes. P's X/SX weights are
(nk,nb); COH weights are (nb,), coinciding with the second specialization per
vertex/head identity. Shape alone is not a sufficient explanation: ablation16 broadcasts the COH weights and still produces64 modules. Sharding or another operand property remains to be isolated. The exact same cached object is reused for SX.

The remainder between the static caller and restore/block measurements is NOT
assigned a guessed cause or cost. Source inspection identifies eager plan
construction as a candidate: every one of32 factory misses creates a projector
(two eagerly warmed GEMM plans) and one Green GEMM plan, versus four factories
in F. New14/15 unprofiled instruments time the factory itself. Ablation17
reuses these unchanged plans by shape, ablation16 normalizes the small weight
operand shape, and ablation13 reuses restore callables across terms. They have now run; results below. None changes production source.

Artifacts are `11_P_full_static_profile/boundary.jsonl`,
`12_F_full_static_profile/boundary.jsonl`, their `boundary_summary.json`,
`nsys_rank0.nsys-rep`, `stats_nvtx_gpu_proj_sum.csv`, and
`stats_nvtx_kern_sum.csv` under the local MoS2 prof_s root. Native ms and
HLO census are reduced in the following section.

### Native warm TT unit and collective census

P11 modules2042/2094, second invocation: TT(1,1) SX body/head;
F12 module1019, second invocation: the same TT(1,1) SX combined call.
The lane extension `extract_nsys_unit.py` correlates native CUDA launches inside
the selected XLA-module NVTX range. Its aggregate ranges agree with native CSV
counts; it refused a missing range during development rather than reporting zero
work. Calls below are GPU projected spans, not host/compiler times.

| operation | P body ms | P head ms | F combined ms |
|---|---:|---:|---:|
| complete unit |2.185950|2.043967|6.757437|
| Green GEMM |1.055104|1.040000|1.211744 (shared)|
| body convolution |0.051008|—|0.050111|
| first band projection |0.394623|0.373151|1.245184 /1.238879 body/head|
| second band projection |0.327680|0.323552|1.197535 /1.178656 body/head|
| NCCL kernel count |60 broadcasts|60 broadcasts|180 broadcasts +2 all-gathers +1 SendRecv|
| NCCL kernel-duration sum ms |0.733663|0.744159|2.928127|

P's pre-G command-buffer range is0.032032ms (5 native kernels; summed
kernel duration0.031903ms), enclosing both child face unfolds, their fused
spin/vertex work and G-buffer setup. This is a **bound on spin-action cost**,
not an isolated spinor einsum measurement. It is only3% of that unit's Green
GEMM span, and cannot explain the132.39s unprofiled static-stage regression.
G→convolution layout conversion is0.013216ms; parent-row selection plus scale
is0.004416ms. No separate dense4×4 GEMM occurs in this sampled pre-G range.

Optimized HLO: all64 P contract_block modules contain zero explicit
collectives; all4 F modules contain4 all-gathers statically across their
conditional branches plus one collective-permute start (the canonical analyzer omitted that async opcode; the supplementary census repairs it). The selected F execution runs2 native all-gathers plus
1 SendRecv. P's broadcasts live inside GEMM FFI and are invisible to HLO.
P body/head TT peak HLO bytes/rank are13,524,977 /9,748,977.

All **300** P restore/add modules are present in the optimized dump and contain
zero explicit HLO collectives. Across300 calls their native projected spans
sum to **6.607031ms**; native kernel-duration sums total **1.673695ms**.
They nevertheless consume95.008s of compiler work in the enclosing restore
boundaries. This justifies prioritizing executable lifetime over a resident
full-q result cache. Native spans and kernel sums are deliberately kept separate.

Actual MoS2 carriers in the new startup/memory receipts are
Q=3,K=9,N=80,s=4,M_C=192,M_T=100,P=4. Thus both families' parent faces
carry2,242,560 bytes/rank; retaining both child faces for both families adds
6,727,680 bytes/rank, for8,970,240 bytes before scratch/aliases. This replaces
the earlier explicitly illustrative equal192/192 estimate for this gate.

Evidence: P11/F12 `census.json`, optimized dumps, native CSVs and
`unit_2042_1.json`, `unit_2094_1.json`, `unit_1019_1.json`.
Canonical analyzers are the sandbox HLO tool and PERF2 collective census; the
single-occurrence extension is registered in `skills/profile_bisp_sigma/SKILL.md`.
P11 eqp0/eqp1 each pass tolerance0 against P05 (90/90 printed rows).

### Complete pinned baseline table

All arms are unprofiled P4, BFC@0.85, JID57966610, exit0. Compiler receipts cover the whole driver, not Sigma alone. Paths are the numbered baseline directories listed above; each owns gwjax.out, driver.rank0.log and driver.1.log. Exact step identifiers are also retained in baseline_steps.txt.

| deck | P rule / tau / other s | F rule / tau / other s | P/F run s | P/F rank0 compiles | P/F compiler s |
|---|---:|---:|---:|---:|---:|
| Si COHSEX I01/I02 | — / — /47.29 | — / — /11.44 |74.21 /38.91|676 /412|47.26 /18.53|
| Si GN I03/I04 |0.09 /4.00 /56.30|6.25 /4.33 /20.64|86.86 /58.64|816 /539|57.35 /27.33|
| MoS2 full static M05/M06 |— /— /150.63|— /— /18.24|204.16 /55.13|1712 /597|147.90 /27.74|
| MoS2 packed bare M07/M08 |— /— /150.61|— /— /19.04|184.39 /54.44|1613 /598|136.65 /27.26|
| MoS2 GN eps5 M09/M10 |0.14 /3.00 /158.16|48.97 /2.50 /27.15|218.30 /116.72|1896 /790|161.28 /39.81|

Both dynamic pairs finish at29/67 nodes, but their selected digests differ despite copied caches. F domains differ at roughly7e-13 Ry at a containment boundary, miss cached rules and rebuild. Equal total node counts do not certify the same nodes. Capture preparation step lx-Xg0-020534-1575957-9680 selects an existing immutable certificate covering the hull of both actual domains through the production lookup guards. Both capture arms replay exactly those certificates (29 and67 nodes); no certificate is enlarged or tolerance loosened. Preparation and replay scripts are in M00_tools and replay_rules/replay.json in each capture arm.

### Unprofiled boundary attribution and isolated ablations

The **static caller** below is compute_static_photon_sigma, nested within the driver's Sigma stage. Do not add its wall to child boundaries or confuse F's11.45s caller with its18.24s whole Sigma stage. These are synchronized host boundaries with compile-counter deltas, without Nsight or HLO dumping. M14/P and M15/F reproduce pinned arithmetic. M16 normalizes weights only; M17 reuses unchanged GEMM/projector plans by endpoint shape; M18 separates centroid restore compilation from canonical Lorentz mixing. All five pass eqp0/eqp1 tolerance0 and90 complete printed Sigma rows against their same-source baselines.

| boundary | P M14 host / compiler s (events) | F M15 host / compiler s (events) | plan reuse M17 host / compiler s (events) | restore shape M18 host / compiler s (events) |
|---|---:|---:|---:|---:|
| restore, all3 terms |64.945 /51.527 (356)|absent|62.610 /49.543 (356)|5.830 /3.986 (85)|
| factory, body+head |54.071 /45.712 (674)|6.281 /5.360 (86)|6.296 /5.366 (86)|52.363 /44.217 (674)|
| contraction, body+head |29.123 /21.485 (65)|2.136 /1.637 (4)|27.923 /20.736 (65)|27.927 /20.814 (65)|
| enclosing static caller |151.373 /120.740 (1135)|11.448 /9.024 (141)|99.935 /77.598 (547)|89.260 /70.986 (864)|

Factory cold median P body/head1719.3/1726.5ms versus F1559.3ms per shape; warm factory lookups are about0.06ms. The difference is primarily **32 versus4 factories**, each building three eagerly warmed GEMM plans, not an intrinsically slower GEMM plan. M17 reduces96 plans to12 with no new resident face or mu-square data. Static executable count remains64.

M16 shape-only normalization still has1135 enclosing compilation events and64 contraction bodies; caller146.783s versus151.373s control is not accepted as a speedup. It leaves the measured cause unresolved and is rejected for production. Remaining specialization must be checked against sharding and other input metadata before a new normalization is proposed.

M18 retains existing C,D accumulation order and all SymMaps rules. It compiles four family transports per term, retains only local callable metadata, and streams source→full→mixed→accumulator. Additional live scalar full-q temporaries are bounded by two block payloads while Python references survive, unlike a16-output cache. For the largest CC block this bound is2,654,208 bytes/rank (2×16×9×192²/4); TT is720,000 bytes/rank. G and interaction objects remain sharded across all P ranks. Actual peak after source integration still requires HLO measurement.

These results rank **shape-class restore compilation** and **shape-only GEMM plan reuse** first for implementation. Their isolated caller savings are62.114s and51.438s respectively; adding isolated savings is only a prediction until the combined gate runs. Resident family faces can remove at most the sampled pre-G32us per body/head TT unit (about3ms over96 similar calls), far below these host costs. The current dense spin action is fused in that bound; a new two-block action is not justified by this capture.

### Dynamic warm nodes on common certified schedules

Both pins are unchanged. All four captures completed on JID57966610, sequential P4. Exact step IDs and reduced native/host/HLO receipts are in dynamic_profile_receipts.json; raw files live in Si I13/I14 and MoS2 M20/M21. Both arms use identical selected certificate files per window and29/67 nodes. Certificate replay is a **performance control**: against Si's original selected rules it changes eqp0 by up to0.518ueV, so it is not an identity gate against the original schedule. A source-change gate uses the same selected schedule on both sides.

| one tau unit | Si P I13 | Si F I14 | MoS2 P M20 | MoS2 F M21 |
|---|---:|---:|---:|---:|
| node count |29|29|67|67|
| compiled complete tau modules |1|1|1|1|
| timed-call compilation events / seconds |1 /0.142032|1 /0.139448|1 /0.151560|1 /0.144964|
| first timed host call ms |194.154|242.957|169.407|163.294|
| warm host median ms |39.503|92.061|2.485|5.418|
| second node projected GPU ms |38.633553|91.150714|1.974432|4.942591|
| second node native kernel sum ms |37.893873|85.287356|1.561376|4.200191|
| HLO peak bytes/rank |2,020,215,829|2,068,595,829|52,074,838|52,901,645|
| explicit HLO collectives/node |0|0|2 collective-permute starts|0|

**The alleged bispinor steady-node inversion is not reproduced.** P is faster on both gate decks. The original stage regression is dominated by static/bare caller compilation and eager planning, with rule-cache misses additionally distorting dynamic totals. The complete tau module does not recompile per node or plan identity. First-call compilation here does not include the separately prewarmed spatial/planning graph; that work belongs to Sigma other.

Si P spends0.970592ms in its parent Green GEMM,16.913465ms in the full-k convolution,4.507422/0.843488ms in its two parent-band projection GEMMs,0.231776ms selecting parent rows and0.002688ms unfolding bands. F spends10.016735ms in full-k Green,16.892479ms in convolution and49.944957/8.946687ms in projection. Both convolve on full K; P saves parent GEMM and projection work. Native NCCL broadcasts/node are96 versus768, summing4.845150 versus56.587261ms.

P's extra open-spin transport range is14.938586ms. Its two dense spin GEMMs sum9.646109ms and three surrounding operator layout kernels sum5.147709ms. This is distinct from the static face spin action's32us enclosing bound: dynamic transport acts on O(K s² M_C²/P) elements. The generic and structurally block-diagonal ablations are reported below; both consume the same supplied spin matrix at the existing symmetry owner.

MoS2 P's corresponding transport range is0.360832ms, including0.225312ms of dense spin GEMMs. Parent Green is0.359424ms, convolution0.108160ms, projection0.624992/0.276832ms and parent-row gather0.016160ms. Two antiunitary transposes are explicit collectives: operator payload7,077,888 bytes/rank at0.031808ms and band payload32,400 bytes/rank at0.006944ms. The literal final-psum-only invariant is **not satisfied by this pre-existing dynamic route**. It is registered, not hidden or bypassed. The optimizations must not introduce another collective.

### Seam accounting and corrected async census

The canonical HLO analyzer omits collective-permute-start. Its unmodified results remain on disk; M00_tools/census_async_starts.py counts optimized async starts once and excludes done operations. Static P's64 contraction and300 restore modules are still collective-free at HLO level under this corrected census. F static modules each have4 all-gathers across branches and1 collective-permute; the selected native execution runs2 all-gathers and1 SendRecv. FFI GEMM broadcasts are additional native communication in both pins.

seam_receipts.json maps exact compiled modules to source contexts. MoS2 static band unfold executes18 times,1.543712ms projected total /0.237376ms native kernel sum, peak537,901 bytes/rank. It has one collective-permute per invocation, with18 native SendRecv kernels; this is the small band seam after parent contraction. There is no per-Lorentz-block band unfold in that static loop.

MoS2's known startup family conversions execute10 times (1.654749ms projected); screening's file-to-packed TT conversions execute32 times (3.953467ms projected), attributed to ZW rather than Sigma. One additional carrier module has insufficient caller metadata for a stage assignment (0.322624ms). Si's bare Sigma input conversions execute32 times through2 compiled modules (4.186365ms projected /2.471006ms kernel sum); each executable has2 all-to-all instructions. Six startup calls cost3.822589ms, and7 calls lacking outer caller metadata cost18.575128ms across the complete driver. These unattributed calls are not silently assigned to Sigma or screening. F's canonical carrier needs no packed-basis conversion.

This bounds the measured seam/device opportunity well below the tens of seconds recovered by compiler lifetime. A writer/reshard fusion would need a new capacity/scaling gate; no seam rule or canonical file order is changed here.

### Further isolated ablations

M22 combines plan reuse and family restore compilation: static caller42.946s,276 compilation events, with exact90-row identity. M30 additionally shares body/head G (without weight normalization):29.662s,244 events,32 contraction modules and48 combined calls; warm combined host median3.269ms. M31 instead normalizes both weight shape **and replicated placement**:29.491s,245 events,32 contraction modules with separate body/head, and32 warm calls per output mode. Each passes eqp0/eqp1 tolerance0 and all90 printed state rows.

M22 input_signatures.jsonl identifies the otherwise identical TT COH call's differing leaf: occupied weights use NamedSharding(P()), while COH weights use SingleDeviceSharding. Shape-only broadcasting cannot unify those signatures. M31 applies an explicit replicated constraint to this small(nk,nb) operand, avoiding a second specialization. It does not replicate any mu-square object.

Si I17's generic elementwise4c rotation preserves eqp0/eqp1 and all256 state rows against the common-certificate I13 control, but its warm native unit is39.206614ms versus38.633553ms. It is rejected as a speedup. A following block-structure ablation validates exact zero off-diagonal2x2 blocks in the supplied typed spin action before omitting zero products; it does not reconstruct determinant or rotation signs.

## Production changes and gates

**Production Sigma GEMM plan reuse passes.** On branch perf/bisp-prof-s-2026-09-06, unmerged. Same-source eqp0/eqp1 and90 complete Sigma rows are identical; static caller97.582s, compiler77.376s and547 events. Existing cache shares plans by mesh, k extent and endpoint shapes; no resident face or operator is added.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/23_P_plan_reuse_candidate; [lx] step lx-Xg4-024713-1877307-4058 exit 0 in 156 s.

**Production full-q restore reuses its symmetry-owner cache.** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0. Static caller 39.204s; 219 compilation events, 29.386s compiler work.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/24_P_restore_candidate; [lx] step lx-Xg4-025324-1917331-9182 exit 0 in 98 s.

**Production occupied and COH weights share shape and placement.** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0. Static caller 25.644s; 188 compilation events, 19.008s compiler work.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/33_P_weight_candidate; [lx] step lx-Xg4-025650-1941125-7057 exit 0 in 88 s.

**Production body and head share one Green function.** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0. Static caller 19.531s; 172 compilation events, 14.348s compiler work.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/34_P_shared_g_candidate; [lx] step lx-Xg4-030030-1963321-6918 exit 0 in 83 s.

**Production Lorentz vertices share one executable per endpoint class.** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0. Static caller 13.542s; 160 compilation events, 9.988s compiler work.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/37_P_shape_class_candidate; [lx] step lx-Xg4-030620-1999625-3851 exit 0 in 71 s.

**Production supplied block-spin action passes fixed-certificate identity; Si warm tau ablation38.634→30.434ms with peak+467644160 bytes/rank** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 256 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/Si/100_bisp_parent_route_2026-09-05/prof_s/20_P_gn_final_profile; [lx] step lx-Xg4-033145-2312576-9256 exit 0 in 154 s.

CPU42 verification: [lx] step lx-Xg0-031951-2245813-3760 exit 0 in 16 s. Three symmetry rotation tests and eleven bispinor route configuration tests pass; only the existing JAX shard_map deprecation warning appears. Artifacts: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/42_cpu_gates/spin_pytest.log and config_pytest.log. This is a focused CPU emulated-mesh gate, not the combined P4 physics gate.

**Final MoS2 full-static gate; Sigma150.63→20.66s** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/35_P_full_static_final; [lx] step lx-Xg4-033420-2328417-3557 exit 0 in 74 s.

**Final Si COHSEX gate; Sigma47.29→13.36s** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 256 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/Si/100_bisp_parent_route_2026-09-05/prof_s/15_P_cohsex_final; [lx] step lx-Xg4-033536-2335119-7259 exit 0 in 46 s.

**Final MoS2 packed-bare gate; Sigma150.61→20.37s** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/25_P_packed_bare_final; [lx] step lx-Xg4-033626-2339875-4951 exit 0 in 60 s.

**Final MoS2 dynamic eps5 gate; Sigma other158.16→29.83s** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/26_P_dynamic_final; [lx] step lx-Xg4-033729-2348577-9137 exit 0 in 95 s.

**Combined P4 Sigma physics gate passes.** On branch perf/bisp-prof-s-2026-09-06, unmerged at ebc98918. Five vertex Green cases, five Sigma chains, all16 ordered current pairs and nonzero Hall head/zero/sign twins pass. Worst all16 relative error6.63e-16; Sigma closure1.15e-13; CT/TC zero-background and sign errors0.

[lx] step lx-Xg4-033906-2358773-6017 exit 0 in 51 s; runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/29_combined_regression/combined_gate.json.

## After-change stage and compiler tables

All times below are unprofiled production tables in each run's `gwjax.out`; counts/seconds are the rank0 final `[compile-cache]` receipts. Compiler seconds are cumulative compiler work, **not elapsed stage time to subtract**. Static parser omission is registered; static rows are transcribed from the printed production table. Dynamic `timing.json` uses the canonical parser; its synthesized `launcher_log` field is not the actual evidence path.

| deck / final run | P before rule/tau/other s | F before rule/tau/other s | P final rule/tau/other s | P before→final total s |
|---|---:|---:|---:|---:|
| Si COHSEX I15 |—/—/47.29|—/—/11.44|—/—/13.36|74.21→40.75|
| Si GN I16 |0.09/4.00/56.30|6.25/4.33/20.64|6.23/3.89/22.97|86.86→62.45|
| MoS2 full static M35 |—/—/150.63|—/—/18.24|—/—/20.66|204.16→70.68|
| MoS2 packed bare M25 |—/—/150.61|—/—/19.04|—/—/20.37|184.39→53.88|
| MoS2 GN eps5 M26 |0.14/3.00/158.16|48.97/2.50/27.15|0.15/2.92/29.83|218.30→88.79|

| deck | P before count/compiler s | F count/compiler s | P final count/compiler s |
|---|---:|---:|---:|
| Si COHSEX |676/47.26|412/18.53|421/20.33|
| Si GN |816/57.35|539/27.33|577/30.71|
| MoS2 full static |1712/147.90|597/27.74|737/42.16|
| MoS2 packed bare |1613/136.65|598/27.26|638/30.98|
| MoS2 GN eps5 |1896/161.28|790/39.81|963/57.51|

M35/I15/M25/M26 pass both EQP files at tolerance0 and every90/256 complete Sigma state row against their original P baseline. I16 differs from original I03 rules (max0.518ueV EQP0); its preserved failed comparator is `eqp0_vs_original_rules.log`. It matches I13's printed outputs, but its node digests differ, so **I16 is not used as the strict fixed-schedule source gate**. I20 replays I13's exact six certificate/node digests and passes both EQP files and all256 state rows; this is the Si GN source-change gate. New common-rule unprofiled controls below separate the cache-miss effect from final P/F performance.

### Measured owner accounting after the fixes

Host rows here are M14/M15/M37, one complete16-block×3-term static caller with heads; native rows are selected TT units, not an average over unequal endpoint classes. The static caller is nested around factories/restores/blocks: do not add it to its children. Plan factories additionally compile eager service support, so their counts are not contraction-module counts.

| boundary | F before | P before | P after | frequency/scaling |
|---|---:|---:|---:|---|
| entire static caller host/compiler s; events |11.448/9.024;141|151.373/120.740;1135|13.542/9.988;160|one caller/run; O(BT) dispatch|
| GEMM factory host/compiler s; events |6.281/5.360;86|54.071/45.712;674|6.302/5.367;86|F4 cold classes; P32→4 cold classes; plans scale with endpoint shape,K,Q,N,s,P|
| block calls / core modules |48/4|96/64|48/4|B16,T3; body/head shared after; distinct plan identities remain separated|
| block host/compiler s |2.136/1.637|29.123/21.485|1.983/1.385|cold plans excluded; original extra support compile makes boundary count65 versus64 cores|
| pooled first/cold block host median ms |474.006|426.785 body;451.660 head|464.316 combined|unequal shape classes pooled; not a like-shape speed ratio|
| pooled warm block host median ms |5.083|4.521 body;2.215 head|2.497 combined|P after44/48 calls warm; F44/48 warm|
| restore host/compiler s |absent at Sigma entry|64.945/51.527|2.259/1.268|300 source contributions and48 outputs unchanged; O(K M_A M_B/P)×source multiplicity|
| selected TT native body+head ms |6.757437 shared|2.185950+2.043967|3.001600 shared|one warm block; sums of separate original modules omit host gaps|
| selected TT native G ms |1.211744 once|1.055104+1.040000|1.043456 once|O(KNs²M_A M_B/P); static G is fullK in both routes|
| selected TT parent/full-band projection ms |1.245184+1.197535 body;1.238879+1.178656 head|0.394623+0.327680 body;0.373151+0.323552 head|0.505024+0.337824 body;0.496800+0.334688 head|P projectsQ vs F K; O(QNs²M_A M_B/P) contractions|
| face transport enclosing native range ms |no parent transport|0.032032 per selected G|0.040832 per shared G|contains both faces, spin/gamma and initialization;192→96 unfolds/run, not a separately timed spin einsum|
| static band unfold |absent|18calls/1.543712ms projected|same18 calls; no seam change|O(KN²), after summation; one CP/call|
| bare Sigma input conversions (Si) |absent|32calls/4.186365ms projected|same input seam|two cached modules;2all-to-all instructions each, independent of tau count|

Final static native class receipts M36: CC4.022591ms/50,119,937 peak bytes, CT3.415488/28,022,609, TC3.347616/27,316,049, TT3.001600/15,255,249. CC/CT/TC/TT execute3/9/9/27 times; selected SX occurrences are1/3/3/9 (zero-based). All four optimized parent core modules have **zero explicit collectives** under the corrected async census. Original TT body/head peaks were13,524,977/9,748,977 separately: sharing G saves execution, but the final combined module's peak is slightly higher than either separate module. These are per-executable HLO peaks, not whole-driver allocator usage.

Final Si capture I20: warm dynamic node30.385825ms versus P38.633553/F91.150714, first host182.489ms, warm host median31.334ms,29calls/one complete tau compilation(0.139382s). Peak2,487,859,989 bytes/rank; zero explicit HLO collectives. Its spin/transport enclosing command-buffer range is6.477497ms versus14.938586 before. Bare TT contraction is one executable for9calls; selected warm TT12 takes12.681075ms versus F31.702557ms, peak314,901,208 bytes/rank and zero explicit HLO collectives. The canonical bare-input all-to-all seam remains outside that kernel.

Receipts: M36/I20 `final_capture_receipt.json`, `unit_*.json`, `boundary_summary.json`, `xla_dump_rank0/hlo_summary.json` and `async_collectives.json`; every receipt contains its actual lx step. The native class comparisons are one selected warm unit per capture, not a repeated statistical claim about small differences.

## Integration disposition and overlap

Adopt S commits in order: `5dd1c775` (GEMM plans), `ebb18981` (restore owner cache), `1beb5711` (weight placement), `fd5ba099` (shared body/head G), `5093d98d` (vertex operands/shape classes), `ebc98918` (supplied spin blocks). All are pushed on `perf/bisp-prof-s-2026-09-06`, unmerged. The last commit has the explicit Si peak-memory tradeoff above; the first five do not require it.

**One restore implementation must win during integration.** ZW independently pushed `87a2bfaa`, adding `_add_photon_block` and retaining100 vertex-specialized restore executables. S's `ebb18981` instead calls the existing symmetry-owner cache directly and deletes the loop-local zero/JIT, without a new helper. Prefer S's shorter owner-cache version for `photon_blocks_full_q`; skip ZW87a2bfaa when adopting its independent chi/tail fixes, or delete `_add_photon_block` while resolving that overlap. Do not retain both paths or add their reported restore savings: this is the same Sigma boundary. No ZW file was edited by this lane. Integration with ZW remains the orchestrator's combined gate, not a result certified by either separate branch.

**Final static native capture verifies four collective-free parent cores and TT3.001600ms** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0. Static caller 31.361s; 160 compilation events, 27.604s compiler work.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/36_P_static_final_profile; [lx] step lx-Xg4-033958-7856-6832 exit 0 in 241 s.

**Final fixed-rule MoS2 capture verifies tau1.785888ms and preserves the two pre-existing collective permutes** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0. Static caller 25.036s; 126 compilation events, 21.810s compiler work.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/38_P_dynamic_final_profile; [lx] step lx-Xg4-034411-36368-8022 exit 0 in 260 s.

**Final unprofiled Si fixed-rule P gate; Sigma27.43s with29 identical nodes** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 256 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/Si/100_bisp_parent_route_2026-09-05/prof_s/21_P_gn_fixed_rules; [lx] step lx-Xg4-034834-61698-3797 exit 0 in 61 s.

Final MoS2 dynamic capture M38 (JID57966610, step lx-Xg4-034411-36368-8022 exit0): one tau executable/67 calls,0.143442s timed-call compilation; first host159.245ms, warm host median2.291017ms; native warm1.785888ms versus original P1.974432/F4.942591. Peak63,803,222 versus52,074,838 bytes/rank, +11,728,384. The two pre-existing collective-permute starts remain. All90 complete Sigma rows and both EQP files are exact against common-rule M20. Current static selection has three CT/TC/TT core executables; all are HLO-collective-free.

Final M36 restore transport census: four symmetry-owner unfold modules1419/1502/1574/1644 execute3/27/27/243 times, respectively, for300 source contributions total. All four have zero explicit HLO collectives. Their projected native transport totals5.533920ms; this excludes separate Lorentz multiplies/additions and therefore is not directly compared to the original fused restore-plus-mix6.607031ms. Source-order restoration remains repeated, while its executable lifetime is fixed. Receipt: M36/restore_owner_native.json. The selected final TT block executes84 native NCCL broadcasts (1.770240ms kernel sum), versus120 across the original separate body/head; HLO-zero does not hide this FFI communication.

## Final common-rule P/F controls and remaining costs

These unprofiled arms use the same selected immutable certificates in both sources; all six Si and eight MoS2 printed node digests match (29/67 dispatches). Production certificate guards remain active. Verbatim receipts are in `final_rule_receipts.txt`; canonical parsed timings and actual step IDs are in `final_fixed_rule_timings.json`. P arms match pinned common-rule P outputs exactly; F arms match pinned common-rule F outputs exactly. Cross-source physics differences are not relaxed into a source-change gate.

| common-rule deck / arms | P rule / tau / other s | F rule / tau / other s | P/F whole Sigma s | P/F whole run s | P/F whole-driver compiles / compiler s |
|---|---:|---:|---:|---:|---:|
| Si GN I21/I22 |0.08/4.07/23.28|0.08/4.26/20.10|27.43/24.44|56.35/52.10|577/31.58 vs539/27.31|
| MoS2 GN M43/M44 |0.12/2.92/29.82|0.11/2.46/26.50|32.86/29.07|88.93/65.75|963/57.47 vs790/39.78|

The final parent Sigma stage remains about3.0s slower on Si GN and3.8s slower on MoS2 GN under matched rules. The leading warm-node inversion is disproved, but whole-stage equality is not claimed. Static restore host work still costs2.259s/caller after fixing compilation; this nearly accounts for the static caller's2.093s residual over F. The dynamic `sigma.tau_sweep` scope includes `integrate_sigma_store` setup, residue reads and omega accumulation, not just the measured tau callable. On MoS2 the complete tau callable is faster while the enclosing sweep remains0.46s slower. The closing executor attribution below separates that residual into cold compilation, pole packing and accumulation; it is not blamed on the faster spin kernel. No inherent tiny-deck or unmeasured P16 limitation is asserted.

## Verification closure

Six production commits pass incremental printed-digit gates; final checks cover all five decks, with exact common-rule controls for both dynamic cases. Combined P4 step `lx-Xg4-033906-2358773-6017` passes the dense vertex/Sigma/all16-current and Hall-head sign/zero oracles. CPU step `lx-Xg0-031951-2245813-3760` passes14 focused tests. All science and captures use authorized57966610 sequentially, one P4 node/leg.

**The full repository gate does not pass.** CPU step `lx-Xg0-035328-91325-4986` exits1: AST suites pass90+34+16=140 tests, but the shared rule allowlist flags13 files unchanged from pinned P, and shared CLAIMS contains12 rows without its evidence token. `41_ast_gate/rule_scope_diff.txt` is empty, proving those flagged files were not changed by these commits; `gate0.log` preserves every finding. Its separate test-evidence check accepts unrelated job57909062, so that check is not substituted for this lane's explicit CPU/P4 receipts. Shared allowlist/ledger files and the evidence guard remain untouched. This is a documented broad-gate failure, not an overall pass.

Raw rank-zero Nsight/HLO artifacts remain under the cited run directories; reduced native units, selected HLO/memory receipts, scripts, canonical rules and load-bearing numerical outputs are committed on this branch. `branch_status.txt` records the checkpoint; `git merge-base --is-ancestor ebc98918 origin/main` returned1, confirming these fixes are unmerged. The orchestrator owns cross-lane integration and its combined gate.

### Closing dynamic executor attribution

Matched-rule M45/M46 synchronize the existing host boundaries without changing arithmetic; M47 additionally times the two in-executor pole packing calls. All three pass both EQP files at tolerance0 and all90 complete state rows against their respective fixed-rule control. JID57966610 steps: P45 `lx-Xg4-035937-127855-7695`, F46 `lx-Xg4-040115-137387-6633`, P47 `lx-Xg4-040611-162030-1739`, all exit0. Raw `boundary.jsonl` and reduced `boundary_summary.json` own these times; the enclosing executor is nested, not additive with its children.

| executor boundary | P45 host s / compiles / compiler s | F46 host s / compiles / compiler s |
|---|---:|---:|
| full executor |3.135303 /32 /1.944396|2.740049 /30 /1.522628|
| tau factory, one call |1.262467 /21 /0.981576|1.256108 /21 /0.980474|
| pole batch read, one batch |0.006388 /0 /0|0.006360 /0 /0|
| accumulator initialization |0.041789 /1 /0.032318|0.041794 /1 /0.032348|
| accumulator precompile |0.051606 /1 /0.039940|0.057544 /1 /0.041988|
| all67 complete tau calls |0.186073 /1 /0.032913|0.403052 /1 /0.034136|
| all67 accumulator adds |0.019856 /0 /0|0.022975 /0 /0|
|8 begin/end windows +finalize |0.000949 /0 /0|0.001023 /0 /0|
| remaining setup/prewarm/window work, by subtraction |1.566176 /8 /0.857648|0.951193 /6 /0.433682|

P47 isolates two `PackedCentroidBasis.pack_operator` calls inside that last bucket:0.152215s, one compilation/0.123126s compiler work. They run once per pole batch for B and Omega, not once per tau; F has no packed-basis seam. A fused reader/reshard could target this boundary, but the entire measured opportunity is0.152s on the gate and requires a wider I/O contract gate. Full-q B/Omega payload scales O(K M_C²/P), independent of band/spin count; at one MoS2 pole the complex128 B payload is1,327,104 bytes/rank (Omega's actual dtype controls its separate payload). No new fusion is justified here solely by these first-call milliseconds.

The residual executor penalty is therefore cold compilation/setup outside the fast node call: P−F compiler work is0.421768s while the node calls save0.216979s. Factory planning and omega accumulation are essentially equal. The remaining setup bucket includes the explicit `.lower(...).compile()` prewarm and small selector/constant preparation; its pieces are not falsely reported as individually timed device kernels. There is no per-node compile storm. These receipts explain why an enclosing sweep may remain slower even after its steady node improves, without asserting that the residual is inherent to a tiny deck.

Run provenance correction: candidate manifests retain donor `source.commit`/instrument fields. The authoritative runtime source is `source_head.txt` plus `source.diff` and `driver.sh`'s guard; `final_run_audit.json` records those artifacts explicitly. Completed science files are preserved. This copy-helper metadata issue is registered in KNOWN_SANDBOX_ERRORS.md.

**Final executor attribution isolates two pole packs at0.152215s and one0.123126s compilation, with exact science** On branch perf/bisp-prof-s-2026-09-06, unmerged. PASS: 90 complete printed state rows identical, including sigCC/sigTT/sigCT. Both EQP files pass tolerance0.

Evidence: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/47_P_executor_seams; [lx] step lx-Xg4-040611-162030-1739 exit 0 in 95 s.

## Shape-class scan on 6x6

Preregistered baseline: orchestrator71ae0bde, fetched and rebased before implementation. Source is compared byte-for-byte against that pin. The optional prior block-spin change is not on the orchestrator pin and is excluded from this experiment. Pool57982945 is authorized; sequential P4 legs precede any available P16 leg. Decks/immutable inputs come from MoS2/42_bisp_scale_2026-09-06, with private copied tmp/rules.

Plan: four ordered endpoint classes; one unvertexed G and its transform per class/term. Distinct W_AB prevents reusing one block's convolution result. Stream canonical gamma-weighted products through a compiled scan, then perform the shared inverse operation/projection on the class sum; do not stack Green tensors. Preserve the existing FFT normalization owner and symmetry actions. Compare printed outputs before accepting changed summation placement. Attribute factories, synchronization fences, restore dispatch and outer Python work before source changes.

### Before-change accounting (6x6, P4)

All source/services/tests equal orchestrator71ae0bde. New run root `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/`; all steps use authorized JID57982945. Baseline01: `lx-Xg4-113321-195974-2219`, exit0, Sigma other21.73s, total73.20s, whole-run compiler674 events/40.89s. Dynamic02: `lx-Xg4-113441-218949-9472`, exit0,116 nodes across8 windows; six cache hits/two rebuilt windows (`parsed.json` and exact node receipts in driver.rank0.log). Capture03: `lx-Xg4-113635-230722-5731`, exit0, native CUDA records exported; EQP tolerance0 and210 complete sector rows identical to01. Python-profile04: `lx-Xg4-114002-258518-9170`, exit0. Profiled walls are not substituted for baseline walls.

| synchronized boundary, 04 | calls | host s | compile events / s | warm median ms |
|---|---:|---:|---:|---:|
| factory |48|6.935026|86 /5.782593|0.079544|
| contraction |48|2.903554|4 /1.590072|7.961281|
| V restores |16|2.438699|24 /1.585057|23.962827|
| W restores |16|0.289386|0 /0|25.205862|
| W−V restores |16|0.548147|4 /0.138982|31.925751|
| outer caller, nested |1|14.395089|138 /9.855657|—|

The outer remainder after those children is1.280278s, including weights, band sums/unfold, closure and instrumentation. cProfile (`04/sigma_host_top.txt`) identifies Python loop own time0.006s, canonical gamma operand construction0.082s, six finish calls0.324s. Its97 `jax.block_until_ready` calls total0.414s; this includes instrument fences and overlaps contraction timing, so it is not an additive host-gap estimate. The48 warm factory lookups are negligible: the6.935s factory total is dominated by four first-use projection/GEMM plans. The300 canonical unfold calls total1.486s and400 block views0.522s, nested within restores. Thus the former2–4s host gap is dispatch/lowering/synchronization spread over these boundaries, not seconds of Python loop arithmetic. The scan removes36 contraction dispatches/term-loop repetitions and the duplicate48 outer sum fences; a subsequent separately gated restore change will target the300 source restores.

Architectural proposal accepted for measurement: move the canonical monomial vertices through the linear transform, build G once/class/term, IFFT it once, scan distinct W_AB products, FFT/project the sum once. Keep head diagnostics in the same factory and prepare their covariant factor orbit once/term. Four class sizes1/3/3/9;12 calls/run rather than48. No Green block axis. The sole FFT normalization owner is extended, with its scalar fused path retained. Additional resident interactions are B_class scalar full-q blocks sharded on both centroid axes, plus one accumulated Sigma; optimized memory will gate the actual lifetime. Existing `SymMaps`, `gamma_apply`, Green and band-projection owners remain authoritative. Printed-digit equivalence and a complex dense class oracle gate the change in summation placement.

### Native warm blocks and optimized HLO before the scan

Capture03 `selected_units.json` owns the reduced figures below. Select the first SX call in each class (occurrences1/3/3/9), after the X calls compiled and warmed each executable. Native projected spans include launch gaps; disjoint CUDA kernel sums are separately classified. All four optimized contraction modules have **zero explicit collectives**, including async starts; NCCL broadcasts inside existing distributed GEMM FFI remain visible and are not called communication-free.

| warm block | projected ms | kernel sum ms | GEMM / NCCL / FFT / other fused ms | optimized bytes/rank |
|---|---:|---:|---|---:|
| CC |29.184572|28.013724|5.777568 /11.611678 /5.379967 /5.244511|1,732,891,144|
| CT |14.252446|13.309374|3.068447 /6.975008 /1.751743 /1.514176|573,907,784|
| TC |14.025886|13.083103|3.055648 /6.756768 /1.755359 /1.515328|573,907,784|
| TT |7.827103|6.251199|1.456032 /3.686367 /0.584832 /0.523968|189,953,736|

The full-class baseline is1/3/3/9 times those per-block costs; this is a scaling estimate, not a measured batched execution. TT therefore offers70.444ms of projected block work per term before batching. Its Green GEMM and two band-projection FFI ranges cost2.866943/1.411616/0.768448ms in CUDA sums; the fused convolution0.715776ms. Canonical unfold, spin/gather and transposes share fused kernels and remain bounded by the non-GEMM/FFT bucket. No standalone spin-rotation latency is invented. The candidate's scan moves vertex gathers onto the Green tensor; their cost must be included in the after capture.

The complex CPU gate passes22 tests on four emulated devices (`07_cpu_scan`, `lx-Xg0-114716-304836-4320`, JID57982945); this substitutes backend FFT/GEMM plumbing and does not certify native GPU execution. First gate06 passed21 but its deliberate Gamma2 mutation reused a previously compiled convolution and therefore did not exercise the mutation. The test now invalidates that executable before injecting the error, and the oracle rejects it. No production arithmetic was changed to satisfy that test.

Allocation fallback01 exited3 after three Slurm query timeouts; fallback02 reached the controller but was refused with QOSMaxSubmitJobPerUserLimit. The P4 candidate remains queued on the authorized shared pool; the named fallback will be retried at ten-minute intervals after that refusal. No other lane's allocation is used.

### Scan-only result and gate

Scan05 (`lx-Xg4-115721-277816-1303`) retains Sigma other21.73s versus baseline21.73s; whole-run compiler674→685 events,40.89→39.95s. Capture08 (`lx-Xg4-115900-384535-8939`) has12 class calls and4 compiled cores versus48/4. Its enclosing caller149 events/26.725 compiler seconds compares with capture03's138/25.007; factory planning falls86→84 events, while eager class input preparation adds13 events outside factories/cores/restores. This explains why dispatch batching alone does not remove the cold host penalty. Profile walls are not stage speedups.

| native warm class | baseline sum of1/3/3/9 blocks, projected ms | scan projected ms | scan CUDA sum ms | scan optimized bytes/rank |
|---|---:|---:|---:|---:|
| CC |29.184572|36.713986|35.482949|1,784,731,404|
| CT |42.757338|20.051494|19.015495|624,712,340|
| TC |42.077658|19.534890|18.499250|624,712,340|
| TT |70.443927|11.931800|10.928640|239,742,932|

The baseline class sums multiply measured single warm blocks; the scan column measures one complete class. All four scanned optimized cores have zero explicit collectives (async starts included). TT is5.90× faster by this projection, CT2.13× and TC2.15×; CC is1.26× slower. CC CUDA GEMM and FFT work are essentially unchanged, while NCCL rises11.612→16.289ms and other fused work5.245→8.062ms. The latter includes applying vertices to G and exposed transform/product operations; NCCL wait variability is not charged entirely to that arithmetic. A singleton specialization could recover at most about23ms/run at these samples, so it is deferred pending a worthwhile complexity/performance tradeoff.

Memory follows the requested one-G class residency: `16 K s^2 M_A M_B/P` bytes/rank, with K36,s4,M_C600,M_T196,P4 giving829,440,000 /270,950,400 /270,950,400 /88,510,464 bytes. Class scalar interactions add51,840,000 /50,803,200 /50,803,200 /49,787,136 bytes. The optimized heap increment closely follows that input stack; it does not multiply Green storage by1/3/3/9. These executable totals include inputs/temporaries and exclude independently resident response/FFT/NCCL resources. Child-face payload is `16 K s nb (M_A+M_B)/P`; parent inputs replace K by n_parent7. Work scales with nb in G/projection and with K,s²,M_A M_B/P in convolution; all centroid-square buffers remain distributed over both mesh axes. Each class unfolds its two endpoints once; no cross-class face cache is introduced.

Both EQP files and210 complete printed sector rows match baseline for static05, capture08 and dynamic09. Dynamic09 (`lx-Xg4-120230-406775-1906`) retains116 nodes, Sigma other31.98s versus31.66s; rule planning6.34→0.15s because the two formerly missing caches were copied after baseline completion, so the total105.11→100.45s is not a scan speedup. CPU07 passes22 dense physics oracles; CPU11 (`lx-Xg0-115808-380235-8363`) passes25 static-owner/head tests. Native12 (`lx-Xg4-120509-421689-1824`) passes the existing full vertex/scalar-face and Hall-head gates. Failed10 was a wrapper permission error before Python; corrected12 uses bash explicitly.

Next ablation: one compiled class producer restores each of the16 source blocks once/term (48/run), instead of300 source restores. Canonical `unfold_isdf_operator` and `mix_lorentz_blocks` remain the only transport/mixing owners. The class producer also creates the tiny vertex operands and interaction stack in that same compiled unit, removing the measured13 extra eager preparation modules. No interaction arrays are retained in the module cache. The source restore function's interface will take packed operands and explicit metadata for one class; the previous per-output implementation will be deleted and its independent restore oracle adapted. This directly addresses the measured host boundary rather than adding a second transport cache or rule.

### Compiled class producer: measured host improvement and cold tradeoff

Restore13 (`lx-Xg4-120912-441486-5686`) passes both EQP files and210 complete sector rows. Sigma other20.46s versus21.73s, whole-run compiler649/39.53s versus674/40.89s. Dynamic16 (`lx-Xg4-121517-471916-2792`) passes the same gate at116 nodes: Sigma other31.71s, tau7.32s, cached rule plan0.17s. Native18 (`lx-Xg4-121816-484589-4806`) and22 CPU restore oracles pass. The source count is48 restores/run,12 class producer dispatches and12 contractions; Python traces only16 source restores when the four producers compile. No second restore rule or cache of interaction arrays is introduced.

Capture15 (`lx-Xg4-121202-456270-6121`) measures113 Sigma-caller compile events/24.136592s versus138/25.006523s in capture03. Four producer modules replace28 restore-related events; factory86→84 and outside work20→21 accounts for the remaining delta. Warm producer median0.837740ms/class; its12 calls total3.808565s including3.411504s compilation. Four scanned cores and four producers all have zero explicit optimized HLO collectives. Producer executable bytes/rank CC/CT/TC/TT are102,216,140 /156,985,180 /173,863,260 /139,388,796, still distributed across all P ranks. Scanned core heaps are unchanged from08. Native SX class projections in15 are32.027033 /17.229896 /16.995690 /10.878397ms, with the same arithmetic and reduced NCCL waiting compared with08.

The matched Python profiles expose the cold tradeoff honestly: baseline04 caller14.395089s, compiler9.855657s; final17 caller14.308313s, compiler11.193339s (`lx-Xg4-121702-480390-2853`). Non-compiler remainder falls4.539432→3.114974s, while compiler work rises1.337682s. That remainder includes lowering/planning and execution, not a pure steady-state metric. The total cold caller is consequently nearly flat in this pair. The unprofiled1.27s stage improvement is one measurement, not a claim that all cold runs improve by that amount.

cProfile records loop own time0.006→0.002s, gamma operand preparation0.082→0.016s and finish/unfold0.324→0.323s. Instrumented `jax.block_until_ready` calls fall97→37, inclusive0.414→0.204s; these include instrumentation and overlap kernel timings. Production source removes84 of its96 per-block fences (48 inner+48 outer become12 inner). The compiled producer removes repeated block views/mixing and eager stack preparation. Remaining first-use GEMM/projection planning is6.857718s/84 compilation events in17, with warm lookups0.083837ms; the `_zeros` fresh-JIT planning issue is registered, not described as warm dispatch overhead.

Repository gate23 (`lx-Xg0-122036-497348-5531`) passes140 AST checks but fails18 rule entries and the shared ledger's12 missing-evidence rows. One new entry required explicit `unroll=1` on the scan; that default is now written explicitly. The photon head `device_put` rule entry predates this work (four calls at the pin, now two after hoisting); no allowlist or shared ledger is changed. A subsequent gate will verify the remaining rule scope. This is not an overall repository-gate pass.

A final bounded producer ablation is justified by17's2.739372s compiler work in four producers: use `mix_lorentz_blocks`' existing coupled tensor contraction instead of its requested-key expansion. That keeps the same sole symmetry owner and reduces the expanded sum graph; its benefit is bounded by those cold seconds, since warm producers already cost below1ms/class. It must pass the same complex restore, printed-digit and HLO gates before adoption.

### Coupled-mixer ablation: rejected

The existing coupled mixer passes static24 (`lx-Xg4-122808-536158-1837`), dynamic27 (`lx-Xg4-123412-568024-9639`), native29 (`lx-Xg4-123714-581890-9106`) and22 CPU oracles. EQP/210-row identity also holds for capture26 and host28. It produces no useful cold improvement: class producer compiler work2.739372s in17 versus2.737722s in28; whole static stage20.46s versus20.53s. Matched warm host producer median worsens0.650788→0.930450ms/class; native-capture host median0.837740→1.079571ms/class. TT producer heap increases139,388,796→143,441,564 bytes/rank. All producer/core optimized HLO collectives remain zero. The three-line source experiment is reverted; its runtime source.diff and measurements are retained, and the canonical service itself is unchanged.

Repository gate30 (`lx-Xg0-123242-556936-6542`) verifies that the new scan-unroll finding is gone. The140 AST checks pass;17 other rule entries and12 shared-ledger missing-evidence rows remain. This remains a broad-gate failure. The adopted source changes for the orchestrator are `1b81c9cc` (scan) and `d994f90d` (compiled class producer plus explicit default unroll). Both are pushed on this branch, unmerged.

P16 attempt: prepare unprofiled and rank0 native controls from private copies of the same6x6 inputs/tmp. Baseline sources will be temporarily restored byte-for-byte to71ae0bde inside this worktree, then the committed candidate restored before its arms. Runtime guards compare every src/services file against the selected pin, and a shell trap restores the candidate on any failed baseline leg. The P16 native captures target exactly the first warm SX TT unit (old block-call index21; new class-call index7), with all other units still timed on the host and all optimized HLO dumped. No other worktree is modified.

### Architectural disposition after the 6x6 measurements

| priority / boundary | disposition and measured or bounded opportunity | additional lifetime / rule ownership |
|---|---|---|
| repeated G/FFT/projection across Lorentz blocks | Adopt the scan: TT70.444→10.878ms per complete warm class in the selected native units; CT42.757→17.230ms and TC42.078→16.996ms. | One G/class, one accumulated Sigma; class scalar inputs add about50MB/rank at P4. Canonical Green, FFT, gamma and band projection owners are reused. |
| per-output restore and eager class preparation | Adopt the compiled producer:300→48 source restores/run,48→12 restore dispatches,25 fewer whole-run compilation events. | Only compiled programs are cached; class arrays are streamed. All centroid-square operands retain both processor axes. |
| GEMM plan dummy initialization | Defer to the service owner:60 `_zeros` calls cost2.653s inclusive during first-use planning. Shared initialization or abstract lowering operands could target part of that bound. | Requires service-wide backend/geometry gates; no claim that all2.653s is removable. Registered at distrib_la/matmul.py:187. |
| resident faces across classes/terms | Defer: all non-GEMM/non-FFT fused core work in capture15 totals under60ms/run, so face reuse can save only a subset. | Four resident full-k family faces would hold73,359,360 bytes/rank on this deck, versus transient class pairs55,296,000 /36,679,680 /18,063,360 bytes. Scales as K s nb(M_C+M_T)/P, while parent inputs scale with n_parent. No additional face-carrier API is justified by this bound. |
| singleton CC specialization | Defer: final selected CC32.027ms versus29.185ms baseline gives about8.5ms/run opportunity, after removing the NCCL-wait difference seen in08. | Could reuse the fused singleton path, but another specialization is not justified by this small gate-deck saving. This is not called an inherent tiny-deck slowdown. |
| coupled tensor Lorentz mixer | Reject: no useful compiler saving, slower warm producer and4,052,768 additional TT bytes/rank. | The canonical service stays unchanged; the run-local source variant is reverted. |
| final band unfold/output seam | Retain: six finish calls cost0.323s, essentially unchanged. | Still once per term/sector, with canonical P-agnostic files and the sole band-operator unfold owner. |

The steady-state opportunities are separated from cold planning. These measurements do not claim a many-second cold caller speedup: the matched Python-profile caller is nearly flat because larger producer/core compilation offsets the removed host work. The main structural result is fewer dispatches, shared Green/projection work and bounded resident memory, with exact science.

Baseline rule audit31 (`lx-Xg0-124825-635859-6424`) ran the provided checker with src/services exactly at71ae0bde. `rule_scope_comparison.json` confirms the same17 finding categories as candidate30, with no added or increased counts; photon head device_put4→2. This establishes the scope of the remaining broad-gate failure without changing the shared allowlist or ledger.

### Final schedule, P16 disposition and handoff

Dynamic02 (`lx-Xg4-113441-218949-9472`) and final16 (`lx-Xg4-121517-471916-2792`), JID57982945, print the same eight node digests, not merely the same116-node count. In printed order the counts/digests are15/9082528ff4f0e7bb,10/c076707524dfd957,17/1255a67225d857e0,15/7f04851aaaa7ba72,14/9ed26293b488db1a,14/81b5a1f08d217250,13/76999d7f2b404689,18/7603613289643180. The exact rows are preserved in `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/dynamic_schedule_receipt.json`; source artifacts are each arm's `driver.rank0.log`. Baseline rebuilt two missing rules; final reused their copies. The planning reduction is therefore excluded from the scan's speedup.

The P16 baseline launch19 on authorized JID57982945 was refused after `--wait 1800` with `LX-POOLFULL`, rc96; **no compute step started**. Evidence: `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/19_P16_baseline/driver.1.log` and `p16_sequence.log`. All four fully free nodes were unavailable. Profile20 and candidate21/22 remain unrun; no P16 identity, timing or scaling claim is made. The trap restored the tested source; `p16_restored_source_check.txt` records an empty successful diff against HEAD, whose production source also matches d994f90d. The earlier fallback allocation was QOS-refused; another lane's pool was not used.

Adopt **1b81c9cc** and **d994f90d** from branch `perf/bisp-prof-s-2026-09-06`: the class scan and compiled class restore producer. The coupled-mixer experiment is rejected and reverted. P4 static/dynamic printed-digit and sector gates, native complex/head gates, rank0 native captures and eight optimized core/producer collective censuses are complete. Zero explicit HLO collectives does not mean communication-free: native DLA NCCL remains. The broad repository gate remains failed on17 inherited rule categories and12 shared-ledger entries, with140 AST checks passing and no new/increased rule counts. Cold planning remains a measured limitation; the accepted structural gain is shared class work and fewer dispatches with bounded memory. These commits are on this branch, not claimed merged into main.

## Phase 3: dynamic faces and compile parity on 6x6

Objective and preregistration: rebase onto fetched parent18196944, retaining the accepted class scan/producer and upstream process-local Gamma placement. Measure the current combined source before changes on private copies of common-basis104–107 and the Si common-rule GN deck. First replace the dynamic CC operator transport with canonical child-face unfolding and a full-child-batch Green GEMM; gather scalar weights without conjugating complex time. Gate both EQP files and complete sector rows and the complex antiunitary oracle. Then compare per-module compiler receipts, prioritizing repeated GEMM dummy initializers/shape-identical plan creation, parent/family kernels, seam preparation and band unfolding. Target P counts/time at or below F; any unmet target remains explicit.

The common-basis104–107 logs contain aggregate compile-cache summaries, not per-module seconds or XLA dumps. New diagnostic legs therefore wrap the same native compiler entry point used by the existing cache counter, retain module identity, backend seconds and source call stacks, and dump XLA HLO. This is run-local instrumentation, with no production env/deck additions. Its total is checked against the production counter. The rank0 native tau capture targets the second invocation; its receipt must show zero compiles before it is called warm.

Run32 is queued on authorized57988457 behind a four-node campaign step. Named allocation attempt34 was refused with QOSMaxSubmitJobPerUserLimit; the receipt is `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/34_P3_allocation/attempt1.log`. No science verdict exists for phase3 yet. Source-only finding: repeated exact C_q construction is in `src/isdf/core.py:_c_q_face_parent`, which creates a fresh `_fused` jit on each call; it is an ISDF/ZW boundary, not a Sigma factory, and is reported here for that owner without editing it.

CPU baseline38 passes23 physics oracles (`driver.1.log`, JID57988457). Run-local candidate39 failed before arithmetic on an unbound manual mesh axis; candidate40, with the static kernel's shard_map wrapper, passes all23 including the complex antiunitary tau sum. Production remains frozen for GPU baselines. CPU initializer probe41 (`lx-Xg0-134320-881449-7187`, JID57988457) measures10 exact repeated `_zeros` calls: incumbent10 compiles/0.240868s, hoisted JAX function1/0.021115s, exact values and sharding. This justifies testing a shape-keyed initializer after the dynamic gate; it does not establish a whole-run improvement.

### Dynamic CC before change

| deck / native capture | nodes | first instrumented host ms / warm median ms | selected warm device ms / CUDA sum ms | transport-heavy CUDA range ms | optimized tau bytes/rank | explicit HLO collectives |
|---|---:|---:|---:|---:|---:|---|
| MoS2 6x6, run33, lx-Xg4-134911-920500-8530 |115|196.056 /33.910|32.384151 /31.615167|14.3603|1,787,348,337|none|
| Si GN, run24, lx-Xg4-135417-946966-8718 |29|195.217 /40.612|39.064191 /38.316097|14.9653|2,019,986,453|none|

Both captures are rank0 on JID57988457 and retain exact EQP/sector rows against their rebased controls (210 MoS2 rows,256 Si rows). The captured unit is invocation1, zero-based, with zero compiles; CUDA startup adds about11s to its host wall, so that outlier is not presented as a tau execution cost. The tables use the median of all warm calls and native launch correlation. Native artifacts: MoS2 `prof_s/33_P3_dynamic_profile/unit_2160_0.json` and Si `prof_s/24_P3_gn_profile/unit_1289_0.json`, with boundary.jsonl, async_collectives.json and canonical HLO census beside them. The native ordinal0 is the only captured occurrence, corresponding to warm host invocation1. DLA NCCL is still present despite zero explicit HLO collectives.

The transport-heavy range includes the open-spin rotation and fused layout/gather work; it is not a standalone spin-only timing. Per-rank complex128 payloads: MoS2 K36,s4,N80,M600,P4 has two child faces55,296,000 bytes and one full operator829,440,000 bytes; Si K64,s4,N32,M480,P4 has two faces31,457,280 bytes and one operator943,718,400 bytes. Parent-face payloads replace K with n_parent7/8. The candidate keeps the final full Green; a smaller transport input does not itself prove a lower optimized peak.

| Speed item | Site | Gate step / artifact | Before → after |
|---|---|---|---|
| 1: dynamic CC child faces | `src/gw/ppm_tau_kernel.py::_get_sigma_kij_kernel` | `lx-Xg0-141347-1062624-1170`; `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/47_cpu_dynamic_faces/` | Complex frequency-weighted antiunitary oracle: 1 passed; CPU compile 10 / 0.3231 s; warm timing unavailable in this oracle. MoS2 boundary operator 829,440,000 → two faces 55,296,000 B/rank; Si 943,718,400 → 31,457,280 B/rank (final G retained). |
| 2: compile parity | `distrib_la.matmul::_zeros`, `matmul_plan::_build_kernel` | `lx-Xg4-141642-1079191-7680`; `prof_s/49_P3_compile_parity_fixed/` | Dynamic 874→759 events (F772); compiler 59.22→55.94 s (baseline32 ordinary, after includes per-module receipts); total98.911→93.096 s (F86.9: target still unmet); 115-node warm median33.910→28.167 ms (before33 host, after49); exact210 EQP0/1 and sector rows. Static duplicate bound649−63−12=574≤F578, not yet measured. First attempt48 failed with a malformed edit/NameError; corrected in49. Module-by-module artifact: `prof_s/dynamic_compile_change.json`; baseline44 HLO seconds are diagnostic-only. |
| 3: retire optional four-spin operator optimization | `symmetry_maps/maps.py:1325`; dynamic caller `ppm_tau_kernel.py` | Dynamic face gate `lx-Xg4-141642-1079191-7680`, `prof_s/49_P3_compile_parity_fixed/` | `ebc98918` is absent from the rebased source and is not to be re-adopted: added Sigma memory0 B/rank, avoiding its measured+467,644,160 B/rank on Si. Dynamic CC and photon static classes now unfold faces. Scalar static-limit/charge callers still use generic operator transport; their remaining calls do not justify reinstating a full-G-sized temporary. Source unchanged. |
| 4: Sigma seams and dispatch | `photon_sigma.py`, `sigma_x_bispinor.py`, `v_q_bispinor.py` | `lx-Xg4-142245-1116304-9394`; `prof_s/50_P3_sigma_seams/` | Static649→570 compiles (F578); matched HLO backend67.482→59.066 s; exact210 EQP0/1 and sector rows. Fused band-unfold/output reshard into one `finish` executable; same-family two-axis packs32→22 dispatches/full sixteen-block read; class fences12→0. No isolated warm-ms receipt, so no host-gap speedup claimed. Four class HLO cores retain zero explicit collectives; centroid-sized transposes at GEMM/FFT ABI and destructive FFT input copy remain (CC copy51,840,000 B/rank), not removable by relabeling a carrier. |
| 1b: remove second Green and audit the actual unit | `greens_function_kernel.py::build_G_tau`; face caller from9cf925d9 | `lx-Xg4-143202-1179356-6474`; `prof_s/55_P3_single_green/` | Exact210 EQP0/1 and sector rows; HLO tau1920:3 cuBLASMp calls=1 Green+2 projections, zero explicit collectives; host warm33.910→28.472 ms, native32.384→26.910 ms. Broadcasts112→200: full-child G costs144, two parent projections56. Peak1,787,348,337→1,781,847,793 B/rank. The extra antiunitary parent Green helper is deleted; generic transport delegates its transpose to SymMaps. The production face path evaluates all child rows in one G batch; it does not derive completed antiunitary G rows from an already-built spatial G. Thus the requested two-total-GEMM/84-broadcast target is not achieved by this data flow; parent-transpose alternative58 is measured separately. Profile compiler755/126.41s includes HLO+Nsight overhead. |
| 5: P16 common-basis arms | Phase3 source, copied private caches | `prof_s/51_P_static_P16` `lx-Xg4-142601-1131204-5353`;52/F-static `lx-Xg4-142753-1149718-4952`;53/P-dynamic `lx-Xg4-142942-1160767-4536`;54/F-dynamic `lx-Xg4-143208-1179945-1205`; JID57988457 | All four arms exact210 EQP0/1 and sector rows against their own P4 controls. Static P/F events542/604, compiler34.59/31.05 s, Sigma17.05/34.03 s, screening62.81/40.56 s, total103.09/98.72 s. Dynamic115 nodes: events758/825, compiler52.31/46.34 s, Sigma44.392/71.904 s, total138.576/147.598 s. No isolated warm-node capture in these legs. Static whole-run target remains unmet in screening. |
| 1c: finished-parent-G transpose ablation, rejected | Run-local `58_P3_parent_transpose/tau_candidate.py`; canonical `SymMaps` transpose | `lx-Xg4-144411-1246644-3392`; `prof_s/58_P3_parent_transpose/` | Exact210 EQP0/1 and sector rows. One parent G, antiunitary children from completed-G transpose:3 GEMMs=1 G+2 projections,84 broadcasts+1 SendRecv, one explicit collective-permute in async HLO census. Native31.706 ms / host warm33.275 ms versus face55 26.910/28.472 ms; peak1,787,347,825 B/rank. Rejected: it restores expensive open-spin operator transport and the forbidden intra-kernel exchange. Face production retained. Deleting the second Green alone cannot yield two total GEMMs while the two-stage band projection remains. |
| 6: coupled current LU, shared existing coordinator | `gw_init.py::_CoupledMu123ZqCoordinator`, `isdf_fitting.py` | `lx-Xg4-144726-1257856-5237`; `prof_s/59_P3_coupled_lu_timed/`; reference56 `lx-Xg4-143508-1192253-5640` | Four ζ files bit-identical, rtol0. Three7-q factors→one21-q factor; cold1,151.202→689.782 ms, factor compiles4→3; three warm solves40.979→34.248 ms (16.4%). Resident Gram/LU payload3,226,944 B/rank unchanged; concatenation temporarily adds3,226,944 B/rank, scaling48 Q μ_packed²/P; existing factor owner retains native pivots. Gate uses default batch_reshard/local-JAX LU despite the pre-factor distributed banner (registered); provider-auto token branch not separately exercised. Run57 timer included781.5ms queued Z work, corrected in59. Whole-fit compiles247→249. ZW repeated-C_q factory remains `isdf/core.py::_c_q_face_parent`, outside Sigma; no second cache added here. |

Phase3 disposition: adopt `9cf925d9` (dynamic faces), `bebb3a9c` (numerical compile keys), `b3bfe6f0` (Sigma seams), `b63a8d77` (single Green helper), and `b0b4a205` (coupled current factor/solve), all pushed on `perf/bisp-prof-s-2026-09-06`, unmerged. The finished-G transpose alternative is rejected by measured slowdown and its collective-permute; the face kernel has three total GEMMs and200 broadcasts, not the requested two/84. Static P4 matched-HLO totals89.69/90.86s beat F, while P16 static totals103.09/98.72s remain slower because screening costs62.81/40.56s. Dynamic P16 totals138.576/147.598s beat F; final unprofiled P4 whole-run advantage is not established. Compile-event targets are met on both P4 and P16; compiler-seconds parity is not. Complete module-class counts, seconds, source sites and dispositions are in `prof_s/compile_class_dispositions.json`. The independent provider-auto coupled-factor gate and isolated P16 warm-unit capture remain unrun under the one-leg protocol; gate59 covers the shipped default batch_reshard route. Under the speed protocol, no suite or gate0 was run.

## Compiler seconds and remaining Sigma costs

| Change/site | Gate and identity | Measurements / disposition |
|---|---|---|
| Rebase b1d8b8f1; map canonical per-class W restores through one traced body (`w_isdf.photon_blocks_full_q`) | JID57992641, lx-Xg4-150851-1389299-3427; `60_P4_mapped_restore/gate.log`: 210 EQP0/EQP1 and sector rows exact | 502 events / 29.895 native compiler s; four restore modules 0.1415/0.2136/0.2083/1.0649 s. New parent-stack upper payloads CC/CT/TC/TT=10,080,000/10,281,600/10,281,600/10,487,232 B per rank, scaling16*blocks*n_qparent*M_A*M_B/P; full-q stack payloads remain51,840,000/52,876,800/52,876,800/53,934,336 B per rank. These are array payload bounds, not peak-memory deltas; see mapped_restore_memory.json. Before HLO-instrumented run50: 570 / 59.066 s, restore 3.1642 s; diagnostic mismatch and integrated ZW changes prohibit attributing the total difference to mapping alone. Warm restore timing unavailable: instrumentation preparation failed before launch; no duplicate gate. Ten largest parent-only modules (class name absent F36) retained in `parent_only_top10.json`. |
| Retire optional ebc98918 spin blocks | Existing run55 one-G exact gate lx-Xg4-143202-1179356-6474 plus current source census `git grep unfold_operator -- src/gw` | No Sigma caller remains: only the unused plan method definition. Generic Green now also uses typed child faces (orchestrator 6a1b8b6c/bf1f13ef). Optional ns=4 specialization remains absent; added Sigma memory 0 B/rank, avoiding its previously measured 467,644,160 B/rank increment. No numerical source change or duplicate gate. |
| Compiler-seconds P4 dynamic follow-through | JID57992641 lx-Xg4-151133-1405561-2426; run61 exact210 EQP0/EQP1 and sectors, copied115-node rule | 672 events /47.06 counter s /44.547 native compiler s; warm tau 29.038 ms; total80.49 s < common-basis F10786.9 s. Static run60 total56.05 s < fixed-main57.1 s. Counter compiler seconds remain above F by1.36 static/1.86 dynamic; whole-run targets met on these observations. |
| Gather/trim parent bands before canonical band unfold (`photon_sigma.finish`) | JID57992641 lx-Xg4-151334-1417519-1781, run66 exact210 EQP0/EQP1 and sector rows | HLO finish:4 full-k all-gathers →2 parent-k all-gathers; heap9,915,572 →1,773,492 B/rank; warm finish median0.759 ms (first221.325 ms including compile). Compiler finish0.186→0.197 s, no compiler win claimed. Four class cores retain zero explicit collectives. Parent band gather scales16*n_parent*nb² bytes, full output16*nk*nb_sigma²; no centroid-square tensor replicated. |
| Ten most expensive parent-only modules after rebase/mapping | run60 native receipts, JID57992641 lx-Xg4-150851-1389299-3427; absent module names relative to F36, not bytecode-equality classes | 1. jit_restore `77e79933` 1.0649s; 2. jit_contract_class `14174887` 0.5031s; 3. jit_contract_class `9a76e9b1` 0.4948s; 4. jit_contract_class `ebf7c6d1` 0.4863s; 5. jit_contract_class `2186a143` 0.4668s; 6. jit_restore `9b54fb94` 0.2136s; 7. jit_restore `206af00e` 0.2083s; 8. jit__photon_q0_factor_orbit `0d9c7048` 0.1960s; 9. jit_body `4113de2d` 0.1758s; 10. jit_body `fb3714c1` 0.1740s. Full stacks/digests in `parent_only_top10_after.json`; map removed repeated restore trace bodies; four different endpoint extents retain four class specializations. Plan captures actual symmetry arrays, so dropping identity from keys without making these runtime operands is unsafe. Remaining head bodies belong to the canonical head owner already optimized by ZW. |
| Remaining class-core/producer copies and >5% tau costs: disposition | run66 optimized HLO/thunks, JID57992641 lx-Xg4-151334-1417519-1781; native tau55 lx-Xg4-143202-1179356-6474 and rejected58 lx-Xg4-144411-1246644-3392 | Class/restore cores: zero explicit collectives. CC core still materializes centroid-major GEMM → spin-axis FFT transpose (829,440,000 B/rank; corresponding tau55 transpose2.468 ms=9.17% of one node, only0.284 s across115 nodes) and destructive interaction-FFT copy (51,840,000 B/rank); canonical `merge_spin_centroid` rules require centroid-major merge to avoid all-to-all, current FFT service requires spin-axis layout. Removing these needs a coordinated service/layout interface change, not deleting JAX transposes or introducing a local spin-major merge. Band seam removed redundant full-k gathers above. Native tau costs: Green9.241 ms, convolution6.797 ms, two projections6.590 ms; embedded NCCL10.325 ms overlaps those categories. Finished-parent-G transpose alternative31.706 ms > child-face26.910 ms and adds collective-permute: rejected. The face-spin GEMM itself is0.287 ms/node (<1.1%), so retaining an additional55,296,000 B/rank of child faces solely to hoist it is below the5% priority threshold. No further measured safe >5% change identified; no new FFI, symmetry rule, or replicated centroid-square tensor introduced. |
| Final P16 matrix (owner close-out) | JID57988457: run62 lx-Xg4-152905-1423784-7552; run64 lx-Xg4-153139-1509986-9665; run65 lx-Xg4-153519-1519234-9727; each210 EQP0/EQP1 and sector rows exact vs own prior arm | P static474 events/30.32 compiler-counter s, Sigma17.26 s, total77.12 s; CC/CT/TC/TT warm class281.40/210.27/138.15/171.71 ms. P/F dynamic675/825 events,44.56/45.94 compiler s, Sigma46.89/67.73 s, total118.25/141.07 s, warm tau135.49/196.53 ms,115 nodes each. F static run63 failed in timing instrumentation (None green_parent); prepared corrected67 canceled before launch on owner instruction, so no new F static comparison. Raw/reduced receipts in p16_followup_summary.json and run dirs. |

## Final disposition

1. Integrated on `feat/bisp-parent-route-2026-09-05` (not claimed on main): mapped canonical restores and parent-band output seam; source matches fetched orchestrator `d4fc26eb`.
2. Measured: P4 static/dynamic totals56.05/80.49 s vs existing F57.13/86.9 s; final P16 dynamic118.25 vs F141.07 s, warm135.49 vs196.53 ms; finish heap9,915,572→1,773,492 B/rank and4→2 gathers, all completed gates210 printed rows exact.
3. Left: P4 compiler-counter excess1.36/1.86 s static/dynamic; retained Green layout transpose2.468 ms/node (829,440,000 B/rank); F P16 static instrumentation retry67 canceled unrun; optional spin-block added Sigma memory0.
4. Remaining source cherry-picks: `[]`; required order `da4d2ab7` then `2310e16d` is already integrated as `026b2992` then `c7e7952e`, and the prior phase3 fixes are present.
5. Do NOT take rejected parent-transpose run58, optional `ebc98918`, failed run63 or unrun67 as production fixes/evidence; do not cherry-pick ancestry-only rebase merges or reapply the already-integrated source commits.

## Reopened fix: parent Green and typed local transport (base c2f69987)

| P4 gate, all JID58001243 | Before → final τ timing | Final module and compile receipts | Printed identity and artifacts |
|---|---|---|---|
| Si charge-only, 8 parents/64 children, 700 nodes; `lx-Xg4-183148-21883-3115` | Regressed90188921 sweep30.27 →20.91s; pre-regression7c5466be21.92s, main22.4s. Final warm24.854ms; sweep/node29.871ms. Before43.243ms is amortized, not a captured warm node. | 2 projection cuBLASMp +1 local parent cuBLAS; zero explicit collectives. Whole-run343 native compile events/22.510s; τ compile0.574s. | 224 EQP0/EQP1/complete sector rows exact vs Si99/31_bisect_scalar_tau/7c5466be. `runs/Si/100_bisp_parent_route_2026-09-05/prof_s/36_scalar_sparse_spin/{gate.log,green_summary.json,boundary.jsonl,xla_dump_rank0/}`. |
| Si SOC GN, 8/64, 29 nodes; `lx-Xg4-183313-24970-9150` | Same-tip child-face control32 warm35.897 →29.987ms. Sweep2.70 →2.80s: no whole-sweep improvement claimed on this short leg. | 2 projection cuBLASMp +1 local parent cuBLAS; zero explicit collectives.406 events/25.077s (control389/23.541s). | 256 complete rows exact vs unchangedc2f69987 control32. Original Si56 is **not** identical at this source tip: both control and candidate differ by max0.518µeV,58/256 EQP0 rows unchanged. `.../prof_s/37_soc_sparse_spin/{gate.log,green_summary.json,xla_dump_rank0/}` and `.../prof_s/32_soc_c2_control_fixed/old56_gate.log`. |
| MoS2 6×6 dynamic, 7/36, 115 nodes; `lx-Xg4-182907-6545-6500` | Previous best child-face55 warm26.910 →24.033ms, meeting26.9ms target. Local parent-G candidate70 with dense structural-zero products32.947ms; omitting only typed zero entries removes8.914ms. Final sweep5.80s. | 2 projection cuBLASMp +2 local parent cuBLAS sites (second conditional on complex weights); zero explicit collectives.595 events/43.333s; τ compile0.725s. | 210 complete rows exact vs common-basis106, using its copied replay_rules and original replay hook. `runs/MoS2/42_bisp_scale_2026-09-06/prof_s/72_dynamic_sparse_spin/{gate.log,green_summary.json,boundary.jsonl,xla_dump_rank0/}`. Before best55: JID57992641, `lx-Xg4-143202-1179356-6474`; source/reference distinction retained. |

`build_G` now contracts Q parents, supplies the conjugated-face parent operator only for antiunitary children with complex weights, and delegates rectangular endpoint transport to the existing symmetry owner. Static class kernels and dynamic τ use this same parent-G/transport algorithm. Static and screening callers retain their parent distributed GEMM plans; dynamic Σ hoists band replication of its two small faces outside the sweep and uses local JAX GEMM, deleting its distributed Green plan and full-child-face unfold. No spin-extent switch, duplicated rotation rule, new FFI, environment dial or deck key. The two spin contractions omit only entries that are identically zero in the supplied typed matrix; this is what recovers the four-spinor speed. The kernel census includes async collective starts: zero explicit collectives inside τ, with communication internal to the two existing projection FFIs unchanged. No distributed GEMM or processor exchange occurs in transport.

Per-rank complex128 bytes at P4 (actual charge-only HLO has **ns=2**, not the requested description's ns=1):

| Deck (Q,K,ns,M,N) | Two band-replicated parent faces | Previous two full-child faces | One Q-parent operator | One transported K-child operator | Final τ temporary bytes |
|---|---:|---:|---:|---:|---:|
| Si charge-only (8,64,2,840,80) |17,203,200|68,812,800|90,316,800|722,534,400|1,806,336,260|
| Si SOC (8,64,4,480,32) |7,864,320|31,457,280|117,964,800|943,718,400|2,359,296,512|
| MoS2 (7,36,4,600,80) |21,504,000|55,296,000|161,280,000|829,440,000|2,073,601,540|

For a square processor mesh the face pair scales as32 Q ns M N/sqrt(P), the old child pair as32 K ns M N/P, and G as16 Q(or K) ns² M²/P. The original projection carrier remains live: new replicated faces add the full pair shown to residency; the τ argument-size increase relative to distributed Green faces is half that pair at P4 (MoS2 +10,752,280B including the selector). All centroid-square objects remain on all P ranks. The conditional conjugated-face G adds one parent-operator buffer; the captured Si Green plans have no antiunitary child selection, despite global SOC symmetry containing antiunitary rows; MoS2 exercises it. Sparse spin action changes temporary memory by **−414,720,000B/rank** on MoS2 (70→72) and **+471,859,200B/rank** on Si SOC (34→37): the latter is a measured speed/memory tradeoff, not zero-cost spin action. Do not reapply optional ebc98918; the canonical generic action here replaces that specialization.

Native compiler receipts are distinct from `[compile-cache]` boundary counters: final calls are prewarmed (zero boundary misses), and whole-run/compiler totals include earlier planning. In scalar36 the parser charges15.36s to rule planning/prewarm; total76.91s is **not** a claimed whole-run speedup. The HLO analyzer uses analysis index0000 for the captured executable; absent buffer-assignment files must not be read as zero memory—`tau_memory.txt` supplies executable memory statistics. No Nsight timing attribution or ns=1 runtime gate is claimed for this reopened fix.

Stopped candidates:27 failed a parent/full-k Green-plan extent;33 failed after changing the shared screening carrier's placement (reverted);68 used the wrong replay hook;31 lacked archived pyproject.toml.28/30's Si56 comparison failure was reproduced by unchanged control32, not fixed by quadrature-edge pinning. Final Si SOC and control use the same copied schedule with a run-local ≤1e-10Ry certificate-edge tolerance; production quadrature logic is unchanged. No additional legs or suites are pending.
