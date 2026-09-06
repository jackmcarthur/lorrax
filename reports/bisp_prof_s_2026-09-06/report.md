# BISP-PROF-S — parent-route Sigma profiling

Heavy investigation; **measurements in progress on authorized campaign pool 57966610**. Branch `perf/bisp-prof-s-2026-09-06`, unmerged. Production source remains unchanged at P=`9f569c4bf75bad40e4f5895946874b4c503e4410`; F is read-only `wt_main_de8dcfbc_fixed` at `e1559a071e244b4f049c924781b668d9e1560739`. The first production plan-reuse fix has passed its same-deck P4 gate; subsequent changes and combined verification are in progress.

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

Static captures are complete; their reduced receipts appear below. Dynamic captures are running with common certified schedules. We use PERF2's `cudaProfilerStart/Stop` inside `runtime.run_main_and_finalize`, with `jax.effects_barrier()` before stop. The prior installed nsys path is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys`; it is available and produced valid CUDA captures in this allocation. Capture rank0 while all four ranks execute the same science. Export `nvtx_gpu_proj_sum`, `nvtx_kern_sum`, `cuda_gpu_kern_sum`; verify actual CUDA records, then isolate one warm static block and one warm tau node. Native kernel sums and projected spans must remain separate; no nested-range summation.

## Saved Sigma device profile

Static P/F native timings and kernel classes are reported below; dynamic reduction is pending. Fused spin/gather costs are bounded by their enclosing native range, not presented as separately measured kernels.

### HLO collective census per stage

Static optimized dumps have been analyzed with `tools/hlo/analyze_hlo_dump.py` and PERF2's collective census; results below. Count async starts once and identify loop multipliers. HLO-zero does not establish communication-free distributed GEMM FFI. The parent orchestrator report explicitly says the literal final-psum-only criterion is not met; this lane has not disproved it. In particular, do not relabel transposed antiunitary transport as an ordinary local gather without inspecting the generated program.

## Owner's boundary-accounting table

**Source census at the pinned P/F trees, not measured module or kernel counts.** Let B be selected Lorentz outputs, T=3 terms, H=1 when head diagnostics are active (otherwise 0), K=nk_full, Q=n_parent, N=nb, s=ns, M_C/M_T=packed endpoint extents and p_x p_y=P. A source-level call may be optimized away or fused; the requested ms/unit and seconds/run columns remain unmeasured.

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

`w_isdf.py:2003` owns implementation, but calls from `photon_sigma.contract_lorentz_blocks` occur during Sigma. Charge **those invocations once to Sigma**, including restore/mix compilation there. Screening's own invocations, if present, remain screening. ZW should read this section; this lane has not edited `w_isdf.py` or any screening file. A prospective restore-lifetime fix needs an ownership agreement before either lane edits the shared function.

## Architectural proposals

This ranking is **provisional, by removable invocation count and implementation
complexity**, not measured speedup. No candidate is accepted for production until
baseline device and compile receipts exist. Boundary ms and allocator peaks are
still unknown. The historical MoS2 full-static Sigma wall, 150.14 s at BFC@0.85/P4,
is only a trivial upper bound on any single Sigma optimization's elapsed saving;
it cannot be divided among source calls to manufacture per-call timings.

All candidates retain canonical P-independent files and typed SymMaps actions.
All mu-square intermediates must remain distributed on both mesh axes. The
final-psum-only requirement is a gate to establish, not an invariant certified by
this source inspection. Acceptance requires eqp0/eqp1 tolerance zero, printed
sigCC/sigTT/sigCT identity, optimized HLO payload/collective checks and the combined
P4 leg. Changed summation order is not excused by a loose cross-source tolerance.

| priority / candidate | boundary and numerical work bound | proposed flow, deletion, and disposition |
|---|---|---|
| 1. Restore callable lifetime | 300 restore/add JIT creations plus 48 zero initializers for full static | Reuse compiled restore/add work at its existing owner across terms, preserving C,D order and streaming one output. Deletes loop-local callable creation. No extra resident dense output required. Coordinate implementation in w_isdf.py with ZW; pending measurement and ownership agreement. |
| 2. Share body/head G | 96 G builds become 48 in a full-static run with heads; 192 face unfolds become 96 | Build G once per block/term and feed body and q0 convolutions before releasing it. Deletes separate G construction in q0-only path. Prefer one compiled call returning two small band operators so G lifetime need not escape the kernel. Bounded 50% reduction in G work, not 50% of Sigma wall. Pending measured head cost. |
| 3. Resident family faces | 192 endpoint unfolds become 12 for two faces per family per term, if vertices are applied after canonical unfolding | Unfold four unvertexed faces once at term entry; apply the canonical gamma vertex at each block consumer. Delete in-block repeated child transport. This requires separating the existing plan method's transport and vertex at their current owners, never reproducing their rules in Sigma. Up to 93.75% fewer face-unfold invocations with heads. Pending memory/trace evidence; families and occupations are not assumed interchangeable. |
| 4. Shape-class contraction | 16 body identities plus 16 head identities versus at most four ordered C/T shape classes per output mode | Stream a scan over vertex pairs within a shape class and accumulate band outputs, rather than materializing 16 G tensors. Delete vertex-specialized Python kernel loop/cache entries. A scan only helps if restores and canonical vertex application fit the same ownership boundary. Potential 32 to 8 callable identities is not a compile-count measurement. Reject literal all-G stacking; defer scan until costs of priorities 1–3 are isolated. |
| 5. Two 2x2 spin actions / fused gather | Dense 4x4 has 16 coefficients; two 2x2 blocks have 8 | Consume the two diagonal blocks of the typed action already built by SymMaps; do not recompute parity/det in Sigma. Replace the dense application at the symmetry service owner, shared by consumers. At most halves rotation multiply-add work, with unchanged face bytes; compiler may already eliminate structural zeros. Defer until device capture proves residual dense work and symmetry-owner coordination is established. |
| 6. Restore each block once per run | Current 300 source restores include repeated source blocks across output mixing and terms | A full completed-output cache needs 16 outputs per interaction instead of one. An alternative term-interleaved consumer restores one output's V and W, consumes X/SX/COH consecutively, then frees both; W−V restoration order must preserve printed digits. Deletes repeated term traversal, but source-wise versus completed-output subtraction changes rounding. Defer; reject unbounded full-run cache absent peak-memory proof. |
| 7. Fuse seam with reshard/writer | Seam invocation and ms counts unmeasured; existing PackedCentroidBasis conversion is already compiled | Extend the adjacent existing compiled consumer/writer boundary to consume its permutation once; delete the separate conversion invocation. Keep files canonical and the single basis-map owner. No new layout registry. Defer until HLO establishes an actual redundant copy/reshard and attributes it to Sigma. |
| 8. Change four-spinor face layout | At most removes measured transpose/copy bytes, currently unknown | Retain a single agreed face layout from producer through gamma action and G GEMM; eliminate explicit adapters instead of retaining old/new carriers. May conflict with the projection GEMM's preferred layout. No shape-only argument establishes a speedup. Defer until layout copies are visible in optimized HLO and native ranges. |
| ZW-owned. Coupled three-current zeta kernel | Three channels could share one unfolded face pair and C_q; at most two of three repeated shared constructions disappear | Scan or batch current vertices inside the existing fitting kernel, preserving tiled mu-square storage and one symmetry owner. Deletes per-channel repeated setup. No Sigma implementation proposed: send the bounded candidate through this report for ZW's measurements and gate. |

### Memory bounds for proposal selection

Let K=nk, Q=n_parent, N=nb carrier, s=ns, M_C/M_T=packed family
extents, and P=p_x p_y. For complex128, both resident faces for both
families require `32 K N s (M_C+M_T)/P` bytes/rank; the corresponding
parent carriers require `32 Q N s (M_C+M_T)/P`. Keeping parents for
projection adds the full child amount to their lifetime, not just the difference.
The child-face storage is independent of the Lorentz-block count; retaining vertex
variants would multiply it and is excluded from this proposal.

The historical MoS2 gate log records K=9, Q=3, N=80 and P=4
(`73_parent_full_static/driver.rank0.log`, checkpoint step
`lx-Xg4-203505-2030774-8671`). With s=4 and **illustrative packed**
M_C=M_T=192, parent faces are 2,949,120 bytes/rank, children add
8,847,360 bytes/rank, and combined carriers are 11,796,480 bytes/rank.
192 is the logged logical centroid count; actual packed extents and padded band
carriers must be read from the new runtime receipt before treating these numbers
as actual allocation sizes. These are payload estimates, not peak-memory measurements.

One G body costs `16 K s² M_left M_right/P` bytes/rank
(21,233,664 bytes in that illustrative case); 16 simultaneously live bodies cost
339,738,624 bytes/rank. Body/head reuse should retain one G, while batched
contraction must stream rather than multiply that footprint. One scalar full-q
interaction block costs `16 K M_left M_right/P` bytes/rank
(1,327,104 illustrative bytes). Sixteen completed blocks of equal extent cost
21,233,664 bytes/rank per interaction, or 42,467,328 for V and W. A two-block
term-interleaved consumer instead needs 2,654,208 bytes/rank before convolution
scratch. Distinct family extents replace 16 M² by `(M_C+3 M_T)²`.
These figures grow quadratically in centroids, so a small-deck fit cannot justify
an all-block cache for large systems. Spin-action specialization leaves face
payload unchanged; seam fusion and layout changes have no certified allocator
savings until alias/copy lifetimes are measured. Coupled zeta memory depends on
its real-grid tile and belongs in ZW's accounting.

No larger-deck/P16 result exists here. Therefore no remaining slowdown is labelled
inherent to tiny decks, and none of these work bounds is presented as a measured
speedup or an integration recommendation.

## Ablations

Completed run-local arithmetic-preserving experiments appear in the unprofiled accounting table below. Production source is still pinned while the dynamic captures run. Failed experiment M13 initialized JAX before the communicator; it is retained as a failed variant, registered in KNOWN_SANDBOX_ERRORS.md, and corrected in M19.

## Verification and current disposition

All ten pinned baseline arms and both static captures completed. Static captures and successful ablations pass tolerance-zero eqp0/eqp1 and all 90 complete printed state rows in sigma_diag.dat, including CC/TT/CT. Dynamic capture and production implementation gates are pending; no production fix is recommended for adoption yet.

Pool 57966610 is explicitly authorized by the user and shared with this campaign. Every arm uses one node sequentially. The earlier failed allocation is historical infrastructure evidence, not a current blocker. All writes, including local run/issue/claim registers, stay within this worktree; shared sandbox ledgers are read-only under the lane's explicit scope.

## Authorized measurements — 2026-09-06

User authorized BISP-orch pool57966610. Arms run sequentially, one P4 node each,
BFC@0.85, source arithmetic pinned as above. No production edits yet.

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

P's extra open-spin transport range is14.938586ms. Its two dense spin GEMMs sum9.646109ms and three surrounding operator layout kernels sum5.147709ms. This is distinct from the static face spin action's32us enclosing bound: dynamic transport acts on O(K s² M_C²/P) elements. A generic elementwise application of the **same supplied spin matrix** is now under ablation at the existing symmetry owner; it requires no determinant or block-diagonal rule in Sigma.

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
