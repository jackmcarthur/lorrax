# BISP-PROF-S — parent-route Sigma profiling

Heavy investigation; **measurements in progress on authorized campaign pool 57966610**. Branch `perf/bisp-prof-s-2026-09-06`, unmerged. Production source remains unchanged at P=`9f569c4bf75bad40e4f5895946874b4c503e4410`; F is read-only `wt_main_de8dcfbc_fixed` at `e1559a071e244b4f049c924781b668d9e1560739`. No performance fix or adoption recommendation has passed a gate.

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

Each has `manifest.yaml`, `cohsex.in`, `rankwrap.sh`, `driver.sh`, `run.sh`, copied `tmp/`, and `inputs.sha256`. None has run. `run.sh JID` pins every dispatch explicitly and refuses to overwrite an attempt. Source checks compare `src/services` against the code pin inside each rank. Host baselines should precede separately copied profiler variants; Nsight instrumentation walls are not baseline timing evidence.

## Valid rank-0 device captures

**Pending; no device timing or kernel-class census claimed.** Use PERF2's `cudaProfilerStart/Stop` inside `runtime.run_main_and_finalize`, with `jax.effects_barrier()` before stop. The prior installed nsys path is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys`; its availability in this allocation is untested. Capture rank0 while all four ranks execute the same science. Export `nvtx_gpu_proj_sum`, `nvtx_kern_sum`, `cuda_gpu_kern_sum`; verify actual CUDA records, then isolate one warm static block and one warm tau node. Native kernel sums and projected spans must remain separate; no nested-range summation.

## Saved Sigma device profile

Pending for both P/F: G GEMMs, face gather/permutation and spin action, convolution FFTs, projection GEMMs, parent-row gather, band unfold, W restore and Lorentz mixing, seam pack/unpack, NCCL kernels. No first-vs-warm ms or compile seconds invented from Python call counts.

### HLO collective census per stage

Pending optimized rank0 dumps and `tools/hlo/analyze_hlo_dump.py`. Count async starts once and identify loop multipliers. HLO-zero does not establish communication-free distributed GEMM FFI. The parent orchestrator report explicitly says the literal final-psum-only criterion is not met; this lane has not disproved it. In particular, do not relabel transposed antiunitary transport as an ordinary local gather without inspecting the generated program.

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

Not run. A first priority is a restore-only compilation-lifetime ablation preserving the exact C,D accumulation order and the existing typed symmetry helpers. Follow with static vertex-shape specialization and body/head G reuse only if traces justify them. The open-spin dynamic operator route must be measured independently; static child-face duplication is not evidence for a dynamic-node change.

## Final verification and disposition

Incomplete: no new P/F drivers, Nsight capture, HLO census, ablation, source edit, printed-digit parity gate, or combined P4 test has run. There are no source commits for the orchestrator to adopt. The checkpoint accounting and prepared arms are useful preparation, not completion of the requested profiling/fix.

One prescribed allocation attempt failed with `QOSMaxSubmitJobPerUserLimit`; artifact `runs/Si/100_bisp_parent_route_2026-09-05/prof_s/00_allocation/alloc.log`. No allocation JID was issued. `scontrol show job 57955075` reports invalid job id (saved alongside allocation log). Other visible pools were not used. User input requested: authorized JID or permission for another allocation attempt after capacity clears. No GPU test was substituted with login-node compute.

Writes, including this report and local run/issue registers, stay in the assigned worktree under the explicit modify-only-worktree restriction. Shared sandbox ledgers remain read-only. The legacy `~/lorrax_service_phase/BUILD_NOTES.md` and `~/lorrax_bse_perf_2026-08-08/INDEX.md` were not present at the two explicit home aliases checked; no recursive home/shared-root search was performed. Current Perlmutter contract, PERF2 report/capture recipe, and supplied completed wrappers were read instead.

Preparation verification: all five P/F copied `tmp/` SHA256 lists are identical; `bash -n` passes all30 launcher scripts. These are file/shell checks, not CPU/GPU science gates. Large copied inputs remain ignored; committed checksums and the bounded `00_allocation/prepare_baselines.py` preserve their reconstruction recipe.

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
