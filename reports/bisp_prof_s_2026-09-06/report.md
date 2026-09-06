# BISP-PROF-S — parent-route Sigma profiling

Heavy investigation; **incomplete, awaiting an authorized compute allocation**. Branch `perf/bisp-prof-s-2026-09-06`, unmerged. Production source remains unchanged at P=`9f569c4bf75bad40e4f5895946874b4c503e4410`; F is read-only `wt_main_de8dcfbc_fixed` at `e1559a071e244b4f049c924781b668d9e1560739`. No performance fix or adoption recommendation has passed a gate.

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

## Ablations

Not run. A first priority is a restore-only compilation-lifetime ablation preserving the exact C,D accumulation order and the existing typed symmetry helpers. Follow with static vertex-shape specialization and body/head G reuse only if traces justify them. The open-spin dynamic operator route must be measured independently; static child-face duplication is not evidence for a dynamic-node change.

## Final verification and disposition

Incomplete: no new P/F drivers, Nsight capture, HLO census, ablation, source edit, printed-digit parity gate, or combined P4 test has run. There are no source commits for the orchestrator to adopt. The checkpoint accounting and prepared arms are useful preparation, not completion of the requested profiling/fix.

One prescribed allocation attempt failed with `QOSMaxSubmitJobPerUserLimit`; artifact `runs/Si/100_bisp_parent_route_2026-09-05/prof_s/00_allocation/alloc.log`. No allocation JID was issued. `scontrol show job 57955075` reports invalid job id (saved alongside allocation log). Other visible pools were not used. User input requested: authorized JID or permission for another allocation attempt after capacity clears. No GPU test was substituted with login-node compute.

Writes, including this report and local run/issue registers, stay in the assigned worktree under the explicit modify-only-worktree restriction. Shared sandbox ledgers remain read-only. The legacy `~/lorrax_service_phase/BUILD_NOTES.md` and `~/lorrax_bse_perf_2026-08-08/INDEX.md` were not present at the two explicit home aliases checked; no recursive home/shared-root search was performed. Current Perlmutter contract, PERF2 report/capture recipe, and supplied completed wrappers were read instead.

Preparation verification: all five P/F copied `tmp/` SHA256 lists are identical; `bash -n` passes all30 launcher scripts. These are file/shell checks, not CPU/GPU science gates. Large copied inputs remain ignored; committed checksums and the bounded `00_allocation/prepare_baselines.py` preserve their reconstruction recipe.
