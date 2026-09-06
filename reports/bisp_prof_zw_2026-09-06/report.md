# BISP-PROF-ZW — heavy investigation, measurements in progress

On branch `perf/bisp-prof-zw-2026-09-06`, unmerged. Baseline P is exactly `9f569c4bf75bad40e4f5895946874b4c503e4410`; fixed-main F is exactly `e1559a071e244b4f049c924781b668d9e1560739`. Production source is unchanged. Pinned P/F measurements and strict-identity ablations are recorded in the live checkpoint sections below; the initial allocation-blocked audit is retained as preregistration history. No production fix has been adopted yet.

## Objective and preregistered candidates

Measure parent-route charge/current ζ and screening before changing production code. Preregistered candidates: repeated family unfold; spinor rotation and vertex contraction; coupled three-channel stacking; plan-identity compilation misses; family seam conversions; repeated full-q block restores. Retain only changes supported by device and compile receipts and same-source printed-digit eqp0/eqp1 and sigCC/sigTT/sigCT identity. No Sigma kernel edits.

Method: PSIIRR-PERF2 report and its `01*/payload.sh` and `profile_driver.py`, read in the sandbox. Use P4, one rank per GPU, cold persistent cache, and BFC@0.85 matching the supplied rank wrappers. Separate unprofiled stage walls from profiled device times. Copy the same ζ/rule cache to both sources and record actual selected node/weight digests; cache-directory equality alone is insufficient. An additional new fit variant is necessary to measure ζ arithmetic, because the supplied parent checkpoint arms reuse ζ.

## Before any source change

### Historical checkpoint harvest (not pinned-tip measurements)

The ten original `driver.rank0.log` stage tables and `driver.1.log` launch receipts were read from disk; complete paths, launcher commits, allocator lines, timing dictionaries, SHA256 log digests, and available node receipts are in `checkpoint_receipts.json` beside this report. All timing values below are **BFC@0.85**, as declared by those run-local wrappers. Screening is chi0 + W + screening support, excluding the separately printed W persist/head row; Sigma is rule planning + tau sweep + other. Absent ζ stage entries mean no fit stage was recorded, not a zero-cost fit.

| System/mode | arm | ζ s | screening s | W persist/head s | Sigma s | total s | original lx step (exit 0) |
|---|---|---:|---:|---:|---:|---:|---|
| Si cohsex | P | not recorded | 6.32 | 0.56 | 47.41 | 74.29 | `lx-Xg4-211207-2283249-4810` |
| Si cohsex | F | 7.6 | 5.25 | 0.09 | 11.35 | 56.32 | `lx-Xg4-200631-1862494-2046` |
| Si gn | P | not recorded | 7.57 | 0.55 | 66.05 | 91.46 | `lx-Xg4-203651-2042362-1297` |
| Si gn | F | 7.35 | 6.39 | 0.09 | 32.12 | 76.97 | `lx-Xg4-200444-1853904-8909` |
| MoS2 full_static | P | not recorded | 31.93 | 0.00 | 150.14 | 198.64 | `lx-Xg4-203505-2030774-8671` |
| MoS2 full_static | F | 8.2 | 12.65 | 0.00 | 18.40 | 70.98 | `lx-Xg4-200418-1851221-8871` |
| MoS2 packed_bare | P | not recorded | 16.19 | 0.00 | 149.95 | 182.66 | `lx-Xg4-204419-2081026-7329` |
| MoS2 packed_bare | F | 8.14 | 16.42 | 0.00 | 18.58 | 85.17 | `lx-Xg4-200532-1857555-6507` |
| MoS2 dynamic_eps5 | P | not recorded | 34.22 | 0.60 | 266.30 | 338.98 | `lx-Xg4-210723-2245574-4379` |
| MoS2 dynamic_eps5 | F | not recorded | 17.25 | 0.29 | 114.68 | 165.86 | `lx-Xg4-210725-2246074-4013` |

**Provenance limitation:** original P launch receipts print `fc7825a8`, and their manifests explicitly mention `source.diff`. Original F static/Si receipts print `de8dcfbc`; F dynamic prints `afeab2d0`. These checkpoint walls reproduce the dispatch, but cannot be relabeled P=`9f569c4b` versus F=`e1559a07`. Newly pinned measurements remain required. The fixed-main checkout itself was verified at `e1559a07` without mutation.

### Compile versus steady state

The harvested logs contain the persistent-cache-OFF announcement but no compilation-count/compile-seconds receipt. No compilation count, warm per-unit time, or compile-adjusted wall can be inferred from them. In particular the lack of a compile-summary receipt is not zero compilations. MoS2 static logs report one screening catalog entry with **11 nodes** on both arms. The MoS2 Gamma cubature receipt reports orders (16,24,32), with (1536,3456,6144) cubature nodes; these are not screening tau nodes. Dynamic logs report individual generated rules (Si 6-node and MoS2 8-node probe rules), which do not establish the full Sigma schedule count. Total dynamic schedule and weight identity remain unmeasured.

## Valid rank-0 device captures

None collected in this lane: allocation failed before a step could launch. PERF2's working recipe is explicit `cudaProfilerStart/Stop`, effects barrier before stop, and Nsight `--capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --sample=none --cpuctxsw=none -t cuda,nvtx,osrt`. The historical binary path is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys`; it is not yet verified in a new runtime step. Capture rank zero and use native projected-span and kernel-duration CSV separately. A trace of one unit must include that unit's real kernel records before it is accepted.

### HLO collective census per stage

Pending real optimized dumps. `tools/hlo/analyze_hlo_dump.py` and PERF2 `tools/profile_collective_census.py` are available in the sandbox/reference worktree. Source loop counts below are **not HLO instruction counts or executed GPU launch counts**. XLA can hoist loop-invariant unfolds and eliminate exact-zero Lorentz terms; HLO and native records must decide what survives. FFI communication is a separate census.

## Owner's boundary-accounting table

Source locations below refer to P=`9f569c4b`; F references were read only at `e1559a07`. Let K=nk, Qk=n_parent, Qq=q-IBZ count, N=nb, S=ns, M_C/M_T=packed family extents, R=tile width, B=band chunks, H=screening tau nodes, and P=rank count. Every measured-ms and total-seconds boundary entry is pending; no stage wall is assigned to a component by subtraction.

| boundary | F | P source count/scope | scaling; measured ms / total s |
|---|---|---|---|
| ζ parent band projectors | full-k route | `_z_q_face_parent._projectors`: two band GEMMs per band chunk; shared by all three current channels | O(Qk N S² M_T R/P); pending |
| ζ child transport | different full-k construction | `_tile_tail.channel_tail` streams S² output spin pairs and two endpoints; `unfold_block` scans max_sources for each; current channel scan repeats this tail three times | source upper loop count 2·3·S²·max_sources operator-block unfolds/current tile, 2·S²·max_sources/charge tile; O(K M R/P) per scalar spin-source transport; pending |
| ζ FFT tail | full-k tail exists | two IFFTs per output spin pair, one final FFT per channel | source 2·3·S² IFFTs + 3 FFTs/current tile; charge 2·S² + 1; same K,M,R factors and transform cost; pending |
| coupled stacking | full-k coupled owner | three channel outputs from a scan; shared parent projectors; optional flattening of (3,Qq) into solve batch | O(3 Qq M_T R/P) result carrier; pending |
| C_q build/solve | canonical extent/full-k input | `_c_q_face_parent`: two parent projectors, two typed open-spin operator unfolds, gamma contraction; solve at packed extent | projector O(Qk N S² M²/P); dense solve extent M, not logical-prefix extent; pending |
| ζ final file seam | canonical file staging | `fit_zeta_to_h5:1939` calls unpack once per output file, after complete G-flat accumulation, not per real tile | O(Qq M N_G/P) per file; 1 charge + 3 current files when all fitted; pending |
| centroid seam compilation | full-k carrier | `_axis_kernel` and `_operator_kernel` already cached by `PackedCentroidBasis` | PERF2 cached-seam fix is already present; no duplicate implementation proposed; pending |
| χ face unfold | F consumes existing full-k faces | paired-plan `_build_Gv_Gc` invokes left/right `unfold_face` before both G builds; one pair in each block's tau body | source 32·H endpoint invocations over sixteen blocks, possibly hoisted by XLA; O(K N M S²/P) spin action and O(K N M S/P) face traffic; pending |
| χ G and orientations | F also has two orientations | two Green builds and two FFTs/node; reverse uses transposed views of those SAME Green tensors | 32·H Green builds over sixteen blocks; not 64·H; O(K N S² M_left M_right/P); pending |
| χ kernel factory keys | vertex pair participates in F too | `_get_chi_minimax_kernel:184` includes both plan identities, both shapes and vertex pair | sixteen Python keys for sixteen distinct pairs at fixed geometry; orientations do not double this to 32; actual compiled-module counts pending |
| packed Dyson | full q in F | full_static uses q-IBZ and one packed solve; packed bare builds no current χ and no packed Dyson | Qq versus K dense matrices of extent M_C+3M_T; pending |
| full-q restore + Lorentz mix | F has already-full-q packed operator | `photon_blocks_full_q` loops source blocks for each requested output: CC 1, CT/TC 3, TT 9 | 100 source additions/full 16-output pass, 99 for current-only 15-output pass, 81 for 9 TT outputs; O(K M_left M_right/P) per surviving restore; pending |
| restore JIT factories | no counterpart full-q restore | fresh zero JIT/output and fresh donating add JIT/source contribution, inside the generator | up to 16 zero factories + 100 add factories/full pass; three static X/SX/COH passes give 300 add factories; not a measured compile count; pending |

### Answers clarified by source inspection

1. **ζ_T does not apply a dense 4×4 spinor einsum to a whole child wavefunction face in its tile kernel.** It first contracts raw-parent projectors, then streams operator spin blocks using the service's `open_spin_block_coefficient` and `unfold_operator_local` (`isdf/core.py:3238–3318`). The same projector data is transported again across output pairs and the three vertex channels. An S=2 comparison therefore must measure this streamed operator schedule, not price it as the wavefunction `unfold_face` einsum. `max_sources` depends on support in the typed spin action, so do not substitute S² blindly.

2. **The dense per-child spin action is present in χ's wavefunction unfold.** `CentroidKUnfoldPlan.unfold_face` delegates to `symmetry_maps.unfold_wavefunction_local`; `maps.py:1121` applies `einsum('qac,q...c->q...a', ...)` with (K,4,4) on the gathered, phased face. Arithmetic grows as S² for fixed other extents; observed S=4 versus S=2 speed ratio is unmeasured. The whole face here is one endpoint with both band and centroid sharding, not a replicated global face.

3. **A pair of plans selects a materially different χ path.** At `w_isdf.py:502`, `gemm_plan.nq` is K when plans are paired. The two endpoints unfold before full-k Green GEMMs. A single plan instead contracts parent Greens and unfolds the operator through `build_G_tau`. Comparing paired versus single plans is not just timing tuple lookup, and cross-family rectangles cannot be treated as same-family Hermitian operators without a derivation.

4. **No demonstrated plan-id cache miss yet.** `_fit_one_rchunk_cache` includes `id(k_unfold_plan)`; the parent Z cache also includes plan/store identity and static tile extent, while tile permutation/wrap tables are runtime operands. The coupled coordinator calls `_z_q_face_parent(coupled_mu123=True)` directly and may supply prebuilt solved ζ, bypassing the ordinary fit kernel. It is wrong to assume three ordinary `_fit_one_rchunk` compilations simply because three current output files exist. Actual plan reuse and trace counts require receipts.

5. **Restore ownership for lane S:** the only production call to `photon_blocks_full_q` is `photon_sigma.contract_lorentz_blocks:163`. `compute_photon_sigma` consumes it for X, SX, COH; `sigma_x_bispinor` also uses the shared consumer for bare TT exchange. Charge/current centroid transport and Lorentz mixing are owned in `w_isdf.py`, but their execution is on the Sigma side. Charge them once to Sigma, not again to screening. This report is the coordination artifact; no Sigma source was edited.

6. **Gamma is a separate screening operation.** `compute_static_photon_response` applies `complete_static_slab_photon_q0` after q-IBZ body construction and sanity checks. Its cubature/body timings must be split from the sixteen χ blocks and distributed Dyson. The bare mode skips the latter two but still has packed-body assembly and Gamma completion.

## Architectural proposals

**Preregistered ranking, not a measured selection.** Compute access still prevents ranking by measured ms/complexity. The following ranks use source-count ceilings only; every device-ms entry is unmeasured. A speedup bound on one component is not a bound on the entire stage. In particular, a compiler-hoisted or zero-eliminated operation has no recoverable cost. No candidate is accepted for production without the required baseline and ablation.

For concrete byte bounds use the harvested MoS2 gate: K=9, Qk=3, N=80, S=4, P=4, logical charge/current extents 192/98. The centroid text header authenticates 98 current points; packed extents M_C/M_T must still be read from a new driver receipt. Thus the byte examples below are **logical-extent lower bounds, not measured allocations or asserted packed extents**. complex128 costs 16 bytes. Both band and centroid dimensions of a face are distributed, so one face costs F(M)=16 K S N M/P bytes/rank. The paired family endpoints have a payload lower bound 2 F(192+98)=6,681,600 bytes/rank; their parent counterparts cost 2,227,200 bytes/rank. Packed padding increases both linearly. Array payload sums do not include aliases, compiler temporaries, or allocator arenas.

| Priority / candidate | Boundary and numerical ceiling | Decision before measurement |
|---|---|---|
| 1. Reuse compiled restore additions; stream output blocks | Up to 100 addition factories per full pass, 300 across X/SX/COH; repeated fixed signatures can in principle reuse executables. Device unfold work unchanged. | First low-complexity ablation; cannot yet give compile seconds saved. |
| 2. Unfold family faces once per term | 32 endpoint invocations per block pass become 4 distinct endpoint faces (two families, two orientations): at most 8x fewer unfolds if already hoisted out of H nodes, at most 8H otherwise. | Measure hoisting first; resident full-child lifetime changes the carrier contract. |
| 3. Two 2x2 spin-action blocks | Dense 4x4 has 16 complex products/output-vector action versus 8 for two 2x2 blocks: at most 2x arithmetic saving for that action alone; identical output traffic. | Must live at the symmetry service owner and authenticate block support; no local det/sign formula. |
| 4. Batched Lorentz contractions | Family pairs have 1,3,3,9 output blocks. Reusing their G/FFT work can reduce 16 pairs of Green builds to 4 pairs/node, at most 4x for that component. | Needs rectangular family shape classes and memory gates; not a sixteen-block resident accumulator by default. |
| 5. Coupled current ζ tail | Parent projectors and C_q are already shared. Three channel tail traversals remain; at most 3x saving on shareable tail transport/FFT work, zero further projector/C_q saving. | Requires a streamed contraction design retaining the same reduction order. |
| 6. Fuse file seams with neighbouring work | Final ζ unpack occurs four times/all families, not once per tile. Existing compiled seam cache already removes eager-module proliferation. | Defer until a residual seam or reshard is measured; no quantified remaining benefit. |
| 7. Change four-spinor carrier layout | A transpose may be a bitcast or a materialized copy; the latter costs at least one read plus one write of the face payload. | Defer until optimized HLO proves a copy; source moveaxis alone is not evidence. |

**1 — Restore executable reuse and streaming.** Keep one full-q output accumulator and the existing source-block ordering. Reuse the compiled addition at the existing response/plan owner across consumers, passing changing data as operands; the symmetry service still owns centroid transport and Lorentz mixing. Delete the fresh JIT factory inside each contribution loop. Before/after explicit output payload stays D(A,B)=16 K M_A M_B/P bytes/rank (TT lower bound 345,744 bytes); parent block payload stays 16 Qq M_A M_B/P. This targets compilation, not repeat device transport. Gate all CC/CT/TT rows and EQP at zero tolerance, plus optimized HLO for the same output lifetime and collective shapes. Caching all full-q blocks is a different proposal: one full response needs 16 K (M_C+3M_T)^2/P bytes/rank, lower bound 8,503,056 bytes, versus one largest output lower bound 1,327,104 bytes. Retaining both V and W doubles the former; reject that uncontrolled lifetime increase until a scaling-safe consumer stream is designed. Any consumer-order change belongs with lane S and is communicated through this report.

**2 — Family face reuse.** At the term boundary call each family's existing typed unfold for each endpoint orientation once, then feed full children to the four family-pair contractions. Keep raw parents for other stages; delete repeated paired unfolds from the block kernels for this route. Before: parent family payload lower bound 2,227,200 bytes plus temporary child endpoints needed by the active block (largest logical CC pair 4,423,680 bytes). After: parents plus persistent children lower bound 8,908,800 bytes/rank. This is a lifetime bound, not necessarily a peak increase: XLA may already retain children through a tau scan. Scaling is O((Qk+K) S N (M_C+M_T)/P), independent of block count; the removed work scales with block count and possibly H. Use the symmetry service unchanged and no new in-kernel collectives. Gate child values, per-block chi, printed EQP/sectors, and the larger-deck/P16 HLO memory envelope before adoption.

**3 — Block-structured spin action.** Let the service derive two diagonal spin subblocks from its authenticated four-spinor action; apply them at its existing wavefunction-unfold owner after the same gather, phase and TRS operation. Delete the dense action on this supported shape through a general structured-action implementation, without an `if bispinor` branch or another rotation rule. Before/after output face payload is F(M); action table decreases from 16 K to 8 K complex entries if a compact representation is adopted (2,304 to 1,152 bytes on this gate). Scratch and transpose costs remain unknown. A reordered summation can change printed digits even for exact block zeros; require arbitrary complex broken-TR and spatial+TR action tests, direct action parity, and the complete science gate. Folding the action into a gather is not automatically free: coefficients still require products/sums.

**4 — Batched Lorentz contraction.** Compute each family-pair Green and FFT pair once per tau, then stream that pair's 1/3/3/9 canonical gamma contractions through a block axis. Delete the independent repeated G/FFT block kernels only after the new owner handles all four shape classes. Each G costs 16 K S² M_A M_B/P bytes/rank; a CC G is at least 21,233,664 bytes here, a TT G 5,531,904 bytes. A pair is twice that. Keeping one family pair and streaming its outputs avoids multiplying this by sixteen; keeping nine TT accumulators instead of one adds at least 2,765,952 bytes/rank. The output packed response already exists but its aliasing cannot be assumed. Scaling remains O(K S² M_A M_B/P), plus explicitly bounded block accumulators; no square tensor on a subset of ranks. Gate both orientations, unequal family extents, HLO live buffers/collectives, and final EQP/sectors. Compilation can still specialize on shape and rule length; four family classes do not prove exactly four compiled modules.

**5 — Coupled ζ tail.** Keep shared parent projectors and C_q, but stream one unfolded output spin pair through contributions to all three current-channel accumulators before moving to the next spin pair. Delete the outer channel repetition of identical transports where the typed vertex algebra permits it; retain the service's spin coefficients and permutation owner. Before/after final three-channel Z payload is 3·16 Qq M_T R/P bytes/rank, with a potential extra concurrent two channel accumulators if the incumbent schedule streamed them. Actual tile R and packed M_T are absent from the checkpoint receipts, so no numerical byte value can be authenticated yet. FFT reuse and reduction order need derivation; current channels have different vertex permutations/phases. Gate saved ζ, downstream EQP/sectors, and per-tile HLO memory. No second coupled-projector implementation is proposed.

**6 — Seam fusion.** Feed the packed output of the preceding reshard directly into the existing canonical writer conversion, preserving canonical, P-agnostic files and the existing padding receipt. Delete an intervening materialization only if the trace shows it. ζ final payload is 16 Qq M N_G/P bytes/rank per file; full operator payload is D(A,B). Before/after memory can only be bounded symbolically until N_G, packed extents and aliasing are measured. Gate byte-identical files, cross-P readback and optimized HLO. Do not implement another pack/unpack beside PackedCentroidBasis.

**7 — Carrier layout.** Select one endpoint layout accepted by the existing GEMM and FFT owners, and move unavoidable conversions to the producer. Delete repeated materialized transposes; preserve typed axes and raw-parent file layout. Payload stays F(M); eliminating a copy can save up to one additional face's live storage per endpoint, but a bitcast saves zero bytes and zero traffic. Establish the actual copy census first. Gate spin/vertex action, all family rectangles, HLO memory and printed science identity. This has the broadest consumer impact and weakest present evidence.

All proposals retain canonical files and single-owner symmetry rules, and require the no-replicated-square invariant and a parent-kernel collective census. None claims a tiny-deck limitation or a larger-deck speedup: there has been no larger-deck or P16 performance measurement.

## Ablations

No ablation executed or source candidate selected. Prioritized experiments after pool access:

- Profile paired χ once cold and repeated warm, then inspect whether its face unfolds are hoisted out of tau scan. Compare full-static F/P at identical nodes and shapes before proposing family reuse. Retaining full child carriers requires explicit memory/lifetime accounting; it cannot silently replace the parent-carrier contract.
- Capture current ζ with more than one tile in a new variant that actually fits ζ. Count streamed spin-source transports, FFTs, channel reuse and compiled signatures. Do not change contraction/reduction order without the identity gate.
- Capture a single restored TT output and a full restore pass, preserving the one-output-block lifetime. Measure zero-coefficient source elimination and fresh-JIT cost before designing a cache at an existing owner. Caching all nine full-q TT blocks would enlarge the live set and is not an accepted shortcut.
- Count family seam calls at loading and write boundaries; cached seam kernels already exist, so repeat PERF2's hypothesis only if a different owner bypasses them.

## Final verification and disposition

**Incomplete, blocked by compute access. No fix accepted, no tests run, no Nsight/HLO evidence, no printed-digit gate, no speedup claim.** Production code is unchanged because the dispatch requires measured baselines and ablations before edits. Outstanding: pinned P/F baseline legs for all five decks, actual schedule matching, ζ fit variants, cold/warm split, native unit profiles, HLO census, justified ablations, selected source fixes, identity gates, and final P4 combined verification.

## Environment receipts

Evidence directory: `runs/Si/100_bisp_parent_route_2026-09-05/prof_zw/00_environment/` in the sandbox. The one requested allocation attempt, named `lx-alloc-jackm-BISP-PROF-ZW`, failed with `QOSMaxSubmitJobPerUserLimit`; see `allocation.log`. It produced no job id and no science step. `scontrol show job 57955075` returns `Invalid job id specified`; see `fallback_pool.log`. No other lane's allocation was used, no second allocation was attempted, and no sbatch was submitted. An authorized live JID or authorization to retry allocation is required to continue GPU work. This is a scheduler refusal, not an automatic approval-review rejection.

Historical home-side `BUILD_NOTES.md` and `lorrax_bse_perf_2026-08-08` corpus paths do not exist in this environment. The current sandbox invariant requires JAX/JAXLIB 0.9.x and `lorrax_A`, superseding the preamble's historical J070 suggestion. Missing home references did not cause a pause; the allocation refusal did. `RUNS_INFLIGHT.md` and issue ledgers record this lane's state.

Ledger: claim **981**, `claims/0981.md`; lane marked blocked before GPU launch.

## Resumed dispatch, 2026-09-06

The renewed dispatch authorized one new allocation attempt. At 07:45–07:50 UTC the named BISP-PROF-ZW allocation again failed with `QOSMaxSubmitJobPerUserLimit`; fallback57955075 remained invalid. Other visible jobs belong to FFREAL2, FFREAL3, BISP-orch and PSIIRR-landing, so none was used. A permitted JID has been requested. This is a scheduler/pool-ownership block, not an approval-tool rejection.

Prepared two new **unrun** full-static arms inside this worktree at `runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/{02_parent_static,03_fixed_static}/`. Both copy the exact same parent73 deck and entire tmp; SHA256 receipts match. The deck retains restart=false and existing ζ reuse, so these arms measure screening, not fresh ζ fitting. Source-tree hashes refuse drift from P9f569c4b/Fe1559a07 while allowing documentary commits; actual HEAD is printed in each rank. BFC@0.85, cache-off, P4, source wrappers, manifests, artifact checks and a single pre-launch-expiry retry are prepared. Shell syntax passed; no runtime or science validation is claimed. `01_prepare/prepare.py` refuses existing arm directories, so rerunning it cannot mutate a completed run. Preparation writes only this lane's worktree.

Architecture candidates are now explicit above, with source-count ceilings and byte lower bounds distinguished from missing measurements. Pinned measurements, native captures, ablations, fixes and science gates remain outstanding.

## Live measurement checkpoint — pool57966610 authorized


Branch perf/bisp-prof-zw-2026-09-06, unmerged; production src/services unchanged from9f569c4b.
P4, BFC@0.85, JID57966610, lx-Xg4-012024-1218242-6951 exit0,209s launch.
Artifact: runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/06_parent_static/driver.rank0.log.
Screening31.21s; Sigma150.16s; total201.36s. Rank0 compiler receipt1712 compiles/148.01s.
EQP1 90/90 printed digits equal parent73, tools/eqp_ab.py tolerance0.
Not yet a P/F comparison; cumulative compiler work is not a wall partition.

### P/F baseline, claim998


P4 BFC@0.85; sequential JID57966610 steps P lx-Xg4-012024-1218242-6951 exit0, F lx-Xg4-012412-1251362-6346 exit0.
On branch perf/bisp-prof-zw-2026-09-06 unmerged, production source9f569c4b, versus fixed-main e1559a07 read-only.
Artifacts runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/{06_parent_static,07_fixed_static}/driver.rank0.log.

| quantity | P | F |
|---|---:|---:|
| screening s |31.21|12.90|
| Sigma s |150.16|18.12|
| total s |201.36|51.24|
| rank0 XLA compiles |1712|597|
| rank0 compiler work s |148.01|27.65|

Same copied tmp and dipole; zeta reused, no new fit measured. EQP1 via tools/eqp_ab.py:82/90 exact, max0.001micro-eV; tolerance0 comparison FAIL. Cross-source comparison is not the fix identity gate. No attribution to individual kernels until unit receipts.

### Unit attribution before changes — MoS2 full static

P04 lx-Xg4-012535-1263108-7463 exit0 and F05 lx-Xg4-013223-1306959-6243 exit0, JID57966610, sequential P4 BFC@0.85. `prof_zw/{04_parent_units,05_fixed_units}/` contains unit_timings.rank*.json, native Nsight CSV, nsys_rank0.nsys-rep, optimized HLO and analyzer/census JSON. These runs repeat each chi block once, return the first result, and require exact equality with the repeat. Nsight captures only the warm (1,1) block. Full-run walls with HLO/Nsight are **not** baseline performance walls.

| measured quantity | P | F |
|---|---:|---:|
| sixteen cold chi public calls, s |31.823|8.855|
| compile count inside those calls |144|43|
| compiler work inside those calls, s |27.473|7.005|
| sixteen warm chi public calls, s |0.411|0.419|
| warm new compilations |0|0|
| warm TT compiled module, ms |22.971|23.792|
| warm TT tau-body median, ms |2.069|2.120|
| two Green GEMMs/block projected sum, ms |21.275|22.105|

Both TT blocks have11 identical nodes/weights, digest7cb9e8744c3d6d41cd322d2bcbdd1ed1d79214408cde742b97c9fd272602ce3a. P parent current face shape=(3,4,100,80); full K=9. P's16 blocks reuse the same charge/current plan identities, and every warm public call adds zero compiles. HLO nevertheless contains16 distinct vertex integrator modules, not32 for orientations: P0464…1169. TT P0686 has20,736,109-byte per-rank compiler peak and zero explicit HLO collectives. Its actual body contains the family gather/phase and dense four-spinor GEMM inside an11-iteration while; XLA did not hoist them. Both endpoint spin actions fuse into one batched GEMM,11 launches/block. The printed P/F cold regression is real; **a warm chi slowdown is not seen in this capture**.

Native rank0 CUDA kernel-duration sums for the TT block: face gather/phase/concatenate0.045024ms (11 kernels), spin GEMM0.145472ms (11 kernels), Green GEMM arithmetic2.417981ms (396 kernels), NCCL Broadcast7.935001ms (792 kernels), FFT kernels0.283360ms (46 kernels), FFT scales0.136064ms (23 kernels), Green-layout transposes0.296128ms (22 kernels). HLO labels and native NVTX scopes own these classes; sums are not additive with their containing projected ranges. Both Green GEMMs use the full9 k rows in P and F; the parent route does not reduce that contraction here.

Restore host attribution in P04: V16 outputs32.567s/120 compiles/28.677 compiler seconds; W16 outputs32.735s/116 compiles/28.748 compiler seconds; W-V16 outputs32.885s/116 compiles/29.050 compiler seconds. The fresh zero and addition factories recompile through all three consumers. These98.187s are measured in the instrumented run; do not subtract them from unprofiled P06. Restores are charged to Sigma, not screening.

Architectural ranking update: restore executable reuse and fewer chi shape specializations target measured tens of compiler seconds. On this TT gate, even eliminating all repeated face gather and rotation saves only0.190496ms of kernel time/block (0.83% of its22.971ms projected span); two2x2 spin actions can save at most half the0.145472ms spin arithmetic, before new launch/layout costs. Full-family residency and a new spin-action representation therefore rank below compilation fixes. This is a measured bound on this deck, not a claim of a large-deck/P16 speedup. Green/FFT reuse across Lorentz blocks remains a worthwhile structural candidate because the containing GEMMs dominate the device unit.

### Restore executable reuse ablation, claim1004


On branch perf/bisp-prof-zw-2026-09-06, unmerged; production unchanged9f569c4b.
Run-local ablation08: JID57966610, lx-Xg4-013500-1326885-7821 exit0, P4 BFC@0.85.
Artifacts runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/08_restore_ablation/{unit_timings.rank0.json,eqp0_ab.txt,eqp1_ab.txt,sectors.csv}.
EQP0/1 each90/90 printed-digit identity and all sigCC/sigTT/sigCT rows exact versus P06.
V/W/W-V restore passes:120/16/16 compiles,20.143760/0.773553/0.833868s; zero factories remain per output.
Whole driver161.02s, final1520 XLA compiles/116.71s compiler work. BaselineP06:201.36s,1712/148.01s.
Ablation adds warm chi repeats, so uninstrumented after-source repetition remains owed. No source implementation accepted yet.
