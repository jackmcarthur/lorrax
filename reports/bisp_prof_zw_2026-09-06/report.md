# BISP-PROF-ZW — heavy investigation, blocked before GPU profiling

On branch `perf/bisp-prof-zw-2026-09-06`, unmerged. Baseline P is exactly `9f569c4bf75bad40e4f5895946874b4c503e4410`; fixed-main F is exactly `e1559a071e244b4f049c924781b668d9e1560739`. Production source is unchanged. This is a source audit and checkpoint harvest, **not a completed performance investigation or a proposed fix**.

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
