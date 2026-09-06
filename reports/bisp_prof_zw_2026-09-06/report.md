# BISP-PROF-ZW — measured compilation and coupled-current fixes

Work in progress on branch `perf/bisp-prof-zw-2026-09-06`, **unmerged**. Pinned before-change P=`9f569c4bf75bad40e4f5895946874b4c503e4410`; fixed-main F=`e1559a071e244b4f049c924781b668d9e1560739`. Adoptable pushed fixes so far: restore executable reuse `87a2bfaa`, chi vertex shape classes `0f8fbc3e`. MoS2 full-static screening31.21→18.59s; total201.36→146.56s; EQP0/1 and every CC/TT/CT row exact. Coupled-current tail sharing now passes its production fresh-fit and downstream identity gate; its warm tile is39.31ms versus51.00ms before.

All new science uses authorized campaign pool **57966610**, `--wait 1800`, one P4 node at a time, one rank/GPU, BFC@0.85, persistent compilation cache OFF and `LORRAX_DEBUG_PRINT=1`. F and P are sequential. No Sigma source was edited. Restore execution is charged to the Sigma consumer, not screening. Paths below are relative to the sandbox unless prefixed `reports/`; M=`runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/`, S=`runs/Si/100_bisp_parent_route_2026-09-05/prof_zw/`, D=`runs/DEV/112_bisp_prof_zw_codex_2026-09-06/`.

## Objective and preregistered candidates

Reproduce PSIIRR-PERF2's ordering: pinned stage baselines; compiler receipts and first/warm units; real rank0 Nsight kernels; optimized HLO/collective census; boundary accounting; isolated ablations; shortest maintainable source changes with strict identity gates. The initial source audit and historical checkpoint table are retained in `preregistration_and_checkpoints.md`, not relabeled as pinned measurements.

Preregistered boundaries: parent-face I/O seams, child gathers/phases/spin action, current vertex permutations, coupled channel stacking, parent-plan identity keys, sixteen chi blocks, packed q-IBZ Dyson, Gamma completion, and per-consumer full-q restores. All new runs copy decks and private tmp/dipole inputs. The ordinary deck arms reuse zeta; auxiliary fresh-fit arms request the existing r_chunk_size4096 key. No new deck key, environment dial, vendor implementation or symmetry rule was introduced.

## Before any source change

Unprofiled stage walls, seconds P/F. Screening=chi0+W+screening support; Sigma=rule plan+tau sweep+other. W persist/head is separate. Missing fresh-zeta stages are not zero-cost fits.

| deck | arms | screening P/F s | Sigma P/F s | total P/F s |
|---|---|---:|---:|---:|
| Si GN-PPM | 13_parent_gn/14_fixed_gn | 7.99/6.79 | 63.73/33.06 | 90.78/58.51 |
| Si COHSEX | 15_parent_cohsex/16_fixed_cohsex | 6.39/5.75 | 48.33/11.65 | 72.97/39.54 |
| MoS2 full static | 06_parent_static/07_fixed_static | 31.21/12.90 | 150.16/18.12 | 201.36/51.24 |
| MoS2 packed bare | 17_parent_bare/18_fixed_bare | 16.67/11.64 | 150.60/18.61 | 184.49/48.20 |
| MoS2 dynamic eps1e-5 | 19_parent_dynamic/20_fixed_dynamic | 32.80/16.58 | 161.92/78.70 | 217.54/116.34 |

| deck | P/F rank0 compilation count | P/F compiler work s | schedule receipt |
|---|---:|---:|---|
| Si GN-PPM |816/539|59.13/28.28|13 screening nodes;29 Sigma node dispatches,6 windows; different selected Sigma digests|
| Si COHSEX |676/412|48.44/18.63|13 screening nodes; no dynamic Sigma schedule|
| MoS2 full static |1712/597|148.01/27.65|11 screening nodes, identical tau/weight digest in unit arms04/05|
| MoS2 packed bare |1613/598|137.11/27.07|11 charge-screening nodes; no current chi or packed Dyson|
| MoS2 dynamic |1896/790|161.08/39.99|11 screening nodes;67 Sigma node dispatches,8 windows; two different selected Sigma digests|

Baseline06 step `lx-Xg4-012024-1218242-6951` exit0;07 `lx-Xg4-012412-1251362-6346` exit0 (claims996/998). Eight arms13–20: `lx-Xg4-015056-1447137-8575` exit0 (claim1030), `D/remaining.lx.log` and `D/baseline_analysis/host.json`. The latter reuses PERF2's existing census parser through analysis symlinks. Each arm has source guards, input SHA256 receipts, driver.rank*.log and artifacts.

**Dynamic whole-run matching limitation:** copying the same rule directory did not force identical rule choices. Si P uses29 dispatches/24 branch-distinct nodes; F has several cache misses and different digests. MoS2 has67 dispatches but F regenerated two rules;49.18s of its Sigma wall is planning. These are observed controls, not a matched Sigma speedup comparison. Guarded PERF2 single-rule replays32/33 are prepared; their original containment/error/noise guards are retained. Matched chi and same-source fix gates do not borrow a relaxed quadrature comparison.

Across P/F06/07, EQP1 is82/90 printed-digit exact, maximum0.001 micro-eV. Cross-source numerical differences are not the acceptance threshold. Every retained source change instead compares to P at tolerance0.

### Compile versus steady state

The compile-cache receipts measure cumulative compiler work, not an additive partition of elapsed wall. HLO/Nsight runs have substantial extra compilation overhead and are excluded from the stage speedup table.

| synchronized public chi unit, all16 blocks | P04 | F05 | shape-class ablation11 |
|---|---:|---:|---:|
| sum of first calls, s |31.822985|8.854682|6.336320|
| new compiles in first calls |144|43|48|
| compiler work in first calls, s |27.473072|7.005439|3.832625|
| sum of repeat warm calls, s |0.410999|0.419334|0.390107|
| repeat-call new compiles |0|0|0|

P and F each originally compile16 vertex integrator modules, **not32** for two orientations. The reverse orientation reuses the two Green tensors. Charge/current plan identities stay stable across all16 calls; there is no observed within-run plan-id churn. P's paired plans select a different algorithm: unfold endpoint faces and contract full K=9 Green tensors. The single-plan route instead transports parent Green operators; it is not just cheaper tuple handling.

## Valid rank-0 device captures

P04 `lx-Xg4-012535-1263108-7463`, F05 `lx-Xg4-013223-1306959-6243`, exit0. Both use11 tau nodes and identical node+weight SHA256 `7cb9e8744c3d6d41cd322d2bcbdd1ed1d79214408cde742b97c9fd272602ce3a`. Native capture brackets one **warm TT(1,1) block** with cudaProfilerStart/Stop and an effects barrier. Binary verified in runtime: `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys`. Flags: `--capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --sample=none --cpuctxsw=none -t cuda,nvtx,osrt`. Real kernel records and native CSV tables exist in each run; these are not empty wrapper traces.

| TT unit | P | F | currency |
|---|---:|---:|---|
| complete module |22.970702ms|23.792180ms|GPU projected span|
| tau-body median |2.068830ms|2.119807ms|11 projected instances|
| two containing Green GEMM ranges/block |21.275342ms|22.105268ms|sum of projected ranges|
| P face gather/phase/concatenate |0.045024ms,11 launches|absent parent unfold|CUDA kernel-duration sum|
| P dense four-spinor GEMM |0.145472ms,11 launches|absent parent unfold|CUDA kernel-duration sum|
| P Green GEMM arithmetic |2.417981ms,396 launches|see native CSV|CUDA kernel-duration sum|
| P NCCL Broadcast |7.935001ms,792 launches|see native CSV|CUDA kernel-duration sum|
| P FFT arithmetic / scales |0.283360/0.136064ms,46/23 launches|see native CSV|CUDA kernel-duration sum|
| P Green-layout transposes |0.296128ms,22 launches|see native CSV|CUDA kernel-duration sum|

Do not add kernel sums to containing projected ranges. The parent face shape is(3,4,100,80); gathered children have K=9. Optimized HLO keeps the spin action **inside** the11-node while. It fuses the two endpoints into one GEMM over c128[9,4000,4] and a[9,4,4] action. Thus repeated unfold exists, but its measured gather+rotation cost is only0.190496ms/block,0.83% of the containing span. P has no measured warm chi slowdown on this gate. Both P and F Green contractions still use all9 k rows.

### Fresh charge and current zeta tiles

P09 `lx-Xg4-013859-1352582-8047`, F10 `lx-Xg4-014103-1371100-7581`, sharing ablation12 `lx-Xg4-014750-1427817-6542`, exit0. These intentionally stop after zeta. All request4096 points; P makes12 whole-orbit tiles with3844 carrier slots each (46080 real points), F12 contiguous tiles with a4096 first shape and1024 remainder. Per-call work differs.

| unit | P09 | F10 | shared-left12 |
|---|---:|---:|---:|
| first charge fit / warm median, ms |3003/35.210|2188/29.672|same algorithm|
| charge first-call compiles over all tiles |17|23|see receipt|
| first current Z / warm median, ms |1555/50.215|1241/33.194|warm39.001|
| current Z compiles over all tiles |2|6|2|
| new compiles in repeat calls |0|0|0|
| warm current Z device span, ms |51.001603|30.535343|39.313741|
| per-rank current-Z HLO peak, bytes |452208158|779081668|452172878|
| scalar spin-source transports/current tile |384|different full-k schedule|256|
| k FFT transforms/current tile |99|different full-k schedule|67|

Zeta does **not** rotate a whole child wavefunction by4×4 in the tile. It contracts open-spin parent projectors, then streams typed operator transports. Before:2 endpoints×3 channels×16 spin pairs×4 sources=384 transports,96 inverse FFTs+3 final FFTs. After: share the left transported/IFFT face across three channels, retaining each channel's reduction order;256 transports and64 inverse+3 final FFTs. Parent projectors and C_q were already shared; no additional saving is claimed there.

Device span improves22.9%, warm host median22.3%. Projector GEMM arithmetic stays1.3655ms (10 launches). Dominant transport kernel classes fall16.203+8.942+5.293→10.715+6.036+3.460ms; new channel updates cost3.534ms. Saved charge and all3 current HDF5 files are bit exact against09 under `compare_zeta_h5.py --rtol0` (charge1,130,688 values;577,122/current file; max_abs0, nonfinite0; claim1031, comparison executed in step015056). Production fresh-fit plus downstream gates24/25 PASS: step lx-Xg4-021818-1673870-3826 exit0, all four files bit exact, EQP0/1 each90/90 exact, allCC/TT/CT exact. Production warm tile39.309827ms, host median39.029ms.

### HLO collective census per stage

Each capture has optimized dumps and `tools/hlo/analyze_hlo_dump.py` output; PERF2 `profile_collective_census.py` supplements its omitted async all-reduce-start. Zero explicit HLO collectives does not mean zero FFI communication.

| stage / module | static optimized HLO collectives | execution frequency / per-rank peak |
|---|---|---|
| P chi TT0686 |0|11 tau nodes;20,736,109B|
| F chi TT0600 |0|11 tau nodes;25,136,736B|
| P zeta current0447 |1 all-to-all,1 all-gather-start,1 all-reduce-start|five band-loop executions each;452,208,158B|
| shared-left current0447 |same|no new tail collectives;452,172,878B|
| P packed Dyson1212 |0 explicit|q-IBZ3, packed492;11,619,136B|
| F packed Dyson0643 |0 explicit|full q9, packed492;34,857,408B|
| P full-q restore addition modules |0 explicit|300 modules identified by jit(add)/jit(_do_unfold); maximum1,804,081B|

Eighteen unrelated jit_add modules are excluded from the restore count. Static-source counts are not native launch counts: typed local restore paths remove explicit centroid collectives here; distributed GEMM/Dyson internals retain their existing service communication. Gamma-specific device timing and host seam receipts are being extended in after-unit arm26.

## Owner's boundary-accounting table

K=nk=9, Qk=n_parent=3, Qq=3, N=nb=80, S=ns=4, M_C=192, M_T=100 packed (98 logical), P=4, H=11, R=3844, B=5 band chunks. Values below are measured on the MoS2 gate or explicitly bounded; no stage wall is assigned to a component by subtracting unrelated runs.

| boundary | F / P presence and count | measured cost / whole-run accounting | scaling |
|---|---|---|---|
| parent family face load/seam | F full-k; P two raw-parent families and endpoint orientations | host seam receipts pending26; PackedCentroidBasis executable caches already exist |face payload16 Qk S N M/P; F replacesQk byK|
| zeta parent projectors | F full-k; P2 GEMMs/band chunk shared across channels |P current1.3655ms arithmetic/tile, unchanged|O(Qk N S² M R/P)|
| zeta operator transports | F different full-k tail; P384→256/current tile,4608→3072 per12-tile fit |dominant classes30.438→20.211ms/tile; total current device51.002→39.314ms|O(K M R/P) per supported scalar source; support owned by SymMaps|
| zeta FFT/channel accumulation | both; P99→67 transforms/tile |whole unit saves11.688ms/tile; cannot add again to transport saving|O(K logK M R/P) per transform,3 channel outputs|
| C_q and solve | both; P parent C_q at packed extent; one shared coupled setup |current debug solved-channel tail often3ms/solve; no isolated C_q speedup claimed|C_q O(Qk N S² M²/P), solve extentM not logical prefix|
| final zeta unpack/write | F canonical staging; P once/output file,4 total, not per real tile |P current HDF5 writes7.84–8.01ms/file in09; unpack cost not isolated yet|16 Qq M N_G/P, N_G=1963; no block/tau multiplier|
| chi endpoint unfold | F absent; P2 endpoints×16 blocks×H=352 source invocations |TT0.190496ms/block gather+spin kernel sum; ≤3.048ms/16 blocks if TT-sized, not a mixed-family measurement|O(K N M S²/P) spin, O(K N M S/P) traffic|
| chi Green/FFT pair | both fullK;2 Greens/node/block,352 total |P TT21.275ms/block projected containing range; F22.105|O(K N S² M_A M_B/P), H×blocks|
| vertex executables |F/P initially16 integrators; two orientations inside each |P cold144 compiles/27.473 compiler s; shape classes48/3.833 in ablation; repeats0|shape/rule-length/plan signatures;4 family classes after fix|
| packed Dyson |F9q;P3q,one solve/full-static response;bare skips|HLO peak34.857→11.619MB; warm timing pending26|O(Qq (M_C+3M_T)³/P), quadratic carriers onallP|
| Gamma completion |both, separate from chi/Dyson |cubature orders16/24/32 (1536/3456/6144 nodes); isolated host timer pending26|packed low-rank update and tiny channel algebra; no chi-H multiplier|
| restore and Lorentz mixing |F already fullq;P100 source contributions/16 outputs/pass,300 acrossV/W/W−V |profile04 V/W/W−V32.567/32.735/32.885s including compilation; restore-reuse08 warm later passes0.774/0.834s |O(K M_A M_B/P) per contribution, source multiplicity1/3/9|
| restore factories |P fresh addition/output factories;F no counterpart |120/116/116 compiles before;120/16/16 after; source21 removes200 run compilations |fixed source-signature classes, consumers multiply only before fix|

The restore timer is in the sole `photon_sigma.contract_lorentz_blocks` consumer. Lane S must not add these seconds a second time. A two-plan chi call and a single-plan call have different Green data flows, so no isolated “tuple overhead” number is valid. The ns4 action has twice the arithmetic of two ns2 blocks, but an actual ns2 scalar-route timing has not been measured; no empirical ns4/ns2 ratio is asserted.

## Architectural proposals

Ranked by measured/bounded gain per added complexity. All retained changes keep INVARIANTS6: quadratic objects on allP ranks; parent tails gain no collectives; canonical P-agnostic files; one symmetry owner. Byte figures are array payloads unless labeled HLO peak; allocator arenas/aliases are separate.

1. **Accept: reusable restore executables, stream outputs.** Removes fresh addition-function identity across consumers: source gate21 saves200 compiles and38.17 Sigma-side seconds. Keep parent packed V/W, one output accumulator, unchanged source order, typed unfold_isdf_operator and mix_lorentz_blocks; delete inner donating-JIT factory. Before/after one TT output360,000B/rank, largest CC1,327,104B, parent TT120,000B; O(K M_A M_B/P), no resident cache added. Gate21 EQP0/1 and sectors exact. Full data caching is deferred: one full packed-q response8,714,304B/rank, bothV/W17,428,608B versus one output; this grows quadratically. Reordering consumers to restore each source once requires lane S coordination and new lifetime/identity gates. Compiler reuse is implemented; data reuse is not claimed.

2. **Accept: chi family shape classes with runtime canonical vertices.** Removes96 run compilations and12.14 screening seconds beyond restore-only gate21. Data flow keeps each existing Green/orientation contraction; canonical gamma_perm_phase tables become replicated tiny runtime operands. Cache distinguishes identity/nonidentity at each endpoint, shapes, plans and rule signatures. Delete captured per-vertex constants from the factory. Main array sizes unchanged; four integer/complex vertex vectors are a few hundred bytes replicated/rank, independent ofM,N,K,P. TT HLO peak before20,736,109B; after pending26. Literal complex Lehmann oracle and exact gate22 retain Γ2 phases and rectangular families.

3. **Accept: coupled zeta tail.** Removes128 transports and32 transforms/tile, saves11.688ms measured warm unit (22.9%). Stream one left child projector/IFFT across the three channel updates, then advance the spin pair; right transports keep canonical vertex permutations and each channel's summation order. Delete the outer repeated channel tail. Three full-k Z outputs41,515,200B/rank atR3844; final q-IBZ solved outputs13,838,400B. Same O(K M_T R/P) scaling; HLO peak452,208,158→452,172,878B. No C_q/projector change or collectives. Bit-exact four-file ablation gate passes; fresh source gate24/25 passes all four files and downstream EQP/sectors exactly.

4. **Defer resident family faces on this evidence.** Removes352 endpoint unfolds/response, leaving4 endpoint faces per term. Upper TT-sized kernel saving3.048ms/response; exact mixed-family aggregate is not claimed. Keep raw parents, unfold both endpoint orientations once/family through existing typed owner, then supply child carriers to family-pair kernels. Before parent payload2,242,560B plus largest active child pair4,423,680B=6,666,240B/rank. After parents+four resident family endpoints8,970,240B. ScalingO((Qk+K) S N(M_C+M_T)/P), saving grows withH×blocks. Delete paired unfold from those kernels only after a new carrier contract and memory gate. For the observed submillisecond component, that change is not justified ahead of compilation fixes. Required future gate: child parity including antiunitary actions, mixed-family chi, HLO memory, exact EQP/sectors and larger-deck measurement.

5. **Defer batched Lorentz Green/FFT reuse; shape-class compilation already accepted separately.** Compute one Green/FFT pair per family pair and tau; stream that pair's1/3/3/9 vertex contractions, deleting repeated G/FFT construction. At most16→4 Green pairs, bounded75% of that component. On this gate the entire16-block warm public work is0.411s, so even its total elimination cannot remove the residual multisecond stage; a TT-sized Green ceiling is0.255s/response. One TT Green5,760,000B/rank, pair11,520,000B; CC pair42,467,328B. Nine TT accumulators add2,880,000B versus one; O(K S² M_A M_B/P), withH reuse and a fixed small Lorentz axis. Requires a new block-axis accumulation contract, rectangular classes, both orientations, HLO live-buffer/collective gate and exact science. Benefit may become significant for largerM,N,H; no unmeasured large-system claim is made.

6. **Defer two2×2 spin blocks / gather folding.** Dense action costs0.145472ms/TT block; half-arithmetic ceiling0.072736ms before launches/traffic (≤0.317% of measured block span). The symmetry service must authenticate and own compact subblocks; no local det/sign formula. Delete dense action only at that owner, preserving gather, lattice phase and antiunitary operations. Face payload unchanged (TT1,152,000B/rank/endpoint); action table2304→1152B onK9. ArithmeticO(K N M S²/P) becomes half for this authenticated support; traffic remainsO(K N M S/P). Gate arbitrary complex spatial+TR action, direct face identity and final printed science. Actual ns2 scalar timing remains unrun.

7. **Defer seam fusion and carrier-layout changes.** The final zeta unpack happens4 times, and the existing PackedCentroidBasis conversion factories are already cached. Current file write≈8ms is not all removable unpack time. Fuse only a demonstrated intermediate with its neighboring reshard/writer; delete that materialization and preserve canonical files, logical padding and cross-P readback. Payload16 Qq M N_G/P; current padded staging2,355,600B/rank, charge4,522,752B. New face layout payload stays16 K S N M/P; one TT face1,152,000B. Actual Green-layout transposes cost0.296128ms/TT block; layout changes have a broad consumer contract and cannot be justified by source moveaxis alone. Gate byte-identical files, typed axes, both families, memory/collectives and EQP/sectors. No second conversion owner.

These are decisions for the measured gate, not an assertion that a tiny-deck slowdown is inherent. No larger-deck or P16 measurement has been made, and no such speedup is claimed.

## Ablations and after-change gates

| concern | run / step under57966610 | exact gate | measured outcome |
|---|---|---|---|
| restore run-local ablation |08 / lx-Xg4-013500-1326885-7821|EQP0/1 and sectors exact|V/W/W−V compiles120/16/16; total161.02s (extra warm chi calls)|
| chi shape-class run-local ablation |11 / lx-Xg4-014352-1399430-3018|EQP0/1 and sectors exact|screening19.62s; warm16 blocks0.390s|
| zeta shared-left run-local ablation |12 / lx-Xg4-014750-1427817-6542|four HDF5 files bit exact|warm device51.002→39.314ms|
| restore source87a2bfaa |21 / lx-Xg4-020725-1591215-3890|EQP0/1 each90/90; allCC/TT/CT exact|screening30.73s, Sigma111.99s, total159.61s;1512 compiles/116.27s|
| restore+chi source0f8fbc3e |22 / lx-Xg4-021236-1630539-3473|same exact gate|screening18.59s, Sigma111.85s, total146.56s;1416 compiles/106.07s|
| CPU focused oracle gate |23 / lx-Xg0-021609-1656988-8320|23 tests passed,2 warnings|photon vertices, centroid parent unfold/basis, four emulated CPU devices|

All steps above exit0. Claims1004,1007,1031,1033,1039 own the corresponding verdicts. P4 source artifacts contain source.diff/source.sha256. Tests are scoped: CPU oracles do not replace the actual GPU EQP/sectors gate. Wider source gates and fresh-fit end-to-end checks remain in progress.

## Verification and disposition

Adopt pushed commits87a2bfaa and0f8fbc3e on this lane branch; the orchestrator integrates. They retain the parent route and physical arithmetic while removing measured compilation overhead. Warm chi is already comparable to F on the tested deck; current-zeta transport remains slower than F even after the measured tail sharing and still needs a wider scaling experiment before any inherent limitation is asserted. Sigma-side residual work belongs to lane S; only restore seconds are attributed here.

Before-change sandbox gate0 fails inherited rule allowlist findings and historical CLAIMS rows without evidence tokens (`D/gate0_before.log`); the AST suites passed. This is registered in KNOWN_SANDBOX_ERRORS and is not presented as a passed gate. Failed arm02 omitted implicit dipole.h5 and is preserved;06 replaced it. Earlier allocation refusals and the final pool authorization are in the archived preregistration; no unauthorized pool was used.

Pending before final disposition: after-unit26 census; remaining deck identity gates27–30; guarded F schedule controls32/33; final branch status, archive and ledger closure. Instruments are reviewable under this report's instruments directory and in each immutable run directory. No broad filesystem search, main push or external message was used.

### Production current-zeta gate24/25

Fresh source24 and reference25 ran sequentially in JID57966610 step lx-Xg4-021818-1673870-3826 exit0. Reference25 consumes the pristine09 fresh canonical files in the same downstream source;24 builds all four anew. Files, EQP0/1 and all sectors are exact, so the fit change is isolated from prior cached-deck zeta conditioning. Instruments/Nsight affect whole driver walls; only the captured warm tile is the performance claim.
