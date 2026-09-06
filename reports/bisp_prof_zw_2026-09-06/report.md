# BISP-PROF-ZW — measured compilation and coupled-current fixes

**6x6 follow-up:** source commits `6effddb5` + `9e31a1a7` are pushed on `perf/bisp-prof-zw-green-reuse-2026-09-06`, unmerged, on orchestrator production pin71ae0bde. One Green/FFT pair per family class reduces P4 warm chi2.724→1.092s and cached screening25.98→24.67s; EQP0/1 and all sectors are printed exact. Fresh common current-fit transaction P/F17.332/16.760s; fresh files are finite but not identical. See [Chi Green reuse and fresh zeta on6x6](#chi-green-reuse-and-fresh-zeta-on-6x6) for the new evidence, memory and pool scope. The sections preceding that follow-up describe the earlier3x3/Si task and its historical commit pins.

Measured disposition on branch `perf/bisp-prof-zw-2026-09-06`, **unmerged**. Pinned before-change P=`9f569c4bf75bad40e4f5895946874b4c503e4410`; fixed-main F=`e1559a071e244b4f049c924781b668d9e1560739`. Adoptable pushed fixes: restore executable reuse `87a2bfaa`, chi vertex shape classes `0f8fbc3e`, coupled-current tail sharing `f811f734`. MoS2 full-static screening31.21→18.59s; total201.36→146.56s; EQP0/1 and every CC/TT/CT row exact. Coupled-current tail sharing now passes its production fresh-fit and downstream identity gate; its warm tile is39.31ms versus51.00ms before.

The initial investigation uses authorized campaign pool **57966610**, `--wait 1800`, one P4 node at a time, one rank/GPU, BFC@0.85, persistent compilation cache OFF and `LORRAX_DEBUG_PRINT=1`. F and P are sequential. No Sigma source was edited. Restore execution is charged to the Sigma consumer, not screening. Paths below are relative to the sandbox unless prefixed `reports/`; M=`runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/`, S=`runs/Si/100_bisp_parent_route_2026-09-05/prof_zw/`, D=`runs/DEV/112_bisp_prof_zw_codex_2026-09-06/`.

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

**Dynamic whole-run matching limitation:** copying the same rule directory did not force identical rule choices. Si P uses29 dispatches/24 branch-distinct nodes; F has several cache misses and different digests. MoS2 has67 dispatches but F regenerated two rules;49.18s of its Sigma wall is planning. These are observed controls, not a matched Sigma speedup comparison. Guarded replays32/33 refused certificates outside their exact stored boxes. The successful matched pinned controls38–41 below resolve this limitation using existing certificates accepted for both sources; no guard was weakened. Matched chi and same-source fix gates do not borrow a relaxed quadrature comparison.

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
| Green GEMM arithmetic |2.417981ms,396 launches|2.467295ms,396 launches|CUDA kernel-duration sum|
| NCCL Broadcast |7.935001ms,792 launches|7.935321ms,792 launches|CUDA kernel-duration sum|
| FFT arithmetic / scales |0.283360/0.136064ms,46/23 launches|0.291520/0.135648ms,46/23 launches|CUDA kernel-duration sum|
| Green-layout transposes |0.296128ms,22 launches|0.290112ms,22 launches|CUDA kernel-duration sum|

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

Zeta does **not** rotate a whole child wavefunction by4×4 in the tile. It contracts open-spin parent projectors, then streams typed operator transports. Before:2 endpoints×3 channels×16 spin pairs×4 sources=384 transports,96 inverse FFTs+3 final FFTs. After: share the left transported/IFFT face across three channels, retaining each channel's reduction order;256 transports and64 inverse+3 final FFTs. The Z-tile parent projectors were already shared. C_q is vertex-dependent and is built separately once per channel; no C_q saving is claimed.

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

Eighteen unrelated jit_add modules are excluded from the restore count. Static-source counts are not native launch counts: typed local restore paths remove explicit centroid collectives here; distributed GEMM/Dyson internals retain their existing service communication. After-unit26 confirms four chi shape-class modules and100 restore modules. Gamma host receipts are in26 and34/35; no isolated Gamma native-kernel speedup is claimed.

## Owner's boundary-accounting table

K=nk=9, Qk=n_parent=3, Qq=3, N=nb=80, S=ns=4, M_C=192, M_T=100 packed (98 logical), P=4, H=11, R=3844, B=5 band chunks. Values below are measured on the MoS2 gate or explicitly bounded; no stage wall is assigned to a component by subtracting unrelated runs.

| boundary | F / P presence and count | measured cost / whole-run accounting | scaling |
|---|---|---|---|
| parent family face load/seam | F full-k; P two raw-parent families and endpoint orientations | unprofiled P charge/current loads1.6/0.6s versus F3.2/0.9s; seam detail in26 |face payload16 Qk S N M/P; F replacesQk byK|
| zeta parent projectors | F full-k; P2 GEMMs/band chunk shared across channels |P current1.3655ms arithmetic/tile, unchanged|O(Qk N S² M R/P)|
| zeta operator transports | F different full-k tail; P384→256/current tile,4608→3072 per12-tile fit |dominant classes30.438→20.211ms/tile; total current device51.002→39.314ms|O(K M R/P) per supported scalar source; support owned by SymMaps|
| zeta FFT/channel accumulation | both; P99→67 transforms/tile |whole unit saves11.688ms/tile; cannot add again to transport saving|O(K logK M R/P) per transform,3 channel outputs|
| C_q and solve | both; P parent C_q at packed extent; one Gram setup per channel, reused across tiles |current solve warm P/F3.228/3.098ms; public C_q repeats still compile once/call (36/37,43/44)|C_q O(Qk N S² M²/P), solve extentM not logical prefix|
| final zeta unpack/write | F canonical staging; P once/output file,4 total, not per real tile |P current HDF5 writes7.84–8.01ms/file; final unpack3calls/1compile, warmed0.430ms/file (36)|16 Qq M N_G/P, N_G=1963; no block/tau multiplier|
| chi endpoint unfold | F absent; P2 endpoints×16 blocks×H=352 source invocations |TT0.190496ms/block gather+spin kernel sum; ≤3.048ms/16 blocks if TT-sized, not a mixed-family measurement|O(K N M S²/P) spin, O(K N M S/P) traffic|
| chi Green/FFT pair | both fullK;2 Greens/node/block,352 total |P TT21.275ms/block projected containing range; F22.105|O(K N S² M_A M_B/P), H×blocks|
| vertex executables |F/P initially16 integrators; two orientations inside each |P cold144 compiles/27.473 compiler s; shape classes48/3.833 in ablation; repeats0|shape/rule-length/plan signatures;4 family classes after fix|
| packed Dyson |F9q;P3q,one solve/full-static response;bare skips|native warm F/P72.686→23.859ms; HLO peak34.857→11.619MB (34/35)|O(Qq (M_C+3M_T)³/P), quadratic carriers onallP|
| Gamma completion |both, separate from chi/Dyson |cubature orders16/24/32 (1536/3456/6144 nodes); host P/F2.780/2.851s,21compiles each (34/35)|packed low-rank update and tiny channel algebra; no chi-H multiplier|
| restore and Lorentz mixing |F already fullq;P100 source contributions/16 outputs/pass,300 acrossV/W/W−V |profile04 V/W/W−V32.567/32.735/32.885s including compilation; restore-reuse08 warm later passes0.774/0.834s |O(K M_A M_B/P) per contribution, source multiplicity1/3/9|
| restore factories |P fresh addition/output factories;F no counterpart |120/116/116 compiles before;120/16/16 after; source21 removes200 run compilations |fixed source-signature classes, consumers multiply only before fix|

The restore timer is in the sole `photon_sigma.contract_lorentz_blocks` consumer. Lane S must not add these seconds a second time. A two-plan chi call and a single-plan call have different Green data flows, so no isolated “tuple overhead” number is valid. The ns4 action has twice the arithmetic of two ns2 blocks, but an actual ns2 scalar-route timing has not been measured; no empirical ns4/ns2 ratio is asserted.

## Architectural proposals

Ranked by measured/bounded gain per added complexity. All retained changes keep INVARIANTS6: quadratic objects on allP ranks; parent tails gain no collectives; canonical P-agnostic files; one symmetry owner. Byte figures are array payloads unless labeled HLO peak; allocator arenas/aliases are separate.

1. **Accept: reusable restore executables, stream outputs.** Removes fresh addition-function identity across consumers: source gate21 saves200 compiles and38.17 Sigma-side seconds. Keep parent packed V/W, one output accumulator, unchanged source order, typed unfold_isdf_operator and mix_lorentz_blocks; delete inner donating-JIT factory. Before/after one TT output360,000B/rank, largest CC1,327,104B, parent TT120,000B; O(K M_A M_B/P), no resident cache added. Gate21 EQP0/1 and sectors exact. Full data caching is deferred: one full packed-q response8,714,304B/rank, bothV/W17,428,608B versus one output; this grows quadratically. Reordering consumers to restore each source once requires lane S coordination and new lifetime/identity gates. Compiler reuse is implemented; data reuse is not claimed.

2. **Accept: chi family shape classes with runtime canonical vertices.** Removes96 run compilations and12.14 screening seconds beyond restore-only gate21. Data flow keeps each existing Green/orientation contraction; canonical gamma_perm_phase tables become replicated tiny runtime operands. Cache distinguishes identity/nonidentity at each endpoint, shapes, plans and rule signatures. Delete captured per-vertex constants from the factory. Main array sizes unchanged; four integer/complex vertex vectors are a few hundred bytes replicated/rank, independent ofM,N,K,P. TT HLO peak20,736,109→21,312,541B:576,432B extra compiler scratch despite the small vertex inputs. Literal complex Lehmann oracle and exact gate22 retain Γ2 phases and rectangular families.

3. **Accept: coupled zeta tail.** Removes128 transports and32 transforms/tile, saves11.688ms measured warm unit (22.9%). Stream one left child projector/IFFT across the three channel updates, then advance the spin pair; right transports keep canonical vertex permutations and each channel's summation order. Delete the outer repeated channel tail. Three full-k Z outputs41,515,200B/rank atR3844; final q-IBZ solved outputs13,838,400B. Same O(K M_T R/P) scaling; HLO peak452,208,158→452,172,878B. No C_q or Z-projector change and no added collectives; each current Gram remains distinct. Bit-exact four-file ablation gate passes; fresh source gate24/25 passes all four files and downstream EQP/sectors exactly.

4. **Reject resident family faces for this gate after ablation46.** Removes352 endpoint unfolds/response, leaving4 endpoint faces per term. Upper TT-sized kernel saving3.048ms/response; exact mixed-family aggregate is not claimed. Keep raw parents, unfold both endpoint orientations once/family through existing typed owner, then supply child carriers to family-pair kernels. Before parent payload2,242,560B plus largest active child pair4,423,680B=6,666,240B/rank. After parents+four resident family endpoints8,970,240B. ScalingO((Qk+K) S N(M_C+M_T)/P), saving grows withH×blocks. Delete paired unfold from those kernels only after a new carrier contract and memory gate. Ablation46 measured0.430s family setup and0.431s warm16-block work versus0.390s in11; screening19.37s versus19.62s does not justify the new lifetime/API on this gate. Required future gate: child parity including antiunitary actions, mixed-family chi, HLO memory, exact EQP/sectors and larger-deck measurement.

5. **Defer batched Lorentz Green/FFT reuse; shape-class compilation already accepted separately.** Compute one Green/FFT pair per family pair and tau; stream that pair's1/3/3/9 vertex contractions, deleting repeated G/FFT construction. At most16→4 Green pairs, bounded75% of that component. On this gate the entire16-block warm public work is0.411s, so even its total elimination cannot remove the residual multisecond stage; a TT-sized Green ceiling is0.255s/response. One TT Green5,760,000B/rank, pair11,520,000B; CC pair42,467,328B. Nine TT accumulators add2,880,000B versus one; O(K S² M_A M_B/P), withH reuse and a fixed small Lorentz axis. Requires a new block-axis accumulation contract, rectangular classes, both orientations, HLO live-buffer/collective gate and exact science. Benefit may become significant for largerM,N,H; no unmeasured large-system claim is made.

6. **Defer two2×2 spin blocks / gather folding.** Dense action costs0.145472ms/TT block; half-arithmetic ceiling0.072736ms before launches/traffic (≤0.317% of measured block span). The symmetry service must authenticate and own compact subblocks; no local det/sign formula. Delete dense action only at that owner, preserving gather, lattice phase and antiunitary operations. Face payload unchanged (TT1,152,000B/rank/endpoint); action table2304→1152B onK9. ArithmeticO(K N M S²/P) becomes half for this authenticated support; traffic remainsO(K N M S/P). Gate arbitrary complex spatial+TR action, direct face identity and final printed science. Actual ns2 scalar timing remains unrun.

7. **Defer seam fusion and carrier-layout changes.** The final zeta unpack happens4 times, and the existing PackedCentroidBasis conversion factories are already cached. Current file write≈8ms is not all removable unpack time; the measured warmed unpack is0.430ms/file, three current files per fit. Fuse only a demonstrated intermediate with its neighboring reshard/writer; delete that materialization and preserve canonical files, logical padding and cross-P readback. Payload16 Qq M N_G/P; current padded staging2,355,600B/rank, charge4,522,752B. New face layout payload stays16 K S N M/P; one TT face1,152,000B. Actual Green-layout transposes cost0.296128ms/TT block; layout changes have a broad consumer contract and cannot be justified by source moveaxis alone. Gate byte-identical files, typed axes, both families, memory/collectives and EQP/sectors. No second conversion owner.

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

All steps above exit0. Claims1004,1007,1031,1033,1039 own the corresponding verdicts. P4 source artifacts contain source.diff/source.sha256. Tests are scoped: CPU oracles do not replace the actual GPU EQP/sectors gate. Fresh-fit24/25 and wider gates27–30 also pass exact identity, as recorded below. Claims1049,1055,1056,1066,1073,1077,1078 cover the later measurements.

## Verification and disposition

Adopt pushed commits `87a2bfaa`, `0f8fbc3e` and `f811f734` in that order on this lane branch; the orchestrator integrates. They retain the parent route and physical arithmetic while removing measured compilation overhead. Warm chi is already comparable to F on the tested deck; current-zeta transport remains slower than F even after the measured tail sharing and still needs a wider scaling experiment before any inherent limitation is asserted. Sigma-side residual work belongs to lane S; only restore seconds are attributed here.

Before-change sandbox gate0 fails inherited rule allowlist findings and historical CLAIMS rows without evidence tokens (`D/gate0_before.log`); the AST suites passed. This is registered in KNOWN_SANDBOX_ERRORS and is not presented as a passed gate. Failed arm02 omitted implicit dipole.h5 and is preserved;06 replaced it. Earlier allocation refusals and the final pool authorization are in the archived preregistration; no unauthorized pool was used.

The after-unit census, all five deck identity gates, fresh-fit gate and matched pinned dynamic controls are complete. The empirical ns2 GPU comparison and larger-deck/P16 scaling experiment remain unrun. Deferred proposals carry bounds, not claims of inherent scaling limitations. Instruments are reviewable under this report's instruments directory and in each immutable run directory. No broad filesystem search, main push or external message was used.

### Production current-zeta gate24/25

Fresh source24 and reference25 ran sequentially in JID57966610 step lx-Xg4-021818-1673870-3826 exit0. Reference25 consumes the pristine09 fresh canonical files in the same downstream source;24 builds all four anew. Files, EQP0/1 and all sectors are exact, so the fit change is isolated from prior cached-deck zeta conditioning. Instruments/Nsight affect whole driver walls; only the captured warm tile is the performance claim.

### C_q setup clarification

Pinned09 prints four CCT calls: charge3.0s; current1/2/3 4.0/1.9/1.9s (includes1.2s GEMM planning each current call). F10 prints3.1s and2.7/1.7/1.7s. `isdf_fitting.py:793` explicitly passes each current vertex to C_q. Sharing a single numerical Gram across vertices would change the method; sharing its common projector construction is a separate possible optimization. Direct C_q receipts36/37 and solve-owner receipts43/44 below separate these costs. The archived initial audit incorrectly called C_q shared; this section corrects that statement.

### After-unit26 census

JID57966610 step lx-Xg4-022703-1727110-4335 exit0; same-source EQP0/1 and sectors exact. Optimized HLO now has4 vertex integrator modules (0464/0504/0560/0612), all0 explicit collectives, and100 reusable restore addition modules versus300 before; restore maximum1,804,081B remains unchanged. TT after peak21,312,541B versus20,736,109B before. Native warm TT24.162491ms; sum16 warm public calls0.428895s,0 compiles. Baseline P/F captures22.970702/23.792180ms and warm sums0.410999/0.419334s; these single captures do not establish a small steady-state difference. Additional five-repeat host checks are included in34/35. The accepted gain is cold compilation/stage wall, not a claimed chi arithmetic speedup.

Arm26 public host seams:36 pack_axis calls1.958555s/10 compiles;6 unpack_axis1.207277s/6;1 pack_operator0.244801s/1. The four raw-parent endpoint unpack calls occur at the file seam, with shapes(3,80,4,M)/(3,4,M,80); they are not repeated in each chi block. Later operator-axis pack calls reuse existing compiled converters (26 warm calls total≈a few ms). Packed Dyson first2.211696s/5 compiles, repeat0.026387s/0; Gamma completion2.789255s/21 compiles. These profiled host timers are not baseline walls or isolated native kernel sums. M/26_after_units/{unit_timings.rank0.json,census.json,stats_nvtx_gpu_proj_sum.csv} owns the receipts.

### Wider source gates27–30

All same-source EQP0/1 and CC/TT/CT rows pass exact identity (Si256 rows/file; MoS2 90). JID57966610 steps27 lx-Xg4-023200-1761689-4535;28 lx-Xg4-023338-1770882-8474;29 lx-Xg4-023501-1779986-4088;30 lx-Xg4-023737-1798141-4536, all exit0. Each arm carries its canonical parser outputs. Si GN29 and MoS2 dynamic67 node dispatches retain every baseline rule digest.

| deck | after screening / Sigma / total s | after compiles / compiler s |
|---|---:|---:|
| Si GN27 |7.68 /61.51 /91.44|816 /57.45|
| Si COHSEX28 |6.14 /47.29 /72.95|676 /47.36|
| MoS2 packed bare29 |16.20 /112.64 /146.00|1413 /105.59|
| MoS2 dynamic30 |19.84 /121.99 /164.88|1602 /118.91|

These cached-deck Si paths do not invoke the changed full packed-photon block loop; their compile counts are unchanged and no Si speedup is claimed. Remaining Si Sigma regression belongs to lane S. MoS2 bare gains restore reuse only; full static/dynamic also gain current-chi shape reuse. Guarded F replay32 correctly refused: its requested box differs by7.3e-13Ry but lies outside the selected immutable certificate. The common-certificate selector and successful controls38–41 below retain all acceptance guards.

### Warm Dyson and repeated chi,34/35

F34 step lx-Xg4-024231-1845178-7018 exit0 and P35 lx-Xg4-024435-1858899-4547 exit0, JID57966610. Each block has five warm repeats with0 new compilations. TT public host median P24.131629ms (24.004–25.752), F27.127103ms (26.922–27.841); all80 warm calls sumP2.081135s/F2.333986s. These repeat samples do not support a persistent parent warm-chi penalty.

Packed Dyson native warm module: P23.859014ms /1383 GPU ops versus F72.686282ms /4189; q-IBZ3 versus fullq9 at extent492. Both first public solves take≈2.328s and5 compiles; warm public calls include capture/synchronization overhead (P37.044ms/F87.560ms), so the native module is the device comparison. Gamma public completion P2.779566s/21 compiles versus F2.850549s/21. Gamma has the same owner/cost here; no separate Gamma fix is justified. Raw native tables and HLO/census remain in34/35.

CPU31 passed all6 parent-Z/C_q parity cases, including coupled ns4 currents, scalar/ns2, nonsymmorphic glide and antiunitary transport. JID57966610 step lx-Xg0-025019-1899125-1232 exit0. Its guarded common-certificate selector found an existing immutable rule covering both P and F for every Si/MoS2 window; no certificate was enlarged or recomputed. Matched pinned controls38–41 completed using those files (29/67 dispatches, every digest identical across P/F).

### Direct C_q and final file seam receipts36/37

P36 step lx-Xg4-025057-1902256-9775 and F37 lx-Xg4-025229-1912766-5738 exit0, JID57966610. Every exact repeated C_q call compiles one fresh module; P first calls also each count1, while F first charge counts3 including auxiliaries and its current first calls each count1. P charge first/repeat655/606ms; current696/646,686/680,688/683ms. A repeated call is therefore not warm at this public factory; its compiler work is≈0.51–0.58s/call. Natural production executes C_q once per channel, not once per tile. The fresh `_fused` C_q factory is registered as a remaining source inefficiency. It is not a demonstrated plan-id miss in the Z tile cache.

P current final unpack:3 calls,1 compilation,0.258712s total; warmed per-file median0.430ms. Charge final unpack:1 call,1 compilation,0.582228s. The final write is≈8ms/file as separately recorded by the HDF5 owner. F has no PackedCentroidBasis seam. Fusing the already-warm current unpack could at most recover submillisecond work per file on this deck; a broad writer/layout rewrite is not justified. The36/37 solve wrapper observed the optional stacked alias, not the exercised per-channel core owner; its zero records are not evidence of zero solves. Corrected direct-owner43/44 receipts are reported below.

F C_q charge first/repeat622/367ms (3/1 compiles); current504/472,479/477,518/484ms (1/1 each). P direct current C_q is about0.17–0.21s/call slower cold than F on this instrumented deck; this setup difference is separate from the improved Z tiles. A common current C_q projector/coordinator could also remove two repeated≈1.2s planning steps, but its three numerical Grams must remain distinct. No such extension was implemented without an isolated sharing gate.

### Direct solve-owner receipts43/44

P43 step lx-Xg4-025755-1947913-6796 exit0; F44 lx-Xg4-025921-1956174-1436 exit0, JID57966610. Actual core.solve_zeta calls: charge24 (12 production tiles plus12 harness repeats), current36 (12 tiles×3 distinct channels). P charge first754ms,6 compiles over all calls, warm median1.711ms; F737ms,10 compiles, warm1.815ms. P current solve extent100, first864ms,5 compiles over36 calls, warm3.228ms; F logical extent98 with padded output100, first842ms,10 compiles, warm3.098ms. F has4096/remainder1024 shapes; P one3844 tile shape. This is shape/remainder compilation, not a new solve compilation for every current channel or tile.

The gate uses hoisted distributed getrf once per current channel with separate block-cyclic pivots. The existing optional stacked raw-array solve explicitly excludes those pivots/provider factors. Simply enabling it would not produce one safe coupled solve on this path. Its entire warm per-tile solve budget is≈3×3.228=9.683ms; a dispatch-only saving is smaller. A new batched hoisted-factor/pivot contract is deferred; it must retain the three distinct Grams and the existing factor owner. Current Gram payload120,000B/rank/channel at q-IBZ,360,000B across three; no replicated-square factor is admissible. This avoids claiming either an already-shared numerical C_q or an unmeasured coupled-solve speedup.

### Resident-family ablation46 and disposition

JID57966610 step lx-Xg4-030157-1973270-9283 exit0. This run-local ablation unfolds both endpoint orientations once for each family, invokes the existing full-face chi kernel with runtime canonical vertices, and releases the four children after packing the response. It preserves every typed action and exact EQP0/1 (90/90 each) plus CC/TT/CT rows. Charge endpoints4,423,680B and current endpoints2,304,000B total6,727,680B/rank, in addition to retained parents2,242,560B. The source352 per-node endpoint invocations become4 per response.

Those family calls cost0.216687+0.212832=0.429519s,2 compiles/0.274316 compiler seconds. First16 chi calls5.732589s,50compiles/3.459082 compiler seconds; warm16 calls0.430670s,0compiles. Comparable shape-class ablation11: first6.336320s/48compiles/3.832625 compiler seconds, warm0.390107s. Including the new family call leaves only≈0.174s cold host reduction, while warmed calls do not improve in this sample. Whole screening19.37s versus19.62s, not a convincing gain for the added lifetime and API contract. Production remains the shorter paired-parent path. This is a measured rejection for this gate, not an inherent tiny-deck claim; larger systems may change the decision.

Failed harness variants42 (bootstrap order) and45 (wrong face descriptor order) are preserved and registered in KNOWN_SANDBOX_ERRORS.46 uses the existing green_face_kernel_kwargs owner to change only the k extent. Neither failed variant is numerical evidence.

### Parent-family load accounting

Unprofiled baseline06 versus07, already cited step012024/012412: production stage rows report charge/current centroid loads P1.6/0.6s, F3.2/0.9s. The scoped loader/transform/insert triplets execute5 times per family in P and15 in F (10 versus30 total). First charge loader/gflat-to-centroid/reshard-insert P0.4/0.4/0.8s, F0.5/0.3/1.1s; later rows often print0.0s at0.1s resolution, which is not evidence of zero device cost. Parent loading is faster here and is not the observed regression. The separate canonical endpoint unpack/pack receipts in26 distinguish file-seam compilation from loader work.

At fixedK,N,M the dense spin action arithmetic scales S² (ns4/ns2=4 by flop count), and face payload scalesS (ratio2). The authenticated two2×2 representation halves dense ns4 arithmetic, but no empirical ns2 GPU comparison was run. Zeta's streamed operator-pair count is a different schedule and is not priced as this wavefunction einsum.

### Matched pinned dynamic controls38–41

These follow-up controls restore exact P9f569c4b production bytes in the assigned worktree, then restore the gated tip after the batch. Each P arm refuses any src/services difference from9f569c4b; F remains e1559a07. They were measured after implementation, not relabeled as chronologically before-change runs. Existing immutable certificates from31 are accepted by the unchanged rule-cache owner for both recorded boxes. Private replay directories and `common_rules.sha256.json` match byte-for-byte, and every actual node/weight digest in the driver agrees. Planning now takes0.08s in both Si arms and0.12/0.13s in MoS2.

| deck / P,F arms | screening P/F s | Sigma P/F s | total P/F s | compile count P/F | compiler work P/F s | Sigma schedule |
|---|---:|---:|---:|---:|---:|---|
| Si GN38/39 |7.86/6.71|60.10/24.41|87.60/51.17|816/539|57.27/27.35|29 dispatches,6 windows, identical digests|
| MoS2 dynamic40/41 |32.30/16.15|159.88/28.80|215.99/65.39|1896/790|161.17/39.76|67 dispatches,8 windows, identical digests|

Authorized JID57966610:38 step `lx-Xg4-030517-1994260-5532`;39 `lx-Xg4-030652-2002797-2728`;40 `lx-Xg4-030750-2008903-9194`;41 `lx-Xg4-031136-2194473-3936`, all exit0. Claim1077; each arm's `lx_attempt1.log`, `driver.rank0.log` and parser outputs, plus `D/common_control_verdict.json` and `D/baseline_analysis/host.json`. Cross-source Si EQP0/1 each256/256 printed exact; MoS2 EQP0 80/90 and EQP1 81/90, maximum0.001 micro-eV. These source comparisons do not weaken the exact same-P acceptance gate for any fix.

### Final before/after stage accounting and remaining work

This table compares each original P baseline to its same-deck, same-rule P fix gate. Cached zeta means the coupled-tail gain is measured separately by fresh-fit24/25. Compiler seconds remain cumulative compiler work, not elapsed-time subtraction.

| deck | P before/after screening s | P before/after Sigma s | P before/after total s | before/after compile count | identity |
|---|---:|---:|---:|---:|---|
| Si GN13→27 |7.99/7.68|63.73/61.51|90.78/91.44|816/816|EQP0/1 and sectors exact|
| Si COHSEX15→28 |6.39/6.14|48.33/47.29|72.97/72.95|676/676|exact|
| MoS2 full static06→22 |31.21/18.59|150.16/111.85|201.36/146.56|1712/1416|exact|
| MoS2 packed bare17→29 |16.67/16.20|150.60/112.64|184.49/146.00|1613/1413|exact|
| MoS2 dynamic19→30 |32.80/19.84|161.92/121.99|217.54/164.88|1896/1602|exact, all67 rule dispatches unchanged|

The leading hypothesis is partly confirmed: sixteen vertex-specific executable classes and repeated restore factories caused substantial cold work. Re-unfolding the endpoint faces also occurs inside each tau node, but its measured cost does not explain the warm chi wall. Steady TT host medians P/F24.13/27.13ms and native warm Dyson23.86/72.69ms show no parent penalty in those units. The improved current-Z device tile remains39.31ms versus F30.54ms, with different tile shapes and operator-transport schedules; C_q setup also remains slower and publicly recompiles on repeat. Charge-fit warm medians P/F35.21/29.67ms likewise do not demonstrate an advantage for the parent tile. No claim of an inherent small-system penalty or unmeasured larger-system gain is made.

Residual whole-run P/F differences are substantial: full-static P after146.56s versus F51.24s, including Sigma111.85s versus18.12s. Lane S owns that remaining Sigma attribution; the38.31s reduction in this lane's baseline-to-two-fix Sigma wall includes restore work and must not be counted again. A few fresh output factories remain in restores, and natural C_q setup is once per channel. Proposed batching, resident data, compact spin actions and layout changes retain their measured bounds above and are not silently added to production.

Checkpoint scope: 23 focused CPU tests in23 and6 parent-Z/C_q parity tests in31 pass, plus the GPU file/science gates. Sandbox gate0 AST suites pass90/90,34/34,16/16. Its aggregate remains FAIL on the same16 rule/ledger findings before and after; `D/gate0_before.log`, `D/gate0_after.log`, and `D/final_run_audit.json` retain this distinction. Failed harness/control variants02,32,33,42,45 remain preserved; their replacements succeeded. This is numerical identity against the pinned route, not an independent certification of inherited bispinor physics. No main push or Sigma source edit occurred.

### Durable evidence and integration handoff

CFS archive: `/global/cfs/cdirs/m4598/jackm/lorrax_evidence/bisp_prof_zw_2026-09-06/evidence.tar`; 17,601 files, 348,641,280 archive bytes, SHA256 `6ad595eda36bb279ae454ac0f99b140ab4c857b231eb2b40563630dffe91d2f0`. Every member was verified against `manifest.json`; `verification.json` reports17,601 verified,0 missing,0 mismatched. The archive retains native Nsight captures, named rank-zero optimized HLO/memory reports, all driver/step receipts, selected canonical zeta/rules, exact source snapshots and parser owners. Full WFN inputs and other-rank HLO remain in their original run locations. `archive_receipt.json` beside this report gives the bounded scope. The final report is also copied beside the CFS archive after closure.

Cherry-pick `87a2bfaa`, then `0f8fbc3e`, then `f811f734`; the later report-only commit requires no production adoption. All source changes are pushed on `perf/bisp-prof-zw-2026-09-06`, unmerged; the final branch-status check fetches origin and confirms the source tip is not an ancestor of origin/main. Production source bytes are restored to the gated tip after38–41. No live lane steps remain, and the shared campaign pool is left allocated for its owner.

## Chi Green reuse and fresh zeta on 6x6

Active follow-up, 2026-09-06. Rebased production source is exactly orchestrator71ae0bde (covariant Gamma, sign-aware transverse ridge, tau exchange removal and integrated chi/zeta fixes); historical sections above remain pinned to their original sources. Fresh unprofiled P/F arms47/48 copy BISP-SCALE02/03 inputs under runs/MoS2/42_bisp_scale_2026-09-06/prof_zw. Candidate: one Green/FFT pair per tau and C-C/C-T/T-C/T-T class, with1/3/3/9 ordered vertex contractions; unchanged q-IBZ output, Ward contact and Dyson. Baseline source is frozen while these runs execute. P4 pool57982945; optional P16 only if all four nodes are available.

### Rebased P4 baselines and preregistered class memory

Claim1256: P47 step `lx-Xg4-113113-163792-9427` and F52 `lx-Xg4-113511-221323-6412`, JID57982945, exit0. All physical input checksums match. F48 failed on the supplied unsupported `sigma_quadrature_reduction_steps` key; replacement52 omits only that inactive static-Sigma setting. No reference deck was edited.

| fresh full driver | P47, production71ae0bde | F52,e1559a07 |
|---|---:|---:|
| charge zeta s |17.74|13.75|
| transverse zeta s |18.21|inside ISDF setup bucket; separate orchestration timing follows|
| ISDF setup/I/O s |9.09|24.19 (includes transverse fit)|
| screening s |30.94|17.37|
| Sigma s |21.27|21.28|
| total s |109.51|89.70|
| rank0 compiles / compiler s |849 /55.74|782 /46.66|

The gate has K36,7 raw parents/q-IBZ rows,80 bands,ns4, logical C/T597/194 and P4 carriers600/196;11 minimax nodes. Baseline source performs352 Green builds and352 Green FFTs over sixteen blocks. The proposed four-class sweep performs88 of each, preserving all16 forward/reverse vertex completions and16 final q FFTs.

One Green has16 K ns² M_A M_B/P bytes: P4 CC829,440,000; CT/TC270,950,400; TT88,510,464. Two are live per class, independent of the number of tau nodes. Nine TT R-space accumulators require49,787,136B/rank versus5,531,904B for a singleton; q-IBZ outputs require9,680,832B for that class. Packing retains the existing all-rank accumulator39,517,632B, with one class consumed before the next is built. These are payload estimates pending optimized HLO, not an allocator-peak claim. Actual P16 carriers and measured peaks will be recorded separately; at unchanged extents the payload divides by four.

The private kernel parameter changes from `vertex_pair` to `vertex_pairs` and returns a tuple of class outputs. The established public singleton function delegates to the plural class entry. The sole photon packer accepts an explicit block order so C-C,C-T,T-C,T-T residency is enforceable without a second packer, symmetry owner or dispatch layer. Ward subtraction remains at the existing TT q-IBZ consumer, and the Dyson is untouched.

Pool continuation: own authorized fallback allocation57986810 succeeded (00_environment/allocation_attempt1.log). Waiting shared-pool probe51 was cancelled before it created a driver or science step; replacement61 and cached baseline62 run sequentially on the owned node. Shared57982945 remains the optional P16 pool.

Publication: the rebased continuation is published as `perf/bisp-prof-zw-green-reuse-2026-09-06` to preserve the campaign's no-force-push rule; the earlier published profiling branch remains intact. The assigned worktree/local branch is unchanged.

### First class implementation: exact and faster warm, but cold compilation regresses

Owned pool57986810. Cached baseline62 `lx-Xg4-115421-357311-4476` exit0: screening25.98s,total60.68s,562 compiles/33.89 compiler seconds. Candidate64 `lx-Xg4-115931-388157-6151` exit0: screening27.75s,total58.77s,562/36.75. Both EQP files210/210 and all CC/TT/CT rows are exact. The total-wall drop includes unrelated bring-up variation; **screening itself is1.77s slower cold**, not a stage speedup.

Profile61 `lx-Xg4-115151-325927-9809` and65 `lx-Xg4-120037-396487-6986` exit0, with three synchronized warm repeats. Baseline mean complete16-block sweep2.724131s; four classes≈1.0425s, a substantial steady-state reduction. The T-T class's nine blocks take a native136.267897ms versus107.428298ms for one original T-T block. Both use the identical11-node/weight SHA256 `2a101e324c7ee44477a03066bdd54b4f6e04395416d15e00dd5aa4f6912cf872`.

Optimized HLO65 contains four vertex-class modules, each with exactly two distributed Green GEMM calls inside the tau body, zero explicit collectives, and2+n_vertices static FFT calls (two Green FFTs per tau plus each final q FFT). Peaks CC2,606,656,548B; CT1,179,007,644B; TC908,057,756B; TT407,508,828B. These include compiler temporaries beyond the payload model; all centroid-quadratic operands retain both mesh axes. No extra family Green pairs appear when n_vertices=9.

The first implementation unrolls the class's vertex contractions. First-call compiler work rises from9.247814s for the original sixteen calls to12.533002s for the four class calls in the instrumented comparison. The next bounded ablation replaces that unrolling with a small vertex loop while keeping the shared Green pair and ordered per-block tau sums. This preserves the measured warm benefit's data flow and targets the observed cold regression.

CPU63 passes13 focused tests, including all four classes against literal complex densities, parent glide/TR transport and scalar contour kernels. Its fresh-file comparison uses `compare_zeta_h5.py --rtol0`: charge maximum absolute3.563056e-4/normalized6.340472e-7; current1 7.412283e-9/1.018844e-11; current2 8.466901e-6/1.020736e-8; current3 5.300709e-9/7.176702e-12; every file finite, none bit-identical across P/F. Fresh cross-source EQP0/1 each12/210 printed exact, maximum4.470 micro-eV. These cross-source results are not substituted for the exact same-source class acceptance gate.


### Streamed vertex loop: accepted P4 implementation

Claim1280, owned JID57986810: CPU70 `lx-Xg0-120729-433343-6355`, cached71 `lx-Xg4-120925-443097-6650`, profile72 `lx-Xg4-121029-449406-2068`, all exit0. The small vertex loop shares the Green pair and preserves each block's tau accumulation order. Both EQP files remain210/210 exact and all CC/TT/CT rows match at the printed digit against62. The13 focused CPU oracle/contour tests pass.

| cached P4 measurement | baseline62/61 | streamed71/72 |
|---|---:|---:|
| screening wall s |25.98|24.67|
| total wall s |60.68|56.85|
| whole-driver compiles / compiler work s |562 /33.89|562 /33.69|
| first complete chi sweep s |14.169094|12.627803|
| first-sweep compile events / compiler work s |48 /9.247814|48 /9.499476|
| mean warm complete chi sweep s |2.724131|1.092205|
| warm compilation events |0|0|

The accepted version reduces warm chi time59.9% and cold screening5.0% on this deck. Compiler work is cumulative, not a quantity subtracted from stage wall. Instrumented and uninstrumented timings occupy separate rows. The unrolled candidate64/65 is retained as an ablation: it is slightly faster warm but loses on cold compilation.

Warm complete class means, baseline singleton sums -> streamed class, in ms: CC449.675 ->437.134; CT627.431 ->244.984; TC628.755 ->242.692; TT1018.269 ->167.394. Three warm repeats per unit, including one captured TT repeat. Native rank-zero Nsight TT: original one block107.428298ms (6462 GPU operations), streamed nine blocks148.173355ms (7297 operations). The capture is a valid warm CUDA-profiler range; it is not nine times the singleton cost.

Optimized HLO72 and census.json: four class modules, two distributed Green GEMMs and one dense parent spin action in each tau body, zero explicit HLO collectives; static FFT sites3/5/5/11. Distributed service collectives are not inferred absent from this HLO census. Per-rank executable peak bytes CC2,606,656,548; CT949,829,044; TC949,828,532; TT386,840,612. One class is consumed before the next; centroid-quadratic arrays retain both mesh axes. Ward placement, ordered orientation, q-IBZ and Dyson code remain unchanged.

### Fresh P/F zeta at a common transaction boundary

Claim1280, owned pool57986810: P68 `lx-Xg4-121237-460008-3992`, F69 `lx-Xg4-121327-464470-6651`, exit0 after intentional post-zeta stop. Inputs and empty private tmp stores are identical physically. Charge wraps fit_zeta_to_h5; transverse spans coupled-coordinator construction through finish_channel(mu=3), including all three writes and provenance. This avoids comparing F's historical stage bucket with P's separate bucket or adding overlapping thread timers.

| fresh transaction | P68 wall s | F69 wall s | P/F compile events | P/F compiler work s |
|---|---:|---:|---:|---:|
| charge |15.769318|13.571303|86 /98|5.988780 /6.184606|
| three current channels |17.331768|16.760236|105 /122|9.241678 /9.621647|

Tile profiles49/50 use the same46080 grid points but the automatic tile widths differ. P charge: four11520-point tiles, warm990.394/989.531/989.731/989.700ms. F charge:26958/19122 points, warm1538.662/1097.658ms. P current: three15360-point coupled tiles, warm876.659/874.540/874.737ms. F current:21320/21320/3440 points, warm961.349/947.199/189.916ms. All warm tile repeats compile zero times. Total warm tile work is about3.959/2.636s (charge P/F),2.626/2.098s (current P/F); the shorter P current tile alone is not evidence of a fit speedup.

P first charge tiles4613.039/991.086/990.294/990.744ms, first current2495.195/874.512/873.988ms; F first charge4117.258/2846.232ms, first current2261.911/944.132/1129.117ms. Shape changes explain the F remainder compilation. Native representative current tile P874.449030ms/884 GPU operations versus F948.829273ms/1313 operations; optimized peak7,024,623,572 versus7,885,377,476 bytes. Each HLO has one static all-to-all, all-gather and all-reduce outside the streamed parent tail; these site counts are not dynamic launch counts.

All four canonical files were compared in63 using compare_zeta_h5.py; the numerical differences and finite counts above are the fresh P/F result. No fresh-fit identity is claimed. The exact acceptance gate for the chi source change uses the same copied P zeta files and11-node schedule.


### Native kernel accounting and remaining fit cost

Native receipts in61/72 (`stats_cuda_gpu_kern_sum.csv`, `stats_nvtx_gpu_proj_sum.csv`) separate service communication from explicit HLO collectives. Kernel durations below are sums over the captured rank-zero range, not additive wall-time partitions; ranges may overlap and module projections include idle gaps.

| kernel class, summed ms | old single TT block61 | new nine-block TT class72 |
|---|---:|---:|
| NCCL service kernels |53.769|48.119|
| GEMMs including spin action |17.127|17.121|
| FFTs and normalization |8.886|9.069|
| transpose fusions |9.113|49.239|
| dynamic accumulator updates |0|1.428|
| other kernels/fusions |1.420|2.414|

The two endpoint Green service ranges project42.627+42.532ms before and40.482+39.951ms after. Each range executes11 times, not99. Both captures have3168 NCCL launches and1595 GEMM launches across the range: retaining nine outputs does not multiply distributed Green construction. The smaller dense spin GEMM executes11 times and totals≈1.05ms. Its phase/gather/transposes are fused; no independent seam or spin-only time is invented from a generic fusion name. Streaming adds99 accumulator-update launches,1.428ms; the new vertex input transpose totals44.997ms. This is the measured tradeoff for the reduction352→88 Green builds/Green FFTs per sweep. Warm64/65 versus72 isolates the compile/vertex-loop tradeoff described above.

Fresh current tiles49/50 remain bounded by different layouts and tile widths. Across each captured range, transpose-fusion kernel sums P545.361ms/F67.590ms, GEMMs24.266/384.874ms, FFT+normalization85.384/118.527ms, dynamic updates67.323/48.798ms, NCCL4.580/27.039ms. These labels are generated-kernel classes, not a new attribution of every transpose to spin transport. P's reduced GEMM work does not imply a faster same-grid fit: the full warm-tile sums and common transaction walls show the remaining cost. No zeta source change is proposed from this unequal-tile capture alone.

Follow-up comparison73 (`lx-Xg0-122412-518038-2249`, JID57986810, exit0) runs compare_zeta_h5.py against every file produced by the common transaction timers68/69. All four are finite and reproduce the same maximum differences as63. A zero-tolerance comparison fails for each P/F pair; that numerical result is retained rather than treating the comparison process exit0 as identity.

The baseline71ae0bde and accepted-source sandbox gate0 logs are byte-identical (`D/gate0_six_pinned.log`, `D/gate0_six_after.log`): AST suites pass; the aggregate fails on inherited rule-allowlist findings and twelve pre-existing ledger rows. This is a scoped before/after check, not a claim of a clean repository-wide gate.


### Fixed-main warm screening control

Claim1283: fresh F75 `lx-Xg4-122636-528904-3333`, JID57986810, exit0; both EQP files210/210 and CC/TT/CT rows are exact to unprofiled F52. Supplemental cached F74 was refused by the fixed-main slab-COHSEX restart gate (step `lx-Xg4-122512-523232-8565`, exit1); its artifacts are preserved and the harness limitation is registered.75 uses the supported fresh path, with no source/gate bypass.

| synchronized chi unit | F75 | P baseline61 | P accepted72 |
|---|---:|---:|---:|
| first complete sweep s |11.337390|14.169094|12.627803|
| first compile events / compiler s |39 /7.814892|48 /9.247814|48 /9.499476|
| warm CC sweep ms |441.704|449.675|437.134|
| warm CT sweep ms |600.504|627.431|244.984|
| warm TC sweep ms |601.011|628.755|242.692|
| warm TT sweep ms |976.076|1018.269|167.394|
| warm complete sweep s |2.619295|2.724131|1.092205|
| warm compile events |0|0|0|

Parent Green reuse now beats fixed-main's unchanged singleton loop in steady state. Native F TT one-block projection103.312531ms/6464 GPU operations, versus P old107.428298ms/6462 and P nine-block class148.173355ms/7297. F HLO peaks CC2,706,078,096B; CT939,855,392; TC939,854,880; TT325,283,968. P baseline -> accepted peaks CC2,606,656,548 ->2,606,656,548; CT874,187,764 ->949,829,044; TC874,187,252 ->949,828,532; TT292,797,140 ->386,840,612. The largest chi executable remains CC; the TT class's extra outputs increase its own peak without exceeding CC.

Residual cold gap: the uninstrumented parent cached screening24.67s is still above F fresh17.37s, with different supported fresh/restart setup paths. The direct first-call chi sweep remains1.290s slower than F, while warmed chi is1.527s faster. This measurement justifies the class change but does not attribute the entire screening-stage remainder to chi or claim a full-driver speedup against F. Per-stage buckets and their support work must stay separate from the isolated unit receipts.

Tile-capture provenance:49 `lx-Xg4-113659-233239-8665`,50 `lx-Xg4-113936-250273-5479`, shared JID57982945, both successful intentional post-zeta runs. CPU63 is `lx-Xg0-115754-378712-4689`, owned JID57986810. The bounded six_run_audit.json records every launch, artifact scope and failed/unlaunched variant; reports/instruments and six_*_receipts retain the parser inputs.


### Follow-up disposition and integration

Claim1284. Adopt source commits `6effddb5` and `9e31a1a7`, in that order, from pushed branch `perf/bisp-prof-zw-green-reuse-2026-09-06` (unmerged). The second commit replaces the first commit's unrolled vertex contractions with the measured streamed loop. Both are based on orchestrator71ae0bde; no Sigma source change or new symmetry rule is included. The historical87a2bfaa restore patch is not part of this follow-up recommendation; orchestrator already owns the newer restore implementation.

The accepted data flow is two Green tensors and their FFTs per family pair per tau, then1/3/3/9 ordered vertex contractions into per-class distributed accumulators. The public singleton wrapper uses the same owner. The old per-block Green path and duplicated packer lifetime are replaced, not retained as selectable production paths. Green payload scales16 K ns² M_A M_B/P per endpoint, and the class accumulators/output scale16 n_vertices K M_A M_B/P. Band count and parent count enter the endpoint face payload16 n_parent ns M_A nb/P and the Green build work, not the quadratic Green storage. No centroid-quadratic object is replicated onto fewer than all ranks; files remain canonical/P-agnostic; typed symmetry actions, q-IBZ output, ordered orientation, Ward contact and packed Dyson stay with their existing owners.

P16 was requested in66 with `--jid57982945 --wait1800 -N4 -G4 -n16`. After approximately ten minutes queued, the shared pool had7/16 GPUs free across its four nodes (`D/shared_pool_p16_final_status.log`); this lane's waiting client was cancelled before it launched a science step. The owned fallback57986810 has one node. Thus P16 before/after, measured P16 carrier extents and P16 peaks are **unrun**, not inferred from P4. The baseline pin was restored to the accepted source in a finally block; git diff HEAD -- src services tests is empty and p16_source_restored.txt records restoration. No shared pool or other lane job was cancelled.

Final validation scope: all same-source P4 science gates are printed exact,13 CPU cases pass, fresh P/F zeta comparisons explicitly retain nonzero differences, native captures are valid, and before/after sandbox gate0 has identical inherited findings. Failed48/74 and unlaunched51/66 remain preserved. The warm chi penalty is removed on the6x6 deck; the common fresh zeta transaction is still slower on P, and cold screening retains support/compilation overhead outside the isolated chi improvement. No inherent tiny-system explanation or unmeasured P16 speedup is asserted.


Six-by-six durable archive: `/global/cfs/cdirs/m4598/jackm/lorrax_evidence/bisp_prof_zw_6x6_2026-09-06/evidence.tar`; 6746 files, 1304565760 bytes, SHA256 `3af4b6f04af224bf8a69fa1e27a400bb825e2a8fa3b8fede67f66e15985c973c`. Every member verified: 6746 checked, zero missing/mismatched. The archive retains this follow-up's native captures, rank-zero optimized HLO, original canonical P/F zeta, step/parser receipts and exact source snapshots. The final report and archive receipt are also copied beside it after closure. Owned pool57986810 was released after confirming that only its extern step remained (`D/six_pool_release.log`); shared57982945 remains its owner's allocation.


## Phase 3: zeta tile table and screening attribution on 6x6

Production base ORCH443e95be plus this lane’s rebased chi Green reuse (32fc693a,00efb049); phase3 changes and gates are in the speed-protocol table below. Original P4 zeta carrier C597/T194 is kept distinct from SCALE’s common P4/P16 carrier C597/T200; fixed-main refuses T194 atP16. P16 static/dynamic P/F80–83 reuse one canonical P zeta copy and the matched certified dynamic-rule schedule. Shared authorized pool57988457, sequential arms. The earlier P/F zeta difference verdict is reopened for an input/operator/solver decomposition; finite output alone is not a numerical explanation.


### Phase3 P16 baseline matrix before source changes

Claim1302; authorized pool57988457, all four steps exit0. [lx] step lx-Xg4-130452-691599-3051 exit 0 in 82 s; [lx] step lx-Xg4-130616-707933-7813 exit 0 in 102 s; [lx] step lx-Xg4-130800-715299-4243 exit 0 in 117 s; [lx] step lx-Xg4-131001-721896-8573 exit 0 in 210 s. Source dc7fd463 is the rebased continuation of the phase2 accepted source on443e95be.

| P16 arm | screening s | Sigma s | total s | rank0 compiles / compiler seconds |
|---|---:|---:|---:|---:|
|80 parent static|31.63|22.31|75.86|646 /34.03|
|81 fixed static|39.43|33.92|96.46|604 /28.85|
|82 parent dynamic|37.19|44.90|112.37|897 /52.55|
|83 fixed dynamic|111.15|66.84|203.57|825 /43.27|

Screening sums chi0,W,screening-support buckets; dynamic W persist/head is separately0.84/0.36s. Sigma includes rule planning,tau andother work. Fresh fit is excluded: allfour private tmp directories start from90_P_static_P16_baseline/tmp, and dynamic arms preserve the matching replay certificates. Logical C597/T200 differs from phase2'sC597/T194; no cross-carrier gain is claimed. P/F numerical comparison and complete screening attribution follow. Native/HLO arm85 stalled after the warm Dyson capture; it is preserved as incomplete and replaced by a host-only profile.


### Complete screening attribution before fixes

Claim1306. [lx] step lx-Xg4-131855-756227-1095 exit 0 in 73 s; [lx] step lx-Xg4-132011-760415-3756 exit 0 in 67 s; [lx] step lx-Xg4-131846-737877-4603 exit 0 in 81 s; [lx] step lx-Xg4-132011-760413-4258 exit 0 in 102 s; [lx] step lx-Xg4-132436-780659-5842 exit 0 in 97 s; [lx] step lx-Xg4-132615-787511-6795 exit 0 in 104 s; [lx] step lx-Xg4-132808-798316-6364 exit 0 in 72 s; [lx] step lx-Xg4-132928-803095-7332 exit 0 in 52 s; [lx] step lx-Xg4-133021-808305-6677 exit 0 in 45 s; [lx] step lx-Xg0-133339-824889-8569 exit 0 in 11 s.86/87 and90/91 are synchronized host-only profiles; they include one explicit warm Dyson repeat, listed separately. The first response row includes compilation and all support owners; subtracting the1.092s warm chi result from a cold25s stage would misattribute chi compilation to other work.

| owner | P4 P/F wall s | P16 P/F wall s | P/F compile events | P4 P/F compiler work s |
|---|---:|---:|---:|---:|
| bare read + packed operator build |1.983/0.968|1.868/0.917|25/17|1.552/0.599|
| complete chi construction |8.679/8.471|14.636/26.976|49/44|5.412/4.100|
| first packed Dyson |1.837/2.989|2.836/5.072|5/10|0.387/0.669|
| explicit warm Dyson repeat |0.154/0.785|0.538/2.702|0/0|0/0|
| head response (before cell completion) |9.383/2.671|9.403/2.849|92/28|6.459/1.599|
| Gamma cell completion |2.456/1.359|2.288/1.419|24/21|1.249/0.977|
| whole response including warm repeat |25.940/18.682|33.038/41.404|213/138|16.206/9.065|

The residual after direct-child timing subtraction is bounded orchestration/sanity/vector packing; nested reader/block timings are retained in owner_timings.rank0.json and must not be added again. The warm Dyson ratio is5.08 atP4 and5.02 atP16, consistent with7 versus36 q rows. Parent packed extents are1212 atP4 and1280 atP16 (C608/T224 atP16), so the q-count alone is not an exact flop ratio. The main cold parent penalty is the head-response per-star loop, not the q-IBZ LU.

95 resident-head ablation: reuse the existing typed child owner, concatenate its stars, and contract head wings once. Head9.383->6.822s and92->77 compilation events; full response25.940->23.404s; both210-row EQP files and every CC/TT/CT row are printed exact. This prototype still pays the old per-star unfolds, so a direct full-child group through the same owner is the bounded next implementation. AtP4 a full pair of charge faces costs2*16*K*ns*M_C*nb/P=55,296,000B; a12-child star costs18,432,000B. AtP16 the full pair costs14,008,320B withM_C608. This is linear in centroids, not a replicated quadratic operator.

### Zeta owners and the non-one-to-one boundary

93/94 use the original logicalC597/T194 inputs and retain canonical normal-equation shards on allfour ranks. Representative native charge tile P968.194ms versus F1503.018ms, but tile widths are11520 versus26958 points. P projector GEMMs52.129ms (10 launches), F702.893ms (160); P transpose fusions649.842ms versus F126.255; FFT+normalization96.359/225.246; NCCL3.844/41.313; other fusion kernels141.548/346.537. HLO executable peaks10,601,691,180/13,979,802,464B per rank. Across the same46080points, warm charge producer P3.879s versus approximately F2.568s; do not compare unequal single tiles as a fit speedup.

Current-family native/clean unit profiles49/50 remain applicable to unchanged zeta source: P874.449ms at15360points versus F948.829ms at21320; whole warm grid2.626/2.098s. Current projector GEMMs24.266/384.874ms, transpose fusions545.361/67.590, FFT+normalization85.384/118.527, dynamic updates67.323/48.798, NCCL4.580/27.039. All raw warm tiles compile zero times.93/94 additionally repeat nested solve owners; their outer F tile timings include inner repeats and are not substituted for these clean timings.

C_q public calls recreate the fused producer in both sources: P charge first1.153s/repeat1.141s, each1compile (0.841/0.811compiler seconds); F first0.939s/4compiles and repeat0.625s/1compile. Current first P1.414/1.083/1.066s, F0.732/0.719/0.691s; each repeated current C_q still compiles once. This is once per channel in the natural run, not a per-r-tile factorization. Charge factor P3.552s/4compiles, F4.507s/11; current factors P0.937/0.311/0.313s, F1.859/0.344/0.351s. Warm solve applications are≈14ms/charge P tile and≈12ms/current tile; F≈28/21ms for the two charge sizes and≈15/5ms for full/remainder current sizes. Zeta write calls total4.04/4.12ms; canonical unpack calls P0.783s including0.511compiler seconds, while the final two warm calls are0.672/0.532ms. Writer completion/file close remains separately visible in the production host rows; write_slab return time alone is not durable HDF5 wall.

The bispinor parent producer contracts open-spin parent projectors, then transports every scalar source spin block. ns4 has16 output pairs, each with four structurally supported source pairs from the typed spin action:64 geometric transports for one endpoint in a charge tile. Coupled current shares16 left outputs but forms48 right outputs, multiplying the same geometric table/phase work again. The ns2 charge route has only4 output pairs. Parent GEMM savings survive; repeated gather/phase fusions consume them. The prepared-transport candidate moves the same service-owned row maps and endpoint phases outside these spin scans, retaining the original per-block accumulation order. Its added per-rank payload is O(K*(M_A/local_x+R_tile/local_y)) for phases/maps, independent of nb and output spin-pair count; no full spin-operator carrier or new collective is introduced. It deletes the duplicate geometric preparation from each scalar transport call and retains a single service owner. A same-source four-file zeta comparison and the existing literal parent-unfold/orbit-tile tests gate it.

### Numerical cause of the fresh P/F zeta differences

99 compares canonical logical matrix tiles without gathering a quadratic operator to rank0. Charge C differs by5.855e-18 absolute /1.369e-15 normalized (relative Frobenius2.839e-15), while its distributed pseudoinverse C+ differs by404.445 absolute /6.969e-7 normalized. Replacing only the parent's charge factor with the fixed-main C+ in97 reduces the zeta difference from3.563e-4 /6.340e-7 normalized to3.115e-8 /5.543e-11. Thus the dominant charge difference comes from amplified normal-equation construction/eigensolve regrouping in C+, not from a wrong parent RHS transport. The counterfactual does not pretend to separate the tiny C construction roundoff from the eigensolver's response to it; it isolates the factor from the RHS and its application.

Current normal matrices differ by≈1.1e-17 absolute /3.3e-11 normalized; their traces explicitly have signs+,-,+. Fixed-main's unsigned ridge is the registered Gamma2 defect; current parent source has af85d474's correction. Restoring only the legacy positive ridge in96 reduces mu2's P/F zeta difference from8.467e-6 /1.021e-8 to4.859e-9 /5.858e-12. Charge,mu1,mu3 remain bit-identical to the unmodified parent in this ablation. The remaining mu1/mu3 differences are7.412e-9 /1.019e-11 and5.301e-9 /7.177e-12. These are measured roundoff-sized residuals after the known regularization difference is isolated; the obsolete fixed-main ridge is not a correction to adopt.

#### Accepted head batch and rejected table-only ζ experiment

The table-only geometric preparation in runs 100–102 passed 32 CPU oracles and bit-for-bit comparison of all four ζ files, but fresh charge/current transactions were 15.651829/17.324310 s against 15.769318/17.331768 s. This does not establish a useful speedup; the source experiment was deleted. Repeated bulk transports, rather than construction of their small index/phase tables, remain the target.

Run 103 replaces parent-star head contraction with one child batch in the same parent-star row order, through the existing typed unfold owner. Head response is **3.562213 s, 39 compiles, 2.269327 compiler seconds**, versus 9.383337 s, 92 compiles, 6.458740 compiler seconds in 86. Complete screening is **21.105296 s, 160 compiles, 12.416341 compiler seconds**, versus 25.939867 s, 213 compiles, 16.206024 compiler seconds. χ fluctuated upward to 9.670716 s, so the head saving itself is 5.821124 s. Packed Dyson and Gamma completion remain unchanged.

The resident pair of child faces is linear in centroid count: `32 K ns M nb / P` bytes (two complex128 carriers), 55,296,000 bytes at P4 and 14,008,320 bytes at P16; the largest former P4 star pair is 18,432,000 bytes. No quadratic operator is gathered, no symmetry action is duplicated, and no parent-kernel collective is added. The default iterator still streams parent stars for other consumers.

Gate: all 210 rows of each eqp file and all sigCC/sigTT/sigCT rows exact to the printed digit against 86. P16 static/dynamic source gates subsequently passed in104/105. Evidence: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/103_P4_head_batch/gate.log`, `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/103_P4_head_batch/owner_timings.rank0.json`; `[lx] step lx-Xg4-135208-937359-9179 exit 0 in 69 s`. Claim 1321, on branch `perf/bisp-prof-zw-phase3-2026-09-06`, unmerged.

#### Collective census and memory scope

Claim 1328; `[lx] step lx-Xg0-135805-970220-1637 exit 0 in 14 s`. Run107 passes ten explicit face-wing CPU oracles (one provenance/import test excluded). Run106 preserves the failure before that provenance assertion: importing the dipole driver requires the absent CPU FFT library, the already-registered import-time startup defect. No broad CPU pass is claimed.

`tools/hlo/analyze_hlo_dump.py` processed84/85 in107. Four P class modules0520/0553/0602/0647 and four F shape-class modules0455/0471/0503/0531 have **zero explicit XLA collectives**; the existing distributed GEMM service still owns its internal communication. P executable peaks by CC/CT/TC/TT are2,606,656,548 /987,231,284 /987,231,028 /418,140,068 bytes per rank; F2,706,078,096 /957,342,752 /957,342,240 /337,566,592. P static FFT call sites are3/5/5/11 per class, F3 each per single-block shape-class module: static sites must not be confused with dynamic launch counts. F85 stopped after chi/Dyson; these completed modules do not certify a complete F head census.

Charge ζ module0238 in93 and0229 in94 each contain one all-to-all site and one all-gather site in band loading, with no new collective in the geometric tail. Their maximum recorded collective outputs are61,931,520 and745,334,784 bytes respectively at the different tile widths. Native NCCL timings above also include dynamic executions. `six_analysis/phase3_census.json` retains exact module paths, sizes and operation records.

The runtime memory probe after the first charge tile reports BFC peak11.67GB(P) /16.87GB(F), with nvidia-smi24.37/30.29GB; by current-channel completion, process-lifetime BFC peaks remain11.67/16.87GB and nvidia-smi reaches24.67/30.39GB. These are rank0 process-lifetime peaks in instrumented93/94, not isolated current-stage live bytes. The executable peaks quoted above and in the ζ table are separately the XLA static allocation reports.

### Speed protocol items

| Item | Site | Gate and evidence | Before / after |
|---|---|---|---|
| Fuse the two owner-local operator gathers | `symmetry_maps/maps.py:_permute_isdf_operator_axes_local` | `[lx] step lx-Xg4-141044-1051748-9811 exit 0 in 92 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/108_P4_flat_gather`; claim 1329 | All four zeta files bit-identical at rtol=0. Charge warm fit 990.394 -> 701.711 ms (85.972 -> 60.912 us/point); current warm tile 874.449 -> 648.471 ms (56.930 -> 42.218 us/point); current native transpose 545.361 -> 44.911 ms, GEMM 24.266 -> 24.152 ms, FFT 85.384 -> 85.285 ms, NCCL 4.580 -> 4.556 ms. Warm compiles 0; current first tile 2 compiles/1.134 compiler s. HLO traces identify the removed take_along_axis gather transposes, not spin GEMMs or seam writes. No resident source-spin cache or memory-plan increase needed. |
| Skip declared zero bare blocks and pack equal-family axes together | `w_isdf.py:compute_static_photon_response; v_q_bispinor.py:BispinorVqReader.get_tile` | `[lx] step lx-Xg4-141942-1067651-7078 exit 0 in 87 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/109_P4_bare_pack`; claim 1333 | eqp0/eqp1 210 rows and all sectors exact against103. Bare build 2.075932 -> 1.093192 s; 25 -> 10 compiles, 1.596640 -> 0.735713 compiler s. Warm Dyson 0.156549 -> 1.898566 s flags shared-pool interference: whole-stage 38.913642 s is not a clean speedup measurement. No extra rerun under speed protocol. |
| Close the P16 static/dynamic matrix | `head batch; runs80–83 and104–105` | `[lx] step lx-Xg4-140734-1015085-3094 exit 0 in 122 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/105_P16_dynamic_head_batch`; claim 1334 | P/F baselines static screening31.63/39.43 s and dynamic37.19/111.15 s are complete. Head after-change static/dynamic eqp0/eqp1 and sectors exact in104/105. After screening48.56/39.94 s is retained as shared-pool timing, not an uncontended improvement. Dynamic repeats the115-dispatch certified schedule; no new suite. |
| Classify P/F zeta discrepancies and defer stacked solves | `canonical C_q and four zeta files; distributed solve owner` | `[lx] step lx-Xg0-133339-824889-8569 exit 0 in 11 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/99_cpu_normal_comparison`; claim 1335 | Worst absolute-difference q-IBZ rows: charge2, mu1/mu2/mu3 all6. Normalized max differences6.340e-7/1.019e-11/1.021e-8/7.177e-12. Fixed C+ substitution lowers charge to5.543e-11: sensitive Gram/eigensolver regrouping. Legacy ridge substitution lowers mu2 to5.858e-12: registered unsigned-ridge defect in F. Three-solve batching rejected: ideal saving at most72 ms/run (about0.42% of17.33s), with prior batching arithmetic drift; preserve existing solve boundaries. |
| Cache exact parent Gram specializations | `isdf/core.py:_c_q_face_parent` | `[lx] step lx-Xg4-142358-1103266-6657 exit 0 in 52 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/110_P4_cq_cache`; claim 1336 | All four zeta files bit-identical at rtol=0. Exact-repeat charge C_q 1.141 -> 0.060766 s and 1 -> 0 compiles; current repeats1.022/1.105/1.056 -> 0.009489/0.009235/0.024868 s, all0 compiles. First charge1 compile/0.758290 compiler s. Key includes shapes, dtypes, mesh, GEMM, vertices and the authenticated plan (its identity contract is preserved); no new cache registry. No natural-run saving from eliminating repeats that the driver does not perform is claimed. |
| Complete fresh9x9 timing | `same SCALE9x9 inputs; parent111 and fixed-main43` | `[lx] step lx-Xg4-142454-1130317-1157 exit 0 in 63 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/111_P4_nine_fresh`; claim 1338 | Fresh charge P19.063494s (87 compiles/6.176104 compiler s) versus F33.546s. Sequential current fallback P11.1+7.9+8.0=27.0s versus F10.844+7.545+7.624=26.013s; current remains about1s slower. All four parent zeta files completed. Parent runtime BFC peak9.53GB/rank0, nvidia-smi22.58GB. F receipt lx-Xg4-113755-240209-8497,43_F_static_P4_9x9_fresh/attempt1.log. No unmeasured inherent-slowdown claim. |
| Reuse one current C_q executable across vertices | `src/isdf/core.py; services/distrib_la/src/distrib_la/matmul_plan.py` | `[lx] step lx-Xg4-143720-1207560-9131 exit 0 in 65 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/115_P4_nine_vertex_cq`; claim 1350 | 9x9 P4 all four zeta files bit-identical to 111 (rtol=0). Equivalent GemmPlan equality excludes generated callable identities; current vertex operands are dynamic. mu2/mu3 first C_q 1.277/1.319 s (112) -> 0.02147/0.02086 s, compiler 1.06/1.09 s -> zero; mu1 warm 0.02047 s, zero compiles. Current stage 27.0 -> 25.6 s including three ~0.02 s repeat probes, versus F 26.013 s. Peak unchanged 9.53 GB BFC / 22.58 GB nvidia-smi per rank. Charge/current shape classes and authenticated symmetry-plan identity retained; no extra resident tensors. |
| Reuse the resident parent zeta tile graph across current channels | `isdf/core.py:_z_q_face_parent` | `[lx] step lx-Xg4-145507-1285127-6609 exit 0 in 70 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/117_P4_nine_vertex_zq`; claim 1357 | 9x9 P4 all four zeta files bit-identical to111 (rtol=0). Runtime vertex operands and shape-keyed resident psi(r) inputs replace vertex/store identity specializations; callback path retains store identity. mu2/mu3 first Zq build1289/1287 ->294/295 ms; warm278 ->279 ms at6584 points. Current stage25.6 ->24.3 s versus F26.013 s (same repeat probes in115/117). Whole post-zeta compile receipts200/15.41s ->201/14.51s: one more tiny module but less compiler work, not fewer total events. BFC peak stays9.53GB/rank0; device-wide nvidia-smi peak31.28GB in this shared-pool leg is not an isolated process peak. No new resident wavefunction cache, solve route, or symmetry rule; q ordering and all same-source values unchanged. |
| Preserve phase3 evidence | CFS archive | `/global/cfs/cdirs/m4598/jackm/lorrax_evidence/bisp_prof_zw_phase3_2026-09-06/evidence.tar` | 4910 members verified; SHA256 `21b29b9e34014512e1851e401be8b0fb8910ebd9672d2d6837d52199a23a5e8d` |
| Attribute the screening remainder after head and bare fixes | `w_isdf.compute_static_photon_response; owner_timings.rank0.json` | `[lx] step lx-Xg4-143605-1200135-3759 exit 0 in 66 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/113_P4_screen_remainder`; claim 1355 | P4 owner wall19.331980 s includes head repeat0.122126 and Dyson repeat0.154228; production-equivalent19.055625 s versus F17.897025 (87). Exact eqp0/eqp1 and all sectors versus109. Compiler work11.144366 versus9.064710 s, 148 versus138 events. Head first3.680596 s, warm0.122126 s/zero compiles. Gamma2.220386 s/24 events: orbit first0.296138 s, later0.128-0.247 ms/zero compiles; moments first0.444060 s, warm2.011-2.218 ms/zero compiles. Required covariant orbit absent in F. Warm q-IBZ Dyson0.154228 versus full-q0.784884 s. Common residual1.451077 versus1.439338 s includes vector packing, sanity, and cubature setup; nested reads/seams are inside bare0.928065 s and never counted again. |
| Close 9x9 solve and numerical-difference disposition | `distributed LU owner; four fixed_main.log comparisons` | `[lx] step lx-Xg4-145507-1285127-6609 exit 0 in 70 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/117_P4_nine_vertex_zq`; claim 1358 | Representative warm solve P130ms/6584 points at mu204 versus F154ms/8112 at mu200: 19.745 versus18.984 us/point, ratio1.0401 matches (204/200)^2=1.0404. Stacking keeps252 production tile-application Getrs calls per run (3 channels x12 q x7 tiles); it does not remove this measured arithmetic/transport cost and needs at least128941056 extra RHS bytes plus3995136 LU bytes per rank, excluding pivots/temporaries. Retain solve boundary. Four9x9 normalized max P/F differences C/T1/T2/T3=3.780045e-8/1.001386e-11/1.690066e-8/1.213026e-11; max absolute2.387533e-5/5.891690e-9/9.786190e-6/6.361486e-9, worst q-IBZ rows0/11/10/11. All same-source comparisons are bit-exact. Counterfactual causal isolation is measured on6x6 (96/97/99), not repeated on9x9; the9x9 pattern is consistent with factor sensitivity and the known fixed-main unsigned Gamma2 ridge defect. |
| Complete P16 screening attribution after the accepted fixes | `screening owners; P16 restart static` | `[lx] step lx-Xg4-150412-1206973-2332 exit 0 in 111 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/114_P16_screen_remainder`; claim 1362 | eqp0/eqp1 and all CC/TT/CT rows exact versus104. Owner40.911938 s includes warm head0.202961 and Dyson1.814747; production-equivalent38.894230 versus F38.702014 s (91). Bare0.941037/10 compiles; chi24.839368/51; Dyson first4.142622/6; head first4.250661/39; Gamma3.178536/24; residual1.542006/18. Total148 events/10.588118 compiler s versus F138/8.516197. Head warm zero compiles. Warm Dyson was0.538368 in90, now1.814747: runtime variability limits any steady-state P16 speedup attribution; no inherent-slowdown assertion and no extra rerun. P16 static/dynamic P/F matrix80-83 and identity104-105 remain complete. |

### Follow-up screening attribution

`113_P4_screen_remainder`: `[lx] step lx-Xg4-143605-1200135-3759 exit 0 in 66 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/113_P4_screen_remainder/owner_timings.rank0.json`.

`87_F4_screen_host`: `[lx] step lx-Xg4-132011-760415-3756 exit 0 in 67 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/87_F4_screen_host/owner_timings.rank0.json`.

`114_P16_screen_remainder`: `[lx] step lx-Xg4-150412-1206973-2332 exit 0 in 111 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/114_P16_screen_remainder/owner_timings.rank0.json`.

`91_F16_screen_host`: `[lx] step lx-Xg4-132011-760413-4258 exit 0 in 102 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/91_F16_screen_host/owner_timings.rank0.json`.

| Owner | P4 parent wall / events / compiler s | P4 fixed-main | P16 parent | P16 fixed-main |
|---|---:|---:|---:|---:|
| Bare reads + packed assembly | 0.928065 / 10 / 0.688079 | 0.967658 / 17 / 0.598505 | 0.941037 / 10 / 0.632634 | 0.916616 / 17 / 0.574607 |
| χ construction | 8.911002 / 51 / 5.488656 | 8.471081 / 44 / 4.100452 | 24.839368 / 51 / 4.921662 | 26.975558 / 44 / 3.630688 |
| Dyson first | 1.864499 / 6 / 0.396475 | 2.989131 / 10 / 0.668727 | 4.142622 / 6 / 0.337793 | 5.072333 / 10 / 0.673350 |
| Dyson warm probe | 0.154228 / 0 / 0.000000 | 0.784884 / 0 / 0.000000 | 1.814747 / 0 / 0.000000 | 2.701918 / 0 / 0.000000 |
| Head first | 3.680596 / 39 / 2.184208 | 2.671261 / 28 / 1.599230 | 4.250661 / 39 / 2.317334 | 2.849028 / 28 / 1.552762 |
| Γ completion | 2.220386 / 24 / 1.246123 | 1.358555 / 21 / 0.977146 | 3.178536 / 24 / 1.275353 | 1.419097 / 21 / 0.981356 |
| Other owner work | 1.451077 / 18 / 1.140824 | 1.439338 / 18 / 1.120649 | 1.542006 / 18 / 1.103342 | 1.469382 / 18 / 1.103435 |
| Whole owner, probes removed | 19.055625 / 148 / 11.144366 | 17.897025 / 138 / 9.064710 | 38.894230 / 148 / 10.588118 | 38.702014 / 138 / 8.516197 |
| Nested bare reads/seams | 0.710160 / 7 / 0.533727 | 0.554597 / 12 / 0.299020 | 0.712917 / 7 / 0.478928 | 0.508380 / 12 / 0.290433 |
| Nested Γ orbit | 0.297645 / 1 / 0.196613 | 0.000000 / 0 / 0.000000 | 0.311358 / 1 / 0.197575 | 0.000000 / 0 / 0.000000 |
| Nested Γ moment kernels | 0.448288 / 1 / 0.333643 | 0.446360 / 1 / 0.332154 | 0.513346 / 1 / 0.357973 | 0.456693 / 1 / 0.336942 |

Only direct owners plus residual sum to the whole stage; the final three rows are nested and must not be added again. Compiler seconds are measured work, not an additional wall-time term. Diagnostic head/Dyson repeats are removed from the whole-owner row.

Head warm probes: 113_P4_screen_remainder: 0.122126 s, 114_P16_screen_remainder: 0.202961 s.

The extra P4 compiler work is 2.080 s while the production-equivalent wall excess is 1.159 s. The fused family χ classes cost more to compile than fixed-main's single-block modules but retain the measured warm sweep advantage (2.724→1.092 s, fixed-main2.619 s). Head construction now compiles39 modules once and repeats in0.122 s; the former per-parent-star compile loop is gone. Γ's required covariant factor orbit is absent from fixed-main: its first call costs0.296 s, subsequent calls0.128–0.247 ms, with only one compiled orbit module. The three moment rules likewise share one module. Thus the entire Γ difference is not a parent/full-k comparison of the same algorithm. Residual vector packing, sanity and cubature setup costs about1.45 s on both routes, including about1.1 s compiler work. Nested reads include seam conversion; no second seam/restore bill is added. Full-q block views/restores occur after the screening owner returns and remain on the Σ side of the accounting boundary.

The P16 after-change warm Dyson differs by3.37× from the earlier parent receipt; retain the row as an observed shared-campaign timing, not an uncontended source regression or speedup. No extra run was made under the speed protocol. The P4/P16 head pair storage remains55,296,000/14,008,320 bytes per rank, scaling as32 K ns M nb/P. The two new ζ cache changes add no wavefunction or quadratic carrier and preserve typed-plan identity, distributed ownership and canonical files.

Disposition: adopt `61753655` and `569199bc` in addition to the earlier `04aea1c3`, `d5da53d5`, `71a753c6`, `5b172eb0`, all on `perf/bisp-prof-zw-phase3-2026-09-06` (unmerged). Current9×9 fitting moved27.0→25.6→24.3 s versus fixed-main26.013 s; all four same-source ζ files are bit-identical. Screening attribution and the P16 static/dynamic matrix are complete. Solve batching is rejected by its unchanged provider work and added residency, and the fresh P/F differences are quantified with the6×6 counterfactual classification kept distinct from9×9 inference. All backlog items are dispositioned; no suite or gate0 was run.

Follow-up evidence: `/global/cfs/cdirs/m4598/jackm/lorrax_evidence/bisp_prof_zw_phase3_followup_2026-09-06/evidence.tar`; 371 members verified, 1293455360 bytes, SHA256 `211335bfb80762958c65a4e47bb8cb539db550cb5383be03bcc070c92d1a430b`; source snapshot `8d3c92260dfcbce7a7156df974794c3577e4fce3`. The final report and run audit are also stored beside the archive.
| Scan ordered Gamma images inside the low-rank update owner | `head_correction.complete_static_slab_photon_q0; photon_layout.add_photon_q0_low_rank` | `[lx] step lx-Xg4-152905-1415820-6044 exit 0 in 82 s`; `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/118_P4_gamma_orbit_scan`; claim 1367 | P4 eqp0/eqp1 and all sectors exact versus113. Gamma2.220386 ->1.862763 s, 24 ->22 compiles; compiler1.246123 ->1.281437 s. Ten ordered24-image scans replace240 host updates and barriers without changing addition order or the existing orbit-factor residency. First scan0.211292 s/one compile; warm0.00418-0.00432 s/zero compiles. Certificate already once,27.8 us; cubature already shared. Next remaining boundary is eager small-field fold indexing/scattering. |

| Scan four Γ coordinate folds in one compiled body | `head_correction._fold_photon_q0_response` | `[lx] step lx-Xg4-153348-1514559-1957 exit 0 in 69 s`; `runs/MoS2/42_bisp_scale_2026-09-06/prof_zw/123_P4_gamma_fold_scan/{gate.log,sector_gate.txt,owner_timings.rank0.json}` | eqp0/eqp1 and sectors exact versus118; Γ1.862763→1.650514 s,22→15 compile events, compiler1.281437→1.152133 s; source9544ed5c. |

Final disposition: gated changes are pushed on `perf/bisp-prof-zw-phase3-2026-09-06`, unmerged; current Cq/tile reuse and both Γ scans are ready for integration.
Measured effect: current9×9 fit27.0→24.3 s (F26.013); Γ6×6P4 completion2.220386→1.862763→1.650514 s and24→22→15 compiles; final gate123 eqp0/eqp1 and all sectors exact.
Left: Γ remains0.291959 s above F (covariant orbit absent in F); prior head cold3.680596 vs2.671261 s but warm0.122126 s; bare0.928065 vs0.967658 s; latest Γ P16 and missing9×9 unit matrix unrun,124 cancelled rc143.
Ordered remaining source cherry-picks: `61753655`, `569199bc`, `7dfacf64`, `9544ed5c`; earlier04aea1c3/d5da53d5/71a753c6/5b172eb0 are represented in orchestrator d4fc26eb by bfb461d9/7c5466be/7bfdb3d5/4e263a8d; preserve its newer26f40be5 coupled-LU implementation when resolving current-fit overlap.
Do NOT take profiling wrappers, cancelled119/124 or unrun120–122 as evidence, unapplied architecture drafts, or older solve-batching rejection as a verdict on newer26f40be5; keep the sign-aware ridge; existing-unit table is `runs/DEV/116_bisp_scale_codex_2026-09-06/ZW_ROWS.md`.
