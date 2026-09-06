# BISP-PROF-ZW — measured compilation and coupled-current fixes

Measured disposition on branch `perf/bisp-prof-zw-2026-09-06`, **unmerged**. Pinned before-change P=`9f569c4bf75bad40e4f5895946874b4c503e4410`; fixed-main F=`e1559a071e244b4f049c924781b668d9e1560739`. Adoptable pushed fixes: restore executable reuse `87a2bfaa`, chi vertex shape classes `0f8fbc3e`, coupled-current tail sharing `f811f734`. MoS2 full-static screening31.21→18.59s; total201.36→146.56s; EQP0/1 and every CC/TT/CT row exact. Coupled-current tail sharing now passes its production fresh-fit and downstream identity gate; its warm tile is39.31ms versus51.00ms before.

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
