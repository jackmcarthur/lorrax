| Unit | Charge PERF2 P/F; Na P16 claim842 | Bispinor 6×6 P4 P/F | Bispinor 6×6 P16 P/F | Arithmetic / logical bytes per rank | Collectives and compilation; mapping verdict |
|---|---|---|---|---|---|
| ζ charge tile | 0.516 native; 0.846 fit-loop wall | 1.616 observed time/point, unequal tiles | Native capture exists; timing excluded for overlap | Parent projectors leading 16 Q ns² M R nb/P real FLOPs; full-K transform tails remain; output16 K M R/P bytes | P4 P/F: five AG and five A2A/tile each; AR5/80. Not the same workload or tile width; no matched-tile speedup claim. |
| ζ current tile | No separate current unit | Coupled P versus three F channels: 0.967 observed time/point | Both routes coupled at P16; timing excluded | Same projector law per channel; P couples three channels, F P4 repeats the full-face tail three times | P4 coupled P5 AR versus F3×80 AR; per-tile counts differ from full-fit counts. Bispinor-specific sharing, not a scalar one-to-one row. |
| χ0 build | 0.764 stage; 0.145 screening including W | CC warm432.659/426.835 =1.014 | CC warm2243.668/2444.546 =0.918 | 22 distributed GEMMs/block at the 11-node static screening rule; full-K work, Θ(K nb ns² M_A M_B/P) per quadrature unit | P4 3168/P16 6336 broadcasts per block on BOTH routes. No Q/K reduction here; inherent full-child summation. |
| W restore | Four A2A on both charge routes; Na not isolated | Not separately isolated from restore/FFT | Same limitation | Full-K operator16 K M_A M_B/P bytes; transform Θ(K log K M_A M_B/P) | Do not count the charge route's four A2A as removed. Native restore modules below have no NCCL; transform/layout collectives belong to their surrounding modules. |
| τ node | 0.346 native; Na0.0717 host | 32.502/55.089 =0.590 native;33.454/56.034 host | 67.816/228.036 =0.297 native | P has TWO G GEMMs:16 Q d_A d_B nb/P; F8 K d_A d_B nb/P. Add projection and full-K convolution | P4 Broadcast112/432; P16 224/864. Integrated parent has four distributed GEMMs, F three. NOT the historic charge operation count. |
| Static block | No like-for-like charge ratio | CC0.267, CT0.249, TC0.258, TT0.257 native | CC0.263, CT0.209, TC0.310, TT0.282 | G8 K d_A d_B nb/P in BOTH; projection uses q=Q (parent), K (full) | P4 Broadcast200/720; P16 400/1440. F includes two extra head-diagnostic projections; ratios are not pure parent-route gains. |
| Band projection | 0.125 native; Na not isolated | (5.565+0.935)/(28.839+4.201)=0.197 | Q/K arithmetic class; constituent native thunks retained in trace, no isolated ratio asserted | 8 q (b d_A d_B+b² d_B)/P; q=Q or K; b30/P4,32/P16 | Two SUMMA calls:4 sqrt(P) q broadcasts; maps to charge Q/K reduction. |
| Antiunitary band unfold | Not isolated | Zero per-τ CP after integration71 | Zero per-τ CP | Final band tensor16 K b²/P; old per-node band exchange129600 bytes at P4 | Old P4: two CP/node, G161280000 bytes/exchanging rank plus band129600. New: second G GEMM and delayed final band unfold. This is a trade-off, not free deletion. |
| Seam conversions / operator spin transport | 139 eager→10 compiled modules; same38 A2A | Full-K convolution ratio0.997; parent spin/seam region14.360ms/node, not isolated rotations | Full-K tails persist | Two local spin rotations16 K ns³ M_A M_B/P FLOPs; full G16 K ns² M_A M_B/P bytes | Local rotations have no NCCL; ns=1 scalar cost differs. P4 CC rotations3.318 GFLOP/rank. Conversion compile events are priced in the module census. |
| Ordinary charge restores | Retired bridges enumerated by PERF2 | Parent typed restores remain; no second removal credited | Same | Θ(K M²/P) output traffic independent of nb | Native P4 small band restore ~0.05ms/call, no NCCL. Restore code is not the antiunitary G exchange. |
| Lorentz restores + mixing | Bispinor-only | 300 source restores/static run; TT ~0.012ms native/restore; host output-restoration aggregate median21.4ms | Same source multiplicity, no isolated timing asserted | Three terms ×100 contributions: output traffic3×16 K(M_C+9M_T)²/P bytes; mixing Θ(K M²/P) | CC/CT/TC/TT counts3/27/27/243, no NCCL inside restore kernels. Launch floor stays300; increasing K alone does not remove asymptotic cost. |
| Per-family faces / current channels | Bispinor-only | Four shape families; 16 ordered Lorentz blocks | P16 has greater carrier padding | Face storage Θ(Q nb ns M/sqrt(P)); current factor3, ordered vertex factor16; all typed actions from SymMaps | Shape families cause distinct modules; current coupling cuts repeat work. No new shape-identical kernel cache claimed beyond measured compile experiment. |
| Packed Dyson | Bispinor-only | Warm LU156.046/797.148=0.196; complete host167.376/820.453=0.204 | Warm LU513.159/2567.270=0.200 | L=M_C+3M_T; A+LU+2TRSM leading(56/3)qL³/P FLOPs; stored matrix16qL²/P bytes | P16 warm LU P/F: AR35/180, Broadcast259/1332, AG140/720, SendRecv144/738. Pivot traffic is data-dependent. Q/K reduction with padding. |
| Γ completion | Bispinor-only covariance correction | ~0.02ms/call,10 calls, zero NCCL | Not independently timed | Rank4 ×12 typed operations; Θ(n_sym rank L²/P), independent of nb, one-off | One P-only compiled module. Fixed-main lacks this physics fix; cannot erase it to force parity. |
| Head attribution | Diagnostic, not a charge gain | Requested false; P off, F erroneously active | Same source discrepancy | Two extra F band-projection GEMMs/static block | F photon_sigma.py:380–423 ignores false when q0 factors exist. Counted in actual static native traffic; registered source defect. |

BISP-SCALE, branch `perf/bisp-scale-2026-09-06`, **unmerged**. Here K=nk=36, Q=n_parent=7, nb=80 GW bands, ns=4 physical components, d_A=ns M_A, p=sqrt(P); all matrix bytes assume complex128. QE has82 bands. All numerical ratios above are diagnostic because the required exact P/F science gate fails. Native timings are rank-0 GPU kernel spans, not whole-stage walls; host rows synchronize their timed work. FLOPs are leading analytic real arithmetic, not hardware counters. Logical operand bytes are not measured vendor wire bytes; Nsight's kernel census does not expose exact cuBLASMp/cuSOLVERMp wire payloads. This report does not invent those missing bytes.

Charge references are harvested from sandbox `reports/psi_irr_perf2_2026-09-05/report.md` (job57941637) and CLAIMS842; the separate claims/0842.md file is absent. PERF2 used cuda_async@0.85, this lane BFC@0.85. Charge reference walls/ratios and Na screening include different work and are not substitutes for a bispinor exact gate.

## Owner verdict

**No: the bispinor route does not map one-to-one onto all historic charge improvements.** At36 k/7 parents and16 GPUs, the clean baseline dynamic parent run is127.64s versus142.84s full-k (10.6% lower wall,1.119× speedup); the exact-to-parent direct-zero experiment reaches118.79s (16.8% lower,1.202×). At4 GPUs the corresponding cold parent remains slower, and even81 k/12 parents does not demonstrate a baseline cold-run crossover (70.10 versus69.03s). The warm kernels do benefit strongly, especially parent projection and packed Dyson, but this deck has not made the entire P4 driver kernel-dominated. The integrated antiunitary τ path buys removal of exchanges with a second Green GEMM. Static G and convolution remain full-k; spinor operator transport, three current channels and Lorentz mixing add work absent from a scalar comparison. P/F residuals reach0.895µeV, so **none of these is a certified equivalent-science speedup**.

## Sources, deck and evidence contract

Started at S4a691b67; cherry-picked ZW0f8fbc3e/f811f734 as cee4011d/3046b911, retaining only the conflicting absent ZW report. After fetching ORCH, measurements switched to integrated af85d474 and then **71ae0bde** for the clean matrices and native static/χ/τ profiles. Final source verification uses integration**18196944** via merge2d427434 plus compile change**902d01ca**. Fixed-main **e1559a07** remained read-only. No performance comparison silently mixes those source states. The integration includes covariant Γ completion and the signed paired-Gram ridge change absent from F; their individual contributions to the residual have not been isolated.

All run paths below are relative to `runs/MoS2/42_bisp_scale_2026-09-06/` (R); analysis paths D mean `runs/DEV/116_bisp_scale_codex_2026-09-06/`. Every arm has its deck, manifest, source receipt and runner. QE6×6 is R00_qe_6x6, with full12-operation group and7 stored IBZ points, claim1214, step57982945/lx-Xg0-105033-2302113-3374 exit0. GW reports `Spatial group : 12 operations`, seven stored parents and `Centroid orbit: CLOSED`. The common basis has logical charge597/current200, selected as a whole orbit to satisfy fixed-main P16 divisibility; the original logical194 current deck refuses F at P16. P4 physical carriers are P600/204 and F600/200; P16 P608/224 and F608/208. Different padding is included in all reported arithmetic.

Both modes use full head, distributed linalg, low_mem_bands=true and copied identical ζ files, restart=false. Dynamic uses the same115 certified quadrature nodes, eps1e-5/reduction_steps0, D03_common_rules replay; containment/error/noise checks remain enforced. F lacks the reduction_steps key, so its deck omits that unsupported spelling. Source profiles use rank0 Nsight/HLO only and exact warm-repeat checks. Full matrix and profile arms all compare EQP0/EQP1 and sigCC/sigTT/sigCT through the sandbox parsers at zero printed tolerance. All prototype/profile versus own-source-baseline gates pass; all P/F gates fail (P4/P16 static max0.895µeV, dynamic EQP1 max0.675µeV, sectors1µeV; 9×9 max0.766µeV).

## Clean stage and compile tables

Compilation is a separately measured receipt and overlaps the stage it compiles; do not add compiler seconds to total run. D/final_P4.json and final_P16.json contain every named stage, not just the following selection.

### P4: master103, job57982945/lx-Xg4-123003-543237-1984 exit0 (655s), claim1294

| Arm | Total s | Screening support s | Sigma other s | Tau sweep s | Compile events | Compiler s |
|---|---:|---:|---:|---:|---:|---:|
|104_P_static_P4_baseline|67.11|26.59|20.15|—|674|38.06|
|105_F_static_P4_baseline|57.13|17.18|20.61|—|578|30.31|
|106_P_dynamic_P4_baseline|96.51|19.30|31.34|7.29|894|57.49|
|107_F_dynamic_P4_baseline|86.87|14.40|29.94|8.93|772|45.21|
|108_P_static_P4_zeros|64.30|26.07|18.09|—|610|35.95|
|109_P_dynamic_P4_zeros|92.84|19.09|28.56|7.15|798|54.28|
|110_P_static_P4_placement|62.35|25.02|17.98|—|589|35.64|
|111_P_dynamic_P4_placement|92.20|18.89|28.45|6.77|777|53.93|

### P16: master120, job57988457/lx-Xg4-124919-641781-9910 exit0 (927s), claim1296

| Arm | Total s | Screening support s | Sigma other s | Tau sweep s | Compile events | Compiler s |
|---|---:|---:|---:|---:|---:|---:|
|90_P_static_P16_baseline|93.61|48.18|22.57|—|646|33.77|
|91_F_static_P16_baseline|94.37|38.71|32.67|—|604|28.78|
|92_P_dynamic_P16_baseline|127.64|38.89|33.87|11.63|897|52.34|
|93_F_dynamic_P16_baseline|142.84|33.64|42.11|27.35|825|43.34|
|94_P_static_P16_zeros|88.22|45.49|20.77|—|582|31.83|
|95_P_dynamic_P16_zeros|118.79|35.91|30.58|9.9|801|49.31|
|99_P_static_P16_placement|87.35|45.58|20.38|—|561|31.44|
|100_P_dynamic_P16_placement|123.46|38.19|30.73|11.03|780|49.07|

Surface-placement follow-up121/122 (job57988457/lx-Xg4-131335-724532-5279 exit0, clean, claim1303) gives static62.11s/584 events/35.53 compiler seconds and dynamic91.69s/772 events/53.86s. Fixed P4 has578/772. Nine-grid124/125 (job57988457/lx-Xg4-131617-747058-1237 exit0, clean, claim1304) gives P/F70.10/69.03s; screening29.14/21.89s; Sigma21.40/23.60s; compiles641/606 and compiler37.47/31.56s. Baseline9×9 remains compiler-heavy. The later final-code9×9 arm132 is67.83s/556 events, exact against124; it is a source gate, not a contemporaneous clean P/F pair establishing a crossover.

## Native unit receipts

The P4 profile matrix72–75 is job57982945/lx-Xg4-130807-715734-2513 exit0 (926s); P16 matrix66–69 is57988457/lx-Xg4-132155-757259-3784 exit0 (1478s). Both have zero overlapping GPU steps in D/gpu_overlap_audit.json. Each profile matches its own baseline exactly. Inspect each run's `boundary.jsonl`, `compile_modules.jsonl`, `census.json`, `native_instances.json`, `stats_nvtx_gpu_proj_sum.csv`, `stats_nvtx_kern_sum.csv` and `xla_dump_rank0/hlo_summary.json`. The recovered PERF2 collective census omits CP, so the HLO and native SendRecv census supplement it.

| Native unit, ms | P4 P | P4 F | P16 P | P16 F |
|---|---:|---:|---:|---:|
| Static CC block median |28.485|106.774|143.982|546.886|
| Static CT block median |13.926|55.959|73.547|352.474|
| Static TC block median |13.660|53.024|129.869|418.577|
| Static TT block median |7.773|30.253|61.835|219.259|
| χ CC warm repeat |432.659|426.835|2243.668|2444.546|
| τ node median,115 calls |32.502|55.089|67.816|228.036|
| Packed static Dyson LU, last/warm call |156.046|797.148|513.159|2567.270|

The four static families execute16 blocks per term and three terms. Fixed-main adds two head-attribution projection calls; its actual native broadcast count therefore exceeds the three-GEMM body-only model in the analytic JSON. P4 F also emits three SendRecv/block; P16 F three AG/block, reflecting the padded band boundary. They are not the old antiunitary CP pair. χ rows contain22 distributed GEMMs for the11-node screening quadrature; equal full-K broadcast counts disprove a claimed universal Q/K work reduction.

Source71ae0bde `greens_function_kernel.py:176–179` builds G and the conjugated-face G transpose separately. P4 optimized τ module1943 has four cuBLASMp calls. The extra G executes6.4512GFLOP/rank and produces161280000bytes/rank, adding28 broadcasts. Earlier af85 module2166 had84 Broadcast+2 SendRecv/node; now112 Broadcast+0 CP. Two G native thunks are1.984+1.965ms versus F10.067ms; projection5.565+0.935 versus28.839+4.201ms; full-K convolution6.781 versus6.801ms. Compiler peak1,787,348,337B new versus1,787,452,481B old is almost unchanged. Claim1307 records the discrepancy with the integration commit's “sole Green GEMM” wording. This is a generic antiunitary compute-for-exchange choice, not an inevitable bispinor multiplicity, and no unauthorized third source experiment was introduced.

Packed Dyson at P16 stores P1280/F1232 transverse-plus-charge axes. The warm native LU peaks are45,875,200/218,566,656B. Construction A has two HLO all-gathers per q chunk; each logical gathered result is16 q_chunk L²/sqrt(P) bytes/rank, with new received fraction1−1/sqrt(P). This existing panel replication has been registered at w_isdf.py:846–847; it does not satisfy a stronger all-intermediates-O(L²/P) invariant. Vendor pivot SendRecv counts are measured, not inferred by multiplying Q/K.

## Fresh ζ, kept separate from copied-ζ GW walls

Fresh fits intentionally stop successfully after fit_zeta and produce all four ζ HDF5 files; no EQP is claimed for those fit-only arms. P4 master117 (113P/114F), job57982945/lx-Xg4-132429-779844-9647 exit0, and actual F tile-boundary capture128/master129, job57982945/lx-Xg4-133148-815088-6247 exit0, pass the GPU overlap audit. The fresh-fit profile budget is15GB rather than the copied-run30GB; automatic tile plans differ. Nsight and extra warm-repeat calls contaminate whole-fit wall comparisons by design.

| P4 fresh unit | Tile points | Warm native ms | Compiler peak bytes | Native collective count/tile |
|---|---:|---:|---:|---|
| P charge |5124|465.897|4,948,420,284|5AG+5SendRecv(A2A)+5AR|
| F charge full tile |12022|676.200|7,441,419,104|5AG+5SendRecv(A2A)+80AR|
| P coupled current |6584|435.490|3,403,297,204|5AG+5SendRecv(A2A)+5AR|
| F each sequential current full tile |9390|~213.88|4,143,000,952|5AG+5SendRecv(A2A)+80AR|

P current warm host436.649ms versus F three times214.576ms; after dividing by tile points, the observed throughput ratio is0.967. Charge time/point ratio is1.616. Neither is a matched-tile intrinsic kernel comparison. F tails are charge10014 points and current8520, with separate compiled modules. P coupled T emits seven tiles; F three sequential channels each emit five. The P4 coupled transverse stage is33.7s under capture/repeats; its nested μ1/2/3 stages overlap and must not be summed. The initial unprofiled old-basis fresh P fit reports charge15.68s/T17.38s; F charge13.76s and transverse inside setup23.84s. Those old logical194/codeaf85 values are not substituted for common-basis P16 measurements.

Fresh P16 master118 (115P/116F), job57988457/lx-Xg4-140325-924900-4910 exit0, overlaps another GPU step. Both fits complete, but its times are excluded. P uses coupled T tiles23040 points, F coupled T39016+7064; F charge46080 points. Unlike P4, F chooses coupling here, so the P4 three-sequential-channel count cannot be extrapolated to P16. A repeat is recorded separately in master139; its final audit is reported in the supplement below.

## Experiments and source disposition

**Restore multiplicity rejected.** Run-local resident prototype18 restores16 source blocks per term (48/run) while leaving100 mixes per term. It refuses a resident carrier above5% of existing memory_per_device_gb. In clean combined step57982945/lx-Xg4-110752-42766-3793 exit0, baseline17 takes65.24s total/20.08s Sigma; resident18 takes65.71/19.84s, both674 compile events, exact EQP0/EQP1/sectors. The standalone ordering was reversed, so the total-time win is not repeatable. Resident carrier measured203,233,536B/rank at original P4 C600/T196. Formula16 K(M_C+3M_T)²/P bytes per retained term is O(KM²/P), independent of nb/ns/Q, on top of parent storage O(QM²/P). Increasing nb or ns makes it relatively less important against G; increasing K alone does not. No production residency change is retained; no P16 resident experiment is justified after the required P4 retention criterion fails.

**Compile experiment retained, parity only partial.** Commit902d01ca replaces fresh jitted zero lambdas with directly sharded allocation, places already-host typed symmetry tables directly, slices their host rows before placement, and directly places absent surface/row tables. It adds no cache/decorator/API/env/deck key and no host gather of distributed physics arrays. The named module census is `evidence/final_compile_module_delta.csv`; raw per-module digests/compiler seconds are in profile compile_modules.jsonl. Direct zeros remove64 static/96 dynamic events; host-table placement removes21 more; surface placement removes five more. Shape-family differences remain, including typed unfold, padding, Γ covariance and head child slicing. The aggregate six-event static P4 excess is not uniquely attributable to six redundant modules; profile instrumentation also changes counts, so its module totals are not silently substituted for plain-driver receipts.

Final source gates at integration18196944+902d01ca are exact to parent baselines for all five arms130–135. P4 static584 versus F578 still misses the requested target; P4 dynamic772=772; P16 static556<604 and dynamic775<825; 9×9 static556<606. D/final_source_gates.json records all stages. P16 master136 job57988457/lx-Xg4-141411-1064403-7247 exits0. P4 master133 job57988457/lx-Xg4-135842-973036-7653 completes all three science arms then fails default pytest collection. CPU core D08 selects76 cells:66 passed,6 skipped,4 failures for absent host FFI; targeted D09 yields11 passed,1 skipped,1 same-environment import failure. A full core pass is not claimed. Claim1331 records these limits alongside the source identity gates.

## Audit and remaining acceptance gaps

Slurm accounting disproved several earlier full-GPU timings. Claim1270 supersedes timing interpretations of claim1255 and the old9×9/P16 arms; historical tables are retained in `historical_measurements.md`, not used above. The clean matrices103/120, surface127,9×9 master126 and profile76/70 intervals are uncontended. Dedicated allocation57988457 was obtained on attempt5 after four QOS refusals; no extra unauthorized pool was used. Other campaign steps subsequently entered that pool, so every follow-up is independently audited. CPU-only validation ran during final source gates; those walls are not the clean comparative matrix.

The owner question has a measured negative answer to universal one-to-one mapping. Acceptance remains incomplete: exact P/F identity fails; static P4 compile parity is six events short; a kernel-dominated cold P4 driver has not been demonstrated at6×6 or9×9; clean P16 fresh-ζ timing is not yet available; exact vendor wire-byte accounting is unavailable from the captured kernel census; CPU/default-core environment gates are not green. These are explicit limits, not assumed passes or zero-cost cells. Source defects outside the two experiments are registered in KNOWN_LORRAX_ISSUES.md; launcher/test-scaffolding problems in KNOWN_SANDBOX_ERRORS.md. No control physics was changed to manufacture identity.

## Final fresh-fit audit and canonical H5 gate

Repeat master139, job57988457/lx-Xg4-141942-1078641-7260 exit0 in253s, also overlapped other full-GPU steps (57988457.79/.80); the name `clean` is an intended experiment label, not its audit verdict. All four fit files exist on both sources. Timing is excluded for both P16 fresh captures; optimized HLO/native counts remain available in137/138. No third repeat is called a measurement without an uncontended interval.

Canonical `tools/compare_zeta_h5.py` compared all seven q rows of113P versus128F in D11_zeta_identity_fixed, job57988457/lx-Xg0-142104-1104429-4025 exit0. Charge has8,207,556 complex values and each current2,749,600; no nonfinite values. Normalized max residuals are charge6.34047e-7, μ1=1.09546e-11, μ2=1.78433e-8, μ3=1.35232e-11. All fail the parser's1e-13 tolerance. This is additional evidence against claiming fresh-fit P/F identity; copied-ζ GW arms deliberately eliminate this input difference. D10 was a failed-path runner attempt and is not evidence for a ζ comparison.

Final native/fresh-fit claim: 1337. The source-change gate remains claim1331; all claims explicitly say branch unmerged. The final report supersedes the historical in-progress narrative.

P16 fresh-fit count supplement (repeat137/138; timings excluded): P charge module163 and coupled current437 each emit5AR+5AG+5SendRecv per tile. F charge266 emits80AR+5AG+5SendRecv; coupled current606/680 emits5AR+5AG+5SendRecv. Thus the coupled-current collective count maps one-to-one at P16 even though it differs radically from sequential F at P4. Parent221 versus F296 compile events describe these instrumented fit-only captures, not whole-driver compile parity. Native instance receipts and HLO summaries exist for both repeat arms.
