# PSIIRR-PERF2 — one-order centroid profiling

Completed measurements, 2026-09-05. On branch `perf/psi-irr-packed-order-profiling-2026-09-05`, unmerged. **Retain cached compiled I/O conversions:** Si P4 walls101.12/98.17s before,92.36/90.72s after;638→509 XLA compilations. Both EQP files and saved zeta are exact to leg20. Claims831/833. No zeta-arithmetic or tau-kernel speedup is claimed.

Evidence root below is `runs/Si/99_psi_irr_zeta_2026-09-05/perf2/`. All launches used `lx run --jid 57941637 --wait 900`, P4 one rank/GPU, cuda_async@0.85; CPU14 explicitly used `-G 0 -n 1`. No forbidden pool, sbatch, source-control mutation, or main push was used. A=7e1ae83d in this lane's worktree; B=read-only ff636b64; C=read-only0bcabe7c with the supplied receipt patch. `source.txt`, `source.diff`, decks and cache checksums accompany the arms.

## Objective and preregistered candidates

Repeat the previous PERF stage, tile, device and ablation comparisons for A/B/C before source changes. Count communication at its actual execution frequency and distinguish arithmetic, communication and host compilation. Test only measured candidates at their existing owner: I/O conversion compilation/fusion, remaining gathers, or small projector GEMMs. No new physics, vendor FFI, deck key, environment dial or replicated large carrier.

The exact leg20 deck is the full-run template. The auxiliary tile deck adds the previous3456 r-chunk key: A/B produce five2768-slot orbit tiles; C produces four3456-point tiles. These are the same requested chunks, not equal amounts of work per invocation. Original full-deck arms have two tiles; C compiles a distinct remainder shape. The source snapshot and all A/B/C captures were completed before editing `centroid_basis.py`.

**Matched quadrature needs actual node receipts.** Every initial arm copied all twelve leg20 rule-cache files. A01/C07 selected379nodes; B02 selected378 because its slightly different spectral box admitted a smaller certificate. New B09/B10 replay A's eleven selected rules by nearest recorded window (distance checked below1e-4 Ry), through the **unchanged containment, error and noise guards**. All379 node counts and all eleven node digests match A01 and C07, including weights from the same immutable files. C11 uses that replay too. No certificate was enlarged or relaxed. The directories retain the complete original cache and the eleven single-rule lookup directories. A04/A16 and candidates13/17 select the same379 schedule normally.

C03 refused the newer `sigma_quadrature_reduction_steps` planning key before science. New C07/C11 omit only that unsupported key; the cached rules make it inactive. Counter-based replay attempts05/06/08 failed because planning is partitioned across ranks; failed artifacts remain. Missing `claims/0826.md` and these harness issues are recorded in KNOWN_SANDBOX_ERRORS. The existing CLAIMS826 row and the originating round4 report supplied the missing reference context.

## Before any source change

Host entries below use unprofiled04/10/11; device entries use rank0 Nsight01/09/07. Those files carry separate scopes. Initial A01/B02 were placed on nodes with existing CPU steps, so their host walls are not baseline timing evidence; B09 replaces B02 for matched device comparison. Profiled host walls are excluded from speedup selection.

| before-change quantity | A ONE-ORDER | B round3 | C main |
|---|---:|---:|---:|
| warm Z host ms (five/four-tile arm) |313,228,246,244|333,216,283,262|434,167,165|
| warm solve host ms |102,24,23,25|19,66,18,20|25,25,26|
| first Z compile+execute ms |3291|3929|1880|
| warm whole tiles ms |421,258,276,276|387,289,308,289|468,200,215|
| peak tiled Z HLO GiB |5.62|5.62|3.61|
| tiled arena high-water GB |6.09|6.09|3.96|
| full-deck zeta s |15.72|14.97|15.11|
| chi0 / W s |4.60 /4.67|5.77 /4.79|6.02 /4.80|
| W persist+head s |2.58|0.29|0.29|
| Sigma rule plan s |0.60|0.60|0.59|
| Sigma tau sweep s |17.31|15.45|30.60|
| Sigma other s |31.64|30.74|27.47|
| total run s |101.12|92.51|110.46|
| XLA compilations / compile wall s |638 /54.80|527 /48.47|520 /46.68|
| rank0 tau module median ms (379 calls) |24.1821|24.3262|69.8066|
| tiled Z device projected median ms |87.6082|87.5869|169.7714|
| solve_all_at_once projected median ms |6.3467|7.7531|8.0949|

The compile counts/walls come from the runtime compile-cache receipts. They are cumulative compiler work, not a partition of elapsed wall time. B09/C07 EQP differences from A20 are respectively1.410/5.409 and0.608/2.548micro-eV for eqp0/1; repeated B10/C11 reproduce them. Zeta normalized maxima are9.014e-8 and1.624e-8. This is the known conditioning/basis-order comparison across different algorithms, well inside the2meV gate; it is **not** printed-digit parity. Candidate acceptance below uses tolerance0 and bit-exact zeta, without borrowing this relaxed cross-source comparison.

Full Sigma substage detail, seconds from the production stage table:

| stage | A04 | B10 | C11 |
|---|---:|---:|---:|
| rule planning |0.60|0.60|0.59|
| tau sweep,379 nodes |17.31|15.45|30.60|
| Sigma other |31.64|30.74|27.47|
| chi0 |4.60|5.77|6.02|
| W |4.67|4.79|4.80|
| W persist + head |2.58|0.29|0.29|

Artifacts: `driver_rank0.log`, approved `parse_lorrax_sigma_run.py` output where present, and `census.json` host fields. Same-source repeat16 is reported below because one host sample cannot resolve allocator/compile/scheduling noise.

## Valid rank-0 device captures

Explicit cudaProfilerStart/Stop before runtime finalization yields real kernel records. A01 step `lx-Xg4-125238-1592193-3339` exit0; B09 `lx-Xg4-130420-1672810-1736` exit0; C07 `lx-Xg4-125458-1603341-6744` exit0. Native `stats_nvtx_gpu_proj_sum.csv` and `stats_nvtx_kern_sum.csv` exist for full and tiled runs. `tools/hlo/analyze_hlo_dump.py` produces each rank0 `hlo_summary.json/md`; the lane's registered `tools/profile_collective_census.py` joins program ids, source-function tables and native device rows, supplementing the analyzer's missing async all-reduce-start count. It does not infer FFI communication from HLO.

| operation, tiled rank0 | A01 | B09 | C07 | instrument/scope |
|---|---:|---:|---:|---|
| complete Z module |87.608ms median|87.587ms|169.771ms|projected span, includes cold;5/5/4 calls; modules0210/0227/0297 jit_fn|
| two projector GEMMs, total/tile |5.285ms|5.276ms|52.445ms|sum of projected GEMM ranges divided by tile count; A/B25 calls of each GEMM, C80 each|
| paired IFFTs |3.514 /3.500ms|3.512 /3.499ms|4.356 /4.358ms|median per transform;4 of each per tile, fft19/20|
| final FFT |2.639ms|2.634ms|3.274ms|one/tile, fft14;9 transforms/tile total|
| local tail fusion |1.923ms x4|1.929ms x4|2.388ms x4|loop_add_fusion.1 A/B, .2 C; projected medians; C is different arithmetic|
| old crop all-gather |absent|**absent already**|absent|optimized HLO; B has rank-local prefix-crop modules, not the old replicated crop|
| RHS bridge outside Z |absent|1.911ms median|absent|B0248 jit__kernel,2 all-to-alls/tile, plus local crop dispatches|
| solve_all_at_once |6.347ms|7.753ms|8.095ms|projected module median;0235/0260/0322|
| warm G-flat accumulation/write row |4ms|3–33ms|5ms|host rchunk debug rows; final HDF5 write/unpack is separate below|

A/B Z GEMMs are a small fraction of the87.6ms tile. The FFT/tail still acts on full k, and A/B tile code and5.62GiB peak are identical. Replacing the eight-parent GEMMs is therefore rejected as this lane's priority. C's larger GEMM work is already removed by the parent route; round4 does not remove it again. Neither the old crop-gather defect nor its PERF restore-fusion candidate should be transplanted here: B's prefix-crop implementation already removed that all-gather.

## Saved Sigma device profile

Same379-node full-run captures, rank0 projected medians. Complete module ids: A1472, B1285, C1238, all `jit__tau`.

| region | A | B | C |
|---|---:|---:|---:|
| complete compiled node |24.182ms|24.326ms|69.807ms|
| Green GEMM FFI12 |1.618ms|1.620ms|14.445ms|
| full-k convolution FFI13 |14.379ms|14.381ms|14.236ms|
| first projection GEMM FFI14 |3.783ms|3.750ms|29.505ms|
| second projection GEMM FFI15 |0.816ms|0.986ms|7.336ms|
| parent-row selection gather |0.1796ms|0.1790ms|absent|
| explicit HLO collectives/node |0|0|0|

**Every NCCL family observed inside the tau node:** only `ncclDevKernel_Broadcast_RING_LL`. No SendRecv, AllGather, ReduceScatter or AllReduce kernel appears inside any of the three sampled nodes. Distributed GEMM FFI owns the broadcasts; zero HLO collectives does not mean communication-free.

| enclosing GEMM | broadcasts/node A/B/C | kernel sum ms/node A/B/C | sum over379 nodes A/B/C, s |
|---|---|---|---|
| Green FFI12 |32 /32 /256|1.096 /1.068 /9.229|0.415 /0.405 /3.498|
| projection FFI14 |32 /32 /256|3.166 /3.135 /24.523|1.200 /1.188 /9.294|
| projection FFI15 |32 /32 /256|0.449 /0.619 /4.426|0.170 /0.234 /1.677|
| total |96 /96 /768|4.711 /4.821 /38.178|1.786 /1.827 /14.469|

These are native CUDA kernel-duration sums, not projected spans and not bandwidth measurements: NCCL may wait for another rank and overlap other streams. The total broadcast launches are36384/36384/291072. Do not add these rows to the containing GEMM or tau ranges. The unchanged convolution and the already-small parent GEMMs explain why A and B's node times coincide. No per-node change is justified by round4's bridge removal.

### HLO collective census per stage

Each tuple is **(all-to-all, all-gather, reduce-scatter, all-reduce/psum)**. An async start/done pair counts once. Counts are static instructions unless a loop multiplier or separate module is stated. All-reduce includes reductions other than sums when the HLO uses that primitive. The complete per-module source and line census is in each `census.json/md` and analyzer JSON; small receipt/assertion transfers outside these stage modules are not silently assigned to the stage.

| unit and owning module | A | B | C | dynamic qualification |
|---|---|---|---|---|
| Z tile:0210/0227/0297 jit_fn |(1,1,0,1)|same|same|inside five-band-chunk loop:5 of each/tile; A/B parent payload, C full-k payload|
| RHS restore: B0248 jit__kernel |(0,0,0,0)|(2,0,0,0)|(0,0,0,0)|outside Z;2 tiles in full run,5 in tile arm|
| C_q build: A0087/B0125 jit__fused |(0,0,0,0)|(4,0,0,0)|(0,0,0,0)|one C_q restore in B; GEMM FFI broadcasts separate|
| chi0: A0561/0662, B0490/0591, C0589/0690 |(0,0,0,0)|(4,0,0,0)|(4,0,0,0)|once after each build, **not per chi tau node**;2 builds,13 and8 integration nodes|
| W solve: A0587/0664, B0516/0593, C0615/0692 |(6,0,0,0)|same|same|one per solve,2 solves; solver redistribution, not a canonical bridge|
| W unfold: A0638/B0567/C0666 jit__do_unfold |(4,0,0,0)|same|same|one static full-q unfold; probe solve remains reduced|
| W symmetry covariance check: A0600/B0529/C0628 jit_fn |(0,2,0,1)|same|same|37 calls; plus separate scalar reductions A0591/0627, B0520/0556, C0619/0655|
| dynamic tau: A1472/B1285/C1238 |(0,0,0,0)|same|same|379 calls; FFI census above|
| static exchange: A0775/B0648/C0747 jit_sigma_sx |(0,0,0,0)|(4,0,0,0)|(0,0,0,0)|one exchange kernel|
| invalid-static G restore: B1092 jit__kernel |(0,0,0,0)|(4,0,0,0)|(0,0,0,0)|two calls in this Si run|
| one pole batch, basis conversion only |(8,0,0,0)|(0,0,0,2)|(0,0,0,0)|two square tensors B and matrix Omega; B1246 gather lowers to all-reduce, not all-gather; A1401…1463 eager seam modules|

Both W solves together plus the one unfold execute16 all-to-alls in each source. Native NCCL sums for these are40.280/44.134/41.704ms A/B/C. The symmetry checks remain too; ONE-ORDER removes basis bridges, not solver redistribution or physical q-grid transport. Their separate source labels prevent counting every all-to-all in W as an unfixed basis conversion.

### I/O-seam cost in A

This is the requested relocated communication row, expanded by owner. Full A01, rank0. Every operation remains distributed over both processor axes. Operator conversions are4 all-to-alls each; single-axis conversions are2. Canonical files are written unchanged.

| seam owner | modules | all-to-alls/run | projected module sum ms | CUDA kernel sum ms | NCCL kernel sum ms |
|---|---|---:|---:|---:|---:|
| zeta write unpack |0273…0285,7 modules|2|135.979|12.875|12.735|
| V pack |0404…0434,16 modules|4|40.572|39.480|35.879|
| restart tensors write |0461…0521,28 modules|8|48.100|46.642|43.015|
| W0 unpack |0666…0720,28 modules|8|71.297|69.873|65.882|
| pole write unpack |1243…1297,28 modules|8|89.516|87.337|80.261|
| pole read pack, one batch |1401…1463,32 modules|8|44.659|42.476|35.279|
| total |139 eager modules|38|430.122|298.683|273.051|

W0 has both the full-q write and reduced-q pre-unfold capture; it is two operator conversions, not one. This scalar GN-PPM run has matrix Omega and B, no B_odd. More poles/batches or an odd residue increase the corresponding row. Final zeta unpack happens once after accumulation, not per tile. The debug per-tile “write” row is not that final I/O seam.

The123ms difference between zeta's projected and kernel sums is a dependency gap; it is not123ms of communication. Cold rank skew also lengthens some NCCL kernels below. These measured costs are charged honestly as observed synchronization time, not a claim about wire bandwidth or an isolated latency constant.

## Owner's collective-accounting table

The brief's historical assumptions are checked against the sampled sources. C does **not** restore G per tau: it builds canonical G directly. The four historical parent-G exchanges were retired before B, so their removal cannot explain an A-vs-B gain. Similarly B's old crop all-gather was already gone. The table's zeros mean absence in these sources, not an unmeasured putative saving.

| family | C present/count | B present/count | A present/count | measured cost per unit and total Si run |
|---|---|---|---|---|
| G canonical restore/tau |no,0|no,0; removed in round3|no,0|0ms in all three. Historical PERF parent capture had4 x~5.5ms/node; that is a different source and schedule, not new A/B evidence.|
| chi restore/build |yes,4 A2A|yes,4 A2A|no,0|C27.596ms NCCL over2 builds (13.798ms/build mean); B275.233ms over2 (137.616ms mean). B warm8-node build6.015ms, cold13-node build269.218ms: do not extrapolate the cold wait as steady bandwidth.|
| zeta RHS exchange/tile |no,0|yes,2 A2A|no,0|B full run6.017ms NCCL/2 tiles=3.008ms/tile; tiled projected bridge median1.911ms including local work.|
| C_q restore |no,0|yes,4 A2A once|no,0|B159.513ms NCCL, once; first exchange is dominated by cold synchronization.|
| static-Sigma restores |no,0|yes,4 in SX +4 per invalid-static G|no,0|SX four explicit ranges sum22.104ms; two invalid-static restores146.266ms NCCL (73.133ms/call mean); total168.370ms.|
| pole-batch pack gathers |no,0|yes,2 axis gathers/operator;2 operators/batch; lowers to2 reductions/batch|replaced by I/O seam,8 A2A/batch|B gather modules total22.587ms projected, including16.330ms NCCL reduction; A44.659ms projected /35.279ms NCCL for the batch. Not an automatic device-speed improvement.|
| duplicate canonical parent face pair |not this parent-route duplication|yes,one pair|no|B18 synchronized construction1.250ms host,8,560,640 carrier bytes/rank (two4,280,320-byte faces); no dedicated collective module. Creation timing is a host measurement, not a made-up device collective cost.|

B18 uses a rank0 JAX-profiler trace and a `canonical_parent_faces` annotation around the actual unmodified function; receipt shapes are(8,80,2,836) and(8,2,836,80). The pair can alias upstream storage, so the carrier byte count is not an allocator-increment claim. GW and trace completed; the inherited tolerance0 cross-source EQP check then failed at1.410micro-eV, as expected from B09/B10. Step18 is a construction-cost receipt, not a passing performance-change leg.

**Reconciliation.** The removed B bridge NCCL durations sum about0.625s over this cold Si run, with large cold waits; A adds about0.273s of I/O NCCL. Neither is remotely a379-times-per-run change. Meanwhile the new eager seams create139 separate modules and increase whole-run compilation from B's527 to A's638. A04 spends2.58s persisting W0 versus B's0.29s, and10.98s in ISDF setup/I/O versus7.96s. These host costs can erase a subsecond communication reduction. The expected big C→B gain is already visible in the tau node,69.807→24.326ms, driven by Green/projection work on8 rather than64 rows and8x fewer broadcast kernels. Round4 changes neither that convolution nor those row counts. This explains the originating leg20's near-equal A/B stage totals without asserting that communication has no cost.

**Scaling and regimes.** Let M=mu_packed, K=nk, Q=n_parent, N=nb, R=tile points, and P=square-grid process count. Each dense operator payload is O(K M²/P) per rank (or Q in a reduced-q seam); an open-spin G adds its spin-block factor. Restores multiply that payload by the number of axes and invocations, with network latency depending on sqrt(P), topology and the collective implementation. Historical G restoration multiplied it by N_tau and mattered even for Si; chi restoration multiplies it by N_chi_build, and static restoration by the small static call count. Z RHS exchange is O(Q M R/P) per tile, independent of N; projector band work is O(Q N M R/P) and remaining projector collectives repeat across band chunks. Full-k FFT/tail remains O(K log K M R/P). Sigma Green/projection work grows with Q and N (one projection also carries N² M work), while the unchanged convolution grows with K M²/P and full-k FFT factors. Parent-face retention is O(Q N M/P),8.56MB/rank here; at large N/M/Q it is a memory-lifetime concern even when construction is1ms.

A's zeta seam scales with stored G-flat size Q M N_G/P once per fit, not per R tile. V/W0/operator pole seams scale with Q or K times M²/P; pole seams also multiply by pole-batch width and number of batches. Compilation/dispatch is largely shape/signature/operation-count cost, independent of the number of entries until compiler lowering and buffer planning become expensive. Cached seams therefore target repeated I/O, many batches and cold executable proliferation. At1000 atoms, dense payload and lifetime can dominate that fixed host cost; the no-replication structure and removed per-build bridges remain valuable even if Si's wall gain is small. This lane measured P4, mu836/1672/3344 and q8/64 in the seam probe; no P>4, nb ladder, parent-count ladder or1000-atom timing is claimed.

## Ablations

Probe12 tests the original eager owner against compiling its exact conversion, alternating execution order ten times and blocking outputs. All twelve shape/direction cases are bit-exact. Representative warm medians: q8,mu836 axis pack~390→5ms; q8 square operator pack~808→2.5ms; q64 square unpack~770→17ms. HLO retains2/4 all-to-alls and no all-gather. Larger mu1672/3344 cases are recorded sample-by-sample in `12_seam_probe/lx_attempt1.log` and named HLO files. This is synthetic distributed-array evidence; cold/host figures are not pure GPU kernel latency.

The selected change caches compiled kernels in `PackedCentroidBasis` by basis identity, axis/spec/direction; operator axes compile together. JAX specializes shape/dtype. Validation, maps, prefix pad/crop and zero-pad algebra remain at the same owner. No consumer changes. A shape change compiles its own correct executable; no array values enter Python cache keys.

The previous lane's ablation-style table, now with all three source baselines and the measured candidate:

| source/arm | Z cold ms | warm total tile ms | tiled zeta stage s | peak Z HLO GiB | arena GB |
|---|---:|---|---:|---:|---:|
| A04 baseline |3291|421,258,276,276|16.4|5.62|6.09|
| B10 round3 |3929|387,289,308,289|16.4|5.62|6.09|
| C11 main |1880|468,200,215|13.0|3.61|3.96|
| A13 cached seams |2961|210,177,222,265|15.7|5.62|6.09|
| A16 baseline repeat |3273|185,272,184,184|15.8|5.62|6.09|
| A17 cached repeat |2745|231,226,220,222|14.8|5.62|6.09|

No settled Z or warm-tile speedup is accepted: the before/after warm samples overlap and reverse order. Z arithmetic and peak6037367604bytes/rank are unchanged; full-deck Z peak14961902580bytes/rank is also unchanged. The retained gain is at the seams:

| matched cold full run | baseline04 | baseline16 | cached13 | cached17 |
|---|---:|---:|---:|---:|
| total s |101.12|98.17|92.36|90.72|
| XLA modules |638|638|509|509|
| compile work s |54.80|54.37|47.56|47.21|
| ISDF setup+I/O s |10.98|9.87|8.22|8.49|
| W persist+head s |2.58|2.56|0.91|0.86|
| Sigma other s |31.64|31.09|30.13|28.46|

Median total falls8.13%; adjacent repeat16→17 falls7.59%. Rank0 profiles01→15 show139 eager seam modules→10 compiled modules, with the **same38 dynamic all-to-alls**. Largest compiled seam peak901457184bytes/rank; all centroid endpoints remain sharded over P. No seam all-gather, reduce-scatter or all-reduce appears. The ten module ids in candidate15 are0273,0392,0419,0421,0429,0431,0576,0578,1101,1205;1101 and1205 execute twice. Compile129 fewer modules explains the stable host benefit without attributing unrelated tau/allocator timing fluctuations to this change.

Rejected: another restore/crop rewrite (already retired in B); eight-row GEMM rewrite (about5.3ms/tile); parent-row gather rewrite (0.18ms/node, small beside14.4ms convolution); changing W solver collectives or full-k convolution (new algorithm/design, not justified by the measured bridge problem). No vendor-specific microkernel was introduced.

## Final verification and disposition

**Retain only cached compiled I/O-seam conversions.** Candidate13 step `lx-Xg4-131041-1719481-9569` exit0; profile15 `lx-Xg4-131054-1721508-1372` exit0; baseline16 `lx-Xg4-131644-1761215-9498` exit0; final17 `lx-Xg4-131922-1779949-5644` exit0 in126s. Baseline16 was run by temporarily restoring only this lane's source file to HEAD, after candidate jobs and CPU14 finished; candidate source was reapplied before17. Each arm retains its exact source.diff. Controls remained read-only.

CPU14 `lx-Xg0-131450-1747364-1087` exit0 in86s: **519 passed,2 skipped,1 strict xfail**,83.37s. Scope is exactly the requested centroid_basis, grouped_layout, centroid_k_unfold, isdf_zq_parent_parity, sigma_parent_projection, padding_owner_static, w_isdf_padding, mpa_store, restart_geometry_contract and symmetry_maps test groups. Four emulated CPU devices; no P>4 or full-suite claim. Existing kernel parity assertions and probe12's exact round trips pass; the source change alters compilation boundaries only.

Candidates13/15/17 each compare against original20 using `tools/eqp_ab.py --tol-uev 0` and `tools/compare_zeta_h5.py --rtol 1e-10`: **eqp0 and eqp1 each224/224 printed-digit identical; all3,932,544 zeta values bit-identical, max_abs0, nonfinite0.** There are no micro-eV differences to explain or waived gates for the accepted change. Frozen sources B/C differences are separately reported above.

Sandbox `tools/gate0.sh ABS_WORKTREE` was run. It is **red**, not passed: its rules gate reports17 inherited findings in other source files and its ledger check names rows254,341,349,384,514,519,523,625,672,694,720,823. No finding names the changed production file. The exact log is `runs/DEV/104_psi_irr_perf2_codex_2026-09-05/gate0.log`; KNOWN_SANDBOX_ERRORS records it. This lane does not repair unrelated source/scaffolding. `tools/branch_status.py --repo ABS_WORKTREE` was also run; the existing unmerged pile is preserved in the adjacent branch_status.log and is not merged here.

## Published state

Claims831/833, the manifest's `psiirr_perf2_steps` and the PSIIRR-PERF2 RUNS_INFLIGHT row own the ledger. This report and the census tool accompany the source change on `perf/psi-irr-packed-order-profiling-2026-09-05`; exact commits and durable archive receipts are appended below after publication. The owner lane may cherry-pick; no merge into its branch or main is performed.

Durable rank0 evidence archive: `/global/cfs/cdirs/m4598/jackm/lorrax_evidence/psi_irr_perf2_2026-09-05/rank0_evidence.tar`,706017280bytes, SHA256 `ec02493f6697aff821cc91bb731015b933a512434e96df73eb4b698d5d6c9231`. It contains the selected raw rank0 Nsight/JAX traces, native CSV, optimized HLO/memory reports, all run scripts/receipts and comparison logs; the final report, claims and gate logs are alongside it. Large repeated science intermediates and other-rank traces remain in scratch and are not needed to substantiate the excerpted results.
