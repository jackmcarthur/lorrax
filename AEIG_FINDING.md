# AEIG finding — accepted under ruling47b

**The answer is two independent defects, and an unnecessary eigenvector problem.** cuSolverMp0.9.1 returns nonorthogonal vectors for the repeated-spectrum reproducer even under a direct vendor call with status/info zero. Our non-donating binding also overwrote its input. Native repair **ec3e5d4e** preserves the input; **05341b40** replaces only the metric inverse square root with guarded coupled Newton–Schulz. The genuine Gram and Ritz eigenvector solves remain unchanged.

**Si and Na pass the owner's |DeltaSigma|<2meV gate, and every Na rank's full-driver peak is unchanged. Ready for ALAND to merge ec3e5d4e and05341b40 and rebuild the native library.** Test assembly **69a530ad6ddeed24de6a18283ac0c9006ff0293c** merges those changes with accepted ALAND2e9ceb8a on branch `lane/sp-aeig-2026-09-07`; it is not a request to merge the whole test-assembly commit. Source changes are pushed. Claims2079 and2088 supersede the earlier strict acceptance labels; no remaining-discrepancy investigation is requested.

Evidence **R** = `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/338_aeig_20260909`. Exchange **D** = `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/152_shared_pole_push_2026-09-07`.

## Acceptance measurements

| Check | Measured result | Job.step / artifact relative to R |
|---|---|---|
| Si, full distributed construction | All8K identical; both original refusals clear; normalized metric residual9.46199725146e-17 | 58128417.30 /12_coupled_si_p4/strict_assessment.json |
| Si analytic Sigma change | max0.00573271300760meV, **PASS** under2meV | same; superseding15_ruling47b_si/assessment.json, claim2079 |
| Na production Sigma_c cube change | max**0.123467230301968meV**, all41×29×86×86 stored complex entries, **PASS** | baseline58128417.31/candidate58128417.32;16_na_assessment/acceptance.json |
| Na exact-model Sigma change vs accepted ALAND08 | max1.35733608751e-8meV,29external×3×3frontier×41energies/all512internalq | 58128417.32 /14_na_coupled_p16/cd48/production.npz and16_na_assessment/acceptance.json |
| Na CD48 diagonal RMS | **0.282992393053667meV**, matching target0.282992meV; full-precision accepted reference0.282992393056295meV | 58128417.32 /14_na_coupled_p16/cd48/receipt.json; reference ALAND08,58128243.18 |
| Na every-rank peak | **8,517,691,768B on each of16ranks in both arms**, non-increase PASS | 58128417.31 and.32 /13_na_baseline_p16/production/measure_rank*.json and14_na_coupled_p16/production/measure_rank*.json |
| Na constructor | All29K identical,1061–1589; norm bound1.55730813617e-8–3.77246062755e-7;1–2iterations; max normalized ZAZ-I residual1.00808926701e-16 | 58128417.32 /16_na_assessment/model_summary.json |

Both Na arms are **cold**, with empty rule caches and persistent compilation cache disabled, on the same four SP-A5 nodes,16A100-SXM4-40GB GPUs. The deck uses **low_mem_bands=true**, matching the accuracy/peak reference; these are not timing-performance claims. The baseline is accepted ALAND2e9ceb8a, and the Na reference deck has `linalg=local`; the Si refusal repair was separately tested with full distributed direction selection and reduction. Peaks are the same whole-driver JAX allocator high-water measure used by ALAND; external native/host allocations are excluded. Startup source closure names `/pscratch/sd/j/jackm/wt_sp_aeig` in both runs. No warm arm was executed.

The full-production comparison is independent of the analytic exact-model comparison. The newly regenerated cold baseline differs from historical ALAND08 production by at most0.0394765895207meV over the full stored cube; neither equality nor offset fitting is required. The accepted2meV bar is satisfied by the actual fresh production pair as well as the exact-model reference comparison. Do not reinterpret the old1e-9 invariant or0.001meV path thresholds as current blockers: their raw failed labels remain immutable in historical artifacts, with acceptance superseded by ruling47b.

## Na CD48 scope and complete score

**58128417.32**, `R/14_na_coupled_p16/cd48/receipt.json`: all29parents and external representatives, all512internalq, frontier bands3–5 zero-based,41global energies−5…+5eV relative to FD9mu,eta0.25eV,fixedG,86logical/88carrierbands. Energy-trapezoid RMS and uniform external representatives; no offset subtraction. The canonical comparator computes production_exact−CD8_model_exact−stored(CD48_direct−CD8_model). This is the established correlated CD48 translation, not a new direct-W integration or absolute-CD certificate.

| Metric | meV |
|---|---:|
| Full RMS |0.163883184418420|
| Diagonal RMS |0.282992393053667|
| Off-diagonal RMS |0.015626897950725|
| Sampled maximum absolute CD48 proxy error |2.399890556714872|
| Own-energy RMS,49in-window states |0.403174831207114|
| Own-energy maximum |1.498455517651546|
| Per-state constant RMS |0.188026484251971|
| Remainder RMS |0.211496420173213|

The constant squared fraction is0.441457122229. The sampled absolute CD48 maximum was already2.399890556502312meV in accepted ALAND08; it is not the candidate-minus-reference change used for replacement acceptance. Both the own-energy CD48 maximum and the full production-pair change are below2meV. Exact state checks reproduce physical energies, rows,chemical potential and padded-wavefunction norm; hashes and factor-action controls are in the canonical receipt.

J=K per parent, damping fraction0 throughout; Gram condition8.67013172086e7–9.99061544469e7. All29 pass V-whitened passivity at0.25eV: largest eigenvalue0.966469641969, minimum−8.5701e-17(roundoff). Total padded compact model storage660,986,096B. Per-parent J/K,condition,storage,passivity and metric receipts: `R/16_na_assessment/model_summary.json`, job58128417.32. No SC stability or broadening-fit claim; poles are rebuilt normally.

## Direct vendor reproduction and version/wiring audit

Define orthogonality `||V†V-I||F/sqrt(n)` and residual `||Aoriginal V-V diag(lambda)||F/||Aoriginal||F`. Complex circulant inputs are exactly Hermitian; all large arrays stay tiled. A freshly regenerated original operand is used in residuals to exclude the independent overwrite defect.

| Spectrum | n | Orthogonality | Residual | Job.step / R artifact |
|---|---:|---:|---:|---|
| Uniform[0,2] |4176|6.885291e-15|4.335820e-15|58128417.23 /04_uniform_campaign_p4/result.json|
| Uniform1±1e-8 |4176|6.66765e-15|2.35707e-15|same|
| Uniform1±1e-12 |4176|4.03118e-15|3.64606e-15|same|
| Identity |4176|0|0|same|
|2166ones plus2010uniform[0,2], direct vendor call |4176|**0.04376881095324079**|3.219455e-15|**58128417.27 /08_native_controls_p4/direct_result.json**|
|2086ones plus2010uniform[0,2], service control |4096|0.04419417382415921|3.30601e-15|58128417.22 /03_block_control_p4/result_4096.json|

The original AREDUCE failure was0.04376881095324085/residual2.13166e-15, job58128417.6, Run334/07_si_eigen_residual_p4/eigen.json. Mixed repeated spectra contracted to1±1e-8 or1±1e-12 also pass (58128417.21,02_campaign_native_p4/result.json), so small spectral width alone is not sufficient to trigger the failure. The failing matrices have `||V†V-I||F²≈8`. A tiny residual does not certify an orthonormal eigenbasis.

The direct C++ reproducer uses ordinary cudaMalloc and checks device info, bypassing XLA FFI/scratch while retaining the established communicator/grid: `R/07_native_repair_p4/direct.cc`, `R/08_native_controls_p4/direct_probe.py`. **Vendor status=0 and device info=0 on all4ranks despite the failure.** Patched FFI reproduces identical numerical diagnostics. Source/native/object hashes are in07_native_repair_p4/build_receipt.json, build58128417.26; tested native SHA256 prefix71f39c3c5761eafb. This is not a separately initialized MPI executable.

Linked library **cuSolverMp0.9.1**: `/global/common/software/m4598/jackm/lorrax_cuda13_runtime/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcusolverMp.so.0`, NCCL2.29.3. Its own version call/header and actual mapped-library paths were checked. Newest advertised module `nvidia/26.5` bundles older0.8.0. Named inventory: `R/local_inventory.json`. No newer available version established and no install performed. NVIDIA's [release notes](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html) list0.9.1 as latest at this audit;0.9.0's non-power-of-two STEDC fix and0.9.1's large-N(>65536 atP2/P4) vector fix do not explain this4096/4176case.

Wiring audit against NVIDIA's [SYEVD contract](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd) and [sample](https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuSOLVERMp/mp_syevd.c):

- SYEVD divide-and-conquer,jobV,LOWER,complex128; no ORFAC/tolerance parameter is exposed.
- Matching square A/Q descriptors, starts1, source(0,0), blocks2088 at4176 and2048 at4096, valid2×2geometry. Row-major tiles and process mapping solve a transpose; the service conjugate-transposes raw Q back to column vectors. Complex residual controls validate the convention.
- Queried device/host bytes are honoured. Private stream/event joins connect XLA to the vendor. Unregistered workspace is supported; the sample additionally illustrates registration and explicit info checks.
- **Independent caller-contract bug:** the non-donating handler overwrote A. Repair ec3e5d4e copies its XY tile into scratch and accounts for alignment/bytes in the public query. P4.27 red twin rejects old input change3.01199401889; repaired input change0 and service controls PASS. AtR4176 scratch558483712vendor+69755904private operand=628239616B/rank.
- Production device-info handling remains an audit gap. Reading info would not catch this demonstrated numerical failure, because direct info is zero.

Vendor defect registered explicitly in `S/KNOWN_LORRAX_ISSUES.md` at `src/ffi/cpp/cusolvermp/eigh_ffi.cc:175`, with version/reproducer, claim2067; exchange receipt `D/exchange/aeig/VENDOR_BUG_RECEIPT.md`. The Gram-diagonal refusal was caused by input mutation; the retained-moment refusal by the metric correction. They must not be attributed to one failure.

## Coupled correction and TENPLUS

For A=Hermitian(z†Gz)+identity on discarded columns, use Y0=A,Z0=I,T=(3I-ZY)/2,Ynext=YT,Znext=TZ through the resolved GEMM. Guard the initial infinity bound d<1; `d_next<=d²` determines the count once from d and target=min(existing tolerance,32eps64), with zero steps within target. No residual-convergence loop or fitted knob; receipt `||ZAZ-I||F` and its normalized value. Invalid direct calls refuse before iteration with the measured norm; traced calls take zero steps and propagate a compulsory failed predicate. P4 controls58128417.29,11_coupled_si_p4/controls.json: bounds0,1e-8,.2,.8,.99 select0,1,5,8,12steps; maximum analytic inverse-root error1.7764e-15; bound1refuses. No second inverse-root implementation remains in source.

`D/exchange/arch/AEIG_TENPLUS.md` answers all six requested points, claim2080: one **untested10–14s** hypothesis replaces the direction dilation with W†W at the existing squared singular cutoff. It attacks ACON's measured18.31459s direction band(58128397.19), but squares conditioning and changes matrix dimensions, so ruling48c reserves it for a ruling before implementation. ACON/distrib_la own it; no duplicate work or heuristic/cache/dial added. There is no demonstrated ten-second saving in the metric-only change, and no performance claim from these low_mem_bands=true accuracy legs.

Investigation, requested replacement, Si/Na acceptance, vendor registration and TENPLUS response are complete. ALAND owns integration of the two ready commits. The vendor bug itself remains upstream; no message was sent to NVIDIA.
