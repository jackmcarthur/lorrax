# AEIG: repeated eigenvalues expose a failure; input mutation is a separate defect

Heavy investigation, 2026-09-09. **Hypothesis 1 holds in the narrower sense of a repeated-eigenvalue failure through the cuSolverMp service. Small spectral width alone does not reproduce it. Hypothesis 2 also holds independently: the non-donating eigensolver overwrites its input. Hypothesis 3 is not an available remedy: both the original failure and these tests already load cuSolverMp 0.9.1, the latest published release.**

Keep AREDUCE's guarded Newton–Schulz metric inverse root (`c0f3e7c`, with capacity follow-up `d411d92`, on branch `lane/sp-areduce-2026-09-07`). Repair input preservation at the native service boundary. Leave the genuine Gram/Ritz eigenproblems in place. Do not claim that the iteration fixes both reported refusals: the Gram-diagonal refusal has a different cause.

Evidence root **R** = `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/338_aeig_20260909`. All new numerical measurements are P4, one node/four processes/four A100 40 GB GPUs, `cuda_async@0.85`, numerical source `b16fe23ff2d831809ffa53f0765cc0fde3fd0f41` in this worktree (leg04 HEAD `d8bbd7fc` adds only this report). Provisioned branch was `arch/sp-aeig-2026-09-09`; publication is on requested `lane/sp-aeig-2026-09-07`. No numerical source change or installation was made.

## Controlled synthetic comparison

Construct complex Hermitian circulants from a prescribed real spectrum using its inverse Fourier transform. Enforce conjugate coefficient pairs exactly. The input has exactly zero Hermiticity defect. All matrix arrays stay XY sharded; only vectors/scalars reach the host. Residuals use a freshly regenerated input **after** the solve, avoiding the overwritten operand. Orthogonality is independently computed with JAX matrix products, not the campaign's native GEMM diagnostics.

Define orthogonality = `||V†V-I||F/sqrt(n)` and residual = `||A_original V-V diag(lambda)||F/||A_original||F`.

| Spectrum | n | Orthogonality | Residual | Job.step / artifact relative to R |
|---|---:|---:|---:|---|
| Uniform [0,2] | 4176 | 6.88529e-15 | 4.33582e-15 | 58128417.23 / `04_uniform_campaign_p4/result.json` |
| Uniform 1 ± 1e-8 | 4176 | 6.66765e-15 | 2.35707e-15 | same |
| Uniform 1 ± 1e-12 | 4176 | 4.03118e-15 | 3.64606e-15 | same |
| Exact identity | 4176 | 0 | 0 | same |
| 2166 eigenvalues at 1; remaining 2010 uniform [0,2] | 4176 | **0.04376881095324079** | **3.21946e-15** | 58128417.21 / `02_campaign_native_p4/result.json` |
| Same mixed multiplicity, contracted to 1 ± 1e-8 | 4176 | 4.58017e-15 | 2.33367e-15 | same |
| Same mixed multiplicity, contracted to 1 ± 1e-12 | 4176 | 3.92201e-15 | 2.97941e-15 | same |
| 2086 eigenvalues at 1; remaining 2010 uniform [0,2] | 4096 | **0.04419417382415921** | **3.30601e-15** | 58128417.22 / `03_block_control_p4/result_4096.json` |
| Repeat mixed 4176 case, fresh process | 4176 | **0.04376881095324079** | **3.21946e-15** | 58128417.22 / `03_block_control_p4/result_4176.json` |

The original AREDUCE measurement is 0.04376881095324085 / 2.13166e-15, job58128417.6, Run334/07_si_eigen_residual_p4/eigen.json. Both synthetic failures have `||V†V-I||F² ≈ 8`. At4176 the maximum Gram error is1.6699353 and four column norms are wrong by more than1e-10; at4096 two norms are wrong. Eigenvalues remain accurate to6e-15. A small eigen residual alone therefore does not certify the orthonormal basis.

This reproduces the failure signature without physical data, ISDF conditioning, native GEMM, or a non-Hermitian input. It does **not** isolate NVIDIA's internal algorithm from every possible FFI/lifetime interaction: a standalone direct-C vendor reproducer and device-info instrumentation are still useful follow-ups. The pattern is not simply monotonic in spectral width. For A≈I a small residual is especially weak evidence, because many nonorthogonal vectors approximately satisfy Av≈v.

Leg01 used the runtime FFI `bc71923a826255cc…`: a login-side FFI environment assignment was reset by launch setup. Leg02/03 explicitly pinned the campaign FFI **inside the payload**, matching the original failure (`03cefb38e54c3f1a…`). Every result records actual `/proc/self/maps` paths; `hashes.json` records full library/probe hashes. Both FFI builds load the same cuSolverMp0.9.1. Leg04 (58128417.23) repeated all uniform cases with the campaign FFI, reproducing every listed numerical diagnostic exactly. The table uses leg04, so the uniform-versus-repeated comparison holds the FFI binary fixed; leg01 is retained as a separately labelled control.

## Version and wiring audit

Actual library: `/global/common/software/m4598/jackm/lorrax_cuda13_runtime/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcusolverMp.so.0`. Runtime banner calls `cusolverMpGetVersion` and prints **0.9.1**, NCCL **2.29.3**, NCCL communicator path, column-major2×2 grid. The installed header independently says0.9.1. Original AREDUCE log07 prints the same versions. Campaign native build receipt is Run322/20_sigma_native/build_receipt.json (58109169.17).

The newest advertised machine compiler module is `nvidia/26.5`; its HPC SDK path is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5`, and its bundled cuSolverMp is **0.8.0**, older than the runtime override. Modules advertised:22.7,23.1,23.9,24.5,25.5,25.9,26.5. This is an inventory of advertised modules and named runtime paths, not an exhaustive filesystem search. See R/local_inventory.json. No module change or install was performed.

NVIDIA's [release notes](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html) identify0.9.1 (August11,2026) as latest. Version0.9.0 fixed STEDC failures with non-power-of-two blocks;0.9.1 fixed incorrect vectors at much larger sizes (P2/P4: N>65536). Neither explains this case: the loaded version includes both fixes, and the failure also occurs at block2048. The notes do not name a remaining clustered-eigenvalue fix. Switching to the26.5 module would not upgrade the solver. A future version trial requires a coherent cuSolverMp/cuBLASMp/NCCL dependency closure, fresh workspace queries and the same P4 reproducer; merely changing module labels is insufficient.

| Contract item | LORRAX implementation and assessment |
|---|---|
| Routine | `src/ffi/cpp/cusolvermp/eigh_ffi.cc:104` calls `cusolverMpSyevd`, jobz='V', CUDA_C_64F here. Divide-and-conquer, not SYEVDX/MRRR; no ORFAC or convergence-tolerance argument exists on this API. |
| Triangle | LOWER for query and solve. With row-major JAX tiles and column-major process mapping the service intentionally solves A transpose; `plan.py:_eigh_columns` conjugate-transposes raw Q back to column eigenvectors. Complex synthetic residuals validate that convention in these cases. |
| Descriptor | n×n, source(0,0), start indices1, `mb=nb=n/2`, `llda=n/2`; A/Q aligned. At4176:2088, at4096:2048. Valid one-block-per-rank square geometry. Smaller block overrides would require actual block-cyclic redistribution and are not a safe workaround. |
| Workspace | Query returns device/host bytes at each call; device scratch allocates queried bytes, host allocation grows to queried size and the queried sizes are passed to solve. No visible dropped factor or under-allocation in this path. |
| Grid/stream | One rank/device; NCCL directly for0.9.1. Private nondefault CUDA stream with event joins to/from XLA. Correct NCCL ABI branch for installed version. |
| Input preservation | **FAIL:** handler const-casts A for an overwriting routine, but `_cusolvermp.py:323–332` declares no donation/alias and no defensive copy. Every new case mutates A; relative changes0.99988–1.40244. This is a real caller-contract violation. |
| Completion info | Handler ignores `ctx->d_info` (`eigh_ffi.cc:150`), treating returned status as sufficient. Vendor example copies device info to host after synchronization and checks zero. This is an audit gap; no new measurement here reads device info. |

NVIDIA's [SYEVD API](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd) promises orthonormal output, accepts the chosen matching square descriptors and exposes no cluster-reorthogonalization dial. Its [sample](https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuSOLVERMp/mp_syevd.c), saved as R/vendor_sample.c, uses column-major native storage and default blocks32, explicit synchronization/info checks, and registered NCCL workspace where supported. LORRAX uses XLA scratch and event joins; unregistered workspace is supported. The consequential differences are the missing caller-preserving copy and unobserved info, not a discovered uplo or workspace-size mismatch.

## Existing inverse-root gate evidence and the two refusals

The AREDUCE correction computes X_next = X(3I-MX²)/2, four iterations starting atI, Hermitianized after each step. A row-sum bound `||I-M||inf<1` certifies SPD/convergence; an explicit `||XMX-I||F/sqrt(R)` residual gate checks the actual four-step result. Four steps are not sufficient for every matrix satisfying the bound, so the residual refusal remains essential. No poles freeze and the genuine Ritz solve is untouched.

Reused evidence root **A** = `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/334_areduce_20260909`:

| Gate | Original measured evidence, not an AEIG rerun |
|---|---|
| All8 Si identicalK, invariants<1e-9 | 58128417.10, A/09_na_routes_p4/si_gauge.json: maximum1.66541587996e-11. |
| Capacity-forced mixed Si reduction passes | 58128417.14, A/14_aot_p4: q0–3 distributed, q4–7 local, all9 math checks; same1.6654e-11 model maximum. |
| Si analytic Sigma path<1e-3meV | 58128417.17, A/16_sigma_p4/si_pair.json: max2.84295663843e-5meV, RMS8.71957267029e-7meV, all8external×6×6bands×133energies/all64internalq. |
| Na all29 model invariants | 58128417.10, A/09_na_routes_p4/gauge.json: max2.72670441464e-10, allK identical. |

These close the **retained-moment refusal** for distributed reduction of the same locally selected directions. They do not close the full distributed direction-selection refusal.

ADIRSEL supplies that separate causal attribution: **58128397.17**, `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/336_adirsel_20260909/04_si_causal_control_p4/diagnosis.json`. Copying only the eigensolver operand preserves M1 and changes q0 Gram minimum from−5776.67410808 (18 negative diagonal entries) to+1.37756351233 (none), with all39 direction ranks unchanged. It stops before reduction and is not a full repaired-model/Sigma gate.

Therefore the brief's premise that one metric eigensolve causes **both** refusals is contradicted by the causal evidence. The service copy must be implemented with corresponding workspace/capacity accounting, followed by fullSiP4 identicalK/1e-9/Sigma1e-3meV gates before claiming both repaired. AEIG adds no second inverse-root implementation and does not claim that outstanding repair completed.

No new W model, full447-call Sigma score, CD score, P16 scaling, SC stability, or performance improvement is claimed. Investigation backlog is complete; production input-preservation repair and independent vendor-C isolation remain named follow-ups.
