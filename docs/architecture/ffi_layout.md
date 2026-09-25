# The FFI layer

How LORRAX reaches vendor libraries: the core kernel operations and the
engine each has on each hardware class ([kernel operations](#kernel-operations)),
every native target (the [kernel catalog](#kernel-catalog)), the layers, the
two build legs and their acceptance gates, which library serves each engine on
each machine, which cuSOLVERMp selects which communication path, which FFT
engine the host library binds, the C++ phdf5 defaults, and how to tell the
native-layer failure modes apart.

This page owns the *native boundary*, the kernel-operations table and the
kernel catalog. Owner rulings are in [`decisions.md`](decisions.md), the
SlabIO contract in [`slab_io.md`](slab_io.md), knob spellings and defaults in
[`../dev/env_vars.md`](../dev/env_vars.md). See the
[register](../index.md#register).

## Kernel operations

One row per core operation: what it computes, where the physics uses it, the
engine on each hardware class, whether a plain-XLA implementation exists, and
the test that pins it. A plain-XLA route is one built from XLA ops alone, with
no LORRAX native library; where the column says *none*, a platform without a
LORRAX library (ROCm today) has no engine for that operation. Verified against
the code at `origin/main` 9379b694 (2026-09-25).

| operation | where it acts | NVIDIA (CUDA leg) | CPU (host leg) | plain XLA route | gate | code |
|---|---|---|---|---|---|---|
| **χ₀ = G ⋆ G**, one τ node: `χ₀(R) += α_τ Σ_ab conj(Gᶜ_ab(R))·Gᵛ_ab(R)`, one FFT_k after the τ sum | gwjax screening (`gw.w_isdf`) → χ₀_q(μ,ν;τ) | mode 11 on the k-box stage for the identity-vertex step response on a raw-parent plan; every other χ₀ kernel (vertex pairs, Fermi–Dirac occupations, distinct left and right centroid sets, the fractional contour) takes mode 3 on unfolded Greens with the spin trace in XLA ([router](#k-convolution-router-and-the-mathdx-family)) | typed unfold and spin trace in XLA, host flat-k transforms | none | `tests/multi_device/kconv_router_p4.py`, `tests/test_kconv_chi_unfold.py` | `ffi.fft.make_kconv_chi_unfold`; `gw.w_isdf._get_chi_minimax_kernel` |
| **Σ = G ⋆ W**: `Σ_k = FFT_k[IFFT_k Ĝ · W(R)]`, Ĝ the Green unfolded on load | gwjax Σ: `gw.ppm_tau_kernel` (τ stream), `gw.cohsex_sigma` → Σ_k(μ,ν;τ) at the caller's k rows | mode 7 on the raw-parent Green (typed unfold and spin action on load); mode 2 on the k-box stage for a full-k operand ([router](#k-convolution-router-and-the-mathdx-family)) | typed unfold in XLA (a full-k copy), then `lorrax_mklfft_gw_conv` | none | `kconv_router_p4.py`, `tests/test_kconv_klead_unfold.py` | `ffi.fft.make_kconv_klead_unfold`, `make_kconv_klead` |
| **Four-current Σ**: `Σ = FFT_k Σ_AB γ_A (IFFT_k Ĝ) γ_B† ∘ W_AB(R)` | gwjax bispinor Σ: `gw.cohsex_sigma`, `gw.centroid_k_unfold` | mode 8: resident arm when a whole `ns²` spin group fits the opt-in shared memory, else the k-box split arm | mode 7's host composition, the γ block sum in XLA | none | `kconv_router_p4.py`, `tests/test_kconv_lorentz_unfold.py` | `ffi.fft.make_kconv_lorentz_unfold` |
| **W, V wedge → R space**: `Y_k = IFFT_k(L_k Ô_k R_k†)`, Ô read from the irreducible q wedge | the W and V operands of the two Σ rows: `gw.screening`, `gw.ppm_tau_kernel`, `gw.cohsex_sigma` | mode 9; a full-zone operand takes mode 3 | unfold in XLA (`apply_unfold_load_tables_local`), then the host flat-k transform | none | `kconv_router_p4.py`, `tests/test_kfft_klead_unfold.py` | `ffi.fft.make_kfft_klead_unfold`; `symmetry_maps.QirrOperator` |
| **k-axis FFT**: `Y = s·FFT^±_k X`, k leading or trailing | χ₀(R) → χ₀(q) (`gw.w_isdf`), the q→0 head (`gw.qsgw_head`), `gw.wavefunction_bundle`, band interpolation (`bandstructure.htransform`, `.orbital`), BSE (k trailing) | mode 3 on the k-box stage (k leading), mode 5 (k trailing) | `lorrax_mklfft_flat_k`, the FFTW3 advanced interface bound by `dlsym` ([§3c](#3c-which-fft-engine-the-host-library-binds)); k trailing: XLA transposes around it. `LORRAX_FFT_FFI=0` refuses | none | `kconv_router_p4.py`, `tests/test_fft_flat_k_numerics.py`, GATE 5, GATE 8 | `ffi.fft.make_kfft_klead`, `make_kfft_kminor`; `common.fft_helpers.make_flat_k_fft` |
| **ζ-fit pair Gram** `C_q`: the pair convolution `s·FFT_k Σ_ab φ_a φ_b conj(IFFT_k P^L)·IFFT_k P^R` of the band projectors `P_k(μ,ν)` (mode 0 in the [mode table](#k-convolution-router-and-the-mathdx-family)) | gwjax face-ψ ζ fit (`gw.isdf_fitting`, one C_q per channel), `gw.downfold` | mode 1, the typed parent load ([mode 1](#parent-load-isdf-pair-convolution-mode-1)); mode 0 on full-k operands | typed parent load in XLA (a full-k open-spin copy per side), host transforms, spin contraction in XLA | none | `kconv_router_p4.py`, `tests/test_isdf_parent_conv.py`, `tests/test_isdf_zq_parent_parity.py` | `ffi.fft.make_fused_conv_kparent`, `make_fused_conv_kpair` ← `isdf.core.c_q_from_psi_sm`, `c_q_downfold` |
| **Route-G plane pair contraction**: mode 0 read from the D-plane FFT output, Bloch phase and L/R split applied on load | gwjax charge ζ fit, route G: `isdf.zeta_mubatch`, `isdf.pair_kernels`, `gw.centroid_k_unfold` → the Z_q store | mode 6 | phase, split and transpose in XLA, then the host pair tail | none | `kconv_router_p4.py`, `tests/test_kconv_plane.py` | `ffi.fft.make_fused_conv_kplane` |
| **Route-G plane FFT**: `Y = FFT₂(P)`, P the plane scattered from its occupied cylinder | gwjax charge ζ fit, route G (`isdf.zeta_mubatch`) | mode 10 iff both sides split into thread FFTs of ≤ 40 points and plane + row tables fit the opt-in shared memory: squares to 100 on sm_80/87, 78 on sm_86/89/120, 119 on sm_90/100; every other plane takes the XLA route ([mode 10](#plane-fft-with-gather-on-load-mode-10)) | the XLA route | static-run concatenate, then `jnp.fft.fftn` | `tests/multi_device/plane_fft_gather_p4.py`, `tests/test_plane_fft_gather.py` | `LocalFourierPlan(in_gather=…)` → `ffi.fft.make_plane_fft_gather` |
| **BSE W-term k-convolution**: `U = s·FFT_k(IFFT_k X · K_R)`, k trailing | bse matvecs: `bse.bse_stack_matvec`, `bse.bse_ring_comm` (the W term on trial vectors) | mode 4, with a complex64 image for the fp32-GMRES arm | XLA moves k to the front, host transforms and product, k moves back; complex128 only | none | `kconv_router_p4.py` | `ffi.fft.make_kconv_kminor`, `make_local_kconv_kminor` |
| **Sphere ↔ box 3-D FFT**: ψ_nk(G) ↔ ψ_nk(r), densities, plane-wave matrix elements | ψ at the centroids (`common.wfn_transforms`), the kmeans valence density (`psp.get_DFT_mtxels`), the V_H, kin_ion and dipole matrix elements (`common.mtxel_sweep`), the QSGW density (`gw.qsgw_density`), DFT operators (`psp.dft_operators`) | XLA `fft` → cuFFT inside jaxlib | XLA:CPU `fft` | this row: `jnp.fft` inside the caller's `shard_map`; no FFI route reaches it | none on numerics; `tests/test_fft_shardmap_context.py` checks only that call sites sit inside a `shard_map` | `common.fft_helpers.local_fftn3`, `local_ifftn3`, `make_sharded_fftn_3d`, `make_sharded_ifftn_3d` |
| **Separable local DFT with supports**: `y = R_out·F·E_in·x`, ≤ 3 axes | no production caller besides the route-G plane FFT's `in_gather` form | `lorrax_fourier_plan_mathdx`: cuBLAS ZGEMM axes, the fused cuBLASDx pair when it fits the opt-in shared memory, one cuFFT group; startup refuses an nvidia-mathdx other than 25.6.0 (`GATE mathdx-pair-wheel`); cuFFT/cuBLAS sizes below 2³¹ (`GATE fourier-plan-int32`) ([plan](#local-fourier-plan-localfourierplan)) | the XLA leg | `dot_general` GEMM axes and one `jnp.fft` group, on every non-CUDA lowering | `tests/test_fourier_plan.py` (GPU and cpu) | `common.fourier_plan.LocalFourierPlan` ([service](../dev/fourier_plan.md)) |
| **Green build GEMM**: `G_k(μ,ν;τ) = Σ_n ψ_nk(μ) w_n(τ) ψ*_nk(ν)` at the k parents | every Green of gwjax: χ₀ (`gw.w_isdf`), Σ (`gw.ppm_tau_kernel`, `gw.cohsex_sigma`), the ζ-fit projectors (`gw.isdf_fitting`) → the parent Green `(n_parent, μ, s, ν, s')` | cuBLASMp batched GEMM through `distrib_la.gemm_plan(layout='face')`; an active band range through `cublasmp_[prepared_]active_range_gemm` | none: the face plan resolves `scalapack`, whose batched GEMM handler does not exist, and refuses | `gemm_plan(layout='axis')` (`jnp.matmul`); gwjax builds its ψ carriers in the face layout, so the Green build never takes it | `services/distrib_la/tests/test_distrib_la_multiproc.py`, `test_active_gemm_range.py`, `test_local_active_gemm_range.py`, `tests/multi_device/active_band_sigma_gate.py` | `gw.greens_function_kernel.build_G_parents`; `distrib_la.gemm_plan` ([active GEMM ranges](../dev/active_gemm_ranges.md)) |
| **Band projection**: `O_k,mn = Σ ψ*_mk(μ) O_k(μ,ν) ψ_nk(ν)` | Σ_mn from Σ_k(μ,ν) (`gw.ppm_tau_kernel`, `gw.cohsex_sigma`, `gw.photon_sigma`, `gw.mpa.sector_sigma`); BSE W decode (`bse.bse_ring_comm`), `common.zeta_projection` | face layout: two planned cuBLASMp GEMMs; legacy body: XLA einsums and two `psum_scatter`s | face layout: none (as the Green build); legacy body: the right contraction on `lorrax_mklblas_gemm_batch` (CBLAS, [§3a](#3a-the-dependency-matrix)) | the legacy body's einsums (`LORRAX_BANDS_GEMM_FFI=0` on cpu) | `tests/test_contract_bands.py`, `active_band_sigma_gate.py` | `common.contract_bands.contract_bands_block_reshard` ([vendor GEMM](../dev/vendor_gemm_service.md)) |
| **Hermitian eigensolve**, batched local or distributed | the charge ζ factor (rank-truncating eigh of each C_q, dense and replicated under both `linalg` layouts), the QSGW `H_k` (`gw.sc_iteration.qp_eigh`) | local: `jnp.linalg.eigh` (cuSOLVER in jaxlib); `linalg = distributed`: cuSOLVERMp `syevd` | local: LAPACK in jaxlib; distributed: ScaLAPACK `p?heevd`/`p?syevd`; host SLATE eigh always refuses at resolve (bug L-2) | `jnp.linalg.eigh` | `pytest -m distrib_la` contract tests; nothing observes which vendor answered | `distrib_la.plan('eigh')`, `dispatch_batched_eigh`; the charge factor: `isdf.cplus.factor` ([`distrib_la`](../services/distrib_la.md), [deck dial](../dev/linalg_ffi.md)) |
| **Dyson solve and dense factorizations**: `W_q = (1 − v_q χ₀_q)⁻¹ v_q` by LU; the transverse ζ LU; Cholesky on explicit request | gwjax screening (`gw.w_isdf.solve_w`), the response bank (`gw.response_bank`), the shared-pole head, the transverse ζ factor | local: per-q `jax.scipy.linalg.lu_factor`/`lu_solve` (cuSOLVER in jaxlib); distributed: cuSOLVERMp batched `solve_lu`/`getrf`/`getrs`, Cholesky `potrf`/`potrs`. `LORRAX_LU_NO_PIVOT` turns cuSOLVERMp pivoting off with no gate. The fused `cublasmp_batched_w_solve` has no production caller | local: LAPACK in jaxlib; distributed: ScaLAPACK `p?getrf`/`p?getrs`; Cholesky: host SLATE `potrf`/`trsm` | the local LU; Cholesky `native2d` | contract tests, `tests/test_transverse_factor_hoist.py`, `tests/multi_device/dyson_panel_p4.py`; nothing observes which vendor answered | `distrib_la.plan('solve_lu')`, `plan('cholesky')` ([targets](#dense-linear-algebra-targets)) |
| **Active-subspace kernels**: store, projected eigh, project, reconstruct, Gram, CGS2 orthogonalization | Davidson (`psp.run_nscf`), Lanczos (`bse.bse_lanczos`, `bse.exciton_bands`) | `lorrax_active_subspace_*` (cuBLAS, cuSOLVER, NCCL) | `CpuSubspacePlan`: NumPy/LAPACK through `jax.pure_callback` | none | `services/distrib_la/tests/test_active_subspace.py`, `tests/test_subspace_orthogonalize.py` | `distrib_la.plan_subspace`, `plan_orthogonalization` ([Davidson](../services/davidson.md)) |
| **Contour accumulator**: `A[o,q,m,n] += p[o]·c[q,m,n]` | the response bank's Laplace/KMS streams (`gw.w_isdf` with selected q rows) | `lorrax_contour_accumulate`, complex128, at most `4·65535` outputs ([small kernels](#small-cuda-kernels)) | none: refuses (`GATE ffi-handler`) | none | `tests/multi_device/contour_accumulator_p4.py` | `gw.contour_accumulator` |
| **Spin rotation**: `G ← U_k G U_k†` per `(k, μ, ν)` spin block | the typed unfold of a parent operator to full k (`symmetry_maps.unfold_spin_centroid_operator` ← `gw.greens_function_kernel.build_G`, `gw.photon_sigma`) | `lorrax_symmetry_spin_rotate_centroid` for `ns ∈ {2, 4}`, complex128; otherwise the einsums | the einsums | JAX einsums (`_rotate_open_spin_centroid_operator`) | the mode-7 reference chain of `kconv_router_p4.py` | `symmetry_maps._spin_rotation` |
| **Parallel HDF5 slab I/O** | every sharded array read or written through `file_io.slab_io` | `phdf5_{read, read_kchunk_union, write, write_independent}`, staged through the CUDA runtime | the same handlers on the host leg | none: one transport, and a deployment that cannot serve it refuses at open | `tests/multi_device/phdf5_*`, `slab_io_*_p4.py`; GATE 7, GATE 10 | `ffi.io` ← `file_io.slab_io` ([§5](#5-parallel-hdf5-the-ffi-side), [SlabIO](slab_io.md)) |

**Every mathdx kernel** (the k-axis rows and mode 10) is NVRTC-built for the
device's own `sm_<cc>`, and `require_kconv` probe-compiles one at startup. On
CUDA every k-grid axis of a k-axis operation is at most 40
(`GATE mathdx-kconv-axis`), and its resident k-row or k-box tile must fit the
device's opt-in shared memory; the cpu leg has neither limit
([refusals](#k-convolution-router-and-the-mathdx-family)). Operands are
complex128; modes 2–5 also take complex64 on CUDA.

**Gaps.**

* No CPU engine: the Green build GEMM and the face-layout band projection
  (gwjax builds its ψ carriers in the face layout), and the contour
  accumulator.
* No plain-XLA route: every k-axis operation, the face-layout GEMMs, the
  active-subspace kernels, the contour accumulator and slab I/O. A platform
  without a LORRAX native library has no engine for them.

## Kernel catalog

Every native entry point is an XLA FFI target (a string a loader registers)
or a ctypes C entry point. From a target string:

1. **Symbol.** Its row in `ffi_loader._CUDA_TARGET_SYMBOLS` /
   `_HOST_TARGET_SYMBOLS`, or in `distrib_la.loader`'s tables of the same
   names (the distributed linear algebra and the active subspace), gives the
   handler symbol. The spin rotation is the one target its door registers
   itself (`symmetry_maps._spin_rotation`).
2. **File.** `git grep -n 'XLA_FFI_DEFINE_HANDLER_SYMBOL' -- src/ffi/cpp`
   lists every handler with its file; the phdf5 handlers are spelled
   `LRX_PHDF_HANDLER(<X>)` (`<X>Ffi` on CUDA, `<X>HostFfi` on host).
3. **Door.** `git grep -n <target>` finds the Python constant and its
   `ffi_call`.

Startup refuses a provider without a target the run needs: the router's and
the Fourier plan's targets (`require_kconv`, `require_fourier_plan`), the
contour accumulator and spin rotation symbols (`ffi_loader.require_cuda_handlers`,
`GATE ffi-handler`), the host FFT and GEMM (their `Gate`s).

The [operations table](#kernel-operations) owns what each family computes,
where it is used, what selects it and its gate; this table maps each family to
its sources, its build and its target strings.

| family | source (`src/ffi/cpp/`) | leg; build | targets (`lorrax_…`) |
|---|---|---|---|
| mathdx k-convolution | `cufft/kconv_mathdx_cuda_ffi.cc`, `cufft/kbox_stage.cuh` | CUDA; NVRTC for the device's own `sm_<cc>` at first use, disk-cached | `mathdx_kconv_{pair, parent, plane, klead, klead_unfold_block, klead_lorentz_conj, chi_unfold, kminor}`, `mathdx_kfft_{klead, klead_unfold, kminor}`; older trees: `mathdx_kconv_klead_{unfold, lorentz}[_rows]` |
| plane FFT (mode 10) | the same file (`kPlaneSrc`) | CUDA; NVRTC | `mathdx_plane_fft_gather` |
| Local Fourier plan | `cufft/fourier_plan_cuda_ffi.cc`, `cufft/fourier_plan.cu` | CUDA; C++ (cuBLAS, cuFFT), nvcc remap kernel, NVRTC cuBLASDx pair | `fourier_plan_mathdx`; older trees: `fourier_plan` |
| NVRTC build service | `common/nvrtc_build.{h,cc}`, `common/lrx_async_gather.h`, `cufft/kbox_stage_src.h.in` | CUDA; C++ | none; gate `tests/multi_device/kconv_cubin_cache_check.py` ("Disk cubin cache" in the [router](#k-convolution-router-and-the-mathdx-family)) |
| host flat-k FFT | `fftw/fft_flat_k_ffi.cc` | host; C++, the FFTW3 ABI bound by `dlsym` | `mklfft_flat_k`, `mklfft_gw_conv` |
| host CBLAS GEMM | `cblas/gemm_batch_ffi.cc` | host; C++ | `mklblas_gemm_batch` |
| distributed dense LA | `cusolvermp/`, `cublasmp/batched_gemm_ffi.cc`, `scalapack/`, `slate/` | CUDA: cuSOLVERMp, cuBLASMp; host: ScaLAPACK, SLATE | [below](#dense-linear-algebra-targets) |
| local active-range GEMM | `cublas/local_active_gemm_ffi.cc` | CUDA; C++ | `cublas_local_active_range_gemm`, `cublas_local_prepared_active_range_gemm` |
| active subspace | `active_subspace/active_{eigh,ops}.cc` | CUDA; C++ (cuBLAS, cuSOLVER, NCCL) | `active_subspace_{store, eigh, project, reconstruct, gram, ortho, distributed_ortho, subtract, subtract_gram}` |
| parallel HDF5 | `phdf5/` | both; C++ (HDF5, MPI; CUDA-runtime staging on the CUDA leg) | `phdf5_{read, read_kchunk_union, write, write_independent}`; `phdf5_read_kchunk` has no caller |
| contour accumulator | `response/contour_accumulate{.cu,_ffi.cc}` | CUDA; nvcc | `contour_accumulate` |
| spin rotation | `symmetry/spin_rotate{.cu,_ffi.cc}` | CUDA; nvcc | `symmetry_spin_rotate_centroid` |
| fused W-solve | `cublasmp/batched_w_solve_ffi.cc`, `cublasmp/w_solve_kernels.cu` | CUDA; C++ and nvcc | `cublasmp_batched_w_solve`; door `ffi.cublasmp.batched_fused_w_solve`, gated by the `distrib_la` contract W-solve cell and `tests/bench/cublasmp_w_solve_test.py` |

**Architectures.** Every nvcc TU carries SASS for sm_80, 86, 89, 90, 100
and 120 and compute_80/compute_120 PTX (§2); every NVRTC kernel compiles for
the device's own `sm_<cc>`. Launch sizes (rows and planes per block, k-box
tiles, the pair's threads) come from the device's opt-in shared memory, its
shared memory per SM and its SM count, except the fixed budgets named in the
router section. `cp.async` (sm_80+) is the only architecture-specific
instruction (`common/lrx_async_gather.h`, `cufft/kbox_stage.cuh`).

### Dense linear algebra targets

`distrib_la` owns these doors, their selection and their refusals
([`distrib_la`](../services/distrib_la.md)); the handlers are here.

| target (`lorrax_…`) | symbol, file | `distrib_la` entry | selected by |
|---|---|---|---|
| `cusolvermp_eigh` | `EighMpFfi`, `cusolvermp/eigh_ffi.cc` | `_cusolvermp.distributed_eigh` | `linalg = distributed` on CUDA |
| `cusolvermp_batched_potrf`, `_potrs` | `CusolverMpBatched{Potrf,Potrs}Ffi`, `cusolvermp/batched_{potrf,potrs}_ffi.cc` | `batched_distributed_cholesky`, `_potrs` | an explicit `cholesky` request (the charge ζ factor is `rank_truncate`) |
| `cusolvermp_batched_solve_lu`, `_getrf`, `_getrs` | `CusolverMpBatched{SolveLu,Getrf,Getrs}Ffi`, `cusolvermp/batched_solve_lu_ffi.cc` | `batched_distributed_solve_lu`; `distrib_la.factor` / `solve` | the W Dyson solve under `linalg = distributed`; the hoisted transverse ζ factor |
| `cublasmp_batched_gemm` | `CublasMpBatchedGemmFfi`, `cublasmp/batched_gemm_ffi.cc` | `distrib_la.matmul`, `gemm_plan(layout='face')` | the CUDA provider route |
| `cublasmp_active_range_gemm`, `_prepared_active_range_gemm` | `CublasMp{,Prepared}ActiveRangeGemmFfi`, same file | `GemmPlan` with `enable_active_range` | a traced or a host-known K interval |
| `scalapack_eigh` | `ScalapackEighHostFfi`, `scalapack/eigh_ffi.cc` | `_scalapack.distributed_eigh` | `linalg = distributed` on cpu |
| `scalapack_batched_solve_lu`, `_getrf`, `_getrs` | `ScalapackBatched{SolveLu,Getrf,Getrs}HostFfi`, `scalapack/{solve_lu,getrf_getrs}_ffi.cc` | `batched_distributed_solve_lu`; `factor` / `solve` | `solve_lu` distributed on cpu |
| `slate_potrf`, `_trsm`, `_batched_potrf`, `_batched_trsm` | `Slate*HostFfi`, `slate/host_ffi.cc` | `_slate.*` | an explicit `cholesky = slate` on cpu |
| `slate_eigh` (host) | `SlateEighHostFfi`, `slate/host_ffi.cc` | — | always refused at resolve (SIGSEGV, bug L-2) |
| `slate_*` (CUDA) | `slate/{eigh,potrf,trsm,batched_potrf,batched_trsm}_ffi.cc` | `_slate.*` | built only when a `gpu_backend=cuda` SLATE is found; the `lorrax_A` bundle has none |

The ctypes C entry points: the cuSOLVERMp/NCCL grid context
(`cusolvermp/c_api.cc`), the MPI mesh context SLATE and ScaLAPACK share
(`slate/context.cc`), the phdf5 lifecycle (`phdf5/api.cc`), the workspace
queries (`lrx_eigh_workspace_bytes`, `lrx_gemm_workspace_bytes`,
`lrx_active_eigh_lwork`) and the two stamps (§2). The host leg's copies end
in `_host` (`common/c_abi.h`).

### Small CUDA kernels

* **Contour accumulator.** `A[o, q, m, n] += p[o]·c[q, m, n]`, complex128,
  in place (operand 0 aliased), the accumulator at `P(None, None, 'x', 'y')`
  and the correlation at `P(None, 'x', 'y')`. Each multiply and add is
  rounded separately (`__dmul_rn`, `__dadd_rn`), so the result equals the
  XLA stream it replaces byte for byte. 128 threads, four outputs per thread;
  more than `4·65535` outputs refuse.
* **Spin rotation.** `G ← U_k G U_k†` on each `(k, μ, ν)` spin block,
  `ns ∈ {2, 4}`, complex128, in place, 128 threads; any other spin width or
  dtype takes the JAX einsums.
* **Fused W-solve.** `W = X (I − X† pref·χ X)⁻¹ X†` with `X X† = V`, per q,
  in one handler: potrf, cuBLASMp GEMMs, the `I − T` kernel, potrf, two trsm
  and a GEMM (16×16-thread helper kernels, `gridDim.z = n_q ≤ 65535`).

---

## 1. The layers

Five, outermost first. Each one can refuse; none silently substitutes.

| # | Layer | Lives in | Job |
|---|---|---|---|
| 1 | **Consumer** | `src/gw/`, `src/file_io/`, `src/bse/`, … | states *logical* intent: shapes, not strides |
| 2 | **Service facade** (Python) | `src/ffi/io.py`, `fft.py`, `gemm.py`; `services/distrib_la/` | owns the call grammar, builds the descriptor, picks the backend |
| 3 | **Gate** | `src/ffi/gate.py` (env dials); `distrib_la.resolve` (deck choices); `ffi.fft.require_kconv` (router) | announce-or-refuse: an explicit request that cannot be honoured refuses, never downgrades |
| 4 | **XLA FFI custom call** | `src/ffi/common/ffi_loader.py`, `distrib_la.loader`, both over `lxkit.native_provider` | locates and attests the `.so`, registers its handler symbols |
| 5 | **C++ handler + vendor library** | `src/ffi/cpp/<vendor>/` | the MPI-IO / BLAS / FFT / solver call |

**A vendor dependency enters only through a facade with run-time resolution
and an announced refusal.** The FFI is required
([`decisions.md`](decisions.md), 2026-08-01): a missing library is a startup
refusal naming the `.so`, never a demotion to a Python path. Portability
fallbacks *inside* a handler (the FFTW3 `dlsym` ladder, batched versus plain
CBLAS) are how the required layer stays buildable everywhere; each announces
which entry it bound.

### Python-side module map

* **Real modules:** `ffi/io.py` (parallel HDF5), `ffi/fft.py` (the host
  flat-k FFT gate, the [k-convolution router](#k-convolution-router-and-the-mathdx-family),
  the plane door and the Fourier-plan call), `ffi/gemm.py` (host batched
  GEMM), `ffi/gate.py`, `ffi/common/ffi_loader.py`, and `ffi/cublasmp/`
  (cuBLASMp GEMM and the fused W-solve; reached by the bench drivers and one
  `distrib_la` contract cell, by no production path). Distributed dense
  linear algebra and the active subspace are `services/distrib_la`, which
  opens the same two libraries through its own `distrib_la.loader`.
* **Target table only:** `ffi/cufft/` repeats the `lorrax_mathdx_*` rows of
  `ffi_loader._CUDA_TARGET_SYMBOLS`. Nothing imports it.
* **Re-export shims:** `ffi.phdf5` → `ffi.io`, `ffi.mklfft` → `ffi.fft`,
  `ffi.mklblas` → `ffi.gemm`, `ffi.cusolvermp` → `distrib_la._cusolvermp`.
  New code imports the real module. A shim is deleted when
  `git grep -nE "ffi\.<shim>" -- src services tests ':!src/ffi'` is empty.

---

## 2. The one C++ tree

`src/ffi/cpp/`: one `CMakeLists.txt`, both platform legs behind an explicit
selector.

```
src/ffi/cpp/
├── CMakeLists.txt           -DLORRAX_FFI_PLATFORM=cuda|host; FATAL_ERROR when unset
├── build.sh  build_host.sh  leg build scripts (container CUDA leg; generic host leg)
├── exports_cuda.map  exports_host.map     version scripts: LORRAX internals are local
├── gate_one_mpi.sh  gate_one_hdf5.sh  gate_one_fftw.sh  gate_one_odr.py
├── run_shifter.sh  in_container.sh  select_gpu.sh    container composition
├── stage/                   vendor stage scripts, seal_bundle.py, stamp_provenance.sh
├── common/                  plumbing every family shares: C ABI naming, ABI and build-config
│                            stamps, the ctx registry, env/announce and thread-pin helpers,
│                            the NVRTC build service and the async-gather device header
├── phdf5/  slate/                                    both legs
├── fftw/  cblas/  scalapack/                         host leg
└── cusolvermp/  cublasmp/  cublas/  cufft/  active_subspace/  response/  symmetry/   CUDA leg
```

`cufft/` holds the mathdx k-convolution family and the Fourier plan (cuFFT's
one caller). `slate/context.cc` is the MPI mesh context SLATE and ScaLAPACK
share. The host targets keep their historical `lorrax_mklfft_*` /
`lorrax_mklblas_*` spellings. Basenames repeat across directories
(`api.cc`, `context.cc`, `ctx.h`, `eigh_ffi.cc`, `batched_potrf_ffi.cc`), so
includes name the directory.

| leg | CMake | library | feature options (all default `ON`) |
|---|---|---|---|
| host | `-DLORRAX_FFI_PLATFORM=host` | `liblorrax_ffi_host.so` | `LORRAX_FFI_HAVE_PHDF5`, `LORRAX_HOST_HAVE_SCALAPACK`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_HOST_HAVE_FFTW3` |
| CUDA | `-DLORRAX_FFI_PLATFORM=cuda` | `liblorrax_ffi.so` | `LORRAX_FFI_HAVE_CAL`, `LORRAX_FFI_HAVE_PHDF5`; SLATE is probed at `LORRAX_SLATE_INSTALL_DIR` |
| unset | — | `FATAL_ERROR` naming both legs | — |

* **Host leg.** CUDA-free by construction: its sources compile with
  `LORRAX_FFI_NO_CUDA`, and GATE 3 refuses a CUDA-stack `DT_NEEDED`. A feature
  group whose dependency is absent is skipped with a `STATUS` line. The FFT
  handler is always built (it declares the FFTW3 ABI itself);
  `LORRAX_HOST_HAVE_FFTW3` only records a found FFTW3 as the run-time
  `dlopen` hint `LORRAX_FFTW3_SO_HINT` and never links it. The GEMM handler is
  built when a CBLAS header and provider are found. The link uses
  `-Wl,--no-undefined`.
* **CUDA leg.** The complete NVIDIA stack (owner, 2026-09-25): nvcc,
  cuBLAS, cuBLASMp, cuSOLVERMp, cuSOLVER, cuFFT, NVRTC and NCCL, always, with
  every CUDA family of the [catalog](#kernel-catalog); nvidia-mathdx is a
  run-time wheel. `-DLORRAX_FFI_HAVE_CUBLASMP=OFF` or
  `-DLORRAX_FFI_HAVE_CUFFT=OFF` refuses at configure, and so does a missing
  cuFFT or NVRTC. The nvcc TUs compile for `CMAKE_CUDA_ARCHITECTURES`,
  default `80;86-real;89-real;90-real;100-real;120` (SASS for each, PTX for
  compute_80 and compute_120). `LORRAX_FFI_HAVE_CAL` is §4.
* **Both legs.** `$ORIGIN` is first in `RPATH`. `exports_{cuda,host}.map`
  localise every LORRAX-owned symbol, and the host leg's C entry points carry
  a `_host` suffix (`cpp/common/c_abi.h`), so the two libraries define no
  LORRAX name in common when both are `dlopen`ed `RTLD_GLOBAL`. Each leg
  exports `lorrax_ffi_{cuda,host}_abi_version` and
  `lorrax_ffi_{cuda,host}_build_config`.

`LORRAX_FFI_PLATFORM` is a CMake cache variable; no Python module reads it.

### 2a. Build entry points

| machine | leg | entry |
|---|---|---|
| Perlmutter `lorrax_A` (bare-host CUDA 13) | CUDA | the runtime's recipe outside this repository: `-DLORRAX_FFI_HAVE_CAL=OFF` against the NCCL-native cuSOLVERMp, no device SLATE, MPI from `config/perlmutter/ffi_mpi.sh` |
| Perlmutter | host | `config/perlmutter/build_ffi_host.sh`, bare metal: pins MPI through `ffi_mpi.sh`, loads `cray-hdf5-parallel/1.14.3.7`, captures the cray-fftw path as the hint, and unloads `cray-libsci`, `cray-fftw`, `craype-accel-nvidia80` and `cudatoolkit` before configure |
| Shifter sites | CUDA | `src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh` |
| Frontera | host / CUDA | `config/frontera/build_ffi_host.sh` / `config/frontera/build_ffi.sh` |
| anywhere else | host | `bash src/ffi/cpp/build_host.sh` |

`config/perlmutter/ffi_mpi.sh` pins the one MPI both Perlmutter legs link:
`cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`, the MPI the phdf5 stage and the
SLATE host install need) and the LibSci that links it,
`cray-libsci/25.09.0`. Both legs are loaded into one GPU process, so they
must name the same `libmpi`. To move to another MPI, change that file only.

### 2b. The build contract: `scripts/verify_ffi_build.sh`

Every build path ends in `scripts/verify_ffi_build.sh [--leg host|cuda]
<so>`. [`../building_ffi.md`](../building_ffi.md#the-verify-contract) owns the
gate list and the acceptance test; the invariant each gate checks:

| gate | invariant | where it can run |
|---|---|---|
| 0 | every backend in `LORRAX_FFI_EXPECT_BACKENDS` exports a handler and the build stamp agrees. The default is the leg's full set, so a build that lost one fails | anywhere |
| 1 | one MPI runtime in the closure (`gate_one_mpi.sh`: `ldd`, deduplicated by `realpath`) | the run environment |
| 2 | one BLAS vendor and one threading flavour in `DT_NEEDED` | anywhere |
| 3 | the host leg links nothing from the CUDA stack | anywhere |
| 4 | the closure resolves (`ldd -r`) | the run environment |
| 5 | zero undefined `fftw_` symbols and zero `fftw` in `DT_NEEDED` | anywhere |
| 6 | every OpenMP entry in `DT_NEEDED` is `libgomp`, `libiomp5` or `libomp` | anywhere |
| 7 | one HDF5 SOVERSION, and the runtime provides it (`gate_one_hdf5.sh`; `LORRAX_FFI_EXPECT_HDF5_SOVERSION`, `LORRAX_FFI_EXPECT_PEER_SO` for the cross-leg check) | ELF halves anywhere; mapped-object half in the run environment |
| 8 | after one real FFT, exactly one FFTW3 engine is mapped and it is the staged one (`gate_one_fftw.sh`) | host leg, in a process that imports jax (`LORRAX_GATE_FFTW_PY` or `LORRAX_FFTW3_STAGE`) |
| 9 | no LORRAX internal on the dynamic table; every host `lrx_*` entry is suffixed | `build.sh`, `config/perlmutter/build_ffi_host.sh` |
| 10 | a CUDA-capable process with both libraries open completes a host phdf5 read (`gate_one_odr.py`) | a GPU node, both pins set |
| 11 | the exported ABI equals `src/ffi/cpp/common/lorrax_ffi_abi.h` | anywhere |

A gate that cannot run in the current environment prints `GATE COULD NOT
RUN` and is counted apart from passes. `LORRAX_FFI_VERIFY_STRICT=1` makes it
a failure; use it for certification inside an allocation.
`LORRAX_FFI_VERIFY=off` disables the verifier with an announcement, and a
library built that way is not certifiable.

### 2c. The deployable unit is one sealed pair

The two legs become one production provider only through
`src/ffi/cpp/stage/seal_bundle.py`. It publishes a new, non-overwriting
directory holding both libraries under `lib/`, the listed private
redistributables, and one `lorrax_ffi_bundle.json`. The manifest binds the
pair, the handler ABI, the full source revision, each file's size and SHA-256,
each ELF SONAME and `DT_NEEDED` record, and the dependency-first private
closure.

Only cuSOLVERMp, cuBLASMp, CAL, SLATE (and its ScaLAPACK API), BLAS++,
LAPACK++ and NVSHMEM may be sealed as private libraries. MPI, site HDF5, the
system and compiler runtimes, CUDA runtime and driver libraries, and NCCL
belong to the machine runtime; `seal_bundle.py` refuses them as private
inputs. At load,
`lxkit.native_provider` rehashes both legs and the closure, preloads each
private library by exact path (so no run script owns a library search path),
checks the live ABI symbol's origin with `dladdr`, and refuses any mapped
engine-private provider the manifest does not name.

`source.revision` is provenance, not a demand that the active checkout have
the same SHA. Compatibility is the exported ABI, the live feature and target
probes, and the build contract. A different ABI always refuses. An unsealed
build-tree library still loads and prints `LEGACY-UNSEALED` with its hash; it
is not production attestation. The Perlmutter `lorrax_A` module selects one
sealed bundle through `LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`.

### 2d. How the loader selects a library

`ffi_loader.get_lib(platform)` (and `distrib_la.loader`, through the same
`lxkit.native_provider` policy):

* **Candidates**, in order: the pin (`LORRAX_FFI_SO` for CUDA,
  `LORRAX_FFI_HOST_SO` for cpu), the in-tree `src/ffi/cpp/build/` or
  `build_host/`, then each `sys.path` directory.
* **Pins.** A pin that is not a file refuses. When a selected leg belongs to
  a sealed bundle, both pins or neither must be set, and both legs must come
  from one manifest; a partial override or mixed providers refuse.
* **ABI.** A stamped library with a different ABI refuses (`FfiAbiMismatch`).
  An unstamped one is announced once and loads, unless
  `LORRAX_FFI_ABI_STRICT=1`.
* **Load order.** In a CUDA-capable process (the first `JAX_PLATFORMS`
  entry, if set, is `cuda` or `gpu`; `CUDA_VISIBLE_DEVICES` is not empty; an
  NVIDIA device node is visible), opening the host library opens the CUDA
  library first, so the CUDA build wins the shared SLATE/BLAS++ SONAMEs.
  After each `dlopen` the loader refuses a process with more than one mapped
  MPI runtime.
* **Probe.** `probe_target(target, platform)` gives one of three reasons:
  unknown target, library could not be loaded, or library loaded but does not
  export the handler. Every gate refusal quotes it.

---

## 3. What each machine provides

The [operations table](#kernel-operations) owns which engine serves each
operation on each hardware class, and its gate. This section owns which
library satisfies that engine on each machine. Versions and launch recipes:
[Perlmutter](../environment/machines/perlmutter.md),
[Frontera](../environment/machines/frontera.md).

### 3a. The dependency matrix

One row per dependency LORRAX does not implement itself: who serves it on
each machine, what else could, and what proves it built right.

| Routine | Perlmutter | Frontera | Alternatives | Check |
|---|---|---|---|---|
| **Flat-k FFT, host leg** | FFTW3 ABI by `dlsym` → cray-fftw | MKL's FFTW3 export, already resident | the candidate ladder (§3c) | GATE 5 (load time), GATE 8 (engine identity) |
| **Host band-block GEMM** (`ffi.gemm`) | LibSci CBLAS; LibSci has no `cblas_?gemm_batch`, so the handler loops plain `cblas_?gemm` | MKL CBLAS, batched entry | any CBLAS provider (MKL, LibSci, OpenBLAS, BLIS); the chosen entry is announced at first use | GATE 2 |
| **Distributed dense solvers** | cuSOLVERMp and cuBLASMp (GPU); ScaLAPACK from LibSci and host SLATE (CPU) | ScaLAPACK from MKL; host SLATE | — | — |
| **Distributed transport** | NCCL for cuSOLVERMp (§4) and cuBLASMp; an `MPI_COMM_WORLD` split in mesh order for SLATE and ScaLAPACK | Intel MPI | — | the `[lorrax cusolverMp] … comm path:` banner |
| **Parallel HDF5** | `cray-hdf5-parallel/1.14.3.7` (`libhdf5_parallel_gnu.so.310`) over `cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`) | phdf5 over Intel MPI | none: one transport | GATE 1, GATE 7 |
| **OpenMP runtime** | `libgomp` | `libiomp5` | `libgomp`, `libiomp5`, `libomp` | GATE 6 |
| **Runtime** | bare-host CUDA 13 (`lorrax_A`); Shifter stages for container sites | apptainer | — | — |

### 3c. Which FFT engine the host library binds

One source, `cpp/fftw/fft_flat_k_ffi.cc`, serves every host FFT through the
FFTW3 advanced interface (`fftw_plan_many_dft`). No FFTW symbol binds at link
time. The engine is resolved at first use:

1. **Already loaded.** `resolve_sym` (`RTLD_DEFAULT`, then `RTLD_NEXT`). On an
   MKL site the ScaLAPACK link line has already loaded MKL's FFTW3 export, so
   the ladder never runs.
2. **The candidate ladder**, `dlopen(RTLD_GLOBAL)`, first hit wins:
   `$LORRAX_FFTW3_SO` → the build's `LORRAX_FFTW3_SO_HINT` (a compile-time
   path) → `libfftw3.so.3` → `libfftw3.so.mpi31.3` → `libmkl_rt.so` →
   `libfftw3.so`.
3. **Refusal.** The handler returns `mklfft: no FFTW3 engine in this
   process`, naming every candidate tried. The startup gate probes only the
   exported handler symbol, so this error arrives at the first host FFT.

On the Perlmutter bare host the hint is cray-fftw's own path. In a Shifter
container `/opt/cray/pe` does not exist, so the engine comes from the
`/lorrax_fftw` stage (`stage/fftw_stage_cray.sh`, mounted by
`run_shifter.sh`).

**Hazard.** CUDA images ship `libcufftw.so`, which exports all three entry
points the ladder binds. Pointing `LORRAX_FFTW3_SO` at it makes the host
handler transform on the GPU, and every FFT check still passes. GATE 8 is
the only check that tells these states apart.

An engine swap is accepted at value-level parity, relative 1e-12 on the Σ
path, never bit-exactness: engines differ in arithmetic order.

---

## 4. The cuSOLVERMp version picks the communication path

cuSOLVERMp changed its grid communicator from CAL (≤ 0.6.x) to NCCL
(≥ 0.7.0). The handler reads the loaded version with `cusolverMpGetVersion`
and selects the path at context creation (`cusolvermp/context.cc`); rank 0
prints `[lorrax cusolverMp] library X.Y.Z, NCCL …, comm path: NCCL|CAL`.

| cuSOLVERMp | ships `cal.h` / `libcal` | comm path | build flag |
|---|---|---|---|
| ≤ 0.6.x | yes | CAL | `-DLORRAX_FFI_HAVE_CAL=ON` (the CMake default) |
| ≥ 0.7.0 | no | NCCL | `-DLORRAX_FFI_HAVE_CAL=OFF` |

* **Every version exports the SONAME `libcusolverMp.so.0`.** A `.so` built
  against one version loads against another without complaint. The run's
  search path must therefore hold exactly one cuSOLVERMp: the sealed bundle
  loads its own by exact path, and `run_shifter.sh` puts exactly one NVHPC
  stage on `LD_LIBRARY_PATH`.
* **A `HAVE_CAL=OFF` build refuses a pre-0.7 library** at context creation. A
  `HAVE_CAL=ON` build carries both paths and `DT_NEEDED` `libcal.so.0`.
* **0.6.x is wrong on a 2-D grid.** With `Px > 1` and `Py > 1` its
  `getrf`/`getrs` return wrong answers, so context creation refuses that
  pairing (`GATE cusolvermp_2d_grid_version`). LORRAX meshes are square, so a
  `HAVE_CAL=ON` build with a pre-0.7 library runs only at P = 1.
* **≥ 0.8 needs NCCL ≥ 2.27** (`ncclCommWindowRegister`); the handler warns
  when the loaded NCCL is older.

**Container stages.** `build.sh` refuses (exit 2) when neither
`LORRAX_NVHPC_ROOT` nor `LORRAX_NVHPC_SUBPATH` names a stage, listing the
stages under `/lorrax_nvhpc` with the flag each needs. It also refuses a stage
without `cal.h` unless `LORRAX_FFI_HAVE_CAL` is set explicitly. The single
source of truth is `LORRAX_NVHPC_SUBPATH` (`config/perlmutter/site_config.sh`,
default `0.7.2_cuda12.9/math_libs/12.9/lib64`). `run_shifter.sh` exports it
with `LORRAX_NVHPC_ROOT` derived from its first component, so a build launched
there agrees with its runs.

---

## 5. Parallel HDF5: the FFI side

[`slab_io.md`](slab_io.md) owns the subsystem: the tile contract, the
launcher requirements, striping, certification, the one-owner-per-file rule
and the measured failure signatures. This section holds only the FFI facts.

* **One transport, no router.** `file_io.slab_io` takes a path, a mode and a
  mesh; a deployment that cannot serve the tile path refuses at open, naming
  the probe that declined ([availability](slab_io.md#availability)).
* **One C++ source, both legs.** The `phdf5/` sources compile into both
  libraries. On the host leg the device staging collapses: the read tail is a
  `memcpy` into the host XLA buffer and the write hands `H5Dwrite` the XLA
  buffer directly. The control-operand stream race
  ([`slab_io.md`](slab_io.md#stream-race)) is therefore CUDA-leg only.
* **The legs' entry points are not interchangeable.** The host leg's C entry
  points end in `_host`, and each leg localises its internals, because one
  `PhdfCtx` name has two struct layouts. A library built without them exports
  both layouts under one name, and the first-loaded library answers for both;
  GATE 9 and GATE 10 catch it.
* **`ffi.io.open_file(path, *, mesh, mode)`** picks the library from the
  mesh's devices and records it per handle, so `close_file` returns through
  the opening library. `mode` has no default. It refuses a mode outside
  `{w, a, r}`, a mesh without both `x` and `y`, and
  `p·q ≠ jax.process_count()`. An already-open path may be opened again only
  when both opens are read-only on the same platform and mesh; they then share
  one native context.

---

## 6. phdf5 defaults

The struct initialisers in `phdf5/ctx.h` are not the effective defaults:
`open_ctx_impl` in `phdf5/context.cc` reassigns every field from the
environment when a file is opened.

| Field | Effective default | Override | Role |
|---|---|---|---|
| `use_collective_read` | `true` | `LORRAX_PHDF5_INDEPENDENT=1` → independent **reads** | tuning; a band-block `read_slabs` is independent regardless ([`slab_io.md`](slab_io.md#tuning)) |
| `use_collective_write` | `true` | `LORRAX_PHDF5_COLLECTIVE_WRITES=0` → independent writes | correctness (§7b) |
| `coll_metadata` | `false` | `LORRAX_PHDF5_COLL_META=1` | non-collective metadata keeps `H5Dcreate`/extend off the collective driver |
| `dedup_replicas` | `true` | `LORRAX_PHDF5_DEDUP_REPLICAS=0` | correctness: one writer per replica group; overlapping selections are undefined under collective writes |
| `align_threshold`, `align_length` | 4 MiB (header: 1 MiB) | `LORRAX_PHDF5_ALIGN_MB` | tuning; independent of the stripe unit |

**Boolean grammar.** `env_flag` (`phdf5/ctx.h`): unset or empty → the
default; otherwise trimmed and lower-cased, and true only for `1`, `true`,
`yes`, `on`. Any other value is false, silently: `=ture` turns collective
writes off. The Python twin, `runtime.env_flags.env_bool`, accepts the same
table but announces an unrecognised value; `tests/test_env_grammar.py` holds
the two in step. Collective-buffering and stripe knobs are in
[`slab_io.md`](slab_io.md#tuning).

---

## 7. Failure modes, and how to tell them apart

### 7a. The PMI-flavour mismatch gives wrong answers

Launched with the wrong PMI for Cray MPICH (`srun --mpi=pmi2` instead of
`cray_shasta`), every rank gets a private singleton `MPI_COMM_WORLD`:
`MPI_Comm_size == 1` while `jax.process_count() == P`. The native checks
cannot see it: `ffi.io.open_file` checks `p·q == jax.process_count()`, and
`shard_index.h::validate_shard_encoding` checks `prod(mesh_shape) ==
ctx->world_size`, where `world_size` *is* `jax.process_count()`. Both compare
JAX to JAX. With disjoint hyperslabs the write completes bit-exact at rc = 0
(sandbox CLAIMS 68); two ranks on one chunk would corrupt silently.

**The guard** (`file_io/_slab_io_ffi._assert_mpi_world`) asks MPI once, at
the first collective open, and compares `MPI_Comm_size(MPI_COMM_WORLD)` with
`jax.process_count()`. The verdict is rank-invariant, so it refuses on every
rank or none.

* A mismatch always refuses. Fix the launcher.
* An MPI world that cannot be probed refuses by default;
  `LORRAX_PHDF5_REQUIRE_MPI_WORLD=0` downgrades that case to a rank-0
  warning.
* `LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK=1` removes the guard. It is a debugging
  escape, never a remedy.

### 7b. The ROMIO collective-buffer OOM

Cray MPICH's collective write can exhaust memory at large per-rank aggregates:

```
Out of memory in .../ad_cray/ad_cray_write_coll.c, line 669
… MPI_Abort … "HDF5: infinite loop closing library"
```

The same line appears when the PMI flavour is wrong (§7a) and collective
writes are on. The two want opposite remedies:

| | genuine collective-buffer OOM | PMI-flavour mismatch |
|---|---|---|
| `MPI_Comm_size(MPI_COMM_WORLD)` | `== jax.process_count()` | `1` on every rank |
| per-rank aggregate | ≳ 1 GB | any |
| `LORRAX_PHDF5_COLLECTIVE_WRITES=0` | fixes it | hides it: silent wrong answers (§7a) |

Check the world size first. `LORRAX_PHDF5_INDEPENDENT=1` changes reads only
and does nothing for a write-side OOM.

### 7c. SONAME aliases that look like two MPIs

In the Shifter container, `stage/phdf5_stage_cray.sh` creates one symlink per
Cray compiler-specific SONAME, `libmpi_gnu_{91,110,123}.so.12`, all pointing
at the container's generic MPICH-ABI `/opt/udiImage/modules/mpich/libmpi.so.12`
(`SHIM_TARGET`). Every variant is one object. On a login node the closure is
incomplete and `ldd` reports several dependencies `not found`, which proves
nothing. Check a library's closure where it runs: inside the container on a
compute node, where `gate_one_mpi.sh` (GATE 1) deduplicates by `realpath`.

### 7d. Bounds-check asymmetry hangs with no traceback

Bounds are tested once, on the logical slab `offset + valid_shape`, which is
replicated, so every rank reaches the same verdict. A test on a rank-local
offset splits the ranks into those that refuse and those that enter the
collective, and the communicator hangs with no HDF5 error. No rank may skip a
collective because of its own error: record it, take part in the teardown,
then raise ([`decisions.md`](decisions.md), 2026-08-04).

---

## 8. Hard invariants

1. **Registered FFI target names and C++ handler symbols do not change.** The
   sets are `_CUDA_TARGET_SYMBOLS` and `_HOST_TARGET_SYMBOLS` in
   `ffi/common/ffi_loader.py`, `distrib_la.loader`'s tables and the spin
   rotation's own registration. Refactors move files, never a target string;
   a changed operand contract is a new target, and the old one stays for
   older trees.
2. **Env knob spellings do not change.** Add an alias instead.
3. **Library names are `liblorrax_ffi.so` and `liblorrax_ffi_host.so`.** A
   change updates every consumer in the same commit.
4. **The two legs share no LORRAX-owned dynamic symbol** (GATE 9, GATE 10).
5. **A stage or build script refuses an unstated environment fact rather than
   guessing it.** `phdf5_stage_cray.sh` refuses an unset `HDF5_DIR` or
   `MPICH_DIR`; `build.sh` refuses an unstated cuSOLVERMp stage, a CAL
   mismatch, and unset `LORRAX_MPI_INCLUDE_DIR` / `LORRAX_MPICH_LIB_DIR`
   (CMake would otherwise fall back to HPC-X Open MPI). What is staged is what every later build links, and a wrong guess
   surfaces much later as a wrong answer or a hang.
6. **A handler ABI change bumps `src/ffi/cpp/common/lorrax_ffi_abi.h`** and
   its mirror `ffi_loader.LORRAX_FFI_ABI_VERSION` together
   (`tests/test_ffi_abi_stamp.py`), and old bundles then refuse.

---

## k-convolution router and the mathdx family

Every k-axis convolution and every k-axis transform in the physics is requested
through one factory in `ffi/fft.py` (re-exported by `common.fft_helpers`). The
factory picks the backend from the mesh platform and nothing else; no
environment variable or deck key selects a route (ruling:
[`decisions.md`](decisions.md), 2026-09-24).

| platform | backend |
|---|---|
| CUDA | nvidia-mathdx: cuFFTDx thread FFTs inside one fused shared-memory pass per k-row, compiled by NVRTC per k-grid |
| cpu | the FFTW3-ABI host plan handlers, composed with XLA elementwise work |
| other | refusal, `GATE kconv-platform` |

Both legs return the same callable contract, so a consumer never branches on
the backend.

**Why this way.** The pair convolution `U_q = Σ_k conj(A_k)·B_{k+q}` over
`N_k` points costs `O(rows·N_k²)` as a direct sum and `O(rows·N_k log N_k)` as
`U = s·FFT_k[conj(IFFT_k A)·IFFT_k B]`. The CUDA kernel keeps each k-row in
shared memory from the inverse transforms through the product (or spin
contraction) to the forward transform, so one convolution reads each operand
from HBM once and writes the result once. The line FFTs are the library's,
specialised per grid when NVRTC compiles the kernel.

| Layer | What |
|---|---|
| 1 consumer | ζ fit (`isdf.core`, `isdf.zeta_mubatch`, `isdf.pair_kernels`, `gw.centroid_k_unfold`), Σ and COHSEX (`gw.ppm_tau_kernel`, `gw.cohsex_sigma`, `gw.screening`), χ₀ (`gw.w_isdf`), BSE (`bse.*`), the real-space pair convolution (`gw.mixed_basis_pair_convolution`, mode 6; unwired), the flat-k transform (`common.fft_helpers.make_flat_k_fft` and its `make_flat_k_ifftn` / `make_flat_k_fftn` / `make_local_flat_k_fftn` wrappers: `gw.w_isdf`, `gw.qsgw_head`, `gw.wavefunction_bundle`, `bandstructure.htransform`, `bandstructure.orbital`). Every door name is unique: `git grep -n <door>` lists its call sites |
| 2 router | `ffi/fft.py`: the doors below. `common.fft_helpers` re-exports them; its `get_donated_kfft_kminor` is `make_kfft_kminor` jitted with its input donated, memoised per `(mesh, kgrid, spec, kind, norm)`, and the caller drops its own reference after the call |
| 3 gate | `require_kconv`, then `require_fourier_plan`, called by `runtime.initialize_communicator_stack` after the FFT and GEMM gates. `require_kconv` on CUDA: the wheel's headers, every `ffi.fft.KCONV_TARGETS` target, and one probe compile (mode 3, k-grid 2×1×1, disk-cached), so a device the installed cuFFTDx cannot compile for refuses at startup (`GATE mathdx-probe`, naming its compute capability and the wheel); cpu: `lorrax_mklfft_flat_k`. `require_fourier_plan` on CUDA: `lorrax_fourier_plan_mathdx` only, since no wheel version is checked at startup (the fused pair checks its wheel at plan build); cpu: nothing (XLA ops). Each factory re-probes its own target (a `LocalFourierPlan` that CUDA can lower probes `lorrax_fourier_plan_mathdx` at construction); operand shapes and dtypes are checked at trace time |
| 4 target | CUDA, in `liblorrax_ffi.so`: the `lorrax_mathdx_*` target of each mode in the mode table below, and four kept for older source trees (`_kconv_klead_unfold_rows`, `_kconv_klead_lorentz_rows`: modes 7 and 8 without the conj-on-load partner and the output spin block; `_kconv_klead_unfold`, `_kconv_klead_lorentz`: every k row stored). `lorrax_fourier_plan_mathdx` and, for older trees, `lorrax_fourier_plan` ([§ Local Fourier plan](#local-fourier-plan-localfourierplan)). cpu, in `liblorrax_ffi_host.so`: `lorrax_mklfft_flat_k`, `lorrax_mklfft_gw_conv` (§3c) |
| 5 handler | CUDA: `cpp/cufft/kconv_mathdx_cuda_ffi.cc`. Modes 0–9 and 11 share one embedded cuFFTDx source (`kSrc`); mode 10 has its own (`kPlaneSrc`); mode 11 also embeds `cufft/kbox_stage.cuh` as the named header `kbox_stage.cuh`, turned into text at configure time (`kbox_stage_src.h.in`). NVRTC compiles an image for the device's own `sm_<cc>` per (CUDA context, mode, `nkx`, `nky`, `nkz`, `ns`, right width, precision, variant) into an in-process cache, backed by the disk cubin cache below. cpu: `cpp/fftw/fft_flat_k_ffi.cc` (`MklFftFlatKHostFfi`, `MklFftGwConvHostFfi`) |

**Doors.** Pick the door whose k position matches the tile you hold. A caller
never transposes to reach another door.

| door | k axis of the operand | CUDA mode | cpu leg |
|---|---|---|---|
| `make_fused_conv_kpair` | 3-D leading `(nkx, nky, nkz, …)` | 0 | two host flat-k inverse transforms, the spin contraction in XLA, one host forward transform |
| `make_fused_conv_kparent` | parent tables | 1 | the typed parent load in XLA, which materialises `(N_k, ns, μ, ν, ns)` per side, then the pair tail |
| `make_fused_conv_kplane` | route-G plane output `(N_k, g, ns, 2c, ns, p)` | 6 | the Bloch phase, split and transpose in XLA, then the pair tail |
| `make_kconv_klead` | flat leading `(N_k, …)` | `prep` 3, `apply` 2 | `prep` is the identity; `apply` is `lorrax_mklfft_gw_conv`, which transforms W itself and holds the R-space T tile only in per-thread compact chunks |
| `make_kconv_klead_unfold` | raw-parent Green `(n_parent, μ, ns, ν, ns)`; output `(len(store_rows), d, μ, d, ν)`, one `d × d` output spin block per call (`spin_block`, default `d = ns`), every other k row transformed and never stored; `conj_partner=True` reads the antiunitary partner as `conj(G)` on the load | 7 (`prep` of `make_kconv_klead`) | `symmetry_maps.apply_unfold_load_tables_local` in XLA (a full-k copy), then the `make_kconv_klead` apply and the row selection |
| `make_kconv_lorentz_unfold` | the same, with the Lorentz blocks `V (N_k, μ, n_A, ν, n_B)` | 8 | the same composition, the γ̃ block sum in XLA, the row selection |
| `make_kfft_klead_unfold` | an interaction on its q wedge `(n_wedge, μ n_l, ν n_r)` (`symmetry_maps.QirrOperator.values`), its partner tile on the pair-transpose rule; output the R-space operand `(N_k, μ n_l, ν n_r)` that `make_kconv_klead`'s prep makes from the full zone | 9 | `apply_unfold_load_tables_local` in XLA (conj or pair-transpose rule, left/right endpoint actions), then the `make_kconv_klead` prep |
| `make_kconv_chi_unfold` | the raw-parent Green pair `Gv`, `Gc` `(n_parent, μ, ns, ν, ns)` and their partners (or `conj(G)` on the load); `acc[o] += α_o Σ_ab conj(Gc'_ab) Gv'_ab` in R space (+ c.c. on a real contour), `acc (n_out, N_k, μ, ν)` in place, `G' = ifftn` of the typed unfold: one τ node of χ₀ with no full-k Green | 11 (k-box stage: one pass, or plane + group-pencil passes chunked over pairs) | the unfold in XLA per operand, the plan-route inverse transform, the spin trace in XLA |
| `make_kfft_klead` | flat leading `(N_k, …)` | 3 | `lorrax_mklfft_flat_k` |
| `make_kconv_kminor` | flat trailing `(…, N_k)` | 4 | XLA moves k to the front, then host inverse transform, product, host forward transform, and k moves back |
| `make_kfft_kminor` | 3-D trailing `(…, nkx, nky, nkz)` | 5 | the same transpose around one host transform |
| `common.fourier_plan.LocalFourierPlan` | none: ≤ 3 spatial axes with per-axis supports; the entry point for sphere↔box and plane transforms (§ Local Fourier plan) | `lorrax_fourier_plan_mathdx`; its `in_gather` form is mode 10 | XLA ops: `dot_general` GEMM axes and one `jnp.fft` group |
| `make_plane_fft_gather` | the backend of `LocalFourierPlan(in_gather=…)`: the route-G cylinder `(…, n_col)` → the transformed plane `(…, n_b, n_c)` | 10 | the XLA route: static-run concatenate, then `jnp.fft.fftn` |

- **Sharding.** The pair, parent, plane and `make_local_*` doors are rank-local
  callables for use inside the caller's `shard_map`; the others wrap their own
  `shard_map`. The k axes are replicated. Specs of the k-leading doors are
  given in the 3-D form, with the three leading axes `None`. For
  `make_kconv_kminor`, `K_R`'s `(d1, d2)` must sit on the same mesh axes as
  X's.
- **Scale.** Every handler takes one total scale `s`, computed in Python from
  `jnp.fft`'s norm conventions (`ffi_fft_scale`, `conv_kpair_scale`). The
  handlers implement no norm of their own. The parent door fixes
  `norm="forward"`, so `s = 1/N_k`.
- **`KConvStored`.** `prep(W)` does everything that depends on W alone, once
  per W. `apply(T, W_prep)` does the rest, once per T. `W_prep` is in the
  backend's own form (R space on CUDA, W unchanged on cpu), so pass it only to
  the `apply` of the same pair.
- **Vertex attributes (modes 0 and 1).** `perm_l`, `perm_r` are permutations
  of `range(ns)` and `phase_l`, `phase_r` are exact monomials in
  `{+1, +i, −1, −i}`; the CUDA leg refuses anything else at factory time.
  `ns ≤ 4`.

The modes of the one handler file. The target column is the string
`ffi_loader._CUDA_TARGET_SYMBOLS` maps to the C++ symbol.

| mode | target (`lorrax_mathdx_…`) | operation | layout | resident banks per k-row |
|---|---|---|---|---|
| 0 pair | `kconv_pair` | `U = s·FFT_k Σ_ab φ_l[a]φ_r[b]·conj(IFFT_k A[…,a,…,b])·IFFT_k B[…,π_l a,…,π_r b]` | `A`, `B` `(nkx,nky,nkz, ns, col, μ, ns)` → `U` `(nkx,nky,nkz, col, μ)` | 3 |
| 1 parent | `kconv_parent` | mode 0 on the typed parent load ([below](#parent-load-isdf-pair-convolution-mode-1)) | `D_l`, `D_r` `(n_parent, ns, μ, ns, ν)` and ten tables → `U` `(N_k, μ, ν)` | 3 |
| 2 klead conv | `kconv_klead` | `U = s·FFT_k(IFFT_k T · V_R[:, None, :, None, :])`, `V_R` already in R space (mode 3 made it) | `T`, `U` `(N_k, a, m_x, b, m_y)`; `V_R` `(N_k, m_x, m_y)` | 1 |
| 3 klead fft | `kfft_klead` | `Y = s·FFT^±_k X` | `(N_k, rows)` | 1 |
| 4 kminor conv | `kconv_kminor` | `U = s·FFT_k(IFFT_k X · K_R[None, :, :, None, None, :])`, `K_R` already in R space (the caller made it with mode 5) | `X` `(d0, d1, d2, d3, d4, N_k)`, `K_R` `(d1, d2, N_k)` → `U` in X's layout (`out_layout=0`) or `(d0, N_k, d3, d1, d4, d2)` (`out_layout=1`) | 1 |
| 5 kminor fft | `kfft_kminor` | `Y = s·FFT^±_k X` | `(rows, N_k)` | 1 |
| 6 plane | `kconv_plane` | mode 0 on the identity plan, loaded from the route-G D-plane FFT output: `P^X = conj(F·D^X)` with the Bloch phase `F[k,g,p]`, L = slots `[0, c)` and R = slots `[c, 2c)` of the `2c` axis, split on load | `D` `(N_k, g, ns, 2c, ns, p)`, `F` `(N_k, g, p)` → `U` `(N_k, c, g·p)` | 3 |
| 7 klead unfold conv | `kconv_klead_unfold_block` | mode 2 on the typed unfold of the raw-parent Green, formed on load through `symmetry_maps.unfold_load_tables`: `Ĝ_k = U_k·[(mph_k·G_{row(k)}[lsrc_k, rsrc_k])·nph_k]·U_k†`, reading the partner `Gt` on an antiunitary row, or `conj(G)` when `conj_src = 1` | `G`, `Gt` `(n_parent, μ·ns, ν·ns)`, `V_R` `(N_k, μ, ν)`, `kout` `(N_k,)` → `U` `(n_out, d, μ, d, ν)`: full-k row `k` stored at `kout[k]` (−1: transformed, not stored), the `d × d` spin block at `(a0, b0)` (`d = ns` stores all) | 1 |
| 8 klead lorentz conv | `kconv_klead_lorentz_conj` | mode 7's load, then the four-current vertex sum in R space: `U = mult·s_f·FFT_k Σ_{A,B} γ_A (s_g·IFFT_k Ĝ) γ_B† ∘ V_R[k, x, A, y, B]`, `γ` signed spin permutations (attributes); one transform of `Ĝ` serves every block | `G`, `Gt` as mode 7, `V_R` `(N_k, μ, n_A, ν, n_B)`, `n_A, n_B ≤ 4` → `U` `(n_out, ns, μ, ns, ν)` through `kout` | 1, a whole `ns²` spin group per block |
| 9 klead unfold fft | `kfft_klead_unfold` | `Y_k = s·IFFT_k(L_k·Ô_k·R_k†)`, `Ô_k` the gathered, phased wedge tile of an interaction (partner tile on an antiunitary row, or its conjugate when `conj_trs = 1`), `L`, `R` the endpoint actions: the R-space operand modes 2, 7 and 8 take | `W`, `Wt` `(n_wedge, μ·n_l, ν·n_r)`, `spin_l` `(N_k, n_l, n_l)`, `spin_r` `(N_k, n_r, n_r)`, `n_l, n_r ≤ 4` → `Y` `(N_k, μ·n_l, ν·n_r)` | 1 |
| 10 plane fft gather | `plane_fft_gather` | the route-G plane transform, gathered on load ([below](#plane-fft-with-gather-on-load-mode-10)) | `F` `(A, S, …, n_col)` → `Y` `(A, n_pg, …, n_b, n_c)` | whole planes |
| 11 klead chi unfold | `kconv_chi_unfold` | one τ node of χ₀: `acc[o] += α_o Σ_ab conj(s_i·IFFT_k Ĝ^c)_ab · (s_i·IFFT_k Ĝ^v)_ab` (+ its conjugate when `complete`), `Ĝ^{v,c}` mode 7's typed unfold of each Green; the forward transform follows the τ sum | `Gv`, `Gvt`, `Gc`, `Gct` `(n_parent, μ·ns, ν·ns)`, `α` `(n_out,)` → `acc` `(n_out, N_k, μ, ν)`, updated in place | the k-box stage: `2ns²` columns per pair |

- **Flat k** is C order, with `kz` fastest.
- **Dtype.** Modes 2–5 take all-complex128 or all-complex64 operands (the
  complex64 image serves the fp32-GMRES BSE arm) and never cast; every other
  mode is complex128 only. The cpu host handlers are complex128 only, so a
  complex64 operand refuses at trace time on a cpu mesh.
- **In place.** Modes 2, 3 and 5, and mode 4 with `out_layout=0`, alias
  operand 0 to the result (`input_output_aliases={0: 0}`); mode 11 aliases
  `acc`. This is safe because each block reads all `N_k` values of its rows
  before it stores any of them.
- **Launch geometry (resident rows: modes 0, 1 and 4–9).** One 256-thread
  block per `rb` rows; a row needs `banks·16·(N_k|1)` bytes of shared memory
  (8 per element for complex64). Modes 0, 1 and 6 take
  `rb = min(16, ⌊B / row⌋)` with `B = min(100 KiB, opt-in)`, the others
  `rb = min(64, ⌊min(48 KiB, opt-in) / row⌋)`; when that is 0, `rb` is what
  the device's opt-in maximum holds. Mode 8 reaches for the opt-in maximum
  until a whole `ns²` spin group fits, and takes the k-box split arm when none
  does. Modes 2 and 3 run on the k-box stage and have no `rb`. Modes 7, 8 and 9 round
  `rb` down to whole spin groups (`ns·n_r` rows; `d²` for a mode-7 spin
  block) when one fits and then load each pair's sources once for the group;
  below one group each bank loads its own row with the same arithmetic.
- **The k-box stage (modes 2, 3, 11, and mode 8's split arm).**
  `kbox_stage.cuh` owns the box transform and its launch rule,
  `kbox_plan(grid, group, operands, elem, opt-in)`, computed from the k-grid
  and the device's opt-in shared memory. Modes 2 and 3 stage whole columns in
  padded shared memory, or, where two columns do not fit a block, run plane
  and pencil passes through the output in place. Mode 8's split arm runs the
  plane and group-pencil passes chunked over pairs through an
  `(N_k, chunk·ns²)` intermediate no larger than its output. For mode 11,
  single pass: `tr`
  whole pairs per block in a padded bank (odd z-line and row strides),
  gathered on load through mode 7's typed unfold, the three axis passes
  (z, y, x) of cuFFTDx thread FFTs, then the spin trace and its accumulation
  into χ_R in the group Mid, one thread per (k, pair). Split arm, when fewer pairs fit: plane passes over
  `(k_y, k_z)` on column tiles, then an R-space x-pencil pass for each spin
  group, chunked over pairs through an `(N_k, chunk·2ns²)` intermediate that
  XLA's scratch allocator grants (at most `scratch_bytes`; the door's default
  is one parent-Green tile).
- **Cost.** Each transform is `O(rows·N_k log N_k)` flops. HBM traffic is one
  read of each operand and one write of the result (the split arms of modes 8
  and 11 add one write and one read of their intermediate). Modes 6–9 and 11 read the
  producer's own buffer (the plane FFT output; the parent Green or the wedge,
  `n_parent/N_k` of the full-k size), so the phased, split or unfolded copy
  that modes 1 and 2 would need is never written. Their gather still reads
  one full tile per full k: the parent tiles are `n_parent/N_k` of the
  full-k footprint, not of the traffic (mode 11 on the 6×6 bispinor reads
  47 GB per τ node against 8.6 GB of parent Greens). The unfold tables (`lsrc`,
  `rsrc` int32, `mph`, `nph` complex128, each `(N_k, μ·ns)`) are closed-over
  host constants sliced per rank. Apart from those two intermediates, which
  XLA's scratch allocator grants, the kernels allocate no device workspace
  beyond dynamic shared memory.
- **Host workspace (cpu leg).** `gw_conv` stages `V_R = IFFT_k W` once per
  call in a reused host arena of `N_k·m_x·m_y·16` bytes, invisible to XLA.

**Refusals.**

| refusal | raised at | condition | fix |
|---|---|---|---|
| `GATE kconv-platform` | startup, factory | the mesh platform is neither CUDA nor cpu | run on a CUDA or cpu mesh |
| `GATE mathdx-headers` | startup (`mathdx_root`); kernel build | no importable `nvidia.mathdx` with `include/cufftdx.hpp` | `pip install nvidia-mathdx`; the `cuda12`/`cuda13` extras of `pyproject.toml` pin it (`==25.6.0`) |
| `GATE kconv-target` | startup, factory | the loaded library lacks the target the router selects | rebuild the library and point `LORRAX_FFI_SO` (CUDA) or `LORRAX_FFI_HOST_SO` (cpu) at it |
| `GATE mathdx-probe` | startup | the mode-3 probe kernel fails to compile or run on this device | an nvidia-mathdx wheel whose cuFFTDx supports the device's compute capability |
| `GATE kconv-kgrid` | factory | the k-grid is not three positive axes | pass the run's `(nkx, nky, nkz)` |
| `GATE mathdx-kconv-axis` | factory; handler | on CUDA, a k-grid axis above 40 (`KCONV_AXIS_MAX`, the fp64 cuFFTDx thread-FFT limit); the cpu leg (FFTW) has no cap | a smaller k-grid |
| `GATE mathdx-kconv-residency` | first call (kernel build) | one resident row, `banks·16·(N_k|1)` bytes (8 per element for complex64), exceeds the device's opt-in shared memory per block. On an A100 (166 912 B) that is `N_k > 3477` for modes 0/1/6 and `N_k > 10431` for the one-bank modes in complex128 | a smaller k-grid; modes 0, 1, 4–7 and 9 have no out-of-core arm |
| `GATE mathdx-kconv-kbox-residency` | first call | modes 2 and 3, and mode 8's split arm: the k-box tile, or its `(k_y, k_z)` plane or pencil, exceeds the opt-in maximum | a smaller k-grid |
| `GATE mathdx-kconv-lorentz-scratch` | apply | mode 8's split arm: XLA's scratch allocator refuses the chunk intermediate | none on the door: the chunk is already at most the output's size |
| `GATE mathdx-kconv-chi-residency` | first call | mode 11: the k-box arm `kbox_plan` picked does not fit the opt-in maximum, or its plane tile splits a spin group | a smaller k-grid; the χ₀ route has no fallback for this grid |
| `GATE mathdx-kconv-chi-scratch` | apply | mode 11's split arm: XLA's scratch allocator refuses the `(N_k, chunk·2ns²)` intermediate | a smaller `scratch_bytes` |
| `k-leading unfold conv: …` | factory; apply | tables cut for another mesh shape; `G` whose parent count or endpoint widths differ from the tables; `Gt=None` on a plan with antiunitary rows | build the tables from the same plan and mesh as `G` (`plan.unfold_load_tables()`); pass `ParentGreen.transpose` |
| `k-conv plane expects …` | trace | `D` or `F` not complex128 `(N_k, g, ns, 2c, ns, p)` / `(N_k, g, p)` | pass the plane FFT output as laid out |
| `LORRAX_FFT_FFI=0` | factory | the cpu leg refuses, and `make_flat_k_fft` refuses on both platforms | unset `LORRAX_FFT_FFI` |

A kernel-build failure (NVRTC compile, missing toolkit headers, module load)
is sticky: the handler caches it per in-process key and returns it on every
later call, naming the stage (`kconv_mathdx (fused cuFFTDx k-convolution):
<stage> failed -- …`).

**Headers and build.** The router passes the wheel's `nvidia/mathdx` directory
(from the `nvidia.mathdx` package spec) to every handler as the string
attribute `mathdx_root`. NVRTC includes `include/` and
`external/cutlass/include` beneath it, plus the CUDA toolkit's `include/` and
`include/cccl`, found beside the loaded libnvrtc. No environment variable names
either path. Building `liblorrax_ffi.so` needs no mathdx: the translation unit
links libnvrtc and resolves the driver API by `dlsym`. CMake compiles this
family and the Fourier plan only when its probe (option
`LORRAX_FFI_HAVE_CUFFT`, default on) finds `cufft.h`, `libcufft`, `nvrtc.h` and
`libnvrtc`; otherwise every `lorrax_mathdx_*` and `lorrax_fourier_plan*` target
is absent and startup refuses (`GATE kconv-target`, or the missing
`lorrax_fourier_plan_mathdx`). libcufft has one caller, the Fourier plan's FFT
group.

**Disk cubin cache.** The images live in `ffi.fft.cubin_cache_dir()`:
`$SCRATCH/.cache/lorrax/kconv_mathdx`, or `~/.cache/lorrax/kconv_mathdx` where
the site defines no `SCRATCH`. The cache is always on, has no knob, and is not
the XLA compile cache (`ISDF_JAX_CACHE_DIR`). One directory serves every world
size, because an image depends on the device and the wheel, not on P.

- **Key.** `common/nvrtc_build.h` owns the rule for every NVRTC-built kernel
  (this family and the Fourier plan's fused pair): FNV-1a over the embedded
  source, the text of each embedded header (`lrx_async_gather.cuh`,
  `kbox_stage.cuh`), the NVRTC options that decide the image (C++ standard,
  architecture, mode, grid, `ns`, rows per block, precision, SM), and the
  whole toolchain that can change an image (`nvrtc::mathdx_toolchain`): the
  cuFFTDx or cuBLASDx, commonDx, CUTLASS and CCCL version headers, the
  nvidia-mathdx wheel's dist-info name, and the NVRTC version with the loaded
  libnvrtc's real path (its patch level). A version header that reads empty
  disables the disk cache for that build rather than dropping out of the
  key. Editing an embedded source or header invalidates its images; file
  names, include paths and the host code are not keyed, so moving a source
  file between directories keeps every image.
- **File.** `kconv_m<mode>_<nkx>x<nky>x<nkz>_ns<ns>[x<n_r>][_c64]_sm<XY>_<key>.cubin`
  and `plan_pair_<N1'>x<K1>_<N2'>x<K2>_sm<XY>_<key>.cubin`, each with a
  `LRXKCONV1` header that carries the key and a hash of the payload.
- **Writes and reads.** A write goes to a unique temporary and is `rename`d
  into place, which is atomic on one filesystem, so concurrent ranks each
  publish a whole file. A read re-hashes the payload and checks for an ELF
  image; a torn, foreign or non-ELF file, or one the driver refuses to load,
  is deleted, recompiled once and replaced.
- **Cost.** A cold NVRTC build takes 5–7 s per image per process; a disk hit
  takes 5–15 ms (sandbox CLAIMS 2673). Without the cache, a CrI3-class
  run pays about 18 s of NVRTC per process.
- **Receipts.** Under `LORRAX_DEBUG_PRINT=1` the startup `[kconv]` line names
  the backend, the wheel root and the cache directory with its image count
  and size. Every kernel build prints `[kconv_mathdx] disk-cache hit` or
  `NVRTC built …` on rank 0, with the grid, rows per block, shared memory and
  whether the cubin was stored.

**Test-only cpu arm.** `tests/conftest.py` sets `LORRAX_KFFT_CPU_TEST_XLA=1`.
In-process pytest cpu meshes on Perlmutter have no host library, so under it
the cpu leg announces itself and uses `jnp.fft` for its k-axis transforms. It
is never read on CUDA and is never a production route.

**The gate.** `tests/multi_device/kconv_router_p4.py`, run as
`lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/kconv_router_p4.py`,
covers modes 0–9 and 11 at P4 (mode 10 has its own gate, below), including an
odd grid, 8×8×8, the complex64 k-minor image and both k-box arms of mode 11. The tolerance is 1e-13 against the cpu composition and 1e-12
against dense sums or `np.fft`; every check has a red twin that must miss by
more than 1e-3. Modes 6 and 7 must also equal the XLA chains they replace
within 2 ulp of the largest value (bitwise today, reported): mode 6 forms
`F·D` as an XLA:GPU complex multiply (no FMA), and mode 7 spells
`(mph·G)·nph` and `U·G·U†` as the XLA unfold and the spin-rotate kernel
round them. The mode-7 cases include a C3 plan with a general complex spin
action and `q = n/3` phases (`ns` 2 and 4) and `N_k = 196` at `ns = 4`, the
per-bank load. Mode 11 is held within 8 ulp of `max|χ|` of the chain it
replaced (`fftn(conj x) = conj(ifftn x)` makes them equal up to rounding).
The CPU-runnable twins are `tests/test_kconv_klead_unfold.py`,
`test_kconv_lorentz_unfold.py`, `test_kfft_klead_unfold.py`,
`test_kconv_chi_unfold.py`, `test_kconv_plane.py` and
`test_kconv_cpu_axis_cap.py`. `tests/multi_device/kconv_cubin_cache_check.py`
covers the disk cache's key and refusal paths.

A new mode is added in four steps:

1. Add a kernel entry to the embedded source under its `LRX_MODE` value.
2. Add a handler and `XLA_FFI_DEFINE_HANDLER_SYMBOL` in the same translation
   unit. Register its target in `ffi_loader._CUDA_TARGET_SYMBOLS`,
   `ffi.fft.KCONV_TARGETS` (which `require_kconv` checks at startup) and
   `ffi.cufft.CUDA_TARGETS`/`CUDA_SYMBOLS`. A changed operand contract gets a
   new target and the old one stays for older trees; changing an existing
   target's signature is an ABI bump (§8).
3. Add a router factory in `ffi/fft.py` that returns the mathdx call on CUDA
   and the plan-route composition on cpu.
4. Add a case, with a red twin, to `tests/multi_device/kconv_router_p4.py`.

### Plane FFT with gather-on-load (mode 10)

Route G transforms planes whose occupied cells arrive as a compact cylinder
`F (…, n_col)`; `plane_from_col (n_b·n_c,)` names each flat cell's column
(`n_col` = empty). The door returns

```text
Y[…, k_b, k_c] = Σ_{b,c} P[…, b, c] e^{-2πi (b k_b/n_b + c k_c/n_c)},   P = F scattered by plane_from_col, 0 elsewhere
```

(`fftn(P, axes=(-2,-1), norm='backward')`) without writing `P`. Persistent
blocks each hold `PB` planes of `(n_b, n_c|1)` in shared memory (`PB ≤ 8`
planes within 64 KiB, else 1; the grid is capped at the resident count). A
block gathers the occupied rows' cells through `gidx (rows, n_c)` and
`row_of (rows,)`, runs the row FFTs on those rows only, runs the column FFTs
on every column with dead rows read as zero, and stores each plane once,
coalesced. HBM traffic is one read of the cylinder and one write of the
plane.

The kernel is latency-bound, not bandwidth-bound (A100, ncu: one HBM pass at
36–40 % of peak, 25 % of warps active, long-scoreboard and barrier stalls
co-dominant). So when a `(rows, n_c)` staging block per plane also fits the
opt-in budget, the next group's cells are gathered asynchronously
(`src/ffi/cpp/common/lrx_async_gather.h`, cp.async, sm_80+) while the
current group's passes run, and the first row pass reads them from the
staging block. This gives 1.11–1.37× at 25²–80² on A100. The table stays a run-time
argument. Its row count is compiled in, so every index divisor is a constant
(one image per door, like the shape key before it; a cold build is about
6.6 s).

Every line FFT is a cuFFTDx thread FFT (`n ≤ 40`). An axis `n = n1·n2` with
`gcd(n1, n2) = 1` runs as the Good–Thomas two-dimensional DFT: the input sits
at `(n2·i1 + n1·i2) mod n`, output `(k1, k2)` is `X[k]` for `k ≡ k1 (mod n1)`,
`k ≡ k2 (mod n2)`, so there are index maps and no twiddles, and frequency `k`
stays at slot `(n2·(k mod n1) + n1·(k mod n2)) mod n` until the store.
`plane_fft_split` picks the most balanced split, or `(n, 1)` for a prime
power `≤ 40`. Block FFTs are not used because cuFFTDx's fp64 database lacks
45, 54, 75, 90, 150 and 250, which would take Bluestein and a host-built
workspace.

The door decides once, at build, and announces the route by name. Mode 10
serves a plane iff both axes split and the block fits:
`16·n_b·(n_c|1) + 5·n_b + 8·PB + 16 ≤` the device's opt-in shared memory per
block (`ffi.fft.plane_resident_bytes`; the second term is the kernel's static
row tables; a direct handler call past either test refuses as `GATE
mathdx-plane-split` or `GATE mathdx-plane-residency`). Every other plane takes
the XLA route: an axis with
no split (a prime above 40 or a prime power above 40: 41, 49, 64, 81, 121,
125, 128, 250, …) or an oversized block. Largest square served: 100 on
sm_80/87 (163 KiB), 78 on sm_86/89/120 (99 KiB, so 80² takes the XLA route),
119 on sm_90/100 (227 KiB). The block runs 512 threads when one block has the
SM (`⌊smem per SM / (PB·plane + 1 KiB)⌋ = 1`, e.g. 72² and up on A100), else
256 with two blocks per SM; the register cap this sets holds on every sm_80+.
F must be complex128 (`GATE plane-fft-dtype`, on both routes) and
`plane_from_col` in `[0, n_col]`. `fn(F, start, size)` transforms the slab
`F[:, start:start+size]` of `F (A, S, …, n_col)` in place, so the ζ loop's
group slice is not copied. The gates are
`tests/multi_device/plane_fft_gather_p4.py` (GPU parity `≤ 1e-13` over the
QE sides 24–250, the routes, a red twin, the slab form) and
`tests/test_plane_fft_gather.py` (a NumPy model of the passes against
`np.fft.fft2`).

### Local Fourier plan (`LocalFourierPlan`)

`common.fourier_plan.LocalFourierPlan` computes `y = R_out·F·E_in·x` over at
most three local axes; its contract, supports and per-axis GEMM/FFT selection
(`GEMM_CROSSOVER`, the support-fraction bound) are the service page
[`../dev/fourier_plan.md`](../dev/fourier_plan.md). This section owns the CUDA
leg, `cpp/cufft/fourier_plan_cuda_ffi.cc` with the remap kernel in
`fourier_plan.cu`.

* **Legs.** A plan with a GEMM stage chooses its leg when the call is lowered
  (`lax.platform_dependent`). On CUDA it is one `lorrax_fourier_plan_mathdx`
  custom call: cuBLAS ZGEMM with a stride-0 matrix and no transposes, the fused
  pair below, one cuFFT Z2Z plan per contiguous run of FFT axes, and remap
  kernels for embed/restrict. Elsewhere it is XLA ops, so a CPU operand in a GPU
  process takes the XLA leg. A plan with no GEMM stage takes the XLA leg on every
  platform, set at construction: on A100 the custom call's remap + cuFFT arm is
  1.20–1.33× XLA's take-embed + cuFFT at 24²–80² and 24³–64³, and cuFFT itself is
  equal on both legs (claim 2786). `lorrax_fourier_plan`
  (the same handler without the pair's two string attributes) stays in the
  library for older trees.
* **Fused pair.** When the two trailing axes are GEMM axes executed back to
  back, one cuBLASDx kernel per plane does both,
  `Q (N1'×N2') = α·A1 (N1'×K1)·P (K1×K2)·A2ᵀ (K2×N2')`, with the plane, both
  matrices and the intermediate in shared memory: the intermediate never
  reaches HBM and both GEMMs run on DMMA. Chosen when the plan is built, iff
  `16·(N1'K1 + N2'K2 + K1K2 + N1'K2 + N1'N2')` bytes fit the device's opt-in
  shared memory (else two cuBLAS GEMMs); a warp per block when the smaller
  GEMM output has < 128 elements, else 128 threads. NVRTC-built per
  `(N1', K1, N2', K2)` through `common/nvrtc_build` (the mathdx key rule with
  `cublasdx_version.hpp`), disk-cached with the k-convolution images; 8–13 s
  per shape cold. A100, whole plan against the cuBLAS chain: Fe 25³, K = 13,
  sphere→box 0.80–0.82, box→sphere 0.64–0.66; 16³–48³ 0.8–0.9; shapes that do
  not fit (64³, 72³, 96²) are unchanged. Boxes whose long axis is an FFT axis
  (CrI3 80×80×250) never pair. No production caller builds a separable plan on
  main (the ζ site uses `in_gather`). The GEMM rows and the pair are kept for
  real-space GW's sphere↔box transforms at Fe-class boxes; the service page has
  the estimate and the per-device projection.
* **CUTLASS under NVRTC.** The nvidia-mathdx 25.6 wheel's CUTLASS 3.9 declares
  `std::tuple_size`/`tuple_element` variadic under NVRTC; CCCL 3 (CUDA 13)
  declares them with one parameter, and NVRTC refuses the pair. The pair's
  build defines CCCL's include guard `_CUDA_STD___TUPLE_STRUCTURED_BINDINGS_H`
  (CCCL's structured bindings of `cuda::std` types drop out; nothing here uses
  them), no header is patched. Validated for the wheels in
  `ffi.fft.PAIR_MATHDX_WHEELS`. A plan that could pair checks the installed
  wheel at construction (`ffi.fft.pair_build_attrs`). On any other version it
  announces `fused pair off` and runs its GEMM pairs as the cuBLAS chain, so a
  new wheel never stops a run; the k-convolution family needs only the wheel's
  cuFFTDx headers and a probe compile (`require_kconv`).
* **Caches.** The CUDA leg caches plans per (device, attributes, batch) for
  the process: device Fourier matrices (`16·N'·K` bytes per GEMM axis), remap
  tables and cuFFT plans without work areas. Work areas and the ping-pong
  intermediates come from XLA's scratch allocator on every call, so nothing
  the plan uses per call lives outside the pool.
* **Refusals.** `GATE fourier-plan-int32`: a cuFFT or cuBLAS size past
  `2³¹−1`. `GATE mathdx-headers`: no
  `cublasdx.hpp` under `mathdx_root` when a pair is built. The Python
  contract's refusals are on the service page.
* **Determinism.** Reruns are bitwise at fixed device and toolkit, and a
  batch slice equals the same rows of a larger batch; nothing is promised
  across architectures.
* **`in_gather=(plane_from_col, n_col)`** is the route-G plane: mode 10
  (§ above) or its XLA route, with the slab form `plan(F, start, size)`.

The gate is `tests/test_fourier_plan.py` on a GPU: its `'__gemm__'` device
kind forces the GEMM on every axis, so two-axis and three-axis cases run the
fused pair wherever it fits, against `np.fft` at relative 1e-12.

### Parent-load ISDF pair convolution (mode 1)

Mode 1 is the pair convolution with its operands unfolded from the raw parent
k-points inside the load, so no full-k open-spin array is written to HBM. With
`p = irr[k]`, `o = sym[k]`, `m = left[o, μ]`, `n = right[o, ν]`, and `𝒯_k`
complex conjugation when `trs[k] ≠ 0`, the load builds

```text
P_{k,ab}(μ, ν) = conj( Σ_{c,e} coef[k, a·ns+b, c·ns+e] ·
                       𝒯_k( e^{2πi q_p·L_{o,μ}} · D_{p,c,e}(m, n) · e^{−2πi q_p·R_{o,ν}} ) )
```

from `D_l` with `coef_l` on the left and from `D_r` with `coef_r` on the
right, and the kernel then runs mode 0 on `P^L`, `P^R`.

| positional operand | shape | dtype |
|---|---|---|
| `D_l`, `D_r` | `(n_parent, ns, μ_local, ns, ν_local)` logical | complex128 |
| `irr`, `sym` | `(N_k,)` | int32 |
| `left`, `right` owner-local source maps | `(n_ops, μ_local)`, `(n_ops, ν_local)` | int32 |
| `L`, `R` lattice wraps | `(n_ops, μ_local, 3)`, `(n_ops, ν_local, 3)` | float64 |
| `q` parent fractional k | `(n_parent, 3)` | float64 |
| `trs` antiunitary mask | `(N_k,)` | int32 |
| `coef_l`, `coef_r` open-spin coefficients | `(N_k, ns², ns²)` | complex128 |
| result `U` | `(N_k, μ_local, ν_local)` | complex128 |

- **Tables.** They are built by `isdf.core._parent_conv_tables_local` from the
  typed unfold plan and must be authenticated owner-local plan tables. The
  handler checks their shapes and dtypes, not the device-side map values.
- **Layout.** The static attribute `centroid_major` states the physical layout
  of `D`. It is 1 for the CCT build (`c_q_from_psi_sm`): major-to-minor
  `(parent, ν, spin_r, μ, spin_l)`, requested through the `ffi_call` input
  layout `(0, 4, 3, 2, 1)`, so the GEMM's output feeds the kernel with no
  transpose. It is 0 for the ZCT tails and route G: row-major
  `(parent, spin_l, μ, spin_r, ν)`. Only the load's address arithmetic
  differs.
- **Vertex.** Production folds the post-unfold Lorentz vertex into `coef_r`
  (`isdf.core._parent_conv_vertices`, conjugating the phase because the load
  returns a conjugate). The kernel's `perm`/`phase` attributes therefore stay
  the identity, and every channel of one shape reuses one executable.
