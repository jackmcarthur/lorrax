# The FFI layer

This page describes how LORRAX reaches vendor libraries: the core kernel
operations and the engine each has on each hardware class
([kernel operations](#kernel-operations)), every native target (the
[kernel catalog](#kernel-catalog)), the five layers between a physics call
and a vendor routine, the one C++ tree and its two legs, which library serves
each engine on each machine, which cuSOLVERMp selects which communication
path, which FFT engine the host library binds, the C++ parallel-HDF5
defaults, and how to tell the native failure modes apart. It is for anyone
adding a kernel or diagnosing a native failure. The k-axis convolution
kernels have their own page, [k-convolution](kconv.md); building and sealing
the libraries is [Building the FFI libraries](../installation/ffi-build.md);
owner rulings are in [`decisions.md`](decisions.md), the SlabIO contract in
[`slab_io.md`](slab_io.md), and knob spellings in the
[environment-variable registry](../reference/env_vars.md).

## Kernel operations

One row per core operation: what it computes, where the physics uses it, the
engine on each hardware class, whether a plain-XLA implementation exists, and
the test that pins it. A plain-XLA route is one built from XLA ops alone, with
no LORRAX native library; where the column says *none*, a platform without a
LORRAX library (ROCm) has no engine for that operation.

| operation | where it acts | NVIDIA (CUDA leg) | CPU (host leg) | plain XLA route | gate | code |
|---|---|---|---|---|---|---|
| **χ₀ = G ⋆ G**, one τ node: `χ₀(R) += α_τ Σ_ab conj(Gᶜ_ab(R))·Gᵛ_ab(R)`, one FFT_k after the τ sum | gwjax screening (`gw.w_isdf`) → χ₀_q(μ,ν;τ) | mode 11 on the k-box stage for the identity-vertex step response and for each node of the shared-pole direct stream on a raw-parent plan (charge, any occupation; the four-current stream with channel vertices, `ffi.fft.make_kconv_chi_vertex`); every other χ₀ kernel (vertex pairs, the Fermi–Dirac Matsubara kernel, distinct left and right centroid sets, the other fractional-contour pair modes) takes mode 3 on unfolded Greens with the spin trace in XLA ([router](kconv.md#router)) | typed unfold and spin trace in XLA, host flat-k transforms | typed unfold, `jnp.fft` transforms and the spin trace in XLA | `tests/test_kconv_xla_gate.py` (mode 11) | `ffi.fft.make_kconv_chi_unfold`; `gw.w_isdf._get_chi_minimax_kernel`, `gw.w_isdf._get_chi_fractional_contour_kernel_face` |
| **Σ = G ⋆ W**: `Σ_k = FFT_k[IFFT_k Ĝ · W(R)]`, Ĝ the Green unfolded on load | gwjax Σ: `gw.ppm_tau_kernel` (τ stream), `gw.cohsex_sigma` → Σ_k(μ,ν;τ) at the caller's k rows | mode 7 on the raw-parent Green (typed unfold and spin action on load); mode 2 on the k-box stage for a full-k operand ([router](kconv.md#router)) | typed unfold in XLA (a full-k copy), then `lorrax_mklfft_gw_conv` | typed unfold in XLA (a full-k copy), then the transforms and the product | `tests/test_kconv_xla_gate.py` (modes 2, 7) | `ffi.fft.make_kconv_klead_unfold`, `make_kconv_klead` |
| **Four-current Σ**: `Σ = FFT_k Σ_AB γ_A (IFFT_k Ĝ) γ_B† ∘ W_AB(R)` | gwjax bispinor Σ: `gw.cohsex_sigma`, `gw.centroid_k_unfold` | mode 8 on the k-box stage: the single arm when a whole `ns²` spin group of padded columns fits the opt-in shared memory, else the split arm | mode 7's host composition, the γ block sum in XLA | mode 7's composition and the γ block sum in XLA | — | `ffi.fft.make_kconv_lorentz_unfold` |
| **W, V wedge → R space**: `Y_k = IFFT_k(L_k Ô_k R_k†)`, Ô read from the irreducible q wedge | the W and V operands of the two Σ rows: `gw.screening`, `gw.ppm_tau_kernel`, `gw.cohsex_sigma` | mode 9; a full-zone operand takes mode 3 | unfold in XLA (`apply_unfold_load_tables_local`), then the host flat-k transform | unfold in XLA (`apply_unfold_load_tables_local`), then `jnp.fft` | `tests/test_kconv_xla_gate.py` (mode 9) | `ffi.fft.make_kfft_klead_unfold`; `symmetry_maps.QirrOperator` |
| **k-axis FFT**: `Y = s·FFT^±_k X`, k leading or trailing | χ₀(R) → χ₀(q) (`gw.w_isdf`), the q→0 head (`gw.qsgw_head`), `gw.wavefunction_bundle`, band interpolation (`bandstructure.htransform`, `.orbital`), BSE (k trailing) | mode 3 on the k-box stage (k leading), mode 5 (k trailing) | `lorrax_mklfft_flat_k`, the FFTW3 advanced interface bound by `dlsym` ([§3c](#3c-which-fft-engine-the-host-library-binds)); k trailing: XLA transposes around it | `jnp.fft` over the three k axes | `tests/test_kconv_xla_gate.py` (modes 3, 5) | `ffi.fft.make_kfft_klead`, `make_kfft_kminor`; `common.fft_helpers.make_flat_k_fft` |
| **ζ-fit pair Gram** `C_q`: the pair convolution `s·FFT_k Σ_ab φ_a φ_b conj(IFFT_k P^L)·IFFT_k P^R` of the band projectors `P_k(μ,ν)` (mode 0 in the [mode table](kconv.md#modes)) | gwjax face-ψ ζ fit (`gw.isdf_fitting`, one C_q per channel), `gw.downfold` | mode 1, the typed parent load ([mode 1](kconv.md#mode-1)); mode 0 on full-k operands | typed parent load in XLA (a full-k open-spin copy per side), host transforms, spin contraction in XLA | typed parent load in XLA (a full-k open-spin copy per side), the transforms and the spin contraction | `tests/test_kconv_xla_gate.py` (mode 0) | `ffi.fft.make_fused_conv_kparent`, `make_fused_conv_kpair` ← `isdf.core.c_q_from_psi_sm`, `c_q_downfold` |
| **Route-G plane pair contraction**: mode 0 read from the D-plane FFT output, Bloch phase and L/R split applied on load | gwjax charge ζ fit, route G: `isdf.zeta_mubatch`, `isdf.pair_kernels`, `gw.centroid_k_unfold` → the Z_q store | mode 6 | phase, split and transpose in XLA, then the host pair tail | phase, split and transpose in XLA, then mode 0's composition | — | `ffi.fft.make_fused_conv_kplane` |
| **Route-G plane FFT**: `Y = FFT₂(P)`, P the plane scattered from its occupied cylinder | gwjax charge ζ fit, route G (`isdf.zeta_mubatch`) | mode 10 iff both sides split into thread FFTs of ≤ 40 points and plane + row tables fit the opt-in shared memory: squares to 100 on sm_80/87, 78 on sm_86/89/120, 119 on sm_90/100; every other plane takes the XLA route ([mode 10](kconv.md#mode-10)) | the XLA route | static-run concatenate, then `jnp.fft.fftn` | — | `LocalFourierPlan(in_gather=…)` → `ffi.fft.make_plane_fft_gather` |
| **BSE W term, trial-stack matvec**: `U = s·FFT_k(IFFT_k T · W_R)` with `T = Σ_K L R`, `K = min(n_c, n_v)`, formed on the load; the decode's (t, μ) contraction in the store | `bse.bse_stack_matvec.build_bse_stack_matvec`: TDA Lanczos, Davidson and thick-restart Lanczos, TDA FEAST and KPM, the spectral-bound Lanczos of FEAST and KPM with or without `--tda`, Haydock, `bse.exciton_bands`; `build_bse_stack_pair_matvec`: the matrix-free `bse.bse_nontda` solver, which the CLI does not select | the first that serves: `lorrax_mathdx_kconv_klead_outer_decode` (T and U never stored; two 8-warp groups ping-pong on two banks; K sum on the fp64 tensor cores, or the FMA pipe under `LORRAX_BSE_OUTER_KSUM=fma`); `lorrax_mathdx_kconv_klead_outer` with the decode in XLA (`lorrax_mathdx_kconv_klead_outer_ksum` under `LORRAX_BSE_OUTER_KSUM=fma`); the XLA encode and mode 2 (`lorrax_mathdx_kconv_klead`). The route is announced once (`[bse] W term`) | the einsum for T, the plan route of `make_local_kconv_klead`, the decode einsum | the einsum for T, the composition of `make_local_kconv_klead`, the decode einsum | — | `ffi.fft.make_local_kconv_klead_outer_decode`, `make_local_kconv_klead_outer`, `make_local_kconv_klead`, `klead_outer_refusal`, `klead_outer_decode_refusal` ([BSE](bse.md#the-matvec)) |
| **BSE W term, ring matvec**: `U = s·FFT_k(IFFT_k X · K_R)`, k trailing | `bse.bse_ring_comm.build_bse_ring_matvec_full`: the full (non-TDA) operator of FEAST and KPM and of the dense `bse.bse_nontda` build; the screening resolvents of `bse.w_ladder` (`w_bse`), `bse.bse_w_exact` and `bse.w_omega_chain`; the dense (A, B) oracle of the equality gates | mode 4, with a complex64 image for the fp32-GMRES arm | XLA moves k to the front, host transforms and product, k moves back; complex128 only | k moved to the front, the transforms and the product, k moved back | `tests/test_kconv_xla_gate.py` (mode 4) | `ffi.fft.make_kconv_kminor`, `make_local_kconv_kminor` |
| **Sphere ↔ box 3-D FFT**: ψ_nk(G) ↔ ψ_nk(r), densities, plane-wave matrix elements | ψ at the centroids (`common.wfn_transforms`), the kmeans valence density (`psp.get_DFT_mtxels`), the V_H, kin_ion and dipole matrix elements (`common.mtxel_sweep`), the QSGW density (`gw.qsgw_density`), DFT operators (`psp.dft_operators`) | XLA `fft` → cuFFT inside jaxlib | XLA:CPU `fft` | this row: `jnp.fft` inside the caller's `shard_map`; no FFI route reaches it | none on numerics; `tests/test_fft_shardmap_context.py` checks only that call sites sit inside a `shard_map` | `common.fft_helpers.local_fftn3`, `local_ifftn3`, `make_sharded_fftn_3d`, `make_sharded_ifftn_3d` |
| **Separable local DFT with supports**: `y = R_out·F·E_in·x`, ≤ 3 axes | no production caller besides the route-G plane FFT's `in_gather` form | `lorrax_fourier_plan_mathdx`: cuBLAS ZGEMM axes, the fused cuBLASDx pair when it fits the opt-in shared memory, one cuFFT group; startup refuses an nvidia-mathdx other than 25.6.0 (`GATE mathdx-pair-wheel`); cuFFT/cuBLAS sizes below 2³¹ (`GATE fourier-plan-int32`) ([plan](#local-fourier-plan-localfourierplan)) | the XLA leg | `dot_general` GEMM axes and one `jnp.fft` group, on every non-CUDA lowering | — | `common.fourier_plan.LocalFourierPlan` ([service](../dev/fourier_plan.md)) |
| **Green build GEMM**: `G_k(μ,ν;τ) = Σ_n ψ_nk(μ) w_n(τ) ψ*_nk(ν)` at the k parents | every Green of gwjax: χ₀ (`gw.w_isdf`), Σ (`gw.ppm_tau_kernel`, `gw.cohsex_sigma`), the ζ-fit projectors (`gw.isdf_fitting`) → the parent Green `(n_parent, μ, s, ν, s')` | a batched 2-D SUMMA (`distrib_la.panel_matmul`, [bounded face products](../services/distrib_la/api.md#bounded-face-products)): XLA all-gathers of band panels of at most `N_b/p` columns, every k in one exchange, each multiplied by the classic-cuBLAS local active-range GEMM (`distrib_la._active_local_cuda`); the response bank's band-window Greens take the same SUMMA with host-known intervals (`gemm_plan(layout='face').prepare_active_range`) | the same SUMMA with the JAX local interval product (`distrib_la._active_local.active_local_matmul`) | the CPU engine is plain XLA | — | `gw.greens_function_kernel.face_green_product`, `build_G_parents`; `distrib_la.panel_matmul` ([active GEMM ranges](../services/distrib_la/api.md#active-ranges)) |
| **Band projection**: `O_k,mn = Σ ψ*_mk(μ) O_k(μ,ν) ψ_nk(ν)` | Σ_mn from Σ_k(μ,ν) (`gw.ppm_tau_kernel`, `gw.cohsex_sigma`, `gw.photon_sigma`, `gw.mpa.sector_sigma`); BSE W decode (`bse.bse_ring_comm`), `common.zeta_projection` | face layout: stationary-operator stream (XLA dots, ψ collectives only); legacy body: XLA einsums and two `psum_scatter`s | face layout: none; legacy body: the right contraction on `lorrax_mklblas_gemm_batch` (CBLAS, [§3a](#3a-the-dependency-matrix)) | the legacy body's einsums (`LORRAX_BANDS_GEMM_FFI=0` on cpu) | — | `common.contract_bands.contract_bands_block_reshard` ([vendor GEMM](../dev/vendor_gemm_service.md)) |
| **Hermitian eigensolve**, batched local or distributed | the charge ζ factor (rank-truncating eigh of each C_q, dense and replicated under both `linalg` layouts), the QSGW `H_k` (`gw.sc_iteration.qp_eigh`) | local: `jnp.linalg.eigh` (cuSOLVER in jaxlib); `linalg = distributed`: cuSOLVERMp `syevd` | local: LAPACK in jaxlib; distributed: ScaLAPACK `p?heevd`/`p?syevd`; host SLATE eigh always refuses at resolve (bug L-2) | `jnp.linalg.eigh` | nothing observes which vendor answered | `distrib_la.plan('eigh')`; the charge factor: `isdf.cplus.factor` ([`distrib_la`](../services/distrib_la/api.md), [deck dial](../services/distrib_la/backends.md)) |
| **Dyson solve and dense factorizations**: `W_q = (1 − v_q χ₀_q)⁻¹ v_q` by LU; the transverse ζ LU; Cholesky on explicit request | gwjax screening (`gw.w_isdf.solve_w`), the response bank (`gw.response_bank`), the shared-pole head, the transverse ζ factor | local: per-q `jax.scipy.linalg.lu_factor`/`lu_solve` (cuSOLVER in jaxlib); distributed: cuSOLVERMp batched `solve_lu`/`getrf`/`getrs`, Cholesky `potrf`/`potrs`. `LORRAX_LU_NO_PIVOT` turns cuSOLVERMp pivoting off with no gate | local: LAPACK in jaxlib; distributed: ScaLAPACK `p?getrf`/`p?getrs`; Cholesky: host SLATE `potrf`/`trsm` | the local LU; Cholesky `native2d` | nothing observes which vendor answered | `distrib_la.plan('solve_lu')`, `plan('cholesky')` ([targets](#dense-linear-algebra-targets)) |
| **Active-subspace kernels**: store, projected eigh, project, reconstruct, Gram, CGS2 orthogonalization | Davidson (`psp.run_nscf`), Lanczos (`bse.bse_lanczos`, `bse.exciton_bands`) | `lorrax_active_subspace_*` (cuBLAS, cuSOLVER, NCCL) | `CpuSubspacePlan`: NumPy/LAPACK through `jax.pure_callback` | none | — | `distrib_la.plan_subspace`, `plan_orthogonalization` ([Davidson](iterative_eigensolvers.md)) |
| **Parallel HDF5 slab I/O** | every sharded array read or written through `file_io.slab_io` | `phdf5_{read, read_kchunk_union, write, write_independent}`, staged through the CUDA runtime | the same handlers on the host leg | none: one transport, and a deployment that cannot serve it refuses at open | GATE 7, GATE 10 | `ffi.io` ← `file_io.slab_io` ([§5](#5-parallel-hdf5-the-ffi-side), [SlabIO](slab_io.md)) |

**Every mathdx kernel** (the k-axis rows and mode 10) is NVRTC-built for the
device's own `sm_<cc>`, and `require_kconv` probe-compiles one at startup. A
grid the family cannot serve on the device (an axis above 40, a k-box plane
tile beyond the opt-in shared memory, a failed probe; `ffi.fft.mathdx_refusal`)
takes the XLA backend with one warning ([router](kconv.md#router)). Operands are
complex128; modes 2–5 also take complex64 on CUDA.

**Gaps.**

* No CPU engine: the face-layout band projection (gwjax builds its ψ
  carriers in the face layout) and the response bank's prepared active-range
  Green GEMM.
* No plain-XLA route: the face-layout GEMMs, the active-subspace kernels and
  slab I/O. A platform without a LORRAX native library
  has no engine for them.

## Kernel catalog

Every native entry point is an XLA FFI target (a string a loader registers)
or a ctypes C entry point. From a target string:

1. **Symbol.** Its row in `ffi_loader._CUDA_TARGET_SYMBOLS` /
   `_HOST_TARGET_SYMBOLS`, or in `distrib_la.loader`'s tables of the same
   names (the distributed linear algebra and the active subspace), gives the
   handler symbol.
2. **File.** `git grep -n 'XLA_FFI_DEFINE_HANDLER_SYMBOL' -- src/ffi/cpp`
   lists every handler with its file; the phdf5 handlers are spelled
   `LRX_PHDF_HANDLER(<X>)` (`<X>Ffi` on CUDA, `<X>HostFfi` on host).
3. **Python caller.** `git grep -n <target>` finds the Python constant and its
   `ffi_call`.

Startup refuses a provider without a target the run needs: the router's and
the Fourier plan's targets (`require_kconv`, `require_fourier_plan`) and the
host GEMM (its `Gate`).

The [operations table](#kernel-operations) owns what each family computes,
where it is used, what selects it and its gate; this table maps each family to
its sources, its build and its target strings.

| family | source (`src/ffi/cpp/`) | leg; build | targets (`lorrax_…`) |
|---|---|---|---|
| mathdx k-convolution | `cufft/kconv_mathdx_cuda_ffi.cc`, `cufft/kbox_stage.cuh` | CUDA; NVRTC for the device's own `sm_<cc>` at first use, disk-cached | `mathdx_kconv_{pair, parent, plane, klead, klead_unfold_xblock, klead_lorentz_conj, chi_unfold, kminor}`, `mathdx_kfft_{klead, klead_unfold, kminor}`; older trees: `mathdx_kconv_klead_{unfold, lorentz}[_rows]`, `mathdx_kconv_klead_unfold_block` |
| plane FFT (mode 10) | the same file (`kPlaneSrc`) | CUDA; NVRTC | `mathdx_plane_fft_gather` |
| Local Fourier plan | `cufft/fourier_plan_cuda_ffi.cc`, `cufft/fourier_plan.cu` | CUDA; C++ (cuBLAS, cuFFT), nvcc remap kernel, NVRTC cuBLASDx pair | `fourier_plan_mathdx`; older trees: `fourier_plan` |
| NVRTC build service | `common/nvrtc_build.{h,cc}`, `common/lrx_async_gather.h`, `cufft/kbox_stage_src.h.in` | CUDA; C++ | none ([the cubin cache](compilation.md#2-native-kernels-and-nvrtc-images)) |
| host flat-k FFT | `fftw/fft_flat_k_ffi.cc` | host; C++, the FFTW3 ABI bound by `dlsym` | `mklfft_flat_k`, `mklfft_gw_conv` |
| host CBLAS GEMM | `cblas/gemm_batch_ffi.cc` | host; C++ | `mklblas_gemm_batch` |
| distributed dense LA | `cusolvermp/`, `scalapack/`, `slate/` | CUDA: cuSOLVERMp; host: ScaLAPACK, SLATE | [below](#dense-linear-algebra-targets) |
| local active-range GEMM | `cublas/local_active_gemm_ffi.cc` | CUDA; C++ | `cublas_local_active_range_gemm` (C aliased, beta), `cublas_local_active_range_gemm_out` (no C, beta = 0: writes every row), `cublas_local_prepared_active_range_gemm` |
| active subspace | `active_subspace/active_{eigh,ops}.cc` | CUDA; C++ (cuBLAS, cuSOLVER, NCCL) | `active_subspace_{store, eigh, project, reconstruct, gram, ortho, distributed_ortho, subtract, subtract_gram}` |
| parallel HDF5 | `phdf5/` | both; C++ (HDF5, MPI; CUDA-runtime staging on the CUDA leg) | `phdf5_{read, read_kchunk_union, write, write_independent}`; `phdf5_read_kchunk` has no caller |

**Architectures.** Every nvcc TU carries SASS for sm_80, 86, 89, 90, 100
and 120 and compute_80/compute_120 PTX (§2); every NVRTC kernel compiles for
the device's own `sm_<cc>`. Launch sizes (rows and planes per block, k-box
tiles, the pair's threads) come from the device's opt-in shared memory, its
shared memory per SM and its SM count, except the fixed budgets named in the
router section. `cp.async` (sm_80+) is the only architecture-specific
instruction (`common/lrx_async_gather.h`, `cufft/kbox_stage.cuh`).

**Kernel lessons.** Each family's measured record (its largest speedup over
the plain-XLA path, what was tried and did not pay, what limits it) is one
comment block, with sandbox claim ids, at its Python owner. It stays out of
the CUDA sources: the cubin key hashes the embedded kernel text, comments
included (`common/nvrtc_build.h`).

| family | block above |
|---|---|
| k-box k-convolution, modes 2–5, 7–9, 11 | `src/ffi/fft.py:make_kconv_klead_unfold` |
| ζ-fit pair convolution and plane FFT, modes 0, 1, 6, 10 | `src/ffi/fft.py:make_fused_conv_kpair` |
| BSE outer-product load and fused decode | `src/ffi/fft.py:make_local_kconv_klead_outer_decode` |
| local active-range GEMM | `services/distrib_la/src/distrib_la/_active_local_cuda.py:active_local_cuda` |
| 2-D SUMMA Green build | `services/distrib_la/src/distrib_la/_panel_matmul.py:panel_matmul` |
| shared-pole W(τ) synthesis and transposes | `src/gw/mpa/sigma.py:synthesize_shared_pole_parents` |

### Dense linear algebra targets

`distrib_la` owns the calls into these targets, their selection and their refusals
([`distrib_la`](../services/distrib_la/api.md)); the handlers are here.

| target (`lorrax_…`) | symbol, file | `distrib_la` entry | selected by |
|---|---|---|---|
| `cusolvermp_eigh` | `EighMpFfi`, `cusolvermp/eigh_ffi.cc` | `_cusolvermp.distributed_eigh` | `linalg = distributed` on CUDA |
| `cusolvermp_batched_potrf`, `_potrs` | `CusolverMpBatched{Potrf,Potrs}Ffi`, `cusolvermp/batched_{potrf,potrs}_ffi.cc` | `batched_distributed_cholesky`, `_potrs` | an explicit `cholesky` request (the charge ζ factor is `rank_truncate`) |
| `cusolvermp_batched_solve_lu`, `_getrf`, `_getrs` | `CusolverMpBatched{SolveLu,Getrf,Getrs}Ffi`, `cusolvermp/batched_solve_lu_ffi.cc` | `batched_distributed_solve_lu` | the W Dyson solve under `linalg = distributed` |
| `scalapack_eigh` | `ScalapackEighHostFfi`, `scalapack/eigh_ffi.cc` | `_scalapack.distributed_eigh` | `linalg = distributed` on cpu |
| `scalapack_batched_solve_lu`, `_getrf`, `_getrs` | `ScalapackBatched{SolveLu,Getrf,Getrs}HostFfi`, `scalapack/{solve_lu,getrf_getrs}_ffi.cc` | `batched_distributed_solve_lu` | `solve_lu` distributed on cpu |
| `slate_potrf`, `_batched_potrf`, `_batched_trsm` | `Slate*HostFfi`, `slate/host_ffi.cc` | `_slate.*` | an explicit `cholesky = slate` on cpu |
| `slate_eigh` (host) | `SlateEighHostFfi`, `slate/host_ffi.cc` | — | always refused at resolve (SIGSEGV, bug L-2) |
| `slate_*` (CUDA) | `slate/{eigh,potrf,trsm,batched_potrf,batched_trsm}_ffi.cc` | `_slate.*` | built only when a `gpu_backend=cuda` SLATE is found; the `lorrax_A` bundle has none |

The ctypes C entry points: the cuSOLVERMp/NCCL grid context
(`cusolvermp/c_api.cc`), the MPI mesh context SLATE and ScaLAPACK share
(`slate/context.cc`), the phdf5 lifecycle (`phdf5/api.cc`), the workspace
queries (`lrx_eigh_workspace_bytes`, `lrx_active_eigh_lwork`) and the two stamps (§2). The host leg's copies end
in `_host` (`common/c_abi.h`).

---

## 1. The layers

Five, outermost first. Each one can refuse; none silently substitutes.

| # | Layer | Lives in | Job |
|---|---|---|---|
| 1 | **Consumer** | `src/gw/`, `src/file_io/`, `src/bse/`, … | states *logical* intent: shapes, not strides |
| 2 | **Service facade** (Python) | `src/ffi/io.py`, `fft.py`, `gemm.py`; `services/distrib_la/` | owns the call grammar, builds the descriptor, picks the backend |
| 3 | **Gate** | `lxkit.gate`, bound by `src/ffi/gate.py` (env dials and the vendor platform key); `distrib_la.resolve` (deck choices); `ffi.fft.require_kconv` (router) | announce-or-refuse: an explicit request that cannot be honoured refuses, never downgrades |
| 4 | **XLA FFI custom call** | `src/ffi/common/ffi_loader.py`, `distrib_la.loader`, both over `lxkit.native_provider` | locates and attests the `.so`, registers its handler symbols |
| 5 | **C++ handler + vendor library** | `src/ffi/cpp/<vendor>/` | the MPI-IO / BLAS / FFT / solver call |

**A vendor dependency enters only through one facade, with run-time
resolution and an announced refusal, and is gated against the XLA path on the
same device** ([XLA reference ruling](decisions.md#xla-reference)). Where a
route selects a native target, a missing library is a startup refusal naming
the `.so`. Portability fallbacks *inside* a handler (the FFTW3 `dlsym`
ladder, batched versus plain CBLAS) keep the library buildable everywhere;
each announces which entry it bound.

### Python-side module map

* **Real modules:** `ffi/io.py` (parallel HDF5), `ffi/fft.py` (the
  [k-convolution router](kconv.md#router), the plane factory and the
  Fourier-plan call), `ffi/gemm.py` (host batched GEMM), `ffi/gate.py` (the
  binding of `lxkit.gate`) and `ffi/common/ffi_loader.py`. Distributed dense
  linear algebra and the active subspace are `services/distrib_la`, which
  opens the same two libraries through its own `distrib_la.loader`.

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
└── cusolvermp/  cublas/  cufft/  active_subspace/   CUDA leg
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

How each leg is built on each site, the verify contract and its gates, how
the two legs are sealed into one deployable bundle, and how a run selects a
library are [Building the FFI libraries](../installation/ffi-build.md).

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
| **Distributed dense solvers** | cuSOLVERMp (GPU); ScaLAPACK from LibSci and host SLATE (CPU) | ScaLAPACK from MKL; host SLATE | — | — |
| **Distributed transport** | NCCL for cuSOLVERMp (§4); an `MPI_COMM_WORLD` split in mesh order for SLATE and ScaLAPACK | Intel MPI | — | the `[lorrax cusolverMp] … comm path:` banner |
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
JAX to JAX. With disjoint hyperslabs the write completes bit-exact at rc = 0;
two ranks on one chunk would corrupt silently.

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
   `ffi/common/ffi_loader.py` and `distrib_la.loader`'s tables. Refactors move
   files, never a target string;
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
   its mirror `ffi_loader.LORRAX_FFI_ABI_VERSION` together, and old bundles
   then refuse.

---

## 9. The Local Fourier plan's CUDA leg {#local-fourier-plan-localfourierplan}

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
  equal on both legs. `lorrax_fourier_plan`
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
* **Accuracy.** The GEMM axes and the fused pair agree with `np.fft` at
  relative 1e-12.
* **Determinism.** Reruns are bitwise at fixed device and toolkit, and a
  batch slice equals the same rows of a larger batch; nothing is promised
  across architectures.
* **`in_gather=(plane_from_col, n_col)`** is the route-G plane: mode 10
  ([k-convolution mode 10](kconv.md#mode-10)) or its XLA route, with the slab form `plan(F, start, size)`.
