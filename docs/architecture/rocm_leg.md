# The ROCm leg's kernels and their CUDA twins

This page owns the native kernels that the ROCm leg runs where
CUDA uses a vendor library or a CUDA-only kernel, and each one's CUDA twin and
status. The handler catalog and the leg table are in
[The FFI layer](ffi_layout.md#platform-legs). The k-convolution modes are in
[k-convolution](kconv.md). Host-sync properties are in
[Host–device synchronization](host_device_sync.md#legs).

Every number on this page is from MI250X (Frontier). gfx94x (MI300) takes the
MFMA path by the target-id rule, but it has not been measured.

**Rules for a leg kernel.**
- **Parallel legs.** One leg-neutral source where the operation allows it
  (`lrx::gpu`, `common/gpu_blas.h`, run-time compiled text). Every CUDA-only or
  ROCm-only optimization names its twin, or says why there is none.
- **Gated against XLA.** Every vendor or native route is gated against the XLA
  route on the same device, with a stated tolerance (`ffi.common.families`
  `gate`, `tol`).
- **Forward-portable.** Kernel text compiles at run time for the device's own
  target id (hipRTC / NVRTC, `common/rtc_build`). Tile sizes, matrix-core paths
  and routes are chosen at run time from the shape and the target, by a stated
  rule, and there is always a generic path. Nothing works only on gfx90a or
  only on sm_80.
- **Never lower the size ceiling.** A kernel adds no workspace beyond the
  existing capped budgets, no resident copy and no second layout.

## 1 Active-range GEMM, device bounds {#active-gemm}

Σ's W synthesis and G builds multiply only the active K range of each row.
`ACTIVE_LOCAL_ROUTE` (`distrib_la._active_local`) picks the implementation per
leg: CUDA `native`, ROCm `device`, cpu `interval`.

**`device`** (`active_gemm/active_gemm_device_ffi.cc`; targets
`lorrax_active_gemm_device`, `lorrax_active_gemm_device_out`) is one run-time
compiled kernel:
- the K loop is clamped per row from the bounds, which are read on the device;
- the weights are applied on load;
- there is no host read or sync, no operand or weighted copy and no workspace;
- inactive storage is never loaded, and FLOPs run over the active width only;
- invalid bounds give NaN.

**FP64 matrix cores** (lanes mfma, mfma2). Under `LRX_MFMA`, which the host sets
only for gfx90a and gfx94x target ids, the inner product runs on
`v_mfma_f64_16x16x4f64`, with complex128 as the exact 4M form. Every other
target, CUDA included, takes the FMA path from the same text.
- The block tile is chosen per call from (m, n). The 32×32 entry is used when it
  pads ≥ 5 % less than 64×64: m = n = 156 covers 160 instead of 192, while 312
  stays on 64×64.
- The next K slab's loads are issued before the current slab's MFMAs (register
  prefetch, no second LDS buffer).

| Fe V2 (one GCD, µs per call, 64 chained) | w8 MFMA | w9 (32×32 + prefetch) | rocBLAS |
|---|---:|---:|---:|
| W c128[64,156,936], full K | 1125 (10.4 TF/s) | 840 (13.9 TF/s) | 793 (14.7 TF/s) |
| W, synthetic live 0.32 | 442 | 312 | 594 |
| G c128[64,312,36], full K | 324 | 305 | 313 |

Fe's real active fraction is 0.747, FLOP-weighted (the W calls 0.819; the G
calls 0.625 and 0.503). At partial bounds the kernel beats rocBLAS 1.9–3.5×.
Fe V2 P=4 Σ τ other went 8.13 → 7.07 / 7.02 s with MFMA (w8), then 7.00 → 6.67 s
with the w9 tile and prefetch (mean of 2 runs each); A100 `native` is 7.54 s.
eqp was within 2.1e-8 eV (w8) and 4.8e-8 eV (w9) of the reference, at run-to-run
level. Gate: `tests/test_active_local_native_gate.py` (against masked, interval
and dense, same device, 1e-13), 64/64 at P=1 and on every rank at P=4.

**Rejected on ROCm.**
- `native` (rocBLAS behind the leg-neutral `cublas/local_active_gemm_ffi.cc`):
  its per-call bounds read and stream sync drain the stream inside the τ loop
  (Σ τ 13.28 s with the `select` guard).
- rocBLAS over all K with zeroed weights (the `full` guard): 7.99 s, and it reads
  inactive storage.

**CUDA twin.** `native` (cuBLAS, bounds read on the host per call, two guard
`cond`s), measured on A100 as idling the GPU ≤ 1.2 %. The `device` source is
leg-neutral and NVRTC compiles it for sm_80 (FMA path). Adoption means listing
it in the CUDA leg's sources and gating it. A DMMA (`mma.sync.m8n8k4.f64`) arm
is sketched in the kernel header, not written.

## 2 kdft modes 1 and 10 {#kdft}

The ROCm ports of the CUDA mathdx parent-load pair convolution and plane FFT
(lane kconv). The designs are in [k-convolution §9](kconv.md#mode-1) and
[§10](kconv.md#mode-10).
- `lorrax_kdft_kconv_parent` (`kdft/kbox_parent_*`): the CUDA mode-1 typed parent
  load (both umklapp phases, the antiunitary conjugate, the open-spin `coef`
  sum) on the k-box stage in mode 6's tile, one HBM pass. Each thread reads a
  parent source once for all ns² entries.
- `lorrax_kdft_plane_fft_gather` (`kdft/plane_gather_*`): resident planes,
  gather-on-load, the same Good–Thomas splits. Each factor line is one register
  Cooley–Tukey step against compile-time twiddles. A 32-point axis stays on XLA
  (`GATE kdft-plane-slower`, 0.93×).
- Per call, one GCD, XLA → kdft: mode 1 on the Fe tile 4.21 → 1.71 ms (2.45×);
  mode 10 on the Fe 25² slab 20.00 → 15.16 ms (1.32×), and 1.07–3.26× elsewhere.
- Fe V2 P=4: ζ 3.14 → 3.00 s, and the **ζ device peak per rank 10.70 → 6.77 GB**.
- Gate: `tests/test_kdft_parent_gate.py` against `xla_reference`, 1e-12, at P=1
  and P=4.

**CUDA twin.** The mathdx originals, `mathdx_kconv_parent` and
`mathdx_plane_fft_gather`. The mode-10 CUDA branch now goes through the shared
`_plane_gather_call` with identical HLO.

## 3 rocSOLVER batched eigh {#batched-eigh}

`lorrax_rocsolver_batched_eigh` (`rocsolver/batched_eigh_ffi.cc`) makes one
`rocsolver_{z,d}{heevd,syevd}_strided_batched` call per stack; jaxlib's eigh
loops `zheevd` per matrix. `distrib_la.local_eigh.BATCHED_EIGH_ROUTE` holds the
windows:

| dtype | driver | stack ≥ | n | workspace per matrix |
|---|---|---:|---|---:|
| complex128 | `heevd` | 2 | 128–3120 | ≤ 4 n² |
| float64 | `syevd` | 2 | 40–3120 | ≤ 5 n² |

- `heevdj` (Jacobi) wins only at n ≤ 312 and is 2.8× slower than `heevd` at
  16 × 312, so no window takes it.
- The complex lower bound of 128 is a stopgap for hsuite na_sc's c128[4,96,96]
  stack, whose outcome follows the solver's rounding.
- A matrix rocSOLVER leaves non-finite with `info = 0` (its silent failure) is NaN
  in both outputs, so the checked solve refuses it by name.
- `COMPLEX_EIGH_ROUTE` ROCm `reflector` fences the wave-1 `zheevd` defect.
- Gate: `tests/test_rocsolver_eigh_gate.py` (residual, orthogonality,
  eigenvalues and eigenspaces against `jnp.linalg.eigh` on the same GCD,
  1e-12).

Whole-matrix LU uses the one-device capacity route (`distrib_la.resolve`).
When either A or its RHS spans more than 2³¹ bytes, `distrib_la.local_lu`
uses rocSOLVER factorization followed by a blocked rocBLAS triangular solve;
this avoids the offset defect in the unblocked solve. The native handler's
queried workspace and pivot buffer are explicit XLA results. ROCm has no
distributed LU provider: the route refuses when one complete solve cannot fit
the caller's room. The 8192 row is a round-size preference, not an accuracy
or maximum-size bound.

**CUDA twin.** Local eigensolves and LU solves use jaxlib's cuSOLVER
(`families` row `rocsolver`, CUDA `xla`). The room-based capacity route and
cuSOLVERMp serve the local and distributed placements respectively.

## 4 Active subspace on the device {#active-subspace}

`active_subspace/active_devsize.cc` (`NATIVE_HANDLERS` ROCm `devsize`): run-time
compiled kernels that read the active descriptors on the device, with no
per-call host sync. The shifted, reflected capacity eigensolve goes through
`lorrax_rocsolver_batched_eigh`. `distributed_ortho` needs cuSolverMp's
communicator and is CUDA-only; on ROCm the JAX psum route serves it and the
native collective refuses by `GATE subspace_native_collectives`. Gate:
`tests/test_subspace_native_gate.py`. **CUDA twin:** `blas` (`active_ops.cc`,
`active_eigh.cc`), with the descriptors read on the host per call.

## 5 Status by family {#status}

| operation | ROCm | CUDA | status |
|---|---|---|---|
| Σ active-range GEMM | `active_gemm` device (MFMA on gfx90a/gfx94x, FMA elsewhere) | `cublas_local` native | CUDA adoption of `device` pending a gated run |
| k-box k-convolution, modes 2–5, 7–9, 11 | kdft (hipRTC) | mathdx (NVRTC) | both built |
| pair convolution / plane FFT, modes 1, 10 | kdft `kbox_parent`, `plane_gather` | mathdx | both built |
| Local Fourier plan | `fourier_plan_rocm` (rocBLAS + hipFFT, no fused pair) | `fourier_plan_mathdx` | both built |
| batched local eigh | `rocsolver_batched_eigh` | jaxlib cuSOLVER | by design |
| distributed eigh | ELPA above `ONE_DEVICE_MAX_N`, one device below | cuSOLVERMp, or one device within the room | by design |
| distributed LU | one device within the room, refused beyond it | cuSOLVERMp, or one device within the room | no ROCm LU library |
| active subspace | `devsize` | `blas` | both built |
| parallel HDF5 | `phdf5` | `phdf5` | one source |
