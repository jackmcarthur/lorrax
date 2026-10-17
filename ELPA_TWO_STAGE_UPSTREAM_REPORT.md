# ELPA 2026.02.001, AMD GPU: two-stage solver returns wrong eigenvectors on a 4x4 grid for n ≥ 47104

*Draft for the ELPA issue tracker; not filed.*

## Summary

On 16 MPI ranks (4×4 process grid, one AMD MI250X GCD per rank), `ELPA_SOLVER_2STAGE` with
the AMD GPU kernels returns eigenvectors that fail both the residual and the orthogonality
check once the matrix size reaches n = 47104. At n = 46080 it is correct. The eigenvalues
look correct: the extremal eigenvalues agree to six digits with the one-stage solver and
with a correct 2×2 run.

The same matrices are solved correctly by:
- `ELPA_SOLVER_1STAGE` on the same 4×4 grid (n = 47104), and
- `ELPA_SOLVER_2STAGE` on a 2×2 grid (n up to 51200).

ELPA reports no error.

## Environment

| | |
|---|---|
| ELPA | 2026.02.001 (tarball sha256 `a379f27f4dbd27b2ee45017afec656d064301e97150c874649bdfd64957b75ed`), API 20260202 |
| System | OLCF Frontier: AMD MI250X (gfx90a), one GCD per rank, 4 ranks per node, 7 cores per rank |
| ROCm | 7.14.1: rocBLAS 5.5.0, rocSOLVER 3.35.0 (`cd957402`), HIP `libamdhip64.so.7` |
| Compilers | GCC / GFortran 13.3.1 (PrgEnv-gnu 8.7.0) |
| MPI | cray-mpich 9.0.1 (`libmpi_gnu_123.so.12`) |
| BLAS / ScaLAPACK | OpenBLAS 0.3.28 (`USE_THREAD=0 USE_LOCKING=1`); netlib ScaLAPACK 2.2.2, static |

Configure (relevant flags):

```
--enable-amd-gpu-kernels --enable-gpu-streams=amd --with-rocsolver --enable-hipcub
--enable-openmp --enable-runtime-threading-support-checks --enable-allow-thread-limiting
--disable-gpu-ccl --disable-avx512-kernels --disable-sse-assembly-kernels --with-mpi=yes
HIPCCFLAGS="--offload-arch=gfx90a -O3 -std=c++17"
```

Our build applies two small local patches to `src/GPU/ROCm/`:
1. an optional device-allocator hook in `hipMallocFromC`/`hipFreeFromC`;
2. status counting around the rocBLAS/rocSOLVER wrappers.

The failing runs below used patch 1 only, with the hook routing allocations to plain
`hipMalloc`. As far as we know, neither patch changes any computation. A stock build was not
re-tested.

## Reproducer

- **Driver:** a standalone C program (attached: `elpa_bench.c`) that:
  - builds a dense Hermitian matrix A = (G + Gᴴ)/√(2n) from a counter-based Gaussian
    generator (seed 12345), distributed block-cyclically with `nblk` on the `nprow × npcol`
    grid (ScaLAPACK descriptor, `local_nrows`/`local_ncols` from `numroc`);
  - calls `elpa_eigenvectors_{double,double_complex}` with **device pointers** for a, ev, q;
  - checks a sample S of k eigenvector columns on the GPU:
    - residual = ‖A Z_S − Z_S Λ_S‖_F / (‖A‖_F √(k/n)),
    - orthogonality = ‖Zᴴ Z_S − I_S‖_F / √k.
- **Handle settings:** `na = nev = n`, `nblk`, `local_nrows`, `local_ncols`, `mpi_comm_parent`,
  `process_row`, `process_col`; then `elpa_setup`; then
  - `solver = ELPA_SOLVER_2STAGE`, `amd-gpu = 1`, `use_gpu_id = <device>`;
  - `real_kernel = ELPA_2STAGE_REAL_AMD_GPU` / `complex_kernel = ELPA_2STAGE_COMPLEX_AMD_GPU`;
  - `omp_threads` (7 for real, 1 for complex);
  - then `elpa_setup_gpu`.
- **Run:** `srun -N4 -n16 -c7 --gpus-per-task=1 --gpu-bind=closest ./elpa_bench d 47104 128 4 4 1 1024 omp_threads=7`

## Results

Real (`d`) and complex (`z`) double precision, cold single call:

| grid | solver | dtype | nb | omp threads | n | sampled cols | residual | orthogonality | flagged cols | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 4×4 | 2-stage | f64 | 128 | 7 | 30720 | 256 | 1.6e-14 | 1.7e-14 | – | 40.7 |
| 4×4 | 2-stage | c128 | 64 | 1 | 30720 | 256 | 1.9e-14 | 2.0e-14 | – | 73.8 |
| 4×4 | 2-stage | f64 | 128 | 7 | 46080 | 1024 | 1.9e-14 | 2.1e-14 | 0 / 1024 | 93.1 |
| 4×4 | 2-stage | f64 | 128 | 7 | **47104** | 1024 | **1.26e-1** | **8.58e-2** | **1024 / 1024** | 100.8 |
| 4×4 | 2-stage | f64 | 128 | 7 | **51200** | 1024 | **1.09e-1** | **7.01e-2** | **1024 / 1024** | 121.7 |
| 4×4 | 2-stage | f64 | 128 | 7 | **51200** | 256 | **1.14e-1** | **7.07e-2** | – | 122.3 |
| 4×4 | 2-stage | c128 | 64 | 1 | **51200** | 256 | **1.29e-1** | **7.65e-2** | – | 251.8 |
| 4×4 | 2-stage, `gpu_solve_tridi=0` | f64 | 128 | 7 | **47104** | 1024 | **1.26e-1** | **8.58e-2** | **1024 / 1024** | 281.4 |
| 4×4 | **1-stage** | f64 | 128 | 7 | 47104 | 256 | 9.6e-15 | 1.7e-14 | 0 / 256 | 41.5 |
| 2×2 | 2-stage | f64 | 128 | 7 | 51200 | 256 | 2.0e-14 | 2.2e-14 | – | 333.7 |
| 2×2 | 2-stage | c128 | 64 | 1 | 51200 | 256 | 2.3e-14 | 2.4e-14 | – | 673.4 |

"Flagged" means per-column residual or orthogonality above 1e-8; "–" means not recorded for
that run. For the 47104 4×4 two-stage run, the driver prints the first 40 flagged columns
(eigen-indices 0–1794). All 40 have per-column residual 1.5–2.4e-14 and per-column
orthogonality between 4e-3 and 0.12. The large aggregate residual therefore comes from
columns beyond the first 40, which were not printed. The extremal eigenvalues match the
correct runs (47104: [−1.999572, 2.000649] in both the 2-stage 4×4 and the 1-stage 4×4 run).

## Observations

1. **Threshold:** between n = 46080 (correct) and n = 47104 (wrong) on 4×4, i.e. local
   matrix edge 11520 vs 11776 at nb = 128.
2. **Not affected:** 4×4 one-stage at the same n, and 2×2 two-stage up to n = 51200
   (local edge 25600).
3. **Not tied to the GPU tridiagonal solve:** `gpu_solve_tridi = 0` gives the same error to
   all printed digits. Our attempt with `gpu_trans_ev_tridi_to_band = 0` timed out (> 1500 s),
   so it is inconclusive.
4. **Both dtypes and both thread settings affected:** real with 7 OpenMP threads, complex
   with 1. We did not sweep nb at fixed dtype (f64 used 128, c128 used 64).
5. **Our reading:** the defect is in the two-stage band stages, i.e. back-transformation
   tridiagonal → band or band → full, or the bandred / band → tridiagonal reduction at large
   n per rank on a 4×4 grid. We have not isolated it further.

## Not tested

- grids other than 2×2 and 4×4;
- n between 46080 and 47104;
- CPU kernels at the same size;
- NVIDIA GPUs;
- ELPA 2025.06.x.

## Attachments (on request)

- `elpa_bench.c` (driver)
- `build_elpa_hip.sh` (full build recipe with pinned hashes)
- job logs `elpa_dg_5699974.out`, `elpa_ab_5700031.out`, `elpa_sw16_5699940.out`,
  `elpa_loc_5700044.out`, `elpa_sw4_5699939.out`
