# ROCm one-device complex128 LU wrong answer: root cause and fix

Branch `rocm/lufix` (Frontier `~/wt/lufix`, commit `efed2af70` on main `4bae15f55`, **not pushed**).
Patch: `rocm_lufix.patch` (12 files, +700/-17). Harnesses and all step logs: `lurc_harness_and_logs.tgz` (Frontier `~/lurc/`).
Compute: 2 debug jobs × 1 node (5697579 51.5 min, 5697676 8.3 min) = **0.99 node-h** of the 1.5 node-h budget.

## Deviations / scope
- **Fe P=4 bitwise** was run with this branch's Python and **main's** FFI library (`LORRAX_FFI_ROCM_SO` = int10 build). Fe with this branch's own rebuilt `.so` did not finish: it hung in "ISDF basis". **Main hangs the same way** with a cold `ISDF_JAX_CACHE_DIR` (details below), so the hang belongs to main. This is still provisional: I did not root-cause it.
- **hsuite P=4** ran the same way (branch Python, main `.so` via `SO_ROCM`). The new handler is never called at Fe/hsuite sizes, because the span rule keeps n=312 on `jnp.linalg.solve`.
- No SLATE timings at n > 16384 (the comparison through 16384 comes from the existing log `distla_lubigs`). There is no P=4 run of the `one_device` route at n > 8192. Large n was gated only through the standalone local solve.
- The cross-version check against `/opt/rocm-10.0.0` failed to run. Its rocBLAS on this driver gives `hipErrorInvalidImage` and a segfault, so I have no upstream verdict.
- The f64 `jnp.linalg.eigh` audit at n=24576 did not finish in about 17 min, and I cancelled it. That is unexplained and provisional; it may be slow, not wrong.

## 1. Root cause: rocBLAS trsm, not rocSOLVER getrf and not jaxlib
The lead hypothesis was partly right: the bug is a 2^31-**byte** offset. But it is in rocBLAS `{z,d}trsm`, left side, **in-place blocked path**. It is not in getrf, not in the 32-bit API, and not in jaxlib.

Evidence (one MI250X GCD, ROCm 7.14.1 wheels, rocSOLVER 3.35, rocBLAS `.so.5`):
1. **getrf is exact everywhere.** The JAX probe measured ‖PA−LU‖/‖A‖ = 1.5e-13 (c128 12288) and 2.0e-13 (c128 16384). Pivots were valid and `perm` matched `piv`. The C++ harness gave the same factors from `rocsolver_zgetrf`, `rocsolver_zgetrf_64` and `hipsolverZgetrf` (jaxlib's call), with factor_rel ≤ 2.4e-12 through c128 16384 and f64 20480. jaxlib's workspace query matched rocBLAS's (e.g. 135,596,288 B at 16384).
2. **A narrow RHS is exact; a wide RHS is wrong.** `jnp.linalg.solve` with 4 RHS is exact at c128 16384 (3.4e-15). With B = n×n (the `distla_lubig` setup) it is wrong: c128 11776 gives 4.8e-3, 12288 gives 8.5e-3, 16384 gives 7.7e-3. A blocked GEMM check shows the residual GEMM itself is exact (1.9e-15).
3. **The triangular solve alone fails** (XLA TriangularSolve on an LU factor). At c128 12288 it is exact at nrhs 1 and 512 and wrong at nrhs ≥ 1024. At 16384 it is exact at nrhs ≤ 384 and wrong at ≥ 512. In the L solve the bad rows are the last 1280 (11008–12287).
4. **rocBLAS reproduces it directly** (`rs_trsm.cpp`). `rocblas_ztrsm` gives 8.6e-3 at (12288, 4096) with library-managed workspace, a capped workspace or none, and with status 0. **`rocblas_ztrsm_64`, `rocsolver_zgetrs` and `rocsolver_zgetrs_64` are equally wrong.** So the int32 API is not the cause.
5. **Which path runs, and the threshold.** Failing cases use the small-workspace in-place path (about 160·m elements). It is taken when m is a multiple of 128 and the RHS is wide. Cases that use the copy-B path (workspace m·(nrhs+128)) are exact even at large n (c128 11616/11649/11680/11712; f64 16416/23200).
   - The threshold is a byte offset crossing 2^31. Wrong exactly when (m−128)(m+1)·itemsize ≥ 2^31, i.e. the 128-block diagonal offset of A.
   - c128: 11648 exact (2,147,143,680 B) and 11776 wrong.
   - f64: 16384 exact and 16512 wrong; 20480 and 24576 wrong.
   - The prediction was made before the f64 16512 run and confirmed by it.
6. The original `distla_lubig` facts all fit:
   - c128 11616 was exact because 11616 is not a multiple of 128, so rocBLAS took the copy path.
   - f64 16384 was exact because it sits just under the threshold.
   - 12288 and 16384 were wrong.
   - The bug is not specific to complex128: f64 fails from n = 16512.

## 2. Fix: a span-blocked path in LORRAX's FFI
- **`src/ffi/cpp/rocsolver/lu_solve_ffi.cc`** (ROCM leg; rocSOLVER/rocBLAS are resolved from the mapped `librocsolver.so.0`, like the eigh handler) adds target `lorrax_rocsolver_lu_solve`:
  - It runs `rocsolver_{z,d}getrf` on the row-major A read as Aᵀ, with no transpose of A.
  - It then solves X = P L⁻ᵀ U⁻ᵀ B by a blocked trsm (op T, diagonal blocks nb) plus GEMM updates, and finishes with `laswp`, incx = −1.
  - **Rule:** no rocBLAS/rocSOLVER call receives an operand spanning ≥ 2^31 bytes from its own pointer. With L = INT32_MAX/itemsize:
    - nb = min(1024, ⌊L/n⌋);
    - RHS chunks of kc columns satisfy (kc−1)n + n ≤ L;
    - GEMM row chunks satisfy (rows−1)n + k ≤ L.
  - Offsets are formed in 64-bit arithmetic in the handler.
  - Workspace comes from XLA scratch, sized by one rocBLAS size-query pass. There are no syncs.
- **`distrib_la/local_lu.py`** (`LOCAL_LU_ROUTE`, ROCM only; the CUDA cuSOLVER LU was measured exact through 16384, so CUDA has no row):
  - `solve_local` sends a system whose A or B spans **more than 2^31 bytes** to the handler, so every rocBLAS offset stays below 2^31. Otherwise it calls `jnp.linalg.solve` unchanged, so Fe sizes are bit-for-bit main.
  - If the target is missing it refuses with `GATE local_lu_target`.
  - Both `jnp.linalg.solve` call sites in `_batch_reshard.py` now call it.
- Registered in `distrib_la.loader`, `ffi_loader`, `CMakeLists.txt`, the `families` row and `test_platform_legs`. No env vars and no XLA flags.
- **`ONE_DEVICE_MAX_N[('solve_lu','ROCM')]`: 8192 → 27427.** The rule (stated in `resolve.py`) is one GCD's capacity: isqrt(allocator `bytes_limit` 60,183,563,895 B ÷ priced 5·n²·16 B).
  - There is no speed crossover:

    | n (c128) | one-device | SLATE 2x2 |
    |---|---|---|
    | 8192 | 0.72 s | 3.63 s |
    | 12288 | 1.53 s | 10.4 s |
    | 16384 | 4.10 s | 16.1 s |

  - Measured capacity: one-device runs at 20480 (6.2 s), 24576 (13.3 s), 26624 (14.2 s) and 28672 (22.7 s, peak 52.9 GB). 30720 is out of memory.
  - A caller's room still caps the route.
  - **Owner decision needed:** the row is also the rounds' element budget. A no-room caller with a large stack of small matrices now gets up to 27427² elements per round (main allowed 8192²). Room-passing callers (`w_isdf`, `response_bank`) are bounded by the size-route room check for one matrix only. I did not move ROCM to `ONE_DEVICE_BY_ROOM`, because that would make SLATE the incumbent and slow main by 3–7×.

## 3. Gates
- **Standalone on one GCD** (`python -m tests.test_rocsolver_lu_gate --big`: PASS on all 13 cases, relative residual ≤ 1.1e-14):
  - Handler forced at small n: 312×16 c128 2.0e-16; f64 batch; nrhs 3, 100, 5000; n = 4097. Each is within about 2× of `jnp.linalg.solve`'s residual on the same GCD.
  - Routed at large n: c128 11776 6.8e-15, 12288 7.2e-15, 16384 8.1e-15, 20480 8.9e-15, 27427 1.0e-14; f64 16512 3.4e-15, 20480 3.7e-15.
  - The native path at those sizes stays wrong: c128 12288 8.5e-3, f64 16512 1.8e-3.
- **Speed and peak, same harness:**

  | case | handler | native |
  |---|---|---|
  | c128 8192 | 0.715 s, peak 4.36 GB | 0.934 s, peak 6.46 GB |
  | c128 4096 | 0.128 s, peak 1.11 GB | 0.128 s, peak 1.90 GB |
  | f64 16384 | 2.71 s, peak 8.66 GB | 3.64 s, peak 11.28 GB |

  Per-call device peak is lower because A is not transposed.
- **Fe P=4** (branch Python, main `.so`) vs main: eqp0/eqp1 max|d| = 0.0 eV; `sigma_mnk.h5` (14 datasets) and `qp_wfn_rotations.h5` (10) are `array_equal`. `cmp.py` flags the eqp text as "DIFFER", but every numeric field matches. Rank-0 device peak is 6.77 GB on both, and the route lines are unchanged (solve_lu 64×312×312 one_device, 1 round). The per-stage device table is empty on this allocator, so per-stage peak equality was not shown.
- **hsuite P=4:** PASS (398.7 s; branch Python, main `.so`).
- **CPU pytest** (`JAX_PLATFORMS=cpu`, 4 host devices): `test_rocsolver_lu_gate`, `test_distrib_la_one_device{,_measured,_room}`, `test_platform_legs` and `test_rocsolver_eigh_gate` pass. The exception is `test_platform_legs::test_the_rocm_leg_is_declared_and_refused_by_its_file_name`, which **fails identically on main** in a checkout with a built `.so`.
- **ROCm pytest** for the gate file: 6 passed, including the GPU regression `test_large_n_route_is_exact_on_rocm[11776-c128, 16512-f64]` (skipped where there is no ROCm device).

### Open: Fe with a freshly built `.so` hangs (main too)
- With this branch's `.so` (key 51540a…), Fe P=4 blocked in "ISDF basis" for 2–10 min where main took 5.6 s. All ranks sat in `BlockUntilReady`.
- The same happened with that TU removed from the build (key 985f8a…), so the handler is not the cause.
- **Main also hung** with a fresh `FE_CACHE` (still stuck after 360 s).
- Likely cause: a cold path keyed by the native closure identity or compile cache. It is unresolved and needs its own lane before the branch's own `.so` can be Fe-gated.

## 4. eigh audit: ('eigh','ROCM') 16384
`jnp.linalg.eigh` (rocSOLVER zheevd) on c128:

| n | time | peak | residual | orthogonality |
|---|---|---|---|---|
| 20480 | 60.5 s | 35.2 GB | 8.4e-15 | 1.3e-14 |
| 24576 | 120.2 s | 50.75 GB | 8.1e-15 | 1.35e-14 |

zheevd does not take the trsm path. **16384 is therefore a measured-coverage bound** (SLATE eigh is not timed beyond it), **not a correctness bound**. A capacity rule like solve_lu's would raise it, but needs the SLATE comparison first. The f64 24576 run is the open item noted above.

## Not tested
- The one_device route at P=4 for n > 8192 (local solve only).
- c64/f32: the handler serves c128 and f64; the route leaves other dtypes on `jnp.linalg.solve`. By the same mechanism, c64 should fail from n ≥ 16512 with a wide RHS; this is unverified.
- Right-side trsm call sites elsewhere in LORRAX (any `triangular_solve`/`cho_solve` whose triangle spans more than 2^31 bytes is exposed). Only the distrib_la LU path was fixed.
- The CUDA leg (unchanged; its exactness comes from the existing A100 record).
