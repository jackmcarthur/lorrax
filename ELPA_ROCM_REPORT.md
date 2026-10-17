# ELPA as the ROCm distributed eigensolver — report

Branch `elpa` (not pushed) on `caffa8cd7` (main `4bae15f55` + 12 unmerged
lufix/dlascale commits). Frontier MI250X (gfx90a), ROCm 7.14.1, cray-mpich
9.0.1, PrgEnv-gnu 8.7.0. Evidence directories (Frontier): `~/elpa_bench/out/`
(standalone, jobs listed in §9), `/lustre/orion/mat280/scratch/jackmc/elpa_gate/`
(gates). Budget used: **2.973 Frontier node-h** of 3.0 (sacct, 20 jobs);
Perlmutter 0 of 0.5.

## 1. Outcome

ELPA wins, but **one-stage** ELPA, not the two-stage ELPA the task asked to
benchmark. ELPA 2026.02.001's GPU two-stage path is slower than its GPU
one-stage path at every size measured. It also returns non-orthogonal
eigenvectors on a 4x4 grid for n ≥ 47104 (§4). The integrated `elpa` backend
therefore defaults to one-stage (`SOLVER = 1`) and is the ROCm `distributed`
eigh above `ONE_DEVICE_MAX_N` (unchanged, 16384). SLATE stays selectable
(`backend = "slate"`) and keeps `potrf`/`solve_lu`.

Cold wall seconds, one call. ELPA and SLATE: Frontier, P = 4 is 2x2 GCDs,
P = 16 is 4x4. cuSOLVERMp: Perlmutter A100 2x2, from earlier lanes.

| n | dtype | ELPA 1-stage P=4 | ELPA 2-stage P=4 | SLATE P=4 | cuSOLVERMp P=4 (A100) | ELPA 1-stage P=16 | ELPA 2-stage P=16 | SLATE P=16 |
|---|---|---|---|---|---|---|---|---|
| 16384 | c128 | 17.36 (14.10 st) | 43.20 (37.47 st) | 74.7 | – | – | – | – |
| 20480 | c128 | 23.11 | 71.52 (63.14 st) | 121.8 | 15.1 (12.62 st) | – | – | 96.0 |
| 30720 | c128 | 56.46 | 186.57 | aborts | 41.8 (39.39 st) | 27.35 | 73.82 | 218.5 |
| 40960 | c128 | 119.30 | 380.03 | – | 100.8 (98.10 st) | – | – | – |
| 51200 | c128 | 208.54 | 673.42 | – | 167.4 | 80.11 | **wrong** (251.84) | – |
| 16384 | f64 | 11.65 (8.20 st) | 21.94 (18.47 st) | – | – | – | – | – |
| 20480 | f64 | 15.25 | 34.51 (30.74 st) | 84.4 | 11.2 (6.84 st) | – | – | – |
| 30720 | f64 | 33.66 | 88.17 | 213.6 | 20.0 (18.03 st) | 19.87 | 40.69 | – |
| 40960 | f64 | 66.23 | 186.16 | – | 44.0 (39.70 st) | – | – | – |
| 46080 | f64 | – | – | – | – | – | 93.14 ✓ | – |
| 47104 | f64 | – | – | – | – | 41.52 | **wrong** (100.79) | – |
| 51200 | f64 | 114.14 | 333.71 | – | 76.2 | 49.47 | **wrong** (122.25) | – |

"st" = steady (second call on the cached handle). ELPA one-stage is 4.3–8x
faster than SLATE where both ran. It is 1.2–1.5x slower than cuSOLVERMp on
A100 at 40960–51200 (different hardware; for orientation only). Sampled
checks (256 columns): residual ≤ 1.15e-14, orthogonality ≤ 1.9e-14 for every
one-stage row. Through the integrated route (harness, §6): 20480 complex128
26.37 s, float64 14.36 s, full checks ~1e-14.

## 2. Deviations from the task (declared)

1. **One-stage, not two-stage, is the default and the headline.** Two-stage
   was benchmarked as asked (§3, §4), lost on speed everywhere, and is wrong
   at 4x4 n ≥ 47104. It is selectable (`solver=2`), and the backend refuses it
   on p ≥ 4 grids for sides above 46080.
2. **Benchmark coverage reduced (budget).** P = 16 ran only at 30720 and 51200
   (both dtypes), plus f64 46080/47104 for the defect. Most rows are a single
   cold call; steady repeats exist only at 16384 and 20480 (P=4). Checks are
   sampled (256 columns; 1024 columns at 46080/47104/51200 P=16); full checks
   only at n = 4096 and in the harness rows.
3. **No GPU band→tridiagonal stage exists** in ELPA 2026.02.001
   (`gpu_tridiag_band` is commented out). Two-stage `band_to_tridi` runs on
   the host, MPI-parallel. This is part of why two-stage loses.
4. **Build variant:** OpenMP build without RCCL. Configure refuses
   OpenMP + `--enable-gpu-ccl=rccl`. RCCL gave nothing at 2x2 (use_ccl=0 69.3
   vs 68.9 s, non-OpenMP build). complex128 runs with omp_threads=1: OpenMP +
   complex two-stage GPU bandred crashes (rocBLAS concurrent handle / memory
   fault).
5. **Early benchmark rows used superseded builds**: p4a/p4b (non-OpenMP +
   RCCL, prefix key 152c54b1347b) and p4c (OpenMP without the allocator hook,
   062bb9cb5603). The final build script does not reproduce them. All
   headline rows (sw4, sw16, dg, ab, loc, 1s, 1s4, 1s16) used the final prefix
   `elpa-2026.02.001-e65d8bda33f0`.
6. **gfx942 not built**; Perlmutter parity (optional) not run.
7. **Harness rows only at 20480** (c128, f64). 30720 was dropped for budget.
   The first harness attempt (gate5) refused at 20480 with "ELPA kept 10485760
   pooled bytes past the call" (§5). The fix scales the small-allocation
   threshold with n. The rows in §6 are from gate6, on the final leg.
8. **Gate legs differ by ELPA-handler-only changes.** Fe P=4 and hsuite ran
   on leg key 8f8214a5e37d47bf. The ELPA device gate ran on 24fe7b29df47db79
   (adds the 1 MiB small-allocation rule). The harness ran on the final
   c4e6406dd0d6c056 (n-proportional threshold). The changes touch only
   `ElpaEighFfi`, which the Fe/hsuite routes never call (n ≤ bound). At the
   device gate's sizes (n ≤ 1024) the final threshold equals the gated 1 MiB.
   The device gate was not rerun on the final leg.
9. **Invalid gate attempts** (excluded from all results): 5700087 imported
   the lufix tree (piped `source`); 5700139 had an empty srun environment;
   5700163 used stale baseline/main directories and hit a handler bug
   (`real_kernel` set on a one-stage handle, fixed).
10. **Driver RNG ≠ harness RNG.** The standalone driver builds its Hermitian
    test matrix with its own generator. The distribution is the same as the
    harness's, but the matrices are not bitwise equal.
11. **Branch base.** The branch sits on `caffa8cd7`, not on main, because its
    CMake/ctx-registry/distrib_la code builds on the dlascale lane. The patch
    is the one commit on top of `caffa8cd7`.
12. **Per-stage device peaks** on ROCm read the non-resetting
    `peak_bytes_in_use`. "Unchanged" (§7) therefore means equal tables (all
    stages report the run peak), not a per-stage comparison.

## 3. Build

`config/frontier/build_elpa_hip.sh` (in the branch). It pins tarballs by
SHA-256:
- ELPA 2026.02.001:
  `a379f27f4dbd27b2ee45017afec656d064301e97150c874649bdfd64957b75ed`
- netlib ScaLAPACK 2.2.2:
  `a2f0c9180a210bf7ffe126c9cb81099cf337da1a7120ddb4cbe4894eb7b7d022`

It installs into an immutable, content-keyed prefix
`$ROOT/elpa-2026.02.001-<key>`. The key hashes the script, the tarballs,
OpenBLAS, the OpenMP switch, ROCm, the gfx target and PrgEnv/MPI. The build
stages, renames, makes the prefix read-only and verifies it on rerun.

- **Configure:** `--enable-amd-gpu-kernels --enable-gpu-streams=amd
  --with-rocsolver --enable-hipcub --enable-openmp
  --enable-runtime-threading-support-checks --enable-allow-thread-limiting
  --disable-gpu-ccl`. AVX-512 kernels are disabled.
- **BLAS:** the SLATE build's locking OpenBLAS 0.3.28 (one BLAS in the
  process).
- **ScaLAPACK:** linked static, PIC and hidden
  (`-Wl,--exclude-libs,libscalapack.a`). Both legs are RTLD_GLOBAL and the
  host leg carries LibSci's ScaLAPACK, so ELPA exports and imports no
  ScaLAPACK symbols.
- **Allocator hook:** a patch to `rocmFunctions_template.h` exports
  `lorrax_elpa_set_device_allocator`, so ELPA's device mallocs can come from
  a LORRAX pool.
- **Link checks:** one ROCm, one MPI and one BLAS; hook exported; no
  ScaLAPACK symbols. The script writes `LORRAX_ELPA_CLOSURE.json` (24 files,
  2 private, record e6f110f6ca1e67ad).
- **Leg closure:** `build_ffi_rocm.sh` builds or locates the prefix and passes
  `-DLORRAX_ELPA_ROCM_DIR`. The final leg's closure: 29 enforced libraries
  (pinned 3, process 10, rpath 11, private 4, attest 1), with
  `libelpa_openmp.so.19` private and `libopenblas` mapped once. Record
  sha256 `7202849ab076b916…`.

## 4. Two-stage correctness defect (upstream report material)

On a 4x4 grid, two-stage ELPA returns eigenvectors with
**residual ~2e-14 on every sampled column but orthogonality 0.07–0.13** for
n ≥ 47104:

| n | dtype | check | resid | orth |
|---|---|---|---|---|
| 46080 | f64 | 1024 col | 1.9e-14 | 2.1e-14 ✓ |
| 47104 | f64 | 1024 col | 1.26e-1 | 8.6e-2 |
| 51200 | f64 | 1024 col | 1.09e-1 | 7.0e-2 |
| 51200 | c128 | 256 col | 1.29e-1 | 7.7e-2 |

- **Pattern:** the bad columns are scaled eigenvectors, not wrong subspaces.
- **Independent of:** dtype, nb (64/128), OpenMP, and the GPU tridiagonal
  solve (`gpu_solve_tridi=0` gives identical errors, job 5700031).
- **Not affected:** P = 4 (2x2) two-stage is correct to 51200, and one-stage
  at 47104 on 4x4 is correct (resid 9.6e-15, orth 1.7e-14).
- **Localization:** the defect is in two-stage's band stages (tridi→band or
  band→full). A `gpu_trans_ev_tridi_to_band=0` run timed out, so that
  isolation is inconclusive.
- **Status:** not yet reported upstream.

Another known issue: ELPA leaks MPI communicators with GPU kernels (CSCS
alps-uenv PR #368). The backend caches one handle per (ctx, n, nb, dtype,
solver), so this costs one leak per shape.

## 5. Stage times, tuning, memory

**One-stage stages** (ELPA timers, s; forward = GPU Householder
tridiagonalization):

| n, grid | dtype | wall | forward | solve | back |
|---|---|---|---|---|---|
| 20480 2x2 | c128 | 23.11 | 17.43 | 2.24 | 3.40 |
| 30720 2x2 | c128 | 56.46 | 43.58 | 4.80 | 7.98 |
| 40960 2x2 | c128 | 119.30 | 94.94 | 8.60 | 15.53 |
| 51200 2x2 | c128 | 208.54 | 169.10 | 13.95 | 25.21 |
| 51200 2x2 | f64 | 114.14 | 89.48 | 13.74 | 10.58 |
| 51200 4x4 | c128 | 80.11 | 57.71 | 8.62 | 13.59 |
| 51200 4x4 | f64 | 49.47 | 34.43 | 8.38 | 6.22 |

**Two-stage stages, 51200 2x2 c128:** full→band 132.4, band→tridi (host)
107.9, solve 15.1, tridi→band 373.4, band→full 38.3 s.

**Tuning:**
- nb = 64 ≈ 128 for one-stage. For two-stage c128, nb = 64 is best: at 20480,
  nb = 32/64/128/256 took 100.1/68.9/79.8/132.4 s.
- nb must be a power of two. nb = 96 silently falls back to CPU kernels
  (621 s).
- `intermediate_bandwidth=128` ≈ nb128. nbc collectives had no effect.
  use_ccl gave no gain.
- f64 two-stage at 20480 is best with nb = 128 and 7 OpenMP threads: 34.5 s,
  vs 39.1–39.7 s at nb = 64 or without OpenMP.

**Memory, measured per rank, standalone (L = n/p):**
- **Hooked ELPA high-water:** ≈ POOL_PER_LOCAL2·L² + ~2.3–2.7 KB·L, with
  POOL_PER_LOCAL2 = 24 B (c128 one-stage, all two-stage) and 16 B (f64
  one-stage). The backend sizes the pool at the lower envelope of these
  (`_elpa.POOL_PER_LOCAL2`, `POOL_PER_LOCAL = 2048`): 2.54 GB c128 / 1.70 GB
  f64 at 20480 P=4, 15.78 / 10.54 GB at 51200 P=4.
- **Outside any pool:** ELPA's GPU-setup state, persistent per handle,
  0.62–0.63 GB. Unhooked rocSOLVER/rocBLAS transients run 0.1–2.75 GB, with
  2.75 GB at 51200 P=4. Total ≈ 3.4 GB at 51200 P=4, above
  `DISTRIBUTED_LINALG_OUTSIDE_BYTES = 3e9` (owner decision, §8).
- **Device peak, standalone 51200 P=4:** c128 40.2 GB, f64 24.5 GB. In the
  integrated route the projected XLA peak at 51200 c128 P=4 is ≈ 57.7 GB of
  arena. This is **projected, not measured in-process**.
- **Host increment:** one-stage up to 11.0 GB per rank (51200 P=4); two-stage
  up to 32.6 GB.

**Buffers ELPA keeps past a call.** ELPA caches some device buffers on the
handle and keeps them past the solve:
- 262144 B at n = 1024;
- n·32 elements at n = 20480 (10 MiB c128, 5 MiB f64).

Allocations below max(1 MiB, n·64·itemsize) are therefore device mallocs
outside the pool. The handler checks after every call that no pooled bytes
are still live and refuses with an error otherwise. This check caught the
20480 case in gate5.

## 6. Integrated harness rows (gate6, job 5700227, leg c4e6406dd0d6c056)

`scale_distrib_la.py --sizes 20480 --dtypes complex128 float64 --ops eigh --mesh 2x2`.
The route is `distributed` → `elpa`; the ledger line names the size route
(n = 20480 > 16384 → elpa).

| dtype | cold s | resid (full) | orth (full) | arena in-use max | outside max | device peak | pool / pool hw | fallbacks | small hw |
|---|---|---|---|---|---|---|---|---|---|
| c128 | 26.37 | 9.10e-15 | 1.37e-14 | 9.25 GB | 2.94 GB | 11.87 GB | 2.538 / 2.517 GB | 0 | 44.5 MB |
| f64 | 14.36 | 7.58e-15 | 1.22e-14 | 5.05 GB | 4.21 GB* | 9.04 GB | 1.699 / 1.699 GB | 0 | 22.4 MB |

\* f64 ran second in the same process. Its outside figure includes the
cached c128 handle's state (~0.6 GB) and the HIP context and code objects.

For comparison, SLATE's harness rows at the smaller n = 16384 (dlascale
lane) show 5.0–7.9 GB outside the arena and an 11.3–14.3 GB device peak.

## 7. Gates

- **Fe P=4, bitwise** (gate4, job 5700195, leg 8f8214a5e37d47bf). Compared
  `elpa` against `elpa_base` (caffa8cd7) and against main (`int10`,
  4bae15f55); source roots verified in each run log.
  - eqp0/eqp1/eqp_g0w0/sigma_diag are identical.
  - `sigma_mnk.h5` (15 datasets) and `qp_wfn_rotations.h5` (10 datasets) are
    bitwise identical.
  - Per-stage device-peak tables are identical (40 rows, run peak 6.77 GB;
    caveat in §2.12).
  - Host run peak: 6.75 / 6.62 / 6.44 GB (elpa / base / main), i.e. +0.13 GB
    on elpa vs base, likely the larger leg (ELPA + OpenMP runtime);
    not investigated.
  - Walls: 155.5 / 132.5 / 118.4 s. The elpa run compiled 351 programs
    (61.4 s) because its native-closure identity is new, so its compile-cache
    namespace was cold. Base and main hit their caches.
- **hsuite P=4** (gate4): 1 passed on all 4 ranks (440 s).
- **ELPA device gate** (`python -m tests.test_distrib_la_elpa_gate`, gate5,
  job 5700209): PASS 8/8 on all ranks (tol 1e-12). It covers f64/c128,
  padded side (n = 1000 → 1024), eigenvalues-only and a cached-handle second
  call; e.g. f64 n = 1024 res 9.0e-15, orth 3.7e-15, |Δw| 9.7e-15.
- **CPU suites** (login node, ROCm leg moved aside as the leg-refusal test
  requires): 152 passed, 45 skipped. Files: `tests/test_platform_legs.py`,
  `tests/test_distrib_la_elpa_gate.py`, `tests/test_native_closure.py`,
  `tests/test_distrib_la_eigh_layout.py`, `tests/test_distrib_la_one_device*.py`,
  `tests/test_borrowed_slate_loaders.py`.
- **Native closure:** enforced at start-up for the final leg (§3).

## 8. Owner decisions

1. **One-device bound.** ELPA one-stage on 2x2 at 16384 c128 (14.1 s steady)
   beats the one-device program (42.4 s; native kernel 42.1 s, per the
   `resolve.py` rationale). `ONE_DEVICE_MAX_N` is left
   unchanged, which keeps the Fe route and its bitwise results. Lowering it
   would move results for 8192 < n ≤ 16384.
2. **Outside-arena allowance.** ELPA needs ~0.63 GB persistent per handle
   plus up to 2.75 GB transient at 51200 P=4, about 3.4 GB total, above the
   3 GB allowance. Options: raise the allowance on the ROCm `elpa` route, or
   route rocBLAS workspace through `rocblas_set_workspace` into the pool (not
   done).
3. **Workspace copy of A.** The workspace holds a copy of A (ELPA overwrites
   its input). Aliasing A to the output buffer would save n²·itemsize/P per
   rank.
4. **Two-stage.** Keep it refused above 46080 at p ≥ 4 until upstream fixes
   the defect (§4).

## 9. Code (branch `elpa`)

- `config/frontier/build_elpa_hip.sh` (new); `config/frontier/build_ffi_rocm.sh`
  builds or locates the ELPA prefix.
- `src/ffi/cpp/elpa/eigh_ffi.cc` (new): `ElpaEighFfi`.
  - Reuses the SLATE context's communicator (row-major JAX order).
  - Caches handles; the pool allocator lives in the XLA workspace, with
    first-fit, coalescing, the small-malloc rule and the live-after-call
    refusal.
  - `LORRAX_ELPA_POOL_REPORT=1` prints pool stats.
- `src/ffi/cpp/CMakeLists.txt`: `LORRAX_ROCM_HAVE_ELPA`, `LORRAX_ELPA_ROCM_DIR`.
- `services/distrib_la`: `_elpa.py` (new: layout, pool model, block-cyclic
  relabeling, sentinel padding); `resolve.py` (`elpa` backend; ROCm
  `distributed` eigh default with measured rationale); `plan.py`;
  `loader.py`; `bench/scale_distrib_la.py` (elpa layout column).
- `src/ffi/common/ffi_loader.py`, `src/ffi/common/families.py` (new `elpa`
  family); `services/lxkit/src/lxkit/native_provider.py` (elpa private and
  pinned).
- Tests: `tests/test_distrib_la_elpa_gate.py` (new: device gate plus CPU
  layout, pool-envelope and closure tests); `tests/test_platform_legs.py`.
- `UPGRADE_NOTES.md`: 2026-10-17 entry.

**Standalone driver:** `~/elpa_bench/elpa_bench.c` (artifact `elpa_bench.c`).
- Usage: `elpa_bench <z|d> n nb nprow npcol nrep ksample [key=val…]`, with
  `solver=1|2` among the key=val options.
- Prints a `RESULT {json}` line and ELPA's timer tree.

**Jobs:**

| Purpose | Job IDs |
|---|---|
| Two-stage P=4 / tuning | 5699666, 5699688, 5699850, 5699901 |
| Sweeps (two-stage) | 5699939 (P=4), 5699940 (P=16) |
| Defect diagnosis | 5699974, 5700031 |
| One-stage | 5700044, 5700053, 5700073, 5700074 |
| Gates (valid) | 5700195, 5700209, 5700227 |
| Invalid / failed | 5699600, 5699636 (failed setup); 5700087, 5700139, 5700163 (invalid gates, §2.9) |

**Reproduce:**
```
bash config/frontier/build_elpa_hip.sh --print-prefix      # build/locate ELPA
bash config/frontier/build_ffi_rocm.sh                      # ROCm leg with ELPA
srun -N1 -n4 -c7 --gpus-per-task=1 --gpu-bind=closest python -m tests.test_distrib_la_elpa_gate
srun -N1 -n4 -c7 --gpus-per-task=1 --gpu-bind=closest python services/distrib_la/bench/scale_distrib_la.py \
     --sizes 20480 --dtypes complex128 float64 --ops eigh --mesh 2x2
```
