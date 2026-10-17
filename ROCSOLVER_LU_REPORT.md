# ROCm one-device LU wrong answer: root cause, fix, audit

**Branch:** `rocm/lufix` (Frontier `~/wt/lufix`, **not pushed**), 4 commits on main `4bae15f55`:

| commit | content |
|---|---|
| `efed2af70` | LU route |
| `3b457db38` | report |
| `3ac2e4f8d` | triangular routes, f32/c64, round budget, audit |
| this report | this report |

**Patch:** `rocm_lufix.patch`. **Upstream draft (not filed):** `ROCBLAS_TRSM_UPSTREAM_DRAFT.md` and `trsm_repro.cpp`. **Harnesses and every step log:** `lurc_harness_and_logs.tgz`.

**Compute:**

| round | jobs | time | node-h |
|---|---|---|---|
| 1 | 5697579 (frontier07938) 51.5 min; 5697676 (frontier08572) 8.3 min | | 0.99 |
| 2 | 5697722 (frontier00513) 25.0 min; 5697770 pinned to frontier07938, cancelled before it started | | 0.42 |

## Deviations / scope
- **The hang mechanism on frontier07938 is not root-caused.** That node was allocated to another user for about 1.5 h, so I could not attach py-spy to a hung rank there. What I did establish (section 4): the hang is specific to that node and to a cold compile cache, it affects main the same way, and it is absent on a healthy node.
- The f64 `jnp.linalg.eigh` audit at n = 24576 does not finish. A py-spy dump shows it stuck **inside XLA compile** (`CompileAndLoad`) after 12 min, with one GCD at 100% use. This was reproduced on two nodes and is unresolved (section 5).
- There are no SLATE timings at n > 16384. The one_device route was not run at P=4 for n > 8192; large n was gated through the standalone local solves.
- The cross-version check against `/opt/rocm-10.0.0` cannot run on this driver.

## 1. Root cause (rocBLAS trsm, not rocSOLVER getrf, not jaxlib)
The defect is in `rocblas_{s,d,c,z}trsm`, side left, on its **in-place blocked path**. rocBLAS takes that path when m is a multiple of 128 and the RHS is wide; its workspace is about 160·m elements, against m·(nrhs+128) for the copy path.
- **Condition:** it returns a wrong X with status 0 when **(m − 128)(m + 1)·sizeof(T) ≥ 2^31**, i.e. once the byte offset of the last 128-wide diagonal block of A crosses 2^31.
- **Not the int32 API:** `rocblas_ztrsm_64`, `rocsolver_zgetrs` and `rocsolver_zgetrs_64` are equally wrong.
- **Factorization is fine:** `rocsolver_?getrf` was exact at every size measured.
- **Confirmed for all four dtypes** by the minimal repro (rocBLAS 5.5.0.cd957402), with exactly the predicted pairs:

| dtype | exact at | wrong at | residual when wrong |
|---|---|---|---|
| z | 11648 | 11776 | 6.9e-2 |
| d | 16384 | 16512 | 3.2e-2 |
| c | 16384 | 16512 | 4.5e-2 |
| s | 23168 | 23296 | 3.3e-2 |

- The same pairs hold through `jnp.linalg.solve` with nrhs = n, as relative residuals:

| dtype | at the exact size | at the wrong size |
|---|---|---|
| c128 | 1.1e-14 | 4.8e-3 |
| f64 | 6.2e-15 | 2.0e-3 |
| c64 | 8.0e-6 | 2.1e-3 |
| f32 | 4.5e-6 | 1.5e-3 |

## 2. Fix
- **Handlers.** `src/ffi/cpp/rocsolver/lu_solve_ffi.cc` (ROCM leg; rocSOLVER/rocBLAS resolved from the mapped `librocsolver.so.0`; s, d, c, z) provides two targets:
  - `lorrax_rocsolver_lu_solve`: getrf on the row-major A read as Aᵀ, then the blocked solve.
  - `lorrax_rocsolver_tri_solve`, three modes:
    1. apply jax's `(lu, piv)`;
    2. solve L x = b;
    3. solve Lᵀ x = b (Lᴴ is mode 3 on conj(b), with the conjugation fused into the layout copy).
  - **Rule:** no rocBLAS/rocSOLVER call gets an operand spanning ≥ 2^31 bytes from its own pointer:
    - nb = min(1024, ⌊L/n⌋), with L = INT32_MAX/itemsize;
    - RHS chunks satisfy (kc−1)n + n ≤ L;
    - transposed-panel GEMM row chunks satisfy (rows−1)n + k ≤ L.
  - Offsets are formed in 64-bit arithmetic. There are no syncs, workspace comes from XLA scratch, and there are no env vars or XLA flags.
- **Route.** `distrib_la.local_lu` (exported from `distrib_la`):
  - `solve_local`, `lu_solve_local` and `tri_solve_local` route a system whose A or B spans **more than 2^31 bytes**. That is a superset of the defect's condition at every n, which a CPU test asserts for m ≤ 40000.
  - Each takes the caller's **original call** as its below-span `native`, so every site is unchanged bit for bit below the span.
  - ROCM only: the CUDA cuSOLVER LU was measured exact through 16384.
- **Size rows** (`resolve.py`):
  - `ONE_DEVICE_MAX_N[('solve_lu','ROCM')]` = **27427**, the capacity of one GCD: isqrt(allocator `bytes_limit` 60,183,563,895 B ÷ the priced 5·n²·16 B).
  - There is no SLATE crossover: one-device takes 0.72, 1.53 and 4.10 s against SLATE 2×2's 3.63, 10.4 and 16.1 s at n = 8192, 12288 and 16384.
  - c128 28672 ran (22.7 s); 30720 runs out of memory.
- **Owner peak rule.** The new `ONE_DEVICE_ROUND_N` keeps main's per-round element allowance: solve_lu 8192², eigh 16384². `one_device_round_n` = min(selection bound, round row), and it is applied in `_one_device`, `plan._incumbent` and `workspace`.
  - **Effect:** a room-less (or any) stack gets main's round count exactly. A single matrix with 8192 < n ≤ 27427 goes one-device, one matrix per rank per round.
  - **Test** `test_room_less_round_counts_are_mains`: P = 4 and 16 on the Fe/hsuite stacks (64×312, 16×312, 4×96) and on binding stacks (256×2048, 64×4096, 8×8192, eigh 256×4096). Round counts equal main's, both directly and through `plan._incumbent`. At (256×2048, P = 4) main runs 4 rounds and 27427 alone would give 1, so the budget binds there.

## 3. Audit: device trsm-reaching call sites (`src/`, `services/`)
**Routed through `local_lu`.** Their n is the ISDF μ count or a basis size, so any of them can span ≥ 2^31 B once n ≳ 11.6k (c128):
- `gw/w_isdf` Dyson local LU (`lu_factor` + `lu_solve`);
- `isdf/core`: `jnp.linalg.solve(L_reg, Z)`, `lu_solve` (ζ back-solve), and the `lu_solve` on the backward-error probe vector;
- `isdf/galerkin`: `solve_triangular` ×2. The vmapped one becomes one column-batched solve on the routed path only;
- `common/zeta_projection`: the L and Lᴴ triangular solves;
- `bse/w_ladder`: T(I−T)⁻¹;
- `distrib_la/_native2d`: the panel `solve_triangular` (tiles of a 2-D Cholesky can be large at small P).

**Bounded by shape (no change):**
- `gw/head_correction`, `gw/photon_direct_head`: 4×4.
- `gw/mpa/pade_fit`: Padé order.
- `mixing/acceleration`: history depth.
- `psp/vnl_ops`: `inv` of the 3×3 lattice.
- All `np.linalg` / mpmath calls: host LAPACK.

**FFI:**
- SLATE trsm (device and host): SLATE calls BLAS per tile on tile allocations (nb ≤ 512), so each operand spans at most nb·max(nb, …)·16 B ≪ 2^31. **This answers the Dist LA question: SLATE's trsm cannot hit this overflow.** The full blocking of every SLATE path at n = 20k–51k is argued from the tile layout, not measured.
- cuSOLVERMp / ScaLAPACK: not ROCm.
- The rocsolver eigh FFI: heevd/syevd make no left-side trsm call with large m.
- Cholesky factorization itself (`jnp.linalg.cholesky`, rocSOLVER potrf) was not separately gated at large n. Only the triangular solves that follow it are routed.

## 4. The ISDF-basis hang (round 1)
- **frontier07938:** every Fe P=4 run with a **cold** cache namespace hung in "ISDF basis" (5 of 5): the branch library ×3, the branch library without the new TU, and **main's library with a fresh `FE_CACHE`**. Runs with a warm namespace passed (3 of 3, main's library). gdb on hung ranks shows every rank in `BlockUntilReady` / `AwaitBuffersReady` (device execution, not compile).
- **frontier00513 (round 2):** cold and warm runs, with the branch's own library and with main, all pass (ISDF basis 22.7 s cold for the branch, 12.2 s cold for main).
- **Ruled out:**
  - the hipRTC kernel store: built and stored in 0.9 s, and stored by atomic rename;
  - the lxffi key and closure: main's library hung the same way;
  - build or cache locks: none exist on that path.
- **Remaining explanation (provisional):** a node or GCD fault on frontier07938 that only fresh compilations expose. The likely trigger is device-side autotuning choosing kernels that hang there, while cached executables were tuned on healthy nodes. This matches the existing frontier05345 entry in the pool README.
- **Recommendation:** add frontier07938 to the pool's `--exclude` list.

## 5. eigh: `('eigh','ROCM')` 16384
c128 zheevd is exact at 20480 (60.5 s, 35.2 GB) and 24576 (120 s, 50.75 GB). **16384 is a measured-coverage bound, not a correctness bound.** f64 at 24576: compile never finishes (see Deviations), so the f64 side of the row above 16384 is unaudited.

## 6. Gates (round 2, final code, frontier00513)
- **Device gate** (`python -m tests.test_rocsolver_lu_gate --big --thresholds`): all PASS.
  - LU, routed:

    | dtype | n | relative residual |
    |---|---|---|
    | c128 | 11776 / 12288 / 16384 / 20480 / 27427 | 6.8e-15 / 7.2e-15 / 8.1e-15 / 8.9e-15 / 1.0e-14 |
    | f64 | 16512 / 20480 | 3.4e-15 / 3.7e-15 |
    | c64 | 16512 | 4.4e-6 |
    | f32 | 23296 | 2.1e-6 |

  - LU, forced small: 6 cases within about 2× of `jnp.linalg.solve`.
  - Triangular modes: lu, L, Lᴴ forced at n = 1000 and routed at c128 12288 (7.2e-15, 1.0e-17, 4.1e-17); Lᵀ forced at f32 n = 1000; L routed at c64 16512 (6.1e-9).
  - The native threshold report reproduced section 1.
- **Fe P=4 with the branch's own library** (`7dd40e…`) against main, cold and warm, each on a fresh cache: eqp max|d| = 0.0 eV, and all 24 HDF5 datasets are `array_equal`. eqp differs only in the timestamp header. Rank-0 device peak is 6.77 GB on all four runs. The final import layout was re-checked warm: still bitwise.
- **hsuite P=4 with the branch's own library:**
  - cold: PASS, 450.7 s;
  - warm, final code: PASS, 405.5 s;
  - an earlier warm run FAILED at gnppm because I edited `distrib_la/__init__` while it was running (an ImportError against the stale module). It was rerun on stable code.
- **CPU:** 162 passed in the LU/one_device/layering/crossfile set, plus measured/room/eigh-gate/compile-stability/memroute. The one remaining failure, `test_platform_legs::…refused_by_its_file_name`, fails identically on main in a checkout with a built library.
- **ROCm pytest:** `tests/test_rocsolver_lu_gate.py`, 11 passed, including the GPU regressions at c128 11776, f64 16512, c64 16512, f32 23296 and the three triangular routes.

## Not tested
- The one_device route at P=4 above n = 8192.
- The routed call sites inside full GW runs at μ > 11.6k: no such input was available. The routes are covered by the gate's direct calls.
- rocSOLVER potrf at large n.
- The CUDA leg: unchanged, no row.
