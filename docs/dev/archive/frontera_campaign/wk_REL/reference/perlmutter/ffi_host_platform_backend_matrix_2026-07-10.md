# FFI host-platform port — SLATE distributed linalg on the JAX CPU backend

_Branch `agent/ffi-host-platform` on lorrax_C (base `adc2197` = origin/main).
Executes the "FFI CPU story" follow-up spec'd in
`reports/slate_linalg_ffi_2026-07-10/report.md` (P2).  Plan: `PLAN.md`._

## Summary

The five SLATE FFI ops (potrf, trsm, eigh, batched potrf/trsm) now run on the
JAX **CPU backend** through host handler variants in a new **CUDA-free**
`liblorrax_ffi_host.so`, with GPU/CPU switching structured like JAX upstream:
the same target names are registered under `platform="CUDA"` (from
`liblorrax_ffi.so`) and `platform="cpu"` (from the host library), and
`jax.ffi.ffi_call` resolves the handler from the lowering platform — the
wrapper call sites and all layout plumbing (local transposes, MPI rank remap,
mesh/tile validation) carry unchanged.  `distributed_cholesky = slate` now
passes through on a CPU backend (still never auto-picked, still loud on
absence).  lorrax_C's GPU FFI was also rebuilt at main and re-verified first
(23/23 contract tests, 2×2 CLI clean).

## Code changes (commits `56df693`, `ec08b04`, `135241a`, `d30cfa4`)

| File | Change |
|---|---|
| `src/ffi/slate/cpp/host_ffi.cc` | NEW — host variants of all five handlers: `fromScaLAPACK()` on host buffers (same 2-D block-cyclic layout + `GridOrder::Col` as `fromDevices`), `Target::HostTask`, `memcpy` staging, no `PlatformStream` ctx; eigh keeps the `tileGetForReading` copy-out of the 2026-07-10 device fix.  Device TUs untouched. |
| `src/ffi/common/cpp/host/{CMakeLists.txt,build_host.sh}` | NEW — separable CUDA-free build (slate `gpu_backend=none` install + Cray-wrapper MPI only), host-side under the Cray PE; XLA FFI headers staged from the **container's** jaxlib (the runtime the .so must match — the host venv's jax 0.9.1 ships a newer FFI API); build fails if the result links any CUDA-stack library. |
| `src/ffi/common/ffi_loader.py` | Per-platform tables `{so, env override, targets, build hint}`; `get_lib(platform=None)` follows `jax.default_backend()`; CUDA-stack lifecycle helpers pinned to `get_lib("CUDA")`; slate lifecycle works through either library (`SlateCtx` is pure MPI, handles interchangeable). |
| `src/ffi/slate/{context,cholesky,trsm,eigh,batched}.py` | `ensure_registered(mesh)` picks the library from the **mesh's** device platform, so slate on CPU devices works inside a GPU-backend process (how the CPU tests run on GPU nodes). |
| `src/ffi/slate/batched.py` | Fix surfaced by mixed-platform testing: `_JIT_CACHE` keyed on mesh axis names/shape only, so a CPU 1×1 mesh reused the function jitted for the GPU 1×1 mesh.  Key now includes device platform + ids. |
| `src/gw/gw_config.py` | CPU backend: `slate` passes through; `auto`/`cusolvermp` still forced `off` (auto never picks slate; the resolver's auto→cusolvermp branch is therefore unreachable on CPU). |
| `src/isdf/core.py` | `_require_slate_ffi` failure message gains the host build pointer. |
| `tests/test_ffi_linalg_contract.py` | `test_slate_*_cpu` ×8 (same check bodies, 1×1 CPU-device mesh, skipif-clean without the host lib), `test_factor_c_q_slate_matches_reference_cpu`, platform-aware CLI (`JAX_PLATFORMS=cpu` runs slate cells, skips CUDA-only ones, prints `backend=`). |
| `src/ffi/slate/README.md`, `src/ffi/AGENTS.md` | CPU-story section rewritten (supported), layout tree updated. |

## Validation

| Check | Where | Result |
|---|---|---|
| GPU FFI rebuild at main (pre-port baseline) | GPU node (JID 55767308), in-container `--fresh` build | build clean; SLATE found; ldd complete |
| GPU contract suite (baseline) | 1 GPU | **23 passed** (no slate skips) |
| GPU multi-rank CLI 2×2 (baseline) | 4 ranks | **14/14 PASS, 0 failures**; smokes at 1.5e-16..5.5e-16 |
| Host .so build | compute node, Cray PE, no container | one-pass; **readelf NEEDED shows zero CUDA/NCCL** — slate/blaspp/lapackpp/libsci/MPICH/system only |
| CPU contract tests (host handlers) | in-container, GPU node, CPU devices | **7/7 passed** (chol+trsm c128/f64, rect m=16/48, batched, eigh c128/f64) |
| Mixed-platform single process | full contract file | **30 passed** (23 CUDA + 7 cpu) after the jit-cache-key fix |
| Multi-rank CPU, 2×2, 4 MPI ranks | CLI, `JAX_PLATFORMS=cpu` | `backend=cpu`; **all 10 slate cells PASS, 0 failures** (comm rank-remap + transposes verified distributed) |
| Full pytest suite (golden gates) | 1 GPU | first pass **212 passed / 0 failed (4:24)**; final (all review fixes + factor_c_q cpu) **213 passed / 0 failed (4:15)** |
| Multi-rank driver post-review-fix | 4 ranks, 2×2 | `common.slate_cholesky_trsm_test -n 256`: distributed init restored; residuals 3.1e-16 / 1.6e-16 / 5.4e-16 — identical to the pre-port baseline |
| Bare-metal CPU node (Milan, nid004156) | jax 0.9.1 runtime, srun, no container | **ldd clean; 7/7 pytest `-k cpu`; 2×2 CLI `backend=cpu … 0 failures`; 4×1 CLI 8 PASS 0 failures**; no env beyond `PrgEnv-gnu cray-libsci` + `JAX_PLATFORMS=cpu`.  Forward-compat confirmed: .so built with 0.7.2-era XLA FFI headers registers fine on the 0.9.1 runtime |
| Contract file after review fixes | 1 GPU | **31 passed** (incl. `factor_c_q` slate on cpu mesh); CLI 2×2 `0 failures` on BOTH `backend=cpu` and `backend=gpu` |
| Adversarial review (4 dimensions, 15 agents, verify-refutation pass) | workflow | 11 raw findings → **6 confirmed** (4 major, 2 minor) → all fixed in `d30cfa4` or documented; 5 refuted |

## Adversarial review outcome (workflow: 4 finders → per-finding refutation)

Confirmed and FIXED (`d30cfa4`):

1. **[major]** slate lifecycle helpers (`create_slate_context` etc.) resolved
   their library via the default backend, not the mesh's platform — a
   CPU-mesh op on a GPU-default machine without a loadable CUDA .so crashed.
   Helpers now take `platform=`; `slate/context.py` passes the mesh platform
   and remembers it per handle for teardown.
2. **[major]** Five pre-existing multi-rank drivers (`common/slate_*`) called
   bare `get_lib()` before `jax.distributed.initialize`; the new
   `get_lib(None)` → `jax.default_backend()` initialized XLA first, making
   distributed init raise (silently swallowed) — every driver degraded to N
   independent serial runs.  New backend-untouched
   `ffi_loader.platform_from_env()` (first `JAX_PLATFORMS` entry, shared with
   the contract CLI) restores them; 4-rank driver re-verified.
3. **[major]** `WfnLoader._auto_pick_backend` probe: bare `get_lib()` on a
   CPU backend now "succeeded" via the slate-only host library and selected
   the CUDA-only `phdf5` backend → hard crash at open instead of the eager
   fallback.  Probe (and `ffi/phdf5/context.py` ×3) pinned to
   `get_lib("CUDA")`.
4. **[major]** Dual-lib `libslate.so.2` soname collision: on GPU nodes the
   in-process `*_cpu` tests exercise the host handlers against the
   already-loaded cuda-build SLATE (host-side execution — supported config);
   the `gpu_backend=none` binary is what loads on CPU nodes, where it was
   validated bare-metal.  Documented in README + test docstring (no code
   change possible short of renaming sonames).
5. **[minor]** `_require_slate_ffi` probes the default platform — correct for
   every production path (the run mesh IS the default backend's); the
   CPU-mesh-inside-GPU-process corner is API-only.  Accepted, noted here.
6. **[minor]** CLI `JAX_PLATFORMS` parse used substring membership
   (`"cpu,cuda"` edge) — replaced by `platform_from_env()` first-entry parse.

Refuted by the verification pass (no action): CMake cache-var footgun
(build_host.sh passes `-D` every run), MPI stand-in on non-Cray compilers
(configure-time consequences acceptable, slateConfig has no
`find_dependency(MPI)`), readelf direct-NEEDED-only check (transitive CUDA
covered by the CPU-node ldd validation), `gw_config` backend query timing
(pre-existing), two `test_qp_solver_config` cases failing **under a CPU
backend** (pre-existing, unchanged by this diff — noted for a future pass).

## Design notes

- **JAX-upstream parallel**: jaxlib registers `lapack_*` under `platform="cpu"`
  and `cusolver_*` under `platform="CUDA"`; call sites are platform-blind.
  This port mirrors that exactly — one target name, N platform registrations.
- **Why a separate .so**: the CUDA library hard-links cuSOLVERMp/NCCL/cudart
  (fails `ctypes.CDLL` at `libnccl.so.2` on CPU nodes, P2 finding).  The host
  library links slate + MPI only and is the config that carries to non-NVIDIA
  machines.  ScaLAPACK host handlers (the `distributed_lu` axis) later join
  this same library and loader table.
- **Header/runtime version discipline**: XLA FFI headers must not be newer
  than the runtime.  `build_host.sh` stages `jax.ffi.include_dir()` out of the
  Shifter image once (`$HOME/software/lorrax_xla_ffi_headers/<tag>/`) instead
  of trusting whatever python is on PATH.
- **fromScaLAPACK ≡ fromDevices layout**: both interpret the local buffer as
  col-major block-cyclic tiles on a `GridOrder::Col` grid, so the device
  handlers' hard-won layout lessons (one-tile-per-rank invariant, per-dimension
  trsm X tiles, 1×q stride guard) apply verbatim and stay validated by the
  same `validate_tile_layout`.

## Status / next steps

- [x] GPU FFI rebuilt + verified on lorrax_C at origin/main
- [x] Host handlers, CUDA-free build, per-platform registration, config seam
- [x] Single-rank + multi-rank CPU validation in-container
- [x] Full-suite golden gates (212 passed / 0 failed; re-run post-review-fix)
- [x] Bare-metal CPU-node validation (Milan; pytest + 2×2 + 4×1 CLI all clean)
- [x] Adversarial review triaged: 6 confirmed findings fixed/documented
- [x] ScaLAPACK backend for `distributed_lu` on the host platform — landed
  2026-07-11, see the section below (`f0b17f3`)
- [ ] Two pre-existing `test_qp_solver_config` failures under a CPU backend
  (unrelated to this diff; surfaced by the review)
- [ ] Optional perf: SLATE HostTask thread pinning (`OMP_NUM_THREADS`) is
  unconfigured in the CPU path — defaults were fine for the contract sizes.

## Branch state

`agent/ffi-host-platform` on lorrax_C, 4 commits atop `adc2197`
(origin/main): `56df693` port, `ec08b04` jit-cache/device-identity fix,
`135241a` factor_c_q cpu test, `d30cfa4` review fixes.  UNPUSHED.
The in-tree GPU `liblorrax_ffi.so` (rebuilt at main) and the new
`host/build/liblorrax_ffi_host.so` are both current with these sources.

## 2026-07-11 extension — ScaLAPACK host backend for `distributed_lu` (`f0b17f3`)

**Where ScaLAPACK lives on NERSC**: inside **Cray LibSci** (`libsci_gnu_mpi`),
which the CC wrapper links implicitly and `liblorrax_ffi_host.so` already
carries for SLATE's BLAS — `nm -D` shows `pzgetrf_/pzgetrs_/pdgetrf_` plus the
C-BLACS bootstrap (`Csys2blacs_handle`, `Cblacs_gridinit`).  readelf NEEDED is
byte-identical before/after: **zero new dependencies**.  (MKL's ScaLAPACK
exists via `intel-oneapi` modules but targets Intel MPI — wrong ABI for Cray
MPICH; not needed.)

Design: fused per-q `pXgetrf`+`pXgetrs` handler
(`src/ffi/scalapack/cpp/solve_lu_ffi.cc`) mirroring the cusolvermp twin's
contract exactly (`P(None,'x','y')`, inner transposes, A donated/factored in
place, B aliased to X, pivots internal).  The BLACS grid is built once per
mesh from the slate `SlateCtx`'s rank-remapped comm — BLACS "C" grid order
lands JAX shard (mx, my) at grid (mx, my), the same pairing the slate
handlers use for `GridOrder::Col`.  `pXgetrf` requires square descriptor
blocks (`MB == NB`), satisfied with `g = N/max(Px,Py)` on square and BOTH 1-D
mesh orientations — including 1×q, which slate's stride assert forbids.
`distributed_lu = scalapack`: explicit only (never auto-picked), host-only
(GPU meshes get a loud pointer to cusolvermp), passes through the CPU-backend
config force, and flows through the SAME `solve_zeta` branch as
`cusolvermp_lu` (import switch only — ridge, logical-extent slicing, column
padding, reshard identical).

| Check | Result |
|---|---|
| host .so rebuild | clean; readelf NEEDED unchanged (CUDA-free, no new libs) |
| contract file, 1 GPU node (mixed platforms) | **35 passed** (31 prior + 4 scalapack) |
| multi-rank CLI `JAX_PLATFORMS=cpu`, 4 ranks | 2×2, 4×1, **1×4**: 4/4 scalapack cells PASS each, `done: 0 failures` |
| full pytest suite | **217 passed / 0 failed (5:36)** at `f0b17f3`; re-run green after the review-fix guards |
| bare-metal Milan CPU node (nid004149) | **12/12 pytest** (`-k "cpu or scalapack"`, zero skips); 2×2 / 4×1 / 1×4 CLI all `backend=cpu … 0 failures` against bare `/opt/cray/pe` libsci |
| adversarial review of `f0b17f3` | 1 confirmed defect (low-severity, fail-loud-but-late) + 1 stale comment — both fixed; ALL numeric checks refuted with hand-derived index math |

**Review outcome**: the confirmed defect was a seam asymmetry —
`distributed_lu = scalapack` on a GPU-backend run passed config validation and
the resolver, and only failed (with a correct, loud ValueError) at the first
transverse solve, i.e. AFTER the expensive C_q build.  Fixed with parse-time
rejection in `gw_config` (non-CPU backend + scalapack → immediate ValueError)
plus a defense-in-depth device-platform check in the resolver.  Refuted with
explicit math: BLACS "C"-order grid mapping on the remapped comm (2×2/4×1/1×4
per-element offsets all coincide with the transposed JAX shards), numroc-vs-
XLA buffer extents (no OOB on any allowed mesh), ipiv sizing/reuse across q,
in-place A mutation under failed donation (scribble target is the transpose
temporary, same convention as the cusolvermp twin), int32 truncation, the
shared solve_zeta branch's backend-agnosticism, and the BLACS-context
pointer-keyed cache lifetime.

## 2026-07-11 — e2e backend timing + equivalence matrix (bispinor GN-PPM)

Run set: `runs/MoS2/C_bispinor_backend_timing_2026-07-11/` (manifest has full
execution notes).  System: MoS2 3×3 nspinor=2 bispinor GN-PPM (the regression
fixture geometry), 256 charge / 208 transverse centroids — the transverse set
regenerated `--no-orbit` so its count divides the mesh (the fixture's 209 is
odd → the distributed LU would silently fall back); all runs therefore need
`LORRAX_FORCE_FULL_BZ=1` and are **only comparable within this experiment**.
Mesh 2×2 everywhere; GPU = 4×A100 (1 node, in-container jax 0.7.2); CPU = 1
Milan node, 4 MPI ranks × 32 cores (in-container, `JAX_PLATFORMS=cpu`; the
native venv path is broken — see KNOWN_SANDBOX_ERRORS 2026-07-11 on §3.5).

| # | Platform | `distributed_cholesky` | `distributed_lu` | Wall [s] | Recorded [s] | charge chol [s] | transverse solve [s] | σ vs platform baseline |
|---|---|---|---|---|---|---|---|---|
| 00 | GPU | off (sharded_cholesky) | off (per-q lu) | 73 | 52.5 | 2.89 | ~1.5/chan | baseline |
| 01 | GPU | cusolvermp | cusolvermp | 80 | 58.6 | 3.70 | ~3.0/chan | **0.000e+00** |
| 02 | GPU | slate | off | 74 | 54.4 | 4.48 | ~1.5/chan | **0.000e+00** |
| 03 | CPU | off (sharded_cholesky) | off (per-q lu) | 273 | 261.5 | 0.73 | 4.4 | baseline |
| 04 | CPU | slate | off | 274 | 262.6 | 1.56 | 4.4 | see below |
| 05 | CPU | off | **scalapack** | 274 | 260.1 | 0.7 | 3.1 | **0.000e+00** |
| 06 | CPU | slate | **scalapack** | 271 | 258.4 | ~1.5 | ~3.1 | see below |

Reading the numbers honestly: at this fixture scale the solver axes are
noise — `zeta_fit.chunk.z_q_build` dominates (~44 s × 4 channels on CPU),
so wall clock is backend-insensitive (GPU 73–80 s, CPU 271–274 s).  The
table's value is **engagement** (every cell's `path=` banner confirmed, no
silent fallbacks) and **equivalence** (bit-identical σ across backends).
Backend performance discrimination needs production scale (CrI3-class
n_rmu, where cuSOLVERMp's panel-loop advantage is the documented win).

Two main-branch bugs the matrix flushed out:

1. **`lax.pcast` (jax-0.9 API, commit `c7e6695`) broke `sharded_cholesky` on
   multi-rank meshes under the container's jax 0.7.2** — the 1-GPU test
   suite never enters that branch.  Fixed on this branch with a
   version-guarded identity shim (`1421db1`); variant 00 then reproduced the
   FFI backends bit-exactly.
2. **Host-platform ffi_call concurrency**: XLA's CPU runtime executes
   independent ffi_calls concurrently on its thread pool; factor_c_q's
   per-q slate potrf calls raced their MPI collectives on the shared comm —
   intermittent single-q-tile corruption (04/06 first runs off by
   0.65–0.95 eV in sigC with exploded GN invalid-mode counts, while a repeat
   and every single-call contract test were bit-clean).  Fixed with a shared
   serialization mutex across all host handlers (slate + scalapack; the CUDA
   handlers are stream-serialized and unaffected) — commit `4085672`.
   Determinism campaign post-fix: **6/6 e2e trials bit-identical to the
   in-tree baseline** (4× slate, 2× slate+scalapack, Milan node, 2×2 mesh;
   `max|Δ| = 0.000e+00` each, healthy invalid-mode signatures, engagement
   banners confirmed).  Pre-fix rate was 2 corrupt of 3.  With the fix, rows
   04 and 06 of the table are bit-identical to baseline like every other
   cell.

`distributed_lu = scalapack` — the axis this experiment set out to prove —
was **bit-identical to the in-tree baseline on its first e2e run** (05 vs
03 = 0.000e+00) and unaffected by the concurrency bug (its per-channel
solves dispatch sequentially).
