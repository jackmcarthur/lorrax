# Known test failures — full-suite census

Two censuses live in this file.  The **Perlmutter** one is authoritative for
this tree; the **Frontera** one below it is the historical record from
2026-08-01 and is kept because several of today's reds are only legible
against it.

A release ships LISTED known-fails, never unknown ones: every non-passing
test in every leg is accounted for below, and every "it is the environment"
claim carries the arm in which it comes out FALSE.

---

# Perlmutter census — `svc/distrib_la-2026-08-07` @ `d5cac09` (2026-08-07)

**First Perlmutter full-suite census in this tree.**  The Frontera census
below was the only one that existed, and three of its "fixed in this pass"
re-freezes turn out to be *platform-local* — see class **P1**.

| | |
|---|---|
| machine | Perlmutter, 1 node, 4×A100-SXM4-40GB, Shifter `ghcr.io/nvidia/jax:jax-2025-07-21` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax `0.7.0.dev20260807`, python 3.12 |
| tree | `/pscratch/sd/j/jackm/svc_distrib_la/lorrax`, `LORRAX_CHECKOUT` (source-tree line read on every leg) |
| `.so` pins | `LORRAX_FFI_SO=~/software/lorrax_ffi_2026-08-07/liblorrax_ffi.so`, `LORRAX_FFI_HOST_SO=/pscratch/sd/j/jackm/svc_distrib_la/build_host_h200/liblorrax_ffi_host.so` (md5 `4c4422b8…`), `LORRAX_FFTW3_SO=~/software/lorrax_fftw_cray/stage/lib/libfftw3.so.mpi31.3.6.10` — BUILD_NOTES 2026-08-07 |
| jobids | **56447670** (gpu pool, all legs except F), **56446562** (Milan cpu pool) |
| artifacts | `/pscratch/sd/j/jackm/svc_distrib_la/_reports_step6/` — one `.log` + one `.xml` per leg, sizes quoted below |

> **Which commit the legs ran at.**  Every leg below ran at `e9340d1`.  The
> two commits since — `d5cac09` (bench baselines) and `efdbf9a` (this file)
> — touch four `.json` under `services/distrib_la/bench/baselines/` and one
> `.md`.  `pyproject`'s `norecursedirs` excludes `bench`, so neither is
> importable or collectable, and `pytest --collect-only` at `e9340d1` and at
> `efdbf9a` returns the **same 1441 ids, diff empty**.  Legs B, E and E2
> were additionally RE-RUN at `efdbf9a` and came back byte-identical
> (130/108P/22S, 120/120P, 130/127P/3S).

> **AND WHICH COMMIT THE FIXES WERE VERIFIED AT.**  `7a1d64f` (B1) and
> `f7c1b17` (B2) land after this census.  Every leg in *§ FIXED AFTER THE
> CENSUS* below was re-run at `f7c1b17` on the same pins and the same node
> class, artifacts in `_reports_fix/`; the leg table above is left as the
> census measured it, at `e9340d1`.

## Verdicts by leg

| leg | invocation | collected | passed | failed | error | skipped |
|---|---|---|---|---|---|---|
| **A** full suite, services deselected | `lx test -- tests/ --no-services -q -rs -p no:randomly` | 1191 | 1120 | 8 | 1 | 62 |
| **A2** full suite WITH services | `lx test -- -q -rs -p no:randomly` (testpaths = tests + services) | 1441 | 1326 | 9 | 22 | 84 |
| **B** full-suite marker leg | `lx test -- -m distrib_la -q -rs` | 130 | 108 | 0 | 0 | 22 |
| **E** lxkit by path | `lx test -- services/lxkit/tests -q -rs` | 120 | 120 | 0 | 0 | 0 |
| **E2** distrib_la by path | `lx test -- services/distrib_la/tests -q -rs` | 130 | 127 | 0 | 0 | 3 |
| **C** device-hungry, 4 emulated host devices | `lx run env XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python3 -m pytest <9 files>` | 146 | 143 | 1 | 0 | 2 |
| **C2** non-square refusal, 3 emulated devices | `… device_count=3 … -k nonsquare` | 1 | 1 | 0 | 0 | 0 |
| **C3** two-device cells | `… device_count=2 … tests/test_charge_zeta_route.py` | 15 | 7 | 0 | 0 | 8 |
| **D** `extra` tier | `lx test -- -m extra -q -rs` | 26 | 23 | 1 | 0 | 2 |
| **F** L-c REAL 4-process CPU 2×2 | `lx run -N 1 -G 4 -n 4 env JAX_PLATFORMS=cpu python3 …test_distrib_la_multiproc.py --mesh 2x2` | 14 cells | 12 | **0** | 0 | 2 |
| **G** L-c REAL 4-process GPU 2×2 | `lx run -N 1 -G 4 -n 4 … --mesh 2x2` (serialized) | 13 cells | 10 | **0** | 0 | 3 |

Legs F and G are the hostile-geometry / real-multi-process tier and are the
only legs that exercise ScaLAPACK, SLATE and cuSOLVERMp on four real ranks.
Both are clean; their residuals are quoted in
`docs/services/distrib_la.md` § Performance.

### Isolation and falsification legs (not census legs — evidence)

| leg | invocation | result | what it settles |
|---|---|---|---|
| `iso_reds` | every leg-A red, ONE process, `lx run … -m pytest` | 58 / **54 P / 4 F** | 4 of the 9 leg-A reds survive isolation; 5 do not |
| `iso_bse` | the five BSE session files, ONE process | 35 / **35 P** | the A2 21-error cascade is not per-test |
| `xdist_arm` | the SAME six files under `lx test` (xdist, 4 workers) | 24 / 7 P / **17 E** | the cascade is the LAUNCHER |
| `base_xdist` | `xdist_arm` at the branch base `96a6399` | 24 / **24 P** (2 runs) | the cascade is a REGRESSION ON THIS BRANCH |
| `falsify_wfnloader` | `test_no_ffi_at_P_gt_1…` with the FFI pins UNSET | **1 P** | that red is the pin |
| `falsify_aot` | `device.memory_stats()` under `XLA_PYTHON_CLIENT_ALLOCATOR=platform` vs `=bfc` | `None` vs a 10-key dict | that red is the allocator |
| `bisect_fileio` | `tests/test_file_io.py` on the CPU platform at `96a6399` / `b3f3675` / `32e61fe` / HEAD | 42P·1S / 42P·1S / **ABORT** / **ABORT** | names the commit |
| `loadorder` | HEAD, CPU platform, `LORRAX_FFI_SO` unloadable vs pinned | **42P·1S** vs ABORT | names the LINE |
| `bisect_xdist` | `xdist_arm` at `78ddcee` / `6920171` | 24/24 P vs 11P·2F·11E | names the second commit |
| `cvd_probe` | each xdist worker prints its own `CUDA_VISIBLE_DEVICES`, at `78ddcee` and at HEAD | `0,1,2,3` vs `0,0,0,0` | names the mechanism |

---

## FIXED AFTER THE CENSUS — the two reds this branch made

Both are fixed on this branch and re-verified; the diagnosis below is the
census's, unchanged, and each row now carries the arm that closes it.  The
rows stay here rather than disappearing: a census that deletes what it
found cannot be audited against the next one.

**Re-verification, branch tip `f7c1b17`, same node class, same BUILD_NOTES
pins, artifacts in `/pscratch/sd/j/jackm/svc_distrib_la/_reports_fix/`
(one `.log` + one `.xml` per leg):**

| leg | census @ `e9340d1` | fix tip @ `f7c1b17` | reference it must match |
|---|---|---|---|
| `test_file_io.py`, CPU platform, 4 emulated devices | **ABORT** | **42 P / 1 S** | base `96a6399`: 42 P / 1 S |
| cvd probe, 4 xdist workers | `'0','0','0','0'` | **`'0','1','2','3'`** | `78ddcee`: `'0','1','2','3'` |
| xdist arm, 6 gnppm-session files | 7 P / 17 E | **24 / 24 P** | `78ddcee`: 24 / 24 P |
| **B** full-suite `-m distrib_la` | 130 / 108 P / 0 F / 22 S | **140 / 118 P / 0 F / 22 S** | +10 new cells, 0 F, skips unchanged |
| **E** lxkit by path | 120 / 120 P | **120 / 120 P** | unchanged |
| **E2** distrib_la by path | 130 / 127 P / 3 S | **140 / 137 P / 0 F / 3 S** | +10 new cells, 0 F, skips unchanged |
| **A** full suite, services deselected | 1191 / 1120 P / 8 F / 1 E / 62 S | **1211 / 1143 P / 6 F / 0 E / 62 S** | 0 newly red, 3 newly green |
| **A2** full suite with services | 1441 / 1326 P / 9 F / 22 E / 84 S | **1471 / 1381 P / 6 F / 0 E / 84 S** | 0 newly red, **25 newly green** |
| WSL full suite (jax 0.9.1, no FFI) | 1441 / 95 red | **1471 / 95 red** | set-diff EMPTY both directions |
| `python3 tests/test_layering.py` | 75 / 75 | **75 / 75** | unchanged |

The six reds left in A/A2 are P1 (3), P2 (2) and P3 (1) below — every one
pre-existing or an owner row, none from this branch.  Leg B is the tension
point and it holds: a CUDA-capable process still opens CUDA first (the
`-m distrib_la` leg's SLATE cells are green, and
`test_the_blaspp_the_cuda_slate_calls_can_see_the_device` passes there),
while the CPU-platform leg opens the host library and nothing else.

### **B1 — FIXED by `7a1d64f`: `_open_cuda_before_host` broke the host phdf5 path**

| | |
|---|---|
| tests | `tests/test_file_io.py` — `test_read_slab_without_shape_rounds_up_to_the_mesh`, `test_slabio_implicit_pad_write_and_zero_padded_read` and every SlabIO write cell after them, on any leg whose jax platform resolves to **cpu** |
| class | (a) real defect, introduced by this branch — **FIXED, `7a1d64f`** |
| covering leg (census) | none — the GPU legs (A/A2) did not reach it, the CPU leg died in it |
| covering leg (now) | the CPU-platform `test_file_io` leg itself, **42 P / 1 S** at `f7c1b17`, plus the two-armed loader cells on every machine |

`src/ffi/common/ffi_loader.py:576-613` (commit **`32e61fe`**, "the two FFI
libraries share their SLATE, and the first one opened wins") makes
`get_lib("cpu")` call `_open_cuda_before_host()`, which `dlopen`s the CUDA
FFI library `RTLD_GLOBAL` first.  That is correct for SLATE — and it is
measured — but in a process whose jax backend is **cpu** the host phdf5 slab
handlers are then answered across the SONAME boundary.  Symptom at the
handler: the read refuses with

    phdf5 read: logical slab out of bounds
      extent=[2,4,1,6]
      offset_base=[0,0,0,4596944070643295330]
      valid_shape=[3,6,6,4609783128842618077]

Those two integers are IEEE-754 float64 bit patterns (≈0.19 and ≈1.87) read
as `int64` — a different handler's argument layout, which is what "the wrong
library answered" looks like.  Where it does not refuse, it aborts inside
the async writer thread (`common/async_io.py:135` → `_slab_io_ffi.py:1749` →
pjit → `Fatal Python error: Aborted`).

**BISECT, one fast deterministic leg (`tests/test_file_io.py`, CPU platform,
4 emulated host devices), all four arms on the same pins:**

| commit | | result |
|---|---|---|
| `96a6399` | branch base | 42 passed / 1 skipped |
| `b3f3675` | the commit *before* the load-order rule | 42 passed / 1 skipped |
| `32e61fe` | **the load-order rule** | 3 failed, then `Fatal Python error: Aborted` |
| `d5cac09` | HEAD | hang → 300 s wall |

**FALSIFICATION (the arm where the hypothesis comes out false), at HEAD:**
point `LORRAX_FFI_SO` at a path that cannot be `dlopen`ed, so
`_open_cuda_before_host`'s best-effort `get_lib("CUDA")` raises and is
swallowed and the process stays host-only — **42 passed / 1 skipped**, byte
for byte the base result.  Same leg with the CUDA `.so` pinned: 300 s wall.

**FIXED, `7a1d64f`** — "the CUDA-first pre-open is for processes that can
use CUDA, and only those".  Not a revert: the SLATE SONAME collision is
real and `32e61fe`'s evidence stands, so a CUDA-capable process still opens
CUDA first.  The pre-open is now gated on `_process_can_use_cuda()` —
`JAX_PLATFORMS` resolved first-entry-wins plus a visible NVIDIA device
(`CUDA_VISIBLE_DEVICES=""` explicitly masked, else a `/dev/nvidia*` node),
the same two signals in the same order as `runtime._gpu_is_present`.  It is
truthful AT LOAD TIME, which is why it is not `jax.default_backend()`: that
INITIALIZES the XLA backend, so asking it inside a loader call would make
the loader decide the process's platform.  Applied in BOTH loaders
(`src/ffi/common/ffi_loader.py`, `services/distrib_la/src/distrib_la/loader.py`).

VALIDATION: `tests/test_file_io.py`, CPU platform, 4 emulated devices, the
CUDA `.so` **pinned** (not the census's unloadable-`.so` falsification arm)
— **42 passed / 1 skipped**, the base `96a6399` result, at
`_reports_fix/fix_fileio.xml`.  The CUDA arm is unharmed: leg B is 140
cells / 0 failed / 22 skipped and
`test_the_blaspp_the_cuda_slate_calls_can_see_the_device` passes in it.

The four order cells are two-armed now, both sides with a red twin
(`test_a_cpu_platform_process_never_opens_the_cuda_library` +
`test_the_cpu_platform_cell_can_fail`, and the same pair for lorrax's
loader in `tests/test_gpu_pinning.py`), plus an 8-row table per loader
constructing every input of the capability gate.  The CPU-platform arm's
red twin is the load-bearing one: without it that cell stays green on any
machine with no CUDA library to find, which is every WSL leg.

**The ABORT itself is a SECOND defect and it is registered, not fixed —
see L1.**

### **B2 — FIXED by `f7c1b17`: the xdist CONTROLLER narrowed the workers' preset**

| | |
|---|---|
| tests | leg A: `test_bse_bgw_regression::test_bse_matches_frozen_and_bgw`, `test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q`, `test_restart_pad_roundtrip::test_restart_mu_pad_roundtrip` (worker crash).  Leg A2: those plus `test_bse_kgrid` (2) and a 21-error cascade over `test_bse_dense_reference` (12), `test_bse_stack_matvec` (3), `test_bse_w_omega_chain` (2), `test_bse_matvec_opts` (2), `test_bse_w0_resolvent` (2) |
| class | (a) real defect, new on this branch — **root-caused to `6920171`**, second symptom layered on at `32e61fe`; **FIXED, `f7c1b17`** (and `32e61fe`'s share by `7a1d64f`) |
| covering leg (census) | the SINGLE-PROCESS leg: `iso_bse` **35/35 pass**, `iso_reds` passes all of these |
| covering leg (now) | the xdist leg itself — **24 / 24 P**, and leg A2's 22-error cascade is 0 |

Every one of these cells is green in one process and red under `lx test`
(1 node, all GPUs, 4 xdist workers, one GPU pinned per worker).  The
session fixture dies with no traceback immediately after

    [restart_write] W0_qmunu (9, 399, 399) 0.02 GB QUEUED in 0.0 s
    [SlabIO.close] draining 1 pending writes for isdf_tensors_399.h5 …

and, separately, reads come back
`ValueError: INVALID_ARGUMENT: phdf5 read: ctx_handle is null`.

MEASURED, same six files, same launcher, same pins, two runs per arm:

| commit | result |
|---|---|
| `96a6399` (base) | 24 / **24 passed** |
| `b3f3675` | 8 passed / 16 error — but `RESOURCE_EXHAUSTED: Failed to allocate 19.9 GiB on device ordinal 0` |
| `d5cac09` (HEAD) | 7 passed / 17 error — the SlabIO drain death |

TWO regressions are stacked here.  Both are now root-caused.

**The memory one is `6920171` ("tests/conftest: the GPU pin was conditional
on xdist, and that hid a leg"), and it puts EVERY xdist worker on GPU 0.**
Bisected on the same arm, then measured directly with a throwaway probe
test that prints its own `CUDA_VISIBLE_DEVICES` (written into the clone for
one run, deleted; both trees verified clean afterwards):

| commit | gw0 | gw1 | gw2 | gw3 | xdist arm |
|---|---|---|---|---|---|
| `78ddcee` (before) | `'0'` | `'1'` | `'2'` | `'3'` | 24 / **24 passed** |
| `6920171` (after) | `'0'` | `'0'` | `'0'` | `'0'` | 11 P / 2 F / 11 E, `RESOURCE_EXHAUSTED … 19.20GiB on device ordinal 0` |
| `d5cac09` (HEAD) | `'0'` | `'0'` | `'0'` | `'0'` | 7 P / 17 E |

MECHANISM.  The pin used to be written only when `PYTEST_XDIST_WORKER`
started with `gw`.  Unconditional, the xdist **controller** — which has no
worker id, so `pin_one_gpu` returns `devs[0]` — now writes
`CUDA_VISIBLE_DEVICES="0"` into its OWN environ at `tests/conftest.py`
module scope.  The workers inherit that environ, so each of them sees
`preset="0"`, `_visible_gpus` returns a ONE-element list, and
`int(worker_id[2:]) % 1 == 0` for all four.  The fan-out
`tests/harness.py:54-78` documents is dead, and four gnppm sessions land on
one 40 GiB A100 instead of four.

`pin_one_gpu` itself is correct and its unit tests pass — they construct
the preset by hand and never see the controller's write.  The gap is the
caller, and `tests/test_gpu_pinning.py` is where the twin belongs: *four
worker ids must map to four distinct devices even when the controller has
already pinned one*.

**FIXED, `f7c1b17`** — "tests/conftest: the xdist CONTROLLER must not
narrow what its workers inherit".  Not a revert: `6920171`'s three reasons
and its 11 cells are untouched, and a plain non-xdist run is still pinned
(that is the leg `6920171` existed to un-hide).  The controller — and only
the controller — takes no device, detected with pytest-xdist's own
predicate: no `PYTEST_XDIST_WORKER` **and** `config.option.dist != "no"`
(`harness.is_xdist_controller`).  `-n 0` and a missing xdist plugin both
leave `dist == "no"`, so both stay pinned.  The pin moves from conftest
module scope to `pytest_configure`, which is where `config` exists and is
still before collection — hence before the first test-module import, hence
before jax.

VALIDATION, all at `_reports_fix/`:

| arm | result |
|---|---|
| cvd probe, 4 xdist workers | gw0..gw3 = **`'0','1','2','3'`** (`fix_cvd.log`) |
| the same probe against the PRE-FIX conftest (`git show 35f3e06:tests/conftest.py`) | **`'0','0','0','0'`, 1 distinct device of 4 — RED ARM FIRES** (`fix_redarm.log`) |
| xdist arm, the 6 gnppm-session files | **24 / 24 passed** (`fix_xdist.xml`) |
| leg A2 | the 22-error cascade is **0**, 25 cells newly green, 0 newly red |

`tests/test_gpu_pinning.py` grows the controller-does-not-pollute twin —
the census probe frozen into the suite: it copies this conftest + harness
into a tmp rootdir (never writing into the checkout), spawns the real
4-worker arm and reads each worker's own `CUDA_VISIBLE_DEVICES` back out of
a file (a worker's stdout does not reach the controller under xdist, `-s`
included — measured).  It scrubs `PYTEST_XDIST_*` from the child env,
because under `lx test` the cell runs INSIDE a worker and the child's
controller would otherwise inherit `PYTEST_XDIST_WORKER=gw2` and pin
`devs[2]` for all four — measured, and the same defect shape from the other
direction.  Beside it: `test_the_controller_takes_no_device_at_all`, its
red twin, and a 7-row `is_xdist_controller` table whose `("", "no")` row is
the non-xdist leg that must still be pinned.

> The compile cache is NOT the cause of the `ctx_handle is null` reads.
> Arm run: same files with `ISDF_JAX_CACHE_DIR=""` → 4 passed; but the
> control (cache ON, same isolation) is *also* green, so the arm is not
> discriminating and the cache hypothesis is unsupported.  Recorded so
> nobody re-derives it.


## SHIP-LISTED FAILURES

### **L-3 — cuSOLVERMp `eigh` with `compute_evecs=False` fails at every `n`** (library defect, now REFUSED at resolve time)

| | |
|---|---|
| tests | none go red: the combination is refused before it can run.  Contract cells `test_resolve_cusolvermp_eigh_refuses_compute_evecs_false`, `test_resolve_cusolvermp_eigh_compute_evecs_true_is_unchanged`, `test_cusolvermp_wrapper_refuses_compute_evecs_false_without_a_gpu` |
| class | (e) third-party library defect — cuSOLVERMp 0.7.2, not LORRAX code |
| evidence | `XlaRuntimeError: INTERNAL: cusolverMpSyevd failed: status=7` (`CUSOLVER_STATUS_INTERNAL_ERROR`) at **n = 64, 256, 1024, 4096**, real 4-process 2×2 CUDA mesh, jobid **56447670**.  `/pscratch/sd/j/jackm/svc_distrib_la_perf/_reports/perf_gpu2x2_decomp.json` rows `"g2. distributed_eigh jobz='N'"`, reproduced independently in `perf_gpu2x2prof_decomp.json` (the `LORRAX_FFI_PROFILE=1` leg) |
| covering leg | the step-6 eigh investigation's `gpu_decomp` and `gpu_decomp_prof` legs |

`compute_evecs=False` (jobz='N') is a **documented parameter** of
`_cusolvermp.distributed_eigh` and it has never worked.  It is not a
wrapper bug: `cusolverMpSyevd_bufferSize` **succeeds** for jobz='N', and
`src/ffi/cpp/cusolvermp/eigh_ffi.cc:106` is a one-line pass-through
(`const char jobz = compute_evecs ? 'V' : 'N'`), so the library sizes the
eigenvalues-only solve and then fails to run it.

**NOT CHASED**, deliberately: no workaround flag was identified in 0.7.2
and chasing a vendor library bug is out of scope for this branch.

**REFUSED, PERMANENTLY, NOT AS A STOPGAP.**  The owner confirms
(2026-08-07) that LORRAX wants `compute_evecs=True` in every case they
can think of, so this parameter's only remaining value on this backend
was a route to an unexplained `INTERNAL_ERROR` three call frames deep.
`resolve.resolve_backend` guard **2c** refuses it, and
`_cusolvermp.distributed_eigh` refuses it again as its first statement —
both, because `Plan.__call__` forwards `**kwargs` straight to the wrapper
(`plan.py:318`) and a resolve-time-only rule would have a hole in it.
A caller who wants eigenvalues only should pass `compute_evecs=True` and
ignore `Q`, or use `jnp.linalg.eigvalsh`.

### **L-4 — SLATE `eigh` SIGSEGVs at n ≥ 4096 on a multi-rank CUDA mesh** (library defect, now REFUSED at resolve time)

| | |
|---|---|
| tests | none go red: the combination is refused before it can run.  Contract cells `test_resolve_slate_cuda_eigh_refuses_at_4096`, `test_resolve_slate_cuda_eigh_still_resolves_at_2048`, `test_resolve_slate_cpu_eigh_at_4096_still_says_the_L2_thing` |
| class | (e) third-party library defect — the **CUDA sibling of L-2** (SLATE host `heev`), same library, same routine family |
| evidence | `srun: error: nid001088: task 0: Segmentation fault` → step exit **139**, all four ranks down.  jobid **56457930**, `/pscratch/sd/j/jackm/svc_distrib_la_perf/_reports/gpu_slate4096_segv.log` — a leg run for the sole purpose of producing this artifact |
| covering leg | the step-6 `gpu_slate4096` leg.  **NOT** `gpu_cross_size.log:51`, which the brief flagged as overwritten: that file now shows the *skip* line, not the crash |

The size sweep **skips** this cell (`gpu_cross_size.log:35`, "slate nq=0
n=4096 -> SKIPPED (known SIGSEGV at n>=4096)") because a crash there takes
the other 20 rows down with it.  The dedicated leg exists so the skip is
backed by an artifact rather than by a memory.

**SIZE-SCOPED, not a removal.**  SLATE eigh RETURNS on the same 2×2 CUDA
mesh at every smaller size measured — 0.401 / 0.546 / 1.387 / 5.444 s at
n = 64 / 256 / 1024 / 2048 (jobid 56447670).  Refusing the whole backend
would delete a working route, and `distributed` eigh on **ROCm** maps to
slate, so "delete slate" is not on the table either.  Nothing between 2048
and 4096 was tried, so the true threshold is somewhere in (2048, 4096];
4096 is the smallest size measured to crash, and erring toward refusal is
correct when the failure mode is a SIGSEGV with no Python traceback.

**A 1×1 CUDA mesh at n ≥ 4096 is UNMEASURED and is NOT refused.**  The
crash was only ever produced multi-rank.  Guard 2d says so, and a contract
cell pins it — this package refuses what someone has watched fail, not
what seems likely.

### **L1 — the two platform `.so`s cross-wire their phdf5 through RTLD_GLOBAL** (latent, registered by B1's fix)

| | |
|---|---|
| tests | none can reach it today; it is what B1's ABORT *was* |
| class | (b) pre-existing, structural — a cross-`.so` ODR violation in the C++ |
| covering leg | none.  The B1 fix keeps it LATENT for host-only processes by never putting one into the mixed state; it is NOT latent for a CUDA-capable process that also does host phdf5 work |

`liblorrax_ffi.so` and `liblorrax_ffi_host.so` collide on far more than
`libslate`/`libblaspp`.  MEASURED 2026-08-07, `nm -D --defined-only` on the
two BUILD_NOTES-pinned builds (deployed device lib; `build_host_h200`,
md5 `4c4422b8…`): **sixteen symbol names are DEFINED BY BOTH** — the nine
C-linkage `lrx_phdf5_*` / `lrx_slate_*` entry points and seven mangled
`lorrax_ffi::phdf5::{open_ctx,close_ctx,ensure_dataset,ensure_read_buf,
ensure_pinned,ensure_mpi_initialized,open_dataset_ro}`.  Both files are
dlopened `RTLD_GLOBAL`, so once both are open the FIRST one answers those
names for BOTH — including for the other library's own internal calls.

And they are not the same functions.  `src/ffi/cpp/phdf5/ctx.h` compiles
`PhdfCtx` with the CUDA stream / event / pinned-buffer members under
`#ifndef LORRAX_FFI_NO_CUDA`; the host build defines that macro and drops
them.  **One C++ type name, two struct layouts, both exporting
`open_ctx(...) -> PhdfCtx*`.**  A handler from one build handed a
`PhdfCtx*` minted by the other reads its fields at the wrong offsets, which
is exactly what B1's

    offset_base=[0,0,0,4596944070643295330]

(a float64 read as int64) and the xdist arm's `phdf5 read: ctx_handle is
null` look like.

**NOT CHASED, deliberately.**  The fix is in C++ — distinct symbol
namespaces per build, or a hidden-visibility phdf5 core, or `RTLD_LOCAL`
with an explicit re-export set — and it is not this branch's regression:
the two libraries have always shared these names.  What this branch changed
was how many processes get put into the mixed state, and B1's fix takes
CPU-platform processes back out of it.  `test_so_acceptance`'s check 5 is
the ratchet on the premise for the SLATE half; there is no equivalent
ratchet on the phdf5 half yet, and that is the first thing to add if
anyone picks this up.

### **P1 — three Tier-1 pins are Frontera-frozen and cannot be green on both machines**

> **RULED, 2026-08-07 (owner): "the micro-eV level is fine for comparisons
> between machines."**  Two of the three rows below are RESOLVED BY POLICY
> and move to *FIXED BY POLICY* immediately after this table.  The third is
> **not covered by the ruling** and stays ship-listed, for a reason given
> where it is listed — it is not a micro-eV problem.

| tests | class | evidence | disposition |
|---|---|---|---|
| `test_gw_jax_regression::test_gnppm_matches_reference` | (c) stale/relocated pin | 20 / 2484 rows, **max abs diff exactly 1.000e-6 eV** against `atol=1e-6` | **FIXED BY POLICY** — `_XMACHINE_ATOL_EV = 1e-5` |
| `test_gw_jax_regression::test_bispinor_gnppm_matches_reference` | " | 24 / 1620 rows, **max abs diff exactly 1.000e-6 eV** | **FIXED BY POLICY** — same constant |
| `test_sigma_ppm_gates::test_g2_branch_window_tiles_are_frozen` | (b) real behavioural difference, mis-filed under P1 | crossing-core node ladder is **100** here, the frozen array is **98** | **STILL SHIP-LISTED** — see P1b |

The Frontera census (`f485b5a`, 2026-08-01, job 7885154) re-froze all three
from Frontera CLX output.  Its own KNOWN_FAILURES row said the drift was
"EXACTLY one unit in the 6th printed decimal … 20/2484 resp. 24/1620 rows".
This census measures the same rows, the same 1-ULP size, in the other
direction — because the reference `f485b5a` *replaced* is what Perlmutter
produces:

    sigma_diag_gnppm_ref.dat     n=12  sigC=  4.771156   (pre-f485b5a)   == today's ACTUAL
                                       sigC=  4.771155   (f485b5a)       == today's DESIRED
    sigma_diag_bispinor_ref.dat  n=10  sigXC=-16.978381  (pre-f485b5a)   == today's ACTUAL
                                       sigXC=-16.978382  (f485b5a)       == today's DESIRED

Nothing on this branch moved them: the WSL full-suite red-set diff over the
whole branch (`96a6399` → `d5cac09`) is empty in both directions.

#### FIXED BY POLICY — the two float pins

A 6-decimal `.dat` at `atol=1e-6` has no room for a cross-platform ULP, so
"re-freeze on whichever machine ran last" was a permanent ping-pong in
which each re-freeze silently turned the other machine red.  The owner's
ruling ends it: the comparison tolerance for these two cross-machine-frozen
pins is now **`_XMACHINE_ATOL_EV = 1e-5` eV** (`tests/test_gw_jax_regression.py`),
10× the observed drift and five orders below anything physical.  The
constant carries the ruling, the date, and the scope; it is named on two
cells and nowhere else.

**What still anchors these tightly.**  Loosening a *cross-machine* pin does
not loosen the tree.  Same-machine drift is caught by the Si COHSEX
byte-identity gate (`test_si_production_matches_frozen_reference`, exact
text match, early return) and by the external BerkeleyGW anchor
(`test_si_production_matches_berkeleygw`, `_BGW_TOL`, sub-meV MAE against
another code).  Those are the gates that would see a physics change; these
two answer "does the frozen MoS2 output reproduce on a different machine",
and that answer should not turn on the 6th decimal of a text file.

`_assert_matches_reference` now REPORTS the observed max |Δ| and what
fraction of the atol budget it used, on every run, pass or fail — a
tolerance whose headroom is invisible cannot be audited, and this ruling
is exactly the kind that needs auditing later.

**VERIFIED ON PERLMUTTER**, `svc/distrib_la-2026-08-07` @ `52f5024`,
jobid **56457930**, `lx test -N 1 -G 1 -n 1`, serial (`-n 0`), BUILD_NOTES
pins, artifacts `/pscratch/sd/j/jackm/svc_distrib_la/_reports_p1/p1b.{log,xml}`:

| cell | max abs Δ | atol | budget used | cells over | cells differing |
|---|---|---|---|---|---|
| `test_gnppm_matches_reference` | **1.000e-06 eV** | 1e-05 | **10.0 %** | 0 | 36 / 2484 |
| `test_bispinor_gnppm_matches_reference` | **1.000e-06 eV** | 1e-05 | **10.0 %** | 0 | 50 / 1620 |

`2 passed in 69.61 s`.  Both were previously RED.  The drift is exactly
the 1-ULP-of-the-6th-decimal the census measured — no larger — so the
band has a full decade of headroom and is not absorbing anything else.

> **Unit note, because the two tables disagree and should not be read as
> contradicting.**  The census row above says "20 / 2484 rows"; the
> measured row here says "36 / 2484 cells".  2484 is the CELL count
> (414 rows × 6 compared columns), so the census's label was wrong even
> though its denominator was right, and the counts differ (20 vs 36)
> because they were taken from different runs of a GPU-nondeterministic
> last-ULP effect.  The quantity that matters — the max |Δ| — is
> **1.000e-06 eV in both**, and that is the one the band is set against.

**The tolerance was actually exercised.**  Both cells took the
`assert_allclose` path, not the byte-identity early return (the report
distinguishes the two by name).  A green here therefore means "the drift
is inside the band", not "the files happened to match", which is the
difference between a verified ruling and a vacuous one.

**Neither reference was re-frozen.**  Re-freezing is the move that created
this row; the fix is the comparison, not the data.

#### P1b — `test_g2_branch_window_tiles_are_frozen` is NOT a micro-eV row

Filed under P1 by resemblance and it does not belong there.  The
Perlmutter/Frontera disagreement in this cell is the **crossing-core node
ladder: 100 nodes here, 98 in the frozen array** — an integer count of
quadrature τ points, riding in a `float64` `meta` row.  It is not a
rounding difference, it is not in eV, and a tolerance would hide a real
change in how many points the window integrates over.  The 2026-08-07
ruling therefore does not reach it, and applying it here would be
laundering.

**RE-MEASURED in the same leg** (jobid 56457930, `_reports_p1/p1.log`) and
it is WORSE than the row above recorded — not a 100-vs-98 count with
otherwise-matching values, but a **shape mismatch with different
contents**:

    ω≥E_F cond|0|core|t not bit-identical
    (shapes (100,), (98,) mismatch)
     ACTUAL:  [ 2.666561,  6.499882,  6.936903,  8.974977, ...]
     DESIRED: [ 5.442279e-09, 7.894766, 10.45215, 10.63999, ...]

The τ-node *positions* disagree from the first element, so this is two
different quadratures, not one quadrature sampled twice.  Whatever the
answer is, it is not a tolerance.  Stays ship-listed, class (b), still an
**OWNER DECISION**:
either the ladder legitimately differs between the two machines' minimax
tables (in which case the reference is platform-dependent data and needs a
different mechanism than an atol), or one of the two is wrong.  Nobody has
determined which.  `tests/test_sigma_ppm_gates.py` carries this note at the
comparison itself.

### P2 — the chunk-width gauge cells (pre-existing)

| tests | class | evidence |
|---|---|---|
| `test_bse_setup_qchunk::test_values_are_invariant_to_the_chunk_width` | (b) pre-existing | `_maxdiff = 1.374` against `< 1e-10`; red in isolation, red on WSL, red at the branch base |
| `test_bse_setup_qchunk::test_chunk_width_ulp_spread_is_reported` | (b) pre-existing | reports 5 non-zero spreads, first `(2, 2.22e-16, 2.22e-16)`, where the pin expects `[]` |

Already characterized and A/B'd at `5bb4368`; cited, not re-derived.  The
second cell is the first one's instrument and goes red with it.  On WSL the
whole file is red (12 cells) for want of an FFI; on Perlmutter exactly these
two are.

### P3 — `test_wfn_loader_eager::test_no_ffi_at_P_gt_1_refuses_and_names_both_libraries`

Class (d) environment.  The cell's own docstring says it "runs the resolver
… on a tree with no `.so` (which is this checkout)".  Under the BUILD_NOTES
pins the `.so` IS present, so the terminal refusal arm is unreachable and
the cell reports `DID NOT RAISE`.  **FALSIFIED**: the same cell with
`LORRAX_FFI_SO`/`LORRAX_FFI_HOST_SO` unset is **1 passed**.  Not a code
defect; the cell needs to neutralise the pins itself (monkeypatch
`_locate_so`, or `delenv` both) instead of assuming its checkout is bare.

### P4 — `test_staged_reshard::test_red_twin_the_unstaged_chain_emits_the_spmd_warning` (leg C)

Class (d) environment, and it is the *instrument's own* red twin, which is
why it matters more than a normal skip.  The cell asserts that the UNSTAGED
chain emits `Involuntary full rematerialization`; on this XLA it emits
nothing, so the twin fails with its own message: *"the instrument cannot go
red here, so a zero count from the staged path means nothing."*  Every other
remat gate in leg C is green — but that green is now **unfalsifiable on this
stack**.  Covering leg: none on Perlmutter; the Frontera census's leg C is
where the instrument last demonstrably worked.

### P5 — `test_aot_memory::test_predicted_peak_matches_runtime_3d_fft` (leg D, `extra`)

Class (d) environment.  `device.memory_stats()` returns `None`, so
`["peak_bytes_in_use"]` raises `TypeError`.  The site's shifter string
carries `--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform`, and the platform
allocator keeps no arena to report — the runtime banner says so in the same
run ("The live client reports no arena accounting at all").  **FALSIFIED**:
`env XLA_PYTHON_CLIENT_ALLOCATOR=bfc` on the same device returns the full
dict (`bytes_in_use, bytes_limit, …, peak_bytes_in_use, …`).  The cell
should skip-with-reason when `memory_stats()` is `None` rather than
`TypeError`.

### P6 — `test_contract_bands::test_ffi_gemm_plan` crashes the interpreter at 4 emulated devices

Class (b) pre-existing.  `SIGSEGV` inside the TEST's own numpy reference
(`test_contract_bands.py:129 _ref` → `numpy/_core/einsumfunc.py:1194
bmm_einsum`), or `SIGABRT` at `:135 _relerr` → `jax…array.__array__` when
threads are pinned.  Reproducible alone; `OMP_NUM_THREADS=1` /
`MKL_NUM_THREADS=1` does not change it; deselecting only that cell leaves
the file at **8 passed**.  **PRE-EXISTING: the same cell segfaults at the
branch base `96a6399` on the same leg.**

This one costs coverage twice over: at 1 device the file's 9 device-gated
cells SKIP, and at 4 emulated devices this cell kills the process — so it
has no leg at all.  It is excluded from leg C by name, which is why leg C
has a junitxml.

---

## Environment-limited (skips, each with its covering leg)

Leg A, 62 skips.  Leg A2 = these 62 plus leg B's 22.

| n | reason | covering leg |
|---|---|---|
| 32 | `needs >=4 devices: XLA_FLAGS=--xla_force_host_platform_device_count=4` (`test_staged_reshard`, `test_staged_reshard_routes`) | **C** (green) |
| 11 | `needs 4 (emulated) devices, got 1` (`test_contract_bands` 9, `test_projection_lgemm` 2) | **C** for `projection_lgemm`; `test_contract_bands` → **P6, uncovered** |
| 7 + 1 | `needs 4 devices, have 1` / `needs 2 devices, have 1` (`test_charge_zeta_route`) | **C** and **C3** (green) |
| 4 | `needs >=4 devices to build a 2x2 mesh` (`test_sharding_fit`) | **C** (green) |
| 1 | `needs >= 4 devices for a 2x2 mesh; have 1` (`test_file_io`) | the 4-device CPU leg — UNCOVERED at census time (B1 aborted it), **covered since `7a1d64f`: 42 P / 1 S** |
| 1 | `device count 1 is a perfect square; the refusal arm needs a non-square count` | **C2** (1 passed at 3 devices) |
| 1 | `needs 4 (emulated) devices` (`test_sanity_gates_jax::test_check_hermitian_sharded`) | **C** (green) |
| 1 | `P=1: jax.devices()[0] IS this process's device, so the negative control cannot fire` | true P>1 srun leg; **not** the emulated legs |
| 1 | `` `lfs` unavailable here, so the stripe count cannot be read `` | none — the skip says outright it verified nothing |
| 1 | `fit_scissor still accepts the energy arrays positionally` | self-documenting deprecation skip |
| 1 | **`eqp_si_fast_ref.dat` is not frozen yet — freezing a reference is the owner's call.  A candidate generated 2026-08-07 at 04b8bba lives in `/pscratch/sd/j/jackm/si_consolidation_2026-08-07/run_fast_final/` (eqp_si_fast.dat); copy it in to enable this gate.** | **INTENTIONAL, VISIBLE.**  The 20-band fast gate.  Owner's call; NOT frozen by this census |

Leg B / E2, 22 and 3 skips:

| n | reason | covering leg |
|---|---|---|
| 19 (leg B only) | `needs >= 4 devices on platform 'cpu', have 1. Set XLA_FLAGS=… BEFORE the first jax import` | legs **F**/**G** — the real 4-process 2×2, which is the coverage these emulate |
| 2 | `slate host heev SIGSEGVs — bug L-2, see docs/dev/linalg_ffi.md` | none; pinned by the skip itself, carried by design |
| 1 | `cholesky/cusolvermp is not usable on a 1x1 gpu mesh` (needs a true-2D mesh) | leg **G** (`cusolvermp_factor_solve[2x2]` green, residual 6.0e-16) |

Leg D, 2 skips: no CrI3 6×6 30Ry SOC `WFN.h5` reachable (out-of-repo
fixture); `jax.jit` decorator-factory form unsupported on jax 0.7.0.

Legs F/G skips are the cross-platform halves — `scalapack_*` and
`batched_eigh_dispatch` are host-only and skip on G; `cusolvermp_*` are
CUDA-only and skip on F.  Between them every cell runs on one of the two.

---

## Instrument notes (how to run this suite on Perlmutter)

1. **A shifter `--env` beats your exported environment, and `XLA_FLAGS` is
   one.**  The NVIDIA jax image ships its own `XLA_FLAGS`, so
   `XLA_FLAGS=--xla_force_host_platform_device_count=4 lx test …` arrives in
   the container as `XLA_FLAGS=  --xla_gpu_enable_latency_hiding_scheduler=true`
   and `jax.device_count()` is **1**.  The leg then SKIPS its way to green
   and looks identical to a leg that ran.  MEASURED both ways; the form that
   takes is `lx run … env XLA_FLAGS=… python3 -m pytest …`, because `env`
   runs *inside* the container.  Every emulated-multi-device leg here uses
   that form, and the first attempt (recorded, superseded) did not.
2. **`lx test` is xdist.**  One node, all GPUs, four workers, one GPU pinned
   per worker by `tests/conftest.py` — and NOT by its controller, which is
   what **B2** was.  It is why the single-process `lx run … -m pytest` leg
   is the control for every session-fixture red.
3. **Judge by artifacts.**  Leg A exits non-zero (`srun: task 0: Exited with
   exit code 1`) and still wrote a 246 kB junitxml with 1191 cells; leg C's
   first attempt exited 139 and wrote NO xml at all.  The exit code
   distinguished neither.
4. **`pytest-timeout` is not installed in this image** — `--timeout=` is an
   `unrecognized argument` and kills the leg with exit 4.  Wrap the payload
   in `timeout N …` instead.
5. **`lx run --cpu` has no container and therefore no jax.**  The CPU L-c
   leg is `lx run -N 1 -G 4 -n 4 env JAX_PLATFORMS=cpu …`, not `--cpu`.
6. Per-leg env, verbatim, is in
   `/pscratch/sd/j/jackm/svc_distrib_la/{census,census2,recheck*,bisect*}.sh`.

## Cross-machine set-diff (WSL, jax 0.9.1, no FFI)

`96a6399` → `d5cac09`, `python -m pytest -q -p no:randomly`, whole branch:
**1420 → 1441 collected, 95 → 95 red, identical ids, 0 removed, 0 newly
red.**  The WSL leg cannot see any of B1/B2 (no FFI, no CUDA) — that is
exactly why the Perlmutter census had to exist.

Re-measured across the two fixes, `35f3e06` → `f7c1b17`: **1441 → 1471
collected, 95 → 95 red, identical ids, 0 removed, 0 newly red, +30 new
cells all green.**  The 30 are the two-armed load-order cells and the two
capability tables (B1) and the controller cells (B2); the xdist twin skips
on WSL for want of the plugin and runs on Perlmutter.

Re-measured again across the step-6 follow-up, `b425291` → `d880e67`:
**1471 → 1480 collected, 95 → 95 red, 185 → 185 skipped, 0 removed,
0 newly red, 0 newly green, +9 new cells ALL GREEN.**  The 9 are the
cost-notice trio, the L-3 trio and the L-4 trio, all resolve-level and all
runnable with no GPU and no `.so`.  `64 failed + 31 errors = 95` in
2348 s.

> **Diff against `wsl_fix/wsl_fix_pre.xml` (1471), NOT `wsl/wsl_efdbf9a.xml`
> (1441)**, or the +30 cells from the B1/B2 fixes are re-counted as new.
> `b425291` is `f7c1b17` plus one `.md`-only commit, so the `f7c1b17`
> artifact is the right baseline for anything measured after it.  This
> trap cost nothing only because it was noticed before the diff was run.

---
---

# Frontera CLX census, 2026-08-01 — HISTORICAL RECORD

**Kept, not deleted.**  Superseded as the tree's census by the Perlmutter
one above; still the authority for what Frontera CPU measured, and the
document class **P1** is measured against.

Complete `python -m pytest tests/` census on Frontera CLX (in-container,
required-FFI defaults, host .so `build_host_MRG`), 2026-08-01, tree
`bbe6e56` + the fixes committed with that census.  Authoritative run:
**job 7885154** (junit XMLs + run dirs under
`/scratch2/08271/jackmc/pytest_p11/`).  Job 7885150 was the first attempt
and is superseded — its 97 failures were dominated by an instrument error
(see "Instrument notes"), kept only as evidence.

## Verdicts by leg (job 7885154)

| leg | invocation | result |
|---|---|---|
| A2: full suite, bare, 1 device | `pytest tests/ --ignore=tests/test_ffi_linalg_contract.py` | 24 failed / 735 passed / 56 skipped / 26 deselected |
| B2: FFI linalg contract | `srun --mpi=pmi2 -n1` + `config/frontera/mpi_transport_env.sh`, `pytest tests/test_ffi_linalg_contract.py` | **0 failed** / 27 passed / 25 skipped |
| C: 4-device leg | `XLA_FLAGS=--xla_force_host_platform_device_count=4`, the 9 device-hungry files | 1 failed / 144 passed / 2 skipped |
| C2: nonsquare refusal | `...device_count=2`, `-k nonsquare` | 1 passed |
| D: extra tier | `-m extra`, 4 devices | 0 failed / 21 passed / 5 skipped |

> **The A2/B2 invocations above are RECORDED AS RUN (job 7885154) and are no
> longer runnable as written.** `tests/test_ffi_linalg_contract.py` moved to
> `services/distrib_la/tests/test_distrib_la_contract.py` and carries the
> `distrib_la` marker, so naming a path is no longer the way to include or
> exclude it. Today the same two legs are
> `pytest tests/ --no-services` and `srun --mpi=pmi2 -n1` +
> `config/frontera/mpi_transport_env.sh` + `pytest -m distrib_la`.
> `tests/conftest.py` owns those hooks and `tests/test_service_selection.py`
> measures that they select what they claim.

Every leg-A2 failure below was triaged; after the fixes in that commit
the only remaining red was the ring-vma class.

## KNOWN FAILURES (ship-listed, as of 2026-08-01)

| tests | class | evidence | status |
|---|---|---|---|
| 10 ring-transport tests: `test_bse_dense_reference` `{w_positive_control,full_H,DV}[ring]` + `test_nontda_matvec_matches_dense_shao` + `test_nontda_solver_reproduces_dense`, `test_bse_stack_matvec::test_stack_memory_flat_in_n_trials`, `test_bse_w0_resolvent` (2), `test_bse_w_omega_chain` (2) | (b) pre-existing — the old handoff's "bse_ring_comm vma", verified present at that HEAD and now precisely diagnosed | `TypeError: scan body ... carry ... {V:(x,y)} varying manual axes` at `src/bse/bse_ring_comm.py:382` (`_apply_V_ring_only` fori_loop carry `A0` unannotated; jax's error prescribes `lax.pvary` on the initial carry). junitA2_7885154. Serial + simple matvec arms PASS, so the dense-reference physics is still covered; only the ring transport arm is dark | **CLOSED on Perlmutter/jax 0.7.0**: all 10 pass in the single-process `iso_bse` leg of the 2026-08-07 census. The Frontera/jax-version scope of the original diagnosis is unretracted |
| `test_ffi_linalg_contract` under a BARE (no-srun) launch with the host .so loadable: silent interpreter death at import | (d) environment — MPI init without PMI2 glue | CLAIMS 30; reproduced 7885125 step1 (srun WITHOUT transport env also dies). **Upgrade that census added: with `mpi_transport_env.sh` sourced under `srun --mpi=pmi2 -n1` the pytest form is fully GREEN (leg B2, 27 passed)** — the CLI matrix is no longer the only instrument | Not a code bug; invocation contract. The bare leg must deselect the service (`--no-services`); the srun+transport leg (`-m distrib_la`) covers it. On Perlmutter neither death occurs: leg B is 130 cells / 0 failed under plain `lx test` |
| `test_centroid_distribution::test_orbit_path_with_trivial_group_matches_plain_path` | (d) environment — reproduces ONLY on an UNSUPPORTED jax. Corrected 2026-08-05: an earlier revision of this row claimed the defect was version-independent; direct measurement disproved that, see evidence | `jax.errors.UnexpectedTracerError`: an `int64[]` tracer whose creating frame is `src/centroid/orbit_syms.py:241` at **`<module>` scope** escapes the jit of `kmeans_pp_init` (`src/centroid/kmeans_isdf.py:585`) and is raised out of the Lloyd loop at `src/centroid/kmeans_isdf.py:738`. Only the orbit-on (`R=`/`Rinv=`/`tau=`) branch trips. **Seen only under the Shifter image's bundled jax 0.5.3** (Perlmutter GPU census, jobid 56385965, `pytest_gpu.log`: 7 failed / 920 passed). Under the SUPPORTED jax the same test **passes** | Still a **tripwire**, and it did not fire: the 2026-08-07 Perlmutter census runs `lorrax_J070` (jax 0.7.0.dev20260807) and `test_centroid_distribution` is green in legs A, A2 and C |

## Environment-limited on Frontera (skips, each with its covering leg)

| tests | reason | coverage |
|---|---|---|
| 45 device-count skips (leg A2): `test_staged_reshard` (14), `test_staged_reshard_routes` (18), `test_charge_zeta_route` (7), `test_sharding_fit` (4), `test_collectives_distribution`, `test_centroid_distribution`, `test_sanity_gates_jax::test_check_hermitian_sharded` | need >=2/4 emulated devices | leg C (all green after the fix below); nonsquare-refusal cell needs a NON-square count → leg C2 (green) |
| `test_centroid_distribution::test_process_local_mesh_is_addressable` negative control | needs true multi-PROCESS (P>1), not emulated devices | P>1 srun leg (P1 scaling legs); `tests/multi_device/` is likewise srun-driven, never pytest-collected |
| `test_bse_kgrid` (7), `test_wfn_transforms::test_to_box_{ibz,full_bz}_mos2` (2), `test_R_proper_cri3` (1, extra tier) | fixtures pinned to `/pscratch/...` — Perlmutter, machine gone | RESOLVED for `test_bse_kgrid` by running on Perlmutter: it is collected and (single-process) green in the 2026-08-07 census. The census's second name, `test_wfn_loader_eager[mos2]` (3), no longer resolves at this HEAD: that cell moved to `services/wfn_loader/tests/test_wfn_loader_contract.py` on 2026-08-07 and its dead `/pscratch` arm was repointed at the in-repo `gnppm_debug/WFN.h5` twin (survey w1_wfn_loader §6.4 — byte-size identical), so it RUNS on both machines now and the Frontera skip count dropped 3 → 2; the census's green-on-Perlmutter finding for it stands and is no longer machine-dependent. Still OWNER's for the 2 remaining `to_box` cells: restage the MoS2 3×3 640c fixture + WFN.h5 on Frontera (or re-point); until then those self-skip |
| 23 CUDA cells in `test_ffi_linalg_contract`, `-m gpu`-dependent extras (3 cufft + 1 CUDA backend in leg D) | need a CUDA jax backend | P1 GPU leg (rtx); on Perlmutter these are the leg-B cells that run |
| `test_slate_cholesky_trsm_cpu` heev cells (2 skips in leg B2) | slate host heev SIGSEGV — documented bug L-2, `docs/dev/linalg_ffi.md` | pre-existing, pinned by the skip itself; SAME 2 skips on Perlmutter (legs B, E2) |
| 26 deselected (`-m extra` tier) | deselected by repo `addopts` default | leg D ran them: 21 passed / 5 skipped |

## Fixed in the Frontera pass (committed with `f485b5a`)

| tests | root cause | fix | validation |
|---|---|---|---|
| `test_file_io` (12) + `test_compute_all_V_q_g_flat::test_..._rejects_r_space_loader` | (c) stale builders: synthetic `zeta_q.h5` helpers never stamped `zeta_is_done`, and `ZetaLoader` now refuses partial files at open (completeness gate) | builders stamp done (complete synthetic payloads); the flag-behaviour tests pass `zeta_is_done=False` explicitly | GREEN in 7885154 leg A2, and GREEN on Perlmutter |
| `test_zq_from_psi_sm_bit_identity` (6) | (c) `_MockPsiGStore` missing `_bpd_per_bc` | mock mirrors `psi_G_store.py:147` | GREEN in 7885154 leg A2, and GREEN on Perlmutter |
| `test_sigma_ppm_gates::test_g2_branch_window_tiles_are_frozen` | (c) stale pin: G2 npz frozen 2026-07-07; `d011a36` reconditioned the Σc HGL crossing quadrature — crossing-core node ladder changed 103→98 | regenerated via `_regenerate_g2_reference()` (job 7885154 step 0, CPU/f64) | GREEN on Frontera — **and RED on Perlmutter, where the ladder is 100. See class P1** |
| `test_gw_jax_regression::test_gnppm_matches_reference`, `::test_bispinor_gnppm_matches_reference` | platform-migrated pins: refs frozen on Perlmutter GPU (b7654ee); on Frontera CPU/FFI the drift is EXACTLY one unit in the 6th printed decimal (max delta 1.000e-6 eV, sigC/sigXC only; 20/2484 resp. 24/1620 rows) against `atol=1e-6` | re-froze both `sigma_diag_*_ref.dat` from the job-7885154 session outputs | **This is the move class P1 documents. The re-freeze relocated the red rather than removing it: on Perlmutter the same 20/2484 and 24/1620 rows are now off by the same 1.000e-6, and the pre-`f485b5a` reference is bit-equal to what Perlmutter produces today** |
| `test_runtime_distributed::test_set_default_env_defaults` | (a) real gap in `skip_gpu_plugin_discovery` | both branches re-apply the demotion pinning | GREEN on Perlmutter (legs A, A2) |
| `test_charge_zeta_route::test_rank_truncate_refuses_rather_than_downgrading` (leg C) | (c) stale pin vs the R15.1 widening | refusal case moved to `_OVER_FACTOR_CAP` (μ=17000) | GREEN on Perlmutter legs C and C3 |
| `test_contract_bands` (9) + `test_projection_lgemm` (2) failing at 1 device | test defect: `assert n_dev >= 4` instead of the suite-wide `pytest.skip` | `_mesh()` skips below 4 devices | GREEN under Frontera leg C; on Perlmutter `test_projection_lgemm` is green and `test_contract_bands` hits class P6 |

## Old-handoff known-fail list, as verified 2026-08-01

| handoff item | verdict |
|---|---|
| file-IO fixtures | root-caused (zeta_is_done completeness gate vs stale test builders) and FIXED |
| bse_ring_comm vma | present on Frontera/jax at that tree; **not reproducible on Perlmutter / jax 0.7.0 (2026-08-07)** |
| kmeans multi-rank segfault | not reproducible in pytest scope; P>1 thread-main refusal FIXED repo-side (24e4dc3, subsumed e97e8ed); true multi-process kmeans belongs to the P>1 srun leg |
| GN-PPM pred remat | remat gates ALL GREEN in Frontera leg C (32 tests). **On Perlmutter the gates are green but their RED TWIN is dead — see class P4** |

## Frontera instrument notes (the 7885150 lesson)

1. Export the environment INSIDE the container: apptainer does not forward
   the host `LD_LIBRARY_PATH`, and without it the required-FFI gate
   refuses (`libhdf5.so.310` unresolvable) — that single mistake produced
   68 failed + 29 errors in job 7885150.  Pattern: job script
   `/scratch2/08271/jackmc/pytest_p11/run_pytest2.sbatch`.
2. The `distrib_la` service suite must be DESELECTED in the bare leg
   (`pytest tests/ --no-services`; import-time death without PMI2 glue)
   and run under `srun --mpi=pmi2 -n1` with
   `config/frontera/mpi_transport_env.sh` sourced (`pytest -m distrib_la`)
   — green there.  Deselect through the hook, never through a second
   `-m`: `pyproject` sets `addopts = "-m 'not extra'"` and an explicit
   `-m` REPLACES it, silently re-enabling the whole `extra` tier.
3. Per-test timeout: `pytest-timeout` staged on `PYTHONPATH`
   (`--timeout=2400 --timeout-method=signal`); do NOT also pass
   `-p pytest_timeout` (double registration).  It is NOT installed in the
   Perlmutter image — see Perlmutter instrument note 4.
4. e2e regression fixtures run the drivers on the CPU node via
   `ISDF_COHSEX_TEST_PLATFORM=auto` (jax native pick); compile cache under
   `$SCRATCH` keeps the whole suite ~21 min.

---

# eqp0.dat / eqp1.dat mix two bases on the self-consistent path

OPEN, UNMEASURED, 2026-08-05.  Carried forward unchanged; neither census
touches it.  `gw_jax.py:652-654` reads `sigma_c_omega_kij_ry` off the object
`run_sc_driver` returns and emits its diagonal as `sigma_c_omega_diag_ev`.
That field is in the QP basis and is correct there — Σ is Hermitised as
½(Σ(E_n)+Σ(E_m)), which only means anything in the basis whose eigenvalues
are those E_n, so it must not be rotated.  The finalize rotates the five
static Σ fields to the DFT basis and carries the cube through unrotated,
which is deliberate.

The defect is downstream: `compute_eqp_diag` forms
`Δ = kin_ion + V_H + Σ_x + Σ_c(E_DFT) − E_DFT` from three DFT-basis
diagonals plus that one QP-basis diagonal.  The sum is basis-consistent
only at U = identity.  `write_results` is unguarded, so both files are
written on the SC path (unlike `eqp_g0w0.dat`, guarded at
`gw_output.py:846`).  The same mixing reaches `sigma_xc_at_dft_ev`
(`gw_jax.py:600-603`).

The error scales with ‖U − 1‖ and NO ONE HAS MEASURED IT.  To measure:
read `U_mnk` from an SC run's `qp_wfn_rotations.h5`, report the largest
off-diagonal element, and bound the eqp error by the off-diagonal Σ weight
it mixes in.  One-shot runs are unaffected — `solve_qp` is reached only in
the non-SC branch (`gw_jax.py:543`), where the whole object is DFT basis.

`tests/test_sigma_result_basis.py` pins which field is in which basis, so
a new Σ channel cannot join the wrong group silently.  It does not and
cannot catch this: the mixing is in the consumer, not the declaration.
